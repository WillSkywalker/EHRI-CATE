"""LLM4SSH zero-shot subject-indexing backend.

Calls the GRAPHIA LiteLLM proxy at llm.graphia-ssh.eu. The prompt is in English
(per the dataset-level decision to rely on the model's multilingual capability
rather than translate), and the input text is fed in its source language.

Returns a list of (uri, score) predictions ranked by confidence. The harness
in src/ehri_cate/scoring.py decides what top-k to evaluate against.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import litellm

from ..rate_limit import NullRateLimiter, RateLimiter
from ..vocab import Vocab


# Match the body LLM4SSH returns on 429:
#   "Limit resets at: 2026-05-20 19:23:19 UTC"
_RESET_AT_RE = re.compile(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")


def _parse_reset_seconds(message: str) -> float | None:
    """Seconds until the rate-limit reset stated in a 429 body, or None."""
    match = _RESET_AT_RE.search(message)
    if not match:
        return None
    try:
        reset = datetime.fromisoformat(match.group(1).replace(" ", "T")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = (reset - datetime.now(timezone.utc)).total_seconds()
    return max(delta, 0.0)


# A predicted (uri, score). Score is in [0, 1]; the LLM doesn't actually give
# calibrated probabilities, so we use rank-position-derived scores.
Prediction = tuple[str, float]


SYSTEM_PROMPT = """You are a subject indexer for Holocaust-related archival material.

You are given an archival description in any language (English, German, Russian, \
Italian, French, Dutch, Czech, Hebrew, Polish, Hungarian, Ukrainian, Serbo-Croatian, \
or others). You must assign subject terms from the EHRI Terms controlled vocabulary.

You will be given a numbered list of candidate terms with their English preferred labels. \
Pick the most relevant terms (typically 1 to 5). Reply with ONLY a JSON object \
of the form:

  {"labels": [<term_number>, <term_number>, ...]}

Ordered from most to least relevant. Do not invent term numbers; only use numbers \
from the candidate list. Do not include explanations or any other text — JSON only."""


USER_TEMPLATE = """Archival description:
\"\"\"
{text}
\"\"\"

Candidate subject terms:
{candidates}

Reply with a JSON object containing the most relevant term numbers in order."""


@dataclass
class LLM4SSHBackend:
    """Zero-shot LLM subject indexer over a LiteLLM proxy."""

    model: str                       # e.g. "Mistral-Small-3.2-24B-Instruct-2506"
    vocab: Vocab
    candidate_uris: list[str]        # which subset of the vocab to offer as candidates
    api_base: str
    api_key: str
    max_tokens: int = 200
    temperature: float = 0.0
    timeout: float = 120.0
    request_label_lang: str = "en"   # language of candidate labels shown to the LLM
    rate_limiter: RateLimiter | NullRateLimiter = field(default_factory=NullRateLimiter)
    max_429_retries: int = 2

    def __post_init__(self):
        # Pre-build the candidate menu once; it's the same for every doc.
        self._candidates_text, self._number_to_uri = self._build_candidate_menu()

    def _build_candidate_menu(self) -> tuple[str, dict[int, str]]:
        labels = self.vocab.candidate_labels(self.request_label_lang)
        lines: list[str] = []
        number_to_uri: dict[int, str] = {}
        for i, uri in enumerate(self.candidate_uris, start=1):
            label = labels.get(uri, uri)
            lines.append(f"{i}. {label}")
            number_to_uri[i] = uri
        return "\n".join(lines), number_to_uri

    def suggest(self, text: str) -> list[Prediction]:
        if not text.strip():
            return []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                text=text, candidates=self._candidates_text,
            )},
        ]

        attempts = 0
        while True:
            self.rate_limiter.acquire()
            try:
                resp = litellm.completion(
                    model=f"litellm_proxy/{self.model}",
                    api_base=self.api_base,
                    api_key=self.api_key,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
                break
            except litellm.RateLimitError as e:
                attempts += 1
                if attempts > self.max_429_retries:
                    raise
                wait = _parse_reset_seconds(str(e)) or (2 ** attempts)
                # Cap the wait so a clock-skew bug can't hang the run.
                time.sleep(min(wait, 70.0))

        content = resp.choices[0].message.content or ""
        numbers = self._parse_label_numbers(content)
        return self._numbers_to_predictions(numbers)

    @staticmethod
    def _parse_label_numbers(raw: str) -> list[int]:
        # Be lenient: pull the first JSON object that has a "labels" array.
        # Falls back to extracting any integer list if the JSON is malformed.
        match = re.search(r"\{[^{}]*\"labels\"\s*:\s*\[[^\]]*\][^{}]*\}", raw, re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
                labels = obj.get("labels", [])
                return [int(x) for x in labels if isinstance(x, (int, str)) and str(x).strip().lstrip("-").isdigit()]
            except (json.JSONDecodeError, ValueError):
                pass
        # Fallback: any integer-looking tokens, deduped, in order
        seen: list[int] = []
        for tok in re.findall(r"\b\d+\b", raw):
            n = int(tok)
            if n not in seen:
                seen.append(n)
        return seen

    def _numbers_to_predictions(self, numbers: list[int]) -> list[Prediction]:
        preds: list[Prediction] = []
        for rank, n in enumerate(numbers):
            uri = self._number_to_uri.get(n)
            if uri is None:
                continue  # hallucinated number, skip
            # Rank-derived score in (0, 1]; #1 gets 1.0, #2 gets ~0.83, etc.
            score = 1.0 / (1.0 + rank * 0.2)
            preds.append((uri, score))
        return preds


def load_backend_from_env(
    model: str,
    vocab: Vocab,
    candidate_uris: list[str],
    rate_limiter: RateLimiter | NullRateLimiter | None = None,
) -> LLM4SSHBackend:
    """Convenience constructor that reads api_base/api_key from env (loaded from .env)."""
    api_base = os.environ.get("LLM4SSH_API_BASE", "https://llm.graphia-ssh.eu")
    api_key = os.environ["LLM4SSH_API_KEY"]  # raises KeyError if unset — by design
    return LLM4SSHBackend(
        model=model,
        vocab=vocab,
        candidate_uris=candidate_uris,
        api_base=api_base,
        api_key=api_key,
        rate_limiter=rate_limiter or NullRateLimiter(),
    )
