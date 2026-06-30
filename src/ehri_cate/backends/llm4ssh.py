"""LLM4SSH zero-shot subject-indexing backend.

Calls the GRAPHIA LiteLLM proxy at llm.graphia-ssh.eu. The prompt is in English
(per the dataset-level decision to rely on the model's multilingual capability
rather than translate), and the input text is fed in its source language.

Returns a list of (uri, score) predictions ranked by confidence. The harness
in src/ehri_cate/scoring.py decides what top-k to evaluate against.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import litellm

from ..rate_limit import NullRateLimiter, RateLimiter
from ..vocab import Vocab


# The model used when the user passes no --model. DeepSeek-V3.1 is the strongest
# single model on the proxy (best in our eval), so it's the sensible zero-config
# default — "best general-purpose", not the lightest. The live /v1/models list
# carries no size metadata, so the default is pinned here rather than derived.
DEFAULT_MODEL = "DeepSeek-V3.1-vLLM"


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


# Placeholders a user-supplied `user` template must contain. {text} is the
# document and {candidates} is the numbered menu; both are mandatory because the
# numbered-JSON output contract in _parse_label_numbers depends on the menu.
_REQUIRED_USER_PLACEHOLDERS = ("{text}", "{candidates}")


@dataclass(frozen=True)
class PromptTemplate:
    """A (system, user) prompt pair. Defaults to the shipped zero-shot prompt."""

    system: str = SYSTEM_PROMPT
    user: str = USER_TEMPLATE

    def render_user(self, text: str, candidates: str) -> str:
        # Plain replace (not str.format) so a template may contain literal braces
        # — e.g. a JSON example — without tripping format()'s field parser.
        return self.user.replace("{text}", text).replace("{candidates}", candidates)


def load_prompt_template(path: str | Path) -> PromptTemplate:
    """Load a prompt template from a TOML file with `system` and/or `user` keys.

    Either key may be omitted, in which case the shipped default for that part is
    kept — so you can override just the system prompt and leave the user template
    untouched. The `user` template must contain the {text} and {candidates}
    placeholders; failing fast here beats silently producing empty predictions.
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - {"system", "user"}
    if unknown:
        raise ValueError(
            f"prompt template {path}: unexpected key(s) {sorted(unknown)}; "
            f"only 'system' and 'user' are allowed"
        )
    system = data.get("system", SYSTEM_PROMPT)
    user = data.get("user", USER_TEMPLATE)
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError(f"prompt template {path}: 'system' and 'user' must be strings")
    missing = [p for p in _REQUIRED_USER_PLACEHOLDERS if p not in user]
    if missing:
        raise ValueError(
            f"prompt template {path}: 'user' template is missing required "
            f"placeholder(s) {missing}"
        )
    return PromptTemplate(system=system, user=user)


@dataclass
class LLM4SSHBackend:
    """Zero-shot LLM subject indexer over a LiteLLM proxy."""

    model: str                       # e.g. "Mistral-Small-3.2-24B-Instruct-2506"
    vocab: Vocab
    candidate_uris: list[str]        # which subset of the vocab to offer as candidates
    api_base: str
    api_key: str
    prompt: PromptTemplate = field(default_factory=PromptTemplate)
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

    @property
    def candidate_menu(self) -> str:
        """The numbered candidate-term menu sent to the model (fixed per run)."""
        return self._candidates_text

    def build_messages(self, text: str) -> list[dict]:
        """The exact chat messages `suggest` would send for `text`."""
        return [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.render_user(
                text, self._candidates_text,
            )},
        ]

    def suggest(self, text: str) -> list[Prediction]:
        if not text.strip():
            return []

        messages = self.build_messages(text)

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


@dataclass(frozen=True)
class PromptSize:
    """A size report for an assembled prompt. `tokens` is approximate: for the
    LLM4SSH proxy models we don't have the model's real tokenizer, so it's either
    litellm's best-effort count or a ~4-chars-per-token fallback. `chars` is exact."""

    chars: int
    tokens: int
    token_method: str  # "litellm" or "chars/4"


def measure_messages(model: str, messages: list[dict]) -> PromptSize:
    """Size of a chat-message list: exact chars + an approximate token count."""
    chars = sum(len(m.get("content", "") or "") for m in messages)
    try:
        tokens = int(litellm.token_counter(model=f"litellm_proxy/{model}", messages=messages))
        method = "litellm"
    except Exception:  # noqa: BLE001 — tokenizer unknown for many proxy models
        tokens = chars // 4
        method = "chars/4"
    return PromptSize(chars=chars, tokens=tokens, token_method=method)


def load_backend_from_env(
    model: str,
    vocab: Vocab,
    candidate_uris: list[str],
    rate_limiter: RateLimiter | NullRateLimiter | None = None,
    prompt: PromptTemplate | None = None,
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
        prompt=prompt or PromptTemplate(),
        rate_limiter=rate_limiter or NullRateLimiter(),
    )


def list_models(api_base: str, api_key: str, timeout: float = 30.0) -> list[str]:
    """Model IDs the LiteLLM proxy serves, via its OpenAI-style /v1/models endpoint.

    Returns the IDs in the order the proxy lists them. The list is unfiltered: it
    may include non-chat models (embeddings, rerankers), since /v1/models exposes
    no capability metadata to filter on.
    """
    url = api_base.rstrip("/") + "/v1/models"
    resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", []) if isinstance(m, dict) and "id" in m]


def list_models_from_env(timeout: float = 30.0) -> list[str]:
    """list_models() reading api_base/api_key from the environment (loaded from .env)."""
    api_base = os.environ.get("LLM4SSH_API_BASE", "https://llm.graphia-ssh.eu")
    api_key = os.environ["LLM4SSH_API_KEY"]  # raises KeyError if unset — by design
    return list_models(api_base, api_key, timeout=timeout)


def set_debug_logging(debug: bool) -> None:
    """Toggle litellm's verbose logging. Off by default so runs stay quiet; the CLI
    calls this with --debug. The import-time LiteLLM warnings are quieted separately
    (before litellm is imported), since they fire too early for this to catch."""
    logging.getLogger("LiteLLM").setLevel(logging.DEBUG if debug else logging.ERROR)
    litellm.suppress_debug_info = not debug
    if debug and hasattr(litellm, "_turn_on_debug"):
        litellm._turn_on_debug()
