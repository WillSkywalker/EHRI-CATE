"""EHRI-CATE web API: EHRI Terms suggestions for archival descriptions over HTTP.

A thin Flask layer over the same LLM backend `cate label` uses, so a suggestion
from the API equals the CLI's for the same text, model and options. Served by
gunicorn via the root wsgi.py (the layout EHRI's python_web ansible deploys).

The vocab and the in-use candidate list come from the snapshot in resources/,
not data/ (gitignored, so absent on a deployed checkout).
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from werkzeug.exceptions import HTTPException

# Quiet LiteLLM's import-time warnings; must run before litellm is imported via
# the backend (see the same note in cli.py).
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm  # noqa: E402

from .backends.llm4ssh import (  # noqa: E402
    DEFAULT_MODEL,
    LLM4SSHBackend,
    TruncatedResponseError,
    list_models_from_env,
    load_backend_from_env,
)
from .vocab import Vocab  # noqa: E402

SERVICE_TITLE = "EHRI-CATE subject indexing API"

RESOURCES = Path(__file__).parent / "resources"

MAX_TEXT_CHARS = 100_000
MAX_TOP_K = 50

load_dotenv()

app = Flask(__name__)
app.json.sort_keys = False
app.json.ensure_ascii = False

_vocab = Vocab.from_turtle(RESOURCES / "ehri_terms.ttl")
_labels = _vocab.candidate_labels("en")
# Already sorted (see scripts/export_in_use_terms.py), the order the eval used.
_candidates: dict[str, list[str]] = {
    "in-use": (RESOURCES / "in_use_terms.txt").read_text(encoding="utf-8").split(),
    "full": sorted(_vocab.concepts.keys()),
}


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException):
    """Errors stay JSON, in the ``{"detail": …}`` shape."""
    return jsonify(detail=error.description), error.code


@app.get("/")
def root() -> dict[str, Any]:
    """Service description."""
    return {
        "name": SERVICE_TITLE,
        "default_model": DEFAULT_MODEL,
        "vocabulary": "EHRI Terms",
        "candidates": {name: len(uris) for name, uris in _candidates.items()},
        "endpoints": [
            "GET models",
            "POST suggest  {text, model?, top_k?, candidates?, thinking?}",
        ],
    }


@app.get("/models")
def models() -> dict[str, Any]:
    """Models the LLM4SSH proxy currently serves (live), default flagged."""
    try:
        available = list_models_from_env()
    except KeyError:
        abort(503, "LLM4SSH_API_KEY is not configured on the server.")
    except Exception as e:  # noqa: BLE001 — any network/HTTP failure upstream
        abort(502, f"Could not fetch the model list: {e}")
    return {"models": [{"id": m, "default": m == DEFAULT_MODEL} for m in available]}


@app.post("/suggest")
def suggest() -> dict[str, Any]:
    """Ranked EHRI Terms for a text; the HTTP twin of `cate label --json`."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, "Request body must be a JSON object.")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        abort(400, "'text' must be a non-empty string.")
    if len(text) > MAX_TEXT_CHARS:
        abort(400, f"'text' is longer than {MAX_TEXT_CHARS} characters.")

    model = body.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model:
        abort(400, "'model' must be a non-empty string.")

    top_k = body.get("top_k", 5)
    # bool is an int subclass; reject it so `true` isn't read as 1
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        abort(400, f"'top_k' must be an integer between 1 and {MAX_TOP_K}.")

    candidates = body.get("candidates", "in-use")
    if candidates not in _candidates:
        abort(400, f"'candidates' must be one of {sorted(_candidates)}.")

    thinking = body.get("thinking", True)
    if not isinstance(thinking, bool):
        abort(400, "'thinking' must be a boolean.")

    try:
        backend = _backend(model, candidates, thinking)
    except KeyError:
        abort(503, "LLM4SSH_API_KEY is not configured on the server.")

    try:
        preds = backend.suggest(text)
    except (litellm.BadRequestError, litellm.NotFoundError, litellm.PermissionDeniedError) as e:
        abort(400, f"The model proxy rejected the request (unknown or blocked model?): {e}")
    except TruncatedResponseError as e:
        abort(502, str(e))
    except Exception as e:  # noqa: BLE001 — timeouts, rate limits, proxy errors
        abort(502, f"Model call failed: {type(e).__name__}: {e}")

    return {
        "model": model,
        "candidates": candidates,
        "thinking": thinking,
        "suggestions": [
            {"rank": i, "uri": u, "label": _labels.get(u, u), "score": round(s, 4)}
            for i, (u, s) in enumerate(preds[:top_k], start=1)
        ],
    }


@functools.lru_cache(maxsize=32)
def _backend(model: str, candidates: str, thinking: bool) -> LLM4SSHBackend:
    # One backend per option combination: the candidate menu is built once and
    # the backend holds no per-request state, so sharing it across threads is safe.
    return load_backend_from_env(
        model=model, vocab=_vocab, candidate_uris=_candidates[candidates], thinking=thinking,
    )
