"""Web API tests (Flask test client). litellm.completion and the /v1/models fetch
are stubbed — no network; the vocab comes from the committed resources/ snapshot."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

import ehri_cate.app as api
from ehri_cate.backends.llm4ssh import DEFAULT_MODEL

CORPUS = Path("data/manual-descriptions-labelled-with-ehri-terms.json")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM4SSH_API_KEY", "test-key")
    api._backend.cache_clear()
    yield api.app.test_client()
    api._backend.cache_clear()


def _stub_completion(monkeypatch, content='{"labels": [2, 1]}', finish_reason="stop"):
    calls = []
    resp = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])
    monkeypatch.setattr(litellm, "completion", lambda **kw: calls.append(kw) or resp)
    return calls


# -- / and /models ---------------------------------------------------------------


def test_root_describes_service(client):
    body = client.get("/").get_json()
    assert body["default_model"] == DEFAULT_MODEL
    assert body["candidates"]["in-use"] == 550


def test_models_lists_and_flags_default(client, monkeypatch):
    monkeypatch.setattr(api, "list_models_from_env", lambda: ["alpha", DEFAULT_MODEL])
    res = client.get("/models")
    assert res.status_code == 200
    assert res.get_json()["models"] == [
        {"id": "alpha", "default": False}, {"id": DEFAULT_MODEL, "default": True},
    ]


def test_models_without_api_key_is_503(client, monkeypatch):
    def boom():
        raise KeyError("LLM4SSH_API_KEY")
    monkeypatch.setattr(api, "list_models_from_env", boom)
    res = client.get("/models")
    assert res.status_code == 503
    assert "LLM4SSH_API_KEY" in res.get_json()["detail"]


# -- /suggest ----------------------------------------------------------------------


def test_suggest_returns_ranked_terms(client, monkeypatch):
    calls = _stub_completion(monkeypatch)
    res = client.post("/suggest", json={"text": "Letters from Theresienstadt ghetto"})
    assert res.status_code == 200
    body = res.get_json()
    assert (body["model"], body["candidates"], body["thinking"]) == (DEFAULT_MODEL, "in-use", True)
    in_use = api._candidates["in-use"]
    assert [s["uri"] for s in body["suggestions"]] == [in_use[1], in_use[0]]
    assert [s["rank"] for s in body["suggestions"]] == [1, 2]
    assert all(s["label"] for s in body["suggestions"])
    assert calls[0]["model"] == f"litellm_proxy/{DEFAULT_MODEL}"
    assert "extra_body" not in calls[0]


def test_suggest_top_k_truncates(client, monkeypatch):
    _stub_completion(monkeypatch, '{"labels": [1, 2, 3]}')
    res = client.post("/suggest", json={"text": "x", "top_k": 2})
    assert len(res.get_json()["suggestions"]) == 2


def test_suggest_no_thinking_reaches_request(client, monkeypatch):
    calls = _stub_completion(monkeypatch)
    client.post("/suggest", json={"text": "x", "model": "Qwen3.6-27B", "thinking": False})
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("body", [
    None,
    {},
    {"text": "   "},
    {"text": "x" * (api.MAX_TEXT_CHARS + 1)},
    {"text": "x", "top_k": 0},
    {"text": "x", "top_k": True},
    {"text": "x", "candidates": "everything"},
    {"text": "x", "thinking": "no"},
    {"text": "x", "model": ""},
])
def test_suggest_rejects_bad_input(client, monkeypatch, body):
    _stub_completion(monkeypatch)
    res = client.post("/suggest", json=body) if body is not None else client.post(
        "/suggest", data="not json", content_type="text/plain")
    assert res.status_code == 400
    assert res.get_json()["detail"]


def test_suggest_truncated_model_reply_is_502(client, monkeypatch):
    _stub_completion(monkeypatch, content=None, finish_reason="length")
    res = client.post("/suggest", json={"text": "x"})
    assert res.status_code == 502
    assert "max_tokens" in res.get_json()["detail"]


def test_suggest_without_api_key_is_503(client, monkeypatch):
    monkeypatch.delenv("LLM4SSH_API_KEY")
    res = client.post("/suggest", json={"text": "x"})
    assert res.status_code == 503


# -- snapshot parity -------------------------------------------------------------


@pytest.mark.skipif(not CORPUS.exists(), reason="local corpus not present")
def test_in_use_snapshot_matches_corpus():
    # Same list, same order as the CLI derives, so the API's numbered menu is
    # the evaluated prompt.
    from ehri_cate.corpus import load_corpus
    expected = sorted({uri for d in load_corpus(CORPUS) for uri in d.gold_uris})
    assert api._candidates["in-use"] == expected
