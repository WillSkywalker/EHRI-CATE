"""LLM4SSH backend request/response handling for reasoning models: truncation
surfaces as an error (not a silent empty prediction), and --no-thinking reaches
the request. litellm.completion is stubbed — no network, and a tiny fake vocab
stands in for the TTL."""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest

from ehri_cate.backends.llm4ssh import LLM4SSHBackend, TruncatedResponseError

URIS = ["u:1", "u:2", "u:3"]


class _FakeVocab:
    def candidate_labels(self, lang):
        return {u: f"label {u}" for u in URIS}


def _backend():
    return LLM4SSHBackend(
        model="Qwen3.6-27B", vocab=_FakeVocab(), candidate_uris=URIS,
        api_base="http://x", api_key="k",
    )


def _stub_completion(monkeypatch, content, finish_reason):
    resp = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])
    monkeypatch.setattr(litellm, "completion", lambda **kw: resp)


def test_empty_truncated_reply_raises(monkeypatch):
    _stub_completion(monkeypatch, None, "length")
    with pytest.raises(TruncatedResponseError):
        _backend().suggest("some text")


def test_partial_truncated_reply_still_parsed(monkeypatch):
    _stub_completion(monkeypatch, '{"labels": [2, 1', "length")
    assert [u for u, _ in _backend().suggest("some text")] == ["u:2", "u:1"]


def test_fenced_json_reply_parsed(monkeypatch):
    # GLM-5.1 / MiniMax-M2.5 wrap the answer in a ```json fence.
    _stub_completion(monkeypatch, '```json\n{"labels": [3]}\n```', "stop")
    assert [u for u, _ in _backend().suggest("some text")] == ["u:3"]


def _capture_completion(monkeypatch):
    calls = []
    resp = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"labels": [1]}'), finish_reason="stop",
    )])
    monkeypatch.setattr(litellm, "completion", lambda **kw: calls.append(kw) or resp)
    return calls


def test_thinking_on_sends_no_extra_body(monkeypatch):
    calls = _capture_completion(monkeypatch)
    _backend().suggest("some text")
    assert "extra_body" not in calls[0]


def test_no_thinking_disables_via_chat_template_kwargs(monkeypatch):
    calls = _capture_completion(monkeypatch)
    be = _backend()
    be.thinking = False
    be.suggest("some text")
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
