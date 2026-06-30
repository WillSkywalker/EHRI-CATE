"""CLI tests for `cate models` and the default-model validation.

The proxy /v1/models fetch is mocked, so these run without network or data: the
default-validation path fails before any corpus/vocab is read, and we pass dummy
(but existing) --corpus/--vocab files only to satisfy click's exists= checks.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import ehri_cate.cli as cli
from ehri_cate.backends.llm4ssh import DEFAULT_MODEL
from ehri_cate.cli import main


def _dummy_data(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]")
    vocab = tmp_path / "vocab.ttl"
    vocab.write_text("")
    return str(corpus), str(vocab)


# -- cate models ---------------------------------------------------------------


def test_models_lists_and_marks_default(monkeypatch):
    monkeypatch.setattr(cli, "list_models_from_env",
                        lambda *a, **k: ["alpha", DEFAULT_MODEL, "beta"])
    res = CliRunner().invoke(main, ["models"])
    assert res.exit_code == 0, res.output
    lines = res.stdout.strip().splitlines()
    assert lines == ["alpha", f"{DEFAULT_MODEL}  (default)", "beta"]


def test_models_missing_api_key(monkeypatch):
    def boom(*a, **k):
        raise KeyError("LLM4SSH_API_KEY")
    monkeypatch.setattr(cli, "list_models_from_env", boom)
    res = CliRunner().invoke(main, ["models"])
    assert res.exit_code != 0
    assert "LLM4SSH_API_KEY" in res.stderr


def test_models_fetch_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(cli, "list_models_from_env", boom)
    res = CliRunner().invoke(main, ["models"])
    assert res.exit_code != 0
    assert "Could not fetch" in res.stderr


# -- default-model validation --------------------------------------------------


def test_evaluate_default_model_validation_error(tmp_path, monkeypatch):
    # DEFAULT_MODEL absent from the live list → clean error before any data load.
    monkeypatch.setattr(cli, "list_models_from_env", lambda *a, **k: ["something-else"])
    corpus, vocab = _dummy_data(tmp_path)
    res = CliRunner().invoke(main, ["evaluate", "--corpus", corpus, "--vocab", vocab])
    assert res.exit_code != 0
    assert "not available" in res.stderr
    assert "cate models" in res.stderr


def test_label_default_model_validation_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "list_models_from_env", lambda *a, **k: ["something-else"])
    corpus, vocab = _dummy_data(tmp_path)
    res = CliRunner().invoke(
        main, ["label", "--corpus", corpus, "--vocab", vocab], input="some archival text",
    )
    assert res.exit_code != 0
    assert "not available" in res.stderr
