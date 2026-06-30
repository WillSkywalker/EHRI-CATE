"""CLI test for `cate label` (ad-hoc stdin labelling).

Needs the local vocab TTL (gitignored) to build the candidate menu, so it is
skipped when absent. The network call is stubbed — we test the CLI plumbing
(stdin → suggestions → output), not the LLM itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ehri_cate.backends.llm4ssh import LLM4SSHBackend
from ehri_cate.cli import main

VOCAB = Path("data/ehri_terms.ttl")

pytestmark = pytest.mark.skipif(
    not VOCAB.exists(), reason="local vocab TTL not present"
)


def _stub_suggest(monkeypatch):
    # Return the first two candidate URIs so output is deterministic, no network.
    monkeypatch.setattr(
        LLM4SSHBackend, "suggest",
        lambda self, text: [(self.candidate_uris[0], 1.0), (self.candidate_uris[1], 0.83)],
    )


def test_label_stdin_table(monkeypatch):
    monkeypatch.setenv("LLM4SSH_API_KEY", "test-key")
    _stub_suggest(monkeypatch)
    res = CliRunner().invoke(
        main, ["label", "--model", "X", "--candidates", "full", "-k", "2"],
        input="a holocaust-related archival description",
    )
    assert res.exit_code == 0, res.output
    assert "[debug]" in res.stderr           # prompt-size report emitted to stderr
    assert "full prompt ~" in res.stderr
    assert res.output.count("\t") >= 2       # two tab-separated suggestion rows on stdout


def test_label_json(monkeypatch):
    monkeypatch.setenv("LLM4SSH_API_KEY", "test-key")
    _stub_suggest(monkeypatch)
    res = CliRunner().invoke(
        main, ["label", "--model", "X", "--candidates", "full", "--json", "-k", "2"],
        input="some text",
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)         # stdout is pure JSON; debug is on stderr
    assert len(payload) == 2
    assert payload[0]["rank"] == 1
    assert payload[0]["uri"] and payload[0]["label"]


def test_label_empty_input_errors(monkeypatch):
    monkeypatch.setenv("LLM4SSH_API_KEY", "test-key")
    res = CliRunner().invoke(
        main, ["label", "--model", "X", "--candidates", "full"], input="   \n",
    )
    assert res.exit_code != 0
    assert "No input text" in res.stderr
