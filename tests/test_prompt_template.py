"""Tests for the LLM prompt-template loader and rendering."""

from __future__ import annotations

import pytest

from ehri_cate.backends.llm4ssh import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    PromptTemplate,
    load_prompt_template,
    measure_messages,
)


def _write(tmp_path, body: str):
    p = tmp_path / "prompt.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_default_matches_shipped_constants():
    t = PromptTemplate()
    assert t.system == SYSTEM_PROMPT
    assert t.user == USER_TEMPLATE


def test_full_override(tmp_path):
    path = _write(tmp_path, 'system = "be terse"\nuser = "{text}\\n{candidates}"\n')
    t = load_prompt_template(path)
    assert t.system == "be terse"
    assert t.user == "{text}\n{candidates}"


def test_partial_override_keeps_default_for_missing_key(tmp_path):
    # Only `system` given → `user` falls back to the shipped default.
    path = _write(tmp_path, 'system = "custom system"\n')
    t = load_prompt_template(path)
    assert t.system == "custom system"
    assert t.user == USER_TEMPLATE


def test_missing_placeholder_raises(tmp_path):
    path = _write(tmp_path, 'user = "no placeholders here"\n')
    with pytest.raises(ValueError, match="placeholder"):
        load_prompt_template(path)


def test_unknown_key_raises(tmp_path):
    path = _write(tmp_path, 'systemm = "typo"\n')
    with pytest.raises(ValueError, match="unexpected key"):
        load_prompt_template(path)


def test_render_user_is_brace_safe():
    # A literal JSON example in the template must survive rendering — this is why
    # we use str.replace, not str.format.
    t = PromptTemplate(user='{text}\nexample: {"labels": [1, 2]}\n{candidates}')
    out = t.render_user("DOC", "1. a\n2. b")
    assert "DOC" in out
    assert '{"labels": [1, 2]}' in out
    assert "1. a\n2. b" in out


def test_example_template_loads():
    # The shipped example must stay valid.
    t = load_prompt_template("prompts/example.toml")
    assert "{text}" in t.user and "{candidates}" in t.user


def test_measure_messages_counts_chars_exactly():
    msgs = [
        {"role": "system", "content": "abcd"},
        {"role": "user", "content": "efghij"},
    ]
    size = measure_messages("Some-Unknown-Model", msgs)
    assert size.chars == 10  # exact, regardless of tokenizer
    assert size.tokens > 0
    assert size.token_method in ("litellm", "chars/4")
