"""Integration test for the Annif backend wrapper.

This actually trains a (small) TF-IDF model, so it needs the optional `baselines`
deps AND the local data files (corpus + vocab TTL), both gitignored. It is skipped
automatically when either is absent, so the default `uv run pytest` stays fast and
API/data-free in environments that don't have them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("annif")

CORPUS = Path("data/manual-descriptions-labelled-with-ehri-terms.json")
VOCAB = Path("data/ehri_terms.ttl")

pytestmark = pytest.mark.skipif(
    not (CORPUS.exists() and VOCAB.exists()),
    reason="local corpus/vocab data files not present",
)


@pytest.fixture(scope="module")
def trained_tfidf(tmp_path_factory):
    from ehri_cate.backends.annif_backend import AnnifWorkspace
    from ehri_cate.corpus import load_corpus
    from ehri_cate.split import stratified_split

    docs = load_corpus(CORPUS)
    train, test = stratified_split(docs, test_size=0.3, seed=0)
    ws = AnnifWorkspace(tmp_path_factory.mktemp("annif-ws"), VOCAB)
    ws.train(["tfidf"], train[:120])  # small + fast; just exercising the pipeline
    return ws, test


def test_train_marks_model_present(trained_tfidf):
    ws, _ = trained_tfidf
    assert ws.is_trained("tfidf")


def test_suggest_returns_uri_score_pairs(trained_tfidf):
    ws, test = trained_tfidf
    backend = ws.backend("tfidf")
    preds = backend.suggest(test[0].text)
    assert isinstance(preds, list)
    assert preds, "expected at least one suggestion for a non-empty doc"
    uri, score = preds[0]
    assert uri.startswith("http")
    assert isinstance(score, float)
    # ranked descending by score
    assert all(preds[i][1] >= preds[i + 1][1] for i in range(len(preds) - 1))


def test_empty_text_yields_no_suggestions(trained_tfidf):
    ws, _ = trained_tfidf
    backend = ws.backend("tfidf")
    assert backend.suggest("   ") == []
