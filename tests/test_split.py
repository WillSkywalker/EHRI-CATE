"""Tests for the stratified train/test split.

No API calls and no data files — synthetic docs only. Requires the optional
`iterative-stratification` dep (the `baselines` group), so the module is skipped
if that isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("iterstrat")

from ehri_cate.corpus import LabelledDoc
from ehri_cate.split import stratified_split


def _make_docs(n: int = 240) -> list[LabelledDoc]:
    """n docs over 5 labels, each doc carrying 1–2 labels in a repeating pattern so
    every label has plenty of positives for the stratifier."""
    labels = [f"http://x/term/{i}" for i in range(5)]
    docs: list[LabelledDoc] = []
    for i in range(n):
        gold = (labels[i % 5], labels[(i + 1) % 5]) if i % 2 == 0 else (labels[i % 5],)
        docs.append(
            LabelledDoc(
                doc_id=f"doc-{i}",
                repo_id="repo",
                repo_name="Repo",
                language_code="en",
                text=f"document number {i}",
                gold_uris=gold,
            )
        )
    return docs


def test_deterministic_for_fixed_seed():
    docs = _make_docs()
    tr1, te1 = stratified_split(docs, test_size=0.3, seed=0)
    tr2, te2 = stratified_split(docs, test_size=0.3, seed=0)
    assert [d.doc_id for d in tr1] == [d.doc_id for d in tr2]
    assert [d.doc_id for d in te1] == [d.doc_id for d in te2]


def test_disjoint_and_complete():
    docs = _make_docs()
    train, test = stratified_split(docs, test_size=0.3, seed=0)
    train_ids = {d.doc_id for d in train}
    test_ids = {d.doc_id for d in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {d.doc_id for d in docs}
    assert len(train) + len(test) == len(docs)


def test_test_fraction_is_approximately_requested():
    docs = _make_docs(240)
    _, test = stratified_split(docs, test_size=0.3, seed=0)
    assert abs(len(test) / len(docs) - 0.3) < 0.05


def test_all_labels_present_across_both_folds():
    docs = _make_docs()
    train, test = stratified_split(docs, test_size=0.3, seed=0)
    all_labels = {u for d in docs for u in d.gold_uris}
    for fold in (train, test):
        fold_labels = {u for d in fold for u in d.gold_uris}
        assert fold_labels == all_labels  # stratification keeps every label on both sides


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_test_size(bad):
    with pytest.raises(ValueError):
        stratified_split(_make_docs(10), test_size=bad)


def test_rejects_too_few_docs():
    with pytest.raises(ValueError):
        stratified_split(_make_docs(1))
