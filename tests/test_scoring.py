"""Scoring tests on a toy 7-concept DAG.

       A
      / \
     B   C
    /|   |
   D E   F
       \
        G   (G has TWO broader: E and F — exercises the DAG case)
"""

from __future__ import annotations

import pytest

from ehri_cate.scoring import (
    flat_scores,
    flat_f1_at_k,
    hierarchical_scores,
    hierarchical_f1_at_k,
    aggregate,
)
from ehri_cate.vocab import Concept, Vocab


@pytest.fixture
def toy_vocab() -> Vocab:
    concepts = {
        c: Concept(uri=c, pref_labels={"en": c}) for c in "ABCDEFG"
    }
    broader = {
        "B": {"A"},
        "C": {"A"},
        "D": {"B"},
        "E": {"B"},
        "F": {"C"},
        "G": {"E", "F"},  # multi-parent (DAG)
    }
    return Vocab(concepts=concepts, broader=broader)


def test_flat_perfect_match():
    s = flat_scores({"A", "B"}, {"A", "B"})
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_flat_empty_prediction():
    s = flat_scores({"A"}, set())
    assert s.f1 == 0.0


def test_flat_no_overlap():
    s = flat_scores({"A"}, {"B"})
    assert s.f1 == 0.0


def test_flat_topk_dedupes_and_truncates():
    preds = [("A", 0.9), ("A", 0.8), ("B", 0.7), ("C", 0.6), ("D", 0.5), ("E", 0.4)]
    s = flat_f1_at_k({"A", "X"}, preds, k=3)
    # top-3 unique = {A, B, C}; gold = {A, X}; TP=1, FP=2, FN=1
    # precision=1/3, recall=1/2 -> f1 = 2*(1/3)*(1/2)/(1/3+1/2) = 0.4
    assert s.precision == pytest.approx(1 / 3)
    assert s.recall == pytest.approx(1 / 2)
    assert s.f1 == pytest.approx(0.4)


# -- Hierarchical -------------------------------------------------------------


def test_hier_exact_match(toy_vocab):
    s = hierarchical_scores({"D"}, {"D"}, toy_vocab)
    assert s.f1 == 1.0


def test_hier_predict_broader_gets_credit(toy_vocab):
    # Gold = G (deep). Predict B (an ancestor of G via E).
    # G* = {G, E, F, B, C, A}.  P* = {B, A}.
    # intersect = {B, A} (size 2). hP = 2/2 = 1.0, hR = 2/6 ≈ 0.333
    s = hierarchical_scores({"G"}, {"B"}, toy_vocab)
    assert s.precision == pytest.approx(1.0)
    assert s.recall == pytest.approx(2 / 6)
    assert 0 < s.f1 < 1


def test_hier_predict_narrower_gets_credit(toy_vocab):
    # Gold = B. Predict D (a child of B).
    # G* = {B, A}.  P* = {D, B, A}. intersect = {B, A}.
    # hP = 2/3, hR = 2/2 = 1.0
    s = hierarchical_scores({"B"}, {"D"}, toy_vocab)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(1.0)


def test_hier_unrelated_branch_low_score(toy_vocab):
    # Gold = D (under B), predicted = F (under C). Common ancestor = A.
    # G* = {D, B, A}. P* = {F, C, A}. intersect = {A}.
    # hP = 1/3, hR = 1/3, hF1 = 1/3
    s = hierarchical_scores({"D"}, {"F"}, toy_vocab)
    assert s.f1 == pytest.approx(1 / 3)


def test_hier_dag_multiparent(toy_vocab):
    # G has TWO ancestors-paths (E->B->A and F->C->A); both should be in G*.
    assert toy_vocab.ancestors("G") == {"E", "B", "A", "F", "C"}


def test_hier_oov_prediction_is_dropped(toy_vocab):
    # An out-of-vocab predicted URI gets no credit and isn't counted in P* size.
    # Otherwise an unrelated prediction would inflate the denominator.
    s = hierarchical_scores({"D"}, {"NOT_IN_VOCAB"}, toy_vocab)
    # P* = {} (OOV dropped), G* = {D, B, A}. With P* empty: precision=0, recall=0.
    assert s.precision == 0.0
    assert s.recall == 0.0


# -- Aggregation --------------------------------------------------------------


def test_aggregate_perfect_two_docs(toy_vocab):
    gold = [{"D"}, {"F"}]
    preds = [["D"], ["F"]]
    agg = aggregate(gold, preds, toy_vocab)
    assert agg.n_docs == 2
    assert agg.doc_avg_f1 == 1.0
    assert agg.micro_f1 == 1.0
    assert agg.weighted_macro_f1 == 1.0
    assert agg.hierarchical_doc_avg_f1 == 1.0


def test_aggregate_micro_vs_doc_average_diverge(toy_vocab):
    # Two docs, very different cardinalities: micro and doc-avg should diverge.
    # Doc1: gold {A, B, C, D}, predicted {A, B, C, D} -> per-doc F1 = 1.0
    # Doc2: gold {A}, predicted {B} -> per-doc F1 = 0.0
    gold = [{"A", "B", "C", "D"}, {"A"}]
    preds = [["A", "B", "C", "D"], ["B"]]
    agg = aggregate(gold, preds, toy_vocab)
    assert agg.doc_avg_f1 == pytest.approx(0.5)
    # micro: TP=4, FP=1, FN=1 -> P=R=F1 = 4/5
    assert agg.micro_f1 == pytest.approx(0.8)
    # hierarchical doc avg should beat flat doc avg, since B is close to A under our DAG
    assert agg.hierarchical_doc_avg_f1 > agg.doc_avg_f1
