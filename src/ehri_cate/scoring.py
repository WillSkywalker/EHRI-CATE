from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .vocab import Vocab


# A prediction: (uri, score). Higher score = more confident.
Prediction = tuple[str, float]


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f1: float


def _safe_f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _topk_uris(predictions: Iterable[Prediction], k: int) -> list[str]:
    ranked = sorted(predictions, key=lambda x: x[1], reverse=True)
    seen: list[str] = []
    for uri, _ in ranked:
        if uri not in seen:
            seen.append(uri)
        if len(seen) >= k:
            break
    return seen


# -- flat (set-equality) F1 ----------------------------------------------------


def flat_scores(gold: Iterable[str], predicted: Iterable[str]) -> Scores:
    g, p = set(gold), set(predicted)
    if not p and not g:
        return Scores(1.0, 1.0, 1.0)
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    return Scores(precision, recall, _safe_f1(precision, recall))


def flat_f1_at_k(gold: Iterable[str], predictions: Iterable[Prediction], k: int = 5) -> Scores:
    """Set-equality F1 between gold URIs and top-k predicted URIs. The paper's primary metric."""
    return flat_scores(gold, _topk_uris(predictions, k))


# -- hierarchical F1 (Kiritchenko-style) --------------------------------------


def _expand_with_ancestors(uris: Iterable[str], vocab: Vocab) -> set[str]:
    """{uri} ∪ ancestors(uri) for each uri that's in the vocab. Out-of-vocab uris are dropped."""
    expanded: set[str] = set()
    for uri in uris:
        if uri in vocab:
            expanded.add(uri)
            expanded.update(vocab.ancestors(uri))
    return expanded


def hierarchical_scores(gold: Iterable[str], predicted: Iterable[str], vocab: Vocab) -> Scores:
    """Kiritchenko et al. (2005) hierarchical P/R/F1 — expand each label with its ancestors,
    then compute set F1 over the expanded sets. Naturally gives partial credit for predicting
    a broader or narrower concept than the gold."""
    g_star = _expand_with_ancestors(gold, vocab)
    p_star = _expand_with_ancestors(predicted, vocab)
    if not p_star and not g_star:
        return Scores(1.0, 1.0, 1.0)
    tp = len(g_star & p_star)
    precision = tp / len(p_star) if p_star else 0.0
    recall = tp / len(g_star) if g_star else 0.0
    return Scores(precision, recall, _safe_f1(precision, recall))


def hierarchical_f1_at_k(
    gold: Iterable[str], predictions: Iterable[Prediction], vocab: Vocab, k: int = 5
) -> Scores:
    return hierarchical_scores(gold, _topk_uris(predictions, k), vocab)


# -- aggregation across documents ---------------------------------------------


@dataclass(frozen=True)
class AggregateScores:
    """The three averaging strategies used in the paper, plus the hierarchical variant."""

    doc_avg_f1: float           # mean of per-doc F1 — paper's primary
    micro_f1: float             # pooled TP/FP/FN across all docs
    weighted_macro_f1: float    # per-label F1 weighted by gold support
    hierarchical_doc_avg_f1: float  # the new metric (Kiritchenko hF1, doc-averaged)
    n_docs: int


def aggregate(
    per_doc_gold: list[set[str]],
    per_doc_topk: list[list[str]],
    vocab: Vocab,
) -> AggregateScores:
    assert len(per_doc_gold) == len(per_doc_topk)
    n = len(per_doc_gold)

    per_doc_flat = [flat_scores(g, p) for g, p in zip(per_doc_gold, per_doc_topk)]
    doc_avg = mean(s.f1 for s in per_doc_flat) if per_doc_flat else 0.0

    # micro: pool TP/FP/FN
    tp = fp = fn = 0
    for g, p in zip(per_doc_gold, per_doc_topk):
        gs, ps = set(g), set(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = _safe_f1(micro_p, micro_r)

    # weighted macro: per-label F1 weighted by support
    labels: set[str] = set()
    for g in per_doc_gold:
        labels.update(g)
    weighted_macro_f1 = 0.0
    total_support = 0
    for label in labels:
        lt = lp = lf = 0  # per-label TP, FP, FN
        for g, p in zip(per_doc_gold, per_doc_topk):
            in_g, in_p = label in g, label in p
            lt += int(in_g and in_p)
            lp += int(in_p and not in_g)
            lf += int(in_g and not in_p)
        support = lt + lf  # number of docs where this label is gold
        if support == 0:
            continue
        prec = lt / (lt + lp) if (lt + lp) else 0.0
        rec = lt / (lt + lf) if (lt + lf) else 0.0
        weighted_macro_f1 += _safe_f1(prec, rec) * support
        total_support += support
    weighted_macro_f1 = weighted_macro_f1 / total_support if total_support else 0.0

    hier = [hierarchical_scores(g, p, vocab) for g, p in zip(per_doc_gold, per_doc_topk)]
    hier_doc_avg = mean(s.f1 for s in hier) if hier else 0.0

    return AggregateScores(
        doc_avg_f1=doc_avg,
        micro_f1=micro_f1,
        weighted_macro_f1=weighted_macro_f1,
        hierarchical_doc_avg_f1=hier_doc_avg,
        n_docs=n,
    )
