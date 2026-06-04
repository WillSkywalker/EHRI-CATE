"""Deterministic multi-label train/test split.

The supervised Annif baselines (unlike the zero-shot LLMs) need training data, so
we hold out a test set and train baselines on the rest. To stay comparable with
the prior paper (Dermentzi et al. 2025), we use iterative multi-label
stratification (Sechidis et al. 2011) via the `iterative-stratification` package,
which keeps each label's positive rate roughly equal across the two folds — important
for a long-tailed label distribution where a plain random split would leave many
labels entirely on one side.

Both LLM and Annif backends are then evaluated on the SAME held-out test docs, so
the comparison is apples-to-apples even though only the baselines saw the train split.
"""

from __future__ import annotations

from .corpus import LabelledDoc


def stratified_split(
    docs: list[LabelledDoc],
    test_size: float = 0.30,
    seed: int = 0,
) -> tuple[list[LabelledDoc], list[LabelledDoc]]:
    """Split docs into (train, test) by iterative multi-label stratification.

    Deterministic for a fixed ``seed``. The split is disjoint and covers every doc
    exactly once. ``test_size`` is the target test fraction (default 0.30, matching
    the paper's 70/30 split).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(docs) < 2:
        raise ValueError("need at least 2 docs to split")

    # Lazy import: iterative-stratification (and numpy) live in the optional
    # `baselines` dependency group, so the core LLM harness doesn't require them.
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    # Binary label matrix: rows = docs, cols = sorted union of all gold URIs.
    all_labels = sorted({uri for d in docs for uri in d.gold_uris})
    label_index = {uri: i for i, uri in enumerate(all_labels)}
    y = np.zeros((len(docs), len(all_labels)), dtype=np.int8)
    for row, doc in enumerate(docs):
        for uri in doc.gold_uris:
            y[row, label_index[uri]] = 1

    # X is unused by the stratifier (it splits on y); a column of row indices suffices.
    x = np.arange(len(docs)).reshape(-1, 1)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(x, y))

    train = [docs[i] for i in train_idx]
    test = [docs[i] for i in test_idx]
    return train, test
