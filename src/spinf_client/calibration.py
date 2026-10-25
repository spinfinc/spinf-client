"""Calibrating an extraction slot's probability on your own labelled examples.

The raw probability of an extracted answer ranks right answers above wrong ones, but it is under-confident by an amount
that depends on the field (part of the probability goes to other spellings and formats of the same answer). A two-number
correction fitted on ~50 labelled examples of the field makes it calibrated (spinf-serve's `docs/calibration.md`):

    P(correct) = sigmoid(a * logit(p) + b)

    a, b = fit(ps, rights)                       # ps: top-answer p per example; rights: was the top answer right?
    query("amount", "\\nQ: ...\\nA:{$}", calibration=(a, b))   # the API then returns p_calibrated

With 50 labels the calibration error was 0.02-0.10 on every task tested. Without labels, DEFAULT_MAPS (fitted pooled
over tasks) is right on average but can be off by ~0.2 on one field, and over-corrects answers the model gives from
memory rather than from the content.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# label-free fallback maps (a, b), pooled over brand / breed / SQuAD / NQ (research, 2026-09-28): the correction depends
# on the task, hardly on the model
DEFAULT_MAPS = {"spinf-12b": (0.76, 1.42), "spinf-31b": (0.71, 1.34)}


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def apply(p: float, a: float, b: float) -> float:
    return _sigmoid(a * _logit(p) + b)


def fit(
    ps: Sequence[float], rights: Sequence[bool], l2: float = 1e-3, iters: int = 5000, lr: float = 0.2
) -> tuple[float, float]:
    """Logistic regression of rightness on logit(p): -> (a, b). A small pull toward a = 1 keeps tiny samples sane."""
    if len(ps) != len(rights) or not ps:
        raise ValueError("ps and rights must be non-empty and the same length")
    x = [_logit(p) for p in ps]
    y = [1.0 if r else 0.0 for r in rights]
    n = len(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for xi, yi in zip(x, y):
            d = _sigmoid(a * xi + b) - yi
            ga += d * xi
            gb += d
        a -= lr * (ga / n + l2 * (a - 1.0))
        b -= lr * gb / n
    return a, b


def ece(ps: Sequence[float], rights: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error: the mean |confidence - accuracy| over equal-width bins, weighted by bin size."""
    total = 0.0
    for k in range(bins):
        idx = [i for i, p in enumerate(ps) if min(int(p * bins), bins - 1) == k]
        if idx:
            conf = sum(ps[i] for i in idx) / len(idx)
            acc = sum(1.0 if rights[i] else 0.0 for i in idx) / len(idx)
            total += abs(conf - acc) * len(idx) / len(ps)
    return total
