"""Content moderation with the spinf/moderation pack: typed results (SpinfClient.moderate()) and thresholds from your
own data (standard library only).

Every category gets a calibrated 0-1 score, comparable across categories, and its percentile on normal traffic. A
category is flagged at or above its threshold (None = reported, never decides). The decision (safe / unsafe) comes from
one of three modes:

  micro_layer   (default) a small linear layer over the category scores and the general reads, trained on labelled
                data; `threshold` moves its cut (any number; the default is the pack's)
  max           unsafe when the highest deciding category score >= `threshold` (0-1)
  per_category  unsafe when any category is flagged; `thresholds` sets them per category

`thresholds` also apply in the other modes: they decide `flagged`, and a flagged category listed in `critical` (the
pack's default: self-harm and child safety) makes the content unsafe whatever the mode says.

Tuning without retraining: score a sample of your own traffic, then
  rate_thresholds(results, rate=0.005)      per category, the score that flags `rate` of it (no labels needed)
  fit_thresholds(results, labels)           per category, the best F1 on texts you labelled with categories
  best_threshold(scores, unsafe)            one cut: the max score or the micro-layer's score against safe / unsafe
and pass them back to moderate().
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from . import packs

PACK_ID = "spinf/moderation"


@dataclass(frozen=True)
class Category:
    score: float  # calibrated 0-1, comparable across categories
    percentile: float | None  # where the score falls on normal traffic, 0-1
    flagged: bool | None  # score >= the category's threshold; None = no threshold (reported only)


@dataclass
class Moderation:
    """One text's moderation result."""

    id: str
    text: str
    unsafe: bool
    flagged: list[str]  # the categories at or above their threshold, highest score first
    categories: dict[str, Category]
    decision: dict[str, Any]  # as the API returns it: mode, score / threshold or categories, critical, by, warning
    pack: str = PACK_ID
    version: str = ""
    context: str = ""
    queries: list[dict[str, Any]] = field(default_factory=list)  # your own queries' results, from the same pass
    raw: list[dict[str, Any]] | None = None  # the pack's raw query results (include_queries=True): see redecide()

    @property
    def mode(self) -> str:
        return self.decision["mode"]

    @property
    def score(self) -> float | None:
        """The decision's score (max: the highest category score; micro_layer: the layer's value); None in
        per_category."""
        return self.decision.get("score")

    @property
    def critical(self) -> list[str]:
        return self.decision.get("critical", [])

    def redecide(
        self,
        pack_file: dict[str, Any],
        mode: str | None = None,
        threshold: float | None = None,
        thresholds: dict[str, float | None] | None = None,
        critical: list[str] | None = None,
    ) -> Moderation:
        """The same text decided again from its raw answers (scored with include_queries=True), with another mode,
        thresholds or critical list; nothing is scored. pack_file: SpinfClient.get_pack() of this result's version."""
        if self.raw is None:
            raise ValueError("no raw answers: score with include_queries=True")
        if pack_file["version"] != self.version:
            raise ValueError(f"this result is pack version {self.version}, the file is {pack_file['version']}")
        out = packs.evaluate(
            pack_file, packs.answers_from_result(self.raw, pack_file), mode, threshold, thresholds, critical
        )
        pr = {"id": self.pack, "version": self.version, "context": self.context, **out}
        return Moderation.from_result(self.id, self.text, pr, self.queries, self.raw)

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        """The n highest category scores."""
        return sorted(((c, v.score) for c, v in self.categories.items()), key=lambda x: -x[1])[:n]

    @classmethod
    def from_result(
        cls,
        id: str,
        text: str,
        pack_result: dict[str, Any],
        queries: list[dict[str, Any]] | None = None,
        raw: list[dict[str, Any]] | None = None,
    ) -> Moderation:
        """From one `results[].packs[]` entry (or packs.evaluate()'s output with id / version / context added)."""
        cats = {c: Category(v["score"], v["percentile"], v["flagged"]) for c, v in pack_result["categories"].items()}
        dec = pack_result["decision"]
        return cls(
            id=id,
            text=text,
            unsafe=bool(dec["unsafe"]),
            flagged=sorted((c for c, v in cats.items() if v.flagged), key=lambda c: -cats[c].score),
            categories=cats,
            decision=dec,
            pack=pack_result.get("id", PACK_ID),
            version=pack_result.get("version", ""),
            context=pack_result.get("context", ""),
            queries=queries or [],
            raw=raw,
        )


# ---------------------------------------------------------------- thresholds from your data
def best_threshold(scores: list[float], labels: list[bool], beta: float = 1.0) -> float:
    """The cut with the best F-beta when `score >= cut` predicts `label` (beta > 1 favours recall, < 1 precision)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    positives = sum(map(bool, labels))
    if not positives or positives == len(labels):
        raise ValueError("need both positive and negative labels")
    b2 = beta * beta
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    best, cut = -1.0, pairs[0][0]
    for i, (s, y) in enumerate(pairs):
        tp, fp = tp + bool(y), fp + (not y)
        if i + 1 < len(pairs) and pairs[i + 1][0] == s:  # ties flag together
            continue
        f = (1 + b2) * tp / ((1 + b2) * tp + b2 * (positives - tp) + fp)
        if f > best:
            best, cut = f, s
    return cut


def fit_thresholds(
    results: list[Moderation], labels: list[Iterable[str]], beta: float = 1.0, min_positives: int = 5
) -> dict[str, float]:
    """Per category, the threshold with the best F-beta on your labelled texts: labels[i] = the categories text i
    belongs to (empty = acceptable). Categories with fewer than `min_positives` labelled texts are left out (they keep
    the pack's threshold)."""
    if len(results) != len(labels):
        raise ValueError("results and labels differ in length")
    sets = [set(x) for x in labels]
    out = {}
    for c in results[0].categories:
        ys = [c in s for s in sets]
        if min_positives <= sum(ys) < len(ys):
            out[c] = round(best_threshold([r.categories[c].score for r in results], ys, beta), 6)
    return out


def rate_thresholds(
    results: list[Moderation], rate: float = 0.005, categories: Iterable[str] | None = None
) -> dict[str, float]:
    """Per category, the score that `rate` of these texts reach (the pack's defaults flag 0.5% of normal traffic).
    Score a sample of your normal traffic, a few thousand texts for rates under 1%; no labels needed."""
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1")
    out = {}
    for c in categories or results[0].categories:
        xs = sorted(r.categories[c].score for r in results)
        pos = (1 - rate) * (len(xs) - 1)  # linear interpolation between order statistics (numpy's default quantile)
        lo = math.floor(pos)
        hi = min(lo + 1, len(xs) - 1)
        out[c] = round(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo), 6)
    return out
