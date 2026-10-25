"""Prompt packs on this side of the API (standard library only).

The API runs a hosted pack itself (`scoring.packs`: categories and decision come back in `results[].packs`); nothing
here is needed for that. This module re-derives the same result from a pack file (`SpinfClient.get_pack()`) and the
pack's raw query results, for three things:

- re-deciding stored results without scoring again: another mode, thresholds or critical list, from results scored
  with `include_queries` (`Moderation.redecide()`, or `answers_from_result()` + `evaluate()`);
- running an edited copy of a pack: its queries go out as ordinary queries (`request_scoring()`, `wrap()`), the
  decision is made here; an edited pack fails the micro-layer lock and decides with its fallback mode;
- checking the API.

The evaluator itself (`_pipeline`) is spinf-serve's `spinf_serve/packs/pipeline.py`, copied verbatim: the one evaluator
the server, the hosted API and this client run (a test checks the copy against a spinf-serve checkout). It reproduces
the packs' golden fixtures (tests/fixtures). What it computes per content:

  answers     the option probabilities per (query id, combination value F), renormalised over the query's options;
              from the API's p or, when the pack says "answers": "logprob" (0.1.3+), from the full-precision logprobs
  reads       pair: sign * logit(p[0]); option: ln p[j] - mu; any: logsumexp(a[1:]) - a[0];
              set: logsumexp(a[unsafe]) - logsumexp(a[rest]), with a = ln p - mu; then z = (value - mean) / sd
  categories  macro = the mean z of the category's reads ("views": the mean of the per-view means);
              score = sigmoid(a * macro + b); percentile = the score in the normal-traffic quantiles; flagged = score >=
              threshold (threshold None = reported only); a category with `requires` is flagged only while the
              categories it lists are
  decision    max: the highest score of the deciding categories >= one threshold; per_category: any deciding category
              flagged; micro_layer: w . x + b >= threshold over category macros ("c:") and read z's ("r:"), only while
              the lock matches, else the fallback mode; a deciding category has a threshold and not `decides: false`;
              then, in every mode, a flagged category listed in decision.critical forces unsafe (not part of the lock:
              editable like the thresholds)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import _pipeline
from ._pipeline import (  # noqa: F401  (the evaluator's names, re-exported)
    CLIP,
    LOCK_WARNING,
    MODES,
    PAIR_CLIP,
    Answers,
    category_macros,
    interp,
    lock_of,
    micro_layer,
    percentile,
    read_value,
    read_z,
)

X = "{X}"


def load(path: str | Path) -> dict[str, Any]:
    """A pack file saved from SpinfClient.get_pack()."""
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- running a pack file as ordinary queries
def parameter_values(pack: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, str]:
    """The values substituted into a context for a pack with `parameters` (the brand pack: {"brand": "Kestrel"}): the
    caller's, a parameter's default, "" for an optional one. A missing required parameter or an unknown name is a
    ValueError."""
    given = dict(params or {})
    out: dict[str, str] = {}
    for name, spec in (pack.get("parameters") or {}).items():
        v = given.pop(name, spec.get("default"))
        if v is None:
            if spec.get("required", True):
                raise ValueError(f"pack parameter {name!r} is required")
            v = ""
        out[name] = str(v)
    if given:
        raise ValueError(f"{pack['pack']} has no parameter {min(given)!r}")
    return out


def wrap(
    pack: dict[str, Any], text: str, context: str | None = None, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """A text in one of the pack's contexts (default the first), as content parts: the context's text before the
    content and after it, the pack's parameters filled in, as the API wraps an input for a hosted pack."""
    name = context or next(iter(pack["contexts"]))
    if name not in pack["contexts"]:
        raise ValueError(f"{pack['pack']} has no context {name!r}; contexts: {', '.join(pack['contexts'])}")
    before, sep, after = pack["contexts"][name].partition(X)
    if not sep:
        raise ValueError(f"context {name!r} has no {X}")
    for k, v in parameter_values(pack, params).items():
        before, after = before.replace("{" + k + "}", v), after.replace("{" + k + "}", v)
    return [{"type": "text", "text": s} for s in (before, text, after) if s]


def request_scoring(pack: dict[str, Any]) -> dict[str, Any]:
    """The pack's `scoring` object for a local run: its queries in their own group (`<pack>@<version>`, as the API
    runs a hosted pack, so the answers do not depend on the call's other queries), the pack's floor and case settings.
    Send it with the inputs wrapped by wrap(), then evaluate() each result."""
    sc = pack["scoring"]
    group = f"{pack['pack']}@{pack['version']}"
    floor = sc.get("empty_floor", True)
    return {
        "queries": [{"empty_floor": floor, **q, "group": group} for q in sc["queries"]],
        "case_insensitive": bool(sc.get("case_insensitive", False)),
    }


def answers_from_result(result: dict[str, Any] | list[dict[str, Any]], pack: dict[str, Any]) -> Answers:
    """One input's result of /v1/score (or its `queries` list) -> {(query id, F value or None): [p per option]} for the
    pack's queries (the caller's own queries can share the result). Where the probabilities come from is the pack's
    `answers`: "p" (up to 0.1.2: the API's p, rounded to 6 decimals) or "logprob" (0.1.3+: exp(logprob - logsumexp over
    the query's options), full precision: a rare option's p is not stuck on the 1e-6 grid)."""
    ids = {q["id"] for q in pack["scoring"]["queries"]}
    return _pipeline.answers_from_result(result, ids, pack.get("answers", "p"))


# ---------------------------------------------------------------- reads, categories, decision
def evaluate(
    pack: dict[str, Any],
    answers: Answers,
    mode: str | None = None,
    threshold: float | None = None,
    thresholds: dict[str, float | None] | None = None,
    critical: list[str] | None = None,
) -> dict[str, Any]:
    """One content's answers -> {"categories", "decision"}, as the API returns them in `results[].packs[]` (before
    the API's rounding to 6 decimals).

    mode: max | per_category | micro_layer (default: the pack's). threshold: the one threshold of max or micro_layer
    (default: the pack's). thresholds: per-category overrides (None = reported only). critical: the categories that
    force unsafe when flagged (default: the pack's decision.critical; [] = none)."""
    out = _pipeline.evaluate(pack, answers, mode, threshold, thresholds, critical)
    return {"categories": out["categories"], "decision": out["decision"]}
