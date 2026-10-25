"""The HTTP client for POST /v1/score: the spinf scoring API, hosted at api.spinf.com or a spinf-serve of your own."""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Self

import httpx

from . import packs as pk
from .moderation import PACK_ID, Moderation

DEFAULT_URL = "https://api.spinf.com"


class SpinfError(RuntimeError):
    """A rejected request (4xx other than 429): the API's error envelope."""

    def __init__(self, status: int, code: str, message: str, param: str | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.param = status, code, message, param


class SpinfClient:
    """Score and extract through the spinf API: hosted (api.spinf.com, a key) or a spinf-serve of your own (its URL;
    a key only when the server was started with --token).

        client = SpinfClient()                       # SPINF_API_KEY from the environment
        client = SpinfClient(base_url="http://localhost:8000", model="spinf/gemma-4-12b-fp8")   # your own server
        out = client.score(
            content=[text("Apple sold 3.5 million phones.")],
            queries=[query("units", "\\nQ: How many phones?\\nA:{n}"),
                     query("dir", "\\nDid sales go{?}", options=[" up", " down"])],
        )

    Retries 429 / 5xx / timeouts (Retry-After when given); any other 4xx raises SpinfError at once.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "spinf-12b",
        timeout_s: float = 120.0,
        tries: int = 6,
    ):
        key = api_key if api_key is not None else os.environ.get("SPINF_API_KEY")
        self.base_url = (base_url or os.environ.get("SPINF_API_URL") or DEFAULT_URL).rstrip("/")
        if not key and self.base_url == DEFAULT_URL:
            raise ValueError(
                "no API key: pass api_key= or set SPINF_API_KEY (a spinf-serve of your own started without --token needs"
                " none: pass base_url= or set SPINF_API_URL)"
            )
        self.model, self.tries = model, tries
        headers = {"User-Agent": "spinf-client"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10), headers=headers)

    def request(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST a raw /v1/score body; returns the decoded response."""
        return self._call("POST", "/v1/score", body)

    def models(self) -> dict[str, Any]:
        """GET /v1/models: the models, their other accepted names (aliases), status, price and minimum per call."""
        return self._call("GET", "/v1/models")

    def account(self) -> dict[str, Any]:
        """GET /v1/account (hosted API only): the key's account: balance, limits, models enabled."""
        return self._call("GET", "/v1/account")

    def pricing(self) -> dict[str, Any]:
        """GET /v1/account/pricing (hosted API only; a spinf-serve of your own bills nothing and has no such endpoint): the prices of what this account can use, only that (its models, the media it is
        cleared for, extraction if enabled, its content limit), in tokens. See billed_tokens()."""
        return self._call("GET", "/v1/account/pricing")

    def list_packs(self) -> dict[str, Any]:
        """GET /v1/packs: the hosted prompt-packs, their versions, contexts, categories (default thresholds) and decision
        modes."""
        return self._call("GET", "/v1/packs")

    def get_pack(self, id: str = PACK_ID, version: str | None = None) -> dict[str, Any]:
        """GET /v1/packs/{id}: the pack file (default: the latest version), to run or re-decide locally (packs)."""
        return self._call("GET", f"/v1/packs/{id}", params={"version": version} if version else None)

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        err = ""
        for k in range(self.tries):
            wait = min(60.0, 2.0**k) + random.random()
            try:
                r = self.http.request(method, f"{self.base_url}{path}", json=body, params=params)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429 or r.status_code >= 500:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.replace(".", "", 1).isdigit():
                        wait = min(120.0, float(ra)) + random.random()
                    err = f"{r.status_code} {r.text[:200]}"
                else:
                    e = (
                        r.json().get("error")
                        if r.headers.get("content-type", "").startswith("application/json")
                        else None
                    ) or {}
                    raise SpinfError(
                        r.status_code, e.get("code", "error"), e.get("message", r.text[:500]), e.get("param")
                    )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                err = f"{type(e).__name__}: {e}"
            if k < self.tries - 1:
                time.sleep(wait)
        raise RuntimeError(f"spinf: failed after {self.tries} attempts: {err}")

    def score(
        self,
        content: list[dict[str, Any]] | None = None,
        queries: list[dict[str, Any]] | None = None,
        inputs: list[dict[str, Any]] | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        packs: list[dict[str, Any]] | None = None,
        **scoring: Any,
    ) -> dict[str, Any]:
        """One content (`content`: a list of parts) or many (`inputs`: [{"id", "content" | "messages", "queries"?}]),
        scored with the shared `queries` and hosted `packs` (pack(): their categories and decision come back in
        `results[].packs`, see pack_result()). Extra keyword arguments go to `scoring` (empty_floor,
        case_insensitive)."""
        if (content is None) == (inputs is None):
            raise ValueError("pass exactly one of content= or inputs=")
        body: dict[str, Any] = {"model": model or self.model, "scoring": {"queries": queries or [], **scoring}}
        if packs:
            body["scoring"]["packs"] = packs
        if content is not None:
            body["messages"] = [{"role": "user", "content": content}]
        else:
            body["inputs"] = [
                {
                    **{k: v for k, v in i.items() if k != "content"},
                    "messages": i.get("messages") or [{"role": "user", "content": i["content"]}],
                }
                for i in inputs
            ]
        if endpoint:
            body["endpoint"] = endpoint
        return self.request(body)

    def moderate(
        self,
        texts: str | list[str],
        *,
        context: str | None = None,
        mode: str | None = None,
        threshold: float | None = None,
        thresholds: dict[str, float | None] | None = None,
        critical: list[str] | None = None,
        version: str | None = None,
        queries: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        include_queries: bool = False,
        pack_file: dict[str, Any] | None = None,
        batch_size: int = 64,
        endpoint: str | None = None,
    ) -> Moderation | list[Moderation]:
        """Moderate one text (-> Moderation) or many (-> a list, in order) with the spinf/moderation pack.

        context: "ai" (a message to an AI assistant, the default) or "forum" (a post on a platform). mode: max |
        per_category | micro_layer (the default); threshold: the cut of max (0-1) or micro_layer; thresholds:
        per-category overrides ({"profanity": None} = reported only); critical: the categories that make a text unsafe
        whenever flagged ([] = none); version: pin the pack version to keep scores comparable over time. queries: your
        own queries, scored in the same pass (Moderation.queries). include_queries: keep the pack's raw query results
        (Moderation.raw) to re-decide later without scoring again (Moderation.redecide()).

        pack_file: a pack file (get_pack(), possibly edited) instead of the hosted pack: its queries go out as ordinary
        queries and the decision is made here (an edited pack's micro-layer lock fails: the fallback mode decides)."""
        one = isinstance(texts, str)
        items = [texts] if one else list(texts)
        names = list(ids) if ids is not None else [str(i) for i in range(len(items))]
        if len(names) != len(items) or len(set(names)) != len(names):
            raise ValueError("ids must be unique, one per text")
        own = {q["id"] for q in queries or []}
        out: list[Moderation] = []
        for lo, hi in _batches(items, batch_size):
            batch = list(zip(names[lo:hi], items[lo:hi], strict=True))
            if pack_file is None:
                ref = pack(
                    PACK_ID,
                    version,
                    context=context,
                    mode=mode,
                    threshold=threshold,
                    thresholds=thresholds,
                    critical=critical,
                    include_queries=include_queries or None,
                )
                resp = self.score(
                    inputs=[{"id": i, "content": [text(t)]} for i, t in batch],
                    queries=queries,
                    packs=[ref],
                    endpoint=endpoint,
                )
            else:
                scoring = pk.request_scoring(pack_file)
                # the hosted pack's layout: its queries in their own group (own trie, starting on a block, after the
                # caller's), so the pack's answers do not depend on the caller's queries
                group = f"{pack_file['pack']}@{pack_file['version']}"
                pack_queries = [{**q, "empty_floor": False, "group": group} for q in scoring.pop("queries")]
                scoring.pop("empty_floor", None)
                resp = self.score(
                    inputs=[{"id": i, "content": pk.wrap(pack_file, t, context)} for i, t in batch],
                    queries=pack_queries + list(queries or []),
                    model=pack_file["model"],
                    endpoint=endpoint,
                    **scoring,
                )
            by_id = {r["input_id"]: r for r in resp["results"]}
            for i, t in batch:
                res = by_id[i]
                mine = [q for q in res["queries"] if q["id"] in own]
                raw = [q for q in res["queries"] if q["id"] not in own]
                if pack_file is None:
                    pr = pack_result({"results": [res]})
                else:
                    ev = pk.evaluate(
                        pack_file, pk.answers_from_result(raw, pack_file), mode, threshold, thresholds, critical
                    )
                    ctx = context or next(iter(pack_file["contexts"]))
                    pr = {"id": pack_file["pack"], "version": pack_file["version"], "context": ctx, **ev}
                out.append(Moderation.from_result(i, t, pr, mine, raw if include_queries else None))
        return out[0] if one else out

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------- builders
def text(s: str) -> dict[str, Any]:
    return {"type": "text", "text": s}


def query(
    id: str,
    template: str,
    options: list[str] | None = None,
    *,
    combinations: list[dict[str, str]] | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    calibration: tuple[float, float] | None = None,
    generate: bool | dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A query. `{?}` + options = scored; `{*}` (text, write it in quotes: '... is "{*}"'), `{n}` (number) or `{$}`
    (amount) = extraction: the model writes the answer, returned with its probability. Extraction needs the account
    feature (else 403 feature_not_enabled); at most 10 slots (queries x combinations) per content. `max_tokens` 1-64
    (default 16; up to 100 generated), `top_k` 1-20 answers (default 5); `calibration=(a, b)` from calibration.fit()
    adds p_calibrated to the top answer. `generate=True` (greedy) or `generate={"temperature": 0.7, "top_p": 0.8,
    "top_k": 20, "seed": 0, "n": 1}` turns a `{*}` slot into free text generation (a summary, a caption): the answers
    come back as {text, finish} without a probability, same gate, billed as a generated slot (2,000 tokens flat; a
    searched slot is 500); no top_k / calibration with it."""
    q: dict[str, Any] = {"id": id, "template": template, **extra}
    for k, v in (
        ("options", options),
        ("combinations", combinations),
        ("max_tokens", max_tokens),
        ("top_k", top_k),
        ("generate", generate),
    ):
        if v is not None:
            q[k] = v
    if calibration is not None:
        q["calibration"] = {"a": calibration[0], "b": calibration[1]}
    return q


def pack(
    id: str,
    version: str | None = None,
    *,
    context: str | None = None,
    mode: str | None = None,
    threshold: float | None = None,
    thresholds: dict[str, float | None] | None = None,
    critical: list[str] | None = None,
    include_queries: bool | None = None,
) -> dict[str, Any]:
    """A hosted prompt-pack for score(packs=[...]), scored in the same pass as your queries (GET /v1/packs lists them).
    version: default the latest (pin it to keep scores comparable); context: one of the pack's contexts (default the
    first); mode / threshold / thresholds / critical: the decision (default the pack's); include_queries: keep the
    pack's raw query results in results[].queries."""
    p: dict[str, Any] = {"id": id, **_drop_none(version=version, context=context, include_queries=include_queries)}
    decision = _drop_none(mode=mode, threshold=threshold, thresholds=thresholds, critical=critical)
    if decision:
        p["decision"] = decision
    return p


def pack_result(response: dict[str, Any], pack_id: str = PACK_ID, input_index: int = 0) -> dict[str, Any]:
    """A pack's result for one input ({id, version, context, categories, decision}) from a /v1/score response."""
    for p in response["results"][input_index].get("packs", []):
        if p["id"] == pack_id:
            return p
    raise KeyError(pack_id)


def _drop_none(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def _batches(texts: list[str], size: int, max_chars: int = 1_000_000) -> list[tuple[int, int]]:
    """[lo, hi) spans of at most `size` texts and about `max_chars` characters (the body limit is 2 MB)."""
    spans, lo, chars = [], 0, 0
    for i, t in enumerate(texts):
        if i > lo and (i - lo >= size or chars + len(t) > max_chars):
            spans.append((lo, i))
            lo, chars = i, 0
        chars += len(t)
    if texts:
        spans.append((lo, len(texts)))
    return spans


def billed_tokens(
    model_pricing: dict[str, Any],
    text_tokens: int = 0,
    images: dict[str, int] | None = None,
    frames: dict[str, int] | None = None,
    audio_seconds: float = 0.0,
    query_tokens: int = 0,
    floor_tokens: int = 0,
    slots_per_content: list[int] | None = None,
) -> int:
    """Hosted API only: estimate a call's billed tokens from one model's entry of pricing()["models"]: text (content text and video
    timestamps) + images and video frames at the model's rate per detail ({"low": n, ...}) + audio seconds (rounded up)
    + question and floor tokens + extraction slots (per content: the first at first_per_content_tokens, each next at
    next_per_content_tokens), at least min_billed_tokens_per_call. The response's usage.billed_tokens is the truth."""
    media = model_pricing.get("media") or {}
    per_image = (media.get("image") or {}).get("tokens_per_image") or {}
    per_frame = (media.get("video") or {}).get("tokens_per_frame") or {}
    per_second = (media.get("audio") or {}).get("tokens_per_second") or 0
    x = model_pricing.get("extraction") or {}
    total = text_tokens + query_tokens + floor_tokens
    total += sum(per_image[d] * n for d, n in (images or {}).items())
    total += sum(per_frame[d] * n for d, n in (frames or {}).items())
    total += math.ceil(audio_seconds * per_second)
    for n in slots_per_content or []:
        if n > 0:
            total += x["first_per_content_tokens"] + x["next_per_content_tokens"] * (n - 1)
    return max(total, model_pricing.get("min_billed_tokens_per_call", 0))


def answers(response: dict[str, Any], query_id: str, input_index: int = 0, combination: int = 0) -> dict[str, Any]:
    """An extraction slot's result ({values, answers, mass, p_negative?, p_calibrated?}) from a /v1/score response:
    answers best first, each {text, p} (+ value, and currency for {$})."""
    for q in response["results"][input_index]["queries"]:
        if q["id"] == query_id:
            return q["combinations"][combination]
    raise KeyError(query_id)
