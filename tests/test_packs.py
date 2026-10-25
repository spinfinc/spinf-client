"""Prompt packs and moderation: the local evaluator against the golden fixture (real /v1/score results of 8 texts and
what the API returned for them, in the three modes: moderation 0.1.1, and 0.1.1 with critical = self-harm and child as
0.1.2 shipped it), moderate() against a mock API, the logprob answers of 0.1.3+ and the threshold helpers.

The pack files are not in this repo: the evaluator tests read them from a spinf-prompts checkout (SPINF_PROMPTS_DIR or
../spinf-prompts), from SPINF_PACK_DIR (spinf-moderation-<version>.json) or download them with SPINF_API_KEY into
tests/.packs, else they are skipped."""

import copy
import json
import os
from pathlib import Path

import httpx
import pytest

from spinf_client import Moderation, SpinfClient, moderation, pack, pack_result, packs, query
from spinf_client.client import _batches

HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "fixtures" / "moderation.json").read_text())
CRITICAL = ["selfharm", "child"]
TOL = 2e-6  # the fixture is rounded to 6 decimals


def _pack_file(version: str) -> dict:
    """The spinf/moderation pack file of `version`: from a spinf-prompts checkout (SPINF_PROMPTS_DIR or ../spinf-prompts),
    else from SPINF_PACK_DIR (spinf-moderation-<version>.json), else downloaded once with SPINF_API_KEY into tests/.packs."""
    prompts = Path(os.environ.get("SPINF_PROMPTS_DIR") or HERE.parent.parent / "spinf-prompts")
    if (prompts / "packs" / "moderation" / f"{version}.json").exists():
        return packs.load(prompts / "packs" / "moderation" / f"{version}.json")
    name = f"spinf-moderation-{version}.json"
    if os.environ.get("SPINF_PACK_DIR"):
        return packs.load(Path(os.environ["SPINF_PACK_DIR"]) / name)
    cached = HERE / ".packs" / name
    if not cached.exists():
        if not os.environ.get("SPINF_API_KEY"):
            pytest.skip("no pack file: set SPINF_PROMPTS_DIR, SPINF_PACK_DIR or SPINF_API_KEY")
        cached.parent.mkdir(exist_ok=True)
        cached.write_text(json.dumps(SpinfClient().get_pack("spinf/moderation", version)))
    return packs.load(cached)


def _close(got: dict, want: dict) -> None:
    for c, w in want["categories"].items():
        g = got["categories"][c]
        assert g["flagged"] == w["flagged"], c
        assert abs(g["score"] - w["score"]) < TOL and abs(g["percentile"] - w["percentile"]) < TOL, c
    for k, w in want["decision"].items():
        g = got["decision"][k]
        assert abs(g - w) < TOL if isinstance(w, float) else g == w, k


@pytest.mark.parametrize("critical", [None, CRITICAL])
@pytest.mark.parametrize("mode", packs.MODES)
def test_fixture_reproduced(mode, critical):
    pack_file = _pack_file("0.1.1")
    assert pack_file["decision"]["micro_layer"]["lock"] == packs.lock_of(pack_file)
    for case in FIXTURE["cases"]:
        answers = packs.answers_from_result(case["result"], pack_file)
        want = case["expected"]["0.1.1+critical" if critical else "0.1.1"][mode]
        _close(packs.evaluate(pack_file, answers, mode, critical=critical), want)


def test_fixture_0_1_3_logprob():
    """0.1.3 (answers from logprob, fitted on engine fp_2b5214ebdc): the recorded results of the 8 texts, as the hosted
    pack scores them, reproduced in the three modes"""
    fixture = json.loads((HERE / "fixtures" / "moderation-0.1.3.json").read_text())
    pack_file = _pack_file("0.1.3")
    assert pack_file["answers"] == "logprob" and pack_file["engine"] == fixture["engine"]
    for mode in packs.MODES:
        for case in fixture["cases"]:
            _close(
                packs.evaluate(pack_file, packs.answers_from_result(case["result"], pack_file), mode),
                case["expected"][mode],
            )


def test_critical_overrides_and_lock_fallback():
    pack_file = _pack_file("0.1.1")
    cutting = FIXTURE["cases"][3]
    assert cutting["text"].startswith("I've been cutting myself")
    answers = packs.answers_from_result(cutting["result"], pack_file)
    assert packs.evaluate(pack_file, answers)["decision"]["unsafe"] is False  # the micro-layer alone: safe
    out = packs.evaluate(pack_file, answers, critical=CRITICAL)
    assert out["decision"]["unsafe"] is True and out["decision"]["by"] == "critical"
    assert out["decision"]["critical"] == ["selfharm"]
    # a critical category needs a threshold to flag: a None override silences it
    quiet = packs.evaluate(pack_file, answers, thresholds={"selfharm": None}, critical=CRITICAL)
    assert quiet["decision"]["unsafe"] is False

    ex = packs.answers_from_result(FIXTURE["cases"][0]["result"], pack_file)  # "How do I get into my ex's Instagram …"
    out = packs.evaluate(pack_file, ex, "per_category", thresholds={"cyber": None, "spam": 0.0})
    assert out["categories"]["cyber"]["flagged"] is None and out["categories"]["spam"]["flagged"] is True
    assert "spam" in out["decision"]["categories"] and "cyber" not in out["decision"]["categories"]
    out = packs.evaluate(pack_file, ex, "max", threshold=0.99)
    assert out["decision"]["unsafe"] is False and out["decision"]["category"] == "cyber"
    assert packs.evaluate(pack_file, ex, "micro_layer", threshold=5.0)["decision"]["unsafe"] is False

    # an edited pack: the micro-layer no longer applies, its fallback decides and says so
    edited = copy.deepcopy(pack_file)
    edited["categories"]["cyber"]["platt"][1] += 0.1
    out = packs.evaluate(edited, ex, "micro_layer")
    assert out["decision"]["mode"] == "per_category" and out["decision"]["warning"] == packs.LOCK_WARNING
    # thresholds and critical are not in the lock
    edited = copy.deepcopy(pack_file)
    edited["categories"]["cyber"]["threshold"] = 0.5
    edited["decision"]["critical"] = CRITICAL
    assert "warning" not in packs.evaluate(edited, ex, "micro_layer")["decision"]


def test_the_evaluator_is_spinf_serves():
    """_pipeline.py is spinf-serve's spinf_serve/packs/pipeline.py, byte for byte (SPINF_SERVE_DIR or ../spinf-serve)."""
    root = Path(os.environ.get("SPINF_SERVE_DIR") or HERE.parent.parent / "spinf-serve")
    theirs = root / "spinf_serve" / "packs" / "pipeline.py"
    if not theirs.exists():
        pytest.skip("no spinf-serve checkout: set SPINF_SERVE_DIR")
    ours = HERE.parent / "src" / "spinf_client" / "_pipeline.py"
    assert ours.read_text() == theirs.read_text(), "copy spinf-serve's pipeline.py over src/spinf_client/_pipeline.py"


def test_wrap_with_parameters_and_the_local_scoring_object():
    pack = {
        "pack": "acme/brand",
        "version": "0.1.0",
        "parameters": {"brand": {"label": "Brand", "required": True}, "tone": {"label": "Tone", "required": False}},
        "contexts": {"page": "Brand of interest: {brand}. {tone}\n\n{X}\n\nEnd."},
        "scoring": {
            "empty_floor": False,
            "queries": [{"id": "q", "template": "{X} ok? {?}", "options": ["yes", "no"]}],
        },
    }
    parts = packs.wrap(pack, "hello", params={"brand": "Kestrel"})
    assert [p["text"] for p in parts] == ["Brand of interest: Kestrel. \n\n", "hello", "\n\nEnd."]
    with pytest.raises(ValueError, match="required"):
        packs.wrap(pack, "hello")
    with pytest.raises(ValueError, match="no parameter"):
        packs.wrap(pack, "hello", params={"brand": "K", "colour": "red"})
    with pytest.raises(ValueError, match="no context"):
        packs.wrap(pack, "hello", context="mail", params={"brand": "K"})
    sc = packs.request_scoring(pack)
    assert sc["queries"][0]["group"] == "acme/brand@0.1.0" and sc["queries"][0]["empty_floor"] is False
    assert sc["case_insensitive"] is False and "group" not in pack["scoring"]["queries"][0]


def test_logprob_answers():
    """0.1.3+ ("answers": "logprob"): probabilities from the logprobs, renormalised over the options; they agree with
    the API's p where p is precise, and are not stuck on its 1e-6 grid where it is not"""
    pack_file = _pack_file("0.1.1")
    lp_pack = {**pack_file, "answers": "logprob"}
    result = FIXTURE["cases"][0]["result"]
    by_p, by_lp = packs.answers_from_result(result, pack_file), packs.answers_from_result(result, lp_pack)
    assert by_p.keys() == by_lp.keys() and len(by_p) == 106
    tiny = 0
    for k, ps in by_p.items():
        total = sum(ps)
        assert abs(sum(by_lp[k]) - 1) < 1e-9
        for p, q in zip(ps, by_lp[k], strict=True):
            if p / total > 1e-3:
                assert abs(p / total - q) / q < 1e-3
            elif p < 1e-5:
                tiny += 1
                assert q > 0 and q != round(q, 6)  # not on the 1e-6 grid
    assert tiny > 0


def _mock(handler):
    c = SpinfClient(api_key="k", base_url="http://x")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_moderate_hosted():
    seen = []
    expected = FIXTURE["cases"][0]["expected"]["0.1.1+critical"]["max"]

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        results = []
        for inp in body["inputs"]:
            own = [
                {"id": q["id"], "combinations": [{"values": {}, "options": [{"text": " en", "p": 0.9}]}]}
                for q in body["scoring"]["queries"]
            ]
            raw = FIXTURE["cases"][0]["result"]["queries"] if body["scoring"]["packs"][0].get("include_queries") else []
            pr = {"id": "spinf/moderation", "version": "0.1.1", "context": "forum", **expected}
            results.append({"input_id": inp["id"], "queries": own + raw, "packs": [pr]})
        return httpx.Response(200, json={"results": results})

    c = _mock(handler)
    lang = query("lang", "\n\nLanguage:{?}", [" en", " it"])
    one = c.moderate("hello", context="forum", mode="max", threshold=0.8, thresholds={"spam": None}, queries=[lang])
    assert isinstance(one, Moderation) and one.unsafe is True and one.mode == "max" and one.version == "0.1.1"
    assert one.flagged[0] == max(one.flagged, key=lambda k: one.categories[k].score)
    assert one.queries[0]["id"] == "lang" and one.raw is None and one.score == expected["decision"]["score"]
    ref = seen[0]["scoring"]["packs"][0]
    assert ref == {
        "id": "spinf/moderation",
        "context": "forum",
        "decision": {"mode": "max", "threshold": 0.8, "thresholds": {"spam": None}},
    }
    assert seen[0]["scoring"]["queries"][0]["id"] == "lang"
    assert seen[0]["inputs"] == [
        {"id": "0", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    ]

    many = c.moderate(["a", "b", "c"], version="0.1.1", include_queries=True, batch_size=2, ids=["x", "y", "z"])
    assert [m.id for m in many] == ["x", "y", "z"] and len(seen) == 3
    assert [len(b["inputs"]) for b in seen[1:]] == [2, 1]
    assert seen[1]["scoring"]["packs"][0] == {"id": "spinf/moderation", "version": "0.1.1", "include_queries": True}
    assert many[0].queries == [] and len(many[0].raw) == 58  # the pack's raw query results, apart from own queries
    with pytest.raises(ValueError):
        c.moderate(["a", "b"], ids=["x", "x"])


def test_moderate_with_a_pack_file():
    pack_file = _pack_file("0.1.1")
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(
            200, json={"results": [{**case["result"], "input_id": str(i)} for i, case in enumerate(FIXTURE["cases"])]}
        )

    c = _mock(handler)
    texts = [case["text"] for case in FIXTURE["cases"]]
    out = c.moderate(texts, mode="micro_layer", critical=CRITICAL, pack_file=pack_file, include_queries=True)
    body = seen[0]
    assert body["model"] == pack_file["model"] and body["scoring"]["case_insensitive"] is False
    assert "packs" not in body["scoring"] and "empty_floor" not in body["scoring"]
    assert all(q["empty_floor"] is False and q["group"] == "spinf/moderation@0.1.1" for q in body["scoring"]["queries"])
    parts = body["inputs"][0]["messages"][0]["content"]
    assert "".join(p["text"] for p in parts) == pack_file["contexts"]["ai"].replace("{X}", FIXTURE["cases"][0]["text"])
    for m, case in zip(out, FIXTURE["cases"], strict=True):
        want = case["expected"]["0.1.1+critical"]["micro_layer"]
        assert m.unsafe == want["decision"]["unsafe"] and m.context == "ai"
        assert abs(m.categories["hate"].score - want["categories"]["hate"]["score"]) < TOL
        # re-decided from the stored raw results, nothing scored: the micro-layer alone, as 0.1.1 shipped it
        _close(
            {"categories": {}, "decision": m.redecide(pack_file).decision},
            {"categories": {}, "decision": case["expected"]["0.1.1"]["micro_layer"]["decision"]},
        )
    with pytest.raises(ValueError):
        out[0].redecide({**pack_file, "version": "9.9.9"})


def test_pack_builder_and_endpoints():
    assert pack("spinf/moderation") == {"id": "spinf/moderation"}
    assert pack("spinf/moderation", "0.1.1", critical=[], include_queries=True) == {
        "id": "spinf/moderation",
        "version": "0.1.1",
        "include_queries": True,
        "decision": {"critical": []},
    }
    resp = {"results": [{"input_id": "0", "queries": [], "packs": [{"id": "spinf/moderation", "decision": {}}]}]}
    assert pack_result(resp)["id"] == "spinf/moderation"
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={})

    c = _mock(handler)
    c.list_packs(), c.get_pack(), c.get_pack("spinf/moderation", "0.1.1")
    assert seen == [
        ("/v1/packs", {}),
        ("/v1/packs/spinf/moderation", {}),
        ("/v1/packs/spinf/moderation", {"version": "0.1.1"}),
    ]
    assert _batches(["a" * 10] * 5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert _batches(["a" * 10] * 5, 64, max_chars=25) == [(0, 2), (2, 4), (4, 5)]
    assert _batches([], 2) == []


def _result(scores: dict[str, float]) -> Moderation:
    cats = {c: {"score": s, "percentile": None, "flagged": None} for c, s in scores.items()}
    return Moderation.from_result("0", "", {"categories": cats, "decision": {"unsafe": False, "mode": "max"}})


def test_thresholds_from_data():
    assert moderation.best_threshold([0.9, 0.8, 0.7, 0.2, 0.1], [True, True, False, True, False]) == 0.2
    assert moderation.best_threshold([0.9, 0.8, 0.7, 0.2, 0.1], [True, True, False, True, False], beta=0.5) == 0.8
    assert moderation.best_threshold([0.5, 0.5, 0.1], [True, False, False]) == 0.5  # ties flag together
    with pytest.raises(ValueError):
        moderation.best_threshold([0.1, 0.2], [True, True])

    results = [_result({"hate": i / 1000, "spam": 1 - i / 1000}) for i in range(1001)]
    t = moderation.rate_thresholds(results, rate=0.01)
    assert t == {"hate": 0.99, "spam": 0.99}
    assert sum(r.categories["hate"].score >= t["hate"] for r in results) == 11  # 0.99 .. 1.0

    labels = [{"hate"} if i > 900 else set() for i in range(1001)]
    fit = moderation.fit_thresholds(results, labels)
    assert fit == {"hate": 0.901}  # spam has no labelled texts: it keeps the pack's threshold
