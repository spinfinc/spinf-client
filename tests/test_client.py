import json
import math
import random
import shutil
import subprocess

import httpx
import pytest

from spinf_client import SpinfClient, SpinfError, answers, billed_tokens, calibration, media, query, text


def test_target_size_matches_the_engine():
    # the engine (spinf_serve.media.decode_image): whole 48 px blocks, <= the detail budget, aspect kept
    assert media.target_size(1920, 1080, "low") == (528, 288)  # 11 x 6 = 66 blocks
    assert media.target_size(640, 480, "low") == (432, 336)  # 9 x 7 = 63
    assert media.target_size(320, 240, "auto") == (912, 672)  # 19 x 14 = 266 (upscaled; the image probe: 266 tokens)
    for w, h in ((1920, 1080), (640, 480), (100, 3000), (3000, 100)):
        for d, cap in media.DETAIL_TOKENS.items():
            tw, th = media.target_size(w, h, d)
            assert tw % 48 == 0 and th % 48 == 0 and (tw // 48) * (th // 48) <= cap
    assert media.image_tokens(1920, 1080) == 68 and media.video_tokens(1920, 1080, 32) == 32 * 75 - 1


def test_sample_times():
    assert media.sample_times(10, num_frames=4) == [1.25, 3.75, 6.25, 8.75]
    assert len(media.sample_times(10, fps=2)) == 20
    with pytest.raises(ValueError):
        media.sample_times(10)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_frames_from_a_file(tmp_path):
    path = str(tmp_path / "v.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=10", path], check=True
    )
    frames = media.video_frames(path, num_frames=8)
    assert len(frames) == 8 and frames[0][0] == 0.25 and frames[0][1][:2] == b"\xff\xd8"
    part = media.video(frames)
    assert part["video"]["detail"] == "low" and part["video"]["frames"][1]["t"] == 0.75
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(frames[0][1])).size == media.target_size(640, 360, "low")


def test_calibration_fit_recovers_a_shift():
    rng = random.Random(0)
    ps, rights = [], []
    for _ in range(4000):
        p = rng.random() * 0.98 + 0.01
        truth = 1 / (1 + math.exp(-(1.0 * math.log(p / (1 - p)) + 2.0)))  # under-confident by +2 logits
        ps.append(p)
        rights.append(rng.random() < truth)
    a, b = calibration.fit(ps, rights)
    assert abs(a - 1.0) < 0.15 and abs(b - 2.0) < 0.2
    assert calibration.ece([calibration.apply(p, a, b) for p in ps], rights) < calibration.ece(ps, rights)


def test_client_requests_and_errors():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if body["scoring"]["queries"][0]["id"] == "bad":
            return httpx.Response(400, json={"error": {"code": "invalid_template", "message": "no slot", "param": "x"}})
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "input_id": "0",
                        "queries": [
                            {
                                "id": "n",
                                "kind": "number",
                                "combinations": [
                                    {
                                        "values": {},
                                        "answers": [{"text": "3.5 million", "p": 0.6, "value": 3.5e6}],
                                        "mass": 0.7,
                                        "p_negative": 0.0,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        )

    c = SpinfClient(api_key="k", base_url="http://x")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    out = c.score(
        content=[text("Apple sold 3.5 million phones.")], queries=[query("n", "\nA:{n}", calibration=(0.8, 1.5))]
    )
    assert answers(out, "n")["answers"][0]["value"] == 3.5e6
    assert seen[0]["messages"][0]["content"][0]["text"].startswith("Apple")
    assert seen[0]["scoring"]["queries"][0]["calibration"] == {"a": 0.8, "b": 1.5}
    with pytest.raises(SpinfError) as e:
        c.score(content=[text("x")], queries=[query("bad", "no slot")])
    assert e.value.code == "invalid_template" and e.value.status == 400
    c.score(inputs=[{"id": "a", "content": [text("x")]}], queries=[query("n", "\nA:{n}")])
    assert seen[-1]["inputs"][0]["messages"][0]["content"][0]["text"] == "x"


PRICING_12B = {
    "model": "spinf-12b",
    "min_billed_tokens_per_call": 1000,
    "media": {
        "image": {"tokens_per_image": {"low": 100, "auto": 300, "high": 1300}},
        "video": {"tokens_per_frame": {"low": 100, "auto": 300, "high": 1300}, "timestamps": "billed as text"},
        "audio": {"tokens_per_second": 30},
    },
    "extraction": {"first_per_content_tokens": 500, "next_per_content_tokens": 100, "max_per_content": 10},
}


def test_billed_tokens_match_live_bills():
    # live calls 2026-09-29: a 20-frame ad + 1 scored question + 1 brand slot; a news sentence with 3 slots
    assert (
        billed_tokens(PRICING_12B, 150, frames={"low": 20}, query_tokens=23, floor_tokens=14, slots_per_content=[1])
        == 2687
    )
    p31 = {**PRICING_12B, "media": {"video": {"tokens_per_frame": {"low": 200, "auto": 500, "high": 2000}}}}
    assert billed_tokens(p31, 150, frames={"low": 20}, query_tokens=23, floor_tokens=14, slots_per_content=[1]) == 4687
    assert billed_tokens(PRICING_12B, 31, query_tokens=22, floor_tokens=3, slots_per_content=[3]) == 1000
    assert billed_tokens(PRICING_12B, 2000, audio_seconds=10.01, slots_per_content=[2, 0, 1]) == 2000 + 301 + 600 + 500


def test_video_tokens_count_timestamps_per_digit():
    # the 20-frame ad above: 150 text tokens = 10 (intro) + 6 + 19 x 7 (timestamps) + 1 ("\n")
    assert media.video_tokens(854, 480, 20) - 20 * media.image_tokens(854, 480) == 6 + 19 * 7


def test_get_endpoints():
    paths = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append((req.method, req.url.path))
        return httpx.Response(200, json={"models": []})

    c = SpinfClient(api_key="k", base_url="http://x")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    c.pricing(), c.models(), c.account()
    assert paths == [("GET", "/v1/account/pricing"), ("GET", "/v1/models"), ("GET", "/v1/account")]
