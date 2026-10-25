"""Images and video for the API: parts built locally, frames sampled locally (the API never decodes video).

A video is sent as frames with their timestamps: {"type": "video", "video": {"detail", "frames": [{"t", "image_url"}]}}.
The model reads each frame as an image at `detail` (low = up to 70 tokens, auto 280, high 1,120, in whole 48 px blocks),
after its "mm:ss" timestamp, as Gemma's own video prompt. Frames are resized here to the exact size the engine would
resize them to, so the request carries no pixels the model never sees (the API would resize larger ones too, but the
request body is limited to 2 MB: a low-detail 16:9 frame is ~17 KB, ~23 KB in base64, so ~80 frames per call).
"""

from __future__ import annotations

import base64
import io
import json
import math
import subprocess
from typing import Any

SIDE = 48  # one image token = a 48x48 block (16 px patches, 3x3 pooling)
DETAIL_TOKENS = {"low": 70, "auto": 280, "high": 1120}


def target_size(width: int, height: int, detail: str = "low") -> tuple[int, int]:
    """The (width, height) the engine resizes an image to for `detail`: aspect kept, both sides whole 48 px blocks,
    at most DETAIL_TOKENS[detail] blocks (spinf_serve.media.decode_image)."""
    soft = DETAIL_TOKENS[detail]
    f = math.sqrt(soft * SIDE * SIDE / (height * width))
    th, tw = math.floor(f * height / SIDE) * SIDE, math.floor(f * width / SIDE) * SIDE
    if th == 0:
        th, tw = SIDE, min(int(width / height) * SIDE, soft * SIDE)
    if tw == 0:
        tw, th = SIDE, min(int(height / width) * SIDE, soft * SIDE)
    return tw, th


def image_tokens(width: int, height: int, detail: str = "low") -> int:
    """Content tokens of one image (the blocks, plus 2 markers)."""
    tw, th = target_size(width, height, detail)
    return (tw // SIDE) * (th // SIDE) + 2


def _jpeg_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def image(source: str | bytes | Any, detail: str = "auto", resize: bool = True, quality: int = 90) -> dict[str, Any]:
    """An image_url part from a path, bytes or a PIL image; resized to the engine's size for `detail` (needs Pillow)."""
    from PIL import Image

    im = (
        source
        if hasattr(source, "convert")
        else Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    )
    im = im.convert("RGB")
    if resize:
        im = im.resize(target_size(*im.size, detail), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return {"type": "image_url", "image_url": {"url": _jpeg_url(buf.getvalue()), "detail": detail}}


def probe(path: str) -> tuple[int, int, float]:
    """(width, height, duration in seconds) of a video file, from ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    o = json.loads(out)
    return int(o["streams"][0]["width"]), int(o["streams"][0]["height"]), float(o["format"]["duration"])


def sample_times(duration: float, num_frames: int | None = None, fps: float | None = None) -> list[float]:
    """Frame times: `num_frames` evenly over the video (the middle of each of n equal slices; Gemma's own default is 32),
    or `fps` frames per second."""
    if (num_frames is None) == (fps is None):
        raise ValueError("pass exactly one of num_frames or fps")
    n = num_frames if num_frames is not None else max(1, int(duration * fps))
    return [(i + 0.5) * duration / n for i in range(n)]


def video_frames(
    path: str, num_frames: int | None = 32, fps: float | None = None, detail: str = "low", quality: int = 85
) -> list[tuple[float, bytes]]:
    """Sample frames with ffmpeg -> [(t, jpeg bytes)], each already at the engine's size for `detail`."""
    w, h, dur = probe(path)
    tw, th = target_size(w, h, detail)
    q = max(2, min(31, round(31 - quality * 0.29)))  # ffmpeg -q:v 2 (best) .. 31
    out = []
    for t in sample_times(dur, None if fps else num_frames, fps):
        for back in (0.0, 0.5, 1.5):  # a seek that lands on no decodable frame: try a little earlier
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{max(0.0, t - back):.3f}",
                    "-i",
                    path,
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={tw}:{th}:flags=bicubic",
                    "-q:v",
                    str(q),
                    "-f",
                    "image2",
                    "-c:v",
                    "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                check=False,
            )
            if r.returncode == 0 and r.stdout:
                out.append((t, r.stdout))
                break
    return out


def video(frames: list[tuple[float, bytes]] | str, detail: str = "low", **sample: Any) -> dict[str, Any]:
    """A video part from [(t, jpeg bytes)] or from a file path (sampled with video_frames(path, **sample))."""
    if isinstance(frames, str):
        frames = video_frames(frames, detail=detail, **sample)
    return {
        "type": "video",
        "video": {
            "detail": detail,
            "frames": [{"t": round(t, 3), "image_url": {"url": _jpeg_url(b)}} for t, b in frames],
        },
    }


def video_tokens(width: int, height: int, n_frames: int, detail: str = "low") -> int:
    """Content tokens of a video part: per frame its image and its timestamp, one token per digit (" mm:ss ": 7
    tokens, 6 for the first frame). Billing counts the timestamps as text and each frame at the model's frame rate
    (client.pricing(); billed_tokens())."""
    return n_frames * (image_tokens(width, height, detail) + 7) - (1 if n_frames else 0)
