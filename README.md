# spinf-client

Python client for the spinf scoring API: scored queries (`{?}` + options), extraction slots (`{*}` text, `{n}` number,
`{$}` amount), images and video frames, prompt packs and content moderation. One runtime dependency (httpx). The same
client talks to the hosted API at `api.spinf.com` and to a [spinf-serve](https://github.com/spinfinc/spinf-serve) of your own:
the request and the response are the same, only the base URL and the key change.

```bash
pip install "spinf-client[media]"          # the media extra adds Pillow for images; video frames also need ffmpeg / ffprobe on PATH
export SPINF_API_KEY=sk-...                # the hosted API
```

```python
from spinf_client import SpinfClient

client = SpinfClient()                                                                  # api.spinf.com, model "spinf-12b"
client = SpinfClient(base_url="http://localhost:8000", model="spinf/gemma-4-12b-fp8")   # your own spinf-serve
```

A spinf-serve started without `--token` needs no key (`SPINF_API_URL` sets the base URL from the environment); one started
with a token takes it as `api_key`. The hosted model names (`spinf-12b`, `spinf-31b`) are aliases of the Hub names
(`spinf/gemma-4-12b-fp8`, `spinf/gemma-4-31b-fp8`) and both work on both. Account, pricing and billing (`client.account()`,
`client.pricing()`, `billed_tokens()`) and the per-account features (extraction slots, video, the content limit) belong to
the hosted API; a server of your own has every feature on and no bill.

## Score and extract

```python
from spinf_client import SpinfClient, answers, query, text

client = SpinfClient()  # model="spinf-12b" by default (or "spinf-31b", ...)
out = client.score(
    content=[text("Apple sold 3.5 million iPhones in the quarter, up from 3.1 million.")],
    queries=[
        query("dir", "\n\nDid iPhone sales go{?}", options=[" up", " down"]),       # scored
        query("units", "\n\nQ: How many iPhones were sold?\nA:{n}"),                # a number
        query("co", '\n\nThe company in this news is "{*}"', top_k=3),              # text: in quotes, name the thing
    ],
)
answers(out, "units")   # {"answers": [{"text": "3.5 million", "p": 0.62, "value": 3500000.0}, ...], "mass": ..., "p_negative": 0.0}
```

Slots:

| slot | the answer | ends |
|---|---|---|
| `{*}` | any text; write the template with an opening quote: `... is "{*}"` | at the text after the slot (the closing quote) or a newline |
| `{n}` | a number: sign or accounting brackets, thousands separators, decimals, `thousand` / `million` / `billion`, `%` | when the next token cannot continue the number |
| `{$}` | the same with an optional currency symbol (`$ € £ ¥`), returned as `currency` | the same |

Each extraction returns the most likely answers (merged over spellings: text by case / punctuation / spaces, numbers by
value) with their probabilities, `mass` (the probability on all answers found) and, for numbers, `p_negative` (the share
of the top value's magnitude on its negative: brackets in accounting tables are often ambiguous). `max_tokens` 1-64
(default 16; up to 100 generated), `top_k` 1-20 (default 5). At most **10 slots per content** (queries × combinations). Extraction slots have
no empty floor.

`generate=True` on a `{*}` slot writes free text instead of searching short answers (a one-sentence summary, a caption):
greedy by default, or `generate={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 0, "n": 1}` to sample `n` sequences
reproducibly; `max_tokens` caps the length (up to 100). Answers come back as `{text, finish}` ("stop" at the closing quote, a newline
or end of text; "length" at the cap) with no probability; same account feature, billed as a generated slot (2,000 tokens flat against
500 for a searched one); no `top_k` / `calibration` with it.

Writing extraction templates (what the tests showed):

- `{*}`: in quotes, and name what goes there. `The company is "{*}"` reads as an adjective slot ("on track");
  `The company in this news is "{*}"` gives "Apple". Without quotes the model also writes formatting (`<strong>Apple</strong>`).
- `{n}` / `{$}`: no quotes and no filler; end right at the slot (`\nQ: How many ...?\nA:{n}` or `\nUnits sold: {n}`).
- One number slot for integers and decimals: check integer-ness on `value`. Use `{$}` on financial content.
- Take the sign from `p_negative`, not from the top answer.
- When the answers are a known short list, a scored question (`{?}` + options) is cheaper and has an empty floor.

## Calibrate a field

The raw probability ranks right answers above wrong ones but is under-confident by a field-dependent amount. Fit the
correction once per field on ~50 labelled examples:

```python
from spinf_client import calibration

a, b = calibration.fit(ps, rights)   # ps: the top answer's p per example, rights: was it right
query("amount", "\n\nQ: What was the revenue in 2019?\nA:{$}", calibration=(a, b))   # -> p_calibrated
```

Without labels, `calibration.DEFAULT_MAPS[model]` is right on average but can be off by ~0.2 on a field.

## Images and video

```python
from spinf_client import image, video

client.score(content=[image("photo.jpg", detail="auto"), text("\n")],
             queries=[query("what", '\nThis is a photo of "{*}"')])

# frames are sampled here with ffmpeg (the API never decodes video) and resized to the size the model reads
part = video("ad.mp4", num_frames=20, detail="low")          # or fps=1
client.score(content=[text("A 31-second TV commercial:\n"), part, text("\n")],
             queries=[query("brand", '\nThe brand featured in this commercial is "{*}"')])
```

A video is read as Gemma reads video, in place in the content: each frame an image at `detail` (low = up to 70 tokens)
after its `mm:ss` timestamp (7 text tokens). `media.video_tokens(w, h, n)` gives the content tokens (32 low frames of a
16:9 video: 2,399). ~20 low frames were enough for the brand of a Super Bowl ad (0.95 right on the 12B, 0.975 on the
31B). The request body is limited to 2 MB: ~80 low frames (~17 KB each as JPEG).

## Content moderation

The hosted `spinf/moderation` pack gives every text a calibrated 0–1 score per category (sexual, child safety,
violence, self-harm, hate, harassment, extremism, weapons, drugs, crime, fraud, hacking, privacy, spam, profanity,
gore, gibberish, a general score and four topics), its percentile on normal traffic, and a safe / unsafe decision.
The API does all of it in the call; nothing is computed here unless you ask.

```python
r = client.moderate("Immigrants are vermin and should be thrown out.")
r.unsafe, r.flagged            # True, ["hate", "general"]
r.categories["hate"]           # Category(score=0.81, percentile=0.999, flagged=True)
r.top(3), r.decision           # the decision as returned: mode, score, threshold, critical, ...

rs = client.moderate(posts, context="forum")    # "ai" (a message to an assistant, default) or "forum" (a post)
```

Decide your way (all optional, the pack's defaults otherwise):

| argument | what it does |
|---|---|
| `mode` | `micro_layer` (default: a small linear layer over the category scores and general reads), `max` (the highest category score against one threshold) or `per_category` (any category over its own threshold) |
| `threshold` | the one cut of `max` (0–1) or `micro_layer` (the layer's value) |
| `thresholds` | per category, `{"violence": 0.8, "profanity": None}`; `None` = reported, never decides. They set `flagged` in every mode. Defaults: the score 0.5% of normal traffic reaches |
| `critical` | categories that make a text unsafe whenever flagged, whatever the mode (default `["selfharm", "child"]`; `[]` = none) |
| `version` | pin the pack version (`client.list_packs()`) to keep scores comparable over time |
| `queries` | your own queries, scored in the same pass on the same content (`r.queries`) |

Texts are sent in batches of 64 (`batch_size`); the result list keeps their order (`ids=` names them).

**Your thresholds from your data** (no retraining; standard library):

```python
from spinf_client import moderation

sample = client.moderate(my_normal_traffic, context="forum")          # a few thousand typical texts
t = moderation.rate_thresholds(sample, rate=0.002)                    # flag 0.2% of it per category
t = moderation.fit_thresholds(labelled, labels)                       # or: best F1 on texts you labelled
                                                                      # (labels[i] = {"hate", ...}, empty = fine)
cut = moderation.best_threshold([r.score for r in res], is_unsafe)    # one cut for mode="max" / "micro_layer"
client.moderate(texts, mode="per_category", thresholds=t)
```

**Re-decide without scoring again:** score with `include_queries=True` (the pack's raw query results stay on
`r.raw`), then try other modes, thresholds or critical lists locally:

```python
pack_file = client.get_pack("spinf/moderation", r.version)   # the pack file (keep it; it changes only by version)
r.redecide(pack_file, mode="max", threshold=0.8, thresholds={"spam": None}, critical=[])   # -> Moderation
```

The local evaluator (`packs`) is the API's own, in the standard library: a verbatim copy of spinf-serve's
`spinf_serve/packs/pipeline.py`, kept identical by a test; it reproduces the API's scores to 1e-6.

**Edit the pack:** `client.moderate(texts, pack_file=edited)` sends an edited copy's questions as ordinary queries and
decides here. Thresholds and the critical list are yours to change; any change to the questions, reads or calibration
breaks the micro-layer's lock (a sha256 over them), and the pack's fallback mode (`per_category`) decides, with a
`warning` in the decision.

**Other packs and your own queries in one call:** `score()` takes `packs=[pack(id, ...)]` next to `queries`; each
result carries `packs` (`pack_result(out, id)`). `pack()` takes the same decision arguments as `moderate()`.

## Pricing and usage

```python
from spinf_client import billed_tokens

pricing = client.pricing()            # what this account can use: models, media rates, extraction, content limit
m = next(x for x in pricing["models"] if x["model"] == "spinf-12b")
billed_tokens(m, text_tokens=150, frames={"low": 20}, query_tokens=23, floor_tokens=14, slots_per_content=[1])  # 2687
```

Billed tokens = text (content text and timestamps) + images and frames at the model's rate per detail + audio per
second + question and floor tokens + extraction (500 tokens a searched slot, 2,000 a generated one), at least 1,000 a
call, at the model's price per million. The response's `usage.prompt_tokens_details` reports `text_tokens`, `media`
(`images`, `video_frames`, `audio_seconds`, `billed_media_tokens`) and `extraction` (`slots`, `generated_slots`,
`billed_tokens`);
`usage.billed_tokens` is what is charged. `client.models()` and `client.account()` wrap `GET /v1/models` and
`GET /v1/account`.

Errors raise `SpinfError` (`status`, `code`, `message`, `param`): e.g. `403 feature_not_enabled` (extraction not
enabled), `403 media_not_allowed` (video not enabled), `400 too_many_extractions`, `400 content_too_long` (over the
account's content limit), `413 body_too_large`. 429 / 5xx (including `503 warming_up`) and timeouts are retried.
