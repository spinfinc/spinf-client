# Contributing to spinf-client

- **Set up**: `uv sync`, then `uv run pytest` and `uv run ruff check . && uv run ruff format --check src tests`.
- **One runtime dependency** (httpx); media helpers stay behind the `media` extra. Python 3.11 and up.
- **The evaluator is spinf-serve's.** `src/spinf_client/_pipeline.py` is a verbatim copy of spinf-serve's
  `spinf_serve/packs/pipeline.py`, checked by a test against a checkout next to this one (or `SPINF_SERVE_DIR`). Fix it
  there, then copy it here.
- **Both servers.** Everything the client does must work against the hosted API and against a spinf-serve of your own;
  what is hosted-only (account, pricing, billing) says so in its docstring and in the README.
- Keep the public hygiene test green (`tests/test_public_hygiene.py`). Apache 2.0 applies to every contribution.
