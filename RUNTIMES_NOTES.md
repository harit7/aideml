# llm_runtimes integration notes

Additive routing that lets AIDE use two extra model backends besides its usual
API providers, via the vendored `llm_runtimes/` package at the repo root:

- `claudecli-<model>` (e.g. `claudecli-sonnet`, `claudecli-opus`, `claudecli-haiku`)
  — Claude through the local `claude -p` CLI (subscription auth, no API key).
- `local-<alias>` (e.g. `local-qwen`) — a local model served by an in-process
  vLLM engine (default: `Qwen/Qwen3-8B-AWQ`, overridable with
  `LLM_RUNTIMES_LOCAL_HF_ID`).

Both are exposed through a lazily started OpenAI-compatible HTTP server
(`llm_runtimes.ensure_server()`, default `http://127.0.0.1:8399/v1`), so AIDE's
existing OpenAI backend code path is reused unchanged, including function
calling (`func_spec`): the runtime server emulates OpenAI `tool_calls`.

## Files changed

- `aide/backend/__init__.py` — `determine_provider()` now routes model names
  starting with `claudecli-` or `local-` to the OpenAI backend.
- `aide/backend/backend_openai.py` — new `_setup_runtimes_client()` builds an
  OpenAI client with `base_url=llm_runtimes.ensure_server()` and
  `api_key="llm-runtimes"`. `query()` selects that client (and the chat
  completions API) for runtime-prefixed models; all other models use the
  pre-existing code paths untouched. For runtime models, `OPENAI_API_KEY` is
  not required and `max_tokens` is passed through as-is (not renamed to
  `max_output_tokens`).

`llm_runtimes/` itself is vendored and unmodified.

## Usage

Run from the repo root (or put the repo root / an installed `llm_runtimes` on
`PYTHONPATH`); otherwise runtime-prefixed models raise an `ImportError`
explaining this.

### Usual API backends (unchanged)

```bash
export OPENAI_API_KEY=...   # and/or ANTHROPIC_API_KEY, etc.
aide data_dir=my_data goal="Predict X" agent.code.model=o4-mini
```

### Claude CLI backend

Requires a logged-in `claude` CLI on PATH. No API keys needed.

```bash
aide data_dir=my_data goal="Predict X" \
  agent.code.model=claudecli-sonnet \
  agent.feedback.model=claudecli-haiku \
  report.model=claudecli-sonnet
```

### Local vLLM backend

```bash
# optional: export LLM_RUNTIMES_LOCAL_HF_ID=Qwen/Qwen3-8B-AWQ
aide data_dir=my_data goal="Predict X" \
  agent.code.model=local-qwen \
  agent.feedback.model=local-qwen
```

### Programmatic

```python
from aide.backend import query
out = query(system_message=None, user_message="Say hi", model="claudecli-haiku")
```

## Limitations

- `claudecli-*` ignores sampling parameters such as `temperature` (the CLI
  does not expose them), so `agent.code.temp` etc. have no effect there.
- The runtime server reports zeroed token counts
  (`prompt_tokens=0, completion_tokens=0`), so AIDE's per-call token logging
  shows 0 tokens for these models.
- `local-*` defaults to `Qwen/Qwen3-8B-AWQ`; the first `local-*` call loads
  the model into the in-process vLLM engine, which takes a while and needs a
  suitable GPU.
- The runtime server binds a fixed local port (default 8399) shared across
  scaffolds on the machine.
