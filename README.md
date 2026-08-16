# lmm — Local/remote Model Manager & Unified Inference Hub

> One file. Zero dependencies. Every LLM runtime on your machine, **and a
> single inference hub that routes any prompt to any backend** — local *or*
> cloud — with one interface, streaming, fallbacks, and live cost.

`lmm` started as a tiny manager for **all** your LLM runtimes (local: Ollama,
LM Studio, Jan, GPT4All, KoboldCPP, vLLM, llama.cpp, Open WebUI; remote/desktop:
Claude, ChatGPT, Cursor, Perplexity, AnythingLLM, Chatbox, Msty, Devin/Cua). It
discovers what's running, shows live status + GPU + your real Anthropic spend,
recommends local-vs-remote routing, and — the part that solves the actual pain —
**keeps LLM app windows off your taskbar** automatically.

Then it grew into a **unified inference hub**: one command (`lmm ask`), one
OpenAI-compatible proxy (`lmm serve --hub`), user-controlled provider priority
(`ask_order`), automatic fallback, and live cost accounting across *every*
backend. Local OR cloud, one interface.

## Why this exists (first principles)

The real problem isn't "I need an LLM manager app." It's:
1. **Multiple LLM apps each grab a taskbar button**, and you lose track of what
   is actually running.
2. **You shouldn't have to remember *which* provider to call.** `lmm` holds the
   routing brain — `ask_order` + automatic fallback — so one prompt reaches the
   right backend without you thinking about it.
3. **The fix should be a system, not a band-aid.** So `lmm` can auto-hide new
   LLM windows the instant they launch, and serve a stable OpenAI-compatible
   endpoint any client can use.

## Install

```bash
# download the single file
curl -fsSL https://raw.githubusercontent.com/shizukutanaka/lmm/main/lmm.py -o ~/.local/bin/lmm && chmod +x ~/.local/bin/lmm
# or just:  python lmm.py <command>

# install + shell alias (macOS/Linux)
./install.sh
# Windows:  powershell -ExecutionPolicy Bypass -File install.ps1
```

Requires only Python 3.8+ (tkinter optional, for the GUI). No `pip install`.

## Usage

```bash
lmm                 -> open the live GUI dashboard (default)
lmm discover        -> list every detected runtime (CLI)
lmm status          -> live status + GPU memory
lmm models          -> local models installed (Ollama)
lmm cost [--days N] -> measured Anthropic token cost + all-provider estimates
lmm route "task"    -> recommend local vs remote for a task

# --- unified inference hub ---
lmm ask "prompt" [--provider NAME]   -> one-shot inference, auto-routed + fallback
lmm chat [--provider NAME]           -> interactive REPL (keeps conversation history)
lmm serve --hub [--host H] [--port P] -> OpenAI-compatible proxy (zero-config)
lmm models          -> unified model registry (local + every cloud backend)
lmm pull <model>    -> pull a model into the local Ollama store
lmm hub-status       -> probe every backend's health (measures, doesn't assume)
lmm log [N]          -> show last N hub routing events (proof of what it did)
lmm selftest         -> self-prove the hub works (syntax, routing, observability)

# --- management ---
lmm serve <model>   -> pull + expose a local model endpoint (Ollama)
lmm stop <runtime>  -> stop a running runtime
lmm hide <runtime>  -> strip a runtime's taskbar button (Claude/ChatGPT/...)
lmm watch           -> background daemon: auto-hide new LLM windows
lmm autostart       -> register `watch` to run at OS login (zero effort)
lmm dash            -> generate + open a self-contained HTML dashboard
lmm gui             -> open the live GUI dashboard explicitly
lmm config <init|list|get|set|unset> [key] [value] -> manage hub settings (CLI)
lmm examples        -> print a sample config file
```

### The hub, in one minute

```bash
# 1. Ask anything — lmm picks the backend (ask_order), falls back on failure
$ lmm ask "What is 2+2?"
[ask] -> trying local-ollama(implicit) (qwen2.5-coder:7b)
4

# 2. Streaming + conversation (REPL keeps history across turns)
$ lmm chat
you> explain recursion in one sentence.
hub> Recursion is when a function calls itself to solve smaller subproblems.

# 3. A stable OpenAI-compatible endpoint for ANY client (zero config)
$ lmm serve --hub --port 8080
# now:  curl http://localhost:8080/v1/chat/completions  (OpenAI SDK works)

# 4. User-controlled routing priority
$ lmm config set ask_order '["openai","local-ollama(implicit)"]'

# 5. Prove what the hub actually did (measurement, not assumption)
$ lmm log
[OK ] 2026-08-17T02:44:22 ask   -> local-ollama(implicit) prompt='What is 2+2?'
[OK ] 2026-08-17T02:51:18 serve -> local-ollama(implicit) reply='STREAMOK'
```

### Unified model registry

`lmm models` lists models across **every** detected backend — local Ollama *and*
any configured cloud OpenAI-compatible endpoint (`/v1/models`). It measures each
live; unreachable backends are reported, not assumed-present.

```bash
$ lmm models
Ollama (local):
  - qwen2.5-coder:7b
  - qwen2.5-coder:3b
openai (cloud):
  - gpt-4o
  - gpt-4o-mini

$ lmm pull qwen2.5-coder:7b   # stock the local store, then route to it
```

### Self-prove it works (`selftest`)

```bash
$ lmm selftest
lmm selftest — measuring, not trusting:
  [PASS] self syntax (py_compile)
  [PASS] command surface complete
  [PASS] implicit Ollama reachable -- qwen2.5-coder:7b
  [PASS] live ask routing returns reply -- SELFTEST_OK
  [PASS] observability (hub.log writable) -- created

SELFTEST PASS — the hub measures and proves itself.
```

`lmm selftest` runs **real measurements** (not trust): it compiles itself, checks
the command surface, probes Ollama, performs a live routed `ask`, and confirms
`hub.log` is writable. Non-zero exit on any failure — usable as a fleet/CI gate.

`lmm` routes every `ask` / `chat` / `serve --hub` request through the **same**
logic:
1. Explicit `--provider NAME` (if it's a configured provider).
2. Else your `ask_order` list (`lmm config set ask_order [...]`).
3. Else a sensible default (keyword match → configured → implicit running Ollama).
4. Always-on zero-config safety net: a running Ollama is appended automatically.

If a provider errors (network/auth), `lmm` falls through to the next — silently
for unspecified targets, respecting your explicit choice otherwise.

### The taskbar problem, solved

- **One-off:** `lmm hide claude` strips Claude's taskbar button immediately
  (the app keeps running). Works for ChatGPT, Cursor, Perplexity, GPT4All,
  AnythingLLM, Chatbox, Msty, Devin/Cua too.
- **Permanent (recommended):** `lmm autostart` registers a background watcher
  that hides *any* new LLM window the moment it appears. You never touch the
  taskbar again. Disable by deleting the scheduled task / launchd plist /
  systemd unit.

## Examples

```bash
$ lmm              # GUI opens: live table of every runtime + GPU + cost

$ lmm discover
[ON ] Ollama                  free  models=qwen2.5-coder:7b @ http://localhost:11434
[ON ] LM Studio               free  procs=5 @ http://localhost:1234/v1
[ON ] Claude Code (Anthropic) PAID  procs=10 @ api.anthropic.com

$ lmm cost
Anthropic measured usage (all-time, 109 sessions)
----------------------------------------------------------------
[opus] sessions=109
   in=72,970 out=297,680 cache_w=4,155,375 cache_r=53,125,702
   est $181.02
----------------------------------------------------------------
TOTAL est $181.02  (pricing approximate; verify on your Anthropic billing)

$ lmm route "summarize this internal secret doc"
=> recommend: Ollama (local, free, private)
```

## Supported runtimes (16, auto-detected)

Ollama · LM Studio · Jan · GPT4All · AnythingLLM · Chatbox · Msty · KoboldCPP ·
Open WebUI · vLLM · llama.cpp · Claude · ChatGPT · Cursor · Perplexity ·
Devin/Cua. Add your own in `~/.lmm/config.json` (see `lmm examples`).

## Configuration

Works with **zero config** (Ollama is auto-detected as a safety net). Edit
settings from the CLI with `lmm config` — no hand-editing JSON:

```bash
lmm config init                                   # create ~/.lmm/config.json
lmm config set ask_order '["openai","local-ollama(implicit)"]'
lmm config set providers.openai '{"kind":"cloud","base_url":"https://api.openai.com/v1","api_key":"$OPENAI_API_KEY","model":"gpt-4o"}'
lmm config get ask_order
lmm config unset providers.openai
```

`~/.lmm/config.json` (or `~/.config/lmm/config.json`, or `./lmm.config.json`) can
add runtimes, override pricing, register cloud providers, or change routing
keywords. See `lmm examples` for the shape.

## How it works

| Concern        | Approach                                                        |
|----------------|-----------------------------------------------------------------|
| Discovery      | Probe each runtime's process + data dir + endpoint              |
| Secrets        | Only *check* existence of existing credentials; never store one |
| Cost           | Aggregate real tokens from `~/.claude/projects/*.jsonl` × public pricing; cloud providers estimated from your `usage` entries |
| Routing        | One brain (`ask_order` + fallback) shared by `ask`/`chat`/`serve --hub`; implicit Ollama safety net |
| Streaming      | SSE pass-through (`http_post_stream`, stdlib only) — token-by-token to CLI and proxy clients |
| Observability  | Every routed turn logged to `~/.lmm/hub.log`; `lmm log` proves it |
| Health         | `hub-status` probes each backend live (no assumptions)          |
| Taskbar hide   | `WS_EX_TOOLWINDOW` on visible windows (Win); headless launch advised for servers |
| Auto mode      | `watch` daemon hides new LLM windows every few seconds          |
| Portability    | `expanduser` paths, `tasklist`/`pgrep`, `tkinter` for the GUI  |

## License

MIT — see [LICENSE](LICENSE).
