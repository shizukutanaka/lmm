# lmm — Local/remote Model Manager

> One file. Zero dependencies. Every LLM app on your machine, in one place —
> and it never clutters your taskbar.

`lmm` is a tiny, cross-platform manager for **all** your LLM runtimes: local
(Ollama, LM Studio, Jan, GPT4All, KoboldCPP, vLLM, llama.cpp, Open WebUI) and
remote/desktop (Claude, ChatGPT, Cursor, Perplexity, AnythingLLM, Chatbox,
Msty, Devin/Cua). It discovers what's running, shows live status + GPU + your
real Anthropic spend, recommends local-vs-remote routing, and — the part that
solves the actual pain — **keeps LLM app windows off your taskbar**
automatically.

## Why this exists (first principles)

The real problem isn't "I need an LLM manager app." It's:
1. **Multiple LLM apps each grab a taskbar button**, and you lose track of what
   is actually running.
2. **You shouldn't have to remember commands.** State must be *visible*
   (Nielsen heuristic #1 — *Visibility of System Status*).
3. **The fix should be a system, not a band-aid.** So `lmm` can auto-hide new
   LLM windows the instant they launch — no clicking required.

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

```
lmm                 -> open the live GUI dashboard (default)
lmm discover        -> list every detected runtime (CLI)
lmm cli             -> same as discover (explicit CLI mode)
lmm status          -> live status + GPU memory
lmm models          -> local models installed (Ollama)
lmm cost [--days N] -> measured spend: Anthropic logs + lmm's own hub telemetry
lmm route "task"    -> recommend local vs remote (--explain for the score)
lmm fit [model]     -> does it fit in your GPU, and at what context length?
lmm bench           -> measure TTFT / TPOT / throughput per provider
lmm ask "prompt"    -> ask any backend: cached, routed, optionally cascaded
lmm serve <model>   -> pull + expose a local model endpoint (Ollama)
lmm serve --hub     -> OpenAI-compatible proxy over every configured provider
lmm cache           -> prompt-cache stats (--clear to drop it)
lmm stop <runtime>  -> stop a running runtime
lmm hide <runtime>  -> strip a runtime's taskbar button (Claude/ChatGPT/...)
lmm watch           -> background daemon: auto-hide new LLM windows
lmm autostart       -> register `watch` to run at OS login (zero effort)
lmm dash            -> generate + open a self-contained HTML dashboard
lmm gui             -> open the live GUI dashboard explicitly
lmm examples        -> print a sample config file
```

### The hub: one endpoint, and a cheaper bill

Point your apps at `lmm serve --hub` and every request goes through one path —
cache, routing, cascade, metering — across all your local and cloud backends.
Three published techniques do the cost work, implemented with the standard
library alone:

| Layer | Paper | What it does here |
|-------|-------|-------------------|
| Cache | [GPT Semantic Cache](https://arxiv.org/abs/2411.05276) (68.8% fewer API calls) | Exact-hash tier is on by default; an optional semantic tier uses **local** Ollama embeddings, never a paid one |
| Routing | [RouteLLM](https://arxiv.org/abs/2406.18665), ICLR 2025 (40% fewer strong-model calls, <5% quality loss) | Scores the prompt and compares it to a cost threshold `α`; easy prompts try cheap providers first |
| Cascade | [FrugalGPT](https://arxiv.org/abs/2305.05176) (up to 98% cost reduction) | `--cascade` runs cheap models first and escalates only when the answer scores badly |

[vCache](https://arxiv.org/abs/2502.03771) shows a single static similarity
threshold cannot bound false cache hits, so the semantic tier is **opt-in**, its
default threshold is a strict `0.95`, and every near miss is logged so you can
tune it from evidence (`lmm cache`).

RouteLLM and FrugalGPT both learn their scorers from training data. lmm has
neither the data nor the dependencies for that, so it approximates them with
lexical features and a heuristic answer verifier. The thresholds — the part you
actually tune — work exactly as published.

```bash
$ lmm ask --cascade --explain "what is 2+2"
[route] strength=0.03 threshold=0.5 -> local-ollama, deepseek, openai
[cascade] rung0 local-ollama score=0.85 $0.0000 -> accept
[cascade] total $0.0000 over 1 rung(s)
4

$ lmm ask --cascade --explain "what is 2+2"      # asked again
[cache] exact hit (sim=1.000) — $0.0000, saved ~$0.0012
4

$ lmm route --explain "refactor this async scheduler and explain why it deadlocks"
  heavy keyword (refactor)               +0.30
  code marker (async)                    +0.25
  reasoning marker (explain why)         +0.20
  strength s = 0.94  >=  threshold 0.50
=> strong-first: openai, deepseek, local-ollama
```

Nothing here is on by default except the exact-match cache. With no config at
all, `lmm ask` behaves as it always did — it just now records what each call
cost. **Your own `ask_order` always wins**: auto-routing only decides the order
you did not specify.

Privacy outranks price everywhere: a prompt matching your `route.private`
keywords is pinned to local providers, and a cascade on such a prompt will not
escalate off the machine no matter how badly the local model scores.

### Will it even run? `lmm fit`

Routing to a local model is only free if the model actually loads. `lmm fit`
answers that with arithmetic instead of trial and error:

```
weights  = params × bits_per_weight / 8
KV cache = 2 × layers × kv_heads × head_dim × context × bytes_per_element
total    = weights + KV cache + ~0.5 GiB (context, activations, scratch)
```

```bash
$ lmm fit llama3.1:8b --vram 8 --ctx 32768
VRAM budget: 8.0 GiB  [--vram]
KV cache dtype: f16  (2.0 bytes/element)
------------------------------------------------------------------------
  [OVER] llama3.1:8b                    9.02 GiB @ 32,768 ctx
         weights 4.52 + kv 4.00 + overhead 0.50   (8.0B params, 4.85 bpw, 32L, 8 kv-heads x 128)
         fits up to 24,437 tokens of context
         -> fits at 7.02 GiB with --kv q8_0 (llama.cpp --cache-type-k/v q8_0)
```

Two details the naive version of this gets wrong:

- **`kv_heads` is the GQA count, not the query-head count.** Llama 3.1 8B has
  32 query heads but only 8 KV heads — using the query count overstates the KV
  cache by 4×. Architecture metadata comes from Ollama's `/api/show`, so these
  are the model's real numbers, not a guess from its name.
- **bits-per-weight is above the nominal bit count.** k-quants store scale
  factors and keep sensitive tensors wider, so Q4_K_M is ~4.85 bpw, not 4.0.

`--kv q8_0` / `--kv q4_0` model llama.cpp's `--cache-type-k`/`--cache-type-v`,
which halve and quarter the KV cache respectively. `lmm route` now uses these
numbers too: instead of "GPU is under 80% used", it reports which installed
model actually fits in the free VRAM.

GPU detection covers NVIDIA (`nvidia-smi`), AMD (`rocm-smi`) and Apple Silicon
(unified memory, reported at a conservative 75% usable share).

### Streaming

The hub speaks SSE, so `stream: true` works with any OpenAI-compatible client.
The three cost layers have genuinely different relationships with streaming, and
the hub is explicit about each rather than silently breaking one:

| Situation | What the client gets |
|-----------|----------------------|
| Plain request | **True pass-through** — the provider's frames are relayed byte-for-byte, so time-to-first-token is real |
| Cache hit | The stored answer replayed as a synthetic stream — no network, $0 |
| `lmm_cascade` on | Buffered, then replayed as a stream. The verifier has to see the whole answer before it can score it; that's the honest price of scoring |

Details that matter for a proxy:

- **Failover ends at the first byte.** Once output is on the wire a retry would
  duplicate it, so a mid-stream upstream failure is reported in an error frame
  rather than silently retried on another provider.
- **A client that hangs up mid-stream is still metered**, flagged `partial`.
  Those tokens were generated and billed whether or not anyone read them.
  Partial answers are never cached.
- **`stream_options.include_usage` is always requested upstream** so streamed
  calls can be metered — but the resulting usage chunk is withheld from clients
  that didn't ask for it. When a provider ignores the option, tokens are
  estimated and the event is flagged `estimated`.
- **A provider that ignores `stream: true` entirely** — answering with an
  ordinary JSON body — used to look like a clean empty stream: no output, no
  metering, no failover, no explanation. It's now reported as an error, so the
  request fails over to the next provider instead of going silent.

`lmm cost` keeps these distinctions rather than blending them into one number:
a line labelled `HUB MEASURED TOTAL` breaks out `of which ESTIMATED` (tokens
inferred, not reported) and `of which PARTIAL` (streams the client abandoned —
still billed upstream, so still counted), and reports p50/p90 TTFT across
genuinely streamed calls. Buffered cascade responses are excluded from that
TTFT series: they have a first-*byte* time but not a first-*token* one.

### `lmm bench` — the third axis

Price is one axis and memory is another; latency is the third, and it doesn't
follow from either. The standard decomposition from the LLM-serving literature
([Orca](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022;
[vLLM/PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023):

```
TTFT   time to first token       — prefill, compute-bound
TPOT   time per output token     — decode, memory-bandwidth-bound
e2e    = TTFT + TPOT × (tokens − 1)
tok/s  = tokens / e2e
```

TTFT and TPOT are reported separately because prefill and decode have different
bottlenecks — [DistServe](https://arxiv.org/abs/2401.09670) goes as far as
running the two phases on separate hardware for exactly this reason.

```bash
$ lmm bench --runs 3
  provider                        TTFT      TPOT     tok/s       e2e
  local-ollama                     92ms      21.3ms      46.1     2813ms
  openai                          412ms       8.1ms     121.4     1447ms
                             $0.01000 per 1k output tokens
```

The first call is a discarded warm-up: it pays for model load and connection
setup, so including it would measure your machine's cold start rather than its
steady-state serving speed. Results are medians over the measured runs, with
the TTFT range shown so a single lucky sample can't masquerade as a result.

### The taskbar problem, solved

- **One-off:** `lmm hide claude` strips Claude's taskbar button immediately
  (the app keeps running). Works for ChatGPT, Cursor, Perplexity, GPT4All,
  AnythingLLM, Chatbox, Msty, Devin/Cua too.
- **Permanent (recommended):** `lmm autostart` registers a background watcher
  that hides *any* new LLM window the moment it appears. You never touch the
  taskbar again. Disable by deleting the scheduled task / launchd plist /
  systemd unit.

### Examples

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

Detection combines three signals: a running process, an **open API port**, and
an install footprint on disk. The ports are each project's documented default:

| Runtime | Port | | Runtime | Port |
|---|---|---|---|---|
| Ollama | 11434 | | KoboldCPP | 5001 |
| LM Studio | 1234 | | vLLM | 8000 |
| Jan | 1337 | | llama.cpp server | 8080 |
| GPT4All | 4891 | | Open WebUI | 8080 |
| AnythingLLM | 3001 | | | |

Two honesty rules. **llama.cpp and Open WebUI share 8080**, so an open socket
alone can't say which is there — those are only reported as serving when a
matching process is running too. And the desktop-only apps (ChatGPT, Cursor,
Perplexity, Chatbox, Msty, Devin) get **no invented endpoint**: they have no
documented local API, and printing a plausible-looking one would be confident
nonsense.

Probing is a TCP connect, not an HTTP call — no auth, no model load — and all
16 run concurrently, so `lmm discover` finishes in about 0.1s.

## Configuration

Works with zero config. `~/.lmm/config.json` (or `~/.config/lmm/config.json`,
or `./lmm.config.json`) can add runtimes, override pricing, change routing
keywords, or configure the hub. See `lmm examples` for the full shape —
`config.example.json` is generated from it, so the two never drift.

| Key | Purpose |
|-----|---------|
| `providers` | Your backends. Optional `price` (a rate-table key or `{in,out}`) makes the cost numbers exact |
| `ask_order` | Provider priority. Set it and lmm follows it exactly |
| `route_threshold` | RouteLLM's `α`, default `0.5`. `null` disables auto-routing |
| `cascade` | `{enabled, rungs, threshold, max_rungs, judge}` — omit `rungs` and they're built cheapest-first automatically |
| `cache` | `{enabled, semantic, similarity, ttl_hours, max_entries, embed_model, max_temp}` |
| `pricing` / `route` / `usage` / `extra_runtimes` | As before |

## What lmm stores

Two append-only JSONL files under `~/.lmm/`, both plain text you can `cat`:

- `usage.jsonl` — one line per call: provider, model, tokens, USD, cache
  status, cascade rung. This is what `lmm cost` reads, and what makes
  `--days N` meaningful.
- `cache.jsonl` — cached answers (and embeddings, if the semantic tier is on).
  Drop it any time with `lmm cache --clear`.

Still no secrets: API keys live in your config, are used, and are never copied
into either file.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Stdlib `unittest`, no network, no fixtures to install — the same
zero-dependency rule the tool follows. `lmm.py` stays a single distributable
file; the tests live outside it.

## How it works

| Concern        | Approach                                                        |
|----------------|-----------------------------------------------------------------|
| Discovery      | Every registry runtime: process + TCP port probe + data dir, all concurrent |
| Secrets        | Only *check* existence of existing credentials; never store one |
| Cost           | Real tokens from `~/.claude/projects/*.jsonl` × public pricing, plus lmm's own metered hub calls |
| Routing        | Lexical strength score vs a cost threshold; your `ask_order` overrides it |
| Cache          | sha256 of the normalized conversation; optional local-embedding similarity tier |
| Taskbar hide   | `WS_EX_TOOLWINDOW` on visible windows (Win); headless launch advised for servers |
| Auto mode      | `watch` daemon hides new LLM windows every few seconds          |
| Portability    | `expanduser` paths, `tasklist`/`pgrep`, `tkinter` for the GUI   |

## License

MIT — see [LICENSE](LICENSE).
