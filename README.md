# lmm — Local/remote Model Manager & Unified Inference Hub

> Three files, no dependencies. One endpoint in front of every LLM you use —
> and a bill you can actually see.

`lmm` is a tiny, cross-platform manager for **all** your LLM runtimes: local
(Ollama, LM Studio, Jan, GPT4All, KoboldCPP, vLLM, llama.cpp, Open WebUI) and
remote/desktop (Claude, ChatGPT, Cursor, Perplexity, AnythingLLM, Chatbox,
Msty, Devin/Cua). It finds what's running, tells you which models actually fit
in your GPU, and puts one OpenAI-compatible endpoint in front of all of them —
caching, routing and cascading each request to the cheapest model that can
handle it, and metering every call so the savings are measured rather than
claimed. On Windows it also keeps those apps' windows off your taskbar.

## Why this exists (first principles)

Running LLMs locally and remotely at the same time creates four problems, and
`lmm` is organised around them:

1. **You cannot see what you are spending.** Cloud spend arrives a month late
   and local inference looks free until it isn't your GPU. So every call `lmm`
   makes is metered to a file you can `cat`, and anything it could not measure
   is labelled as an estimate rather than blended into a total.
2. **The strongest model is the wrong default.** Most prompts do not need it.
   `lmm` routes on a cost threshold, runs cheap models first and escalates only
   on a bad answer, and reuses answers it already paid for.
3. **You should not have to remember which backend to call.** `lmm` holds the
   routing brain — `ask_order`, automatic fallback, and a reply check that can
   fall through a backend that answered badly — so one prompt reaches the right
   place whether you type `lmm ask`, `lmm chat`, or point an app at
   `lmm serve --hub`.
4. **State must be visible** (Nielsen heuristic #1 — *Visibility of System
   Status*), so you should not have to remember commands to see it. Hence the
   dashboard, `lmm log` and `lmm stats` — and, on Windows, where each LLM app
   grabs its own taskbar button, the ability to keep them out of your way.

## Install

```bash
# clone (or download the three files) and install
git clone https://github.com/shizukutanaka/lmm && cd lmm
./install.sh
# Windows:  powershell -ExecutionPolicy Bypass -File install.ps1

# no clone? fetch the three files into one directory and run it in place
for f in lmm.py backend.py frontend.py; do
  curl -fsSLO https://raw.githubusercontent.com/shizukutanaka/lmm/main/$f
done
python lmm.py <command>
```

Requires only Python 3.8+ (tkinter optional, for the GUI). No `pip install`.

`lmm` is three files, not one: `lmm.py` is a thin entry point over `backend.py`
(the engine — no CLI, no GUI, so it can be tested without a terminal or a
display) and `frontend.py` (argument parsing, the `cmd_*` handlers and the
dashboard). **They must stay in the same directory.** The installer places all
three in `~/.local/share/lmm` and puts a small launcher on your PATH; if you
copy files by hand and miss one, `lmm` says which one rather than throwing an
import traceback.

## Usage

```
lmm                 -> live GUI dashboard (default; falls back to text status headless)
lmm discover        -> list every detected runtime (--save seeds ask_order)
lmm status          -> runtimes + GPU + hub/cache/breaker summary
lmm models          -> models on every running runtime and configured provider
lmm pull <model>    -> pull a model into the local Ollama store
lmm cost [--days N] -> measured spend: Anthropic logs + lmm's own hub telemetry
lmm route "task"    -> recommend local vs remote (--explain for the score)
lmm priority        -> show ask_order, or --optimize it from measured results
lmm fit [model|f.gguf] -> does it fit in your GPU, and at what context length?
lmm bench           -> measure TTFT / TPOT / throughput per provider

# --- asking ---
lmm ask "prompt"    -> one question, cached, routed, optionally cascaded
                       (--cascade, --verify, --explain, --no-cache)
lmm chat            -> interactive REPL; same hub path, so turns are cached+metered
lmm serve <model>   -> pull + expose a local model endpoint (Ollama)
lmm serve --hub     -> OpenAI-compatible proxy over every configured provider
lmm cache           -> prompt-cache stats (--clear to drop it)
lmm stop <runtime>  -> stop a running runtime
lmm hide <runtime>  -> strip a runtime's taskbar button (Claude/ChatGPT/...)
lmm watch           -> background daemon: auto-hide new LLM windows
lmm autostart       -> register `watch` to run at OS login (zero effort)
lmm dash            -> generate + open a self-contained HTML dashboard
lmm gui             -> open the live GUI dashboard (Windows: minimizes to tray)
lmm config <init|list|get|set|unset> [key] [value] -> manage hub settings (CLI)
lmm examples        -> print a sample config file
```

### The hub: one endpoint, and a cheaper bill

Point your apps at `lmm serve --hub` and every request goes through one path —
cache, routing, cascade, metering — across all your local and cloud backends.
`GET /v1/models` lists the **real model ids** of every reachable backend, with
each provider's name kept as a routable alias. Ask for a real id and the hub
routes to the provider that serves it and forwards **that exact model** — a
proxy that silently substitutes some default for the model you named is lying
to you, so naming a model nobody serves is a clear 400, not a quiet fallback.

That claim is tested against the **real `openai` client library**, not just
curl: non-streaming, streaming (where a framing mistake shows up as zero
chunks and no error), the opt-in usage chunk, model listing, and the 401 →
`AuthenticationError` path all run through the actual SDK. Those tests skip
automatically where the SDK isn't installed, so the suite itself stays
zero-dependency.
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

#### Verified mode: let each entry earn the right to answer

A static threshold answers *"are these two prompts close?"* when the question is
*"would this cached answer still be right?"* — and the similarity at which that
flips is different for every prompt. Set `cache.max_error_rate` and lmm switches
to vCache's approach instead: **a per-entry threshold, learned online.**

```json
"cache": { "semantic": true, "max_error_rate": 0.05 }
```

An entry may only answer once there is statistical evidence its error rate is
under your bound. Until then a near neighbour triggers **exploration**: lmm pays
for the real answer, compares it against the one the neighbour would have given,
and records the outcome on that entry. Agreement is measured by embedding both
answers — locally, so labelling is free.

That loop is the whole mechanism, and it separates the two cases a static
threshold cannot tell apart:

| Neighbour | Static threshold | Verified mode |
|---|---|---|
| Genuinely interchangeable | served from request 1 | explores, then serves free once certified |
| Similar wording, different answer | **served from request 1, wrong every time** | never certified, never served |

`lmm cache` reports how many entries are certified and how many observations
back them.

Two honest notes on the implementation. The labels come from comparing answer
embeddings, which is a proxy for correctness, not correctness itself. And the
bound is a [Wilson score](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)
lower bound rather than the paper's calibrated Bayesian posterior — so the
guarantee is "there is statistical evidence the error rate is below δ", not the
paper's tighter one. Wilson is used precisely because the naive estimate reads
2-out-of-2 as 100% correct, which would certify an entry on two lucky draws.

#### Tool calls are answers too

A reply that calls a tool sets `content` to `null` and puts its payload in
`tool_calls`. Scoring that as an empty answer made the cascade escalate through
*every* rung on a request the cheapest model had already answered correctly —
turning a cost reducer into a cost multiplier on exactly the agent traffic a hub
proxies most. So a well-formed tool call is scored as the complete answer it is.

The text checks (length, hedging, truncation) mean nothing for a tool call, so
they aren't applied. What is checked is the real failure mode: a missing function
name, or **arguments that don't parse as JSON** — small models emitting plausible
names with truncated structured output is precisely what constrained decoding
exists to prevent, and it's the tool-call analogue of an unclosed code fence.
A malformed call still escalates; a valid one doesn't.

Tool arguments are billed output tokens even though they never appear in
`content`, so they count toward metering and toward `bench`'s TTFT.

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
  cache by 4×. Architecture metadata comes from the model itself, so these are
  its real numbers, not a guess from its name.
- **bits-per-weight is above the nominal bit count.** k-quants store scale
  factors and keep sensitive tensors wider, so Q4_K_M is ~4.85 bpw, not 4.0.

`--kv q8_0` / `--kv q4_0` model llama.cpp's `--cache-type-k`/`--cache-type-v`,
which halve and quarter the KV cache respectively. `lmm route` now uses these
numbers too: instead of "GPU is under 80% used", it reports which installed
model actually fits in the free VRAM.

#### Point it at a `.gguf` file

`fit` reads Ollama's metadata when you give it a tag, but it also reads **GGUF
files directly** — so it works for LM Studio, llama.cpp, KoboldCPP and Jan
users, and with no runtime running at all:

```bash
$ lmm fit ~/models/llama-3-8b-q4_k_m.gguf --vram 24 --ctx 32768
  [OK  ] llama-3-8b-q4_k_m.gguf         9.03 GiB @ 32,768 ctx
         weights 4.53 (exact from file) + kv 4.00 + overhead 0.50
         (8.03B params, 4.85 bpw ~q4_k_m, 32L, 8 kv-heads x 128)
         fits up to 155,379 tokens of context
```

The file path is the *more accurate* input, not just a convenience. lmm sums
the tensor table for an **exact parameter count** and takes the weights term
from the **real file size**, so nothing is looked up in a bits-per-weight
table — bits-per-weight is instead *measured* (real bytes ÷ real parameters)
and reported as the nearest known quant with a `~`. On this path the only
estimate left in the total is the 0.5 GiB overhead.

Reading stops at the header: metadata and the tensor table, never the weights,
so sizing a 40 GB model touches a few hundred KB. Malformed or hostile files
are refused rather than trusted — the parser caps key counts, tensor counts,
string lengths and array nesting, so a corrupt header can't make it allocate
its way to an OOM. GGUF v1 (32-bit lengths) is rejected with a clear reason.

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

# 3b. Real models, real routing — point any OpenAI SDK at the hub:
$ python -c "from openai import OpenAI; c=OpenAI(base_url='http://localhost:8080/v1',api_key='lmm'); print([m.id for m in c.models.list().data])"
# -> ['qwen2.5-coder:3b', 'qwen2.5-coder:7b', 'qwen2.5-coder:14b']
$ python -c "from openai import OpenAI; c=OpenAI(base_url='http://localhost:8080/v1',api_key='lmm'); print(c.chat.completions.create(model='qwen2.5-coder:3b', messages=[{'role':'user','content':'hi'}]).model)"
# -> qwen2.5-coder:3b   (routed to the exact model you asked for)

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
  [PASS] observability (trail writable)
  [PASS] verify_reply detects hallucination -- hallucinated token
  [PASS] route_and_verify falls back bad->good -- verified ok
  [PASS] doctor command runs -- doctor: HEALTHY
  [PASS] config validate command runs -- config: VALID
  [PASS] secrets command runs -- secrets: CLEAN
  [PASS] stats command runs
  [PASS] priority --optimize runs

SELFTEST PASS — 12 checks, the hub proves itself.
```

`lmm selftest` runs **real measurements** (not trust): it compiles itself, checks
the command surface, probes Ollama, performs a live routed `ask`, confirms
the trail is writable, and runs the diagnostic commands. Non-zero exit on any
failure — usable as a fleet/CI gate. `LMM_SELFTEST_SKIP_LIVE=1` skips the two
checks that need a running backend, which is how CI runs it.

`lmm selftest --guard` is the same run in machine-readable form: it prints
**only failures** and lets the exit code carry the verdict, so a green run is
silent. That is what `guard.sh`, the pre-push hook and the GitHub Actions
workflow invoke.

One distinction matters here: **`doctor` grades your machine, `selftest` grades
lmm.** A laptop with nothing running is an unhealthy machine and a perfectly
working tool, so the gate asserts that `doctor` runs and reaches a verdict — and
prints that verdict — rather than demanding a healthy host. Requiring
`doctor: HEALTHY` made the gate unpassable on any CI runner, which is exactly
where it needs to pass.

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

Edit settings from the CLI with `lmm config` — no hand-editing JSON:

```bash
lmm config init                                   # create ~/.lmm/config.json
lmm config set ask_order '["openai","local-ollama(implicit)"]'
lmm config set providers.openai '{"kind":"cloud","base_url":"https://api.openai.com/v1","api_key":"$OPENAI_API_KEY","model":"gpt-4o"}'
# "$OPENAI_API_KEY" is expanded from the environment when providers are
# resolved, so the secret never sits in the file. If the variable is not
# set, the literal "$NAME" is kept and `lmm doctor` names it — instead of
# the provider's 401 being the only witness.
lmm config get ask_order
lmm config unset providers.openai
```

There are 35 settings in total. **You need five of them**, and only if you
want the hub — everything else has a working default:

| Key | Purpose |
|-----|---------|
| `providers` | Your backends. Without this, `lmm ask` still works against a running Ollama |
| `ask_order` | Provider priority. Set it and lmm follows it exactly, disabling auto-routing |
| `route_threshold` | RouteLLM's `α`, default `0.5`. Lower sends more to cheap models |
| `cascade.enabled` | Run cheap models first and escalate only on a bad answer |
| `cache.enabled` | Reuse answers you already paid for (on by default) |

<details>
<summary><b>The other 30 — reach for these only when you need them</b></summary>

| Key | Purpose |
|-----|---------|
| `providers[].price` | A rate-table key (`"deepseek-chat"`) or `{in, out}`. Makes cost numbers exact instead of matched by model name |
| `cascade.rungs` | Explicit rung order. Omit and they are built cheapest-first automatically |
| `cascade.threshold` / `max_rungs` | Accept score and the ceiling on calls per prompt |
| `cascade.judge` | A provider name to grade answers, on top of the built-in heuristic |
| `cache.semantic` | Fuzzy matching via local embeddings. Off by default — see the verified-mode section for why |
| `cache.max_error_rate` | Switches the semantic tier to vCache's per-entry learned thresholds |
| `cache.confidence` / `min_observations` / `answer_match` | How much evidence certifies an entry |
| `cache.similarity` / `ttl_hours` / `max_entries` / `embed_model` / `max_temp` | Static-threshold tuning, expiry and size |
| `hub.token` / `allow_remote` | Required to bind beyond loopback — see Hub security |
| `retry` | `{attempts, base_ms, cap_ms}` full-jitter backoff per provider |
| `breaker` | `{enabled, threshold, cooldown_s}` circuit breaker for dead providers |
| `pricing` | Override or add rate-table entries |
| `route` | `{private, heavy}` keyword lists. `private` is a hard pin: non-local providers are excluded outright — `ask_order` does not outrank it, a failing local never falls back to the cloud, and with no local provider the request is refused rather than sent |
| `usage` | Hand-entered cloud spend, added on top of what lmm measures itself |
| `extra_runtimes` | Your own runtimes for `discover` and `stop` |

</details>

### Failing over well

Falling through to the next provider on error was the only reliability lmm had,
and it had three holes: a single transient blip abandoned a working provider, a
permanently dead one was re-tried first on *every* request, and `429`'s
`Retry-After` was ignored. Two standard patterns close them:

The breaker's memory outlives the process: state lives in
`~/.lmm/breaker.json`, so five consecutive `lmm ask` runs against a dead
backend pay its timeout at most `threshold` times, not five. Cross-process
writes are last-writer-wins — the same openly-taken trade as the metering
log; the state is advisory, so the worst case is one extra probe. Delete the
file (or wait out `cooldown_s`) when you know the outage is over.

**Retry with full jitter** — a transient failure (`429`, `5xx`, connection
error, timeout) is retried on the *same* provider before failing over, with the
delay drawn uniformly from `[0, min(cap, base·2^attempt)]`. That randomisation
is the point: a fixed exponential schedule makes every client retry in lockstep,
the thundering herd documented in
[Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
(Marc Brooker, AWS) — full jitter is the default in the AWS SDKs for that reason.
A `4xx` is never retried, because it would fail identically. `Retry-After`
(both delay-seconds and HTTP-date forms, per RFC 9110 §10.2.3) is honoured but
capped at `cap_ms`, so a hostile header can't park the hub.

**Circuit breaker** — after `threshold` consecutive failures a provider's
circuit opens and it's skipped for `cooldown_s`, then given one half-open trial;
success closes it. Without this a dead backend charges every single request its
full timeout. The pattern is from Michael Nygard's *Release It!*. The breaker
never skips the last candidate standing — refusing to try anything is worse than
one timeout.

```jsonc
"retry":   { "attempts": 2, "base_ms": 250, "cap_ms": 8000 },  // attempts:1 = off
"breaker": { "enabled": true, "threshold": 3, "cooldown_s": 30 }
```

Breaker state is per-process, so it pays off in the long-lived `serve --hub` and
is harmless in a one-shot `lmm ask`.

## Hub security

The hub proxies to your providers using **your** API keys, so whoever can reach
the port can spend your budget without ever seeing a key. Reachability is
therefore the whole security boundary:

- **Loopback (`127.0.0.1`, the default) is open** — only you can reach it.
- **Binding wider (`--host 0.0.0.0`) is refused** unless you either set
  `hub.token` (the hub then requires `Authorization: Bearer <token>` on every
  request, checked in constant time) or explicitly set `hub.allow_remote: true`
  to accept an unauthenticated bind on a network you trust. The refusal prints a
  ready-to-paste token so the secure path is the easy one.

A 401 from the hub reveals nothing about the expected token, and no error
response ever echoes a provider key.

### A note on working-directory config

`lmm` looks for `lmm.config.json` in the current directory, which is convenient
for per-project settings but means the file can come from any repo you happen to
be inside. Because `extra_runtimes[].models_cmd` is a shell command, honouring
that field from an untrusted directory would be remote code execution on `cd`.
So **`models_cmd` runs only from your own config** (`~/.lmm/config.json` or
`~/.config/lmm/config.json`); from a working-directory file it is ignored with a
warning. Everything else in a local config still applies.

## What lmm stores

Two append-only JSONL files under `~/.lmm/`, both plain text you can `cat`:

- `usage.jsonl` — one line per call: provider, model, tokens, USD, cache
  status, cascade rung. This is what `lmm cost` reads, and what makes
  `--days N` meaningful. Past ~4MB the oldest events are folded into a single
  rollup line whose totals are preserved exactly — so the file stays small
  enough to parse on the GUI's refresh timer, and `lmm cost` reports the same
  numbers before and after. Only per-event detail ages out: TTFT percentiles
  and near-miss similarities always come from the recent raw tail, where
  recency is the point.
- `cache.jsonl` — cached answers (and embeddings, if the semantic tier is on).

Both files are yours to edit, and the readers assume you will: a line that is
not JSON is skipped, and a field with the wrong type (`"usd": "abc"`) costs
only itself — the command still runs and every other line still counts.
Before this was measured, one mangled field killed `lmm cost` and `lmm status`
with a traceback, and one bad timestamp silently discarded every cache entry
after it.
  Drop it any time with `lmm cache --clear`.

Still no secrets: API keys live in your config, are used, and are never copied
into either file.

Writers to both files are serialised within a process (the hub's concurrent
request threads cannot lose each other's events to a compaction or prune). Two
*separate* lmm processes writing at the same instant — say `lmm ask` racing a
running hub's compaction — can still, rarely, drop a statistics line; a
portable cross-process file lock does not exist in the standard library, and
that trade is taken openly rather than papered over.

Version history lives in [CHANGELOG.md](CHANGELOG.md) — useful because
"upgrading" here means re-copying three files with no package manager to tell
you what moved, and the log is how you tell what a newer copy changed.

## Tests & CI

```bash
python3 -m unittest discover -s tests -v
```

Stdlib `unittest`, no network, no fixtures to install — the same
zero-dependency rule the tool follows. The tests live outside the distributable
modules.

The suite carries the claims this README makes, so they cannot quietly stop
being true:

- **Zero dependencies** — every import in all three modules is checked against
  the standard library list. This used to be a CI step that read `lmm.py`
  alone, which after the split would have passed with a third-party import
  sitting in `backend.py`.
- **The install works** — `install.sh` runs into a throwaway `HOME` and the
  resulting command must actually start. The split broke this exact path, and
  nothing caught it.
- **Every subcommand is wired** — the registered subparsers and the dispatch
  branches are compared as sets. `lmm hide` was once a silent no-op and
  `lmm cli` exited 2, both because those two lists disagreed.
- **The layers stay apart** — the engine may not grow a `cmd_*` handler.

Two workflows cover CI. **`.github/workflows/guard.yml`** runs
`lmm selftest --guard`: the tool proving itself on one interpreter, which is
also what `guard.sh` and the pre-push hook run locally. Since `selftest`
runs the unit suite whenever a `tests/` directory sits beside it, all three
gates carry all of it — a checkout cannot be committed, pushed or merged
with a failing test. (Users get three files and no `tests/`, so for them the
check simply does not apply.) The matrix that runs
the suite on Python 3.8 through 3.13 lives at **`ci/github-actions-ci.yml`**
and is not yet in `.github/workflows/`, because pushes from this repo's GitHub
App are rejected without `workflows` permission. Enabling it is a one-line copy
you run yourself:

```bash
mkdir -p .github/workflows && cp ci/github-actions-ci.yml .github/workflows/ci.yml
git add .github && git commit -m "ci: enable GitHub Actions" && git push
```

For contributors there is also a local proof-before-ship gate: enable it once
with `git config core.hooksPath .githooks` (same command on every platform).
`pre-commit` runs `guard.sh` against the working tree, and `pre-push` checks
out the **exact revision being pushed** into a temporary worktree and runs its
selftest there — the working tree may contain unpushed edits that would mask
or cause a failure, so it is the wrong thing to test at push time. Override
with `LMM_SKIP_HOOK=1` when you mean it.

## How it works

| Concern        | Approach                                                        |
|----------------|-----------------------------------------------------------------|
| Discovery      | Every registry runtime: process + TCP port probe + data dir, all concurrent |
| Secrets        | Only *check* existence of existing credentials; never store one |
| Cost           | Real tokens from `~/.claude/projects/*.jsonl` × public pricing, plus lmm's own metered hub calls |
| Routing        | Lexical strength score vs a cost threshold; your `ask_order` overrides it; implicit Ollama safety net |
| Verification   | The reply is scored, not just the pick — `--verify` falls through a backend that answered badly |
| Cache          | sha256 of the normalized conversation; optional local-embedding similarity tier |
| Streaming      | SSE pass-through, stdlib only — token-by-token to the CLI and to proxy clients |
| Observability  | Successes are the metering events themselves; only failures get a trail entry (one writer per fact) — `lmm log`, `lmm stats` and `priority --optimize` join the two |
| Health         | `doctor` grades the machine — including a live probe of each configured provider — and `selftest` grades the tool |
| Taskbar hide   | `WS_EX_TOOLWINDOW` on visible windows (Win); headless launch advised for servers |
| Auto mode      | `watch` daemon hides new LLM windows every few seconds          |
| Portability    | `expanduser` paths, `tasklist`/`pgrep`, `tkinter` for the GUI  |

## License

MIT — see [LICENSE](LICENSE).
