# Changelog

All notable changes to `lmm` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`lmm` ships as plain files with no package manager, so "upgrading" is copying
them over the old ones — and this log is how you tell what a newer copy
changed. As of 1.2.0 there are three of them (`lmm.py`, `backend.py`,
`frontend.py`) and they must stay together; before that there was one. Config
is backward compatible across every version below: new keys have defaults, and
no existing key changed meaning.

## [1.2.0]

Reading models from disk, a cache that can prove it's safe to reuse an answer,
the round of fixes that took the hub from "works" to "holds up under load", and
the merge of the managed-routing line of work into the same tool.

### Changed
- **`lmm` is now three files**: `lmm.py` (entry point) over `backend.py` (the
  engine — no CLI, no GUI, so it is testable without a terminal or a display)
  and `frontend.py` (argument parsing, the `cmd_*` handlers, the dashboard).
  They must live in the same directory. The installers place all three in
  `~/.local/share/lmm` and put a launcher on your PATH.
- `call_provider` takes one signature for every caller: `messages` for a full
  history, `extra` for caller-supplied parameters the hub forwards, and
  `stream=True` for the token-by-token path `lmm ask` and `lmm chat` render.

### Removed
- The merge briefly created two of everything, and the duplicates are gone:
  - **A second router.** Three copies of the same heavy/code/private heuristic
    routed prompts (`prompt_strength`, `score_and_route`, and a third inlined
    in `route_and_verify`), and on the prompts where routing matters most they
    disagreed — "refactor this 500-line module into testable units" scored as
    a *light* task and went to a 3B model. One router remains; `--verify`'s
    closed loop and `priority --optimize` now order candidates through it.
    `ask --auto` is gone with it.
  - **A second observability log.** `hub.log` had no size cap, no compaction
    and no lock — the exact unbounded-growth and lost-write bugs already fixed
    for `usage.jsonl`, alive under another name (measured: 14.7 MB and growing
    at 50k events, vs 3.7 MB capped). The trail now lives in the metering log;
    `lmm log`, `lmm stats` and `priority --optimize` read it from there. An
    old `hub.log` is left where it was but no longer written or read.
  - **Two commands.** `cli` was an alias for `discover`; `hub-status`'s one
    unique capability — probing each configured provider — moved into
    `doctor`, which is where a health probe belongs.
  - The second and third copies of `NAME_TO_KEY` (display name -> provider
    key), which lived inside `resolve_ask_targets` and `optimize_ask_order` —
    the latter's comment admitted it "mirrors" the former. One module
    constant now, and a test forbids a function from growing a private copy.
    The two HTTP model-list readers merged the same way: `fetch_models`
    delegates its OpenAI-compatible branch to `probe_models`, which learned
    to send auth.
  - `cache_prune` (a public wrapper with zero callers anywhere — every
    internal site already goes through the locked variant).
  - `cascade_rungs`'s private branch and its `prompt` parameter: privacy is
    `pin_private`'s job, enforced before rungs are built, and on pinned
    input the branch was measured behaviourally identical to the normal
    path. A second authority that can drift is worse than none — the same
    lesson as the second router.
  - **A second pre-push hook and its scaffolding.** `hooks/pre-push` was the
    unwired ancestor of `.githooks/pre-push`; `setup-hooks.bat` wrapped the
    one command (`git config core.hooksPath .githooks`) that is identical on
    every platform; `lmm_ROADMAP.md` was a 70-feature speculation whose own
    conclusion was that most of it should not be built — and whose "build
    now" list had already been built.

### Added
- The managed-routing commands: `chat` (a REPL that keeps history), `config`
  (init/list/get/set/unset, so settings need no hand-edited JSON), `priority`
  (show `ask_order`, or `--optimize` it from measured results), `pull`, `log`,
  `stats`, `doctor` (which also probes each configured provider), `secrets`,
  and `selftest`.
- `lmm ask --verify` scores the reply after the call and falls through a
  backend that answered badly — the complement of routing, which can only
  guess before the call. It sits alongside the cache/cascade cost path rather
  than replacing it, and orders its candidates through the same router.
- `lmm discover --save` seeds `ask_order` from what is actually running, and
  `lmm models` now lists configured cloud providers as well as local runtimes
  — a cloud backend needs no local process, so discovery alone never saw it.
- Naming a provider that does not exist now prints the names that do.
- The suite carries the README's claims: the zero-dependency rule is checked
  across every module, `install.sh` is run into a throwaway `HOME` and the
  resulting command must start, and the registered subcommands are compared
  against the dispatch branches as sets.
- `lmm fit` reads a `.gguf` file directly (`lmm fit ./model.gguf`), so
  LM Studio / llama.cpp / KoboldCPP users can size a model with no runtime
  running. Weights and parameter count come from the file itself — only the
  ~0.5 GiB overhead is an estimate. Validated against the published
  Llama-3-8B figure (4.0 GiB KV cache at 32K, fp16).
- Verified semantic cache (vCache, arXiv:2502.03771): set
  `cache.max_error_rate` and each entry earns the right to answer by
  accumulating evidence — under a bound — that reusing it was correct, instead
  of trusting one static similarity threshold. Off by default; the static
  threshold remains the default behaviour.
- SSE streaming compatibility is proven against the real `openai` client
  library (non-streaming, streaming, `include_usage`, model listing, and the
  401 path), not just curl. Those tests skip where the SDK is absent, so the
  suite stays zero-dependency.

### Fixed
- **Two thirds of the test suite was spent waiting to stop.** The suite ran
  365 tests in 24.9 s, with an odd cluster of tests at ~0.5 s each. The
  cause was one line: `serve_forever()` inherits a 0.5 s `poll_interval`,
  and `shutdown()` blocks until the loop next looks — so every stub
  backend's teardown cost ~0.48 s, and the suite creates one per test.
  Measured in isolation: 0.451 s at the default, 0.001 s at 0.01 s (450x).
  The fixtures now say what they want: **24.9 s -> 7.4 s (3.4x)** for the
  same 365 tests, zero failures. A structural test forbids the default
  returning — and its own first version was rejected for inspecting only
  *calls*, when the dangerous form is handing `x.serve_forever` to a
  thread as a bare reference; it passed the mutation it was written to
  catch until it learned to see that.
  The hub's own `serve_forever` is left at 0.5 s deliberately, now written
  down rather than inherited: Ctrl-C measured 0.000 s there because SIGINT
  raises through the poll instead of waiting for it, so lowering it would
  buy nothing and only add wakeups.
- **The variety request was half-honoured.** `cache.max_temp` promises
  that an explicitly high temperature means "the caller wants variety, not
  a cache". The store side kept that promise (a hot answer was never
  frozen); the lookup side did not, so a hot repeat of a cached question
  was served the frozen answer — the exact thing the caller asked not to
  get. Both hub paths now bypass the cache for hot requests, with a trace
  line saying so.
- **The privacy pin yielded under pressure — three ways.** "route.private
  pins a prompt to local providers" failed cross-examination badly: with
  `ask_order` set the privacy check was never consulted, so a confidential
  prompt went to whichever cloud the user listed first, with a local model
  sitting right there; without `ask_order` the pin was only a sort, so the
  cloud stayed in the list as a fallback and a failing local leaked the
  prompt; and with no local provider at all it warned — in a trace printed
  after the request had already returned — and sent anyway. One authority
  (`pin_private`) now enforces it on every path (ask, chat, hub, --verify):
  non-local targets are removed, and with nothing local left the request is
  refused outright, because the user opted in by listing the keyword and
  refusing is respecting their own instruction. Proven against a live stub
  that must receive nothing.
- **The circuit breaker was a hub-only story.** "A dead backend stops
  charging every request its full timeout" failed cross-examination twice:
  `lmm ask` never passed a breaker into the request path at all — the
  comment beside HUB_BREAKER claimed ask "makes a fresh one per process",
  and it made none — and the state died with the process, so every CLI run
  re-paid the timeout. The breaker now rides the ask and --verify paths and
  persists to `~/.lmm/breaker.json` (wall-clock cooldowns, last-writer-wins
  across processes, zero writes on healthy traffic). Proven across real
  processes: a second interpreter inherits the first one's open circuit,
  and five asks attempt a dead provider exactly `threshold` times.
- **The documented env-var key pattern sent the literal string.** `lmm
  secrets` prints "move secrets to environment variables; lmm reads them at
  call time", and the README shows `"api_key": "$OPENAI_API_KEY"` — but
  nothing anywhere expanded it. Captured on the wire: `Authorization: Bearer
  $OPENAI_API_KEY`, so following the product's own security advice broke
  auth with no hint why. Expansion now happens at the one gateway from
  config to providers; an UNSET variable stays literal and `lmm doctor`
  reports it by name.
- **Two spenders were off the books.** "Every call lmm makes is metered" was
  false twice over: `ask --verify` called providers directly and metered
  nothing — including rejected replies, which are billed whether the quality
  gate likes them or not — and `lmm bench` spent tokens on every run,
  warm-up included, invisibly. Both now meter under their own sources
  (`verify`, `bench`), so `lmm cost` shows what verifying and measuring have
  cost you.
- **The hub probed liveness on every request — with subprocesses.** Both
  implicit-provider detectors ran per request, even when the request named an
  explicit provider: measured under concurrent load at 2.00 subprocess spawns
  (pgrep/tasklist) plus 1.00 Ollama HTTP probe per request, capping the hub
  at ~153 req/s against a localhost stub. The routing-path wrappers are now
  memoised for 5 s (liveness does not change per-request; death mid-window is
  the circuit breaker's job, a fresh start waits at most the TTL), while
  discover/status/doctor keep reading the detectors directly — for them,
  freshness is the product. Same bench after: **302 req/s**, 0.06 spawns and
  0.03 probes per request, proven by count in a test rather than by timing.
- **A model-id request paid for work it threw away.** Restoring per-model
  routing made every model-id request fetch the backend's model list (one
  full round-trip) and also run the whole default router — an Ollama port
  probe and a `pgrep` — only to discard that result once the id resolved.
  Measured against a localhost stub: P50 22.5 ms vs 12.5 ms for the same
  request by provider name. Model lists are now cached for 30 s per backend
  (a dead backend's empty answer too, for the circuit breaker's reason), and
  the router only runs when it is actually consulted. Same request now:
  **2.0 ms** — faster than the name path, and proven by count, not vibes: a
  test asserts N model-id requests cost exactly one upstream GET.
- **The merge severed two shipped features, and a zero-caller sweep found
  them.** Per-model routing (master's fbbc59e): the hub was back to listing
  provider names instead of real model ids, and a client-picked model id
  neither reached the provider serving it nor was forwarded — worse, an
  unknown model silently routed to some default. `/v1/models` now aggregates
  real ids (provider names stay as routable aliases), a picked id is
  forwarded verbatim to its owner, and an unknown one is a clear 400.
  Minimize-to-tray (master's 35bbf95): `setup_tray` had zero callers, so the
  GUI feature simply vanished; it is wired back on Windows, and on other
  platforms X keeps meaning close — withdrawing with no tray icon to restore
  from would orphan the process.
- **`fetch_models` never worked against this tree's own providers.** Their
  `base_url` convention includes the `/v1` suffix, so appending `/v1/models`
  produced `/v1/v1/models` — a 404 that read as "no models". The tests had
  monkeypatched it, which is how a path bug survives; it now runs against a
  real HTTP stub.
- **The measurement loop was open.** `lmm stats` and `priority --optimize`
  read per-attempt routing outcomes that, after the merge, nothing wrote —
  the readers survived their writer. Failed attempts (which have no usage to
  meter) now get a trail entry from the hub itself, successes are read from
  the metering events that already existed, and the loop is proven by a
  round-trip test: a real `ask` against a live stub must show up in
  `measure_performance()`.
- **`lmm chat` was invisible to `lmm cost`.** It called providers through its
  own private fallback loop — no cache, no metering, no retry, no breaker —
  while the README claimed one path for ask/chat/hub. Chat now streams
  through `hub_stream`, the same generator the hub server uses, and a test
  asserts a chat turn produces a metering event.
- **The cascade accepted hallucinations only the verify gate could see.**
  Script-fused tokens ('propagレーション') were detected
  by `verify_reply` but not by the cascade's scorer, so the cost path could
  accept an answer the quality path rejected. One grader now serves both —
  `verify_reply` is a threshold gate over `verify_answer` — and the fusion
  check applies only to non-Japanese conversations: the old version rejected
  ordinary Japanese technical writing ('APIキー',
  'Pythonコード') as garbage.
- **Splitting the file broke every install.** `install.sh` and `install.ps1`
  still copied `lmm.py` alone, and the README still told you to `curl` it on
  its own, so a fresh install died on `from backend import *` before printing
  anything. The installers now ship all three modules, and an entry point that
  cannot find its siblings names the missing file instead of raising
  `ModuleNotFoundError`.
- **The CI gate could not pass on CI.** `lmm selftest` required
  `doctor: HEALTHY`, but `doctor` grades the *machine* — a runner with no
  backend running is an unhealthy machine and a working tool, which is why the
  two live checks are already skipped there. The gate now asserts that `doctor`
  runs and reaches a verdict, and prints that verdict, so an unhealthy host is
  still visible without failing the build.
- **The pre-push guard blocked every push, worked on exactly one machine,
  and silently skipped its own tree check.** It tested each pushed `.py`
  file as a lone blob — a premise that died with the three-file split, since
  a lone `lmm.py` exits 1 by design; it hardcoded one specific machine's
  temp directory, so it only ran at all on that machine; and it looked for
  `guard.sh` inside `.githooks/`, where it has never been, so the tree-level
  check silently never executed. It now checks out the exact pushed revision
  into a temporary git worktree and runs the selftest there — portable, and
  proven both ways: a good revision passes, a tree with a module deleted is
  blocked.
- **`selftest --guard` ignored its own contract.** It documented "exit code
  only" and then printed the full banner. It now prints failures only, so a
  green run is silent and a red one still says what broke.
- **Cascade treated a tool call as an empty answer.** A reply that calls a
  tool leaves `content` null, so the verifier scored it 0 and escalated
  through every rung — up to ~460× the cost of the one correct call it had
  already received at rung 0. Well-formed tool calls now score as the complete
  answers they are; a call with unparseable JSON arguments still escalates.
- **Concurrent writers lost metering events.** The usage log's compaction and
  the cache's prune are read-modify-replace rewrites; under the hub's
  concurrent request threads an append landing mid-rewrite was erased —
  measured at 21.9% of events lost under 8 writers. In-process locks bring
  that to 0.0%; readers stay lock-free.
- **The usage log grew without bound**, and every reader (the GUI, on a 5s
  timer) parsed all of it — so the tool that shows spend got slower the more
  it was used. Past ~4 MB the old events fold into one rollup line whose
  totals are preserved exactly; the file drops ~92% and `lmm cost` reports
  identical numbers.
- **Headless first contact showed nothing.** The default command is the GUI;
  on a server, container or ssh session that printed one line about a missing
  toolkit and exited. It now falls back to the text status.
- **A closed pipe was a crash.** `lmm discover | head` ended in a
  BrokenPipeError traceback; a reader hanging up early is normal, and now
  exits 0 (Ctrl-C exits 130).

## [1.1.0]

The presentation, daemon and packaging layers catch up to the engine, plus the
reliability and detection work.

### Added
- Retry with full-jitter backoff per provider (honouring `Retry-After`) and a
  circuit breaker that skips a provider after repeated failures, so one dead
  backend stops costing every request its timeout.
- `discover` actually detects all 16 registry runtimes (was 4) by combining a
  process check, an open-port probe and an install footprint — concurrently,
  in ~0.1s. GPU detection covers AMD (`rocm-smi`) and Apple Silicon alongside
  NVIDIA.
- The HTML dashboard and `lmm status` show real telemetry — measured hub
  spend, cache savings, stream TTFT percentiles, breaker state — instead of a
  text dump, and a `Serving` column distinct from `Running`.
- A GitHub Actions workflow (`ci/github-actions-ci.yml`) running the suite on
  Python 3.8–3.13, a zero-config smoke test, and a mechanical check that every
  import resolves to the standard library.

### Changed
- `lmm models` lists models on every running runtime, not only Ollama.
- `route` gives one answer instead of two that could contradict each other.

### Fixed
- The GUI cost label showed prose or an illustrative estimate rather than a
  number; `lmm hide` was a silent no-op; `lmm cli` exited 2; `cmd_cache`
  ignored the user's config. The `cmd_watch` daemon leaked window handles and
  silently stopped hiding windows over time. The GUI froze on every refresh by
  doing subprocess and log work on the Tk thread.
- `measured_tokens` double-counted usage nested in a record.
- `install.ps1` appended a duplicate alias on every run.

### Security
- Closed a working-directory config RCE: `./lmm.config.json`'s `models_cmd`
  ran as a shell command, so any repo you `cd` into could execute code when
  you typed `lmm`. It now runs only from a trusted (home) config.
- The `serve --hub` proxy holds your API keys; binding beyond loopback now
  requires a bearer token or an explicit opt-in, instead of being an open
  relay.

## [1.0.0]

Initial public shape: the one-endpoint cost hub.

### Added
- `serve --hub`: one OpenAI-compatible endpoint in front of every configured
  local and cloud backend, sharing a single request path with `lmm ask`.
- The cost-reduction trio, stdlib-only: prompt cache (GPT Semantic Cache,
  arXiv:2411.05276), cost-threshold routing (RouteLLM, arXiv:2406.18665), and
  a cheap-first cascade (FrugalGPT, arXiv:2305.05176).
- Metering: every call recorded to `~/.lmm/usage.jsonl` with tokens and USD,
  so `lmm cost` reports measured spend and anything estimated is labelled as
  such.
- `lmm fit` sizes a model against free VRAM from the KV-cache arithmetic
  (GQA-aware), and `lmm bench` measures TTFT / TPOT / throughput per provider.
- Runtime discovery, taskbar hiding (Windows), and the tkinter dashboard.

[1.2.0]: https://github.com/shizukutanaka/lmm/releases/tag/v1.2.0
[1.1.0]: https://github.com/shizukutanaka/lmm/releases/tag/v1.1.0
[1.0.0]: https://github.com/shizukutanaka/lmm/releases/tag/v1.0.0
