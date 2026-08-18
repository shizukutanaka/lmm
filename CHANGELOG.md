# Changelog

All notable changes to `lmm` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`lmm` is a single file, so "upgrading" is re-downloading `lmm.py`. This log is
how you tell what a newer copy changed. Config is backward compatible across
every version below: new keys have defaults, and no existing key changed
meaning.

## [1.2.0]

Reading models from disk, a cache that can prove it's safe to reuse an answer,
and the round of fixes that took the hub from "works" to "holds up under load".

### Added
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
