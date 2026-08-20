#!/usr/bin/env python3
# frontend.py - everything a person touches
# ---------------------------------------------------------------------------
# Argument parsing, the cmd_* handlers, and the tkinter dashboard. All of the
# engine lives in backend.py; this layer only decides what to show and what to
# ask for. Keeping the two apart means the engine can be tested without a
# terminal or a display, and this file can change how something is presented
# without touching how it is computed.
#
# Note on `backend.x` vs a bare `x`: `from backend import *` copies values into
# this module at import time, so rebinding backend.x afterwards would not be
# seen here. Anything a test (or `lmm selftest`) substitutes at runtime is
# therefore reached through the module, deliberately.
# ---------------------------------------------------------------------------
import backend
from backend import *

# ------------------------------ routing ------------------------------------
def cmd_route(cfg, task, explain=False):
    """Recommend local vs remote, and with --explain show the RouteLLM-style
    strength score that drives provider ordering."""
    print(f"task: {task}\n=> recommend: {route_task(cfg, task)}")
    if not explain:
        return
    score, feats = prompt_strength(cfg, task)
    thr = cfg.get("route_threshold", DEFAULT_ROUTE_THRESHOLD)
    print("")
    print("strength features (RouteLLM-style, arXiv:2406.18665):")
    for label, w in feats or [("(no features matched)", 0.0)]:
        print(f"  {label:38} {w:+.2f}")
    if is_private(cfg, task):
        print("  private keyword -> local pinned regardless of score")
    rel = ">=" if thr is not None and score >= thr else "<"
    print(f"  strength s = {score:.2f}  {rel}  threshold {thr}")
    targets = resolve_ask_targets(cfg, task, None)
    if targets:
        mode = "strong-first" if thr is not None and score >= thr else "weak-first"
        if cfg.get("ask_order"):
            mode = "user ask_order (auto-routing deferred)"
        print(f"=> {mode}: "
              + ", ".join(n for n, _ in backend.order_targets(cfg, task, targets)))
    else:
        print("=> no providers configured; add 'providers' or start Ollama")
    fit = best_local_fit()
    if fit:
        print(f"   largest installed model that fits: {fit['model']} "
              f"({fit['gib']:.1f} of {fit['budget_gib']:.1f} GiB free "
              f"at {fit['ctx']:,} ctx)")


# ------------------------------ dashboard -----------------------------------
def build_dash(cfg):
    items = backend.discover(cfg)
    gpu = gpu_info()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = ""
    for it in items:
        cls = "on" if it["running"] else "off"
        paid = "PAID" if it.get("paid") else "free"
        models = ", ".join(it.get("models", [])) or "-"
        # `running` means "a window OR a port" — a GUI app with no API and a
        # headless server look identical under it. `serving` (the port answers)
        # is the signal that actually decides whether lmm can route to it.
        serving = "YES" if it.get("serving") else "-"
        rows += (f'<tr class="{cls}"><td>{html.escape(it["name"])}</td>'
                 f'<td>{it["type"]}</td><td>{paid}</td>'
                 f'<td>{"YES" if it["running"] else "no"}</td>'
                 f'<td>{serving}</td>'
                 f'<td>{it.get("procs", 0)}</td><td>{html.escape(models)}</td>'
                 f'<td>{html.escape(str(it.get("endpoint", "-")))}</td></tr>')
    gpu_html = (f'{gpu["name"]} {gpu["used"]}/{gpu["total"]} MiB ({gpu["pct"]}%)'
                if gpu else "n/a")

    # Telemetry as structured cards, sharing hub_cost_stats with the text
    # report. The previous dashboard pasted cost_report()'s entire formatted
    # text into one <div> — a metered hub deserves numbers, not prose.
    card_html = ""
    for label, value, note in dash_cards(cfg):
        card_html += (f'<div class="card"><div class="clabel">{html.escape(label)}</div>'
                      f'<div class="cval">{html.escape(value)}</div>'
                      f'<div class="cnote">{html.escape(note)}</div></div>')
    summary = html.escape(cost_summary(cfg))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>LMM Dashboard</title><style>
body{{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0d1117;color:#e6edf3}}
h1{{padding:16px 20px;margin:0;font-size:18px;background:#161b22;border-bottom:1px solid #30363d}}
.meta{{padding:8px 20px;color:#8b949e;font-size:13px}}
.gpu{{padding:8px 20px;color:#7ee787;font-size:13px}}
.summary{{padding:4px 20px 8px;font-size:13px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 2%}}
.card{{flex:1 1 170px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 14px}}
.clabel{{color:#8b949e;font-size:12px}}
.cval{{color:#7ee787;font-size:20px;font-weight:600;margin:4px 0}}
.cnote{{color:#8b949e;font-size:11px}}
table{{border-collapse:collapse;width:96%;margin:12px 2%}}
th,td{{border:1px solid #30363d;padding:8px 10px;font-size:13px;text-align:left}}
th{{background:#161b22}}
tr.on td:first-child{{color:#7ee787}} tr.off{{opacity:.55}}
</style></head><body>
<h1>🧠 LMM — Local/remote Model Manager</h1>
<div class="meta">generated {now}</div>
<div class="gpu">GPU: {gpu_html}</div>
<div class="summary">{summary}</div>
<div class="cards">{card_html}</div>
<table><tr><th>Runtime</th><th>Type</th><th>Cost</th><th>Running</th>
<th>Serving</th><th>Procs</th><th>Models</th><th>Endpoint</th></tr>
{rows}</table>
</body></html>"""


def dash_cards(cfg):
    """Telemetry cards as (label, value, note) rows, from the same
    hub_cost_stats the text report reads — one aggregation, two renderers."""
    st = hub_cost_stats()
    cards = [("Hub spend",
              f"${st['measured']:,.4f}" if st["calls"] else "$0",
              f"{st['calls']} metered call(s)" if st["calls"] else "no calls yet")]
    hits = st["hits"]["exact"] + st["hits"]["semantic"]
    if hits:
        cards.append(("Cache savings", f"${st['saved_cache']:,.4f}",
                      f"{hits} hit(s): {st['hits']['exact']} exact / "
                      f"{st['hits']['semantic']} semantic"))
    if st["ttfts"]:
        cards.append(("Stream TTFT", f"{median(st['ttfts']):.0f} ms p50",
                      f"p90 {percentile(st['ttfts'], 90):.0f} ms over "
                      f"{len(st['ttfts'])} stream(s)"))
    if st["local_calls"]:
        cards.append(("Local (free)", f"{st['local_calls']} call(s)",
                      f"{st['local_tokens']:,} tokens run at $0"))
    if st["est_usd"] or st["partial_usd"]:
        note = []
        if st["est_usd"]:
            note.append(f"${st['est_usd']:,.4f} estimated")
        if st["partial_usd"]:
            note.append(f"${st['partial_usd']:,.4f} partial")
        # kept apart on purpose: a card labelled measured must not hide guesses
        cards.append(("Not clean measurements", " · ".join(note),
                      "usage inferred, or stream abandoned mid-flight"))
    tripped = [n for n in list(HUB_BREAKER._open_until)
               if HUB_BREAKER.state(n) != "closed"]
    if tripped:
        cards.append(("Circuit breaker", f"{len(tripped)} open",
                      ", ".join(sorted(tripped))))
    return cards


# ------------------------------ commands -----------------------------------
def cmd_discover(cfg, as_json, save=False):
    items = backend.discover(cfg)
    if as_json:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return
    if save:
        # Seed ask_order from what is actually running, so a first-time user
        # gets a working priority list without hand-writing one.
        running = [it["name"] for it in items
                   if it["running"] and it["name"] != "-"]
        if not running:
            print("[discover] no running backends detected - nothing to save.")
            return
        cfg = dict(cfg)
        cfg["ask_order"] = running
        path = save_config(cfg)
        print(f"[discover] saved {len(running)} backend(s) to ask_order:")
        for i, n in enumerate(running, 1):
            print(f"  {i}. {n}")
        print(f"[discover] config written: {path}")
        return
    for it in items:
        flag = "ON " if it["running"] else "off"
        paid = "PAID" if it.get("paid") else "free"
        extra = ""
        if it.get("models"):
            extra += f" models={', '.join(it['models'])}"
        if it.get("procs"):
            extra += f" procs={it['procs']}"
        if it.get("endpoint") and it["endpoint"] != "-":
            extra += f" @ {it['endpoint']}"
        print(f"[{flag}] {it['name']:<30} {paid:<5}{extra}")


def cmd_status(cfg):
    gpu = gpu_info()
    print("GPU:", gpu["name"] if gpu else "n/a",
          f"({gpu['used']}/{gpu['total']} MiB, {gpu['pct']}%)" if gpu else "")
    print("-" * 64)
    # with_models=False: status does not display model lists, so probing every
    # serving runtime's /v1/models just to discard the result was pure latency.
    for it in backend.discover(cfg, with_models=False):
        print(f"{it['name']:<32} running={it['running']!s:<5} "
              f"serving={it.get('serving', False)!s:<5} "
              f"procs={it.get('procs', 0)}")
    # For a tool whose headline feature is a metered hub, `status` should show
    # the hub's state, not only the runtimes'.
    print("-" * 64)
    print("hub:", cost_summary(cfg))
    conf = merged_cache(cfg)
    entries = cache_entries(conf)
    print(f"cache: {len(entries)} live entries"
          + (" (semantic on)" if conf.get("semantic") else ""))
    tripped = [n for n in list(HUB_BREAKER._open_until)
               if HUB_BREAKER.state(n) != "closed"]
    if tripped:
        print("breaker: open ->", ", ".join(sorted(tripped)))


def cmd_models(cfg=None):
    """Every model you can actually reach, from both directions.

    discover() harvests model lists from any OpenAI-compatible /models endpoint
    on a *running* runtime (LM Studio, Jan, KoboldCPP, vLLM, ...), which is what
    you see locally. Configured providers are a second source: a cloud backend
    is reachable without any local process, so it never shows up in discover.
    Listing only one of the two is the tool disagreeing with itself.
    """
    cfg = cfg or {}
    found = False
    for it in backend.discover(cfg, with_models=True):
        ms = it.get("models") or []
        if not ms:
            continue
        found = True
        print(f"{it['name']}:")
        for m in ms:
            print("  -", m)
    for name, prov in (merged_providers(cfg) or {}).items():
        models = fetch_models(prov)
        if not models:
            continue
        found = True
        print(f"{name} ({prov.get('kind', '?')}):")
        for m in models:
            print("  -", m)
    if not found:
        print("no models found on any running runtime or configured provider "
              "(start one, e.g. `lmm serve <model>`, or see `lmm examples`)")


def cmd_fit(model=None, ctx=None, vram=None, kv="f16", as_json=False):
    """Answer the question every local-LLM user actually has: does this model
    run on this machine, and at what context length?"""
    gpu = gpu_info()
    if vram:
        budget, src = float(vram), "--vram"
    elif gpu:
        budget = (gpu["total"] - gpu["used"]) / 1024.0
        src = f"{gpu['name']} ({gpu['total'] - gpu['used']} of {gpu['total']} MiB free)"
    else:
        print("[fit] no GPU detected and no --vram given. "
              "Pass --vram <GiB> to size a machine you do not have in front of you.")
        return

    models = [model] if model else (detect_ollama().get("models") or [])
    if not models:
        print("[fit] no model given and no Ollama models installed. Try: "
              "lmm fit llama3.1:8b --vram 24, or point it at a file: "
              "lmm fit ./model.gguf")
        return

    rows = []
    for name in models:
        # A path to a .gguf is sized from the file itself — no runtime needed,
        # which is what makes `fit` usable for LM Studio / llama.cpp / KoboldCPP
        # users who have weights on disk but nothing registered with Ollama.
        if looks_like_gguf(name):
            got = read_gguf(name)
            if got.get("error"):
                rows.append({"model": name, "error": got["error"]})
                continue
            spec = got
        else:
            spec = ollama_model_info(name)
        if not spec:
            params = params_from_name(name)
            rows.append({"model": name, "error": (
                "no metadata — start Ollama so `lmm fit` can read the real "
                "layer and KV-head counts, or pass a .gguf path"),
                "weights_only_gib": (params * quant_bpw(None) / 8.0 / GIB
                                     if params else None)})
            continue
        want = ctx or spec["ctx_max"] or 4096
        est = estimate_vram(spec, want, None, kv)
        rows.append({"model": name, "ctx": want, "kv_type": kv, "spec": spec,
                     "est": est, "fits": est["total_gib"] <= budget,
                     "max_ctx": max_context_for(spec, budget, None, kv)})

    if as_json:
        print(json.dumps({"budget_gib": round(budget, 2), "source": src,
                          "rows": rows}, indent=2, default=str))
        return

    print(f"VRAM budget: {budget:.1f} GiB  [{src}]")
    print(f"KV cache dtype: {kv}  "
          f"({KV_BYTES.get(kv.lower(), 2.0)} bytes/element)")
    print("-" * 72)
    for r in rows:
        if r.get("error"):
            print(f"  {r['model']:<32} ? {r['error']}")
            if r.get("weights_only_gib"):
                print(f"  {'':<32}   weights alone ~{r['weights_only_gib']:.1f} GiB "
                      f"(assuming q4_k_m)")
            continue
        e, s = r["est"], r["spec"]
        mark = "OK  " if r["fits"] else "OVER"
        label = os.path.basename(r["model"]) if s.get("source") == "gguf" \
            else r["model"]
        print(f"  [{mark}] {label:<28} {e['total_gib']:6.2f} GiB "
              f"@ {r['ctx']:,} ctx")
        # Say where the numbers came from: a GGUF gives exact weights and an
        # exact parameter count, Ollama metadata gives an estimated weights term.
        origin = ("exact from file" if s.get("source") == "gguf"
                  else "estimated from quant table")
        print(f"         weights {e['weights_gib']:.2f} ({origin}) + "
              f"kv {e['kv_gib']:.2f} + overhead {e['overhead_gib']:.2f}   "
              f"({s['params']/1e9:.2f}B params, {e['bpw']:.2f} bpw"
              + (f" {s['quant']}" if s.get("quant") else "")
              + f", {s['layers']}L, {s['kv_heads']} kv-heads x {s['head_dim']})")
        if r["max_ctx"]:
            print(f"         fits up to {r['max_ctx']:,} tokens of context")
        else:
            print("         weights alone do not fit — use a smaller quant or model")
        if not r["fits"] and kv == "f16":
            alt = estimate_vram(s, r["ctx"], None, "q8_0")
            if alt["total_gib"] <= budget:
                print(f"         -> fits at {alt['total_gib']:.2f} GiB with "
                      f"--kv q8_0 (llama.cpp --cache-type-k/v q8_0)")
    print("-" * 72)
    if any(r.get("spec", {}).get("source") == "gguf" for r in rows):
        print("GGUF rows read weights and parameter count from the file, so only "
              "the\n0.5 GiB overhead is an estimate. KV cache is exact for the "
              "given context.")
    else:
        print("Estimates. bits-per-weight are llama.cpp measurements on "
              "LLaMA-family models;\noverhead is a 0.5 GiB middle estimate for "
              "context and scratch buffers.")


def cmd_serve(model):
    if not model:
        print("usage: lmm serve <ollama-model>  e.g. lmm serve qwen2.5-coder:7b")
        return
    print(f"pulling {model} ...")
    r = backend.run(f"ollama pull {model}")
    print((r.stdout.strip() if r else "(pull failed/timeout)"))
    print("endpoint ready: http://localhost:11434  (OpenAI-compatible)")


def cmd_serve_hub(cfg, host, port, quiet=False):
    """Start an OpenAI-compatible proxy that fans out to every configured
    provider (cloud + local). Apps point at this one endpoint; `lmm` routes
    each request. This is the hub: one endpoint, many backends."""
    import http.server, socketserver, threading, hmac, secrets
    provs = merged_providers(cfg)
    if not provs and not backend.local_ollama_provider():
        print("[hub] no providers configured and no local Ollama running. "
              "Start Ollama (`lmm serve <model>`) or add 'providers' to lmm "
              "config (see `lmm examples`). Nothing to proxy.")
        return

    # The `breaker` config key was documented in `lmm examples` but never read:
    # HUB_BREAKER was built from the defaults at import time, so anyone who set
    # a threshold or cooldown had it silently ignored. Apply it here, where the
    # config is finally in hand.
    brk = merged_breaker(cfg)
    HUB_BREAKER.threshold = int(brk.get("threshold", 3))
    HUB_BREAKER.cooldown_s = float(brk.get("cooldown_s", 30))
    hub_breaker = HUB_BREAKER if brk.get("enabled", True) else None

    hub = merged_hub(cfg)
    token = hub.get("token") or None
    allowed, lines = hub_bind_check(host, hub,
                                   lambda: secrets.token_urlsafe(24))
    for line in lines:
        if not quiet:
            print(line)
    if not allowed:
        return

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, frames):
            """Relay SSE frames. No Content-Length: the body length is unknown
            when the headers go out, which is the whole point of streaming.
            Each frame is flushed so the client sees tokens as they arrive
            rather than in one buffered lump at the end."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for frame in frames:
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass          # client hung up mid-stream; nothing to report
            finally:
                # Closing explicitly runs hub_stream's own finally, which is
                # what records the spend on an abandoned stream. Relying on
                # refcounting to finalise the generator happens to work on
                # CPython but is not a language guarantee.
                close = getattr(frames, "close", None)
                if close:
                    close()

        def _authed(self):
            """Constant-time token check. A plain `==` on a secret leaks its
            length and prefix through timing, and this endpoint is reachable by
            whoever the bind allows."""
            if not token:
                return True
            got = self.headers.get("Authorization", "") or ""
            if got.startswith("Bearer "):
                got = got[7:]
            return hmac.compare_digest(got.strip(), token)

        def _deny(self):
            # 401 with no hint about the expected value.
            self._send(401, {"error": {"message": "missing or invalid bearer "
                                                  "token for the lmm hub",
                                       "type": "invalid_request_error"}})

        def do_GET(self):
            if not self._authed():
                self._deny()
                return
            if self.path.rstrip("/").endswith("/v1/models"):
                # Real model ids from every reachable backend, plus the
                # provider names themselves as routable aliases. Returning
                # only provider names — as this once did — meant a client
                # could never pick between two models on the same backend,
                # the exact stub-ids bug fbbc59e fixed and the merge undid.
                data, seen = [], set()
                for n, p in provs.items():
                    for mid in fetch_models(p):
                        if mid not in seen:
                            seen.add(mid)
                            data.append({"id": mid, "object": "model",
                                         "owned_by": n})
                    if n not in seen:
                        seen.add(n)
                        data.append({"id": n, "object": "model",
                                     "owned_by": p["kind"]})
                self._send(200, {"object": "list", "data": data})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                self._deny()
                return
            if not self.path.rstrip("/").endswith("/v1/chat/completions"):
                self._send(404, {"error": "only /v1/chat/completions supported"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length).decode("utf-8", "ignore"))
            except Exception as e:
                self._send(400, {"error": f"bad json: {e}"})
                return
            # Routing, cascade, cache and metering are the SAME code path as
            # `lmm ask` (hub_complete) — one hub, one routing brain. The whole
            # messages array is forwarded, so system prompts and prior turns
            # survive the hop.
            msgs = req.get("messages", [])
            # `model` may be a provider name (an alias we list in /v1/models)
            # or a REAL model id the client picked from that same listing. A
            # provider name routes to that provider's default model; a model
            # id must reach the provider that serves it AND be forwarded
            # verbatim — the client asked for qwen2.5-coder:3b, not for
            # whatever the provider's config happens to default to.
            explicit = req.get("model", "")
            if explicit and explicit not in provs:
                # A model id: map it to its owner directly. Running the full
                # router first and discarding the result cost every model-id
                # request an Ollama port probe and a pgrep — measured as the
                # whole remaining latency gap after the model-list cache.
                owner = resolve_provider_by_model(provs, explicit)
                if owner:
                    targets = [(owner, dict(provs[owner], model=explicit))]
                else:
                    # The client named a model nobody serves. Answering with
                    # some other model anyway — which this handler used to do
                    # by falling back to default routing — is the one thing a
                    # proxy must never be: a silent substitution.
                    self._send(400, {"error": {
                        "message": "unknown model '%s' — GET /v1/models for "
                                   "what this hub serves" % explicit,
                        "type": "invalid_request_error",
                        "code": "model_not_found"}})
                    return
            else:
                targets = resolve_ask_targets(
                    cfg, messages_text(msgs),
                    explicit if explicit in provs else None)
            if not targets:
                self._send(400, {"error": "no provider available for model '%s'" % explicit})
                return
            no_cache = (bool(req.get("lmm_no_cache"))
                        or self.headers.get("X-LMM-No-Cache") is not None)
            extra = {k: req[k] for k in PASSTHROUGH_KEYS if k in req}
            hub_opts = {"cascade": bool(req.get("lmm_cascade")),
                        "cache": not no_cache, "extra": extra, "source": "hub",
                        "breaker": hub_breaker}
            if req.get("stream"):
                hub_opts["client_usage"] = bool(
                    (req.get("stream_options") or {}).get("include_usage"))
                self._stream(hub_stream(cfg, msgs, targets, hub_opts))
                return
            res, _trace = hub_complete(cfg, msgs, targets, hub_opts)
            if isinstance(res, dict) and res.get("error"):
                self._send(502, {"error": "all providers failed: %s" % res["error"]})
                return
            self._send(200, res)

        def log_message(self, *a):
            pass

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    httpd = S((host, port), Handler)
    quiet or print(f"[hub] OpenAI-compatible endpoint: http://{host}:{port}/v1")
    quiet or print(f"[hub] backends: {', '.join(provs)}")
    quiet or print("[hub] Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("[hub] stopped.")


def cmd_stop(runtime, cfg):
    # Every runtime the registry can detect should also be stoppable —
    # `lmm stop cursor` used to answer "unknown runtime" about an app
    # `lmm discover` had just listed. STOP_TABLE stays for its aliases
    # (e.g. "llama-server", which is not a registry key).
    table = {k: list(v.get("procs", [])) for k, v in RUNTIME_REGISTRY.items()}
    table.update(STOP_TABLE)
    for e in cfg.get("extra_runtimes", []):
        table[e["name"].lower()] = e.get("procs", [])
    names = table.get((runtime or "").lower())
    if not names:
        print("unknown runtime. choices:", ", ".join(sorted(table)))
        return
    for n in names:
        if os.name == "nt":
            r = backend.run(f'taskkill.exe /IM "{n}" /F')
        else:
            r = backend.run(f"pkill -f '{n}' || true")
        ok = (r and r.returncode == 0) if r else False
        print(f"stop {n}: {'ok' if ok else 'no-process-or-failed'}")


def cmd_dash(cfg):
    out = os.path.join(backend.HOME, ".lmm_dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_dash(cfg))
    print("dashboard ->", out)
    try:
        webbrowser.open(out)
    except Exception:
        pass


def prune_seen(seen, live, tick, grace=2):
    """Forget window handles absent for `grace` ticks. Because Windows recycles
    HWNDs, a handle we never forget will eventually match a new window that
    reused it and wrongly skip hiding it. Mutates and returns `seen`."""
    for hwnd in [h for h, t in seen.items() if h not in live and tick - t > grace]:
        del seen[hwnd]
    return seen


def watch_list(cfg):
    """(runtime name, lowercase title keyword) pairs the daemon watches.

    Registry entries plus cfg extra_runtimes — cmd_stop already treats those as
    first class and the daemon should agree. Pure, so the composition is
    testable off Windows.
    """
    out = []
    for rt, entry in RUNTIME_REGISTRY.items():
        for kw in entry.get("titles", []):
            out.append((rt, kw.lower()))
    for e in (cfg or {}).get("extra_runtimes", []):
        for kw in e.get("titles", []) or [e.get("name", "")]:
            if kw:
                out.append((e.get("name", "?"), kw.lower()))
    return out


def watch_new_windows(windows, seen, tick, watchlist):
    """Decide what to act on this tick: [(hwnd, title, runtime)] for windows
    not seen before. Also stamps `seen` so the caller only has to prune.

    Separated from the ctypes calls because the DECISION is what regressed
    before (recycled handles being skipped forever), and the decision needs no
    Windows API to verify.
    """
    fresh = []
    for hwnd, title in windows:
        first = hwnd not in seen
        seen[hwnd] = tick
        if not first:
            continue
        low = (title or "").lower()
        rt = next((r for r, kw in watchlist if kw in low), "?")
        fresh.append((hwnd, title, rt))
    return fresh


def cmd_watch(cfg, interval=3.0):
    """Background daemon: automatically strip the taskbar button from any
    newly-spawned LLM-runtime window. This is the root-cause fix (not a
    per-app band-aid): the user never has to click 'hide' — it happens the
    moment an app launches. Idempotent; safe to leave running.
    Press Ctrl-C to stop."""
    if os.name != "nt":
        print("watch is currently Windows-only (taskbar is a Windows concept).")
        return
    print(f"[lmm watch] auto-hiding new LLM windows every {interval}s. Ctrl-C to stop.")
    # One sweep per tick, not one per registry entry: sixteen full EnumWindows
    # passes every few seconds is fifteen too many, and entries with no titles
    # were enumerating the whole desktop just to match nothing. cfg-defined
    # extra_runtimes participate too — cmd_stop already treats them as first
    # class, and the daemon should agree.
    watchlist = watch_list(cfg)          # (runtime name, lowercase keyword)
    all_keywords = [kw for _, kw in watchlist]
    # HWND -> last tick seen. Windows RECYCLES window handles, so a plain
    # ever-growing "seen" set eventually swallows a brand-new window that
    # happens to reuse an old handle — the daemon silently stops working in
    # exactly the always-on scenario `lmm autostart` sets up. Entries that
    # vanish from the enumeration are dropped so a recycled handle counts as
    # new again.
    seen = {}
    tick = 0
    try:
        while True:
            tick += 1
            current = _enum_windows_by_title(all_keywords)
            live = set(h for h, _ in current)
            for hwnd, title, rt in watch_new_windows(current, seen, tick,
                                                     watchlist):
                try:
                    user32 = ctypes.windll.user32
                    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                    new_ex = (ex | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
                    if new_ex != ex:
                        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_ex)
                        print(f"  auto-hidden '{rt}' window: {title!r}")
                except Exception:
                    pass
            prune_seen(seen, live, tick)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[lmm watch] stopped.")


def cmd_autostart():
    """Register `lmm watch` to start automatically with the OS, so the
    taskbar stays clean with zero user action (root-cause fix, not a band-aid).
    Prefers USER-level mechanisms that need no admin rights:
      Win  -> Startup folder shortcut (no UAC)
      mac  -> launchd agent (RunAtLoad)
      Linux-> systemd --user
    Falls back to Task Scheduler only on Windows if the Startup shortcut fails.
    """
    if os.name != "nt":
        # `watch` exits immediately off-Windows, but launchd KeepAlive and
        # systemd Restart=always would restart it forever — registering a
        # service whose only behaviour is a restart loop. Refuse instead.
        print("autostart registers `lmm watch`, which is Windows-only "
              "(the taskbar is a Windows concept). Nothing to register here.")
        return
    me = os.path.abspath(__file__)
    python_exe = sys.executable
    if os.name == "nt":
        # 1) Startup folder shortcut — user-level, no admin needed
        startup = os.path.join(os.environ.get("APPDATA", backend.HOME),
                               "Microsoft", "Windows", "Start Menu",
                               "Programs", "Startup")
        try:
            os.makedirs(startup, exist_ok=True)
            lnk = os.path.join(startup, "lmm-watch.lnk")
            ps = (
                f'$s=(New-Object -ComObject WScript.Shell).CreateShortcut('
                f"'{lnk}');"
                f"$s.TargetPath='{python_exe}';"
                f"$s.Arguments='\"{me}\" watch';"
                f"$s.WindowStyle=7;"  # 7 = minimized
                f"$s.Save()"
            )
            r = backend.run(f'powershell -NoProfile -Command "{ps}"')
            if r and r.returncode == 0 and os.path.exists(lnk):
                print(f"registered lmm watch via Startup folder (no admin needed): {lnk}")
                return
        except Exception as e:
            pass
        # 2) fallback: Task Scheduler (may need admin)
        cmd = (f'schtasks /Create /TN "lmm-watch" /TR '
               f'"{python_exe} \"{me}\" watch" /SC ONLOGON /F')
        r = backend.run(cmd)
        if r and r.returncode == 0:
            print("registered lmm watch with Windows Task Scheduler (runs at logon).")
        else:
            print("autostart registration failed. Manual option: copy this to your "
                  "Startup folder as a shortcut:\n"
                  f'  {python_exe} "{me}" watch')


def gpu_label(gpu):
    """The GPU line for the GUI header. Pure, so the ⚠ threshold is testable
    without a display."""
    if not gpu:
        return "GPU: n/a"
    text = "GPU: %s %s/%s MiB (%s%%)" % (gpu["name"], gpu["used"], gpu["total"],
                                         gpu["pct"])
    # flag a nearly-full card: the next model load is the one that fails
    return text + "  \u26a0" if gpu["pct"] >= 85 else text


def gui_rows(items):
    """(values, tag) per runtime for the GUI table.

    Lifted out of launch_gui's paint() closure so it can be tested without
    tkinter — which is not installed everywhere, and was the excuse for this
    layer having no coverage while it silently drifted.
    """
    rows = []
    for it in items:
        rows.append(((
            it["name"], it["type"],
            "PAID" if it.get("paid") else "free",
            "YES" if it["running"] else "no",
            "YES" if it.get("serving") else "-",
            it.get("procs", 0),
            ", ".join(it.get("models", [])) or "-",
            str(it.get("endpoint", "-")),
        ), "on" if it["running"] else "off"))
    return rows


def gui_model_choices(items):
    """Dropdown values: every model on every discovered runtime, deduped."""
    models = []
    for it in items:
        models.extend(it.get("models") or [])
    return sorted(set(models))


def launch_gui(cfg):
    """Zero-dependency live dashboard (tkinter). Implements the UX principles
    gathered from the research:
      * Nielsen #1 Visibility of System Status — live status, always informed
      * Norman's Gulf of Evaluation — one glance shows state -> next action
      * Backstage vs Frontstage (NN/g) — show cost/VRAM only when it matters
      * Direct manipulation — buttons do the action immediately
      * Trust = Communication — never stores secrets; states are honest
    """
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as e:
        # `lmm` with no arguments lands here on every server, container and
        # WSL box — environments with no tkinter and no display. The tool's
        # whole thesis is visibility of system status, so complaining about a
        # GUI toolkit the user never asked for and showing NOTHING was the
        # worst possible first contact. Degrade to the text status instead.
        print(f"(no GUI here — tkinter unavailable: {e}; showing text status. "
              "`lmm dash` renders the HTML dashboard.)")
        cmd_status(cfg)
        return
    import threading

    try:
        root = tk.Tk()
    except Exception as e:
        # tkinter installed but no display — ssh sessions and CI runners land
        # here, where the import above succeeds and Tk() is what fails. An
        # uncaught TclError traceback is the same broken first contact as a
        # missing tkinter, so it gets the same graceful exit.
        print(f"(no GUI here — no display: {e}; showing text status. "
              "`lmm dash` renders the HTML dashboard.)")
        cmd_status(cfg)
        return
    root.title("LMM — Local/remote Model Manager")
    root.geometry("920x560")

    # Windows: a system-tray icon, so minimizing keeps lmm running in the
    # background (the taskbar-tidiness thesis applied to lmm itself).
    # setup_tray wires X -> withdraw and returns a cleanup; on non-Windows it
    # returns None and X keeps its default meaning — closing a window must
    # never orphan a process that has no tray icon to bring it back.
    tray_cleanup = setup_tray(root)
    if tray_cleanup is None:
        root.protocol("WM_DELETE_WINDOW", root.destroy)

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # --- top bar ---------------------------------------------------------
    top = ttk.Frame(root)
    top.pack(fill="x", padx=8, pady=6)
    ttk.Label(top, text="🧠 LMM Dashboard", font=("Segoe UI", 14, "bold")).pack(side="left")
    gpu_var = tk.StringVar(value="GPU: …")
    ttk.Label(top, textvariable=gpu_var, foreground="#2e7d32").pack(side="right")
    cost_label = ttk.Label(top, text="", foreground="#1565c0")
    cost_label.pack(side="right", padx=12)

    # --- runtime table ---------------------------------------------------
    cols = ("Runtime", "Type", "Cost", "Running", "Serving", "Procs", "Models",
            "Endpoint")
    tree = ttk.Treeview(root, columns=cols, show="headings", height=12)
    widths = (170, 70, 60, 70, 66, 60, 200, 190)
    for c, w in zip(cols, widths):
        tree.heading(c, text=c)
        tree.column(c, width=w, anchor="w")
    tree.pack(fill="both", expand=True, padx=8, pady=4)

    # --- action bar ------------------------------------------------------
    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=8, pady=6)
    ttk.Label(bar, text="Runtime:").pack(side="left")
    rt_var = tk.StringVar()
    rt_cb = ttk.Combobox(bar, textvariable=rt_var, width=18,
                         values=sorted(RUNTIME_REGISTRY))
    rt_cb.pack(side="left", padx=4)
    ttk.Label(bar, text="Model:").pack(side="left")
    mdl_var = tk.StringVar()
    # Populated from discover() on each refresh — the list used to be four
    # hardcoded Ollama tags, stale the day they were written.
    mdl_cb = ttk.Combobox(bar, textvariable=mdl_var, width=22, values=[])
    mdl_cb.pack(side="left", padx=4)

    def act(fn_name):
        rt = rt_var.get().strip().lower()
        if not rt:
            messagebox.showwarning("LMM", "select a runtime first")
            return
        model = mdl_var.get().strip() or "qwen2.5-coder:7b"

        def work():
            # `serve` shells out to `ollama pull`, which can run for minutes on
            # a cold model. On the Tk thread that is an unresponsive window, so
            # every action runs on a worker and reports back via root.after.
            if fn_name == "hide":
                msg = hide_taskbar(rt)
            elif fn_name == "stop":
                cmd_stop(rt, cfg)
                msg = f"stopped {rt} (if running)"
            elif fn_name == "serve":
                cmd_serve(model)
                msg = f"served {model}"
            else:
                msg = "?"
            root.after(0, lambda: (messagebox.showinfo("LMM", msg), refresh()))
        threading.Thread(target=work, daemon=True).start()

    ttk.Button(bar, text="▶ Serve", command=lambda: act("serve")).pack(side="left", padx=2)
    ttk.Button(bar, text="■ Stop", command=lambda: act("stop")).pack(side="left", padx=2)
    ttk.Button(bar, text="⊟ Hide from taskbar", command=lambda: act("hide")).pack(side="left", padx=2)
    ttk.Button(bar, text="⟳ Refresh", command=lambda: refresh()).pack(side="right", padx=2)

    # --- live refresh ----------------------------------------------------
    def gather():
        """Everything slow: subprocesses, socket probes, and a full walk of the
        session logs. Runs on a worker thread — doing it on the Tk thread froze
        the window for the duration of every refresh."""
        gpu = gpu_info()
        try:
            cost_line = cost_summary(cfg)
        except Exception:
            cost_line = ""
        return gpu, cost_line, backend.discover(cfg)

    def paint(data):
        """Everything touching widgets. Tk is not thread-safe, so this only
        ever runs on the main thread, marshalled back via root.after."""
        gpu, cost_line, items = data
        gpu_var.set(gpu_label(gpu))
        if cost_line:
            cost_label.config(text=cost_line)
        for row in tree.get_children():
            tree.delete(row)
        for values, tag in gui_rows(items):
            tree.insert("", "end", values=values, tags=(tag,))
        # keep the user's typed value; only refresh the dropdown choices
        choices = gui_model_choices(items)
        if choices:
            mdl_cb.configure(values=choices)
        tree.tag_configure("on", foreground="#1b5e20")
        tree.tag_configure("off", foreground="#9e9e9e")

    busy = {"running": False}

    def refresh():
        if busy["running"]:
            return               # a slow probe must not queue up behind itself
        busy["running"] = True

        def work():
            try:
                data = gather()
            except Exception:
                data = None
            finally:
                busy["running"] = False
            if data is not None:
                root.after(0, lambda: paint(data))
        threading.Thread(target=work, daemon=True).start()

    refresh()
    # auto-refresh every 5s (visibility of system status, continuously)
    def loop():
        refresh()
        root.after(5000, loop)
    root.after(5000, loop)
    root.mainloop()


def cmd_hide(runtime):

    print(hide_taskbar(runtime), flush=True)


def cmd_examples():
    print(json.dumps({
        "pricing": {"my-model": {"in": 1.0, "out": 2.0, "cw": 1.0, "cr": 0.1}},
        "route": {"private": ["secret", "社内"], "heavy": ["code", "設計"]},
        "providers": {
            "openai": {"api_key": "sk-...", "base_url": "https://api.openai.com/v1",
                       "model": "gpt-4o", "kind": "remote",
                       "price": "openai-gpt4o"},
            "gemini": {"api_key": "AIza...", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "model": "gemini-1.5-pro", "kind": "remote",
                       "price": {"in": 1.25, "out": 5.0}},
            "my-local": {"api_key": "ollama", "base_url": "http://localhost:11434/v1",
                         "model": "qwen2.5-coder:7b", "kind": "local"}
        },
        # Hub security. The hub calls your providers with the keys above, so
        # anyone who can reach it can spend your budget. Loopback is open;
        # binding wider is refused unless you set a token (recommended) or
        # explicitly opt into an unauthenticated bind.
        "hub": {"token": None, "allow_remote": False},
        # Retry a TRANSIENT failure (429, 5xx, timeout) on the same provider,
        # with full-jitter backoff. A 4xx is never retried — it would fail the
        # same way. attempts=1 disables retrying.
        "retry": {"attempts": 2, "base_ms": 250, "cap_ms": 8000},
        # Stop paying a down provider's timeout on every request: after
        # `threshold` consecutive failures it is skipped for `cooldown_s`.
        "breaker": {"enabled": True, "threshold": 3, "cooldown_s": 30},
        # Provider priority. Set it and lmm follows your order exactly;
        # leave it out and route_threshold below decides.
        "ask_order": ["my-local", "openai"],
        # RouteLLM-style cost threshold (arXiv:2406.18665). Prompts scoring
        # below it try the cheap providers first. null disables auto-routing.
        "route_threshold": 0.5,
        # FrugalGPT-style cascade (arXiv:2305.05176): run cheap models first,
        # escalate only when the answer scores below `threshold`.
        "cascade": {
            "enabled": False, "rungs": ["my-local", "openai"],
            "threshold": 0.6, "max_rungs": 3, "judge": None
        },
        # Prompt cache. The semantic tier needs a local embedding model and
        # accepts fuzzy matches, so it is opt-in (see vCache, arXiv:2502.03771).
        "cache": {
            "enabled": True, "semantic": False, "similarity": 0.95,
            "ttl_hours": 168, "max_entries": 2000,
            "embed_model": "nomic-embed-text", "max_temp": 0.3,
            # Set max_error_rate (vCache, arXiv:2502.03771) and each cached
            # entry must earn the right to answer by accumulating evidence
            # that it was correct; null keeps the static-threshold behaviour.
            "max_error_rate": None, "confidence": 0.95,
            "answer_match": 0.92, "min_observations": 3
        },
        # Optional: hand-entered cloud spend, added on top of what lmm
        # measures itself in ~/.lmm/usage.jsonl.
        "usage": {
            "openai": 12.50,
            "gemini": {"in": 1000000, "out": 2000000}
        },
        "extra_runtimes": [
            {
                "name": "vLLM",
                "type": "local",
                "paid": False,
                "procs": ["vllm", "vllm.entrypoints"],
                "installed_paths": ["~/.vllm"],
                "endpoint": "http://localhost:8000/v1",
                "models_cmd": "curl -s http://localhost:8000/v1/models"
            },
            {
                "name": "My Remote Agent",
                "type": "remote",
                "paid": True,
                "procs": ["myagent"],
                "installed_paths": ["~/myagent"],
                "endpoint": "https://api.myagent.example"
            }
        ]
    }, indent=2, ensure_ascii=False))


def cmd_ask(prompt, provider, cfg, cascade=False, no_cache=False,
            explain=False, verify=False):
    """Unified inference over every backend: cache, threshold routing, optional
    cheap-first cascade, and metering - all of it the same code path the hub
    serves, so `lmm ask` and an app pointed at `lmm serve --hub` behave alike.

    --verify measures the reply after the call and falls through to the next
    backend if it is unusable — the complement of routing, which can only
    guess before the call. Both orderings come from the same router.
    """
    if not (prompt or "").strip():
        print("usage: lmm ask \"your question\" [--provider NAME] [--cascade]")
        return
    effective = provider
    if verify and not provider:
        name, vreason, reply = route_and_verify(prompt, cfg, cfg.get("ask_order"))
        if name and reply is not None:
            print(f"[ask] verified-route: {name} -> {vreason}")
            print(reply)
        else:
            print("[ask] verified-route: no backend passed the quality gate "
                  f"({vreason})")
        return
    targets = resolve_ask_targets(cfg, prompt, effective)
    if not targets:
        if provider:
            # Naming a provider that does not exist is a typo, not an outage.
            # Saying which names *are* known turns a dead end into a fix.
            known = sorted(set(
                list((cfg.get("providers") or {}).keys())
                + [it["name"] for it in backend.discover(cfg) if it["running"]]
                + ["local-ollama(implicit)", "local-lmstudio(implicit)"]))
            print(f"[ask] unknown provider '{provider}'.")
            print(f"[ask] known providers: {', '.join(known)}")
            print("[ask] or omit --provider to use ask_order / auto routing.")
        else:
            print("[ask] no provider available. Start Ollama (`lmm serve <model>`) "
                  "or add 'providers' to lmm config (see `lmm examples`).")
        return
    if explain:
        score, feats = prompt_strength(cfg, prompt)
        thr = cfg.get("route_threshold", DEFAULT_ROUTE_THRESHOLD)
        print(f"[route] strength={score:.2f} threshold={thr} -> "
              + ", ".join(n for n, _ in backend.order_targets(cfg, prompt, targets)))
    res, trace = hub_complete(cfg, prompt, targets,
                              {"cascade": cascade, "cache": not no_cache,
                               "source": "ask"})
    for line in trace:               # warnings are never hidden behind --explain
        if explain or line.startswith("[warn]"):
            print(line)
    if isinstance(res, dict) and res.get("error"):
        log_hub({"event": "ask", "provider": effective or "(routed)",
                 "ok": False, "error": res["error"], "prompt": prompt})
        print(f"[ask] all providers failed. last error: {res['error']}")
        return
    try:
        answer = res["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("[ask] unexpected response shape from provider")
        return
    log_hub({"event": "ask", "provider": effective or "(routed)", "ok": True,
             "prompt": prompt, "reply": (answer or "")[:200], "trace": trace})
    print(answer)


def cmd_bench(cfg, provider=None, runs=3, prompt=None, max_tokens=128):
    """Measure TTFT / TPOT / throughput per provider so the local-vs-cloud
    decision has latency data, not just price."""
    prompt = prompt or "Count from 1 to 40, separated by commas."
    provs = merged_providers(cfg)
    if provider:
        targets = [(provider, provs[provider])] if provider in provs else []
        if not targets:
            lo = backend.local_ollama_provider()
            if lo and provider in ("local", "ollama", "local-ollama"):
                targets = [("local-ollama(implicit)", lo)]
    else:
        targets = list(provs.items())
        lo = backend.local_ollama_provider()
        if lo:
            targets.append(("local-ollama(implicit)", lo))
    if not targets:
        print("[bench] no provider available. Start Ollama or add 'providers' "
              "to lmm config (see `lmm examples`).")
        return

    print(f'prompt: "{prompt}"   runs: {runs} (+1 discarded warm-up)   '
          f"max_tokens: {max_tokens}")
    print("-" * 78)
    print(f"  {'provider':<26} {'TTFT':>9} {'TPOT':>9} {'tok/s':>9} {'e2e':>9}")
    print("-" * 78)
    pricing = merged_pricing(cfg)
    for name, prov in targets:
        # The first call pays for model load, connection setup and any cold
        # cache. Including it would measure the machine's startup, not its
        # steady-state serving speed, so it is discarded.
        warm = bench_once(prov, prompt, max_tokens)
        if warm.get("error"):
            print(f"  {name:<26} failed: {warm['error']}")
            continue
        samples = []
        for i in range(max(1, runs)):
            # Vary the prefix per run. Repeating an identical prompt would hit
            # the server's prefix cache (Ollama and vLLM both keep one) and
            # report a cache-hit TTFT instead of a real prefill — the run would
            # measure the cache, not the model.
            r = bench_once(prov, "(%d) %s" % (i, prompt), max_tokens)
            if not r.get("error"):
                samples.append(r)
        if not samples:
            print(f"  {name:<26} warm-up ok but every measured run failed")
            continue
        ttft = median([s["ttft_ms"] for s in samples])
        tpot = median([s["tpot_ms"] for s in samples])
        tps = median([s["tok_per_s"] for s in samples])
        e2e = median([s["e2e_ms"] for s in samples])
        note = " (tokens estimated)" if samples[0]["estimated"] else ""
        print(f"  {name:<26} {ttft:8.0f}ms {tpot:8.1f}ms {tps:9.1f} {e2e:8.0f}ms{note}")
        if len(samples) > 1:
            lo_t = min(s["ttft_ms"] for s in samples)
            hi_t = max(s["ttft_ms"] for s in samples)
            print(f"  {'':<26} TTFT range {lo_t:.0f}-{hi_t:.0f}ms over "
                  f"{len(samples)} runs")
        rate = price_for(prov, prov.get("model"), pricing)
        if rate["out"]:
            per_1k = rate["out"] / 1000.0
            print(f"  {'':<26} ${per_1k:.5f} per 1k output tokens")
    print("-" * 78)
    print("TTFT = time to first token (prefill). TPOT = time per output token "
          "after the first\n(decode). e2e = TTFT + TPOT x (tokens-1). Medians "
          "over the measured runs.")


def cmd_cache(cfg=None, clear=False):
    """Inspect or drop the prompt cache. The near-miss similarities are the
    evidence for tuning cache.similarity — vCache (arXiv:2502.03771) makes the
    case that a static threshold picked blind is not safe.

    It reads the EFFECTIVE config, not the defaults: showing `threshold 0.95` to
    someone who set 0.85 gave them the wrong evidence for the one decision this
    command exists to support, and expiry was judged against the default TTL
    rather than theirs.
    """
    conf = merged_cache(cfg or {})
    if clear:
        try:
            if os.path.isfile(backend.CACHE_LOG):
                os.remove(backend.CACHE_LOG)
            print("[cache] cleared")
        except OSError as e:
            print(f"[cache] could not clear: {e}")
        return
    entries = cache_entries(conf)
    print(f"[cache] {backend.CACHE_LOG}")
    print(f"[cache] enabled={conf.get('enabled')} semantic={conf.get('semantic')} "
          f"similarity={conf.get('similarity')} ttl_hours={conf.get('ttl_hours')} "
          f"max_entries={conf.get('max_entries')}")
    if conf.get("semantic"):
        print(f"[cache] embed model: {conf.get('embed_model')} (local Ollama)")
    delta = conf.get("max_error_rate")
    if delta is None:
        print("[cache] mode: static threshold — every neighbour above "
              f"{conf.get('similarity')} is served")
    else:
        print(f"[cache] mode: verified (vCache) — max_error_rate={delta}, "
              f"confidence={conf.get('confidence')}, "
              f"min_observations={conf.get('min_observations')}")
    print(f"[cache] {len(entries)} live entries, "
          f"{sum(1 for e in entries if e.get('emb'))} with embeddings")
    if delta is not None and entries:
        # Per-entry state is the whole point of the verified mode, so show it.
        certified_n = 0
        for e in entries:
            obs = e.get("obs") or []
            if obs and certified(e, max(o[0] for o in obs), conf)[0]:
                certified_n += 1
        observed = sum(len(e.get("obs") or []) for e in entries)
        print(f"[cache] {certified_n} entr(ies) certified to answer, "
              f"{observed} observation(s) recorded")
    if entries:
        oldest = min(e.get("at", 0) for e in entries)
        print("[cache] oldest: "
              + datetime.datetime.fromtimestamp(oldest).strftime("%Y-%m-%d %H:%M"))
        print(f"[cache] value stored: ${sum(float(e.get('usd', 0) or 0) for e in entries):,.4f}")
    hits = {"exact": 0, "semantic": 0, "near-miss": 0}
    sims = []
    for ev in read_usage():
        if ev.get("rollup"):
            for k in hits:
                hits[k] += (ev.get("hits") or {}).get(k, 0)
            continue                     # sims stay tail-only: recency matters
        h = ev.get("cache")
        if h in hits:
            hits[h] += 1
        if h == "near-miss":
            sims.append(ev.get("similarity", 0.0))
    print(f"[cache] hits: exact={hits['exact']} semantic={hits['semantic']} "
          f"near-miss={hits['near-miss']}")
    if sims:
        print(f"[cache] near-miss similarity: max={max(sims):.3f} "
              f"avg={sum(sims) / len(sims):.3f} "
              f"(threshold {conf.get('similarity')})")


def cmd_priority(cfg, show=False, optimize=False):
    """Management UI: inspect / reorder the routing priority (ask_order).

    Flow the user asked for: discover -> set priority -> real use (ask/chat).
    `lmm priority` interactively reorders; `lmm priority --show` just prints.
    `lmm priority --optimize` re-ranks by MEASURED performance (closed loop).
    """
    if optimize:
        new_order, stat = optimize_ask_order(cfg)
        # display-name -> provider-key normalizer (mirrors optimize_ask_order)
        NAME_TO_KEY = {
            "ollama": "local-ollama(implicit)",
            "lm studio": "local-lmstudio(implicit)",
            "claude code (anthropic)": "claude-code",
            "claude": "claude-code",
        }
        def _norm(n):
            return n if n in stat else NAME_TO_KEY.get(n.lower(), n)
        pricing = merged_pricing(cfg)
        def _cost(key):
            if "local-" in key or key.endswith("(implicit)"):
                return 0.0
            fam = key.split(":")[0].lower()
            p = pricing.get(fam, pricing.get("default", {"out": 15.0}))
            return float(p.get("out", 15.0))
        cfg = dict(cfg)
        cfg["ask_order"] = new_order
        save_config(cfg)
        print("[priority] optimized ask_order by cost-aware measured performance:")
        for i, n in enumerate(new_order, 1):
            s = stat.get(_norm(n), {})
            c = _cost(_norm(n))
            cost_s = f"${c:g}/1M" if c > 0 else "free"
            if s.get("ok", 0) + s.get("fail", 0) == 0:
                ev = f"unmeasured, {cost_s}"
            else:
                tot = s["ok"] + s["fail"]
                ev = f"{s['ok']}/{tot} ok, {s.get('avg_ms', 0)}ms, {cost_s}"
            print(f"  {i}. {n}  [{ev}]")
        print("[priority] now use `lmm ask` — it routes by this evidence-ranked order.")
        return
    if show:
        order = cfg.get("ask_order") or []
        if not order:
            print("[priority] ask_order is empty — using auto-routing.")
            print("[priority] run `lmm discover --save` to seed from detected backends.")
        else:
            print("[priority] current routing order (highest first):")
            for i, n in enumerate(order, 1):
                print(f"  {i}. {n}")
        return
    items = backend.discover(cfg)
    running = [it["name"] for it in items
               if it["running"] and it["name"] != "-"]
    if not running:
        print("[priority] no running backends detected.")
        print("[priority] start Ollama (`lmm serve <model>`) or a cloud provider first.")
        return
    cur = list(cfg.get("ask_order") or [])
    # seed with running backends not already present, keep current order first
    ordered = [n for n in cur if n in running]
    for n in running:
        if n not in ordered:
            ordered.append(n)
    print("[priority] detected backends (set priority by typing order):")
    for i, n in enumerate(running, 1):
        mark = ">" if n in cur else " "
        print(f"  {mark} {i}. {n}")
    print("")
    print("[priority] enter priority order as space/comma separated numbers,")
    print("           e.g. '3 1 2'  ->  backend#3 first, then #1, then #2")
    print("           or 'auto' to keep current order, or 'q' to cancel.")
    try:
        inp = input("[priority] > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[priority] cancelled.")
        return
    if inp.lower() in ("q", "quit", "cancel"):
        print("[priority] cancelled.")
        return
    if inp.lower() == "auto":
        new_order = ordered
    else:
        try:
            picks = [int(x) for x in inp.replace(",", " ").split() if x.strip()]
            new_order = [running[i - 1] for i in picks if 1 <= i <= len(running)]
            if not new_order:
                raise ValueError
        except Exception:
            print("[priority] invalid input — keeping current order.")
            new_order = ordered
    cfg = dict(cfg)
    cfg["ask_order"] = new_order
    path = save_config(cfg)
    print(f"[priority] saved {len(new_order)} backend(s) to ask_order:")
    for i, n in enumerate(new_order, 1):
        print(f"  {i}. {n}")
    print(f"[priority] config written: {path}")
    print("[priority] now use `lmm ask` / `lmm chat` — they route by this order.")


def cmd_pull(model):
    """Pull a model into the local Ollama base (the unified local model store).
    The hub's default backend is Ollama, so `lmm pull` keeps it stocked."""
    if not model:
        print("usage: lmm pull <ollama-model>  e.g. lmm pull qwen2.5-coder:7b")
        return
    print(f"pulling {model} into local Ollama ...")
    r = backend.run(f"ollama pull {model}", timeout=900)
    if r is None or r.returncode != 0:
        # command itself failed/exceptioned
        err = (r.stderr.strip() if r and r.stderr else "(pull failed/timeout)")
        print(err)
    else:
        # success: ollama pull prints progress; confirm presence via list
        confirm = backend.run(f"ollama list", timeout=30)
        ok = confirm and model.split(":")[0] in (confirm.stdout or "") \
            and (model.split(":")[1] if ":" in model else "") in (confirm.stdout or "")
        print((r.stdout.strip() if r.stdout else "") or "pull complete.")
        if not ok:
            print("(warning: model not found in `ollama list` after pull)")
    print("done. Use `lmm models` to confirm, `lmm ask` to route to it.")


def cmd_chat(provider, cfg):
    """Interactive chat REPL over the hub. Keeps conversation history across
    turns and routes every turn through hub_stream — the SAME path `lmm ask`
    and `lmm serve --hub` use, which is what makes the README's "one path"
    claim true rather than aspirational. That buys chat the cache, the
    routing, retry/breaker, and metering; its old private fallback loop had
    none of those, so chat turns were invisible to `lmm cost`.
    Type 'exit'/'quit'/'/exit' to leave."""
    print("lmm chat — type 'exit' to quit. Routes every turn via the hub.")
    brk = merged_breaker(cfg)
    breaker = HUB_BREAKER if brk.get("enabled", True) else None
    messages = []
    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit", "/exit"):
                break
            messages.append({"role": "user", "content": line})
            targets = resolve_ask_targets(cfg, line, provider)
            if not targets:
                print("[chat] no provider available.")
                messages.pop()  # drop the unsendable turn
                continue
            print("hub> ", end="", flush=True)
            parts, err = [], None
            for frame in hub_stream(cfg, list(messages), targets,
                                    {"source": "chat", "breaker": breaker}):
                for raw in frame.split(b"\n"):
                    if not raw.startswith(b"data: "):
                        continue
                    payload = raw[6:].strip()
                    if not payload or payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload.decode("utf-8", "ignore"))
                    except ValueError:
                        continue
                    if isinstance(obj.get("error"), dict):
                        err = obj["error"].get("message", "unknown error")
                        continue
                    piece = chunk_text(obj)
                    if piece:
                        parts.append(piece)
                        print(piece, end="", flush=True)
            print("")
            if parts:
                messages.append({"role": "assistant",
                                 "content": "".join(parts)})
                if err:
                    print(f"[chat] stream ended early: {err}")
            else:
                print(f"[chat] all providers failed: {err or 'no reply'}")
                messages.pop()  # drop the turn we couldn't send
    except Exception as e:
        print(f"[chat] stopped: {e}")


def cmd_log(n, cfg):
    """Show the last n hub-trail events (proof of what the hub actually
    routed/served). Defaults to 20. The trail lives in the metering log —
    trail entries are the ones carrying an "event" key."""
    try:
        n = int(n)
    except Exception:
        n = 20
    events = [e for e in backend.read_usage() if e.get("event")]
    if not events:
        print("[log] no hub events yet. Run `lmm ask` or `lmm serve --hub` first.")
        return
    for e in events[-n:]:
        ts = e.get("ts", "?")
        kind = e.get("event", "?")
        prov = e.get("provider", "-")
        ok = e.get("ok")
        st = "OK " if ok else ("ERR" if ok is False else ".. ")
        extra = ""
        if "error" in e:
            extra = f" err={e['error']}"
        if "prompt" in e:
            extra += f" prompt={e['prompt'][:40]!r}"
        print(f"  [{st}] {ts} {kind:8} -> {prov}{extra}")


def cmd_config(args, cfg):
    """Manage lmm config: init / list / get / set / unset.
    Lets the user freely control hub priority (ask_order) and providers
    without hand-editing JSON. Zero-dep, stdlib only."""
    act = getattr(args, "config_action", None)
    if act == "validate":
        return cmd_validate_config(cfg)
    if act == "init":
        if os.path.exists(os.path.join(backend.HOME, ".lmm", "config.json")):
            print("[config] ~/.lmm/config.json already exists; not overwriting.")
            return
        save_config({})
        print("[config] created ~/.lmm/config.json")
        return
    if act == "list":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    if act == "get":
        key = args.key
        cur = cfg
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        print("null" if cur is None else json.dumps(cur, ensure_ascii=False))
        return
    if act == "set":
        key, val = args.key, args.value
        # parse value: bool/int/float/json else string
        lv = val.lower()
        if lv in ("true", "false"):
            typed = (lv == "true")
        else:
            try:
                typed = json.loads(val)
            except Exception:
                typed = val
        # navigate to parent, create dicts as needed
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                print(f"[config] cannot set '{key}': '{part}' is not a dict")
                return
        node[parts[-1]] = typed
        save_config(cfg)
        print(f"[config] set {key} = {json.dumps(typed, ensure_ascii=False)}")
        return
    if act == "unset":
        key = args.key
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                print(f"[config] key '{key}' not found")
                return
        if isinstance(node, dict) and parts[-1] in node:
            del node[parts[-1]]
            save_config(cfg)
            print(f"[config] unset {key}")
        else:
            print(f"[config] key '{key}' not found")
        return
    print("[config] usage: lmm config <init|list|get|set|unset>")


def cmd_validate_config(cfg):
    """Validate ~/.lmm/config.json against the schema lmm actually relies on.

    Prints one `[OK]/[FAIL] <check> -- <detail>` line per check and exits 1 if
    any check failed, so it is usable as a CI / pre-flight gate. Zero-dep.
    """
    errors = []

    def ok(check, detail=""):
        print(f"[OK] {check} -- {detail}")

    def fail(check, detail=""):
        print(f"[FAIL] {check} -- {detail}")
        errors.append(check)

    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    # 1. config is a dict ----------------------------------------------------
    if not isinstance(cfg, dict):
        fail("config.type", f"expected dict, got {type(cfg).__name__}")
        print("config: INVALID (1 error(s))")
        sys.exit(1)
    ok("config.type", f"dict with {len(cfg)} top-level key(s)")

    # 2. pricing -------------------------------------------------------------
    if "pricing" in cfg:
        pricing = cfg.get("pricing")
        if not isinstance(pricing, dict):
            fail("pricing", f"expected dict, got {type(pricing).__name__}")
        else:
            bad = []
            for fam, v in pricing.items():
                if not isinstance(v, dict):
                    bad.append(f"{fam}: not a dict")
                    continue
                for req in ("in", "out"):
                    if req not in v:
                        bad.append(f"{fam}.{req}: missing")
                    elif not is_num(v[req]):
                        bad.append(f"{fam}.{req}: not numeric ({v[req]!r})")
                for opt in ("cw", "cr"):
                    if opt in v and not is_num(v[opt]):
                        bad.append(f"{opt}: not numeric ({v[opt]!r})")
            if bad:
                fail("pricing", "; ".join(bad))
            else:
                ok("pricing", f"{len(pricing)} entry(ies) have numeric in/out")
    else:
        ok("pricing", "absent (defaults apply)")

    # 3. providers -----------------------------------------------------------
    if "providers" in cfg:
        provs = cfg.get("providers")
        if not isinstance(provs, dict):
            fail("providers", f"expected dict, got {type(provs).__name__}")
        else:
            bad = []
            for name, v in provs.items():
                if not isinstance(v, dict):
                    bad.append(f"{name}: not a dict")
                    continue
                bu = v.get("base_url")
                if not isinstance(bu, str) or not bu.strip():
                    bad.append(f"{name}.base_url: missing/not a string")
                elif not (bu.startswith("http://") or bu.startswith("https://")):
                    bad.append(f"{name}.base_url: not a url ({bu!r})")
                mdl = v.get("model")
                if not isinstance(mdl, str) or not mdl.strip():
                    bad.append(f"{name}.model: missing/empty")
                kind = v.get("kind", "remote")
                if kind not in ("local", "remote"):
                    bad.append(f"{name}.kind: must be local|remote ({kind!r})")
            if bad:
                fail("providers", "; ".join(bad))
            else:
                ok("providers", f"{len(provs)} provider(s) well-formed")
    else:
        ok("providers", "absent")

    # 4. ask_order -----------------------------------------------------------
    if "ask_order" in cfg:
        order = cfg.get("ask_order")
        if not isinstance(order, list):
            fail("ask_order", f"expected list, got {type(order).__name__}")
        else:
            known = set()
            for k in merged_providers(cfg):
                known.add(k.lower())
            running = []
            try:
                for it in backend.discover(cfg):
                    nm = (it.get("name") or "")
                    if it.get("running"):
                        running.append(nm)
                        known.add(nm.lower())
            except Exception as e:
                print(f"[..] ask_order -- discover() failed: {e}")
            # (implicit local bases surface via discover() running=True)
            unresolved = [n for n in order
                          if not (isinstance(n, str) and n.lower() in known)]
            if unresolved:
                fail("ask_order",
                     f"UNRESOLVED: {unresolved} (known providers="
                     f"{sorted(merged_providers(cfg))}, running={running})")
            else:
                ok("ask_order",
                   f"{len(order)} entry(ies) all resolve (running={running})")
    else:
        ok("ask_order", "absent (default order applies)")

    # 5. route ---------------------------------------------------------------
    if "route" in cfg:
        route = cfg.get("route")
        if not isinstance(route, dict):
            fail("route", f"expected dict, got {type(route).__name__}")
        else:
            bad = []
            for key in ("private", "heavy"):
                if key not in route:
                    continue
                v = route[key]
                if not isinstance(v, list):
                    bad.append(f"{key}: expected list, got {type(v).__name__}")
                elif not all(isinstance(x, str) for x in v):
                    bad.append(f"{key}: non-string element(s)")
            if bad:
                fail("route", "; ".join(bad))
            else:
                ok("route", "private/heavy are lists of strings")
    else:
        ok("route", "absent (defaults apply)")

    if errors:
        print(f"config: INVALID ({len(errors)} error(s))")
        sys.exit(1)
    print("config: VALID")


def cmd_secrets(cfg):
    """Scan the config for accidentally-stored secrets. lmm's hard rule: NO
    secret is ever stored or copied — credentials are only CHECKED, never saved.
    This never prints the secret value itself, only a redacted hint, and exits
    non-zero if a real secret is found so it can gate a commit."""
    findings = []

    def redact(val):
        s = str(val)
        if len(s) <= 8:
            return "***"
        return f"{s[:3]}...{s[-4:]}"

    def looks_secret(v):
        s = str(v)
        if not s:
            return False
        if any(s.startswith(p) for p in
               ("sk-", "AKIA", "AIza", "ya29", "Bearer ")):
            return True
        # long high-entropy-ish strings are treated as credentials too
        if len(s) > 20 and any(c.isdigit() for c in s) \
                and any(c.isalpha() for c in s):
            return True
        return False

    for name, v in (cfg.get("providers") or {}).items():
        if not isinstance(v, dict):
            continue
        key = v.get("api_key", "")
        if looks_secret(key):
            findings.append(f"provider {name}: REAL SECRET DETECTED "
                            f"(redacted: {redact(key)})")
    for k, v in cfg.items():
        if k.lower() in ("secret", "token", "password") and v:
            findings.append(f"top-level {k}: SECRET DETECTED "
                            f"(redacted: {redact(v)})")
    if findings:
        print("lmm secrets scan — FINDINGS:")
        for f in findings:
            print(f"  [WARN] {f}")
        print("  move secrets to environment variables; lmm reads them at "
              "call time, never stores them")
        sys.exit(2)
    print("secrets: CLEAN — no stored credentials detected")
    sys.exit(0)


def cmd_doctor(cfg):
    """First-principles health check: don't trust that lmm works — MEASURE it.
    Runs real probes (config loads, Ollama reachable, a backend is running,
    ask_order resolves, each provider answers, trail writable) and exits
    non-zero if any FAILs."""
    fails = []

    def chk(name, ok, detail=""):
        mark = "OK" if ok else "FAIL"
        if not ok:
            fails.append(name)
        print(f"[{mark}] {name}" + (f" -- {detail}" if detail else ""))

    # 1) config loads
    try:
        _ = cfg
        ok_cfg = isinstance(cfg, dict)
    except Exception as e:
        ok_cfg = False
        detail_cfg = str(e)
    chk("config loads", ok_cfg, "" if ok_cfg else detail_cfg)

    # 2) implicit Ollama reachable
    lo = backend.local_ollama_provider()
    chk("implicit Ollama reachable",
        bool(lo) and bool(lo.get("model")),
        (lo.get("model") if lo else "ollama not running"))

    # 3) at least one backend running
    try:
        disc = backend.discover(cfg)
        running = [d["name"] for d in disc if d.get("running")]
        chk("at least one backend running",
            len(running) > 0,
            (", ".join(running) if running else "none running"))
    except Exception as e:
        chk("at least one backend running", False, str(e))

    # 4) ask_order resolves
    order = cfg.get("ask_order") or []
    if not isinstance(order, list) or not order:
        chk("ask_order resolves", False, "ask_order empty or not a list")
    else:
        known = set(n.lower() for n in merged_providers(cfg))
        try:
            run_names = [d["name"] for d in backend.discover(cfg) if d.get("running")]
        except Exception:
            run_names = []
        unresolved = [n for n in order
                      if n.lower() not in known and n not in run_names]
        chk("ask_order resolves",
            len(unresolved) == 0,
            ("unresolved: " + ", ".join(unresolved)) if unresolved else
            f"{len(order)} entr(ies) OK")

    # 5) each configured provider answers (was `lmm hub-status`, folded in:
    #    the machine checks above see local processes, but a cloud provider
    #    has no process to see — only a probe can grade it)
    provs = (cfg.get("providers") or {}) if isinstance(cfg, dict) else {}
    for name, prov in provs.items():
        ok = False
        detail = ""
        try:
            import urllib.request
            url = (prov.get("base_url", "").rstrip("/")) + "/models"
            req = urllib.request.Request(
                url, headers={"Authorization":
                              f"Bearer {prov.get('api_key', '')}"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                ok = resp.status == 200
                detail = f"HTTP {resp.status}"
        except Exception as e:
            detail = str(e)[:60]
        chk(f"provider '{name}' answers", ok, detail)

    # 6) the trail is writable (observability is part of health)
    try:
        log_hub({"event": "doctor", "provider": "(self)", "ok": True,
                 "prompt": "doctor probe"})
        chk("usage trail writable", True)
    except Exception as e:
        chk("usage trail writable", False, str(e))

    print("")
    if fails:
        print(f"doctor: UNHEALTHY ({len(fails)} issue(s)): {', '.join(fails)}")
        sys.exit(1)
    print("doctor: HEALTHY")


def cmd_stats(cfg):
    """Aggregate the hub trail into measured routing statistics.
    Proves (don't trust — measure) HOW well the routing actually performs:
    per-backend success rate, average latency, total attempts, and how often
    the first tried backend succeeded. Zero-dep: pure JSONL scan."""
    # One writer per fact: successes are the metering events meter_call wrote
    # (cache=="miss" — an actual provider call, not a cache answer); failures
    # are the ask_attempt trail entries, which exist because a failure has no
    # usage to meter. Joining them here is what makes this a measurement of
    # every path — ask, chat, cascade and the hub server alike.
    by_provider = {}
    first_rung = {"ok": 0, "fail": 0}
    total = 0
    for e in backend.read_usage():
        if e.get("rollup"):
            continue
        if e.get("event") == "ask_attempt" and not e.get("ok"):
            name = e.get("provider", "?")
            rec = by_provider.setdefault(
                name, {"ok": 0, "fail": 0, "lat": [], "cost": 0.0})
            rec["fail"] += 1
            total += 1
            if isinstance(e.get("latency_ms"), int):
                rec["lat"].append(e["latency_ms"])
            if e.get("rung") == 0:
                first_rung["fail"] += 1
        elif not e.get("event") and e.get("cache") == "miss" \
                and e.get("provider"):
            name = e["provider"]
            rec = by_provider.setdefault(
                name, {"ok": 0, "fail": 0, "lat": [], "cost": 0.0})
            rec["ok"] += 1
            total += 1
            if isinstance(e.get("ms"), int):
                rec["lat"].append(e["ms"])
            if isinstance(e.get("usd"), (int, float)):
                rec["cost"] += e["usd"]
            if e.get("rung") == 0:
                first_rung["ok"] += 1
    if not total:
        print("[stats] no routing attempts measured yet. Run `lmm ask` first.")
        sys.exit(0)

    print(f"lmm stats — {total} routing attempt(s) measured:")
    print(f"{'backend':<32}{'success':>9}{'fail':>6}{'succ%':>8}{'avg_ms':>9}{'cost$':>10}")
    print("-" * 74)
    total_cost = 0.0
    for name, rec in sorted(by_provider.items(),
                            key=lambda kv: -(kv[1]["ok"] + kv[1]["fail"])):
        tot = rec["ok"] + rec["fail"]
        pct = (rec["ok"] / tot * 100) if tot else 0
        avg = int(sum(rec["lat"]) / len(rec["lat"])) if rec["lat"] else 0
        total_cost += rec["cost"]
        print(f"{name:<32}{rec['ok']:>9}{rec['fail']:>6}{pct:>7.0f}%{avg:>9}{rec['cost']:>10.4f}")
    overall_ok = sum(r["ok"] for r in by_provider.values())
    overall_pct = (overall_ok / total * 100) if total else 0
    print("-" * 74)
    print(f"{'TOTAL':<32}{overall_ok:>9}{total - overall_ok:>6}{overall_pct:>7.0f}%{'':>9}{total_cost:>10.4f}")
    ft_total = first_rung["ok"] + first_rung["fail"]
    if ft_total:
        print(f"first-try success rate: {first_rung['ok']}/{ft_total} "
              f"({first_rung['ok'] / ft_total * 100:.0f}%)")
    print(f"total estimated cost: ${total_cost:.4f} "
          f"(local backends are free; cloud costs are measured per actual tokens)")


def cmd_selftest(cfg, guard=False):
    """Self-prove the hub works. Runs real measurements (not trust):
    syntax, command surface, implicit-Ollama reachability, a live `ask`
    routing that must return a real reply, and trail observability.
    Each check is pass/fail; a non-zero failure count means `lmm` is broken.

    guard=True is for callers that only read the exit code (guard.sh, CI).
    It drops the passing lines, because a green run has nothing to say — but
    it keeps the failures, since an exit code alone cannot tell you what
    broke."""
    import subprocess as _sp
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, ok, detail))
        if guard and ok:
            return
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))

    def note(msg):
        if not guard:
            print(msg)

    if not guard:
        print("lmm selftest — measuring, not trusting:")
    # 1) syntax of this very file
    selfp = os.path.abspath(__file__)
    r = _sp.run([sys.executable, "-m", "py_compile", selfp],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    chk("self syntax (py_compile)", r.returncode == 0)

    # 2) command surface present
    import re as _re
    with open(os.path.abspath(__file__), encoding="utf-8") as _f:
        src = _f.read()
    cmds = _re.findall(r"add_parser\(['\"]([\w-]+)['\"]", src)
    needed = ["discover", "status", "models", "cost", "route", "serve",
              "ask", "chat", "config", "log", "selftest", "doctor",
              "stop", "dash", "gui", "watch", "autostart", "hide", "examples"]
    missing = [c for c in needed if c not in cmds]
    chk("command surface complete", not missing,
        ("missing: " + ",".join(missing)) if missing else "")

    # 3) implicit Ollama reachable
    skip_live = os.environ.get("LMM_SELFTEST_SKIP_LIVE", "") in ("1", "true", "yes")
    if skip_live:
        note("  [SKIP] implicit Ollama reachable -- LMM_SELFTEST_SKIP_LIVE=1")
        lo = None
    else:
        lo = backend.local_ollama_provider()
        chk("implicit Ollama reachable", bool(lo),
            (lo["model"] if lo else "ollama not running"))

    # 4) live ask routing returns a real reply
    if skip_live:
        note("  [SKIP] live ask routing returns reply -- LMM_SELFTEST_SKIP_LIVE=1")
    elif lo:
        try:
            gen = call_provider(lo, "Reply with exactly: SELFTEST_OK",
                                stream=True)
            reply = "".join(gen) if not isinstance(gen, dict) else ""
            ok_reply = "SELFTEST_OK" in reply
            chk("live ask routing returns reply", ok_reply,
                (reply[:40] if reply else "no reply"))
        except Exception as e:
            chk("live ask routing returns reply", False, str(e))
    else:
        chk("live ask routing returns reply", False, "no provider")

    # 5) observability: the trail gets written, and lands in the ONE log
    #    (hub.log used to be a second file with no cap, no compaction and no
    #    lock — the bugs already fixed for usage.jsonl, alive under another
    #    name)
    log_hub({"event": "selftest", "provider": "(self)", "ok": True,
             "prompt": "selftest probe"})
    seen = any(e.get("event") == "selftest" for e in backend.read_usage())
    chk("observability (trail writable)", seen)

    # 6) closed-loop verify routing — the hub must PROVE its own core claim
    #    ("measure the answer, don't just trust the pick"). Two deterministic
    #    checks that need NO live backend:
    #    6a) verify_reply rejects a hallucinated token
    bad_ok, bad_reason = verify_reply(
        "explain quantum mechanics",
        "The wavefunction propagレーション describes everything.")
    chk("verify_reply detects hallucination", (not bad_ok), bad_reason)
    #    6b) route_and_verify falls through a bad backend to a good one.
    #        We monkey-patch call_provider with two mock backends so the test
    #        is deterministic and offline (measure, don't trust — even the test).
    #        NOTE: route_and_verify lives in backend.py and references
    #        backend.call_provider, so we must patch the backend module's symbol,
    #        not frontend's globals (after the file split, they are separate).
    real_call = backend.call_provider
    try:
        def _mock(prov, prompt, stream=False):
            if "bad" in prov.get("model", ""):
                return {"choices": [{"message": {
                    "content": "wavefunction is propagレーション thing."}}]}
            return {"choices": [{"message": {
                "content": "The Schrodinger equation iħ∂ψ/∂t=Hψ governs "
                           "quantum state evolution via the Hamiltonian H. "
                           "It describes how the quantum state of a physical "
                           "system changes with time and underlies all of "
                           "quantum mechanics including superposition."}}]}
        backend.call_provider = _mock
        _cfg = dict(cfg)
        _cfg["providers"] = {
            "bad":  {"api_key": "x", "base_url": "http://127.0.0.1:9/v1",
                     "model": "bad-model", "kind": "local"},
            "good": {"api_key": "x", "base_url": "http://127.0.0.1:8/v1",
                     "model": "good-model", "kind": "local"}}
        _cfg["ask_order"] = ["bad", "good"]
        _name, _reason, _reply = route_and_verify(
            "explain quantum mechanics", _cfg, _cfg["ask_order"])
        chk("route_and_verify falls back bad->good",
            _name == "good" and _reply is not None,
            (_reason if _reason else str(_name)))
    except Exception as e:
        chk("route_and_verify falls back bad->good", False, str(e))
    finally:
        backend.call_provider = real_call

    # 7) doctor + config validate + secrets commands are wired & run
    #
    # `doctor` diagnoses the MACHINE; `selftest` proves LMM. A box with no
    # Ollama running is an unhealthy machine and a working lmm, so requiring
    # "doctor: HEALTHY" here made the suite unpassable anywhere without a live
    # backend — including this project's own CI runner, where checks 3 and 4
    # are skipped for exactly that reason. What proves lmm is that doctor runs
    # and reaches a verdict; the verdict itself is reported as the detail so an
    # unhealthy machine is still visible in the output.
    import io as _io
    import contextlib as _cl

    def _run_cmd(fn, *a, **kw):
        """Call a cmd_* handler, capturing stdout and its exit code."""
        buf = _io.StringIO()
        try:
            with _cl.redirect_stdout(buf):
                fn(*a, **kw)
            return buf.getvalue(), 0
        except SystemExit as e:
            return buf.getvalue(), (e.code if isinstance(e.code, int) else 0)

    try:
        doc_out, doc_code = _run_cmd(cmd_doctor, cfg)
        verdict = next((l for l in reversed(doc_out.strip().splitlines())
                        if l.startswith("doctor:")), "")
        chk("doctor command runs", bool(verdict), verdict or f"exit={doc_code}")
    except Exception as e:
        chk("doctor command runs", False, str(e))

    try:
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            cmd_validate_config(cfg)
        val_out = _buf.getvalue()
        chk("config validate command runs", "config: VALID" in val_out,
            val_out.strip().splitlines()[-1] if val_out.strip() else "")
    except SystemExit as e:
        chk("config validate command runs", e.code in (0, None),
            f"exit={e.code}")
    except Exception as e:
        chk("config validate command runs", False, str(e))

    try:
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            cmd_secrets(cfg)
        sec_out = _buf.getvalue()
        # secrets must either be CLEAN or report a finding; never crash
        chk("secrets command runs",
            ("secrets: CLEAN" in sec_out) or ("FINDINGS" in sec_out),
            sec_out.strip().splitlines()[0] if sec_out.strip() else "")
    except SystemExit as e:
        # secrets exits 2 on finding, 0 on clean — both are "ran"
        chk("secrets command runs", e.code in (0, 2), f"exit={e.code}")
    except Exception as e:
        chk("secrets command runs", False, str(e))

    # 8) stats command runs (reads the structured hub log without crashing)
    try:
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            cmd_stats(cfg)
        st_out = _buf.getvalue()
        chk("stats command runs", "lmm stats" in st_out,
            st_out.strip().splitlines()[0] if st_out.strip() else "")
    except SystemExit as e:
        chk("stats command runs", e.code in (0, None), f"exit={e.code}")
    except Exception as e:
        chk("stats command runs", False, str(e))

    # 9) priority --optimize runs (closed-loop: measurement -> re-prioritize)
    try:
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            cmd_priority(cfg, optimize=True)
        po_out = _buf.getvalue()
        chk("priority --optimize runs", "optimized ask_order" in po_out,
            po_out.strip().splitlines()[0] if po_out.strip() else "")
    except SystemExit as e:
        chk("priority --optimize runs", e.code in (0, None), f"exit={e.code}")
    except Exception as e:
        chk("priority --optimize runs", False, str(e))

    fails = sum(1 for _, ok, _ in checks if not ok)
    if fails == 0:
        note("")
        note(f"SELFTEST PASS — {len(checks)} checks, the hub proves itself.")
    else:
        print("")
        print(f"SELFTEST FAIL — {fails} of {len(checks)} check(s) failed. "
              "Fix before trusting the hub.")
    return 1 if fails else 0


def setup_tray(root, tooltip="LMM Dashboard"):
    """Add a Windows system-tray icon so minimizing keeps lmm running in the
    background (zero-dep: pure ctypes / Win32). On non-Windows this is a no-op.
    Returns a cleanup callable, or None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user = ctypes.windll.user32
        shell = ctypes.windll.shell32
        user.LoadIconW.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user.LoadIconW.restype = ctypes.c_void_p
        shell.Shell_NotifyIconW.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        user.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p]
        user.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                        wintypes.WPARAM, wintypes.LPARAM]
        user.DefWindowProcW.restype = ctypes.c_long
    except Exception:
        return None

    WM_TRAYMSG = 0x401
    hwnd = int(root.winfo_id())

    class NOTIFYICONDATA(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("hWnd", ctypes.c_void_p),
            ("uID", ctypes.c_uint),
            ("uFlags", ctypes.c_uint),
            ("uCallbackMessage", ctypes.c_uint),
            ("hIcon", ctypes.c_void_p),
            ("szTip", ctypes.c_wchar * 128),
        ]
    nid = NOTIFYICONDATA()
    nid.cbSize = ctypes.sizeof(NOTIFYICONDATA)
    nid.hWnd = hwnd
    nid.uID = 1
    nid.uFlags = 0x1 | 0x2 | 0x4
    nid.uCallbackMessage = WM_TRAYMSG
    nid.hIcon = user.LoadIconW(0, 32512)
    nid.szTip = tooltip[:127]   # ctypes assigns str into c_wchar array
    shell.Shell_NotifyIconW(0x0, ctypes.byref(nid))

    def restore():
        root.deiconify()
        root.lift()
        root.focus_force()

    def wndproc(h, msg, w, l):
        if msg == WM_TRAYMSG and l == 0x205:   # right-click
            m = user.CreatePopupMenu()
            user.AppendMenuW(m, 0, 1001, "Open")
            user.AppendMenuW(m, 0, 1002, "Quit")
            pt = wintypes.POINT()
            user.GetCursorPos(ctypes.byref(pt))
            cmd = user.TrackPopupMenu(m, 0x100, pt.x, pt.y, 0, hwnd, None)
            if cmd == 1001:
                restore()
            elif cmd == 1002:
                cleanup()
                root.destroy()
            return 0
        if msg == WM_TRAYMSG and l == 0x203:   # double-click
            restore()
            return 0
        return user.DefWindowProcW(ctypes.c_void_p(h), msg, w, l)

    wp = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
                            wintypes.WPARAM, wintypes.LPARAM)(wndproc)
    user.SetWindowLongPtrW(ctypes.c_void_p(hwnd), -4,
                           ctypes.cast(wp, ctypes.c_void_p))
    root.protocol("WM_DELETE_WINDOW", lambda: root.withdraw())  # X -> tray

    def cleanup():
        shell.Shell_NotifyIconW(0x2, ctypes.byref(nid))

    return cleanup


def main():
    ap = argparse.ArgumentParser(
        description="LMM - Local/remote Model Manager (cross-platform, zero-dep)")
    ap.add_argument("-v", "--version", action="store_true", help="show version")
    sub = ap.add_subparsers(dest="cmd")

    # --- inspecting the machine -------------------------------------------
    p = sub.add_parser("discover")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", action="store_true",
                   help="seed ask_order from the backends that are running")
    sub.add_parser("status")
    sub.add_parser("models")
    p = sub.add_parser("pull", help="pull a model into local Ollama")
    p.add_argument("model", nargs="?")
    p = sub.add_parser("fit", help="will this model fit in your GPU, and at what context?")
    p.add_argument("model", nargs="?", help="model tag; omit to check every installed one")
    p.add_argument("--ctx", type=int, default=None, help="context length in tokens")
    p.add_argument("--vram", type=float, default=None, help="override detected free VRAM (GiB)")
    p.add_argument("--kv", default="f16", choices=sorted(set(KV_BYTES)),
                   help="KV cache dtype (llama.cpp --cache-type-k/v)")
    p.add_argument("--json", action="store_true")

    # --- spending and routing ---------------------------------------------
    p = sub.add_parser("cost")
    p.add_argument("--days", type=int, default=30,
                   help="only count the last N days (0 = all-time)")
    p = sub.add_parser("route")
    p.add_argument("task", nargs="?")
    p.add_argument("--explain", action="store_true",
                   help="show the strength-score breakdown and provider order")
    p = sub.add_parser("priority", help="manage routing priority (discover -> set -> use)")
    p.add_argument("--show", action="store_true", help="show current ask_order only")
    p.add_argument("--optimize", action="store_true",
                   help="re-rank ask_order by MEASURED performance (closed loop)")
    p = sub.add_parser("bench", help="measure TTFT / TPOT / throughput per provider")
    p.add_argument("--provider", default=None, help="provider name from config")
    p.add_argument("--runs", type=int, default=3, help="measured runs after the warm-up")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=128)

    # --- asking things ------------------------------------------------------
    p = sub.add_parser("ask")
    p.add_argument("prompt", nargs="*")
    p.add_argument("--provider", default=None, help="provider name from config")
    p.add_argument("--cascade", action="store_true",
                   help="cheap-first cascade: escalate only on a low-scoring answer")
    p.add_argument("--no-cache", action="store_true", help="bypass the prompt cache")
    p.add_argument("--explain", action="store_true",
                   help="show routing score, rung scores and per-call cost")
    p.add_argument("--verify", action="store_true",
                   help="closed loop: measure each reply, fall back if unfit")
    p = sub.add_parser("chat", help="interactive hub chat REPL (keeps history)")
    p.add_argument("--provider", default=None, help="force a provider")

    # --- serving ------------------------------------------------------------
    p = sub.add_parser("serve")
    p.add_argument("model", nargs="?")
    p.add_argument("--hub", action="store_true",
                   help="start OpenAI-compatible proxy over all configured providers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p = sub.add_parser("stop")
    p.add_argument("runtime", nargs="?")
    p = sub.add_parser("cache")
    p.add_argument("--clear", action="store_true", help="delete all cached answers")
    p.add_argument("--stats", action="store_true", help="show cache stats (default)")

    # --- proving it works ---------------------------------------------------
    p = sub.add_parser("selftest", help="self-prove the hub works (measure, don't trust)")
    p.add_argument("--guard", action="store_true",
                   help="machine-readable mode: report failures only, exit code is the answer")
    sub.add_parser("doctor")
    sub.add_parser("secrets")
    sub.add_parser("stats")
    p = sub.add_parser("log", help="show recent hub events (proof of routing)")
    p.add_argument("n", nargs="?", default=20)
    p = sub.add_parser("config", help="manage lmm config (init/list/get/set/unset)")
    p.add_argument("config_action", nargs="?", default="list")
    p.add_argument("key", nargs="?", default=None)
    p.add_argument("value", nargs="?", default=None)

    # --- desktop ------------------------------------------------------------
    sub.add_parser("dash")
    sub.add_parser("gui")
    p = sub.add_parser("watch")
    p.add_argument("--interval", type=float, default=3.0)
    sub.add_parser("autostart")
    sub.add_parser("hide").add_argument("runtime", nargs="?")
    sub.add_parser("examples")

    args = ap.parse_args()
    if args.version:
        print(f"lmm {VERSION}")
        return
    cfg = load_config()
    # No subcommand -> launch the live GUI dashboard. Visibility of system
    # status (Nielsen #1) is the root solution, so the default entry point is
    # the visual dashboard, not a text dump. Use `lmm discover` for CLI output.
    cmd = args.cmd or "gui"
    if cmd == "discover":
        cmd_discover(cfg, getattr(args, "json", False), getattr(args, "save", False))
    elif cmd == "status":
        cmd_status(cfg)
    elif cmd == "models":
        cmd_models(cfg)
    elif cmd == "pull":
        cmd_pull(args.model)
    elif cmd == "fit":
        cmd_fit(args.model, args.ctx, args.vram, args.kv, getattr(args, "json", False))
    elif cmd == "cost":
        print(cost_report(cfg, args.days or None))
    elif cmd == "route":
        cmd_route(cfg, args.task, getattr(args, "explain", False))
    elif cmd == "priority":
        cmd_priority(cfg, getattr(args, "show", False), getattr(args, "optimize", False))
    elif cmd == "bench":
        cmd_bench(cfg, args.provider, args.runs, args.prompt, args.max_tokens)
    elif cmd == "ask":
        cmd_ask(" ".join(getattr(args, "prompt", [])),
                getattr(args, "provider", None), cfg,
                cascade=getattr(args, "cascade", False),
                no_cache=getattr(args, "no_cache", False),
                explain=getattr(args, "explain", False),
                verify=getattr(args, "verify", False))
    elif cmd == "chat":
        cmd_chat(args.provider, cfg)
    elif cmd == "serve":
        if getattr(args, "hub", False):
            cmd_serve_hub(cfg, args.host, args.port)
        else:
            cmd_serve(args.model)
    elif cmd == "stop":
        cmd_stop(args.runtime, cfg)
    elif cmd == "cache":
        cmd_cache(cfg, getattr(args, "clear", False))
    elif cmd == "selftest":
        sys.exit(cmd_selftest(cfg, guard=getattr(args, "guard", False)))
    elif cmd == "doctor":
        cmd_doctor(cfg)
    elif cmd == "secrets":
        cmd_secrets(cfg)
    elif cmd == "stats":
        cmd_stats(cfg)
    elif cmd == "log":
        cmd_log(args.n, cfg)
    elif cmd == "config":
        cmd_config(args, cfg)
    elif cmd == "dash":
        cmd_dash(cfg)
    elif cmd == "gui":
        launch_gui(cfg)
    elif cmd == "watch":
        cmd_watch(cfg, args.interval)
    elif cmd == "autostart":
        cmd_autostart()
    elif cmd == "hide":
        cmd_hide(args.runtime)
    elif cmd == "examples":
        cmd_examples()
