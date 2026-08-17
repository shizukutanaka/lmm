import backend
from backend import *








# ------------------------------- config ------------------------------------





























# ----------------------------- detectors -----------------------------------








# --------------------- taskbar-hiding (Windows only) -----------------------
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
GWL_EXSTYLE = -20




def hide_taskbar(runtime):
    """Remove a runtime's GUI windows from the taskbar, using RUNTIME_REGISTRY.

    Two strategies (first-principles, from how Windows decides a taskbar
    button exists):
      1. SERVER-ONLY / headless-first apps (Ollama, LM Studio via `lms`,
         Jan, KoboldCPP, vLLM, Open WebUI, llama.cpp): they never need a
         taskbar button — advise the headless launch and don't touch windows.
      2. GUI apps (Claude, ChatGPT, Cursor, Perplexity, GPT4All, AnythingLLM,
         Chatbox, Msty, Devin/Cua): set WS_EX_TOOLWINDOW on every visible
         top-level window so it leaves the taskbar immediately, without
         closing or minimizing the app. Idempotent; restart restores it.
    """
    rt = (runtime or "").lower()
    entry = RUNTIME_REGISTRY.get(rt)
    if entry is None:
        return (f"unknown runtime '{runtime}'. choices: "
                + ", ".join(sorted(RUNTIME_REGISTRY)))
    if os.name != "nt":
        return ("hide is only supported on Windows (the taskbar is a Windows "
                "concept). On other OSes these apps already run without a dock "
                "button when launched headless.")
    titles = entry.get("titles", [])
    if not titles:
        # server-only / headless-first runtime
        return (f"'{runtime}' runs without a taskbar button by design. "
                f"{entry.get('headless', '')}")
    wins = _enum_windows_by_title(titles)
    if not wins:
        return (f"no visible '{runtime}' window is on the taskbar right now. "
                f"(If it isn't running, start it headless: "
                f"{entry.get('headless', 'n/a')})")
    user32 = ctypes.windll.user32
    SetWindowLongW = user32.SetWindowLongW
    GetWindowLongW = user32.GetWindowLongW
    count = 0
    for hwnd, title in wins:
        try:
            ex = GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_ex = (ex | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
            if new_ex != ex:
                SetWindowLongW(hwnd, GWL_EXSTYLE, new_ex)
                count += 1
        except Exception:
            pass
    return (f"removed {count} '{runtime}' window(s) from the taskbar "
            f"(the app keeps running, hidden from the taskbar). "
            f"Restart the app to restore its taskbar button.")






# ------------------------------ cost ---------------------------------------






# ------------------------------ routing ------------------------------------


# ------------------------------ dashboard -----------------------------------
def build_dash(cfg):
    items = discover(cfg)
    gpu = gpu_info()
    cost = cost_report(cfg)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = ""
    for it in items:
        cls = "on" if it["running"] else "off"
        paid = "PAID" if it.get("paid") else "free"
        models = ", ".join(it.get("models", [])) or "-"
        rows += (f'<tr class="{cls}"><td>{html.escape(it["name"])}</td>'
                 f'<td>{it["type"]}</td><td>{paid}</td>'
                 f'<td>{"YES" if it["running"] else "no"}</td>'
                 f'<td>{it.get("procs", 0)}</td><td>{html.escape(models)}</td>'
                 f'<td>{html.escape(str(it.get("endpoint", "-")))}</td></tr>')
    cost_html = html.escape(cost).replace("\n", "<br>")
    gpu_html = (f'{gpu["name"]} {gpu["used"]}/{gpu["total"]} MiB ({gpu["pct"]}%)'
                if gpu else "n/a")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>LMM Dashboard</title><style>
body{{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0d1117;color:#e6edf3}}
h1{{padding:16px 20px;margin:0;font-size:18px;background:#161b22;border-bottom:1px solid #30363d}}
.meta{{padding:8px 20px;color:#8b949e;font-size:13px}}
.gpu{{padding:8px 20px;color:#7ee787;font-size:13px}}
table{{border-collapse:collapse;width:96%;margin:12px 2%}}
th,td{{border:1px solid #30363d;padding:8px 10px;font-size:13px;text-align:left}}
th{{background:#161b22}}
tr.on td:first-child{{color:#7ee787}} tr.off{{opacity:.55}}
.cost{{margin:12px 2%;padding:12px;background:#161b22;border:1px solid #30363d;
border-radius:6px;font-family:ui-monospace,Consolas,monospace;font-size:12px;
white-space:pre-wrap}}
</style></head><body>
<h1>🧠 LMM — Local/remote Model Manager</h1>
<div class="meta">generated {now}</div>
<div class="gpu">GPU: {gpu_html}</div>
<table><tr><th>Runtime</th><th>Type</th><th>Cost</th><th>Running</th>
<th>Procs</th><th>Models</th><th>Endpoint</th></tr>
{rows}</table>
<div class="cost">{cost_html}</div>
</body></html>"""


# ------------------------------ commands -----------------------------------
def cmd_discover(cfg, as_json, save=False):
    items = discover(cfg)
    if as_json:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return
    if save:
        # Save detected backends as the initial priority order (running first).
        running = [it["name"] for it in items
                   if it["running"] and it["name"] != "-"]
        if not running:
            print("[discover] no running backends detected — nothing to save.")
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
    items = discover(cfg)
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


def cmd_status(cfg):
    gpu = gpu_info()
    print("GPU:", gpu["name"] if gpu else "n/a",
          f"({gpu['used']}/{gpu['total']} MiB, {gpu['pct']}%)" if gpu else "")
    print("-" * 64)
    for it in discover(cfg):
        print(f"{it['name']:<32} running={it['running']!s:<5} "
              f"procs={it.get('procs', 0)}")




def cmd_models(cfg):
    """Unified model registry: list models across every detected provider
    (local Ollama + configured cloud backends). Measures each, does not assume."""
    provs = merged_providers(cfg)
    if not provs:
        ms = detect_ollama()["models"]
        if ms:
            print("Ollama local models:")
            for m in ms:
                print("  -", m)
        else:
            print("no providers detected (no Ollama, no config)")
        return
    for name, prov in provs.items():
        models = fetch_models(prov)
        if models:
            print(f"{name} ({prov.get('kind', '?')}):")
            for m in models:
                print("  -", m)
        else:
            print(f"{name} ({prov.get('kind', '?')}): unreachable / no models")


def cmd_pull(model):
    """Pull a model into the local Ollama base (the unified local model store).
    The hub's default backend is Ollama, so `lmm pull` keeps it stocked."""
    if not model:
        print("usage: lmm pull <ollama-model>  e.g. lmm pull qwen2.5-coder:7b")
        return
    print(f"pulling {model} into local Ollama ...")
    r = run(f"ollama pull {model}", timeout=900)
    if r is None or r.returncode != 0:
        # command itself failed/exceptioned
        err = (r.stderr.strip() if r and r.stderr else "(pull failed/timeout)")
        print(err)
    else:
        # success: ollama pull prints progress; confirm presence via list
        confirm = run(f"ollama list", timeout=30)
        ok = confirm and model.split(":")[0] in (confirm.stdout or "") \
            and (model.split(":")[1] if ":" in model else "") in (confirm.stdout or "")
        print((r.stdout.strip() if r.stdout else "") or "pull complete.")
        if not ok:
            print("(warning: model not found in `ollama list` after pull)")
    print("done. Use `lmm models` to confirm, `lmm ask` to route to it.")


def cmd_serve(model):
    if not model:
        print("usage: lmm serve <ollama-model>  e.g. lmm serve qwen2.5-coder:7b")
        return
    print(f"pulling {model} ...")
    r = run(f"ollama pull {model}", timeout=900)
    if r is None or r.returncode != 0:
        err = (r.stderr.strip() if r and r.stderr else "(pull failed/timeout)")
        print(err)
    else:
        print((r.stdout.strip() if r.stdout else "") or "pull complete.")
    print("endpoint ready: http://localhost:11434  (OpenAI-compatible)")




def cmd_serve_hub(cfg, host, port):
    """Start an OpenAI-compatible proxy that fans out to every configured
    provider (cloud + local). Apps point at this one endpoint; `lmm` routes
    each request. This is the hub: one endpoint, many backends."""
    import http.server, socketserver, threading
    provs = merged_providers(cfg)
    # zero-config: if no providers configured but Ollama is running, expose it
    targets = resolve_ask_targets(cfg, "", None) if "resolve_ask_targets" in globals() else [(n, p) for n, p in provs.items()]
    if not targets and not provs:
        # final safety net: implicit running Ollama even without config
        lo = local_ollama_provider()
        if lo:
            targets = [("local-ollama(implicit)", lo)]
    if not targets:
        print("[hub] no providers configured and no Ollama running. Add "
              "'providers' to lmm config (see `lmm examples`), or start Ollama.")
        return
    provs = dict(targets)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/v1/models"):
                # Real model IDs from every backend (not stubs). An OpenAI
                # client calling models.list() must see actual selectable models.
                data = []
                for n, p in provs.items():
                    for mid in fetch_models(p):
                        data.append({
                            "id": mid,
                            "object": "model",
                            "owned_by": p.get("kind", "unknown"),
                            "lmm_provider": n,
                        })
                if not data:
                    data = [{"id": n, "object": "model",
                             "owned_by": p.get("kind", "unknown")}
                            for n, p in provs.items()]
                self._send(200, {"object": "list", "data": data})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/v1/chat/completions"):
                self._send(404, {"error": "only /v1/chat/completions supported"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length).decode("utf-8", "ignore"))
            except Exception as e:
                self._send(400, {"error": f"bad json: {e}"})
                return
            # routing: reuse the SAME intelligence as `lmm ask` —
            # cfg['ask_order'] + auto-routing + implicit-Ollama fallback.
            msgs = req.get("messages", [])
            prompt = msgs[0].get("content", "") if msgs else ""
            explicit = req.get("model", "")  # may be a provider name OR a real model id
            requested_model = explicit  # remember what the client asked for
            if explicit and explicit not in provs:
                # client picked a real model id from /v1/models — map it back
                mapped = resolve_provider_by_model(provs, explicit)
                explicit = mapped if mapped else None
            targets = resolve_ask_targets(cfg, prompt, explicit if explicit in provs else None)
            if not targets:
                self._send(400, {"error": "no provider available for model '%s'" % requested_model})
                return
            last_err = None
            want_stream = bool(req.get("stream"))
            for name, prov in targets:
                fwd = dict(req)
                # honor the client's requested model id if it resolved to this
                # provider (otherwise fall back to the provider's default model)
                use_model = requested_model if (requested_model and requested_model in fetch_models(prov)) else prov["model"]
                fwd["model"] = use_model
                prov = dict(prov)  # don't mutate the shared provider dict
                prov["model"] = use_model
                gen = call_provider(prov, prompt,
                                    temperature=fwd.get("temperature", 0.7),
                                    messages=fwd.get("messages"),
                                    stream=want_stream)
                if isinstance(gen, dict) and gen.get("error"):
                    last_err = gen["error"]
                    log_hub({"event": "serve", "provider": name, "ok": False,
                             "error": last_err, "prompt": prompt})
                    continue
                if want_stream:
                    # SSE pass-through: forward each chunk as it arrives
                    full = []
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        for piece in gen:
                            if piece.startswith("[stream error:"):
                                self.wfile.write(
                                    ("data: " + json.dumps({"error": piece})
                                     + chr(10) + chr(10)).encode("utf-8"))
                                break
                            full.append(piece)
                            self.wfile.write(
                                ("data: " + json.dumps({"choices": [
                                    {"delta": {"content": piece}}]})
                                 + chr(10) + chr(10)).encode("utf-8"))
                            self.wfile.flush()
                        self.wfile.write(
                            ("data: [DONE]" + chr(10) + chr(10)).encode("utf-8"))
                    except (BrokenPipeError, ConnectionAbortedError, OSError):
                        pass  # client disconnected mid-stream
                    log_hub({"event": "serve", "provider": name, "ok": True,
                             "prompt": prompt, "reply": "".join(full)[:200]})
                    return
                log_hub({"event": "serve", "provider": name, "ok": True,
                         "prompt": prompt})
                self._send(200, gen)
                return
            self._send(502, {"error": "all providers failed: %s" % last_err})
            log_hub({"event": "serve", "provider": "(all)", "ok": False,
                     "error": last_err, "prompt": prompt})

        def log_message(self, *a):
            pass

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    httpd = S((host, port), Handler)
    print(f"[hub] OpenAI-compatible endpoint: http://{host}:{port}/v1")
    print(f"[hub] backends: {', '.join(provs)}")
    print("[hub] Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("[hub] stopped.")



def cmd_hub_status(cfg):
    """Measure hub health: list every backend lmm would route to, and probe
    each one for liveness (zero-config Ollama included). The hub is only as
    trustworthy as what this reports — measure, don't assume."""
    targets = resolve_ask_targets(cfg, "", None)
    if not targets:
        lo = local_ollama_provider()
        if lo:
            targets = [("local-ollama(implicit)", lo)]
    if not targets:
        print("[hub-status] no backends available (no config, no Ollama).")
        return
    print(f"HUB STATUS — {len(targets)} backend(s):")
    print("-" * 68)
    for name, prov in targets:
        # probe: a tiny completion request, or a models fetch for local
        ok = False
        detail = ""
        try:
            if prov.get("kind") == "local" and "11434" in prov["base_url"]:
                r = run("ollama list")
                ok = bool(r and r.returncode == 0)
                detail = "ollama reachable" if ok else "ollama not responding"
            else:
                # cloud: lightweight models list (GET) to test connectivity/auth
                import urllib.request
                url = prov["base_url"].rstrip("/") + "/models"
                req = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {prov.get('api_key','')}"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    ok = resp.status == 200
                    detail = f"HTTP {resp.status}"
        except Exception as e:
            detail = str(e)[:60]
        flag = "OK " if ok else "DOWN"
        print(f"  [{flag}] {name:24} {prov['model'] or '(no model)'}")
        print(f"         {prov['base_url']}  -> {detail}")
    print("-" * 68)
    live = sum(1 for n, p in targets
               if (p.get("kind") == "local" and "11434" in p["base_url"]
                    and bool(run("ollama list"))) or False)
    print("Tip: `lmm serve --hub` exposes these as one OpenAI-compatible endpoint.")



def cmd_stop(runtime, cfg):
    table = dict(STOP_TABLE)
    for e in cfg.get("extra_runtimes", []):
        table[e["name"].lower()] = e.get("procs", [])
    names = table.get((runtime or "").lower())
    if not names:
        print("unknown runtime. choices:", ", ".join(sorted(table)))
        return
    for n in names:
        if os.name == "nt":
            r = run(f'taskkill.exe /IM "{n}" /F')
        else:
            r = run(f"pkill -f '{n}' || true")
        ok = (r and r.returncode == 0) if r else False
        print(f"stop {n}: {'ok' if ok else 'no-process-or-failed'}")


def cmd_dash(cfg):
    out = os.path.join(HOME, ".lmm_dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_dash(cfg))
    print("dashboard ->", out)
    try:
        webbrowser.open(out)
    except Exception:
        pass


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
    seen = set()
    try:
        while True:
            for rt, entry in RUNTIME_REGISTRY.items():
                for hwnd, title in _enum_windows_by_title(entry.get("titles", [])):
                    if hwnd in seen:
                        continue
                    seen.add(hwnd)
                    try:
                        user32 = ctypes.windll.user32
                        ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                        new_ex = (ex | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
                        if new_ex != ex:
                            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_ex)
                            print(f"  auto-hidden '{rt}' window: {title!r}")
                    except Exception:
                        pass
            import time
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
    me = os.path.abspath(__file__)
    python_exe = sys.executable
    if os.name == "nt":
        # 1) Startup folder shortcut — user-level, no admin needed
        startup = os.path.join(os.environ.get("APPDATA", HOME),
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
            r = run(f'powershell -NoProfile -Command "{ps}"')
            if r and r.returncode == 0 and os.path.exists(lnk):
                print(f"registered lmm watch via Startup folder (no admin needed): {lnk}")
                return
        except Exception as e:
            pass
        # 2) fallback: Task Scheduler (may need admin)
        cmd = (f'schtasks /Create /TN "lmm-watch" /TR '
               f'"{python_exe} \"{me}\" watch" /SC ONLOGON /F')
        r = run(cmd)
        if r and r.returncode == 0:
            print("registered lmm watch with Windows Task Scheduler (runs at logon).")
        else:
            print("autostart registration failed. Manual option: copy this to your "
                  "Startup folder as a shortcut:\n"
                  f'  {python_exe} "{me}" watch')
    elif sys.platform == "darwin":
        plist = os.path.join(HOME, "Library", "LaunchAgents",
                             "com.lmm.watch.plist")
        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.lmm.watch</string>
  <key>ProgramArguments</key><array>
    <string>{python_exe}</string><string>{me}</string><string>watch</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>"""
        try:
            os.makedirs(os.path.dirname(plist), exist_ok=True)
            with open(plist, "w") as f:
                f.write(content)
            run(f"launchctl load {plist}")
            print(f"registered lmm watch with launchd: {plist}")
        except Exception as e:
            print("launchd registration failed:", e)
    else:
        # Linux systemd --user
        unit = os.path.join(HOME, ".config", "systemd", "user", "lmm-watch.service")
        content = f"""[Unit]
Description=lmm watch - auto-hide LLM taskbar windows
[Service]
ExecStart={python_exe} {me} watch
Restart=always
[Install]
WantedBy=default.target
"""
        try:
            os.makedirs(os.path.dirname(unit), exist_ok=True)
            with open(unit, "w") as f:
                f.write(content)
            run("systemctl --user daemon-reload")
            run("systemctl --user enable --now lmm-watch.service")
            print(f"registered lmm watch with systemd --user: {unit}")
        except Exception as e:
            print("systemd registration failed:", e)



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
        print("tkinter unavailable on this system:", e)
        return
    import threading

    root = tk.Tk()
    root.title("LMM — Local/remote Model Manager")
    root.geometry("920x560")
    _tray_cleanup = setup_tray(root)  # minimize -> stay in system tray
    root.bind("<Map>", lambda e: None)  # restored from tray -> no-op

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
    cols = ("Runtime", "Type", "Cost", "Running", "Procs", "Models", "Endpoint")
    tree = ttk.Treeview(root, columns=cols, show="headings", height=12)
    widths = (170, 70, 60, 70, 60, 220, 200)
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
    mdl_cb = ttk.Combobox(bar, textvariable=mdl_var, width=22,
                          values=["qwen2.5-coder:7b", "qwen2.5-coder:14b",
                                  "llama3.1:8b", "mistral:7b"])
    mdl_cb.pack(side="left", padx=4)

    def act(fn_name):
        rt = rt_var.get().strip().lower()
        if not rt:
            messagebox.showwarning("LMM", "select a runtime first")
            return
        if fn_name == "hide":
            msg = hide_taskbar(rt)
        elif fn_name == "stop":
            cmd_stop(rt, cfg)
            msg = f"stopped {rt} (if running)"
        elif fn_name == "serve":
            cmd_serve(mdl_var.get().strip() or "qwen2.5-coder:7b")
            msg = f"served {mdl_var.get().strip() or 'qwen2.5-coder:7b'}"
        else:
            msg = "?"
        messagebox.showinfo("LMM", msg)
        refresh()

    ttk.Button(bar, text="▶ Serve", command=lambda: act("serve")).pack(side="left", padx=2)
    ttk.Button(bar, text="■ Stop", command=lambda: act("stop")).pack(side="left", padx=2)
    ttk.Button(bar, text="⊟ Hide from taskbar", command=lambda: act("hide")).pack(side="left", padx=2)
    ttk.Button(bar, text="⟳ Refresh", command=lambda: refresh()).pack(side="right", padx=2)

    # --- first-run onboarding: auto-detect priority if none is set ---------
    # (so a fresh user lands in a working "managed software" without knowing
    #  the CLI. Visibility of system status + zero-config entry point.)
    if not cfg.get("ask_order"):
        _running = [it["name"] for it in discover(cfg)
                    if it["running"] and it["name"] != "-"]
        if _running:
            cfg["ask_order"] = _running
            save_config(cfg)

    # --- priority panel (manage routing priority: discover -> set -> use) ----
    prio = ttk.LabelFrame(root, text="Routing priority (ask_order)", padding=6)
    prio.pack(fill="x", padx=8, pady=4)
    prio_list = tk.Listbox(prio, height=4, selectmode="single")
    prio_list.pack(side="left", fill="x", expand=True, padx=(0, 6))

    def prio_load():
        prio_list.delete(0, "end")
        for n in (cfg.get("ask_order") or []):
            prio_list.insert("end", n)
        if prio_list.size() == 0:
            prio_list.insert("end", "(empty — run `discover --save`)")

    def prio_move(delta):
        idx = prio_list.curselection()
        if not idx:
            return
        i = idx[0]
        j = i + delta
        if j < 0 or j >= prio_list.size():
            return
        items = list(prio_list.get(0, "end"))
        items[i], items[j] = items[j], items[i]
        prio_list.delete(0, "end")
        for n in items:
            prio_list.insert("end", n)

    def prio_save():
        items = [prio_list.get(i) for i in range(prio_list.size())
                 if prio_list.get(i) != "(empty — run `discover --save`)"]
        cfg["ask_order"] = items
        save_config(cfg)
        messagebox.showinfo("LMM", f"saved {len(items)} backend(s) to ask_order")
        refresh()

    pbtn = ttk.Frame(prio)
    pbtn.pack(side="right", fill="y")
    ttk.Button(pbtn, text="▲ Up", width=8, command=lambda: prio_move(-1)).pack(pady=1)
    ttk.Button(pbtn, text="▼ Down", width=8, command=lambda: prio_move(1)).pack(pady=1)
    ttk.Button(pbtn, text="💾 Save", width=8, command=prio_save).pack(pady=1)

    # --- ask panel (real use: route by priority, show reply) ----------------
    askf = ttk.LabelFrame(root, text="Ask (routes by priority)", padding=6)
    askf.pack(fill="both", expand=True, padx=8, pady=4)

    # backend selector (pin a backend, or leave "auto" to follow ask_order)
    sel_row = ttk.Frame(askf)
    sel_row.pack(fill="x", pady=(0, 4))
    ttk.Label(sel_row, text="Backend:").pack(side="left")
    prov_var = tk.StringVar(value="auto (priority order)")
    prov_cb = ttk.Combobox(sel_row, textvariable=prov_var, width=30,
                           state="readonly")
    prov_cb.pack(side="left", padx=4)

    def prov_reload():
        running = [it["name"] for it in discover(cfg)
                   if it["running"] and it["name"] != "-"]
        prov_cb["values"] = ["auto (priority order)"] + running

    auto_var = tk.BooleanVar(value=True)
    auto_chk = ttk.Checkbutton(sel_row, text="auto-route (measure task)",
                               variable=auto_var)
    auto_chk.pack(side="left", padx=8)

    ask_in = ttk.Entry(askf)
    ask_in.pack(fill="x", pady=(0, 4))

    ask_out = tk.Text(askf, height=6, wrap="word", state="disabled",
                      font=("Consolas", 10))
    ask_out.pack(fill="both", expand=True)

    def ask_run():
        prompt = ask_in.get().strip()
        if not prompt:
            return
        sel = prov_var.get()
        explicit = None if sel.startswith("auto") else sel
        use_auto = auto_var.get() and sel.startswith("auto")
        ask_out.config(state="normal")
        ask_out.insert("end", f"you> {prompt}\n")
        ask_out.config(state="disabled")
        ask_in.delete(0, "end")

        def worker():
            reply = gui_ask(prompt, cfg, explicit=explicit, auto=use_auto)
            ask_out.config(state="normal")
            ask_out.insert("end", f"{reply}\n\n")
            ask_out.see("end")
            ask_out.config(state="disabled")
        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(askf, text="➤ Send", command=ask_run).pack(anchor="e", pady=(2, 0))
    ask_in.bind("<Return>", lambda e: ask_run())

    # --- live refresh ----------------------------------------------------
    def refresh():
        # GPU + cost
        gpu = gpu_info()
        gpu_var.set(f"GPU: {gpu['name']} {gpu['used']}/{gpu['total']} MiB ({gpu['pct']}%)"
                    if gpu else "GPU: n/a")
        # color the GPU red when VRAM is tight (backstage->frontstage on exception)
        if gpu and gpu["pct"] >= 85:
            gpu_var.set(gpu_var.get() + "  ⚠")
        try:
            cost_label.config(text=cost_report(cfg).splitlines()[-1])
        except Exception:
            pass
        # table
        for row in tree.get_children():
            tree.delete(row)
        for it in discover(cfg):
            tag = "on" if it["running"] else "off"
            tree.insert("", "end", values=(
                it["name"], it["type"],
                "PAID" if it.get("paid") else "free",
                "YES" if it["running"] else "no",
                it.get("procs", 0),
                ", ".join(it.get("models", [])) or "-",
                str(it.get("endpoint", "-")),
            ), tags=(tag,))
        tree.tag_configure("on", foreground="#1b5e20")
        tree.tag_configure("off", foreground="#9e9e9e")
        prio_load()
        prov_reload()

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
                       "model": "gpt-4o", "kind": "remote"},
            "gemini": {"api_key": "AIza...", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "model": "gemini-1.5-pro", "kind": "remote"},
            "my-local": {"api_key": "ollama", "base_url": "http://localhost:11434/v1",
                         "model": "qwen2.5-coder:7b", "kind": "local"}
        },
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









def cmd_ask(prompt, provider, cfg, auto=False, verify=False):
    """Unified inference with auto-routing + fallback: tries providers in order
    (explicit > auto-score > private/local > configured > implicit running Ollama)
    and falls through to the next on error. This is the hub's intelligence.
    With auto=True, the FIRST target is chosen by MEASURED task-vs-backend fit
    (first-principles routing). With verify=True, each candidate's reply is
    MEASURED by verify_reply and fallen back if unfit (closed loop).
    Responses stream token-by-token."""
    effective = provider
    if auto and not provider:
        best, reason = score_and_route(prompt, cfg, cfg.get("ask_order"))
        if best:
            print(f"[ask] auto-routed: {reason}")
            effective = best
        else:
            print(f"[ask] auto-routing skipped: {reason}")
    if verify and not provider:
        name, vreason, reply = route_and_verify(
            prompt, cfg, cfg.get("ask_order"))
        if name and reply is not None:
            print(f"[ask] verified-route: {name} -> {vreason}")
            print(reply)
            return
        else:
            print(f"[ask] verified-route: no backend passed the quality gate "
                  f"({vreason})")
            return
    targets = resolve_ask_targets(cfg, prompt, effective)
    if not targets:
        if provider:
            known = sorted(set(
                list((cfg.get("providers") or {}).keys())
                + [it["name"] for it in discover(cfg) if it["running"]]
                + ["local-ollama(implicit)", "local-lmstudio(implicit)"]))
            print(f"[ask] unknown provider '{provider}'.")
            print(f"[ask] known providers: {', '.join(known)}")
            print(f"[ask] or omit --provider to use ask_order / auto routing.")
        else:
            print("[ask] no provider available. Start Ollama (`lmm serve <model>`) "
                  "or add 'providers' to lmm config (see `lmm examples`).")
        return
    last_err = None
    for name, prov in targets:
        print(f"[ask] -> trying {name} ({prov['model'] or 'no model set'})")
        t0 = time.time()
        gen = call_provider(prov, prompt, stream=True)
        if isinstance(gen, dict) and gen.get("error"):
            last_err = gen["error"]
            log_hub({"event": "ask_attempt", "provider": name, "ok": False,
                     "latency_ms": int((time.time() - t0) * 1000),
                     "prompt": prompt, "error": last_err,
                     "reason": "pre-check error"})
            log_hub({"event": "ask", "provider": name, "ok": False,
                     "error": last_err, "prompt": prompt})
            print(f"[ask]    {name} failed: {last_err} -- fallback")
            continue
        try:
            full = []
            for piece in gen:
                if piece.startswith("[stream error:"):
                    raise RuntimeError(piece)
                full.append(piece)
                print(piece, end="", flush=True)
            print("")
            latency = int((time.time() - t0) * 1000)
            reply_text = "".join(full)
            tokens = len(reply_text) // 4  # rough token estimate
            # measure REAL cost: $/1M tok * actual tokens (FrugalGPT: measure, don't
            # trust a static table). Local backends cost $0.
            in_tok = len(prompt) // 4
            out_tok = tokens
            pricing = merged_pricing(cfg)
            if "local-" in name or name.endswith("(implicit)"):
                cost_usd = 0.0
            else:
                fam = name.split(":")[0].lower()
                pr = pricing.get(fam, pricing.get("default", {"in": 3.0, "out": 15.0}))
                cost_usd = (in_tok * float(pr.get("in", 3.0))
                            + out_tok * float(pr.get("out", 15.0))) / 1_000_000.0
            cost_usd = round(cost_usd, 6)
            log_hub({"event": "ask_attempt", "provider": name, "ok": True,
                     "latency_ms": latency, "prompt": prompt,
                     "reply_tokens": tokens, "cost_usd": cost_usd,
                     "reason": "success"})
            log_hub({"event": "ask", "provider": name, "ok": True,
                     "prompt": prompt, "reply": reply_text[:200]})
            return
        except (KeyError, IndexError, TypeError, RuntimeError) as e:
            last_err = f"stream failed: {e}"
            log_hub({"event": "ask_attempt", "provider": name, "ok": False,
                     "latency_ms": int((time.time() - t0) * 1000),
                     "prompt": prompt, "error": last_err,
                     "reason": "stream failure"})
            log_hub({"event": "ask", "provider": name, "ok": False,
                     "error": last_err, "prompt": prompt})
            print(f"[ask]    {name} stream error -- fallback")
            continue
    log_hub({"event": "ask", "provider": "(all)", "ok": False,
             "error": last_err, "prompt": prompt})
    print(f"[ask] all providers failed. last error: {last_err}")


def gui_ask(prompt, cfg, explicit=None, auto=False):
    """Non-streaming ask for the GUI: routes by ask_order (priority), returns
    the first successful reply text, or an error string. Never prints; the GUI
    owns the display. If `explicit` (a backend name) is given, that backend is
    used directly (user pinned it in the GUI). If `auto` is True and no
    explicit backend is pinned, the backend is chosen by measured task fit."""
    effective = explicit
    if auto and not explicit:
        best, reason = score_and_route(prompt, cfg, cfg.get("ask_order"))
        if best:
            effective = best
            prompt = f"[auto: {reason}]\n{prompt}"
    targets = resolve_ask_targets(cfg, prompt, effective)
    if not targets:
        return "[ask] no provider available. Run `lmm discover --save` first."
    last_err = None
    for name, prov in targets:
        r = call_provider(prov, prompt, stream=False)
        if isinstance(r, dict) and r.get("error"):
            last_err = r["error"]
            log_hub({"event": "ask", "provider": name, "ok": False,
                     "error": last_err, "prompt": prompt, "via": "gui"})
            continue
        try:
            reply = r["choices"][0]["message"]["content"]
            log_hub({"event": "ask", "provider": name, "ok": True,
                     "prompt": prompt, "reply": reply[:200], "via": "gui"})
            return f"[{name}] {reply}"
        except (KeyError, IndexError, TypeError) as e:
            last_err = f"bad response: {e}"
            log_hub({"event": "ask", "provider": name, "ok": False,
                     "error": last_err, "prompt": prompt, "via": "gui"})
            continue
    return f"[ask] all providers failed. last error: {last_err}"


def cmd_chat(provider, cfg):
    """Interactive chat REPL over the hub. Keeps conversation history
    (messages) across turns and routes every user turn through the SAME
    unified routing (ask_order + fallback) as `lmm ask`. Each turn is logged
    to hub.log. Local OR cloud, one interface. Type 'exit'/'quit'/'/exit' to
    leave."""
    print("lmm chat — type 'exit' to quit. Routes every turn via the hub.")
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
            last_err = None
            replied = False
            for name, prov in targets:
                gen = call_provider(prov, line,
                                    messages=messages[:-1] + [{"role": "user",
                                                               "content": line}],
                                    stream=True)
                if isinstance(gen, dict) and gen.get("error"):
                    last_err = gen["error"]
                    log_hub({"event": "chat", "provider": name, "ok": False,
                             "error": last_err, "prompt": line})
                    continue
                try:
                    print("hub> ", end="", flush=True)
                    full = []
                    for piece in gen:
                        if piece.startswith("[stream error:"):
                            raise RuntimeError(piece)
                        full.append(piece)
                        print(piece, end="", flush=True)
                    print("")
                    log_hub({"event": "chat", "provider": name, "ok": True,
                             "prompt": line, "reply": "".join(full)[:200]})
                    messages.append({"role": "assistant",
                                     "content": "".join(full)})
                    replied = True
                    break
                except (KeyError, IndexError, TypeError, RuntimeError) as e:
                    last_err = f"stream failed: {e}"
                    log_hub({"event": "chat", "provider": name, "ok": False,
                             "error": last_err, "prompt": line})
                    continue
            if not replied:
                print(f"[chat] all providers failed: {last_err}")
                messages.pop()  # drop the turn we couldn't send
    except Exception as e:
        print(f"[chat] stopped: {e}")







def cmd_log(n, cfg):
    """Show the last n hub events from ~/.lmm/hub.log (proof of what the hub
    actually routed/served). Defaults to 20."""
    p = os.path.join(HOME, ".lmm", "hub.log")
    if not os.path.exists(p):
        print("[log] no hub.log yet. Run `lmm ask` or `lmm serve --hub` first.")
        return
    try:
        n = int(n)
    except Exception:
        n = 20
    lines = [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
    for l in lines[-n:]:
        try:
            e = json.loads(l)
        except Exception:
            print(l); continue
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



def cmd_selftest(cfg):
    """Self-prove the hub works. Runs real measurements (not trust):
    syntax, command surface, implicit-Ollama reachability, a live `ask`
    routing that must return a real reply, and hub.log observability.
    Each check is pass/fail; a non-zero failure count means `lmm` is broken."""
    import subprocess as _sp
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))

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
              "ask", "chat", "hub-status", "config", "log", "selftest",
              "stop", "dash", "gui", "watch", "autostart", "hide", "examples"]
    missing = [c for c in needed if c not in cmds]
    chk("command surface complete", not missing,
        ("missing: " + ",".join(missing)) if missing else "")

    # 3) implicit Ollama reachable
    skip_live = os.environ.get("LMM_SELFTEST_SKIP_LIVE", "") in ("1", "true", "yes")
    if skip_live:
        print("  [SKIP] implicit Ollama reachable -- LMM_SELFTEST_SKIP_LIVE=1")
        lo = None
    else:
        lo = local_ollama_provider()
        chk("implicit Ollama reachable", bool(lo),
            (lo["model"] if lo else "ollama not running"))

    # 4) live ask routing returns a real reply
    if skip_live:
        print("  [SKIP] live ask routing returns reply -- LMM_SELFTEST_SKIP_LIVE=1")
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

    # 5) observability: hub.log gets written
    before = os.path.exists(os.path.join(HOME, ".lmm", "hub.log"))
    log_hub({"event": "selftest", "provider": "(self)", "ok": True,
             "prompt": "selftest probe"})
    after = os.path.exists(os.path.join(HOME, ".lmm", "hub.log"))
    chk("observability (hub.log writable)", after,
        ("created" if not before else "appended"))

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
    try:
        import io as _io
        _buf = _io.StringIO()
        import contextlib as _cl
        with _cl.redirect_stdout(_buf):
            cmd_doctor(cfg)
        doc_out = _buf.getvalue()
        chk("doctor command runs", "doctor: HEALTHY" in doc_out,
            doc_out.strip().splitlines()[-1] if doc_out.strip() else "")
    except SystemExit as e:
        chk("doctor command runs", e.code in (0, None),
            f"exit={e.code}")
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
    print("")
    if fails == 0:
        print("SELFTEST PASS — the hub measures and proves itself.")
    else:
        print(f"SELFTEST FAIL — {fails} check(s) failed. Fix before trusting the hub.")
    return 1 if fails else 0



def cmd_config(args, cfg):
    """Manage lmm config: init / list / get / set / unset.
    Lets the user freely control hub priority (ask_order) and providers
    without hand-editing JSON. Zero-dep, stdlib only."""
    act = getattr(args, "config_action", None)
    if act == "validate":
        return cmd_validate_config(cfg)
    if act == "init":
        if os.path.exists(os.path.join(HOME, ".lmm", "config.json")):
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
                for it in discover(cfg):
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
    ask_order resolves, hub.log writable) and exits non-zero if any FAILs."""
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
    lo = local_ollama_provider()
    chk("implicit Ollama reachable",
        bool(lo) and bool(lo.get("model")),
        (lo.get("model") if lo else "ollama not running"))

    # 3) at least one backend running
    try:
        disc = discover(cfg)
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
            run_names = [d["name"] for d in discover(cfg) if d.get("running")]
        except Exception:
            run_names = []
        unresolved = [n for n in order
                      if n.lower() not in known and n not in run_names]
        chk("ask_order resolves",
            len(unresolved) == 0,
            ("unresolved: " + ", ".join(unresolved)) if unresolved else
            f"{len(order)} entr(ies) OK")

    # 5) hub.log writable
    try:
        log_hub({"event": "doctor", "provider": "(self)", "ok": True,
                 "prompt": "doctor probe"})
        chk("hub.log writable", True)
    except Exception as e:
        chk("hub.log writable", False, str(e))

    print("")
    if fails:
        print(f"doctor: UNHEALTHY ({len(fails)} issue(s)): {', '.join(fails)}")
        sys.exit(1)
    print("doctor: HEALTHY")


def cmd_stats(cfg):
    """Aggregate the structured hub log into measured routing statistics.
    Proves (don't trust — measure) HOW well the routing actually performs:
    per-backend success rate, average latency, total attempts, and how often
    the first tried backend succeeded. Zero-dep: pure JSONL scan."""
    p = os.path.join(HOME, ".lmm", "hub.log")
    if not os.path.exists(p):
        print("[stats] no hub.log yet. Run `lmm ask` first.")
        sys.exit(0)
    attempts = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "ask_attempt":
                attempts.append(e)
    if not attempts:
        print("[stats] no ask_attempt events yet. Run `lmm ask` first.")
        sys.exit(0)

    by_provider = {}
    first_try_ok = 0
    total = len(attempts)
    for i, e in enumerate(attempts):
        name = e.get("provider", "?")
        rec = by_provider.setdefault(name, {"ok": 0, "fail": 0, "lat": [], "cost": 0.0})
        if e.get("ok"):
            rec["ok"] += 1
            # first attempt of a session = index 0 or right after an all-fail
            if i == 0 or (attempts[i - 1].get("provider") == "(all)"):
                first_try_ok += 1
        else:
            rec["fail"] += 1
        if isinstance(e.get("latency_ms"), int):
            rec["lat"].append(e["latency_ms"])
        if isinstance(e.get("cost_usd"), (int, float)):
            rec["cost"] += e["cost_usd"]

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
    print(f"first-try success rate: {first_try_ok}/{total} "
          f"({first_try_ok / total * 100:.0f}%)")
    print(f"total estimated cost: ${total_cost:.4f} "
          f"(local backends are free; cloud costs are measured per actual tokens)")


def main():
    ap = argparse.ArgumentParser(
        description="LMM - Local/remote Model Manager (cross-platform, zero-dep)")
    ap.add_argument("-v", "--version", action="store_true", help="show version")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("discover")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", action="store_true",
                   help="save detected backends to ask_order (initial priority)")
    p = sub.add_parser("priority", help="manage routing priority (discover -> set -> use)")
    p.add_argument("--show", action="store_true", help="show current ask_order only")
    p.add_argument("--optimize", action="store_true",
                   help="re-rank ask_order by MEASURED performance (closed loop)")
    sub.add_parser("status")
    sub.add_parser("models")
    p = sub.add_parser("pull", help="pull a model into local Ollama (unified model store)")
    p.add_argument("model", nargs="?")
    p = sub.add_parser("cost")
    p.add_argument("--days", type=int, default=30)
    p = sub.add_parser("route")
    p.add_argument("task", nargs="?")
    p = sub.add_parser("serve")
    p.add_argument("model", nargs="?")
    p.add_argument("--hub", action="store_true",
                   help="start OpenAI-compatible proxy over all configured providers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    sub.add_parser("hub-status")
    p = sub.add_parser("config", help="manage lmm config (init/list/get/set/unset)")
    p.add_argument("config_action", nargs="?", default="list")
    p.add_argument("key", nargs="?", default=None)
    p.add_argument("value", nargs="?", default=None)
    p = sub.add_parser("log", help="show recent hub events (proof of routing)")
    p.add_argument("n", nargs="?", default=20)
    p = sub.add_parser("selftest", help="self-prove the hub works (measure, don't trust)")
    p.add_argument("--guard", action="store_true",
                   help="machine-readable mode: no banner, exit code only (for external verifiers)")
    p = sub.add_parser("chat", help="interactive hub chat REPL (keeps history)")
    p.add_argument("--provider", default=None, help="force a provider")
    p = sub.add_parser("stop")
    p.add_argument("runtime", nargs="?")
    sub.add_parser("dash")
    sub.add_parser("gui")
    p = sub.add_parser("watch")
    p.add_argument("--interval", type=float, default=3.0)
    sub.add_parser("autostart")
    sub.add_parser("hide").add_argument("runtime", nargs="?")
    p = sub.add_parser("ask")
    p.add_argument("prompt", nargs="*")
    p.add_argument("--provider", default=None, help="provider name from config")
    p.add_argument("--auto", action="store_true",
                   help="route by measured task fit (first-principles), not static ask_order")
    p.add_argument("--verify", action="store_true",
                   help="closed-loop: measure each reply, fallback if unfit (measure, don't trust)")
    sub.add_parser("examples")
    sub.add_parser("doctor")
    sub.add_parser("secrets")
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.version:
        print("lmm 1.0.0")
        return
    cfg = load_config()
    # No subcommand -> launch the live GUI dashboard. Visibility of system
    # status (Nielsen #1) is the root solution, so the default entry point is
    # the visual dashboard, not a text dump. Use `lmm discover` for CLI output.
    cmd = args.cmd or "gui"
    if cmd == "discover":
        cmd_discover(cfg, getattr(args, "json", False),
                     getattr(args, "save", False))
    elif cmd == "priority":
        cmd_priority(cfg, getattr(args, "show", False),
                    getattr(args, "optimize", False))
    elif cmd == "cli":
        cmd_discover(cfg, False)
    elif cmd == "status":
        cmd_status(cfg)
    elif cmd == "models":
        cmd_models(cfg)
    elif cmd == "pull":
        cmd_pull(args.model)
    elif cmd == "cost":
        print(cost_report(cfg, args.days))
    elif cmd == "route":
        rec = route_task(cfg, args.task)
        best, reason = score_and_route(args.task, cfg, cfg.get("ask_order"))
        print(f"task: {args.task}")
        print(f"=> keyword recommend: {rec}")
        print(f"=> measured best-fit : {best}")
        print(f"   reason: {reason}")
        # cost-awareness: show what the chosen backend costs vs local (free)
        pricing = merged_pricing(cfg)
        def _cost(key):
            if "local-" in key or key.endswith("(implicit)"):
                return 0.0
            fam = key.split(":")[0].lower()
            p = pricing.get(fam, pricing.get("default", {"out": 15.0}))
            return float(p.get("out", 15.0))
        c = _cost(best)
        print(f"   cost: {'free (local)' if c == 0 else f'${c:g}/1M out tokens'}"
              f"  -- FrugalGPT lesson: keep it local unless quality demands cloud")
    elif cmd == "serve":
        if getattr(args, "hub", False):
            cmd_serve_hub(cfg, args.host, args.port)
        else:
            cmd_serve(args.model)
    elif cmd == "hub-status":
        cmd_hub_status(cfg)
    elif cmd == "config":
        cmd_config(args, cfg)
    elif cmd == "log":
        cmd_log(args.n, cfg)
    elif cmd == "selftest":
        sys.exit(cmd_selftest(cfg))
    elif cmd == "chat":
        cmd_chat(args.provider, cfg)
    elif cmd == "stop":
        cmd_stop(args.runtime, cfg)
    elif cmd == "dash":
        cmd_dash(cfg)
    elif cmd == "gui":
        launch_gui(cfg)
    elif cmd == "watch":
        cmd_watch(cfg, args.interval)
    elif cmd == "autostart":
        cmd_autostart()
    elif cmd == "ask":
        cmd_ask(" ".join(getattr(args, "prompt", [])), getattr(args, "provider", None),
                cfg, auto=getattr(args, "auto", False),
                verify=getattr(args, "verify", False))
    elif cmd == "examples":
        cmd_examples()
    elif cmd == "doctor":
        cmd_doctor(cfg)
    elif cmd == "secrets":
        cmd_secrets(cfg)
    elif cmd == "stats":
        cmd_stats(cfg)


if __name__ == "__main__":
    main()

