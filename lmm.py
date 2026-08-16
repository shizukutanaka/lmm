#!/usr/bin/env python3
# lmm.py - Local/remote Model Manager
# ---------------------------------------------------------------------------
# One-file, zero-dependency manager for every LLM runtime on your machine.
# Cross-platform (Windows / macOS / Linux). Detects local (Ollama, LM Studio,
# llama.cpp) and remote (Claude Code, your own agents) runtimes, shows live
# status + GPU usage, estimates Anthropic cost from real session logs, and
# recommends local-vs-remote routing for a task.
#
#   No secrets are ever stored. Existing credentials are only *checked*, never
#   copied or saved. All paths are derived from your home directory.
#
# Install (any OS):
#   curl -fsSL https://raw.githubusercontent.com/<you>/lmm/main/lmm.py -o ~/.local/bin/lmm
#   chmod +x ~/.local/bin/lmm
#   # or just:  python lmm.py <command>
#
# Commands:
#   discover [--json]   list every detected runtime
#   status              live status + GPU
#   models              local models installed
#   cost [--days N]     measured Anthropic token cost from session logs
#   route "task ..."    recommend local vs remote for a task
#   serve <model>       pull + expose a local model endpoint (Ollama)
#   stop  <runtime>     stop a running runtime
#   dash                generate + open a self-contained HTML dashboard
#   examples            show a sample config file you can copy
#
# Config (~/.lmm/config.json) lets anyone add their own runtimes, override
# pricing, or change routing keywords. See `lmm examples`.
# ---------------------------------------------------------------------------
import os
import sys
import json
import argparse
import subprocess
import datetime
import html
import webbrowser
import ctypes

HOME = os.path.expanduser("~")

# ---- Public pricing (USD per 1M tokens) — approximate, editable ----
# Claude families (measured from ~/.claude session logs when available):
DEFAULT_PRICING = {
    "opus":   {"in": 15.0,  "out": 75.0,  "cw": 18.75, "cr": 1.50},
    "sonnet": {"in": 3.0,   "out": 15.0,  "cw": 3.75,  "cr": 0.30},
    "haiku":  {"in": 0.25,  "out": 1.25,  "cw": 0.30,  "cr": 0.03},
    "default":{"in": 3.0,   "out": 15.0,  "cw": 3.75,  "cr": 0.30},
    # Cloud API providers (no local session log; estimate-only from token counts):
    # values are [input, output] USD / 1M tokens (cache fields unused -> 0).
    "openai-gpt4o":     {"in": 2.50,  "out": 10.0,  "cw": 0.0, "cr": 0.0},
    "openai-gpt4o-mini":{"in": 0.15,  "out": 0.60,  "cw": 0.0, "cr": 0.0},
    "gemini-1.5-pro":   {"in": 1.25,  "out": 5.0,   "cw": 0.0, "cr": 0.0},
    "gemini-1.5-flash": {"in": 0.075, "out": 0.30,  "cw": 0.0, "cr": 0.0},
    "mistral-large":    {"in": 2.0,   "out": 6.0,   "cw": 0.0, "cr": 0.0},
    "groq-llama":       {"in": 0.59,  "out": 0.79,  "cw": 0.0, "cr": 0.0},
    "deepseek-chat":    {"in": 0.27,  "out": 1.10,  "cw": 0.0, "cr": 0.0},
    "cohere-command":   {"in": 1.0,   "out": 3.0,   "cw": 0.0, "cr": 0.0},
    "together-llama":   {"in": 0.80,  "out": 0.80,  "cw": 0.0, "cr": 0.0},
}

DEFAULT_ROUTE = {
    "private": ["secret", "private", "local", "offline", "社内", "秘密",
                "プライベート", "ローカル", "オフライン"],
    "heavy":   ["code", "design", "refactor", "debug", "architecture",
                "リファクタ", "設計", "コード", "デバッグ", "大規模"],
}

# runtime name -> process names to kill
STOP_TABLE = {
    "ollama":       ["ollama.exe", "ollama", "ollama app.exe", "ollama app"],
    "lmstudio":     ["LM Studio.exe", "lmstudio", "lmstudio.exe"],
    "claude":       ["claude.exe", "claude"],
    "llama-server": ["llama-server.exe", "llama-server", "server.exe", "./server"],
}

# --- RUNTIME REGISTRY: every major / similar LLM service -------------------
# One entry per known desktop / local / remote LLM app. `lmm` uses this to
# detect, hide-from-taskbar, and advise how to launch each headlessly.
# Fields:
#   procs   : process image names to match (cross-platform)
#   titles  : visible window-title substrings that identify its GUI window
#   kind    : "local" | "remote"
#   paid    : bool (display only)
#   headless: how to run WITHOUT a taskbar-occupying GUI, or None
#   appdirs : per-OS candidate app-data dir names (for "installed" detection)
RUNTIME_REGISTRY = {
    "ollama": {
        "procs": ["ollama.exe", "ollama", "ollama app.exe", "ollama app"],
        "titles": [],
        "kind": "local", "paid": False,
        "headless": "Ollama starts as a background service; "
                    "`lmm serve <model>` loads a model with no GUI.",
        "appdirs": ["Ollama"], "data": "~/.ollama",
    },
    "lmstudio": {
        "procs": ["LM Studio.exe", "lmstudio", "lmstudio.exe"],
        "titles": ["LM Studio"],
        "kind": "local", "paid": False,
        "headless": "Run headless: `lms server start` (no GUI, no taskbar).",
        "appdirs": ["LM-Studio", "lmstudio", ".lmstudio"], "data": "~/.lmstudio",
    },
    "claude": {
        "procs": ["claude.exe", "claude"],
        "titles": ["Claude"],
        "kind": "remote", "paid": True,
        "headless": "CLI already has no window when run in a terminal. To keep "
                    "the desktop app off the taskbar use `lmm hide claude`.",
        "appdirs": ["claude", "Claude", ".claude"], "data": "~/.claude",
    },
    "chatgpt": {
        "procs": ["ChatGPT.exe", "chatgpt"],
        "titles": ["ChatGPT"],
        "kind": "remote", "paid": False,
        "headless": "No official headless mode. Use `lmm hide chatgpt` to strip "
                    "its taskbar button (app keeps running).",
        "appdirs": ["ChatGPT"], "data": None,
    },
    "cursor": {
        "procs": ["Cursor.exe", "cursor"],
        "titles": ["Cursor"],
        "kind": "remote", "paid": False,
        "headless": "`lmm hide cursor` removes its taskbar button; app keeps "
                    "running. Editor itself stays usable from the system tray.",
        "appdirs": ["Cursor", ".cursor"], "data": None,
    },
    "perplexity": {
        "procs": ["Perplexity.exe", "perplexity"],
        "titles": ["Perplexity"],
        "kind": "remote", "paid": False,
        "headless": "`lmm hide perplexity` removes its taskbar button.",
        "appdirs": ["Perplexity"], "data": None,
    },
    "jan": {
        "procs": ["Jan.exe", "jan"],
        "titles": ["Jan"],
        "kind": "local", "paid": False,
        "headless": "Headless server: `jan server start` (OpenAI-compatible API).",
        "appdirs": ["Jan", ".jan"], "data": "~/.jan",
    },
    "gpt4all": {
        "procs": ["gpt4all.exe", "gpt4all"],
        "titles": ["GPT4All"],
        "kind": "local", "paid": False,
        "headless": "`lmm hide gpt4all` removes its taskbar button.",
        "appdirs": ["GPT4All", "gpt4all"], "data": None,
    },
    "anythingllm": {
        "procs": ["anythingllm.exe", "anythingllm", "anything-llm"],
        "titles": ["AnythingLLM"],
        "kind": "local", "paid": False,
        "headless": "`lmm hide anythingllm` removes its taskbar button.",
        "appdirs": ["AnythingLLM", "anythingllm"], "data": None,
    },
    "chatbox": {
        "procs": ["Chatbox.exe", "chatbox", "Chatbox AI.exe"],
        "titles": ["Chatbox"],
        "kind": "remote", "paid": False,
        "headless": "`lmm hide chatbox` removes its taskbar button.",
        "appdirs": ["Chatbox"], "data": None,
    },
    "msty": {
        "procs": ["Msty.exe", "msty"],
        "titles": ["Msty"],
        "kind": "local", "paid": False,
        "headless": "`lmm hide msty` removes its taskbar button.",
        "appdirs": ["Msty"], "data": None,
    },
    "koboldcpp": {
        "procs": ["koboldcpp.exe", "koboldcpp", "koboldcpp_cu11.exe"],
        "titles": ["KoboldCpp"],
        "kind": "local", "paid": False,
        "headless": "Runs as a console/server with no taskbar button. "
                    "Open its web UI in a browser.",
        "appdirs": ["KoboldCPP", "koboldcpp"], "data": None,
    },
    "openwebui": {
        "procs": ["open-webui", "open_webui", "uvicorn"],
        "titles": ["Open WebUI"],
        "kind": "local", "paid": False,
        "headless": "Browser-based; runs as a server (no taskbar window). "
                    "Launch: `open-webui serve`.",
        "appdirs": ["OpenWebUI"], "data": None,
    },
    "vllm": {
        "procs": ["vllm", "vllm.entrypoints"],
        "titles": [],
        "kind": "local", "paid": False,
        "headless": "Server-only (`python -m vllm.entrypoints.openai.api_server`). "
                    "No GUI, no taskbar.",
        "appdirs": [".vllm"], "data": None,
    },
    "devin": {
        "procs": ["devin.exe", "devin", "cua-driver.exe", "cua-driver"],
        "titles": ["Cua", "Devin", "AgentCursorOverlay"],
        "kind": "remote", "paid": True,
        "headless": "`lmm hide devin` removes its agent overlay / taskbar entry.",
        "appdirs": ["Devin", "Cua"], "data": None,
    },
    "llamacpp": {
        "procs": ["llama-server.exe", "llama-server", "server.exe", "./server"],
        "titles": [],
        "kind": "local", "paid": False,
        "headless": "Server-only; no GUI. Open the API/UI in a browser.",
        "appdirs": [], "data": None,
    },
}

# runtime name -> window-title keywords for hide (derived; editable)
HIDE_KEYWORDS = {k: v.get("titles", []) for k, v in RUNTIME_REGISTRY.items()}

CLAUDE_PROJECTS = os.path.join(HOME, ".claude", "projects")
CLAUDE_CREDS    = os.path.join(HOME, ".claude", ".credentials.json")


# ----------------------------- low level helpers ---------------------------
def run(cmd):
    """Run a shell command, tolerating Windows CP932 output. Returns a
    subprocess.CompletedProcess-like object with .stdout/.stderr as text, or
    None on failure."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=25)
        r.stdout = r.stdout.decode("utf-8", "ignore") if r.stdout else ""
        r.stderr = r.stderr.decode("utf-8", "ignore") if r.stderr else ""
        return r
    except Exception:
        return None


def proc_count(names):
    """Count running processes whose image name matches any of `names`
    (case-insensitive). Cross-platform."""
    names = [n.lower() for n in names]
    if os.name == "nt":
        total = 0
        for n in names:
            r = run(f'tasklist.exe /FI "IMAGENAME eq {n}" /FO CSV /NH')
            if r and r.stdout:
                total += sum(1 for line in r.stdout.splitlines() if n in line.lower())
        return total
    r = run("pgrep -fl " + " ".join(f"'{n}'" for n in names) + " || true")
    return len([l for l in (r.stdout or "").splitlines() if l.strip()]) if r else 0


def app_data():
    if os.name == "nt":
        return os.environ.get("LOCALAPPDATA", os.path.join(HOME, "AppData", "Local"))
    if sys.platform == "darwin":
        return os.path.join(HOME, "Library", "Application Support")
    return os.environ.get("XDG_DATA_HOME", os.path.join(HOME, ".local", "share"))


def gpu_info():
    r = run("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
    if r and r.stdout.strip():
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        if len(parts) >= 3:
            try:
                u = int(parts[1].replace("MiB", ""))
                t = int(parts[2].replace("MiB", ""))
                return {"name": parts[0], "used": u, "total": t, "pct": round(100 * u / t)}
            except ValueError:
                pass
    return None


# ------------------------------- config ------------------------------------
def load_config():
    cands = [
        os.path.join(HOME, ".lmm", "config.json"),
        os.path.join(HOME, ".config", "lmm", "config.json"),
        "lmm.config.json",
    ]
    for c in cands:
        if os.path.exists(c):
            try:
                with open(c, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def merged_pricing(cfg):
    p = dict(DEFAULT_PRICING)
    for k, v in (cfg.get("pricing") or {}).items():
        p[k] = v
    return p


def merged_route(cfg):
    r = {k: list(v) for k, v in DEFAULT_ROUTE.items()}
    for k, v in (cfg.get("route") or {}).items():
        r[k] = v
    return r


def merged_providers(cfg):
    """Return the cloud/remote provider registry from config.

    Each provider entry: {api_key, base_url, model, kind}. Used by `lmm ask`
    and `lmm serve --hub` so a single `lmm` command can target any backend
    through one OpenAI-compatible interface.
    """
    provs = {}
    for name, v in (cfg.get("providers") or {}).items():
        if not isinstance(v, dict):
            continue
        provs[name] = {
            "api_key": v.get("api_key", ""),
            "base_url": v.get("base_url", "https://api.openai.com/v1"),
            "model": v.get("model", ""),
            "kind": v.get("kind", "remote"),
        }
    return provs


def http_post_json(url, payload, api_key, timeout=60):
    """Minimal OpenAI-compatible chat completion call (zero-dep, stdlib only)."""
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception as e:  # network / auth errors must surface, not silently pass
        return {"error": str(e)}


def call_provider(prov, prompt, temperature=0.7):
    """Send `prompt` to one provider (OpenAI-compatible /chat/completions)."""
    if not prov.get("model"):
        return {"error": "provider has no model set"}
    url = prov["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": prov["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    return http_post_json(url, payload, prov.get("api_key", ""))


def pick_provider_for_task(cfg, task, provs):
    """Choose a provider via the same keyword logic as route_task, but over
    the configured cloud providers. Falls back to the first provider."""
    rt = merged_route(cfg)
    t = (task or "").lower()
    if any(k in t for k in rt.get("private", [])):
        for name, p in provs.items():
            if p.get("kind") == "local":
                return name
    for name in provs:
        if name.lower() in t:
            return name
    return next(iter(provs), None)


# ----------------------------- detectors -----------------------------------
def detect_ollama():
    r = run("ollama list")
    models = []
    if r and r.returncode == 0:
        for ln in r.stdout.strip().splitlines()[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
    procs = proc_count(["ollama.exe", "ollama", "ollama app.exe", "ollama app"])
    running = bool(models) or procs > 0
    return {
        "name": "Ollama", "type": "local", "paid": False,
        "running": running, "procs": procs, "models": models,
        "endpoint": "http://localhost:11434" if running else "-",
        "installed": running or os.path.isdir(os.path.join(HOME, ".ollama")),
    }


def detect_lmstudio():
    ad = app_data()
    paths = [os.path.join(ad, "LM Studio"), os.path.join(ad, "lmstudio"),
             os.path.join(HOME, ".lmstudio")]
    installed = any(os.path.isdir(p) for p in paths)
    procs = proc_count(["LM Studio.exe", "lmstudio", "lmstudio.exe"])
    running = procs > 0
    return {
        "name": "LM Studio", "type": "local", "paid": False,
        "running": running, "procs": procs, "models": [],
        "endpoint": "http://localhost:1234/v1" if running else "-",
        "installed": installed,
    }


def detect_llamacpp():
    procs = proc_count(["llama-server.exe", "llama-server", "server.exe", "./server"])
    return {
        "name": "llama.cpp server", "type": "local", "paid": False,
        "running": procs > 0, "procs": procs, "models": [], "endpoint": "-",
        "installed": False,
    }


def detect_claude():
    procs = proc_count(["claude.exe", "claude"])
    creds = os.path.exists(CLAUDE_CREDS)
    proj = os.path.isdir(CLAUDE_PROJECTS)
    return {
        "name": "Claude Code (Anthropic)", "type": "remote", "paid": True,
        "running": procs > 0, "procs": procs, "models": [],
        "endpoint": "api.anthropic.com", "installed": proj or creds,
        "has_auth": creds,
    }


# --------------------- taskbar-hiding (Windows only) -----------------------
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
GWL_EXSTYLE = -20


def _enum_windows_by_title(keywords):
    """Return list of (hwnd, title) for visible top-level windows whose title
    contains one of `keywords` (case-insensitive). Windows-only. Safe even if
    the window list changes mid-enumeration."""
    if os.name != "nt":
        return []
    try:
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                                         ctypes.c_void_p)
    except Exception:
        return []
    results = []
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    GetWindowTextW = user32.GetWindowTextW
    IsWindowVisible = user32.IsWindowVisible
    buf_t = ctypes.create_unicode_buffer

    def cb(hwnd, lparam):
        try:
            if IsWindowVisible(hwnd):
                n = GetWindowTextLengthW(hwnd)
                if n > 0:
                    buf = buf_t(n + 1)
                    GetWindowTextW(hwnd, buf, n + 1)
                    t = buf.value
                    if any(k.lower() in t.lower() for k in keywords):
                        results.append((int(hwnd), t))
        except Exception:
            pass
        return True

    try:
        user32.EnumWindows.argtypes = [WNDENUMPROC, ctypes.c_void_p]
        user32.EnumWindows.restype = ctypes.c_bool
        user32.EnumWindows(WNDENUMPROC(cb), 0)
    except Exception:
        pass
    return results


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


def detect_extra(cfg):
    out = []
    for e in cfg.get("extra_runtimes", []):
        procs = proc_count(e.get("procs", []))
        installed = any(os.path.exists(p.replace("~", HOME))
                        for p in e.get("installed_paths", []))
        running = procs > 0 or installed
        models = []
        mc = e.get("models_cmd")
        if mc:
            r = run(mc)
            if r and r.returncode == 0:
                models = [l for l in r.stdout.strip().splitlines() if l.strip()]
        out.append({
            "name": e["name"], "type": e.get("type", "local"),
            "paid": e.get("paid", False), "running": running, "procs": procs,
            "models": models, "endpoint": e.get("endpoint", "-"),
            "installed": installed,
        })
    return out


def discover(cfg):
    items = [detect_ollama(), detect_lmstudio(), detect_llamacpp(),
             detect_claude()]
    items += detect_extra(cfg)
    return items


# ------------------------------ cost ---------------------------------------
def model_family(s):
    s = (s or "").lower()
    for fam in ("opus", "sonnet", "haiku"):
        if fam in s:
            return fam
    return "default"


def measured_tokens():
    if not os.path.isdir(CLAUDE_PROJECTS):
        return None
    agg = {}
    total_sessions = 0
    for root, _, files in os.walk(CLAUDE_PROJECTS):
        for f in files:
            if not f.endswith(".jsonl"):
                continue
            total_sessions += 1
            fam = "default"
            seen = False
            path = os.path.join(root, f)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        ms = json.dumps(d)
                        if not seen:
                            for fam_cand in ("opus", "sonnet", "haiku"):
                                if fam_cand in ms.lower():
                                    fam = fam_cand
                                    seen = True
                                    break
                        def walk(o):
                            if isinstance(o, dict):
                                if isinstance(o.get("usage"), dict):
                                    u = o["usage"]
                                    yield (u.get("input_tokens", 0),
                                           u.get("output_tokens", 0),
                                           u.get("cache_creation_input_tokens", 0),
                                           u.get("cache_read_input_tokens", 0))
                                for v in o.values():
                                    yield from walk(v)
                            elif isinstance(o, list):
                                for v in o:
                                    yield from walk(v)
                        for i, o, cw, cr in walk(d):
                            a = agg.setdefault(fam, {"in": 0, "out": 0,
                                                     "cw": 0, "cr": 0, "sessions": set()})
                            a["in"] += i
                            a["out"] += o
                            a["cw"] += cw
                            a["cr"] += cr
                            a["sessions"].add(f)
            except Exception:
                pass
    for a in agg.values():
        a["sessions"] = len(a["sessions"])
    return {"by_family": agg, "total_sessions": total_sessions}


def cost_report(cfg, days=30):
    pricing = merged_pricing(cfg)
    data = measured_tokens()
    if not data:
        return "No Claude session logs found at ~/.claude/projects"
    out = [f"Anthropic measured usage (all-time, {data['total_sessions']} sessions)",
           "-" * 64]
    grand = 0.0
    for fam, a in sorted(data["by_family"].items(),
                         key=lambda x: -(x[1]["in"] + x[1]["out"])):
        p = pricing.get(fam, pricing["default"])
        cin = a["in"] / 1e6 * p["in"]
        cout = a["out"] / 1e6 * p["out"]
        ccw = a["cw"] / 1e6 * p["cw"]
        ccr = a["cr"] / 1e6 * p["cr"]
        cost = cin + cout + ccw + ccr
        grand += cost
        out.append(f"[{fam}] sessions={a['sessions']}")
        out.append(f"   in={a['in']:,} out={a['out']:,} "
                   f"cache_w={a['cw']:,} cache_r={a['cr']:,}")
        out.append(f"   est ${cost:,.2f}  (in ${cin:,.2f} + out ${cout:,.2f} "
                   f"+ cw ${ccw:,.2f} + cr ${ccr:,.2f})")
    out.append("-" * 64)
    out.append(f"TOTAL est ${grand:,.2f}  "
               "(pricing approximate; verify on your Anthropic billing page)")
    # ---- cross-provider estimate (cloud APIs have no local session log) ----
    # Use the local Ollama token volume as a proxy baseline: if the same
    # workload ran on each cloud provider, what would it cost? This makes
    # `lmm cost` cover local->cloud without requiring API telemetry.
    proxy = 0
    if data and data.get("by_family"):
        proxy = max((a["in"] + a["out"] for a in data["by_family"].values()),
                    default=0)
    if proxy <= 0:
        proxy = 1_000_000  # default 1M-token baseline for comparison
    out.append("")
    out.append("=" * 64)
    out.append(f"CROSS-PROVIDER ESTIMATE (baseline = "
               f"{proxy/1e6:.2f}M tok in+out, illustrative)")
    out.append("-" * 64)
    cloud_keys = [k for k in pricing
                  if k not in ("opus", "sonnet", "haiku", "default")]
    for k in sorted(cloud_keys):
        p = pricing[k]
        # assume ~50/50 in/out split of the baseline for a simple comparison
        c = (proxy / 2 / 1e6 * p["in"]) + (proxy / 2 / 1e6 * p["out"])
        out.append(f"  {k:22} ~${c:,.2f}")
    out.append("")
    out.append("Tip: set real cloud usage in lmm config 'usage' to replace "
               "this estimate with measured cost.")
    # ---- measured cloud usage (user-provided, replaces estimate) ----
    # cfg['usage'] accepts either a USD amount per provider:
    #   "usage": {"openai": 12.50, "gemini": 3.20}
    # or token counts priced via the rate table:
    #   "usage": {"openai": {"in": 1000000, "out": 2000000}}
    usage = cfg.get("usage") or {}
    if usage:
        out.append("")
        out.append("=" * 64)
        out.append("MEASURED CLOUD USAGE (from lmm config 'usage')")
        out.append("-" * 64)
        measured = 0.0
        for name, val in sorted(usage.items()):
            if isinstance(val, dict):
                p = pricing.get(name, pricing["default"])
                c = (val.get("in", 0) / 1e6 * p["in"]
                     + val.get("out", 0) / 1e6 * p["out"])
                out.append(f"  {name:22} {val}  = ${c:,.2f}")
            else:
                c = float(val)
                out.append(f"  {name:22} ${c:,.2f}")
            measured += c
        out.append("-" * 64)
        out.append(f"MEASURED CLOUD TOTAL  ${measured:,.2f}")
        out.append(f"ALL-IN TOTAL (Claude + cloud)  ${grand + measured:,.2f}")
    return "\n".join(out)


# ------------------------------ routing ------------------------------------
def route_task(cfg, task):
    t = (task or "").lower()
    rt = merged_route(cfg)
    gpu = gpu_info()
    local_ok = detect_ollama()["running"]
    if any(k in t for k in rt["private"]):
        if local_ok:
            rec = "Ollama (local, free, private)"
        else:
            rec = "start a local runtime first: lmm serve <model>"
    elif any(k in t for k in rt["heavy"]):
        rec = "Claude Code (remote, paid) — best for coding/design"
    else:
        if gpu and gpu["pct"] < 80 and local_ok:
            rec = "Ollama (local, free) — GPU headroom available"
        else:
            rec = "Claude Code (remote, paid)"
    return rec


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
def cmd_discover(cfg, as_json):
    items = discover(cfg)
    if as_json:
        print(json.dumps(items, indent=2, ensure_ascii=False))
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
    for it in discover(cfg):
        print(f"{it['name']:<32} running={it['running']!s:<5} "
              f"procs={it.get('procs', 0)}")


def cmd_models():
    ms = detect_ollama()["models"]
    if ms:
        print("Ollama local models:")
        for m in ms:
            print("  -", m)
    else:
        print("no ollama models (or ollama not running)")


def cmd_serve(model):
    if not model:
        print("usage: lmm serve <ollama-model>  e.g. lmm serve qwen2.5-coder:7b")
        return
    print(f"pulling {model} ...")
    r = run(f"ollama pull {model}")
    print((r.stdout.strip() if r else "(pull failed/timeout)"))
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
                self._send(200, {"object": "list", "data": [
                    {"id": n, "object": "model", "owned_by": p["kind"]}
                    for n, p in provs.items()]})
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
            explicit = req.get("model", "")  # may be a configured provider name
            targets = resolve_ask_targets(cfg, prompt, explicit if explicit in provs else None)
            if not targets:
                self._send(400, {"error": "no provider available for model '%s'" % explicit})
                return
            last_err = None
            for name, prov in targets:
                fwd = dict(req)
                fwd["model"] = prov["model"]
                res = call_provider(prov, prompt, temperature=fwd.get("temperature", 0.7))
                if isinstance(res, dict) and res.get("error"):
                    last_err = res["error"]
                    continue
                self._send(200, res)
                return
            self._send(502, {"error": "all providers failed: %s" % last_err})

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
    _tray_cleanup = setup_tray(root)
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


def local_ollama_provider():
    """Synthesize an implicit local provider from a running Ollama, so `lmm ask`
    works even with no config (zero-config hub). Returns dict or None."""
    d = detect_ollama()
    if not d.get("running"):
        return None
    models = d.get("models") or []
    model = models[0] if models else "qwen2.5-coder:3b"
    return {"api_key": "ollama", "base_url": "http://localhost:11434/v1",
            "model": model, "kind": "local", "_implicit": True}


def resolve_ask_targets(cfg, prompt, explicit):
    """Ordered providers to try. Priority is USER-CONTROLLED via cfg['ask_order']
    (a list of provider names). Falls back to implicit running Ollama.

    Order:  explicit  >  cfg['ask_order']  >  (keyword-matched)  >  implicit
    running Ollama as final safety net. With no ask_order set, the old default
    (keyword match, then all configured, then implicit Ollama) applies.
    """
    provs = merged_providers(cfg)
    if explicit:
        if explicit in provs:
            return [(explicit, provs[explicit])]
        lo = local_ollama_provider()
        if explicit in ("local", "ollama", "local-ollama") and lo:
            return [("local-ollama(implicit)", lo)]
        return []
    order = list(cfg.get("ask_order") or [])
    out, seen = [], set()
    for n in order:                      # user-defined priority, in full
        if n in provs and n not in seen:
            out.append((n, provs[n])); seen.add(n)
    if not order:                       # no ask_order: sensible default
        pk = pick_provider_for_task(cfg, prompt, provs)
        if pk and pk not in seen:
            out.append((pk, provs[pk])); seen.add(pk)
        for n, p in provs.items():
            if n not in seen:
                out.append((n, p)); seen.add(n)
    lo = local_ollama_provider()        # always-on zero-config safety net
    if lo and "local-ollama(implicit)" not in seen:
        out.append(("local-ollama(implicit)", lo))
    return out



def cmd_ask(prompt, provider, cfg):
    """Unified inference with auto-routing + fallback: tries providers in order
    (explicit > private/local > configured > implicit running Ollama) and
    falls through to the next on error. This is the hub's intelligence. Priority is set by cfg['ask_order']."""
    targets = resolve_ask_targets(cfg, prompt, provider)
    if not targets:
        print("[ask] no provider available. Start Ollama (`lmm serve <model>`) "
              "or add 'providers' to lmm config (see `lmm examples`).")
        return
    last_err = None
    for name, prov in targets:
        print(f"[ask] -> trying {name} ({prov['model'] or 'no model set'})")
        res = call_provider(prov, prompt)
        if isinstance(res, dict) and res.get("error"):
            last_err = res["error"]
            print(f"[ask]    {name} failed: {last_err} -- fallback")
            continue
        try:
            msg = res["choices"][0]["message"]["content"]
            print(msg)
            return
        except (KeyError, IndexError, TypeError):
            last_err = "unexpected response"
            print(f"[ask]    {name} returned unexpected shape -- fallback")
            continue
    print(f"[ask] all providers failed. last error: {last_err}")



def main():
    ap = argparse.ArgumentParser(
        description="LMM - Local/remote Model Manager (cross-platform, zero-dep)")
    ap.add_argument("-v", "--version", action="store_true", help="show version")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("discover").add_argument("--json", action="store_true")
    sub.add_parser("status")
    sub.add_parser("models")
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
    sub.add_parser("examples")
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
        cmd_discover(cfg, getattr(args, "json", False))
    elif cmd == "cli":
        cmd_discover(cfg, False)
    elif cmd == "status":
        cmd_status(cfg)
    elif cmd == "models":
        cmd_models()
    elif cmd == "cost":
        print(cost_report(cfg, args.days))
    elif cmd == "route":
        print(f"task: {args.task}\n=> recommend: {route_task(cfg, args.task)}")
    elif cmd == "serve":
        if getattr(args, "hub", False):
            cmd_serve_hub(cfg, args.host, args.port)
        else:
            cmd_serve(args.model)
    elif cmd == "hub-status":
        cmd_hub_status(cfg)
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
        cmd_ask(" ".join(getattr(args, "prompt", [])), getattr(args, "provider", None), cfg)
    elif cmd == "examples":
        cmd_examples()


if __name__ == "__main__":
    main()
