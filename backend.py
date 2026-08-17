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
import time

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
def run(cmd, timeout=25):
    """Run a shell command, tolerating Windows CP932 output. Returns a
    subprocess.CompletedProcess-like object with .stdout/.stderr as text, or
    None on failure (exception/timeout)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
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
def http_post_stream(url, payload, api_key, timeout=120):
    """Streaming OpenAI-compatible chat completion (SSE, zero-dep stdlib).
    Yields content chunks as they arrive. Stops on [DONE]."""
    import urllib.request
    import http.client
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "Accept": "text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = ""
            for raw in resp:
                line = raw.decode("utf-8", "ignore").rstrip(chr(10))
                if not line.startswith("data:"):
                    continue
                payload_s = line[len("data:"):].strip()
                if payload_s == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_s)
                except Exception:
                    continue
                choices = obj.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
    except Exception as e:
        yield f"[stream error: {e}]"
def http_get_json(url, api_key=None, timeout=30):
    """Minimal GET returning parsed JSON, or {'error': ...} on failure."""
    import urllib.request
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception as e:
        return {"error": str(e)}
def call_provider(prov, prompt, temperature=0.7, messages=None, stream=False):
    """Send a completion request to one provider (OpenAI-compatible
    /chat/completions). If `messages` (full history) is given, it is sent
    verbatim; otherwise a single-turn [user: prompt] is used. When `stream`
    is True, returns a generator yielding content chunks (SSE)."""
    if not prov.get("model"):
        return {"error": "provider has no model set"}
    url = prov["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": prov["model"],
        "messages": messages if messages is not None
                     else [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        return http_post_stream(url, payload, prov.get("api_key", ""))
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
def _model_size_b(name):
    """Extract approximate model size in billions from a model id, else None."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", (name or "").lower())
    return float(m.group(1)) if m else None
def backend_catalog(cfg):
    """Build a catalog of every routable backend WITH measured capability
    signals (size in B params, local/remote, paid). This is the first-principles
    view: we route on *measured ability*, not on a static priority list."""
    targets = resolve_ask_targets(cfg, "", None)
    cat = []
    for name, prov in targets:
        size = _model_size_b(prov.get("model"))
        cat.append({
            "name": name,
            "model": prov.get("model"),
            "size_b": size,
            "kind": prov.get("kind", "remote"),
            "paid": (prov.get("kind") == "remote"),
            "base_url": prov.get("base_url"),
        })
    return cat
def score_and_route(task, cfg, ask_order=None):
    """First-principles routing: MEASURE the task, match to MEASURED backend
    ability. ask_order is a *tie-breaker weight*, not a blind master.

    Returns (backend_name, reason) or (None, reason)."""
    t = (task or "").lower()
    cat = backend_catalog(cfg)
    cat = [c for c in cat if c["base_url"]]  # only reachable backends
    if not cat:
        return None, "no reachable backend"
    # task signals
    words = len(t.split())
    heavy = any(k in t for k in
                ["explain", "derive", "proof", "quantum", "analyze", "design",
                 "architecture", "research", "compare", "why", "理由", "説明",
                 "設計", "解析", "比較", "導出", "証明", "考察"])
    code = any(k in t for k in
               ["code", "function", "bug", "refactor", "クラス", "関数",
                "コード", "実装", "デバッグ", "write a", "def ", "```"])
    is_private = any(k in t for k in
                    merged_route(cfg).get("private", []) + ["secret", "社内"])
    # score each backend
    scored = []
    for c in cat:
        s = 0.0
        size = c["size_b"]
        if size is None:
            size = 8.0 if c["kind"] == "remote" else 3.0  # cloud=big, unknown local=mid
        # ability vs task difficulty
        if heavy:
            s += min(size, 70) / 7.0          # bigger model -> much better at hard tasks
        else:
            s += 1.0                          # light task: any model fine
        if code and "coder" in (c["model"] or "").lower():
            s += 2.0                          # code-tuned model bonus
        if is_private and c["kind"] == "local":
            s += 3.0                          # privacy prefers local
        if c["kind"] == "local" and not c["paid"]:
            s += 0.5                          # free is nice
        # ask_order as a tie-breaker weight (priority respected, not obeyed)
        if ask_order:
            try:
                idx = ask_order.index(c["name"])
                s += max(0, (len(ask_order) - idx)) * 0.1
            except ValueError:
                pass
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    return best["name"], f"score={scored[0][0]:.1f} size={best['size_b']}b " \
                         f"task={'heavy' if heavy else 'light'}" \
                         f"{'/code' if code else ''}{'/private' if is_private else ''}"
def verify_reply(task, reply):
    """First-principles QUALITY GATE: don't trust that a backend answered well
    — MEASURE it. Returns (ok, reason). This closes the routing loop: a backend
    is only 'good' if its reply actually satisfies the task, measured by
    proxy signals (we cannot grade meaning, but we can detect failure modes).

    Proxy failure modes (all MEASURED, not assumed):
      - empty / error / refusal
      - hallucinated tokens (mojibake / non-word ASCII-Japanese mixes)
      - task-specific under-delivery (code task with no code block, etc.)
    """
    if not reply or not reply.strip():
        return False, "empty reply"
    t = (task or "").lower()
    r = reply.strip()
    # hallucination / mojibake proxy: latin letter immediately fused to a
    # Japanese kana/kanji with no separator, e.g. 'propagレーション'
    import re
    if re.search(r"[A-Za-z][\u3040-\u30ff\u4e00-\u9fff]|[\u3040-\u30ff\u4e00-\u9fff][A-Za-z]", r):
        return False, "hallucinated token (latin+script fused)"
    # code task must contain a code block or def/class
    if any(k in t for k in ["code", "function", "実装", "クラス", "関数",
                             "def ", "write a", "```"]):
        if "```" not in r and "def " not in r and "class " not in r and "function" not in r:
            return False, "code task but no code block produced"
    # heavy reasoning task: a one-liner is under-delivery — but ONLY for
    # English tasks. Japanese answers can be correct and concise (e.g.
    # "シュレーディンガー方程式です"), so we must not punish short JP replies.
    has_jp = any('\u3040' <= c <= '\u9fff' for c in r)
    if (not has_jp) and any(k in t for k in
                            ["explain", "derive", "proof", "理由", "why"]):
        if len(r) < 120:
            return False, f"reasoning task but reply too short ({len(r)} chars)"
    return True, "ok"
def route_and_verify(task, cfg, ask_order=None, max_tries=3):
    """Closed-loop routing (measure -> run -> verify -> fallback if bad).
    Picks backends by score_and_route ordering, tries each, and if verify_reply
    says the answer is unfit, FALLS BACK to the next — recording the measured
    reason. This is 'measure, don't trust' made operational."""
    t = (task or "").lower()
    cat = backend_catalog(cfg)
    cat = [c for c in cat if c["base_url"]]
    if not cat:
        return None, "no reachable backend", None
    # build an ordered candidate list by reusing score_and_route's score fn
    # (score each catalog entry the same way, sort descending)
    heavy = any(k in t for k in
                ["explain", "derive", "proof", "quantum", "analyze", "design",
                 "architecture", "research", "compare", "why", "理由", "説明",
                 "設計", "解析", "比較", "導出", "証明", "考察"])
    code = any(k in t for k in
               ["code", "function", "bug", "refactor", "クラス", "関数",
                "コード", "実装", "デバッグ", "write a", "def ", "```"])
    is_private = any(k in t for k in
                    merged_route(cfg).get("private", []) + ["secret", "社内"])
    rank = []
    for c in cat:
        s = 0.0
        size = c["size_b"]
        if size is None:
            size = 8.0 if c["kind"] == "remote" else 3.0
        s += min(size, 70) / 7.0 if heavy else 1.0
        if code and "coder" in (c["model"] or "").lower():
            s += 2.0
        if is_private and c["kind"] == "local":
            s += 3.0
        if c["kind"] == "local" and not c["paid"]:
            s += 0.5
        if ask_order:
            try:
                idx = ask_order.index(c["name"])
                s += max(0, (len(ask_order) - idx)) * 0.1
            except ValueError:
                pass
        rank.append((s, c))
    rank.sort(key=lambda x: -x[0])
    order = [c["name"] for _, c in rank]
    targets = resolve_ask_targets(cfg, task, None)
    ordered = [t_ for t_ in targets if t_[0] in order]
    ordered += [t_ for t_ in targets if t_[0] not in order]
    tried = []
    for name, prov in ordered[:max_tries]:
        r = call_provider(prov, task, stream=False)
        if isinstance(r, dict) and r.get("error"):
            tried.append(f"{name} error: {r['error']}")
            continue
        try:
            reply = r["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            tried.append(f"{name} bad response")
            continue
        ok, vreason = verify_reply(task, reply)
        if ok:
            return name, f"verified ok ({vreason})", reply
        tried.append(f"{name} failed verify: {vreason}")
        log_hub({"event": "verify", "provider": name, "ok": False,
                 "reason": vreason, "prompt": task})
    return None, f"all tried failed: {'; '.join(tried)}", None
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
def measure_performance():
    """Scan hub.log for ask_attempt events and return {provider: {ok, fail,
    avg_ms}}. Reuses the same JSONL source cmd_stats reads — measure, don't
    trust: this is the REAL observed routing outcome, not a guess."""
    import os as _os
    import json as _json
    from collections import defaultdict
    p = _os.path.join(HOME, ".lmm", "hub.log")
    stat = defaultdict(lambda: {"ok": 0, "fail": 0, "lat": []})
    if not _os.path.exists(p):
        return stat
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = _json.loads(line)
            except Exception:
                continue
            if e.get("event") != "ask_attempt":
                continue
            n = e.get("provider", "?")
            if e.get("ok"):
                stat[n]["ok"] += 1
            else:
                stat[n]["fail"] += 1
            if isinstance(e.get("latency_ms"), int):
                stat[n]["lat"].append(e["latency_ms"])
    for v in stat.values():
        v["avg_ms"] = int(sum(v["lat"]) / len(v["lat"])) if v["lat"] else 0
    return stat
def optimize_ask_order(cfg):
    """First-principles closed loop: don't trust the static ask_order — MEASURE
    it. Re-rank ask_order by observed success rate and latency, so a backend
    that keeps failing or timing out drops in priority. Unmeasured backends are
    kept but moved to the tail (we have no evidence they're good). The user's
    explicit entries are preserved in identity, only reordered by evidence.

    NOTE: ask_order holds DISPLAY names (e.g. "Ollama") but the hub log records
    PROVIDER keys (e.g. "local-ollama(implicit)"). We normalize via NAME_TO_KEY
    before matching, else every backend looks "unmeasured" (measure, don't trust).

    Cost-aware (FrugalGPT / RouteLLM lesson): a backend that is free AND works is
    infinitely cost-efficient; a paid one must EARN its place via higher success
    or quality. We fold $/1M tokens into the score so local models win when they
    are sufficient, paid models only win when they materially outperform.
    """
    stat = measure_performance()
    pricing = merged_pricing(cfg)
    order = list(cfg.get("ask_order") or [])
    if not order:
        return order, stat
    # display-name -> provider-key normalizer (mirrors resolve_ask_targets)
    NAME_TO_KEY = {
        "ollama": "local-ollama(implicit)",
        "lm studio": "local-lmstudio(implicit)",
        "claude code (anthropic)": "claude-code",
        "claude": "claude-code",
    }
    def norm(n):
        if n in stat:
            return n
        return NAME_TO_KEY.get(n.lower(), n)
    def cost_per_1m(key):
        # local backends are free; cloud keys resolve via merged_pricing
        if "local-" in key or key.endswith("(implicit)"):
            return 0.0
        fam = key.split(":")[0].lower()
        p = pricing.get(fam, pricing.get("default", {"out": 15.0}))
        return float(p.get("out", 15.0))
    scored = []
    for name in order:
        key = norm(name)
        s = stat.get(key, {})
        ok, fail = s.get("ok", 0), s.get("fail", 0)
        total = ok + fail
        if total == 0:
            score = -1.0          # unmeasured: tail, but keep
        else:
            succ = ok / total
            avg = s.get("avg_ms", 0) or 1
            cost = cost_per_1m(key)
            # cost-efficiency: success per dollar (free => 1/cost -> large bonus)
            eff = succ / (cost / 1000.0 + 0.01)
            # success dominates; latency + cost-efficiency are tie-breakers
            score = succ * 10.0 + min(100000 / avg, 5.0) + min(eff / 50.0, 5.0)
        scored.append((score, name))
    # stable sort: highest score first; unmeasured (-1) sink to the tail
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored], stat
def fetch_models(prov):
    """Return a list of model-id strings for a provider, or [] on failure.
    Ollama uses /api/tags; OpenAI-compatible clouds use /v1/models."""
    try:
        base = prov.get("base_url", "")
        if prov.get("kind") == "local" and "11434" in base:
            # Ollama's native tags endpoint (strip any /v1 suffix)
            root = base.replace("/v1", "").rstrip("/")
            r = http_get_json(root + "/api/tags")
            if r and "models" in r:
                return [m.get("name", "?") for m in r["models"]]
        # OpenAI-compatible: /v1/models
        r = http_get_json(base.rstrip("/") + "/v1/models",
                          api_key=prov.get("api_key"))
        if r and "data" in r:
            return [m.get("id", m.get("name", "?")) for m in r["data"]]
    except Exception:
        return []
    return []
def resolve_provider_by_model(provs, model_id):
    """Map a real model id (from /v1/models) back to its provider name.
    Lets OpenAI clients pick a model and have it routed correctly."""
    if not model_id:
        return None
    for n, p in provs.items():
        if model_id in fetch_models(p):
            return n
    return None
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
def local_lmstudio_provider():
    """Synthesize an implicit local provider from a running LM Studio server,
    so `lmm ask` / `lmm serve --hub` can fan out to it too (zero-config hub).
    Returns dict or None."""
    d = detect_lmstudio()
    if not d.get("running"):
        return None
    # base_url WITHOUT /v1: fetch_models() appends /v1/models itself
    # (matching the Ollama convention, which also stores the root URL).
    endpoint = (d.get("endpoint") or "http://localhost:1234/v1").replace("/v1", "").rstrip("/")
    models = d.get("models") or []
    model = models[0] if models else "local-model"
    return {"api_key": "lmstudio", "base_url": endpoint,
            "model": model, "kind": "local", "_implicit": True}
def resolve_ask_targets(cfg, prompt, explicit):
    """Ordered providers to try. Priority is USER-CONTROLLED via cfg['ask_order']
    (a list of provider names). Falls back to implicit running Ollama.

    Order:  explicit  >  cfg['ask_order']  >  (keyword-matched)  >  implicit
    running Ollama as final safety net. With no ask_order set, the old default
    (keyword match, then all configured, then implicit Ollama) applies.

    NOTE: ask_order entries may be either provider keys (from config 'providers')
    or human-readable discover names (e.g. "Ollama"). We normalize both forms.
    """
    # Map discover display names -> provider keys / implicit keys.
    NAME_TO_KEY = {
        "ollama": "local-ollama(implicit)",
        "lm studio": "local-lmstudio(implicit)",
        "claude code (anthropic)": "claude-code",  # if configured as a provider
        "claude": "claude-code",
    }
    provs = merged_providers(cfg)
    # fold in the implicit running-Ollama safety net so explicit references
    # to it (e.g. a model id resolved back to "local-ollama(implicit)")
    # are recognized as a valid target.
    lo = local_ollama_provider()
    if lo and "local-ollama(implicit)" not in provs:
        provs = dict(provs)
        provs["local-ollama(implicit)"] = lo
    # fold in implicit running LM Studio too (zero-config hub: LM Studio is a
    # local base just like Ollama, and should be routable without config).
    ls = local_lmstudio_provider()
    if ls and "local-lmstudio(implicit)" not in provs:
        provs = dict(provs)
        provs["local-lmstudio(implicit)"] = ls
    # also allow a configured provider named 'claude-code' (Anthropic) to be
    # targeted by its discover name.
    provs_norm = {}
    for k, v in provs.items():
        provs_norm[k.lower()] = (k, v)
        provs_norm[NAME_TO_KEY.get(k.lower(), k.lower())] = (k, v)
    # additionally register discover names as keys pointing at their provider
    for discover_name, key in NAME_TO_KEY.items():
        if key in provs and discover_name not in provs_norm:
            provs_norm[discover_name] = (key, provs[key])
    if explicit:
        if explicit in provs:
            return [(explicit, provs[explicit])]
        en = explicit.lower()
        if en in provs_norm:
            k, v = provs_norm[en]
            return [(k, v)]
        if en in ("local", "ollama", "local-ollama") and lo:
            return [("local-ollama(implicit)", lo)]
        return []
    order = list(cfg.get("ask_order") or [])
    out, seen = [], set()
    for n in order:                      # user-defined priority, in full
        nk = n.lower()
        # accept either the provider key or the discover display name
        if n in provs:
            k, v = n, provs[n]
        elif nk in provs_norm:
            k, v = provs_norm[nk]
        else:
            continue
        if k not in seen:
            out.append((k, v)); seen.add(k)
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
    ls = local_lmstudio_provider()
    if ls and "local-lmstudio(implicit)" not in seen:
        out.append(("local-lmstudio(implicit)", ls))
    return out
def save_config(cfg):
    """Write config to ~/.lmm/config.json (primary candidate)."""
    d = os.path.join(HOME, ".lmm")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return p
def log_hub(entry):
    """Append a structured hub event to ~/.lmm/hub.log (JSONL). This is how the
    hub proves what it actually did — measurement, not assumption. The hub is
    only as trustworthy as this log."""
    import datetime
    try:
        d = os.path.join(HOME, ".lmm")
        os.makedirs(d, exist_ok=True)
        entry = dict(entry)
        entry.setdefault("ts", datetime.datetime.now().isoformat(timespec="seconds"))
        with open(os.path.join(d, "hub.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + chr(10))
    except Exception:
        pass
