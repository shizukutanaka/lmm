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
#   cost [--days N]     measured spend: Anthropic session logs + lmm's own hub
#   route "task ..."    recommend local vs remote for a task (--explain)
#   fit [model]         does it fit in your GPU, and at what context length?
#   ask "prompt"        one question, routed across every backend (--cascade)
#   serve <model>       pull + expose a local model endpoint (Ollama)
#   serve --hub         OpenAI-compatible proxy over all configured providers
#   cache               prompt-cache stats (--clear)
#   stop  <runtime>     stop a running runtime
#   hide  <runtime>     strip a runtime's taskbar button (Windows)
#   dash                generate + open a self-contained HTML dashboard
#   examples            show a sample config file you can copy
#
# Config (~/.lmm/config.json) lets anyone add their own runtimes, override
# pricing, or change routing keywords. See `lmm examples`.
#
# The hub aims to be measurably cheaper than always calling the strongest
# model. Three published techniques do the work, all stdlib-only:
#   RouteLLM   arXiv:2406.18665 - score the prompt, route on a cost threshold
#   FrugalGPT  arXiv:2305.05176 - cheap models first, escalate on a low score
#   GPT Sem.   arXiv:2411.05276 - reuse answers instead of re-paying for them
#                (with vCache arXiv:2502.03771 on why fuzzy hits must be opt-in)
# Every call is metered to ~/.lmm/usage.jsonl so the savings are measured
# rather than asserted -- see `lmm cost`.
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import math
import hashlib
import argparse
import subprocess
import datetime
import html
import webbrowser
import ctypes

HOME = os.path.expanduser("~")

# lmm's own state lives here: measured usage telemetry and the prompt cache.
# Both are append-only JSONL so they stay inspectable with `cat`/`tail` and
# never need a database.
LMM_DIR   = os.path.join(HOME, ".lmm")
USAGE_LOG = os.path.join(LMM_DIR, "usage.jsonl")
CACHE_LOG = os.path.join(LMM_DIR, "cache.jsonl")

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

# --- Cost-reduction defaults (see the three papers referenced below) --------
# The hub is only worth having if it is measurably cheaper than calling the
# strongest model every time. Three published techniques, all implemented here
# with the stdlib alone:
#
#   routing  : RouteLLM (Ong et al., ICLR 2025, arXiv:2406.18665) — score the
#              prompt, compare against a cost threshold, send weak-first when
#              the strong model probably would not win.
#   cascade  : FrugalGPT (Chen/Zaharia/Zou, arXiv:2305.05176) — run cheap
#              models first and escalate only when a scorer rejects the answer.
#   cache    : GPT Semantic Cache (arXiv:2411.05276) — reuse answers instead of
#              re-paying for them. vCache (arXiv:2502.03771) shows a single
#              static similarity threshold cannot bound false hits, so the
#              semantic tier is opt-in and its default threshold is strict.
DEFAULT_ROUTE_THRESHOLD = 0.5      # RouteLLM's alpha: s >= alpha -> strong-first

DEFAULT_CASCADE = {
    "enabled":   False,   # opt-in; --cascade turns it on for one call
    "rungs":     [],      # empty -> auto-built cheapest-first from prices
    "threshold": 0.6,     # accept an answer scoring >= this, else escalate
    "max_rungs": 3,       # never spend more than this many calls on one prompt
    "judge":     None,    # provider name for an LLM-as-judge second opinion
}

DEFAULT_CACHE = {
    "enabled":     True,   # tier 1 (exact hash) is safe enough to default on
    "semantic":    False,  # tier 2 needs local embeddings AND accepts fuzzy hits
    "similarity":  0.95,   # strict on purpose — see vCache, arXiv:2502.03771
    "ttl_hours":   168,    # a week; answers go stale, especially about code
    "max_entries": 2000,
    "embed_model": "nomic-embed-text",
    "max_temp":    0.3,    # above this the caller wants variety, not a cache
}

# Cheap lexical features standing in for RouteLLM's learned win-predictor.
# Each entry is (label, weight); the matching logic lives in prompt_strength().
STRENGTH_CODE_MARKERS = ("```", "def ", "class ", "function ", "import ",
                         "select ", "#include", "=>", "();", "async ")
STRENGTH_REASON_MARKERS = ("step by step", "step-by-step", "explain why", "why ",
                           "compare", "trade-off", "tradeoff", "prove", "derive",
                           "なぜ", "理由", "比較", "手順", "設計")

# FrugalGPT's scorer is learned from data; with no training set available we
# approximate it by penalising the failure modes small models actually exhibit.
VERIFY_REFUSAL_MARKERS = ("i can't", "i cannot", "i'm unable", "i am unable",
                          "as an ai", "i don't have access", "sorry, i",
                          "申し訳", "できません", "わかりません", "不明です")
VERIFY_HEDGE_MARKERS = ("i think", "maybe", "probably", "not sure", "might be",
                        "i believe", "たぶん", "おそらく", "かもしれ")

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
    """Detected accelerator memory in MiB: {name, used, total, pct, kind}.

    NVIDIA, AMD and Apple Silicon are all asked in turn — "is there GPU
    headroom" is the question behind every local-vs-cloud decision, and
    answering it only for CUDA users made that decision wrong everywhere else.
    """
    r = run("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
    if r and r.stdout.strip():
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        if len(parts) >= 3:
            try:
                u = int(parts[1].replace("MiB", "").strip())
                t = int(parts[2].replace("MiB", "").strip())
                return {"name": parts[0], "used": u, "total": t,
                        "pct": round(100 * u / t) if t else 0, "kind": "nvidia"}
            except ValueError:
                pass
    # AMD ROCm. --showmeminfo prints "Total Memory (B): N" / "Total Used ...".
    r = run("rocm-smi --showmeminfo vram --csv")
    if r and r.stdout.strip():
        try:
            total = used = 0
            for ln in r.stdout.splitlines():
                low = ln.lower()
                nums = [int(x) for x in ln.replace(",", " ").split() if x.isdigit()]
                if not nums:
                    continue
                if "total" in low and "used" not in low:
                    total = max(total, max(nums))
                elif "used" in low:
                    used = max(used, max(nums))
            if total > 0:
                t, u = total // (1024 ** 2), used // (1024 ** 2)
                return {"name": "AMD GPU (ROCm)", "used": u, "total": t,
                        "pct": round(100 * u / t) if t else 0, "kind": "amd"}
        except (ValueError, ZeroDivisionError):
            pass
    # Apple Silicon: no discrete VRAM — the GPU shares system RAM. Metal caps a
    # process well below installed RAM, so report the usable share, not all of
    # it. (~75% is the conservative default for machines up to 36GB.)
    if sys.platform == "darwin":
        r = run("sysctl -n hw.memsize")
        if r and r.stdout.strip().isdigit():
            total_mib = int(r.stdout.strip()) // (1024 ** 2)
            usable = int(total_mib * 0.75)
            return {"name": "Apple Silicon (unified memory)", "used": 0,
                    "total": usable, "pct": 0, "kind": "apple"}
    return None


# ------------------- VRAM fit: will this model actually run? ----------------
# The local half of every local-vs-cloud decision. Two published numbers do the
# work:
#
#   weights  = params * bits_per_weight / 8
#   kv cache = 2 * n_layers * n_kv_heads * head_dim * ctx * bytes_per_element
#
# The 2 is K and V. n_kv_heads must be the GQA count, not the query-head count
# — Llama 3.1 8B has 32 query heads but 8 KV heads, so assuming the former
# overstates the KV cache by 4x. Verified against the published figure for
# Llama-3-8B at 32K context in fp16: 2*32*8*128*32768*2 = 4.0 GiB.
#
# bits-per-weight is above the nominal bit count because k-quants store scales
# and keep sensitive tensors wider. Values are the llama.cpp measurements on
# LLaMA-family models and are approximate for other architectures.
QUANT_BPW = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0,
    "q8_0": 8.50, "q6_k": 6.59,
    "q5_k_m": 5.69, "q5_k_s": 5.52, "q5_0": 5.54, "q5_1": 6.00,
    "q4_k_m": 4.85, "q4_k_s": 4.58, "q4_0": 4.55, "q4_1": 5.00,
    "q3_k_l": 4.27, "q3_k_m": 3.89, "q3_k_s": 3.50,
    "q2_k": 3.35,
    "iq4_xs": 4.25, "iq3_xs": 3.30, "iq2_xs": 2.40,
}

# KV cache element size. llama.cpp exposes these as --cache-type-k/--cache-type-v;
# q8_0 halves the cache for very little quality loss, q4_0 quarters it.
KV_BYTES = {"f16": 2.0, "q8_0": 1.0, "q8": 1.0, "q4_0": 0.5, "q4": 0.5}

# CUDA/Metal context, activations and scratch buffers. Real usage runs roughly
# 200-800MB depending on batch size and backend; 0.5GiB is a middle estimate.
VRAM_OVERHEAD_GIB = 0.5

GIB = 1024 ** 3


def parse_params(s):
    """'8.0B' / '134.52M' / '7b' -> parameter count as a float."""
    s = (s or "").strip().lower().replace(",", "")
    mult = 1.0
    if s.endswith("b"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def params_from_name(name):
    """Last-resort parameter count from a tag like 'qwen2.5-coder:7b'."""
    token = ""
    for ch in (name or "").lower():
        if ch.isdigit() or ch == ".":
            token += ch
        elif ch in ("b", "m") and token:
            n = parse_params(token + ch)
            if 1e8 <= n <= 2e12:          # ignore version numbers like "2.5"
                return n
            token = ""
        else:
            token = ""
    return 0.0


def ollama_model_info(model):
    """Real architecture metadata from a running Ollama (`POST /api/show`).

    Returns {params, layers, kv_heads, head_dim, ctx_max, quant, source} or
    None. Without this the KV cache cannot be computed honestly, because
    layer/head counts are not derivable from a parameter count.
    """
    body = http_post_json("http://localhost:11434/api/show",
                          {"model": model}, "", timeout=20)
    if not isinstance(body, dict) or body.get("error") or "model_info" not in body:
        return None
    mi = body.get("model_info") or {}
    det = body.get("details") or {}
    arch = mi.get("general.architecture") or det.get("family") or ""

    def g(*suffixes):
        for suf in suffixes:
            for key in (f"{arch}.{suf}", suf):
                if isinstance(mi.get(key), (int, float)):
                    return mi[key]
        return None

    layers = g("block_count")
    heads = g("attention.head_count")
    kv_heads = g("attention.head_count_kv") or heads
    embed = g("embedding_length")
    head_dim = g("attention.key_length")
    if not head_dim and embed and heads:
        head_dim = embed / heads
    params = parse_params(det.get("parameter_size", "")) or params_from_name(model)
    if not (layers and kv_heads and head_dim):
        return None
    return {"params": params, "layers": int(layers), "kv_heads": int(kv_heads),
            "head_dim": int(head_dim), "ctx_max": int(g("context_length") or 0),
            "quant": (det.get("quantization_level") or "").lower(),
            "arch": arch, "source": "ollama"}


def quant_bpw(quant):
    q = (quant or "").strip().lower()
    if q in QUANT_BPW:
        return QUANT_BPW[q]
    for k in sorted(QUANT_BPW, key=len, reverse=True):
        if k in q:
            return QUANT_BPW[k]
    return QUANT_BPW["q4_k_m"]          # Ollama's default pull


def kv_bytes_per_token(spec, kv_type="f16"):
    """2 (K and V) * layers * kv_heads * head_dim * bytes — per token."""
    return (2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"]
            * KV_BYTES.get((kv_type or "f16").lower(), 2.0))


def estimate_vram(spec, ctx, quant=None, kv_type="f16"):
    """Total GiB to run `spec` at `ctx` tokens: weights + KV cache + overhead."""
    bpw = quant_bpw(quant or spec.get("quant"))
    weights = spec.get("params", 0.0) * bpw / 8.0 / GIB
    kv = kv_bytes_per_token(spec, kv_type) * max(ctx, 0) / GIB
    return {"weights_gib": weights, "kv_gib": kv,
            "overhead_gib": VRAM_OVERHEAD_GIB,
            "total_gib": weights + kv + VRAM_OVERHEAD_GIB, "bpw": bpw}


def max_context_for(spec, budget_gib, quant=None, kv_type="f16"):
    """Largest context that still fits in `budget_gib`. 0 means the weights
    alone do not fit."""
    est = estimate_vram(spec, 0, quant, kv_type)
    room = budget_gib - est["weights_gib"] - VRAM_OVERHEAD_GIB
    if room <= 0:
        return 0
    per_tok = kv_bytes_per_token(spec, kv_type) / GIB
    return int(room / per_tok) if per_tok > 0 else 0


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


def merged_cascade(cfg):
    c = dict(DEFAULT_CASCADE)
    c.update(cfg.get("cascade") or {})
    return c


def merged_cache(cfg):
    c = dict(DEFAULT_CACHE)
    c.update(cfg.get("cache") or {})
    return c


def merged_providers(cfg):
    """Return the cloud/remote provider registry from config.

    Each provider entry: {api_key, base_url, model, kind, price}. Used by
    `lmm ask` and `lmm serve --hub` so a single `lmm` command can target any
    backend through one OpenAI-compatible interface.

    `price` is optional and binds the provider to a rate: either a key in the
    pricing table ("deepseek-chat") or an inline {"in": .., "out": ..}. Without
    it we fall back to matching the model name, so existing configs keep
    working — it only makes the cost numbers exact.
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
            "price": v.get("price"),
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


# Request fields we hand straight to the backend. Anything else in an incoming
# hub request is lmm's business, not the provider's.
PASSTHROUGH_KEYS = ("temperature", "max_tokens", "max_completion_tokens", "top_p",
                    "stop", "presence_penalty", "frequency_penalty", "seed",
                    "response_format", "tools", "tool_choice", "n", "user")


def as_messages(prompt):
    """Accept either a bare prompt string or an OpenAI messages array and
    always return a messages array. Callers upstream of the hub deal in whole
    conversations; `lmm ask` deals in one string."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt or [])


def messages_text(messages):
    """Flatten a messages array to plain text for scoring/keying/hashing.
    Content may be a string or the OpenAI content-parts list."""
    parts = []
    for m in as_messages(messages):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):        # content parts: [{type:text,text:..}]
            for seg in c:
                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                    parts.append(seg["text"])
    return "\n".join(parts)


def call_provider(prov, prompt, temperature=0.7, extra=None):
    """Send a prompt (str) or a full messages array to one provider over the
    OpenAI-compatible /chat/completions API.

    Passing the whole array matters: a system prompt and prior turns carry the
    caller's intent, and dropping them silently changes the answer.
    """
    if not prov.get("model"):
        return {"error": "provider has no model set"}
    url = prov["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": prov["model"],
        "messages": as_messages(prompt),
        "temperature": temperature,
    }
    for k in PASSTHROUGH_KEYS:                  # caller-supplied params win
        if extra and k in extra and extra[k] is not None:
            payload[k] = extra[k]
    payload.pop("stream", None)                 # we never stream upstream
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
# Metering: every real call lmm makes records what it actually cost. Without
# this, the routing/cascade/cache savings below would be a claim rather than a
# measurement — and `cost --days N` had nothing dated to filter on.
def price_for(prov, model, pricing):
    """Resolve one provider+model to a {in,out,cw,cr} USD/1M-token rate.

    Order: explicit provider 'price' > model-name match in the rate table >
    provider-name match > default. Local runtimes are free by definition.
    """
    if (prov or {}).get("kind") == "local":
        return {"in": 0.0, "out": 0.0, "cw": 0.0, "cr": 0.0}
    p = (prov or {}).get("price")
    if isinstance(p, dict):
        return {"in": float(p.get("in", 0.0)), "out": float(p.get("out", 0.0)),
                "cw": float(p.get("cw", 0.0)), "cr": float(p.get("cr", 0.0))}
    if isinstance(p, str) and p in pricing:
        return pricing[p]
    hay = (model or "").lower()
    if hay:
        # longest key first so "openai-gpt4o-mini" beats "openai-gpt4o"
        for k in sorted(pricing, key=len, reverse=True):
            if k in ("default",):
                continue
            core = k.split("-")[-1]
            if k in hay or (len(core) > 3 and core in hay):
                return pricing[k]
    return pricing["default"]


def usage_cost(usage, rate):
    """USD for one OpenAI-compatible `usage` block at the given rate."""
    u = usage or {}
    cached = 0
    det = u.get("prompt_tokens_details")
    if isinstance(det, dict):
        cached = det.get("cached_tokens", 0) or 0
    fresh_in = max((u.get("prompt_tokens", 0) or 0) - cached, 0)
    return (fresh_in / 1e6 * rate.get("in", 0.0)
            + (u.get("completion_tokens", 0) or 0) / 1e6 * rate.get("out", 0.0)
            + cached / 1e6 * rate.get("cr", 0.0))


def log_usage(event):
    """Append one metering event to ~/.lmm/usage.jsonl. Never raises: telemetry
    must not be able to break the call it is measuring."""
    try:
        os.makedirs(LMM_DIR, exist_ok=True)
        event = dict(event)
        event.setdefault("ts", datetime.datetime.now().isoformat(timespec="seconds"))
        with open(USAGE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_usage(days=None):
    """Read metering events, optionally only the last `days`."""
    if not os.path.isfile(USAGE_LOG):
        return []
    cutoff = None
    if days:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    out = []
    try:
        with open(USAGE_LOG, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if cutoff:
                    try:
                        if datetime.datetime.fromisoformat(ev.get("ts", "")) < cutoff:
                            continue
                    except Exception:
                        pass
                out.append(ev)
    except Exception:
        return out
    return out


def meter_call(prov_name, prov, model, res, pricing, **extra):
    """Price one provider response and record it. Returns the USD charged."""
    usage = (res or {}).get("usage") if isinstance(res, dict) else None
    rate = price_for(prov, model, pricing)
    usd = usage_cost(usage, rate)
    ev = {"provider": prov_name, "model": model,
          "kind": (prov or {}).get("kind", "remote"),
          "in": (usage or {}).get("prompt_tokens", 0) or 0,
          "out": (usage or {}).get("completion_tokens", 0) or 0,
          "usd": round(usd, 6)}
    ev.update(extra)
    log_usage(ev)
    return usd


def model_family(s):
    s = (s or "").lower()
    for fam in ("opus", "sonnet", "haiku"):
        if fam in s:
            return fam
    return "default"


def measured_tokens(days=None):
    """Aggregate real Anthropic token usage from ~/.claude/projects/*.jsonl.

    `days` filters by session-file mtime — the session logs carry no single
    reliable timestamp field, but the file's last write is when the session
    last ran, which is the granularity `cost --days N` actually needs.
    """
    if not os.path.isdir(CLAUDE_PROJECTS):
        return None
    cutoff = time.time() - days * 86400 if days else None
    agg = {}
    total_sessions = 0
    for root, _, files in os.walk(CLAUDE_PROJECTS):
        for f in files:
            if not f.endswith(".jsonl"):
                continue
            path = os.path.join(root, f)
            if cutoff is not None:
                try:
                    if os.path.getmtime(path) < cutoff:
                        continue
                except OSError:
                    continue
            total_sessions += 1
            fam = "default"
            seen = False
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
                            a["sessions"].add(path)  # full path: same filename
                                                     # recurs across projects
            except Exception:
                pass
    for a in agg.values():
        a["sessions"] = len(a["sessions"])
    return {"by_family": agg, "total_sessions": total_sessions}


def cost_report(cfg, days=30):
    pricing = merged_pricing(cfg)
    data = measured_tokens(days)
    window = f"last {days}d" if days else "all-time"
    grand = 0.0
    if not data or not data.get("by_family"):
        out = [f"No Claude session logs found at {CLAUDE_PROJECTS} ({window})"]
        return "\n".join(out + hub_cost_block(cfg, pricing, days))
    out = [f"Anthropic measured usage ({window}, {data['total_sessions']} sessions)",
           "-" * 64]
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
    return "\n".join(out + hub_cost_block(cfg, pricing, days, grand))


def hub_cost_block(cfg, pricing, days=None, claude_total=0.0):
    """Everything lmm measured itself: real hub spend, what the cache and the
    local runtimes saved, plus any hand-entered cfg['usage']."""
    out = []
    events = read_usage(days)
    window = f"last {days}d" if days else "all-time"
    measured = 0.0

    if events:
        by_prov = {}
        hits = {"exact": 0, "semantic": 0}
        saved_cache = 0.0
        local_calls, local_tokens = 0, 0
        for ev in events:
            hit = ev.get("cache")
            if hit in hits:
                hits[hit] += 1
                saved_cache += float(ev.get("saved_usd", 0.0) or 0.0)
                continue
            name = ev.get("provider", "?")
            a = by_prov.setdefault(name, {"calls": 0, "in": 0, "out": 0, "usd": 0.0,
                                          "kind": ev.get("kind", "remote")})
            a["calls"] += 1
            a["in"] += ev.get("in", 0) or 0
            a["out"] += ev.get("out", 0) or 0
            a["usd"] += float(ev.get("usd", 0.0) or 0.0)
            if ev.get("kind") == "local":
                local_calls += 1
                local_tokens += (ev.get("in", 0) or 0) + (ev.get("out", 0) or 0)
        out.append("")
        out.append("=" * 64)
        out.append(f"HUB MEASURED USAGE ({window}, from {USAGE_LOG})")
        out.append("-" * 64)
        for name, a in sorted(by_prov.items(), key=lambda x: -x[1]["usd"]):
            tag = "free" if a["kind"] == "local" else "paid"
            out.append(f"  {name:22} {a['calls']:4d} calls  "
                       f"in={a['in']:,} out={a['out']:,}  "
                       f"${a['usd']:,.4f} ({tag})")
            measured += a["usd"]
        out.append("-" * 64)
        out.append(f"HUB MEASURED TOTAL  ${measured:,.4f}")

        # The savings side of the ledger — the whole point of the cascade,
        # the router and the cache.
        strong = max((r.get("out", 0.0) for k, r in pricing.items()
                      if k != "default"), default=0.0)
        if local_calls:
            would = local_tokens / 1e6 * strong
            out.append(f"  LOCAL (free)         {local_calls:4d} calls, "
                       f"{local_tokens:,} tok — would cost ~${would:,.4f} "
                       f"at the priciest configured rate")
        if hits["exact"] or hits["semantic"]:
            out.append(f"  CACHE                {hits['exact'] + hits['semantic']:4d} hits "
                       f"(exact {hits['exact']} / semantic {hits['semantic']}) — "
                       f"saved ~${saved_cache:,.4f}")
    else:
        out.append("")
        out.append(f"No hub telemetry yet ({USAGE_LOG}). Run `lmm ask ...` or "
                   "`lmm serve --hub` and spend will be measured automatically.")

    # ---- hand-entered cloud usage (still supported alongside telemetry) ----
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
        manual = 0.0
        for name, val in sorted(usage.items()):
            if isinstance(val, dict):
                p = pricing.get(name, pricing["default"])
                c = (val.get("in", 0) / 1e6 * p["in"]
                     + val.get("out", 0) / 1e6 * p["out"])
                out.append(f"  {name:22} {val}  = ${c:,.2f}")
            else:
                c = float(val)
                out.append(f"  {name:22} ${c:,.2f}")
            manual += c
        out.append("-" * 64)
        out.append(f"MEASURED CLOUD TOTAL  ${manual:,.2f}")
        measured += manual
    if measured:
        out.append("")
        out.append(f"ALL-IN TOTAL (Claude + cloud)  ${claude_total + measured:,.2f}")
    return out


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
              + ", ".join(n for n, _ in order_targets(cfg, task, targets)))
    else:
        print("=> no providers configured; add 'providers' or start Ollama")


def best_local_fit(ctx=4096, limit=6):
    """Largest installed local model that actually fits in free VRAM at `ctx`.

    "Is there GPU headroom" used to be answered with a percentage, which says
    nothing about whether the model you have can load. This checks the real
    number: weights + KV cache + overhead against free memory.
    """
    gpu = gpu_info()
    if not gpu:
        return None
    budget = (gpu["total"] - gpu["used"]) / 1024.0
    best = None
    for name in (detect_ollama().get("models") or [])[:limit]:
        spec = ollama_model_info(name)
        if not spec:
            continue
        est = estimate_vram(spec, ctx)
        if est["total_gib"] <= budget and (best is None or
                                           spec["params"] > best[1]["params"]):
            best = (name, spec, est)
    if not best:
        return None
    return {"model": best[0], "gib": best[2]["total_gib"], "budget_gib": budget,
            "params": best[1]["params"], "ctx": ctx}


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
        fit = best_local_fit() if local_ok else None
        if fit:
            rec = (f"Ollama {fit['model']} (local, free) — fits in "
                   f"{fit['gib']:.1f} of {fit['budget_gib']:.1f} GiB free "
                   f"at {fit['ctx']:,} ctx")
        elif gpu and local_ok:
            rec = ("Claude Code (remote, paid) — no installed local model fits "
                   f"in {(gpu['total'] - gpu['used']) / 1024.0:.1f} GiB free "
                   "(see `lmm fit`)")
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
        print("[fit] no model given and no Ollama models installed. "
              "Try: lmm fit llama3.1:8b --vram 24")
        return

    rows = []
    for name in models:
        spec = ollama_model_info(name)
        if not spec:
            params = params_from_name(name)
            rows.append({"model": name, "error": (
                "no metadata (start Ollama so `lmm fit` can read the real layer "
                "and KV-head counts)"),
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
        print(f"  [{mark}] {r['model']:<28} {e['total_gib']:6.2f} GiB "
              f"@ {r['ctx']:,} ctx")
        print(f"         weights {e['weights_gib']:.2f} + kv {e['kv_gib']:.2f} "
              f"+ overhead {e['overhead_gib']:.2f}   "
              f"({s['params']/1e9:.1f}B params, {e['bpw']:.2f} bpw, "
              f"{s['layers']}L, {s['kv_heads']} kv-heads x {s['head_dim']})")
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
    print("Estimates. bits-per-weight are llama.cpp measurements on LLaMA-family "
          "models;\noverhead is a 0.5 GiB middle estimate for context and scratch "
          "buffers.")


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
    if not provs and not local_ollama_provider():
        print("[hub] no providers configured and no local Ollama running. "
              "Start Ollama (`lmm serve <model>`) or add 'providers' to lmm "
              "config (see `lmm examples`). Nothing to proxy.")
        return

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
            # Routing, cascade, cache and metering are the SAME code path as
            # `lmm ask` (hub_complete) — one hub, one routing brain. The whole
            # messages array is forwarded, so system prompts and prior turns
            # survive the hop.
            msgs = req.get("messages", [])
            explicit = req.get("model", "")  # may be a configured provider name
            targets = resolve_ask_targets(
                cfg, messages_text(msgs), explicit if explicit in provs else None)
            if not targets:
                self._send(400, {"error": "no provider available for model '%s'" % explicit})
                return
            no_cache = (bool(req.get("lmm_no_cache"))
                        or self.headers.get("X-LMM-No-Cache") is not None)
            extra = {k: req[k] for k in PASSTHROUGH_KEYS if k in req}
            res, _trace = hub_complete(cfg, msgs, targets,
                                       {"cascade": bool(req.get("lmm_cascade")),
                                        "cache": not no_cache,
                                        "extra": extra, "source": "hub"})
            if isinstance(res, dict) and res.get("error"):
                self._send(502, {"error": "all providers failed: %s" % res["error"]})
                return
            self._send(200, res)

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
                       "model": "gpt-4o", "kind": "remote",
                       "price": "openai-gpt4o"},
            "gemini": {"api_key": "AIza...", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "model": "gemini-1.5-pro", "kind": "remote",
                       "price": {"in": 1.25, "out": 5.0}},
            "my-local": {"api_key": "ollama", "base_url": "http://localhost:11434/v1",
                         "model": "qwen2.5-coder:7b", "kind": "local"}
        },
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
            "embed_model": "nomic-embed-text", "max_temp": 0.3
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



# ------------------- (2) RouteLLM-style threshold routing -------------------
# Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data",
# ICLR 2025 (arXiv:2406.18665). Their router predicts the probability that the
# strong model wins and converts it into a strong-vs-weak decision with a cost
# threshold alpha; they report ~40% fewer strong-model calls at <5% quality
# loss. Their predictor is trained on Chatbot Arena preferences, which we have
# neither the data nor the dependencies to reproduce, so the win probability is
# approximated by cheap lexical features. The threshold half of the framework
# is exactly as published, and it is the half the user tunes.
def is_private(cfg, text):
    """Privacy beats price: a prompt matching route.private stays local even if
    a cloud model would answer it better."""
    low = (text or "").lower()
    return any(k.lower() in low for k in merged_route(cfg).get("private", []))


def prompt_strength(cfg, prompt):
    """Score how likely a prompt needs the strong model. Returns (score, feats)
    where score is in [0,1] and feats is a list of (label, weight) for
    `lmm route --explain`."""
    text = prompt if isinstance(prompt, str) else messages_text(prompt)
    t = (text or "").strip()
    low = t.lower()
    feats, raw = [], 0.0

    hit = next((k for k in merged_route(cfg).get("heavy", []) if k.lower() in low), None)
    if hit:
        feats.append((f"heavy keyword ({hit})", 0.30)); raw += 0.30
    cm = next((m for m in STRENGTH_CODE_MARKERS if m in low), None)
    if cm:
        feats.append((f"code marker ({cm.strip()})", 0.25)); raw += 0.25
    rm = next((m for m in STRENGTH_REASON_MARKERS if m in low), None)
    if rm:
        feats.append((f"reasoning marker ({rm.strip()})", 0.20)); raw += 0.20
    n = len(t)
    if n > 1000:
        feats.append((f"long prompt ({n} chars)", 0.25)); raw += 0.25
    elif n > 200:
        feats.append((f"medium prompt ({n} chars)", 0.15)); raw += 0.15
    q = low.count("?") + low.count("？")
    if q >= 3:
        feats.append((f"multi-question ({q})", 0.10)); raw += 0.10
    # The short-prompt discount catches "hi" / "what time is it", not a terse
    # but demanding "refactor this" — so it only applies when nothing above
    # already flagged the prompt as hard.
    if n < 40 and q <= 1 and not feats:
        feats.append((f"short one-liner ({n} chars)", -0.30)); raw -= 0.30

    # Logistic squash centred so that a single "heavy" keyword lands exactly on
    # the default threshold — i.e. the pre-existing keyword behaviour is the
    # calibration point, and everything else moves relative to it.
    score = 1.0 / (1.0 + math.exp(-6.0 * (raw - 0.30)))
    return score, feats


def target_price(cfg, prov, pricing=None):
    """Output-token price for one target, used to order cheap->expensive."""
    pricing = pricing or merged_pricing(cfg)
    return price_for(prov, prov.get("model"), pricing)["out"]


def order_targets(cfg, prompt, targets):
    """Reorder candidate providers by predicted need, cheapest-first when the
    prompt looks easy. A user-defined cfg['ask_order'] is absolute and is never
    reordered — auto-routing only decides what was left undecided."""
    if not targets or cfg.get("ask_order"):
        return targets
    thr = cfg.get("route_threshold", DEFAULT_ROUTE_THRESHOLD)
    if thr is None:
        return targets
    text = prompt if isinstance(prompt, str) else messages_text(prompt)
    if is_private(cfg, text):
        return sorted(targets, key=lambda t: 0 if t[1].get("kind") == "local" else 1)
    pricing = merged_pricing(cfg)
    score, _ = prompt_strength(cfg, prompt)
    if score >= thr:                       # strong-first: priciest is the proxy
        return sorted(targets, key=lambda t: -target_price(cfg, t[1], pricing))
    return sorted(targets, key=lambda t: target_price(cfg, t[1], pricing))


# --------------------- (1) FrugalGPT-style LLM cascade ----------------------
# Chen, Zaharia & Zou, "FrugalGPT" (arXiv:2305.05176): query cheap models
# first, score the answer, and only escalate when the score is too low — up to
# 98% cost reduction at GPT-4-level accuracy. Their scoring function is a
# trained DistilBERT regressor. With stdlib-only as a hard constraint we
# instead penalise the failure modes small models actually exhibit, and let the
# user bolt on a real LLM judge via cascade.judge when they want one.
def verify_answer(prompt, answer, cfg=None, judge=None):
    """Score an answer in [0,1] and explain the deductions. Low means the
    cheap model probably failed and the request should escalate."""
    text = (answer or "").strip()
    if not text or len(text.split()) < 3:
        return 0.0, ["empty or near-empty answer"]
    low = text.lower()
    ptext = (prompt if isinstance(prompt, str) else messages_text(prompt)) or ""
    score, why = 1.0, []

    m = next((k for k in VERIFY_REFUSAL_MARKERS if k in low), None)
    if m:
        score -= 0.5; why.append(f"refusal marker ({m.strip()})")
    m = next((k for k in VERIFY_HEDGE_MARKERS if k in low), None)
    if m:
        score -= 0.15; why.append(f"hedging ({m.strip()})")
    if text.count("```") % 2 == 1:
        score -= 0.25; why.append("unclosed code fence")
    elif text[-1] not in ".!?)\"'`}】。！？」…":
        score -= 0.25; why.append("truncated mid-sentence")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and len(lines) - len(set(lines)) >= 2:
        score -= 0.30; why.append("degenerate repetition")
    wants_code = any(m in ptext.lower() for m in STRENGTH_CODE_MARKERS)
    if wants_code and "```" not in text:
        score -= 0.30; why.append("code was asked for, none returned")
    if len(text) < len(ptext) * 0.1 and ptext.count(".") + ptext.count("?") > 1:
        score -= 0.20; why.append("answer far shorter than a multi-part prompt")

    score = max(0.0, min(1.0, score))
    if judge:
        js = judge_answer(judge, ptext, text)
        if js is not None:
            why.append(f"llm judge {js:.2f}")
            score = (score + js) / 2.0
    return score, why


def judge_answer(judge_prov, prompt, answer):
    """Optional second opinion: ask a configured provider to grade the answer.
    Returns a float in [0,1] or None if the judge was unusable."""
    q = ("Grade how well the ANSWER addresses the QUESTION. "
         "Reply with only a number between 0.0 and 1.0, nothing else.\n\n"
         "QUESTION:\n%s\n\nANSWER:\n%s" % (prompt[:4000], answer[:4000]))
    res = call_provider(judge_prov, q, temperature=0.0)
    if not isinstance(res, dict) or res.get("error"):
        return None
    try:
        raw = res["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return None
    num = ""
    for ch in raw:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    try:
        return max(0.0, min(1.0, float(num)))
    except ValueError:
        return None


def cascade_rungs(cfg, targets, prompt=None):
    """Order the rungs cheapest-first, which is what makes a cascade cheap.
    An explicit cascade.rungs list wins; otherwise it is derived from price, so
    a zero-config user still gets local -> cheap cloud -> expensive cloud.

    Escalation must never leak: a prompt matching route.private stays on local
    rungs, because "the cheap model did badly" is not a reason to ship a secret
    to a cloud API.
    """
    casc = merged_cascade(cfg)
    cap = max(1, int(casc.get("max_rungs", 3)))
    text = prompt if isinstance(prompt, str) else messages_text(prompt)
    if text and is_private(cfg, text):
        local = [t for t in targets if t[1].get("kind") == "local"]
        if local:
            return local[:cap]
    named = list(casc.get("rungs") or [])
    if named:
        by_name = dict(targets)
        rungs = [(n, by_name[n]) for n in named if n in by_name]
        if rungs:
            return rungs[:cap]
    pricing = merged_pricing(cfg)
    ordered = sorted(targets, key=lambda t: (0 if t[1].get("kind") == "local" else 1,
                                             target_price(cfg, t[1], pricing)))
    return ordered[:cap]


# ------------------ (3) prompt cache, exact + semantic tiers ----------------
# GPT Semantic Cache (arXiv:2411.05276) reports up to 68.8% fewer API calls by
# reusing answers to semantically similar prompts. vCache (arXiv:2502.03771)
# shows a single static similarity threshold cannot bound the false-hit rate,
# so tier 2 is opt-in, its default threshold is deliberately strict, and every
# near miss is recorded so the threshold can be tuned from evidence.
def normalize_prompt(text):
    """Whitespace-insensitive but indentation-preserving: trailing spaces and
    blank-line runs never change an answer, but leading indentation does (it is
    syntax in Python and YAML), so it is kept."""
    lines = [l.rstrip() for l in (text or "").strip().splitlines()]
    return "\n".join(l for i, l in enumerate(lines) if l or (i and lines[i - 1]))


def cache_key(messages, model):
    payload = normalize_prompt(messages_text(messages)) + "\x00" + (model or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_text(text, model):
    """Embed via the LOCAL Ollama server only. Paying a cloud provider to
    populate a cache that exists to avoid paying cloud providers would be
    self-defeating, so there is deliberately no remote embedding path."""
    body = http_post_json("http://localhost:11434/api/embed",
                          {"model": model, "input": text[:8000]}, "", timeout=20)
    if isinstance(body, dict) and body.get("embeddings"):
        return body["embeddings"][0]
    # older Ollama builds only speak the singular /api/embeddings form
    body = http_post_json("http://localhost:11434/api/embeddings",
                          {"model": model, "prompt": text[:8000]}, "", timeout=20)
    if isinstance(body, dict) and body.get("embedding"):
        return body["embedding"]
    return None


def cache_entries(conf):
    """Live (unexpired) cache entries, oldest first."""
    if not os.path.isfile(CACHE_LOG):
        return []
    ttl = float(conf.get("ttl_hours", 168)) * 3600
    now = time.time()
    out = []
    try:
        with open(CACHE_LOG, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if ttl > 0 and now - float(e.get("at", 0)) > ttl:
                    continue
                out.append(e)
    except Exception:
        pass
    return out


def cache_lookup(cfg, messages, model):
    """Return (entry, how, similarity). `how` is 'exact', 'semantic' or None."""
    conf = merged_cache(cfg)
    if not conf.get("enabled", True):
        return None, None, 0.0
    entries = cache_entries(conf)
    if not entries:
        return None, None, 0.0
    key = cache_key(messages, model)
    for e in reversed(entries):                  # newest wins
        if e.get("key") == key:
            return e, "exact", 1.0
    if not conf.get("semantic"):
        return None, None, 0.0
    text = normalize_prompt(messages_text(messages))
    emb = embed_text(text, conf.get("embed_model", "nomic-embed-text"))
    if not emb:
        return None, None, 0.0
    best, best_sim = None, 0.0
    for e in entries:
        if e.get("model") != model or not e.get("emb"):
            continue
        sim = cosine(emb, e["emb"])
        if sim > best_sim:
            best, best_sim = e, sim
    thr = float(conf.get("similarity", 0.95))
    if best and best_sim >= thr:
        return best, "semantic", best_sim
    if best:                                     # near miss: evidence for tuning
        log_usage({"provider": "cache", "model": model, "kind": "local",
                   "in": 0, "out": 0, "usd": 0.0, "cache": "near-miss",
                   "similarity": round(best_sim, 4), "threshold": thr})
    return None, None, best_sim


def cache_store(cfg, messages, model, result, usd=0.0, temperature=None):
    """Store one answer. `temperature` is the caller's EXPLICIT setting, or
    None if they never set one — only an explicit high temperature means the
    caller wants a fresh sample every time."""
    conf = merged_cache(cfg)
    if not conf.get("enabled", True):
        return
    if temperature is not None and float(temperature) > float(conf.get("max_temp", 0.3)):
        return                                   # caller wants variety, not reuse
    text = normalize_prompt(messages_text(messages))
    entry = {"at": time.time(), "key": cache_key(messages, model), "model": model,
             "usd": round(usd, 6), "result": result}
    if conf.get("semantic"):
        emb = embed_text(text, conf.get("embed_model", "nomic-embed-text"))
        if emb:
            entry["emb"] = emb
    try:
        os.makedirs(LMM_DIR, exist_ok=True)
        with open(CACHE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return
    cache_prune(conf)


def cache_prune(conf):
    """Rewrite the log when it outgrows max_entries, keeping the newest."""
    cap = int(conf.get("max_entries", 2000))
    try:
        entries = cache_entries(conf)
        if len(entries) <= cap * 1.5:
            return
        keep = entries[-cap:]
        tmp = CACHE_LOG + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for e in keep:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, CACHE_LOG)
    except Exception:
        pass


# ------------------------- unified completion path --------------------------
def hub_complete(cfg, messages, targets, opts=None):
    """One request path shared by `lmm ask` and `lmm serve --hub`:

        cache lookup -> threshold routing -> cascade -> metering -> cache store

    Returns (result, trace). `result` is an OpenAI-shaped response dict (or one
    carrying an "error"); `trace` is human-readable lines for --explain.
    """
    opts = opts or {}
    pricing = merged_pricing(cfg)
    casc = merged_cascade(cfg)
    extra = dict(opts.get("extra") or {})
    # None means "the caller never asked for a temperature". Only an explicit
    # high temperature signals a want for variety, so only that should suppress
    # caching — lmm's own 0.7 default must not disable the cache by accident.
    req_temp = extra.get("temperature")
    temp = 0.7 if req_temp is None else req_temp
    source = opts.get("source", "ask")
    trace = []

    use_cache = opts.get("cache", True)
    use_cascade = opts.get("cascade", casc.get("enabled", False))

    targets = order_targets(cfg, messages, targets)
    if not targets:
        return {"error": "no provider available"}, trace

    # Privacy is pinned in order_targets/cascade_rungs, but only if a local
    # provider exists at all. When none does, say so out loud rather than
    # quietly shipping a prompt the user flagged as private to a remote API.
    if (is_private(cfg, messages_text(as_messages(messages)))
            and not any(t[1].get("kind") == "local" for t in targets)):
        trace.append("[warn] prompt matches route.private but no local provider "
                     "is running — it will be sent to a remote API "
                     "(start one with `lmm serve <model>`)")

    # The cache key identifies the REQUEST, not whoever ended up answering it:
    # a cascade may answer from rung 2, and next time that answer should be
    # served for the same question. Ordering is deterministic given cfg, so the
    # first target is a stable identity for the request.
    cache_model = targets[0][1].get("model")

    # ---- (1) cache -----------------------------------------------------
    if use_cache:
        probe_model = cache_model
        entry, how, sim = cache_lookup(cfg, messages, probe_model)
        if entry:
            saved = float(entry.get("usd", 0.0) or 0.0)
            log_usage({"provider": "cache", "model": probe_model, "kind": "local",
                       "in": 0, "out": 0, "usd": 0.0, "cache": how,
                       "similarity": round(sim, 4), "saved_usd": round(saved, 6),
                       "source": source})
            trace.append(f"[cache] {how} hit (sim={sim:.3f}) — $0.0000, "
                         f"saved ~${saved:.4f}")
            return entry["result"], trace

    # ---- (2)+(3) routing already applied; now walk the rungs -----------
    rungs = cascade_rungs(cfg, targets, messages) if use_cascade else targets
    judge = None
    if use_cascade and casc.get("judge"):
        judge = merged_providers(cfg).get(casc["judge"])
    threshold = float(casc.get("threshold", 0.6))
    spent = 0.0
    best = None                      # (score, result, name) fallback
    last_err = None

    for i, (name, prov) in enumerate(rungs):
        res = call_provider(prov, messages, temperature=temp, extra=extra)
        if isinstance(res, dict) and res.get("error"):
            last_err = res["error"]
            trace.append(f"[{'cascade' if use_cascade else 'ask'}] "
                         f"rung{i} {name} failed: {last_err} -> next")
            continue
        try:
            answer = res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            last_err = "unexpected response shape"
            trace.append(f"[ask] rung{i} {name} returned an unexpected shape -> next")
            continue

        if not use_cascade:
            usd = meter_call(name, prov, prov.get("model"), res, pricing,
                             source=source, cache="miss", rung=i, accepted=True)
            trace.append(f"[ask] {name} ({prov.get('model')}) ${usd:.4f}")
            if use_cache:
                cache_store(cfg, messages, cache_model, res, usd, req_temp)
            return res, trace

        score, why = verify_answer(messages, answer, cfg, judge)
        accept = score >= threshold or i == len(rungs) - 1
        usd = meter_call(name, prov, prov.get("model"), res, pricing,
                         source=source, cache="miss", rung=i,
                         score=round(score, 3), accepted=bool(accept))
        spent += usd
        verdict = "accept" if score >= threshold else "escalate"
        trace.append(f"[cascade] rung{i} {name} score={score:.2f} ${usd:.4f} "
                     f"-> {verdict}" + (f" ({', '.join(why)})" if why else ""))
        if best is None or score > best[0]:
            best = (score, res, name)
        if score >= threshold:
            if use_cache:
                cache_store(cfg, messages, cache_model, res, spent, req_temp)
            trace.append(f"[cascade] total ${spent:.4f} over {i + 1} rung(s)")
            return res, trace

    if best:
        # Every rung scored low. FrugalGPT keeps the best-scoring answer rather
        # than trusting whichever model happened to be last.
        trace.append(f"[cascade] no rung cleared {threshold:.2f}; "
                     f"returning best ({best[2]}, score={best[0]:.2f}), "
                     f"total ${spent:.4f}")
        if use_cache:
            cache_store(cfg, messages, cache_model, best[1], spent, req_temp)
        return best[1], trace
    return {"error": last_err or "all providers failed"}, trace


def cmd_ask(prompt, provider, cfg, cascade=False, no_cache=False, explain=False):
    """Unified inference over every backend: cache, threshold routing, optional
    cheap-first cascade, and metering — all of it the same code path the hub
    serves, so `lmm ask` and an app pointed at `lmm serve --hub` behave alike."""
    if not (prompt or "").strip():
        print("usage: lmm ask \"your question\" [--provider NAME] [--cascade]")
        return
    targets = resolve_ask_targets(cfg, prompt, provider)
    if not targets:
        print("[ask] no provider available. Start Ollama (`lmm serve <model>`) "
              "or add 'providers' to lmm config (see `lmm examples`).")
        return
    if explain:
        score, feats = prompt_strength(cfg, prompt)
        thr = cfg.get("route_threshold", DEFAULT_ROUTE_THRESHOLD)
        print(f"[route] strength={score:.2f} threshold={thr} -> "
              + ", ".join(n for n, _ in order_targets(cfg, prompt, targets)))
    res, trace = hub_complete(cfg, prompt, targets,
                              {"cascade": cascade, "cache": not no_cache,
                               "source": "ask"})
    for line in trace:               # warnings are never hidden behind --explain
        if explain or line.startswith("[warn]"):
            print(line)
    if isinstance(res, dict) and res.get("error"):
        print(f"[ask] all providers failed. last error: {res['error']}")
        return
    try:
        print(res["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        print("[ask] unexpected response shape from provider")


def cmd_cache(clear=False):
    """Inspect or drop the prompt cache. The near-miss similarities are the
    evidence for tuning cache.similarity — vCache (arXiv:2502.03771) makes the
    case that a static threshold picked blind is not safe."""
    if clear:
        try:
            if os.path.isfile(CACHE_LOG):
                os.remove(CACHE_LOG)
            print("[cache] cleared")
        except OSError as e:
            print(f"[cache] could not clear: {e}")
        return
    entries = cache_entries(DEFAULT_CACHE)
    print(f"[cache] {CACHE_LOG}")
    print(f"[cache] {len(entries)} live entries, "
          f"{sum(1 for e in entries if e.get('emb'))} with embeddings")
    if entries:
        oldest = min(e.get("at", 0) for e in entries)
        print("[cache] oldest: "
              + datetime.datetime.fromtimestamp(oldest).strftime("%Y-%m-%d %H:%M"))
        print(f"[cache] value stored: ${sum(float(e.get('usd', 0) or 0) for e in entries):,.4f}")
    hits = {"exact": 0, "semantic": 0, "near-miss": 0}
    sims = []
    for ev in read_usage():
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
              f"(threshold {DEFAULT_CACHE['similarity']})")



def main():
    ap = argparse.ArgumentParser(
        description="LMM - Local/remote Model Manager (cross-platform, zero-dep)")
    ap.add_argument("-v", "--version", action="store_true", help="show version")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("discover").add_argument("--json", action="store_true")
    sub.add_parser("cli", help="same as discover (explicit CLI mode)")
    sub.add_parser("status")
    sub.add_parser("models")
    p = sub.add_parser("cost")
    p.add_argument("--days", type=int, default=30,
                   help="only count the last N days (0 = all-time)")
    p = sub.add_parser("route")
    p.add_argument("task", nargs="?")
    p.add_argument("--explain", action="store_true",
                   help="show the strength-score breakdown and provider order")
    p = sub.add_parser("serve")
    p.add_argument("model", nargs="?")
    p.add_argument("--hub", action="store_true",
                   help="start OpenAI-compatible proxy over all configured providers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
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
    p.add_argument("--cascade", action="store_true",
                   help="cheap-first cascade: escalate only on a low-scoring answer")
    p.add_argument("--no-cache", action="store_true", help="bypass the prompt cache")
    p.add_argument("--explain", action="store_true",
                   help="show routing score, rung scores and per-call cost")
    p = sub.add_parser("fit", help="will this model fit in your GPU, and at what context?")
    p.add_argument("model", nargs="?", help="model tag; omit to check every installed one")
    p.add_argument("--ctx", type=int, default=None, help="context length in tokens")
    p.add_argument("--vram", type=float, default=None, help="override detected free VRAM (GiB)")
    p.add_argument("--kv", default="f16", choices=sorted(set(KV_BYTES)),
                   help="KV cache dtype (llama.cpp --cache-type-k/v)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("cache")
    p.add_argument("--clear", action="store_true", help="delete all cached answers")
    p.add_argument("--stats", action="store_true", help="show cache stats (default)")
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
        print(cost_report(cfg, args.days or None))
    elif cmd == "route":
        cmd_route(cfg, args.task, getattr(args, "explain", False))
    elif cmd == "serve":
        if getattr(args, "hub", False):
            cmd_serve_hub(cfg, args.host, args.port)
        else:
            cmd_serve(args.model)
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
    elif cmd == "hide":
        cmd_hide(args.runtime)
    elif cmd == "ask":
        cmd_ask(" ".join(getattr(args, "prompt", [])),
                getattr(args, "provider", None), cfg,
                cascade=getattr(args, "cascade", False),
                no_cache=getattr(args, "no_cache", False),
                explain=getattr(args, "explain", False))
    elif cmd == "fit":
        cmd_fit(args.model, args.ctx, args.vram, args.kv, getattr(args, "json", False))
    elif cmd == "cache":
        cmd_cache(getattr(args, "clear", False))
    elif cmd == "examples":
        cmd_examples()


if __name__ == "__main__":
    main()
