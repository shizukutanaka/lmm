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
lmm cost [--days N] -> measured Anthropic token cost from your session logs
lmm route "task"    -> recommend local vs remote for a task
lmm serve <model>   -> pull + expose a local model endpoint (Ollama)
lmm stop <runtime>  -> stop a running runtime
lmm hide <runtime>  -> strip a runtime's taskbar button (Claude/ChatGPT/...)
lmm watch           -> background daemon: auto-hide new LLM windows
lmm autostart       -> register `watch` to run at OS login (zero effort)
lmm dash            -> generate + open a self-contained HTML dashboard
lmm gui             -> open the live GUI dashboard explicitly
lmm examples        -> print a sample config file
```

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

## Configuration

Works with zero config. `~/.lmm/config.json` (or `~/.config/lmm/config.json`,
or `./lmm.config.json`) can add runtimes, override pricing, or change routing
keywords. See `lmm examples` for the shape.

## How it works

| Concern        | Approach                                                        |
|----------------|-----------------------------------------------------------------|
| Discovery      | Probe each runtime's process + data dir + endpoint              |
| Secrets        | Only *check* existence of existing credentials; never store one |
| Cost           | Aggregate real tokens from `~/.claude/projects/*.jsonl` × public pricing |
| Taskbar hide   | `WS_EX_TOOLWINDOW` on visible windows (Win); headless launch advised for servers |
| Auto mode      | `watch` daemon hides new LLM windows every few seconds          |
| Portability    | `expanduser` paths, `tasklist`/`pgrep`, `tkinter` for the GUI   |

## License

MIT — see [LICENSE](LICENSE).
