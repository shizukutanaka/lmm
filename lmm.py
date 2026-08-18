#!/usr/bin/env python3
# lmm.py - Local/remote Model Manager
# ---------------------------------------------------------------------------
# Thin entry point. Implementation split into two focused layers:
#   backend.py  — LLM execution, routing, verification, the cascade, the prompt
#                 cache, metering, detection and providers (no CLI, no GUI)
#   frontend.py — CLI argument parsing (cmd_*) and the tkinter GUI dashboard
# lmm.py only wires them together and launches the entry point. It re-exports
# both, so `import lmm` still hands you the whole surface in one name.
#
# Design (first-principles): separation of concerns — a THIN frontend talks to a
# THIN backend. No secrets are ever stored; credentials are only checked, never
# saved. Cross-platform (Windows / macOS / Linux), stdlib only.
#
# Commands (no subcommand -> the live GUI dashboard):
#   gui                 live GUI dashboard (tkinter; the default command)
#   discover [--json]   list every detected runtime (--save seeds ask_order)
#   cli                 same as discover (explicit CLI mode)
#   status              runtimes + GPU + hub/cache/breaker summary
#   models              models on every running runtime and configured provider
#   pull <model>        pull a model into local Ollama
#   fit [model|*.gguf]  does it fit in your GPU, and at what context length?
#   cost [--days N]     measured spend: Anthropic session logs + lmm's own hub
#   route "task ..."    recommend local vs remote for a task (--explain)
#   priority            show or re-rank ask_order by MEASURED performance
#   bench               measure TTFT / TPOT / throughput per provider
#   ask "prompt"        one question, routed across every backend
#                       (--cascade, --auto, --verify, --explain, --no-cache)
#   chat                interactive REPL that keeps conversation history
#   serve <model>       pull + expose a local model endpoint (Ollama)
#   serve --hub         OpenAI-compatible proxy over all configured providers
#   hub-status          probe every backend the hub would route to
#   stop  <runtime>     stop a running runtime
#   cache               prompt-cache stats (--clear)
#   selftest [--guard]  self-prove the hub works (measure, don't trust)
#   doctor              diagnose configuration and reachability
#   secrets             scan the config for anything that must not be there
#   stats               measured routing outcomes from the hub log
#   log [n]             recent hub events (proof of what was routed where)
#   config <action>     manage lmm config (init/list/get/set/unset)
#   hide  <runtime>     strip a runtime's taskbar button (Windows)
#   watch               daemon: auto-hide new LLM windows (Windows)
#   autostart           register `watch` at login (Windows)
#   dash                generate + open a self-contained HTML dashboard
#   examples            show a sample config file you can copy
#
# Config (~/.lmm/config.json) lets anyone add their own runtimes, override
# pricing, or change routing keywords. See `lmm examples`.
# ---------------------------------------------------------------------------
import os
import sys

# lmm is three files that must travel together. Copying lmm.py alone used to be
# the documented install, so a missing sibling is a likely mistake rather than a
# broken checkout — say which file is missing and where it should be, instead of
# letting a bare ModuleNotFoundError be the whole answer.
try:
    from backend import *          # noqa: F401,F403  — the engine
    from frontend import *         # noqa: F401,F403  — the CLI and GUI
    from frontend import main
except ImportError as _e:
    _here = os.path.dirname(os.path.abspath(__file__))
    _missing = [f for f in ("backend.py", "frontend.py")
                if not os.path.isfile(os.path.join(_here, f))]
    if not _missing:
        raise
    sys.stderr.write(
        "lmm: missing %s next to lmm.py in %s\n"
        "lmm is three files - lmm.py, backend.py and frontend.py - and they\n"
        "must sit in the same directory. Re-run the installer, or fetch them\n"
        "together:\n"
        "  for f in lmm.py backend.py frontend.py; do curl -fsSLO "
        "https://raw.githubusercontent.com/shizukutanaka/lmm/main/$f; done\n"
        % (" and ".join(_missing), _here))
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # `lmm discover | head` — the reader closed first. That is how pipes
        # are supposed to work, not an error; a traceback here fails basic
        # UNIX table manners. Python would also complain again while flushing
        # stdout at exit, so hand it a sink first.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
    except KeyboardInterrupt:
        print()
        sys.exit(130)          # the conventional 128+SIGINT
