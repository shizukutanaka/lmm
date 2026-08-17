#!/usr/bin/env python3
# lmm.py - Local/remote Model Manager
# ---------------------------------------------------------------------------
# Thin entry point. Implementation split into two focused layers:
#   backend.py  — LLM execution, routing, measurement, detection, providers
#                 (zero-dependency, framework-free core logic)
#   frontend.py — CLI argument parsing (cmd_*) and the tkinter GUI dashboard
# lmm.py only wires them together and launches the entry point.
#
# Design (first-principles): separation of concerns — a THIN frontend talks to a
# THIN backend. No secrets are ever stored; credentials are only checked, never
# saved. Cross-platform (Windows / macOS / Linux), stdlib only.

from backend import *
from frontend import main

if __name__ == "__main__":
    main()
