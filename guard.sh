#!/usr/bin/env bash
# Local equivalent of .github/workflows/guard.yml — run BEFORE push/commit.
# Mirrors the server-side self-guard so lmm can prove itself OFFLINE,
# independent of GitHub Actions. Even if the remote CI is skipped, the pushed
# code must still self-prove locally.
#
# Usage:  bash guard.sh
# Exit 0 = self-test passes, non-zero = broken.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || { echo "[guard] cannot cd to repo"; exit 1; }

export LMM_SELFTEST_SKIP_LIVE="1"   # no live Ollama on this box needed
echo "[guard] lmm self-proves (measure, don't trust) — selftest --guard"
if python lmm.py selftest --guard; then
  echo "[guard] ALL OK"
  exit 0
else
  echo "[guard] FAILED — lmm.py did not self-prove; do not push"
  exit 1
fi
