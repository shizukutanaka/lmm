#!/usr/bin/env bash
# lmm installer (macOS / Linux). Copies lmm.py to ~/.local/bin/lmm and adds
# an alias to your shell rc if not already present.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/lmm.py"
DEST_DIR="${HOME}/.local/bin"
DEST="${DEST_DIR}/lmm"
mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
chmod +x "$DEST"
echo "installed -> $DEST"

# try common rc files
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
  if [ -f "$rc" ] && ! grep -q "alias lmm=" "$rc"; then
    printf '\n# LMM - Local/remote Model Manager\nalias lmm="%s"\n' "$DEST" >> "$rc"
    echo "added alias to $rc (restart shell or: source $rc)"
  fi
done
echo "done. Try: lmm discover"
