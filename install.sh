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
added=0
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
  if [ -f "$rc" ]; then
    if ! grep -q "alias lmm=" "$rc"; then
      printf '\n# LMM - Local/remote Model Manager\nalias lmm="%s"\n' "$DEST" >> "$rc"
      echo "added alias to $rc (restart shell or: source $rc)"
    fi
    added=1
  fi
done
# without this, a user with no ~/.bashrc or ~/.zshrc got no alias and no hint
if [ "$added" -eq 0 ]; then
  echo "no ~/.bashrc or ~/.zshrc found; add manually:  alias lmm=\"$DEST\""
fi
echo "done. Try: lmm discover"
