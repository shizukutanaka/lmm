#!/usr/bin/env bash
# lmm installer (macOS / Linux).
#
# lmm is three files that import each other — lmm.py (entry), backend.py
# (engine) and frontend.py (CLI + GUI) — so they are installed together into a
# library directory, and a small launcher goes on your PATH. Copying only
# lmm.py, as this script used to, produced an `lmm` that could not start.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="${HOME}/.local/share/lmm"
DEST_DIR="${HOME}/.local/bin"
DEST="${DEST_DIR}/lmm"

for f in lmm.py backend.py frontend.py; do
  if [ ! -f "${SRC_DIR}/${f}" ]; then
    echo "install: ${f} not found next to this script — incomplete checkout?" >&2
    exit 1
  fi
done

mkdir -p "$LIB_DIR" "$DEST_DIR"
cp "${SRC_DIR}/lmm.py" "${SRC_DIR}/backend.py" "${SRC_DIR}/frontend.py" "$LIB_DIR/"
echo "installed library -> $LIB_DIR"

# A launcher, not a copy: it keeps the three files together and picks up an
# upgrade to the library without being reinstalled itself.
cat > "$DEST" <<EOF
#!/usr/bin/env bash
exec "\${LMM_PYTHON:-python3}" "${LIB_DIR}/lmm.py" "\$@"
EOF
chmod +x "$DEST"
echo "installed launcher -> $DEST"

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
