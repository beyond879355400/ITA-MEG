#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python}"
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not executable: $PYTHON_BIN" >&2
  exit 2
fi

# Existing JSON configs resolve their relative paths from the formal thesis root.
cd -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_epoch1_matrix.py" "$@"
