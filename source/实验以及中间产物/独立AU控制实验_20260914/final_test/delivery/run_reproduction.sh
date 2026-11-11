#!/usr/bin/env bash
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
# Keep the actual venv interpreter path; do not resolve it to system Python.
PY="$HOME/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python"
if [ ! -x "$PY" ]; then
  echo "REFUSED: original verified Python environment is unavailable: $PY" >&2
  exit 2
fi
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" -B "$HERE/run_reproduction.py" "$@"
