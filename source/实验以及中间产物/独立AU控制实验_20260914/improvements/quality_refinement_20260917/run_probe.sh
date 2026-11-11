#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$HERE/runtime_repair_v2${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -B "$HERE/run_probe.py" "$@"
