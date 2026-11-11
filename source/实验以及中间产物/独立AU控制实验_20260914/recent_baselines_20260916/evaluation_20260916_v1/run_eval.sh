#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../runtime_local_v2/bin/python"
export PYTHONPATH="$HERE/insightface_built_v1:$HERE/runtime_overlay_v1"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$HERE/overlay_tmp_v1"
if [[ "${1:-}" == "--extra-bridge" ]]; then
  shift
  exec "$PY" -B "$HERE/extra_bridge.py" "$@"
fi
exec "$PY" -B "$HERE/evaluate_new.py" "$@"
