#!/usr/bin/env bash
# Existing-artifact review only. This entry never invokes a model/generation queue.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REVIEW_PYTHON:-}" ]]; then
  PYTHON_BIN="$REVIEW_PYTHON"
elif [[ -x "$HOME/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python" ]]; then
  PYTHON_BIN="$HOME/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "复核未启动：未找到可执行Python；请设置REVIEW_PYTHON为Python 3.10+解释器完整路径。" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
exec "$PYTHON_BIN" -B "$SCRIPT_DIR/review_driver.py" "$@"
