#!/usr/bin/env bash
# Single fixed request, demo-only. Default dry-run never loads models.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REPRODUCE_PYTHON:-$HOME/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "未启动：需要当前Mac冻结环境中的Python、合法数据和权重；不自动下载或安装。" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
exec "$PYTHON_BIN" -B "$SCRIPT_DIR/demo_driver.py" "$@"
