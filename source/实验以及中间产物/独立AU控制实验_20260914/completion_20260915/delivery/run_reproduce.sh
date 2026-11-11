#!/usr/bin/env bash
# Default is dry-run. Model generation additionally requires explicit stage and terminal guards.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REPRODUCE_PYTHON:-$HOME/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "未启动：当前冻结Mac环境的Python不存在。请先准备合法数据/权重与原运行环境；本入口不会安装或下载。" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
exec "$PYTHON_BIN" -B "$SCRIPT_DIR/reproduce_driver.py" "$@"
