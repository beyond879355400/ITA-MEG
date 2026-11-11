#!/bin/zsh
set -euo pipefail
HERE=${0:A:h}
EXP=${HERE:h:h:h}
export PYTHONPATH="$EXP/improvements/quality_refinement_20260917/runtime_repair_v2"
export PYTHONPYCACHEPREFIX="$HERE/pycache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
exec '/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python' -u "$HERE/run_diagnostic.py" "$@"
