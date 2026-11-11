#!/usr/bin/env bash
# Reproduce the fixed4 development smoke cases locally; no metrics or new cases.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/env_overlay_v1/bin/python"
if [[ "${1:-}" != "--execute" ]]; then
  echo 'Plan only: CPU FP32, fixed4 smoke_dev4 cases, mode1, smoothing off,4 lossless frames each, select native frame0000.'
  echo 'No models loaded. No weights downloaded. Original author checkpoint/inputs are hash-checked.'
  echo 'To run after closing other memory-heavy inference jobs:'
  echo 'PERFORMRECAST_GPU_RUN_GO=1 bash run_local_smoke4.sh --execute'
  exit 0
fi
[[ "${PERFORMRECAST_GPU_RUN_GO:-}" == 1 ]] || { echo 'Explicit compute gate absent'; exit 2; }
[[ -x "${PYTHON}" ]] || { echo 'The checked overlay Python is missing'; exit 2; }
"${PYTHON}" -B -c 'import shutil,sys; sys.exit(0 if shutil.disk_usage(sys.argv[1]).free >= 8*1024**3 else "Less than8 GiB free; no model run")' "${ROOT}"
LOCK="${ROOT}/.local_smoke_compute.lock"
mkdir "${LOCK}" || { echo 'Another local smoke run holds the lock; do not remove it while running'; exit 2; }
trap 'rmdir "${LOCK}"' EXIT
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/reruns/${TAG}"
mkdir -p "${ROOT}/reruns"
mkdir "${OUT}"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${PYTHON}" -u -B "${ROOT}/run_smoke.py" --device cpu --output "${OUT}/first" --execute 2>&1 | tee "${OUT}/first.log"
"${PYTHON}" -u -B "${ROOT}/run_remaining.py" --output "${OUT}/remaining" --execute 2>&1 | tee "${OUT}/remaining.log"
echo "Complete. First-case receipt: ${OUT}/first/receipt.json"
echo "Remaining-case receipt: ${OUT}/remaining/receipt.json"
