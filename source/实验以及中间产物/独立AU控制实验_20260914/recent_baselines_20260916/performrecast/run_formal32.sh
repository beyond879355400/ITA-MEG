#!/usr/bin/env bash
# Explicitly gated, append-only fixed32 native PerformRecast batch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/env_overlay_v2/bin/python"
RUN_DIR="${FORMAL_RUN_DIR:-${ROOT}/formal_runs/final32_v3}"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:-}" in
  --prepare)
    "${PYTHON}" -B "${ROOT}/formal_batch.py" prepare --run-dir "${RUN_DIR}" --python "${PYTHON}"
    ;;
  --run|--resume)
    [[ "${PERFORMRECAST_COMPUTE_GO:-}" == 1 ]] || { echo 'COMPUTE_GO absent; no model loaded'; exit 2; }
    [[ -f "${RUN_DIR}/frozen_protocol.json" ]] || { echo 'Prepare and review fixed protocol first'; exit 2; }
    mkdir -p "${RUN_DIR}/logs"
    LOG="${RUN_DIR}/logs/session_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    [[ ! -e "${LOG}" ]] || { echo 'Log collision; no overwrite'; exit 2; }
    ARGS=(run --run-dir "${RUN_DIR}" --execute)
    [[ "$1" == --resume ]] && ARGS+=(--resume)
    "${PYTHON}" -u -B "${ROOT}/formal_batch.py" "${ARGS[@]}" 2>&1 | tee "${LOG}"
    ;;
  *)
    echo 'Plan only.32 original cases,1 deterministic selected frame/case; no seed aliases.'
    echo 'Prepare (no model): bash run_formal32.sh --prepare'
    echo 'After root compute slot: PERFORMRECAST_COMPUTE_GO=1 bash run_formal32.sh --run'
    echo 'Interrupted resume: PERFORMRECAST_COMPUTE_GO=1 bash run_formal32.sh --resume'
    ;;
esac
