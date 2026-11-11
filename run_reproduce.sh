#!/usr/bin/env bash
# No package installation or model execution occurs implicitly.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
mode="${1:-audit}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  audit) exec "$PYTHON_BIN" -B "$ROOT/scripts/audit_records.py" "$@" ;;
  test)
    "$PYTHON_BIN" -B -m unittest discover -s "$ROOT/tests" -v
    "$PYTHON_BIN" -B "$ROOT/source/实验以及中间产物/独立AU控制实验_20260914/statistics/test_paired_subject_statistics.py"
    exec "$PYTHON_BIN" -B "$ROOT/source/实验以及中间产物/盲审补充实验_20261023/generation/test_ablation_runner.py" ;;
  *)
    echo "Usage: bash run_reproduce.sh [audit [--output NEW.json]|test]" >&2
    echo "For actual training/generation/evaluation see docs/REPRODUCING.md." >&2
    exit 2 ;;
esac
