#!/usr/bin/env bash
# Read-only metadata/model/code integrity audit. Never reads image/video payloads.
# Exit 0: verified package metadata; 2: integrity/dependency failure; 3: pending.
set -u
set -o pipefail
if [ "$#" -ne 0 ]; then
  echo 'REFUSED: read-only verification takes no arguments and never executes/repeats the generation queue.'
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
FINAL="$ROOT/final_test"
PROTOCOL="$FINAL/frozen_protocol.json"
EXPECTED_PROTOCOL=41d0f77e1307fd7da22e212a2485101fa6ab9662f11b02ad63fdb94d0bceda55
for dependency in jq rg; do
  command -v "$dependency" >/dev/null 2>&1 || { echo "FAIL missing dependency: $dependency"; exit 2; }
done
if command -v sha256sum >/dev/null 2>&1; then
  digest() { sha256sum -- "$1" | cut -d ' ' -f 1; }
elif command -v shasum >/dev/null 2>&1; then
  digest() { shasum -a 256 -- "$1" | cut -d ' ' -f 1; }
else
  echo 'FAIL need sha256sum or shasum'; exit 2
fi
checked=0; skipped_media=0; failures=0; pending=0
fail() { failures=$((failures + 1)); if [ "$failures" -le 20 ]; then echo "FAIL $*"; fi; }
wait_for() { pending=$((pending + 1)); echo "PENDING $*"; }
check_file() {
  local expected=$1 path=$2 actual
  case "$path" in /*) ;; *) path="$ROOT/$path" ;; esac
  case "$path" in *.png|*.jpg|*.jpeg|*.bmp|*.webp|*.gif|*.avi|*.mp4|*.mov|*.mkv)
    skipped_media=$((skipped_media + 1)); return ;;
  esac
  if [ ! -f "$path" ]; then fail "missing: $path"; return; fi
  actual=$(digest "$path") || { fail "hash failed: $path"; return; }
  if [ "$actual" != "$expected" ]; then fail "SHA mismatch: $path"; else checked=$((checked + 1)); fi
}

if [ ! -f "$PROTOCOL" ]; then echo 'PENDING frozen_protocol.json absent'; exit 3; fi
check_file "$EXPECTED_PROTOCOL" "$PROTOCOL"
if [ "$failures" -ne 0 ]; then exit 2; fi
jq -e '.status=="frozen_final_test" and .extraction_authorized==true and
  (.case_ids|length)==32 and .sampling_seeds==[101,202,303] and
  (.models|length)==8 and .freeze_receipt.selected_steps==20' "$PROTOCOL" >/dev/null || {
  echo 'FAIL frozen protocol contract'; exit 2;
}
# These checks include selected checkpoint files, training-source bindings, all
# generation entry sources and evaluator/native wrapper seals recorded at freeze.
# Exclude video paths before opening anything; image assets are also skipped.
while IFS=$'\t' read -r expected path; do
  check_file "$expected" "$path"
done < <(jq -r '[.inputs[], .execution_sources[], .models[].artifacts[],
  (.freeze_receipt.verified_file_shas | to_entries[] | {path:.key,sha256:.value})]
  | unique_by(.path,.sha256)[] | [.sha256,.path] | @tsv' "$PROTOCOL")

# Discover current-run metadata only. Do not select images, inspect models, or
# mistake development/runtime-check fixtures for terminal final-test outputs.
raw_count=0; source_count=0; index_count=0; stats_count=0; all_methods='[]'
while IFS= read -r path; do
  case "$path" in
    "$ROOT"/data_prepare/final_test_extractions/*/raw_frame_manifest.json)
      if jq -e --arg p "$EXPECTED_PROTOCOL" '.protocol_sha256==$p and .status=="raw_frames_extracted"' "$path" >/dev/null; then raw_count=$((raw_count+1)); fi ;;
    "$FINAL"/source_caches/*/source_cache_manifest.json)
      if jq -e --arg p "$EXPECTED_PROTOCOL" '.protocol_sha256==$p and (.status=="complete" or .status=="completed_with_failures")' "$path" >/dev/null; then source_count=$((source_count+1)); fi ;;
    "$FINAL"/generation_runs/*/generation_index.json|"$FINAL"/external_runs/*/generation_index.json)
      if jq -e --arg p "$EXPECTED_PROTOCOL" '.protocol_sha256==$p and (.status=="complete" or .status=="completed_with_failures") and .usable_for_evaluation!=false' "$path" >/dev/null; then
        index_count=$((index_count+1))
        all_methods=$(jq -cn --argjson a "$all_methods" --argjson b "$(jq -c '.method_names' "$path")" '$a+$b|unique')
      fi ;;
    "$FINAL"/*/statistics/paired_subject_results.json)
      # Final seal below must include this file and its paired input; result
      # presence alone is not treated as successful statistical validation.
      stats_count=$((stats_count+1)) ;;
  esac
done < <(rg --files "$ROOT/data_prepare" "$FINAL" -g raw_frame_manifest.json -g source_cache_manifest.json -g generation_index.json -g paired_subject_results.json)
[ "$raw_count" -gt 0 ] || wait_for 'raw extraction manifest'
[ "$source_count" -gt 0 ] || wait_for 'source cache manifest'
if ! jq -e --argjson names "$all_methods" '([.models[].method]|sort)==($names|sort)' "$PROTOCOL" >/dev/null; then wait_for 'terminal generation indexes covering all eight methods'; fi
[ "$stats_count" -gt 0 ] || wait_for 'final paired_subject_results.json'

# Standard sha256sum manifest, to be sealed by root AFTER final outputs exist.
# This script never creates it and never labels an unsealed result as verified.
SEAL="$HERE/SHA256SUMS"
if [ ! -f "$SEAL" ]; then
  wait_for 'delivery/SHA256SUMS final-artifact seal (not yet produced)'
else
  sealed_stats=0; sealed_input=0; sealed_metrics=0; seal_count=0
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    expected=${line:0:64}; path=${line:66}
    if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]] || [ "${line:64:2}" != '  ' ] || [ -z "$path" ]; then fail 'invalid SHA256SUMS row'; continue; fi
    case "$path" in /*|../*|*/../*) fail "seal path must be EXP-relative: $path"; continue;; esac
    case "$path" in *.json|*.md|*.csv|*.tsv|*.txt|*.log|*.py|*.sh) ;; *) fail "non-metadata final artifact excluded: $path"; continue;; esac
    check_file "$expected" "$path"; seal_count=$((seal_count+1))
    case "$path" in */paired_subject_results.json) sealed_stats=$((sealed_stats+1));;
      */paired_subject_input.json) sealed_input=$((sealed_input+1));;
      */all_case_seed_metrics.json) sealed_metrics=$((sealed_metrics+1));; esac
  done < "$SEAL"
  [ "$sealed_stats" -gt 0 ] && [ "$sealed_input" -gt 0 ] && [ "$sealed_metrics" -gt 0 ] || wait_for 'seal must include paired input/results and all_case_seed_metrics'
  [ "$seal_count" -gt 0 ] || wait_for 'SHA256SUMS has no artifacts'
fi
echo "SUMMARY checked=$checked skipped_media=$skipped_media failures=$failures pending=$pending raw_manifests=$raw_count source_caches=$source_count generation_indexes=$index_count statistical_outputs=$stats_count"
echo 'BOUNDARY image/video bytes skipped; no decoding, model execution, installs, writes, or algorithmic re-evaluation.'
if [ "$failures" -gt 0 ]; then exit 2; fi
if [ "$pending" -gt 0 ]; then exit 3; fi
echo 'PASS protocol/models/code and sealed final metadata; media-content integrity is not audited by this script.'
