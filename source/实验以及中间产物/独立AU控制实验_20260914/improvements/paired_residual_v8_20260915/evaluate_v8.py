#!/usr/bin/env python3
"""Dev8 paired-residual diagnostic; frozen v5 CPU evaluators, 120 logical rows.

metadata/validate/prepare and import use only the standard library. Models run
only in explicit openface/facenet/all phases after terminal generation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V5 = HERE.parent / 'quality_diagnosis_v5_20260915'
V5_CORE = V5 / 'evaluate_diagnostic.py'
V5_CORE_SHA = 'cb293b700e58e6a7321913c43577a82f8761bfb6c9e1d1181d534dcfd023f294'
V5_RUN = V5 / 'runs/source_init_202609151525'
V5_INPUTS = {
    'cases.json': '7daad59d5540d7b49243cb03da38f7aaed5d2c4294c8cb46ac7ce58b46d43145',
    'result.json': '4e92651f47abadfefdf0be2a91600544c1142cbbd99ebab411b2bea0b22177ce',
}
CONFIGS = {
    'paired_g025': {'gain': .25, 'strength': .75, 'au_scale': 1.0, 'ip_scale': .9, 'guidance_scale': 7.5},
    'paired_g050': {'gain': .5, 'strength': .75, 'au_scale': 1.0, 'ip_scale': .9, 'guidance_scale': 7.5},
    'paired_g100': {'gain': 1.0, 'strength': .75, 'au_scale': 1.0, 'ip_scale': .9, 'guidance_scale': 7.5},
}
SOLVER = 'explicit_first_order_rho_on_DEIS_query_grid'
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
METHODS = [method + '_' + config for config in CONFIGS
           for method in ('baseline', 'bidirectional')]
BASE_REFERENCES = ['source_real', 'target_real', 'baseline_text2img',
                   'bidirectional_text2img', 'vae_reconstruction']
IMG2IMG_REFERENCES = [f'{method}_img2img_s{strength}' for strength in ('055', '075')
                      for method in ('baseline', 'bidirectional')]
REFERENCES = BASE_REFERENCES + IMG2IMG_REFERENCES
CASE_KIND = 'v8_development_paired_cases'
RESULT_KIND = 'v8_paired_residual_diagnostic'


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive_v5_adapter():
    """Exact, SHA-gated metadata bindings only; no scoring/model rule edits."""
    raw = V5_CORE.read_bytes()
    require(sha_bytes(raw) == V5_CORE_SHA, 'Frozen v5 diagnostic adapter changed')
    source = raw.decode('utf-8')
    changes = {
        "METHODS = [f'{method}_img2img_s{strength}' for strength in ('035', '055', '075')\n"
        "           for method in ('baseline', 'bidirectional')]": 'METHODS = ' + repr(METHODS),
        "REFERENCES = ['source_real', 'target_real', 'baseline_text2img',\n"
        "              'bidirectional_text2img', 'vae_reconstruction']": 'REFERENCES = ' + repr(REFERENCES),
        "'v5_development_diagnostic_cases'": repr(CASE_KIND),
        "'v5_source_initialization_diagnostic'": repr(RESULT_KIND),
        "            if method == 'vae_reconstruction':\n"
        "                require(Path(ref['path']).is_relative_to(run), 'VAE reconstruction must belong to this run')\n":
            "            # VAE path+SHA bound to the frozen v5 run by reference_lineage.\n",
        "len(items) == 88, 'Duplicate logical record'": "len(items) == 120, 'Duplicate logical record'",
        "'logical_count': 88": "'logical_count': 120",
        '40 <= count <= 88': '72 <= count <= 120',
        "generation_methods = ['baseline_text2img', 'bidirectional_text2img'] + METHODS":
            "generation_methods = ['baseline_text2img', 'bidirectional_text2img'] + " + repr(IMG2IMG_REFERENCES) + ' + METHODS',
        "'logical_records': 88": "'logical_records': 120",
    }
    for old, new in changes.items():
        require(source.count(old) == 1, 'Ambiguous v8 metadata binding: ' + old)
        source = source.replace(old, new)
    # Count adaptation must never leak old upper-bound/summary constants.
    require(not any(isinstance(node, ast.Constant) and type(node.value) is int and node.value == 88
                    for node in ast.walk(ast.parse(source))), 'Stale v5 count remains')
    return source, [{'old': old, 'new': new} for old, new in changes.items()]


DERIVED_SOURCE, ADAPTATIONS = derive_v5_adapter()
CORE = types.ModuleType('v8_sha_pinned_v5_evaluator')
# The v5 adapter derives its output scope and prepared evaluator seal from this
# path; all its original dependency paths still resolve below the same EXP.
CORE.__file__ = str(Path(__file__).resolve())
exec(compile(DERIVED_SOURCE, str(V5_CORE), 'exec'), CORE.__dict__)
_check_policy = CORE.check_policy
_reference_lineage = CORE.reference_lineage
_validate = CORE.validate


def provenance():
    return {'v5_core': str(V5_CORE), 'v5_core_sha256': V5_CORE_SHA,
            'derived_adapter_sha256': sha_bytes(DERIVED_SOURCE.encode('utf-8')),
            'v8_entry_sha256': CORE.sha(__file__),
            'v5_run': str(V5_RUN), 'v5_input_sha256': V5_INPUTS,
            'adaptations': ADAPTATIONS,
            'entry_origin': 'v6 evaluate_v6.py metadata adaptation only',
            'entry_origin_sha256': 'e72173eb31f4101eb7e5b586e59cea5c3aae68ece44455c77fa211814e65ad67',
            'solver_declaration': SOLVER,
            'solver_validation_scope': 'metadata and trace/call-count consistency only; not an independent numerical solver proof',
            'scoring_model_selection_standardization_changed': False}


def check_policy():
    require(CORE.sha(V5_CORE) == V5_CORE_SHA, 'Frozen v5 diagnostic adapter changed')
    return _check_policy()


def frozen_cases():
    """Construct references only from the two hash-frozen v5 manifests."""
    check_policy()
    for name, digest in V5_INPUTS.items():
        require(CORE.sha(V5_RUN / name) == digest, 'Frozen v5 input changed: ' + name)
    cases = CORE.read(V5_RUN / 'cases.json')['cases']
    result = CORE.read(V5_RUN / 'result.json')
    require(result['status'] == 'complete' and result['test_data_used'] is False,
            'Frozen v5 run is not a completed dev diagnostic')
    records = {(row['case_id'], row['method']): row for row in result['records']}
    for case in cases:
        for method in IMG2IMG_REFERENCES:
            row = records[(case['case_id'], method)]
            require(row['status'] == 'complete', 'Missing complete v5 img2img reference')
            for key in ('subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5', 'prompt'):
                require(row[key] == case[key], 'Frozen v5 row/case mismatch: ' + key)
            case[method] = {'path': row['output'], 'sha256': row['sha256']}
    return cases


def reference_lineage(cases):
    original = {case['case_id']: case for case in frozen_cases()}
    require(len(cases) == 8 and {case['case_id'] for case in cases} == set(original),
            'v8 must use the exact frozen v5 dev8 case IDs')
    bindings = []
    for case in cases:
        old = original[case['case_id']]
        # All existing v5 case fields and old references must be unchanged;
        # additional mask/generation metadata is permitted on the v8 case.
        for key, value in old.items():
            require(case.get(key) == value, 'v8 differs from frozen v5 case field: ' + key)
        for method in REFERENCES:
            bindings.append({'case_id': case['case_id'], 'role': method, **old[method]})
    lineage = _reference_lineage(cases)
    lineage.update(v5_inputs={name: {'path': str(V5_RUN / name), 'sha256': digest}
                              for name, digest in V5_INPUTS.items()},
                   v5_reference_bindings=bindings, same_sha_different_path_allowed=False,
                   vae_reference='reused unchanged from frozen v5 run; not required in v8 run')
    return lineage


def validate_solver_metadata(run):
    """Check generator declarations/trace metadata, never call its solver."""
    run = CORE.scoped_path(run, HERE)
    paths = [run / 'configuration.json', run / 'runtime.json']
    hashes = {str(path): CORE.sha(path) for path in paths}
    config, runtime = [CORE.read(path) for path in paths]
    require(config.get('kind') == RESULT_KIND and config.get('test_data_used') is False and
            config.get('training_performed') is False, 'Invalid paired generator declaration')
    require(config.get('solver') == SOLVER and config.get('scheduler_step_calls') == 0 and
            config.get('final_rho') == 0, 'Wrong paired first-order solver declaration')
    require(config.get('guidance_scale') == 7.5 and config.get('au_scale') == 1 and config.get('ip_scale') == .9,
            'Frozen CFG/AU/IP scale changed')
    require(config.get('target_pixels_supplied') is False and config.get('post_generation_pixel_blend') is False,
            'Generation input/output boundary changed')
    require(runtime.get('scheduler') == 'DEISMultistepScheduler_grid_only' and
            runtime.get('scheduler_step_calls') == 0, 'DEIS grid must not be declared multistep execution')
    levels = runtime['time_levels']
    require(len(levels) == 15 and [level['t'] for level in levels] == TIMESTEPS, 'Paired query-grid mismatch')
    for index, level in enumerate(levels):
        next_t = TIMESTEPS[index + 1] if index < 14 else None
        require(level['next_t'] == next_t, 'Paired next timestep mismatch')
        require(all(isinstance(level[key], (int, float)) and CORE.math.isfinite(level[key])
                    for key in ('alpha', 'sigma', 'rho', 'next_alpha', 'next_sigma', 'next_rho')),
                'Nonfinite paired level metadata')
        require(level['rho'] > level['next_rho'] >= 0, 'Nondecreasing paired rho metadata')
        if index < 14:
            require(level['next_rho'] == levels[index + 1]['rho'], 'Adjacent paired rho mismatch')
    require(levels[-1]['next_rho'] == 0 and levels[-1]['next_alpha'] == 1 and levels[-1]['next_sigma'] == 0,
            'Missing explicit clean endpoint')
    result = CORE.read(run / 'result.json')
    binding = result['configuration']
    require(Path(binding['path']).resolve() == paths[0] and binding['sha256'] == hashes[str(paths[0])],
            'Generation configuration binding changed')
    require(all(CORE.sha(path) == digest for path, digest in hashes.items()), 'Solver declarations changed during validation')
    return {'solver': SOLVER, 'scheduler_step_calls': 0, 'configuration_runtime_hashes': hashes,
            'time_levels': levels, 'guidance_scale': 7.5, 'au_scale': 1.0, 'ip_scale': .9,
            'residual_gain_is_not_au_scale': True, 'validation_scope': 'metadata only; scoring unchanged'}


def validate(run):
    document = _validate(run)
    for item in document['items']:
        if item['role'] != 'new_generation':
            continue
        row = item['generation_record']
        config = row['method'].split('_', 1)[1]
        require(row.get('config') == config, 'Generated config/method mismatch')
        require(all(row.get(key) == value for key, value in CONFIGS[config].items()),
                'Generated config parameters mismatch')
        require(row.get('nominal_steps') == 20, 'Expected frozen 20 nominal steps')
        if row['status'] == 'complete':
            require(row.get('actual_steps') == 15 and row.get('actual_timesteps') == TIMESTEPS,
                    'Unexpected paired actual query points')
            require(row.get('source_calls') + row.get('source_cache_reuses') == 15 and
                    row.get('target_calls') == row.get('source_prediction_requests') == 15 and
                    row.get('scheduler_step_calls') == 0, 'Paired branch call-count metadata differs')
            trace = row['step_trace']
            require(len(trace) == 15 and [step['t'] for step in trace] == TIMESTEPS and
                    trace[-1]['next_t'] is None and trace[-1]['next_rho'] == 0,
                    'Paired trace is incomplete or missing the clean endpoint')
            require(row.get('finite') is True and all(step.get('finite') is True for step in trace),
                    'Successful paired run must retain finite trace declarations')
    document['paired_solver_metadata'] = validate_solver_metadata(run)
    document['adapter_provenance'] = provenance()
    document['configs'] = CONFIGS
    return document


CORE.check_policy = check_policy
CORE.reference_lineage = reference_lineage
CORE.validate = validate


def metadata():
    """Verify historical refs and evaluator code; never import model runtimes."""
    cases = frozen_cases()
    lineage = reference_lineage(cases)
    for case in cases:
        for method in REFERENCES:
            CORE.checked_image(case[method])
    return {'status': 'metadata_verified_no_models_run', 'test_used': False,
            'case_count': len(cases), 'reference_count': 72, 'new_generated_planned': 48,
            'logical_records': 120, 'case_kind': CASE_KIND, 'result_kind': RESULT_KIND,
            'methods': METHODS, 'references': REFERENCES, 'configs': CONFIGS,
            'solver': SOLVER, 'actual_query_points': TIMESTEPS, 'residual_gain_is_not_au_scale': True,
            'original_metric_columns': CORE.METRICS, 'policy_sha256': CORE.POLICY_SHA,
            'reference_lineage': lineage, 'adapter_provenance': provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('metadata', 'validate', 'prepare', 'openface', 'facenet', 'merge', 'all'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase in ('metadata', 'validate'):
        if args.phase == 'validate':
            require(args.run is not None, '--run required')
            document = validate(args.run)
        else:
            document = metadata()
        print(json.dumps({key: value for key, value in document.items() if key != 'items'}, ensure_ascii=False))
        return
    require(args.output is not None, '--output required; fresh directory below v8 diagnostic')
    if args.phase in ('prepare', 'all'):
        require(args.run is not None, '--run required')
        CORE.prepare(args.run, args.output)
    if args.phase in ('openface', 'all'):
        CORE.openface(args.output)
    if args.phase in ('facenet', 'all'):
        CORE.facenet(args.output)
    if args.phase in ('merge', 'all'):
        CORE.merge(args.output)


if __name__ == '__main__':
    main()

