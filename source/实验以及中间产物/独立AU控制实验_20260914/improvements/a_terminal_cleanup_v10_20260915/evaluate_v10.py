#!/usr/bin/env python3
"""Frozen CPU metrics for v10; audit 15 DEIS calls plus 0/1 terminal UNet call."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V9_SOURCE = HERE.parent / 'a_texture_schedule_v9_20260915/evaluate_v9.py'
V9_SOURCE_SHA = '74945dcbf11ef6fa998c8049c47e997b421e13c5d48270798416bd6a47fb2f12'
V9_CASES = HERE.parent / 'a_texture_schedule_v9_20260915/runs/late_au_202609151925/cases.json'
V9_CASES_SHA = '2f7221ac4df6e909eb18af4cd313566e0422a82591310c9715bb904b98224c94'
CONFIGS = {name: {'strength': .75, 'au_scale': 1.5, 'terminal_unet_calls': calls}
           for name, calls in [('terminal_last_eps', 0), ('terminal_t0', 1), ('terminal_t0_lp', 1)]}
CASE_KIND = 'v10_development_terminal_cases'
RESULT_KIND = 'v10_terminal_diagnostic'
FILTER = {'sigma_latent': .5, 'radius': 2, 'kernel_size': 5, 'padding': 'reflect',
          'normalization': 'conv(mask*delta)/conv(mask)', 'denominator_epsilon': 1e-8,
          'outside_mask_zero': True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive_v9():
    raw = V9_SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == V9_SOURCE_SHA, 'Frozen v9 evaluator changed')
    source = raw.decode('utf-8')
    old = "CONFIGS = {\n    name: {'strength': .75, 'au_scale_initial': 1.5, 'au_scale_late': late,\n           'au_late_max_t': 250}\n    for name, late in [('late_au100', 1.0), ('late_au050', .5), ('late_au000', 0.0)]\n}"
    replacements = {old: 'CONFIGS = ' + repr(CONFIGS),
                    "CASE_KIND = 'v9_development_late_au_cases'": 'CASE_KIND = ' + repr(CASE_KIND),
                    "RESULT_KIND = 'v9_late_au_diagnostic'": 'RESULT_KIND = ' + repr(RESULT_KIND)}
    for old, new in replacements.items():
        require(source.count(old) == 1, 'Ambiguous v10 metadata replacement: ' + old)
        source = source.replace(old, new)
    return source, [{'old': old, 'new': new} for old, new in replacements.items()]


DERIVED_V9_SOURCE, ADAPTATIONS = derive_v9()
BASE = types.ModuleType('v10_sha_pinned_v9_metadata')
BASE.__file__ = str(Path(__file__).resolve())
exec(compile(DERIVED_V9_SOURCE, str(V9_SOURCE), 'exec'), BASE.__dict__)
CORE = BASE.CORE
METHODS, REFERENCES, TIMESTEPS = BASE.METHODS, BASE.REFERENCES, BASE.TIMESTEPS
_base_policy = BASE.check_policy
_base_lineage = BASE.reference_lineage
_base_provenance = BASE.provenance
_base_metadata = BASE.metadata
# This primitive is the same frozen v5 validator, before v6/v9 generation-metadata guards.
_primitive_validate = BASE.ADAPTER._validate
_frozen_merge = BASE._base_merge


def check_policy():
    require(CORE.sha(V9_SOURCE) == V9_SOURCE_SHA, 'Frozen v9 evaluator changed')
    return _base_policy()


def reference_lineage(cases):
    lineage = _base_lineage(cases)
    require(CORE.sha(V9_CASES) == V9_CASES_SHA, 'Frozen v9 cases changed')
    original = CORE.read(V9_CASES)
    require(original['kind'] == 'v9_development_late_au_cases' and original['test_data_used'] is False,
            'V9 case source is not dev-only')
    mapping = {case['case_id']: case for case in original['cases']}
    require(len(cases) == len(mapping) == 8 and {case['case_id'] for case in cases} == set(mapping),
            'V10 must preserve exact v9 dev8 cases')
    for case in cases:
        for key, value in mapping[case['case_id']].items():
            require(case.get(key) == value, 'V10 differs from frozen v9 case field: ' + key)
    lineage.update(v9_case_manifest={'path': str(V9_CASES), 'sha256': V9_CASES_SHA},
                   all_old_v9_case_fields_preserved=True)
    return lineage


def provenance():
    previous = _base_provenance()
    previous.pop('v9_entry_sha256', None)
    return {**previous, 'v9_frozen_entry': str(V9_SOURCE), 'v9_frozen_entry_sha256': V9_SOURCE_SHA,
            'v10_entry_sha256': CORE.sha(__file__), 'v10_metadata_adaptations': ADAPTATIONS,
            'v10_derived_v9_sha256': hashlib.sha256(DERIVED_V9_SOURCE.encode()).hexdigest(),
            'metric_model_selection_standardization_changed': False}


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def audit_terminal(row, old):
    config = row['config']
    extra = CONFIGS[config]['terminal_unet_calls']
    for key in ('source_latent_sha256', 'noise_sha256'):
        require(row.get(key) == old[key], 'Generated A binding changed: ' + key)
    require(row.get('actual_timesteps') == TIMESTEPS, 'DEIS timesteps differ from A')
    counts = {'actual_steps': 15, 'denoising_steps': 15, 'deis_unet_calls': 15,
              'terminal_unet_calls': extra, 'total_unet_calls': 15 + extra}
    require(all(type(row.get(key)) is int and row[key] == value for key, value in counts.items()),
            'Actual denoising/terminal/total UNet counts disagree')
    trace = row.get('unet_call_trace')
    require(isinstance(trace, list) and len(trace) == 15 + extra, 'Missing actual UNet calls')
    for index, entry in enumerate(trace):
        require(isinstance(entry, dict), 'UNet trace entry must be a record')
        require(entry.get('phase') == ('deis' if index < 15 else 'terminal') and
                entry.get('t') == (TIMESTEPS[index] if index < 15 else 0), 'UNet phase/timestep mismatch')
        require(entry.get('au_scale') == 1.5 and entry.get('cfg_batch') == 2,
                'UNet AU scale or CFG batch differs from A')
        require(valid_sha(entry.get('raw_epsilon_sha256')) and valid_sha(entry.get('prompt_embeds_sha256')),
                'Missing raw epsilon or prompt tensor hash')
    require(len({entry['prompt_embeds_sha256'] for entry in trace}) == 1, 'Prompt embeddings change within generation')
    terminal = row.get('terminal_trace')
    require(isinstance(terminal, dict) and terminal.get('mode') == config, 'Wrong terminal mode')
    require(terminal.get('last_deis_timestep') == 50 and
            terminal.get('epsilon_source') == ('last_t50' if extra == 0 else 'extra_t0'),
            'Wrong terminal epsilon source')
    require(terminal.get('scheduler_step_index_before') == terminal.get('scheduler_step_index_after') == 20,
            'Terminal cleanup must not call or advance scheduler.step')
    require(terminal.get('extra_cfg') == (7.5 if extra else None), 'Terminal CFG mismatch')
    require(terminal.get('filter') == (FILTER if config == 'terminal_t0_lp' else None), 'Terminal filter changed')
    for key in ('rho0', 'alpha0', 'sigma0'):
        require(type(terminal.get(key)) in (int, float) and math.isfinite(terminal[key]),
                'Invalid terminal coefficient: ' + key)
    require(terminal['alpha0'] > 0 and terminal['sigma0'] >= 0 and terminal['rho0'] >= 0 and
            math.isclose(terminal['rho0'], terminal['sigma0'] / terminal['alpha0'], rel_tol=1e-6, abs_tol=1e-8),
            'Terminal coefficient relationship inconsistent')
    for key in ('input_latent_sha256', 'epsilon_cfg_sha256', 'last_deis_raw_epsilon_sha256',
                'clean_latent_sha256', 'post_filter_clean_latent_sha256', 'projected_latent_sha256'):
        require(valid_sha(terminal.get(key)), 'Missing terminal tensor hash: ' + key)
    require(terminal['last_deis_raw_epsilon_sha256'] == trace[14]['raw_epsilon_sha256'],
            'Last DEIS epsilon hash differs from actual call trace')
    require(terminal.get('terminal_raw_epsilon_sha256') == (trace[-1]['raw_epsilon_sha256'] if extra else None),
            'Terminal epsilon hash differs from actual call trace')
    if config != 'terminal_t0_lp':
        require(terminal['post_filter_clean_latent_sha256'] == terminal['clean_latent_sha256'],
                'Unexpected filter for an unfiltered candidate')
    projection = row.get('projection_trace')
    require(isinstance(projection, list) and len(projection) == 15 and
            all(entry.get('t') == t for entry, t in zip(projection, TIMESTEPS)), 'Invalid projection trace')
    require(projection[:14] == old['projection_trace'][:14], 'Pre-terminal projected DEIS states differ from A')
    require(projection[-1].get('next_t') is None, 'Final projection must be terminal')
    require(projection[-1].get('latent_sha256') == terminal['projected_latent_sha256'],
            'Final projection does not bind terminal cleanup output')
    if row['group'] == 'no_edit':
        require(terminal['projected_latent_sha256'] == row['source_latent_sha256'],
                'No-edit projection must restore the exact source latent')
    return {'case_id': row['case_id'], 'method': row['method'], 'config': config, **counts,
            'a_condition_and_noise_preserved': True, 'scheduler_advanced_by_cleanup': False}


def validate(run):
    document = _primitive_validate(run)
    configuration = BASE.generation_configuration(run, document)
    for key, value in {'denoising_steps': 15, 'scheduler_sigmas_mutated': False,
                       'terminal_scheduler_step_calls': 0, 'terminal_add_noise_calls': 0,
                       'terminal_filter': FILTER}.items():
        require(configuration.get(key) == value, 'Terminal configuration declaration changed: ' + key)
    _, old_records = BASE.frozen_v7()
    audit = []
    for item in document['items']:
        if item['role'] != 'new_generation':
            continue
        row = item['generation_record']
        family, config = row['method'].split('_', 1)
        require(row.get('config') == config, 'Generated method/config mismatch')
        require(row.get('strength') == .75 and row.get('au_scale') == 1.5 and row.get('ip_scale') == .9 and
                row.get('nominal_steps') == 20, 'Generated A config changed')
        old = old_records[(row['case_id'], family + '_smooth_s075_a150')]
        for key in ('condition_hashes', 'region_mask', 'prompt'):
            require(row.get(key) == old[key], 'Generated A binding changed: ' + key)
        if row['status'] == 'complete':
            audit.append(audit_terminal(row, old))
    document.update(adapter_provenance=provenance(), configs=CONFIGS, terminal_call_audit=audit,
                    fair_baseline_pairs={config: ['baseline_' + config, 'bidirectional_' + config] for config in CONFIGS},
                    compute_budget_note='15 DEIS UNet calls plus 0/1 explicitly disclosed terminal call; total 15/16. actual_steps means denoising steps only.')
    return document


def metadata():
    report = _base_metadata()
    report.update(adapter_provenance=provenance(), terminal_counts=CONFIGS,
                  compute_budget_note='15 DEIS calls, plus terminal_last_eps:0 / terminal_t0:1 / terminal_t0_lp:1; total 15/16.')
    return report


def merge(output):
    _frozen_merge(output)
    output = CORE.scoped_path(output, HERE)
    report = CORE.read(output / 'comparison/results.json')
    stats = BASE.supplemental_statistics(report)
    stats.update(frozen_comparison_sha256=CORE.sha(output / 'comparison/results.json'),
                 adapter_provenance=provenance(), terminal_counts=CONFIGS,
                 compute_budget_note='Same terminal operation within each baseline/bidirectional pair; 15 versus 16 total UNet calls across candidate configurations is disclosed.')
    CORE.save(output / 'comparison/v10_statistics.json', stats)


BASE.check_policy = BASE.ADAPTER.check_policy = CORE.check_policy = check_policy
BASE.reference_lineage = BASE.ADAPTER.reference_lineage = CORE.reference_lineage = reference_lineage
BASE.provenance = BASE.ADAPTER.provenance = provenance
CORE.validate = validate
CORE.merge = merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('metadata', 'validate', 'prepare', 'openface', 'facenet', 'merge', 'all'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase in ('metadata', 'validate'):
        require(args.phase != 'validate' or args.run is not None, '--run required')
        report = metadata() if args.phase == 'metadata' else validate(args.run)
        print(json.dumps({key: value for key, value in report.items() if key != 'items'}, ensure_ascii=False))
        return
    require(args.output is not None, 'Fresh v10 --output required')
    if args.phase in ('prepare', 'all'):
        require(args.run is not None, '--run required')
        CORE.prepare(args.run, args.output)
    if args.phase in ('openface', 'all'):
        CORE.openface(args.output)
    if args.phase in ('facenet', 'all'):
        CORE.facenet(args.output)
    if args.phase in ('merge', 'all'):
        merge(args.output)


if __name__ == '__main__':
    main()
