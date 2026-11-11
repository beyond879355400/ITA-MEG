#!/usr/bin/env python3
"""Verify the single-option development rerun without inference or score selection."""
import argparse
import json
from pathlib import Path
import struct

from prepare_dev_contracts import HERE, sha
from run_liveportrait_dev19_absolute import OLD_CONFIG, OLD_CONFIG_SHA, CORE_SHA, MANIFEST_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    binding = json.loads((root / 'single_factor_binding.json').read_text())
    old = json.loads(OLD_CONFIG.read_text())
    new = json.loads((root / 'configuration.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    motion = json.loads((root / 'motion_summary.json').read_text())
    assert sha(OLD_CONFIG) == OLD_CONFIG_SHA
    assert binding['returncode'] == 0 and binding['generation_executed'] and not binding['test_used']
    for key in ['wrapper_sha256', 'helper_sha256', 'manifest_sha256', 'asset_manifest_sha256',
                'model_files', 'source_files', 'cases', 'upstream_commit', 'model_revision']:
        assert new[key] == old[key], ('Unexpected non-option change', key)
    old_parameters = old['policy']['effective_author_cli_parameters']
    new_parameters = new['policy']['effective_author_cli_parameters']
    assert [k for k in old_parameters if old_parameters[k] != new_parameters[k]] == ['flag_relative_motion']
    assert old_parameters['flag_relative_motion'] is True and new_parameters['flag_relative_motion'] is False
    assert new_parameters['animation_region'] == 'all' and new['policy']['device'] == 'cpu'
    assert sha(root / 'snapshots/run_liveportrait_dev19_core.py') == CORE_SHA
    assert sha(root / 'snapshots/run_liveportrait_dev19_absolute.py') == binding['entry_sha256']
    assert sha(root / 'snapshots/derived_runner.py') == binding['derived_source_sha256']
    assert motion['manifest_sha256'] == MANIFEST_SHA and motion['all_fresh_execution']
    assert not motion['prior_single_case_reused'] and motion['device'] == 'cpu'
    assert summary['case_count'] == summary['completed_count'] == len(summary['results']) == len(motion['records']) == 19
    cases = {r['case_id']: r for r in new['cases']}
    assert len(cases) == 19
    unique_outputs = set()
    for row, probe in zip(summary['results'], motion['records']):
        case = cases[row['case_id']]
        assert row['case_id'] == probe['case_id'] and probe['source_sha256'] == case['source_sha256']
        assert probe['driver_sha256'] == case['target_sha256'] and not probe['observer_changes_tensors']
        assert case['seed'] == 101 and case['split'] == 'dev'
        assert sha(case['source']) == case['source_sha256'] and sha(case['target']) == case['target_sha256']
        if row['status'] == 'success':
            assert Path(row['output']).is_relative_to(root) and row['output'] not in unique_outputs
            unique_outputs.add(row['output'])
            assert sha(row['output']) == row['output_sha256'] == probe['output_sha256']
            header = Path(row['output']).read_bytes()[:24]
            assert header[:8] == b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II', header[16:24]) == (case['source_width'], case['source_height'])
            assert len(probe['warp_decode_observations']) == 1
    print(json.dumps({'status': 'PASS', 'only_generation_option_changed': 'flag_relative_motion',
                      'case_count': 19, 'success_count': summary['success_count'],
                      'failed_count': summary['failed_count'], 'all_inputs_and_models_unchanged': True,
                      'fresh_unique_outputs': len(unique_outputs), 'old_configuration_unchanged': True,
                      'source_pixel_copies': sum(p.get('source_pixel_copy', False) for p in motion['records']),
                      'generation_seconds_sum': sum(r['elapsed_seconds'] for r in summary['results']),
                      'test_used': False}))


if __name__ == '__main__':
    main()
