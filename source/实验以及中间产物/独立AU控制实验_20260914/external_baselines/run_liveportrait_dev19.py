#!/usr/bin/env python3
"""Fresh CPU execution of every frozen raw dev19 request, with per-case motion audit."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

from prepare_dev_contracts import HERE, OLD, REPOS, sha, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    assert document['schema_version'] == 3 and len(document['cases']) == 19 and not document['test_used']
    assert document['old_12_rows_exactly_preserved'] and document['sampling_seeds'] == [101]
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      NO_ALBUMENTATIONS_UPDATE='1', MPLCONFIGDIR=str(HERE / 'cache/matplotlib'),
                      XDG_CACHE_HOME=str(HERE / 'cache/xdg'), TORCH_HOME=str(HERE / 'cache/torch'),
                      PYDANTIC_DISABLE_PLUGINS='__all__', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    runtime = HERE / 'runtime'
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location('liveportrait_native_dev19', runtime / 'run_liveportrait.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    base = runtime / ('runs/liveportrait_dev19_raw_native_' + datetime.now().strftime('%Y%m%d%H%M'))
    out = base
    index = 2
    while out.exists():
        out = base.with_name(base.name + f'-{index:02d}')
        index += 1
    active = {'case_id': None, 'events': []}
    probes = []
    native_initialize, native_execute = wrapper.initialize, wrapper.execute_case
    def initialize(*values, **kwargs):
        import torch
        torch.set_num_threads(2)
        pipeline, detector, base_args, torch, metadata = native_initialize(*values, **kwargs)
        live = pipeline.live_portrait_wrapper
        original = live.warp_decode
        def observed(feature, source, driving):
            delta = (driving - source).detach().float().cpu()
            active['events'].append({'case_id': active['case_id'], 'source_shape': list(source.shape), 'driving_shape': list(driving.shape),
                                     'exact_equal': bool(torch.equal(source, driving)), 'delta_rms': float(delta.square().mean().sqrt()),
                                     'delta_max_abs': float(delta.abs().max()), 'changed_scalar_count': int(torch.count_nonzero(delta)),
                                     'scalar_count': delta.numel()})
            return original(feature, source, driving)
        live.warp_decode = observed
        return pipeline, detector, base_args, torch, metadata
    def execute(row, output, *values, **kwargs):
        active['case_id'], active['events'] = row['case_id'], []
        result = native_execute(row, output, *values, **kwargs)
        probe = {'case_id': row['case_id'], 'request_case_id': row['request_case_id'], 'group': row['group'],
                 'source_sha256': row['source_sha256'], 'driver_sha256': row['target_sha256'], 'status': result['status'],
                 'warp_decode_observations': active['events'][:], 'observer_changes_tensors': False, 'test_used': False}
        if result['status'] == 'success':
            import numpy as np
            from PIL import Image
            source = np.asarray(Image.open(row['source']).convert('RGB'))
            output_image = np.asarray(Image.open(result['output']).convert('RGB'))
            assert source.shape == output_image.shape
            probe.update(output=result['output'], output_sha256=result['output_sha256'],
                         source_pixel_copy=bool(np.array_equal(source, output_image)),
                         pixel_mean_absolute_difference=float(np.abs(output_image.astype(float) - source.astype(float)).mean()))
        save(output / 'cases' / row['case_id'] / 'motion_probe.json', probe)
        probes.append(probe)
        return result
    wrapper.initialize, wrapper.execute_case = initialize, execute
    command = ['--repo', str(REPOS['liveportrait']), '--checkpoint', str(OLD / 'models/liveportrait_author'),
               '--manifest', str(args.manifest), '--output', str(out), '--device', 'cpu',
               '--detection-policy', 'author_default', '--execute']
    code = wrapper.main(command)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['case_count'] == summary['completed_count'] == len(probes) == 19
    save(out / 'motion_summary.json', {'case_count': 19, 'records': probes, 'manifest_sha256': sha(args.manifest),
              'source_sha256': sha(__file__), 'all_fresh_execution': True, 'prior_single_case_reused': False,
              'device': 'cpu', 'test_used': False, 'returncode': code})
    (out / 'snapshots/run_liveportrait_dev19.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(out), 'case_count': 19, 'success_count': summary['success_count'],
                      'failed_count': summary['failed_count'], 'motion_probes': len(probes), 'device': 'cpu'}, ensure_ascii=False), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
