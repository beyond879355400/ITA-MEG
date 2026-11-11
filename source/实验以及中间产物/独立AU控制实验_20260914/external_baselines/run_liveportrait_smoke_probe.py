#!/usr/bin/env python3
"""One CPU native image-driven case, with read-only motion observations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from prepare_dev_contracts import HERE, OLD, REPOS, sha, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    assert document['schema_version'] == 2 and not document['test_used']
    case = next(row for row in document['cases'] if row['group'] != 'no_edit')
    assert case['seed'] == 101 and case['split'] == 'dev'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1', HF_HUB_OFFLINE='1',
                      MPLCONFIGDIR=str(HERE / 'cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'cache/xdg'),
                      TORCH_HOME=str(HERE / 'cache/torch'), PYDANTIC_DISABLE_PLUGINS='__all__')
    sys.dont_write_bytecode = True
    runtime = HERE / 'runtime'
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location('current_native_liveportrait_wrapper', runtime / 'run_liveportrait.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    native_initialize = wrapper.initialize
    observations = []
    actual_parameters = {}
    def initialize(*values, **kwargs):
        import torch
        torch.set_num_threads(2)
        pipeline, detector, base_args, torch, metadata = native_initialize(*values, **kwargs)
        live = pipeline.live_portrait_wrapper
        cfg = live.inference_cfg
        for key in ('flag_relative_motion', 'animation_region', 'driving_option', 'flag_eye_retargeting',
                    'flag_lip_retargeting', 'flag_stitching', 'driving_multiplier'):
            actual_parameters[key] = getattr(cfg, key)
        actual_parameters['lip_array_shape'] = list(cfg.lip_array.shape)
        actual_parameters['lip_array_sha256'] = __import__('hashlib').sha256(cfg.lip_array.tobytes()).hexdigest()
        original = live.warp_decode
        def observed(feature_3d, kp_source, kp_driving):
            delta = (kp_driving - kp_source).detach().float().cpu()
            observations.append({'kp_source_shape': list(kp_source.shape), 'kp_driving_shape': list(kp_driving.shape),
                                 'source_to_driving_exact_equal': bool(torch.equal(kp_source, kp_driving)),
                                 'delta_rms': float(delta.square().mean().sqrt()), 'delta_max_abs': float(delta.abs().max()),
                                 'changed_scalar_count': int(torch.count_nonzero(delta)), 'total_scalar_count': delta.numel()})
            return original(feature_3d, kp_source, kp_driving)
        live.warp_decode = observed
        return pipeline, detector, base_args, torch, metadata
    wrapper.initialize = initialize
    argv = ['--repo', str(REPOS['liveportrait']), '--checkpoint', str(OLD / 'models/liveportrait_author'),
            '--manifest', str(args.manifest), '--case-id', case['case_id'], '--output', str(args.output),
            '--device', 'cpu', '--detection-policy', 'author_default', '--execute']
    code = wrapper.main(argv)
    summary = json.loads((args.output / 'summary.json').read_text())
    record = summary['results'][0]
    probe = {'case_id': case['case_id'], 'returncode': code, 'native_parameters': actual_parameters,
             'warp_decode_observations': observations, 'observer_changes_model_tensors': False,
             'observer_source_sha256': sha(__file__), 'native_pipeline_sha256': sha(REPOS['liveportrait'] / 'src/live_portrait_pipeline.py'),
             'mode_expected': 'source PNG + driving PNG; flag_is_driving_video=False', 'test_used': False}
    if record['status'] == 'success':
        import numpy as np
        from PIL import Image
        source = np.asarray(Image.open(case['source']).convert('RGB'))
        output = np.asarray(Image.open(record['output']).convert('RGB'))
        assert source.shape == output.shape
        probe.update(output=record['output'], output_sha256=sha(record['output']), source_sha256=case['source_sha256'],
                     pixel_exact_source_copy=bool(np.array_equal(source, output)),
                     pixel_mean_absolute_difference=float(np.abs(output.astype(float) - source.astype(float)).mean()),
                     pixel_changed_count=int(np.count_nonzero(output != source)))
    save(args.output / 'motion_probe.json', probe)
    (args.output / 'snapshots/run_liveportrait_smoke_probe.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(probe, ensure_ascii=False), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
