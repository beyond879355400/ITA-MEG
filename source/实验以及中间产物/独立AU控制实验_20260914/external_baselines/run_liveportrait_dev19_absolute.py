#!/usr/bin/env python3
"""One native option only: relative=False for every frozen dev19 image pair."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

from prepare_dev_contracts import HERE, REPOS, sha, save

CORE = HERE / 'run_liveportrait_dev19.py'
CORE_SHA = '9d76419edcc647509dbc731caa84d4ac4653d77fff67cd456ea05e059d6c0a28'
MANIFEST_SHA = 'bbb12e349ea149cfd51fb8eb547c37a772b5b94157029390b640a0433052bbb4'
OLD_CONFIG = HERE / 'runtime/runs/liveportrait_dev19_raw_native_202609142041/configuration.json'
OLD_CONFIG_SHA = '4c36a6e3e18155274d0802163fd97847fb19f7c542bd8f1031611d284111177a'
WRAPPER_SHA = '7d227e268f0e00c079d9b4e84609c72896a6867d8b57922a7660378fa0a6a558'
PREFIX = 'liveportrait_dev19_raw_absolute_'


def expected_parameters():
    assert sha(OLD_CONFIG) == OLD_CONFIG_SHA
    old = json.loads(OLD_CONFIG.read_text())['policy']['effective_author_cli_parameters']
    assert old['flag_relative_motion'] is True and old['animation_region'] == 'all'
    new = dict(old, flag_relative_motion=False)
    assert [k for k in old if old[k] != new[k]] == ['flag_relative_motion']
    return old, new


def bind_absolute_parameters(wrapper):
    old, new = expected_parameters()
    native_parameters, native_initialize = wrapper.author_parameters, wrapper.initialize
    def parameters(repo, device='mps'):
        assert device == 'cpu'
        actual = native_parameters(repo, device)
        assert actual == old, 'Native defaults changed since the relative dev19 run'
        return dict(new)
    def initialize(*args, **kwargs):
        pipeline, detector, base_args, torch, metadata = native_initialize(*args, **kwargs)
        cfg = pipeline.live_portrait_wrapper.inference_cfg
        for key in ('flag_relative_motion', 'animation_region', 'flag_stitching', 'driving_multiplier',
                    'flag_eye_retargeting', 'flag_lip_retargeting', 'flag_normalize_lip', 'flag_pasteback'):
            assert getattr(cfg, key) == new[key], ('Effective native setting mismatch', key)
        assert pipeline.live_portrait_wrapper.device == 'cpu'
        metadata['single_native_option_override'] = {'flag_relative_motion': {'old': True, 'new': False}}
        metadata['driver_pose_available'] = True
        return pipeline, detector, base_args, torch, metadata
    wrapper.author_parameters, wrapper.initialize = parameters, initialize


def derive_source():
    assert sha(CORE) == CORE_SHA
    source = CORE.read_text()
    edits = [
        ('    spec.loader.exec_module(wrapper)\n', '    spec.loader.exec_module(wrapper)\n    bind_absolute_parameters(wrapper)\n'),
        ('runs/liveportrait_dev19_raw_native_', 'runs/' + PREFIX),
        ('snapshots/run_liveportrait_dev19.py', 'snapshots/run_liveportrait_dev19_absolute.py'),
    ]
    for old, new in edits:
        assert source.count(old) == 1, ('Pinned core adaptation ambiguous', old)
        source = source.replace(old, new)
    ast.parse(source)
    return source


def preflight(manifest):
    assert sha(manifest) == MANIFEST_SHA
    doc = json.loads(manifest.read_text())
    assert len(doc['cases']) == 19 and doc['sampling_seeds'] == [101] and not doc['test_used']
    assert all(r['split'] == 'dev' and r['seed'] == 101 for r in doc['cases'])
    assert sha(HERE / 'runtime/run_liveportrait.py') == WRAPPER_SHA
    sys.path.insert(0, str(HERE / 'runtime'))
    spec = importlib.util.spec_from_file_location('absolute_preflight_native', HERE / 'runtime/run_liveportrait.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.git_check(REPOS['liveportrait'])
    old, new = expected_parameters()
    assert wrapper.author_parameters(REPOS['liveportrait'], 'cpu') == old
    bind_absolute_parameters(wrapper)
    assert wrapper.author_parameters(REPOS['liveportrait'], 'cpu') == new
    derived = derive_source()
    return {'status': 'PASS', 'case_count': 19, 'manifest_sha256': sha(manifest),
            'old_configuration_sha256': OLD_CONFIG_SHA, 'core_runner_sha256': CORE_SHA,
            'wrapper_sha256': WRAPPER_SHA, 'entry_sha256': sha(__file__),
            'only_parameter_change': {'flag_relative_motion': {'old': True, 'new': False}},
            'unchanged_parameters': {k: v for k, v in old.items() if k != 'flag_relative_motion'},
            'new_effective_parameters': new, 'test_used': False, 'generation_executed': False,
            'input_group': 'source_plus_real_driver',
            'native_absolute_all_behavior': 'driver expression, rotation and translation; source scale retained',
            'derived_source_sha256': __import__('hashlib').sha256(derived.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    check = preflight(args.manifest)
    if args.preflight:
        print(json.dumps(check, ensure_ascii=False), flush=True)
        return
    source = derive_source()
    before = set((HERE / 'runtime/runs').glob(PREFIX + '*'))
    namespace = {'__name__': 'pinned_dev19_absolute_runner', '__file__': str(Path(__file__).resolve()),
                 'bind_absolute_parameters': bind_absolute_parameters}
    exec(compile(source, str(CORE), 'exec'), namespace)
    sys.argv = [str(Path(__file__).resolve()), '--manifest', str(args.manifest)]
    code = 1
    try:
        namespace['main']()
        code = 0
    except SystemExit as exc:
        code = exc.code
    finally:
        created = set((HERE / 'runtime/runs').glob(PREFIX + '*')) - before
        if len(created) == 1:
            out = created.pop()
            (out / 'snapshots').mkdir(exist_ok=True)
            (out / 'snapshots/run_liveportrait_dev19_core.py').write_bytes(CORE.read_bytes())
            (out / 'snapshots/derived_runner.py').write_text(source)
            (out / 'snapshots/run_liveportrait_dev19_absolute.py').write_bytes(Path(__file__).read_bytes())
            save(out / 'parameter_change_preflight.json', check)
            config = json.loads((out / 'configuration.json').read_text())
            assert config['policy']['effective_author_cli_parameters'] == check['new_effective_parameters']
            assert sha(OLD_CONFIG) == OLD_CONFIG_SHA
            save(out / 'single_factor_binding.json', {**check, 'generation_executed': True, 'returncode': code,
                 'prior_relative_run': str(OLD_CONFIG.parent), 'new_run': str(out),
                 'all_cases_same_settings': True, 'old_config_unchanged': True})
    raise SystemExit(code)


if __name__ == '__main__':
    main()
