#!/usr/bin/env python3
"""Build strict dev-only native-AU/image-driver contracts and CPU preflight plans."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
OLD = EXPERIMENT.parent / '近期外部对比_20260914'
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
PYTHON = Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'
REPOS = {'magicface': Path.home() / 'Projects/github/weimengting/MagicFace',
         'faceadapter': Path.home() / 'Projects/github/FaceAdapter/Face-Adapter',
         'liveportrait': Path.home() / 'Projects/github/KlingAIResearch/LivePortrait'}
COMMITS = {'magicface': 'bcaca61620450681233529d472c452091c25c92b',
           'faceadapter': '6e313885188e05740a7c40e12c1b9015c748bf08',
           'liveportrait': '9b294b3d0536135442ea73cb01e6cb3ca7029dd3'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def resident(path):
    path = Path(path)
    if not path.is_file():
        return {'path': str(path), 'status': 'missing'}
    s = path.stat()
    unavailable = bool(getattr(s, 'st_flags', 0) & 0x40000000) or (s.st_size > 0 and s.st_blocks == 0)
    return {'path': str(path), 'status': 'dataless_do_not_read' if unavailable else 'resident', 'bytes': s.st_size,
            'blocks': s.st_blocks, 'mtime_ns': s.st_mtime_ns}


def save(path, payload):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def install_wrappers():
    runtime = HERE / 'runtime'
    assets = ('run_faceadapter.py', 'faceadapter_detection_policy.py', 'run_liveportrait.py',
              'liveportrait_detection_policy.py', 'second_baseline_review/liveportrait_required_assets.json')
    bindings = []
    for name in assets:
        source, target = OLD / name, runtime / name
        if resident(source)['status'] != 'resident':
            raise ValueError('Wrapper source not resident: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        if target.exists() and sha(target) != expected:
            raise ValueError('Installed wrapper differs; use versioned runtime instead')
        if not target.exists():
            shutil.copyfile(source, target)
        bindings.append({'source': str(source), 'runtime': str(target), 'sha256': expected, 'content_modified': False})
    return runtime, bindings


def magic_resources():
    home = Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace'
    old = EXPERIMENT.parent / 'disfa_au_augmentation_20260723/artifacts/magicface_official_preprocess_20260725'
    candidates = {
        'ID_enc/config.json': [old / 'generator_models/ID_enc/config.json', home / 'ID_enc/config.json'],
        'ID_enc/diffusion_pytorch_model.safetensors': [EXPERIMENT / 'models/magicface/ID_enc/diffusion_pytorch_model.safetensors', old / 'generator_models/ID_enc/diffusion_pytorch_model.safetensors', home / 'ID_enc/diffusion_pytorch_model.safetensors'],
        'denoising_unet/config.json': [old / 'generator_models/denoising_unet/config.json', home / 'denoising_unet/config.json'],
        'denoising_unet/diffusion_pytorch_model.safetensors': [old / 'generator_models/denoising_unet/diffusion_pytorch_model.safetensors', home / 'denoising_unet/diffusion_pytorch_model.safetensors'],
        'text_encoder/config.json': [old / 'generator_models/text_encoder/config.json', home / 'sd15_cache/text_encoder/config.json'],
        'text_encoder/model.safetensors': [home / 'sd15_cache/text_encoder/model.safetensors', old / 'generator_models/text_encoder/model.safetensors'],
        '79999_iter.pth': [home / 'third_party_files/79999_iter.pth', home / '79999_iter.pth', old / 'models/79999_iter.pth'],
        'd3dfr_res50_nofc.pth': [OLD / 'models/face_adapter/third_party/d3dfr_res50_nofc.pth', old / 'models/d3dfr_res50_nofc.pth'],
        'BFM_model_front.mat': [OLD / 'models/face_adapter/third_party/BFM_model_front.mat', old / 'models/BFM_model_front.mat'],
        'scrfd_10g_bnkps.onnx': [OLD / 'models/face_adapter/third_party/models/antelopev2/scrfd_10g_bnkps.onnx', old / 'models/scrfd_10g_bnkps.onnx'],
    }
    for name in ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors', 'scheduler/scheduler_config.json',
                 'tokenizer/merges.txt', 'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json'):
        candidates[name] = [EXPERIMENT / 'models/magicface_base' / name, home / 'sd15_cache' / name]
    resources = []
    for name, paths in candidates.items():
        records = [resident(path) for path in paths]
        selected = next((row for row in records if row['status'] == 'resident'), None)
        # Hash only resident bytes; this never hydrates a cloud placeholder.
        if selected is not None:
            selected = dict(selected, sha256=sha(selected['path']))
            if name == 'ID_enc/diffusion_pytorch_model.safetensors':
                assert selected['bytes'] == 3438167536 and selected['sha256'] == 'c5e676f3774849e861c13098e94972e830be47a5b1881dfd25291e0e71821f51'
        resources.append({'component': name, 'selected': selected, 'candidates': records})
    return {'resources': resources, 'missing_resident_components': [row['component'] for row in resources if row['selected'] is None],
            'preprocessing': 'fresh source-derived SCRFD crop + BiSeNet background + D3DFR/BFM contour; no target-image input',
            'fresh_backgrounds_prepared': False, 'generation_executed': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-preflight', action='store_true')
    args = p.parse_args()
    for name, repo in REPOS.items():
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMITS[name]
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    request_path = EXPERIMENT / 'development_previews/dev12_manifest.json'
    cache_path = EXPERIMENT / 'training_cache/expanded_v2_202609141711/manifest.json'
    requests, cache = json.loads(request_path.read_text()), json.loads(cache_path.read_text())
    assert not requests['test_data_used'] and not cache['test_data_used'] and requests['cache_sha256'] == cache['cache_sha256']
    assert len(requests['cases']) == 12 and requests['sampling_seeds'] == [101]
    pairs = {row['case_id']: row for row in cache['pairs']}
    from PIL import Image
    inspected, native, driver = {}, [], []
    for case in requests['cases']:
        pair = pairs[case['case_id']]
        assert case['split'] == pair['split'] == 'dev'
        source_au, target_au = case['source_au_vector_0_5'], case['target_au_vector_0_5']
        assert len(source_au) == len(target_au) == 12 and all(float(v).is_integer() and 0 <= v <= 5 for v in source_au + target_au)
        assert target_au == pair['target_au_vector_0_5']
        expected_source = pair.get('source_au_vector_0_5', [pair.get('source_aus', {}).get(f'au{au}') for au in AUS])
        assert source_au == expected_source
        images = {}
        for role in ('source', 'target'):
            entry = pair['aligned_' + role]
            path = Path(entry['aligned_path']).resolve()
            assert EXPERIMENT in path.parents and resident(path)['status'] == 'resident' and sha(path) == entry['aligned_sha256']
            if entry['aligned_sha256'] not in inspected:
                with Image.open(path) as image:
                    image.load()
                    assert image.mode == 'RGB' and image.size == (512, 512)
                inspected[entry['aligned_sha256']] = {'path': str(path), 'sha256': entry['aligned_sha256'], 'mode': 'RGB', 'size': [512, 512]}
            images[role] = inspected[entry['aligned_sha256']]
        for seed in requests['sampling_seeds']:
            common = {'case_id': case['case_id'] + f'_seed{seed}', 'request_case_id': case['case_id'], 'subject': case['subject'],
                      'split': 'dev', 'group': case['group'], 'seed': seed, 'au_order': AUS,
                      'source': images['source']['path'], 'source_sha256': images['source']['sha256'],
                      'source_au_vector_0_5': source_au, 'target_au_vector_0_5': target_au}
            native.append({**common, 'au_delta_native': [t - s for s, t in zip(source_au, target_au)],
                           'input_group': 'source_plus_manual_au', 'background_policy': 'derive_from_this_source_only',
                           'prompt': 'A close up of a person.', 'native_inference_steps': 50})
            driver.append({**common, 'target': images['target']['path'], 'target_sha256': images['target']['sha256'],
                           'input_group': 'source_plus_real_driver', 'target_role': 'real_observed_driver_not_generated_output'})
    assert all('target' not in row and 'target_path' not in row and 'target_image_path' not in row for row in native)
    base = HERE / ('contracts/dev12_' + datetime.now().strftime('%Y%m%d%H%M'))
    output = base
    index = 2
    while output.exists():
        output = base.with_name(base.name + f'-{index:02d}')
        index += 1
    output.mkdir(parents=True)
    provenance = {'schema_version': 1, 'scope': 'development_only', 'test_used': False, 'sampling_seeds': requests['sampling_seeds'],
                  'request_manifest_sha256': sha(request_path), 'pair_cache_manifest_sha256': sha(cache_path),
                  'old_generated_outputs_or_metrics_used': False}
    save(output / 'magicface_dev12.json', {**provenance, 'input_group': 'source_plus_manual_au', 'cases': native})
    save(output / 'image_driver_dev12.json', {**provenance, 'input_group': 'source_plus_real_driver', 'cases': driver})
    save(output / 'input_shape_checks.json', {'unique_images': list(inspected.values()), 'requests': 12, 'shape': [3, 512, 512],
                                              'magicface_delta_shape': [1, 12], 'au_delta_min': min(min(r['au_delta_native']) for r in native),
                                              'au_delta_max': max(max(r['au_delta_native']) for r in native), 'test_used': False})
    runtime, bindings = install_wrappers()
    save(output / 'wrapper_bindings.json', bindings)
    magic = magic_resources()
    save(output / 'magicface_resources.json', magic)
    stamp = output.name
    commands = {
        'faceadapter': [str(PYTHON), '-B', str(runtime / 'run_faceadapter.py'), '--repo', str(REPOS['faceadapter']),
                        '--checkpoint', str(OLD / 'models/face_adapter'), '--base-model', str(OLD / 'models/stable-diffusion-v1-5'),
                        '--vae-model', str(OLD / 'models/sd-vae-ft-mse'), '--manifest', str(output / 'image_driver_dev12.json'),
                        '--output', str(runtime / 'runs' / ('faceadapter_' + stamp)), '--device', 'cpu', '--preload-final-components',
                        '--detection-policy', 'shared_fallback', '--preflight'],
        'liveportrait': [str(PYTHON), '-B', str(runtime / 'run_liveportrait.py'), '--repo', str(REPOS['liveportrait']),
                         '--checkpoint', str(OLD / 'models/liveportrait_author'), '--manifest', str(output / 'image_driver_dev12.json'),
                         '--output', str(runtime / 'runs' / ('liveportrait_' + stamp)), '--device', 'cpu',
                         '--detection-policy', 'shared_fallback', '--preflight'],
    }
    save(output / 'preflight_commands.json', commands)
    preflights = {}
    if args.run_preflight:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(HERE / 'cache/pycache'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
                   HF_HOME=str(HERE / 'cache/huggingface'), TORCH_HOME=str(HERE / 'cache/torch'),
                   MPLCONFIGDIR=str(HERE / 'cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'cache/xdg'))
        for name, command in commands.items():
            with (output / (name + '_preflight.log')).open('x') as log:
                run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            preflights[name] = {'returncode': run.returncode, 'command': command}
    save(output / 'summary.json', {'output': str(output), 'request_count': 12, 'native_au_cases': len(native), 'driver_cases': len(driver),
                                   'unique_real_images_checked': len(inspected), 'source_only_magicface': True,
                                   'magicface_missing_resident': magic['missing_resident_components'], 'preflights': preflights,
                                   'generation_run': False, 'test_used': False, 'weights_downloaded_or_modified': False})
    print(json.dumps({'output': str(output), 'magicface_missing_resident': magic['missing_resident_components'],
                      'preflight_returncodes': {k: v['returncode'] for k, v in preflights.items()}, 'generation_run': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
