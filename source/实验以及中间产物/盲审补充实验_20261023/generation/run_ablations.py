#!/usr/bin/env python3
"""Frozen P/S ablation. Preflight/freeze never import torch or load models."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
from ablation_core import ARMS, SEEDS, execution_grid, make_controller, need, sampling_callback, select_cases, tensor_sha

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[1]
EXP = ARTIFACTS / '独立AU控制实验_20260914'
COMPLETION = EXP / 'completion_20260915'
OLD = COMPLETION / 'generation_runs/selected_v11_32_v1'
TRAIN = EXP / 'runs/v4_slot_2k_bidir_step200_202609141920'
ARM_NAMES = {'P1S1': 'full', 'P0S1': 'no_projection', 'P1S0': 'no_smoothing', 'P0S0': 'both_off'}
DATALLESS = 0x40000000


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    path = Path(path).resolve()
    need(HERE in path.parents, 'All new output must stay in the new generation directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # Same-directory hard-link publication is atomic and refuses overwrite.
    # An interrupted temporary file is retained as evidence, never parsed on resume.
    os.link(temporary, path)
    temporary.unlink()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def check_ref(item):
    path = Path(item['path'])
    need(path.is_file() and sha(path) == item['sha256'], 'Content changed/missing: ' + str(path))
    return path


def stat_file(path, expected=None, role=None):
    path = Path(path)
    item = {'path': str(path), 'role': role, 'expected_sha256': expected, 'exists': path.is_file()}
    if item['exists']:
        stat = path.stat()
        item.update(bytes=stat.st_size, blocks=stat.st_blocks, dataless=bool(getattr(stat, 'st_flags', 0) & DATALLESS))
    return item


def runtime_asset_paths():
    hub = Path.home() / '.cache/huggingface/hub'
    rv = hub / 'models--SG161222--Realistic_Vision_V4.0_noVAE/snapshots/1685907c0283c7278ba26c5fe561506f564b48d3'
    ff = hub / 'models--Tvaranka--fineface/snapshots/3bd764b8f6f9b113b5e1d878cbfb499d71c04446'
    ip = hub / 'models--h94--IP-Adapter-FaceID/snapshots/43907e6f44d079bf1a9102d9a6e56aef7a219bae'
    result = {'rv/' + name: rv / name for name in (
        'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
        'text_encoder/config.json', 'text_encoder/model.safetensors',
        'scheduler/scheduler_config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt',
        'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json')}
    result.update({'fineface/au_attention': ff / 'sd15_attn_processors.ckpt',
                   'fineface/au_encoder': ff / 'sd15_au_encoder.ckpt',
                   'faceid': ip / 'ip-adapter-faceid-portrait-v11_sd15.bin'})
    return result


def bind_session_caches(session):
    os.environ.update(HF_HOME=str(session / 'cache/huggingface'), TORCH_HOME=str(session / 'cache/torch'),
                      MPLCONFIGDIR=str(session / 'cache/matplotlib'), PYTHONDONTWRITEBYTECODE='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.pycache_prefix = str(session / 'cache/pycache')
    sys.dont_write_bytecode = True


def documents():
    config = read(OLD / 'configuration.json')
    check_ref(config['source_cache'])
    cache = read(config['source_cache']['path'])
    runtime = read(OLD / 'runtime.json')
    bindings = read(TRAIN / 'bindings.json')
    cases = select_cases(cache)
    need(cache['kind'] == 'completion_v11_source_feature_cache' and cache['status'] == 'complete', 'Incomplete source cache')
    need(cache['protocol_sha256'] == config['freeze']['sha256'], 'Source protocol binding changed')
    need(cache['sampling_seeds'] == SEEDS and config['policy']['guidance_scale'] == 7.5, 'Original seeds/CFG changed')
    need(cache['target_pixels_read'] is False and cache['target_latents_computed'] is False, 'Source-only cache contract failed')
    need(runtime['source_anchor'] == cache['source_anchor'] == bindings['source_anchor'], 'Source anchor metadata mismatch')
    need(runtime['public_assets'] == bindings['public_assets'], 'Old runtime/checkpoint public assets differ')
    return config, cache, runtime, bindings, cases


def input_inventory(config, cache, runtime, bindings, cases, include_old_images=True):
    result = []
    for name, path in [('old_configuration', OLD / 'configuration.json'), ('old_runtime', OLD / 'runtime.json'),
                       ('training_bindings', TRAIN / 'bindings.json'), ('checkpoint_seal', TRAIN / 'checkpoint_00000200.json'),
                       ('old_result', OLD / 'result.json')]:
        result.append(stat_file(path, role=name))
    for role, item in [('frozen_protocol', config['freeze']), ('source_cache', config['source_cache']),
                       ('source_features', cache['features_file']), ('source_latents', runtime['source_latents']),
                       ('checkpoint', runtime['checkpoints']['bidirectional']), ('source_measurements', config['source_measurements'])]:
        result.append(stat_file(item['path'], item['sha256'], role))
    for kind in ('execution_sources', 'reused_sources'):
        for role, item in config['generation'][kind].items():
            result.append(stat_file(item['path'], item['sha256'], kind + '/' + role))
    for name, expected in bindings['sources'].items():
        result.append(stat_file(EXP / name, expected, 'trained_source/' + name))
    for name, path in runtime_asset_paths().items():
        result.append(stat_file(path, bindings['public_assets'][name]['sha256'], 'runtime_asset/' + name))
    for source in cache['sources']:
        item = source['canonical_source']
        result.append(stat_file(item['path'], item['sha256'], 'source_image'))
    for case in cases:
        folder = OLD / 'masks' / case['case_id']
        for name in ('edit64.npy', 'cleanup64.npy', 'receipt.json'):
            result.append(stat_file(folder / name, role='source_mask'))
        if include_old_images:
            for seed in SEEDS:
                result.append(stat_file(OLD / 'images' / case['case_id'] / f'bidirectional_local_lp_s050_seed{seed}.png', role='old_full_parity_image'))
    for path in (HERE / 'ablation_core.py', Path(__file__)):
        result.append(stat_file(path, role='new_implementation'))
    # These dynamically imported files enforce their own embedded historical SHA pins.
    for relative in ('improvements/a_terminal_cleanup_v10_20260915/run_terminal.py',
                     'improvements/region_preserving_v6_20260915/run_region_diagnostic.py',
                     'improvements/quality_diagnosis_v5_20260915/run_diagnostic.py',
                     'improvements/region_preserving_v6_20260915/region_mask.py'):
        result.append(stat_file(EXP / relative, role='terminal_dependency'))
    return result


def preflight(output=None):
    docs = documents()
    inventory = input_inventory(*docs)
    result = {'kind': 'preservation_ablation_preflight', 'created_utc': stamp(), 'models_loaded': False,
              'gpu_started': False, 'cases': 24, 'seeds': SEEDS, 'arms': ARMS, 'planned': 288,
              'first_frozen_case': docs[-1][0]['case_id'], 'files': inventory,
              'missing': [v for v in inventory if not v['exists']],
              'needs_hydration': [v for v in inventory if v.get('dataless')],
              'excluded_from_runtime': ['404MB training cache; anchor reused with checkpoint tensor binding',
                                        'base/unidirectional/MLP checkpoints; only bidirectional step200 loaded',
                                        'source feature encoders; previously bound cached source features reused']}
    if output:
        write_new(output, result)
    print(json.dumps({**{k: result[k] for k in ('kind', 'models_loaded', 'gpu_started', 'planned', 'first_frozen_case')},
                      'missing_count': len(result['missing']), 'needs_hydration_count': len(result['needs_hydration']),
                      'hydration_bytes': sum(r['bytes'] for r in result['needs_hydration']),
                      'report': None if output is None else str(Path(output).resolve())}, ensure_ascii=False))
    return result


def freeze(path):
    config, cache, runtime, bindings, cases = documents()
    inventory = input_inventory(config, cache, runtime, bindings, cases)
    missing = [r['path'] for r in inventory if not r['exists']]
    need(not missing, 'Missing files: ' + repr(missing))
    # Hydration is deliberately not silently initiated by freeze.
    need(not any(r.get('dataless') for r in inventory), 'Run preflight and hydrate listed files before freezing')
    verified = []
    for item in inventory:
        current = ref(item['path'])
        need(item['expected_sha256'] is None or current['sha256'] == item['expected_sha256'], 'Historical SHA mismatch: ' + item['path'])
        verified.append({**current, 'role': item['role']})
    old_result = read(OLD / 'result.json')
    need(old_result['status'] == 'complete', 'Historical full run incomplete')
    old_rows = {(r['case_id'], r['seed']): r for r in old_result['records']
                if r['variant'] == 'bidirectional' and r['group'] == 'nonzero_target'}
    need(len(old_rows) == 72, 'Historical full arm must have all 72 outputs')
    references, masks = {}, {}
    for case in cases:
        cid = case['case_id']
        for seed in SEEDS:
            row = old_rows[cid, seed]
            need(row['status'] == 'complete', 'Historical full output failed')
            historical_mask_receipt = check_ref(row['mask_receipt'])
            need(historical_mask_receipt.resolve() == (OLD / 'masks' / cid / 'receipt.json').resolve(), 'Historical mask path changed')
            image = {'path': row['output'], 'sha256': row['sha256']}
            check_ref(image)
            references[f'{cid}::{seed}'] = {**image, 'condition_hashes': row['condition_hashes'],
                'noise_sha256': row['noise_sha256'], 'source_latent_sha256': row['source_latent_sha256']}
        folder = OLD / 'masks' / cid
        receipt = read(folder / 'receipt.json')
        need(receipt['source_only'] and not receipt['target_geometry_used'] and not receipt['generated_pixels_used'], 'Mask source boundary changed')
        for name in ('edit_mask', 'cleanup_mask'):
            check_ref(receipt[name])
        masks[cid] = {'receipt': ref(folder / 'receipt.json'), 'edit': receipt['edit_mask'], 'cleanup': receipt['cleanup_mask'],
                      'edit_tensor_sha256': receipt['edit_mask_tensor_sha256'], 'cleanup_tensor_sha256': receipt['cleanup_mask_tensor_sha256']}
    policy = dict(config['policy'])
    policy.update(version='review_20261023_preservation_factorial_v1', variant_order=['bidirectional'],
                  intervention='P: 14 intermediate plus final projection; S: terminal local_filtered_clean only',
                  zero_edit='not included; only original 24 nonzero requests', source_latent='reuse original tensor after SHA check')
    doc = {'kind': 'frozen_preservation_factorial_v1', 'created_utc': stamp(), 'status': 'frozen',
           'cases': cases, 'arms': ARMS, 'arm_names': ARM_NAMES, 'seeds': SEEDS, 'planned': 288,
           'selection': 'all original 24 nonzero requests in original cache order; no score/image selection',
           'policy': policy, 'inputs': verified, 'old_configuration': ref(OLD / 'configuration.json'),
           'source_cache': ref(config['source_cache']['path']), 'source_features': ref(cache['features_file']['path']),
           'source_latents': ref(runtime['source_latents']['path']), 'old_runtime': ref(OLD / 'runtime.json'),
           'training_bindings': ref(TRAIN / 'bindings.json'), 'checkpoint': ref(runtime['checkpoints']['bidirectional']['path']),
           'masks': masks, 'parity_references': references, 'new_training': False,
           'independent_new_subject_test': False, 'target_pixels_used_for_generation': False,
           'success_criteria': '288 accountable slots; full arm exact image SHA AND pixel parity for all 72 historical references; no omitted failures'}
    write_new(path, doc)
    print(json.dumps({'status': 'frozen_no_models', 'freeze': ref(path), 'planned': 288}, ensure_ascii=False))


def load_original():
    sys.path.insert(0, str(EXP))
    sys.path.insert(0, str(COMPLETION / 'generation'))
    spec = importlib.util.spec_from_file_location('review_original_run_grid', COMPLETION / 'generation/run_grid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_runtime(plan, session):
    """Reuse unchanged model classes, cache and conditions; load only one learned adapter."""
    original = load_original()
    torch = original.environment('mps')
    bind_session_caches(session)
    import importlib.metadata
    import numpy as np
    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DEISMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from modeling.frozen_backbone import FrozenConditionedDenoiser, RV
    from modeling.safe_token_residual_adapter import RelativeRMSConfig
    from train_centered_adapter import load_checkpoint, adapter_from_checkpoint, verify_source_anchor
    # The historical trainer resets cache variables at import; restore new scope.
    bind_session_caches(session)
    cache = read(check_ref(plan['source_cache']))
    bindings = read(check_ref(plan['training_bindings']))
    old_runtime = read(check_ref(plan['old_runtime']))
    packages = {name: importlib.metadata.version(name) for name in bindings['environment']['packages']}
    need(packages == bindings['environment']['packages'], 'Package versions differ from original frozen environment')
    features = torch.load(check_ref(plan['source_features']), map_location='cpu', weights_only=True)
    state = load_checkpoint(check_ref(plan['checkpoint']))
    need(state['variant'] == 'bidirectional' and state['global_step'] == 200 and state['diagnostic'] is False, 'Wrong trained checkpoint')
    need(state['bindings'] == bindings, 'Checkpoint bindings differ from historical training file')
    need(set(features) == {'source_anchor', 'sources', 'conditions'}, 'Unexpected cache field/target latent')
    verify_source_anchor(features['source_anchor'], cache['source_anchor'], state['bindings']['source_anchor'])
    for case in plan['cases']:
        values = features['conditions'][case['case_id']]
        need(values['source_aus'].tolist() == case['source_au_vector_0_5'] and
             values['target_aus'].tolist() == case['target_au_vector_0_5'] and values['prompt'] == case['prompt'], 'Frozen condition changed')
        for key in ('text_tokens', 'text_pooled', 'source_aus', 'target_aus'):
            need(torch.isfinite(values[key]).all().item() and values[key].dtype == torch.float32, 'Invalid cached condition')
    latents = torch.load(check_ref(plan['source_latents']), map_location='cpu', weights_only=True)
    source_receipts = {r['source_id']: r for r in old_runtime['source_vae_records']}
    need(set(latents) == set(source_receipts), 'Original source latent set mismatch')
    for sid, latent in latents.items():
        need(source_receipts[sid]['status'] == 'complete' and latent.shape == (1, 4, 64, 64) and
             latent.dtype == torch.float32 and tensor_sha(latent) == source_receipts[sid]['latent_sha256'], 'Original source latent mismatch')

    class Runtime(original.Runtime):
        def __init__(self):
            self.torch, self.features, self.device, self.output = torch, features, 'mps', session
            self.v11 = original.module('v11')
            bind_session_caches(session)
            adapter = adapter_from_checkpoint(state, self.device).eval().requires_grad_(False)
            cfg = state['effective_config']
            trust = RelativeRMSConfig(enabled=cfg['trust_region'], rho_au=cfg['rho_au'], rho_ip=cfg['rho_ip'], reference_rms_floor=cfg['reference_rms_floor'])
            self.adapters = {'bidirectional': (adapter, trust)}
            self.backbone = FrozenConditionedDenoiser(self.device, checkpointing=False)
            vae = AutoencoderKL.from_pretrained(str(RV / 'vae'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
            text = CLIPTextModel.from_pretrained(str(RV / 'text_encoder'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
            self.pipe = StableDiffusionImg2ImgPipeline(vae=vae, text_encoder=text,
                tokenizer=CLIPTokenizer.from_pretrained(str(RV / 'tokenizer'), local_files_only=True), unet=self.backbone.unet,
                scheduler=DEISMultistepScheduler.from_pretrained(str(RV / 'scheduler'), local_files_only=True),
                safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            def finite_decode(module, args, result):
                need(torch.isfinite(result).all().item(), 'Nonfinite VAE decode')
            self.decode_guard = vae.decoder.register_forward_hook(finite_decode)
            self.sources = {sid: (latent.to(self.device), source_receipts[sid]['reconstruction']) for sid, latent in latents.items()}
            write_new(session / 'runtime.json', {'device': self.device, 'dtype': 'torch.float32', 'packages': packages,
                'python': platform.python_version(), 'source_vae_records': list(source_receipts.values()),
                'source_latent_mode': 'original source-only tensors verified against original runtime SHA',
                'source_anchor': cache['source_anchor'], 'checkpoint': plan['checkpoint'], 'loaded_adapter_variants': ['bidirectional'],
                'training_cache_loaded': False, 'target_encoder_calls': 0, 'scheduler': type(self.pipe.scheduler).__name__,
                'scheduler_config': dict(self.pipe.scheduler.config), 'public_assets': bindings['public_assets']})

        def sample_arm(self, prepared, seed, arm):
            source = prepared['source']
            noise = torch.randn(source.shape, generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32).to(self.device)
            controller = make_controller(self.v11, self.backbone.unet, source, prepared['edit'], prepared['cross'], torch, prepared['cleanup'], arm['smoothing'])
            callback, history = sampling_callback(controller, source, noise, prepared['edit'], torch, arm['projection'])
            receipt = {'condition_hashes': prepared['condition_hashes'], 'source_latent_sha256': tensor_sha(source),
                       'noise_sha256': tensor_sha(noise), 'edit_mask_sha256': tensor_sha(prepared['edit']),
                       'cleanup_mask_sha256': tensor_sha(prepared['cleanup']), 'mask_receipt': prepared['mask_receipt']}
            try:
                with torch.no_grad():
                    image = self.pipe(prompt=prepared['prompt'], image=source, strength=.75, num_inference_steps=20,
                        guidance_scale=7.5, negative_prompt='', generator=torch.Generator(device='cpu').manual_seed(seed),
                        cross_attention_kwargs=prepared['cross'], callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs=['latents', 'prompt_embeds']).images[0]
                controller.assert_complete()
                need(len(history) == 15, 'Wrong callback count')
                return image, {**receipt, **controller.record(), 'projection_trace': history,
                               'projection_count': sum(v['projection_applied'] for v in history)}
            except BaseException as error:
                error.generation_receipt = {**receipt, **controller.record(), 'projection_trace': history}
                raise
            finally:
                controller.close()

    masks = {}
    for case in plan['cases']:
        item = plan['masks'][case['case_id']]
        edit = np.load(check_ref(item['edit']), allow_pickle=False)
        cleanup = np.load(check_ref(item['cleanup']), allow_pickle=False)
        need(edit.shape == cleanup.shape == (64, 64) and edit.dtype == cleanup.dtype == np.float32, 'Mask shape/dtype changed')
        need(hashlib.sha256(edit.tobytes()).hexdigest() == item['edit_tensor_sha256'] and
             hashlib.sha256(cleanup.tobytes()).hexdigest() == item['cleanup_tensor_sha256'], 'Original mask tensor changed')
        masks[case['case_id']] = edit, cleanup, item['receipt']
    return Runtime(), cache, masks


def pixel_parity(path, reference):
    import numpy as np
    from PIL import Image
    check_ref(reference)
    with Image.open(path) as generated, Image.open(reference['path']) as original:
        same_format = generated.mode == original.mode and generated.size == original.size
        a, b = np.asarray(generated), np.asarray(original)
        equal = same_format and np.array_equal(a, b)
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16)) if a.shape == b.shape else None
    return {'reference': reference, 'sha256_equal': sha(path) == reference['sha256'], 'pixels_equal': bool(equal),
            'max_absolute_pixel_difference': None if delta is None else int(delta.max()),
            'mean_absolute_pixel_difference': None if delta is None else float(delta.mean()),
            'passed': bool(equal and sha(path) == reference['sha256'])}


def validate_completed(record):
    need(record['status'] == 'success' and record['output_path'] and record['output_sha256'], 'Incomplete resume record')
    check_ref({'path': record['output_path'], 'sha256': record['output_sha256']})
    if record['arm'] == 'full':
        need(record.get('parity', {}).get('passed') is True, 'Historical full parity failed; cannot resume as success')


def execute(freeze_path, output, smoke=False, resume=False):
    plan = read(freeze_path)
    need(plan['kind'] == 'frozen_preservation_factorial_v1' and plan['status'] == 'frozen', 'Explicit frozen plan required')
    need(plan['arms'] == list(ARMS) and plan['seeds'] == SEEDS and plan['planned'] == 288, 'Frozen intervention grid changed')
    need(select_cases({'cases': plan['cases']}) == plan['cases'], 'Frozen cases invalid')
    for item in plan['inputs']:
        check_ref(item)
    out = Path(output).resolve()
    need(HERE in out.parents and out != HERE, 'Output must be within new generation directory')
    config = {'freeze': ref(freeze_path), 'smoke': smoke, 'planned': 4 if smoke else 288,
              'slot_plan': [{'case_id': c['case_id'], 'subject': c['subject'], 'seed': s, 'arm': ARM_NAMES[a['name']]} for c, s, a in execution_grid(plan['cases'], smoke)]}
    if resume:
        need(out.is_dir() and read(out / 'configuration.json') == config, 'Resume configuration differs')
    else:
        need(not out.exists(), 'Refusing existing output; explicit --resume required')
        write_new(out / 'configuration.json', config)
    # mkdir is the exclusive run lock; never clear a stale lock automatically.
    lock = out / '.active-lock'
    lock.mkdir()
    try:
        write_new(lock / 'owner.json', {'pid': os.getpid(), 'started_utc': stamp()})
        return execute_locked(plan, config, out, smoke)
    finally:
        (lock / 'owner.json').unlink(missing_ok=True)
        lock.rmdir()


def execute_locked(plan, config, out, smoke):
    sessions = out / 'sessions'
    session = sessions / stamp()
    session.mkdir(parents=True)
    records = {}
    resume_read_errors = []
    for path in sorted((out / 'attempts').glob('*/*/record.json')):
        try:
            record = read(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # Keep damaged historical receipts; retry the planned slot into a new attempt.
            resume_read_errors.append({'path': str(path), 'sha256': sha(path), 'error': str(error)})
            continue
        if record['status'] == 'success':
            validate_completed(record)
            records[record['execution_id']] = record
    if resume_read_errors:
        write_new(session / 'resume_read_errors.json', resume_read_errors)
    runtime = None
    try:
        runtime, cache, masks = build_runtime(plan, session)
        sources = {s['source_id']: s for s in cache['sources']}
        with (out / 'generation_records.jsonl').open('a', encoding='utf-8', buffering=1) as journal:
            for case, seed, arm in execution_grid(plan['cases'], smoke):
                cid, name = case['case_id'], ARM_NAMES[arm['name']]
                key = f'{cid}::{seed}::{name}'
                if key in records:
                    continue
                attempt = out / 'attempts' / f'{cid}__{seed}__{name}' / stamp()
                attempt.mkdir(parents=True)
                source = sources[case['source_id']]
                row = {'execution_id': key, 'case_id': cid, 'subject': case['subject'], 'split': case['split'],
                       'seed': seed, 'arm': name, 'switches': arm, 'variant': 'bidirectional',
                       'source_path': source['canonical_source']['path'], 'source_sha256': source['canonical_source']['sha256'],
                       'source_au12': case['source_au_vector_0_5'], 'target_au12': case['target_au_vector_0_5'],
                       'source_frame_1based': case['source_frame_1based'], 'target_frame_1based': case.get('target_frame_1based'),
                       'output_path': None, 'output_sha256': None, 'started_utc': stamp(), 'status': 'failed'}
                write_new(attempt / 'started.json', row)
                started = time.monotonic()
                fatal_parity = False
                try:
                    prepared = runtime.prepare('bidirectional', case, source, masks[cid])
                    image, receipt = runtime.sample_arm(prepared, seed, arm)
                    row.update(receipt)
                    need(image.mode == 'RGB' and image.size == (512, 512), 'Native512RGB required')
                    image_path = attempt / 'image.png'
                    with image_path.open('xb') as stream:
                        image.save(stream, format='PNG')
                        stream.flush()
                        os.fsync(stream.fileno())
                    row.update(output_path=str(image_path), output_sha256=sha(image_path))
                    reference = plan['parity_references'][f'{cid}::{seed}']
                    for field in ('condition_hashes', 'source_latent_sha256', 'noise_sha256'):
                        need(row[field] == reference[field], 'Paired generation input changed: ' + field)
                    if name == 'full':
                        row['parity'] = pixel_parity(image_path, reference)
                        fatal_parity = not row['parity']['passed']
                        need(not fatal_parity, 'Full-arm historical PNG parity failed; all images preserved, stop before further inference')
                    row['status'] = 'success'
                except (Exception, KeyboardInterrupt) as error:
                    # Any full-arm failure leaves the historical protocol unverified.
                    # Do not spend GPU time on the other arms after that gate fails.
                    fatal_parity = fatal_parity or name == 'full'
                    row.update(getattr(error, 'generation_receipt', {}))
                    row.update(error=str(error), traceback=traceback.format_exc(), interrupted=isinstance(error, KeyboardInterrupt))
                row['seconds'] = time.monotonic() - started
                write_new(attempt / 'record.json', row)
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                records[key] = row
                print(json.dumps({k: row[k] for k in ('case_id', 'seed', 'arm', 'status', 'seconds')}, ensure_ascii=False), flush=True)
                if fatal_parity or row.get('interrupted'):
                    break
    except (Exception, KeyboardInterrupt) as error:
        write_new(session / 'fatal_error.json', {'error': str(error), 'traceback': traceback.format_exc()})
    finally:
        if runtime:
            runtime.close()
    final = []
    source_metadata = {s['source_id']: s for s in read(plan['source_cache']['path'])['sources']}
    for case, seed, arm in execution_grid(plan['cases'], smoke):
        key = f"{case['case_id']}::{seed}::{ARM_NAMES[arm['name']]}"
        source = source_metadata[case['source_id']]['canonical_source']
        final.append(records.get(key, {'execution_id': key, 'case_id': case['case_id'], 'subject': case['subject'],
                                      'seed': seed, 'arm': ARM_NAMES[arm['name']], 'status': 'failed',
                                      'source_path': source['path'], 'source_sha256': source['sha256'],
                                      'source_au12': case['source_au_vector_0_5'], 'target_au12': case['target_au_vector_0_5'],
                                      'source_frame_1based': case['source_frame_1based'], 'target_frame_1based': case.get('target_frame_1based'),
                                      'execution_started': False, 'error': 'not_started', 'output_path': None, 'output_sha256': None}))
    successes = sum(r['status'] == 'success' for r in final)
    result = {'kind': 'preservation_factorial_result', 'freeze': config['freeze'], 'smoke': smoke,
              'run_root': str(out), 'slot_plan': config['slot_plan'], 'cases': plan['cases'],
              'resume_read_errors': resume_read_errors,
              'status': 'complete' if successes == config['planned'] else 'incomplete', 'planned': config['planned'],
              'completed': successes, 'failed_or_not_started': len(final) - successes,
              'all_slots_accounted': len(final) == config['planned'], 'records': final,
              'all_full_arm_parity_passed': all(r.get('parity', {}).get('passed') for r in final if r['arm'] == 'full'),
              'same_protocol_complete': successes == config['planned'] and all(r.get('parity', {}).get('passed') for r in final if r['arm'] == 'full'),
              'new_training': False, 'target_pixels_used_for_generation': False}
    write_new(session / 'result.json', result)
    print(json.dumps({'status': result['status'], 'completed': successes, 'planned': config['planned'],
                      'result': ref(session / 'result.json')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--freeze', type=Path, metavar='NEW_JSON')
    group.add_argument('--execute', type=Path, metavar='FROZEN_JSON')
    parser.add_argument('--output', type=Path, help='Preflight report file or execution run directory, inside generation/')
    parser.add_argument('--smoke', action='store_true', help='First frozen nonzero case, seed101, all four arms')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.output)
    elif args.freeze:
        need(not args.smoke and not args.resume, 'Freeze always records complete 288-slot protocol')
        freeze(args.freeze)
    else:
        need(args.output is not None, '--execute requires --output')
        result = execute(args.execute.resolve(), args.output, args.smoke, args.resume)
        raise SystemExit(0 if result['status'] == 'complete' else 2)


if __name__ == '__main__':
    main()
