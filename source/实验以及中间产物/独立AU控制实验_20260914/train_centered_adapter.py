"""Resumable v4 source-centered train/dev entry. No training occurs on import.

Default mode is a bounded diagnostic and is ineligible for final selection.
This script never reads final-test images or any legacy generated output.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(ROOT / 'cache/huggingface'), TORCH_HOME=str(ROOT / 'cache/torch'), MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'))
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT / 'cache/pycache')

import numpy as np
import torch

MODEL_VERSION = 'v4_source_centered_protected_au_relative_rms'
CHECKPOINT_KIND = 'independent_au_adapter_resumable_v4_source_centered'
AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: str
    split_path: str
    pairs_manifest: str
    model_version: str = MODEL_VERSION
    variant: str = 'bidirectional'
    readout: str = 'global'
    patch_scale: str = 'unit_mean'
    trust_region: bool = True
    rho_au: float = 0.1
    rho_ip: float = 0.025
    reference_rms_floor: float = 1e-6
    diagnostic: bool = True
    max_steps: int = 64
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    identity_weight: float = 0.1
    au_weight: float = 1.0
    residual_weight: float = 1e-4
    aux_snr_min: float = 0.5
    aux_snr_max: float = 20.0
    au_qualification: str | None = str(ROOT / 'au_supervisor/layer4_v1/qualification.json')
    device: str = 'mps'
    seed: int = 20260914
    noise_seed: int = 20260915
    order_seed: int = 20260916
    dev_seed: int = 20260917
    dev_case_manifest: str | None = None
    dev_timesteps: tuple = (50, 250, 500, 750, 950)
    checkpoint_every: int = 32
    validate_every: int = 32
    cpu_threads: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'dev_timesteps', tuple(self.dev_timesteps))
        if self.model_version != MODEL_VERSION or self.variant not in ('bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp'):
            raise ValueError('Unsupported model version or variant')
        if self.readout not in ('global', 'slot') or self.patch_scale not in ('raw', 'unit_mean'):
            raise ValueError('Unknown readout/patch factor')
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Unknown device')
        for name in ('max_steps', 'batch_size', 'checkpoint_every', 'validate_every', 'cpu_threads'):
            if getattr(self, name) < 1:
                raise ValueError(name + ' must be positive')
        if self.diagnostic and self.max_steps > 128:
            raise ValueError('Diagnostic runs are bounded to 128 total steps')
        for name in ('learning_rate', 'gradient_clip', 'rho_au', 'rho_ip', 'reference_rms_floor'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name in ('weight_decay', 'identity_weight', 'au_weight', 'residual_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if not 0 < self.aux_snr_min <= self.aux_snr_max or not math.isfinite(self.aux_snr_max):
            raise ValueError('Invalid auxiliary SNR window')
        if not self.dev_timesteps or any(not isinstance(t, int) or not 0 <= t <= 999 for t in self.dev_timesteps):
            raise ValueError('Development timesteps must lie in [0,999]')
        if self.au_weight > 0 and not self.au_qualification:
            raise ValueError('Positive AU loss requires qualified train-only supervisor')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resolve_source_anchor(train, dev=()):
    """Choose only from real TRAIN zero-target records; dev can only fail checks."""
    from prepare_training_cache import make_prompt
    neutral_prompt = make_prompt([0] * 12)
    for role, rows in (('train', train), ('dev', dev)):
        for row in rows:
            if row['split'] != role or row['source_aus'].shape != (12,) or torch.count_nonzero(row['source_aus']).item():
                raise ValueError('v4 integration accepts only recorded zero-AU sources, never inferred nonzero sources')
    neutral = sorted((row for row in train if torch.count_nonzero(row['target_aus']).item() == 0), key=lambda row: row['case_id'])
    if not neutral:
        raise ValueError('No real neutral TRAIN target available for the source-text anchor')
    anchor = neutral[0]['text_pooled'].detach().cpu().clone()
    if anchor.shape != (768,) or not torch.isfinite(anchor).all() or anchor.dtype != torch.float32:
        raise ValueError('Invalid source-text anchor tensor')
    for row in list(train) + list(dev):
        zero_target = torch.count_nonzero(row['target_aus']).item() == 0
        if zero_target != (row['prompt'] == neutral_prompt):
            raise ValueError('Neutral prompt and manual zero-AU target disagree')
        if zero_target and not torch.equal(row['text_pooled'].cpu(), anchor):
            raise ValueError('Neutral cached text embeddings are not identical')
    binding = {'policy': 'lexicographically first real zero-AU TRAIN target; all neutral embeddings must match',
               'anchor_train_case_id': neutral[0]['case_id'], 'neutral_train_case_ids': [row['case_id'] for row in neutral],
               'source_prompt': neutral_prompt, 'text_pooled_sha256': tensor_sha256(anchor), 'shape': [768], 'dtype': 'torch.float32',
               'source_au_policy': 'all twelve source AU values must be exactly zero'}
    return anchor, binding


def verify_source_anchor(anchor, actual_binding, expected_binding):
    if actual_binding != expected_binding or tensor_sha256(anchor) != expected_binding.get('text_pooled_sha256'):
        raise ValueError('Source-text anchor content or training provenance changed')


def create_centered_adapter(variant, model_config, device='cpu'):
    from modeling.safe_token_residual_adapter import TokenAdapterConfig, SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter
    cfg = TokenAdapterConfig(**model_config) if isinstance(model_config, dict) else model_config
    expected_direction = 'bidirectional' if variant == 'mlp' else variant
    expected_ff = 3 if variant in ('condition_reads_image', 'image_reads_condition') else 2
    if variant not in ('bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp') or cfg.direction != expected_direction or cfg.ff_multiplier != expected_ff:
        raise ValueError('Model variant and recorded direction/parameter matching disagree')
    model = SafeTokenPooledMLPResidualAdapter(cfg, 1032) if variant == 'mlp' else SafeTokenCrossModalResidualAdapter(cfg)
    bias = model.slot_readout[-1].bias
    if bias is None or bias.numel() != cfg.hidden_dim:
        raise ValueError('Unexpected common output-offset parameter')
    bias.requires_grad_(False)
    bias.grad = None
    return model.to(device)


def adapter_from_checkpoint(payload, device='cpu'):
    if payload.get('kind') != CHECKPOINT_KIND or payload.get('model_version') != MODEL_VERSION:
        raise ValueError('v4 requires a source-centered checkpoint, not a v3 checkpoint')
    model = create_centered_adapter(payload['variant'], payload['model_config'], device)
    model.load_state_dict(payload['adapter'], strict=True)
    return model


def centered_batch_conditions(model, batch, source_anchor, base_au, base_ip, trust):
    from modeling.source_centered_adapter import forward_source_centered
    from modeling.safe_token_residual_adapter import condition_residuals
    if torch.count_nonzero(batch['source_aus']).item():
        raise ValueError('This integrated v4 runner only supports known zero-AU sources')
    source_text = source_anchor[None].expand(len(batch['target_aus']), -1)
    output = forward_source_centered(model, batch['image_patches'], batch['text_pooled'], batch['target_aus'], source_text, batch['source_aus'])
    noedit = (batch['target_aus'] == batch['source_aus']).all(-1) & (batch['text_pooled'] == source_text).all(-1)
    if noedit.any().item() and (torch.count_nonzero(output.au_delta[noedit]).item() or torch.count_nonzero(output.image_delta[noedit]).item()):
        raise RuntimeError('Identical source/target conditions did not produce zero raw residual')
    # One limiter operation. Assemble conditions from that exact effective
    # residual rather than applying a second limiter to already bounded data.
    effective = condition_residuals(base_au, base_ip, output, trust_region=trust)
    explicit = output.explicit_au_dims
    values = {'au_embedding': torch.cat((base_au[..., :explicit], base_au[..., explicit:] + effective['au_delta'][..., explicit:].to(base_au.dtype)), dim=-1),
              'image_embedding': base_ip + effective['image_delta'].to(base_ip.dtype)}
    return output, effective, values


def effective_config(config):
    # Paths are protected by content bindings. Only target run length may grow;
    # LR is constant, so extending max_steps does not alter an LR schedule.
    ignored = {'cache_dir', 'split_path', 'pairs_manifest', 'au_qualification', 'dev_case_manifest', 'max_steps'}
    return {key: value for key, value in asdict(config).items() if key not in ignored}


def write_new(path, writer):
    """Atomic publish without replacing an existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.partial-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # fails atomically if destination already exists
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_new(path, payload):
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    write_new(path, lambda handle: handle.write(encoded))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    return value


def capture_rng(noise_generator, order_generator, device):
    np_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'algorithm': np_state[0], 'keys': torch.from_numpy(np_state[1].astype(np.int64)), 'position': np_state[2], 'has_gauss': np_state[3], 'cached_gaussian': np_state[4]}, 'torch_cpu': torch.get_rng_state(), 'torch_mps': torch.mps.get_rng_state().cpu() if device == 'mps' else None, 'torch_cuda': [state.cpu() for state in torch.cuda.get_rng_state_all()] if device == 'cuda' else None, 'private_noise': noise_generator.get_state(), 'private_order': order_generator.get_state(), 'device': device}


def restore_rng(state, noise_generator, order_generator, device):
    if state['device'] != device:
        raise ValueError('Exact resume cannot silently change device')
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['algorithm'], n['keys'].cpu().numpy().astype(np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device == 'mps':
        if state['torch_mps'] is None:
            raise ValueError('Missing MPS RNG state')
        torch.mps.set_rng_state(state['torch_mps'].cpu())
    if device == 'cuda':
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    noise_generator.set_state(state['private_noise'].cpu())
    order_generator.set_state(state['private_order'].cpu())


def next_indices(state, count, batch_size, generator):
    if count < 1:
        raise ValueError('Empty training set')
    if not state['order'] or state['cursor'] == count:
        if state['order']:
            state['epoch'] += 1
        state['order'] = torch.randperm(count, generator=generator).tolist()
        state['cursor'] = 0
    if sorted(state['order']) != list(range(count)) or not 0 <= state['cursor'] < count:
        raise ValueError('Invalid saved data order/cursor')
    indices = state['order'][state['cursor']:state['cursor'] + batch_size]
    state['cursor'] += len(indices)
    return indices


def gradient_diagnostics(model):
    """Before clipping: absent modules/gradients remain null, never fake zero."""
    directional = []
    for index, block in enumerate(getattr(model, 'blocks', [])):
        item = {'block': index}
        for name in ('condition_from_image', 'image_from_condition'):
            module = getattr(block, name, None)
            item[name] = None if module is None else {part: None if getattr(module, part).weight.grad is None else float(getattr(module, part).weight.grad.detach().norm().cpu()) for part in ('q', 'k', 'v')}
        directional.append(item)
    gradients = [p.grad.detach().cpu() for p in model.parameters() if p.requires_grad and p.grad is not None]
    return {'directional_gradient_norms': directional if directional else None, 'gradient_scale': 'before clipping', 'trainable_parameter_tensors': sum(p.requires_grad for p in model.parameters()), 'nonnull_gradient_parameter_tensors': len(gradients), 'nonzero_gradient_parameter_tensors': sum(bool(torch.count_nonzero(g).item()) for g in gradients)}


def validate_resume(payload, config, bindings):
    if payload.get('kind') != CHECKPOINT_KIND or payload.get('model_version') != MODEL_VERSION or payload.get('variant') != config.variant:
        raise ValueError('Checkpoint version/variant mismatch; no MLP/attention substitution')
    if fingerprint(payload['effective_config']) != fingerprint(effective_config(config)):
        raise ValueError('Effective training configuration changed')
    if payload['bindings_fingerprint'] != fingerprint(bindings) or payload['bindings'] != bindings:
        raise ValueError('Source/data/split/model/environment binding changed')
    if not 0 <= payload['global_step'] < config.max_steps:
        raise ValueError('Resume target must be beyond the saved step')
    count = payload['train_count']
    state = payload['data_state']
    if sorted(state['order']) != list(range(count)) or not 0 <= state['cursor'] <= count or state['epoch'] < 0:
        raise ValueError('Saved data order is malformed')
    if not payload.get('optimizer') or not payload.get('rng'):
        raise ValueError('Checkpoint is not a full resumable state')


def save_checkpoint(path, model, optimizer, config, bindings, step, data_state, noise_rng, order_rng, train_count, model_config):
    payload = {'kind': CHECKPOINT_KIND, 'model_version': MODEL_VERSION, 'variant': config.variant, 'effective_config': effective_config(config), 'bindings': bindings, 'bindings_fingerprint': fingerprint(bindings), 'global_step': step, 'train_count': train_count, 'data_state': dict(data_state), 'model_config': model_config, 'adapter': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()), 'rng': cpu_tree(capture_rng(noise_rng, order_rng, config.device)), 'diagnostic': config.diagnostic, 'eligible_for_final_selection': not config.diagnostic, 'au_supervision_enabled': config.au_weight > 0}
    write_new(path, lambda handle: torch.save(payload, handle))
    json_new(Path(path).with_suffix('.json'), {'kind': CHECKPOINT_KIND, 'checkpoint_sha256': sha256(path), 'step': step, 'variant': config.variant, 'model_version': MODEL_VERSION, 'bindings_fingerprint': payload['bindings_fingerprint']})
    return payload


def load_checkpoint(path):
    path = Path(path)
    seal = read_json(path.with_suffix('.json'))
    if seal['kind'] != CHECKPOINT_KIND or sha256(path) != seal['checkpoint_sha256']:
        raise ValueError('Checkpoint integrity seal failed')
    return torch.load(path, map_location='cpu', weights_only=True)


def load_data(config):
    cache = Path(config.cache_dir).resolve()
    split_path, pairs_path = Path(config.split_path).resolve(), Path(config.pairs_manifest).resolve()
    if any(ROOT not in path.parents for path in (cache, split_path, pairs_path)):
        raise ValueError('Only this independent experiment data may be read')
    manifest = read_json(cache / 'manifest.json')
    split, pair_document = read_json(split_path), read_json(pairs_path)
    if manifest.get('test_data_used') is not False or manifest.get('old_generated_images_or_metrics_used') is not False:
        raise ValueError('Cache must explicitly exclude test/legacy generated data')
    if split.get('test_locked') is not True:
        raise ValueError('A frozen subject split is required')
    split_sha = sha256(split_path)
    for declared in (manifest.get('subject_split_sha256'), pair_document.get('subject_split_sha256')):
        if declared is not None and declared != split_sha:
            raise ValueError('Cache/pair plan declares another frozen subject split')
    groups = {key: set(split['groups'][key]) for key in ('train', 'dev', 'test')}
    if any(groups[a] & groups[b] for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test'))):
        raise ValueError('Overlapping frozen subject groups')
    if sha256(pairs_path) != manifest['source_manifest_sha256'] or sha256(cache / 'features.pt') != manifest['cache_sha256']:
        raise ValueError('Cache/source manifest hash mismatch')
    samples = torch.load(cache / 'features.pt', map_location='cpu', weights_only=True)['samples']
    pairs = {x['case_id']: x for x in pair_document['pairs']}
    if len(pairs) != len(pair_document['pairs']) or len(samples) != len(pairs) or len({r['case_id'] for r in samples}) != len(samples):
        raise ValueError('Duplicate/missing data cases')
    for row in samples:
        pair = pairs[row['case_id']]
        role = row['split']
        if role not in ('train', 'dev') or row['subject'] not in groups[role] or row['subject'] in groups['test'] or pair['split'] != role or pair['subject'] != row['subject']:
            raise ValueError('Sample does not belong to its frozen train/dev role')
        if pair['au_order'] != list(AUS) or not torch.equal(row['target_aus'], torch.tensor(pair['target_au_vector_0_5'], dtype=row['target_aus'].dtype)):
            raise ValueError('Target AU/vector order mismatch')
        if 'source_au_vector_0_5' in pair:
            source_values = pair['source_au_vector_0_5']
            paths = [pair[side]['raw_path'] for side in ('source', 'target')]
            for side in ('source', 'target'):
                image_record = pair[side]
                if image_record['split'] != role or image_record['subject'] != row['subject']:
                    raise ValueError('Pair source/target role differs')
                expected_labels = [image_record['input']['labels']['au' + str(a)] for a in AUS]
                if expected_labels != pair[side + '_au_vector_0_5']:
                    raise ValueError('Manual AU labels differ from pair vector')
        else:
            source_values = [pair['source_aus']['au' + str(a)] for a in AUS]
            paths = [pair['source_path'], pair['target_path']]
            if [pair['target_aus']['au' + str(a)] for a in AUS] != pair['target_au_vector_0_5']:
                raise ValueError('Manual target labels differ from vector')
        if not torch.equal(row['source_aus'], torch.tensor(source_values, dtype=row['source_aus'].dtype)):
            raise ValueError('Source AU mismatch')
        for path in paths:
            if ROOT not in Path(path).resolve().parents:
                raise ValueError('Pair points outside new raw-data preparation')
        expected = {'image_patches': (49, 512), 'text_pooled': (768,), 'text_tokens': (77, 768), 'target_aus': (12,), 'base_faceid': (512,), 'source_identity': (512,)}
        for key, shape in expected.items():
            if row[key].shape != shape or not torch.isfinite(row[key]).all():
                raise ValueError('Invalid cached feature: ' + key)
        if ((row['target_aus'] < 0) | (row['target_aus'] > 5)).any() or not torch.isfinite(row['latent_mean']).all() or not torch.isfinite(row['latent_std']).all() or (row['latent_std'] < 0).any():
            raise ValueError('Invalid target intensity/latent distribution')
        latent_size = int(manifest['resolution']) // 8
        if row['latent_mean'].shape != (4, latent_size, latent_size) or row['latent_std'].shape != row['latent_mean'].shape:
            raise ValueError('Cached latent dimensions do not match recorded resolution')
    train = sorted((r for r in samples if r['split'] == 'train'), key=lambda r: r['case_id'])
    dev = sorted((r for r in samples if r['split'] == 'dev'), key=lambda r: r['case_id'])
    if not train or not dev:
        raise ValueError('Both train and development cases are required')
    _, source_anchor_binding = resolve_source_anchor(train, dev)
    bindings = {'cache': sha256(cache / 'features.pt'), 'cache_manifest': sha256(cache / 'manifest.json'), 'split': sha256(split_path), 'pairs': sha256(pairs_path), 'train_case_ids': [r['case_id'] for r in train], 'all_dev_case_ids': [r['case_id'] for r in dev], 'source_anchor': source_anchor_binding}
    dev, subset_info = select_development_cases(config.dev_case_manifest, dev, bindings['cache'], bindings['split'])
    bindings.update(dev_case_ids=[r['case_id'] for r in dev], development_subset=subset_info)
    return train, dev, groups, bindings


def select_development_cases(path, dev, cache_sha, split_sha):
    if path is None:
        return dev, {'policy': 'all development cases', 'manifest_sha256': None}
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError('Development manifest must belong to this independent experiment')
    document = read_json(path)
    if document.get('status') != 'frozen_for_development' or document.get('test_data_used') is not False or document.get('cache_sha256') != cache_sha or document.get('subject_split_sha256') != split_sha:
        raise ValueError('Development subset must be frozen against this cache and subject split')
    cases = document['cases']
    mapping = {row['case_id']: row for row in dev}
    selected = []
    seen = set()
    for item in cases:
        case_id = item['case_id']
        if case_id in seen or case_id not in mapping or item.get('split') != 'dev' or item.get('subject') != mapping[case_id]['subject']:
            raise ValueError('Development subset has duplicate/non-dev cases or subject drift')
        row = mapping[case_id]
        for source, key in (('source_aus', 'source_au_vector_0_5'), ('target_aus', 'target_au_vector_0_5')):
            if key in item and not torch.equal(row[source], torch.tensor(item[key], dtype=row[source].dtype)):
                raise ValueError('Development subset changed manual AU labels')
        seen.add(case_id)
        selected.append(row)
    if not selected:
        raise ValueError('Empty development subset')
    return selected, {'policy': document['selection'], 'manifest_sha256': sha256(path), 'all_dev_count': len(dev), 'selected_count': len(selected)}


def validate_au_qualification(config, groups, split_sha):
    if config.au_weight == 0:
        return None, {'enabled': False, 'reason': 'AU weight is zero; no AU supervisor is loaded'}
    path = Path(config.au_qualification).resolve()
    if ROOT not in path.parents:
        raise ValueError('AU qualification must belong to this independent experiment')
    report = read_json(path)
    if report.get('qualified') is not True or report.get('training_complete') is not True or report.get('test_used') is not False or report.get('subject_split_sha256') != split_sha:
        raise ValueError('AU supervisor has not qualified on this train/dev split')
    gate = report.get('gate', {})
    if gate.get('passed') is not True or not gate.get('checks') or not all(value is True for value in gate['checks'].values()):
        raise ValueError('AU qualification criteria are incomplete or failed')
    train_subjects, dev_subjects = set(report['train_subjects']), set(report['dev_subjects'])
    if not train_subjects or not dev_subjects or not train_subjects <= groups['train'] or not dev_subjects <= groups['dev'] or train_subjects & dev_subjects or (train_subjects | dev_subjects) & groups['test']:
        raise ValueError('AU supervisor subject leakage')
    expected = {'num_aus': 12, 'hidden_dim': 128, 'image_size': 224, 'intensity_scale': 5.0}
    if report.get('architecture') != expected:
        raise ValueError('AU supervisor architecture mismatch')
    preprocess = report.get('preprocess', {})
    if preprocess.get('sha256') != sha256(ROOT / 'modeling/au_regressor.py'):
        raise ValueError('AU preprocessing source changed')
    checkpoint = Path(report['checkpoint_path']).resolve()
    if ROOT not in checkpoint.parents or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Qualified AU checkpoint hash mismatch')
    return checkpoint, {'enabled': True, 'qualification_sha256': sha256(path), 'checkpoint_sha256': report['checkpoint_sha256'], 'gate': gate, 'architecture': expected, 'train_subjects': sorted(train_subjects), 'dev_subjects': sorted(dev_subjects), 'test_used': False}


def source_and_asset_bindings(au_info):
    from modeling.frozen_backbone import RV, FF, IP, ARC_WEIGHTS, FINEFACE_REPO, FACE_REPO, check_repo
    check_repo(FINEFACE_REPO, '6e9cc8df1180092891c1f55e4d4a06110759dac6')
    check_repo(FACE_REPO, '6e313885188e05740a7c40e12c1b9015c748bf08')
    sources = [Path(__file__), ROOT / 'modeling/frozen_backbone.py', ROOT / 'modeling/cross_modal_adapter.py', ROOT / 'modeling/token_residual_adapter.py', ROOT / 'modeling/safe_token_residual_adapter.py', ROOT / 'modeling/source_centered_adapter.py', ROOT / 'modeling/au_regressor.py', ROOT / 'prepare_training_cache.py']
    assets = {f'rv/{name}': RV / name for name in ('unet/config.json', 'unet/diffusion_pytorch_model.safetensors', 'vae/config.json', 'vae/diffusion_pytorch_model.safetensors', 'text_encoder/config.json', 'text_encoder/model.safetensors', 'scheduler/scheduler_config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json')}
    assets.update({'fineface/au_attention': FF / 'sd15_attn_processors.ckpt', 'fineface/au_encoder': FF / 'sd15_au_encoder.ckpt', 'faceid': IP / 'ip-adapter-faceid-portrait-v11_sd15.bin', 'iresnet100': ARC_WEIGHTS})
    identity_root = ROOT.parent / 'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l'
    assets.update({f'cached_faceid/{name}': identity_root / name for name in ('det_10g.onnx', 'w600k_r50.onnx')})
    packages = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'diffusers', 'transformers', 'Pillow')}
    result = {'sources': {str(p.relative_to(ROOT)): sha256(p) for p in sources}, 'public_assets': {key: {'sha256': sha256(path), 'bytes': path.stat().st_size} for key, path in assets.items()}, 'public_code_revisions': {'fineface': '6e9cc8df1180092891c1f55e4d4a06110759dac6', 'faceadapter': '6e313885188e05740a7c40e12c1b9015c748bf08'}, 'environment': {'python': platform.python_version(), 'packages': packages, 'platform': platform.platform()}, 'au_supervision': au_info}
    return result, sources


def run(config, output, resume=None):
    from modeling.frozen_backbone import FrozenConditionedDenoiser, FrozenImageTextEncoders
    from modeling.safe_token_residual_adapter import TokenAdapterConfig, SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter, RelativeRMSConfig, augment_condition_embeddings, condition_residuals
    from modeling.au_regressor import AUIntensityRegressor, AURegressorConfig
    train, dev, groups, bindings = load_data(config)
    source_anchor, source_anchor_binding = resolve_source_anchor(train, dev)
    verify_source_anchor(source_anchor, source_anchor_binding, bindings['source_anchor'])
    au_checkpoint, au_info = validate_au_qualification(config, groups, bindings['split'])
    extra, source_files = source_and_asset_bindings(au_info)
    bindings.update(extra)
    payload = load_checkpoint(resume) if resume else None
    if payload is not None:
        validate_resume(payload, config, bindings)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'snapshots').mkdir()
    for source in source_files:
        destination = output / 'snapshots' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    json_new(output / 'configuration.json', asdict(config))
    json_new(output / 'bindings.json', bindings)
    result = {'status': 'running', 'model_version': MODEL_VERSION, 'variant': config.variant, 'diagnostic': config.diagnostic, 'eligible_for_final_selection': not config.diagnostic, 'formal_sota_evidence': False, 'test_data_used': False, 'old_generated_images_or_metrics_used': False, 'au_supervision': au_info, 'parent_checkpoint': None if resume is None else {'path': str(Path(resume).resolve()), 'sha256': sha256(resume)}}
    try:
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if config.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        if config.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        device = torch.device(config.device)
        source_anchor = source_anchor.to(device)
        sync = torch.mps.synchronize if config.device == 'mps' else torch.cuda.synchronize if config.device == 'cuda' else lambda: None
        backbone = FrozenConditionedDenoiser(config.device, checkpointing=True)
        if backbone.noise_scheduler.config.prediction_type != 'epsilon' or backbone.noise_scheduler.config.num_train_timesteps != 1000:
            raise ValueError('This trainer requires the frozen epsilon/1000-step scheduler')
        for module in backbone.unet.modules():
            if isinstance(module, torch.nn.Dropout) and module.p != 0:
                raise ValueError('Checkpointing train-mode backbone must have zero dropout')
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
                raise ValueError('Frozen backbone BN must not update')
        encoders = FrozenImageTextEncoders(config.device) if config.identity_weight or config.au_weight else None
        au_model = None
        if au_checkpoint is not None:
            au_payload = torch.load(au_checkpoint, map_location='cpu', weights_only=True)
            au_model = AUIntensityRegressor(AURegressorConfig(**au_payload['config'])).to(device)
            au_model.load_state_dict(au_payload['model_state_dict'], strict=True)
            au_model.freeze_for_generator_loss()
        torch.manual_seed(config.seed)
        model_config = TokenAdapterConfig(direction=config.variant if config.variant != 'mlp' else 'bidirectional', ff_multiplier=3 if config.variant in ('condition_reads_image', 'image_reads_condition') else 2, readout=config.readout, patch_scale=config.patch_scale)
        model = create_centered_adapter(config.variant, model_config, config.device)
        trust = RelativeRMSConfig(enabled=config.trust_region, rho_au=config.rho_au, rho_ip=config.rho_ip, reference_rms_floor=config.reference_rms_floor)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
        frozen = [backbone] + ([encoders] if encoders is not None else []) + ([au_model] if au_model is not None else [])
        if any(p.requires_grad for module in frozen for p in module.parameters()):
            raise RuntimeError('A frozen model parameter is trainable')
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        if optimized != {id(p) for p in model.parameters() if p.requires_grad}:
            raise RuntimeError('Optimizer membership differs from trainable adapter')
        noise_rng = torch.Generator(device='cpu').manual_seed(config.noise_seed)
        order_rng = torch.Generator(device='cpu').manual_seed(config.order_seed)
        data_state = {'epoch': 0, 'order': [], 'cursor': 0}
        step = 0
        alpha_table = backbone.noise_scheduler.alphas_cumprod.to(device)
        fields_to_batch = ('image_patches', 'text_pooled', 'text_tokens', 'target_aus', 'source_aus', 'base_faceid', 'source_identity', 'latent_mean', 'latent_std')
        def batch(rows):
            return {key: torch.stack([row[key] for row in rows]).to(device) for key in fields_to_batch}
        def noises(b, generator, timesteps=None):
            clean = b['latent_mean'] + b['latent_std'] * torch.randn(b['latent_mean'].shape, generator=generator).to(device)
            noise = torch.randn(clean.shape, generator=generator).to(device)
            t = torch.randint(0, 1000, (len(clean),), generator=generator).to(device) if timesteps is None else torch.full((len(clean),), timesteps, device=device, dtype=torch.long)
            return noise, t, backbone.noise_scheduler.add_noise(clean, noise, t)
        def predict(b, noisy, t, enabled):
            base_au, base_ip = backbone.baseline_conditions(b['target_aus'], b['base_faceid'])
            if not enabled:
                return backbone(noisy, t, b['text_tokens'], base_au, base_ip), None, None
            adapted, effective, values = centered_batch_conditions(model, b, source_anchor, base_au, base_ip, trust)
            if not torch.equal(values['au_embedding'][..., :12], base_au[..., :12]):
                raise RuntimeError('Explicit AU prefix changed')
            return backbone(noisy, t, b['text_tokens'], values['au_embedding'], values['image_embedding']), adapted, effective
        def validate(label):
            model.eval()
            records = []
            with torch.no_grad():
                for row in dev:
                    b = batch([row])
                    for tvalue in config.dev_timesteps:
                        seed = int(fingerprint([config.dev_seed, row['case_id'], tvalue])[:15], 16)
                        noise, t, noisy = noises(b, torch.Generator(device='cpu').manual_seed(seed), tvalue)
                        baseline, _, _ = predict(b, noisy, t, False)
                        adapted, _, _ = predict(b, noisy, t, True)
                        records.append({'case_id': row['case_id'], 'subject': row['subject'], 'timestep': tvalue, 'noise_seed': seed, 'base_noise_mse': float((baseline-noise).square().mean().cpu()), 'adapter_noise_mse': float((adapted-noise).square().mean().cpu())})
            model.train()
            json_new(output / f'dev_{label}.json', {'records': records, 'base_mean': sum(r['base_noise_mse'] for r in records)/len(records), 'adapter_mean': sum(r['adapter_noise_mse'] for r in records)/len(records), 'scope': 'fixed teacher-forced denoising diagnostics, not final generation quality'})
        with torch.no_grad():
            b = batch([train[0]])
            _, t, noisy = noises(b, torch.Generator(device='cpu').manual_seed(999), 250)
            base, _, _ = predict(b, noisy, t, False)
            initial, _, _ = predict(b, noisy, t, True)
            zero_error = float((base-initial).abs().max().cpu())
            if not torch.equal(base, initial):
                raise RuntimeError('Fresh v4 does not exactly preserve the real frozen denoiser')
        result.update(zero_init_denoiser_max_abs_difference=zero_error, model_config=asdict(model_config), total_adapter_parameters=model.parameter_counts()['total'], trainable_parameters=model.parameter_counts()['trainable'], optimized_parameters=sum(p.numel() for group in optimizer.param_groups for p in group['params']), frozen_common_offset_parameters=model.slot_readout[-1].bias.numel(), source_anchor=source_anchor_binding, adapter_forward_calls_per_request=2, noise_timestep_sampling='uniform integers 0..999 inclusive', auxiliary_snr_window=[config.aux_snr_min, config.aux_snr_max], au_loss_scale='raw MSE divided by 25', residual_regularization='effective bounded source-centered residual; learned AU tail only; mean over actual IP tokens')
        if payload is not None:
            model.load_state_dict(payload['adapter'], strict=True)
            optimizer.load_state_dict(payload['optimizer'])
            step, data_state = payload['global_step'], payload['data_state']
            if payload['train_count'] != len(train) or payload['model_config'] != asdict(model_config):
                raise ValueError('Resume model/data shape changed')
            restore_rng(payload['rng'], noise_rng, order_rng, config.device)
        validate(f'entry_{step:08d}')
        started = time.monotonic()
        with (output / 'training_trace.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            while step < config.max_steps:
                indices = next_indices(data_state, len(train), config.batch_size, order_rng)
                rows = [train[index] for index in indices]
                b = batch(rows)
                noise, t, noisy = noises(b, noise_rng)
                optimizer.zero_grad(set_to_none=True)
                sync(); tick = time.monotonic()
                predicted, adapted, effective = predict(b, noisy, t, True)
                noise_loss = (predicted-noise).square().mean()
                alpha = alpha_table[t]
                snr = alpha / (1-alpha)
                eligible = (snr >= config.aux_snr_min) & (snr <= config.aux_snr_max)
                identity_loss = torch.zeros((), device=device)
                au_loss = torch.zeros((), device=device)
                x0 = decoded = identity = pred_au = None
                if encoders is not None and eligible.any().item():
                    a = alpha[eligible].view(-1, 1, 1, 1)
                    x0 = (noisy[eligible]-(1-a).sqrt()*predicted[eligible])/a.sqrt()
                    decoded = encoders.decode_latents(x0).clamp(-1, 1)
                    if config.identity_weight:
                        identity = encoders.identity_embedding(decoded)
                        identity_loss = (1-torch.nn.functional.cosine_similarity(identity, b['source_identity'][eligible])).sum()/len(rows)
                    if au_model is not None:
                        pred_au = au_model(decoded, input_range='minus_one_one')
                        au_loss = ((pred_au/5-b['target_aus'][eligible]/5).square().mean(-1)).sum()/len(rows)
                residual = effective['au_delta'][..., 12:].square().mean() + effective['image_delta'].square().mean()
                loss = noise_loss + config.identity_weight*identity_loss + config.au_weight*au_loss + config.residual_weight*residual
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                if any(p.grad is not None for module in frozen for p in module.parameters()):
                    raise RuntimeError('Frozen parameter accumulated a gradient')
                if model.slot_readout[-1].bias.grad is not None:
                    raise RuntimeError('Structurally cancelled common offset must stay frozen')
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise RuntimeError('Nonfinite adapter gradient')
                diagnostics = gradient_diagnostics(model) if step < 8 or (step+1) % config.checkpoint_every == 0 or step+1 == config.max_steps else None
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                if not torch.isfinite(gradient_norm).item():
                    raise RuntimeError('Nonfinite total gradient norm')
                optimizer.step(); sync()
                step += 1
                record = {'step': step, 'epoch': data_state['epoch'], 'case_ids': [r['case_id'] for r in rows], 'timesteps': t.detach().cpu().tolist(), 'snr': snr.detach().cpu().tolist(), 'auxiliary_eligible': eligible.detach().cpu().tolist(), 'loss': float(loss.detach().cpu()), 'noise_mse_all_samples': float(noise_loss.detach().cpu()), 'identity_loss_zero_outside_window': float(identity_loss.detach().cpu()), 'au_loss_zero_outside_window': float(au_loss.detach().cpu()), 'effective_residual_penalty': float(residual.detach().cpu()), 'unclipped_gradient_norm': float(gradient_norm.cpu()), 'frozen_grad_nonnull_count': 0, 'seconds': time.monotonic()-tick}
                record['gradient_diagnostics'] = diagnostics
                record['source_centered_noedit'] = ((b['target_aus'] == b['source_aus']).all(-1) & (b['text_pooled'] == source_anchor[None]).all(-1)).detach().cpu().tolist()
                log.write(json.dumps(record)+'\n'); print(json.dumps(record), flush=True)
                if step % config.validate_every == 0 or step == config.max_steps:
                    validate(f'step_{step:08d}')
                if step % config.checkpoint_every == 0 or step == config.max_steps:
                    save_checkpoint(output/f'checkpoint_{step:08d}.pt', model, optimizer, config, bindings, step, data_state, noise_rng, order_rng, len(train), asdict(model_config))
                del loss, predicted, adapted, effective, noise_loss, identity_loss, au_loss, residual, x0, decoded, identity, pred_au
        result.update(status='complete', global_step=step, epoch=data_state['epoch'], training_seconds=time.monotonic()-started, note='Train/dev completion only; diagnostic runs cannot select a final model.')
    except Exception as error:
        result.update(status='fail', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    json_new(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    config = TrainConfig(**read_json(args.config))
    output = args.output or ROOT / 'runs' / f'train_{MODEL_VERSION}_{config.variant}_{datetime.now().strftime("%Y%m%d%H%M")}'
    output = output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Choose a NEW run directory inside this independent experiment; resume also appends a new run')
    print('OUTPUT=' + str(output), flush=True)
    result = run(config, output, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
