"""Fixed V11 completion contract. No model or image import at module load."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
EXP = COMPLETION.parent
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
VARIANTS = ['base', 'bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp']
SEEDS = [101, 202, 303]
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
POLICY = {
    'version': 'selected_v11_complete_grid_v1', 'variant_order': VARIANTS,
    'sampling_seeds': SEEDS, 'checkpoint_step': 200, 'width': 512, 'height': 512,
    'source_init_strength': .75, 'nominal_steps': 20, 'actual_timesteps': TIMESTEPS,
    'scheduler': 'DEISMultistepScheduler', 'guidance_scale': 7.5, 'au_scale': 1.5,
    'ip_scale': .9, 'negative_prompt': '', 'cleanup_sigma': .5,
    'terminal': 'last_t50_cfg_epsilon', 'deis_unet_calls': 15, 'terminal_unet_calls': 0,
    'source_latent': 'VAE posterior mean times scaling_factor',
    'projection': 'first14 next_t source plus same noise; terminal V11 clean then source projection once',
    'dtype': 'torch.float32', 'source_device': 'mps', 'generation_device': 'mps',
    'target_pixels_used': False, 'target_latents_computed': False,
    'zero_edit': 'source VAE reconstruction by construction, not learned identity evidence',
}
CKPTS = {
    'bidirectional': ('v4_slot_2k_bidir_step200_202609141920', '7fcc25b704fca8c0a4e46a171096df8fea940b40a246d4939193e64f533b4e30'),
    'condition_reads_image': ('v4_slot_2k_condition_reads_image_step200_202609142024', '969def9470fb6a8ba4ff5b37e11b601144101e563e25ff3c26e09e4fdb71396b'),
    'image_reads_condition': ('v4_slot_2k_image_reads_condition_step200_202609142032', '99b565be708493e23a5a5d11829353cbc88cba3c2d78a388be49534703bd695f'),
    'mlp': ('v4_slot_2k_mlp_step200_202609142000', 'f58336567246ecd2f6818ea4ae30f947914af4e9c2296177b86169b95d6aa6f5'),
}
PINS = {
    'v11': ('improvements/a_local_cleanup_v11_20260915/run_local.py', '7f872acfb9d6114c4da4e25ee35c4157727be1c3a30822227c64c77e1c68f78a'),
    'v11_geometry': ('improvements/a_local_cleanup_v11_20260915/cleanup_mask.py', '913623ca6c2dd76f0a3bca5a5316a6fb10339a3fe0d506944286fba32ee41be9'),
    'v7_geometry': ('improvements/smooth_regions_v7_20260915/region_mask.py', 'ca173d310425c6c97ba4350a21cdd5b3d4f08a961fdb87b9c134e1b3029ef824'),
    'source_runtime': ('final_test/prepare_source_cache.py', '3f12b67deefdf890696739e469fa29647485d3f7e9e45a83a45995970d6dadae'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verified_ref(value, label):
    path = Path(value['path']).resolve()
    need(EXP in path.parents, label + ' must be inside experiment')
    need(sha(path) == value['sha256'], label + ' SHA changed')
    return path


def pin(role):
    relative, expected = PINS[role]
    path = EXP / relative
    need(sha(path) == expected, role + ' fixed implementation changed')
    return path


def module(role):
    path = pin(role)
    if role == 'source_runtime':
        sys.path.insert(0, str(EXP / 'final_test'))
        sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location('completion_fixed_' + role, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def execution_spec():
    checkpoints = {}
    for variant, (folder, expected) in CKPTS.items():
        path = EXP / 'runs' / folder / 'checkpoint_00000200.pt'
        need(sha(path) == expected, variant + ' checkpoint changed')
        seal = read(path.with_suffix('.json'))
        need(seal['checkpoint_sha256'] == expected and seal['step'] == 200 and seal['variant'] == variant,
             'Early200 seal mismatch')
        checkpoints[variant] = ref(path)
    return {'policy': POLICY, 'checkpoints': checkpoints,
            'execution_sources': {name: ref(HERE / name) for name in ('common.py', 'geometry.py', 'run_grid.py', 'prepare_cache.py')},
            'reused_sources': {role: ref(pin(role)) for role in PINS}}


def validate_labels(document):
    cases = document['cases']
    need(len(cases) == 32 and len({c['case_id'] for c in cases}) == 32, 'Exactly32 frozen cases required')
    need(len({c['subject'] for c in cases}) == 8, 'Exactly8 subjects required')
    for case in cases:
        need(re.fullmatch(r'[A-Za-z0-9_-]+', case['case_id']), 'Unsafe case ID')
        need(case['split'] == 'postfreeze_case_validation', 'Explicit postfreeze validation split required')
        need(case['au_order'] == AUS, 'AU order changed')
        source, target = case['source_au_vector_0_5'], case['target_au_vector_0_5']
        need(len(source) == len(target) == 12 and not any(source), 'Only known zeroAU source supported')
        need(all(type(v) in (int, float) and 0 <= v <= 5 for v in target), 'Invalid targetAU labels')
    return cases


def load_freeze(path, expected_sha):
    need(sha(path) == expected_sha, 'Explicit freeze SHA mismatch')
    document = read(path)
    need(document['kind'] == 'completion_v11_frozen_protocol' and document['status'] == 'frozen', 'Freeze must be explicit')
    need(document['generation'] == execution_spec(), 'Generation policy/implementation/checkpoint differs from freeze')
    label_path = verified_ref(document['label_plan'], 'frozen labels')
    cases = validate_labels(read(label_path))
    need(document['case_ids'] == [c['case_id'] for c in cases], 'Frozen case order changed')
    return document, cases


def environment(device):
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(EXP / 'cache/huggingface'),
                      TORCH_HOME=str(EXP / 'cache/torch'), MPLCONFIGDIR=str(EXP / 'cache/matplotlib'))
    sys.path.insert(0, str(EXP))
    import torch
    need(torch.__version__.split('+')[0] == '2.1.1', 'Frozen torch2.1.1 required')
    need(device in ('cpu', 'mps'), 'Unsupported device')
    if device == 'mps':
        need(torch.backends.mps.is_available(), 'MPS required for frozen generation')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch


def load_anchor():
    """Reads TRAIN/DEV tensors, never final cases or target pixels."""
    import torch
    from train_centered_adapter import load_checkpoint, resolve_source_anchor, verify_source_anchor
    path = EXP / 'runs' / CKPTS['bidirectional'][0] / 'checkpoint_00000200.pt'
    payload = load_checkpoint(path)
    features = EXP / 'training_cache/train2k_batched_202609141853/features.pt'
    need(sha(features) == payload['bindings']['cache'], 'Training anchor features changed')
    data = torch.load(features, map_location='cpu', weights_only=True)['samples']
    need(all(s['split'] in ('train', 'dev') for s in data), 'Unexpected anchor role')
    anchor, binding = resolve_source_anchor([s for s in data if s['split'] == 'train'], [s for s in data if s['split'] == 'dev'])
    verify_source_anchor(anchor, binding, payload['bindings']['source_anchor'])
    return anchor, binding, payload
