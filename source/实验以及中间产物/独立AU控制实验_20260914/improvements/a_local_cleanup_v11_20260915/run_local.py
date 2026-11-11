"""V11 fixed source-only local cleanup of V10 last-epsilon clean latent;32dev."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V10 = HERE.parent / 'a_terminal_cleanup_v10_20260915'
V7 = HERE.parent / 'smooth_regions_v7_20260915'
V10_SHA = '3b11fb90c69276200906a156e7cd126c07727d393140829fce69858ce2f3d9a8'
MASK_MANIFEST = HERE / 'masks/first8/manifest.json'
MODES = {'local_lp_s050': .5, 'local_lp_s075': .75}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


need(sha(V10 / 'run_terminal.py') == V10_SHA, 'Frozen V10 implementation changed')
spec = importlib.util.spec_from_file_location('v11_private_frozen_v10', V10 / 'run_terminal.py')
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)
spec_mask = importlib.util.spec_from_file_location('v11_cleanup_geometry', HERE / 'cleanup_mask.py')
geometry = importlib.util.module_from_spec(spec_mask)
spec_mask.loader.exec_module(geometry)
finite, tensor_sha = v10.finite, v10.tensor_sha


def filter_metadata(sigma):
    need(sigma in MODES.values(), 'Only the two fixed local filters are authorized')
    radius = math.ceil(3 * sigma)
    return {'sigma_latent': sigma, 'radius': radius, 'kernel_size': 2 * radius + 1, 'padding': 'reflect',
            'normalization': 'conv(editmask*delta)/conv(editmask)', 'denominator_epsilon': 1e-8,
            'denominator_fallback': 'original_delta',
            'local_blend': 'clean + cleanupmask * (filtered_delta - original_delta)',
            'outside_cleanup_exact': True, 'final_projection': 'original_editmask_once'}


def local_filtered_clean(clean, source, edit_mask, cleanup_mask, sigma, torch):
    """Weighted normalized residual convolution, blended only on local support."""
    import torch.nn.functional as functional
    metadata = filter_metadata(sigma)
    for name, value in (('clean', clean), ('source', source), ('editmask', edit_mask), ('cleanupmask', cleanup_mask)):
        finite(value, torch, name)
    expected_mask = (clean.shape[0], 1, clean.shape[2], clean.shape[3])
    need(clean.ndim == 4 and clean.shape == source.shape and edit_mask.shape == cleanup_mask.shape == expected_mask,
         'Local cleanup shape mismatch')
    need(min(clean.shape[-2:]) > metadata['radius'], 'Spatial dimensions too small for reflect filter')
    need(edit_mask.min().item() >= 0 and edit_mask.max().item() <= 1 and
         cleanup_mask.min().item() >= 0 and cleanup_mask.max().item() <= 1, 'Invalid mask range')
    need(not torch.any((cleanup_mask > 0) & (edit_mask <= 0)).item(), 'Cleanup support must lie inside edit support')
    if not torch.any(cleanup_mask > 0).item():
        return clean.clone()
    delta = clean - source
    finite(delta, torch, 'original clean residual')
    radius = metadata['radius']
    axis = torch.arange(-radius, radius + 1, device=clean.device, dtype=clean.dtype)
    weights = torch.exp(-.5 * (axis / sigma) ** 2)
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None]
    pad = [radius] * 4
    numerator = functional.conv2d(functional.pad(edit_mask * delta, pad, mode='reflect'),
                                 kernel.repeat(clean.shape[1], 1, 1, 1), groups=clean.shape[1])
    denominator = functional.conv2d(functional.pad(edit_mask, pad, mode='reflect'), kernel)
    finite(numerator, torch, 'normalized numerator'); finite(denominator, torch, 'normalized denominator')
    candidate = numerator / denominator.clamp_min(metadata['denominator_epsilon'])
    filtered_delta = torch.where(denominator > metadata['denominator_epsilon'], candidate, delta)
    mixed = clean + cleanup_mask * (filtered_delta - delta)
    result = torch.where(cleanup_mask > 0, mixed, clean)
    finite(result, torch, 'local filtered clean latent')
    outside = (cleanup_mask == 0).expand_as(clean)
    need(torch.equal(result[outside], clean[outside]), 'Outside-local latent must remain exact clean')
    return result


class LocalController(v10.TerminalController):
    def __init__(self, unet, mode, source, mask, cross, torch, cleanup_mask, cleanup_ref, original_a_prefix):
        need(mode in MODES, 'Unknown local cleanup config')
        # Keep the inherited mode at terminal_last_eps: no extra t0 forward.
        super().__init__(unet, 'terminal_last_eps', source, mask, cross, torch)
        self.local_mode, self.cleanup_sigma = mode, MODES[mode]
        self.cleanup_mask, self.cleanup_ref = cleanup_mask, cleanup_ref
        self.original_a_prefix, self.a_prefix_verified = original_a_prefix, False

    def correct(self, latent, pipeline, prompt_embeds):
        clean = super().correct(latent, pipeline, prompt_embeds)
        result = local_filtered_clean(clean, self.source, self.mask, self.cleanup_mask, self.cleanup_sigma, self.torch)
        self.terminal_trace.update(mode=self.local_mode, base_mode='terminal_last_eps',
            base_terminal_clean_latent_sha256=tensor_sha(clean), post_filter_clean_latent_sha256=tensor_sha(result),
            filter=filter_metadata(self.cleanup_sigma), cleanup_mask=self.cleanup_ref,
            outside_cleanup_clean_exact=True)
        return result

    def record(self):
        return {**super().record(), 'cleanup_sigma': self.cleanup_sigma, 'cleanup_mask': self.cleanup_ref,
                'original_a_projection_prefix': self.original_a_prefix, 'a_prefix_verified': self.a_prefix_verified}


def load_verified_cleanup():
    import numpy as np
    need(MASK_MANIFEST.exists(), 'Prepare source-only V11 cleanup masks before generation')
    doc = json.loads(MASK_MANIFEST.read_text())
    need(doc['kind'] == 'v11_source_only_cleanup_masks' and doc['status'] == 'complete' and doc['case_count'] == 8,
         'Expected complete8case cleanup manifest')
    need(doc['source_only'] and not doc['test_used'] and not doc['target_geometry_used'] and not doc['generated_pixels_used'],
         'Source-only cleanup contract failed')
    need(doc['code']['sha256'] == sha(HERE / 'cleanup_mask.py') and
         doc['policy_sha256'] == geometry.canonical_sha(geometry.POLICY) and doc['policy'] == geometry.POLICY,
         'Frozen cleanup code/policy changed')
    for item in doc['inputs'].values():
        need(sha(item['path']) == item['sha256'], 'Cleanup manifest input changed')
    cases, records, edits, _ = geometry.verified_documents()
    entries = {entry['case_id']: entry for entry in doc['cases']}
    need(set(entries) == {case['case_id'] for case in cases}, 'Cleanup case set changed')
    arrays = {}
    for case in cases:
        entry = entries[case['case_id']]
        need(entry['source_sha256'] == case['source_real']['sha256'] and sha(entry['path']) == entry['sha256'],
             'Cleanup source/mask binding changed')
        need(sha(entry['receipt']['path']) == entry['receipt']['sha256'], 'Cleanup receipt changed')
        receipt = json.loads(Path(entry['receipt']['path']).read_text())
        edit_ref = edits[case['case_id']]['mask64']
        need(sha(edit_ref['path']) == edit_ref['sha256'], 'Frozen V7 mask changed')
        expected, expected_receipt, _ = geometry.build_mask(case, records[case['case_id']], np.load(edit_ref['path'], allow_pickle=False))
        actual = np.load(entry['path'], allow_pickle=False)
        need(np.array_equal(actual, expected) and all(receipt[k] == value for k, value in expected_receipt.items()),
             'Cleanup mask/receipt does not reproduce from fixed source geometry')
        need(receipt['source_image_unchanged'] and receipt['raster_protected_pixels_overlap'] == 0,
             'Source preservation/protected-region declaration failed')
        arrays[case['case_id']] = actual
    return entries, arrays


def local_inputs():
    original = v10.inputs_with_a()
    frozen = json.loads(geometry.V10_CASES.read_text())['cases']
    need(len(original) == len(frozen) == 8 and all(a == b for a, b in zip(original, frozen)), 'V10 original case fields changed')
    entries, _ = load_verified_cleanup()
    for case in frozen:
        case['cleanup_mask'] = entries[case['case_id']]
    return frozen


def original_a_prefix(case_id, family):
    return v10.a_records()[case_id, family + '_smooth_s075_a150']['projection_trace'][:14]


raw_kernel = v10.raw_kernel
source = v10.source
changes = [
    ("CONFIGS = [('terminal_last_eps', .75, 1.5), ('terminal_t0', .75, 1.5), ('terminal_t0_lp', .75, 1.5)]",
     "CONFIGS = [('local_lp_s050', .75, 1.5), ('local_lp_s075', .75, 1.5)]", 1),
    ("'v10_development_terminal_cases'", "'v11_development_local_cleanup_cases'", 1),
    ("'v10_terminal_diagnostic'", "'v11_local_cleanup_diagnostic'", 2),
    ("'v10_complete'", "'v11_complete'", 1),
    ("New v10 run required", "New v11 run required", 1),
    ("mask_refs = load_prepared_masks(mask_manifest, cases)",
     "mask_refs = load_prepared_masks(mask_manifest, cases)\n    cleanup_refs,cleanup_arrays=load_verified_cleanup()", 1),
    ("controller=TerminalController(backbone.unet,config,src,mask,cross,torch)",
     "controller=LocalController(backbone.unet,config,src,mask,cross,torch,torch.from_numpy(cleanup_arrays[c['case_id']])[None,None].to('mps'),cleanup_refs[c['case_id']],original_a_prefix(c['case_id'],method))", 1),
    ("verify_a_prefix(c['case_id'],method,history)",
     "verify_a_prefix(c['case_id'],method,history)\n                        controller.a_prefix_verified=True", 1),
    ("if len(results)==6:", "if len(results)==4:", 1),
    ("'planned':48", "'planned':32", 1),
]
for old, new, count in changes:
    need(source.count(old) == count, 'Ambiguous private V11 derivation: ' + old)
    source = source.replace(old, new)
core = {'__name__': 'v11_local_kernel', '__file__': str(Path(__file__).resolve()), 'V7': V7,
        'LocalController': LocalController, 'load_verified_cleanup': load_verified_cleanup,
        'original_a_prefix': original_a_prefix, 'verify_a_prefix': v10.verify_a_prefix}
exec(compile(source, str(v10.KERNEL), 'exec'), core)
original_save = core['save']


def save_metadata(path, doc):
    if Path(path).name == 'configuration.json':
        doc['configs'] = [{'name': mode, 'strength': .75, 'au_scale': 1.5, 'terminal_unet_calls': 0,
                           'cleanup_sigma': sigma} for mode, sigma in MODES.items()]
        doc.update(v10_implementation={'path': str(V10 / 'run_terminal.py'), 'sha256': V10_SHA},
            kernel_reference={'path': str(v10.KERNEL), 'sha256': v10.KERNEL_SHA},
            derived_kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
            cleanup_mask_manifest={'path': str(MASK_MANIFEST), 'sha256': sha(MASK_MANIFEST)},
            cleanup_geometry_code={'path': str(HERE / 'cleanup_mask.py'), 'sha256': sha(HERE / 'cleanup_mask.py')},
            cleanup_policy_sha256=geometry.canonical_sha(geometry.POLICY),
            frozen_v10_cases={'path': str(geometry.V10_CASES), 'sha256': geometry.CASES_SHA},
            original_a_result={'path': str(v10.A_RESULT), 'sha256': v10.A_RESULT_SHA},
            hypothesis='Fixed source-only local latent residual regularization, not anatomical causal segmentation or RGB retouching',
            terminal_formula='V10 terminal_last_eps clean=(L-sigma0*CFG_epsilon_t50)/alpha0',
            local_cleanup='clean + cleanupmask*(normalized_filtered_delta-original_delta); outside cleanup exact clean',
            filter_configs={mode: filter_metadata(sigma) for mode, sigma in MODES.items()},
            denoising_steps=15, terminal_unet_calls=0, total_unet_calls=15,
            projection='First14 original A next_t projection; last_eps clean/local cleanup then original editmask*clean+(1-editmask)*source',
            scheduler_sigmas_mutated=False, terminal_scheduler_step_calls=0, terminal_add_noise_calls=0,
            terminal_output_is_hypothesis=True)
    original_save(path, doc)


core['inputs'], core['save'] = local_inputs, save_metadata
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'prepared_no_models', 'cases': len(local_inputs()), 'planned': 32,
                          'configs': list(MODES), 'deis_calls_per_image': 15, 'extra_terminal_calls': [0, 0]}))
    else:
        need(args.run is not None, 'Explicit new run directory required')
        core['execute'](args.run.resolve(), V7 / 'masks/first8/manifest.json')
