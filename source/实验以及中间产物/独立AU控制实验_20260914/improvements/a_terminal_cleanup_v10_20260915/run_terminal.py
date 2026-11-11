"""V10 controlled terminal-cleanup hypothesis; frozen A trajectory, no RGB editing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V6 = HERE.parent / 'region_preserving_v6_20260915'
V7 = HERE.parent / 'smooth_regions_v7_20260915'
V9_CASES = HERE.parent / 'a_texture_schedule_v9_20260915/runs/late_au_202609151925/cases.json'
V9_CASES_SHA = '2f7221ac4df6e909eb18af4cd313566e0422a82591310c9715bb904b98224c94'
A_RESULT = V7 / 'runs/smooth_202609151710/result.json'
A_RESULT_SHA = 'e59ce11ad87ff6bba91c8131019997833fee0f482a3df364e2d6f28e80aa079d'
KERNEL = V6 / 'run_region_diagnostic.py'
KERNEL_SHA = '85236dfc8803fb66d8f2440a240475a10c59f189ab3df059a44389f13db1fcec'
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
MODES = ('terminal_last_eps', 'terminal_t0', 'terminal_t0_lp')
FILTER = {'sigma_latent': .5, 'radius': 2, 'kernel_size': 5, 'padding': 'reflect',
          'normalization': 'conv(mask*delta)/conv(mask)', 'denominator_epsilon': 1e-8,
          'outside_mask_zero': True}


def need(value, message):
    if not value:
        raise ValueError(message)


def finite(tensor, torch, label):
    need(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all().item(), 'Nonfinite or invalid ' + label)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cfg_epsilon(raw, reference, torch, guidance=7.5):
    finite(raw, torch, 'raw epsilon')
    need(raw.shape == (reference.shape[0] * 2, *reference.shape[1:]), 'CFG epsilon batch/shape mismatch')
    need(guidance == 7.5, 'Frozen CFG7.5 required')
    unconditional, conditional = raw.chunk(2)
    result = unconditional + guidance * (conditional - unconditional)
    finite(result, torch, 'CFG epsilon')
    return result


def terminal_coefficients(scheduler, reference, torch):
    need(scheduler.config.prediction_type == 'epsilon' and scheduler.config.thresholding is False,
         'Terminal clean formula requires epsilon without thresholding')
    rho = scheduler.sigmas[-1].detach().to(device=reference.device, dtype=reference.dtype)
    finite(rho, torch, 'terminal rho')
    need(rho.numel() == 1 and rho.item() > 0, 'Expected actual nonzero DEIS terminal rho')
    alpha, sigma = scheduler._sigma_to_alpha_sigma_t(rho)
    finite(alpha, torch, 'terminal alpha'); finite(sigma, torch, 'terminal sigma')
    need(alpha.item() > 0 and sigma.item() > 0, 'Terminal VP coefficients must be positive')
    need(torch.allclose(sigma / alpha, rho, atol=1e-7, rtol=1e-6), 'rho is not sigma/alpha')
    return alpha, sigma, rho


def estimate_clean(latent, epsilon, alpha, sigma, torch):
    finite(latent, torch, 'terminal input latent'); finite(epsilon, torch, 'terminal epsilon')
    finite(alpha, torch, 'clean alpha'); finite(sigma, torch, 'clean sigma')
    need(latent.shape == epsilon.shape and alpha.numel() == sigma.numel() == 1 and alpha.item() > 0 and sigma.item() >= 0,
         'Clean estimate shape/coefficient mismatch')
    clean = (latent - sigma * epsilon) / alpha
    finite(clean, torch, 'clean estimate')
    return clean


def filtered_clean_residual(clean, source, mask, torch):
    """Only latent residual smoothing, normalized by the fixed editing support."""
    import torch.nn.functional as functional
    for name, value in (('clean', clean), ('source', source), ('mask', mask)):
        finite(value, torch, 'filter ' + name)
    need(clean.shape == source.shape and clean.ndim == 4 and mask.shape ==
         (clean.shape[0], 1, clean.shape[2], clean.shape[3]), 'Filter latent/mask shape mismatch')
    need(mask.min().item() >= 0 and mask.max().item() <= 1 and min(clean.shape[-2:]) > FILTER['radius'],
         'Filter mask range or spatial shape invalid')
    if not torch.any(mask > 0).item():
        return source.clone()
    axis = torch.arange(-FILTER['radius'], FILTER['radius'] + 1, device=clean.device, dtype=clean.dtype)
    weights = torch.exp(-.5 * (axis / FILTER['sigma_latent']) ** 2)
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None]
    pad = [FILTER['radius']] * 4
    delta = clean - source
    finite(delta, torch, 'unfiltered latent residual')
    numerator = functional.conv2d(functional.pad(mask * delta, pad, mode='reflect'),
                                 kernel.repeat(clean.shape[1], 1, 1, 1), groups=clean.shape[1])
    denominator = functional.conv2d(functional.pad(mask, pad, mode='reflect'), kernel)
    finite(numerator, torch, 'filtered numerator'); finite(denominator, torch, 'filtered denominator')
    normalized = numerator / denominator.clamp_min(FILTER['denominator_epsilon'])
    normalized = torch.where((denominator > FILTER['denominator_epsilon']) & (mask > 0), normalized,
                             torch.zeros_like(normalized))
    result = source + normalized
    finite(result, torch, 'filtered clean latent')
    return result


class TerminalController:
    """Read-only raw-UNet capture; the extra call cannot overwrite the t50 record."""
    def __init__(self, unet, mode, source, mask, cross, torch):
        need(mode in MODES, 'Unknown terminal mode')
        self.unet, self.mode, self.source, self.mask, self.cross, self.torch = unet, mode, source, mask, cross, torch
        self.phase, self.calls, self.terminal_trace = 'deis', [], None
        self.last_raw = self.last_text = self.terminal_raw = None
        self.last_kwargs = None
        self.guard = unet.register_forward_hook(self.capture, with_kwargs=True)

    def capture(self, module, args, kwargs, output):
        torch = self.torch
        raw = output[0] if isinstance(output, tuple) else output.sample
        finite(args[0], torch, 'UNet input'); finite(raw, torch, 'UNet raw epsilon')
        text = kwargs['encoder_hidden_states']
        finite(text, torch, 'UNet text tokens')
        need(args[0].shape == raw.shape == (self.source.shape[0] * 2, *self.source.shape[1:]), 'UNet CFG shape mismatch')
        need(text.shape[0] == 2 * self.source.shape[0], 'Expected already concatenated negative/positive text')
        need(kwargs['cross_attention_kwargs'] is self.cross and self.cross['au_scale'] == 1.5
             and self.cross['ip_scale'] == .9, 'Original A cross conditions/scales changed')
        t = int(args[1])
        self.calls.append({'phase': self.phase, 't': t, 'au_scale': 1.5, 'cfg_batch': int(raw.shape[0]),
                           'raw_epsilon_sha256': tensor_sha(raw), 'prompt_embeds_sha256': tensor_sha(text)})
        if self.phase == 'deis':
            self.last_raw, self.last_text = raw.detach().clone(), text.detach().clone()
            self.last_kwargs = {k: kwargs.get(k) for k in ('timestep_cond', 'added_cond_kwargs')}
        else:
            need(self.phase == 'terminal' and t == 0, 'Unexpected extra UNet call')
            self.terminal_raw = raw.detach().clone()

    def correct(self, latent, pipeline, prompt_embeds):
        torch = self.torch
        need([r['t'] for r in self.calls] == TIMESTEPS and all(r['phase'] == 'deis' for r in self.calls),
             'Terminal cleanup requires exactly the original fifteen DEIS UNet calls')
        need(pipeline.do_classifier_free_guidance and pipeline.guidance_scale == 7.5
             and pipeline.cross_attention_kwargs is self.cross, 'Extra forward CFG/cross contract changed')
        need(self.last_kwargs == {'timestep_cond': None, 'added_cond_kwargs': None},
             'This frozen A pipeline requires no timestep_cond or extra image condition')
        need(prompt_embeds.shape == self.last_text.shape and torch.equal(prompt_embeds, self.last_text),
             'Callback must provide the same already-concatenated negative/positive text')
        alpha, sigma, rho = terminal_coefficients(pipeline.scheduler, latent, torch)
        index_before = pipeline.scheduler.step_index
        sigma_before = tensor_sha(pipeline.scheduler.sigmas)
        model_history_before = [None if v is None else tensor_sha(v) for v in pipeline.scheduler.model_outputs]
        epsilon = cfg_epsilon(self.last_raw, latent, torch)
        if self.mode != 'terminal_last_eps':
            self.phase = 'terminal'
            try:
                model_input = pipeline.scheduler.scale_model_input(torch.cat([latent, latent]), 0)
                finite(model_input, torch, 'terminal UNet input')
                raw = pipeline.unet(model_input, 0, encoder_hidden_states=prompt_embeds,
                    timestep_cond=None, cross_attention_kwargs=self.cross, added_cond_kwargs=None, return_dict=False)[0]
                need(self.terminal_raw is not None and torch.equal(raw, self.terminal_raw), 'Extra epsilon capture mismatch')
                epsilon = cfg_epsilon(raw, latent, torch)
            finally:
                self.phase = 'deis'
        need(pipeline.scheduler.step_index == index_before and tensor_sha(pipeline.scheduler.sigmas) == sigma_before
             and [None if v is None else tensor_sha(v) for v in pipeline.scheduler.model_outputs] == model_history_before,
             'Terminal cleanup must not mutate DEIS time/history/sigmas')
        clean = estimate_clean(latent, epsilon, alpha, sigma, torch)
        filtered = filtered_clean_residual(clean, self.source, self.mask, torch) if self.mode == 'terminal_t0_lp' else clean
        self.terminal_trace = {'mode': self.mode, 'last_deis_timestep': 50,
            'rho0': float(rho.item()), 'alpha0': float(alpha.item()), 'sigma0': float(sigma.item()),
            'epsilon_source': 'last_t50' if self.mode == 'terminal_last_eps' else 'extra_t0',
            'input_latent_sha256': tensor_sha(latent), 'epsilon_cfg_sha256': tensor_sha(epsilon),
            'last_deis_raw_epsilon_sha256': tensor_sha(self.last_raw),
            'terminal_raw_epsilon_sha256': None if self.terminal_raw is None else tensor_sha(self.terminal_raw),
            'clean_latent_sha256': tensor_sha(clean), 'post_filter_clean_latent_sha256': tensor_sha(filtered),
            'extra_cfg': None if self.mode == 'terminal_last_eps' else 7.5,
            'filter': dict(FILTER) if self.mode == 'terminal_t0_lp' else None,
            'scheduler_step_index_before': index_before, 'scheduler_step_index_after': pipeline.scheduler.step_index}
        return filtered

    def finalize_projection(self, projected):
        finite(projected, self.torch, 'final projected latent')
        need(self.terminal_trace is not None, 'Missing terminal cleanup trace')
        self.terminal_trace['projected_latent_sha256'] = tensor_sha(projected)
        if not self.torch.any(self.mask > 0).item():
            need(self.torch.equal(projected, self.source), 'Empty mask must exactly restore source latent')

    def assert_complete(self):
        expected = 0 if self.mode == 'terminal_last_eps' else 1
        need([r['t'] for r in self.calls if r['phase'] == 'deis'] == TIMESTEPS, 'Original DEIS call trace changed')
        need([r['t'] for r in self.calls if r['phase'] == 'terminal'] == ([0] if expected else []), 'Wrong extra call count')
        need(self.terminal_trace and 'projected_latent_sha256' in self.terminal_trace, 'Incomplete terminal trace')

    def record(self):
        deis = sum(r['phase'] == 'deis' for r in self.calls)
        extra = sum(r['phase'] == 'terminal' for r in self.calls)
        return {'denoising_steps': deis, 'deis_unet_calls': deis, 'terminal_unet_calls': extra,
                'total_unet_calls': len(self.calls), 'unet_call_trace': self.calls, 'terminal_trace': self.terminal_trace}

    def close(self):
        self.guard.remove()


raw_kernel = KERNEL.read_bytes()
need(hashlib.sha256(raw_kernel).hexdigest() == KERNEL_SHA, 'Frozen V6 kernel changed')
source = raw_kernel.decode()
changes = [
    ("CONFIGS = [('masked_s055_a100', .55, 1.), ('masked_s075_a100', .75, 1.), ('masked_s075_a150', .75, 1.5)]",
     "CONFIGS = [('terminal_last_eps', .75, 1.5), ('terminal_t0', .75, 1.5), ('terminal_t0_lp', .75, 1.5)]", 1),
    ("'v6_development_region_preserving_cases'", "'v10_development_terminal_cases'", 1),
    ("'v6_region_preserving_diagnostic'", "'v10_terminal_diagnostic'", 2),
    ("'v6_complete'", "'v10_complete'", 1),
    ("New v6 run required", "New v10 run required", 1),
    ("HERE/'region_mask.py'", "V7/'region_mask.py'", 2),
    ("import numpy as np\n    from region_mask import build_mask",
     "import numpy as np\n    sys.path.insert(0,str(V7))\n    from region_mask import build_mask", 1),
    ("projected=project_latents(kwargs['latents'],src,noise,mask,pipeline.scheduler,nt,torch)",
     "terminal_input=kwargs['latents']\n                        if nt is None:\n                            terminal_input=controller.correct(terminal_input,pipeline,kwargs['prompt_embeds'])\n                        projected=project_latents(terminal_input,src,noise,mask,pipeline.scheduler,nt,torch)\n                        if nt is None: controller.finalize_projection(projected)", 1),
    ("started=time.monotonic()\n                    try:",
     "started=time.monotonic()\n                    controller=TerminalController(backbone.unet,config,src,mask,cross,torch)\n                    try:", 1),
    ("callback_on_step_end_tensor_inputs=['latents']", "callback_on_step_end_tensor_inputs=['latents','prompt_embeds']", 1),
    ("need(len(history)==int(20*strength),'Wrong actual step count')",
     "need(len(history)==int(20*strength),'Wrong actual step count')\n                        controller.assert_complete()\n                        verify_a_prefix(c['case_id'],method,history)", 1),
    ("row.update(seconds=time.monotonic()-started,actual_steps=len(history),actual_timesteps=[r['t'] for r in history],projection_trace=history)",
     "finally:\n                        controller.close()\n                    row.update(seconds=time.monotonic()-started,actual_steps=len(history),actual_timesteps=[r['t'] for r in history],projection_trace=history,**controller.record())", 1),
]
for old, new, count in changes:
    need(source.count(old) == count, 'Ambiguous V10 kernel derivation: ' + old)
    source = source.replace(old, new)
core = {'__name__': 'v10_terminal_kernel', '__file__': str(Path(__file__).resolve()),
        'V7': V7, 'TerminalController': TerminalController}
exec(compile(source, str(KERNEL), 'exec'), core)
original_inputs, original_save = core['inputs'], core['save']


def a_records():
    need(core['sha'](A_RESULT) == A_RESULT_SHA, 'Original A result changed')
    doc = core['v5'].read(A_RESULT)
    need(doc['status'] == 'complete' and doc['test_data_used'] is False, 'Original A must be complete dev data')
    return {(r['case_id'], r['method']): r for r in doc['records']}


def inputs_with_a():
    original = original_inputs()
    need(core['sha'](V9_CASES) == V9_CASES_SHA, 'Frozen V9 case fields changed')
    doc = core['v5'].read(V9_CASES)
    need(doc['kind'] == 'v9_development_late_au_cases' and doc['case_count'] == 8 and doc['test_data_used'] is False,
         'Expected all eight original development cases')
    cases, previous = doc['cases'], {c['case_id']: c for c in original}
    need({c['case_id'] for c in cases} == set(previous), 'Frozen cases differ')
    records = a_records()
    for c in cases:
        need(all(c[k] == v for k, v in previous[c['case_id']].items()), 'Original case fields changed')
        for family in ('baseline', 'bidirectional'):
            row = records[c['case_id'], family + '_smooth_s075_a150']
            need(row['status'] == 'complete' and row['strength'] == .75 and row['au_scale'] == 1.5
                 and row['actual_steps'] == 15 and row['actual_timesteps'] == TIMESTEPS, 'Original A protocol changed')
            need(c['a_' + family] == {'path': row['output'], 'sha256': row['sha256']}
                 and core['sha'](row['output']) == row['sha256'], 'Original A reference image changed')
    return cases


def verify_a_prefix(case_id, family, history):
    prior = a_records()[case_id, family + '_smooth_s075_a150']['projection_trace']
    need(history[:14] == prior[:14], 'First fourteen projected DEIS states must exactly match original A')


def save_metadata(path, doc):
    if Path(path).name == 'configuration.json':
        doc['configs'] = [{'name': mode, 'strength': .75, 'au_scale': 1.5,
                           'terminal_unet_calls': 0 if mode == 'terminal_last_eps' else 1} for mode in MODES]
        doc.update(kernel_reference={'path': str(KERNEL), 'sha256': KERNEL_SHA},
            derived_kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
            original_a_result={'path': str(A_RESULT), 'sha256': A_RESULT_SHA},
            frozen_v9_cases={'path': str(V9_CASES), 'sha256': V9_CASES_SHA},
            hypothesis='Controlled terminal cleanup diagnostic, not a claimed DEIS implementation bug fix',
            terminal_formula='clean=(L-sigma0*epsilon_cfg)/alpha0; alpha0,sigma0=DEIS._sigma_to_alpha_sigma_t(sigmas[-1])',
            terminal_input='Last DEIS output before final mask/source projection; actual nonzero terminal rho',
            terminal_filter=dict(FILTER), denoising_steps=15,
            projection='First14 original next_t projection; terminal cleanup then original mask*clean+(1-mask)*source',
            scheduler_sigmas_mutated=False, terminal_scheduler_step_calls=0, terminal_add_noise_calls=0,
            terminal_output_is_hypothesis=True)
    original_save(path, doc)


core['inputs'], core['save'], core['verify_a_prefix'] = inputs_with_a, save_metadata, verify_a_prefix
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly run 48 new dev generations after review')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'prepared_no_models', 'cases': len(inputs_with_a()), 'planned': 48,
                          'configs': list(MODES), 'deis_calls_per_image': 15, 'extra_terminal_calls': [0, 1, 1]}))
    else:
        need(args.run is not None, 'Explicit new run directory required')
        core['execute'](args.run.resolve(), V7 / 'masks/first8/manifest.json')
