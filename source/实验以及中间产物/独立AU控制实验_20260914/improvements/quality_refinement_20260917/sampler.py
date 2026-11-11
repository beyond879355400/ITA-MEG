"""Source-only V11 quality probes; no model imports or loading at import time.

Public interface: sample(runtime, prepared, seed, *, guidance, steps).
The caller owns the already-loaded legacy Runtime and exclusive MPS access.
Only CFG and nominal step count vary. All files are append-only in this task.
"""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
STRENGTH = .75
AU_SCALE = 1.5
IP_SCALE = .9
CLEANUP_SIGMA = .5


def need(value, message):
    if not value:
        raise ValueError(message)


def parameters(guidance, steps):
    need(isinstance(guidance, (float, int)) and not isinstance(guidance, bool)
         and math.isfinite(guidance) and guidance > 1,
         'Quality probes require finite CFG > 1')
    need(type(steps) is int and steps in (20, 40), 'Authorized nominal steps are20or40')
    return float(guidance), steps


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(tensor, torch, label):
    need(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all().item(),
         'Nonfinite or invalid ' + label)


def cfg_epsilon(raw, reference, torch, guidance):
    finite(raw, torch, 'raw epsilon')
    need(raw.shape == (2 * reference.shape[0], *reference.shape[1:]), 'CFG batch/shape mismatch')
    unconditional, conditional = raw.chunk(2)
    # Preserve the exact legacy arithmetic order for CFG7.5 parity.
    result = unconditional + guidance * (conditional - unconditional)
    finite(result, torch, 'guided epsilon')
    return result


def actual_timesteps(pipeline, steps, device):
    need(getattr(pipeline.scheduler, 'order', None) == 1, 'Expected DEIS pipeline order1')
    times, count = pipeline.get_timesteps(steps, STRENGTH, device)
    values = [int(t) for t in times]
    need(count == int(steps * STRENGTH) and len(values) == count,
         'Unexpected strength/actual timestep count')
    need(values and all(a > b >= 0 for a, b in zip(values, values[1:])),
         'Expected strictly decreasing nonnegative timesteps')
    return times, values


class QualityController:
    """Capture every native UNet forward and reuse the last full-CFG epsilon."""

    def __init__(self, runtime, prepared, guidance, steps):
        self.runtime, self.torch, self.pipe = runtime, runtime.torch, runtime.pipe
        self.source, self.edit = prepared['source'], prepared['edit']
        self.cleanup, self.cross = prepared['cleanup'], prepared['cross']
        self.guidance, self.steps = parameters(guidance, steps)
        self.calls, self.values, self.times = [], None, None
        self.last_raw = self.last_text = self.last_kwargs = self.terminal_trace = None
        self.history = []
        self.guard = runtime.backbone.unet.register_forward_hook(self.capture, with_kwargs=True)

    def bind(self):
        if self.values is None:
            self.times, self.values = actual_timesteps(self.pipe, self.steps, self.source.device)
            self.times = self.times.clone()
        return self.times, self.values

    def capture(self, module, args, kwargs, output):
        torch = self.torch
        times, values = self.bind()
        raw = output[0] if isinstance(output, tuple) else output.sample
        finite(args[0], torch, 'UNet input'); finite(raw, torch, 'UNet raw epsilon')
        text = kwargs['encoder_hidden_states']
        finite(text, torch, 'UNet text')
        need(args[0].shape == raw.shape == (self.source.shape[0] * 2, *self.source.shape[1:]),
             'Native UNet CFG input/output shape changed')
        need(text.shape[0] == 2 * self.source.shape[0], 'Expected negative/positive text batch')
        need(kwargs['cross_attention_kwargs'] is self.cross
             and self.cross['au_scale'] == AU_SCALE and self.cross['ip_scale'] == IP_SCALE,
             'AU/IP conditions changed')
        need(self.pipe.do_classifier_free_guidance and self.pipe.guidance_scale == self.guidance,
             'Pipeline CFG differs from terminal CFG')
        t = int(args[1]); index = len(self.calls)
        need(index < len(values) and t == values[index], 'Unexpected UNet timestep/call')
        self.calls.append({'phase': 'deis', 't': t, 'au_scale': AU_SCALE,
                           'guidance_scale': self.guidance, 'cfg_batch': int(raw.shape[0]),
                           'raw_epsilon_sha256': tensor_sha(raw),
                           'prompt_embeds_sha256': tensor_sha(text)})
        self.last_raw, self.last_text = raw.detach().clone(), text.detach().clone()
        self.last_kwargs = {k: kwargs.get(k) for k in ('timestep_cond', 'added_cond_kwargs')}

    def correct(self, latent, pipeline, prompt_embeds):
        torch = self.torch
        _, values = self.bind()
        need([r['t'] for r in self.calls] == values, 'Terminal correction before complete native trajectory')
        need(pipeline is self.pipe and pipeline.cross_attention_kwargs is self.cross
             and pipeline.guidance_scale == self.guidance, 'Terminal pipeline/config changed')
        need(self.last_kwargs == {'timestep_cond': None, 'added_cond_kwargs': None},
             'Unexpected extra UNet conditioning')
        need(prompt_embeds.shape == self.last_text.shape and torch.equal(prompt_embeds, self.last_text),
             'Terminal text differs from the actual last forward')
        # Reuse the original tested numerical operations, not its CFG7.5 controller.
        v10 = self.runtime.v11.v10
        alpha, sigma, rho = v10.terminal_coefficients(pipeline.scheduler, latent, torch)
        scheduler_index = pipeline.scheduler.step_index
        sigma_sha = tensor_sha(pipeline.scheduler.sigmas)
        old_history = [None if x is None else tensor_sha(x) for x in pipeline.scheduler.model_outputs]
        epsilon = cfg_epsilon(self.last_raw, latent, torch, self.guidance)
        clean = v10.estimate_clean(latent, epsilon, alpha, sigma, torch)
        result = self.runtime.v11.local_filtered_clean(
            clean, self.source, self.edit, self.cleanup, CLEANUP_SIGMA, torch)
        need(pipeline.scheduler.step_index == scheduler_index
             and tensor_sha(pipeline.scheduler.sigmas) == sigma_sha
             and [None if x is None else tensor_sha(x) for x in pipeline.scheduler.model_outputs] == old_history,
             'Terminal operation changed DEIS history/sigmas/index')
        self.terminal_trace = {
            'mode': 'local_lp_s050', 'base_mode': 'terminal_last_eps',
            'last_deis_timestep': values[-1], 'epsilon_source': 'last_actual_unet',
            'rho0': float(rho.item()), 'alpha0': float(alpha.item()), 'sigma0': float(sigma.item()),
            'guidance_scale': self.guidance, 'extra_cfg': None,
            'input_latent_sha256': tensor_sha(latent), 'epsilon_cfg_sha256': tensor_sha(epsilon),
            'last_deis_raw_epsilon_sha256': tensor_sha(self.last_raw),
            'terminal_raw_epsilon_sha256': None, 'clean_latent_sha256': tensor_sha(clean),
            'base_terminal_clean_latent_sha256': tensor_sha(clean),
            'post_filter_clean_latent_sha256': tensor_sha(result),
            'filter': self.runtime.v11.filter_metadata(CLEANUP_SIGMA),
            'outside_cleanup_clean_exact': True,
            'scheduler_step_index_before': scheduler_index,
            'scheduler_step_index_after': pipeline.scheduler.step_index}
        return result

    def callback(self, pipeline, index, timestep, kwargs):
        torch = self.torch
        times, values = self.bind()
        need(pipeline is self.pipe and index == len(self.history) and int(timestep) == values[index],
             'Unexpected callback sequence')
        need(len(self.calls) == index + 1, 'UNet/callback count mismatch')
        latents = kwargs['latents']
        finite(latents, torch, 'denoising latent')
        next_t = times[index + 1] if index + 1 < len(times) else None
        if next_t is None:
            latents = self.correct(latents, pipeline, kwargs['prompt_embeds'])
        reference = self.source if next_t is None else pipeline.scheduler.add_noise(
            self.source, self.noise, next_t.reshape(1))
        # Arithmetic is deliberately identical to legacy sampling_callback.
        projected = self.edit * latents + (1 - self.edit) * reference
        finite(projected, torch, 'source projection')
        if next_t is None:
            self.terminal_trace['projected_latent_sha256'] = tensor_sha(projected)
            if not torch.any(self.edit > 0).item():
                need(torch.equal(projected, self.source), 'Zeroedit must recover source latent')
        entry = {'t': int(timestep), 'next_t': None if next_t is None else int(next_t),
                 'latent_sha256': tensor_sha(projected)}
        if self.trace_dir is not None:
            path = self.trace_dir / f'step_{index:02d}_t{int(timestep):04d}.pt'
            cpu = projected.detach().cpu().contiguous()
            with path.open('xb') as stream:
                torch.save(cpu, stream)
            entry.update(latent_file=str(path), latent_file_sha256=file_sha(path))
        self.history.append(entry)
        return {'latents': projected}

    def assert_complete(self):
        _, values = self.bind()
        need([r['t'] for r in self.calls] == values and [r['t'] for r in self.history] == values,
             'Incomplete generation trace')
        need(self.terminal_trace and 'projected_latent_sha256' in self.terminal_trace,
             'Missing final projection')

    def record(self):
        return {'sampler_version': 'quality_cfg_steps_v1', 'guidance_scale': self.guidance,
                'source_init_strength': STRENGTH, 'au_scale': AU_SCALE, 'ip_scale': IP_SCALE,
                'cleanup_sigma': CLEANUP_SIGMA, 'nominal_steps': self.steps,
                'denoising_steps': len(self.history), 'actual_steps': len(self.history),
                'actual_timesteps': self.values, 'deis_unet_calls': len(self.calls),
                'terminal_unet_calls': 0, 'total_unet_calls': len(self.calls),
                'unet_call_trace': self.calls, 'projection_trace': self.history,
                'terminal_trace': self.terminal_trace,
                'trace_directory': None if self.trace_dir is None else str(self.trace_dir),
                'target_pixels_used': False, 'target_latents_computed': False}

    def close(self):
        self.guard.remove()


def sample(runtime, prepared, seed, *, guidance: float, steps: int):
    """Return the native PIL output and JSON-safe receipt, with every latent saved."""
    guidance, steps = parameters(guidance, steps)
    need(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    torch, source = runtime.torch, prepared['source']
    need(prepared['cross']['au_scale'] == AU_SCALE and prepared['cross']['ip_scale'] == IP_SCALE,
         'Prepared AU/IP scales must remain1.5/.9')
    finite(source, torch, 'source latent')
    need(source.dtype == torch.float32 and tuple(source.shape) == (1, 4, 64, 64),
         'Expected native FP32 source latent1x4x64x64')
    trace_dir = None
    if prepared.get('quality_trace_dir') is not None:
        trace_dir = Path(prepared['quality_trace_dir']).resolve()
        need(ROOT in trace_dir.parents, 'Quality trace output must remain inside this task directory')
        trace_dir.mkdir(parents=True, exist_ok=False)
    noise = torch.randn(source.shape, generator=torch.Generator(device='cpu').manual_seed(seed),
                        dtype=torch.float32).to(runtime.device)
    controller = QualityController(runtime, prepared, guidance, steps)
    controller.noise, controller.trace_dir = noise, trace_dir
    receipt = {'condition_hashes': dict(prepared['condition_hashes']),
               'source_latent_sha256': tensor_sha(source), 'noise_sha256': tensor_sha(noise),
               'mask_receipt': prepared['mask_receipt'],
               'source_vae_reconstruction': prepared['reconstruction']}
    try:
        with torch.no_grad():
            image = runtime.pipe(prompt=prepared['prompt'], image=source, strength=STRENGTH,
                num_inference_steps=steps, guidance_scale=guidance, negative_prompt='',
                generator=torch.Generator(device='cpu').manual_seed(seed),
                cross_attention_kwargs=prepared['cross'], callback_on_step_end=controller.callback,
                callback_on_step_end_tensor_inputs=['latents', 'prompt_embeds']).images[0]
        controller.assert_complete()
        need(image.mode == 'RGB' and image.size == (512, 512), 'Expected native512RGB output')
        receipt.update(controller.record(), status='complete')
        if trace_dir is not None:
            with (trace_dir / 'receipt.json').open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return image, receipt
    except Exception as error:
        receipt.update(controller.record(), status='failed', error=str(error))
        if trace_dir is not None:
            with (trace_dir / 'failure.json').open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        error.generation_receipt = receipt
        raise
    finally:
        controller.close()
