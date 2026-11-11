"""Infrastructure probe: gradients through frozen public denoiser, no experiment data."""
import argparse
import contextlib
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = Path.home() / 'Projects/github/tvaranka/fineface'
HUB = Path.home() / '.cache/huggingface/hub'
RV = HUB / 'models--SG161222--Realistic_Vision_V4.0_noVAE/snapshots/1685907c0283c7278ba26c5fe561506f564b48d3'
FF = HUB / 'models--Tvaranka--fineface/snapshots/3bd764b8f6f9b113b5e1d878cbfb499d71c04446'
IP = HUB / 'models--h94--IP-Adapter-FaceID/snapshots/43907e6f44d079bf1a9102d9a6e56aef7a219bae'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.latent_size not in (16, 32, 64) or not 1 <= args.steps <= 3:
        raise ValueError('Bounded infrastructure probe only')
    stamp = datetime.now().strftime('%Y%m%d%H%M')
    output = ROOT / 'runtime' / ('condition_gradients_' + stamp)
    suffix = 2
    while output.exists():
        output = ROOT / 'runtime' / ('condition_gradients_' + stamp + f'-{suffix:02d}')
        suffix += 1
    output.mkdir(parents=True)
    print('PROBE_OUTPUT=' + str(output), flush=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      HF_HOME=str(ROOT / 'cache/huggingface'), TORCH_HOME=str(ROOT / 'cache/torch'),
                      MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'), NO_ALBUMENTATIONS_UPDATE='1')
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(ROOT / 'cache/pycache')
    sys.path.insert(0, str(REPO))
    result = {'status': 'running', 'infrastructure_only': True, 'uses_old_generated_images_or_metrics': False,
              'uses_real_training_or_test_data': False, 'optimizer_steps': 0, 'args': vars(args)}
    shutil.copyfile(__file__, output / 'probe_source.py')
    try:
        with (output / 'run.log').open('w') as logfile, contextlib.redirect_stdout(logfile), contextlib.redirect_stderr(logfile):
            import torch
            from diffusers import UNet2DConditionModel
            from fineface.au_attention import hack_unet_attn_layers, AUIPAttnProcessor2_0
            from fineface.fineface_pipeline import AUEncoder
            from fineface.fineface_ip_pipeline import MLPProjModel
            torch.manual_seed(20260914)
            if args.device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable')
            device = torch.device(args.device)
            sync = torch.mps.synchronize if args.device == 'mps' else lambda: None
            weights = [RV / 'unet/diffusion_pytorch_model.safetensors', FF / 'sd15_attn_processors.ckpt',
                       FF / 'sd15_au_encoder.ckpt', IP / 'ip-adapter-faceid-portrait-v11_sd15.bin']
            for path in weights:
                stat = path.stat()
                if getattr(stat, 'st_flags', 0) & 0x40000000:
                    raise RuntimeError('Model is a cloud placeholder: ' + str(path))
            result['weights'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in weights]
            result['fineface_commit'] = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
            started = time.monotonic()
            unet = UNet2DConditionModel.from_pretrained(str(RV / 'unet'), local_files_only=True, torch_dtype=torch.float32)
            unet.set_default_attn_processor()
            hack_unet_attn_layers(unet, AUIPAttnProcessor2_0)
            au_state = torch.load(FF / 'sd15_attn_processors.ckpt', map_location='cpu', weights_only=True)
            incompat = unet.load_state_dict(au_state, strict=False)
            if incompat.unexpected_keys:
                raise RuntimeError('Unexpected AU adapter keys: ' + str(incompat.unexpected_keys))
            ip_state = torch.load(IP / 'ip-adapter-faceid-portrait-v11_sd15.bin', map_location='cpu', weights_only=True)
            ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
            ip_incompat = ip_layers.load_state_dict(ip_state['ip_adapter'], strict=False)
            if ip_incompat.unexpected_keys:
                raise RuntimeError('Unexpected IP adapter keys: ' + str(ip_incompat.unexpected_keys))
            au_encoder = AUEncoder(hidden_dim=756, clip_dim=768, pad_zeros=False)
            au_encoder.load_state_dict(torch.load(FF / 'sd15_au_encoder.ckpt', map_location='cpu', weights_only=True))
            id_encoder = MLPProjModel(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=16)
            id_encoder.load_state_dict(ip_state['image_proj'])
            for module in (unet, au_encoder, id_encoder):
                module.requires_grad_(False)
                module.to(device)
            unet.train()
            unet.enable_gradient_checkpointing()
            au_encoder.eval()
            id_encoder.eval()
            sync()
            result.update(torch=torch.__version__, dtype='float32', initialization_seconds=time.monotonic() - started,
                          denoiser_parameters=sum(p.numel() for p in unet.parameters()),
                          trainable_denoiser_parameters=sum(p.numel() for p in unet.parameters() if p.requires_grad),
                          gradient_checkpointing=True)
            rows = []
            for i in range(args.steps):
                # Synthetic tensors deliberately do not yield scientific performance results.
                raw_au = torch.tensor([[1., 0., 2., 0., 0., 0., 1., 0., 0., 0., 0., 0.]], device=device)
                au = au_encoder(raw_au).detach().requires_grad_(True)
                identity = id_encoder(torch.zeros(1, 512, device=device)).detach().requires_grad_(True)
                text = torch.randn(1, 77, 768, device=device)
                latent = torch.randn(1, 4, args.latent_size, args.latent_size, device=device)
                expected_noise = torch.randn_like(latent)
                sync()
                begin = time.monotonic()
                predicted = unet(latent, torch.tensor([500], device=device), encoder_hidden_states=text,
                                 cross_attention_kwargs={'au_embedding': au, 'au_scale': 1.0,
                                                         'image_embedding': identity, 'ip_scale': 0.9}).sample
                loss = (predicted - expected_noise).square().mean()
                sync()
                forward_end = time.monotonic()
                loss.backward()
                sync()
                row = {'step': i, 'forward_seconds': forward_end - begin,
                       'backward_seconds': time.monotonic() - forward_end,
                       'synthetic_loss': float(loss.detach().cpu()),
                       'au_gradient_norm': float(au.grad.norm().cpu()) if au.grad is not None else None,
                       'identity_gradient_norm': float(identity.grad.norm().cpu()) if identity.grad is not None else None,
                       'finite': bool(torch.isfinite(loss)) and au.grad is not None and identity.grad is not None
                                 and bool(torch.isfinite(au.grad).all()) and bool(torch.isfinite(identity.grad).all()),
                       'frozen_weight_gradients_absent': all(p.grad is None for p in unet.parameters())}
                if args.device == 'mps':
                    row['sampled_allocated_bytes_after_backward'] = torch.mps.current_allocated_memory()
                    row['sampled_driver_bytes_after_backward'] = torch.mps.driver_allocated_memory()
                rows.append(row)
                print(json.dumps(row), flush=True)
                del predicted, loss, au, identity, text, latent, expected_noise
                if not row['finite'] or not row['frozen_weight_gradients_absent'] or row['au_gradient_norm'] <= 0 or row['identity_gradient_norm'] <= 0:
                    raise RuntimeError('Condition gradients did not pass')
            result.update(status='pass', measurements=rows)
    except Exception as error:
        result.update(status='fail', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
