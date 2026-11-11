"""Pinned public components used by new experiments, never an old ITA checkpoint."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
FINEFACE_REPO = Path.home() / 'Projects/github/tvaranka/fineface'
FACE_REPO = Path.home() / 'Projects/github/FaceAdapter/Face-Adapter'
HUB = Path.home() / '.cache/huggingface/hub'
RV = HUB / 'models--SG161222--Realistic_Vision_V4.0_noVAE/snapshots/1685907c0283c7278ba26c5fe561506f564b48d3'
FF = HUB / 'models--Tvaranka--fineface/snapshots/3bd764b8f6f9b113b5e1d878cbfb499d71c04446'
IP = HUB / 'models--h94--IP-Adapter-FaceID/snapshots/43907e6f44d079bf1a9102d9a6e56aef7a219bae'
ARC_WEIGHTS = ROOT.parent / '近期外部对比_20260914/models/face_adapter/third_party/insightface_glint360k.pth'
AU_ORDER = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_repo(path, expected):
    found = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
    if found != expected or changed:
        raise RuntimeError('Public source revision or cleanliness changed: ' + str(path))


def fineface_classes():
    check_repo(FINEFACE_REPO, '6e9cc8df1180092891c1f55e4d4a06110759dac6')
    sys.path.insert(0, str(FINEFACE_REPO))
    from fineface.au_attention import hack_unet_attn_layers, AUIPAttnProcessor2_0
    from fineface.fineface_pipeline import AUEncoder
    from fineface.fineface_ip_pipeline import MLPProjModel
    return hack_unet_attn_layers, AUIPAttnProcessor2_0, AUEncoder, MLPProjModel


def freeze(module, device):
    return module.requires_grad_(False).eval().to(device)


class FrozenConditionedDenoiser(nn.Module):
    def __init__(self, device='mps', checkpointing=True):
        super().__init__()
        hack, processor, AUEncoder, MLPProjModel = fineface_classes()
        self.device = torch.device(device)
        self.unet = UNet2DConditionModel.from_pretrained(str(RV / 'unet'), local_files_only=True, torch_dtype=torch.float32)
        self.unet.set_default_attn_processor()
        hack(self.unet, processor)
        self.au_load = self.unet.load_state_dict(torch.load(FF / 'sd15_attn_processors.ckpt', map_location='cpu', weights_only=True), strict=False)
        if self.au_load.unexpected_keys:
            raise RuntimeError('Unexpected public AU adapter keys')
        state = torch.load(IP / 'ip-adapter-faceid-portrait-v11_sd15.bin', map_location='cpu', weights_only=True)
        self.ip_load = nn.ModuleList(self.unet.attn_processors.values()).load_state_dict(state['ip_adapter'], strict=False)
        if self.ip_load.unexpected_keys:
            raise RuntimeError('Unexpected public FaceID adapter keys')
        self.au_encoder = AUEncoder(hidden_dim=756, clip_dim=768, pad_zeros=False)
        self.au_encoder.load_state_dict(torch.load(FF / 'sd15_au_encoder.ckpt', map_location='cpu', weights_only=True))
        self.ip_proj = MLPProjModel(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=16)
        self.ip_proj.load_state_dict(state['image_proj'])
        for module in (self.unet, self.au_encoder, self.ip_proj):
            freeze(module, self.device)
        if checkpointing:
            self.unet.train()
            self.unet.enable_gradient_checkpointing()
        self.noise_scheduler = DDPMScheduler.from_pretrained(str(RV / 'scheduler'), local_files_only=True)

    def baseline_conditions(self, aus_0_5, buffalo_l_embeddings):
        # Preserve the original public encoder's 0--5 input scale.
        return self.au_encoder(aus_0_5), self.ip_proj(buffalo_l_embeddings)

    def forward(self, noisy_latents, timesteps, text_tokens, au_tokens, image_tokens, au_scale=1.0, ip_scale=0.9):
        return self.unet(noisy_latents, timesteps, encoder_hidden_states=text_tokens,
                         cross_attention_kwargs={'au_embedding': au_tokens, 'au_scale': au_scale,
                                                 'image_embedding': image_tokens, 'ip_scale': ip_scale}).sample


class FrozenImageTextEncoders(nn.Module):
    def __init__(self, device='mps'):
        super().__init__()
        self.device = torch.device(device)
        check_repo(FACE_REPO, '6e313885188e05740a7c40e12c1b9015c748bf08')
        arc_path = FACE_REPO / 'third_party/insightface_backbone_conv.py'
        # Do not import upstream infer.py: it globally disables autograd.
        spec = importlib.util.spec_from_file_location('new_experiment_public_arcface', arc_path)
        arc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arc)
        self.arcface = arc.iresnet100()
        self.arcface.load_state_dict(torch.load(ARC_WEIGHTS, map_location='cpu', weights_only=True), strict=True)
        self.vae = AutoencoderKL.from_pretrained(str(RV / 'vae'), local_files_only=True, torch_dtype=torch.float32)
        self.text_encoder = CLIPTextModel.from_pretrained(str(RV / 'text_encoder'), local_files_only=True, torch_dtype=torch.float32)
        self.tokenizer = CLIPTokenizer.from_pretrained(str(RV / 'tokenizer'), local_files_only=True)
        for module in (self.arcface, self.vae, self.text_encoder):
            freeze(module, self.device)

    def encode_text(self, prompts):
        tokens = self.tokenizer(prompts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        result = self.text_encoder(tokens.input_ids.to(self.device))
        return result.last_hidden_state, result.pooler_output

    def image_features(self, normalized_rgb):
        image = torch.nn.functional.interpolate(normalized_rgb, size=(112, 112), mode='bilinear', align_corners=False)
        patches = self.arcface(image)
        identity = torch.nn.functional.normalize(self.arcface(image, return_id512=True), dim=-1)
        return patches, identity

    def identity_embedding(self, normalized_rgb):
        # Gradients with respect to generated pixels remain enabled.
        image = torch.nn.functional.interpolate(normalized_rgb, size=(112, 112), mode='bilinear', align_corners=False)
        return torch.nn.functional.normalize(self.arcface(image, return_id512=True), dim=-1)

    def encode_latents(self, normalized_rgb):
        posterior = self.vae.encode(normalized_rgb).latent_dist
        scale = self.vae.config.scaling_factor
        return posterior.mean * scale, posterior.std * scale

    def decode_latents(self, latents):
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample
