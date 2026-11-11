"""Dev-only MagicFace native AU contract; no model loading or auto-download."""
import math

AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
PROMPT = 'A close up of a person.'


def validate_case(case):
    if case['split'] != 'dev' or case['input_group'] != 'source_plus_manual_au':
        raise ValueError('This entry is frozen for source+AU development only')
    for forbidden in ('target', 'target_path', 'target_image_path', 'driver', 'driver_path', 'target_sha256'):
        if forbidden in case:
            raise ValueError('MagicFace must not receive a target/driver image field')
    if case['au_order'] != AUS:
        raise ValueError('AU order mismatch')
    source, target = case['source_au_vector_0_5'], case['target_au_vector_0_5']
    if len(source) != 12 or len(target) != 12:
        raise ValueError('Expected full 12AU vectors')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 5 for v in source + target):
        raise ValueError('Manual AU intensities must be finite 0..5')
    delta = [t - s for s, t in zip(source, target)]
    if delta != case['au_delta_native']:
        raise ValueError('Native AU delta must be target minus source, without scale conversion')
    if case['native_inference_steps'] != 50 or case['prompt'] != PROMPT:
        raise ValueError('Native sampling contract changed')
    return delta


def cpu_source_payload(case):
    """Only read a frozen source and form CPU source/AU tensors for preflight."""
    from pathlib import Path
    import hashlib
    import torch
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    delta = validate_case(case)
    path = Path(case['source'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != case['source_sha256']:
        raise ValueError('Source hash changed')
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (512, 512):
            raise ValueError('Require frozen RGB512 source')
        source = to_tensor(image).unsqueeze(0)
    au = torch.tensor(delta, dtype=torch.float32).unsqueeze(0)
    assert source.device.type == au.device.type == 'cpu'
    return {'source': source, 'au': au}


def native_inference_kwargs(case, source_tensor, background_tensor, prompt_embeds, *,
                            background_derived_from_source_sha256, generator):
    """Bind a preloaded official pipeline call; no target-image argument exists.

    Fresh background/contour must be produced by the official source-side
    preprocessing before this entry can be called. This function does not
    create a fake background or substitute a target image.
    """
    import torch
    delta = validate_case(case)
    if background_derived_from_source_sha256 != case['source_sha256']:
        raise ValueError('Background provenance must be the same frozen source')
    if tuple(source_tensor.shape) != (1, 3, 512, 512) or tuple(background_tensor.shape) != (1, 3, 512, 512):
        raise ValueError('Native source/background shape must be B1 RGB512')
    return {'prompt_embeds': prompt_embeds, 'source': source_tensor, 'bg': background_tensor,
            'au': torch.tensor(delta, dtype=torch.float32).unsqueeze(0), 'num_inference_steps': 50,
            'guidance_scale': 7.5, 'image_guidance_scale': 1.5, 'generator': generator}
