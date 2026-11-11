"""New train/dev-only aligned-image and frozen-feature cache; no test or old outputs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(ROOT / 'cache/huggingface'),
                  TORCH_HOME=str(ROOT / 'cache/torch'), MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'))
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT / 'cache/pycache')

AU_WORDS = {
    1: 'raised inner eyebrows', 2: 'raised outer eyebrows', 4: 'lowered brows',
    5: 'raised upper eyelids', 6: 'raised cheeks', 9: 'a wrinkled nose',
    12: 'raised lip corners', 15: 'lowered lip corners', 17: 'a raised chin',
    20: 'stretched lips', 25: 'parted lips', 26: 'a lowered jaw',
}


def make_prompt(values):
    pieces = []
    for au, value in zip((1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26), values):
        if value > 0:
            degree = 'slightly ' if value <= 2 else 'moderately ' if value <= 3 else 'strongly '
            pieces.append(degree + AU_WORDS[au])
    return 'A tightly cropped photograph of a face with ' + (', '.join(pieces) if pieces else 'a neutral expression') + '.'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = p.parse_args()
    args.output = args.output.resolve()
    if ROOT not in args.output.parents or args.output.exists():
        raise ValueError('A new output directory inside this experiment is required')
    import numpy as np
    import torch
    import cv2
    from PIL import Image, ImageOps
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import estimate_norm
    from modeling.frozen_backbone import FrozenImageTextEncoders, sha256, AU_ORDER, RV, ARC_WEIGHTS
    document = json.loads(args.manifest.read_text())
    pairs = document['pairs']
    subjects = {'train': set(), 'dev': set()}
    for row in pairs:
        if row['split'] not in subjects:
            raise ValueError('This cache builder must not read test inputs')
        subjects[row['split']].add(row['subject'])
        if row['au_order'] != list(AU_ORDER):
            raise ValueError('AU order drift')
    if subjects['train'] & subjects['dev']:
        raise ValueError('Subject leakage across train/dev')
    args.output.mkdir(parents=True)
    aligned_dir = args.output / 'aligned'
    aligned_dir.mkdir()
    shutil.copyfile(__file__, args.output / 'cache_builder_snapshot.py')
    identity_root = ROOT.parent / 'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface'
    face = FaceAnalysis(name='buffalo_l', root=str(identity_root), allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.15)
    encoders = FrozenImageTextEncoders(args.device)
    image_cache, prepared, cache = {}, [], []
    started = time.monotonic()

    def frame(path, expected):
        path = Path(path).resolve()
        if ROOT not in path.parents or sha256(path) != expected:
            raise ValueError('New raw frame path/hash mismatch')
        if expected in image_cache:
            return image_cache[expected]
        rgb = np.asarray(Image.open(path).convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        attempts, detected = [], []
        for variant in ('original', 'autocontrast_cutoff_1'):
            detector_rgb = rgb if variant == 'original' else np.asarray(ImageOps.autocontrast(Image.fromarray(rgb), cutoff=1))
            for threshold in (0.5, 0.3, 0.15, 0.1):
                face.det_model.det_thresh = threshold
                found = face.get(cv2.cvtColor(detector_rgb, cv2.COLOR_RGB2BGR))
                attempts.append({'variant': variant, 'threshold': threshold, 'candidates': len(found)})
                if found:
                    detected = found
                    break
            if detected:
                break
        face.det_model.det_thresh = 0.15
        if not detected:
            raise RuntimeError('Face detection failed; do not discard the fixed case: ' + str(path))
        selected = max(detected, key=lambda x: float((x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])))
        transform112 = estimate_norm(selected.kps, image_size=112)
        transform = transform112 * (512.0 / 112.0)
        aligned = cv2.warpAffine(bgr, transform, (512, 512), flags=cv2.INTER_LINEAR)
        aligned_path = aligned_dir / (expected + '.png')
        if not cv2.imwrite(str(aligned_path), aligned):
            raise RuntimeError('Could not write aligned frame')
        array = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
        with torch.no_grad():
            patches, id_ref = encoders.image_features(tensor)
            latent_mean, latent_std = encoders.encode_latents(tensor)
        record = {'raw_path': str(path), 'raw_sha256': expected, 'aligned_path': str(aligned_path),
                  'aligned_sha256': sha256(aligned_path), 'affine_original_to_512': transform.tolist(),
                  'detection_attempts': attempts, 'face_bbox': selected.bbox.tolist(),
                  'alignment': 'ArcFace canonical112 template scaled to512; original RGB warped, not contrast-adjusted'}
        value = {'record': record, 'patches': patches[0].cpu(), 'id_reference': id_ref[0].cpu(),
                 'latent_mean': latent_mean[0].cpu(), 'latent_std': latent_std[0].cpu(),
                 'base_faceid': torch.from_numpy(np.asarray(selected.normed_embedding).copy()).float()}
        image_cache[expected] = value
        return value

    for row in pairs:
        source = frame(row['source_path'], row['source_sha256'])
        target = frame(row['target_path'], row['target_sha256'])
        prompt = make_prompt(row['target_au_vector_0_5'])
        with torch.no_grad():
            tokens, pooled = encoders.encode_text([prompt])
        cache.append({'case_id': row['case_id'], 'split': row['split'], 'subject': row['subject'], 'prompt': prompt,
                      'image_patches': source['patches'], 'source_identity': source['id_reference'],
                      'base_faceid': source['base_faceid'], 'text_tokens': tokens[0].cpu(), 'text_pooled': pooled[0].cpu(),
                      'target_aus': torch.tensor(row['target_au_vector_0_5'], dtype=torch.float32),
                      'source_aus': torch.tensor([row['source_aus']['au'+str(x)] for x in AU_ORDER], dtype=torch.float32),
                      'latent_mean': target['latent_mean'], 'latent_std': target['latent_std']})
        prepared.append({**row, 'prompt': prompt, 'aligned_source': source['record'], 'aligned_target': target['record']})
        print('CACHED', row['case_id'], flush=True)
    cache_path = args.output / 'features.pt'
    torch.save({'schema_version': 1, 'samples': cache}, cache_path)
    manifest = {'status': 'complete', 'source_manifest_sha256': sha256(args.manifest), 'samples': len(pairs),
                'unique_images': len(image_cache), 'subject_splits': {k: sorted(v) for k, v in subjects.items()},
                'device': args.device, 'resolution': 512, 'patch_shape': list(cache[0]['image_patches'].shape),
                'base_faceid_encoder': 'buffalo_l/w600k_r50', 'patch_and_id_loss_encoder': 'IResNet100/glint360k',
                'cache_path': str(cache_path), 'cache_sha256': sha256(cache_path), 'elapsed_seconds': time.monotonic()-started,
                'model_assets': {'rv_snapshot': str(RV), 'arc_weights_sha256': sha256(ARC_WEIGHTS)},
                'old_generated_images_or_metrics_used': False, 'test_data_used': False, 'pairs': prepared}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'pairs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
