#!/usr/bin/env python3
"""Make fixed4-frame lossless source/driver clips; no generative model use."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(output):
    from PIL import Image
    policy_path = ROOT / 'smoke_policy.json'
    policy = json.loads(policy_path.read_text())
    manifest = (ROOT / policy['input_manifest']).resolve()
    if sha(manifest) != policy['input_manifest_sha256']:
        raise RuntimeError('Frozen smoke manifest changed')
    data = json.loads(manifest.read_text())
    case = data['cases'][0]
    if case['case_id'] != policy['case_id']:
        raise RuntimeError('First-case binding changed')
    output.mkdir(parents=True, exist_ok=False)
    bundle = manifest.parent.parent
    record = {'policy_sha256': sha(policy_path), 'case_id': case['case_id'], 'models_loaded': False, 'inputs': {}}
    for role in ('source', 'target'):
        path = bundle / case[role+'_path']
        if sha(path) != case[role+'_sha256']:
            raise RuntimeError('Frozen input image changed: '+role)
        copied = output / (role + '.png')
        shutil.copy2(path, copied)
        image = Image.open(path).convert('RGB')
        clip = output / (role+'.mov')
        command = ['ffmpeg','-nostdin','-v','error','-loop','1','-framerate','25','-i',str(path),
                   '-frames:v','4','-c:v','png','-pix_fmt','rgb24','-an',str(clip)]
        subprocess.run(command, check=True)
        decoded = subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(clip),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        expected = image.tobytes() * 4
        if decoded != expected:
            raise RuntimeError('Lossless clip pixel parity failed: '+role)
        record['inputs'][role] = {'original_path': str(path), 'original_sha256': sha(path),
                'copied_path': str(copied), 'clip': str(clip), 'clip_sha256': sha(clip),
                'width': image.width, 'height': image.height, 'frames': 4, 'fps': 25,
                'every_decoded_frame_exact_original_rgb': True, 'ffmpeg_command': command}
    record['status'] = 'complete'
    (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({'status':'plan_only','output':str(args.output),'models_loaded':False}))
    else:
        print(json.dumps(execute(args.output.resolve()),ensure_ascii=False,indent=2))
