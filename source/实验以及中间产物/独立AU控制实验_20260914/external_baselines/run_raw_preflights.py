#!/usr/bin/env python3
"""Run only resource/manifest CPU preflights for raw-v2 image-driven methods."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

from prepare_dev_contracts import HERE, OLD, REPOS, PYTHON, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contracts', type=Path, required=True)
    args = p.parse_args()
    contracts = args.contracts.resolve()
    manifest = contracts / 'image_driver_dev12.json'
    doc = json.loads(manifest.read_text())
    assert HERE in contracts.parents and doc['schema_version'] == 2 and not doc['test_used']
    runtime = HERE / 'runtime'
    base = contracts / ('native_preflights_' + datetime.now().strftime('%Y%m%d%H%M'))
    out = base
    n = 2
    while out.exists():
        out = base.with_name(base.name + f'-{n:02d}')
        n += 1
    out.mkdir()
    unique = contracts.name + '_' + out.name
    commands = {
        'faceadapter': [str(PYTHON), '-B', str(runtime / 'run_faceadapter.py'), '--repo', str(REPOS['faceadapter']),
                        '--checkpoint', str(OLD / 'models/face_adapter'), '--base-model', str(OLD / 'models/stable-diffusion-v1-5'),
                        '--vae-model', str(OLD / 'models/sd-vae-ft-mse'), '--manifest', str(manifest),
                        '--output', str(runtime / 'runs' / ('faceadapter_' + unique)), '--device', 'cpu',
                        '--preload-final-components', '--detection-policy', 'author_default', '--preflight'],
        'liveportrait': [str(PYTHON), '-B', str(runtime / 'run_liveportrait.py'), '--repo', str(REPOS['liveportrait']),
                         '--checkpoint', str(OLD / 'models/liveportrait_author'), '--manifest', str(manifest),
                         '--output', str(runtime / 'runs' / ('liveportrait_' + unique)), '--device', 'cpu',
                         '--detection-policy', 'author_default', '--preflight'],
    }
    save(out / 'commands.json', commands)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONPYCACHEPREFIX=str(HERE / 'cache/pycache'),
               HF_HOME=str(HERE / 'cache/huggingface'), TORCH_HOME=str(HERE / 'cache/torch'),
               MPLCONFIGDIR=str(HERE / 'cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'cache/xdg'))
    records = {}
    for name, command in commands.items():
        with (out / (name + '.log')).open('x') as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        records[name] = {'returncode': run.returncode, 'command': command}
    save(out / 'summary.json', {'records': records, 'generation_run': False, 'test_used': False})
    print(json.dumps({'output': str(out), 'returncodes': {k: v['returncode'] for k, v in records.items()}, 'generation_run': False}))


if __name__ == '__main__':
    main()
