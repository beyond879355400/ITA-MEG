#!/usr/bin/env python3
"""APFS clonefile layout only; never re-download or replace a different model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from prepare_dev_contracts import HERE, sha, resident, save


def clone(source, target, expected):
    source, target = Path(source), Path(target)
    assert resident(source)['status'] == 'resident' and sha(source) == expected
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == expected
        mode = 'existing_verified'
    else:
        run = subprocess.run(['cp', '-c', str(source), str(target)], capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError('APFS clonefile failed; no full-copy fallback chosen: ' + run.stderr)
        mode = 'cp_-c_APFS_clonefile'
    a, b = source.stat(), target.stat()
    assert (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino) and a.st_size == b.st_size and sha(target) == expected
    return {'source': str(source), 'target': str(target), 'bytes': b.st_size, 'sha256': expected, 'mode': mode,
            'source_device_inode': [a.st_dev, a.st_ino], 'target_device_inode': [b.st_dev, b.st_ino], 'independent_inode': True}


def main():
    inventory_path = HERE / 'resource_restore/magicface_inventory_verified_202609141950.json'
    inventory = json.loads(inventory_path.read_text())
    assert inventory['all_required_files_resident'] and not inventory['missing_resident_components']
    root = HERE / 'magicface_runtime'
    if (root / 'layout_seal.json').exists():
        raise RuntimeError('Layout already sealed')
    free_before = shutil.disk_usage(HERE).free
    records = []
    for row in inventory['resources']:
        name = row['component']
        if name.startswith(('ID_enc/', 'denoising_unet/', 'text_encoder/')):
            target = root / 'models/generator' / name
        elif name.startswith(('vae/', 'scheduler/', 'tokenizer/')):
            target = root / 'models/base' / name
        else:
            target = root / 'models/preprocess' / name
        records.append(dict(clone(row['selected']['path'], target, row['selected']['sha256']), component=name))
    legacy = Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace'
    official_blobs = {'third_party/model_resnet_d3dfr.py': 'ad6e51184a78bb304634e5eb905b239885f19623',
                      'third_party/d3dfr/bfm.py': '1f9db24399f25e0ce20d30cbc03a252e48f4787b'}
    auxiliary = []
    for name, blob in official_blobs.items():
        source = legacy / name
        data = source.read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob
        auxiliary.append(dict(clone(source, root / 'auxiliary' / name, sha(source)), official_git_blob_sha1=blob,
                              official_revision='mengtingwei/magicface@c3e825225734f5ef075fbcbe7040b7cc8ccca174'))
    save(root / 'layout_seal.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'inventory_sha256': sha(inventory_path),
                                   'models': records, 'auxiliary_sources': auxiliary, 'script_sha256': sha(__file__),
                                   'disk_free_before': free_before, 'disk_free_after': shutil.disk_usage(HERE).free,
                                   'downloads': False, 'old_directories_modified': False, 'environment_modified': False})
    print(json.dumps({'layout': str(root), 'model_files': len(records), 'auxiliary_source_files': len(auxiliary),
                      'all_independent_inodes': all(r['independent_inode'] for r in records + auxiliary),
                      'downloads': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
