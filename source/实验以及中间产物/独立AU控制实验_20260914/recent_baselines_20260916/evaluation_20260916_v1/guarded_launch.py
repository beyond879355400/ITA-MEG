"""Execute frozen evaluator bytes only after length+SHA verification.

Call this file through the one-line Python -c read/verify bootstrap documented
in README, and require structured output/receipts, never exit0 alone.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
SHA='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'


def checked(path,expected):
    path=Path(path);size=path.stat().st_size;data=path.read_bytes()
    if not data or len(data)!=size or hashlib.sha256(data).hexdigest()!=expected:
        raise RuntimeError('Empty/truncated/changed source refused: '+str(path))
    return data


spec=json.loads(checked(HERE/'protocol_v1.json',SHA))
sys.path[:0]=[str(HERE),str(HERE/'insightface_built_v1'),str(HERE/'runtime_overlay_v1')]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1',TMPDIR=str(HERE/'overlay_tmp_v1'),PYTHONNOUSERSITE='1')
module=types.ModuleType('protocol');module.__file__=str(HERE/'protocol.py');sys.modules['protocol']=module
exec(compile(checked(module.__file__,spec['execution_sources']['protocol.py']['sha256']),module.__file__,'exec'),module.__dict__)
evaluation=types.ModuleType('evaluate_new');evaluation.__file__=str(HERE/'evaluate_new.py');sys.modules['evaluate_new']=evaluation
exec(compile(checked(evaluation.__file__,spec['execution_sources']['evaluate_new.py']['sha256']),evaluation.__file__,'exec'),evaluation.__dict__)
args=sys.argv[1:]
if args and args[0]=='--extra-bridge':
    name='extra_bridge.py';args=args[1:]
    addon=json.loads((HERE/'additional_bridge_source.json').read_text())
    data=checked(HERE/name,addon['sha256'])
else:
    name='evaluate_new.py';data=None
sys.argv=[str(HERE/name),*args]
if data is None:evaluation.main()
else:exec(compile(data,str(HERE/name),'exec'),{'__name__':'__main__','__file__':str(HERE/name)})
