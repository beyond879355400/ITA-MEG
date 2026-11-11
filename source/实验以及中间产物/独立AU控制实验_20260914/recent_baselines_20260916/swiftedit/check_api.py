"""Read-only import/API inventory: never instantiate a model or allocate MPS tensors."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(HERE / 'versions/device_v1'))
names = ['torch','torchvision','torchaudio','diffusers','transformers','huggingface-hub',
         'accelerate','ftfy','tensorboard','Jinja2','einops','numpy','safetensors']
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
report = {'kind': 'swiftedit_import_only_probe', 'packages': packages, 'models_constructed': False,
          'inference_performed': False, 'tensor_device': 'none', 'network_downloads': False}
try:
    import torch
    import models
    import infer
    report.update(status='import_pass', mps_available=torch.backends.mps.is_available(),
                  classes=[name for name in ('InverseModel','AuxiliaryModel','IPSBV2Model') if hasattr(models,name)],
                  source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (HERE/'versions/device_v1/models.py', HERE/'versions/device_v1/infer.py')})
    if packages['ftfy'] is None:
        report['required_before_model_load'] = ['ftfy absent: stage it into a separate environment/layer after CACHE_READY']
except Exception as error:
    report.update(status='import_failed', error=repr(error))
print(json.dumps(report, ensure_ascii=False, indent=2))
