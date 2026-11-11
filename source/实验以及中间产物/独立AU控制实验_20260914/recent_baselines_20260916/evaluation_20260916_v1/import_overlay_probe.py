"""Import-only verified evaluation overlay. Reject any pretrained model load."""
import importlib.metadata as metadata
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
EXP=HERE.parent.parent
paths=[HERE/'insightface_built_v1',HERE/'runtime_overlay_v1',EXP/'tools/facenet/runtime']
sys.path[:0]=[str(p)for p in paths]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  TMPDIR=str(HERE/'overlay_tmp_v1'),MPLCONFIGDIR=str(HERE/'overlay_tmp_v1/matplotlib'),XDG_CACHE_HOME=str(HERE/'overlay_tmp_v1/xdg'))
started=time.monotonic()


def no_network(*args,**kwargs):raise RuntimeError('Import-only probe must not access network')
socket.socket.connect=no_network
import torch
def no_weights(*args,**kwargs):raise RuntimeError('Import-only probe must not load pretrained weights')
torch.load=no_weights
import torchvision
import numpy
import PIL
import cv2
import onnxruntime
onnxruntime.InferenceSession.__init__=no_weights
import insightface
import facenet_pytorch
import scipy
import skimage

names={'torch':torch,'torchvision':torchvision,'numpy':numpy,'Pillow':PIL,'opencv-python':cv2,'onnxruntime':onnxruntime,
       'insightface':insightface,'facenet_pytorch_source':facenet_pytorch,'scipy':scipy,'scikit-image':skimage}
report={'status':'import_only_pass','models_constructed':False,'pretrained_loads_forbidden':True,'network_forbidden':True,
        'python':sys.executable,'seconds':time.monotonic()-started,'imports':{},'sys_path_prefix':sys.path[:4]}
for n,m in names.items():
    p=Path(m.__file__);st=p.stat();data=p.read_bytes()
    if len(data)!=st.st_size:raise RuntimeError('Dataless empty/short imported file: '+str(p))
    report['imports'][n]={'version':getattr(m,'__version__',None),'path':str(p),'file_bytes':len(data),
                          'sha256':hashlib.sha256(data).hexdigest(),'flags':st.st_flags}
with (HERE/'overlay_import_receipt.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False))
