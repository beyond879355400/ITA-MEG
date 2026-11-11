#!/usr/bin/env python3
"""Tiny operator compatibility only; no checkpoints or face models loaded."""
import argparse
import json
import os
from pathlib import Path


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('PERFORMRECAST_GPU_RUN_GO')!='1':
        raise SystemExit('GPU_RUN_GO absent; no tensor operations run')
    if a.output.exists():
        raise SystemExit('Append-only report path required')
    import torch
    import torch.nn.functional as F
    report={'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'models_loaded':False,'tests':[]}
    for name in ('grid_sample_3d','conv3d'):
        try:
            x=torch.zeros((1,1,2,3,3),device='mps',dtype=torch.float32)
            if name=='grid_sample_3d':
                y=F.grid_sample(x,torch.zeros((1,2,3,3,3),device='mps'),align_corners=False)
            else:
                y=F.conv3d(x,torch.ones((1,1,1,1,1),device='mps'))
            torch.mps.synchronize()
            report['tests'].append({'op':name,'status':'PASS','shape':list(y.shape)})
        except Exception as e:
            report['tests'].append({'op':name,'status':'UNSUPPORTED_OR_ERROR','error':type(e).__name__+': '+str(e)})
    report['recommended_device']='mps' if all(x['status']=='PASS' for x in report['tests']) else 'cpu'
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
