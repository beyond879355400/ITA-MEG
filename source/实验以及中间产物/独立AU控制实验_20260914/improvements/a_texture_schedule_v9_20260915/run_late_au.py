"""A-candidate late-AU schedule, derived from frozen v6 kernel; outputs versioned."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
V6=HERE.parent/'region_preserving_v6_20260915'
V7=HERE.parent/'smooth_regions_v7_20260915'
KERNEL=V6/'run_region_diagnostic.py'
EXPECTED='85236dfc8803fb66d8f2440a240475a10c59f189ab3df059a44389f13db1fcec'

def late_scale(t,late):
    if late not in (0.,.5,1.): raise ValueError('Only three fixed late values')
    return late if t is not None and int(t)<=250 else 1.5

raw=KERNEL.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED: raise RuntimeError('Original v6 kernel changed')
source=raw.decode()
changes=[
("CONFIGS = [('masked_s055_a100', .55, 1.), ('masked_s075_a100', .75, 1.), ('masked_s075_a150', .75, 1.5)]",
 "CONFIGS = [('late_au100', .75, 1.), ('late_au050', .75, .5), ('late_au000', .75, 0.)]",1),
("'v6_development_region_preserving_cases'","'v9_development_late_au_cases'",1),
("'v6_region_preserving_diagnostic'","'v9_late_au_diagnostic'",2),
("'v6_complete'","'v9_complete'",1),
("HERE/'region_mask.py'","V7/'region_mask.py'",2),
("import numpy as np\n    from region_mask import build_mask", "import numpy as np\n    sys.path.insert(0,str(V7))\n    from region_mask import build_mask",1),
("name=method+'_'+config; cross['au_scale']=au_scale", "name=method+'_'+config; cross['au_scale']=1.5\n                    actual_scales=[]\n                    def observe_scale(module,args,kwargs):\n                        t_seen=int(args[1])\n                        value=kwargs['cross_attention_kwargs']['au_scale']\n                        need(value==late_scale(t_seen,au_scale),'Actual UNet AU schedule mismatch')\n                        actual_scales.append({'t':t_seen,'au_scale':value})\n                    scale_guard=backbone.unet.register_forward_pre_hook(observe_scale,with_kwargs=True)",1),
("history.append({'t':int(t),'next_t':None if nt is None else int(nt),'latent_sha256':v5.tensor_sha(projected)})",
 "history.append({'t':int(t),'next_t':None if nt is None else int(nt),'latent_sha256':v5.tensor_sha(projected),'au_scale_used':cross['au_scale']})\n                        cross['au_scale']=late_scale(None if nt is None else int(nt),au_scale)",1),
("row.update(method=name,config=config,seed=101,strength=strength,au_scale=au_scale,ip_scale=.9,nominal_steps=20,",
 "row.update(method=name,config=config,seed=101,strength=strength,au_scale_initial=1.5,au_scale_late=au_scale,au_late_max_t=250,ip_scale=.9,nominal_steps=20,",1),
("row.update(seconds=time.monotonic()-started,actual_steps=len(history),actual_timesteps=[r['t'] for r in history],projection_trace=history)",
 "scale_guard.remove()\n                    row.update(seconds=time.monotonic()-started,actual_steps=len(history),actual_timesteps=[r['t'] for r in history],projection_trace=history,au_scale_trace=actual_scales)",1),
]
for old,new,count in changes:
    if source.count(old)!=count:raise RuntimeError('Ambiguous schedule derivation: '+old)
    source=source.replace(old,new)
# Do not execute the old CLI. Only the small late-schedule changes above differ.
core={'__name__':'v9_late_au_kernel','__file__':str(Path(__file__).resolve()),'V7':V7,'late_scale':late_scale}
exec(compile(source,str(KERNEL),'exec'),core)
original_inputs=core['inputs']; original_save=core['save']

def inputs_with_a():
    cases=original_inputs(); path=V7/'runs/smooth_202609151710/result.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='e59ce11ad87ff6bba91c8131019997833fee0f482a3df364e2d6f28e80aa079d':raise RuntimeError('Original A output changed')
    records={(r['case_id'],r['method']):r for r in json.loads(path.read_text())['records']}
    for c in cases:
        for method,name in [('baseline','a_baseline'),('bidirectional','a_bidirectional')]:
            r=records[c['case_id'],method+'_smooth_s075_a150']
            if r['status']!='complete' or core['sha'](r['output'])!=r['sha256']:raise RuntimeError('A image changed')
            c[name]={'path':r['output'],'sha256':r['sha256']}
    return cases

def save_metadata(path,doc):
    if Path(path).name=='configuration.json':
        doc['configs']=[{'name':n,'strength':s,'au_scale_initial':1.5,'au_scale_late':a,'au_late_max_t':250} for n,s,a in core['CONFIGS']]
        doc['schedule']='AU1.5 for t>250; named late value for t<=250; actual UNet pre-hook logs every used scale'
        doc['kernel_reference']={'path':str(KERNEL),'sha256':EXPECTED}
        doc['derived_kernel_sha256']=hashlib.sha256(source.encode()).hexdigest()
    original_save(path,doc)

core['inputs']=inputs_with_a;core['save']=save_metadata
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps({'status':'prepared_no_models','cases':len(inputs_with_a()),'configs':core['CONFIGS']}));sys.exit(0)
    if a.run is None:raise ValueError('Explicit new run required')
    core['execute'](a.run.resolve(),V7/'masks/first8/manifest.json')
