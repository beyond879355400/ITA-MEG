"""Explicit, append-only quality matrix; default metadata preflight, no models.

expansion32: five internal variants x32 frozen requests x3 seeds =480 outputs.
dev19: five internal variants x19 existing development requests x1 seed =95.
No target images are read; no old code, weights or results are changed.
"""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import traceback

W = Path(__file__).resolve().parent
EXP = W.parent.parent
COMPLETION = EXP/'completion_20260915'
GEN = COMPLETION/'generation'
VARIANTS = ['base','bidirectional','condition_reads_image','image_reads_condition','mlp']
SCOPE_SEEDS = {'expansion32':[101,202,303], 'dev19':[101]}
BASE_CONFIG = COMPLETION/'generation_runs/selected_v11_32_v1/configuration.json'
BASE_CONFIG_SHA = 'af9243692fe842fb1f4126e077a3af00b5964fcf8acafb5c48a1ad01c430c61a'
sys.dont_write_bytecode = True


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path)}


def read(path):
    with Path(path).open(encoding='utf-8') as stream: return json.load(stream)


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def validate_config(value,scope):
    need(scope in SCOPE_SEEDS,'Unknown scope')
    need(type(value) is dict and set(value)=={'id','guidance','steps','seeds'},
         'Config must contain exactly id,guidance,steps,seeds; no target paths or hidden overrides')
    need(isinstance(value['id'],str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value['id']),
         'Config id must be a safe unique name')
    g=value['guidance']
    need(type(g) in (int,float) and math.isfinite(g) and g>1,'Finite CFG>1 required')
    need(type(value['steps']) is int and value['steps'] in (20,40),'Only nominal20or40 supported')
    need(value['seeds']==SCOPE_SEEDS[scope] and all(type(x) is int for x in value['seeds']),
         'Scope requires exact frozen seed order: '+str(SCOPE_SEEDS[scope]))
    return {'id':value['id'],'guidance':float(g),'steps':value['steps'],'seeds':list(value['seeds'])}


def legacy():
    sys.path[:0]=[str(GEN),str(EXP),str(COMPLETION/'delivery')]
    import common,run_grid,run_dev_completion
    need(common.VARIANTS==VARIANTS,'Frozen variant order changed')
    return common,run_grid,run_dev_completion


def load_inputs(scope):
    c,grid,dev=legacy()
    if scope=='dev19':
        inputs=dev.plan()
        return {'scope':scope,'dev':inputs,'cases':inputs['cases'],'sources':inputs['sources'],
                'refs':inputs['refs'],'legacy_spec':inputs['core_generation']}
    need(sha(BASE_CONFIG)==BASE_CONFIG_SHA,'Original32case configuration changed')
    base=read(BASE_CONFIG)
    refs={k:base[k] for k in ('freeze','source_cache','source_measurements')}
    for role,value in refs.items(): c.verified_ref(value,role)
    _,labels=c.load_freeze(refs['freeze']['path'],refs['freeze']['sha256'])
    cache,_=grid.load_cache(Path(refs['source_cache']['path']),refs['source_cache']['sha256'],
                            labels,refs['freeze']['sha256'],load_tensors=False)
    measurements=read(refs['source_measurements']['path'])
    need(measurements['protocol_sha256']==refs['freeze']['sha256']
         and measurements['source_cache_sha256']==refs['source_cache']['sha256']
         and measurements['source_only'] is True,'Source geometry/cache binding mismatch')
    mapping={r['case_id']:r for r in measurements['records'] if r.get('method')=='source_real'}
    need(len(mapping)==len(measurements['records']) and set(mapping)==set(cache['case_ids']),
         'One source-only measurement per frozen request required')
    return {'scope':scope,'cases':cache['cases'],'sources':cache['sources'],'cache':cache,
            'labels':labels,'measurements':mapping,'refs':refs,'legacy_spec':c.execution_spec()}


def matrix_cells(cases,config,scope):
    """Pure manifest construction; one record for every planned cell."""
    rows=[]; seen=set()
    for case in cases:
        cid=case['case_id']
        need(isinstance(cid,str) and re.fullmatch(r'[A-Za-z0-9_-]+',cid),'Unsafe case id')
        need(cid not in seen,'Duplicate case id'); seen.add(cid)
        for variant in VARIANTS:
            for seed in config['seeds']:
                method=variant+'_'+config['id']
                rows.append({'case_id':cid,'subject':case['subject'],'split':case['split'],
                    'pool':scope,'variant':variant,'method':method,'config_id':config['id'],'seed':seed,
                    'execution_id':f'{method}::{cid}::seed{seed}',
                    'source_aus_0_5':case['source_au_vector_0_5'],
                    'target_aus_0_5':case['target_au_vector_0_5'],
                    'group':'nonzero_target' if any(case['target_au_vector_0_5']) else 'no_edit',
                    'output':None,'sha256':None,'execution_started':False,'status':'planned'})
    need(len({r['execution_id'] for r in rows})==len(rows),'Duplicate execution ids')
    return rows


def preflight(scope,config_path):
    config_path=Path(config_path).resolve()
    config=validate_config(read(config_path),scope)
    inputs=load_inputs(scope)
    expected_cases=32 if scope=='expansion32' else 19
    need(len(inputs['cases'])==expected_cases,'Scope case count mismatch')
    rows=matrix_cells(inputs['cases'],config,scope)
    plan={'kind':'quality_full_internal_matrix','status':'planned','scope':scope,'config':config,
          'case_ids':[c['case_id'] for c in inputs['cases']],
          'subjects':sorted({c['subject'] for c in inputs['cases']}),
          'variant_order':VARIANTS,'seed_order':config['seeds'],'planned':len(rows),
          'case_selection':'All frozen cases in original order; no outcome-based filtering',
          'source_strength':.75,'au_scale':1.5,'ip_scale':.9,'cleanup_sigma':.5,'negative_prompt':'',
          'target_pixels_supplied':False,'target_latents_computed':False,'training_performed':False,
          'all_attempts_retained':True,'zeroedit_caching':False,'inputs':inputs['refs'],
          'original_generation_spec':inputs['legacy_spec'],'requested_config':ref(config_path),
          'runner':ref(__file__),'sampler':ref(W/'sampler.py')}
    return inputs,config,rows,plan


def prepare_masks(inputs,output):
    c,grid,dev=legacy()
    import numpy as np
    from geometry import build_masks
    masks,records={},[]
    sources={s['source_id']:s for s in inputs['sources']}
    for case in inputs['cases']:
        cid=case['case_id']
        try:
            if inputs['scope']=='dev19':
                subset=dict(inputs['dev']); subset['cases']=[case]
                one,receipts=dev.prepare_masks(subset,output)
                masks.update(one); records.extend(receipts)
            else:
                edit,cleanup,receipt=build_masks(case,sources[case['source_id']],inputs['measurements'][cid])
                folder=output/'masks'/cid; folder.mkdir(parents=True,exist_ok=False)
                for name,array in (('edit64',edit),('cleanup64',cleanup)):
                    with (folder/(name+'.npy')).open('xb') as stream: np.save(stream,array,allow_pickle=False)
                receipt.update(edit_mask=ref(folder/'edit64.npy'),cleanup_mask=ref(folder/'cleanup64.npy'))
                save(folder/'receipt.json',receipt)
                masks[cid]=edit,cleanup,ref(folder/'receipt.json')
                records.append({'case_id':cid,'status':'complete','receipt':masks[cid][2]})
        except Exception as error:
            records.append({'case_id':cid,'status':'failed','error':str(error),'traceback':traceback.format_exc()})
    save(output/'mask_receipts.json',records)
    return masks,records


def load_features(inputs,torch):
    c,grid,dev=legacy()
    if inputs['scope']=='dev19': return dev.source_only_features(inputs['dev'],torch)
    refs=inputs['refs']
    cache,features=grid.load_cache(Path(refs['source_cache']['path']),refs['source_cache']['sha256'],
        inputs['labels'],refs['freeze']['sha256'],load_tensors=True)
    return features,cache,{'reused_source_only_cache':refs['source_cache'],'target_latents_computed':False}


def active_quality_generators():
    watched={'run_probe.py','run_matrix.py'}
    output=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True)
    active=[]
    for line in output.splitlines():
        try:
            pid,command=line.strip().split(None,1); pid=int(pid); args=shlex.split(command)
        except ValueError: continue
        if pid==os.getpid() or '--execute' not in args: continue
        hits=[Path(a).name for a in args if a.endswith('.py') and Path(a).name in watched]
        if hits: active.append({'pid':pid,'script':hits[0]})
    return active


def append(stream,row):
    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()


def generate_matrix(runtime,inputs,config,planned,output,masks,sample_fn):
    """Continue per-cell failures; retain untouched planned cells on fatal interrupt."""
    cases={c['case_id']:c for c in inputs['cases']}
    sources={s['source_id']:s for s in inputs['sources']}
    records=[]; preparations={}; started=time.monotonic(); fatal=None
    config_ref=ref(output/'configuration.json')
    with (output/'generation_records.jsonl').open('x',buffering=1) as log, (output/'events.jsonl').open('x',buffering=1) as events:
        try:
            for base in planned:
                row=dict(base); case=cases[row['case_id']]; source=sources[case['source_id']]
                row.update(source_real=source['canonical_source'],configuration=config_ref)
                key=row['case_id'],row['variant']
                if key not in preparations:
                    try:
                        need(source['status']=='complete' and case.get('condition_status','complete')=='complete',
                             'Source/condition cache preparation failed')
                        need(case['case_id'] in masks,'Source mask preparation failed')
                        prepared=runtime.prepare(row['variant'],case,source,masks[case['case_id']])
                        preparations[key]=(prepared,None)
                    except Exception as error: preparations[key]=(None,str(error))
                prepared,error=preparations[key]
                t=time.monotonic()
                if error:
                    row.update(status='not_started',error=error,seconds=0.)
                else:
                    row['execution_started']=True
                    append(events,{'event':'started','execution_id':row['execution_id'],
                                   'local':datetime.now().astimezone().isoformat()})
                    one=dict(prepared)
                    one['quality_trace_dir']=output/'traces'/case['case_id']/row['method']/f'seed{row["seed"]}'
                    image=None
                    try:
                        image,receipt=sample_fn(runtime,one,row['seed'],guidance=config['guidance'],steps=config['steps'])
                        need(image.size==(512,512) and image.mode=='RGB','Native512RGB output required')
                        path=output/'images'/case['case_id']/f'{row["method"]}_seed{row["seed"]}.png'
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with path.open('xb') as stream: image.save(stream,format='PNG')
                        row.update(output=str(path),sha256=sha(path),receipt=receipt)
                        if row['group']=='no_edit':
                            from PIL import Image
                            with Image.open(prepared['reconstruction']['path']) as original:
                                need(image.tobytes()==original.convert('RGB').tobytes(),'Zeroedit source reconstruction parity failed')
                        row['status']='complete'
                    except Exception as failure:
                        row.update(status='failed',error=str(failure),traceback=traceback.format_exc())
                        if hasattr(failure,'generation_receipt'): row['receipt']=failure.generation_receipt
                    finally:
                        if image is not None: image.close()
                    row['seconds']=time.monotonic()-t
                records.append(row); append(log,row)
                print(json.dumps({'stage':'generated','done':len(records),'planned':len(planned),
                    'case':row['case_id'],'variant':row['variant'],'seed':row['seed'],
                    'status':row['status'],'seconds':row['seconds']},ensure_ascii=False),flush=True)
        except BaseException as error:
            fatal={'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc()}
            for remaining in planned[len(records):]:
                row=dict(remaining); row.update(status='not_started',error='Run interrupted: '+str(error),seconds=0.)
                records.append(row); append(log,row)
    need(len(records)==len(planned),'Matrix accounting incomplete')
    return records,fatal,time.monotonic()-started


def run(inputs,config,planned,plan,output):
    c,grid,dev=legacy()
    from reproduce_driver import exclusive_lock,active_generators
    need(not active_generators() and not active_quality_generators(),'Another generator is active')
    with exclusive_lock():
        need(not active_generators() and not active_quality_generators(),'Another generator started')
        output=output.resolve()
        need(output.parent==W/'runs' and not output.exists(),'Use a new immediate Q/runs directory')
        output.mkdir(parents=True,exist_ok=False)
        plan.update(started_local=datetime.now().astimezone().isoformat())
        save(output/'configuration.json',plan)
        save(output/'cases.json',{'kind':'quality_matrix_cases','pool':inputs['scope'],
            'cases':inputs['cases'],'sources':inputs['sources'],'target_pixels_supplied':False})
        save(output/'planned_matrix.json',{'records':planned,'planned':len(planned)})
        runtime=None
        try:
            torch=c.environment('mps')
            masks,_=prepare_masks(inputs,output)
            features,cache,receipts=load_features(inputs,torch)
            save(output/'feature_receipts.json',receipts)
            import sampler
            need(sha(sampler.__file__)==plan['sampler']['sha256'],'Sampler changed after preflight')
            print(json.dumps({'stage':'loading_models','planned':len(planned)}),flush=True)
            runtime=grid.Runtime(features,cache,output,torch)
            records,fatal,elapsed=generate_matrix(runtime,inputs,config,planned,output,masks,sampler.sample)
            counts=dict(Counter(r['status'] for r in records))
            status='interrupted' if fatal else ('complete' if counts.get('complete',0)==len(planned) else 'completed_with_failures')
            result={'kind':'quality_matrix_result','status':status,'scope':inputs['scope'],
                'planned':len(planned),'completed':counts.get('complete',0),'counts':counts,'records':records,
                'configuration':ref(output/'configuration.json'),'runtime':ref(output/'runtime.json'),
                'target_pixels_supplied':False,'training_performed':False,
                'all_planned_cases_variants_seeds_retained':True,'zeroedit_caching':False,
                'seconds':elapsed,'fatal_error':fatal}
            save(output/'result.json',result)
            return 0 if status=='complete' else 1
        except BaseException as error:
            save(output/'fatal_error.json',{'error':str(error),'type':type(error).__name__,
                 'traceback':traceback.format_exc(),'all_saved_artifacts_retained':True})
            if not (output/'result.json').exists():
                rows=[dict(r,status='not_started',error='Run-level preparation failure: '+str(error)) for r in planned]
                if not (output/'generation_records.jsonl').exists():
                    with (output/'generation_records.jsonl').open('x') as log:
                        for row in rows: append(log,row)
                save(output/'result.json',{'kind':'quality_matrix_result','status':'failed_runtime',
                    'planned':len(planned),'completed':0,'records':rows,
                    'configuration':ref(output/'configuration.json'),'target_pixels_supplied':False,
                    'training_performed':False,'all_planned_cases_variants_seeds_retained':True})
            raise
        finally:
            if runtime is not None: runtime.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--scope',choices=list(SCOPE_SEEDS),required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    inputs,config,rows,plan=preflight(args.scope,args.config)
    print(json.dumps({'stage':'preflight','scope':args.scope,'config':config,'cases':len(inputs['cases']),
        'variants':VARIANTS,'planned':len(rows),'models_loaded':False,'target_pixels_supplied':False},ensure_ascii=False),flush=True)
    if not args.execute: return 0
    need(args.output is not None,'--execute requires an explicit new --output')
    return run(inputs,config,rows,plan,args.output)


if __name__=='__main__': raise SystemExit(main())
