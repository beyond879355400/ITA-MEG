#!/usr/bin/env python3
"""Full frozen dev12 × two-method CPU evaluation; no generation or tuning."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'v1'))
import evaluate_development as helper
AUS=tuple(helper.AUS)
PREVIEW=ROOT/'development_previews/v3_global_raw_dev64_202609141730'
CACHE=ROOT/'training_cache/expanded_v2_202609141711'
TOOLS=ROOT/'tools/openface'
EXPECTED_MODELS={'det_10g.onnx':'5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',
                 'w600k_r50.onnx':'4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'}

def mean(values):return sum(values)/len(values) if values else None
def digest(path):return helper.digest(path)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def au_metrics(pred,source_pred,source_gt,target_gt):
    c=[i for i in range(12) if source_gt[i]!=target_gt[i]];u=[i for i in range(12) if i not in c]
    base={'changed_aus':[AUS[i] for i in c],'unchanged_aus':[AUS[i] for i in u],
          'target_mae_changed':None,'delta_mae_changed':None,'unchanged_source_relative_drift':None,
          'target_mae_all12':None,'delta_mae_all12':None,'measured_delta_12':None,'direction_correct_fraction':None}
    if pred is None:return base
    base['target_mae_changed']=mean([abs(pred[i]-target_gt[i]) for i in c]);base['target_mae_all12']=mean([abs(pred[i]-target_gt[i]) for i in range(12)])
    if source_pred is None:return base
    delta=[pred[i]-source_pred[i] for i in range(12)];wanted=[target_gt[i]-source_gt[i] for i in range(12)]
    base.update(delta_mae_changed=mean([abs(delta[i]-wanted[i]) for i in c]),delta_mae_all12=mean([abs(delta[i]-wanted[i]) for i in range(12)]),
                unchanged_source_relative_drift=mean([abs(delta[i]) for i in u]),measured_delta_12=delta,
                direction_correct_fraction=mean([int(delta[i]*wanted[i]>0) for i in c]),
                response_gain_changed=[delta[i]/wanted[i] for i in c])
    return base

def prepare():
    result=json.loads((PREVIEW/'result.json').read_text());settings=json.loads((PREVIEW/'settings.json').read_text())
    cases=json.loads((PREVIEW/'frozen_cases.json').read_text());meta=json.loads((CACHE/'manifest.json').read_text())
    records=[json.loads(x) for x in (PREVIEW/'generation_records.jsonl').read_text().splitlines()]
    assert result['status']=='complete' and result['images']==24 and result['cases']==12 and result['methods']==2
    assert not result['test_data_used'] and not meta['test_data_used'] and not cases['test_data_used']
    assert settings['cache_sha256']==meta['cache_sha256']==cases['cache_sha256']==digest(CACHE/'features.pt')
    assert settings['case_manifest_sha256']==digest(PREVIEW/'frozen_cases.json')
    assert len(settings['methods'])==1 and settings['sampling_seeds']==cases['sampling_seeds']==[101]
    method_name=settings['methods'][0]['name'];assert method_name.startswith('bidirectional_step64_')
    pairs={p['case_id']:p for p in meta['pairs']};generation={}
    split=json.loads(Path(meta['subject_split_path']).read_text())['groups']
    for g in records:
        key=(g['case_id'],g['method'],g['seed'])
        assert key not in generation and g['status']=='complete' and g['method'] in ('baseline',method_name)
        assert g['seed']==101 and g['split']=='dev' and g['subject'] in split['dev'] and g['subject'] not in split['test']
        path=Path(g['output']).resolve();assert PREVIEW in path.parents and digest(path)==g['sha256']
        generation[key]=g
    items=[]
    for case in cases['cases']:
        cid=case['case_id'];assert re.fullmatch(r'[A-Za-z0-9_.-]+',cid)
        pair=pairs[cid];assert pair['subject']==case['subject'] and pair['split']=='dev'
        sg=pair['source_au_vector_0_5'];tg=pair['target_au_vector_0_5']
        assert case['source_au_vector_0_5']==sg and case['target_au_vector_0_5']==tg
        assert pair['au_order']==list(AUS)
        for side,values in [('source',sg),('target',tg)]:
            assert pair['aligned_'+side]['input']['au_vector_0_5']==values
        for method in ('source_real','target_real','baseline','new_v3'):
            if method.endswith('_real'):
                side=method.split('_')[0];path=PREVIEW/cid/(method+'.png');expected=pair['aligned_'+side]['aligned_sha256'];original_method=method
            else:
                original_method='baseline' if method=='baseline' else method_name;g=generation[(cid,original_method,101)]
                assert g['source_aus_0_5']==sg and g['target_aus_0_5']==tg and g['prompt']==pair['prompt']
                path=Path(g['output']);expected=g['sha256']
            assert digest(path)==expected
            items.append({'case_id':cid,'subject':case['subject'],'group':case['group'],'split':'dev','method':method,'original_method':original_method,
                          'seed':101 if method in ('baseline','new_v3') else None,'path':str(path),'sha256':expected,
                          'source_labels_12':sg,'target_labels_12':tg})
    assert len(items)==48 and len({r['case_id'] for r in items})==12
    return items,settings,meta

def openface(item,out):
    folder=out/'openface'/item['case_id']/item['method'];folder.mkdir(parents=True)
    command=[str(TOOLS/'FaceLandmarkImg'),'-f',item['path'],'-aus','-mloc',str(TOOLS/'build/bin/model/main_clnf_general.txt'),'-out_dir',str(folder)]
    started=time.monotonic()
    try:
        with (folder/'run.log').open('x') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        files=list(folder.glob('*.csv'));parsed=helper.parse_openface(files[0]) if len(files)==1 else {'parse_status':'missing_or_multiple_csv','vector':None,'confidence':None,'raw_rows':[],'face_count':None}
        if p.returncode:parsed.update(parse_status='process_failed',vector=None)
        parsed.update(returncode=p.returncode,csv_files=[str(p) for p in files])
    except Exception as error:parsed={'parse_status':'error','error':str(error),'vector':None,'confidence':None,'raw_rows':[],'face_count':None}
    assert digest(item['path'])==item['sha256']
    record={**item,**parsed,'seconds':time.monotonic()-started,'command':command};save(folder/'parsed.json',record);return record

def identity(items,out):
    import cv2
    import numpy as np
    from PIL import Image
    from insightface.app import FaceAnalysis
    modelroot=ROOT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface'
    for name,expected in EXPECTED_MODELS.items():assert digest(modelroot/'models/buffalo_l'/name)==expected
    with (out/'identity_runtime.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        app=FaceAnalysis(name='buffalo_l',root=str(modelroot),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1,det_size=(640,640),det_thresh=.5)
    rows=[];embeddings={}
    for item in items:
        assert digest(item['path'])==item['sha256']
        bgr=cv2.imread(item['path'],cv2.IMREAD_COLOR);assert bgr is not None
        assert np.array_equal(bgr,np.asarray(Image.open(item['path']).convert('RGB'))[:,:,::-1])
        padded=cv2.copyMakeBorder(bgr,128,128,128,128,cv2.BORDER_CONSTANT,value=(128,128,128))
        found=app.get(padded);key=(item['case_id'],item['method'])
        row={**item,'face_count':len(found),'faces':[{'bbox':f.bbox.tolist(),'score':float(f.det_score),'kps':f.kps.tolist()} for f in found],
             'identity_cosine':None,'valid_single_face':False,'padding_px':128,'padding_value':128,'detector_threshold':.5}
        if len(found)==1:
            emb=np.asarray(found[0].normed_embedding,dtype=np.float64)
            if emb.shape==(512,) and np.isfinite(emb).all() and np.linalg.norm(emb)>0:
                embeddings[key]=emb/np.linalg.norm(emb);row['valid_single_face']=True
        rows.append(row)
    for r in rows:
        k=(r['case_id'],r['method']);s=(r['case_id'],'source_real')
        if k in embeddings and s in embeddings:r['identity_cosine']=float(np.dot(embeddings[k],embeddings[s]))
    payload={'model_sha256':EXPECTED_MODELS,'provider':'CPUExecutionProvider','padding_px':128,'padding_value':128,'detector_size':[640,640],
             'detector_threshold':.5,'fallback_used':False,'embeddings_saved':False,'test_used':False,
             'identity_space':'buffalo_l conditioning encoder space; not fully independent',
             'versions':{k:importlib.metadata.version(k) for k in ('insightface','onnxruntime','numpy','opencv-python','Pillow')},'records':rows}
    save(out/'identity_results.json',payload);return {(r['case_id'],r['method']):r for r in rows}

def summarize(rows):
    keys=['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','target_mae_all12','delta_mae_all12','direction_correct_fraction','identity_cosine']
    def one(part):
        d={'planned':len(part),'au_valid':sum(r['vector'] is not None for r in part),'identity_single_face':sum(r['identity']['valid_single_face'] for r in part),
           'identity_pairs_valid':sum(r['identity_cosine'] is not None for r in part)}
        for key in keys:
            values=[r[key] for r in part if r[key] is not None];d[key]={'mean':mean(values),'n':len(values)}
        return d
    aggregate={}
    for method in ('source_real','target_real','baseline','new_v3'):
        selected=[r for r in rows if r['method']==method]
        aggregate[method]={scope:one([r for r in selected if scope=='all' or (r['group']=='no_edit')==(scope=='no_edit')]) for scope in ('all','edited','no_edit')}
        aggregate[method]['per_subject']={s:{scope:one([r for r in selected if r['subject']==s and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit'))]) for scope in ('all','edited','no_edit')} for s in sorted({r['subject'] for r in selected})}
    mapping={(r['case_id'],r['method']):r for r in rows};paired=[]
    for cid in sorted({r['case_id'] for r in rows}):
        a,b=mapping[(cid,'baseline')],mapping[(cid,'new_v3')]
        paired.append({'case_id':cid,'subject':a['subject'],'group':a['group'],**{k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in keys}})
    differences={}
    for scope in ('all','edited','no_edit'):
        chosen=[r for r in paired if scope=='all' or (r['group']=='no_edit')==(scope=='no_edit')]
        bysubject={s:{k:mean([r[k] for r in chosen if r['subject']==s and r[k] is not None]) for k in keys} for s in sorted({r['subject'] for r in chosen})}
        differences[scope]={'per_subject':bysubject,'subject_equal_mean':{k:mean([v[k] for v in bysubject.values() if v[k] is not None]) for k in keys},
                            'paired_valid_cases':{k:sum(r[k] is not None for r in chosen) for k in keys}}
    return {'by_method':aggregate,'paired_new_minus_baseline':paired,'paired_difference_summary':differences,
            'difference_direction':'negative is lower error/drift; positive is higher identity/direction fraction',
            'aggregation':'case metrics first; report subject-equal paired mean and coverage; no missing imputation'}

def main():
    output=HERE/'run';assert not output.exists()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',MPLCONFIGDIR=str(HERE/'cache/matplotlib'),XDG_CACHE_HOME=str(HERE/'cache/xdg'),
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPYCACHEPREFIX=str(HERE/'cache/pycache'))
    sys.dont_write_bytecode=True;sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']
    items,settings,meta=prepare();seal=json.loads((TOOLS/'tool_seal.json').read_text())
    for entry in [seal['binary'],seal['launcher'],*seal['model_files']]:assert digest(TOOLS/entry['path'])==entry['sha256']
    output.mkdir();save(output/'protocol.json',{'items':items,'test_used':False,'sota_conclusion':False,'workers':2,'au_order':AUS,
          'au_policy':'original canonical512, officialCLNF FaceLandmarkImg static AU, singleton finite0-5; continuous confidence retained, no outcome-selected cutoff',
          'identity_policy':'all canonical images gray128 padding128, FaceAnalysis CPU det640 threshold0.5, singleton only, no fallback',
          'C_definition':'manual source12AU != target12AU; no_edit is separate, C metrics NA when empty',
          'input_preview_result_sha256':digest(PREVIEW/'result.json'),'input_records_sha256':digest(PREVIEW/'generation_records.jsonl'),
          'cache_manifest_sha256':digest(CACHE/'manifest.json'),'evaluator_sha256':digest(__file__),'parser_helper_sha256':digest(Path(helper.__file__)),
          'tool_seal_sha256':digest(TOOLS/'tool_seal.json'),'settings':settings})
    (output/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes());(output/'parser_helper_snapshot.py').write_bytes(Path(helper.__file__).read_bytes())
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(openface,item,output) for item in items]
        for task in as_completed(tasks):
            row=task.result();rows.append(row);print(json.dumps({'done':len(rows),'case_id':row['case_id'],'method':row['method'],'au':row['parse_status']}),flush=True)
    ids=identity(items,output);sources={r['case_id']:r['vector'] for r in rows if r['method']=='source_real'}
    for row in rows:
        row.update(au_metrics(row['vector'],sources[row['case_id']],row['source_labels_12'],row['target_labels_12']))
        row['identity']=ids[(row['case_id'],row['method'])];row['identity_cosine']=row['identity']['identity_cosine']
    rows.sort(key=lambda r:(r['case_id'],r['method']));report=summarize(rows)
    report.update(status='complete',test_used=False,sota_conclusion=False,records=rows,au_order=AUS,planned_generated=24,planned_real_controls=24)
    save(output/'results.json',report)
    print(json.dumps({'status':'complete','all48_retained':len(rows)==48,'generated_coverage':{k:report['by_method'][k]['all'] for k in ('baseline','new_v3')},
                      'paired_edited':report['paired_difference_summary']['edited'],'paired_no_edit':report['paired_difference_summary']['no_edit']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
