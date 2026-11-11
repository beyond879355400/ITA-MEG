#!/usr/bin/env python3
"""Fixed v2 OpenFace primary-face selection by68-landmark bbox area only."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;TOOLS=ROOT/'tools/openface'
POLICY_SHA='d29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
AUS12=[1,2,4,5,6,9,12,15,17,20,25,26];AUS17=[1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def parse(path):
    with path.open() as f:raw=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(f)]
    candidates=[]
    for index,row in enumerate(raw):
        box=area=None
        try:
            x=[float(row[f'x_{i}']) for i in range(68)];y=[float(row[f'y_{i}']) for i in range(68)]
            if all(math.isfinite(v) for v in x+y):
                box=[min(x),min(y),max(x),max(y)];area=(box[2]-box[0])*(box[3]-box[1]);area=area if area>0 else None
        except (KeyError,ValueError):pass
        candidates.append({'row_index':index,'face_id':row.get('face'),'bbox68':box,'area68':area,'confidence_raw':row.get('confidence')})
    eligible=[c for c in candidates if c['area68'] is not None]
    selected=max(eligible,key=lambda c:(c['area68'],-c['row_index'])) if eligible else None
    result={'raw_rows':raw,'face_count':len(raw),'all_candidates':candidates,'selected_row':None if selected is None else selected['row_index'],
            'vector':None,'selected_au17':None,'confidence':None,'parse_status':'no_face' if not raw else 'no_valid_landmark_geometry',
            'selection':'maximum68-landmark bbox area; tie first raw row; no metric score used'}
    areas=sorted((c['area68'] for c in eligible),reverse=True)
    result['second_to_first_area_ratio']=areas[1]/areas[0] if len(areas)>1 else None
    if selected is None:return result
    row=raw[selected['row_index']];values={}
    for au in AUS17:
        try:v=float(row[f'AU{au:02d}_r']);v=v if math.isfinite(v) and 0<=v<=5 else None
        except (KeyError,ValueError):v=None
        values[str(au)]=v
    try:confidence=float(row['confidence']);confidence=confidence if math.isfinite(confidence) else None
    except (KeyError,ValueError):confidence=None
    result.update(selected_au17=values,confidence=confidence,parse_status='selected_face_invalid_values')
    if confidence is not None and all(values[str(au)] is not None for au in AUS12):
        result.update(vector=[values[str(au)] for au in AUS12],parse_status='primary_face_finite')
    return result

def measure(item,output):
    identity=item.get('sample_id') or (item['case_id']+'__'+item['method']);folder=output/'raw'/identity;folder.mkdir(parents=True)
    command=[str(TOOLS/'FaceLandmarkImg'),'-f',item['path'],'-aus','-2Dfp','-mloc',str(TOOLS/'build/bin/model/main_clnf_general.txt'),'-out_dir',str(folder)]
    start=time.monotonic()
    try:
        with (folder/'run.log').open('x') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        files=list(folder.glob('*.csv'));result=parse(files[0]) if len(files)==1 else {'parse_status':'missing_or_multiple_csv','vector':None,'selected_au17':None,'raw_rows':[],'all_candidates':[],'selected_row':None,'face_count':None,'confidence':None}
        if p.returncode:result.update(parse_status='process_failed',vector=None,selected_au17=None)
        result.update(returncode=p.returncode,csv_files=[str(path) for path in files])
    except Exception as error:result={'parse_status':'error','error':str(error),'vector':None,'selected_au17':None,'raw_rows':[],'all_candidates':[],'selected_row':None,'face_count':None,'confidence':None}
    unchanged=sha(item['path'])==item['sha256']
    if not unchanged:result.update(parse_status='input_changed',vector=None,selected_au17=None)
    result.update(item);result.update(policy_sha256=POLICY_SHA,seconds=time.monotonic()-start,input_unchanged=unchanged,command=command)
    if 'labels12' in item:result['labels']=item['labels12']
    save(folder/'parsed.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['real240','dev19']);args=p.parse_args()
    assert sha(HERE/'policy.json')==POLICY_SHA;policy=json.loads((HERE/'policy.json').read_text())
    name='real_dev240_items.json' if args.mode=='real240' else 'dev19_items.json';manifest=json.loads((HERE/name).read_text())
    assert sha(HERE/name)==policy['inputs'][name]['sha256'] and not manifest['test_used']
    items=manifest['items'];assert all(r['split']=='dev' and sha(r['path'])==r['sha256'] for r in items)
    if args.mode=='dev19':
        for dependency in ('real_dev_openface/summary.json','facenet_real12/summary.json'):
            ready=json.loads((HERE/dependency).read_text());assert ready['status']=='complete' and ready['policy_sha256']==POLICY_SHA
    output=HERE/('real_dev_openface' if args.mode=='real240' else 'dev19_openface');assert not output.exists();output.mkdir()
    seal=json.loads((TOOLS/'tool_seal.json').read_text());assert sha(TOOLS/'tool_seal.json')==policy['openface']['tool_seal_sha256']
    for item in [seal['binary'],seal['launcher'],*seal['model_files']]:assert sha(TOOLS/item['path'])==item['sha256']
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    save(output/'protocol.json',{'policy_sha256':POLICY_SHA,'input_manifest_sha256':sha(HERE/name),'script_sha256':sha(__file__),'tool_seal_sha256':sha(TOOLS/'tool_seal.json'),
         'phase':args.mode,'items':items,'test_used':False,'workers':2,'prior_reports_modified':False})
    (output/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records=[];start=time.monotonic()
    with (output/'per_image.jsonl').open('x',encoding='utf-8',buffering=1) as stream,ThreadPoolExecutor(max_workers=2) as workers:
        for future in as_completed([workers.submit(measure,item,output) for item in items]):
            result=future.result();records.append(result);stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            if len(records)%20==0:print(json.dumps({'done':len(records),'planned':len(items),'valid':sum(r['vector'] is not None for r in records)}),flush=True)
    records.sort(key=lambda r:r.get('sample_id') or (r['case_id'],r['method']))
    payload={'status':'complete','policy_sha256':POLICY_SHA,'planned':len(items),'valid':sum(r['vector'] is not None for r in records),
             'multi_candidate_images':sum((r['face_count'] or 0)>1 for r in records),'seconds':time.monotonic()-start,'test_used':False,'records':records}
    if args.mode=='real240':
        sys.path.insert(0,str(HERE.parent/'real_dev_calibration'));from calibrate_openface import summarize
        stats=summarize(records);payload['calibration']=stats
        # Independent direct MAE recomputation; not merely rereading the same helper.
        valid=[r for r in records if r['vector'] is not None]
        direct=sum(sum(abs(a-b) for a,b in zip(r['vector'],r['labels']))/12 for r in valid)/len(valid)
        assert abs(direct-stats['macro_mae'])<1e-12
        oldpath=HERE.parent/'real_dev_calibration/openface_202609141729/per_frame.jsonl'
        old={r['sample_id']:r for r in [json.loads(line) for line in oldpath.read_text().splitlines()]}
        common=[r for r in records if old[r['sample_id']]['vector'] is not None and r['vector'] is not None]
        payload['legacy_singleton_comparison']={'source_sha256':sha(oldpath),'old_valid':sum(r['vector'] is not None for r in old.values()),
            'common_valid':len(common),'exact_equal_12au':sum(r['vector']==old[r['sample_id']]['vector'] for r in common),
            'newly_valid_sample_ids':[r['sample_id'] for r in records if r['vector'] is not None and old[r['sample_id']]['vector'] is None],
            'changed_vectors':[{'sample_id':r['sample_id'],'old':old[r['sample_id']]['vector'],'new':r['vector']} for r in common if r['vector']!=old[r['sample_id']]['vector']]}
    save(output/'results.json',payload)
    summary={k:v for k,v in payload.items() if k!='records'};save(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('calibration','legacy_singleton_comparison')},indent=2))

if __name__=='__main__':main()
