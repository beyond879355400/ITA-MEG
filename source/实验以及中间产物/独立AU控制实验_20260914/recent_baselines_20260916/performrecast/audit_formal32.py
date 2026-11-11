#!/usr/bin/env python3
"""Model-free final32 byte, frame, and finite-value verification; no reruns."""
import argparse,collections,json,math,time
from pathlib import Path
from formal_batch import read,sha,png_size,new_json

PIN='99dd906c8b60656c5c2734d0f7a32714e0b67b9ae381d3e51ddc8f0c306da460'
EVAL='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'

def audit(folder):
    started=time.monotonic();protocol_path=folder/'frozen_protocol.json';index_path=folder/'generation_index.json'
    if sha(protocol_path)!=PIN:raise RuntimeError('Frozen protocol changed')
    protocol=read(protocol_path);index=read(index_path)
    if index['protocol_sha256']!=PIN or index['evaluation_protocol_sha256']!=EVAL:raise RuntimeError('Index protocol differs')
    if index['planned_cases']!=32 or index['seed_aliases']!=0 or index['selected_frame']!=0:raise RuntimeError('Index quantity/selection differs')
    if [r['case_id'] for r in index['rows']]!=protocol['case_order']:raise RuntimeError('Missing/reordered cases')
    if len(set(r['case_id'] for r in index['rows']))!=32:raise RuntimeError('Duplicated case')
    counts=collections.Counter();pngs=0;checks=[];durations=[];subjects=collections.Counter()
    for row,case in zip(index['rows'],protocol['cases']):
        terminal=read(Path(row['receipt_path']));counts[terminal['status']]+=1;subjects[case['subject']]+=1
        if row['status']!=terminal['status'] or terminal['case_id']!=case['case_id'] or terminal['protocol_sha256']!=PIN:raise RuntimeError('Terminal binding differs')
        if row['seed']!=101 or row['seed_alias'] is not False:raise RuntimeError('Random-run alias introduced')
        for role,key in [('source','source'),('target','target_reference')]:
            if row[key]!=case[role+'_absolute_path'] or row[role+'_sha256']!=case[role+'_sha256']:raise RuntimeError('Input binding differs')
        if terminal['status']=='failed':
            if row['output'] is not None or not terminal.get('error'):raise RuntimeError('Failure not retained')
            checks.append({'case_id':case['case_id'],'status':'failed','error':terminal['error']});continue
        if terminal['status']!='complete':raise RuntimeError('Nonterminal case')
        selected=Path(terminal['attempt'])/'native_crop_png/frame0000.png'
        if row['output']!=str(selected) or row['sha256']!=terminal['output_sha256'] or sha(selected)!=row['sha256']:raise RuntimeError('Selected PNG differs')
        if len(terminal['outputs'])!=12 or len(terminal['finite_generated_tensors'])!=4:raise RuntimeError('Original frame count differs')
        for item in terminal['outputs']:
            path=Path(item['path'])
            if sha(path)!=item['sha256'] or png_size(path)!=item['size']:raise RuntimeError('Native PNG hash or size differs')
            pngs+=1
        for item in terminal['finite_generated_tensors']:
            if not all(math.isfinite(item[k]) for k in ('min','max')):raise RuntimeError('Nonfinite generated tensor')
        if any(not item['decoded4_frames_exact'] for item in terminal['inputs'].values()):raise RuntimeError('Lossless input gate missing')
        durations.append(terminal['seconds']);checks.append({'case_id':case['case_id'],'status':'complete','output_sha256':row['sha256'],'native_pngs_verified':12})
    if dict(subjects)!={s:4 for s in protocol['case_audit']['subjects']}:raise RuntimeError('Subject balance changed')
    if counts['complete']!=index['successful_cases'] or counts['failed']!=index['failed_cases']:raise RuntimeError('Summary count differs')
    summary={'kind':'performrecast32_completed_native_output_audit','status':'pass','protocol_sha256':PIN,'evaluation_protocol_sha256':EVAL,
             'index_path':str(index_path),'index_sha256':sha(index_path),'cases':32,'subjects':dict(subjects),'counts':dict(counts),
             'native_pngs_verified':pngs,'case_checks':checks,'case_seconds':durations,'timing_use':'execution_record_only_not_method_speed_ranking',
             'audit_seconds':round(time.monotonic()-started,3),'models_loaded':False}
    new_json(folder/'completion_audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('case_checks','case_seconds')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();audit(a.run_dir.resolve())
