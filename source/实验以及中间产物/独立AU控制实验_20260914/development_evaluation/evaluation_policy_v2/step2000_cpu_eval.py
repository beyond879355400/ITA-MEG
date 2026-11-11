#!/usr/bin/env python3
"""Minimal actual-run entry: gate, frozen CPU evaluators, same-metric merge."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import itertools
import json
import os
from pathlib import Path
import re
import sys
from step2000_gate import HERE,ROOT,POLICY_SHA,LABELS,REFERENCES,require,read,sha,save,validate,prepare,check_input_metadata
from openface_v2 import measure,TOOLS

def checked(group):
    require(re.fullmatch('step2000_[a-z0-9_]+',group) is not None,'Invalid group name');out=HERE/group
    seal=read(out/'items_seal.json');doc=read(out/'items.json');require(seal['policy_sha256']==POLICY_SHA and sha(out/'items.json')==seal['sha256'],'Input seal mismatch')
    require(sha(out/'reference_records.json')==seal['reference_records_sha256'],'Reference records changed')
    for item in doc['items']:check_input_metadata(item)
    actual=validate(doc['run']);actual.pop('reference_records');require(actual==doc,'Source run/config/weights/images changed since preparation')
    return out,doc

def derived_identity_source(kind,group,digest):
    """Only input directory/hash/count changes in pinned CPU-only evaluator cores."""
    require(re.fullmatch('step2000_[a-z0-9_]+',group) is not None and re.fullmatch('[0-9a-f]{64}',digest) is not None,'Unsafe adapter binding')
    if kind=='facenet':
        core=HERE/'run_facenet_single_directions.py';require(sha(core)=='5d8e1916bb5bcc9afc307980438b5f2f3fb0610f5dde12141f90e85efa7ebc2d','FaceNet core changed');source=core.read_text()
        replacements={"root = HERE / 'single_directions'":f"root = HERE / '{group}'","INPUT_SHA = 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'":f"INPUT_SHA = '{digest}'"}
        for a,b in replacements.items():require(source.count(a)==1,'Ambiguous FaceNet adaptation');source=source.replace(a,b)
        class Count(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) is int and node.value==57:node.value=95
                return node
        source=ast.unparse(ast.fix_missing_locations(Count().visit(ast.parse(source))))+'\n'
    else:
        require(kind=='buffalo','Unknown identity evaluator');core=HERE/'append_buffalo.py';require(sha(core)=='96daabea1e618a993de22ef684a4cbdcc6822911191a424fc457ea19012af88a','Buffalo core changed');source=core.read_text()
        replacements={"manifest=HERE/'single_directions/items.json'":f"manifest=HERE/'{group}/items.json'","HERE/'single_directions/items_seal.json'":f"HERE/'{group}/items_seal.json'","out=HERE/'single_directions/buffalo'":f"out=HERE/'{group}/buffalo'","len(items)==57":"len(items)==95"}
        for a,b in replacements.items():require(source.count(a)==1,'Ambiguous buffalo adaptation');source=source.replace(a,b)
    ast.parse(source);return core,source

def identity(kind,group):
    out,doc=checked(group);core,source=derived_identity_source(kind,group,sha(out/'items.json'));require(not(out/kind).exists(),'Evaluation output exists')
    scope={'__name__':'step2000_pinned_cpu_identity','__file__':str(Path(__file__).resolve())};exec(compile(source,str(core),'exec'),scope);scope['main']()
    (out/kind/'core_snapshot.py').write_bytes(core.read_bytes());(out/kind/'derived_evaluator_snapshot.py').write_text(source)
    save(out/kind/'adaptation.json',{'policy_sha256':POLICY_SHA,'core_sha256':sha(core),'derived_sha256':sha(out/kind/'derived_evaluator_snapshot.py'),'entry_sha256':sha(__file__),
          'changes':'only input directory, frozen manifest SHA and95 logical records; face-selection/model/preprocessing unchanged','test_used':False})

def openface(group):
    out,doc=checked(group);folder=out/'openface';require(not folder.exists(),'OpenFace output exists')
    for name in ['real_dev_openface/summary.json','facenet_real12/summary.json']:
        ready=read(HERE/name);require(ready['status']=='complete' and ready['policy_sha256']==POLICY_SHA,'Calibration not ready')
    policy=read(HERE/'policy.json');tool=read(TOOLS/'tool_seal.json');require(sha(TOOLS/'tool_seal.json')==policy['openface']['tool_seal_sha256'],'OpenFace tool seal changed')
    for item in [tool['binary'],tool['launcher'],*tool['model_files']]:require(sha(TOOLS/item['path'])==item['sha256'],'OpenFace binary/model changed')
    items=[r for r in doc['items'] if r['method'] in LABELS.values()];require(len(items)==76,'Expected76 new output measurements')
    folder.mkdir();save(folder/'protocol.json',{'policy_sha256':POLICY_SHA,'items_sha256':sha(out/'items.json'),'items':items,'script_sha256':sha(__file__),'helper_sha256':sha(HERE/'openface_v2.py'),'test_used':False})
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');rows=[]
    with (folder/'per_image.jsonl').open('x',buffering=1) as stream,ThreadPoolExecutor(max_workers=2) as pool:
        for job in as_completed([pool.submit(measure,item,folder) for item in items]):
            row=job.result();rows.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'planned':76,'valid':sum(r['vector'] is not None for r in rows),'records':sorted(rows,key=lambda r:(r['case_id'],r['method']))}
    save(folder/'results.json',result);save(folder/'summary.json',{k:v for k,v in result.items() if k!='records'})

def enrich_new_measurement(au_row,source_row,target_row,face_row,buffalo_row):
    """Keep the new AU parser result; old source is only a numeric reference."""
    from summarize_v2 import e,extra_errors,EXTRA
    require(au_row['source_labels_12']==source_row['source_labels_12'] and au_row['target_labels_12']==source_row['target_labels_12'],'New output labels differ from source reference')
    require(face_row['sha256']==buffalo_row['sha256']==au_row['sha256'],'New measurement identity binding differs')
    merged=dict(au_row)
    merged.update(e.au_metrics(au_row['vector'],source_row['vector'],au_row['source_labels_12'],au_row['target_labels_12']))
    merged.update(facenet_primary=face_row,buffalo_auxiliary=buffalo_row,identity_cosine=face_row['identity_cosine'],buffalo_identity_cosine=buffalo_row['identity_cosine'])
    merged['extra5'],merged['extra5_proxy_mae_to_real_target']=extra_errors(au_row['selected_au17'],target_row['selected_au17'],source_row['selected_au17'])
    for a in EXTRA:merged[f'extra_AU{a}_error_to_real_target']=merged['extra5'][str(a)]['absolute_error_to_target']
    for key in ('vector','raw_rows','selected_au17','confidence','all_candidates','selected_row','parse_status'):
        if key in au_row:require(merged[key]==au_row[key],'Fresh AU measurement overwritten: '+key)
    return merged

def merge(group):
    from summarize_v2 import e,compare,extra_errors,other_compare,strata_helper,EXTRA
    out,doc=checked(group);rows=read(out/'reference_records.json')['records'];source={r['case_id']:r for r in rows if r['method']=='source_real'};target={r['case_id']:r for r in rows if r['method']=='target_real'}
    au=read(out/'openface/results.json');fn=read(out/'facenet/results.json');bf=read(out/'buffalo/results.json')
    require(all(r['policy_sha256']==POLICY_SHA and r['test_used'] is False for r in [au,fn,bf]),'Mixed evaluation policy');require(au['status']==bf['status']=='complete' and fn['summary']['status']=='complete','Evaluation incomplete')
    require(len(au['records'])==76 and len(fn['records'])==len(bf['records'])==95,'Incomplete evaluator rows')
    fm={(r['case_id'],r['method']):r for r in fn['records']};bm={(r['case_id'],r['method']):r for r in bf['records']}
    for r in au['records']:
        key=(r['case_id'],r['method']);s=source[r['case_id']];t=target[r['case_id']]
        require(fm[key]['sha256']==bm[key]['sha256']==r['sha256']==sha(r['path']),'Cross-evaluator output mismatch')
        rows.append(enrich_new_measurement(r,s,t,fm[key],bm[key]))
    methods=list(REFERENCES)+list(LABELS.values());require(len(rows)==209 and len({(r['case_id'],r['method']) for r in rows})==209,'Wrong merged row count')
    cohorts={c:[r for r in rows if c=='all' or r['cohort']==c] for c in ['all','original','additional']};pairs={a+'_vs_'+b:[a,b] for a,b in itertools.combinations(methods,2)}
    extra_keys=['buffalo_identity_cosine','extra5_proxy_mae_to_real_target']+[f'extra_AU{a}_error_to_real_target' for a in EXTRA]
    coverage={m:{scope:{'planned':len(part:=[r for r in rows if r['method']==m and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit'))]),'AU_valid':sum(r['vector'] is not None for r in part),
                       'FaceNet_valid':sum(r['identity_cosine'] is not None for r in part),'buffalo_valid':sum(r['buffalo_identity_cosine'] is not None for r in part)} for scope in ['all','edited','no_edit']} for m in methods}
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'same_training_protocol_sota_ranking':False,'methods':methods,'case_count':19,'generated_count':171,'records':rows,'coverage':coverage,
       'input_groups':{'source_plus_text_manual_AU':['baseline']+list(LABELS.values()),'source_plus_manual_AU_native':['magicface'],'source_plus_real_driver':['liveportrait','faceadapter','liveportrait_absolute']},
       'all_methods_common_valid':{c:compare(part,methods) for c,part in cohorts.items()},'pairwise':{name:{c:compare(part,ms) for c,part in cohorts.items()} for name,ms in pairs.items()},
       'au_level_all_methods':strata_helper.stratify(rows,methods),'au_level_pairwise':{name:strata_helper.stratify(rows,ms) for name,ms in pairs.items()},
       'extra_diagnostics':{'all_methods':other_compare(rows,methods,extra_keys),'pairwise':{name:other_compare(rows,ms,extra_keys) for name,ms in pairs.items()}},
       'gate_input_sha256':sha(out/'items.json'),'old_step200_reports_overwritten':False,'script_sha256':sha(__file__)}
    folder=out/'comparison';folder.mkdir(exist_ok=False);save(folder/'results.json',result)
    # Reuse the independent arithmetic/geometry verifier with only output-folder binding.
    core=HERE/'verify_nine_comparison.py';text=core.read_text();old="choices=('comparison_nine',)";require(text.count(old)==1,'Verifier interface changed')
    text=text.replace(old,f"choices=('{group}/comparison',)");scope={'__name__':'step2000_independent_verifier','__file__':str(core)}
    previous=sys.argv;sys.argv=[str(core),group+'/comparison']
    try:exec(compile(text,str(core),'exec'),scope);scope['main']()
    finally:sys.argv=previous
    (folder/'verifier_snapshot.py').write_text(text)
    save(folder/'verification_binding.json',{'core_sha256':sha(core),'derived_sha256':sha(folder/'verifier_snapshot.py'),'changes':'only comparison-folder CLI binding; arithmetic and geometry checks unchanged','test_used':False})
    print(json.dumps({'status':'complete','group':group,'record_count':209,'results_sha256':sha(folder/'results.json')},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('validate','prepare','openface','facenet','buffalo','merge'));p.add_argument('--run',type=Path);p.add_argument('--group');a=p.parse_args()
    if a.phase in ('validate','prepare'):
        require(a.run is not None,'Supply actual completed generation --run')
        if a.phase=='validate':doc=validate(a.run);print(json.dumps({'status':doc['status'],'run':doc['run'],'logical_count':doc['logical_count']},ensure_ascii=False))
        else:require(a.group is not None,'New output --group required');print(prepare(a.run,a.group))
    else:
        require(a.group is not None,'Prepared --group required')
        if a.phase=='openface':openface(a.group)
        elif a.phase in ('facenet','buffalo'):identity(a.phase,a.group)
        else:merge(a.group)
if __name__=='__main__':main()
