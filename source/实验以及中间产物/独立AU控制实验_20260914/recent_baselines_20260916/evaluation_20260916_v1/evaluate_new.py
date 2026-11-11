"""Versioned incremental measurement and case-level Holm5 reporting.

Default commands validate metadata only. Any real prepare/measurement/export
requires both --execute and root approval pinned to this protocol SHA.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
from protocol import *


def new_directory(path):
    p=Path(path).resolve();need(p.is_relative_to(HERE)and not p.exists(),'Output must be new evaluation subdirectory')
    p.mkdir(parents=True);return p


def prepare(protocol_path,protocol_sha,index_refs,output):
    control=load_protocol(protocol_path,protocol_sha);c,e,cases,old,refs=context()
    indexes=[read(verified(r))for r in index_refs];planned,normalized=normalize(cases,indexes,protocol_sha)
    out=new_directory(output);items=[];availability=[];meta=old['case_metadata']
    def add(row,image,role):
        try:
            item=e.inspect_image_item(row,image,role);items.append(item)
            availability.append({'key':list(key(row)),'status':'available','role':role,'image_size':item['image_size']})
        except Exception as error:
            availability.append({'key':list(key(row)),'status':'unavailable','role':role,'error':str(error)})
    for r in refs['cases']:
        for role in ['source','target']:
            row={**meta[r['case_id']],'method':role+'_real','seed':0};reference=r[role+'_reference']
            if reference['status']=='complete':add(row,reference['image'],'real_'+role)
            else:availability.append({'key':list(key(row)),'status':'reference_failed','role':'real_'+role,'error':reference.get('error')})
    for p in planned:
        g=normalized[key(p)]
        if g['status']=='complete':
            add({**meta[p['case_id']],**p,'execution_id':g['execution_id'],'execution_started':True},g['output'],'generated')
        else:availability.append({'key':list(key(p)),'status':g['status'],'role':'generated','error':g['failure_reason']})
    need(len({key(r)for r in items})==len(items),'Duplicate measurement items')
    document={'kind':'recent_external_measurement_inputs_v1','status':'prepared','data_role':ROLE,
              'independent_final_test':False,'previously_exposed_subjects':True,'test_used':True,
              'protocol':ref(protocol_path),'protocol_sha256':protocol_sha,'measurement_policy_sha256':POLICY_SHA,
              'generation_indexes':index_refs,'case_metadata':meta,'planned':planned,
              'normalized_generation_records':list(normalized.values()),'items':items,'availability':availability,
              'old_metrics':ref(pin('old_metrics')),'old_items':ref(pin('old_items')),'references':ref(pin('references')),
              'physical_planned_outputs':{'swiftedit':96,'performrecast':32},
              'seed_minus1_is_not_actual_random_seed':True,'algorithm_metadata_only_adapter':True}
    save(out/'items.json',document)
    save(out/'items_seal.json',{'sha256':sha(out/'items.json'),'policy_sha256':POLICY_SHA,'protocol_sha256':protocol_sha})
    return out


def job_control(job):
    job=Path(job).resolve();need(job.is_relative_to(HERE),'Job outside versioned directory')
    doc=read(job/'items.json');seal=read(job/'items_seal.json')
    load_protocol(doc['protocol']['path'],doc['protocol']['sha256'])
    need(sha(job/'items.json')==seal['sha256']and doc['protocol_sha256']==seal['protocol_sha256']==doc['protocol']['sha256'],'Job seal changed')
    need(seal['policy_sha256']==doc['measurement_policy_sha256']==POLICY_SHA,'Measurement policy changed')
    for r in doc['generation_indexes']+[doc['old_metrics'],doc['old_items'],doc['references']]:verified(r)
    return job,doc


def set_cache(job):
    for key0 in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','TORCH_HOME']:
        p=job/'cache'/key0.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key0]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def measurement_runtime_guard(runtime_receipt,protocol_sha,for_bridge=False):
    r=read(verified(runtime_receipt))
    need(r.get('granted_by')=='root'and r.get('protocol_sha256')==protocol_sha,'Root runtime binding required')
    allowed=['original_frozen_environment','bridged_environment_passed']+(['bridge_candidate']if for_bridge else [])
    need(r.get('runtime_status')in allowed,'Runtime not original or bridged')
    executable=Path(sys.executable).absolute()
    need(str(executable)==r.get('python_executable'),'Interpreter differs from approved runtime')
    if r['runtime_status']=='bridged_environment_passed':
        bridge=read(verified(r['bridge_receipt']))
        need(bridge.get('status')=='pass'and bridge.get('measurement_policy_sha256')==POLICY_SHA,'Bridge comparison not passed')
    return r


def measure(job,kind,runtime_receipt):
    job,doc=job_control(job);runtime=measurement_runtime_guard(runtime_receipt,doc['protocol_sha256'],doc['kind']=='recent_external_bridge_inputs_v1');c,e=legacy()
    out=job/kind;need(not out.exists(),'Never overwrite measurements');set_cache(job)
    for item in doc['items']:verified({'path':item['path'],'sha256':item['sha256']})
    if kind=='openface':
        out.mkdir();records,physical=e.openface_items(doc['items'],out)
        save(out/'results.json',{'status':'complete','records':records,'unique_measurements':physical,
             'policy_sha256':POLICY_SHA,'data_role':ROLE,'protocol_sha256':doc['protocol_sha256'],
             'measurement_policy_sha256':POLICY_SHA,'input_sha256':sha(job/'items.json')})
    elif not doc['items']:
        out.mkdir();save(out/'results.json',{'records':[],'policy_sha256':POLICY_SHA,'summary':{'status':'complete'},'no_models_called':True})
    else:
        sources=len({r['case_id']for r in doc['items']if r['method']=='source_real'})
        core,text=e.derive_identity(kind,job,sha(job/'items.json'),len(doc['items']),sources)
        ns={'__name__':'recent_external_frozen_identity','__file__':str(Path(__file__).resolve())}
        exec(compile(text,str(core),'exec'),ns);ns['main']()
        with (out/'derived_snapshot.py').open('x')as f:f.write(text)
    save(out/'binding.json',{'protocol_sha256':doc['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,
       'input_sha256':sha(job/'items.json'),'results_sha256':sha(out/'results.json'),'runtime_receipt':runtime_receipt,
       'algorithm_and_detector_unchanged':True,'deterministic_seed_key_semantics':'-1 not a random draw'})
    return out/'results.json'


def prepare_bridge(protocol_path,protocol_sha,output):
    load_protocol(protocol_path,protocol_sha);c,e,cases,old,refs=context()
    cid=cases[0]['case_id'];source_items=[r for r in old['items']if r['case_id']==cid and
        (r['method']in ['source_real','target_real']or (r['method']==CHOSEN and r['seed']==101))]
    need(len(source_items)==3,'Fixed bridge requires source,target,selected seed101')
    for row in source_items:e.inspect_image_item(row,{'path':row['path'],'sha256':row['sha256']},
          'real_source'if row['method']=='source_real'else 'real_target'if row['method']=='target_real'else 'generated')
    out=new_directory(output)
    doc={'kind':'recent_external_bridge_inputs_v1','status':'prepared','data_role':ROLE,'test_used':True,
         'protocol':ref(protocol_path),'protocol_sha256':protocol_sha,'measurement_policy_sha256':POLICY_SHA,
         'generation_indexes':[],'old_metrics':ref(pin('old_metrics')),'old_items':ref(pin('old_items')),
         'references':ref(pin('references')),'case_metadata':{cid:old['case_metadata'][cid]},'items':source_items,
         'planned':[{'method':CHOSEN,'subject':cases[0]['subject'],'case_id':cid,'seed':101}],
         'bridge_selection':'manifest first case and seed101, no choice by score'}
    save(out/'items.json',doc);save(out/'items_seal.json',{'sha256':sha(out/'items.json'),'policy_sha256':POLICY_SHA,'protocol_sha256':protocol_sha})
    return out


def bridge_compare(job):
    job,doc=job_control(job);need(doc['kind']=='recent_external_bridge_inputs_v1','Not a bridge job')
    tolerances=read(doc['protocol']['path'])['environment_bridge'];mismatches=[];checks=[]
    def compare_value(label,left,right,tolerance):
        if left is None or right is None:ok=left is None and right is None
        elif isinstance(left,(int,float))and isinstance(right,(int,float)):ok=abs(left-right)<=tolerance
        else:ok=left==right
        checks.append({'field':label,'old':left,'new':right,'tolerance':tolerance,'pass':ok})
        if not ok:mismatches.append(label)
    expected_keys={key(r)for r in doc['items']}
    for kind in ['openface','facenet','buffalo']:
        b=read(job/kind/'binding.json');need(b['protocol_sha256']==doc['protocol_sha256']and b['results_sha256']==sha(job/kind/'results.json')and b['input_sha256']==sha(job/'items.json'),'Bridge binding changed')
        original=read(OLD/'all_methods_v1'/kind/'results.json');current=read(job/kind/'results.json')
        oldmap={key(r):r for r in original['records']};newmap={key(r):r for r in current['records']}
        need(set(newmap)==expected_keys,'Missing bridge row')
        for k in sorted(expected_keys):
            a,z=oldmap[k],newmap[k];need(a['sha256']==z['sha256'],'Bridge image mismatch')
            prefix=kind+':'+k[0]
            if kind=='openface':
                for f in ['face_count','selected_row','parse_status']:compare_value(prefix+':'+f,a.get(f),z.get(f),0)
                compare_value(prefix+':candidate_count',len(a.get('all_candidates',[])),len(z.get('all_candidates',[])),0)
                for au in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]:
                    compare_value(prefix+':AU'+str(au),(a.get('selected_au17')or{}).get(str(au)),(z.get('selected_au17')or{}).get(str(au)),tolerances['OpenFace_AU_absolute_tolerance'])
                for i,(aa,zz)in enumerate(zip(a.get('all_candidates',[]),z.get('all_candidates',[]))):
                    compare_value(prefix+f':candidate{i}:row',aa.get('row_index'),zz.get('row_index'),0)
                    if aa.get('bbox68')is None or zz.get('bbox68')is None:compare_value(prefix+f':candidate{i}:bbox',aa.get('bbox68'),zz.get('bbox68'),0)
                    else:
                        for j,(x,y)in enumerate(zip(aa['bbox68'],zz['bbox68'])):compare_value(prefix+f':candidate{i}:bbox{j}',x,y,tolerances['geometry_absolute_tolerance'])
            else:
                for f in ['face_count','selected_index','status']:compare_value(prefix+':'+f,a.get(f),z.get(f),0)
                compare_value(prefix+':identity_cosine',a.get('identity_cosine'),z.get('identity_cosine'),tolerances['FaceNet_and_buffalo_cosine_absolute_tolerance'])
                aa=a.get('candidates',a.get('all_candidates',[]));zz=z.get('candidates',z.get('all_candidates',[]))
                compare_value(prefix+':candidate_count',len(aa),len(zz),0)
                for i,(x,y)in enumerate(zip(aa,zz)):
                    for j,(v,w)in enumerate(zip(x.get('bbox',[]),y.get('bbox',[]))):compare_value(prefix+f':bbox{i}.{j}',v,w,tolerances['geometry_absolute_tolerance'])
    result={'status':'pass'if not mismatches else 'failed','measurement_policy_sha256':POLICY_SHA,
            'protocol_sha256':doc['protocol_sha256'],'fixed_case_id':next(iter(doc['case_metadata'])),
            'selected_seed':101,'tolerances':tolerances,'checks':checks,'mismatches':mismatches,
            'models_run_in_this_compare':False,'mixing_new_runtime_allowed':not mismatches}
    save(job/'bridge_receipt.json',result);return job/'bridge_receipt.json'


def flat_rows(metrics,generation,availability):
    result=[]
    for r in metrics:
        k=key(r);g=generation.get(k,{});a=availability.get(k,{})
        result.append({'method':r['method'],'subject':r['subject'],'case_id':r['case_id'],
            'seed':None if r['seed']==-1 else r['seed'],'seed_not_applicable':r['method']=='performrecast',
            'group':r['group'],'generation_status':g.get('status','missing'),'input_availability':a.get('status','unknown'),
            'execution_id':g.get('execution_id'),'alias_of':g.get('alias_of'),'failure_reason':g.get('failure_reason')or a.get('error'),
            'output':g.get('output'),'au_parse_status':(r.get('AU_measurement')or{}).get('parse_status'),
            'facenet_status':(r.get('FaceNet_measurement')or{}).get('status'),
            'buffalo_status':(r.get('buffalo_measurement')or{}).get('status'),
            **{m:r.get(m)for m in METRICS}})
    return result


def write_csv(path,rows,fields):
    with Path(path).open('x',newline='',encoding='utf-8-sig')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(r.get(k),ensure_ascii=False,separators=(',',':'))if isinstance(r.get(k),(dict,list))else r.get(k)for k in fields})


def export(job):
    job,doc=job_control(job);c,e=legacy();data=[];available={key(r):r for r in doc['items']}
    for kind in ['openface','facenet','buffalo']:
        b=read(job/kind/'binding.json');m=read(job/kind/'results.json')
        need(b['protocol_sha256']==doc['protocol_sha256']and b['measurement_policy_sha256']==POLICY_SHA and b['input_sha256']==sha(job/'items.json'),'Measurement binding mismatch')
        need(b['results_sha256']==sha(job/kind/'results.json'),'Measurement file changed')
        need(len({key(r)for r in m['records']})==len(m['records'])and {key(r)for r in m['records']}==set(available),'Missing/duplicate evaluator rows')
        for r in m['records']:need(r['sha256']==available[key(r)]['sha256'],'Measured image differs')
        data.append(m)
    new_rows=c.legacy('evaluation_export').merge_measurements(doc,*data)
    old_metric_doc=read(pin('old_metrics'));old_items=read(pin('old_items'))
    old_rows=old_metric_doc['records'];need(len(old_rows)==768,'Legacy denominator changed')
    all_rows=copy.deepcopy(old_rows)+new_rows
    generations={key(r):r for r in old_items['normalized_generation_records']+doc['normalized_generation_records']}
    need(len(all_rows)==896 and len(generations)==896,'Expected768legacy+96Swift+32Perform actual logical rows')
    availability={tuple(r['key']):r for r in old_items['availability']+doc['availability']}
    per_image=flat_rows(all_rows,generations,availability)
    case_input,coverage=case_rows(all_rows,generations,doc['case_metadata'])
    case_input['metadata'].update(protocol_sha256=doc['protocol_sha256'],measurement_policy_sha256=POLICY_SHA,
        reused_old_metrics=doc['old_metrics'],new_measurements_only=True,old_seven_comparisons_unchanged=True,
        comparison_input_groups={'magicface':'source_manual_AU','faceadapter':'source_real_driver','liveportrait':'source_real_driver',
                                 'swiftedit':'source_text','performrecast':'source_real_target_driver'},
        same_training_or_input_protocol=False)
    stats=c.legacy('statistics').analyze(case_input)
    stats=c.enrich_coverage(stats,case_input,doc['case_metadata'])
    stats['analysis_count_semantics']='frozen statistics engine seed-key counts are case-mean rows here, not physical seed generations; consult case_seed_coverage and all_per_image'
    gallery=gallery_plan(case_input,doc['case_metadata'],per_image)
    out=new_directory(job/'statistics')
    save(out/'all_case_seed_metrics.json',{'status':'complete','protocol_sha256':doc['protocol_sha256'],
        'records':all_rows,'per_image_table':per_image,'legacy_metrics_sha256':PINS['old_metrics'][1],
        'legacy_measurements_recomputed':False,'availability':list(availability.values())})
    save(out/'case_metric_input.json',case_input);save(out/'case_seed_coverage.json',coverage)
    stats['input_sha256']=sha(out/'case_metric_input.json');save(out/'paired_subject_results.json',stats)
    save(out/'gallery_selection.json',gallery)
    write_csv(out/'all_per_image.csv',per_image,list(per_image[0]))
    subject_rows=[]
    for m,values in stats['raw_method_statistics'].items():
        for metric,summary in values.items():
            for subject,value in summary['subject_means'].items():
                subject_rows.append({'method':m,'metric':metric,'subject':subject,'value':value})
    write_csv(out/'subject_metrics.csv',subject_rows,['method','metric','subject','value'])
    summary=[{'method':m,'metric':metric,'n_case_means':row['n_seed_outputs'],**{k:v for k,v in row.items()if k not in ['case_means','subject_means','n_seed_outputs']}}
              for m,values in stats['raw_method_statistics'].items()for metric,row in values.items()]
    write_csv(out/'all_method_summary.csv',summary,['method','metric','mean','n_subjects','n_cases','n_case_means'])
    save(out/'export_receipt.json',{'status':'complete','protocol':doc['protocol'],'new_rows':128,'old_rows':768,
        'case_analysis_rows':320,'new_model_inference':False,'evaluation_run_separately':True,
        'statistics_source':ref(EXP/'statistics/paired_subject_statistics.py'),
        'outputs':{p.name:sha(p)for p in out.iterdir()if p.is_file()}})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['emit-protocol','preflight','prepare','openface','facenet','buffalo','export',
                                   'bridge-prepare','bridge-openface','bridge-facenet','bridge-buffalo','bridge-compare'])
    p.add_argument('--protocol',type=Path);p.add_argument('--protocol-sha');p.add_argument('--output',type=Path)
    p.add_argument('--generation-index',type=Path,action='append',default=[]);p.add_argument('--generation-sha',action='append',default=[])
    p.add_argument('--job',type=Path);p.add_argument('--approval',type=Path);p.add_argument('--runtime-receipt',type=Path)
    p.add_argument('--runtime-sha');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.phase=='emit-protocol':
        need(a.output and a.output.resolve().is_relative_to(HERE),'Protocol output must be inside new evaluation directory')
        save(a.output,specification());print(json.dumps({'status':'protocol_only_no_models','output':ref(a.output)}));return
    if a.phase in ['preflight','prepare','bridge-prepare']:
        load_protocol(a.protocol,a.protocol_sha);_,_,cases,_,_=context()
        need(len(a.generation_index)==len(a.generation_sha),'Index SHA count mismatch')
        indexes=[{'path':str(x.resolve()),'sha256':s}for x,s in zip(a.generation_index,a.generation_sha)]
        if indexes:normalize(cases,[read(verified(r))for r in indexes],a.protocol_sha)
        if a.phase=='preflight'or not a.execute:
            print(json.dumps({'status':'metadata_only','new_plan':128,'Perform_physical':32,'Swift_physical':96,'indices_checked':bool(indexes),'models_loaded':False}));return
        gate(a.approval,a.protocol_sha,a.phase)
        result=prepare_bridge(a.protocol,a.protocol_sha,a.output)if a.phase=='bridge-prepare'else prepare(a.protocol,a.protocol_sha,indexes,a.output)
    else:
        job,doc=job_control(a.job)
        if not a.execute:print(json.dumps({'status':'metadata_only','phase':a.phase,'items':len(doc['items']),'models_loaded':False}));return
        gate(a.approval,doc['protocol_sha256'],a.phase)
        if a.phase=='export':result=export(job)
        elif a.phase=='bridge-compare':result=bridge_compare(job)
        else:
            need(a.runtime_receipt and a.runtime_sha,'Explicit original/bridged runtime receipt required')
            result=measure(job,a.phase.removeprefix('bridge-'),{'path':str(a.runtime_receipt.resolve()),'sha256':a.runtime_sha})
    print(json.dumps({'status':'complete','output':str(result)},ensure_ascii=False))


if __name__=='__main__':main()
