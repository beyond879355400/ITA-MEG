"""Export frozen metrics to seed->case->subject statistics, with all denominators."""
import argparse
import copy
import json
from pathlib import Path
import sys
from evaluation_contract import *
from evaluation_measurements import job_control

STATISTICS_SHA='f0f1baaf08464ac2e46bfe607ae7ea34ae1c49f200288b57b45e32df2a44d4ed'

def merge_measurements(document,au,face,buffalo):
    sys.path.insert(0,str(V2))
    from summarize_v2 import e,extra_errors,EXTRA
    # Three logical image identities include seed: never collapse stochastic draws.
    def mapping(rows):
        result={(r['case_id'],r['method'],r['seed']):r for r in rows};require(len(result)==len(rows),'Duplicate evaluator logical record');return result
    am=mapping(au['records']);fm=mapping(face['records']);bm=mapping(buffalo['records']);out=[]
    for planned in document['planned']:
        key=(planned['case_id'],planned['method'],planned['seed']);ar=am.get(key);fr=fm.get(key);br=bm.get(key);meta=document['case_metadata'][planned['case_id']]
        sr=am.get((planned['case_id'],'source_real',0));tr=am.get((planned['case_id'],'target_real',0))
        row={**copy.deepcopy(meta),**planned,'AU_measurement':ar,'FaceNet_measurement':fr,'buffalo_measurement':br}
        values=None if ar is None else ar['vector'];source_values=None if sr is None else sr['vector']
        row.update(e.au_metrics(values,source_values,meta['source_labels_12'],meta['target_labels_12']))
        row['identity_cosine']=fr.get('identity_cosine') if fr else None;row['buffalo_identity_cosine']=br.get('identity_cosine') if br else None
        row['extra5'],row['extra5_proxy_mae_to_real_target']=extra_errors(ar.get('selected_au17') if ar else None,tr.get('selected_au17') if tr else None,sr.get('selected_au17') if sr else None)
        for a in EXTRA:row[f'extra_AU{a}_error_to_real_target']=row['extra5'][str(a)]['absolute_error_to_target']
        out.append(row)
    return out

def export(job):
    job,doc,control=job_control(job,'final_evaluation_export',__file__)
    au=read(job/'openface/results.json');face=read(job/'facenet/results.json');buff=read(job/'buffalo/results.json')
    require(au['status']=='complete' and au['protocol_sha256']==control['protocol_sha256'] and au['measurement_policy_sha256']==POLICY_SHA,'AU evaluation incomplete/wrong protocol')
    for kind,data in [('facenet',face),('buffalo',buff)]:
        binding=read(job/kind/'final_binding.json');require(binding['protocol_sha256']==control['protocol_sha256'] and binding['measurement_policy_sha256']==POLICY_SHA and data.get('policy_sha256')==POLICY_SHA,'Identity measurement binding mismatch')
    available={(r['case_id'],r['method'],r['seed']):r for r in doc['items']}
    for data in (au,face,buff):
        keys=[]
        for r in data['records']:
            key=(r['case_id'],r['method'],r['seed']);keys.append(key);require(key in available and r['sha256']==available[key]['sha256'],'Evaluator image key/SHA changed')
        require(len(keys)==len(set(keys)) and set(keys)==set(available),'Every available input must keep an evaluator row, including invalid detections')
    rows=merge_measurements(doc,au,face,buff);normalized={row_key(r):r for r in doc['normalized_generation_records']}
    counts={'all':len(doc['case_metadata']),'edited':sum(r['group']=='edited' for r in doc['case_metadata'].values()),'no_edit':sum(r['group']=='no_edit' for r in doc['case_metadata'].values())}
    metadata={'protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,'evaluation_policy_sha256':control['evaluation_sha256'],
      'raw_manifest_sha256':control['raw_manifest_sha256'],'prepared_input_sha256':sha(job/'items.json'),'case_scope_counts':counts,
      'planned_applicable_outputs_per_method_metric':{name:counts[definition['scope']]*len(control['protocol']['sampling_seeds']) for name,definition in control['evaluation']['metrics'].items()},
      'scope_NA_is_not_generation_failure':True,'input_groups':{m:control['sampling']['method_settings'][m].get('input_group') for m in frozen_header(control['protocol'])},
      'source_targets_remain_reference_only':True,'script_sha256':sha(__file__)}
    document=statistics_document(control['protocol'],control['evaluation'],doc['planned'],normalized,rows,doc['execution_accounting'],metadata)
    out=job/'statistics';out.mkdir(exist_ok=False);save(out/'paired_subject_input.json',document)
    save(out/'all_case_seed_metrics.json',{'status':'complete','protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,'records':rows,
           'availability':doc['availability'],'case_scope_counts':counts,'execution_accounting':doc['execution_accounting']})
    statistics_path=EXP/'statistics/paired_subject_statistics.py';require(sha(statistics_path)==STATISTICS_SHA,'Statistics algorithm changed')
    save(out/'statistics_command.json',{'input':str(out/'paired_subject_input.json'),'input_sha256':sha(out/'paired_subject_input.json'),'statistics_script':str(statistics_path),
      'statistics_sha256':STATISTICS_SHA,'output_suggested':str(out/'paired_subject_results.json'),'execution_performed':False,
      'command_arguments':['--input',str(out/'paired_subject_input.json'),'--output',str(out/'paired_subject_results.json')]})
    (out/'exporter_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return out/'paired_subject_input.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();print(export(a.job))
