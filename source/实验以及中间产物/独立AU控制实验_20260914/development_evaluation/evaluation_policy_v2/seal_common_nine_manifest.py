#!/usr/bin/env python3
"""All19×9 real outputs and input-information groups, no pending placeholders."""
import json
from pathlib import Path
from openface_v2 import HERE,ROOT,POLICY_SHA,sha,save

def main():
    folder=HERE/'comparison_nine';result=json.loads((folder/'results.json').read_text());assert result['status']=='complete' and result['policy_sha256']==POLICY_SHA and not result['test_used']
    methods=result['methods'];assert len(methods)==9 and set(methods)=={'baseline','current_model','condition_reads_image','image_reads_condition','mlp_model','magicface','liveportrait','faceadapter','liveportrait_absolute'}
    by={(r['case_id'],r['method']):r for r in result['records']};request=ROOT/'development_evaluation/dev_coverage_audit/dev19_manifest_2k.json'
    assert sha(request)=='90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28';requests=json.loads(request.read_text());assert not requests['test_data_used']
    cachepath=ROOT/'training_cache/train2k_batched_202609141853/manifest.json';cache=json.loads(cachepath.read_text());pairs={r['case_id']:r for r in cache['pairs'] if r['split']=='dev'}
    group={m:g for g,ms in result['input_groups'].items() for m in ms};cases=[]
    from PIL import Image
    for req in requests['cases']:
        cid=req['case_id'];pair=pairs[cid];s=by[(cid,'source_real')];t=by[(cid,'target_real')]
        assert s['source_labels_12']==req['source_au_vector_0_5'] and s['target_labels_12']==req['target_au_vector_0_5'] and s['subject']==req['subject']
        refs={}
        for role,record in [('source',s),('target',t)]:
            aligned=pair['aligned_'+role];assert record['sha256']==sha(record['path'])==aligned['aligned_sha256'] and sha(aligned['raw_path'])==aligned['raw_sha256']
            refs[role]={'evaluation_path':record['path'],'evaluation_sha256':record['sha256'],'raw_frame_path':aligned['raw_path'],'raw_frame_sha256':aligned['raw_sha256'],
                        'manual_AU_0_5':req[role+'_au_vector_0_5'],'label_frame':aligned['input']['label_frame'],'camera':aligned['input']['camera']}
        outputs=[]
        for m in methods:
            r=by[(cid,m)];assert r['split']=='dev' and r['subject']==req['subject'] and sha(r['path'])==r['sha256']
            with Image.open(r['path']) as im:im.load();size=list(im.size);mode=im.mode
            outputs.append({'method':m,'original_method':r['original_method'],'input_group':group[m],'extra_real_target_driver':m in ['liveportrait','faceadapter','liveportrait_absolute'],
               'output_path':r['path'],'output_sha256':r['sha256'],'native_size':size,'mode':mode,'request_seed_key':101,
               'seed_role':'pairing key only; deterministic author inference' if m.startswith('liveportrait') else 'sampler seed',
               'original_inference_input':r.get('inference_input'),'AU_measurement_valid':r['vector'] is not None,'FaceNet_measurement_valid':r['identity_cosine'] is not None,
               'buffalo_auxiliary_valid':r['buffalo_identity_cosine'] is not None})
        cases.append({'case_id':cid,'subject':req['subject'],'split':'dev','group':req['group'],'reference_frames':refs,'outputs':outputs})
    assert len(cases)==19 and sum(len(c['outputs']) for c in cases)==171
    doc={'status':'frozen_complete','policy_sha256':POLICY_SHA,'test_used':False,'case_count':19,'generated_count':171,'setting_count':9,'LP_is_one_method_two_settings':True,'methods':methods,'input_groups':result['input_groups'],
         'same_training_protocol_sota_ranking':False,'native_output_evaluation':'no additional alignment/resizing/enhancement before native evaluator preprocessing',
         'request_manifest_path':str(request),'request_manifest_sha256':sha(request),'cache_manifest_sha256':sha(cachepath),'evaluation_results_sha256':sha(folder/'results.json'),
         'cases':cases,'script_sha256':sha(__file__)}
    save(folder/'case_output_manifest.json',doc);save(folder/'case_output_manifest_seal.json',{'path':str(folder/'case_output_manifest.json'),'sha256':sha(folder/'case_output_manifest.json'),'policy_sha256':POLICY_SHA})
    print(json.dumps({'status':'complete','cases':19,'generated':171,'sha256':sha(folder/'case_output_manifest.json')},ensure_ascii=False))
if __name__=='__main__':main()
