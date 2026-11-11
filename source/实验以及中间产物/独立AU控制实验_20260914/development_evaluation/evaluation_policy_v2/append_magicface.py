#!/usr/bin/env python3
"""Await all19 fresh native MagicFace outputs, then apply fixed CPU policy v2."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
from openface_v2 import HERE,ROOT,TOOLS,POLICY_SHA,AUS12,sha,save,measure
OUT=HERE/'external_magicface'
RUN=ROOT/'external_baselines/runs/magicface_dev19_raw_native_202609142101'
CONTRACT=ROOT/'external_baselines/contracts/raw_dev19_v3_202609142040/magicface_dev19.json'
def read(p):return json.loads(p.read_text())

def prepare():
    assert sha(HERE/'policy.json')==POLICY_SHA
    summary=read(RUN/'generate_summary.json')
    if summary['processed_count']!=19:raise RuntimeError('Full19 output not complete; do not substitute smoke outputs')
    assert summary['success_count']==summary['case_count']==19 and summary['failed_count']==0 and not summary['test_used']
    config=read(RUN/'configuration.json');contract=read(CONTRACT);assert config['manifest_sha256']==sha(CONTRACT) and config['cases']==contract['cases']
    assert not config['old_single_case_outputs_reused'] and not contract['test_used'] and config['seeds']==[101]
    assert config['runner_sha256']==sha(RUN/'runner_snapshot.py') and contract['input_group']=='source_plus_manual_au'
    initial=read(HERE/'dev19_items.json')['items'];sources={r['case_id']:r for r in initial if r['method']=='source_real'}
    cachepath=ROOT/'training_cache/train2k_batched_202609141853/manifest.json';cache=read(cachepath);assert sha(cachepath)==contract['cache_manifest_sha256']
    pairs={r['case_id']:r for r in cache['pairs'] if r['split']=='dev'};items=[];seen=set();provenance=[]
    from PIL import Image
    for native in config['cases']:
        cid=native['request_case_id'];s=sources[cid];pair=pairs[cid];case_dir=RUN/'cases'/native['case_id'];identity=read(case_dir/'case.json');g=read(case_dir/'generation.json')
        assert cid not in seen and identity['case']==native and identity['manifest_sha256']==sha(CONTRACT);seen.add(cid)
        assert native['split']=='dev' and native['seed']==101 and native['subject']==s['subject'] and native['au_order']==AUS12
        assert native['source_au_vector_0_5']==s['source_labels_12'] and native['target_au_vector_0_5']==s['target_labels_12']
        assert 'target' not in native and 'target_image_path' not in native and not identity['target_image_read'] and not g['target_image_read']
        assert sha(native['source'])==native['source_sha256']==pair['aligned_source']['raw_sha256']
        assert sha(s['path'])==s['sha256']==pair['aligned_source']['aligned_sha256']
        assert g['status']=='complete' and not g['test_used'] and g['case_id']==native['case_id'] and g['request_case_id']==cid and g['subject']==s['subject']
        assert g['seed']==101 and g['inference_steps']==50 and g['guidance_scale']==7.5 and g['image_guidance_scale']==1.5 and g['scheduler']=='UniPCMultistepScheduler'
        assert g['source_raw_sha256']==native['source_sha256'] and g['source_aus']==s['source_labels_12'] and g['target_aus']==s['target_labels_12']
        assert g['au_variation_sent']==native['au_delta_native']==[t-a for a,t in zip(s['source_labels_12'],s['target_labels_12'])]
        assert g['runner_sha256']==config['runner_sha256'] and g['model_layout_seal_sha256']==config['layout_seal_sha256']
        for loading in ['id_loading','denoising_loading']:assert all(not v for v in g[loading].values())
        image=case_dir/'generated.png';assert Path(g['output']).resolve()==image.resolve() and sha(image)==g['output_sha256']
        with Image.open(image) as im:im.load();size=list(im.size);assert im.mode=='RGB' and size==[512,512]
        items.append({**s,'method':'magicface','original_method':'magicface','path':str(image),'sha256':g['output_sha256'],'native_output_size':size,
           'input_group':'source_plus_manual_au_native','manual_au_native_condition':True,'extra_driver_available':False,'inference_input':native,
           'evaluation_input':'original native generated512 RGB; no extra alignment, resize, padding or enhancement before evaluator'})
        provenance.append({'case_id':cid,'case_json_sha256':sha(case_dir/'case.json'),'generation_json_sha256':sha(case_dir/'generation.json'),
                           'repo_commit':g['repo_commit'],'native_pipeline_sha256':g['native_pipeline_sha256'],'source_raw_sha256':g['source_raw_sha256'],
                           'source_crop_sha256':g['source_crop_sha256'],'background_sha256':g['background_sha256'],'target_image_read':False})
    assert len(items)==19 and seen==set(sources)
    OUT.mkdir(exist_ok=False);doc={'status':'frozen_appendix','policy_sha256':POLICY_SHA,'test_used':False,'items':list(sources.values())+items,'generated_count':19,'real_source_count':19,
       'summary_sha256':sha(RUN/'generate_summary.json'),'configuration_sha256':sha(RUN/'configuration.json'),'contract_sha256':sha(CONTRACT),
       'common_reference_manifest_sha256':sha(HERE/'dev19_items.json'),'cache_manifest_sha256':sha(cachepath),'provenance':provenance,
       'coordinate_note_sha256':sha(HERE/'INPUT_COORDINATE_NOTE.md'),'script_sha256':sha(__file__),'native_generation_config':config}
    save(OUT/'items.json',doc);save(OUT/'items_seal.json',{'path':str(OUT/'items.json'),'sha256':sha(OUT/'items.json'),'policy_sha256':POLICY_SHA})
    print(json.dumps({'status':'ready','records':38,'manifest_sha256':sha(OUT/'items.json')},ensure_ascii=False),flush=True)

def evaluate():
    assert sha(HERE/'policy.json')==POLICY_SHA
    document=read(OUT/'items.json');assert sha(OUT/'items.json')==read(OUT/'items_seal.json')['sha256'] and not document['test_used']
    for name in ['real_dev_openface/summary.json','facenet_real12/summary.json']:
        ready=read(HERE/name);assert ready['status']=='complete' and ready['policy_sha256']==POLICY_SHA
    items=[r for r in document['items'] if r['method']=='magicface'];assert len(items)==19 and all(r['split']=='dev' and sha(r['path'])==r['sha256'] for r in items)
    policy=read(HERE/'policy.json');seal=read(TOOLS/'tool_seal.json');assert sha(TOOLS/'tool_seal.json')==policy['openface']['tool_seal_sha256']
    for asset in [seal['binary'],seal['launcher'],*seal['model_files']]:assert sha(TOOLS/asset['path'])==asset['sha256']
    out=OUT/'openface';out.mkdir(exist_ok=False);save(out/'protocol.json',{'policy_sha256':POLICY_SHA,'input_manifest_sha256':sha(OUT/'items.json'),'items':items,'test_used':False,
        'script_sha256':sha(__file__),'openface_helper_sha256':sha(HERE/'openface_v2.py'),'tool_seal_sha256':sha(TOOLS/'tool_seal.json'),'workers':2})
    (out/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes());os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    records=[]
    with (out/'per_image.jsonl').open('x',buffering=1) as log,ThreadPoolExecutor(max_workers=2) as workers:
        for task in as_completed([workers.submit(measure,item,out) for item in items]):
            r=task.result();records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps({'done':len(records),'valid':sum(r['vector'] is not None for r in records)}),flush=True)
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'records':sorted(records,key=lambda r:r['case_id']),'planned':19,'valid':sum(r['vector'] is not None for r in records)}
    save(out/'results.json',result);save(out/'summary.json',{k:v for k,v in result.items() if k!='records'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','evaluate'));a=p.parse_args();prepare() if a.mode=='prepare' else evaluate()
