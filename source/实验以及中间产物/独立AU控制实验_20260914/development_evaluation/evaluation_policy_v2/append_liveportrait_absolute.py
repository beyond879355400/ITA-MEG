#!/usr/bin/env python3
"""Evaluate new native external outputs; real-driver input group stays explicit."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
from openface_v2 import HERE,ROOT,TOOLS,POLICY_SHA,AUS12,sha,save,measure
OUT=HERE/'external_liveportrait_absolute'
RUNS=ROOT/'external_baselines/runtime/runs'
GROUPS={'liveportrait_absolute':'liveportrait_dev19_raw_absolute_202609142146'}
CONTRACT=ROOT/'external_baselines/contracts/raw_dev19_v3_202609142040/image_driver_dev19.json'
def read(p):return json.loads(p.read_text())

def prepare():
    assert sha(HERE/'policy.json')==POLICY_SHA
    initial=read(HERE/'dev19_items.json')['items'];sources={r['case_id']:r for r in initial if r['method']=='source_real'};targets={r['case_id']:r for r in initial if r['method']=='target_real'}
    cachepath=ROOT/'training_cache/train2k_batched_202609141853/manifest.json';cache=read(cachepath);pairs={r['case_id']:r for r in cache['pairs'] if r['split']=='dev'}
    contract=read(CONTRACT);assert not contract['test_used'] and contract['input_group']=='source_plus_real_driver' and len(contract['cases'])==19
    assert sha(cachepath)==contract['cache_manifest_sha256'];contractrows={r['request_case_id']:r for r in contract['cases']};assert set(contractrows)==set(sources)
    items=[];provenance={};seen=set()
    from PIL import Image
    for method,name in GROUPS.items():
        run=RUNS/name;summary=read(run/'summary.json');config=read(run/'configuration.json')
        assert summary['case_count']==summary['success_count']==19 and summary['failed_count']==0 and len(summary['results'])==19
        assert config['manifest_sha256']==sha(CONTRACT)
        oldrun=RUNS/'liveportrait_dev19_raw_native_202609142041';oldconfig=read(oldrun/'configuration.json');binding=read(run/'single_factor_binding.json')
        assert binding['status']=='PASS' and binding['old_configuration_sha256']==sha(oldrun/'configuration.json') and binding['all_cases_same_settings']
        assert binding['only_parameter_change']=={'flag_relative_motion':{'old':True,'new':False}}
        oldcopy=json.loads(json.dumps(oldconfig));newcopy=json.loads(json.dumps(config))
        assert oldcopy['policy']['effective_author_cli_parameters'].pop('flag_relative_motion') is True
        assert newcopy['policy']['effective_author_cli_parameters'].pop('flag_relative_motion') is False
        assert oldcopy==newcopy,'Any change beyond relative_motion is prohibited'
        oldinputs={r['input']['request_case_id']:r['input'] for r in read(oldrun/'summary.json')['results']}
        assert oldinputs=={r['input']['request_case_id']:r['input'] for r in summary['results']}
        configured={r['request_case_id']:r for r in config['cases']};assert configured==contractrows
        outputs=[]
        for r in summary['results']:
            native=r['input'];cid=native['request_case_id'];s=sources[cid];t=targets[cid];pair=pairs[cid]
            assert (cid,method) not in seen and r['status']=='success';seen.add((cid,method))
            assert native==configured[cid] and native['split']=='dev' and native['seed']==101 and native['subject']==s['subject']
            assert native['source_au_vector_0_5']==s['source_labels_12'] and native['target_au_vector_0_5']==s['target_labels_12'] and native['au_order']==AUS12
            assert native['target_role']=='same_original_raw_observed_target_frame; never_generated'
            for side,ref in [('source',s),('target',t)]:
                aligned=pair['aligned_'+side];assert sha(native[side])==native[side+'_sha256']==aligned['raw_sha256']
                assert Path(native[side]).resolve()==Path(aligned['raw_path']).resolve() and ref['sha256']==aligned['aligned_sha256']==sha(ref['path'])
            image=Path(r['output']).resolve();assert image.is_relative_to(run.resolve()) and sha(image)==r['output_sha256'] and image.name=='generated.png'
            with Image.open(image) as im:im.load();size=list(im.size);mode=im.mode
            assert mode=='RGB' and size==[1024,768]
            item={**s,'method':method,'original_method':'liveportrait','path':str(image),'sha256':r['output_sha256'],'native_output_size':size,
                  'input_group':'source_plus_real_driver','manual_au_native_condition':False,'extra_driver_available':True,'inference_input':native,
                  'source_reference_space':'common canonical512 of exact same original source frame','target_reference_space':'common canonical512 of exact same original target frame',
                  'evaluation_input':'original native generated RGB; no extra alignment, resize, padding or beautification before evaluator'}
            items.append(item);outputs.append({'case_id':cid,'output_sha256':r['output_sha256'],'size':size})
        provenance[method]={'run':str(run),'summary_sha256':sha(run/'summary.json'),'configuration_sha256':sha(run/'configuration.json'),
                            'contract_sha256':sha(CONTRACT),'single_factor_binding_sha256':sha(run/'single_factor_binding.json'),'independent_only_relative_motion_diff':True,'upstream_commit':config['upstream_commit'],'native_generation_policy':config['policy'],'outputs':outputs}
    assert len(items)==19 and all((cid,m) in seen for cid in sources for m in GROUPS)
    OUT.mkdir(exist_ok=False)
    doc={'status':'frozen_appendix','policy_sha256':POLICY_SHA,'test_used':False,'items':list(sources.values())+items,'generated_count':19,'real_source_count':19,
         'generation_input_group':'source_plus_real_driver','same_manual_AU_input_ranking':False,'case_manifest_sha256':contract['request_manifest_sha256'],
         'common_reference_manifest_sha256':sha(HERE/'dev19_items.json'),'cache_manifest_sha256':sha(cachepath),'provenance':provenance,'script_sha256':sha(__file__),
         'coordinate_note_sha256':sha(HERE/'INPUT_COORDINATE_NOTE.md')}
    save(OUT/'items.json',doc);save(OUT/'items_seal.json',{'path':str(OUT/'items.json'),'sha256':sha(OUT/'items.json'),'policy_sha256':POLICY_SHA})
    print(json.dumps({'status':'ready','count':38,'generated':19,'manifest_sha256':sha(OUT/'items.json')},ensure_ascii=False),flush=True)

def evaluate():
    assert sha(HERE/'policy.json')==POLICY_SHA
    doc=read(OUT/'items.json');assert sha(OUT/'items.json')==read(OUT/'items_seal.json')['sha256'] and not doc['test_used']
    for name in ('real_dev_openface/summary.json','facenet_real12/summary.json'):
        r=read(HERE/name);assert r['status']=='complete' and r['policy_sha256']==POLICY_SHA
    items=[r for r in doc['items'] if r['method'] in GROUPS];assert len(items)==19 and all(r['split']=='dev' and sha(r['path'])==r['sha256'] for r in items)
    policy=read(HERE/'policy.json');seal=read(TOOLS/'tool_seal.json');assert sha(TOOLS/'tool_seal.json')==policy['openface']['tool_seal_sha256']
    for asset in [seal['binary'],seal['launcher'],*seal['model_files']]:assert sha(TOOLS/asset['path'])==asset['sha256']
    out=OUT/'openface';out.mkdir(exist_ok=False);save(out/'protocol.json',{'policy_sha256':POLICY_SHA,'input_manifest_sha256':sha(OUT/'items.json'),'items':items,'test_used':False,
        'script_sha256':sha(__file__),'openface_helper_sha256':sha(HERE/'openface_v2.py'),'tool_seal_sha256':sha(TOOLS/'tool_seal.json'),'workers':2})
    (out/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes());os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    records=[]
    with (out/'per_image.jsonl').open('x',buffering=1) as log,ThreadPoolExecutor(max_workers=2) as workers:
        for task in as_completed([workers.submit(measure,item,out) for item in items]):
            r=task.result();records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps({'done':len(records),'valid':sum(r['vector'] is not None for r in records)}),flush=True)
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'records':sorted(records,key=lambda r:(r['case_id'],r['method'])),'planned':19,'valid':sum(r['vector'] is not None for r in records)}
    save(out/'results.json',result);save(out/'summary.json',{k:v for k,v in result.items() if k!='records'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','evaluate'));a=p.parse_args();prepare() if a.mode=='prepare' else evaluate()
