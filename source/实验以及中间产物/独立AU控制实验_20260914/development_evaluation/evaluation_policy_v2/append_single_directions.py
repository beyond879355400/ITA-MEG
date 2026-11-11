#!/usr/bin/env python3
"""Append the two frozen single-direction variants under unchanged policy v2."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys
from openface_v2 import HERE,ROOT,TOOLS,POLICY_SHA,sha,save,measure

OUT=HERE/'single_directions'
PREVIEW=ROOT/'development_previews/v4_slot_2k_singles200_dev19_202609142041'
VARIANTS=('condition_reads_image','image_reads_condition')
def read(path):return json.loads(path.read_text())
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,str(HERE.parent))
prior=module(HERE.parent/'v4_slot_2k_mlp200_dev19/evaluate.py','trace_helpers')

def metadata():
    original=ROOT/'development_previews/v4_slot_2k_step200_dev19_202609141941'
    mlp=ROOT/'development_previews/v4_slot_2k_mlp200_dev19_202609142015'
    configs=[(original,read(original/'settings.json')['methods'][0]),(mlp,read(mlp/'settings.json')['methods'][0])]
    configs += [(PREVIEW,m) for m in read(PREVIEW/'settings.json')['methods']]
    settings=read(original/'settings.json');base=configs[0][1];checks=[]
    for preview,method in configs:
        s=read(preview/'settings.json')
        for k in ('cache_sha256','case_manifest_sha256','steps','guidance_scale','sampling_seeds','au_scale','ip_scale','public_assets','source_anchor'):
            assert s[k]==settings[k],(preview.name,k)
        scheduler=read(preview/'runtime_scheduler.json');reference=read(original/'runtime_scheduler.json')
        # Diffusers records a set-derived diagnostic list; order is not a sampler parameter.
        for obj in (scheduler,reference):obj['config']['_use_default_values']=sorted(obj['config']['_use_default_values'])
        assert scheduler==reference
        cf=method['effective_config'];ignored={'variant','checkpoint_every','validate_every'}
        assert {k:v for k,v in cf.items() if k not in ignored}=={k:v for k,v in base['effective_config'].items() if k not in ignored}
        mc=method['model_config'];ignored_mc={'direction','ff_multiplier'}
        assert {k:v for k,v in mc.items() if k not in ignored_mc}=={k:v for k,v in base['model_config'].items() if k not in ignored_mc}
        assert mc['ff_multiplier']==(3 if cf['variant'] in VARIANTS else 2)
        gs=[json.loads(x) for x in (preview/'generation_records.jsonl').read_text().splitlines()]
        g=next(g for g in gs if g['method']==method['name']);checkpoint=Path(g['checkpoint'])
        assert sha(checkpoint)==method['checkpoint_sha256'];r=read(checkpoint.parent/'result.json');seal=read(checkpoint.with_suffix('.json'))
        assert r['status']=='complete' and r['global_step']==seal['step']==200 and sha(checkpoint)==seal['checkpoint_sha256']
        b=read(checkpoint.parent/'bindings.json');trace=prior.trace_to(checkpoint)
        checks.append({'variant':cf['variant'],'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'bindings':b,'trace':trace,
                       'trainable_parameters':r['trainable_parameters'],'model_config':mc,'effective_config':cf,
                       'generation_source_sha256':sha(preview/'generation_source.py')})
    schedule_keys=('step','case_ids','timesteps','auxiliary_eligible')
    traces=[[{k:r[k] for k in schedule_keys} for r in c['trace']] for c in checks]
    assert len(traces[0])==200 and all(t==traces[0] for t in traces)
    for item in checks[1:]:
        for k in ('cache','cache_manifest','split','pairs','public_assets','au_supervision','train_case_ids','all_dev_case_ids','dev_case_ids','sources'):
            assert item['bindings'][k]==checks[0]['bindings'][k],(item['variant'],k)
    return {'status':'pass','test_used':False,'policy_sha256':POLICY_SHA,'train_schedule_all200_equal':True,
            'schedule_fields':schedule_keys,'datasets_losses_centering_readout_seeds_equal':True,'sampling_settings_equal':True,
            'intentional_differences':'variant; single-direction FF multiplier3 vs bidirectional2 for parameter matching; checkpoint/validation cadence logged',
            'scheduler_normalization':'Only set-derived diagnostic _use_default_values sorted; all actual sampler parameters compared unchanged',
            'methods':[{k:v for k,v in c.items() if k not in ('trace','bindings')} for c in checks]}

def prepare():
    assert sha(HERE/'policy.json')==POLICY_SHA
    audit=metadata();OUT.mkdir(exist_ok=False);save(OUT/'metadata_audit.json',audit)
    initial=read(HERE/'dev19_items.json')['items'];lookup={(r['case_id'],r['method']):r for r in initial}
    source={r['case_id']:r for r in initial if r['method']=='source_real'}
    settings=read(PREVIEW/'settings.json');variant={m['name']:m['effective_config']['variant'] for m in settings['methods']}
    assert set(variant.values())==set(VARIANTS)
    result=read(PREVIEW/'result.json');assert result['status']=='complete' and result['images']==38 and not result['test_data_used']
    assert sha(PREVIEW/'frozen_cases.json')==settings['case_manifest_sha256']==read(ROOT/'development_previews/v4_slot_2k_step200_dev19_202609141941/settings.json')['case_manifest_sha256']
    generations=[json.loads(line) for line in (PREVIEW/'generation_records.jsonl').read_text().splitlines()];items=[];seen=set();noedit=[]
    original=ROOT/'development_previews/v4_slot_2k_step200_dev19_202609141941'
    prompts={g['case_id']:g['prompt'] for g in [json.loads(l) for l in (original/'generation_records.jsonl').read_text().splitlines()] if g['method']=='baseline'}
    for g in generations:
        cid=g['case_id'];v=variant[g['method']];s=source[cid];key=(cid,v)
        assert key not in seen and g['status']=='complete' and g['split']=='dev' and g['seed']==101;seen.add(key)
        assert g['subject']==s['subject'] and g['source_aus_0_5']==s['source_labels_12'] and g['target_aus_0_5']==s['target_labels_12'] and g['prompt']==prompts[cid]
        assert sha(g['output'])==g['sha256'] and Path(g['output']).resolve().is_relative_to(PREVIEW)
        for role in ('source','target'):assert sha(PREVIEW/cid/(role+'_real.png'))==lookup[(cid,role+'_real')]['sha256']
        item={**s,'method':v,'original_method':g['method'],'path':g['output'],'sha256':g['sha256']};items.append(item)
        if s['group']=='no_edit':noedit.append({'case_id':cid,'method':v,'PNG_SHA_equal_baseline':g['sha256']==lookup[(cid,'baseline')]['sha256']})
    assert len(items)==38 and all((cid,v) in seen for cid in source for v in VARIANTS)
    assert len(noedit)==8 and all(r['PNG_SHA_equal_baseline'] for r in noedit)
    document={'status':'frozen_appendix','policy_sha256':POLICY_SHA,'test_used':False,'items':list(source.values())+items,
              'generated_count':38,'real_source_count':19,'source_initial_manifest_sha256':sha(HERE/'dev19_items.json'),
              'generation_manifest_sha256':sha(PREVIEW/'generation_records.jsonl'),'case_manifest_sha256':settings['case_manifest_sha256'],
              'metadata_audit_sha256':sha(OUT/'metadata_audit.json'),'script_sha256':sha(__file__),
              'input_coordinate_note_sha256':sha(HERE/'INPUT_COORDINATE_NOTE.md'),'no_edit_PNG_checks':noedit}
    save(OUT/'items.json',document);save(OUT/'items_seal.json',{'path':str(OUT/'items.json'),'sha256':sha(OUT/'items.json'),'policy_sha256':POLICY_SHA})
    print(json.dumps({'metadata':'pass','items':57,'noedit_equal':8,'manifest_sha256':sha(OUT/'items.json')},ensure_ascii=False),flush=True)

def evaluate():
    assert sha(HERE/'policy.json')==POLICY_SHA
    doc=read(OUT/'items.json');assert sha(OUT/'items.json')==read(OUT/'items_seal.json')['sha256'] and doc['policy_sha256']==POLICY_SHA
    for dependency in ('real_dev_openface/summary.json','facenet_real12/summary.json','comparison/results.json'):
        r=read(HERE/dependency);assert r['status']=='complete' and r['policy_sha256']==POLICY_SHA
    items=[r for r in doc['items'] if r['method'] in VARIANTS];assert len(items)==38 and all(r['split']=='dev' and sha(r['path'])==r['sha256'] for r in items)
    policy=read(HERE/'policy.json');seal=read(TOOLS/'tool_seal.json');assert sha(TOOLS/'tool_seal.json')==policy['openface']['tool_seal_sha256']
    for asset in [seal['binary'],seal['launcher'],*seal['model_files']]:assert sha(TOOLS/asset['path'])==asset['sha256']
    output=OUT/'openface';output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'policy_sha256':POLICY_SHA,'input_manifest_sha256':sha(OUT/'items.json'),'script_sha256':sha(__file__),
                               'openface_helper_sha256':sha(HERE/'openface_v2.py'),'items':items,'test_used':False,'workers':2})
    (output/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    records=[]
    with (output/'per_image.jsonl').open('x',buffering=1) as log,ThreadPoolExecutor(max_workers=2) as pool:
        for task in as_completed([pool.submit(measure,item,output) for item in items]):
            r=task.result();records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            print(json.dumps({'done':len(records),'valid':sum(r['vector'] is not None for r in records)}),flush=True)
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'planned':38,'valid':sum(r['vector'] is not None for r in records),'records':sorted(records,key=lambda r:(r['case_id'],r['method']))}
    save(output/'results.json',result);save(output/'summary.json',{k:v for k,v in result.items() if k!='records'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','evaluate'));a=p.parse_args()
    prepare() if a.mode=='prepare' else evaluate()
