"""Bounded generation-only quality probe on already exposed DEV cases.

Reuses the frozen weight/source interfaces; modified inference is separately named.
No real target pixels enter generation. No existing output or source is overwritten.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
from datetime import datetime
W=Path(__file__).resolve().parent;EXP=W.parents[1];GEN=EXP/'completion_20260915/generation'
sys.dont_write_bytecode=True
sys.path[:0]=[str(GEN),str(EXP),str(EXP/'completion_20260915/delivery')]
import common,run_dev_completion as dev,run_grid
CONFIGS=[
 {'id':'v11_reference','guidance':7.5,'steps':20},
 {'id':'cfg5_steps20','guidance':5.0,'steps':20},
 {'id':'cfg3p5_steps20','guidance':3.5,'steps':20},
 {'id':'cfg5_steps40','guidance':5.0,'steps':40},
]
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--configs',nargs='+');a=ap.parse_args()
    started=time.monotonic();inputs=dev.plan()
    selected=[c for c in inputs['cases'] if c['case_id'] in inputs['old_cases'] and any(c['target_au_vector_0_5'])]
    zero=next(c for c in inputs['cases'] if not any(c['target_au_vector_0_5']))
    cases=selected+[zero];assert len(selected)==4 and len({c['subject'] for c in selected})==4
    configs=CONFIGS if a.configs is None else [c for c in CONFIGS if c['id'] in a.configs]
    assert configs and (a.configs is None or len(configs)==len(set(a.configs)))
    plan={'kind':'dev_quality_guidance_probe','status':'planned','baseline_thesis_commit':'75b8b72',
      'case_selection':'All four edited requests in existing selected V11 DEV8, plus first zero-edit control; no target examples chosen by new results',
      'cases':[{'case_id':c['case_id'],'subject':c['subject'],'source_aus_0_5':c['source_au_vector_0_5'],'target_aus_0_5':c['target_au_vector_0_5']} for c in cases],
      'configs':configs,'seed':101,'planned':len(cases)*len(configs),'training_performed':False,
      'source_strength':.75,'au_scale':1.5,'ip_scale':.9,'cleanup_sigma':.5,'negative_prompt':'',
      'test_cases_used_for_configuration_selection':False,'all_attempts_retained':True,'inputs':inputs['refs'],
      'runner':common.ref(__file__)}
    print(json.dumps({'stage':'preflight','cases':plan['cases'],'configs':configs,'planned':plan['planned'],'seconds':time.monotonic()-started},ensure_ascii=False),flush=True)
    if not a.execute:return
    assert a.output and a.output.resolve().parent==W/'runs' and not a.output.exists(),'Explicit new W/runs output required'
    from reproduce_driver import exclusive_lock,active_generators
    assert not active_generators(),'Another native generator is active'
    with exclusive_lock():
      assert not active_generators(),'Another native generator started'
      output=a.output.resolve();output.mkdir(parents=True)
      import sampler
      plan['sampler']=common.ref(sampler.__file__);plan['started_local']=datetime.now().astimezone().isoformat()
      save(output/'configuration.json',plan)
      torch=common.environment('mps')
      masks,mask_records=dev.prepare_masks(inputs,output)
      features,cache,feature_receipts=dev.source_only_features(inputs,torch)
      save(output/'cases.json',{'kind':'quality_dev_cases','pool':'dev19','cases':cases,'sources':inputs['sources'],'target_pixels_supplied':False})
      save(output/'feature_receipts.json',feature_receipts);save(output/'mask_receipts.json',mask_records)
      print(json.dumps({'stage':'loading_models','seconds':time.monotonic()-started}),flush=True)
      runtime=None;records=[]
      try:
        runtime=run_grid.Runtime(features,cache,output,torch);sources={s['source_id']:s for s in cache['sources']}
        print(json.dumps({'stage':'models_ready','seconds':time.monotonic()-started}),flush=True)
        (output/'images').mkdir()
        with (output/'generation_records.jsonl').open('x',buffering=1) as log:
          for case in cases:
            prepared=runtime.prepare('bidirectional',case,sources[case['source_id']],masks[case['case_id']])
            for cfg in configs:
              row={'case_id':case['case_id'],'subject':case['subject'],'split':'dev','pool':'dev19',
                'method':'bidirectional_'+cfg['id'],'config_id':cfg['id'],'variant':'bidirectional','seed':101,
                'source_aus_0_5':case['source_au_vector_0_5'],'target_aus_0_5':case['target_au_vector_0_5'],
                'source_real':sources[case['source_id']]['canonical_source'],'configuration':common.ref(output/'configuration.json'),
                'output':None,'sha256':None,'status':'started'}
              t=time.monotonic();prepared['quality_trace_dir']=output/'traces'/case['case_id']/cfg['id']/'seed101'
              try:
                image,receipt=sampler.sample(runtime,prepared,101,guidance=cfg['guidance'],steps=cfg['steps'])
                assert image.size==(512,512) and image.mode=='RGB'
                path=output/'images'/(case['case_id']+'_'+cfg['id']+'_seed101.png');image.save(path)
                row.update(status='complete',output=str(path),sha256=common.sha(path),receipt=receipt)
                if cfg['id']=='v11_reference':
                  expected=inputs['expected'][case['case_id'],'bidirectional']
                  row['reference_parity']={'png_sha_equal':row['sha256']==expected['sha256'],'expected_output':expected['output'],
                    'conditions_equal':receipt['condition_hashes']==expected['condition_hashes'],
                    'source_equal':receipt['source_latent_sha256']==expected['source_latent_sha256'],
                    'noise_equal':receipt['noise_sha256']==expected['noise_sha256'],
                    'trajectory_equal':[{k:r[k] for k in ('t','next_t','latent_sha256')} for r in receipt['projection_trace']]==expected['projection_trace']}
                  if not all(row['reference_parity'][k] for k in ('png_sha_equal','conditions_equal','source_equal','noise_equal','trajectory_equal')):
                    raise RuntimeError('Reference parity failed; retain output as diagnostic and stop rather than tuning on changed runtime')
                if not any(case['target_au_vector_0_5']):
                  from PIL import Image
                  with Image.open(prepared['reconstruction']['path']) as ref:
                    assert image.tobytes()==ref.convert('RGB').tobytes(),'Zero edit did not preserve source VAE reconstruction'
                image.close()
              except Exception as e:
                row.update(status='failed',error=str(e),traceback=traceback.format_exc())
                row['seconds']=time.monotonic()-t;records.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n')
                save(output/'fatal_error.json',{'error':str(e),'failed_record':row,'all_attempts_retained':True});raise
              row['seconds']=time.monotonic()-t;records.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n')
              print(json.dumps({'stage':'generated','done':len(records),'planned':plan['planned'],'case':case['case_id'],'config':cfg['id'],'seconds':row['seconds'],'status':row['status']},ensure_ascii=False),flush=True)
        save(output/'result.json',{'kind':'quality_probe_result','status':'complete','records':records,'planned':plan['planned'],
          'configuration':common.ref(output/'configuration.json'),'runtime':common.ref(output/'runtime.json'),
          'target_pixels_supplied':False,'training_performed':False,'seconds':time.monotonic()-started})
      finally:
        if runtime is not None:runtime.close()
if __name__=='__main__':main()
