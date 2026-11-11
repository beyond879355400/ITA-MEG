"""Generate four explicitly qualitative examples with one fixed setting, no metrics."""
from pathlib import Path
import argparse,json,sys,time,traceback
W=Path(__file__).resolve().parent;EXP=W.parents[1];GEN=EXP/'completion_20260915/generation'
sys.dont_write_bytecode=True;sys.path[:0]=[str(GEN),str(EXP),str(EXP/'completion_20260915/delivery')]
import common,run_grid,reproduce_driver as rep,sampler

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    cfg=common.read(W/'qualitative_config.json');frozen,refs=rep.frozen_inputs()
    _,labels=common.load_freeze(refs['freeze']['path'],refs['freeze']['sha256'])
    cache,_=run_grid.load_cache(refs['source_cache']['path'],refs['source_cache']['sha256'],labels,refs['freeze']['sha256'],False)
    mapping={c['case_id']:c for c in cache['cases']};cases=[mapping[c] for c in cfg['case_ids']]
    assert len(cases)==4 and cfg['included_in_existing_main_statistics'] is False and cfg['guidance']==3.5 and cfg['steps']==20
    print(json.dumps({'stage':'preflight','cases':cfg['case_ids'],'guidance':cfg['guidance'],'steps':cfg['steps'],'outputs':4,'qualitative_only':True}),flush=True)
    if not a.execute:return
    assert a.output and a.output.resolve().parent==W/'runs' and not a.output.exists()
    assert not rep.active_generators()
    with rep.exclusive_lock():
      out=a.output.resolve();out.mkdir(parents=True);(out/'images').mkdir()
      common.save(out/'configuration.json',{'kind':'four_qualitative_examples','parameters':cfg,'config':common.ref(W/'qualitative_config.json'),
        'inputs':refs,'runner':common.ref(__file__),'sampler':common.ref(sampler.__file__),'target_pixels_supplied':False,
        'scope':'No replacement of original quantitative measurements; four outputs for labelled qualitative figures only'})
      torch=common.environment('mps');cache,features=run_grid.load_cache(refs['source_cache']['path'],refs['source_cache']['sha256'],labels,refs['freeze']['sha256'],True)
      from geometry import build_masks
      import numpy as np
      measurements=common.read(refs['source_measurements']['path'])
      assert measurements['protocol_sha256']==refs['freeze']['sha256'] and measurements['source_cache_sha256']==refs['source_cache']['sha256'] and measurements['source_only']
      measures={r['case_id']:r for r in measurements['records'] if r['method']=='source_real'};sources={s['source_id']:s for s in cache['sources']}
      original=common.read(EXP/'completion_20260915/generation_runs/selected_v11_32_v1/result.json')
      old={r['case_id']:r for r in original['records'] if r['method']=='bidirectional_local_lp_s050' and r['seed']==cfg['seed']}
      common.save(out/'cases.json',{'cases':cases,'target_pixels_supplied':False})
      runtime=None;records=[];started=time.monotonic()
      try:
        print(json.dumps({'stage':'loading_models'}),flush=True)
        runtime=run_grid.Runtime(features,cache,out,torch)
        print(json.dumps({'stage':'models_ready','seconds':time.monotonic()-started}),flush=True)
        for case in cases:
          cid=case['case_id'];source=sources[case['source_id']]
          edit,cleanup,mr=build_masks(case,source,measures[cid]);folder=out/'masks'/cid;folder.mkdir(parents=True)
          for name,array in [('edit',edit),('cleanup',cleanup)]:
            with (folder/(name+'.npy')).open('xb') as f:np.save(f,array,allow_pickle=False)
          common.save(folder/'receipt.json',mr);mask=(edit,cleanup,common.ref(folder/'receipt.json'))
          prepared=runtime.prepare('bidirectional',case,source,mask);prepared['quality_trace_dir']=out/'traces'/cid
          t=time.monotonic();image,receipt=sampler.sample(runtime,prepared,cfg['seed'],guidance=cfg['guidance'],steps=cfg['steps'])
          original_row=old[cid];assert common.sha(original_row['output'])==original_row['sha256']
          for key in ('condition_hashes','source_latent_sha256','noise_sha256'):assert receipt[key]==original_row[key],key
          path=out/'images'/(cid+'_qual_cfg3p5_seed101.png');image.save(path);image.close()
          row={'case_id':cid,'subject':case['subject'],'seed':cfg['seed'],'status':'complete','output':str(path),'sha256':common.sha(path),
            'source':source['canonical_source'],'target_aus_0_5':case['target_au_vector_0_5'],'configuration':common.ref(out/'configuration.json'),
            'reference_output':{'path':original_row['output'],'sha256':original_row['sha256']},'unchanged_conditions_source_noise':True,
            'qualitative_only':True,'included_in_existing_main_statistics':False,'receipt':receipt,'seconds':time.monotonic()-t}
          records.append(row);common.save(out/(cid+'_receipt.json'),row)
          print(json.dumps({'stage':'generated','done':len(records),'planned':4,'case':cid,'seconds':row['seconds']}),flush=True)
        common.save(out/'result.json',{'kind':'qualitative_examples','status':'complete','records':records,'planned':4,
          'configuration':common.ref(out/'configuration.json'),'runtime':common.ref(out/'runtime.json'),'included_in_existing_main_statistics':False})
      except BaseException as e:
        common.save(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'completed_records':records,'incomplete':True});raise
      finally:
        if runtime:runtime.close()
if __name__=='__main__':main()
