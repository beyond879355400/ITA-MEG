"""Resume only missing keys; allocator-only release before each case reseed."""
import argparse,gc,json,sys
from pathlib import Path
import run_formal as formal

core=formal.core;HERE=formal.HERE


def completed_map(index,verify_outputs=True):
    core.need(index.get('kind')=='recent_baseline_generation_index_v1' and index.get('method')=='swiftedit','Wrong resume method/kind')
    core.need(index['case_manifest_sha256']==formal.MANIFEST_SHA and index['sampling_seeds']==formal.SEEDS,'Resume protocol mismatch')
    case_map={r['case_id']:r for r in formal.stage_manifest()['cases']}
    expected={(name,s) for name in case_map for s in formal.SEEDS}
    seen=set();done={}
    for row in index['records']:
        key=(row['case_id'],row['seed'])
        core.need(key in expected and key not in seen,'Duplicate or unexpected resume key')
        seen.add(key)
        original=case_map[row['case_id']]
        core.need(row['source_sha256']==original['source_sha256'] and row['target_prompt']==original['prompt'] and
                  row['source_au_vector_0_5']==original['source_au_vector_0_5'] and row['target_au_vector_0_5']==original['target_au_vector_0_5'] and not row['target_image_used'],'Resume input/label mismatch')
        if row['status']=='complete':
            core.need(row['primary_output_role']=='target_edit' and row['execution_started'],'Invalid prior completion')
            if verify_outputs:core.need(core.sha(row['output']['path'])==row['output']['sha256'],'Previously completed PNG changed')
            done[key]=row
    core.need(seen==expected,'Prior terminal matrix incomplete')
    return done


def pending_cases(cases,seed,done):
    return [r for r in cases if (r['case_id'],seed) not in done]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=formal.SEEDS,required=True)
    p.add_argument('--prior-index',required=True);p.add_argument('--run-name',required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    prior=Path(args.prior_index).resolve();index=core.read(prior);done=completed_map(index)
    formal.configure(args.seed)
    original_cases=core.fixed_cases;all_cases=formal.stage_manifest()['cases'];pending=pending_cases(all_cases,args.seed,done)
    core.need(pending,'No missing requests for this seed')
    def selected(start=0,count=None):
        core.need(start==0 and count==len(pending),'Resume must include every missing key in original order')
        return pending_cases(original_cases(0,32),args.seed,done)
    core.fixed_cases=selected
    original_gate=core.gate;gate_calls=0
    def boundary_gate(key):
        nonlocal gate_calls
        original_gate(key)
        if key=='gpu_run_go':
            gate_calls+=1
            if gate_calls>1:
                import torch
                torch.mps.synchronize()
                before={'allocated':torch.mps.current_allocated_memory(),'driver':torch.mps.driver_allocated_memory()}
                gc.collect();torch.mps.empty_cache()
                after={'allocated':torch.mps.current_allocated_memory(),'driver':torch.mps.driver_allocated_memory()}
                path=formal.SWIFT/'runs'/args.run_name/'allocator_events.jsonl'
                with path.open('a',buffering=1) as f:f.write(json.dumps({'boundary':gate_calls-1,'before':before,'after':after,'before_reseed':True})+'\n')
    core.gate=boundary_gate
    original_preflight=core.preflight
    def preflight(*a,**kw):
        doc=original_preflight(*a,**kw)
        doc.update(resume_wrapper_sha256=core.sha(Path(__file__)),prior_index={'path':str(prior),'sha256':core.sha(prior)},
                   preserved_completed_keys=len(done),missing_keys_this_seed=[r['case_id'] for r in pending],
                   resume_reason='allocator resource management; no quality-based selection or repeat of completed requests',
                   allocator_policy='case boundary before reseed: synchronize;gc.collect;empty_cache. Live model tensors/dtype/params/watermark unchanged.')
        return doc
    core.preflight=preflight
    args.start=0;args.count=len(pending);args.device='mps'
    if args.execute:core.execute(args)
    else:print(json.dumps(preflight(0,len(pending)),indent=2))


if __name__=='__main__':main()
