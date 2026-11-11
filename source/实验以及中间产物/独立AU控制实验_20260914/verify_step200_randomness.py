"""CPU-only shared RNG/order verification of the four real step200 parents."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
import run_epoch1_matrix as matrix
import train_centered_adapter as trainer


def tensor_sha(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def equal_tree(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal_tree(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(equal_tree(a,b) for a,b in zip(left,right))
    return type(left)==type(right) and left==right


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    prepared=matrix.preflight(args.manifest)
    checkpoints={};paths={};traces={};chains={}
    def load(path):
        path=Path(path).resolve()
        if trainer.ROOT not in path.parents:raise ValueError('Checkpoint outside this experiment')
        if path not in checkpoints:checkpoints[path]=trainer.load_checkpoint(path)
        return checkpoints[path]
    def trace_chain(path,visited=None):
        path=Path(path).resolve();visited=set() if visited is None else visited
        if path in visited:raise ValueError('Checkpoint parent cycle')
        visited.add(path);state=load(path);result=trainer.read_json(path.parent/'result.json')
        rows=[];chain=[]
        parent=result.get('parent_checkpoint')
        if parent is not None:
            parent_path=Path(parent['path']).resolve()
            if trainer.sha256(parent_path)!=parent['sha256']:raise ValueError('Parent-chain checksum mismatch')
            if load(parent_path)['global_step']>=state['global_step']:raise ValueError('Parent step does not precede current step')
            rows,chain=trace_chain(parent_path,visited)
        current=[json.loads(line) for line in (path.parent/'training_trace.jsonl').read_text().splitlines()]
        rows+=current;chain.append({'path':str(path),'sha256':trainer.sha256(path),'step':state['global_step'],'trace_rows':len(current)})
        if [row['step'] for row in rows]!=list(range(1,state['global_step']+1)):raise ValueError('Trace chain does not exactly cover all steps')
        return rows,chain
    for task in prepared['tasks']:
        path=Path(task['parent_checkpoint']);variant=task['variant'];paths[variant]=path
        load(path);traces[variant],chains[variant]=trace_chain(path)
    reference=load(paths['bidirectional']);config=reference['effective_config'];n=reference['train_count']
    comparisons={}
    for variant,path in paths.items():
        state=load(path)
        common_seeds={name:state['effective_config'][name] for name in ('seed','noise_seed','order_seed','dev_seed')}
        expected={name:config[name] for name in common_seeds}
        trace_equal=[{k:r[k] for k in ('step','case_ids','timesteps')} for r in traces[variant]]==[{k:r[k] for k in ('step','case_ids','timesteps')} for r in traces['bidirectional']]
        check={'global_step':state['global_step'],'private_noise_equal':torch.equal(state['rng']['private_noise'],reference['rng']['private_noise']),
               'private_order_equal':torch.equal(state['rng']['private_order'],reference['rng']['private_order']),'data_state_equal':state['data_state']==reference['data_state'],
               'shared_seeds_equal':common_seeds==expected,'all_200_case_ids_and_timesteps_equal':trace_equal,
               'train_case_ids_equal':state['bindings']['train_case_ids']==reference['bindings']['train_case_ids'],
               'cache_and_split_equal':all(state['bindings'][k]==reference['bindings'][k] for k in ('cache','split')),
               'python_rng_equal':equal_tree(state['rng']['python'],reference['rng']['python']),
               'numpy_rng_equal':equal_tree(state['rng']['numpy'],reference['rng']['numpy']),
               'architecture_dependent_global_torch_cpu_rng_equal':torch.equal(state['rng']['torch_cpu'],reference['rng']['torch_cpu']),
               'private_noise_sha256':tensor_sha(state['rng']['private_noise']),'private_order_sha256':tensor_sha(state['rng']['private_order']),
               'data_state_sha256':trainer.fingerprint(state['data_state'])}
        required=('private_noise_equal','private_order_equal','data_state_equal','shared_seeds_equal','all_200_case_ids_and_timesteps_equal','train_case_ids_equal','cache_and_split_equal')
        if not all(check[key] for key in required):raise ValueError('Shared training randomness mismatch: '+variant)
        comparisons[variant]=check
    origin_path=Path(chains['bidirectional'][0]['path']);origin=load(origin_path)
    if origin['global_step']!=8 or len(chains['bidirectional'])!=2:raise ValueError('Expected the explicit bidirectional 8 -> 200 chain')
    noise=torch.Generator(device='cpu').manual_seed(config['noise_seed']);order=torch.Generator(device='cpu').manual_seed(config['order_seed'])
    state={'epoch':0,'order':[],'cursor':0};streams={name:hashlib.sha256() for name in ('posterior_epsilon','target_epsilon','timesteps')}
    step8_match=None
    for step in range(1,201):
        ids=trainer.next_indices(state,n,config['batch_size'],order)
        posterior=torch.randn((len(ids),4,64,64),generator=noise)
        target=torch.randn((len(ids),4,64,64),generator=noise)
        timesteps=torch.randint(0,1000,(len(ids),),generator=noise)
        expected_cases=[reference['bindings']['train_case_ids'][i] for i in ids]
        row=traces['bidirectional'][step-1]
        if row['case_ids']!=expected_cases or row['timesteps']!=timesteps.tolist():raise ValueError('Reconstructed private draws disagree with actual trace')
        for key,value in (('posterior_epsilon',posterior),('target_epsilon',target),('timesteps',timesteps)):streams[key].update(value.numpy().tobytes())
        if step==8:step8_match=torch.equal(noise.get_state(),origin['rng']['private_noise']) and torch.equal(order.get_state(),origin['rng']['private_order']) and state==origin['data_state']
    full_match=torch.equal(noise.get_state(),reference['rng']['private_noise']) and torch.equal(order.get_state(),reference['rng']['private_order']) and state==reference['data_state']
    # Independent restart specifically from the archived step8 private states.
    noise.set_state(origin['rng']['private_noise']);order.set_state(origin['rng']['private_order']);state=copy.deepcopy(origin['data_state'])
    for step in range(9,201):
        ids=trainer.next_indices(state,n,config['batch_size'],order)
        torch.randn((len(ids),4,64,64),generator=noise);torch.randn((len(ids),4,64,64),generator=noise);torch.randint(0,1000,(len(ids),),generator=noise)
    resumed_match=torch.equal(noise.get_state(),reference['rng']['private_noise']) and torch.equal(order.get_state(),reference['rng']['private_order']) and state==reference['data_state']
    if not all((step8_match,full_match,resumed_match)):raise ValueError('Private RNG replay failed')
    output={'status':'passed','manifest_sha256':trainer.sha256(args.manifest),'comparisons':comparisons,'parent_chains':chains,
            'seed_to_step8_matches_real_parent':step8_match,'seed_to_step200_matches_all_four':full_match,'step8_to_step200_replay_matches':resumed_match,
            'reconstructed_common_draw_stream_sha256':{k:h.hexdigest() for k,h in streams.items()},
            'shared_scope':'private posterior/noise/timestep streams, fixed data order and fixed-dev seed; not architecture-dependent parameter initialization',
            'noise_draw_hash_scope':'reconstructed from recorded seed, shape and frozen algorithm; raw training noise was not separately logged',
            'global_torch_cpu_note':'Different architectures consume different CPU RNG counts during parameter initialization; this is not the private data/noise RNG.',
            'cpu_only':True,'models_instantiated':False,'mps_rng_restored':False,'training_started':False,'script_sha256':trainer.sha256(__file__)}
    trainer.json_new(args.output,output)
    print(json.dumps({'status':'passed','comparisons':comparisons,'seed_to_step8':step8_match,'seed_to_step200':full_match,'step8_to_step200':resumed_match,'output':str(args.output)},indent=2))


if __name__=='__main__':main()
