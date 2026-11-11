"""Bounded CPU consistency check of four EXISTING DEV sources, not final test.

Thresholds are fixed in source before execution. No VAE/UNet/generation/MPS.
Uses the production SourceRuntime low-level implementation, not a CLI bypass.
"""
from pathlib import Path
import json
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from prepare_source_cache import SourceRuntime,digest,save_json,new_directory

CACHE=ROOT/'training_cache/train2k_batched_202609141853'
CACHE_SHA='7ede6afdfc6948ba4da845309033edc700466d80fc91bb5921410b780e8133e0'
CHECKPOINT=ROOT/'runs/epoch1_matrix_202609142138/training_runs/v4_bidirectional_step2000_202609142138/checkpoint_00002000.pt'
CHECKPOINT_SHA='953b6cab096823a72e1b00ade58a62718d31172fe83793bea477cbe9df0fb4dc'
DEV_SUBJECTS=('SN004','SN012','SN017','SN018')
THRESHOLDS={'canonical_pixel_max_abs':0,'base_faceid_max_abs':1e-6,
            'image_patches_max_abs':1e-4,'source_identity_max_abs':1e-4,
            'text_tokens_max_abs':1e-4,'text_pooled_max_abs':1e-4}


def main():
    import numpy as np
    import torch
    from PIL import Image
    from train_centered_adapter import load_checkpoint
    from prepare_training_cache import make_prompt
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if digest(CACHE/'features.pt')!=CACHE_SHA or digest(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Pinned development cache/checkpoint changed')
    meta=json.loads((CACHE/'manifest.json').read_text())
    if meta['test_data_used'] is not False or meta['cache_sha256']!=CACHE_SHA:raise ValueError('Not the frozen train/dev cache')
    payload=load_checkpoint(CHECKPOINT)
    samples=torch.load(CACHE/'features.pt',map_location='cpu',weights_only=True)['samples']
    if any(row['split'] not in ('train','dev') for row in samples):raise ValueError('Test record in development cache')
    sample_map={row['case_id']:row for row in samples};pairs={}
    for pair in sorted(meta['pairs'],key=lambda p:p['case_id']):
        if pair['split']=='dev':pairs.setdefault(pair['subject'],pair)
    if tuple(sorted(pairs))!=DEV_SUBJECTS:raise ValueError('Development subjects differ from fixed four')
    output=new_directory(HERE/'runtime_checks','cpu_dev_sources')
    receipt={'status':'started','thresholds_fixed_before_run':THRESHOLDS,'threshold_purpose':'numerical implementation consistency, not method performance',
        'cache_sha256':CACHE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,'selected_case_ids':[pairs[s]['case_id'] for s in DEV_SUBJECTS],
        'selection':'first lexicographic dev case per frozen dev subject; source identical for all cases of a subject',
        'source_runtime_sha256':digest(HERE/'prepare_source_cache.py'),'checker_sha256':digest(__file__),
        'device':'cpu','test_pixels_read':False,'vae_loaded':False,'unet_loaded':False,'mps_used':False}
    save_json(output/'started.json',receipt)
    runtime=SourceRuntime('cpu',payload);records=[];started=time.monotonic()
    for subject in DEV_SUBJECTS:
        pair=pairs[subject];original=pair['aligned_source'];cached=sample_map[pair['case_id']]
        if original['split']!='dev':
            raise ValueError('Unexpected source role')
        if cached['split']!='dev' or cached['subject']!=subject or any(pair['source_au_vector_0_5']):raise ValueError('Wrong dev source')
        for path_key,sha_key in (('raw_path','raw_sha256'),('aligned_path','aligned_sha256')):
            if digest(original[path_key])!=original[sha_key]:raise ValueError('Existing DEV source SHA changed')
        with Image.open(original['raw_path']) as image:raw=image.convert('RGB').copy()
        canonical,tensors,alignment=runtime.source(raw)
        with Image.open(original['aligned_path']) as image:old=np.asarray(image.convert('RGB'))
        prompt=make_prompt(pair['target_au_vector_0_5'])
        if prompt!=cached['prompt']:raise ValueError('Cached label-derived prompt changed')
        tokens,pooled=runtime.text(prompt)
        metrics={'canonical_pixel_max_abs':int(np.abs(np.asarray(canonical).astype(np.int16)-old.astype(np.int16)).max()),
            **{key+'_max_abs':float((tensors[key]-cached[key]).abs().max()) for key in ('base_faceid','image_patches','source_identity')},
            'text_tokens_max_abs':float((tokens-cached['text_tokens']).abs().max()),
            'text_pooled_max_abs':float((pooled-cached['text_pooled']).abs().max())}
        checks={key:value<=THRESHOLDS[key] for key,value in metrics.items()}
        row={'subject':subject,'case_id':pair['case_id'],'source_raw_sha256':original['raw_sha256'],
             'old_canonical_sha256':original['aligned_sha256'],'prompt':prompt,'max_abs':metrics,'checks':checks,
             'passed':all(checks.values()),'alignment':alignment}
        records.append(row);save_json(output/(subject+'.json'),row)
        print(json.dumps({'subject':subject,'passed':row['passed'],'max_abs':metrics}),flush=True)
    result={**receipt,'status':'complete','passed':all(r['passed'] for r in records),'records':records,
            'environment':runtime.environment,'source_assets':runtime.assets,'seconds':time.monotonic()-started}
    save_json(output/'result.json',result)
    print(json.dumps({'report':str(output/'result.json'),'passed':result['passed']}),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
