"""New train/dev generator pairs from newly decoded, manually labelled frames.

The plan is fixed before feature computation. No detector score or generator
output is used to choose the pair set. Final test subjects are rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(ROOT/'cache/huggingface'),
                  TORCH_HOME=str(ROOT/'cache/torch'), MPLCONFIGDIR=str(ROOT/'cache/matplotlib'))
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT/'cache/pycache')
AU_ORDER = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plan(args):
    if args.output.exists():
        raise ValueError('Plan requires a new directory')
    alignment = json.loads(args.alignment.read_text())
    split = json.loads(args.split.read_text())
    if alignment['subject_split_sha256'] != digest(args.split) or alignment['test_count'] != 0:
        raise ValueError('Alignment/split identity mismatch')
    groups = {}
    for sample in alignment['samples']:
        if sample['split'] not in ('train', 'dev') or sample['status'] != 'aligned':
            raise ValueError('Do not silently discard failures or accept test samples')
        if sample['input']['au_order'] != AU_ORDER:
            raise ValueError('AU ordering changed')
        groups.setdefault((sample['split'], sample['subject']), []).append(sample)
    pairs, counts = [], []
    for (subset, subject), rows in sorted(groups.items()):
        zeros = sorted((r for r in rows if not any(r['input']['au_vector_0_5'])), key=lambda r:r['input']['frame'])
        if not zeros:
            raise ValueError('No genuinely neutral source for '+subject)
        source = zeros[(len(zeros)-1)//2]
        targets = sorted((r for r in rows if max(r['input']['au_vector_0_5']) <= 3), key=lambda r:r['input']['frame'])
        for target in targets:
            case_id = 'fresh_v2_'+subset+'_'+subject+f"_s{source['input']['frame']:05d}_t{target['input']['frame']:05d}"
            pairs.append({'case_id':case_id, 'split':subset, 'subject':subject, 'au_order':AU_ORDER,
                          'source':source, 'target':target,
                          'source_au_vector_0_5':source['input']['au_vector_0_5'],
                          'target_au_vector_0_5':target['input']['au_vector_0_5'],
                          'is_no_edit':not any(target['input']['au_vector_0_5'])})
        counts.append({'split':subset,'subject':subject,'source_frame':source['input']['frame'],
                       'pairs':len(targets),'no_edit':sum(not any(r['input']['au_vector_0_5']) for r in targets)})
    document = {'stage':'train_dev_pair_plan','test_data_used':False,'old_generated_images_or_metrics_used':False,
                'selection':'Per subject use temporal lower median of newly sampled all-12-AU-zero frames as source; retain ALL newly sampled targets with maximum manual AU <=3, including no-edit controls. No output/score selection.',
                'alignment_manifest':str(args.alignment.resolve()),'alignment_sha256':digest(args.alignment),
                'subject_split_path':str(args.split.resolve()),'subject_split_sha256':digest(args.split),
                'subject_split_document':split, 'counts':counts,'pairs':pairs}
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'prepare_source.py')
    (args.output/'pair_plan.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pairs':len(pairs),'subjects':len(groups),'counts':counts,'plan_sha256':digest(args.output/'pair_plan.json')},indent=2))


def cache(args):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import norm_crop
    from modeling.frozen_backbone import FrozenImageTextEncoders, RV, ARC_WEIGHTS
    from prepare_training_cache import make_prompt
    plan_path = args.output/'pair_plan.json'
    document = json.loads(plan_path.read_text())
    if document['test_data_used'] or digest(document['alignment_manifest']) != document['alignment_sha256'] or digest(document['subject_split_path']) != document['subject_split_sha256']:
        raise ValueError('Plan source binding failed')
    if (args.output/'features.pt').exists() or (args.output/'manifest.json').exists():
        raise ValueError('Never overwrite an existing cache')
    pairs = document['pairs']
    images = {}
    for pair in pairs:
        if pair['split'] not in ('train','dev'):
            raise ValueError('No test accepted')
        for role in ('source','target'):
            row = pair[role]
            for path_key, hash_key in (('aligned_path','aligned_sha256'),('raw_path','raw_sha256')):
                path = Path(row[path_key]).resolve()
                if ROOT not in path.parents or digest(path) != row[hash_key]:
                    raise ValueError('New image binding mismatch: '+str(path))
            images[row['sample_id']] = row
    start = time.monotonic()
    encoder = FrozenImageTextEncoders(args.device)
    identity_path = ROOT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l/w600k_r50.onnx'
    recognizer = get_model(str(identity_path), providers=['CPUExecutionProvider'])
    recognizer.prepare(ctx_id=-1)
    sources = {p['source']['sample_id'] for p in pairs}
    targets = {p['target']['sample_id'] for p in pairs}
    features = {}
    with torch.no_grad():
        for index, (sample_id,row) in enumerate(sorted(images.items())):
            rgb = np.asarray(Image.open(row['aligned_path']).convert('RGB'),dtype=np.float32)/127.5-1
            tensor = torch.from_numpy(rgb.transpose(2,0,1).copy())[None].to(args.device)
            feature = {}
            if sample_id in sources:
                patches, identity = encoder.image_features(tensor)
                feature.update(patches=patches[0].cpu(),source_identity=identity[0].cpu())
                raw_bgr = cv2.imread(row['raw_path'])
                original_crop = norm_crop(raw_bgr,np.asarray(row['face_kps'],dtype=np.float32),image_size=112)
                embedding = recognizer.get_feat(original_crop).reshape(-1)
                norm = np.linalg.norm(embedding)
                if not np.isfinite(embedding).all() or norm <= 0:
                    raise RuntimeError('Invalid condition embedding')
                feature['base_faceid'] = torch.from_numpy((embedding/norm).copy()).float()
            if sample_id in targets:
                mean,std = encoder.encode_latents(tensor)
                feature.update(latent_mean=mean[0].cpu(),latent_std=std[0].cpu())
            features[sample_id] = feature
            if index % 20 == 0:
                print('IMAGES_CACHED',index+1,len(images),flush=True)
        samples, metadata_pairs, prompt_cache = [],[],{}
        for pair in pairs:
            source = features[pair['source']['sample_id']]
            target = features[pair['target']['sample_id']]
            prompt = make_prompt(pair['target_au_vector_0_5'])
            if prompt not in prompt_cache:
                tokens,pooled = encoder.encode_text([prompt])
                prompt_cache[prompt] = (tokens[0].cpu(),pooled[0].cpu())
            tokens,pooled = prompt_cache[prompt]
            samples.append({'case_id':pair['case_id'],'split':pair['split'],'subject':pair['subject'],'prompt':prompt,
                            'image_patches':source['patches'],'source_identity':source['source_identity'],
                            'base_faceid':source['base_faceid'],'text_tokens':tokens,'text_pooled':pooled,
                            'target_aus':torch.tensor(pair['target_au_vector_0_5'],dtype=torch.float32),
                            'source_aus':torch.tensor(pair['source_au_vector_0_5'],dtype=torch.float32),
                            'latent_mean':target['latent_mean'],'latent_std':target['latent_std']})
            metadata_pairs.append({**{k:v for k,v in pair.items() if k not in ('source','target')},'prompt':prompt,
                                   'aligned_source':pair['source'],'aligned_target':pair['target']})
    torch.save({'schema_version':2,'samples':samples},args.output/'features.pt')
    model_assets = {'rv_snapshot':str(RV),'arc_weights_sha256':digest(ARC_WEIGHTS),
                    'faceid_condition_weights_sha256':digest(identity_path),
                    'vae_sha256':digest(RV/'vae/diffusion_pytorch_model.safetensors'),
                    'text_encoder_sha256':digest(RV/'text_encoder/model.safetensors')}
    manifest = {'status':'complete','schema_version':2,'test_data_used':False,'old_generated_images_or_metrics_used':False,
                'source_manifest_sha256':digest(plan_path),'source_manifest_path':str(plan_path.resolve()),
                'subject_split_sha256':document['subject_split_sha256'],'subject_split_path':document['subject_split_path'],
                'subject_splits':{k:sorted({p['subject'] for p in pairs if p['split']==k}) for k in ('train','dev')},
                'samples':len(samples),'unique_images':len(images),'patch_shape':[49,512],'resolution':512,
                'device':args.device,'cache_path':str((args.output/'features.pt').resolve()),'cache_sha256':digest(args.output/'features.pt'),
                'model_assets':model_assets,'base_faceid_encoder':'buffalo_l/w600k_r50',
                'patch_and_id_loss_encoder':'IResNet100/glint360k','elapsed_seconds':time.monotonic()-start,'pairs':metadata_pairs}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='pairs'},ensure_ascii=False,indent=2))


def finalize_existing_cache(args):
    """Recover metadata only after completed features and a provenance-path error.

    It does not alter feature tensors or repeat generation. All plan labels,
    subject IDs, image files and the complete tensor schema are revalidated.
    """
    import torch
    from modeling.frozen_backbone import RV, ARC_WEIGHTS
    from prepare_training_cache import make_prompt
    if (args.output/'manifest.json').exists():
        raise ValueError('Cannot replace a completed manifest')
    plan_path=args.output/'pair_plan.json'
    document=json.loads(plan_path.read_text())
    if document['test_data_used'] or digest(document['alignment_manifest'])!=document['alignment_sha256'] or digest(document['subject_split_path'])!=document['subject_split_sha256']:
        raise ValueError('Invalid source binding')
    payload=torch.load(args.output/'features.pt',map_location='cpu',weights_only=True)
    samples=payload['samples']
    if len(samples)!=len(document['pairs']) or payload['schema_version']!=2:
        raise ValueError('Incomplete cache')
    expected_shapes={'image_patches':(49,512),'source_identity':(512,), 'base_faceid':(512,),
                     'text_tokens':(77,768),'text_pooled':(768,),'target_aus':(12,),'source_aus':(12,),
                     'latent_mean':(4,64,64),'latent_std':(4,64,64)}
    metadata,images=[],set()
    for sample,pair in zip(samples,document['pairs']):
        if pair['split'] not in ('train','dev') or any(sample[k]!=pair[k] for k in ('case_id','subject','split')):
            raise ValueError('Sample identity changed')
        for field,shape in expected_shapes.items():
            if tuple(sample[field].shape)!=shape or not torch.isfinite(sample[field]).all():
                raise ValueError('Invalid cached tensor: '+field)
        for side in ('source','target'):
            row=pair[side]
            if not torch.equal(sample[side+'_aus'],torch.tensor(pair[side+'_au_vector_0_5'],dtype=torch.float32)):
                raise ValueError('Labels changed')
            for path_key,hash_key in (('raw_path','raw_sha256'),('aligned_path','aligned_sha256')):
                path=Path(row[path_key]).resolve()
                if ROOT not in path.parents or digest(path)!=row[hash_key]:
                    raise ValueError('Image binding changed')
            images.add(row['sample_id'])
        prompt=make_prompt(pair['target_au_vector_0_5'])
        if sample['prompt']!=prompt:
            raise ValueError('Prompt mismatch')
        metadata.append({**{k:v for k,v in pair.items() if k not in ('source','target')},'prompt':prompt,
                         'aligned_source':pair['source'],'aligned_target':pair['target']})
    identity_path=ROOT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l/w600k_r50.onnx'
    manifest={'status':'complete','schema_version':2,'test_data_used':False,'old_generated_images_or_metrics_used':False,
              'source_manifest_sha256':digest(plan_path),'source_manifest_path':str(plan_path.resolve()),
              'subject_split_sha256':document['subject_split_sha256'],'subject_split_path':document['subject_split_path'],
              'subject_splits':{k:sorted({p['subject'] for p in document['pairs'] if p['split']==k}) for k in ('train','dev')},
              'samples':len(samples),'unique_images':len(images),'patch_shape':[49,512],'resolution':512,'device':args.device,
              'cache_path':str((args.output/'features.pt').resolve()),'cache_sha256':digest(args.output/'features.pt'),
              'base_faceid_encoder':'buffalo_l/w600k_r50','patch_and_id_loss_encoder':'IResNet100/glint360k',
              'model_assets':{'rv_snapshot':str(RV),'arc_weights_sha256':digest(ARC_WEIGHTS),
                              'faceid_condition_weights_sha256':digest(identity_path),
                              'vae_sha256':digest(RV/'vae/diffusion_pytorch_model.safetensors'),
                              'text_encoder_sha256':digest(RV/'text_encoder/model.safetensors')},
              'metadata_recovery':{'reason':'Feature computation and features.pt write completed, then provenance code requested nonexistent .bin weights instead of actual .safetensors. Metadata rebuilt with actual filenames; tensors unchanged.',
                                   'elapsed_seconds':None,'complete_tensor_schema_and_all_labels_verified':True,
                                   'execution_source_sha256':digest(args.output/'prepare_source.py'),
                                   'finalization_source_sha256':digest(__file__)},'pairs':metadata}
    shutil.copyfile(__file__,args.output/'metadata_recovery_source.py')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='pairs'},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('plan','cache','finalize'))
    p.add_argument('--alignment',type=Path)
    p.add_argument('--split',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','mps'),default='mps')
    args=p.parse_args()
    args.output=args.output.resolve()
    if ROOT not in args.output.parents:
        raise ValueError('Output must be inside current experiment')
    if args.phase=='plan':
        if args.alignment is None or args.split is None:
            p.error('plan requires --alignment and --split')
        plan(args)
    elif args.phase=='cache':
        cache(args)
    else:
        finalize_existing_cache(args)


if __name__=='__main__':
    main()
