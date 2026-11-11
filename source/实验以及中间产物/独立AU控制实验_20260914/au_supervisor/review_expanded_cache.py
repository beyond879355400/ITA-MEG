#!/usr/bin/env python3
"""Read-only review of root's 472 new pairs and one CPU feature recomputation."""
import collections
import json
from pathlib import Path
import sys
from align_real_frames import HERE,EXPERIMENT,SPLIT_SHA,environment,sha,write_json


def main():
    environment();sys.path.insert(0,str(EXPERIMENT))
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import norm_crop
    from modeling.frozen_backbone import FrozenImageTextEncoders,ARC_WEIGHTS,RV
    torch.set_num_threads(2)
    folder=EXPERIMENT/'training_cache/expanded_v2_202609141711'
    mf=folder/'manifest.json';m=json.loads(mf.read_text());plan=json.loads((folder/'pair_plan.json').read_text())
    assert sha(folder/'pair_plan.json')==m['source_manifest_sha256']
    assert sha(m['subject_split_path'])==SPLIT_SHA==m['subject_split_sha256']
    assert not m['test_data_used'] and not m['old_generated_images_or_metrics_used']
    assert sha(folder/'prepare_source.py')==m['metadata_recovery']['execution_source_sha256']
    assert sha(folder/'metadata_recovery_source.py')==m['metadata_recovery']['finalization_source_sha256']
    assert sha(EXPERIMENT/'prepare_expanded_training_cache.py')==sha(folder/'metadata_recovery_source.py')
    original_cache_sha=sha(m['cache_path']);assert original_cache_sha==m['cache_sha256']
    payload=torch.load(m['cache_path'],map_location='cpu',weights_only=True);samples=payload['samples']
    assert len(samples)==len(m['pairs'])==472 and payload['schema_version']==2
    alignment=json.loads(Path(plan['alignment_manifest']).read_text());assert sha(plan['alignment_manifest'])==plan['alignment_sha256']
    byid={r['sample_id']:r for r in alignment['samples']}
    split=json.loads(Path(m['subject_split_path']).read_text())['groups'];groups=collections.defaultdict(list)
    for r in alignment['samples']:groups[(r['split'],r['subject'])].append(r)
    expected=set()
    for (role,subject),rows in groups.items():
        assert role in ('train','dev') and subject in split[role] and subject not in split['test']
        zero=sorted([r for r in rows if not any(r['input']['au_vector_0_5'])],key=lambda r:r['input']['frame'])
        source=zero[(len(zero)-1)//2]
        for target in rows:
            if max(target['input']['au_vector_0_5'])<=3:expected.add((role,subject,source['sample_id'],target['sample_id']))
    actual=set();checked_images=set();counts=collections.Counter();subjects=collections.defaultdict(set);no_edit=0
    shapes={'image_patches':(49,512),'source_identity':(512,), 'base_faceid':(512,), 'text_tokens':(77,768),'text_pooled':(768,),
            'target_aus':(12,),'source_aus':(12,), 'latent_mean':(4,64,64),'latent_std':(4,64,64)}
    for p,s in zip(m['pairs'],samples):
        assert all(p[k]==s[k] for k in ('case_id','split','subject','prompt'))
        assert p['subject'] in split[p['split']] and p['subject'] not in split['test']
        for role in ('source','target'):
            r=p['aligned_'+role]
            assert r==byid[r['sample_id']] and r['split']==p['split'] and r['subject']==p['subject']
            if r['sample_id'] not in checked_images:
                for path_key,hash_key in [('raw_path','raw_sha256'),('aligned_path','aligned_sha256')]:assert sha(r[path_key])==r[hash_key]
                checked_images.add(r['sample_id'])
            assert torch.equal(s[role+'_aus'],torch.tensor(r['input']['au_vector_0_5'],dtype=torch.float32))
        assert not any(p['source_au_vector_0_5']) and max(p['target_au_vector_0_5'])<=3
        assert torch.equal(s['source_aus'],torch.zeros(12))
        for k,shape in shapes.items():assert tuple(s[k].shape)==shape and torch.isfinite(s[k]).all()
        assert (s['latent_std']>0).all()
        for k in ('base_faceid','source_identity'):assert abs(float(s[k].norm())-1)<1e-5
        actual.add((p['split'],p['subject'],p['aligned_source']['sample_id'],p['aligned_target']['sample_id']))
        counts[p['split']]+=1;subjects[p['split']].add(p['subject']);no_edit+=int(p['is_no_edit'])
    assert expected==actual and len(actual)==472 and counts=={'train':343,'dev':129}
    assert len(subjects['train'])==15 and len(subjects['dev'])==4
    # Choose first nonzero-target pair by the existing frozen order, not by a score.
    index=next(i for i,p in enumerate(m['pairs']) if any(p['target_au_vector_0_5']))
    p=m['pairs'][index];s=samples[index];source=p['aligned_source'];target=p['aligned_target']
    identity_path=EXPERIMENT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l/w600k_r50.onnx'
    assert sha(identity_path)==m['model_assets']['faceid_condition_weights_sha256']
    assert sha(ARC_WEIGHTS)==m['model_assets']['arc_weights_sha256']
    assert sha(RV/'vae/diffusion_pytorch_model.safetensors')==m['model_assets']['vae_sha256']
    assert sha(RV/'text_encoder/model.safetensors')==m['model_assets']['text_encoder_sha256']
    encoder=FrozenImageTextEncoders('cpu')
    def pixels(row):
        image=np.asarray(Image.open(row['aligned_path']).convert('RGB'),dtype=np.float32)/127.5-1
        return torch.from_numpy(image.transpose(2,0,1).copy())[None]
    with torch.no_grad():
        patches,identity=encoder.image_features(pixels(source))
        mean,std=encoder.encode_latents(pixels(target))
        tokens,pooled=encoder.encode_text([p['prompt']])
    recognizer=get_model(str(identity_path),providers=['CPUExecutionProvider']);recognizer.prepare(ctx_id=-1)
    crop=norm_crop(cv2.imread(source['raw_path']),np.asarray(source['face_kps'],dtype=np.float32),image_size=112)
    emb=recognizer.get_feat(crop).reshape(-1);emb/=np.linalg.norm(emb)
    fresh={'image_patches':patches[0],'source_identity':identity[0],'latent_mean':mean[0],'latent_std':std[0],
           'text_tokens':tokens[0],'text_pooled':pooled[0],'base_faceid':torch.from_numpy(emb.copy()).float()}
    comparisons={}
    for k,v in fresh.items():
        old=s[k];error=(v-old).abs()
        close=bool(torch.allclose(v,old,atol=.001,rtol=.001))
        comparisons[k]={'max_abs_difference':float(error.max()),'mean_abs_difference':float(error.mean()),'allclose_atol_1e_3_rtol_1e_3':close}
        assert close,(k,comparisons[k])
    assert sha(m['cache_path'])==original_cache_sha and sha(mf)==sha(folder/'manifest.json')
    report={'status':'passed','review_script_sha256':sha(__file__),'manifest_sha256':sha(mf),'cache_sha256_before_and_after':original_cache_sha,
            'source_plan_sha256':sha(folder/'pair_plan.json'),'counts':dict(counts),'subjects':{k:sorted(v) for k,v in subjects.items()},
            'test_used':False,'raw_and_aligned_image_hashes_verified':len(checked_images),'all_sources_truly_zero_12au':True,
            'complete_gt_only_pair_selection_reconstructed':True,'no_edit_pairs':no_edit,
            'all_tensor_shapes_labels_finite_verified':True,'latent_semantics':'aligned TARGET posterior mean/std, confirmed for one real nonzero target on CPU',
            'base_faceid_semantics':'buffalo_l w600k_r50 on original-raw112 norm_crop; distinct from IResNet100 patch/identity-loss stream',
            'recompute_case_id':p['case_id'],'recompute_selection':'first nonzero true-target AU vector in frozen pair order',
            'recompute_device':'cpu','recomputed_feature_comparisons':comparisons,
            'recovery_source_note':'original writer has erroneous .bin provenance paths after torch.save; metadata finalizer has no tensor write and correct .safetensors hashes'}
    write_json(HERE/'expanded_cache_review.json',report);print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
