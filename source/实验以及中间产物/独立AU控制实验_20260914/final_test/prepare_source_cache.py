"""Final-test SOURCE-only features. No target-image reader or VAE is present.

CLI always uses the production accepted label plan. Injected runtimes in the
Python API exist solely for synthetic unit tests, never as a CLI bypass.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',
                  HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'),
                  MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
from protocol_gate import (PRODUCTION,validate_control,digest,read_json,resolve_path,
                           verify_runtime_source,require_declared_artifact)


def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n')


def new_directory(parent,prefix):
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    stem=prefix+'_'+datetime.now().strftime('%Y%m%d%H%M')
    for number in range(1,10000):
        path=parent/(stem if number==1 else stem+'-'+str(number).zfill(2))
        try:path.mkdir();return path
        except FileExistsError:pass
    raise RuntimeError('Could not allocate a new non-overwriting output directory')


def verify_ref(ref,label):
    path=resolve_path(ref['path'])
    if digest(path)!=ref['sha256']:raise ValueError(label+' SHA changed')
    return path


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime_environment():
    names=('torch','torchvision','numpy','diffusers','transformers','Pillow','opencv-python','insightface','onnxruntime')
    return {'python':platform.python_version(),'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in names}}


def validate_source_feature_settings(control,requested_device=None):
    spec=control['sampling'].get('source_features',{})
    expected={'implementation','device','dtype','anchor_checkpoint','training_cache_manifest','training_cache_features'}
    if set(spec)!=expected or spec.get('implementation')!='training_compatible_source_features_v1':
        raise ValueError('Source-feature policy fields must be explicit; no ignored settings')
    if spec['device'] not in ('cpu','mps') or spec['dtype']!='torch.float32':raise ValueError('Source features require explicit CPU/MPS FP32 policy')
    if requested_device is not None and requested_device!=spec['device']:raise ValueError('Source-cache device conflicts with frozen policy')
    return spec


def load_train_anchor(control):
    """Read only frozen TRAIN/DEV cache tensors and a declared final checkpoint."""
    import torch
    from train_centered_adapter import load_checkpoint,MODEL_VERSION,resolve_source_anchor,verify_source_anchor
    spec=validate_source_feature_settings(control)
    checkpoint=require_declared_artifact(control,spec['anchor_checkpoint'])
    payload=load_checkpoint(checkpoint)
    if payload.get('model_version')!=MODEL_VERSION:raise ValueError('Source anchor checkpoint is not v4')
    meta_path=verify_ref(spec['training_cache_manifest'],'training cache manifest')
    features_path=verify_ref(spec['training_cache_features'],'training cache features')
    if ROOT not in meta_path.parents or ROOT not in features_path.parents:raise ValueError('Training cache outside experiment')
    meta=read_json(meta_path)
    if meta.get('test_data_used') is not False or meta.get('status')!='complete':raise ValueError('Train anchor cache has test/unfinished records')
    if meta['cache_sha256']!=spec['training_cache_features']['sha256'] or payload['bindings']['cache']!=meta['cache_sha256']:
        raise ValueError('Training cache does not match checkpoint')
    if meta['subject_split_sha256']!=control['protocol']['inputs']['subject_split']['sha256'] or payload['bindings']['split']!=meta['subject_split_sha256']:
        raise ValueError('Training/final split changed')
    samples=torch.load(features_path,map_location='cpu',weights_only=True)['samples']
    if any(row['split'] not in ('train','dev') or row['subject'] in control['label_plan']['test_subjects'] for row in samples):
        raise ValueError('Anchor cache contains test subjects')
    anchor,binding=resolve_source_anchor([r for r in samples if r['split']=='train'],[r for r in samples if r['split']=='dev'])
    verify_source_anchor(anchor,binding,payload['bindings']['source_anchor'])
    del samples
    for relative,expected in payload['bindings']['sources'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Frozen model/training source changed: '+relative)
    receipt={'anchor_checkpoint':{'path':str(checkpoint),'sha256':spec['anchor_checkpoint']['sha256']},
             'training_cache_manifest':{'path':str(meta_path),'sha256':spec['training_cache_manifest']['sha256']},
             'training_cache_features':{'path':str(features_path),'sha256':spec['training_cache_features']['sha256']},
             'binding':binding,'test_anchor_selection':False}
    return anchor,receipt,payload


def source_asset_paths():
    from modeling.frozen_backbone import RV,ARC_WEIGHTS
    identity=ROOT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l'
    paths={f'rv/{name}':RV/name for name in ('text_encoder/config.json','text_encoder/model.safetensors',
           'tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json')}
    paths.update(iresnet100=ARC_WEIGHTS)
    paths.update({f'cached_faceid/{name}':identity/name for name in ('det_10g.onnx','w600k_r50.onnx')})
    return paths


class SourceRuntime:
    """Exact train preprocessing primitives; loads IResNet+CLIP, never VAE."""
    def __init__(self,device,payload):
        import importlib.util
        import torch
        from insightface.model_zoo import get_model
        from transformers import CLIPTextModel,CLIPTokenizer
        from modeling.frozen_backbone import RV,FACE_REPO,ARC_WEIGHTS,check_repo,freeze
        self.device=torch.device(device)
        self.dtype='torch.float32';self.environment=runtime_environment()
        assets=source_asset_paths()
        self.assets={}
        for name,path in assets.items():
            expected=payload['bindings']['public_assets'][name]
            if digest(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:
                raise ValueError('Source feature asset differs from training: '+name)
            self.assets[name]={'path':str(path),**expected}
        check_repo(FACE_REPO,'6e313885188e05740a7c40e12c1b9015c748bf08')
        spec=importlib.util.spec_from_file_location('final_source_iresnet',FACE_REPO/'third_party/insightface_backbone_conv.py')
        arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
        self.arcface=arc.iresnet100()
        self.arcface.load_state_dict(torch.load(ARC_WEIGHTS,map_location='cpu',weights_only=True),strict=True)
        freeze(self.arcface,self.device)
        self.text_encoder=freeze(CLIPTextModel.from_pretrained(str(RV/'text_encoder'),local_files_only=True,torch_dtype=torch.float32),self.device)
        self.tokenizer=CLIPTokenizer.from_pretrained(str(RV/'tokenizer'),local_files_only=True)
        self.detector=get_model(str(assets['cached_faceid/det_10g.onnx']),providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=-1,input_size=(640,640),det_thresh=.15)
        self.recognizer=get_model(str(assets['cached_faceid/w600k_r50.onnx']),providers=['CPUExecutionProvider'])
        self.recognizer.prepare(ctx_id=-1)

    def source(self,image):
        import cv2
        import numpy as np
        import torch
        from PIL import Image,ImageOps
        from insightface.utils.face_align import estimate_norm,norm_crop
        from modeling.frozen_backbone import FrozenImageTextEncoders
        rgb=np.asarray(image);bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);attempts=[];found=None
        for variant in ('original','autocontrast_cutoff1_detection_only'):
            candidate=rgb if variant=='original' else np.asarray(ImageOps.autocontrast(image,cutoff=1))
            for threshold in (.5,.3,.15,.1):
                self.detector.det_thresh=threshold
                boxes,kps=self.detector.detect(cv2.cvtColor(candidate,cv2.COLOR_RGB2BGR),max_num=0,metric='default')
                attempts.append({'variant':variant,'threshold':threshold,'candidates':len(boxes),
                                 'boxes_xyxy_score':boxes.tolist(),'keypoints':None if kps is None else kps.tolist()})
                if len(boxes):
                    index=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
                    found=(boxes[index],kps[index]);break
            if found is not None:break
        if found is None:raise RuntimeError('No source face detected; source/cases remain failed')
        bbox,kps=found;matrix=estimate_norm(kps,image_size=112)*(512./112.)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite source alignment')
        aligned=cv2.warpAffine(bgr,matrix,(512,512),flags=cv2.INTER_LINEAR)
        canonical=Image.fromarray(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB))
        tensor=torch.from_numpy((np.asarray(canonical,dtype=np.float32)/127.5-1).transpose(2,0,1).copy())[None].to(self.device)
        with torch.no_grad():patches,identity=FrozenImageTextEncoders.image_features(self,tensor)
        vector=self.recognizer.get_feat(norm_crop(bgr,kps,image_size=112)).reshape(-1)
        norm=float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm<=0:raise ValueError('Invalid base FaceID feature')
        features={'image_patches':patches[0].cpu(),'source_identity':identity[0].cpu(),
                  'base_faceid':torch.from_numpy((vector/norm).copy()).float()}
        alignment={'affine_original_to_512':matrix.tolist(),'face_kps_original':kps.tolist(),
                   'face_bbox_original':bbox[:4].tolist(),'detection_score':float(bbox[4]),'detector_attempts':attempts}
        return canonical,features,alignment

    def text(self,prompt):
        import torch
        from modeling.frozen_backbone import FrozenImageTextEncoders
        with torch.no_grad():tokens,pooled=FrozenImageTextEncoders.encode_text(self,[prompt])
        return tokens[0].cpu(),pooled[0].cpu()


def source_only_plan(control):
    """Extract source metadata without touching target paths (even with stat)."""
    sources={};cases=[]
    from prepare_training_cache import make_prompt
    for row in control['raw_manifest']['cases']:
        inputs=row['generation_input'];raw=inputs['source_image']
        if any(inputs['source_au_vector_0_5']):raise ValueError('v4 production accepts only known all-zero source AU')
        source_id=row['subject']+'_source_f'+str(raw['label_frame_1based']).zfill(5)
        source={'source_id':source_id,'subject':row['subject'],'split':'test',
                'source_frame_1based':raw['label_frame_1based'],'raw_source':dict(raw)}
        if source_id in sources and sources[source_id]!=source:raise ValueError('Shared source metadata is inconsistent')
        sources[source_id]=source
        cases.append({'case_id':row['case_id'],'subject':row['subject'],'split':'test','source_id':source_id,
                      'source_au_vector_0_5':list(inputs['source_au_vector_0_5']),'target_au_vector_0_5':list(inputs['target_au_vector_0_5']),
                      'target_reference_image':dict(row['target_reference_image']),
                      'prompt':make_prompt(inputs['target_au_vector_0_5'])})
    return list(sources.values()),cases


def check_tensor(value,shape,label):
    import torch
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Invalid source-only tensor: '+label)


def build_source_cache(control,output,anchor,anchor_receipt,runtime):
    """The sole image opening operation is raw_source; no target reader exists."""
    import torch
    from PIL import Image
    sources,cases=source_only_plan(control)
    check_tensor(anchor,(768,),'source_anchor')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'canonical_sources').mkdir()
    features={'source_anchor':anchor.cpu(),'sources':{},'conditions':{}}
    prompt_cache={}
    for source in sources:
        raw=source['raw_source'];path=resolve_path(raw['path'])
        try:
            if digest(path)!=raw['sha256']:raise ValueError('Raw source SHA changed')
            with Image.open(path) as opened:
                if opened.size!=(raw['width'],raw['height']) or opened.mode!=raw['mode'] or raw['mode']!='RGB':
                    raise ValueError('Raw source size/mode differs from manifest')
                image=opened.copy()
            canonical,tensors,alignment=runtime.source(image)
            if canonical.size!=(512,512) or canonical.mode!='RGB':raise ValueError('Expected canonical512 RGB source')
            for key,shape in {'image_patches':(49,512),'source_identity':(512,),'base_faceid':(512,)}.items():check_tensor(tensors[key],shape,key)
            if set(tensors)!={'image_patches','source_identity','base_faceid'}:raise ValueError('Unexpected source tensor or target latent')
            path_out=output/'canonical_sources'/(source['source_id']+'.png');canonical.save(path_out)
            if digest(path)!=raw['sha256']:raise ValueError('Raw source changed during processing')
            features['sources'][source['source_id']]=tensors
            source.update(status='complete',canonical_source={'path':str(path_out.resolve()),'sha256':digest(path_out),'width':512,'height':512,'mode':'RGB'},alignment=alignment)
        except Exception as error:
            source.update(status='failed',canonical_source=None,error=type(error).__name__+': '+str(error))
    for case in cases:
        prompt=case['prompt']
        try:
            if prompt not in prompt_cache:
                tokens,pooled=runtime.text(prompt)
                check_tensor(tokens,(77,768),'text_tokens');check_tensor(pooled,(768,),'text_pooled')
                prompt_cache[prompt]=(tokens,pooled)
            tokens,pooled=prompt_cache[prompt]
            if not any(case['target_au_vector_0_5']) and not torch.equal(pooled,anchor):
                raise ValueError('Neutral text encoding differs from frozen TRAIN anchor')
            features['conditions'][case['case_id']]={'text_tokens':tokens,'text_pooled':pooled,
                'source_aus':torch.tensor(case['source_au_vector_0_5'],dtype=torch.float32),
                'target_aus':torch.tensor(case['target_au_vector_0_5'],dtype=torch.float32),'prompt':prompt}
            case['condition_status']='complete'
        except Exception as error:case.update(condition_status='failed',error=type(error).__name__+': '+str(error))
    feature_path=output/'features.pt'
    with feature_path.open('xb') as handle:torch.save(features,handle)
    manifest={'kind':'final_test_source_feature_cache','status':'complete' if all(s['status']=='complete' for s in sources) and all(c['condition_status']=='complete' for c in cases) else 'completed_with_failures',
        'protocol_sha256':control['protocol_sha256'],'raw_manifest':{'path':control['raw_manifest_path'],'sha256':control['raw_manifest_sha256']},
        'features_file':{'path':str(feature_path.resolve()),'sha256':digest(feature_path)},'source_anchor':anchor_receipt,
        'case_ids':control['case_ids'],'sampling_seeds':control['sampling_seeds'],'sources':sources,'cases':cases,
        'source_assets':runtime.assets,'device':str(runtime.device),'dtype':runtime.dtype,'environment':runtime.environment,
        'source_cache_code_sha256':digest(__file__),'protocol_gate_sha256':digest(HERE/'protocol_gate.py'),
        'target_pixels_read':False,'target_latents_computed':False,'vae_loaded':False,
        'preprocessing':'TRAIN canonical112 affine scaled512; original pixels only; buffalo_l normalized FaceID; IResNet100 49 patches; fixed CLIP prompt tokenizer77'}
    shutil.copyfile(__file__,output/'source_cache_snapshot.py')
    save_json(output/'source_cache_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True);parser.add_argument('--raw-manifest',type=Path,required=True)
    parser.add_argument('--raw-manifest-sha',required=True);parser.add_argument('--device',choices=('cpu','mps'))
    args=parser.parse_args()
    if not args.protocol.is_file() or digest(args.protocol)!=args.protocol_sha:raise ValueError('Explicit frozen protocol SHA required')
    control=validate_control(args.protocol,args.raw_manifest,args.raw_manifest_sha)
    verify_runtime_source(control,'protocol_gate',HERE/'protocol_gate.py');verify_runtime_source(control,'source_cache',Path(__file__))
    device=validate_source_feature_settings(control,args.device)['device']
    anchor,receipt,payload=load_train_anchor(control)
    runtime=SourceRuntime(device,payload)
    # Reserve a name only; build_source_cache creates it exclusively.
    reserved=new_directory(ROOT/'final_test/source_caches','source');reserved.rmdir()
    manifest=build_source_cache(control,reserved,anchor,receipt,runtime)
    print(json.dumps({'status':manifest['status'],'manifest':str(reserved/'source_cache_manifest.json'),'sha256':digest(reserved/'source_cache_manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
