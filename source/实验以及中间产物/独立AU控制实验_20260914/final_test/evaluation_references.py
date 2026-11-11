"""Separate CPU-only real-target reference alignment; never called by generator."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
from evaluation_contract import HERE,EXP,POLICY_SHA,load_control,require,read,sha,save,ref_path,resolve

DETECTOR_SHA='5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'
DETECTOR_ROOT=EXP/'au_supervisor/models/detector_only'

def validate_source_cache(control,cache):
    require(cache.get('status') in ('complete','completed_with_failures'),'Source cache is not terminal')
    require(cache.get('kind')=='final_test_source_feature_cache' and cache.get('target_pixels_read') is False and cache.get('target_latents_computed') is False and cache.get('vae_loaded') is False,'Source cache crossed target/VAE boundary')
    require(cache.get('protocol_sha256')==control['protocol_sha256'] and cache.get('raw_manifest',{}).get('sha256')==control['raw_manifest_sha256'],'Source cache belongs to another test protocol/raw manifest')
    sources=cache.get('sources',[]);mapping={s['source_id']:s for s in sources};require(len(mapping)==len(sources),'Duplicate source ID')
    cases=cache.get('cases',[]);require([r['case_id'] for r in cases]==control['protocol']['case_ids'],'Source-cache complete case order changed')
    case_map={r['case_id']:r for r in cases}
    for c in cases:
        truth=control['case_map'][c['case_id']];g=truth['generation_input'];s=mapping[c['source_id']]
        require(c['subject']==truth['subject']==s['subject'],'Source subject mismatch')
        require(c['source_au_vector_0_5']==g['source_au_vector_0_5'] and c['target_au_vector_0_5']==g['target_au_vector_0_5'],'Source-cache condition labels changed')
        require(s['raw_source']['sha256']==g['source_image']['sha256'] and resolve(s['raw_source']['path'])==resolve(g['source_image']['path']),'Canonical source not bound to original source frame')
        require(c.get('target_reference_image')==truth['target_reference_image'],'Target reference metadata changed in source cache')
    return mapping,case_map

def make_target_aligner(output):
    # Imports/models occur only after load_control has accepted the frozen protocol.
    import cv2
    import numpy as np
    from PIL import Image,ImageOps
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import estimate_norm
    detector=DETECTOR_ROOT/'models/buffalo_l/det_10g.onnx';require(sha(detector)==DETECTOR_SHA,'Reference detector changed')
    with (output/'detector.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        face=FaceAnalysis(name='buffalo_l',root=str(DETECTOR_ROOT),allowed_modules=['detection'],providers=['CPUExecutionProvider'])
        face.prepare(ctx_id=-1,det_size=(640,640),det_thresh=.15)
    require(set(face.models)=={'detection'},'Reference alignment must load detection only')
    def align(ref):
        path=ref_path(ref);rgb=np.asarray(Image.open(path).convert('RGB'));bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);attempts=[];selected=None
        for variant in ('original','autocontrast_cutoff1_detection_only'):
            detection_rgb=rgb if variant=='original' else np.asarray(ImageOps.autocontrast(Image.fromarray(rgb),cutoff=1))
            for threshold in (.5,.3,.15,.1):
                face.det_model.det_thresh=threshold;found=face.get(cv2.cvtColor(detection_rgb,cv2.COLOR_RGB2BGR));candidates=[]
                for i,f in enumerate(found):
                    box=[float(v) for v in f.bbox];kps=[[float(v) for v in xy] for xy in f.kps];area=(box[2]-box[0])*(box[3]-box[1])
                    finite=all(math.isfinite(v) for v in box+[z for xy in kps for z in xy])
                    candidates.append({'index':i,'bbox':box if finite else None,'kps':kps if finite else None,'area':area if finite and area>0 else None,
                                       'score':float(f.det_score) if math.isfinite(float(f.det_score)) else None})
                attempts.append({'variant':variant,'threshold':threshold,'candidates':candidates})
                valid=[r for r in candidates if r['area'] is not None]
                if valid:selected=max(valid,key=lambda x:(x['area'],-x['index']));break
            if selected:break
        face.det_model.det_thresh=.15
        if selected is None:return {'status':'failed','error':'No reference face detected','detector_attempts':attempts,'raw_image':ref}
        matrix=estimate_norm(np.asarray(selected['kps'],dtype=np.float32),image_size=112)*(512./112.);require(np.isfinite(matrix).all(),'Nonfinite target alignment')
        warped=cv2.warpAffine(bgr,matrix,(512,512),flags=cv2.INTER_LINEAR);dest=output/'canonical_targets'/('sha_'+ref['sha256']+'.png');dest.parent.mkdir(exist_ok=True)
        require(not dest.exists(),'No reference overwrite');require(cv2.imwrite(str(dest),warped),'Reference PNG write failed');require(sha(path)==ref['sha256'],'Raw target changed')
        return {'status':'complete','image':{'path':str(dest),'sha256':sha(dest),'width':512,'height':512,'mode':'RGB'},'raw_image':ref,
                'detector_attempts':attempts,'affine_original_to_512':matrix.tolist(),'selected_face':selected,'alignment_role':'evaluation_target_only'}
    return align

def execute(protocol_path,raw_path,raw_sha,source_cache_path,source_cache_sha,output):
    control=load_control(protocol_path,raw_path,raw_sha,'final_evaluation_references',__file__)
    source_path=ref_path({'path':str(source_cache_path),'sha256':source_cache_sha});cache=read(source_path);sources,case_sources=validate_source_cache(control,cache)
    output=Path(output).resolve();require(output.is_relative_to(HERE) and not output.exists(),'New final_test reference output directory required');output.mkdir(parents=True)
    for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','TORCH_HOME','TMPDIR'):
        folder=output/'cache'/key.lower();folder.mkdir(parents=True);os.environ[key]=str(folder)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');sys.dont_write_bytecode=True
    source_refs={};by_raw={};rows=[]
    # Source canonical files are reused; no VAE or target-dependent source encoder.
    from PIL import Image
    for sid,s in sources.items():
        result={'status':'failed','source_id':sid,'subject':s['subject'],'raw_image':s['raw_source'],'error':s.get('error','source canonical unavailable')}
        try:
            require(s.get('status') in ('complete','aligned') and isinstance(s.get('canonical_source'),dict),'Source-cache alignment failed')
            ref=s['canonical_source'];path=ref_path(ref)
            with Image.open(path) as image:image.load();require(image.mode=='RGB' and image.size==(512,512),'Source canonical not512 RGB')
            result={'status':'complete','source_id':sid,'subject':s['subject'],'image':dict(ref),'raw_image':s['raw_source'],'reused_source_cache':True}
            by_raw[s['raw_source']['sha256']]=result
        except Exception as error:result.update(error_type=type(error).__name__,error=str(error))
        source_refs[sid]=result
    align=make_target_aligner(output);targets={}
    for cid in control['protocol']['case_ids']:
        raw=control['case_map'][cid];meta=control['cases'][cid];source=source_refs[case_sources[cid]['source_id']];target_raw=raw['target_reference_image'];digest=target_raw['sha256']
        if digest not in targets:
            try:
                if digest in by_raw:targets[digest]={**by_raw[digest],'evaluation_target_same_raw_as_source':True}
                else:targets[digest]=align(target_raw)
            except Exception as error:targets[digest]={'status':'failed','raw_image':target_raw,'error_type':type(error).__name__,'error':str(error)}
        rows.append({**meta,'source_reference':source,'target_reference':targets[digest],'target_passed_to_generator':False})
    result={'kind':'final_test_evaluation_references','status':'complete','protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,
      'raw_manifest':{'path':control['raw_manifest_path'],'sha256':raw_sha},'source_cache_manifest':{'path':str(source_path),'sha256':source_cache_sha},'cases':rows,
      'planned_cases':len(rows),'source_valid_cases':sum(r['source_reference']['status']=='complete' for r in rows),'target_valid_cases':sum(r['target_reference']['status']=='complete' for r in rows),
      'source_references_reencoded':False,'targets_only_used_for_evaluation':True,'script_sha256':sha(__file__),'detector_sha256':DETECTOR_SHA}
    save(output/'references.json',result);(output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());return output/'references.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--raw-manifest',type=Path,required=True);p.add_argument('--raw-sha',required=True)
    p.add_argument('--source-cache',type=Path,required=True);p.add_argument('--source-cache-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(execute(a.protocol,a.raw_manifest,a.raw_sha,a.source_cache,a.source_cache_sha,a.output))
if __name__=='__main__':main()
