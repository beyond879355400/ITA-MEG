#!/usr/bin/env python3
"""Same frozen auxiliary buffalo policy, additional external dev19 groups with native image sizes."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
from openface_v2 import HERE,ROOT,POLICY_SHA,sha,save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',choices=('external_liveportrait_absolute',),required=True);args=parser.parse_args()
    assert sha(HERE/'policy.json')==POLICY_SHA
    policy=json.loads((HERE/'policy.json').read_text());params=policy['identity_auxiliary']
    manifest=HERE/args.folder/'items.json';seal=json.loads((HERE/args.folder/'items_seal.json').read_text())
    assert sha(manifest)==seal['sha256'] and seal['policy_sha256']==POLICY_SHA
    document=json.loads(manifest.read_text());items=document['items'];assert not document['test_used'] and len(items)==(57 if args.folder=='external_driver' else 38)
    for name in ('real_dev_openface/summary.json','facenet_real12/summary.json'):
        ready=json.loads((HERE/name).read_text());assert ready['status']=='complete' and ready['policy_sha256']==POLICY_SHA
    out=HERE/args.folder/'buffalo';out.mkdir(exist_ok=False)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',MPLCONFIGDIR=str(out/'cache/matplotlib'),XDG_CACHE_HOME=str(out/'cache/xdg'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode=True
    import cv2
    import numpy as np
    from PIL import Image
    from insightface.app import FaceAnalysis
    modelroot=ROOT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface'
    for name,expected in params['model_sha256'].items():assert sha(modelroot/'models/buffalo_l'/name)==expected
    with (out/'runtime.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        app=FaceAnalysis(name='buffalo_l',root=str(modelroot),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider']);app.prepare(ctx_id=-1,det_size=(640,640),det_thresh=.5)
    cache={};features={};records=[]
    for item in items:
        assert item['split']=='dev' and sha(item['path'])==item['sha256'];digest=item['sha256']
        if digest not in cache:
            bgr=cv2.imread(item['path'],cv2.IMREAD_COLOR);assert bgr is not None and np.array_equal(bgr,np.asarray(Image.open(item['path']).convert('RGB'))[:,:,::-1])
            padded=cv2.copyMakeBorder(bgr,128,128,128,128,cv2.BORDER_CONSTANT,value=(128,128,128));faces=app.get(padded);candidates=[]
            for index,face in enumerate(faces):
                box=[float(v) for v in face.bbox];valid=all(math.isfinite(v) for v in box) and box[2]>box[0] and box[3]>box[1]
                candidates.append({'index':index,'bbox':[v if math.isfinite(v) else None for v in box],'area':(box[2]-box[0])*(box[3]-box[1]) if valid else None,
                                   'detection_score':float(face.det_score) if math.isfinite(float(face.det_score)) else None,
                                   'landmarks':[[float(v) if math.isfinite(float(v)) else None for v in point] for point in face.kps]})
            eligible=[r for r in candidates if r['area'] is not None];chosen=max(eligible,key=lambda r:(r['area'],-r['index'])) if eligible else None
            record={'face_count':len(faces),'all_candidates':candidates,'selected_index':chosen['index'] if chosen else None,'status':'no_valid_face' if chosen is None else 'selected_invalid',
                    'actual_input_shape':list(bgr.shape),'detector_input_shape':list(padded.shape)}
            if chosen is not None and chosen['detection_score'] is not None:
                f=np.asarray(faces[chosen['index']].normed_embedding,dtype=np.float64);norm=float(np.linalg.norm(f))
                if f.shape==(512,) and np.isfinite(f).all() and math.isfinite(norm) and norm>0:features[digest]=f/norm;record.update(status='valid_primary_face',embedding_norm=norm)
            areas=sorted((c['area'] for c in eligible),reverse=True);record['second_to_first_area_ratio']=areas[1]/areas[0] if len(areas)>1 else None
            cache[digest]=record
        assert sha(item['path'])==digest
        records.append({**item,**cache[digest],'identity_cosine':None,'policy_sha256':POLICY_SHA})
    sources={r['case_id']:r['sha256'] for r in records if r['method']=='source_real'}
    for r in records:
        a=r['sha256'];b=sources[r['case_id']]
        if a in features and b in features:r['identity_cosine']=float(np.dot(features[a],features[b]))
    save(out/'protocol.json',{'policy_sha256':POLICY_SHA,'input_manifest_sha256':sha(manifest),'script_sha256':sha(__file__),'parameters':params,'test_used':False,
                             'coordinate_note_sha256':sha(HERE/'INPUT_COORDINATE_NOTE.md'),'prior_script_sha256':sha(HERE/'buffalo_auxiliary.py')})
    save(out/'results.json',{'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'records':records,'logical_count':len(records),'unique_images_measured':len(cache),
                             'valid_unique_images':len(features),'embeddings_saved':False,'primary_identity_metric':False})
    (out/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'complete':True,'records':len(records),'unique':len(cache),'valid':len(features)}),flush=True)

if __name__=='__main__':main()
