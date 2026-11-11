#!/usr/bin/env python3
"""Freeze the explicitly authorized post-dev/pre-test measurement policy v2."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def save(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    old=ROOT/'development_evaluation/v4_slot_2k_mlp200_dev19/run/results.json'
    records=json.loads(old.read_text())['records'];fields=('case_id','subject','method','original_method','path','sha256','split','group','cohort','source_labels_12','target_labels_12')
    items=[{k:r[k] for k in fields} for r in records]
    assert len(items)==95 and len({r['case_id'] for r in items})==19
    assert {r['method'] for r in items}=={'source_real','target_real','baseline','current_model','mlp_model'}
    assert all(r['split']=='dev' and sha(r['path'])==r['sha256'] for r in items)
    realfile=ROOT/'au_supervisor/alignment_v1/aligned_manifest.json';alignment=json.loads(realfile.read_text())
    real=[{'sample_id':r['sample_id'],'subject':r['subject'],'split':'dev','path':r['aligned_path'],'sha256':r['aligned_sha256'],
           'labels12':r['input']['au_vector_0_5']} for r in alignment['samples'] if r['split']=='dev']
    assert len(real)==240 and all(sha(r['path'])==r['sha256'] for r in real)
    fnfile=ROOT/'tools/facenet/runs/real_dev_202609141815/protocol.json';fnreal=json.loads(fnfile.read_text())['images']
    assert len(fnreal)==12 and all(sha(r['path'])==r['sha256'] for r in fnreal)
    save(HERE/'dev19_items.json',{'items':items,'source_results_sha256':sha(old),'selection':'all original95 records, only image/label identity fields retained; no outcome selection','test_used':False})
    save(HERE/'real_dev240_items.json',{'items':real,'source_alignment_sha256':sha(realfile),'test_used':False})
    save(HERE/'facenet_real12_items.json',{'images':fnreal,'source_protocol_sha256':sha(fnfile),'test_used':False})
    evidence=ROOT/'external_baselines/evaluation/primary_face_sensitivity_202609142028'
    sensitivity=json.loads((evidence/'results.json').read_text())
    assert len(sensitivity['records'])==6 and all(r['openface']['all_17au_unchanged_by_2Dfp'] for r in sensitivity['records'])
    policy={'policy_id':'evaluation_policy_v2','schema_version':2,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
       'phase':'revised_after_development_diagnostics_before_final_test','introduced_after_dev_results':True,'final_test_accessed':False,
       'revision_reason':'Singleton-only validity conflated tiny texture candidates with failure of a dominant face in a single-person task. Replace outcome-independent geometry selection consistently across real calibration and all methods. Use a separate identity encoder as primary.',
       'legacy_reports_preserved':True,'legacy_policy':'singleton-only OpenFace/identity validity; buffalo used as main available identity metric',
       'source_controls':{'primary_face_choice':'largest geometry area, never identity similarity/AU error/model rank','retain_all_candidates':True,'retain_all_raw_OpenFace_rows':True,
                          'tie_break':'first returned row/candidate index','invalid_geometry':'retain but not rankable; if no finite positive-area geometry then NA',
                          'selected_invalid_values':'affected metric NA; never switch to smaller face to improve a score','duplicate_image_sha_reuse_allowed':True},
       'openface':{'input':'original canonical512 RGB image; no padding/enhancement','backend':'FaceLandmarkImg official CLNF general static AU',
                   'flags':['-aus','-2Dfp','-mloc','build/bin/model/main_clnf_general.txt'],
                   'selection':'largest un-clipped bbox formed by all68 x/y landmarks; geometry-only',
                   'confidence':'retain continuous value; no confidence cutoff added','primary_au_order':[1,2,4,5,6,9,12,15,17,20,25,26],
                   'all_au_order':[1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45],
                   'primary_validity':'selected confidence and all12 AU finite, AU values in[0,5]','tool_seal_sha256':sha(ROOT/'tools/openface/tool_seal.json'),
                   'output_flag_evidence':{'directory':str(evidence.relative_to(ROOT)),'results_sha256':sha(evidence/'results.json'),'protocol_sha256':sha(evidence/'protocol.json'),
                                           'scope':'six-image paired flag check; each corresponding raw row all17AU unchanged, plus source inspection of output-only flag'}},
       'identity_primary':{'name':'FaceNet InceptionResnetV1 VGGFace2, facenet-pytorch','role':'independent of generator condition and identity-loss encoders',
                    'input':'canonical512 PIL RGB, no padding','detector':'MTCNN','image_size':160,'margin':0,'min_face_size':20,
                    'thresholds':[.6,.7,.7],'factor':.709,'keep_all_for_audit':True,'selection':'largest un-clipped detector bbox area; retain all boxes/areas/probabilities/landmarks',
                    'post_process':True,'standardization':'(pixel_0_to_255 -127.5)/128.0, exactly once in author extract()','model_mode':'eval,512D L2-normalized',
                    'device':'cpu','threads':2,'identity_acceptance_threshold':None,'tool_seal_sha256':sha(ROOT/'tools/facenet/tool_seal.json')},
       'identity_auxiliary':{'name':'buffalo_l/w600k_r50','role':'conditioning encoder space, auxiliary only','input':'canonical BGR, uniform128px gray128 padding',
                    'api':'FaceAnalysis detection+recognition CPUExecutionProvider','detector_size':[640,640],'detector_threshold':.5,
                    'selection':'largest un-clipped detector bbox area; retain all candidates','threshold_search':False,'fallback':False,
                    'model_sha256':{'det_10g.onnx':'5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91','w600k_r50.onnx':'4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'}},
       'primary_metrics':{'AU_units':'raw0-5','C':'manual target12 != source12, not selected by predictions','no_edit':'C empty -> C metricsNA; output invariance tested separately',
                    'reported':['target_C_MAE','delta_C_MAE','unchanged_AU_source_relative_drift','all12_MAE','direction_correct_fraction','FaceNet_identity_cosine'],
                    'paired_policy':'per metric common valid cases; cases then subject-equal aggregation; everyn/subjectcount explicit; all failures retained'},
       'extra5_diagnostics':{'aus':[7,10,14,23,45],'primary':False,'reference':'same real target selected-face OpenFace measurements, not invented zero manual labels',
                    'report_all5':True,'missing_reference_or_output':'NA','no_isolated_AU45_selection':True},
       'calibration_order':['all240 realdev OpenFace','original12 unique real images FaceNet (parallel allowed)','then all95 dev19 items for three methods and real references'],
       'inputs':{name:{'path':name,'sha256':sha(HERE/name)} for name in ('dev19_items.json','real_dev240_items.json','facenet_real12_items.json')},
       'freeze_script_sha256':sha(Path(__file__))}
    save(HERE/'policy.json',policy)
    save(HERE/'policy_seal.json',{'policy_id':policy['policy_id'],'policy_sha256':sha(HERE/'policy.json'),'frozen_at_utc':policy['frozen_at_utc'],'final_test_accessed':False})
    print(json.dumps({'policy':str(HERE/'policy.json'),'sha256':sha(HERE/'policy.json'),'items95':len(items),'real240':len(real),'facenet_real12':len(fnreal)},indent=2))

if __name__=='__main__':main()
