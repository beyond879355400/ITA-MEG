"""Lossless native index adapter; no generation or numerical measurement."""
from pathlib import Path
import json
import extent_io as io

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
NATIVE=NEW/'performrecast/formal_runs/final32_v3/generation_index.json'
NATIVE_SHA='32e0b1210454b67657eaf6ac48e0bb87c4b57b85561cc94c87b0ae0d8f9243a4'
CASES=NEW/'preservation/bundle_v2_20260916/inputs/postfreeze32.json'
CASE_SHA='fa88ca09eb8253ce61ba046cb8d9b4c4affaef6cc421893cedb487c09522413b'
PROTOCOL_SHA='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'

def main():
    native=io.read_json(NATIVE,NATIVE_SHA);cases=io.read_json(CASES,CASE_SHA)['cases']
    assert native['kind']=='performrecast32_selected_frame_index'
    assert native['evaluation_protocol_sha256']==PROTOCOL_SHA
    assert native['status']=='complete'and native['case_level_deterministic_outputs']is True
    assert native['seed_aliases']==0 and native['selected_frame']==0
    assert [r['case_id']for r in native['rows']]==[c['case_id']for c in cases]and len(cases)==32
    rows=[]
    for r,c in zip(native['rows'],cases):
        assert r['subject']==c['subject']and r['status']=='complete'
        assert r['source_sha256']==c['source_sha256']and r['target_sha256']==c['target_sha256']
        assert r['source_aus_0_5']==c['source_au_vector_0_5']and r['target_aus_0_5']==c['target_au_vector_0_5']
        output=Path(r['output']);assert output.name=='frame0000.png'and output.parent.name=='native_crop_png'
        assert io.sha(output)==r['sha256']
        rows.append({'method':'performrecast','case_id':r['case_id'],'subject':r['subject'],
            'split':c['split'],'group':c['group'],'seed':None,'seed_not_applicable':True,
            'native_initialization_seed':r['seed'],'source_sha256':r['source_sha256'],'target_sha256':r['target_sha256'],
            'source_au_vector_0_5':r['source_aus_0_5'],'target_au_vector_0_5':r['target_aus_0_5'],
            'target_image_used':True,'primary_output_role':'native_crop_frame0000','status':'complete',
            'execution_id':'performrecast:'+r['case_id']+':'+output.parents[1].name,
            'execution_started':True,'output':{'path':str(output),'sha256':r['sha256']},
            'native_receipt':io.ref(r['receipt_path']),'native_row_index':len(rows),'failure_reason':None})
    assert len({r['execution_id']for r in rows})==32
    result={'kind':'recent_baseline_generation_index_v1','method':'performrecast','status':'complete',
            'protocol_sha256':PROTOCOL_SHA,'case_manifest_sha256':CASE_SHA,'case_ids':[c['case_id']for c in cases],
            'input_group':'source_real_target_driver','sampling_seeds':[],
            'native_generation_protocol_sha256':native['protocol_sha256'],'native_index':io.ref(NATIVE),
            'adapter':io.ref(__file__),'deterministic_seed_semantics':'native101 initialization only; no stochastic replication',
            'records':rows}
    out=HERE/'performrecast_normalized_index_v1.json'
    with out.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'pass','records':len(rows),'output':io.ref(out),'images_changed':False,'models_called':False}))

if __name__=='__main__':main()
