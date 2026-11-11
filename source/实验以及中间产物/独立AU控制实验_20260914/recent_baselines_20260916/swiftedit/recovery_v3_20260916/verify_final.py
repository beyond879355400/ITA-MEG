"""Post-generation integrity only; no model, scores, or image modification."""
import argparse,json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;NEW=HERE.parent.parent
sys.path.insert(0,str(NEW))
from runtime_io_v1 import hash_exact
import run_formal as formal
from PIL import Image
import numpy as np


def verify(index_path):
    index_path=Path(index_path);index=json.loads(index_path.read_text())
    cases={r['case_id']:r for r in formal.stage_manifest()['cases']}
    assert index['status']=='complete' and index['method']=='swiftedit'
    assert index['case_manifest_sha256']==formal.MANIFEST_SHA
    assert index['protocol_sha256']=='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'
    expected={(name,s) for name in cases for s in formal.SEEDS};seen=set();times=[];outputs=[]
    sequence=['inverse_vae_encoder','inverse_unet','generation_unet','generation_vae_decoder','generation_vae_decoder']
    for row in index['records']:
        key=(row['case_id'],row['seed']);assert key in expected and key not in seen;seen.add(key)
        case=cases[row['case_id']];receipt=row['native_receipt']
        assert row['status']=='complete' and row['execution_started'] and row['primary_output_role']=='target_edit'
        assert not row['target_image_used'] and row['source_sha256']==case['source_sha256'] and row['target_prompt']==case['prompt']
        assert row['source_au_vector_0_5']==case['source_au_vector_0_5'] and row['target_au_vector_0_5']==case['target_au_vector_0_5']
        assert receipt['seed']==row['seed'] and receipt['case_id']==row['case_id'] and receipt['defaults']==formal.core.DEFAULTS and receipt['finite']
        assert [r['module'] for r in receipt['forward_trace']]==sequence and all(r['finite'] for r in receipt['forward_trace'])
        edited=next(r for r in receipt['files'] if r['role']=='target_edit')
        assert row['output']=={'path':edited['path'],'sha256':edited['sha256']}
        for file in receipt['files']:
            path=Path(file['path']);check=hash_exact(path,path.stat().st_size,file['sha256'])
            with Image.open(path) as image:assert image.size==(512,512) and image.mode=='RGB';image.verify()
            outputs.append(check)
        mask=receipt['author_mask'];path=Path(mask['path']);hash_exact(path,path.stat().st_size,mask['sha256'])
        values=np.load(path,allow_pickle=False)
        assert values.shape==(64,64) and np.isfinite(values).all() and np.isin(values,[0,1]).all()
        assert float(values.mean())==mask['fraction']
        times.append(receipt['seconds'])
    assert seen==expected and len(index['records'])==96
    prior_path=Path(index['prior_index']['path']);hash_exact(prior_path,prior_path.stat().st_size,index['prior_index']['sha256'])
    prior=json.loads(prior_path.read_text());old={(r['case_id'],r['seed']):r for r in prior['records'] if r['status']=='complete'}
    current={(r['case_id'],r['seed']):r for r in index['records']}
    assert len(old)==17 and all(current[key]==row for key,row in old.items())
    return {'status':'pass','index':str(index_path),'requests':96,'unique_keys':96,'seeds':formal.SEEDS,
            'preserved_prior_complete_rows':len(old),'new_complete_rows':96-len(old),'png_count':len(outputs),'mask_count':96,
            'all_output_hashes_dimensions_crc_finite_and_native_call_sequences_valid':True,'source_and_label_protocol_unchanged':True,
            'native_seconds':{'min':min(times),'median':statistics.median(times),'max':max(times),'sum':sum(times)},
            'base_source_status':index['base_source_status'],'quality_scores_computed':False,'model_loaded':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('index');args=parser.parse_args()
    result=verify(args.index)
    path=Path(args.index).parent/'generation_integrity.json'
    with path.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
