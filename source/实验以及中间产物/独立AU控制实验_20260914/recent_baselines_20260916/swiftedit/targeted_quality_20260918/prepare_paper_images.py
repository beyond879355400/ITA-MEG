from pathlib import Path
import hashlib,json,shutil
T=Path(__file__).resolve().parent;S=T.parent;E=S.parent.parent;ROOT=E.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['aligned_input_probe_v1','paper_aligned_remaining5_v1'];records=[]
dest=ROOT/'figures/swiftedit_input_aligned_20260918';dest.mkdir(exist_ok=False)
for name in names:
    folder=T/'runs'/name;result=json.loads((folder/'result.json').read_text());config=json.loads((folder/'configuration.json').read_text())
    assert result['status']=='complete' and config['native_table_configuration'] is False
    assert config['seed']==101 and config['defaults']=={'scale_ta':1,'scale_edit':.2,'scale_non_edit':1,'clamp_rate':3.,'mask_threshold':.5}
    for r in result['records']:
        assert r['status']=='complete'
        im=folder/r['case_id']/'target_edit.png';assert sha(im)==r['stage_files']['target_edit.png']
        out=dest/(r['case_id']+'.png');shutil.copy2(im,out);assert sha(out)==sha(im)
        records.append({'case_id':r['case_id'],'method':'swiftedit','new_copy':True,'candidate_path':str(im),
            'source_sha256':sha(im),'formal_path':out.relative_to(ROOT).as_posix(),
            'source_image_sha256':r['source_sha256'],'native_table_configuration':False,
            'configuration_path':str(folder/'configuration.json'),'configuration_sha256':sha(folder/'configuration.json')})
assert len(records)==8 and len({r['case_id'] for r in records})==8
(T/'paper_images_manifest.json').write_text(json.dumps({'status':'eight_aligned_input_candidates_byte_verified','records':records},ensure_ascii=False,indent=2))
print('PASS: 8 actual outputs, same weights/defaults/seed, source-aligned input; no old statistics altered')
