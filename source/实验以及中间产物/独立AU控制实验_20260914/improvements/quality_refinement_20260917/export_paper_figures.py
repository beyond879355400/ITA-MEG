"""Copy verified raw outputs only; emit a patch for two complete five-method figures."""
from pathlib import Path
import hashlib,json,re,shutil
from PIL import Image
Q=Path(__file__).resolve().parent;EXP=Q.parents[1];ROOT=EXP.parents[1]
C=Q/'figures_candidate/bound_cfg35_v3';M=json.loads((C/'bound_manifest.json').read_text())
R=json.loads((Q/'runs/qualitative_four_v1/result.json').read_text());rows={r['case_id']:r for r in R['records']}
assert R['status']=='complete' and len(rows)==4 and R['included_in_existing_main_statistics'] is False
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dest=ROOT/'figures/quality_samples_20260917';assert not dest.exists();dest.mkdir()
replacements={};receipt=[]
roles=['source','target','bidirectional_local_lp_s050','magicface','faceadapter','liveportrait','swiftedit','performrecast']
for case in M['cases']:
    assert [t['role'] for t in case['tiles']]==roles
    row=rows[case['case_id']];assert row['status']=='complete' and row['seed']==101 and row['qualitative_only']
    assert row['unchanged_conditions_source_noise'] and row['receipt']['guidance_scale']==3.5
    source=next(t for t in case['tiles'] if t['role']=='source')
    with Image.open(source['path']) as a,Image.open(row['source']['path']) as b:
        assert a.size==b.size==(512,512) and a.convert('RGB').tobytes()==b.convert('RGB').tobytes()
    for tile in case['tiles']:
        src=Path(tile['path']);assert sha(src)==tile['sha256']
        if tile['role']=='bidirectional_local_lp_s050':
            assert str(src)==row['output'] and tile['sha256']==row['sha256']
            assert tile['display_geometry']['geometry_is_new_image_detection'] is False
        name=tile['formal_destination_filename'];out=dest/name;assert not out.exists()
        shutil.copy2(src,out);assert sha(out)==sha(src)
        rel=out.relative_to(ROOT).as_posix();replacements[tile['tex_path']]=rel
        receipt.append({'case_id':case['case_id'],'role':tile['role'],'source':str(src),'destination':rel,
            'sha256':sha(out),'copied_exact':True,'trim_bp':tile['display_geometry']['trim_bp'],
            'new_generated_output':tile['role']=='bidirectional_local_lp_s050',
            'evaluation_missing':tile['display_geometry'].get('missing_evaluation',False)})
patch='*** Begin Patch\n'
for incoming,name in [('five_methods_pair1.tex','quality_comparison_expansion.tex'),('five_methods_pair2.tex','quality_comparison_diagnostics.tex')]:
    text=(C/incoming).read_text()
    for old,new in replacements.items():text=text.replace(old,new)
    text=text.replace('\\begingroup\n','\\begingroup\n\\fontsize{9pt}{11pt}\\selectfont\\setlength{\\parskip}{0pt}\\setlength{\\parindent}{0pt}\n',1)
    assert '/Users/' not in text and '../' not in text and '占位' not in text
    assert len(re.findall(r'\\QualityFaceTile\{',text))==16
    assert all(text.count('{'+n+'}')==2 for n in ['MagicFace','Face-Adapter','LivePortrait','SwiftEdit','PerformRecast'])
    path=ROOT/'figures'/name;assert not path.exists()
    patch+='*** Add File: '+str(path)+'\n'+''.join('+'+l+'\n' for l in text.splitlines())
patch+='*** End Patch\n';(Q/'paper_figures.patch').write_text(patch)
(Q/'export_receipt.json').write_text(json.dumps({'status':'assets_verified_patch_ready','source_manifest':str(C/'bound_manifest.json'),
    'source_manifest_sha256':sha(C/'bound_manifest.json'),'new_images':4,'original_images_reused':28,
    'all_five_baselines_per_case':True,'png_pixels_modified':False,'records':receipt},ensure_ascii=False,indent=2))
print(json.dumps({'copied':len(receipt),'new_images':4,'patch':str(Q/'paper_figures.patch')},ensure_ascii=False))
