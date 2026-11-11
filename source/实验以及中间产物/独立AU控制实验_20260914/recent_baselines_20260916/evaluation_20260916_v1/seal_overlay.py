"""Metadata/content seal of this small overlay only; no model calls."""
import hashlib
import json
from pathlib import Path
import prepare_overlay as p

folders=[p.SITE,p.HERE/'insightface_built_v1',p.HERE/'build_tools_v1']
records=[]
for folder in folders:
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        p.need(not path.is_symlink(),'Unexpected package symlink')
        st=path.stat();p.need(not st.st_flags&0x40000000,'SF_DATALESS overlay file')
        records.append({'path':str(path.relative_to(p.HERE)),'bytes':st.st_size,'sha256':p.sha(path),
                        'nlink':st.st_nlink,'flags':st.st_flags})
total=sum(r['bytes']for r in records);p.need(total<p.CAP,'Overlay budget exceeded');p.guard()
source=p.HERE/'overlay_import_receipt.json';probe=json.loads(source.read_text());p.need(probe['status']=='import_only_pass','Import not passed')
for package in ['torch','torchvision','numpy']:
    p.need(str(p.CORE)in probe['imports'][package]['path'],'Core was shadowed')
manifest={'status':'import_only_pass_bridge_required','total_bytes':total,'files':records,'no_models_run':True,
          'import_receipt':{'path':str(source),'sha256':p.sha(source)},'core_unchanged':True,
          'opencv_origin_note':'Old site contains both GUI/headless4.11 records; old binary SF_DATALESS read timed out. Root authorized official opencv-python4.11.0.86 wheel, contingent on both bridges.',
          'insightface_origin':{'PyPI_version':'0.7.3','sdist_sha256':'f191f719612ebb37018f41936814500544cd0f86e6fcd676c023f354c668ddf7',
                               'compiled_private_no_source_patch':True,'old_read_failures_retained':True},
          'launcher':{'path':str(p.HERE/'run_eval.sh'),'sha256':p.sha(p.HERE/'run_eval.sh')},
          'python_executable':str(p.CORE/'bin/python'),'protocol_sha256':'281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f',
          'root_runtime_approval_not_created':True}
p.save(p.HERE/'overlay_seal.json',manifest)
print(json.dumps({'status':manifest['status'],'bytes':total,'files':len(records),'seal_sha256':p.sha(p.HERE/'overlay_seal.json')}))
