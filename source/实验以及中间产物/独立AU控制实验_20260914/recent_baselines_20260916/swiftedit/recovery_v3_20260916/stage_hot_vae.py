"""Root-approved single fixed VAE repair; ordinary byte files, never clones."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;SWIFT=HERE.parent
sys.path.insert(0,str(SWIFT))
from stage_components import get_one
from stage_weights import read,save_new,sha,need,now

root=SWIFT/'hot_assets_v1';root.mkdir(exist_ok=True)
manifest=read(SWIFT/'aux_components.json')
selected=[row for row in manifest['components'] if row['role']=='sd21_community' and row['path'].startswith('vae/')]
need(len(selected)==2,'Expected exact VAE config+weights')
files=[]
for original in selected:
    row={k:v for k,v in original.items() if k!='local_reuse_source'}
    row['local_path']=row['path']
    actual=get_one(row,root)
    files.append({**actual,'original_path':str(SWIFT/'weights/auxiliary'/original['local_path']),
                  'fixed_url':row['url'],'expected_hash':row['hash'],'expected_hash_kind':row['hash_kind']})
receipt={'status':'complete','at':now(),'authority':'root explicitly authorized only334MB exactSHA repair after oldclone+source confirmed dataless',
         'method':'ordinary sequential URL bytes, no APFS clone, no hardlink','old_assets_modified':False,
         'required_weight_sha256':'a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815',
         'files':files,'aux_components_sha256':sha(SWIFT/'aux_components.json')}
save_new(root/'migration_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
