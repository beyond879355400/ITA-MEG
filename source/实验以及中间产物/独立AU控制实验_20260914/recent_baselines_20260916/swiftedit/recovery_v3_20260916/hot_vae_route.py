"""Only the explicitly approved equal-SHA VAE replacement; no numeric change."""
from pathlib import Path

WEIGHT_SHA='a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815'
CONFIG_SHA='424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903'


def factory_route(original,old_root,new_root):
    def load(path,*args,**kwargs):
        if str(path)==str(old_root) and kwargs.get('subfolder')=='vae':
            path=str(new_root)
        return original(path,*args,**kwargs)
    return load


def install(core,swift):
    old=swift/'weights/auxiliary/sd21_community';hot=swift/'hot_assets_v1'
    mapping={old/'vae/diffusion_pytorch_model.safetensors':hot/'vae/diffusion_pytorch_model.safetensors',
             old/'vae/config.json':hot/'vae/config.json'}
    expected={'diffusion_pytorch_model.safetensors':(334643276,WEIGHT_SHA),'config.json':(553,CONFIG_SHA)}
    real_sha=core.sha;original_verify=core.verify_assets
    def verify():
        receipt=core.read(hot/'migration_receipt.json')
        core.need(receipt['status']=='complete' and receipt['required_weight_sha256']==WEIGHT_SHA,'Hot VAE receipt missing')
        for target in mapping.values():
            size,digest=expected[target.name]
            core.need(target.stat().st_size==size and real_sha(target)==digest,'Hot VAE size/SHA mismatch')
        # Existing verification retains every declaration/key/size/proof check.
        # For exactly these two paths it verifies replacement bytes, never
        # falsely claims the retained dataless clone was read successfully.
        core.sha=lambda path:real_sha(mapping.get(Path(path),Path(path)))
        try:paths,checked=original_verify()
        finally:core.sha=real_sha
        for row in checked:
            source=Path(row['path'])
            if source in mapping:
                row['declared_original_path']=str(source);row['path']=str(mapping[source])
                row['original_clone_rehashed']=False;row['same_sha_repair']=True
        return paths,checked
    core.verify_assets=verify
    from diffusers import AutoencoderKL
    AutoencoderKL.from_pretrained=factory_route(AutoencoderKL.from_pretrained,old,hot)
