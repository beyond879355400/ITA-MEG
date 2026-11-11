"""CPU-only negative tests of the three repaired generation entry guards."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock
from align_real_frames import HERE,EXPERIMENT,environment,sha,write_json

def main():
    environment();sys.path.insert(0,str(EXPERIMENT))
    import train_adapter
    source=EXPERIMENT/'generate_adapter_development.py'
    spec=importlib.util.spec_from_file_location('guard_reader',source);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    checkpoint=EXPERIMENT/'runs/v3_global_raw_smoke_202609141727/checkpoint_00000008.pt'
    state=train_adapter.load_checkpoint(checkpoint)
    root=HERE/'adapter_contract_review'
    cfg=json.loads((EXPERIMENT/'configs/v3_global_raw_dev64.json').read_text())
    cases=json.loads(Path(cfg['dev_case_manifest']).read_text());cases['synthetic_negative_test']='metadata-only change must invalidate exact case contract'
    changed=root/'synthetic_changed_cases_DO_NOT_USE.json';write_json(changed,cases)
    outcomes={}
    for kind in ('cfg_disabled','changed_public_asset','changed_case_manifest'):
        args=['runner','--cache',cfg['cache_dir'],'--cases',str(changed) if kind=='changed_case_manifest' else cfg['dev_case_manifest'],
              '--checkpoint',str(checkpoint),'--output',str(root/('must_not_create_'+kind)),'--guidance-scale','1' if kind=='cfg_disabled' else '7.5']
        assets=json.loads(json.dumps(state['bindings']['public_assets']))
        if kind=='changed_public_asset':assets[next(iter(assets))]['sha256']='0'*64
        try:
            with mock.patch.object(sys,'argv',args),mock.patch.object(train_adapter,'source_and_asset_bindings',return_value=({'public_assets':assets},[])):
                runner.main()
        except ValueError as error:outcomes[kind]={'rejected':True,'error':str(error)}
        else:raise AssertionError('Guard did not reject '+kind)
        assert not (root/('must_not_create_'+kind)).exists()
    report={'status':'passed','generation_source_sha256':sha(source),'device':'cpu','models_loaded':False,'test_images_used':False,'checks':outcomes}
    write_json(root/'repaired_guards_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
