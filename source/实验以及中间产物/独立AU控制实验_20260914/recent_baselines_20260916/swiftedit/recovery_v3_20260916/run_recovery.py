"""New runtime/gate wrapper; old experiment runner and outputs stay unchanged."""
import argparse,json,sys,os,tempfile
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
SWIFT=HERE.parent
TMP_ROOT=SWIFT.parent/'tmp_swiftedit_recovery_v3'
TMP_ROOT.mkdir(exist_ok=True)
for key in ['TMPDIR','TEMP','TMP']:os.environ[key]=str(TMP_ROOT)
tempfile.tempdir=str(TMP_ROOT)
sys.path.insert(0,str(SWIFT))
import run_first as base
base.MANIFEST=HERE/'verified_inputs/smoke_dev4.json'

old_read=base.read
def current_read(path):
    path=Path(path)
    return old_read(HERE/'approvals.json') if path.resolve()==(SWIFT/'approvals.json').resolve() else old_read(path)

def current_gate(key):
    entry=old_read(HERE/'approvals.json').get(key)
    base.need(isinstance(entry,dict) and entry.get('granted_by')=='root' and entry.get('message'),'Current root gate missing: '+key)

base.read=current_read;base.gate=current_gate
old_preflight=base.preflight
def preflight(*args,**kwargs):
    result=old_preflight(*args,**kwargs)
    receipt=old_read(HERE/'runtime_import_pass.json')
    base.need(receipt['status']=='pass' and not receipt['old_environment_imported'],'Fresh runtime import gate absent')
    base.need(Path(sys.prefix).resolve()==(SWIFT.parent/'runtime_local_v2').resolve(),'Use the new independent runtime')
    base.need(base.MANIFEST.stat().st_size==6680 and base.sha(base.MANIFEST)=='e4aecce6e6e0206f0788bfa24a6eaa90bd3dc2e2f8c36a2245575a35f5db1d09','Frozen manifest copy mismatch')
    result.update(recovery_wrapper_sha256=base.sha(Path(__file__)),runtime_import_receipt_sha256=base.sha(HERE/'runtime_import_pass.json'),
                  runtime_requirements_lock_sha256=base.sha(HERE/'requirements.lock'),runtime_interpreter=sys.executable,
                  development_only_before_formal_go=True,old_experiment_modified=False,temporary_root=str(TMP_ROOT))
    return result
base.preflight=preflight


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--verify-assets',action='store_true')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=1)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--run-name',default='recovery_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args()
    base.need(Path(args.run_name).name==args.run_name and args.run_name not in ['.','..'],'Unsafe run name')
    if args.execute:base.execute(args)
    else:print(json.dumps(preflight(args.start,args.count,args.verify_assets),ensure_ascii=False,indent=2))
