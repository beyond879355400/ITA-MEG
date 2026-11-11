"""Explicit200/2000 IP1.0 gates; original0.9/IP2000 gates remain byte-identical."""
import importlib.util
from pathlib import Path
import re
import sys
import ip10_gate as _ip
import step2000_gate as _old

HERE=_old.HERE;ROOT=_old.ROOT;POLICY_SHA=_old.POLICY_SHA
REFERENCES=('magicface','liveportrait','faceadapter','liveportrait_absolute')
VARIANTS=_old.VARIANTS
require=_old.require;read=_old.read;sha=_old.sha;save=_old.save;GateError=_old.GateError
check_input_metadata=_old.check_input_metadata
IP_GATE_SHA='e4eee92fb8dd919f003ebe63501b990ca60435eab288c270cb5b00d0b188c3fd'
LEGACY_AUDIT_SHA='105d4ef94c9c15096295b0c557de4eb320c89ce5016f667011e56809e4928f61'

def valid_budget(budget):require(type(budget) is int and budget in (200,2000),'Explicit budget must be200 or2000')
def labels_for_budget(budget):
    valid_budget(budget)
    return {'baseline':f'baseline_ip10_b{budget}',**{v:f'{v}_step{budget}_ip10' for v in VARIANTS}}

def _trace200(checkpoint):
    # This already-verified helper reconstructs the real parent chain (bidir8+192),
    # instead of inventing a nonexistent200-step parent for the200 budget.
    path=HERE.parent/'v4_slot_2k_mlp200_dev19/evaluate.py'
    require(sha(path)=='fb86ccd1820118d90ecb45450628d9a1059d13aaf24f3b6f6971c008ee589cf9','Sealed parent-trace helper changed')
    sys.path.insert(0,str(HERE.parent))
    spec=importlib.util.spec_from_file_location('ip_budget_parent_trace',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.trace_to(checkpoint)

def gate_source(budget):
    valid_budget(budget);require(sha(_ip.__file__)==IP_GATE_SHA,'Previously reviewed IP gate changed')
    core,text=_ip.derive_gate()
    def change(a,b):
        nonlocal text
        require(text.count(a)==1,'Ambiguous explicit-budget adaptation: '+a);text=text.replace(a,b)
    change("LABELS={'baseline':'baseline_ip10',**{v:v+'_step2000_ip10' for v in VARIANTS}}",f'LABELS={labels_for_budget(budget)!r}')
    change("return {'baseline':'baseline_ip10',**method_map}",f"return {{'baseline':'baseline_ip10_b{budget}',**method_map}}")
    change("'parent_step200_sha256':oldmethod['sha256']","'reference_step200_checkpoint_sha256':oldmethod['sha256']")
    if budget==200:
        change("method['name']==f'source_centered_{v}_step2000_{digest[:8]}'","method['name']==f'source_centered_{v}_step200_{digest[:8]}'")
        change("final['global_step']==seal['step']==2000","final['global_step']==seal['step']==200")
        change("expected_parent=Path(oldmethod['checkpoint']);require(final['parent_checkpoint']['sha256']==oldmethod['sha256'] and Path(final['parent_checkpoint']['path']).resolve()==expected_parent.resolve(),'Not resumed from sealed corresponding step200')",
               "expected_parent=Path(oldmethod['checkpoint']);require(method['checkpoint_sha256']==oldmethod['sha256'] and checkpoint.resolve()==expected_parent.resolve(),'Not the exact sealed corresponding200 checkpoint')")
        change("trace=[json.loads(l) for l in (checkpoint.parent/'training_trace.jsonl').read_text().splitlines()]","trace=_trace200(checkpoint)")
        change("require([r['step'] for r in trace]==list(range(201,2001)),'Expected the full resumed1800-step trace')",
               "require([r['step'] for r in trace]==list(range(1,201)),'Expected exact complete200-step parent-assembled trace')")
        change("'resumed_schedule_all_equal':True","'training_schedule_all_equal':True")
        change("'status':'validated_complete_ip10_step2000'","'status':'validated_complete_ip10_step200'")
    namespace={'__name__':f'ip_explicit_budget_{budget}','__file__':str(Path(__file__).resolve()),'_trace200':_trace200}
    exec(compile(text,str(core),'exec'),namespace)
    return text,namespace

def checkpoint_budget_match(budget,new,reference):
    valid_budget(budget);newmap={r['variant']:r['checkpoint_sha256'] for r in new};refmap={r['variant']:r['checkpoint_sha256'] for r in reference}
    require(set(newmap)==set(VARIANTS) and len(new)==len(reference)==4 and newmap==refmap,'Exactly the corresponding four same-budget checkpoints required')

def reference_for_budget(budget,reference_group):
    valid_budget(budget)
    if budget==200:
        require(reference_group in (None,'sealed_step200'),'Budget200 uses its sealed200 evidence, not a2000 reference')
        require(sha(_old.OLD_AUDIT)==LEGACY_AUDIT_SHA and sha(_old.REFERENCE)==_old.REFERENCE_SHA,'Sealed200 evidence changed')
        verification=read(_old.REFERENCE.parent/'independent_verification.json')
        require(verification['status']=='pass' and verification['input_sha256']==_old.REFERENCE_SHA,'Sealed200 comparison lacks independent verification')
        methods=read(_old.OLD_AUDIT)['methods']
        reference=[{'variant':m['variant'],'checkpoint_sha256':m['sha256']} for m in methods]
        return reference,{'reference_ip09_group':'sealed_step200','reference_ip09_kind':'frozen_four200_checkpoint_audit_and_comparison',
          'reference_ip09_comparison_sha256':_old.REFERENCE_SHA,'reference_ip09_checkpoint_audit_sha256':LEGACY_AUDIT_SHA}
    require(isinstance(reference_group,str) and re.fullmatch('step2000_[a-z0-9_]+',reference_group),'Budget2000 requires its actual verified0.9 group')
    folder=HERE/reference_group;old=read(folder/'items.json');seal=read(folder/'items_seal.json');result=read(folder/'comparison/results.json');check=read(folder/'comparison/independent_verification.json')
    require(sha(folder/'items.json')==seal['sha256'] and result['status']=='complete' and check['status']=='pass' and check['input_sha256']==sha(folder/'comparison/results.json'),'2000reference not sealed/complete/verified')
    actual=_old.validate(old['run']);actual.pop('reference_records');require(actual==old,'2000reference generation changed')
    return old['checkpoints'],{'reference_ip09_group':reference_group,'reference_ip09_kind':'completed_verified2000_group',
      'reference_ip09_input_sha256':sha(folder/'items.json'),'reference_ip09_comparison_sha256':sha(folder/'comparison/results.json')}

def validate(run,budget,reference_group=None):
    valid_budget(budget);reference,receipt=reference_for_budget(budget,reference_group)
    text,namespace=gate_source(budget);new=namespace['validate'](run);checkpoint_budget_match(budget,new['checkpoints'],reference)
    new.update(**receipt,expected_budget=budget,method_labels=labels_for_budget(budget),input_scale=1.0,
       baseline_scope='all19 fresh baseline images from this sameIP1.0 run; no0.9 or cross-budget baseline measurement reuse',
       gate_parent_core_sha256=_ip.CORE_SHA,ip_gate_sha256=IP_GATE_SHA,
       budget_gate_source_sha256=__import__('hashlib').sha256(text.encode()).hexdigest())
    return new

def prepare(run,group,budget,reference_group=None):
    valid_budget(budget);require(isinstance(group,str) and re.fullmatch('ipbudget_[a-z0-9_]+',group),'New ipbudget_ group required')
    output=HERE/group;require(not output.exists(),'Never overwrite previous budget outputs')
    doc=validate(run,budget,reference_group);references=doc.pop('reference_records')
    require(len(doc['items'])==114 and len(references)==114 and all(r['method']!='baseline' for r in references),'Unexpected baseline/reference mix')
    output.mkdir();save(output/'reference_records.json',{'records':references,'policy_sha256':POLICY_SHA,'source_sha256':_old.REFERENCE_SHA,'test_used':False})
    save(output/'items.json',doc);save(output/'items_seal.json',{'sha256':sha(output/'items.json'),'reference_records_sha256':sha(output/'reference_records.json'),'policy_sha256':POLICY_SHA})
    (output/'gate_snapshot.py').write_bytes(Path(__file__).read_bytes());(output/'derived_budget_gate.py').write_text(gate_source(budget)[0]);return output
