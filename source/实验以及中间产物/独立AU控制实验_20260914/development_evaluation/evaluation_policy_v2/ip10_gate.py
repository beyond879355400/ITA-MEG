"""Pinned step2000 gate adaptation for one explicit IP1.0 development setting."""
from pathlib import Path
import re
import step2000_gate as _old

CORE_SHA='1af22a95e6f2bec4a1403494523763b33e2ab61112fec37eb9aa915e9be1eef1'
GENERATOR_SHA='fd23603a960e8294e0b4ce4976f0c69e1e7e05ece79064d95adedea62ffde252'

def derive_gate():
    core=Path(_old.__file__).resolve();_old.require(_old.sha(core)==CORE_SHA,'Frozen0.9 gate changed');text=core.read_text()
    def change(a,b):
        nonlocal text
        _old.require(text.count(a)==1,'Ambiguous IP gate adaptation: '+a);text=text.replace(a,b)
    change("LABELS={v:v+'_step2000' for v in VARIANTS}","LABELS={'baseline':'baseline_ip10',**{v:v+'_step2000_ip10' for v in VARIANTS}}")
    change("REFERENCES=('baseline','magicface','liveportrait','faceadapter','liveportrait_absolute')","REFERENCES=('magicface','liveportrait','faceadapter','liveportrait_absolute')")
    change("GENERATOR_SHA='fbba652deaf7f1dbdb3972799b00a010e5782eaedddab40f85d4154941e9a83f'",f"GENERATOR_SHA='{GENERATOR_SHA}'")
    change('    methods=settings.get(\'methods\',[]);require',"    require(settings.get('skip_baseline') is False,'IP1.0 requires a fresh same-run baseline')\n    methods=settings.get('methods',[]);require")
    change('    for key in (\'cache_sha256\',',"    require(settings.get('ip_scale')==1.0 and type(settings.get('ip_scale')) in (int,float),'Explicit IP1.0 required')\n    require(settings.get('setting_status')=='development sensitivity only; not the final parameter choice','Sensitivity scope missing')\n    reference_settings={**reference_settings,'ip_scale':1.0}\n    for key in ('cache_sha256',")
    change('def record_gate(records,settings,cases,source_rows,prompts):',"def record_gate(records,settings,cases,source_rows,prompts):\n    require(all(type(r.get('ip_scale')) in (int,float) and r.get('ip_scale')==1.0 for r in records),'Every generated record must state IP1.0')")
    change('    return method_map',"    return {'baseline':'baseline_ip10',**method_map}")
    change('method_map=record_gate(records,settings,cases,source,prompts)',"method_map=record_gate(records,settings,cases,source,prompts)\n    base={r['case_id']:r for r in records if r['method']=='baseline'}\n    require(len(base)==19,'All19 fresh IP1.0 baselines required')")
    change("        if g['method']=='baseline':require(g['sha256']==base[cid]['sha256'],'Regenerated baseline differs from sealed reference');continue", "        # The baseline is this run's IP1.0 output, never the0.9 reference.")
    change("require(len(items)==76,'Need76 new model outputs')","require(len(items)==95,'Need95 fresh IP1.0 baseline+model outputs')")
    change("'generated_count':76,'real_source_count':19,'logical_count':95","'generated_count':95,'real_source_count':19,'logical_count':114")
    change("'status':'validated_complete_step2000'","'status':'validated_complete_ip10_step2000'")
    return core,text

_core,_text=derive_gate()
_namespace={'__name__':'ip10_pinned_gate','__file__':str(Path(__file__).resolve())}
exec(compile(_text,str(_core),'exec'),_namespace)
for _name in ('HERE','ROOT','POLICY_SHA','LABELS','REFERENCES','VARIANTS','GateError','require','read','sha','save','check_input_metadata','input_metadata','document_gate','record_gate'):
    globals()[_name]=_namespace[_name]

def checkpoint_match(new,old):
    left={r['variant']:r['checkpoint_sha256'] for r in new['checkpoints']};right={r['variant']:r['checkpoint_sha256'] for r in old['checkpoints']}
    require(set(left)==set(VARIANTS) and left==right,'IP1.0 must use exactly the verified0.9 four checkpoints')

def validate(run,reference_group):
    require(isinstance(reference_group,str) and re.fullmatch('step2000_[a-z0-9_]+',reference_group) is not None,'Explicit verified0.9 reference group required')
    folder=HERE/reference_group;old=read(folder/'items.json');seal=read(folder/'items_seal.json')
    require(sha(folder/'items.json')==seal['sha256'] and seal['policy_sha256']==POLICY_SHA,'0.9 reference input seal changed')
    report=read(folder/'comparison/results.json');verification=read(folder/'comparison/independent_verification.json')
    require(report['status']=='complete' and verification['status']=='pass' and verification['input_sha256']==sha(folder/'comparison/results.json'),'0.9 reference evaluation not complete/independently verified')
    actual=_old.validate(old['run']);actual.pop('reference_records');require(actual==old,'0.9 generation reference changed')
    new=_namespace['validate'](run);checkpoint_match(new,old)
    new.update(reference_ip09_group=reference_group,reference_ip09_input_sha256=sha(folder/'items.json'),reference_ip09_comparison_sha256=sha(folder/'comparison/results.json'),
               input_scale=1.0,baseline_scope='fresh baseline from this same IP1.0 run; no0.9 baseline measurements reused',gate_core_sha256=CORE_SHA)
    return new

def prepare(run,group,reference_group):
    require(re.fullmatch('ip10_[a-z0-9_]+',group) is not None,'New ip10_ group required');output=HERE/group;require(not output.exists(),'No overwrite')
    doc=validate(run,reference_group);references=doc.pop('reference_records');require(len(references)==114 and all(r['method']!='baseline' for r in references),'Old baseline cannot enter IP1.0 references')
    output.mkdir();save(output/'reference_records.json',{'records':references,'policy_sha256':POLICY_SHA,'source_sha256':_namespace['REFERENCE_SHA'],'test_used':False})
    save(output/'items.json',doc);save(output/'items_seal.json',{'sha256':sha(output/'items.json'),'reference_records_sha256':sha(output/'reference_records.json'),'policy_sha256':POLICY_SHA})
    (output/'gate_snapshot.py').write_bytes(Path(__file__).read_bytes());(output/'derived_gate_snapshot.py').write_text(_text);return output
