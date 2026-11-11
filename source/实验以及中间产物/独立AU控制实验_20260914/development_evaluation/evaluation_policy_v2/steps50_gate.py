"""One candidate plus fresh baseline: dev20->50 sampling-step gate only."""
import hashlib
from pathlib import Path
import re
import ip_budget_gate as _budget
import step2000_gate as _old

HERE=_old.HERE;ROOT=_old.ROOT;POLICY_SHA=_old.POLICY_SHA
LABELS={'baseline':'baseline_steps50','bidirectional':'bidirectional_step200_steps50'}
REFERENCES=()
require=_old.require;read=_old.read;sha=_old.sha;save=_old.save;check_input_metadata=_old.check_input_metadata
GateError=_old.GateError
GENERATOR_SHA='fd23603a960e8294e0b4ce4976f0c69e1e7e05ece79064d95adedea62ffde252'
CANDIDATE_SHA='7fcc25b704fca8c0a4e46a171096df8fea940b40a246d4939193e64f533b4e30'
BUDGET_GATE_SHA='f2506601f16e926354b4139011cd66cf4347fd36e3715898297419e3b669ed35'

def scheduler50(actual,reference):
    require(actual.get('steps')==50 and reference.get('steps')==20,'Require explicit50 vs verified20 sampler steps')
    reference={**reference,'steps':50};require(_old.clean_scheduler(actual)==_old.clean_scheduler(reference),'Non-step scheduler setting changed')

def derive_source():
    require(sha(_budget.__file__)==BUDGET_GATE_SHA,'Reviewed budget gate changed');text,unused=_budget.gate_source(200)
    def change(a,b):
        nonlocal text
        require(text.count(a)==1,'Ambiguous step-only gate adaptation: '+a);text=text.replace(a,b)
    change("VARIANTS=('bidirectional','condition_reads_image','image_reads_condition','mlp')","VARIANTS=('bidirectional',)")
    change('LABELS='+repr(_budget.labels_for_budget(200)),'LABELS='+repr(LABELS))
    change("REFERENCES=('magicface','liveportrait','faceadapter','liveportrait_absolute')","REFERENCES=()")
    change("require(len(methods)==4,'Exactly four complete step2000 structures required')","require(len(methods)==1,'Exactly one selectedbidir200 checkpoint required')")
    change('len(set(variants))==4','len(set(variants))==1')
    change("count=4 if settings['skip_baseline'] else 5","count=1 if settings['skip_baseline'] else 2")
    change("settings.get('ip_scale')==1.0","settings.get('ip_scale')==0.9")
    change("reference_settings={**reference_settings,'ip_scale':1.0}","reference_settings={**reference_settings,'ip_scale':0.9,'steps':50}")
    change("r.get('ip_scale')==1.0","r.get('ip_scale')==0.9")
    change("return {'baseline':'baseline_ip10_b200',**method_map}","return {'baseline':'baseline_steps50',**method_map}")
    change("require(clean_scheduler(read(run/'runtime_scheduler.json'))==clean_scheduler(read(OLD_PREVIEW/'runtime_scheduler.json')),'Actual sampler changed')",
           "_scheduler50(read(run/'runtime_scheduler.json'),read(OLD_PREVIEW/'runtime_scheduler.json'))")
    change("require(len(settings.get('development_manifest_binding',[]))==4", "require(len(settings.get('development_manifest_binding',[]))==1")
    change("require(len(items)==95,'Need95 fresh IP1.0 baseline+model outputs')","require(len(items)==38,'Need38 fresh50-step baseline+candidate images')")
    change("'generated_count':95,'real_source_count':19,'logical_count':114","'generated_count':38,'real_source_count':19,'logical_count':57")
    change("'status':'validated_complete_ip10_step200'","'status':'validated_complete_steps50_bidir200'")
    text=text.replace('IP1.0','IP0.9/50steps')
    namespace={'__name__':'selected_candidate_steps50_gate','__file__':str(Path(__file__).resolve()),'_trace200':_budget._trace200,'_scheduler50':scheduler50}
    exec(compile(text,str(_old.__file__),'exec'),namespace)
    return text,namespace

def validate(run):
    text,namespace=derive_source();doc=namespace['validate'](run)
    require(len(doc['checkpoints'])==1 and doc['checkpoints'][0]['variant']=='bidirectional' and doc['checkpoints'][0]['checkpoint_sha256']==CANDIDATE_SHA,'Selected candidate changed')
    old_run=_old.OLD_PREVIEW;new_run=Path(doc['run'])
    require(sha(old_run/'generation_records.jsonl')=='23a0fe43595146f08c20355f0f132f825b409a60ad2cc275fa3949ae5d5abffd','Frozen20-step condition records changed')
    old_records=[__import__('json').loads(line) for line in (old_run/'generation_records.jsonl').read_text().splitlines()]
    new_records=[__import__('json').loads(line) for line in (new_run/'generation_records.jsonl').read_text().splitlines()]
    old={(r['case_id'],r['method'],r['seed']):r for r in old_records}
    for r in new_records:
        key=(r['case_id'],r['method'],r['seed']);require(key in old and r['condition_hashes']==old[key]['condition_hashes'],'AU/IP condition tensor hashes differ from20-step reference')
    doc.update(sampling_steps=50,reference_sampling_steps=20,candidate_sha256=CANDIDATE_SHA,input_scale=.9,
        only_sampling_steps_changed=True,condition_hashes_equal20=True,reference20_generation_records_sha256=sha(old_run/'generation_records.jsonl'),
        steps_binding='settings.steps=runtime_scheduler.steps=50; pinned generator passes args.steps to everypipe call; per-row steps field is not fabricated',
        gate_source_sha256=hashlib.sha256(text.encode()).hexdigest())
    return doc

def prepare(run,group):
    require(isinstance(group,str) and re.fullmatch('steps50_[a-z0-9_]+',group),'New steps50_ group required');out=HERE/group;require(not out.exists(),'No overwrite')
    doc=validate(run);refs=doc.pop('reference_records');require(len(refs)==38 and all(r['method'] in ('source_real','target_real') for r in refs),'Only realreferences permitted')
    require(len(doc['items'])==57,'Source19+new38 inputs required')
    out.mkdir();save(out/'reference_records.json',{'records':refs,'policy_sha256':POLICY_SHA,'test_used':False})
    save(out/'items.json',doc);save(out/'items_seal.json',{'sha256':sha(out/'items.json'),'reference_records_sha256':sha(out/'reference_records.json'),'policy_sha256':POLICY_SHA})
    (out/'gate_snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'derived_gate_snapshot.py').write_text(derive_source()[0]);return out
