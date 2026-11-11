"""Additional conservative first-edited environment bridge; v1 stays immutable."""
import argparse
import json
from pathlib import Path
from protocol import *
import evaluate_new as entry


def prepare(protocol_path,protocol_sha,out):
    load_protocol(protocol_path,protocol_sha);c,e,cases,old,refs=context()
    selected=next(r for r in cases if r['source_au_vector_0_5']!=r['target_au_vector_0_5'])
    cid=selected['case_id']
    items=[r for r in old['items']if r['case_id']==cid and
           (r['method']in ['source_real','target_real']or r['method']==CHOSEN and r['seed']==101)]
    need(len(items)==3,'Expected first edited source/target/selected output')
    for r in items:e.inspect_image_item(r,{'path':r['path'],'sha256':r['sha256']},
        'real_source'if r['method']=='source_real'else 'real_target'if r['method']=='target_real'else 'generated')
    out=entry.new_directory(out)
    doc={'kind':'recent_external_bridge_inputs_v1','status':'prepared','data_role':ROLE,'test_used':True,
         'protocol':ref(protocol_path),'protocol_sha256':protocol_sha,'measurement_policy_sha256':POLICY_SHA,
         'generation_indexes':[],'old_metrics':ref(pin('old_metrics')),'old_items':ref(pin('old_items')),
         'references':ref(pin('references')),'case_metadata':{cid:old['case_metadata'][cid]},'items':items,
         'planned':[{'method':CHOSEN,'subject':selected['subject'],'case_id':cid,'seed':101}],
         'bridge_selection':'first edited manifest case/seed101, root-added conservative environment acceptance; no score selection',
         'additional_bridge_source':ref(__file__)}
    save(out/'items.json',doc);save(out/'items_seal.json',{'sha256':sha(out/'items.json'),'policy_sha256':POLICY_SHA,'protocol_sha256':protocol_sha})
    return out


def combine(protocol_path,protocol_sha,first,second,out):
    load_protocol(protocol_path,protocol_sha);_,_,cases,_,_=context()
    expected=[cases[0]['case_id'],next(r['case_id']for r in cases if r['source_au_vector_0_5']!=r['target_au_vector_0_5'])]
    docs=[read(verified(ref(p)))for p in [first,second]]
    for d,cid in zip(docs,expected):
        need(d['fixed_case_id']==cid and d['selected_seed']==101,'Bridge case changed')
        need(d['protocol_sha256']==protocol_sha and d['measurement_policy_sha256']==POLICY_SHA,'Bridge policy changed')
    ok=all(d['status']=='pass'and not d['mismatches']for d in docs)
    need(Path(out).resolve().is_relative_to(HERE),'Receipt outside scope')
    save(out,{'status':'pass'if ok else 'failed','protocol_sha256':protocol_sha,'measurement_policy_sha256':POLICY_SHA,
              'fixed_case_ids':expected,'selected_seed':101,'both_bridges_required':True,'bridge_receipts':[ref(first),ref(second)],
              'mixing_new_runtime_allowed':ok,'source':ref(__file__)})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','combine']);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha',required=True);p.add_argument('--approval',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first',type=Path);p.add_argument('--second',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    load_protocol(a.protocol,a.protocol_sha)
    if not a.execute:print(json.dumps({'status':'metadata_only','models_loaded':False,'extra_bridge':'first edited fixed request seed101'}));return
    gate(a.approval,a.protocol_sha,'bridge-extra-'+a.phase)
    result=prepare(a.protocol,a.protocol_sha,a.output)if a.phase=='prepare'else combine(a.protocol,a.protocol_sha,a.first,a.second,a.output)
    print(json.dumps({'output':str(result)}))


if __name__=='__main__':main()
