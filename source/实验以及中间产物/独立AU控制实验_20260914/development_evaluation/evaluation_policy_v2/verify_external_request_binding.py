#!/usr/bin/env python3
"""Explicit request-manifest binding supplement; frozen old manifests unchanged."""
import json
from openface_v2 import HERE,ROOT,POLICY_SHA,sha,save

def main():
    request=ROOT/'development_evaluation/dev_coverage_audit/dev19_manifest_2k.json';r=json.loads(request.read_text());assert not r['test_data_used']
    expected='90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28';assert sha(request)==expected
    cases={c['case_id']:c for c in r['cases']};assert len(cases)==19 and r['sampling_seeds']==[101]
    manifest=HERE/'external_driver/items.json';d=json.loads(manifest.read_text());assert d['policy_sha256']==POLICY_SHA
    contract=ROOT/'external_baselines/contracts/raw_dev19_v3_202609142040/image_driver_dev19.json';c=json.loads(contract.read_text());assert c['request_manifest_sha256']==expected
    checked=[]
    for item in d['items']:
        q=cases[item['case_id']];assert item['subject']==q['subject'] and item['source_labels_12']==q['source_au_vector_0_5'] and item['target_labels_12']==q['target_au_vector_0_5'] and item['group']==q['group']
        checked.append({'case_id':item['case_id'],'method':item['method'],'matching_request':True})
    output=HERE/'external_driver/request_binding_supplement.json'
    save(output,{'status':'pass','policy_sha256':POLICY_SHA,'test_used':False,'request_manifest_path':str(request),'request_manifest_sha256':expected,
          'source_items_sha256':sha(manifest),'source_native_contract_sha256':sha(contract),'checked_logical_records':len(checked),'records':checked,'script_sha256':sha(__file__),
          'reason':'items.json optional case_manifest_sha256 was null because common reference document uses another schema; existing hash-locked native contract already bound exact19 requests. This supplement exposes and verifies that binding explicitly, without changing any measurement or frozen file.'})
    print(json.dumps({'status':'pass','request_manifest_sha256':expected,'checked':len(checked)}))
if __name__=='__main__':main()
