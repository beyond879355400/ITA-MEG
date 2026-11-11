"""Frozen final-test CPU evaluators. No images/models are read on import."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from evaluation_contract import *

CORES={'openface':('openface_v2.py','5446237ff2fba672bd06ec62820bf671d9deb4c776d83a28d18060a581b9f908'),
       'facenet':('run_facenet_single_directions.py','5d8e1916bb5bcc9afc307980438b5f2f3fb0610f5dde12141f90e85efa7ebc2d'),
       'buffalo':('append_buffalo.py','96daabea1e618a993de22ef684a4cbdcc6822911191a424fc457ea19012af88a')}

def prepare_job(protocol,raw,raw_sha,references,reference_sha,index_refs,output):
    control=load_control(protocol,raw,raw_sha,'final_evaluation_measurements',__file__)
    rp=ref_path({'path':str(references),'sha256':reference_sha});refs=read(rp)
    require(refs.get('kind')=='final_test_evaluation_references' and refs.get('status')=='complete' and refs.get('protocol_sha256')==control['protocol_sha256'] and refs.get('measurement_policy_sha256')==POLICY_SHA,'Real reference binding mismatch')
    require(refs.get('raw_manifest',{}).get('sha256')==raw_sha and [r['case_id'] for r in refs['cases']]==control['protocol']['case_ids'],'Reference case/raw binding changed')
    index_paths=[ref_path(r) for r in index_refs];docs=[read(p) for p in index_paths]
    plan,normalized,accounting=normalize_outputs(control['protocol'],control['sampling'],control['cases'],docs,control['protocol_sha256'],raw_sha)
    source_cache_path=ref_path(refs['source_cache_manifest']);source_cache=read(source_cache_path)
    source_bound=source_cache_index_binding(control['sampling'],refs,source_cache,docs)
    output=Path(output).resolve();require(output.is_relative_to(HERE) and not output.exists(),'New final_test evaluation directory required')
    output.mkdir(parents=True);items=[];availability=[]
    def add_image(meta,image_ref,role):
        item=clean_item({**meta,'path':image_ref.get('path'),'sha256':image_ref.get('sha256')});check_item(item)
        key=[item.get('method'),item.get('subject'),item.get('case_id'),item.get('seed')]
        try:
            path=ref_path(image_ref)
            from PIL import Image
            with Image.open(path) as image:image.load();require(image.mode=='RGB','Native measurement input must be decodableRGB; do not transform corrupt images')
            item['path']=str(path);items.append(item);availability.append({'key':key,'status':'available','role':role,'image':image_ref})
        except Exception as error:availability.append({'key':key,'status':'unavailable','role':role,'image':image_ref,'error_type':type(error).__name__,'error':str(error)})
    for ref_case in refs['cases']:
        meta=control['cases'][ref_case['case_id']]
        require(ref_case['subject']==meta['subject'] and ref_case['source_labels_12']==meta['source_labels_12'] and ref_case['target_labels_12']==meta['target_labels_12'],'Reference labels changed')
        for role in ('source','target'):
            ref=ref_case[role+'_reference'];row={**meta,'method':role+'_real','seed':0}
            if ref['status']=='complete':add_image(row,ref['image'],'real_'+role)
            else:availability.append({'key':[role+'_real',meta['subject'],meta['case_id'],0],'status':'unavailable','role':'real_'+role,'reference_error':ref})
    for p in plan:
        key=row_key(p);r=normalized.get(key)
        if r is None:availability.append({'key':list(key),'status':'missing_output','role':'generated'});continue
        if r['status']!='complete':availability.append({'key':list(key),'status':r['status'],'role':'generated','generation_record':r});continue
        add_image({**control['cases'][p['case_id']],**p,'split':'test','execution_id':r['execution_id'],'execution_started':r['execution_started'],'alias_of':r.get('alias_of')},r['output'],'generated')
    unique={(r['case_id'],r['method'],r['seed']) for r in items};require(len(unique)==len(items),'Duplicate measurement input')
    doc={'kind':'final_test_measurement_inputs','status':'prepared','data_role':'final_test','test_used':True,'protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,'items':items,
      'planned':plan,'normalized_generation_records':list(normalized.values()),'availability':availability,'execution_accounting':accounting,'case_metadata':control['cases'],
      'protocol_path':control['protocol_path'],'raw_manifest':{'path':control['raw_manifest_path'],'sha256':raw_sha},'reference_manifest':{'path':str(rp),'sha256':reference_sha},
      'generation_indexes':index_refs,'source_cache_bound_methods':source_bound,'evaluation_policy':{'path':control['evaluation_path'],'sha256':control['evaluation_sha256']},'script_sha256':sha(__file__)}
    save(output/'items.json',doc);save(output/'items_seal.json',{'sha256':sha(output/'items.json'),'policy_sha256':POLICY_SHA,'measurement_policy_sha256':POLICY_SHA,'protocol_sha256':control['protocol_sha256']})
    (output/'prepare_snapshot.py').write_bytes(Path(__file__).read_bytes());return output

def job_control(job,role='final_evaluation_measurements',path=None):
    job=Path(job).resolve();require(job.is_relative_to(HERE),'Job outside final_test');doc=read(job/'items.json');seal=read(job/'items_seal.json')
    control=load_control(doc['protocol_path'],doc['raw_manifest']['path'],doc['raw_manifest']['sha256'],role,path or __file__)
    require(sha(job/'items.json')==seal['sha256'] and doc['protocol_sha256']==seal['protocol_sha256']==control['protocol_sha256'],'Prepared job changed')
    require(doc['evaluation_policy']['sha256']==control['evaluation_sha256'],'Frozen metrics/family changed')
    ref_path(doc['reference_manifest'])
    for r in doc['generation_indexes']:ref_path(r)
    for item in doc['items']:check_item(item)
    return job,doc,control

def derive_identity(kind,job,input_sha,count,source_count):
    require(kind in ('facenet','buffalo') and type(count) is int and count>=0 and 0<=source_count<=32,'Invalid identity adapter count')
    require(re.fullmatch('[0-9a-f]{64}',str(input_sha)) is not None,'Invalid frozen measurement-input SHA')
    core=V2/CORES[kind][0];require(sha(core)==CORES[kind][1],'Frozen identity algorithm changed');text=core.read_text()
    if kind=='facenet':
        changes={"root = HERE / 'single_directions'":f'root = Path({str(job)!r})',
          "INPUT_SHA = 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'":f"INPUT_SHA = '{input_sha}'",
          "not document['test_used']":"document['data_role']=='final_test'"}
        changes.update({"(row['case_id'], row['method'])":"(row['case_id'], row['method'], row['seed'])",
          "source_sha = sources[row['case_id']]":"source_sha = sources.get(row['case_id'])",
          "HERE / 'shared_cache/matplotlib'":"root / 'cache/matplotlib'","HERE / 'shared_cache/xdg'":"root / 'cache/xdg'"})
    else:
        changes={"manifest=HERE/'single_directions/items.json'":f"manifest=Path({str(job/'items.json')!r})",
          "HERE/'single_directions/items_seal.json'":f"Path({str(job/'items_seal.json')!r})","out=HERE/'single_directions/buffalo'":f"out=Path({str(job/'buffalo')!r})",
          "sources[r['case_id']]":"sources.get(r['case_id'])","not document['test_used']":"document['data_role']=='final_test'"}
    for old,new in changes.items():require(text.count(old)==1,'Ambiguous identity-schema adaptation: '+old);text=text.replace(old,new)
    class Adapter(ast.NodeTransformer):
        def visit_Dict(self,node):
            self.generic_visit(node)
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='test_used' and isinstance(node.values[i],ast.Constant) and node.values[i].value is False:node.values[i]=ast.Constant(value=True)
            return node
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==57:node.value=count
            elif kind=='facenet' and type(node.value) is int and node.value==19:node.value=source_count
            elif node.value=='dev':node.value='test'
            return node
    tree=ast.fix_missing_locations(Adapter().visit(ast.parse(text)));return core,ast.unparse(tree)+'\n'

def measure_identity(job,kind):
    job,doc,control=job_control(job);out=job/kind;require(not out.exists(),'Never overwrite measurements')
    items=doc['items']
    if not items:
        out.mkdir();save(out/'results.json',{'policy_sha256':POLICY_SHA,'protocol_sha256':control['protocol_sha256'],'records':[],'test_used':True,'summary':{'status':'complete'},'no_models_called':True})
        save(out/'final_binding.json',{'protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,'no_models_called':True});return
    # Final gates/hash availability are rechecked; actual decode failures stay in per-image records.
    for item in items:require(sha(item['path'])==item['sha256'],'Measurement image changed after plan seal')
    source_count=len({r['case_id'] for r in items if r['method']=='source_real'});core,text=derive_identity(kind,job,sha(job/'items.json'),len(items),source_count)
    sys.path.insert(0,str(V2));namespace={'__name__':'frozen_final_identity_adapter','__file__':str(Path(__file__).resolve())}
    exec(compile(text,str(core),'exec'),namespace);namespace['main']()
    (out/'core_snapshot.py').write_bytes(core.read_bytes());(out/'derived_snapshot.py').write_text(text)
    save(out/'final_binding.json',{'protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,'core_sha256':sha(core),'derived_sha256':sha(out/'derived_snapshot.py'),
          'adaptations':'test role/scope metadata, input root/hash/count, seed-key uniqueness, missing-source NA, final_test cache paths only; detector/model/normalization unchanged'})

def measure_openface(job):
    job,doc,control=job_control(job);out=job/'openface';require(not out.exists(),'Never overwrite measurements');out.mkdir()
    core=V2/CORES['openface'][0];require(sha(core)==CORES['openface'][1],'OpenFace helper changed')
    spec=importlib.util.spec_from_file_location('frozen_final_openface',core);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    policy=read(V2/'policy.json');seal=read(module.TOOLS/'tool_seal.json');require(sha(module.TOOLS/'tool_seal.json')==policy['openface']['tool_seal_sha256'],'OpenFace toolseal changed')
    for r in [seal['binary'],seal['launcher'],*seal['model_files']]:require(sha(module.TOOLS/r['path'])==r['sha256'],'OpenFace asset changed')
    unique={}
    for item in doc['items']:
        require(sha(item['path'])==item['sha256'],'Image changed after seal')
        unique.setdefault(item['sha256'],clean_item({**item,'sample_id':'sha_'+item['sha256']}))
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');physical=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(module.measure,item,out) for item in unique.values()]):physical.append(f.result())
    bysha={r['sha256']:r for r in physical};records=[]
    for item in doc['items']:
        parsed=bysha[item['sha256']];records.append({**parsed,**clean_item(item),'measurement_source_path':parsed['path']})
    save(out/'results.json',{'status':'complete','test_used':True,'protocol_sha256':control['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,'records':records,'unique_measurements':physical,
      'planned_available_logical_images':len(doc['items']),'unique_images_measured':len(unique),'core_sha256':sha(core),'input_sha256':sha(job/'items.json')})
    (out/'core_snapshot.py').write_bytes(core.read_bytes())

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','openface','facenet','buffalo'));p.add_argument('--job',type=Path);p.add_argument('--protocol',type=Path)
    p.add_argument('--raw-manifest',type=Path);p.add_argument('--raw-sha');p.add_argument('--references',type=Path);p.add_argument('--references-sha');p.add_argument('--generation-index',type=Path,action='append',default=[]);p.add_argument('--generation-sha',action='append',default=[]);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.phase=='prepare':
        require(len(a.generation_index)==len(a.generation_sha)>0,'Explicit completed generation indexes with their actualSHA required')
        refs=[{'path':str(p),'sha256':s} for p,s in zip(a.generation_index,a.generation_sha)]
        print(prepare_job(a.protocol,a.raw_manifest,a.raw_sha,a.references,a.references_sha,refs,a.output))
    elif a.phase=='openface':measure_openface(a.job)
    else:measure_identity(a.job,a.phase)
if __name__=='__main__':main()
