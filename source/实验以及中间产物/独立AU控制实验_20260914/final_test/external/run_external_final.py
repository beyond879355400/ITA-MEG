#!/usr/bin/env python3
"""Frozen final-test external inference. Defaults to no work without complete control files."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from contracts import HERE,EXP,THESIS,require,resolve,executable_path,sha,signature,project,update_index,failed_run_index
from native_runtime import validate_runtime,verify_core_files,verify_lease,verify_job_inputs,execute_job
sys.path.insert(0,str(HERE.parent))
from protocol_gate import validate_control,verify_runtime_source

SOURCE_ROLES={'protocol_gate':HERE.parent/'protocol_gate.py','external_entry':Path(__file__).resolve(),
              'external_contracts':HERE/'contracts.py','external_native':HERE/'native_runtime.py'}


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def save_index(path,value):
    temporary=Path(str(path)+'.tmp')
    with temporary.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    temporary.replace(path)


def bound_control(protocol,protocol_sha,raw,raw_sha,method):
    # Header refusal precedes any raw-manifest or asset access.
    require(Path(protocol).is_file(),'Final protocol absent; test remains locked')
    doc=json.loads(Path(protocol).read_text())
    require(doc.get('status')=='frozen_final_test' and doc.get('extraction_authorized') is True,'Final protocol is not frozen/authorized')
    require(sha(protocol)==protocol_sha,'Explicit protocol SHA mismatch')
    control=validate_control(protocol,raw,raw_sha)
    for role,path in SOURCE_ROLES.items():verify_runtime_source(control,role,path)
    require(method in control['models'],'External method not in the frozen complete model set')
    settings=control['models'][method]['settings']
    projection=project(method,settings,control['raw_manifest'],protocol_sha,raw_sha)
    require(projection['case_ids']==control['case_ids'] and projection['sampling_seeds']==control['sampling_seeds'],'Final request grid changed')
    lease=validate_runtime(settings,control['models'][method]['artifacts'])
    return control,settings,projection,lease


def worker_binding(context):
    # Control JSON and recorded asset-stat lease only; no target-image read for MF.
    protocol=Path(context['protocol_path']);raw=Path(context['raw_manifest_path']);sampling=Path(context['sampling_path'])
    doc=json.loads(protocol.read_text())
    require(doc.get('status')=='frozen_final_test' and doc.get('extraction_authorized') is True,'Worker protocol is not frozen')
    require(sha(protocol)==context['protocol_sha256'] and sha(raw)==context['raw_manifest_sha256'],'Worker protocol/raw binding changed')
    require(sha(sampling)==context['sampling_sha256'],'Worker sampling settings changed')
    settings=json.loads(sampling.read_text())['method_settings'][context['method']]
    require(settings==context['settings'],'Worker parameters differ from frozen method')
    for role,path in SOURCE_ROLES.items():
        ref=doc.get('execution_sources',{}).get(role,{})
        require(resolve(ref.get('path'))==path and ref.get('sha256')==sha(path),'Worker runtime source not frozen')
    expected=project(context['method'],settings,json.loads(raw.read_text()),context['protocol_sha256'],context['raw_manifest_sha256'])
    selected=[row for row in expected['cases'] if row['execution_id']==context['job']['execution_id']]
    require(selected==[context['job']],'Worker request was not in the complete frozen projection')
    verify_core_files();verify_lease(context['asset_lease'])


def execute_worker(context_path,context_sha,output):
    require(sha(context_path)==context_sha,'Worker context SHA mismatch')
    context=json.loads(Path(context_path).read_text());out=Path(output).resolve()
    require(out.is_relative_to(HERE.parent/'external_runs'),'Final worker output outside isolated final-test directory')
    worker_binding(context)
    require(Path(sys.executable).absolute()==executable_path(context['settings']['runtime']['python_executable']),'Worker must use the frozen interpreter path without resolving its venv symlink')
    result=execute_job(context,out)
    print(json.dumps({'execution_id':result['execution_id'],'status':result['status'],'execution_started':result['generation_started']},ensure_ascii=False),flush=True)
    return 0 if result['status']=='complete' else 1


def run(args):
    control,settings,projection,lease=bound_control(args.protocol,args.protocol_sha256,args.raw_manifest,args.raw_manifest_sha256,args.method)
    report={'status':'validated_frozen_external_plan','protocol_sha256':control['protocol_sha256'],'raw_manifest_sha256':control['raw_manifest_sha256'],
            'method':args.method,'implementation':settings['implementation'],'case_count':32,'logical_count':96,
            'physical_executions_planned':len(projection['cases']),'seed_execution':settings['seed_execution'],
            'test_pixels_opened':False,'models_called':False,'parameters':settings['native_parameters']}
    if args.phase=='validate':print(json.dumps(report,ensure_ascii=False,indent=2));return 0
    require(args.output is not None,'Explicit new final output directory required')
    out=args.output.resolve();require(out.is_relative_to(HERE.parent/'external_runs'),'Output must stay below final_test/external_runs')
    require(not out.exists(),'Output already exists; choose a new run (no overwrite or silent retry)')
    out.mkdir(parents=True)
    config={'protocol_path':control['protocol_path'],'protocol_sha256':control['protocol_sha256'],
            'raw_manifest_path':control['raw_manifest_path'],'raw_manifest_sha256':control['raw_manifest_sha256'],
            'sampling_path':control['sampling_path'],'sampling_sha256':control['sampling_sha256'],
            'method':args.method,'settings':settings,'asset_lease':lease,
            'source_shas':{role:sha(path) for role,path in SOURCE_ROLES.items()},'test_data_used':True}
    save_new(out/'configuration.json',config);save_new(out/'projection.json',projection);save_new(out/'validation.json',report)
    for role,path in SOURCE_ROLES.items():
        (out/'snapshots').mkdir(exist_ok=True);(out/'snapshots'/path.name).write_bytes(path.read_bytes())
    receipts={};index_path=out/'generation_index.json';save_index(index_path,update_index(projection,receipts,control['raw_manifest_path']))
    lock=out/'.run.lock'
    with lock.open('x') as f:json.dump({'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat()},f)
    try:
        for number,job in enumerate(projection['cases'],1):
            verify_lease(lease)
            # Source/driver content hashes are checked by the worker before model calls.
            # This keeps a missing/corrupt per-case input as an explicit failed receipt.
            context={**copy.deepcopy(config),'job':job};context_path=out/'contexts'/(job['case_id']+'.json');save_new(context_path,context)
            job_out=out/'executions'/job['case_id'];logs=out/'logs';logs.mkdir(exist_ok=True)
            command=[str(executable_path(settings['runtime']['python_executable'])),'-B',str(Path(__file__).resolve()),'worker',
                     '--context',str(context_path),'--context-sha256',sha(context_path),'--output',str(job_out)]
            with (logs/(job['case_id']+'.log')).open('x') as logfile:
                result=subprocess.run(command,cwd=THESIS,stdout=logfile,stderr=subprocess.STDOUT,check=False)
            path=job_out/'worker_result.json'
            if path.is_file():receipt=json.loads(path.read_text())
            else:
                receipt={'execution_id':job['execution_id'],'status':'failed','output':None,'generation_started':(job_out/'generation_started.json').exists(),
                         'error':{'error_type':'WorkerExitedWithoutReceipt','returncode':result.returncode}}
            require(receipt['execution_id']==job['execution_id'],'Worker execution binding mismatch')
            receipt['worker_command']=command;receipt['worker_context_sha256']=sha(context_path);receipt['process_returncode']=result.returncode
            receipts[job['execution_id']]=receipt
            save_new(out/'receipts'/(job['case_id']+'.json'),receipt)
            save_index(index_path,update_index(projection,receipts,control['raw_manifest_path']))
            print(json.dumps({'method':args.method,'done':number,'planned':len(projection['cases']),'case_id':job['request_case_id'],'seed':job['seed'],'status':receipt['status']},ensure_ascii=False),flush=True)
        # Closing check re-hashes actual model assets; source/driver bytes checked per job.
        for asset in lease['assets']:require(sha(asset['path'])==asset['sha256'],'Model asset changed during final run')
    except BaseException as error:
        save_new(out/'run_failure.json',{'error_type':type(error).__name__,'error':str(error),'test_data_used':True})
        save_index(index_path,failed_run_index(projection,receipts,control['raw_manifest_path'],error))
        raise
    finally:lock.unlink(missing_ok=True)
    index=update_index(projection,receipts,control['raw_manifest_path'],terminal=True);save_index(index_path,index)
    print(json.dumps({k:index[k] for k in ('status','physical_executions','unique_output_images','successful_logical_records','failed_logical_records')},ensure_ascii=False),flush=True)
    return 0 if index['status']=='complete' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('validate','execute','worker'))
    parser.add_argument('--protocol',type=Path);parser.add_argument('--protocol-sha256');parser.add_argument('--raw-manifest',type=Path);parser.add_argument('--raw-manifest-sha256')
    parser.add_argument('--method');parser.add_argument('--output',type=Path);parser.add_argument('--context',type=Path);parser.add_argument('--context-sha256')
    args=parser.parse_args()
    if args.phase=='worker':
        require(args.context is not None and args.context_sha256 and args.output is not None,'Bound worker context/output required')
        return execute_worker(args.context,args.context_sha256,args.output)
    require(all(v is not None for v in (args.protocol,args.protocol_sha256,args.raw_manifest,args.raw_manifest_sha256,args.method)),'Explicit frozen protocol/raw SHA and method required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
