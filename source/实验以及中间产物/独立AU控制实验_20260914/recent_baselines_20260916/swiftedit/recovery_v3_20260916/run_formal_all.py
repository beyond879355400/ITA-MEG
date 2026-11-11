"""Serial3seed supervisor, append-only terminal rows and complete96row index."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import run_formal as formal

HERE=formal.HERE;SWIFT=formal.SWIFT;core=formal.core


def normalized(case,seed,run_name,receipt=None,started=False,reason=None):
    out=None
    if receipt:
        file=next(x for x in receipt['files'] if x['role']=='target_edit')
        core.need(Path(file['path']).is_file() and core.sha(file['path'])==file['sha256'],'Primary output SHA mismatch')
        out={'path':file['path'],'sha256':file['sha256']}
    return {'case_id':case['case_id'],'subject':case['subject'],'split':'postfreeze_case_validation','seed':seed,
            'source_sha256':case['source_sha256'],'source_au_vector_0_5':case['source_au_vector_0_5'],
            'target_au_vector_0_5':case['target_au_vector_0_5'],'target_image_used':False,'target_prompt':case['prompt'],
            'primary_output_role':'target_edit','status':'complete' if receipt else 'failed' if started else 'not_started',
            'execution_id':run_name+'__'+case['case_id'],'execution_started':bool(receipt) or started,'output':out,
            **({'native_receipt':receipt} if receipt else {'failure_reason':reason or 'upstream technical stop'})}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--protocol-sha256',required=True)
    args=p.parse_args();core.need(len(args.protocol_sha256)==64,'Frozen evaluation protocol SHA required')
    grant=core.read(HERE/'approvals.json').get('formal_batch_go')
    core.need(grant and grant['granted_by']=='root','Formal computeGO absent')
    core.need(grant['protocol_sha256']==args.protocol_sha256 and grant['case_manifest_sha256']==formal.MANIFEST_SHA and grant['sampling_seeds']==formal.SEEDS,'Formal freeze differs from root approval')
    core.need(core.sha(formal.NEW/'evaluation_20260916_v1/protocol_v1.json')==args.protocol_sha256,'Frozen evaluation file SHA mismatch')
    cases=formal.stage_manifest()['cases'];out=Path(args.output).resolve();out.mkdir(exist_ok=False)
    group=out.name;terminal=[];halt=False;started_at=datetime.now(timezone.utc).isoformat()
    core.save_new(out/'configuration.json',{'group':group,'started_at':started_at,'case_manifest_sha256':formal.MANIFEST_SHA,
        'protocol_sha256':args.protocol_sha256,'sampling_seeds':formal.SEEDS,'case_ids':[r['case_id'] for r in cases],
        'planned':96,'supervisor_sha256':core.sha(Path(__file__)),'formal_wrapper_sha256':core.sha(HERE/'run_formal.py'),
        'runtime_lock_sha256':core.sha(HERE/'requirements.lock'),'source_status':formal.core.STATUS})
    with (out/'terminal_index.jsonl').open('x',buffering=1) as live:
        def emit(row):
            terminal.append(row);live.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');live.flush();os.fsync(live.fileno())
            print(json.dumps({'terminal':len(terminal),'case_id':row['case_id'],'seed':row['seed'],'status':row['status']}),flush=True)
        for seed in formal.SEEDS:
            name=group+'_seed'+str(seed);run=SWIFT/'runs'/name;seen=set()
            if halt:
                for case in cases:emit(normalized(case,seed,name,reason='prior seed technical failure; not silently retried'))
                continue
            core.need(shutil.disk_usage(SWIFT).free>=core.RESERVE,'Disk reserve before seed process')
            command=[sys.executable,'-B','-u',str(HERE/'run_formal.py'),'--seed',str(seed),'--execute','--run-name',name]
            with (out/('seed'+str(seed)+'.log')).open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                interrupted=False
                while process.poll() is None:
                    if shutil.disk_usage(SWIFT).free<core.RESERVE and not interrupted:
                        process.send_signal(signal.SIGINT);interrupted=True
                    for case in cases:
                        if case['case_id'] in seen:continue
                        receipt=run/(case['case_id']+'_receipt.json')
                        if receipt.exists():
                            try:row=json.loads(receipt.read_text())
                            except (json.JSONDecodeError,OSError):continue
                            emit(normalized(case,seed,name,row));seen.add(case['case_id'])
                    time.sleep(2)
                code=process.wait()
            status=core.read(run/'result.json') if (run/'result.json').exists() else {'status':'failed','error':'No terminal result; see seed log'}
            starts=set()
            if (run/'events.jsonl').exists():
                for line in (run/'events.jsonl').read_text().splitlines():
                    try:event=json.loads(line)
                    except ValueError:continue
                    if event.get('stage')=='case_start':starts.add(event['case_id'])
            for case in cases:
                if case['case_id'] in seen:continue
                path=run/(case['case_id']+'_receipt.json')
                if path.exists():emit(normalized(case,seed,name,core.read(path)))
                else:emit(normalized(case,seed,name,started=case['case_id'] in starts,reason=status.get('error','seed process stopped')))
            halt=bool(code or status['status']!='complete')
    core.need(len(terminal)==96 and len({(r['case_id'],r['seed']) for r in terminal})==96,'Incomplete terminal matrix')
    index={'kind':'recent_baseline_generation_index_v1','method':'swiftedit',
           'status':'complete' if all(r['status']=='complete' for r in terminal) else 'completed_with_failures',
           'protocol_sha256':args.protocol_sha256,'case_manifest_sha256':formal.MANIFEST_SHA,
           'case_ids':[r['case_id'] for r in cases],'input_group':'source_text','sampling_seeds':formal.SEEDS,
           'base_source_status':core.STATUS,'records':terminal,'started_at':started_at,'finished_at':datetime.now(timezone.utc).isoformat()}
    core.save_new(out/'generation_index.json',index)
    print(json.dumps({'status':index['status'],'rows':96,'index':str(out/'generation_index.json')}),flush=True)


if __name__=='__main__':main()
