"""One-shot fixed four-variant v4 continuation, step 200 -> 2000.

Standard library only. --dry-run never imports Torch or starts a child process.
No scheduling, retry loop, parallel training, or arbitrary task commands.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

EXPERIMENT=Path(__file__).resolve().parent
PROJECT=EXPERIMENT.parent.parent
VARIANTS=('bidirectional','condition_reads_image','image_reads_condition','mlp')
MODEL_VERSION='v4_source_centered_protected_au_relative_rms'
CHECKPOINT_KIND='independent_au_adapter_resumable_v4_source_centered'
MATRIX_KIND='v4_slot_train2k_epoch1_matrix'


class QueueExecutionError(RuntimeError):
    """An output/process orchestration error after queue execution was entered."""


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def invalid(value):raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def write_new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def project_file(value,project,experiment,label):
    if not isinstance(value,str) or not value.strip():raise ValueError(label+' is not filled in the execution manifest')
    raw=Path(value);path=(raw if raw.is_absolute() else project/raw).resolve()
    if experiment not in path.parents or not path.is_file():raise ValueError(label+' must be an existing file inside this experiment: '+str(path))
    return path


def checked_sha(path,expected,label):
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError(label+' SHA is not filled or invalid')
    actual=digest(path)
    if actual!=expected:raise ValueError(label+' SHA changed')
    return actual


def validate_task(task,trainer_sha,project,experiment):
    variant=task['variant']
    config_path=project_file(task.get('config_path'),project,experiment,variant+' config')
    parent=project_file(task.get('parent_checkpoint'),project,experiment,variant+' parent checkpoint')
    checked_sha(config_path,task.get('config_sha256'),variant+' config')
    checked_sha(parent,task.get('parent_sha256'),variant+' parent checkpoint')
    config=read_json(config_path)
    if config.get('variant')!=variant or config.get('max_steps')!=2000 or config.get('diagnostic') is not False or config.get('batch_size')!=1:
        raise ValueError(variant+' must use its fixed non-diagnostic batch1 total-2000 config')
    if config.get('model_version',MODEL_VERSION)!=MODEL_VERSION or config.get('readout')!='slot' or config.get('patch_scale')!='unit_mean':
        raise ValueError(variant+' config is not the frozen v4 slot/unit_mean experiment')
    seal=read_json(parent.with_suffix('.json'))
    if seal.get('kind')!=CHECKPOINT_KIND or seal.get('model_version')!=MODEL_VERSION or seal.get('variant')!=variant or seal.get('step')!=200 or seal.get('checkpoint_sha256')!=task['parent_sha256']:
        raise ValueError(variant+' parent sidecar is not the specified v4 step200 checkpoint')
    previous=read_json(parent.parent/'configuration.json')
    if previous.get('max_steps')!=200 or previous.get('variant')!=variant:
        raise ValueError(variant+' parent run configuration is not step200')
    for key,value in config.items():
        # The trainer fingerprints JSON: 1 and 1.0 are distinct even though
        # Python numeric equality says they are equal. Mirror that distinction.
        serialized=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        old_serialized=json.dumps(previous.get(key),sort_keys=True,separators=(',',':'),allow_nan=False)
        if key!='max_steps' and (key not in previous or serialized!=old_serialized):
            raise ValueError(variant+' continuation changes parent setting: '+key)
    completion=read_json(parent.parent/'result.json')
    if completion.get('status')!='complete' or completion.get('global_step')!=200 or completion.get('variant')!=variant:
        raise ValueError(variant+' parent run has not completed step200')
    snapshot=parent.parent/'snapshots/train_centered_adapter.py'
    checked_sha(snapshot,trainer_sha,variant+' parent trainer snapshot')
    return {'variant':variant,'config_path':str(config_path),'config_sha256':task['config_sha256'],
            'parent_checkpoint':str(parent),'parent_sha256':task['parent_sha256'],
            'start_step':200,'target_step':2000,'remaining_steps':1800,'config':config}


def preflight(manifest_path,project=PROJECT,experiment=EXPERIMENT):
    project=Path(project).resolve();experiment=Path(experiment).resolve()
    manifest_path=Path(manifest_path).resolve()
    if experiment not in manifest_path.parents:raise ValueError('Execution manifest must be in this experiment')
    manifest=read_json(manifest_path)
    if manifest.get('kind')!=MATRIX_KIND or manifest.get('parent_step')!=200 or manifest.get('target_step')!=2000:
        raise ValueError('Only the fixed v4 step200 -> 2000 matrix is supported')
    tasks=manifest.get('tasks',[])
    if [row.get('variant') for row in tasks]!=list(VARIANTS):raise ValueError('Exactly the fixed four variants, in their recorded order, are required')
    trainer=experiment/'train_centered_adapter.py'
    trainer_sha=checked_sha(trainer,manifest.get('trainer_sha256'),'Trainer')
    resolved=[validate_task(row,trainer_sha,project,experiment) for row in tasks]
    protocols=[{k:v for k,v in row['config'].items() if k!='variant'} for row in resolved]
    if any(value!=protocols[0] for value in protocols[1:]):raise ValueError('The four continuation configs differ beyond variant')
    return {'kind':MATRIX_KIND,'manifest_path':str(manifest_path),'manifest_sha256':digest(manifest_path),
            'trainer_path':str(trainer),'trainer_sha256':trainer_sha,'project_root':str(project),'experiment_root':str(experiment),
            'tasks':resolved,'total_remaining_steps':7200,'parent_step':200,'target_step':2000}


def minute_directory(parent,prefix,minute=None,create=True):
    parent=Path(parent);stamp=minute or datetime.now().strftime('%Y%m%d%H%M')
    if not re.fullmatch('[0-9]{12}',stamp):raise ValueError('Timestamp must be precise to the minute')
    index=1
    while True:
        path=parent/(prefix+'_'+stamp+('' if index==1 else '-'+str(index).zfill(2)))
        if create:
            try:path.mkdir(parents=True,exist_ok=False);return path
            except FileExistsError:index+=1
        elif not path.exists():return path
        else:index+=1


def execute_queue(prepared,python_bin=sys.executable,run_process=subprocess.run):
    experiment=Path(prepared['experiment_root']);project=Path(prepared['project_root'])
    queue=minute_directory(experiment/'runs','epoch1_matrix')
    tasks=[]
    for row in prepared['tasks']:
        run_path=minute_directory(queue/'training_runs','v4_'+row['variant']+'_step2000',create=False)
        tasks.append({**row,'run_directory':str(run_path)})
    write_new_json(queue/'queue_manifest.json',{**prepared,'python_bin':str(python_bin),'tasks':tasks})
    states={row['variant']:'planned' for row in tasks}
    for row in tasks:write_new_json(queue/'task_status'/row['variant']/'planned.json',row)
    failed=None
    with (queue/'events.jsonl').open('x',encoding='utf-8',buffering=1) as events:
        for row in tasks:
            variant=row['variant'];status_dir=queue/'task_status'/variant
            if failed is not None:
                states[variant]='not_started'
                write_new_json(status_dir/'not_started.json',{'status':'not_started','reason':'Earlier task failed: '+failed})
                continue
            command=[str(python_bin),'-u',prepared['trainer_path'],'--config',row['config_path'],'--resume',row['parent_checkpoint'],'--output',row['run_directory']]
            try:
                # Recheck immediately before EACH launch, not only at queue creation.
                checked_sha(prepared['manifest_path'],prepared['manifest_sha256'],'Execution manifest')
                checked_sha(prepared['trainer_path'],prepared['trainer_sha256'],'Trainer')
                validate_task(row,prepared['trainer_sha256'],project,experiment)
                started={'status':'running','variant':variant,'command':command,'cwd':str(project),'started_at':datetime.now().isoformat(timespec='seconds')}
                write_new_json(status_dir/'started.json',started);events.write(json.dumps(started)+'\n')
                print(json.dumps({'variant':variant,'status':'running','run_directory':row['run_directory']},ensure_ascii=False),flush=True)
                env=dict(os.environ,PYTHONUNBUFFERED='1')
                with (status_dir/'process.log').open('x',encoding='utf-8') as log:
                    process=run_process(command,cwd=str(project),stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
                if process.returncode!=0:raise RuntimeError('Trainer exited with code '+str(process.returncode))
                result=read_json(Path(row['run_directory'])/'result.json')
                if result.get('status')!='complete' or result.get('global_step')!=2000 or result.get('variant')!=variant or result.get('model_version')!=MODEL_VERSION:
                    raise RuntimeError('Trainer did not report successful step2000 completion')
                finished={'status':'complete','variant':variant,'global_step':2000,'run_directory':row['run_directory']}
                states[variant]='complete'
            except Exception as error:
                failed=variant;states[variant]='failed'
                finished={'status':'failed','variant':variant,'error_type':type(error).__name__,'error':str(error),'run_directory':row['run_directory']}
            write_new_json(status_dir/'finished.json',finished);events.write(json.dumps(finished)+'\n')
            print(json.dumps(finished,ensure_ascii=False),flush=True)
    summary={'status':'failed' if failed else 'complete','queue_directory':str(queue),'tasks':states,'failed_variant':failed,
             'old_runs_overwritten':False,'automatic_retry':False}
    write_new_json(queue/'result.json',summary)
    return summary


def run_manifest(manifest_path,dry_run=False,project=PROJECT,experiment=EXPERIMENT,run_process=subprocess.run):
    prepared=preflight(manifest_path,project,experiment)
    if dry_run:
        return {'status':'dry_run_passed','training_started':False,'models_loaded':False,'filesystem_outputs_created':False,**prepared}
    try:
        return execute_queue(prepared,run_process=run_process)
    except Exception as error:
        raise QueueExecutionError(str(error)) from error


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        summary=run_manifest(args.manifest,args.dry_run)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return 0 if summary['status'] in ('complete','dry_run_passed') else 1
    except Exception as error:
        execution_error=isinstance(error,QueueExecutionError)
        print(json.dumps({'status':'execution_error' if execution_error else 'preflight_failed','training_started':None if execution_error else False,
                          'inspect_queue_artifacts':execution_error,'error_type':type(error).__name__,'error':str(error)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
