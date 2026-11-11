"""Authorized consumption of ONLY this run's five verified download parts.

No archive is deleted until a completed output file has been flushed, fsynced,
and rehashed. Parts retired by the reader are closed before becoming eligible.
Failure preserves all completed/partial outputs, remaining parts and journal.
"""
from datetime import datetime
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

import stage_weights as stage


class RetiringReader(stage.MultipartReader):
    def __init__(self, paths):
        super().__init__(paths)
        self.retired=[]

    def readinto(self,target):
        before=self.index
        result=super().readinto(target)
        # Base class closes old stream before advancing index.
        self.retired.extend(self.paths[before:self.index])
        return result


def sync_directory(path):
    fd=os.open(str(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def sync_directory_chain(path,root):
    path=Path(path).resolve();root=Path(root).resolve()
    stage.need(path==root or root in path.parents,'Directory durability path escaped output')
    while True:
        sync_directory(path)
        if path==root:break
        path=path.parent
    sync_directory(root.parent)


def durable_json(path,document):
    with Path(path).open('x') as f:
        json.dump(document,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    sync_directory(Path(path).parent)


def extract_consume(download_root,out,entries,index,reserve=stage.RESERVE):
    """Testable core. Production main always uses unchanged8GiB reserve."""
    download_root=Path(download_root).resolve();out=Path(out)
    out.mkdir(exist_ok=False,parents=True)
    sync_directory(out.parent)
    paths=[download_root/row['name'] for row in entries]
    approved={str(path):row for path,row in zip(paths,entries)}
    deleted=[];completed=[];reader=RetiringReader(paths)
    durable_json(out/'archive_index.json',index)
    journal=(out/'consumption_events.jsonl').open('x',buffering=1)
    sync_directory(out)
    def event(**row):
        journal.write(json.dumps({'at':stage.now(),**row},ensure_ascii=False,allow_nan=False)+'\n')
        journal.flush();os.fsync(journal.fileno())
    def validate(path):
        path=Path(path)
        stage.need(str(path) in approved and path.parent==download_root and not path.is_symlink(), 'Part outside exact allowlist or symlink')
        row=approved[str(path)]
        stage.need(path.is_file() and path.stat().st_size==row['size'] and stage.sha(path)==row['digest'][7:], 'Part changed since approval')
        return row
    def retire_completed():
        for path in list(reader.retired):
            if str(path) in deleted:continue
            stage.need(len(deleted)<len(paths) and path==paths[len(deleted)], 'Part consumption order differs')
            row=validate(path)
            event(event='delete_intent',path=str(path),bytes=row['size'],sha256=row['digest'][7:],
                  last_completed_file=completed[-1] if completed else None,reader_closed=True)
            path.unlink()
            sync_directory(download_root)
            deleted.append(str(path))
            event(event='part_deleted',path=str(path),bytes=row['size'],sha256=row['digest'][7:])
    try:
        # Prevalidation takes place before any destructive action.
        for path in paths:validate(path)
        event(event='start',parts=entries,reserve_bytes=reserve,free_bytes=shutil.disk_usage(out).free)
        with io.BufferedReader(reader,buffer_size=1024*1024) as stream:
            with tarfile.open(fileobj=stream,mode='r|*') as tar:
                observed=[]
                for member in tar:
                    relative=stage.safe_member(member)
                    observed.append({'name':member.name,'bytes':member.size,'directory':member.isdir()})
                    stage.need(observed==index['members'][:len(observed)],'Archive differs from inspected member list')
                    target=out.joinpath(*relative.parts)
                    stage.need(target.resolve()==out.resolve() or out.resolve() in target.resolve().parents,'Tar path escape')
                    if member.isdir():
                        target.mkdir(parents=True,exist_ok=True);continue
                    stage.need(shutil.disk_usage(out).free>=member.size+reserve,'Member would violate8GiB reserve')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    h=__import__('hashlib').sha256();written=0
                    with tar.extractfile(member) as source,target.open('xb') as dest:
                        while True:
                            stage.need(shutil.disk_usage(out).free>=reserve,'Disk reserve reached; partial output retained')
                            chunk=source.read(4*1024*1024)
                            if not chunk:break
                            stage.need(shutil.disk_usage(out).free>=reserve+len(chunk),'Upcoming chunk would violate8GiB reserve')
                            dest.write(chunk);h.update(chunk);written+=len(chunk)
                        dest.flush();os.fsync(dest.fileno())
                    stage.need(written==member.size and stage.sha(target)==h.hexdigest(),'Extracted member size/SHA mismatch')
                    sync_directory_chain(target.parent,out)
                    record={'path':str(target.relative_to(out)),'bytes':written,'sha256':h.hexdigest()}
                    completed.append(record);event(event='member_durable_verified',**record)
                    retire_completed()
                stage.need(observed==index['members'] and sum(row['bytes'] for row in completed)==index['expanded_bytes'],'Incomplete archive')
                # Consume gzip/tar padding to EOF. This closes/retire final part
                # only after all output members are durable and checked.
                while stream.read(1024*1024):pass
        reader.close()
        # All outputs are independently reverified before any final cached part.
        for row in completed:
            stage.need(stage.sha(out/row['path'])==row['sha256'],'Final output changed')
        stage.need(reader.index==len(paths) and reader.retired==paths,'Reader did not retire all parts in order')
        retire_completed()
        receipt={'status':'complete','at':stage.now(),'files':completed,'expanded_bytes':index['expanded_bytes'],
                 'merged_archive_written':False,'source_parts_deleted':True,'deleted_parts':deleted,
                 'original_parts':entries,'consumption_journal':'consumption_events.jsonl'}
        durable_json(out/'extraction_receipt.json',receipt)
        event(event='complete',free_bytes=shutil.disk_usage(out).free,expanded_bytes=index['expanded_bytes'])
        return receipt
    except BaseException as error:
        reader.close()
        event(event='failed',error=str(error),completed_files=len(completed),deleted_parts=deleted)
        durable_json(out/'extraction_failure.json',{'status':'failed','error':str(error),'completed_files':completed,
                     'deleted_parts':deleted,'original_parts':entries,'remaining_parts':[str(p) for p in paths if p.exists()]})
        raise
    finally:
        reader.close();journal.close()


def main():
    stage.gate('cache_ready');stage.gate('consume_new_parts')
    stage.need(shutil.disk_usage(stage.HERE).free>=stage.RESERVE,'Disk reserve already reached')
    entries=stage.assets();paths=stage.verified_parts()
    index=stage.inspect(paths)
    saved=stage.read(stage.HERE/'archive_index_preextract.json')
    stage.need(index['members']==saved['members'] and index['expanded_bytes']==saved['expanded_bytes'],'Prior inspection differs')
    out=stage.HERE/'weights'/('official_v1_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    receipt=extract_consume(stage.HERE/'downloads',out,entries,index)
    print(json.dumps({'status':receipt['status'],'output':str(out),'expanded_bytes':receipt['expanded_bytes'],
                      'free_bytes':shutil.disk_usage(stage.HERE).free}),flush=True)


if __name__=='__main__':main()
