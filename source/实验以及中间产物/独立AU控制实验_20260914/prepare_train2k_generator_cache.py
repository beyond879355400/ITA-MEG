"""Bind expanded training targets to unchanged source anchors and development cases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('plan','bind-dev'))
    parser.add_argument('--train-plan',type=Path,required=True)
    parser.add_argument('--alignment',type=Path,required=True)
    parser.add_argument('--previous-cache',type=Path,required=True)
    parser.add_argument('--previous-dev',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    if ROOT not in args.output.parents:
        raise ValueError('Require output in this experiment')
    old_plan=load(args.previous_cache/'pair_plan.json')
    old_meta=load(args.previous_cache/'manifest.json')
    old_dev=load(args.previous_dev)
    new_plan=load(args.train_plan)
    alignment=load(args.alignment)
    split=load(args.split)
    split_sha=sha(args.split)
    if not split['test_locked'] or old_meta['subject_split_sha256']!=split_sha or old_dev['subject_split_sha256']!=split_sha or alignment['subject_split_sha256']!=split_sha:
        raise ValueError('Subject split changed')
    if old_dev['cache_sha256']!=old_meta['cache_sha256'] or old_meta['test_data_used'] or old_dev['test_data_used']:
        raise ValueError('Previous developer reference is invalid')
    if alignment['status']!='complete' or alignment['failed']!=0 or alignment['count']!=2000:
        raise ValueError('New training alignment must complete without dropping cases')
    train=new_plan['pairs']
    dev=[p for p in old_plan['pairs'] if p['split']=='dev']
    if len(train)!=2000 or len(dev)!=129:
        raise ValueError('Unexpected data size; do not silently change the experiment')
    old_sources={p['subject']:p['source'] for p in old_plan['pairs']}
    seen=set()
    for pair in train+dev:
        role=pair['split']
        if role not in ('train','dev') or pair['subject'] not in split['groups'][role] or pair['subject'] in split['groups']['test']:
            raise ValueError('Split leakage')
        if pair['case_id'] in seen:
            raise ValueError('Duplicate case ID')
        seen.add(pair['case_id'])
        if pair['source']!=old_sources[pair['subject']] or any(pair['source_au_vector_0_5']):
            raise ValueError('Fixed true-neutral source anchor changed')
        if role=='train' and max(pair['target_au_vector_0_5'])>3:
            raise ValueError('Expanded target is outside the fixed low/moderate range')
    if args.phase=='plan':
        if args.output.exists():
            raise ValueError('Append a new cache directory')
        document={'stage':'expanded_train_with_unchanged_development','test_data_used':False,
            'old_generated_images_or_metrics_used':False,
            'selection':'Use the sealed 2000 train-only pair plan verbatim and the original 129 dev pairs verbatim. All 19 original source anchors remain unchanged. No new dev/test selection.',
            'alignment_manifest':str(args.alignment.resolve()),'alignment_sha256':sha(args.alignment),
            'train_pair_plan_path':str(args.train_plan.resolve()),'train_pair_plan_sha256':sha(args.train_plan),
            'previous_pair_plan_path':str((args.previous_cache/'pair_plan.json').resolve()),
            'previous_pair_plan_sha256':sha(args.previous_cache/'pair_plan.json'),
            'previous_development_manifest_path':str(args.previous_dev.resolve()),'previous_development_manifest_sha256':sha(args.previous_dev),
            'subject_split_path':str(args.split.resolve()),'subject_split_sha256':split_sha,
            'subject_split_document':split,'counts':{'train':len(train),'dev':len(dev)},'pairs':train+dev}
        args.output.mkdir(parents=True)
        (args.output/'pair_plan.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
        shutil.copyfile(__file__,args.output/'merge_plan_source.py')
        shutil.copyfile(ROOT/'prepare_expanded_training_cache.py',args.output/'prepare_source.py')
        print(json.dumps({'status':'plan_complete','train':len(train),'dev':len(dev),'pair_plan_sha256':sha(args.output/'pair_plan.json'),'old_dev_file_unchanged':True},indent=2))
    else:
        meta=load(args.output/'manifest.json')
        plan=load(args.output/'pair_plan.json')
        if meta['status']!='complete' or meta['subject_split_sha256']!=split_sha or meta['source_manifest_sha256']!=sha(args.output/'pair_plan.json') or sha(args.output/'features.pt')!=meta['cache_sha256']:
            raise ValueError('New feature cache is not verified')
        if plan['previous_development_manifest_sha256']!=sha(args.previous_dev):
            raise ValueError('Original developer file changed')
        current={p['case_id']:p for p in meta['pairs'] if p['split']=='dev'}
        original={p['case_id']:p for p in old_meta['pairs'] if p['split']=='dev'}
        if current!=original:
            raise ValueError('Development metadata/content was changed')
        rebound=dict(old_dev)
        rebound.update(cache_sha256=meta['cache_sha256'],
                       previous_cache_sha256=old_dev['cache_sha256'],
                       source_development_manifest_sha256=sha(args.previous_dev),
                       rebinding_reason='Only training data/cache identity expanded; cases, order, labels, groups and sampling seeds are unchanged.')
        destination=args.output/'dev12_manifest.json'
        if destination.exists():
            raise ValueError('Cannot overwrite a rebound developer manifest')
        destination.write_text(json.dumps(rebound,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'status':'dev_binding_complete','path':str(destination),'sha256':sha(destination),'same_cases_and_seeds':rebound['cases']==old_dev['cases'] and rebound['sampling_seeds']==old_dev['sampling_seeds']},indent=2))


if __name__=='__main__':
    main()
