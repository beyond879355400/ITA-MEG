#!/usr/bin/env python3
"""CPU-only union of original600 and expanded2000 true training frames."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;ROOT=SUP.parent
sys.path.insert(0,str(SUP))
from align_real_frames import sha,write_json,SPLIT_SHA

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def assert_resident(path):
    st=Path(path).stat()
    assert not(st.st_flags&0x40000000) and st.st_blocks>0,'Nonresident input: '+str(path)

def main():
    output=HERE/'data_union.json';assert not output.exists()
    oldfile=SUP/'alignment_v1/aligned_manifest.json';newfile=ROOT/'data_prepare/generator_train_v3_2k/alignment/aligned_manifest.json'
    old=json.loads(oldfile.read_text());new=json.loads(newfile.read_text());splitfile=ROOT/'data_prepare/au_supervision_v2/subjects_frozen.json'
    assert old['subject_split_sha256']==new['subject_split_sha256']==sha(splitfile)==SPLIT_SHA
    groups=json.loads(splitfile.read_text())['groups'];original=[r for r in old['samples'] if r['split']=='train'];dev=[r for r in old['samples'] if r['split']=='dev']
    assert len(original)==600 and len(dev)==240 and len(new['samples'])==2000
    byhash={};duplicates=[]
    for origin,rows in [('original_supervisor600',original),('expanded_generator2000',new['samples'])]:
        for row in rows:
            assert row['split']=='train' and row['subject'] in groups['train'] and row['subject'] not in groups['test'] and row['status']=='aligned'
            labels=row['input']['au_vector_0_5'];assert len(labels)==12 and all(0<=v<=5 for v in labels)
            for p,h in [('raw_path','raw_sha256'),('aligned_path','aligned_sha256')]:assert_resident(row[p]);assert sha(row[p])==row[h]
            key=row['raw_sha256'];origin_record={'origin':origin,'sample_id':row['sample_id'],'subject':row['subject'],'label_frame':row['input']['frame']}
            if key in byhash:
                kept=byhash[key]
                assert kept['subject']==row['subject'] and kept['input']['au_vector_0_5']==labels,'Duplicate raw image has conflicting identity/labels'
                assert kept['aligned_sha256']==row['aligned_sha256'],'Duplicate raw image has inconsistent canonical preprocessing'
                kept['union_origins'].append(origin_record);duplicates.append({'raw_sha256':key,'kept_sample_id':kept['sample_id'],'duplicate':origin_record})
            else:byhash[key]={**row,'union_origins':[origin_record]}
    train=sorted(byhash.values(),key=lambda r:(r['subject'],r['input']['frame'],r['raw_sha256']))
    assert {r['subject'] for r in train}==set(groups['train']) and {r['subject'] for r in dev}==set(groups['dev'])
    assert not set(r['raw_sha256'] for r in train)&set(r['raw_sha256'] for r in dev)
    for row in dev:
        assert row['subject'] not in groups['test'] and row['status']=='aligned'
        assert_resident(row['aligned_path']);assert sha(row['aligned_path'])==row['aligned_sha256']
    high_original={r['raw_sha256'] for r in original if max(r['input']['au_vector_0_5'])>3}
    assert high_original <= set(byhash)
    parentfile=SUP/'layer4_v1/qualification.json';parent=json.loads(parentfile.read_text())
    assert parent['qualified'] and parent['training_complete'] and not parent['test_used'] and parent['subject_split_sha256']==SPLIT_SHA
    assert sha(parent['checkpoint_path'])==parent['checkpoint_sha256']=='3dd3f4da559fc2a9a915713815847bebf481df09cc46ffccdbbf60bdfb88ed49'
    data={'protocol':'au_supervisor_real_train_union_v2','subject_split_sha256':SPLIT_SHA,'split_path':str(splitfile),'train_count':len(train),'dev_count':240,'test_used':False,
          'train_subjects':groups['train'],'dev_subjects':groups['dev'],'train':train,'dev':dev,
          'sources':{'original_alignment_manifest':{'path':str(oldfile),'sha256':sha(oldfile)},'expanded_alignment_manifest':{'path':str(newfile),'sha256':sha(newfile)}},
          'duplicate_policy':'deduplicate ORIGINAL RGB PNG SHA; require equal subject, full12AU and canonical SHA; retain original600 record on overlap',
          'duplicates':duplicates,'original600_high_gt_frames_preserved':len(high_original),'dev_rows_canonical_sha256':fingerprint(dev),
          'union_spacing_note':'old600 frames are retained, so the union is not globally constrained to16-frame gaps; this is AU fitting, not a new generator pairing set',
          'prepare_script_sha256':sha(__file__)}
    write_json(output,data)
    config={'data_manifest':str(output),'data_manifest_sha256':sha(output),'output_dir':str(HERE/'training'),
            'parent_qualification':str(parentfile),'parent_qualification_sha256':sha(parentfile),'parent_checkpoint':parent['checkpoint_path'],'parent_checkpoint_sha256':parent['checkpoint_sha256'],
            'subject_split_sha256':SPLIT_SHA,'model_config':parent['architecture'],'model_source_sha256':parent['preprocess']['sha256'],
            'device':'mps','seed':20260914,'epochs':30,'batch_size':16,'layer4_lr':.0001,'head_lr':.0005,'weight_decay':.0001,
            'optimizer_policy':'new AdamW with empty state because training data changed; not exact resume of previous training',
            'batchnorm':'running statistics frozen eval; layer4 affine trainable','loss':'same full12AU normalized MSE=rawMSE/25',
            'qualification_gate':parent['predeclared_gate'],'constant_baselines':'recompute train mean/median/zero from union labels only; same fixed240dev and original gate rules',
            'parent_entry_evaluation':True,'selection':'qualified checkpoints first, then smallest dev macroMAE, activeMAE, earliest epoch; parent entry reported separately',
            'scope':'AU supervisor extension and qualification; warmstart plus additional optimization means this is not a pure causal sample-count ablation',
            'test_used':False}
    write_json(HERE/'train_config.json',config)
    per_subject={s:sum(r['subject']==s for r in train) for s in groups['train']}
    report={'status':'pass','original_train':600,'expanded_train':2000,'raw_sha_duplicates_removed':len(duplicates),'union_train':len(train),
            'dev_unchanged':240,'dev_rows_canonical_sha256':fingerprint(dev),'high_gt_old_frames_preserved':len(high_original),'subject_counts':per_subject,
            'data_manifest_sha256':sha(output),'config_sha256':sha(HERE/'train_config.json'),'parent_checkpoint_sha256':parent['checkpoint_sha256'],
            'test_used':False,'mps_used':False,'images_copied_or_modified':False}
    write_json(HERE/'prepare_qa.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
