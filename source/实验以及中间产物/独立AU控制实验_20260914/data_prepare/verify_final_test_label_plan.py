"""Independent CSV/metadata-only verification; does not import the planner."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

AUS=(1,2,4,5,6,9,12,15,17,20,25,26)
SUBJECTS=('SN003','SN005','SN008','SN009','SN021','SN024','SN031','SN032')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify(directory):
    directory=Path(directory)
    inventory=json.loads((directory/'label_inventory.json').read_text())
    plan=json.loads((directory/'plan_3_targets_per_subject.json').read_text())
    assert tuple(plan['test_subjects'])==SUBJECTS and plan['sampling_seeds_proposed']==[101,202,303]
    assert plan['generation_configuration_frozen'] is False and plan['test_images_prepared']==0
    labels={subject:{} for subject in SUBJECTS}
    for record in inventory['label_records']:
        assert record['subject'] in SUBJECTS and record['au'] in AUS
        assert digest(record['path'])==record['sha256']
        column={}
        with Path(record['path']).open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.reader(handle):
                if not row:continue
                frame,value=map(int,row)
                assert frame not in column and 0<=value<=5
                column[frame]=value
        assert sorted(column)==list(range(1,len(column)+1))
        assert record['au'] not in labels[record['subject']]
        labels[record['subject']][record['au']]=column
    assert len(inventory['label_records'])==96
    old_inventory=json.loads(Path(inventory['video_inventory_path']).read_text())
    assert digest(inventory['video_inventory_path'])==inventory['video_inventory_sha256']
    old_headers={(r['subject'],r['camera']):r for r in old_inventory}
    checked_headers={}
    for record in inventory['video_headers']:
        base={k:v for k,v in record.items() if k not in ('subject','camera','label_frames','frame_count_matches_labels','video_metadata_sha256')}
        assert canonical(base)==record['video_metadata_sha256']
        stat=os.stat(record['file']['path'])
        assert stat.st_size==record['file']['size_bytes'] and stat.st_mtime_ns==record['file']['mtime_ns']
        assert record['decoder_used'] is False and record['movie_packet_bodies_read'] is False
        original=old_headers[(record['subject'],record['camera'])]
        assert record['frames']==int(original['video_stream']['nb_frames'])
        assert record['header']['width']==original['video_stream']['width'] and record['header']['height']==original['video_stream']['height']
        checked_headers[(record['subject'],record['camera'])]=record
    frequency=collections.Counter();cell_subjects=collections.defaultdict(set);rows=[];case_ids=set();available=set()
    for subject in SUBJECTS:
        assert set(labels[subject])==set(AUS)
        lengths={len(column) for column in labels[subject].values()};assert len(lengths)==1
        total=lengths.pop();zero_runs=[];start=None;eligible=0
        for frame in range(1,total+2):
            values=[labels[subject][a][frame] for a in AUS] if frame<=total else None
            if values is not None and not any(values):
                if start is None:start=frame
            elif start is not None:
                zero_runs.append((start,frame-1));start=None
            if values is not None and 0<max(values)<=3:
                eligible+=1;available.update((a,v) for a,v in zip(AUS,values) if v>0)
        selected_source=plan['source_intervals'][subject]
        max_length=max(end-begin+1 for begin,end in zero_runs)
        assert selected_source['length']==max_length
        interval=(selected_source['start_frame_1based'],selected_source['end_frame_1based'])
        assert interval in zero_runs and interval[1]-interval[0]+1==max_length
        source=(interval[0]+interval[1])//2
        assert source==selected_source['source_frame_1based']
        assert eligible==inventory['subjects'][subject]['eligible_frames']
        cases=[c for c in plan['cases'] if c['subject']==subject]
        nonzero=[c for c in cases if c['kind']=='nonzero_target'];noedit=[c for c in cases if c['kind']=='source_noedit']
        assert len(cases)==4 and len(nonzero)==3 and len(noedit)==1
        targets=sorted(c['target_frame_1based'] for c in nonzero)
        gaps=[b-a for a,b in zip(targets,targets[1:])];assert min(gaps)>=16
        for case in cases:
            assert case['case_id'] not in case_ids;case_ids.add(case['case_id'])
            assert case['split']=='test' and case['images_prepared'] is False and case['source_frame_1based']==source
            assert case['source_frame_zero_based']==source-1 and case['target_frame_zero_based']==case['target_frame_1based']-1
            source_values=[labels[subject][a][source] for a in AUS]
            target_values=[labels[subject][a][case['target_frame_1based']] for a in AUS]
            assert source_values==case['source_au_vector_0_5']==[0]*12
            assert target_values==case['target_au_vector_0_5']
            header=checked_headers[(subject,case['camera'])]
            assert header['frames']==total and case['video_metadata_sha256']==header['video_metadata_sha256'] and case['video_path']==header['file']['path']
            if case['kind']=='nonzero_target':
                assert 0<max(target_values)<=3
                for au,value in zip(AUS,target_values):
                    if value>0:frequency[(au,value)]+=1;cell_subjects[(au,value)].add(subject)
            else:assert case['target_frame_1based']==source and target_values==source_values
        rows.append({'subject':subject,'camera':cases[0]['camera'],'source_interval':list(interval),'source':source,'targets':targets,'minimum_nonzero_target_gap':min(gaps),
                     'minimum_source_target_gap':min(abs(source-frame) for frame in targets)})
    all_cells={(au,level) for au in AUS for level in (1,2,3)}
    assert set(frequency)==all_cells==available
    assert {tuple(x) for x in plan['coverage']['covered_cells']}==all_cells and plan['coverage']['missing_cells']==[]
    return {'status':'passed','manual_label_files_rechecked':96,'video_headers_crosschecked':len(checked_headers),'cases':len(case_ids),'nonzero_targets':24,'noedit_cases':8,
        'coverage_count':36,'coverage_upper_bound':36,'coverage_optimal_by_complete_coverage':True,'missing_cells':[],
        'per_subject':rows,'coverage_frequencies':[{'au':a,'intensity':v,'cases':frequency[(a,v)],'subjects':sorted(cell_subjects[(a,v)])} for a,v in sorted(all_cells)],
        'SN005_headers':{camera:record['frames'] for (subject,camera),record in checked_headers.items() if subject=='SN005'},
        'video_header_bytes_read_by_planner':sum(r['header_bytes_read'] for r in inventory['video_headers']),
        'test_images_decoded':0,'test_images_prepared':0,'models_called':0,'generation_configuration_frozen':False,
        'plan_sha256':digest(directory/'plan_3_targets_per_subject.json'),'inventory_sha256':digest(directory/'label_inventory.json'),
        'verification_source_sha256':digest(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    result=verify(args.directory)
    with (args.directory/'independent_verification.json').open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
