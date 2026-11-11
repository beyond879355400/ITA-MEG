"""Final-test LABEL-ONLY planning. Never decode frames or call a pixel model.

Reads 12 manual AU text files for each fixed test subject and AVI hdrl bytes
only. It never reads movi packet bodies, invokes a decoder, or creates images.
"""
import argparse
import collections
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.pycache_prefix=str(ROOT/'cache/pycache')
from prepare_disfa_pilot import labels_for_subject,contiguous_runs,file_state

SUBJECTS=('SN003','SN005','SN008','SN009','SN021','SN024','SN031','SN032')
AUS=(1,2,4,5,6,9,12,15,17,20,25,26)
SEED=20260914
SPLIT_SHA='b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d'
LABEL_ROOT=Path.home()/'Desktop/Cursor/geme-main/DISFA/ActionUnit_Labels'
CELLS=[(au,level) for au in AUS for level in (1,2,3)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def stable(*values):
    return hashlib.sha256(('|'.join(map(str,(SEED,)+values))).encode()).hexdigest()


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def avi_header_only(path):
    """Parse avih/strh inside hdrl; do not enter/read a movi data list."""
    state=file_state(path)
    if not state.get('resident'):raise ValueError('Video must already be resident for header-only inspection')
    ranges=[];headers={};streams=[]
    with Path(path).open('rb') as handle:
        def read_at(offset,count,reason):
            if count>64:raise ValueError('Header-only read exceeded its fixed small read size')
            handle.seek(offset);data=handle.read(count)
            if len(data)!=count:raise ValueError('Truncated AVI metadata')
            ranges.append({'offset':offset,'bytes':count,'reason':reason})
            return data
        signature=read_at(0,12,'RIFF AVI signature')
        if signature[:4]!=b'RIFF' or signature[8:]!=b'AVI ':raise ValueError('Not a supported AVI RIFF file')
        def scan(start,end):
            position=start
            while position+8<=end:
                chunk=read_at(position,8,'header chunk identifier/length');name=chunk[:4];size=struct.unpack('<I',chunk[4:])[0]
                body=position+8;next_position=body+size+(size%2)
                if next_position>end+1:raise ValueError('AVI header chunk exceeds enclosing list')
                if name==b'LIST':
                    kind=read_at(body,4,'header list kind')
                    if kind in (b'hdrl',b'strl',b'odml'):
                        if size>2*1024*1024:raise ValueError('Unusually large AVI header; manual metadata review required')
                        scan(body+4,body+size)
                    elif kind==b'movi':
                        raise ValueError('Refusing to inspect movie packet data')
                elif name==b'avih':
                    values=struct.unpack('<14I',read_at(body,56,'AVIMAINHEADER fields'))
                    headers.update(microseconds_per_frame=values[0],avih_total_frames=values[4],initial_frames=values[5],streams=values[6],width=values[8],height=values[9])
                elif name==b'strh':
                    raw=read_at(body,56,'AVISTREAMHEADER fields')
                    if raw[:4]==b'vids':
                        scale,rate,start_frame,length=struct.unpack_from('<4I',raw,20)
                        streams.append({'type':'vids','codec_fourcc':raw[4:8].decode('ascii',errors='replace'),'scale':scale,'rate':rate,'start':start_frame,'length':length})
                elif name==b'dmlh':
                    headers['odml_total_frames']=struct.unpack('<I',read_at(body,4,'OpenDML frame count'))[0]
                position=next_position
        position=12;found=False
        while position+12<=min(state['size_bytes'],2*1024*1024):
            head=read_at(position,8,'top-level chunk identifier/length');size=struct.unpack('<I',head[4:])[0]
            if head[:4]==b'LIST':
                kind=read_at(position+8,4,'top-level list kind')
                if kind==b'hdrl':
                    if size>2*1024*1024:raise ValueError('Header list too large')
                    scan(position+12,position+8+size);found=True;break
                if kind==b'movi':raise ValueError('No header found before movie data')
            position+=8+size+(size%2)
        if not found or len(streams)!=1:raise ValueError('Missing or ambiguous video stream header')
    stream=streams[0]
    if headers.get('avih_total_frames')!=stream['length'] or headers.get('odml_total_frames',stream['length']) not in (0,stream['length']):
        raise ValueError('AVI frame counts disagree; no guessed mapping')
    if not stream['scale'] or not stream['rate'] or stream['start']!=0 or headers.get('initial_frames')!=0:
        raise ValueError('Unsupported AVI timing/start offset')
    payload={'file':state,'header':headers,'video_stream':stream,'frames':stream['length'],'read_ranges':ranges,'header_bytes_read':sum(r['bytes'] for r in ranges),
             'decoder_used':False,'movie_packet_bodies_read':False,'checksum_scope':'header and file metadata only, not compressed-video content'}
    return {**payload,'video_metadata_sha256':canonical_sha(payload)}


def source_interval(vectors,subject):
    runs=contiguous_runs([frame for frame,values in vectors.items() if not any(values.values())])
    if not runs:raise ValueError(subject+' has no all-12-AU-zero source interval')
    length=max(end-start+1 for start,end in runs)
    tied=[run for run in runs if run[1]-run[0]+1==length]
    start,end=min(tied,key=lambda run:stable('source',subject,*run))
    return {'start_frame_1based':start,'end_frame_1based':end,'length':length,'source_frame_1based':(start+end)//2,'midpoint_rule':'floor((start+end)/2)',
            'maximum_length_ties':len(tied),'tie_break_seed':SEED}


def cell_indices(vector):return [index*3+value-1 for index,value in enumerate(vector) if value in (1,2,3)]


def inventory(output):
    split_path=HERE/'au_supervision_v2/subjects_frozen.json'
    if sha(split_path)!=SPLIT_SHA:raise ValueError('Frozen subject split changed')
    split=json.loads(split_path.read_text())
    if tuple(split['groups']['test'])!=SUBJECTS or split.get('test_locked') is not True:raise ValueError('Do not change fixed test subjects')
    video_inventory_path=HERE/'pilot_v1_02/raw_video_inventory.json'
    known=json.loads(video_inventory_path.read_text());video_map={(r['subject'],r['camera']):r for r in known if r['subject'] in SUBJECTS}
    subjects={};candidates=[];records=[];video_records=[];available=set()
    for subject in SUBJECTS:
        vectors,_,labels=labels_for_subject(LABEL_ROOT,subject);records.extend(labels)
        interval=source_interval(vectors,subject);camera=None;chosen=None
        for option in ('right','left'):
            record=avi_header_only(Path(video_map[(subject,option)]['path']))
            record.update(subject=subject,camera=option,label_frames=len(vectors),frame_count_matches_labels=record['frames']==len(vectors))
            video_records.append(record)
            if record['frame_count_matches_labels']:camera=option;chosen=record;break
        if chosen is None:raise ValueError(subject+' has no camera header matching all manual label frames')
        subset=[]
        for frame,labels in vectors.items():
            vector=[labels['au'+str(a)] for a in AUS]
            if 0<max(vector)<=3:
                covered=cell_indices(vector);available.update(covered)
                subset.append({'subject':subject,'frame':frame,'vector':vector,'cells':covered,'tie_hash':stable('target',subject,frame)})
        if len(subset)<4:raise ValueError(subject+' cannot provide four eligible nonzero target frames')
        candidates.extend(subset)
        subjects[subject]={'manual_label_frames':len(vectors),'source_interval':interval,'source_au_vector_0_5':[0]*12,'camera':camera,
            'video_path':chosen['file']['path'],'video_metadata_sha256':chosen['video_metadata_sha256'],'eligible_frames':len(subset),
            'eligible_cells':sorted({cell for row in subset for cell in row['cells']})}
        print(json.dumps({'subject':subject,'source':interval,'camera':camera,'frames':chosen['frames'],'eligible':len(subset)}),flush=True)
    payload={'status':'label_inventory_only','test_subjects':list(SUBJECTS),'subject_split_sha256':SPLIT_SHA,'au_order':list(AUS),'label_records':records,
        'video_inventory_path':str(video_inventory_path),'video_inventory_sha256':sha(video_inventory_path),'video_headers':video_records,'subjects':subjects,'candidates':candidates,
        'available_cells':sorted(available),'unavailable_cells':[list(CELLS[i]) for i in range(36) if i not in available],
        'test_images_decoded':0,'test_images_prepared':0,'models_called':0,'sampling_seeds_proposed':[101,202,303]}
    save(output/'label_inventory.json',payload)
    return payload


def solve(payload,per_subject,time_limit):
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import csc_matrix,vstack
    candidates=sorted(payload['candidates'],key=lambda row:(SUBJECTS.index(row['subject']),row['tie_hash']))
    n=len(candidates);rows=[];cols=[];data=[];lower=[];upper=[]
    def add(indices,values,lo,hi):
        row=len(lower);rows.extend([row]*len(indices));cols.extend(indices);data.extend(values);lower.append(lo);upper.append(hi)
    by_subject={subject:[] for subject in SUBJECTS}
    by_cell={cell:[] for cell in range(36)}
    for index,row in enumerate(candidates):
        by_subject[row['subject']].append(index)
        for cell in row['cells']:by_cell[cell].append(index)
    for subject,indices in by_subject.items():
        add(indices,[1]*len(indices),per_subject,per_subject)
        ordered=sorted(indices,key=lambda index:candidates[index]['frame'])
        right=0;previous_window=None
        for left,index in enumerate(ordered):
            right=max(right,left)
            while right<len(ordered) and candidates[ordered[right]]['frame']-candidates[index]['frame']<16:right+=1
            window=ordered[left:right]
            if len(window)>1 and window!=previous_window:add(window,[1]*len(window),-np.inf,1)
            previous_window=window
    for cell,indices in by_cell.items():add(indices+[n+cell],[1]*len(indices)+[-1],0,np.inf)
    A=csc_matrix((data,(rows,cols)),shape=(len(lower),n+36),dtype=float)
    A.indices=A.indices.astype(np.int32);A.indptr=A.indptr.astype(np.int32)
    constraint=LinearConstraint(A,np.asarray(lower),np.asarray(upper));bounds=Bounds(np.zeros(n+36),np.ones(n+36))
    objective=np.r_[np.zeros(n),-np.ones(36)]
    options={'disp':False,'presolve':True,'mip_rel_gap':0.,'time_limit':float(time_limit),'threads':1}
    begin=time.monotonic();primary=milp(objective,integrality=np.ones(n+36),bounds=bounds,constraints=constraint,options=options)
    if primary.x is None:raise RuntimeError('No feasible label plan found: '+primary.message)
    chosen=[candidates[i] for i in range(n) if primary.x[i]>.5]
    covered=set(cell for row in chosen for cell in row['cells']);coverage=len(covered)
    upper_bound=len(payload['available_cells'])
    primary_proven=primary.status==0 or coverage==upper_bound
    primary_info={'status':int(primary.status),'message':primary.message,'coverage':coverage,'theoretical_available_cell_upper_bound':upper_bound,'coverage_optimal_proven':primary_proven,
                  'mip_gap':None if getattr(primary,'mip_gap',None) is None else float(primary.mip_gap),'seconds':time.monotonic()-begin}
    print(json.dumps({'per_subject':per_subject,'primary':primary_info}),flush=True)
    # Preserve maximum coverage before lower-intensity and seeded tie breaking.
    extra=csc_matrix((np.ones(36),(np.zeros(36),np.arange(n,n+36))),shape=(1,n+36))
    B=vstack((A,extra),format='csc');B.indices=B.indices.astype(np.int32);B.indptr=B.indptr.astype(np.int32)
    fixed=LinearConstraint(B,np.r_[lower,coverage],np.r_[upper,coverage])
    costs=np.asarray([73792*max(row['vector'])+64*sum(row['vector'])+(i+1)/(n+1) for i,row in enumerate(candidates)]+[0.]*36)
    begin=time.monotonic();refined=milp(costs,integrality=np.ones(n+36),bounds=bounds,constraints=fixed,options=options)
    if refined.x is not None:chosen=[candidates[i] for i in range(n) if refined.x[i]>.5]
    refined_info={'status':int(refined.status),'message':refined.message,'used':refined.x is not None,'objective':None if refined.fun is None else float(refined.fun),
                  'mip_gap':None if getattr(refined,'mip_gap',None) is None else float(refined.mip_gap),'seconds':time.monotonic()-begin}
    selected={subject:sorted((row for row in chosen if row['subject']==subject),key=lambda row:row['frame']) for subject in SUBJECTS}
    for subject,targets in selected.items():
        if len(targets)!=per_subject or any(b['frame']-a['frame']<16 for a,b in zip(targets,targets[1:])):raise ValueError('MILP assignment violated quota/spacing')
    actual=set(cell for row in chosen for cell in row['cells'])
    if len(actual)<coverage:raise ValueError('Refinement lost primary coverage')
    return {'nonzero_targets_per_subject':per_subject,'primary':primary_info,'refinement':refined_info,'selected':selected,
            'covered_cells':[list(CELLS[i]) for i in sorted(actual)],'missing_cells':[list(CELLS[i]) for i in range(36) if i not in actual],
            'coverage_count':len(actual),'coverage_denominator':36,'full_coverage':len(actual)==36}


def make_plan(payload,solution):
    cases=[]
    for subject in SUBJECTS:
        source=payload['subjects'][subject];frame=source['source_interval']['source_frame_1based']
        common={'subject':subject,'split':'test','camera':source['camera'],'video_path':source['video_path'],'video_metadata_sha256':source['video_metadata_sha256'],
                'source_frame_1based':frame,'source_frame_zero_based':frame-1,'source_au_vector_0_5':[0]*12,'au_order':list(AUS),'images_prepared':False}
        cases.append({**common,'case_id':f'planned_test_{subject}_noedit_f{frame:05d}','kind':'source_noedit','target_frame_1based':frame,'target_frame_zero_based':frame-1,'target_au_vector_0_5':[0]*12})
        for target in solution['selected'][subject]:
            cases.append({**common,'case_id':f'planned_test_{subject}_target_f{target["frame"]:05d}','kind':'nonzero_target','target_frame_1based':target['frame'],
                'target_frame_zero_based':target['frame']-1,'target_au_vector_0_5':target['vector'],'covered_cells':[list(CELLS[c]) for c in target['cells']],'tie_hash':target['tie_hash']})
    return {'status':'label_only_proposal_waiting_generation_config_freeze','subject_split_sha256':SPLIT_SHA,'test_subjects':list(SUBJECTS),
        'nonzero_targets_per_subject':solution['nonzero_targets_per_subject'],'sampling_seeds_proposed':[101,202,303],'cases':cases,'coverage':solution,
        'source_intervals':{s:payload['subjects'][s]['source_interval'] for s in SUBJECTS},'selection_seed':SEED,'minimum_nonzero_target_frame_gap':16,
        'selection_inputs':'manual labels only; no model scores, output images or detector results',
        'frame_index_policy':'label frame n is 1-based; planned zero-based index is n-1; camera header length must match labels, no empirical shift guessed',
        'source_rule':'midpoint of longest all-12-AU-zero interval, hash tie-break seed20260914',
        'test_images_decoded':0,'test_images_prepared':0,'models_called':0,'generation_configuration_frozen':False,'not_a_completed_test':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--time-limit',type=int,default=60)
    args=parser.parse_args();output=args.output.resolve()
    if ROOT not in output.parents or output.exists():raise ValueError('Use a new in-experiment label-only output directory')
    output.mkdir(parents=True);begin=time.monotonic()
    payload=inventory(output)
    print('CANDIDATES '+str(len(payload['candidates']))+' AVAILABLE '+str(len(payload['available_cells'])),flush=True)
    solution3=solve(payload,3,args.time_limit);plan3=make_plan(payload,solution3);save(output/'plan_3_targets_per_subject.json',plan3)
    plans=[plan3]
    if not solution3['full_coverage']:
        solution4=solve(payload,4,args.time_limit);plan4=make_plan(payload,solution4);save(output/'plan_4_targets_per_subject.json',plan4);plans.append(plan4)
    save(output/'summary.json',{'status':'label_only_planning_complete','plans':[{'nonzero_targets_per_subject':p['nonzero_targets_per_subject'],'cases':len(p['cases']),
        'coverage':p['coverage']['coverage_count'],'missing_cells':p['coverage']['missing_cells'],'coverage_optimal_proven':p['coverage']['primary']['coverage_optimal_proven']} for p in plans],
        'elapsed_seconds':time.monotonic()-begin,'planner_sha256':sha(__file__),'label_reader_sha256':sha(HERE/'prepare_disfa_pilot.py'),
        'test_images_decoded':0,'test_images_prepared':0,'models_called':0,'generation_configuration_frozen':False})
    print(json.dumps({'output':str(output),'plans':[(p['nonzero_targets_per_subject'],p['coverage']['coverage_count']) for p in plans]}),flush=True)


if __name__=='__main__':main()
