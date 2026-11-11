"""Fix a small balanced developer preview before generating new outputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('New in-experiment output required')
    metadata=json.loads((args.cache/'manifest.json').read_text())
    if metadata['test_data_used']:
        raise ValueError('Test is excluded')
    dev=[p for p in metadata['pairs'] if p['split']=='dev']
    chosen=[]
    for subject in sorted({p['subject'] for p in dev}):
        rows=[p for p in dev if p['subject']==subject]
        edited=[p for p in rows if any(p['target_au_vector_0_5'])]
        lowest=min(max(p['target_au_vector_0_5']) for p in edited)
        groups={
            'no_edit':[p for p in rows if p['aligned_source']['sample_id']==p['aligned_target']['sample_id']],
            'lowest_available':[p for p in edited if max(p['target_au_vector_0_5'])==lowest],
            'second_edited':edited,
        }
        for group,candidates in groups.items():
            candidates=[p for p in candidates if p['case_id'] not in {r['case_id'] for r in chosen}]
            if not candidates:
                raise ValueError('Missing predeclared condition for '+subject+':'+group)
            pair=min(candidates,key=lambda p:hashlib.sha256(('20260914:'+p['case_id']).encode()).hexdigest())
            chosen.append({'case_id':pair['case_id'],'split':'dev','subject':subject,'group':group,
                           'source_au_vector_0_5':pair['source_au_vector_0_5'],
                           'target_au_vector_0_5':pair['target_au_vector_0_5']})
    document={'status':'frozen_for_development','test_data_used':False,'cache_sha256':metadata['cache_sha256'],
              'subject_split_sha256':metadata['subject_split_sha256'],
              'selection':'Per dev subject: one exact source=no-edit request, one edited request with lowest available maximum AU, then one different edited request. Minimum SHA256(seed20260914:case_id) within each manual-label group; no generated scores used.',
              'planning_note':'An earlier in-memory max-AU=1 requirement failed for SN004 before any manifest/output; replaced by lowest genuinely available level. No labels are changed.',
              'sampling_seeds':[101], 'cases':chosen}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(document,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
