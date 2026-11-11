"""Compare frozen source/target labels against original DISFA annotation text."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


def validate_plan(plan):
    if plan.get('kind') != 'frozen_preservation_factorial_v1' or plan.get('status') != 'frozen':
        raise ValueError('Explicit frozen study plan required')
    cases = plan.get('cases')
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError('Exactly24 frozen nonzero cases required')
    ids, counts = set(), {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Case object required')
        cid, subject = case.get('case_id'), case.get('subject')
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', cid) or cid in ids:
            raise ValueError('Unique safe case id required')
        if not isinstance(subject, str) or not re.fullmatch(r'SN[0-9]{3}', subject):
            raise ValueError('DISFA subject id required')
        ids.add(cid)
        counts[subject] = counts.get(subject, 0) + 1
        if case.get('au_order') != AUS:
            raise ValueError('Fixed12 AU order required')
        for role in ('source', 'target'):
            frame = case.get(role + '_frame_1based')
            values = case.get(role + '_au_vector_0_5')
            if type(frame) is not int or frame < 1:
                raise ValueError('Explicit positive 1-based integer frame required')
            if not isinstance(values, list) or len(values) != 12 or not all(
                    type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 5 for v in values):
                raise ValueError('Twelve finite0-5 labels required')
        if any(case['source_au_vector_0_5']) or not any(case['target_au_vector_0_5']):
            raise ValueError('Original neutral-source, nonzero-target contract required')
    if len(counts) != 8 or set(counts.values()) != {3}:
        raise ValueError('Eight subjects with3 requests each required')


def parse_labels(text):
    result = {}
    for cells in csv.reader(text.splitlines()):
        if not cells:
            continue
        if len(cells) != 2:
            raise ValueError('Expected explicit frame,intensity pairs')
        frame, value = map(int, cells)
        if frame < 1 or value not in range(6) or frame in result:
            raise ValueError('Invalid/duplicate frame or AU intensity')
        result[frame] = value
    if not result:
        raise ValueError('Empty manual labels')
    return result


def verify(freeze, labels, output):
    freeze, labels, output = Path(freeze).resolve(), Path(labels).resolve(), Path(output).resolve()
    if HERE not in output.parents or output.exists():
        raise ValueError('New output inside supplement required')
    raw = freeze.read_bytes()
    plan = json.loads(raw)
    validate_plan(plan)
    cache, sources, rows = {}, [], []
    for case in plan['cases']:
        subject = case['subject']
        for role in ('source', 'target'):
            frame = case[role + '_frame_1based']
            expected = case[role + '_au_vector_0_5']
            if len(expected) != len(case['au_order']):
                raise ValueError('AU dimension mismatch')
            for au, value in zip(case['au_order'], expected):
                key = (subject, au)
                if key not in cache:
                    path = labels / subject / f'{subject}_au{au}.txt'
                    data = path.read_bytes()
                    cache[key] = parse_labels(data.decode('utf-8'))
                    sources.append({'subject': subject, 'au': au, 'path': str(path),
                                    'sha256': hashlib.sha256(data).hexdigest(), 'frames': len(cache[key])})
                actual = cache[key].get(frame)
                rows.append({'case_id': case['case_id'], 'subject': subject, 'role': role,
                             'frame_1based': frame, 'au': au, 'frozen_value': value,
                             'original_manual_value': actual, 'matches': actual is not None and actual == value})
    if len(rows) != 576 or len(sources) != 96:
        raise ValueError('Incomplete576-check/96-annotation-file accounting')
    failures = [row for row in rows if not row['matches']]
    result = {'status': 'pass' if not failures else 'label_mismatch',
              'freeze_sha256': hashlib.sha256(raw).hexdigest(), 'original_annotation_files': sources,
              'case_role_au_checks': len(rows),
              'unique_real_frames': len({(r['subject'], r['frame_1based']) for r in rows}),
              'unique_frame_au_checks': len({(r['subject'], r['frame_1based'], r['au']) for r in rows}),
              'mismatch_count': len(failures), 'mismatches': failures, 'records': rows,
              'scope': 'Existing DISFA human annotations; not new generated-image manual annotations'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('status', 'case_role_au_checks', 'unique_real_frames',
                                           'unique_frame_au_checks', 'mismatch_count')}))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.freeze, args.labels, args.output) else 1)
