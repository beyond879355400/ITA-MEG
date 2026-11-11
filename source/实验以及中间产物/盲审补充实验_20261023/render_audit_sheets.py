"""Read-only raw-image QA contact sheets; never edit experimental images."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / '独立AU控制实验_20260914'
ARMS = ('full', 'no_projection', 'no_smoothing', 'both_off')
LABELS = ('Source', 'Real target: reference', 'Full P1S1', 'No projection P0S1',
          'No smoothing P1S0', 'Both off P0S0')
SEEDS = (101, 202, 303)
REF_SHA = 'e93c575050f2fa0a3b1e312830fe4b622bc33d0a3fa76efeaf77c7484a168761'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_image(path, expected):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError(f'Image SHA changed: {path}')
    with Image.open(path) as image:
        image.load()
        if image.mode != 'RGB' or image.size != (512, 512):
            raise ValueError(f'Expected native RGB512: {path}')
        return image.copy()


def journal_snapshot(path):
    """Snapshot complete binary lines; a partial UTF-8 tail is still pending."""
    raw = Path(path).read_bytes()
    boundary = raw.rfind(b'\n') + 1
    complete = raw[:boundary]
    rows = {}
    for line in complete.splitlines():
        row = json.loads(line.decode('utf-8', errors='strict'))
        rows[(row['case_id'], row['seed'], row['arm'])] = row
    return rows, {'complete_prefix_bytes': len(complete), 'pending_tail_bytes': len(raw) - len(complete),
                  'complete_prefix_sha256': hashlib.sha256(complete).hexdigest()}


def render(journal, output):
    journal, output = Path(journal).resolve(), Path(output).resolve()
    if HERE not in journal.parents or HERE not in output.parents:
        raise ValueError('Read current experiment and write QA inside this supplement only')
    if output.exists():
        raise FileExistsError('Use a new QA snapshot; no overwrite')
    refs_path = EXP / 'completion_20260915/evaluation/references_v1/references.json'
    if digest(refs_path) != REF_SHA:
        raise ValueError('Historical reference manifest changed')
    refs = json.loads(refs_path.read_text(encoding='utf-8'))
    cases = [c for c in refs['cases'] if c['source_labels_12'] != c['target_labels_12']]
    rows, snapshot = journal_snapshot(journal)
    groups = defaultdict(list)
    for case in cases:
        groups[case['subject']].append(case)
    output.mkdir(parents=True)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    font = ImageFont.truetype(font_path, 12)
    title_font = ImageFont.truetype(font_path, 15)
    cell, gap, row_height = 160, 8, 194
    receipt = {'kind': 'raw_image_visual_audit', 'journal': str(journal),
               'journal_snapshot': snapshot,
               'selection': 'all frozen subjects/cases and all three seeds; no score selection',
               'display': 'whole RGB512 image resized to square160 with identical LANCZOS; no crop or retouch',
               'references_sha256': REF_SHA, 'sheets': [], 'checked_outputs': [], 'pending_or_failed': []}
    for subject, subject_cases in groups.items():
        canvas = Image.new('RGB', (6 * (cell + gap) + gap, 50 + len(subject_cases) * 3 * row_height), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 4), subject + ' | All fixed cases / seeds | Visual QA, not blinded human AU rating', fill='black', font=title_font)
        for col, label in enumerate(LABELS):
            draw.text((gap + col * (cell + gap), 28), label, fill='black', font=font)
        for case_index, case in enumerate(subject_cases):
            cid = case['case_id']
            for seed_index, seed in enumerate(SEEDS):
                y = 50 + (case_index * 3 + seed_index) * row_height
                aus = ','.join(f'{au}:{v:g}' for au, v in zip((1,2,4,5,6,9,12,15,17,20,25,26), case['target_labels_12']) if v)
                draw.text((gap, y), f'{cid} | seed {seed} | AU {aus}', fill='black', font=font)
                for col in range(6):
                    x = gap + col * (cell + gap)
                    if col < 2:
                        ref = case['source_reference' if col == 0 else 'target_reference']['image']
                        image = checked_image(ref['path'], ref['sha256'])
                    else:
                        arm = ARMS[col - 2]
                        row = rows.get((cid, seed, arm))
                        if not row or row['status'] != 'success':
                            status = 'pending' if not row else row['status']
                            draw.rectangle((x, y+22, x+cell, y+22+cell), fill='#eeeeee')
                            draw.text((x+8, y+85), status, fill='#a00000', font=title_font)
                            receipt['pending_or_failed'].append({'case_id': cid, 'seed': seed, 'arm': arm, 'status': status})
                            continue
                        image = checked_image(row['output_path'], row['output_sha256'])
                        receipt['checked_outputs'].append({'case_id': cid, 'seed': seed, 'arm': arm,
                                                           'path': row['output_path'], 'sha256': row['output_sha256']})
                    canvas.paste(image.resize((cell, cell), Image.Resampling.LANCZOS), (x, y + 22))
        path = output / (subject + '.png')
        with path.open('xb') as stream:
            canvas.save(stream, format='PNG')
        receipt['sheets'].append({'subject': subject, 'path': str(path), 'sha256': digest(path)})
    receipt['checked_output_count'] = len(receipt['checked_outputs'])
    receipt['missing_count'] = len(receipt['pending_or_failed'])
    with (output / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'checked': receipt['checked_output_count'], 'pending_or_failed': receipt['missing_count'],
                      'sheets': len(receipt['sheets']), 'receipt': str(output / 'receipt.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.journal, args.output)
