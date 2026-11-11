"""Render every sealed dev request for visual inspection, never alter outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
METHODS = [('source_real', '真实源图'), ('target_real', '真实目标'),
           ('baseline', '公开基底'), ('current_model', '双向模块（200步）'),
           ('liveportrait', 'LivePortrait'), ('faceadapter', 'Face-Adapter')]
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    doc = json.loads(args.results.read_text())
    if doc['test_used'] or doc['case_count'] != 19:
        raise ValueError('Only the existing dev19 sealed comparison is supported')
    rows = doc['records']
    by_key = {(r['case_id'], r['method']): r for r in rows}
    cases = list(dict.fromkeys(r['case_id'] for r in rows))
    if len(cases) != 19 or len(by_key) != len(rows):
        raise ValueError('Duplicate or missing cases')
    used = [by_key[(case, method)] for case in cases for method, _ in METHODS]
    if any('_step200_' not in Path(by_key[(case, 'current_model')]['path']).name for case in cases):
        raise ValueError('This fixed-caption renderer is only for step200 development images')
    for row in used:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Input image SHA changed: ' + row['path'])
    args.output.mkdir(parents=True, exist_ok=False)
    font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    tile, gap, row_h, top = 256, 10, 318, 42
    width = len(METHODS) * (tile + gap) + gap
    output_files = []
    for start in range(0, len(cases), 3):
        subset = cases[start:start + 3]
        sheet = Image.new('RGB', (width, top + row_h * len(subset)), 'white')
        draw = ImageDraw.Draw(sheet)
        for column, (_, label) in enumerate(METHODS):
            draw.text((gap + column * (tile + gap), 10), label, fill='black', font=font)
        for index, case in enumerate(subset):
            target = by_key[(case, 'target_real')]
            labels = target['target_labels_12']
            action = '，'.join(f'AU{au}={value:g}' for au, value in zip(AUS, labels) if value)
            title = f"{case}  |  {action or '不编辑'}"
            y = top + index * row_h
            draw.text((gap, y), title, fill='black', font=small)
            for column, (method, _) in enumerate(METHODS):
                row = by_key[(case, method)]
                with Image.open(row['path']) as original:
                    original = original.convert('RGB')
                    display = original.copy()
                    display.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                x = gap + column * (tile + gap)
                sheet.paste(display, (x + (tile-display.width)//2, y+30+(tile-display.height)//2))
        path = args.output / f'dev_cases_{start+1:02d}_{start+len(subset):02d}.png'
        sheet.save(path)
        output_files.append({'path': str(path.resolve()), 'sha256': digest(path), 'cases': subset})
    for row in used:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Input image changed during rendering')
    manifest = {'purpose': 'development visual review only', 'test_used': False,
                'renderer_sha256': digest(__file__), 'trained_steps_in_caption': 200,
                'source_results_sha256': digest(args.results), 'cases': cases,
                'methods': METHODS, 'display_only_resize': 'PIL thumbnail LANCZOS, aspect ratio preserved',
                'source_images_modified': False, 'outputs': output_files}
    with (args.output/'manifest.json').open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'complete', 'cases': len(cases), 'sheets': output_files}, ensure_ascii=False))


if __name__ == '__main__':
    main()
