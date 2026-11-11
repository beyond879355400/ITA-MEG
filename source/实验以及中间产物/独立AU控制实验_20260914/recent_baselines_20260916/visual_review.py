"""Contact sheets for every declared baseline output; never edits source PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent


def sha(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        remaining = before.st_size
        while remaining:
            block = f.read(min(1024 * 1024, remaining))
            assert block, 'Short read; retry materialization without accepting empty content'
            h.update(block)
            remaining -= len(block)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--expected', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    document = json.loads(args.index.read_text())
    assert document['status'] in ('complete', 'completed_with_failures')
    rows = document['records']
    assert len(rows) == args.expected
    assert len({(r['case_id'], r['seed']) for r in rows}) == len(rows)
    output = args.output.resolve()
    assert output.is_relative_to(HERE) and not output.exists()
    font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    font = ImageFont.truetype(font_path, 20)
    heading = ImageFont.truetype(font_path, 28)
    output.mkdir(parents=True)
    sheets, checked = [], []
    for offset in range(0, len(rows), 6):
        sheet = Image.new('RGB', (1060, 1810), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 12), f"{document['method']}：全量输出检查 {offset+1}—{min(offset+6,len(rows))}", font=heading, fill='black')
        for index, row in enumerate(rows[offset:offset+6]):
            x, y = 12 + (index % 2)*524, 62 + (index // 2)*580
            seed = '确定性输出' if row['seed'] is None else f"随机种子 {row['seed']}"
            draw.text((x, y), row['case_id'], font=font, fill='black')
            draw.text((x, y+25), seed, font=font, fill='black')
            record = {'case_id': row['case_id'], 'seed': row['seed'], 'status': row['status']}
            if row['status'] == 'complete':
                reference = row['output']
                path = Path(reference['path'])
                assert sha(path) == reference['sha256'], 'Source image hash mismatch'
                with Image.open(path) as source:
                    assert source.format == 'PNG' and source.mode == 'RGB'
                    original_size = source.size
                    tile = ImageOps.contain(source, (512, 512), Image.Resampling.LANCZOS)
                    sheet.paste(tile, (x+(512-tile.width)//2, y+52+(512-tile.height)//2))
                record.update(path=str(path), sha256=reference['sha256'], image_size=original_size)
            else:
                draw.text((x+12, y+130), f"未生成：{row['status']}", font=heading, fill='#a00000')
                record['failure_reason'] = row.get('failure_reason')
            checked.append(record)
        filename = output/f"sheet_{offset//6+1:02d}.png"
        sheet.save(filename)
        sheets.append({'path': str(filename), 'sha256': sha(filename)})
    (output/'manifest.json').write_text(json.dumps({
        'kind': 'all_output_visual_contact_sheets_v1', 'source_index': str(args.index),
        'source_index_sha256': sha(args.index), 'records': checked, 'sheets': sheets,
        'source_images_modified': False, 'image_selection': 'every declared record in index order',
        'display_only': 'aspect-preserving fit, no crop, no enhancement; not used for metrics',
        'visual_inspection_done': False,
    }, ensure_ascii=False, indent=2))
    print(json.dumps({'rows': len(rows), 'sheets': len(sheets), 'output': str(output)}))


if __name__ == '__main__':
    main()
