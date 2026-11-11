"""All four fixed edited cases, original images only; compact comparison view."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); run = a.run.resolve()
    assert run.parent == HERE / 'runs'
    doc = json.loads((run / 'cases.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    assert result['status'] == 'complete' and result['cases']['sha256'] == sha(run / 'cases.json')
    cases = [r for r in doc['cases'] if r['group'] == 'lowest_available']
    assert [r['subject'] for r in cases] == ['SN004','SN012','SN017','SN018']
    rows = {(r['case_id'], r['method']): r for r in result['records']}
    cols = [('source_real', '源图'), ('target_real', '真实目标'), ('bidirectional_text2img', '原双向：噪声起点'),
            ('bidirectional_img2img_s035', '源图起点：强度0.35'), ('bidirectional_img2img_s055', '源图起点：强度0.55'),
            ('bidirectional_img2img_s075', '源图起点：强度0.75')]
    font = '/System/Library/Fonts/STHeiti Medium.ttc'
    title, body, small = [ImageFont.truetype(font, n) for n in (30, 22, 18)]
    cell, gap, margin = 256, 10, 20
    width = margin * 2 + cell * 6 + gap * 5
    page = Image.new('RGB', (width, 110 + 4 * 323 + 48), 'white'); draw = ImageDraw.Draw(page)
    draw.text((margin, 12), '源图初始化诊断｜4名开发被试的全部固定编辑案例', font=title, fill='#15263e')
    draw.text((margin, 52), '同一双向检查点；不训练、不修图。这里只比较初始化方式，全部零编辑与基底对照另见完整图。', font=small, fill='#44556a')
    for i, (_, name) in enumerate(cols):
        draw.text((margin + i * (cell + gap), 80), name, font=body, fill='#15263e')
    receipt = []
    for ri, c in enumerate(cases):
        y = 112 + ri * 323
        active = ', '.join(f'AU{au}={v}' for au, v in zip([1,2,4,5,6,9,12,15,17,20,25,26], c['target_aus_0_5']) if v)
        draw.text((margin, y), c['subject'] + '  ' + active, font=body, fill='#263951')
        for ci, (key, _) in enumerate(cols):
            ref = c.get(key)
            if ref is None:
                r = rows[c['case_id'], key]; ref = {'path': r['output'], 'sha256': r['sha256']}
            assert sha(ref['path']) == ref['sha256']
            im = Image.open(ref['path']); assert im.mode == 'RGB' and im.size == (512,512)
            page.paste(im.resize((cell,cell), Image.Resampling.LANCZOS), (margin + ci * (cell + gap), y + 32))
            receipt.append({'case_id': c['case_id'], 'column': key, **ref})
        draw.line((margin, y + 302, width - margin, y + 302), fill='#dde3eb')
    draw.text((margin, page.height - 36), '固定seed101；开发集诊断，不是独立测试。三档实际去噪步数7 / 11 / 15；原图与所有指标均保留。', font=small, fill='#44556a')
    output = HERE / 'rendered' / (run.name + '_compact')
    output.mkdir(parents=True, exist_ok=False)
    page.save(output / 'all_four_edited_cases.png')
    with (output / 'index.json').open('x') as f:
        json.dump({'status':'rendered_pending_review','source_cells':receipt,
                   'image':{'path':str(output / 'all_four_edited_cases.png'), 'sha256':sha(output / 'all_four_edited_cases.png')},
                   'script_sha256':sha(__file__), 'retouching':False, 'case_selection':'all four preselected edited cases'},f,ensure_ascii=False,indent=2)
    print(output / 'all_four_edited_cases.png')
