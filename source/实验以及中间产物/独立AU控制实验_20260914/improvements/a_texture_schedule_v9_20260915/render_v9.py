"""Terminal-only V9 layouts: every dev case, source-fixed debug crops, no models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / 'region_preserving_v6_20260915/render_v6.py'
TEMPLATE_SHA = '107143fe267144d9c8ac5b500faa5395f9179008038855ccdc0367133ab690d0'
V7RUN = HERE.parent / 'smooth_regions_v7_20260915/runs/smooth_202609151710'
V7_RESULT_SHA = 'e59ce11ad87ff6bba91c8131019997833fee0f482a3df364e2d6f28e80aa079d'
V7_CASES_SHA = '5335ff4c2b41585b4ab83ddfe1253edf1de127b01ddb9190cea542c0078c3c90'
CONFIGS = (('late_au100', 1.), ('late_au050', .5), ('late_au000', 0.))
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
ROI_POLICY = {
    'SN012': {'label': '嘴与下巴', 'indices': list(range(5, 12)) + list(range(48, 68))},
    'SN018': {'label': '眉毛与眉心', 'indices': list(range(17, 28))},
}
ROI_PADDING = 20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


if sha(TEMPLATE) != TEMPLATE_SHA:
    raise ValueError('Frozen scientific renderer changed')
spec = importlib.util.spec_from_file_location('v9_readonly_scientific_layout', TEMPLATE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
layout = base.layout
need, read, ref = base.need, base.read, base.ref
SUBJECTS, GROUPS, CELL, GAP, MARGIN, AUS = base.SUBJECTS, base.GROUPS, base.CELL, base.GAP, base.MARGIN, base.AUS


def columns(overview=False):
    cols = [
        {'key': 'source_real', 'kind': 'reference', 'lines': ['来源图', '原始对齐图', '仅参照 · 无去噪']},
        {'key': 'target_real', 'kind': 'reference', 'lines': ['真实目标', '同一固定案例', '仅参照 · 非生成图输入']},
        {'key': 'a_bidirectional', 'kind': 'a', 'lines': ['原A · 双向', '15步均AU×1.5', '强度.75 / 名义20实际15']},
    ]
    if not overview:
        cols.append({'key': 'a_baseline', 'kind': 'a', 'lines': ['原A · 基底', '15步均AU×1.5', '强度.75 / 名义20实际15']})
    for config, late in CONFIGS:
        methods = [('bidirectional', '双向')] if overview else [('baseline', '基底'), ('bidirectional', '双向')]
        for method, label in methods:
            cols.append({'key': method + '_' + config, 'kind': 'v9', 'config': config, 'late': late,
                         'lines': [f'V9 · {label} · 末AU×{late:g}', '前10步AU×1.5', f'末5步AU×{late:g} / 实际15步']})
    need(len(cols) == (6 if overview else 10), 'Unexpected display column count')
    return cols


def bound_doc(run, result, key, filename):
    path = run / filename
    binding = result[key]
    need(Path(binding['path']).resolve() == path and sha(path) == binding['sha256'], key + ' binding changed')
    return read(path)


def load_inputs(run):
    need(run.parent == HERE / 'runs', 'Input must be a direct v9 runs child')
    need((run / 'result.json').is_file(), 'No terminal result.json; active generation cannot be rendered')
    result = read(run / 'result.json')
    need(result.get('kind') == 'v9_late_au_diagnostic' and result.get('status') in
         ('complete', 'completed_with_failures'), 'Not a terminal V9 result')
    need(result.get('test_data_used') is False and result.get('planned') == 48, 'Expected dev-only 48 outputs')
    document = bound_doc(run, result, 'cases', 'cases.json')
    config = bound_doc(run, result, 'configuration', 'configuration.json')
    need(document.get('kind') == 'v9_development_late_au_cases' and document.get('test_data_used') is False,
         'Wrong cases protocol')
    expected = [{'name': n, 'strength': .75, 'au_scale_initial': 1.5, 'au_scale_late': a,
                 'au_late_max_t': 250} for n, a in CONFIGS]
    need(config.get('kind') == 'v9_late_au_diagnostic' and config.get('configs') == expected, 'Schedule matrix changed')
    need(all(config.get(k) == v for k, v in {'seed': 101, 'nominal_steps': 20, 'guidance_scale': 7.5,
         'ip_scale': .9, 'test_data_used': False, 'training_performed': False,
         'post_generation_pixel_blend': False}.items()), 'Shared configuration changed')
    cases = document['cases']
    need(document.get('case_count') == len(cases) == 8 and len({c['case_id'] for c in cases}) == 8,
         'All eight unique cases required')
    need(all(c['split'] == 'dev' for c in cases), 'Non-dev case')
    need(Counter((c['subject'], c['group']) for c in cases) ==
         Counter((s, g) for s in SUBJECTS for g in GROUPS), 'All four subjects and both case groups required')
    need(sha(V7RUN / 'result.json') == V7_RESULT_SHA and sha(V7RUN / 'cases.json') == V7_CASES_SHA,
         'Original A metadata changed')
    a_result = read(V7RUN / 'result.json')
    need(a_result['status'] == 'complete' and a_result['test_data_used'] is False, 'A must be complete dev data')
    a_rows = {(r['case_id'], r['method']): r for r in a_result['records']}
    a_cases = {c['case_id']: c for c in read(V7RUN / 'cases.json')['cases']}
    for case in cases:
        previous = a_cases[case['case_id']]
        need(all(case[k] == previous[k] for k in ('subject', 'split', 'group', 'source_real', 'target_real',
             'source_aus_0_5', 'target_aus_0_5', 'prompt', 'region_mask')), 'Case/source/target differs from A')
        for method in ('baseline', 'bidirectional'):
            row = a_rows[case['case_id'], method + '_smooth_s075_a150']
            need(row['status'] == 'complete' and row['strength'] == .75 and row['au_scale'] == 1.5
                 and row['actual_timesteps'] == TIMESTEPS, 'Original A schedule differs')
            need(case['a_' + method] == {'path': row['output'], 'sha256': row['sha256']}, 'A image binding differs')
    allowed = {c['key']: c for c in columns() if c['kind'] == 'v9'}
    known = {c['case_id']: c for c in cases}
    records = {}
    for row in result['records']:
        key = row['case_id'], row['method']
        need(key[0] in known and key[1] in allowed and key not in records, 'Unexpected/duplicate generated row')
        col, case = allowed[key[1]], known[key[0]]
        need(all(row[k] == case[k] for k in ('subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5', 'prompt')),
             'Generated row does not match case')
        need(all(row.get(k) == v for k, v in {'config': col['config'], 'seed': 101, 'strength': .75,
             'au_scale_initial': 1.5, 'au_scale_late': col['late'], 'au_late_max_t': 250,
             'nominal_steps': 20, 'ip_scale': .9}.items()), 'Record label/schedule differs')
        need(Path(row['output']).resolve() == run / 'images' / key[0] / (key[1] + '.png'),
             'Generated image must belong to this exact run/case/method')
        if row['status'] == 'complete':
            need(row['actual_steps'] == 15 and row['actual_timesteps'] == TIMESTEPS, 'Unexpected actual steps')
            trace = [{'t': t, 'au_scale': 1.5 if t > 250 else col['late']} for t in TIMESTEPS]
            need(row['au_scale_trace'] == trace, 'Actual UNet schedule differs from visible label')
            if case['group'] == 'no_edit':
                need(row['sha256'] == case['vae_reconstruction']['sha256'], 'Noedit differs from source VAE reconstruction')
        records[key] = row
    need(len(records) == 48, 'All 48 terminal rows required, including failures')
    need(result['completed'] == sum(r['status'] == 'complete' for r in records.values()), 'Completed count mismatch')
    if result['status'] == 'complete':
        need(result['completed'] == 48, 'Complete status must include 48 successful rows')
    return cases, records, result


def cell_source(case, col, records):
    if col['kind'] in ('reference', 'a'):
        value = case[col['key']]
        return {'path': value['path'], 'expected_sha256': value['sha256'],
                'generation_status': 'reference', 'actual_steps': 15 if col['kind'] == 'a' else None}
    row = records.get((case['case_id'], col['key']))
    if row is None:
        return {'path': None, 'expected_sha256': None, 'generation_status': 'missing_record', 'actual_steps': None}
    return {'path': row.get('output'), 'expected_sha256': row.get('sha256'),
            'generation_status': row['status'], 'actual_steps': row.get('actual_steps'), 'error': row.get('error')}


def render_page(name, selected, cols, records, output, fonts, overview=False):
    width = 2 * MARGIN + len(cols) * CELL + (len(cols) - 1) * GAP
    height = 256 + len(selected) * 398 + 120
    page = Image.new('RGB', (width, height), layout.BACKGROUND)
    draw = ImageDraw.Draw(page)
    title = '4个编辑案例 · 原A与3种末段AU' if overview else selected[0]['subject'] + ' · 零编辑与编辑全配置'
    draw.text((MARGIN, 18), 'V9｜' + title, font=fonts[42], fill=layout.TEXT)
    draw.text((MARGIN, 74), '新候选：前10步AU×1.5，末5步分别AU×1 / ×0.5 / ×0；名义20步，强度.75，实际15步。', font=fonts[22], fill=layout.MUTED)
    draw.text((MARGIN, 112), '完整原图等比排版，不裁脸、不修图、不筛图；固定seed101 / CFG7.5 / IP.9；仅开发诊断。', font=fonts[22], fill=layout.MUTED)
    for j, col in enumerate(cols):
        x = MARGIN + j * (CELL + GAP)
        draw.rounded_rectangle((x, 157, x + CELL, 252), radius=8,
                               fill='#e0e9f5' if col['kind'] == 'v9' else '#e6e9ef')
        for k, line in enumerate(col['lines']):
            layout.centered(draw, line, x, 164 + k * 28, fonts[20])
    cells = []
    for i, case in enumerate(selected):
        top = 267 + i * 398
        changed = [f'AU{au}:{s:g}→{t:g}' for au, s, t in zip(AUS, case['source_aus_0_5'], case['target_aus_0_5']) if s != t]
        group = '零编辑' if case['group'] == 'no_edit' else '编辑'
        note = ' / '.join(changed) or '空mask回到来源VAE重建，为构造保证'
        draw.text((MARGIN, top), f"{case['subject']} · {group}｜{case['case_id']}｜{note}", font=fonts[22], fill=layout.TEXT)
        for j, col in enumerate(cols):
            info = layout.paste_cell(page, draw, MARGIN + j * (CELL + GAP), top + 37,
                                     cell_source(case, col, records), fonts)
            info.update(case_id=case['case_id'], subject=case['subject'], group=case['group'], row=i,
                        column=j, column_key=col['key'], column_label=col['lines'], crop=False)
            cells.append(info)
    y = height - 100
    draw.line((MARGIN, y, width - MARGIN, y), fill='#cbd3df', width=2)
    draw.text((MARGIN, y + 15), '原A始终AU×1.5；V9仅降低实际t≤250的末5步AU权重，AU标签与其它条件不变。', font=fonts[20], fill=layout.MUTED)
    draw.text((MARGIN, y + 48), '缺失、失败、SHA不符保留占位。全8案例、全部48新图见4张主体详图；原图与页面SHA见index。', font=fonts[20], fill=layout.MUTED)
    path = output / (name + '.png')
    with path.open('xb') as handle:
        page.save(handle, format='PNG')
    return {**ref(path), 'kind': 'overview' if overview else 'subject', 'size': list(page.size),
            'case_ids': [c['case_id'] for c in selected], 'columns': cols, 'cells': cells}


def roi_bounds(points, indices, padding=ROI_PADDING):
    need(len(points) == 68 and all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in points),
         'Invalid source 68 points')
    chosen = [points[i] for i in indices]
    bounds = [math.floor(min(p[0] for p in chosen) - padding), math.floor(min(p[1] for p in chosen) - padding),
              math.ceil(max(p[0] for p in chosen) + padding), math.ceil(max(p[1] for p in chosen) + padding)]
    clipped = [max(0, min(512, n)) for n in bounds]
    need(clipped[0] < clipped[2] and clipped[1] < clipped[3], 'Empty source-fixed ROI')
    return bounds, clipped


def source_roi(case):
    binding = case['region_mask']['receipt']
    need(sha(binding['path']) == binding['sha256'], 'Source geometry receipt changed')
    doc = read(binding['path'])
    need(doc['case_id'] == case['case_id'] and doc['source_sha256'] == case['source_real']['sha256']
         and Path(doc['source_path']).resolve() == Path(case['source_real']['path']).resolve(), 'ROI belongs to another source')
    need(doc['source_only'] is True and doc['test_used'] is False and doc['target_geometry_used'] is False,
         'ROI must use source-only development geometry')
    policy = ROI_POLICY[case['subject']]
    raw, clipped = roi_bounds(doc['source_landmarks_68_512'], policy['indices'])
    return {'case_id': case['case_id'], 'subject': case['subject'], 'label': policy['label'],
            'source': case['source_real'], 'receipt': binding, 'source_landmarks_sha256': doc['source_landmarks_sha256'],
            'zero_based_landmark_indices': policy['indices'], 'padding_pixels': ROI_PADDING,
            'bbox_xyxy_unclipped': raw, 'bbox_xyxy': clipped, 'source_size': [512, 512],
            'coordinate_rule': 'floor(min-20),ceil(max+20),clip[0,512]; same box for every method',
            'source_only': True, 'output_adaptive': False, 'for_display_only': True, 'sent_to_evaluation': False}


def paste_crop(page, draw, x, y, source, roi, fonts, size=512):
    info = dict(source)
    info.update(crop=True, crop_box_xyxy=roi['bbox_xyxy'], debug_only=True, sent_to_evaluation=False)
    draw.rectangle((x, y, x + size, y + size), fill='white', outline='#cbd3df', width=2)
    error = None
    if source['generation_status'] not in ('reference', 'complete'):
        error = '生成失败／缺失记录，保留占位'
    elif not source.get('path') or not Path(source['path']).is_file():
        error = '原图文件缺失，保留占位'
    else:
        info['actual_sha256'] = sha(source['path'])
        if info['actual_sha256'] != source.get('expected_sha256'):
            error = '原图SHA不符，保留占位'
        else:
            try:
                with Image.open(source['path']) as original:
                    need(original.size == (512, 512), 'Debug ROI requires canonical 512x512 images')
                    original.load()
                    cropped = original.crop(tuple(roi['bbox_xyxy'])).convert('RGB')
                    scale = min(size / cropped.width, size / cropped.height)
                    shown = cropped.resize((round(cropped.width * scale), round(cropped.height * scale)), Image.Resampling.LANCZOS)
                    px, py = x + (size - shown.width) // 2, y + (size - shown.height) // 2
                    page.paste(shown, (px, py))
                    info.update(source_size=list(original.size), crop_size=list(cropped.size), display_scale=scale,
                                display_box=[px, py, shown.width, shown.height], render_status='shown')
            except (OSError, ValueError) as failure:
                error = '原图读取／尺寸失败，保留占位'
                info['render_error'] = str(failure)
    if error:
        draw.text((x + 18, y + size // 2), error, font=fonts[22], fill=layout.ERROR)
        info.update(render_status='placeholder', placeholder_reason=error)
    digest = info.get('actual_sha256', '')
    draw.text((x + 14, y + size + 10), '固定来源ROI · 仅展示裁剪 · SHA ' + digest[:12], font=fonts[18], fill=layout.MUTED)
    return info


def render_debug(cases, records, output, fonts):
    selected = [next(c for c in cases if c['subject'] == s and c['group'] == 'lowest_available') for s in ROI_POLICY]
    cols = columns(True)[2:]
    size, gap = 512, 18
    width, height = MARGIN * 2 + 4 * size + 3 * gap, 174 + 2 * 625 + 94
    page = Image.new('RGB', (width, height), layout.BACKGROUND)
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, 18), 'DEBUG｜SN012嘴下巴 / SN018眉心 · 来源固定ROI放大', font=fonts[42], fill=layout.TEXT)
    draw.text((MARGIN, 78), '原A与3个新双向使用同一来源关键点框，固定外扩20px；仅展示裁剪，不修图、不按结果重选、不送评价。', font=fonts[22], fill=layout.MUTED)
    for j, col in enumerate(cols):
        draw.text((MARGIN + j * (size + gap) + 12, 129), col['lines'][0], font=fonts[25], fill=layout.TEXT)
    cells, rois = [], []
    for i, case in enumerate(selected):
        roi = source_roi(case)
        rois.append(roi)
        top = 184 + i * 625
        draw.text((MARGIN, top), f"{case['subject']} · {roi['label']}｜source bbox {roi['bbox_xyxy']}｜{case['case_id']}", font=fonts[22], fill=layout.TEXT)
        for j, col in enumerate(cols):
            info = paste_crop(page, draw, MARGIN + j * (size + gap), top + 42,
                              cell_source(case, col, records), roi, fonts, size)
            info.update(case_id=case['case_id'], subject=case['subject'], group=case['group'], row=i,
                        column=j, column_key=col['key'], column_label=col['lines'])
            cells.append(info)
    draw.text((MARGIN, height - 69), '新候选前10步AU×1.5；末5步分别AU×1 / ×0.5 / ×0。原A15步均AU×1.5；均名义20、实际15步。', font=fonts[22], fill=layout.MUTED)
    draw.text((MARGIN, height - 37), '局部观察不能代替全图、AU动作、身份与非目标漂移验收；全8case无筛选见主体详图。', font=fonts[20], fill=layout.MUTED)
    path = output / 'debug_source_fixed_rois.png'
    with path.open('xb') as handle:
        page.save(handle, format='PNG')
    return {**ref(path), 'kind': 'debug_crop_only', 'size': list(page.size), 'case_ids': [c['case_id'] for c in selected],
            'columns': cols, 'cells': cells, 'rois': rois, 'sent_to_evaluation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Terminal direct V9 runs child; no models loaded')
    args = parser.parse_args()
    run = args.run.resolve()
    cases, records, result = load_inputs(run)
    bindings = {str(p): sha(p) for p in (run / 'cases.json', run / 'result.json', run / 'configuration.json',
                                        V7RUN / 'cases.json', V7RUN / 'result.json', Path(__file__), TEMPLATE, base.HELPER)}
    fonts, font_path = layout.make_fonts()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = HERE / 'rendered' / f'{run.name}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    edited = [next(c for c in cases if c['subject'] == s and c['group'] == 'lowest_available') for s in SUBJECTS]
    pages = [render_page('overview_all_four_edited', edited, columns(True), records, output, fonts, True)]
    pages += [render_page(s + '_all_cases_all_methods', base.ordered_cases(cases, [s]), columns(), records, output, fonts)
              for s in SUBJECTS]
    pages.append(render_debug(cases, records, output, fonts))
    observed = {cell['path']: cell['actual_sha256'] for page in pages for cell in page['cells'] if cell.get('actual_sha256')}
    receipts = {r['receipt']['path']: r['receipt']['sha256'] for r in pages[-1]['rois']}
    need(all(sha(p) == h for p, h in {**bindings, **observed, **receipts}.items()), 'Input changed during rendering')
    detail_cells = [c for p in pages if p['kind'] == 'subject' for c in p['cells']]
    need(len(detail_cells) == 80 and len({(c['case_id'], c['column_key']) for c in detail_cells}) == 80,
         'All eight cases and ten columns must be represented')
    overview = {k: pages[0][k] for k in ('path', 'sha256', 'size')}
    manifest = {'kind': 'v9_all_case_scientific_layouts', 'status': 'rendered_pending_visual_review',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run), 'generation_status': result['status'],
        'models_loaded': False, 'test_data_used': False, 'source_files_modified': False,
        'source_hashes_rechecked_after_render': True, 'selection': 'all8dev, all4subjects, bothgroups, all3configs, bothmethods; no quality filtering',
        'layout': {'overview_rows': 4, 'overview_columns': 6, 'subject_pages': 4, 'subject_rows': 2, 'subject_columns': 10,
                   'whole_image_letterbox': True, 'retouch': False, 'debug_rows': 2, 'debug_columns': 4,
                   'debug_crop_only': True, 'debug_crops_sent_to_evaluation': False},
        'planned_generated_images': 48, 'generation_record_count': len(records), 'case_ids': [c['case_id'] for c in base.ordered_cases(cases, SUBJECTS)],
        'shown_detail_cells': sum(c['render_status'] == 'shown' for c in detail_cells),
        'placeholder_detail_cells': sum(c['render_status'] == 'placeholder' for c in detail_cells),
        'metadata_and_code_bindings': [{'path': p, 'sha256': h} for p, h in bindings.items()],
        'all_observed_image_bindings': [{'path': p, 'sha256': h} for p, h in sorted(observed.items())],
        'source_geometry_receipt_bindings': [{'path': p, 'sha256': h} for p, h in receipts.items()],
        'font': ref(font_path), 'overview': overview, 'pages': pages}
    index = output / 'index.json'
    with index.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': manifest['status'], 'index': ref(index), 'overview': overview, 'page_count': len(pages),
                      'placeholder_detail_cells': manifest['placeholder_detail_cells']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
