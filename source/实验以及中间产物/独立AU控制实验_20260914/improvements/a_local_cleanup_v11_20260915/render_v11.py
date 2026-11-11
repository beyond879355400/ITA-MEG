"""V11 complete scientific layouts:32new plus A and uniform V10 global-LP refs."""
import hashlib
import math
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V9_RENDERER = HERE.parent / 'a_texture_schedule_v9_20260915/render_v9.py'
V9_RENDERER_SHA = '507b23e265e9988da214a987f6dabad3f39e576e42657cbeacbe61ac7e675335'
V10RUN = HERE.parent / 'a_terminal_cleanup_v10_20260915/runs/terminal_202609151940'
V10_RESULT_SHA = '09c4ba3ce187f4cd6cf68b499f2fc5ac27482737b7133344bdf63e129e30f697'
V10_CASES_SHA = '227a371225b21066656863cabcede26a3ababb7c6ca785645e02515387ed6982'
MODES = (('local_lp_s050', .5, 2), ('local_lp_s075', .75, 3))
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


need(sha(V9_RENDERER) == V9_RENDERER_SHA, 'Frozen V9 layout changed')
source = V9_RENDERER.read_text()
CHANGES = [
    ('Terminal-only V9 layouts: every dev case, source-fixed debug crops, no models.',
     'Terminal-only V11 layouts:32new, all dev cases, unchanged source-fixed debug crops, no models.'),
    ('4个编辑案例 · 原A与3种末段AU', '4个编辑案例 · 原A / 全域LP / 局部LP'),
    ("'V9｜'", "'V11｜'"),
    ('新候选：前10步AU×1.5，末5步分别AU×1 / ×0.5 / ×0；名义20步，强度.75，实际15步。',
     'V11局部σ.5/.75：AU×1.5、强度.75、15DEIS+0额外UNet；V10全域LP参照多1次t0，共16次UNet。'),
    ('完整原图等比排版，不裁脸、不修图、不筛图；固定seed101 / CFG7.5 / IP.9；仅开发诊断。',
     '仅整图排版；基于已看V10 dev的固定工程正则，不是未见dev的独立结果；不保证RGB眉眼唇完全不变。'),
    ('原A始终AU×1.5；V9仅降低实际t≤250的末5步AU权重，AU标签与其它条件不变。',
     'V11采用来源几何与统一AU触发，局部平滑眉心/唇下下巴的latent残差；不是RGB修图或已验证的自然度提升。'),
    ('缺失、失败、SHA不符保留占位。全8案例、全部48新图见4张主体详图；原图与页面SHA见index。',
     '缺失、失败、SHA不符保留占位。全8案例、全部32新图与统一V10参照见4张主体详图；全部SHA见index。'),
    ('新候选前10步AU×1.5；末5步分别AU×1 / ×0.5 / ×0。原A15步均AU×1.5；均名义20、实际15步。',
     'AU始终×1.5；原A/V11局部LP为15次UNet，V10全域LP为16次；名义20、实际DEIS15步。'),
    ('Terminal direct V9 runs child; no models loaded', 'Terminal direct V11 runs child; no models loaded'),
    ("'v9_all_case_scientific_layouts'", "'v11_all_case_scientific_layouts'"),
    ("V7RUN / 'cases.json', V7RUN / 'result.json', Path(__file__), TEMPLATE, base.HELPER)",
     "V7RUN / 'cases.json', V7RUN / 'result.json', Path(__file__), TEMPLATE, base.HELPER, V9_RENDERER)"),
    ("    fonts, font_path = layout.make_fonts()\n    stamp =", "    bindings.update(extra_binding_refs(cases, run))\n    fonts, font_path = layout.make_fonts()\n    stamp ="),
    ("'planned_generated_images': 48, 'generation_record_count': len(records)",
     "'forward_budget': {'v11_deis': 15, 'v11_extra': 0, 'v11_total': 15, 'v10_global_lp_total': 16},\n        'planned_generated_images': 32, 'generation_record_count': len(records)"),
    ("'all8dev, all4subjects, bothgroups, all3configs, bothmethods; no quality filtering'",
     "'all8dev, all4subjects, bothgroups, both2configs, bothmethods; uniform A and V10globalLP refs; no quality filtering'"),
    ("cell_source(case, col, records), fonts)\n            info.update(case_id=case['case_id']",
     "cell_source(case, col, records), fonts)\n            if col['kind'] in ('v11','v10_lp'): annotate_budget(page, draw, MARGIN + j * (CELL + GAP), top + 37, info, fonts)\n            info.update(case_id=case['case_id']"),
]
for old, new in CHANGES:
    need(source.count(old) == 1, 'Ambiguous V11 layout adaptation: ' + old)
    source = source.replace(old, new)
source = source.replace("col['kind'] == 'v9'", "col['kind'] == 'v11'")
R = ModuleType('v11_from_frozen_v9_layout')
R.__dict__.update(__file__=str(Path(__file__).resolve()), V9_RENDERER=V9_RENDERER)
exec(compile(source, str(V9_RENDERER), 'exec'), R.__dict__)


def columns(overview=False):
    cols = [
        {'key': 'source_real', 'kind': 'reference', 'lines': ['来源图', '原始对齐图', '仅参照 · 无去噪']},
        {'key': 'target_real', 'kind': 'reference', 'lines': ['真实目标', '同一固定案例', '仅参照 · 非生成图输入']},
        {'key': 'a_bidirectional', 'kind': 'a', 'lines': ['原A · 双向', 'AU×1.5 / 强度.75', '15 DEIS / 总UNet15']},
    ]
    if not overview:
        cols.append({'key': 'a_baseline', 'kind': 'a', 'lines': ['原A · 基底', 'AU×1.5 / 强度.75', '15 DEIS / 总UNet15']})
    references = [('bidirectional', '双向')] if overview else [('baseline', '基底'), ('bidirectional', '双向')]
    for family, label in references:
        cols.append({'key': 'v10_global_lp_' + family, 'kind': 'v10_lp',
                     'lines': [f'V10全域LP · {label}', 'σ.5 / 全编辑mask', '15 DEIS + 1额外UNet']})
    for mode, sigma, radius in MODES:
        families = [('bidirectional', '双向')] if overview else [('baseline', '基底'), ('bidirectional', '双向')]
        for family, label in families:
            cols.append({'key': family + '_' + mode, 'kind': 'v11', 'config': mode, 'sigma': sigma, 'radius': radius,
                         'lines': [f'V11 {label} · 局部σ{sigma:g}', '源几何 / AU×1.5', '15 DEIS + 0额外UNet']})
    need(len(cols) == (6 if overview else 10), 'Expected six/ten uniform columns')
    return cols


def filter_spec(sigma, radius):
    return {'sigma_latent': sigma, 'radius': radius, 'kernel_size': 2 * radius + 1, 'padding': 'reflect',
            'normalization': 'conv(editmask*delta)/conv(editmask)', 'denominator_epsilon': 1e-8,
            'denominator_fallback': 'original_delta',
            'local_blend': 'clean + cleanupmask * (filtered_delta - original_delta)',
            'outside_cleanup_exact': True, 'final_projection': 'original_editmask_once'}


def validate_terminal_row(row, col):
    need(all(row.get(k) == v for k, v in {'config': col['config'], 'seed': 101, 'strength': .75,
         'au_scale': 1.5, 'ip_scale': .9, 'nominal_steps': 20, 'cleanup_sigma': col['sigma']}.items()),
         'V11 visible configuration differs')
    if row['status'] != 'complete':
        return
    need(all(row.get(k) == v for k, v in {'actual_steps': 15, 'denoising_steps': 15, 'deis_unet_calls': 15,
         'terminal_unet_calls': 0, 'total_unet_calls': 15, 'actual_timesteps': TIMESTEPS}.items()), 'Wrong V11 call budget')
    calls = row['unet_call_trace']
    need([(c['phase'], c['t']) for c in calls] == [('deis', t) for t in TIMESTEPS]
         and all(c['cfg_batch'] == 2 and c['au_scale'] == 1.5 for c in calls), 'V11 UNet phase/CFG trace differs')
    terminal = row['terminal_trace']
    need(terminal['mode'] == col['config'] and terminal['last_deis_timestep'] == 50
         and terminal['epsilon_source'] == 'last_t50' and terminal['extra_cfg'] is None
         and terminal['terminal_raw_epsilon_sha256'] is None, 'V11 must reuse t50 epsilon without extra UNet')
    need(terminal['scheduler_step_index_before'] == terminal['scheduler_step_index_after'] == 20,
         'Terminal cleanup moved the scheduler')
    need(all(math.isfinite(terminal[k]) and terminal[k] > 0 for k in ('alpha0', 'sigma0', 'rho0'))
         and math.isclose(terminal['rho0'], terminal['sigma0'] / terminal['alpha0'], rel_tol=1e-6),
         'Invalid nonzero terminal VP coefficients')
    need(terminal['projected_latent_sha256'] == row['projection_trace'][-1]['latent_sha256'], 'Final latent binding differs')
    need(terminal['filter'] == filter_spec(col['sigma'], col['radius']), 'Local filter differs from fixed normalized rule')
    need(terminal['base_mode'] == 'terminal_last_eps' and terminal['outside_cleanup_clean_exact'] is True,
         'Expected unchanged last-epsilon clean outside local support')


def frozen_v10():
    need(sha(V10RUN / 'result.json') == V10_RESULT_SHA and sha(V10RUN / 'cases.json') == V10_CASES_SHA,
         'Fixed V10 comparison metadata changed')
    result = R.read(V10RUN / 'result.json')
    need(result['status'] == 'complete' and result['planned'] == result['completed'] == 48
         and result['test_data_used'] is False, 'V10 references must be complete development data')
    cases = R.bound_doc(V10RUN, result, 'cases', 'cases.json')['cases']
    rows = {(r['case_id'], r['method']): r for r in result['records']}
    need(len(rows) == 48 and len(cases) == 8, 'V10 reference count mismatch')
    return {c['case_id']: c for c in cases}, rows


def load_inputs(run):
    need(run.parent == HERE / 'runs' and (run / 'result.json').is_file(), 'Input must be a terminal direct V11 runs child')
    result = R.read(run / 'result.json')
    need(result.get('kind') == 'v11_local_cleanup_diagnostic' and result.get('status') in
         ('complete', 'completed_with_failures'), 'Not a terminal V11 diagnostic')
    need(result.get('test_data_used') is False and result.get('planned') == 32, 'Expected all32 development outputs')
    doc = R.bound_doc(run, result, 'cases', 'cases.json')
    config = R.bound_doc(run, result, 'configuration', 'configuration.json')
    need(doc['kind'] == 'v11_development_local_cleanup_cases' and doc['case_count'] == 8
         and doc['test_data_used'] is False, 'Wrong V11 dev-case schema')
    need(config['kind'] == 'v11_local_cleanup_diagnostic', 'Wrong V11 configuration kind')
    expected_configs = [{'name': n, 'strength': .75, 'au_scale': 1.5, 'terminal_unet_calls': 0, 'cleanup_sigma': s}
                        for n, s, _ in MODES]
    need(config['configs'] == expected_configs and config['filter_configs'] ==
         {n: filter_spec(s, r) for n, s, r in MODES}, 'Fixed local-filter configuration matrix differs')
    need(all(config.get(k) == v for k, v in {'guidance_scale': 7.5, 'ip_scale': .9, 'seed': 101,
         'nominal_steps': 20, 'denoising_steps': 15, 'post_generation_pixel_blend': False,
         'test_data_used': False, 'training_performed': False}.items()), 'Original A common protocol differs')
    previous, old_rows = frozen_v10()
    cases = doc['cases']
    need(len(cases) == 8 and len({c['case_id'] for c in cases}) == 8
         and {c['case_id'] for c in cases} == set(previous), 'All eight original cases required')
    for case in cases:
        need({k: v for k, v in case.items() if k != 'cleanup_mask'} == previous[case['case_id']],
             'Original V10 fields/source/A refs changed')
        need('cleanup_mask' in case, 'Missing source-only local cleanup binding')
    allowed = {c['key']: c for c in columns() if c['kind'] == 'v11'}
    known, records = {c['case_id']: c for c in cases}, {}
    for row in result['records']:
        key = row['case_id'], row['method']
        need(key[0] in known and key[1] in allowed and key not in records, 'Unexpected/duplicate V11 row')
        case = known[key[0]]
        need(all(row[k] == case[k] for k in ('subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5',
             'prompt', 'region_mask')), 'Generated row changed original case fields')
        family = 'bidirectional' if key[1].startswith('bidirectional_') else 'baseline'
        need(row['condition_hashes'] == case[family + '_text2img_condition_hashes'], 'Raw condition tensors changed')
        need(Path(row['output']).resolve() == run / 'images' / key[0] / (key[1] + '.png'), 'Wrong V11 image path')
        need(row['cleanup_mask'] == case['cleanup_mask'], 'Record local-support binding differs from case')
        validate_terminal_row(row, allowed[key[1]])
        if row['status'] == 'complete':
            prefix = old_rows[case['case_id'], family + '_terminal_last_eps']['projection_trace'][:14]
            need(row['a_prefix_verified'] is True and row['original_a_projection_prefix'] == row['projection_trace'][:14] == prefix,
                 'Original A first14 projected states changed')
            need(row['terminal_trace']['cleanup_mask'] == case['cleanup_mask'], 'Terminal cleanup support differs from case')
        if row['status'] == 'complete' and case['group'] == 'no_edit':
            need(row['sha256'] == case['vae_reconstruction']['sha256'], 'No-edit differs from source VAE reconstruction')
        records[key] = row
    need(len(records) == 32 and result['completed'] == sum(r['status'] == 'complete' for r in records.values()),
         'All32 terminal rows and correct completed count required')
    if result['status'] == 'complete':
        need(result['completed'] == 32, 'Complete must mean all32 successful')
    # Build display-only copies; cases.json and the user's original fields are never rewritten.
    display_cases = []
    for original in cases:
        case = dict(original)
        for family in ('baseline', 'bidirectional'):
            row = old_rows[case['case_id'], family + '_terminal_t0_lp']
            need(row['status'] == 'complete' and row['au_scale'] == 1.5 and row['strength'] == .75
                 and row['actual_steps'] == row['denoising_steps'] == 15 and row['terminal_unet_calls'] == 1
                 and row['total_unet_calls'] == 16, 'V10 LP reference has a different protocol')
            case['v10_global_lp_' + family] = {'path': row['output'], 'sha256': row['sha256']}
        display_cases.append(case)
    return display_cases, records, result


def extra_binding_refs(cases, run):
    refs = {str(V10RUN / 'result.json'): V10_RESULT_SHA, str(V10RUN / 'cases.json'): V10_CASES_SHA}
    config = R.read(run / 'configuration.json')
    script = config['script']
    need(Path(script['path']).resolve().parent == HERE, 'V11 generator must belong to this version directory')
    refs[script['path']] = script['sha256']
    for key in ('cleanup_mask_manifest', 'cleanup_geometry_code'):
        refs[config[key]['path']] = config[key]['sha256']
    manifest = R.read(config['cleanup_mask_manifest']['path'])
    need(manifest['source_only'] is True and manifest['test_used'] is False and manifest['target_geometry_used'] is False
         and manifest['generated_pixels_used'] is False, 'Local cleanup must use source geometry only')
    entries = {c['case_id']: c for c in manifest['cases']}
    for case in cases:
        binding = case['cleanup_mask']
        need(entries[case['case_id']] == binding, 'Local support differs from prepared source-only manifest')
        refs[binding['path']] = binding['sha256']
        if isinstance(binding.get('receipt'), dict):
            refs[binding['receipt']['path']] = binding['receipt']['sha256']
    need(all(sha(p) == h for p, h in refs.items()), 'V11 cleanup/reference/code binding changed')
    return refs


original_cell_source = R.cell_source
def cell_source(case, col, records):
    if col['kind'] == 'v10_lp':
        value = case[col['key']]
        return {'path': value['path'], 'expected_sha256': value['sha256'], 'generation_status': 'complete',
                'actual_steps': 15, 'denoising_steps': 15, 'terminal_unet_calls': 1, 'total_unet_calls': 16,
                'comparison_reference': 'fixed V10 full-editmask t0 LP; not a V11 candidate'}
    info = original_cell_source(case, col, records)
    if col['kind'] == 'v11':
        row = records.get((case['case_id'], col['key']), {})
        info.update(denoising_steps=row.get('denoising_steps'), terminal_unet_calls=row.get('terminal_unet_calls'),
                    total_unet_calls=row.get('total_unet_calls'))
    return info


def annotate_budget(page, draw, x, y, info, fonts):
    draw.rectangle((x, y + R.CELL + 10, x + R.CELL, y + R.CELL + 39), fill=R.layout.BACKGROUND)
    caption = (f"DEIS{info['denoising_steps']} + 终端{info['terminal_unet_calls']} / UNet{info['total_unet_calls']}"
               if info['generation_status'] == 'complete' else '生成失败／未完整记录调用')
    R.layout.centered(draw, caption, x, y + R.CELL + 14, fonts[18], R.layout.MUTED)
    info['display_step_caption'] = caption


R.columns, R.load_inputs, R.cell_source = columns, load_inputs, cell_source
R.annotate_budget, R.extra_binding_refs = annotate_budget, extra_binding_refs
if __name__ == '__main__':
    R.main()
