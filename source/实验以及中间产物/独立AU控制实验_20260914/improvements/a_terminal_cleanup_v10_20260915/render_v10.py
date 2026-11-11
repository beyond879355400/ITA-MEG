"""V10 terminal-run scientific layouts; same frozen source-only ROIs as V9."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V9_RENDERER = HERE.parent / 'a_texture_schedule_v9_20260915/render_v9.py'
V9_RENDERER_SHA = '507b23e265e9988da214a987f6dabad3f39e576e42657cbeacbe61ac7e675335'
RUNNER = HERE / 'run_terminal.py'
RUNNER_SHA = '3b11fb90c69276200906a156e7cd126c07727d393140829fce69858ce2f3d9a8'
MODES = (('terminal_last_eps', '末步ε外推', 0), ('terminal_t0', 't0重新估计', 1),
         ('terminal_t0_lp', 't0+latent低通', 1))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


need(sha(RUNNER) == RUNNER_SHA, 'Frozen V10 controller changed')
spec = importlib.util.spec_from_file_location('v10_render_frozen_metadata', RUNNER)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
need(sha(V9_RENDERER) == V9_RENDERER_SHA, 'Frozen V9 renderer changed')
source = V9_RENDERER.read_text()
CHANGES = [
    ('Terminal-only V9 layouts: every dev case, source-fixed debug crops, no models.',
     'Terminal-only V10 layouts: every dev case, unchanged source-fixed debug crops, no models.'),
    ('4个编辑案例 · 原A与3种末段AU', '4个编辑案例 · 原A与3种末端处理'),
    ("'V9｜'", "'V10｜'"),
    ('新候选：前10步AU×1.5，末5步分别AU×1 / ×0.5 / ×0；名义20步，强度.75，实际15步。',
     '全部AU×1.5、强度.75；15次原A的DEIS前向；末步ε外推不加调用，t0与t0低通各加1次UNet。'),
    ('原A始终AU×1.5；V9仅降低实际t≤250的末5步AU权重，AU标签与其它条件不变。',
     'V10仅处理最后DEIS输出，再沿原mask与source混合；t0/低通共16次UNet，末步外推15次。'),
    ('新候选前10步AU×1.5；末5步分别AU×1 / ×0.5 / ×0。原A15步均AU×1.5；均名义20、实际15步。',
     'AU始终×1.5；原A与末步外推15次UNet；t0/低通16次。名义20、实际DEIS15步；仅末端假设检验。'),
    ('Terminal direct V9 runs child; no models loaded', 'Terminal direct V10 runs child; no models loaded'),
    ("'v9_all_case_scientific_layouts'", "'v10_all_case_scientific_layouts'"),
    ("V7RUN / 'cases.json', V7RUN / 'result.json', Path(__file__), TEMPLATE, base.HELPER)",
     "V7RUN / 'cases.json', V7RUN / 'result.json', Path(__file__), TEMPLATE, base.HELPER, V9_RENDERER, RUNNER)"),
    ("'planned_generated_images': 48, 'generation_record_count': len(records)",
     "'terminal_budget': {'deis_steps': 15, 'terminal_last_eps_extra': 0, 'terminal_t0_extra': 1, 'terminal_t0_lp_extra': 1},\n        'planned_generated_images': 48, 'generation_record_count': len(records)"),
    ("cell_source(case, col, records), fonts)\n            info.update(case_id=case['case_id']",
     "cell_source(case, col, records), fonts)\n            if col['kind'] == 'v10': annotate_budget(page, draw, MARGIN + j * (CELL + GAP), top + 37, info, fonts)\n            info.update(case_id=case['case_id']"),
]
for old, new in CHANGES:
    need(source.count(old) == 1, 'Ambiguous V10 layout adaptation: ' + old)
    source = source.replace(old, new)
# Only the new-column kind changes. The untouched image/ROI layout functions are
# reused verbatim except the explicit budget caption line above.
source = source.replace("col['kind'] == 'v9'", "col['kind'] == 'v10'")
R = ModuleType('v10_from_frozen_v9_layout')
R.__dict__.update(__file__=str(Path(__file__).resolve()), V9_RENDERER=V9_RENDERER, RUNNER=RUNNER)
exec(compile(source, str(V9_RENDERER), 'exec'), R.__dict__)


def columns(overview=False):
    cols = [
        {'key': 'source_real', 'kind': 'reference', 'lines': ['来源图', '原始对齐图', '仅参照 · 无去噪']},
        {'key': 'target_real', 'kind': 'reference', 'lines': ['真实目标', '同一固定案例', '仅参照 · 非生成图输入']},
        {'key': 'a_bidirectional', 'kind': 'a', 'lines': ['原A · 双向', 'AU×1.5 / 强度.75', '15 DEIS / 总UNet15']},
    ]
    if not overview:
        cols.append({'key': 'a_baseline', 'kind': 'a', 'lines': ['原A · 基底', 'AU×1.5 / 强度.75', '15 DEIS / 总UNet15']})
    for mode, label, extra in MODES:
        families = [('bidirectional', '双向')] if overview else [('baseline', '基底'), ('bidirectional', '双向')]
        for family, family_label in families:
            cols.append({'key': family + '_' + mode, 'kind': 'v10', 'config': mode,
                         'terminal_unet_calls': extra,
                         'lines': [f'V10 · {family_label} · {label}', 'AU×1.5 / 强度.75', f'15 DEIS + {extra}额外UNet']})
    need(len(cols) == (6 if overview else 10), 'Expected six/ten display columns')
    return cols


def validate_terminal_row(row, col):
    extra = col['terminal_unet_calls']
    expected = {'config': col['config'], 'seed': 101, 'strength': .75, 'au_scale': 1.5, 'ip_scale': .9,
                'nominal_steps': 20}
    need(all(row.get(k) == v for k, v in expected.items()), 'Terminal record label/config mismatch')
    if row['status'] != 'complete':
        return
    need(all(row.get(k) == v for k, v in {'actual_steps': 15, 'denoising_steps': 15, 'deis_unet_calls': 15,
         'terminal_unet_calls': extra, 'total_unet_calls': 15 + extra, 'actual_timesteps': frozen.TIMESTEPS}.items()),
         'Completed terminal row has incorrect visible call budget')
    calls = row['unet_call_trace']
    need([(c['phase'], c['t']) for c in calls] == [('deis', t) for t in frozen.TIMESTEPS]
         + ([('terminal', 0)] if extra else []), 'UNet phase/time trace differs')
    need(all(c['cfg_batch'] == 2 and c['au_scale'] == 1.5 for c in calls), 'Full CFG/AU call trace differs')
    terminal = row['terminal_trace']
    need(terminal['mode'] == col['config'] and terminal['last_deis_timestep'] == 50
         and terminal['extra_cfg'] == (7.5 if extra else None), 'Terminal operation/CFG differs')
    need(terminal['filter'] == (frozen.FILTER if col['config'] == 'terminal_t0_lp' else None), 'Filter rule differs')
    need(terminal['scheduler_step_index_before'] == terminal['scheduler_step_index_after'] == 20,
         'Terminal stage moved scheduler index')
    need(all(math.isfinite(terminal[k]) and terminal[k] > 0 for k in ('alpha0', 'sigma0', 'rho0')),
         'Invalid nonzero terminal coefficients')
    need(math.isclose(terminal['rho0'], terminal['sigma0'] / terminal['alpha0'], rel_tol=1e-6),
         'Terminal rho differs from VP sigma/alpha')
    need(terminal['projected_latent_sha256'] == row['projection_trace'][-1]['latent_sha256'],
         'Final displayed output does not match terminal projection record')


def load_inputs(run):
    need(run.parent == HERE / 'runs', 'Input must be a direct V10 runs child')
    need((run / 'result.json').is_file(), 'No terminal result.json; active generation cannot be rendered')
    result = R.read(run / 'result.json')
    need(result.get('kind') == 'v10_terminal_diagnostic' and result.get('status') in
         ('complete', 'completed_with_failures'), 'Not a terminal V10 diagnostic')
    need(result.get('test_data_used') is False and result.get('planned') == 48, 'Expected all48 dev-only outputs')
    doc = R.bound_doc(run, result, 'cases', 'cases.json')
    config = R.bound_doc(run, result, 'configuration', 'configuration.json')
    need(doc['kind'] == 'v10_development_terminal_cases' and doc['case_count'] == 8
         and doc['test_data_used'] is False, 'Wrong V10 development-case schema')
    cases = doc['cases']
    need(cases == frozen.inputs_with_a(), 'Cases/labels/A refs differ from the complete frozen V9 fields')
    expected_configs = [{'name': n, 'strength': .75, 'au_scale': 1.5, 'terminal_unet_calls': e} for n, _, e in MODES]
    need(config['kind'] == 'v10_terminal_diagnostic' and config['configs'] == expected_configs, 'Config matrix differs')
    need(all(config.get(k) == v for k, v in {'guidance_scale': 7.5, 'ip_scale': .9, 'seed': 101,
         'nominal_steps': 20, 'denoising_steps': 15, 'post_generation_pixel_blend': False,
         'test_data_used': False, 'training_performed': False}.items()), 'Original A common protocol differs')
    need(config['script']['sha256'] == RUNNER_SHA and Path(config['script']['path']).resolve() == RUNNER,
         'Generated run was not made by the frozen controller')
    allowed = {c['key']: c for c in columns() if c['kind'] == 'v10'}
    known = {c['case_id']: c for c in cases}
    records = {}
    for row in result['records']:
        key = row['case_id'], row['method']
        need(key[0] in known and key[1] in allowed and key not in records, 'Unexpected/duplicate generated row')
        case = known[key[0]]
        need(all(row[k] == case[k] for k in ('subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5',
             'prompt', 'region_mask')), 'Generated row changed original case fields')
        family = 'bidirectional' if key[1].startswith('bidirectional_') else 'baseline'
        need(row['condition_hashes'] == case[family + '_text2img_condition_hashes'], 'Raw AU/IP condition hashes changed')
        need(Path(row['output']).resolve() == run / 'images' / key[0] / (key[1] + '.png'), 'Image belongs to another run/case/method')
        validate_terminal_row(row, allowed[key[1]])
        if row['status'] == 'complete' and case['group'] == 'no_edit':
            need(row['sha256'] == case['vae_reconstruction']['sha256'], 'No-edit differs from source VAE reconstruction')
        records[key] = row
    need(len(records) == 48 and result['completed'] == sum(r['status'] == 'complete' for r in records.values()),
         'All48 terminal records and correct completed count required')
    if result['status'] == 'complete':
        need(result['completed'] == 48, 'Complete run must contain all48 successful rows')
    return cases, records, result


original_cell_source = R.cell_source
def cell_source(case, col, records):
    info = original_cell_source(case, col, records)
    if col['kind'] == 'v10':
        row = records.get((case['case_id'], col['key']), {})
        info.update(denoising_steps=row.get('denoising_steps'), terminal_unet_calls=row.get('terminal_unet_calls'),
                    total_unet_calls=row.get('total_unet_calls'))
    return info


def annotate_budget(page, draw, x, y, info, fonts):
    draw.rectangle((x, y + R.CELL + 10, x + R.CELL, y + R.CELL + 39), fill=R.layout.BACKGROUND)
    if info['generation_status'] == 'complete':
        caption = f"DEIS{info['denoising_steps']} + 终端{info['terminal_unet_calls']} / UNet{info['total_unet_calls']}"
    else:
        caption = '生成失败／未完整记录调用'
    R.layout.centered(draw, caption, x, y + R.CELL + 14, fonts[18], R.layout.MUTED)
    info['display_step_caption'] = caption


R.columns, R.load_inputs, R.cell_source, R.annotate_budget = columns, load_inputs, cell_source, annotate_budget
if __name__ == '__main__':
    R.main()
