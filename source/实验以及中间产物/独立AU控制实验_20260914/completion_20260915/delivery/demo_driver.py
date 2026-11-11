"""One frozen source-only request via unchanged native Runtime; never a new experiment.

There is intentionally no independent denoiser, scheduler loop, condition math,
model loader or mask formula here. The native full-input gate and Runtime are
used unchanged. Default dry-run imports no model/torch/numpy/image libraries.
"""
import argparse
import contextlib
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
import time
import traceback
import reproduce_driver as reproduce

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
GEN = COMPLETION / 'generation'
OUTPUT_ROOT = HERE / 'demo_runs'
CASE_ID = 'postfreeze_SN003_s00869_t01203'
VARIANT = 'bidirectional'
METHOD = VARIANT + '_local_lp_s050'
SEED = 101
ORIGINAL_RUN = COMPLETION / 'generation_runs/selected_v11_32_v1'
need, read, sha, save = reproduce.need, reproduce.read, reproduce.sha, reproduce.save


def choose_output(name=None):
    root = OUTPUT_ROOT.resolve()
    need(HERE.resolve() in root.parents, 'Demo输出不能逃离delivery目录。')
    if name:
        need(re.fullmatch(r'demo_[A-Za-z0-9_-]+', name), '--run-name只能是demo_开头的安全新目录名。')
        names = [name]
    else:
        stem = 'demo_' + datetime.now().astimezone().strftime('%Y%m%d%H%M%S')
        names = [stem] + [stem + '-' + str(i).zfill(2) for i in range(2, 10000)]
    for candidate in names:
        path = root / candidate
        if not path.exists():
            return path
    raise ValueError('Demo输出目录已存在；拒绝覆盖。')


def make_plan(name=None):
    frozen, refs = reproduce.frozen_inputs()
    label_ref = frozen['label_plan']
    need(sha(label_ref['path']) == label_ref['sha256'], '冻结案例清单SHA已改变。')
    labels = read(label_ref['path'])['cases']
    edited = [r for r in labels if any(r['target_au_vector_0_5'])]
    need(edited and edited[0]['case_id'] == CASE_ID, '固定首个非零案例改变；不能按结果替换demo案例。')
    case = edited[0]
    need(case['subject'] == 'SN003' and case['target_frame_1based'] == 1203 and not any(case['source_au_vector_0_5']), 'Demo来源/目标AU标签不匹配。')
    output = choose_output(name)
    return {'kind': 'fixed_v11_single_image_demo', 'demo_only': True, 'included_in_statistics': False,
        'logical_outputs_added_to_main': 0, 'physical_outputs_added_to_main': 0,
        'case_id': CASE_ID, 'subject': 'SN003', 'variant': VARIANT, 'method': METHOD, 'seed': SEED,
        'selection': 'First nonzero case in already frozen order, fixed by request before demo execution; not selected using generated quality',
        'source_au_vector_0_5': case['source_au_vector_0_5'], 'target_au_vector_0_5': case['target_au_vector_0_5'],
        'output_directory': str(output), 'inputs': refs, 'label_plan': label_ref,
        'core_functions': ['run_grid.load_freeze', 'run_grid.load_cache', 'geometry.build_masks', 'run_grid.Runtime', 'Runtime.prepare', 'Runtime.sample', 'returned PIL.Image.save'],
        'native_preparation_note': 'Unchanged Runtime loads four checkpoints and prepares all eight SOURCE VAE latents; only one sample call is made.',
        'expected_denoising_calls': 15, 'extra_terminal_calls': 0,
        'target_pixels_supplied': False, 'target_latents_supplied': False, 'parameters_changed': False,
        'full_protocol_logical_outputs_unchanged': 768, 'full_protocol_physical_jobs_unchanged': 704,
        'sampler_source_sha256': frozen['generation']['execution_sources']['run_grid.py']['sha256'],
        'demo_entry_sha256': sha(__file__), 'created_local': datetime.now().astimezone().isoformat()}


def original_reference(plan):
    path = ORIGINAL_RUN / 'result.json'
    need(path.is_file(), '完整原生成结果尚未终态，不能做demo对照。')
    digest = sha(path)
    result = read(path)
    need(result['status'] in ('complete', 'completed_with_failures') and result['planned'] == 480 and len(result['records']) == 480, '完整原480结果不满足终态条件。')
    config_ref = result['configuration']
    need(sha(config_ref['path']) == config_ref['sha256'], '原生成配置SHA不匹配。')
    config = read(config_ref['path'])
    for key in ('freeze', 'source_cache', 'source_measurements'):
        need(config[key] == plan['inputs'][key], 'Demo与原实验输入不同：' + key)
    matches = [r for r in result['records'] if r['case_id'] == CASE_ID and r['method'] == METHOD and r['seed'] == SEED]
    need(len(matches) == 1 and matches[0]['status'] == 'complete', '固定demo原结果必须有且只有一个成功匹配，失败时不能换案例。')
    row = matches[0]
    need(row['source_aus_0_5'] == plan['source_au_vector_0_5'] and row['target_aus_0_5'] == plan['target_au_vector_0_5'], '原demo请求AU不一致。')
    need(ORIGINAL_RUN in Path(row['output']).resolve().parents and sha(row['output']) == row['sha256'], '原成功输出实体/SHA不匹配。')
    need(sha(path) == digest, '读取期间原完整结果发生变化。')
    return row, {'path': str(path), 'sha256': digest}


def load_native():
    # Only called beyond the terminal+busy+shared-lock guards.
    sys.path.insert(0, str(GEN))
    import run_grid
    import geometry
    return run_grid, geometry


def compare_output(path, expected):
    actual_sha = sha(path)
    need(sha(expected['output']) == expected['sha256'], '原输出在比较前改变。')
    exact_file = actual_sha == expected['sha256']
    result = {'new_png_sha256': actual_sha, 'original_png_sha256': expected['sha256'], 'png_sha_equal': exact_file,
        'decoded_pixels_checked': False, 'decoded_pixels_equal': None}
    if exact_file:
        result['matched'] = True
        return result
    # Encoding-only fallback never changes either image; exact pixels, no tolerance.
    from PIL import Image
    with Image.open(path) as current, Image.open(expected['output']) as original:
        same_format = current.mode == original.mode == 'RGB' and current.size == original.size == (512, 512)
        equal = same_format and current.tobytes() == original.tobytes()
    result.update(decoded_pixels_checked=True, decoded_pixels_equal=equal, matched=equal)
    return result


def produce_one(runtime, case, source, mask, output, expected):
    """Only routing and artifact validation; native functions own all generation math."""
    path = output / (CASE_ID + '_' + METHOD + '_seed101_demo.png')
    need(not path.exists(), 'Demo图已存在，不能覆盖。')
    prepared = runtime.prepare(VARIANT, case, source, mask)
    image, receipt = runtime.sample(prepared, SEED)
    need(image.mode == 'RGB' and image.size == (512, 512), '原生生成结果不是512RGB。')
    image.save(path)  # Same save method used by the frozen execute_grid.
    comparison = compare_output(path, expected)
    trace_checks = {key: receipt[key] == expected[key] for key in
        ('condition_hashes', 'source_latent_sha256', 'noise_sha256', 'projection_trace')}
    trace_checks['calls_15_0_15'] = (receipt['deis_unet_calls'], receipt['terminal_unet_calls'], receipt['total_unet_calls']) == (15, 0, 15)
    passed = comparison['matched'] and all(trace_checks.values())
    return {'status': 'demo_verified' if passed else 'demo_mismatch', 'demo_only': True, 'included_in_statistics': False,
        'output': {'path': str(path), 'sha256': sha(path)}, 'native_receipt': receipt,
        'image_comparison': comparison, 'trace_checks': trace_checks, 'matched': passed,
        'sampler_calls': 1, 'logical_outputs_added_to_main': 0, 'physical_outputs_added_to_main': 0}


def native_demo(plan, output, expected):
    core, geometry = load_native()
    refs = plan['inputs']
    _, labels = core.load_freeze(Path(refs['freeze']['path']), refs['freeze']['sha256'])
    torch = core.environment('mps')
    cache, features = core.load_cache(Path(refs['source_cache']['path']), refs['source_cache']['sha256'], labels, refs['freeze']['sha256'], load_tensors=True)
    cases = [c for c in cache['cases'] if c['case_id'] == CASE_ID]
    need(len(cases) == 1, '固定demo case在已校验完整缓存中缺失/重复。')
    case = cases[0]
    sources = [s for s in cache['sources'] if s['source_id'] == case['source_id']]
    need(len(sources) == 1 and sources[0]['status'] == 'complete' and case['condition_status'] == 'complete', 'Demo源或条件缓存失败。')
    source = sources[0]
    measurements = read(refs['source_measurements']['path'])
    need(measurements['source_only'] and measurements['protocol_sha256'] == refs['freeze']['sha256'] and measurements['source_cache_sha256'] == refs['source_cache']['sha256'], '源几何绑定不匹配。')
    measured = [r for r in measurements['records'] if r.get('method') == 'source_real' and r['case_id'] == CASE_ID]
    need(len(measured) == 1, 'Demo只接受其固定source_real几何。')
    edit, cleanup, mask_receipt = geometry.build_masks(case, source, measured[0])
    import numpy as np
    old_mask = read(core.verified_ref(expected['mask_receipt'], 'original demo mask receipt'))
    for name, array, reference in (('edit64', edit, old_mask['edit_mask']), ('cleanup64', cleanup, old_mask['cleanup_mask'])):
        original = core.verified_ref(reference, 'original ' + name)
        need(np.array_equal(array, np.load(original, allow_pickle=False)), 'Demo源mask必须和原结果严格相同。')
        with (output / (name + '.npy')).open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
    mask_receipt.update(edit_mask=core.ref(output / 'edit64.npy'), cleanup_mask=core.ref(output / 'cleanup64.npy'),
        demo_only=True, original_masks_exact=True)
    save(output / 'mask_receipt.json', mask_receipt)
    mask = (edit, cleanup, core.ref(output / 'mask_receipt.json'))
    runtime = None
    try:
        # Entire native Runtime constructor is retained, no alternate/simplified model loader.
        runtime = core.Runtime(features, cache, output, torch)
        result = produce_one(runtime, case, source, mask, output, expected)
        result['runtime'] = core.ref(output / 'runtime.json')
        return result
    finally:
        if runtime is not None:
            runtime.close()


def execute(plan):
    guard = reproduce.execution_guard()
    if guard['status'] != 'READY':
        return guard, guard['exit_code']
    with reproduce.exclusive_lock():
        guard = reproduce.execution_guard()
        if guard['status'] != 'READY':
            return guard, guard['exit_code']
        reproduce.frozen_inputs()
        expected, original_result = original_reference(plan)
        output = Path(plan['output_directory']).resolve()
        need(output.parent == OUTPUT_ROOT.resolve() and HERE.resolve() in output.parents and not output.exists(), 'Demo输出目录无效或已存在。')
        output.mkdir(parents=True)
        save(output / 'plan.json', {**plan, 'original_result': original_result,
            'original_output': {'path': expected['output'], 'sha256': expected['sha256']}})
        started = time.monotonic()
        result = {'kind': plan['kind'], 'demo_only': True, 'included_in_statistics': False, 'case_id': CASE_ID,
            'method': METHOD, 'seed': SEED, 'original_result': original_result, 'status': 'started'}
        save(output / 'started.json', result)
        code = 1
        try:
            with (output / 'demo.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result.update(native_demo(plan, output, expected))
            reproduce.frozen_inputs()
            need(sha(original_result['path']) == original_result['sha256'], 'Demo期间原完整结果改变。')
            code = 0 if result['matched'] else 1
        except KeyboardInterrupt:
            result.update(status='interrupted', error='用户中断；demo产物保留，不改正式结果。')
            code = 130
        except Exception as error:
            result.update(status='failed', error=str(error), traceback=traceback.format_exc(),
                native_partial_receipt=getattr(error, 'generation_receipt', None))
            code = 1
        finally:
            result.update(seconds=time.monotonic() - started, exit_code=code, main_704_768_records_modified=False,
                end_local=datetime.now().astimezone().isoformat(), saved_demo_artifacts_preserved=True)
            save(output / 'demo_result.json', result)
        return result, code


def main():
    parser = argparse.ArgumentParser(description='固定一张同协议复现demo；默认dry-run，不运行模型。')
    parser.add_argument('--execute', action='store_true', help='仅主队列完整终态并经显式授权后生成固定一张')
    parser.add_argument('--run-name', help='可选demo_开头的新名称，已存在则拒绝')
    args = parser.parse_args()
    plan = make_plan(args.run_name)
    print('DEMO-ONLY：固定%s / %s / seed101，原生Runtime单次sample；不进入704/768实验统计。' % (CASE_ID, VARIANT), flush=True)
    print('复用原完整源码；无目标图/目标latent输入，不改条件、权重、采样或局部正则。', flush=True)
    print('输出：' + plan['output_directory'], flush=True)
    if not args.execute:
        print('DRY_RUN：未加载模型、未创建demo输出。完整主队列终态后，main审阅决定是否执行一次复现自检。', flush=True)
        print('届时显式命令：' + shlex.join([str(reproduce.EXPECTED_PYTHON), '-B', str(Path(__file__).resolve()), '--execute', '--run-name', Path(plan['output_directory']).name]), flush=True)
        return 0
    result, code = execute(plan)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print('Demo未启动或安全拒绝：' + str(error), file=sys.stderr)
        raise SystemExit(2)
