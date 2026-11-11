"""Teacher-facing, append-only wrapper around existing non-ML verifiers."""
import argparse
import contextlib
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
REVIEW = COMPLETION / 'review_tools'
FREEZE = COMPLETION / 'frozen_protocol.json'
FREEZE_SHA = '498298977f3589248084e37d053ec7cd8f37f71b35e914849465081a3df9af33'
DEFAULT_JOB = COMPLETION / 'evaluation/all_methods_v1'
DEFAULT_QUEUE = COMPLETION / 'execution_v1/result.json'
DEV_GENERATION = COMPLETION / 'dev19_runs/dev19_202609152322/result.json'
DEV_COMPARISON = COMPLETION / 'evaluation/dev19_tools/dev19_202609152322/comparison/results.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def reserve_report(root):
    root = Path(root).resolve()
    need(root == HERE or HERE in root.parents, '复核输出必须位于completion/delivery内部。')
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime('review_%Y%m%d%H%M%S')
    for number in range(1, 10000):
        path = root / (stamp if number == 1 else stamp + '-' + str(number).zfill(2))
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('无法分配不覆盖旧文件的复核目录。')


def terminal_check(requirements):
    """Read terminal metadata only; never inspect images/CSVs before acceptance."""
    pending = []
    for label, path, statuses in requirements:
        path = Path(path)
        if not path.is_file():
            pending.append({'stage': label, 'path': str(path), 'reason': '终态文件尚未产生'})
            continue
        try:
            document = read(path)
        except (ValueError, OSError) as error:
            pending.append({'stage': label, 'path': str(path), 'reason': '元数据暂不可完整读取：' + str(error)})
            continue
        if statuses is not None:
            status = document.get('status', document.get('summary', {}).get('status'))
            if status not in statuses:
                pending.append({'stage': label, 'path': str(path), 'reason': '尚非可复核终态', 'status': status})
    return pending


def final_requirements(job, queue_result):
    accepted = ('complete', 'completed_with_failures')
    return [
        ('完整计算队列', queue_result, ('computation_complete_pending_independent_and_visual_review', 'computation_complete_pending_review')),
        ('本方480条生成', COMPLETION / 'generation_runs/selected_v11_32_v1/result.json', accepted),
        ('MagicFace生成', COMPLETION / 'external_runs/magicface_postfreeze_v1/generation_index.json', accepted),
        ('FaceAdapter生成', COMPLETION / 'external_runs/faceadapter_postfreeze_v1/generation_index.json', accepted),
        ('LivePortrait生成', COMPLETION / 'external_runs/liveportrait_postfreeze_v1/generation_index.json', accepted),
        ('评价输入清单', job / 'items.json', None),
        ('评价输入封存', job / 'items_seal.json', None),
        ('OpenFace评价', job / 'openface/results.json', ('complete',)),
        ('FaceNet评价', job / 'facenet/results.json', ('complete',)),
        ('buffalo评价', job / 'buffalo/results.json', ('complete',)),
        ('全案例指标导出', job / 'statistics/all_case_seed_metrics.json', ('complete',)),
        ('主体配对统计输入', job / 'statistics/paired_subject_input.json', None),
        ('主体配对统计结果', job / 'statistics/paired_subject_results.json', None),
    ]


def load_verifier(name, filename):
    # Both imported modules are read-only arithmetic auditors, never inference entries.
    path = REVIEW / filename
    need(path.is_file(), '复核实现不存在：' + str(path))
    sys.path.insert(0, str(REVIEW))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(output, job, queue_result, dev_only=False):
    summary = {'kind': 'teacher_existing_artifact_review', 'start_local': datetime.now().astimezone().isoformat(),
        'python': sys.executable, 'python_version': sys.version, 'source': str(HERE / 'review_driver.py'),
        'source_sha256': sha(HERE / 'review_driver.py'), 'mode': 'dev_only' if dev_only else 'dev_and_terminal32',
        'models_loaded_or_called': 0, 'model_generation_performed': False, 'training_performed': False,
        'prior_results_modified': False, 'verifiers': {}, 'checks': {}, 'output': str(output)}
    errors, waiting = [], []
    dev_pending = terminal_check([
        ('开发集95条生成', DEV_GENERATION, ('complete',)),
        ('开发集完整比较', DEV_COMPARISON, ('complete',)),
    ])
    if dev_pending:
        waiting += dev_pending
        summary['checks']['dev19'] = {'status': 'PENDING', 'details': dev_pending}
    else:
        try:
            path = REVIEW / 'verify_dev19.py'
            summary['verifiers']['dev19'] = {'path': str(path), 'sha256': sha(path)}
            with (output / 'dev19.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = load_verifier('teacher_dev19_verifier', 'verify_dev19.py').verify()
            need(result['status'] == 'verified', '开发集复核没有返回verified。')
            save(output / 'dev19_verification.json', result)
            summary['checks']['dev19'] = {'status': 'PASS', 'report': 'dev19_verification.json', 'denominators': result['denominators'],
                'recomputed': '生成SHA/轨迹、原CSV选行、AU/extra算术、4主体等权及36AU×level',
                'identity_scope': '身份结果绑定/选行/标量回读；未重新计算embedding'}
        except Exception as error:
            errors.append('dev19: ' + str(error))
            summary['checks']['dev19'] = {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc()}
    if dev_only:
        summary['checks']['postfreeze32'] = {'status': 'NOT_REQUESTED', 'reason': '--dev-only，不代表32例验证已完成'}
    else:
        final_pending = terminal_check(final_requirements(job, queue_result))
        if final_pending:
            waiting += final_pending
            summary['checks']['postfreeze32'] = {'status': 'PENDING', 'details': final_pending,
                'reason': '完整生成/评价/统计尚未终态；没有对进行中图像或部分数字做终验'}
        else:
            try:
                need(FREEZE.is_file() and sha(FREEZE) == FREEZE_SHA, '正式冻结协议SHA不匹配。')
                if importlib.util.find_spec('numpy') is None:
                    raise RuntimeError('终态统计复核需要NumPy；请使用具有NumPy的现有解释器，不会自动安装依赖。')
                path = REVIEW / 'verify_completed.py'
                summary['verifiers']['postfreeze32'] = {'path': str(path), 'sha256': sha(path),
                    'statistics_helper': {'path': str(REVIEW / 'statistics_check.py'), 'sha256': sha(REVIEW / 'statistics_check.py')}}
                with (output / 'postfreeze32.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = load_verifier('teacher_terminal32_verifier', 'verify_completed.py').verify(job, FREEZE, FREEZE_SHA)
                need(result['status'] == 'PASS', '32例终态复核没有返回PASS。')
                save(output / 'postfreeze32_verification.json', result)
                summary['checks']['postfreeze32'] = {'status': 'PASS', 'report': 'postfreeze32_verification.json',
                    'planned_logical_outputs': result['planned_logical_outputs'], 'planned_physical_jobs': result['planned_physical_jobs'],
                    'successful_physical_jobs': result['successful_physical_jobs'],
                    'recomputed': '人工标签/SHA、CSV选行/AU指标、三seed主体聚合、50k bootstrap、符号翻转与Holm校正',
                    'identity_scope': 'FaceNet/buffalo绑定和标量回读；不重新计算embedding或视觉质量'}
            except Exception as error:
                errors.append('postfreeze32: ' + str(error))
                summary['checks']['postfreeze32'] = {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc()}
    status = 'FAIL' if errors else 'PENDING' if waiting else 'PASS_DEV_ONLY' if dev_only else 'PASS'
    code = 1 if errors else 3 if waiting else 0
    summary.update(status=status, exit_code=code, errors=errors, pending=waiting, end_local=datetime.now().astimezone().isoformat(),
        interpretation='完整性/数字复核通过不等于重新生成，也不代表全部生成或测量都成功；失败和NA分母由原协议保留。')
    save(output / 'summary.json', summary)
    lines = ['教师实验复核（仅已有产物，无模型运行）', '状态：' + status, '输出：' + str(output), '']
    for name, result in summary['checks'].items():
        lines.append(name + '：' + result['status'])
    if waiting:
        lines += ['', '待完成：完整32例生成、评价、统计终态尚未齐备。入口不会重跑、接管或改写任何计算队列。',
                  '任务完成后再次运行相同命令；届时会追加新的复核目录，不覆盖本次记录。']
    if errors:
        lines += ['', '复核错误：'] + errors
    lines += ['', summary['interpretation'], '身份embedding未重算；不包含生成图视觉质量终审。']
    with (output / 'summary.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return summary, code


def main():
    parser = argparse.ArgumentParser(description='已有实验结果的完整性与数字复核；默认绝不运行模型。')
    parser.add_argument('--dev-only', action='store_true', help='只复核已完成19例开发集，不宣称32例完成')
    parser.add_argument('--job', type=Path, default=DEFAULT_JOB, help='同一冻结协议下的终态评价目录')
    parser.add_argument('--queue-result', type=Path, default=DEFAULT_QUEUE, help='只读完整计算终态回执路径')
    parser.add_argument('--output-root', type=Path, default=HERE / 'review_runs', help='delivery内部追加报告根目录')
    args = parser.parse_args()
    need(sys.version_info >= (3, 10), '需要Python 3.10或以上；不会自动安装或更换解释器。')
    job, queue = args.job.resolve(), args.queue_result.resolve()
    need(COMPLETION in job.parents and COMPLETION in queue.parents, '评价/队列回执须位于当前completion目录，不允许跨实验混用。')
    output = reserve_report(args.output_root)
    print('仅复核已有产物，不运行模型；本次追加目录：' + str(output), flush=True)
    started = time.monotonic()
    summary, code = run_checks(output, job, queue, args.dev_only)
    print('复核状态：' + summary['status'] + '；耗时 %.2f 秒。' % (time.monotonic() - started), flush=True)
    if code == 3:
        print('待完成：完整32例生成/评价/统计尚未终态。已完成部分的复核报告已保存；没有重跑或改写队列。', flush=True)
    elif code:
        print('复核失败：请查看summary.json中的具体错误；未自动修改原始结果。', flush=True)
    print('摘要：' + str(output / 'summary.txt'), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
