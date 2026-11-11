"""Explicit staged same-protocol regeneration. Default prints commands only.

No model imports; only --execute may spawn an already frozen native generator,
after the original full queue is terminal, known generation processes are absent,
and a process-lifetime reproduction lock has been acquired.
"""
import argparse
import contextlib
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
import review_driver as review

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
EXP = COMPLETION.parent
THESIS = COMPLETION.parents[2]
OUTPUT_ROOT = HERE / 'reproduction_runs'
LOCK_PATH = HERE / '.teacher_reproduction.lock'
INPUTS = {
    'freeze': (COMPLETION / 'frozen_protocol.json', '498298977f3589248084e37d053ec7cd8f37f71b35e914849465081a3df9af33'),
    'raw_manifest': (COMPLETION / 'data/raw_202609152329/raw_frame_manifest.json', 'f3139d03d7b7808aea47cd46aed2f0107e77bf443f69f476ed845f8a1c1875ec'),
    'source_cache': (COMPLETION / 'source_caches/selected_v11_v1/manifest.json', 'ae4ba1c8ec9cb7604999857d9b73f2754d4bddd997821ad89fd9d71d11583dca'),
    'source_measurements': (COMPLETION / 'evaluation/source_v1/results.json', '9071f9226a0d18b6a7e2eba2c48fb2425e31297e26bbefb63e42844fa96c40be'),
}
COUNTS = {'ours': (480, 480), 'magicface': (96, 96), 'faceadapter': (96, 96), 'liveportrait': (32, 96)}
EXPECTED_PYTHON = Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'
need, read, sha, save = review.need, review.read, review.sha, review.save


def frozen_inputs():
    refs = {}
    for role, (path, expected) in INPUTS.items():
        need(path.is_file() and sha(path) == expected, '固定输入不匹配：' + role)
        refs[role] = {'path': str(path), 'sha256': expected}
    frozen = read(INPUTS['freeze'][0])
    need(frozen['status'] == 'frozen' and frozen['generation']['policy']['sampling_seeds'] == [101, 202, 303], '冻结设置或三seed已改变。')
    need(read(INPUTS['raw_manifest'][0])['status'] == 'raw_frames_extracted', '原始帧未完整就绪。')
    need(read(INPUTS['source_cache'][0])['status'] == 'complete' and read(INPUTS['source_measurements'][0])['status'] == 'complete', '来源缓存/测量未完整就绪。')
    for item in list(frozen['generation']['execution_sources'].values()) + list(frozen['external']['sources'].values()):
        need(sha(item['path']) == item['sha256'], '冻结生成入口源码已改变：' + item['path'])
    return frozen, refs


def choose_name(stage, explicit=None):
    root = OUTPUT_ROOT.resolve()
    need(HERE.resolve() in root.parents, '复跑报告根目录不能通过符号链接逃离delivery。')
    if explicit:
        need(re.fullmatch(r'reproduce_[A-Za-z0-9_-]+', explicit), '--run-name必须是reproduce_开头的英文/数字/下划线/短横线名称。')
        candidates = [explicit]
    else:
        stem = 'reproduce_' + datetime.now().astimezone().strftime('%Y%m%d%H%M%S') + '_' + stage
        candidates = [stem] + [stem + '-' + str(n).zfill(2) for n in range(2, 10000)]
    for name in candidates:
        delivery = root / name
        native = delivery / 'outputs' if stage == 'ours' else COMPLETION / 'external_runs' / ('teacher_reproduction_' + name.removeprefix('reproduce_'))
        if not delivery.exists() and not native.exists():
            return name, delivery, native
    raise ValueError('该新输出名称已存在；不会覆盖，请选择另一个名称。')


def timing_estimate(stage):
    # Completed full-stage timing receipts only; never new metrics/images.
    if stage in ('ours', 'magicface', 'faceadapter'):
        path = COMPLETION / 'execution_v1' / (stage + '.receipt.json')
        document = read(path)
        need(document['phase'] == stage and document['returncode'] == 0 and document['seconds'] > 0, '完整阶段耗时回执未成功终态：' + stage)
        total = document['seconds']
        count = 1
        reference_kind = 'completed_full_stage_wall_seconds'
        scope = '同冻结主实验完整生成阶段墙钟回执，含该阶段载入/预处理/生成/保存及校验；不含后续CPU评价统计'
        practical = '本机该完整阶段已完成%.2f秒（约%.2f小时），作为同协议复跑量级参考' % (total, total / 3600)
    else:
        path = COMPLETION / 'external_runs/liveportrait_postfreeze_v1/generation_index.json'
        document = read(path)
        need(document['status'] == 'complete', 'LivePortrait历史耗时参考尚未完整。')
        values = [r['seconds'] for r in document['execution_receipts']]
        need(len(values) == 32 and all(value > 0 for value in values), 'LP32物理worker耗时记录不完整。')
        total, count = sum(values), len(values)
        reference_kind = 'sum_completed_native_worker_seconds'
        scope = '同冻结32物理worker累计耗时，不是主队列阶段墙钟；96逻辑行含64确定性别名'
        practical = '本机32物理worker已完成累计约%.2f分钟（冻结配置CPU），未计额外调度开销' % (total / 60)
    return {'reference': {'path': str(path), 'sha256': sha(path)}, 'reference_count': count,
        'reference_kind': reference_kind, 'observed_completed_seconds': total, 'observed_completed_hours': total / 3600,
        'amortized_seconds_per_physical_output': total / COUNTS[stage][0],
        'scope': scope, 'practical_estimate': practical, 'warning': '仅该次本机已完成量级，复跑会随负载变化；非跨机器性能承诺，不含后续统一评价。'}


def command_plan(stage, explicit_name=None):
    need(stage in COUNTS, '未知重生成阶段。')
    frozen, refs = frozen_inputs()
    name, delivery, native = choose_name(stage, explicit_name)
    fixed = ['--freeze', refs['freeze']['path'], '--freeze-sha', refs['freeze']['sha256']]
    if stage == 'ours':
        executable = str(EXPECTED_PYTHON)
        command = [executable, '-B', str(COMPLETION / 'generation/run_grid.py'), *fixed,
            '--source-cache', refs['source_cache']['path'], '--source-cache-sha', refs['source_cache']['sha256'],
            '--source-measurements', refs['source_measurements']['path'], '--source-measurements-sha', refs['source_measurements']['sha256'],
            '--output', str(native), '--execute']
        result_name = 'result.json'
    else:
        executable = frozen['external']['methods'][stage]['settings']['runtime']['python_executable']
        command = [executable, '-B', str(COMPLETION / 'external/run_external.py'), *fixed,
            '--raw-manifest', refs['raw_manifest']['path'], '--raw-sha', refs['raw_manifest']['sha256'],
            '--method', stage, '--output', str(native), '--execute']
        result_name = 'generation_index.json'
    need(Path(executable).is_file(), '冻结解释器不存在：' + executable)
    need('run_completion_queue.py' not in ' '.join(command), '禁止调用静态v1队列。')
    return {'kind': 'teacher_same_protocol_regeneration', 'stage': stage, 'run_name': name,
        'delivery_directory': str(delivery), 'native_output_directory': str(native),
        'delivery_output_relative_link': None if stage == 'ours' else os.path.relpath(native, delivery),
        'expected_result': str(native / result_name), 'command': command, 'shell_command': shlex.join(command),
        'cwd': str(THESIS), 'inputs': refs, 'case_count': 32, 'sampling_seeds': [101, 202, 303],
        'planned_physical_outputs': COUNTS[stage][0], 'planned_logical_outputs': COUNTS[stage][1],
        'planned_alias_rows': COUNTS[stage][1] - COUNTS[stage][0], 'timing': timing_estimate(stage),
        'parameters_changed': False, 'new_cases_added': False, 'automatic_evaluation': False,
        'source_sha256': sha(__file__), 'created_local': datetime.now().astimezone().isoformat()}


def active_generators():
    watched = {'run_completion_queue.py', 'run_grid.py', 'run_external.py', 'run_dev_completion.py'}
    output = subprocess.check_output(['/bin/ps', '-axo', 'pid=,command='], text=True)
    active = []
    for line in output.splitlines():
        try:
            pid_text, command = line.strip().split(None, 1)
            arguments = shlex.split(command)
        except ValueError:
            continue
        if '--execute' not in arguments and '--worker' not in arguments:
            continue
        scripts = [Path(arg).name for arg in arguments if arg.endswith('.py') and Path(arg).name in watched]
        if scripts:
            # Never persist unrelated full process arguments.
            active.append({'pid': int(pid_text), 'script': scripts[0]})
    return active


def execution_guard():
    waiting = review.terminal_check(review.final_requirements(review.DEFAULT_JOB, review.DEFAULT_QUEUE))
    if waiting:
        return {'status': 'PENDING', 'exit_code': 3, 'reason': '完整主队列及生成/评价/统计未终态，禁止重生成以免并发占用。', 'pending': waiting}
    need(platform.system() == 'Darwin' and platform.machine() == 'arm64', '仅支持当前Mac arm64冻结环境，不是跨平台包。')
    need(Path(sys.executable).absolute() == EXPECTED_PYTHON.absolute(), '执行时必须使用当前冻结的MagicFace Python环境。')
    active = active_generators()
    if active:
        return {'status': 'BUSY', 'exit_code': 4, 'reason': '发现仍在运行的生成入口；本次未启动模型。', 'active': active}
    return {'status': 'READY', 'exit_code': 0}


@contextlib.contextmanager
def exclusive_lock():
    HERE.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('另一教师复跑持有独占锁；禁止并行运行。') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute_plan(plan):
    guard = execution_guard()
    if guard['status'] != 'READY':
        return guard, guard['exit_code']
    with exclusive_lock():
        # Recheck after obtaining the lock to close two-wrapper dispatch races.
        guard = execution_guard()
        if guard['status'] != 'READY':
            return guard, guard['exit_code']
        frozen_inputs()
        output, native = Path(plan['delivery_directory']), Path(plan['native_output_directory'])
        need(not output.exists() and not native.exists(), '输出目录已经存在，拒绝覆盖。')
        need(output.parent == OUTPUT_ROOT.resolve() and HERE.resolve() in output.resolve().parents, 'delivery复跑目录不合法。')
        output.mkdir(parents=True)
        save(output / 'plan.json', plan)
        if plan['delivery_output_relative_link'] is not None:
            (output / 'outputs').symlink_to(plan['delivery_output_relative_link'], target_is_directory=True)
        receipt = {'kind': plan['kind'], 'stage': plan['stage'], 'run_name': plan['run_name'],
            'plan_sha256': sha(output / 'plan.json'), 'command': plan['command'], 'start_local': datetime.now().astimezone().isoformat(),
            'native_output_directory': str(native), 'prior_outputs_modified': False, 'parameters_changed': False,
            'main_queue_terminal_before_start': True, 'status': 'started'}
        save(output / 'started.json', receipt)
        process, code = None, 1
        started = time.monotonic()
        try:
            with (output / (plan['stage'] + '.log')).open('x') as log:
                process = subprocess.Popen(plan['command'], cwd=THESIS, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait()
            receipt['returncode'] = code
            result_path = Path(plan['expected_result'])
            if result_path.is_file():
                result = read(result_path)
                receipt['native_result'] = {'path': str(result_path), 'sha256': sha(result_path)}
                receipt['native_status'] = result.get('status')
                receipt['status'] = 'complete' if code == 0 and result.get('status') == 'complete' else 'completed_with_failures'
            else:
                receipt.update(status='failed_no_terminal_result', native_result=None)
            frozen_inputs()
            if receipt['status'] != 'complete':
                code = code or 1
        except KeyboardInterrupt:
            receipt.update(status='interrupted', error='用户中断；已保存的阶段产物保留，不覆盖原实验。')
            code = 130
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        except Exception as error:
            receipt.update(status='failed', error=str(error), traceback=traceback.format_exc())
            code = 1
        finally:
            receipt.update(end_local=datetime.now().astimezone().isoformat(), seconds=time.monotonic() - started,
                exit_code=code, saved_stage_artifacts_preserved=True, evaluation_performed=False)
            save(output / 'receipt.json', receipt)
        return receipt, code


def main():
    parser = argparse.ArgumentParser(description='同冻结协议重生成；默认dry-run，仅打印精确命令。')
    parser.add_argument('--stage', choices=list(COUNTS), help='默认dry-run展示ours；execute必须明确选择一个阶段')
    parser.add_argument('--execute', action='store_true', help='仅主队列完整终态后，显式执行所选整个阶段')
    parser.add_argument('--run-name', help='可选reproduce_开头的新目录名；已有名称拒绝覆盖')
    args = parser.parse_args()
    if args.execute and args.stage is None:
        parser.error('--execute必须同时显式提供--stage；不会默认启动480图。')
    plan = command_plan(args.stage or 'ours', args.run_name)
    print('同一冻结协议：32案例、原始AU/seed/参数不变；不会调用静态v1队列。', flush=True)
    print('阶段：%s；%d物理输出、%d逻辑记录。' % (plan['stage'], plan['planned_physical_outputs'], plan['planned_logical_outputs']), flush=True)
    print('预计时间：' + plan['timing']['practical_estimate'], flush=True)
    print('delivery：' + plan['delivery_directory'], flush=True)
    print('原生产物：' + plan['native_output_directory'], flush=True)
    print('精确命令（当前%s）：\n%s' % ('申请执行' if args.execute else '仅dry-run，不执行', plan['shell_command']), flush=True)
    if not args.execute:
        waiting = review.terminal_check(review.final_requirements(review.DEFAULT_JOB, review.DEFAULT_QUEUE))
        print('DRY_RUN：没有运行模型、没有创建输出目录。' + (' 主队列仍未终态，当前不允许execute。' if waiting else ' 主队列已终态，execute还会核查独占锁和运行进程。'), flush=True)
        return 0
    result, code = execute_plan(plan)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print('重生成未启动或已安全拒绝：' + str(error), file=sys.stderr)
        raise SystemExit(2)
