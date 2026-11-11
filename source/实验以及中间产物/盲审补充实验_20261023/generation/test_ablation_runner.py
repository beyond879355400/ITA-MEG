"""CPU/stdlib control tests. These do not establish image generation parity."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ablation_core as core
import run_ablations as runner


class Scalar:
    def __init__(self, value):
        self.value = float(value)
    def __int__(self): return int(self.value)
    def reshape(self, *_): return self
    def __mul__(self, other): return Scalar(self.value * number(other))
    def __rmul__(self, other): return self * other
    def __add__(self, other): return Scalar(self.value + number(other))
    def __radd__(self, other): return self + other
    def __sub__(self, other): return Scalar(self.value - number(other))
    def __rsub__(self, other): return Scalar(number(other) - self.value)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self): return struct.pack('<d', self.value)
    def all(self): return self
    def item(self): return self.value


def number(value):
    return value.value if isinstance(value, Scalar) else value


class Scheduler:
    def __init__(self):
        self.timesteps = [Scalar(v) for v in [999, 949, 899, 849, 799] + core.TIMESTEPS]
        self.add_noise_calls = []
    def add_noise(self, source, noise, t):
        self.add_noise_calls.append(int(t))
        return source + noise + int(t) / 1000


class BaseController:
    def __init__(self, *args):
        self.terminal_trace = {}
        self.correct_count = 0
        self.finalized = None
    def correct(self, latent, pipeline, embeds):
        self.correct_count += 1
        return latent + 100
    def finalize_projection(self, result):
        self.finalized = result


TORCH = SimpleNamespace(isfinite=lambda _: Scalar(1))


class SwitchTests(unittest.TestCase):
    def run_case(self, projection, smoothing, mask=.25):
        calls = []
        def filter_clean(clean, source, edit, cleanup, sigma, torch):
            calls.append((edit.value, cleanup.value, sigma))
            return clean + 7
        v11 = SimpleNamespace(v10=SimpleNamespace(TerminalController=BaseController),
                              local_filtered_clean=filter_clean, filter_metadata=lambda sigma: {'sigma': sigma})
        source, noise, edit, cleanup = Scalar(10), Scalar(2), Scalar(mask), Scalar(.125)
        controller = core.make_controller(v11, None, source, edit, None, TORCH, cleanup, smoothing)
        callback, history = core.sampling_callback(controller, source, noise, edit, TORCH, projection)
        pipeline = SimpleNamespace(scheduler=Scheduler())
        output = []
        for i, t in enumerate(core.TIMESTEPS):
            output.append(callback(pipeline, i, Scalar(t), {'latents': Scalar(i + 1), 'prompt_embeds': None})['latents'])
        return controller, history, pipeline.scheduler, calls, output, (edit, cleanup)

    def test_full_matches_original_control_formula(self):
        controller, trace, scheduler, calls, values, masks = self.run_case(True, True)
        expected = [.25 * (i + 1) + .75 * (12 + core.TIMESTEPS[i + 1] / 1000) for i in range(14)]
        expected.append(.25 * (15 + 100 + 7) + .75 * 10)
        self.assertEqual([v.value for v in values], expected)
        self.assertEqual(len(scheduler.add_noise_calls), 14)
        self.assertEqual(sum(r['projection_applied'] for r in trace), 15)
        self.assertEqual(calls, [(.25, .125, .5)])
        self.assertEqual(controller.correct_count, 1)

    def test_projection_off_removes_all_projection_not_terminal(self):
        controller, trace, scheduler, calls, values, masks = self.run_case(False, True)
        self.assertEqual([v.value for v in values[:14]], list(range(1, 15)))
        self.assertEqual(values[-1].value, 122)
        self.assertEqual(scheduler.add_noise_calls, [])
        self.assertEqual(sum(r['projection_applied'] for r in trace), 0)
        self.assertEqual(controller.correct_count, 1)
        self.assertEqual(calls, [(.25, .125, .5)])
        self.assertEqual([m.value for m in masks], [.25, .125])
        self.assertEqual(controller.terminal_trace['filter']['final_projection'], 'disabled')

    def test_smoothing_off_keeps_terminal_and_projection(self):
        controller, trace, scheduler, calls, values, _ = self.run_case(True, False)
        self.assertEqual(calls, [])
        self.assertEqual(values[-1].value, .25 * 115 + .75 * 10)
        self.assertEqual(controller.correct_count, 1)
        self.assertEqual(len(scheduler.add_noise_calls), 14)
        self.assertEqual(controller.terminal_trace['base_mode'], 'terminal_last_eps')
        self.assertIsNone(controller.terminal_trace['filter'])

    def test_both_off_preserves_exact_uncorrected_steps_and_terminal(self):
        controller, trace, scheduler, calls, values, _ = self.run_case(False, False)
        self.assertEqual([v.value for v in values], list(range(1, 15)) + [115])
        self.assertEqual(calls, [])
        self.assertEqual(scheduler.add_noise_calls, [])
        self.assertFalse(controller.terminal_trace['projection_enabled'])

    def test_empty_edit_mask_not_changed_to_full_when_projection_off(self):
        _, _, _, calls, _, masks = self.run_case(False, True, mask=0)
        self.assertEqual(calls, [(0, .125, .5)])
        self.assertEqual(masks[0].value, 0)

    def test_changed_timestep_rejected(self):
        controller = BaseController()
        callback, _ = core.sampling_callback(controller, Scalar(1), Scalar(2), Scalar(.5), TORCH, True)
        with self.assertRaisesRegex(ValueError, 'timesteps'):
            callback(SimpleNamespace(scheduler=Scheduler()), 0, Scalar(748), {'latents': Scalar(1)})


class GridAndIntegrityTests(unittest.TestCase):
    def cases(self):
        return [{'case_id': f'case_{i}', 'subject': f'SN{i // 3:03}',
                 'source_au_vector_0_5': [0] * 12, 'target_au_vector_0_5': [1] + [0] * 11} for i in range(24)]

    def test_complete_grid_all_288_slots_once(self):
        selected = core.select_cases({'cases': self.cases()})
        grid = core.execution_grid(selected)
        self.assertEqual(len(grid), 288)
        self.assertEqual(len({(c['case_id'], s, a['name']) for c, s, a in grid}), 288)

    def test_smoke_is_first_case_first_seed_all_arms(self):
        grid = core.execution_grid(self.cases(), smoke=True)
        self.assertEqual([(c['case_id'], s) for c, s, _ in grid], [('case_0', 101)] * 4)
        self.assertEqual([a['name'] for _, _, a in grid], ['P1S1', 'P0S1', 'P1S0', 'P0S0'])

    def test_zero_requests_excluded_without_sort_or_score(self):
        cases = self.cases()
        zero = {**cases[0], 'case_id': 'zero', 'target_au_vector_0_5': [0] * 12}
        self.assertEqual(core.select_cases({'cases': [zero] + cases}), cases)

    def test_changed_subject_population_rejected(self):
        cases = self.cases()
        cases[0]['subject'] = 'new_subject'
        with self.assertRaisesRegex(ValueError, '8 original subjects'):
            core.select_cases({'cases': cases})

    def test_write_outside_new_directory_rejected(self):
        with self.assertRaisesRegex(ValueError, 'new generation directory'):
            runner.write_new(runner.HERE.parent / 'forbidden.json', {})

    def test_hash_tamper_and_resume_parity_blocked(self):
        with tempfile.TemporaryDirectory(dir=runner.HERE) as folder:
            path = Path(folder) / 'test.json'
            runner.write_new(path, {'fixture': True})
            item = runner.ref(path)
            self.assertEqual(runner.check_ref(item), path)
            with self.assertRaises(FileExistsError):
                runner.write_new(path, {'fixture': False})
            with self.assertRaisesRegex(ValueError, 'Content changed'):
                runner.check_ref({**item, 'sha256': '0' * 64})
            with self.assertRaisesRegex(ValueError, 'parity failed'):
                runner.validate_completed({'status': 'success', 'output_path': str(path), 'output_sha256': item['sha256'], 'arm': 'full'})

    def run_failure_fixture(self, corrupt_record=False):
        with tempfile.TemporaryDirectory(dir=runner.HERE) as folder:
            root = Path(folder)
            cases = self.cases()
            for case in cases:
                case.update(source_id='source', source_frame_1based=1, target_frame_1based=2, split='postfreeze_case_validation')
            source = {'source_id': 'source', 'canonical_source': {'path': '/fixture/source.png', 'sha256': 'fixture'}}
            cache = {'sources': [source]}
            runner.write_new(root / 'cache.json', cache)
            calls = []
            class FakeRuntime:
                def prepare(self, *args):
                    calls.append(args[1]['case_id'])
                    raise RuntimeError('fixture full condition failure')
                def close(self): pass
            plan = {'cases': cases, 'source_cache': runner.ref(root / 'cache.json')}
            config = {'freeze': {'path': 'fixture', 'sha256': 'fixture'}, 'planned': 4,
                      'slot_plan': [{'case_id': c['case_id'], 'subject': c['subject'], 'seed': s, 'arm': runner.ARM_NAMES[a['name']]}
                                    for c, s, a in core.execution_grid(cases, True)]}
            masks = {c['case_id']: None for c in cases}
            if corrupt_record:
                damaged = root / 'run/attempts/case_0__101__full/interrupted/record.json'
                damaged.parent.mkdir(parents=True)
                damaged.write_text('{"status":', encoding='utf-8')
            with patch.object(runner, 'build_runtime', return_value=(FakeRuntime(), cache, masks)):
                result = runner.execute_locked(plan, config, root / 'run', True)
            self.assertEqual(calls, ['case_0'])
            self.assertEqual(len(result['records']), 4)
            self.assertEqual(result['completed'], 0)
            self.assertFalse(result['same_protocol_complete'])
            self.assertEqual([r.get('error') for r in result['records'][1:]], ['not_started'] * 3)
            self.assertTrue(all(r['source_sha256'] == 'fixture' for r in result['records']))
            self.assertEqual(len(result['resume_read_errors']), int(corrupt_record))
            if corrupt_record:
                self.assertEqual(damaged.read_text(), '{"status":')

    def test_any_full_failure_stops_other_arms_but_accounts_all_slots(self):
        self.run_failure_fixture()

    def test_corrupt_record_preserved_and_resume_retries_planned_slot(self):
        self.run_failure_fixture(corrupt_record=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
