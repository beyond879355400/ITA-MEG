"""Dry-run, terminal guard, lock and failure-preservation tests; no real process dispatch."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import reproduce_driver as r


class ReproductionEntryTests(unittest.TestCase):
    def test_exact_full_stage_commands_keep_original_protocol(self):
        for stage, counts in r.COUNTS.items():
            plan = r.command_plan(stage, 'reproduce_synthetic_readonly_' + stage)
            self.assertEqual((plan['planned_physical_outputs'], plan['planned_logical_outputs']), counts)
            self.assertEqual(plan['sampling_seeds'], [101, 202, 303])
            self.assertEqual(plan['case_count'], 32)
            self.assertIn('--execute', plan['command'])
            self.assertNotIn('--case', plan['command'])
            self.assertNotIn('--case-id', plan['command'])
            self.assertNotIn('run_completion_queue.py', plan['shell_command'])
            if stage != 'ours':
                self.assertIn(r.COMPLETION / 'external_runs', Path(plan['native_output_directory']).parents)
                target = Path(plan['delivery_directory']) / plan['delivery_output_relative_link']
                self.assertEqual(target.resolve(), Path(plan['native_output_directory']))

    def test_default_dryrun_creates_no_output_or_model_process(self):
        path = r.OUTPUT_ROOT / 'reproduce_synthetic_default_dryrun'
        self.assertFalse(path.exists())
        with patch.object(sys, 'argv', ['reproduce_driver.py', '--run-name', path.name]), patch.object(r.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(r.main(), 0)
        launch.assert_not_called()
        self.assertFalse(path.exists())

    def test_execute_requires_explicit_stage(self):
        with patch.object(sys, 'argv', ['reproduce_driver.py', '--execute']), patch.object(r.subprocess, 'Popen') as launch, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                r.main()
        self.assertEqual(error.exception.code, 2)
        launch.assert_not_called()

    def test_nonterminal_queue_blocks_before_directory_or_process(self):
        plan = {'stage': 'ours'}
        with patch.object(r, 'execution_guard', return_value={'status': 'PENDING', 'exit_code': 3}), patch.object(r.subprocess, 'Popen') as launch:
            result, code = r.execute_plan(plan)
        self.assertEqual(code, 3)
        self.assertEqual(result['status'], 'PENDING')
        launch.assert_not_called()

    def test_existing_name_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as directory, patch.object(r, 'OUTPUT_ROOT', Path(directory)):
            name = 'reproduce_existing'
            (Path(directory) / name).mkdir()
            with self.assertRaises(ValueError):
                r.choose_name('ours', name)

    def test_lock_prevents_second_concurrent_reproduction(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as directory, patch.object(r, 'LOCK_PATH', Path(directory) / 'lock'):
            with r.exclusive_lock():
                with self.assertRaises(RuntimeError):
                    with r.exclusive_lock():
                        self.fail('Second lock unexpectedly acquired')

    def test_failed_mock_stage_preserves_receipt_and_log(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as directory:
            root = Path(directory)
            output = root / 'reproduce_failure'
            plan = {'kind': 'synthetic', 'stage': 'ours', 'run_name': output.name, 'delivery_directory': str(output),
                'native_output_directory': str(output / 'outputs'), 'delivery_output_relative_link': None,
                'expected_result': str(output / 'outputs/result.json'), 'command': ['MODEL_PROCESS_NEVER_EXECUTED']}
            with patch.object(r, 'OUTPUT_ROOT', root), patch.object(r, 'execution_guard', return_value={'status': 'READY', 'exit_code': 0}), patch.object(r, 'exclusive_lock', return_value=contextlib.nullcontext()), patch.object(r, 'frozen_inputs', return_value=({}, {})), patch.object(r.subprocess, 'Popen') as launch:
                launch.return_value.wait.return_value = 1
                receipt, code = r.execute_plan(plan)
            self.assertEqual(code, 1)
            self.assertEqual(receipt['status'], 'failed_no_terminal_result')
            self.assertTrue((output / 'plan.json').is_file())
            self.assertTrue((output / 'ours.log').is_file())
            self.assertTrue((output / 'receipt.json').is_file())
            self.assertTrue(receipt['saved_stage_artifacts_preserved'])

    def test_active_generator_check_reports_only_pid_and_script(self):
        snapshot = '123 python -B /some/run_grid.py --execute\n124 python -B /some/unrelated.py\n'
        with patch.object(r.subprocess, 'check_output', return_value=snapshot):
            self.assertEqual(r.active_generators(), [{'pid': 123, 'script': 'run_grid.py'}])


if __name__ == '__main__':
    unittest.main(verbosity=2)
