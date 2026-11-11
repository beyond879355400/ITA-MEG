"""Synthetic tests of append-only/reporting/terminal gates; never imports models."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import review_driver as r


class ReviewEntryTests(unittest.TestCase):
    def test_same_root_append_never_reuses_directory(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as directory:
            first = r.reserve_report(directory)
            second = r.reserve_report(directory)
            self.assertNotEqual(first, second)
            sentinel = first / 'sentinel'
            sentinel.write_text('preserved')
            r.reserve_report(directory)
            self.assertEqual(sentinel.read_text(), 'preserved')

    def test_missing_and_running_metadata_stay_pending(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as directory:
            running = Path(directory) / 'running.json'
            r.save(running, {'status': 'running'})
            rows = r.terminal_check([('missing', Path(directory) / 'missing.json', ('complete',)), ('running', running, ('complete',))])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]['status'], 'running')

    def test_default_pending_does_not_import_terminal_verifier(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as directory:
            out = Path(directory)
            waiting = [{'stage': 'synthetic running', 'reason': 'not terminal'}]
            with patch.object(r, 'terminal_check', return_value=waiting), patch.object(r, 'load_verifier') as loader:
                summary, code = r.run_checks(out, r.DEFAULT_JOB, r.DEFAULT_QUEUE)
            loader.assert_not_called()
            self.assertEqual(code, 3)
            self.assertEqual(summary['status'], 'PENDING')
            self.assertFalse(summary['model_generation_performed'])
            self.assertEqual(summary['models_loaded_or_called'], 0)
            self.assertTrue((out / 'summary.json').is_file())

    def test_output_cannot_escape_delivery(self):
        with self.assertRaises(ValueError):
            r.reserve_report(r.COMPLETION / 'unrelated_reports')


if __name__ == '__main__':
    unittest.main(verbosity=2)
