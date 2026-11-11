"""Synthetic files/commands only; no original completion/model/media is read."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_reproduction as r


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return r.sha(path)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        base = r.HERE / 'synthetic_reproduction_tests'
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.final = self.root / 'final_test'
        self.original = self.final / 'execution_202609150734'
        protocol_sha = put(self.final / 'frozen_protocol.json', {'fixture': True})
        font_sha = put(self.final / 'run_with_fontcache.py', '# synthetic font wrapper\n')
        stat_source = '''import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
source=Path(a.input).read_bytes()
out={'metadata':json.loads(source)['metadata'],'input_sha256':hashlib.sha256(source).hexdigest(),'statistics_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixture':True}
with Path(a.output).open('x') as h:json.dump(out,h)
'''
        statistics_sha = put(self.root / 'statistics/paired_subject_statistics.py', stat_source)
        queue_source = f'''import json
PY={sys.executable!r}
PSHA={protocol_sha!r}
def main():
    RUN.mkdir(parents=True,exist_ok=False)
    with (RUN/'complete.json').open('x') as h:json.dump({{'fixture':True}},h)
'''
        queue_sha = put(self.final / 'run_frozen_queue.py', queue_source)
        stats = self.original / 'measurements/statistics'
        input_sha = put(stats / 'paired_subject_input.json', {'metadata': {'protocol_sha256': protocol_sha}, 'fixture': True})
        result_sha = put(stats / 'paired_subject_results.json', {'metadata': {'protocol_sha256': protocol_sha},
            'input_sha256': input_sha, 'statistics_source_sha256': statistics_sha})
        put(self.original / 'complete.json', {'status': 'computed_pending_independent_review',
            'protocol_sha256': protocol_sha, 'statistics': {'path': str(stats / 'paired_subject_results.json'), 'sha256': result_sha}})
        self.context = r.Context(self.root, sys.executable, protocol_sha, queue_sha, font_sha, statistics_sha, 'unused')

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, mode, **kwargs):
        return r.execute(mode, self.context, busy_check=kwargs.get('busy_check', lambda _: []))

    def test_statistics_new_directory_originals_unchanged(self):
        old = {str(p): p.read_bytes() for p in self.original.rglob('*.json')}
        first = self.execute('statistics')
        second = self.execute('statistics')
        self.assertNotEqual(first['directory'], second['directory'])
        self.assertFalse(first['models_run'])
        self.assertFalse((self.final / '.reproduction.lock').exists())
        self.assertEqual(old, {str(p): p.read_bytes() for p in self.original.rglob('*.json')})

    def test_full_import_only_redirects_run(self):
        queue = (self.final / 'run_frozen_queue.py').read_bytes()
        report = self.execute('full')
        self.assertTrue(Path(report['complete']['path']).is_file())
        self.assertNotEqual(Path(report['queue_run']), self.original)
        self.assertEqual(queue, (self.final / 'run_frozen_queue.py').read_bytes())
        self.assertTrue((self.original / 'complete.json').is_file())

    def test_missing_completion_has_no_writes_or_subprocess(self):
        (self.original / 'complete.json').unlink()
        before = set(self.root.rglob('*'))
        with patch.object(r.subprocess, 'run', side_effect=AssertionError('subprocess called')):
            for mode in ('statistics', 'full'):
                with self.assertRaisesRegex(ValueError, 'not complete'):
                    self.execute(mode)
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_busy_or_lock_refuses_new_run(self):
        with self.assertRaisesRegex(ValueError, 'still running'):
            self.execute('full', busy_check=lambda _: [{'pid': 123}])
        self.assertFalse((self.final / 'reproduction_runs').exists())
        put(self.final / '.reproduction.lock', 'existing owner')
        with self.assertRaises(FileExistsError):
            self.execute('statistics')
        self.assertEqual((self.final / '.reproduction.lock').read_text(), 'existing owner')

    def test_source_input_output_hash_tamper_refused(self):
        for path in (self.final / 'run_frozen_queue.py',
                     self.original / 'measurements/statistics/paired_subject_input.json',
                     self.original / 'measurements/statistics/paired_subject_results.json'):
            old = path.read_bytes()
            path.write_bytes(b'tampered fixture')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                self.execute('statistics')
            path.write_bytes(old)
        self.assertFalse((self.final / 'reproduction_runs').exists())

    def test_failure_retains_lock_and_failure_receipt(self):
        with patch.object(r.subprocess, 'run', side_effect=OSError('synthetic launch failure')):
            with self.assertRaises(OSError):
                self.execute('statistics')
        self.assertTrue((self.final / '.reproduction.lock').is_file())
        self.assertEqual(len(list((self.final / 'reproduction_runs').glob('*/reproduction_failed.json'))), 1)
        with self.assertRaises(FileExistsError):
            self.execute('full')

    def test_default_verify_readonly_and_conflicting_flags(self):
        with patch.object(r, 'execute', side_effect=AssertionError('execution called')), \
             patch.object(r, 'check'), patch.object(r.subprocess, 'run') as runner:
            runner.return_value.returncode = 3
            self.assertEqual(r.main([]), 3)
            self.assertEqual(r.main(['--verify']), 3)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                r.main(['--full', '--statistics'])
            self.assertEqual(error.exception.code, 2)

    def test_process_probe_flags_busy_generators(self):
        rows = '999999 python /fixture/run_frozen_queue.py --execute\n999998 python /fixture/run_reproduction.py --full\n'
        with patch.object(r.subprocess, 'check_output', return_value=rows):
            self.assertEqual(len(r.active_commands(self.context)), 2)


if __name__ == '__main__':
    unittest.main()
