import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from render_audit_sheets import HERE, journal_snapshot


class SnapshotTests(unittest.TestCase):
    def snapshot(self, data):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path = Path(folder) / 'journal.jsonl'
            path.write_bytes(data)
            return journal_snapshot(path)

    def line(self):
        return (json.dumps({'case_id': '测试', 'seed': 101, 'arm': 'full'}, ensure_ascii=False) + '\n').encode()

    def test_complete(self):
        data = self.line()
        rows, receipt = self.snapshot(data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(receipt['pending_tail_bytes'], 0)
        self.assertEqual(receipt['complete_prefix_sha256'], hashlib.sha256(data).hexdigest())

    def test_partial_utf8_tail(self):
        data = self.line()
        rows, receipt = self.snapshot(data + b'{"case_id":"\xe7\xbb')
        self.assertEqual(len(rows), 1)
        self.assertEqual(receipt['complete_prefix_bytes'], len(data))
        self.assertGreater(receipt['pending_tail_bytes'], 0)

    def test_only_partial(self):
        rows, receipt = self.snapshot(b'\xe7\xbb')
        self.assertEqual(rows, {})
        self.assertEqual(receipt['pending_tail_bytes'], 2)

    def test_bad_complete_utf8_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            self.snapshot(b'\xe7\xbb\n')

    def test_bad_complete_json_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            self.snapshot(b'{invalid}\n')


if __name__ == '__main__':
    unittest.main()
