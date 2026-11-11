import unittest
import json
from pathlib import Path
import tempfile
from verify_manual_labels import HERE, AUS, parse_labels, validate_plan, verify


class ManualLabelTests(unittest.TestCase):
    def test_explicit_frames_not_line_position(self):
        self.assertEqual(parse_labels('1,0\r\n3,2\r\n'), {1: 0, 3: 2})

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            parse_labels('1,0\n1,2\n')

    def test_invalid_intensity_rejected(self):
        with self.assertRaises(ValueError):
            parse_labels('1,6\n')

    def test_zero_frame_rejected(self):
        with self.assertRaises(ValueError):
            parse_labels('0,1\n')

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            parse_labels('')

    def good_plan(self):
        return {'kind': 'frozen_preservation_factorial_v1', 'status': 'frozen', 'cases': [
            {'case_id': f'case_{s}_{i}', 'subject': f'SN{s:03d}', 'au_order': list(AUS),
             'source_frame_1based': 1, 'target_frame_1based': i + 2,
             'source_au_vector_0_5': [0]*12, 'target_au_vector_0_5': [1] + [0]*11}
            for s in range(1, 9) for i in range(3)]}

    def test_good_plan(self):
        validate_plan(self.good_plan())

    def test_empty_verify_cannot_pass_or_write(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root = Path(folder)
            freeze, output = root / 'freeze.json', root / 'result.json'
            freeze.write_text(json.dumps({'cases': []}))
            with self.assertRaises(ValueError):
                verify(freeze, root / 'labels', output)
            self.assertFalse(output.exists())

    def test_incomplete_plan_cannot_pass(self):
        for field, value in [('au_order', []), ('source_frame_1based', True),
                             ('target_au_vector_0_5', []), ('subject', '../other')]:
            with self.subTest(field=field):
                plan = self.good_plan()
                plan['cases'][0][field] = value
                with self.assertRaises(ValueError):
                    validate_plan(plan)

    def test_duplicate_case_rejected(self):
        plan = self.good_plan()
        plan['cases'][1]['case_id'] = plan['cases'][0]['case_id']
        with self.assertRaises(ValueError):
            validate_plan(plan)


if __name__ == '__main__':
    unittest.main()
