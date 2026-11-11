"""Synthetic tests of independently implemented dev19 verification arithmetic."""
import csv
from pathlib import Path
import tempfile
import unittest
import verify_dev19 as v


class VerificationTests(unittest.TestCase):
    def test_largest68bbox_and_tie_first_are_recomputed_from_csv(self):
        with tempfile.TemporaryDirectory(dir=v.HERE) as folder:
            path = Path(folder) / 'synthetic.csv'
            rows = []
            for face, scale in enumerate((1., 2., 2.)):
                row = {'face': str(face), 'confidence': '.98'}
                for i in range(68):
                    row['x_' + str(i)] = str(i * scale)
                    row['y_' + str(i)] = str(i * scale)
                for au in v.AUS17:
                    row[f'AU{au:02d}_r'] = str(face + .1)
                rows.append(row)
            with path.open('x') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = v.independent_openface(path)
            self.assertEqual(result['selected_row'], 1)
            self.assertEqual(result['vector'], [1.1] * 12)
            self.assertEqual(result['second_to_first_area_ratio'], 1.)

    def test_changed_metrics_noedit_na_and_extra_reference_are_distinct(self):
        source = {'vector': [0.] * 12, 'selected_au17': {str(a): 0. for a in v.AUS17}}
        predicted = {'vector': [1.] * 12, 'selected_au17': {str(a): 1. for a in v.AUS17}}
        target = {'vector': [.5] * 12, 'selected_au17': {str(a): .5 for a in v.AUS17}}
        values = v.metrics(predicted, source, target, [0] * 12, [0] * 12)
        self.assertIsNone(values['target_mae_changed'])
        self.assertIsNone(values['delta_mae_changed'])
        self.assertEqual(values['unchanged_source_relative_drift'], 1.)
        self.assertEqual(values['extra5_proxy_mae_to_real_target'], .5)
        changed = v.metrics(predicted, source, target, [0] * 12, [2] + [0] * 11)
        self.assertEqual(changed['target_mae_changed'], 1.)
        self.assertEqual(changed['delta_mae_changed'], 1.)
        self.assertEqual(changed['direction_correct_fraction'], 1.)

    def test_equal_subject_not_equal_case_weighting_and_common_denominator(self):
        rows = []
        for cid, subject, a, b in (('c1', 's1', 0., 1.), ('c2', 's1', 2., 3.), ('c3', 's2', 9., 10.)):
            rows += [{'case_id': cid, 'subject': subject, 'method': method, 'metric': value} for method, value in (('a', a), ('b', b))]
        result = v.aggregate(rows, ['a', 'b'], 'metric')
        self.assertEqual(result['subject_equal_means'], {'a': 5., 'b': 6.})
        self.assertEqual(result['new_minus_reference'], 1.)
        rows[-1]['metric'] = None
        common = v.aggregate(rows, ['a', 'b'], 'metric')
        self.assertEqual(common['planned_cases'], 3)
        self.assertEqual(common['common_cases'], 2)
        self.assertEqual(common['subjects'], ['s1'])

    def test_inprogress32path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'in-progress32'):
            v.allowed(v.COMPLETION / 'generation_runs' / 'unseen32' / 'image.png')

    def test_numeric_mismatch_and_duplicate_denominator_fail(self):
        with self.assertRaises(ValueError):
            v.same(.8518, .8519)
        with self.assertRaises(ValueError):
            v.mapping([{'case_id': 'c', 'method': 'm'}] * 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
