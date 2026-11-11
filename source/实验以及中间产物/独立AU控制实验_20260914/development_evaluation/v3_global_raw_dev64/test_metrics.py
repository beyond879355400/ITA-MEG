import unittest
from evaluate import au_metrics,summarize

class MetricsTest(unittest.TestCase):
    def test_no_edit_empty_C(self):
        r=au_metrics([0]*12,[0]*12,[0]*12,[0]*12)
        self.assertIsNone(r['target_mae_changed']);self.assertIsNone(r['delta_mae_changed'])
        self.assertEqual(r['unchanged_source_relative_drift'],0)
    def test_changed_not_preselected_au(self):
        source=[0]*12;target=[0]*12;target[11]=2
        r=au_metrics([1]*12,[.5]*12,source,target)
        self.assertEqual(r['changed_aus'],[26]);self.assertEqual(r['target_mae_changed'],1)
        self.assertEqual(r['delta_mae_changed'],1.5);self.assertEqual(r['unchanged_source_relative_drift'],.5)
    def test_source_failure_preserves_absolute_C(self):
        target=[1]+[0]*11;r=au_metrics([0]*12,None,[0]*12,target)
        self.assertEqual(r['target_mae_changed'],1);self.assertIsNone(r['delta_mae_changed'])
    def test_prediction_failure_is_missing(self):
        r=au_metrics(None,[0]*12,[0]*12,[1]*12)
        self.assertIsNone(r['target_mae_changed']);self.assertIsNone(r['unchanged_source_relative_drift'])
    def test_decreasing_request_has_correct_direction(self):
        r=au_metrics([0]*12,[1]*12,[1]*12,[0]*12)
        self.assertEqual(r['direction_correct_fraction'],1);self.assertEqual(r['delta_mae_changed'],0)

if __name__=='__main__':unittest.main()
