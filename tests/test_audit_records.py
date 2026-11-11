import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("auditor", ROOT/"scripts/audit_records.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class AuditTests(unittest.TestCase):
    def test_subject_equal_not_image_equal(self):
        data = [
            dict(method="m",subject="a",case_id="a1",seed="1",group="edited",x="0"),
            dict(method="m",subject="a",case_id="a1",seed="2",group="edited",x="2"),
            dict(method="m",subject="a",case_id="a2",seed="1",group="edited",x="3"),
            dict(method="m",subject="b",case_id="b1",seed="1",group="edited",x="10")]
        self.assertEqual(audit.aggregate(data,"m","x","edited"),(6.0,2,3))
    def test_missing_values_are_not_zero(self):
        rows=[dict(method="m",subject="a",case_id="a",seed="1",group="edited",x="")]
        self.assertEqual(audit.aggregate(rows,"m","x"),(None,0,0))
    def test_duplicate_keys_fail(self):
        r=dict(method="m",subject="a",case_id="a",seed="1",group="edited",x="1")
        with self.assertRaises(ValueError):audit.aggregate([r,r],"m","x")
    def test_nonfinite_values_fail(self):
        for x in ["NaN","inf","-inf"]:
            with self.assertRaises(ValueError):audit.number(x)
    def test_metric_mismatch_fails(self):
        with self.assertRaises(ValueError):audit.check(.2,.3,"test")
    def test_holm(self):
        got=audit.holm([.01,.04,.03])
        for a,b in zip(got,[.03,.06,.06]):self.assertAlmostEqual(a,b)
    def test_none_does_not_equal_zero(self):
        with self.assertRaises(ValueError):audit.check(None,0,"missing")
    def test_repository_archived_records(self):
        got=audit.audit(ROOT)
        self.assertEqual(got["status"],"PASS")
        self.assertEqual(got["main_measurement_rows"],896)
        self.assertEqual(got["ablation_measurement_rows"],288)
        self.assertFalse(got["models_loaded"])

if __name__ == "__main__":
    unittest.main()
