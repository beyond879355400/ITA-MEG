"""Static38-image entry checks; no actual images/models."""
import ast
import sys
import unittest
import steps50_cpu_eval as entry


class Steps50CpuTests(unittest.TestCase):
    def test_exact_counts(self):
        tree=ast.parse(entry.derived_source());f={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertIn('len(items) == 38',f['openface']);self.assertIn("len(au['records']) == 38",f['merge'])
        self.assertIn("len(bf['records']) == 57",f['merge']);self.assertIn('len(rows) == 76',f['merge'])
        self.assertIn("'generated_count': 38",f['merge']);self.assertIn('node.value = 57',f['derived_identity_source'])
        self.assertNotIn("'source_plus_real_driver'",f['merge'])
    def test_neural_and_metric_helpers_unchanged(self):
        old=ast.parse(entry.CORE.read_text());new=ast.parse(entry.derived_source())
        a={x.name:x for x in old.body if isinstance(x,ast.FunctionDef)};b={x.name:x for x in new.body if isinstance(x,ast.FunctionDef)}
        self.assertEqual(ast.dump(a['enrich_new_measurement'],include_attributes=False),ast.dump(b['enrich_new_measurement'],include_attributes=False))
        self.assertIn("steps50_[a-z0-9_]+",entry.derived_source())
    def test_import_is_not_model_execution(self):
        self.assertFalse(any(n in sys.modules for n in ('torch','cv2','PIL','onnxruntime')))


if __name__=='__main__':unittest.main(verbosity=2)
