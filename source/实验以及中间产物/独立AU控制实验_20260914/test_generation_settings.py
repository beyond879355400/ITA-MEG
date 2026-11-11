"""CPU parameter/source gates only; does not call main or load any model."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
import generate_centered_settings_development as settings
import generate_centered_development as original
from test_generation_manifest_gate import ManifestGateTests as _OriginalGateTests


class SettingsManifestTests(_OriginalGateTests):
    def validate(self,doc,bindings=True):
        path=self.root/'input.json';self.write(path,doc)
        return settings.validate_development_manifest(path,doc,self.samples,'cache','split',self.bindings if bindings else None)


class SettingsParameterTests(unittest.TestCase):
    def test_only_two_explicit_numeric_scales(self):
        for value in ('0.9',.9,'1.0',1):
            result=settings.conditioning_scales(value)
            self.assertEqual(result,{'au_scale':1.0,'ip_scale':float(value)})
        for value in (True,False,None,'nan',float('inf'),0,-1,1.1,.9001,'auto'):
            with self.subTest(value=value),self.assertRaises(argparse.ArgumentTypeError):settings.conditioning_scales(value)

    def test_cli_requires_explicit_ip_scale(self):
        parser=settings.create_argument_parser();base=['--cache','unused','--cases','unused','--output','unused']
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parser.parse_args(base)
        for value in ('0.9','1.0'):
            args=parser.parse_args(base+['--ip-scale',value]);self.assertEqual(args.ip_scale,float(value))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parser.parse_args(base+['--ip-scale','1.5'])

    def test_all_other_sampling_defaults_unchanged_and_no_torch_import(self):
        before=set(sys.modules)
        args=settings.create_argument_parser().parse_args(['--cache','unused','--cases','unused','--output','unused','--ip-scale','1.0'])
        self.assertEqual(args.steps,20);self.assertEqual(args.guidance_scale,7.5)
        self.assertEqual(args.preview_group,'all');self.assertFalse(args.skip_baseline);self.assertEqual(args.checkpoint,[])
        self.assertNotIn('torch',set(sys.modules)-before)

    def test_original_bytes_and_manifest_gate_unchanged(self):
        self.assertEqual(hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),'fbba652deaf7f1dbdb3972799b00a010e5782eaedddab40f85d4154941e9a83f')
        def function_tree(module,name):
            return next(node for node in ast.parse(Path(module.__file__).read_text()).body if isinstance(node,ast.FunctionDef) and node.name==name)
        self.assertEqual(ast.dump(function_tree(original,'validate_development_manifest'),include_attributes=False),ast.dump(function_tree(settings,'validate_development_manifest'),include_attributes=False))

    def test_actual_cross_kwargs_and_metadata_use_the_same_scales(self):
        tree=ast.parse(Path(settings.__file__).read_text())
        assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)]
        cross=next(node.value for node in assignments if any(isinstance(t,ast.Name) and t.id=='cross' for t in node.targets))
        self.assertIsInstance(cross,ast.Dict)
        self.assertTrue(any(key is None and isinstance(value,ast.Name) and value.id=='scales' for key,value in zip(cross.keys,cross.values)))
        record=next(node.value for node in assignments if any(isinstance(t,ast.Name) and t.id=='record' for t in node.targets))
        value=next(value for key,value in zip(record.keys,record.values) if isinstance(key,ast.Constant) and key.value=='ip_scale')
        self.assertEqual(ast.dump(value),ast.dump(ast.parse("scales['ip_scale']",mode='eval').body))
        metadata=next(node for node in ast.walk(tree) if isinstance(node,ast.Dict) and any(isinstance(key,ast.Constant) and key.value=='setting_status' for key in node.keys))
        self.assertTrue(any(key is None and isinstance(value,ast.Name) and value.id=='scales' for key,value in zip(metadata.keys,metadata.values)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(SettingsParameterTests),unittest.defaultTestLoader.loadTestsFromTestCase(SettingsManifestTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x') as handle:json.dump({'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'passed':result.wasSuccessful(),
            'cpu_only':True,'models_called':False,'images_generated':0,'test_pixels_or_metrics_read':False,
            'settings_generator_sha256':hashlib.sha256(Path(settings.__file__).read_bytes()).hexdigest(),'original_generator_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()},handle,indent=2)
    raise SystemExit(not result.wasSuccessful())
