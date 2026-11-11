"""Model-free wrapper contract tests: never load pretrained tensors."""
import ast
import hashlib
import importlib
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_first as run
import stage_components as components


class RunnerTests(unittest.TestCase):
    def test_fixed_source_only_manifest_and_parameters(self):
        row=run.fixed_cases()[0]
        self.assertEqual(row['case_id'],'fresh_v2_dev_SN004_s00745_t04266')
        self.assertEqual(run.SEED,101)
        self.assertEqual(run.DEFAULTS,dict(scale_ta=1,scale_edit=.2,scale_non_edit=1,clamp_rate=3.,mask_threshold=.5))
        self.assertEqual(row['prompt'],'A tightly cropped photograph of a face with slightly raised inner eyebrows, slightly raised outer eyebrows, slightly lowered brows, slightly lowered lip corners, slightly parted lips, slightly a lowered jaw.')

    def test_target_reference_never_opened(self):
        seen=[]
        original=run.sha
        def checked(path):
            seen.append(str(path));return original(path)
        with patch.object(run,'sha',side_effect=checked):
            run.fixed_cases(0,4)
        data=run.read(run.MANIFEST)
        for row in data['cases']:
            self.assertNotIn(str(run.BUNDLE/row['target_path']),seen)
        tree=ast.parse(Path(run.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant):
                self.assertNotEqual(node.slice.value,'target_path')

    def test_selection_is_bounded_and_not_result_driven(self):
        for start,count in [(-1,1),(4,1),(0,0),(0,5),(3,2)]:
            with self.assertRaises(ValueError):run.fixed_cases(start,count)

    def test_execution_gate_before_asset_or_model_access(self):
        with patch.object(run,'gate',side_effect=ValueError('Waiting for root permission')):
            with patch.object(run,'verify_assets') as assets:
                with self.assertRaisesRegex(ValueError,'permission'):run.execute(SimpleNamespace())
                assets.assert_not_called()

    def test_safe_load_and_no_network_restore(self):
        calls=[]
        def loader(*args,**kwargs):calls.append((args,kwargs));return 'safe'
        fake=SimpleNamespace(load=loader);connect=socket.socket.connect
        with run.offline_and_safe_load(fake):
            self.assertEqual(fake.load('x',map_location='cuda',weights_only=False),'safe')
            with self.assertRaisesRegex(RuntimeError,'Network'):
                socket.socket.connect(None,('example.invalid',443))
        self.assertIs(fake.load,loader);self.assertIs(socket.socket.connect,connect)
        self.assertEqual(calls[0][1],dict(map_location='cpu',weights_only=True))

    def test_git_blob_verification(self):
        with tempfile.TemporaryDirectory(dir=run.HERE,prefix='synthetic_hash_') as temp:
            path=Path(temp)/'x';path.write_bytes(b'hello\n')
            expected=hashlib.sha1(b'blob 6\0hello\n').hexdigest()
            self.assertEqual(components.content_hash(path,'git_blob_sha1'),expected)
            self.assertEqual(components.content_hash(path,'sha256'),hashlib.sha256(b'hello\n').hexdigest())

    def test_no_path_escape(self):
        with self.assertRaises(ValueError):run.relative_inside(run.HERE,'../../escape')
        with self.assertRaises(ValueError):run.relative_inside(run.HERE,'/tmp/outside')

    def test_preflight_is_model_free_and_discloses_source(self):
        with patch.object(run,'verify_assets') as assets:
            doc=run.preflight()
            self.assertFalse(doc['model_construction_performed'])
            self.assertFalse(doc['eligible_for_thesis_main_table'])
            self.assertEqual(doc['base_source_status'],components.STATUS)
            assets.assert_not_called()

    def test_v2_changes_only_proven_overwritten_constructor(self):
        old=(run.HERE/'versions/device_v1/models.py').read_text().rstrip()
        new=(run.VERSION/'models.py').read_text().rstrip()
        old=old.replace('self.unet = UNet2DConditionModel.from_pretrained(\n            pretrained_model_name_path\n        ).to(self.device)',
                        'self.unet = UNet2DConditionModel.from_config(\n            UNet2DConditionModel.load_config(pretrained_model_name_path)\n        ).to(self.device)')
        self.assertEqual(old,new)
        for name in ['infer.py','src/attention_processor.py','src/mask_attention_processor.py','src/mask_ip_controller.py']:
            self.assertEqual((run.HERE/'versions/device_v1'/name).read_bytes().rstrip(),(run.VERSION/name).read_bytes().rstrip())
        proof=run.read(run.HERE/'logs/full_ip_meta_coverage.json')
        self.assertTrue(proof['strict_load_passed'])
        self.assertEqual(proof['state_key_count'],754)
        self.assertEqual(len(proof['shared_alias_groups']),32)
        self.assertEqual(proof['persistent_buffer_count'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
