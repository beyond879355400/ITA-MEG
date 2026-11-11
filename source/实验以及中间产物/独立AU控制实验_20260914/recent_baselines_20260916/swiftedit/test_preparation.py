"""CPU-only mask and archive tests. No official model or network call."""
import ast
import copy
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
import stage_weights as stage

HERE = Path(__file__).resolve().parent


def actual_mask_function():
    path = HERE / 'versions/device_v1/infer.py'
    tree = ast.parse(path.read_text())
    binary = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'to_binary')
    edit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'edit_image')
    selected, enabled = [], False
    for node in edit.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            enabled = True
        if enabled:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'input_sb':
                break
            selected.append(copy.deepcopy(node))
    function = ast.parse('def mask(predict_inverted_code,clamp_rate=3.,mask_threshold=.5,device="cpu"):\n pass').body[0]
    function.body = selected + [ast.Return(value=ast.Name(id='mask12', ctx=ast.Load()))]
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[binary,function], type_ignores=[])), str(path), 'exec'), scope)
    return scope['mask']


class PreparationTests(unittest.TestCase):
    def test_nonzero_mask_matches_official_formula_exactly(self):
        prediction = torch.arange(128,dtype=torch.float32).reshape(2,4,4,4) / 8
        prediction[1,:,2:,2:] += 20
        source, target = prediction.chunk(2)
        diff = (source-target).abs().mean(dim=[0,1]); maximum = float(diff.mean()*3)
        expected = (diff.clamp(0,maximum)/maximum > .5).float()
        self.assertTrue(torch.equal(actual_mask_function()(prediction), expected))

    def test_exact_zero_difference_is_finite_zero_mask(self):
        p = torch.ones(2,4,4,4)
        mask = actual_mask_function()(p)
        self.assertTrue(torch.isfinite(mask).all())
        self.assertEqual(torch.count_nonzero(mask).item(), 0)

    def test_nonfinite_difference_is_rejected(self):
        for bad in (float('nan'), float('inf')):
            p = torch.ones(2,4,4,4); p[1,0,0,0] = bad
            with self.assertRaises(ValueError):
                actual_mask_function()(p)

    def test_official_defaults_and_generation_route_preserved(self):
        path = HERE/'versions/device_v1/infer.py'; tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='edit_image')
        self.assertEqual([ast.literal_eval(v) for v in fn.args.defaults], [1,.2,1,3.,.5,'mps'])
        text = path.read_text()
        self.assertIn('ip_sb_model.gen_img(', text)
        self.assertIn('.latent_dist.sample()', text)
        self.assertIn('where=["mid_blocks", "up_blocks"]', text)
        self.assertNotIn('"cuda"', text)
        self.assertNotIn('return pil_img_cond', text)

    def test_multifile_reader_and_stream_archive(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='synthetic_archive_') as temp:
            root = Path(temp); memory = io.BytesIO()
            with tarfile.open(fileobj=memory,mode='w:gz') as tar:
                payload = b'not a real checkpoint'
                info = tarfile.TarInfo('swiftedit_weights/model.bin'); info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            data = memory.getvalue(); cuts = [data[:17],data[17:39],data[39:]]; paths=[]
            for i, part in enumerate(cuts):
                path=root/('part'+str(i));path.write_bytes(part);paths.append(path)
            with io.BufferedReader(stage.MultipartReader(paths)) as combined:
                self.assertEqual(combined.read(),data)
            info=stage.inspect(paths)
            self.assertEqual(info['expanded_bytes'],len(payload))
            self.assertEqual(info['members'][0]['name'],'swiftedit_weights/model.bin')
            self.assertFalse((root/'merged.tar.gz').exists())

    def test_unsafe_tar_members_rejected(self):
        for name, kind in [('../escape',tarfile.REGTYPE),('/escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE)]:
            info=tarfile.TarInfo(name);info.type=kind
            with self.assertRaises(ValueError):stage.safe_member(info)

    def test_official_asset_totals_and_sha_digests(self):
        entries=stage.assets()
        self.assertEqual(sum(row['size'] for row in entries),9748940800)
        self.assertTrue(all(row['digest'].startswith('sha256:') for row in entries))

    def test_download_and_extract_are_gated(self):
        with patch.object(stage,'read',return_value={'cache_ready':None}):
            for action in (stage.download,stage.extract):
                with self.assertRaisesRegex(ValueError,'Waiting for root permission'):action()

    def test_attention_code_exact_except_patch_serialized_eof_whitespace(self):
        lock=stage.read(HERE/'source_lock.json')
        repo=Path.home()/'Projects/github/Qualcomm-AI-research/SwiftEdit'
        for path,digest in lock['files'].items():
            if path.startswith('src/') and path.endswith('.py'):
                original=(repo/path).read_bytes()
                self.assertEqual(hashlib.sha256(original).hexdigest(),digest)
                self.assertEqual((HERE/'versions/device_v1'/path).read_bytes().rstrip(),original.rstrip())


if __name__=='__main__':
    unittest.main(verbosity=2)
