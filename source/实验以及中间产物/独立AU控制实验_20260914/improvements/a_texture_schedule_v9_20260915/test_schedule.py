"""CPU-only tests of the actual derived V9 callbacks; no models/test images."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('late_au_under_test', HERE / 'run_late_au.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
DERIVED = ast.parse(runner.source)
ORIGINAL = ast.parse(runner.raw.decode())
EXECUTE = next(n for n in DERIVED.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
CONFIG_LOOP = next(n for n in ast.walk(EXECUTE) if isinstance(n, ast.For)
                   and ast.unparse(n.target) == '(config, strength, au_scale)')
CALLBACK = next(n for n in CONFIG_LOOP.body if isinstance(n, ast.FunctionDef) and n.name == 'step_end')
PIPELINE = (Path(sys.prefix) / 'lib/python3.10/site-packages/diffusers/pipelines/'
            'stable_diffusion/pipeline_stable_diffusion_img2img.py')
PIPE_TREE = ast.parse(PIPELINE.read_text())
PIPE_CLASS = next(n for n in PIPE_TREE.body if isinstance(n, ast.ClassDef)
                  and n.name == 'StableDiffusionImg2ImgPipeline')
PIPE_CALL = next(n for n in PIPE_CLASS.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
PROPERTY = next(n for n in PIPE_CLASS.body if isinstance(n, ast.FunctionDef)
                and n.name == 'cross_attention_kwargs')
PROPERTY_NS = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(PROPERTY)], type_ignores=[])),
             str(PIPELINE), 'exec'), PROPERTY_NS)
TIMESTEPS = torch.tensor([999, 949, 899, 849, 799, 749, 699, 649, 599, 549,
                         500, 450, 400, 350, 300, 250, 200, 150, 100, 50])


class TinyUNet(torch.nn.Module):
    def forward(self, latent, t, cross_attention_kwargs):
        return latent + cross_attention_kwargs['au_scale']


class FakeScheduler:
    timesteps = TIMESTEPS

    @staticmethod
    def add_noise(source, noise, t):
        return source + noise * (t[0].float() / 1000)


class TinyPipeline:
    cross_attention_kwargs = PROPERTY_NS['cross_attention_kwargs']

    def __init__(self, unet):
        self.unet = unet
        self.scheduler = FakeScheduler()

    def run(self, source, cross_attention_kwargs, callback):
        # Same reference assignment and execution order verified against installed source.
        self._cross_attention_kwargs = cross_attention_kwargs
        assert self.cross_attention_kwargs is cross_attention_kwargs
        latent = source.clone()
        for i, t in enumerate(self.scheduler.timesteps[5:]):
            latent = self.unet(latent, t, cross_attention_kwargs=self.cross_attention_kwargs)
            latent = callback(self, i, t, {'latents': latent})['latents']
        return latent


def build_loop_fixture():
    """Reuse actual init, hook, callback and cleanup inside a genuine repeated loop."""
    start = ast.parse('''
def run_fixture():
    outputs = []
    backbone = SimpleNamespace(unet=TinyUNet())
    pipeline = TinyPipeline(backbone.unet)
    for case in range(2):
        for method in ('baseline', 'bidirectional'):
            cross = {'ip_scale': .9}
            for config, strength, au_scale in CONFIGS:
                pass
    return outputs, backbone.unet
''')
    target = next(n for n in ast.walk(start) if isinstance(n, ast.For)
                  and ast.unparse(n.target) == '(config, strength, au_scale)')
    noise_index = next(i for i, n in enumerate(CONFIG_LOOP.body)
                       if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'noise')
    target.body = copy.deepcopy(CONFIG_LOOP.body[:noise_index])
    target.body += ast.parse('''
src = torch.full((1, 1, 2, 2), 0.25 + case)
noise = torch.ones_like(src)
mask = torch.full_like(src, float(case))
history = []
''').body
    target.body += [copy.deepcopy(CALLBACK)]
    target.body += ast.parse('''
latent = pipeline.run(src, cross, step_end)
scale_guard.remove()
outputs.append({'case': case, 'method': method, 'late': au_scale,
    'trace': actual_scales, 'history': history, 'final_scale': cross['au_scale'],
    'latent': latent, 'source': src})
''').body
    namespace = dict(torch=torch, SimpleNamespace=SimpleNamespace, TinyUNet=TinyUNet,
                     TinyPipeline=TinyPipeline, CONFIGS=runner.core['CONFIGS'],
                     need=runner.core['need'], late_scale=runner.late_scale,
                     project_latents=runner.core['project_latents'], v5=runner.core['v5'])
    exec(compile(ast.fix_missing_locations(start), '<actual-v9-callback-fixture>', 'exec'), namespace)
    return namespace['run_fixture']


class TestSchedule(unittest.TestCase):
    def test_kernel_pin_and_fixed_configs(self):
        self.assertEqual(hashlib.sha256(runner.raw).hexdigest(), runner.EXPECTED)
        self.assertEqual(runner.core['CONFIGS'], [('late_au100', .75, 1.),
                         ('late_au050', .75, .5), ('late_au000', .75, 0.)])

    def test_threshold_and_reset(self):
        for late in (0., .5, 1.):
            self.assertEqual(runner.late_scale(251, late), 1.5)
            self.assertEqual(runner.late_scale(250, late), late)
            self.assertEqual(runner.late_scale(0, late), late)
            self.assertEqual(runner.late_scale(None, late), 1.5)
        with self.assertRaises(ValueError):
            runner.late_scale(50, 1.5)

    def test_actual_scheduler_uses_exact_last_five_timesteps(self):
        scheduler_path = PIPELINE.parents[2] / 'schedulers/scheduling_deis_multistep.py'
        tree = ast.parse(scheduler_path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'set_timesteps')
        method = copy.deepcopy(method)
        for arg in method.args.args:
            arg.annotation = None
        namespace = {'torch': torch, 'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(scheduler_path), 'exec'), namespace)
        runtime = json.loads((runner.V7 / 'runs/smooth_202609151710/runtime.json').read_text())
        config = runtime['scheduler_config']
        self.assertEqual((config['timestep_spacing'], config['num_train_timesteps'],
                          config['use_karras_sigmas']), ('linspace', 1000, False))
        scheduler = SimpleNamespace(config=SimpleNamespace(**config),
                                    alphas_cumprod=torch.linspace(.999, .001, 1000))
        namespace['set_timesteps'](scheduler, 20, 'cpu')
        self.assertTrue(torch.equal(scheduler.timesteps, TIMESTEPS))
        self.assertEqual(scheduler.timesteps[5:][scheduler.timesteps[5:] <= 250].tolist(),
                         [250, 200, 150, 100, 50])

    def test_installed_pipeline_shares_cross_and_calls_hook_before_callback(self):
        code = ast.unparse(PIPE_CALL)
        self.assertIn('self._cross_attention_kwargs = cross_attention_kwargs', code)
        self.assertEqual(ast.unparse(PROPERTY.body[0]), 'return self._cross_attention_kwargs')
        self.assertIn('cross_attention_kwargs=self.cross_attention_kwargs', code)
        self.assertLess(code.index('noise_pred = self.unet('),
                        code.index('callback_outputs = callback_on_step_end('))
        self.assertIn('latent_model_input, t,', code)

    def test_actual_hooks_across_cases_methods_configs(self):
        outputs, unet = build_loop_fixture()()
        self.assertEqual(len(outputs), 12)
        self.assertEqual(len(unet._forward_pre_hooks), 0)
        for row in outputs:
            self.assertEqual([r['t'] for r in row['trace']], TIMESTEPS[5:].tolist())
            self.assertEqual([r['au_scale'] for r in row['trace']], [1.5] * 10 + [row['late']] * 5)
            self.assertEqual([r['au_scale_used'] for r in row['history']],
                             [r['au_scale'] for r in row['trace']])
            self.assertEqual([r['next_t'] for r in row['history']], TIMESTEPS[6:].tolist() + [None])
            self.assertEqual(row['final_scale'], 1.5)
            if row['case'] == 0:
                self.assertTrue(torch.equal(row['latent'], row['source']))

    def test_actual_hook_fails_on_wrong_used_scale(self):
        fixture = build_loop_fixture()
        fixture.__globals__['late_scale'] = lambda t, late: -1
        with self.assertRaisesRegex(ValueError, 'Actual UNet AU schedule mismatch'):
            fixture()

    def test_projection_is_unchanged_and_uses_next_t(self):
        get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == 'project_latents')
        self.assertEqual(ast.dump(get(DERIVED)), ast.dump(get(ORIGINAL)))
        source = torch.ones(1, 1, 2, 2)
        mask = torch.zeros_like(source)
        projection = runner.core['project_latents']
        self.assertTrue(torch.equal(projection(source * 99, source, source, mask,
                        FakeScheduler(), torch.tensor(200), torch), source * 1.2))
        self.assertTrue(torch.equal(projection(source * 99, source, source, mask,
                        FakeScheduler(), None, torch), source))

    def test_noedit_and_original_input_guards_retained(self):
        original_inputs = next(n for n in ORIGINAL.body if isinstance(n, ast.FunctionDef) and n.name == 'inputs')
        derived_inputs = next(n for n in DERIVED.body if isinstance(n, ast.FunctionDef) and n.name == 'inputs')
        self.assertEqual(ast.dump(original_inputs), ast.dump(derived_inputs))
        self.assertIn("need(row['sha256']==c['vae_reconstruction']['sha256'],'Empty mask must reconstruct source')", runner.source)
        self.assertEqual(runner.source.count("V7/'region_mask.py'"), 2)
        self.assertIn("need(condition_hashes==c[method+'_text2img_condition_hashes']", runner.source)

    def test_a_reference_mapping_and_hash_failure_without_image_reads(self):
        records = json.loads((runner.V7 / 'runs/smooth_202609151710/result.json').read_text())['records']
        by_key = {(r['case_id'], r['method']): r for r in records}
        ids = sorted({r['case_id'] for r in records})
        hashes = {r['output']: r['sha256'] for r in records}
        fresh_cases = lambda: [{'case_id': cid} for cid in ids]
        with patch.object(runner, 'original_inputs', fresh_cases), patch.dict(runner.core, sha=hashes.__getitem__):
            mapped = runner.inputs_with_a()
        self.assertEqual(len(mapped), 8)
        for case in mapped:
            for method in ('baseline', 'bidirectional'):
                old = by_key[case['case_id'], method + '_smooth_s075_a150']
                self.assertEqual(case['a_' + method], {'path': old['output'], 'sha256': old['sha256']})
        with patch.object(runner, 'original_inputs', fresh_cases), patch.dict(runner.core, sha=lambda _: 'bad'):
            with self.assertRaisesRegex(RuntimeError, 'A image changed'):
                runner.inputs_with_a()


if __name__ == '__main__':
    unittest.main(verbosity=2)
