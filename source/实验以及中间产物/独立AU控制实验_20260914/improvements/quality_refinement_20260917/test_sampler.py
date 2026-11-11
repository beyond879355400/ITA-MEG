"""CPU synthetic tests; no real model, image dataset, MPS or old venv import."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

import sampler as q

torch.set_num_threads(1)
E = q.ROOT.parent.parent


def pure_definitions(path, names, namespace):
    """Load only named pure definitions, never a legacy module's top-level loader."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


TIMES20 = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
v10dict = pure_definitions(E/'improvements/a_terminal_cleanup_v10_20260915/run_terminal.py',
    ['need', 'finite', 'tensor_sha', 'cfg_epsilon', 'terminal_coefficients', 'estimate_clean', 'TerminalController'],
    {'hashlib': hashlib, 'TIMESTEPS': TIMES20,
     'MODES': ('terminal_last_eps', 'terminal_t0', 'terminal_t0_lp')})
v10 = SimpleNamespace(**v10dict)
v11dict = pure_definitions(E/'improvements/a_local_cleanup_v11_20260915/run_local.py',
    ['need', 'filter_metadata', 'local_filtered_clean'],
    {'math': __import__('math'), 'finite': v10.finite, 'MODES': {'local_lp_s050': .5, 'local_lp_s075': .75}})
v11 = SimpleNamespace(v10=v10, **v11dict)
oldgrid = pure_definitions(E/'completion_20260915/generation/run_grid.py',
    ['fixed_controller', 'sampling_callback'], {'need': q.need, 'tensor_sha': q.tensor_sha, 'TIMESTEPS': TIMES20})


class Scheduler:
    order = 1
    config = SimpleNamespace(prediction_type='epsilon', thresholding=False)

    def configure(self, steps):
        self.timesteps = torch.from_numpy(np.linspace(0, 999, steps+1).round()[::-1][:-1].copy().astype(np.int64))
        self.sigmas = torch.linspace(1., .03, steps+1)
        self.model_outputs = [None, None]
        self.step_index = steps-int(steps*.75)

    def _sigma_to_alpha_sigma_t(self, rho):
        alpha = 1 / (rho**2 + 1)**.5
        return alpha, rho * alpha

    def add_noise(self, source, noise, t):
        rho = t.float().reshape(-1,1,1,1)/1000 + .03
        alpha, sigma = self._sigma_to_alpha_sigma_t(rho)
        return alpha*source + sigma*noise

    def step(self, epsilon, sample):
        self.model_outputs = [self.model_outputs[-1], epsilon.clone()]
        self.step_index += 1
        return sample - .04*epsilon


class UNet(torch.nn.Module):
    def forward(self, x, t, *, encoder_hidden_states, cross_attention_kwargs,
                timestep_cond=None, added_cond_kwargs=None, return_dict=False):
        negative, positive = x.chunk(2)
        raw = torch.cat((negative*.1 + int(t)*.00002,
                         positive*.13 + int(t)*.00002 + .005))
        return (raw,)


class Pipeline:
    def __init__(self, unet, fail_at=None):
        self.unet, self.scheduler, self.fail_at = unet, Scheduler(), fail_at

    def get_timesteps(self, steps, strength, device):
        count=min(int(steps*strength),steps)
        return self.scheduler.timesteps[(steps-count)*self.scheduler.order:], count

    def __call__(self, *, prompt, image, strength, num_inference_steps, guidance_scale,
                 negative_prompt, generator, cross_attention_kwargs, callback_on_step_end,
                 callback_on_step_end_tensor_inputs):
        assert strength == .75 and negative_prompt == ''
        self.scheduler.configure(num_inference_steps)
        self.guidance_scale=guidance_scale; self.do_classifier_free_guidance=True
        self.cross_attention_kwargs=cross_attention_kwargs
        times,_=self.get_timesteps(num_inference_steps,strength,'cpu')
        noise=torch.randn(image.shape,generator=generator,dtype=torch.float32)
        latent=self.scheduler.add_noise(image,noise,times[:1])
        text=torch.zeros((2,77,768))
        for i,t in enumerate(times):
            raw=self.unet(torch.cat((latent,latent)),t,encoder_hidden_states=text,
                cross_attention_kwargs=cross_attention_kwargs,timestep_cond=None,
                added_cond_kwargs=None,return_dict=False)[0]
            if i == self.fail_at: raise RuntimeError('synthetic fail')
            neg,pos=raw.chunk(2)
            epsilon=neg+guidance_scale*(pos-neg)
            latent=self.scheduler.step(epsilon,latent)
            latent=callback_on_step_end(self,i,t,{'latents':latent,'prompt_embeds':text})['latents']
        self.final_latent=latent.clone()
        rgb=((latent[0,:3].permute(1,2,0).numpy().clip(-1,1)+1)*127.5).astype(np.uint8)
        return SimpleNamespace(images=[Image.fromarray(rgb).resize((512,512))])


def runtime(fail_at=None):
    unet=UNet()
    return SimpleNamespace(torch=torch,device='cpu',backbone=SimpleNamespace(unet=unet),
                           pipe=Pipeline(unet,fail_at),v11=v11)


def prepared(zero=False):
    source=torch.linspace(-.5,.5,4*64*64).reshape(1,4,64,64)
    edit=torch.zeros((1,1,64,64)) if zero else torch.linspace(0,1,64*64).reshape(1,1,64,64)
    cleanup=torch.zeros_like(edit); cleanup[...,25:36,26:37]=.6
    if zero: cleanup.zero_()
    cross={'au_embedding':torch.zeros((2,1,768)), 'image_embedding':torch.zeros((2,16,768)),
           'au_scale':1.5,'ip_scale':.9}
    return {'source':source,'edit':edit,'cleanup':cleanup,'cross':cross,'prompt':'fixed AU text',
            'condition_hashes':{'a':'fixture'},'mask_receipt':{'case_id':'synthetic'},
            'reconstruction':{'path':'synthetic_not_real_image'}}


def legacy_sample(rt,p,seed):
    noise=torch.randn(p['source'].shape,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
    ctrl=oldgrid['fixed_controller'](v11,rt.backbone.unet,p['source'],p['edit'],p['cross'],torch,p['cleanup'])
    callback,history=oldgrid['sampling_callback'](ctrl,p['source'],noise,p['edit'],torch)
    try:
        with torch.no_grad():
            image=rt.pipe(prompt=p['prompt'],image=p['source'],strength=.75,num_inference_steps=20,
                guidance_scale=7.5,negative_prompt='',generator=torch.Generator(device='cpu').manual_seed(seed),
                cross_attention_kwargs=p['cross'],callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=['latents','prompt_embeds']).images[0]
        ctrl.assert_complete()
        return image,history,ctrl.record()
    finally: ctrl.close()


class SamplerTests(unittest.TestCase):
    def test_parameters(self):
        self.assertEqual(q.parameters(5,40),(5.,40))
        for g,n in [(float('nan'),20),(1,20),(True,20),(5,21),(5,20.)]:
            with self.assertRaises(ValueError): q.parameters(g,n)

    def test_timestep_counts(self):
        rt=runtime()
        for steps,count,last in [(20,15,50),(40,30,25)]:
            rt.pipe.scheduler.configure(steps)
            _,values=q.actual_timesteps(rt.pipe,steps,'cpu')
            self.assertEqual((len(values),values[0],values[-1]),(count,749,last))
        rt.pipe.scheduler.configure(20)
        self.assertEqual(q.actual_timesteps(rt.pipe,20,'cpu')[1],TIMES20)

    def test_cfg_parameter_not_hardcoded(self):
        r=torch.ones((1,4,2,2)); raw=torch.cat((r,r*3))
        self.assertTrue(torch.equal(q.cfg_epsilon(raw,r,torch,3.5),r*8))
        self.assertTrue(torch.equal(q.cfg_epsilon(raw,r,torch,5),r*11))

    def test_exact_legacy_cpu_oracle_parity(self):
        a,p=runtime(),prepared(); old_image,old_history,old_receipt=legacy_sample(a,p,101)
        b=runtime(); image,receipt=q.sample(b,p,101,guidance=7.5,steps=20)
        self.assertEqual(old_image.tobytes(),image.tobytes())
        self.assertEqual(old_history,receipt['projection_trace'])
        self.assertTrue(torch.equal(a.pipe.final_latent,b.pipe.final_latent))
        for key in ('epsilon_cfg_sha256','base_terminal_clean_latent_sha256',
                    'post_filter_clean_latent_sha256','projected_latent_sha256'):
            self.assertEqual(old_receipt['terminal_trace'][key],receipt['terminal_trace'][key])
        self.assertEqual(len(b.backbone.unet._forward_hooks),0)

    def test_40_steps_actual_terminal(self):
        image,r=q.sample(runtime(),prepared(),101,guidance=5,steps=40)
        self.assertEqual(r['actual_steps'],30)
        self.assertEqual(r['terminal_trace']['last_deis_timestep'],25)
        self.assertEqual(r['total_unet_calls'],30)
        self.assertEqual(r['terminal_unet_calls'],0)
        self.assertEqual(r['terminal_trace']['guidance_scale'],5)
        json.dumps(r,allow_nan=False)

    def test_zeroedit_exact_source_for_all_probes(self):
        for g,n in [(7.5,20),(5,20),(3.5,20),(5,40)]:
            rt,p=runtime(),prepared(zero=True)
            _,r=q.sample(rt,p,101,guidance=g,steps=n)
            self.assertTrue(torch.equal(rt.pipe.final_latent,p['source']))

    def test_full_step_tensor_archive_and_collision_refusal(self):
        base=q.ROOT/'cpu_test_artifacts'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            rt,p=runtime(),prepared(); path=Path(folder)/'new_trace'; p['quality_trace_dir']=path
            _,r=q.sample(rt,p,101,guidance=5,steps=20)
            self.assertEqual(len(list(path.glob('step_*.pt'))),15)
            for item in r['projection_trace']:
                value=torch.load(item['latent_file'],weights_only=True,map_location='cpu')
                self.assertEqual(q.tensor_sha(value),item['latent_sha256'])
            with self.assertRaises(FileExistsError): q.sample(runtime(),p,101,guidance=5,steps=20)

    def test_fail_keeps_partial_receipt_and_removes_hook(self):
        rt=runtime(fail_at=2)
        with self.assertRaises(RuntimeError) as caught: q.sample(rt,prepared(),101,guidance=5,steps=20)
        r=caught.exception.generation_receipt
        self.assertEqual(r['status'],'failed'); self.assertEqual(len(r['projection_trace']),2)
        self.assertEqual(r['total_unet_calls'],3)
        self.assertEqual(len(rt.backbone.unet._forward_hooks),0)

    def test_immutable_source_conditions(self):
        p=prepared(); before={k:q.tensor_sha(p[k]) for k in ('source','edit','cleanup')}
        cond={k:q.tensor_sha(v) for k,v in p['cross'].items() if isinstance(v,torch.Tensor)}
        q.sample(runtime(),p,101,guidance=5,steps=20)
        self.assertEqual(before,{k:q.tensor_sha(p[k]) for k in before})
        self.assertEqual(cond,{k:q.tensor_sha(p['cross'][k]) for k in cond})

    def test_invalid_cross_rejected(self):
        p=prepared(); p['cross']['au_scale']=1
        rt=runtime()
        with self.assertRaises(ValueError): q.sample(rt,p,101,guidance=5,steps=20)
        self.assertEqual(len(rt.backbone.unet._forward_hooks),0)


if __name__=='__main__': unittest.main(verbosity=2)
