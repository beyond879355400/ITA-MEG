"""CPU/MPS state and gate tests; never loads a generation model."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import train_adapter as trainer


class TrainerStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='trainer_state_', dir=trainer.ROOT)
        self.path = Path(self.temporary.name)
        self.config = trainer.TrainConfig(cache_dir=str(self.path), split_path=str(self.path/'split.json'), pairs_manifest=str(self.path/'pairs.json'), device='cpu', max_steps=8)

    def tearDown(self):
        self.temporary.cleanup()

    def test_invalid_configuration(self):
        for changes in ({'model_version':'v1'}, {'variant':'other'}, {'rho_au':0}, {'max_steps':129}, {'dev_timesteps':[1000]}, {'au_weight':1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def _rng_roundtrip(self, device):
        random.seed(8); np.random.seed(8); torch.manual_seed(8)
        noise = torch.Generator().manual_seed(23)
        order = torch.Generator().manual_seed(24)
        def sample():
            return (random.random(), np.random.randn(3), torch.randn(3), torch.randn(3, generator=noise), torch.randperm(7,generator=order), torch.randn(3,device=device).cpu())
        state = trainer.capture_rng(noise, order, device)
        expected = sample()
        sample()
        trainer.restore_rng(state, noise, order, device)
        actual = sample()
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        for left, right in zip(expected[2:], actual[2:]):
            self.assertTrue(torch.equal(left, right))

    def test_cpu_rng_roundtrip(self):
        self._rng_roundtrip('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_rng_roundtrip(self):
        self._rng_roundtrip('mps')

    def test_order_crosses_epoch_without_dropping(self):
        rng = torch.Generator().manual_seed(19)
        state = {'epoch':0,'order':[],'cursor':0}
        first = trainer.next_indices(state, 7, 3, rng)
        saved, saved_rng = copy.deepcopy(state), rng.get_state()
        remainder = [trainer.next_indices(state,7,3,rng) for _ in range(3)]
        rng.set_state(saved_rng)
        resumed = [trainer.next_indices(saved,7,3,rng) for _ in range(3)]
        self.assertEqual(remainder, resumed)
        self.assertEqual(sorted(first + remainder[0] + remainder[1]), list(range(7)))
        self.assertEqual(state['epoch'], 1)

    def _synthetic_checkpoint(self):
        torch.manual_seed(34)
        model = torch.nn.Linear(3,2)
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)
        noise = torch.Generator().manual_seed(3)
        order = torch.Generator().manual_seed(4)
        state = {'epoch':0,'order':[],'cursor':0}
        def step(m, opt, n, o, s):
            ids = trainer.next_indices(s,5,2,o)
            x = torch.randn(len(ids),3,generator=n)
            target = torch.randn(len(ids),2,generator=n)
            opt.zero_grad(set_to_none=True)
            loss=(m(x)-target).square().mean()
            loss.backward();opt.step()
            return loss.detach().clone(),ids
        step(model,optimizer,noise,order,state)
        bindings={'split':'frozen','sources':{'model':'exact'},'train_case_ids':list('abcde')}
        path=self.path/'checkpoint.pt'
        trainer.save_checkpoint(path,model,optimizer,self.config,bindings,1,state,noise,order,5,{'test':'linear'})
        return path,model,optimizer,noise,order,state,bindings,step

    def test_adam_and_rng_exact_resume(self):
        path,model,opt,noise,order,state,bindings,step=self._synthetic_checkpoint()
        saved=trainer.load_checkpoint(path)
        trainer.validate_resume(saved,self.config,bindings)
        expected_loss, expected_ids=step(model,opt,noise,order,state)
        expected=trainer.cpu_tree(model.state_dict())
        restored=torch.nn.Linear(3,2)
        restored.load_state_dict(saved['adapter'])
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=.1)
        restored_opt.load_state_dict(saved['optimizer'])
        trainer.restore_rng(saved['rng'],noise,order,'cpu')
        actual_loss,actual_ids=step(restored,restored_opt,noise,order,saved['data_state'])
        self.assertTrue(torch.equal(expected_loss,actual_loss))
        self.assertEqual(expected_ids,actual_ids)
        for key,value in restored.state_dict().items():
            self.assertTrue(torch.equal(expected[key],value))

    def test_resume_rejects_drift_but_allows_length_extension(self):
        path,_,_,_,_,_,bindings,_=self._synthetic_checkpoint()
        payload=trainer.load_checkpoint(path)
        trainer.validate_resume(payload,replace(self.config,max_steps=16),bindings)
        for changed in (replace(self.config,variant='mlp'),replace(self.config,readout='slot'),replace(self.config,learning_rate=.001)):
            with self.assertRaises(ValueError):trainer.validate_resume(payload,changed,bindings)
        with self.assertRaises(ValueError):trainer.validate_resume(payload,self.config,{**bindings,'split':'another'})
        with self.assertRaises(ValueError):trainer.validate_resume(payload,self.config,{**bindings,'sources':{'model':'changed'}})

    def test_never_overwrites_and_seal_detects_corruption(self):
        path,model,opt,noise,order,state,bindings,_=self._synthetic_checkpoint()
        before=trainer.sha256(path)
        with self.assertRaises(FileExistsError):
            trainer.save_checkpoint(path,model,opt,self.config,bindings,2,state,noise,order,5,{})
        self.assertEqual(before,trainer.sha256(path))
        with path.open('ab') as handle:handle.write(b'corruption')
        with self.assertRaises(ValueError):trainer.load_checkpoint(path)

    def test_noau_ablation_eligible_only_outside_diagnostic(self):
        self.assertNotIn('max_steps',trainer.effective_config(self.config))
        config=replace(self.config,diagnostic=False)
        m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters())
        payload=trainer.save_checkpoint(self.path/'noau.pt',m,o,config,{},1,{'epoch':0,'order':[0],'cursor':1},torch.Generator(),torch.Generator(),1,{})
        self.assertTrue(payload['eligible_for_final_selection'])
        self.assertFalse(payload['au_supervision_enabled'])

    def test_real_failed_qualification_is_rejected(self):
        split_path=trainer.ROOT/'data_prepare/au_supervision_v2/subjects_frozen.json'
        groups={k:set(v) for k,v in trainer.read_json(split_path)['groups'].items()}
        failed=trainer.ROOT/'au_supervisor/head_only_v1/qualification.json'
        config=replace(self.config,au_weight=1,au_qualification=str(failed))
        with self.assertRaisesRegex(ValueError,'has not qualified'):
            trainer.validate_au_qualification(config,groups,trainer.sha256(split_path))

    def test_development_subset_rejects_test_and_duplicates(self):
        rows=[{'case_id':'dev_A','subject':'A','source_aus':torch.zeros(12),'target_aus':torch.ones(12)}]
        document={'status':'frozen_for_development','test_data_used':False,'cache_sha256':'cache','subject_split_sha256':'split','selection':'preplanned labels','cases':[{'case_id':'dev_A','split':'dev','subject':'A'}]}
        path=self.path/'dev.json'
        path.write_text(json.dumps(document))
        selected,binding=trainer.select_development_cases(path,rows,'cache','split')
        self.assertEqual(len(selected),1)
        self.assertEqual(binding['manifest_sha256'],trainer.sha256(path))
        for cases in ([document['cases'][0]]*2,[{'case_id':'test_A','split':'test','subject':'A'}]):
            document['cases']=cases
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):trainer.select_development_cases(path,rows,'cache','split')

    def test_gradient_diagnostics_distinguishes_absent_none_zero(self):
        from modeling.safe_token_residual_adapter import TokenAdapterConfig, SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter
        single=SafeTokenCrossModalResidualAdapter(TokenAdapterConfig(direction='condition_reads_image',ff_multiplier=3))
        report=trainer.gradient_diagnostics(single)
        self.assertIsNone(report['directional_gradient_norms'][0]['image_from_condition'])
        self.assertIsNone(report['directional_gradient_norms'][0]['condition_from_image']['q'])
        module=single.blocks[0].condition_from_image
        module.q.weight.grad=torch.zeros_like(module.q.weight)
        module.v.weight.grad=torch.ones_like(module.v.weight)
        report=trainer.gradient_diagnostics(single)
        self.assertEqual(report['directional_gradient_norms'][0]['condition_from_image']['q'],0)
        self.assertGreater(report['directional_gradient_norms'][0]['condition_from_image']['v'],0)
        self.assertEqual(report['nonnull_gradient_parameter_tensors'],2)
        self.assertEqual(report['nonzero_gradient_parameter_tensors'],1)
        mlp=SafeTokenPooledMLPResidualAdapter(TokenAdapterConfig(),1032)
        self.assertIsNone(trainer.gradient_diagnostics(mlp)['directional_gradient_norms'])


if __name__=='__main__':
    unittest.main(verbosity=2)
