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
import train_centered_adapter as trainer


class TrainerStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='trainer_state_', dir=trainer.ROOT)
        self.path = Path(self.temporary.name)
        self.config = trainer.TrainConfig(cache_dir=str(self.path), split_path=str(self.path/'split.json'), pairs_manifest=str(self.path/'pairs.json'), device='cpu', max_steps=8,au_weight=0,au_qualification=None)

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


class CenteredIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):torch.manual_seed(20260914)

    def neutral_rows(self):
        from prepare_training_cache import make_prompt
        pooled=torch.randn(768)
        row={'case_id':'train_neutral_b','split':'train','subject':'train_B','source_aus':torch.zeros(12),'target_aus':torch.zeros(12),'prompt':make_prompt([0]*12),'text_pooled':pooled}
        first={**row,'case_id':'train_neutral_a','subject':'train_A','text_pooled':pooled.clone()}
        dev={**row,'case_id':'dev_neutral','split':'dev','subject':'dev_C','text_pooled':pooled.clone()}
        return [row,first],[dev]

    def test_anchor_is_train_only_stable_and_all_neutral_match(self):
        train,dev=self.neutral_rows()
        anchor,binding=trainer.resolve_source_anchor(train,dev)
        self.assertEqual(binding['anchor_train_case_id'],'train_neutral_a')
        self.assertEqual(binding['neutral_train_case_ids'],['train_neutral_a','train_neutral_b'])
        again,again_binding=trainer.resolve_source_anchor(list(reversed(train)),[])
        self.assertTrue(torch.equal(anchor,again));self.assertEqual(binding,again_binding)
        with self.assertRaisesRegex(ValueError,'No real neutral TRAIN'):
            trainer.resolve_source_anchor([],dev)
        changed=[{**dev[0],'text_pooled':dev[0]['text_pooled']+.01}]
        with self.assertRaisesRegex(ValueError,'not identical'):
            trainer.resolve_source_anchor(train,changed)

    def test_anchor_tamper_and_nonzero_source_rejected(self):
        train,dev=self.neutral_rows();anchor,binding=trainer.resolve_source_anchor(train,dev)
        trainer.verify_source_anchor(anchor,binding,binding)
        with self.assertRaisesRegex(ValueError,'changed'):
            trainer.verify_source_anchor(anchor+.1,binding,binding)
        with self.assertRaisesRegex(ValueError,'changed'):
            trainer.verify_source_anchor(anchor,{**binding,'anchor_train_case_id':'dev_neutral'},binding)
        dev[0]['source_aus']=torch.ones(12)
        with self.assertRaisesRegex(ValueError,'zero-AU sources'):
            trainer.resolve_source_anchor(train,dev)

    def test_all_variants_load_strictly_and_cancelled_offset_frozen(self):
        from modeling.safe_token_residual_adapter import TokenAdapterConfig
        from dataclasses import asdict
        counts={}
        for variant in ('bidirectional','condition_reads_image','image_reads_condition','mlp'):
            cfg=TokenAdapterConfig(direction=variant if variant!='mlp' else 'bidirectional',ff_multiplier=3 if variant in ('condition_reads_image','image_reads_condition') else 2,readout='global',patch_scale='unit_mean')
            model=trainer.create_centered_adapter(variant,cfg)
            self.assertFalse(model.slot_readout[-1].bias.requires_grad)
            self.assertEqual(model.parameter_counts()['total']-model.parameter_counts()['trainable'],128)
            state={'kind':trainer.CHECKPOINT_KIND,'model_version':trainer.MODEL_VERSION,'variant':variant,'model_config':asdict(cfg),'adapter':model.state_dict()}
            loaded=trainer.adapter_from_checkpoint(state)
            self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items()))
            self.assertFalse(loaded.slot_readout[-1].bias.requires_grad)
            counts[variant]=model.parameter_counts()['trainable']
            with self.assertRaises(ValueError):
                trainer.adapter_from_checkpoint({**state,'model_version':'v3_protected_au_relative_rms'})
            if variant!='mlp':
                with self.assertRaises(ValueError):
                    trainer.adapter_from_checkpoint({**state,'model_config':{**asdict(cfg),'direction':'mlp'}})
        self.assertEqual(counts,{'bidirectional':840500,'condition_reads_image':840244,'image_reads_condition':840244,'mlp':840380})

    def test_real_v3_checkpoint_rejected_before_loading(self):
        old=trainer.ROOT/'runs/v3_global_raw_dev64_202609141727/checkpoint_00000064.pt'
        with self.assertRaisesRegex(ValueError,'integrity seal'):
            trainer.load_checkpoint(old)

    def integration_probe(self,device):
        from modeling.safe_token_residual_adapter import TokenAdapterConfig,RelativeRMSConfig
        cfg=TokenAdapterConfig(image_dim=24,text_dim=32,hidden_dim=16,au_hidden_dim=16,au_feature_dim=24,condition_dim=32,readout='global',patch_scale='unit_mean')
        model=trainer.create_centered_adapter('bidirectional',cfg,device)
        torch.nn.init.normal_(model.au_output.weight,std=.03);torch.nn.init.normal_(model.image_output.weight,std=.03)
        anchor=torch.randn(32,device=device)
        b={'image_patches':torch.randn(2,49,24,device=device),'text_pooled':anchor[None].repeat(2,1),'source_aus':torch.zeros(2,12,device=device),'target_aus':torch.zeros(2,12,device=device)}
        base_au,base_ip=torch.randn(2,1,32,device=device),torch.randn(2,16,32,device=device)
        trust=RelativeRMSConfig(enabled=True,rho_au=.1,rho_ip=.025)
        out,effective,values=trainer.centered_batch_conditions(model,b,anchor,base_au,base_ip,trust)
        self.assertTrue(torch.equal(values['au_embedding'],base_au));self.assertTrue(torch.equal(values['image_embedding'],base_ip))
        loss=(values['au_embedding']*torch.randn_like(base_au)).sum()+(values['image_embedding']*torch.randn_like(base_ip)).sum();loss.backward()
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad).item()==0 for p in model.parameters()))
        model.zero_grad(set_to_none=True);b['target_aus'][:,0]=1;b['text_pooled']=b['text_pooled']+torch.randn_like(b['text_pooled'])*.1
        out,effective,values=trainer.centered_batch_conditions(model,b,anchor,base_au,base_ip,trust)
        out.source_output.au_delta.retain_grad();out.target_output.au_delta.retain_grad()
        loss=values['au_embedding'].square().mean()+values['image_embedding'].square().mean();loss.backward()
        self.assertGreater(float(out.source_output.au_delta.grad.norm().cpu()),0)
        self.assertTrue(torch.equal(out.source_output.au_delta.grad,-out.target_output.au_delta.grad))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in model.parameters()),2)
        self.assertIsNone(model.slot_readout[-1].bias.grad)
        self.assertTrue(torch.equal(values['au_embedding'][...,:12],base_au[...,:12]))
        b['source_aus'][:,0]=1
        with self.assertRaisesRegex(ValueError,'zero-AU'):
            trainer.centered_batch_conditions(model,b,anchor,base_au,base_ip,trust)

    def test_cpu_centered_batch_integration(self):self.integration_probe('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_centered_batch_integration(self):self.integration_probe('mps')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        trainer.json_new(args.report,{'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'skipped':result.skipped,'passed':result.wasSuccessful(),'trainer_sha256':trainer.sha256(trainer.ROOT/'train_centered_adapter.py'),'generator_sha256':trainer.sha256(trainer.ROOT/'generate_centered_development.py'),'training_started':False,'generation_models_loaded':False})
    raise SystemExit(not result.wasSuccessful())
