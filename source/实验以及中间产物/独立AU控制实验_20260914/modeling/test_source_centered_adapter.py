"""Synthetic CPU/MPS structural checks only; no training or generation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.pycache_prefix=str(HERE/'test_pycache')
import torch
from source_centered_adapter import forward_source_centered, SourceCenteredResidualOutput
from safe_token_residual_adapter import TokenAdapterConfig, SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter, RelativeRMSConfig, condition_residuals, augment_condition_embeddings, relative_rms_trust_region

SEED=20260914
OBSERVATIONS={}


def configuration(**changes):
    return replace(TokenAdapterConfig(),image_dim=24,text_dim=32,hidden_dim=16,num_heads=4,au_hidden_dim=16,au_feature_dim=24,condition_dim=32,**changes)


def make_model(variant,config,nonzero=True):
    model=SafeTokenPooledMLPResidualAdapter(config,128) if variant=='mlp' else SafeTokenCrossModalResidualAdapter(replace(config,direction=variant,ff_multiplier=3 if variant!='bidirectional' else 2))
    if nonzero:
        torch.nn.init.normal_(model.au_output.weight,std=.04)
        torch.nn.init.normal_(model.image_output.weight,std=.04)
    return model


def inputs(config,device='cpu'):
    return (torch.randn(2,49,config.image_dim,device=device),torch.randn(2,config.text_dim,device=device),torch.rand(2,12,device=device)*3)


class SourceCenteredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):torch.manual_seed(SEED)

    def test_frozen_source_hashes(self):
        for name,expected in [('cross_modal_adapter.py','dab32f9c0969682281f7d51559c3ccefea05dbc93edd54fa8561ed5c802d40f4'),('token_residual_adapter.py','42b0dd3e4bb1287e4484d9700cfcf3798fe4064cb9f8691468944e79409bf146'),('safe_token_residual_adapter.py','546c3b2e856f096959b838976afd00ad613cd11629d7b418d2e2419de5642f24')]:
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),expected)
        self.assertEqual(hashlib.sha256((HERE.parent/'train_adapter.py').read_bytes()).hexdigest(),'79bfa85282d94294bdf6a620b3a87e091929592e6bf17a8128e9cabeafc9a3b3')

    def test_all_variants_factors_noedit_returns_baseline_without_new_parameters(self):
        counts={}
        for variant in ('bidirectional','condition_reads_image','image_reads_condition','mlp'):
            for readout in ('global','slot'):
                for scale in ('raw','unit_mean'):
                    config=configuration(readout=readout,patch_scale=scale)
                    model=make_model(variant,config)
                    parameters=[id(p) for p in model.parameters()]
                    state={k:v.clone() for k,v in model.state_dict().items()}
                    image,text,au=inputs(config)
                    output=forward_source_centered(model,image,text,au,text.clone(),au.clone(),return_attention=True)
                    self.assertIsInstance(output,SourceCenteredResidualOutput)
                    self.assertEqual(output.image_delta.shape,(2,1 if readout=='global' else 16,32))
                    self.assertEqual(output.au_delta.shape,(2,1,32))
                    self.assertEqual(torch.count_nonzero(output.au_delta).item(),0)
                    self.assertEqual(torch.count_nonzero(output.image_delta).item(),0)
                    base_au,base_ip=torch.randn(2,1,32),torch.randn(2,16,32)
                    for tr in (RelativeRMSConfig(),RelativeRMSConfig(enabled=True,rho_au=.1,rho_ip=.025)):
                        result=augment_condition_embeddings(base_au,base_ip,output,trust_region=tr)
                        self.assertTrue(torch.equal(result['au_embedding'],base_au))
                        self.assertTrue(torch.equal(result['image_embedding'],base_ip))
                    self.assertEqual(parameters,[id(p) for p in model.parameters()])
                    self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
                    counts[variant+'_'+readout+'_'+scale]=sum(p.numel() for p in model.parameters())
        OBSERVATIONS['synthetic_parameter_counts']=counts

    def test_zero_initialization_preserves_baseline_even_for_edits(self):
        config=configuration();model=make_model('bidirectional',config,nonzero=False)
        image,text,au=inputs(config)
        output=forward_source_centered(model,image,text,au,text+1,torch.zeros_like(au))
        self.assertEqual(torch.count_nonzero(output.au_delta).item(),0)
        self.assertEqual(torch.count_nonzero(output.image_delta).item(),0)
        loss=(output.au_delta*torch.randn_like(output.au_delta)).sum()+(output.image_delta*torch.randn_like(output.image_delta)).sum()
        loss.backward()
        self.assertGreater(model.au_output.weight.grad.norm().item(),0)
        self.assertGreater(model.image_output.weight.grad.norm().item(),0)

    def test_equal_au_different_text_is_not_gated(self):
        config=configuration();model=make_model('bidirectional',config)
        image,text,au=inputs(config)
        output=forward_source_centered(model,image,text+.5,au,text,au)
        self.assertGreater(output.au_delta.abs().max().item(),0)
        self.assertGreater(output.image_delta.abs().max().item(),0)

    def test_known_nonzero_source_au_and_downward_edit_synthetic(self):
        config=configuration();model=make_model('bidirectional',config)
        image,text,_=inputs(config)
        source=torch.full((2,12),2.)
        target=torch.zeros_like(source)
        output=forward_source_centered(model,image,text,target,text,source)
        self.assertGreater(output.au_delta.abs().max().item(),0)
        same=forward_source_centered(model,image,text,source,text,source)
        self.assertEqual(torch.count_nonzero(same.au_delta).item(),0)

    def test_dropout_must_be_zero_in_train_and_eval(self):
        model=make_model('bidirectional',configuration(dropout=.1))
        image,text,au=inputs(model.config)
        for mode in (True,False):
            model.train(mode)
            with self.assertRaisesRegex(ValueError,'dropout=0'):
                forward_source_centered(model,image,text,au,text,au)

    def test_raw_difference_then_exactly_one_existing_tr_and_prefix(self):
        config=configuration(readout='slot');model=make_model('bidirectional',config)
        image,text,au=inputs(config)
        output=forward_source_centered(model,image,text,au,text*.5,torch.zeros_like(au))
        self.assertTrue(torch.equal(output.au_delta,output.target_output.au_delta-output.source_output.au_delta))
        self.assertTrue(torch.equal(output.image_delta,output.target_output.image_delta-output.source_output.image_delta))
        self.assertTrue(torch.equal(output.image_delta,output.candidate_slot_deltas))
        self.assertEqual(torch.count_nonzero(output.au_delta[...,:12]).item(),0)
        base_au,base_ip=torch.randn(2,1,32),torch.randn(2,16,32)
        base_au[...,:12]=au[:,None]
        trust=RelativeRMSConfig(enabled=True,rho_au=.1,rho_ip=.025)
        effective=condition_residuals(base_au,base_ip,output,trust_region=trust)
        expected_au=relative_rms_trust_region(output.target_output.au_delta[...,12:]-output.source_output.au_delta[...,12:],base_au[...,12:],.1)
        expected_ip=relative_rms_trust_region(output.target_output.image_delta-output.source_output.image_delta,base_ip,.025)
        self.assertTrue(torch.equal(expected_au,effective['au_delta'][...,12:]))
        self.assertTrue(torch.equal(expected_ip,effective['image_delta']))
        result=augment_condition_embeddings(base_au,base_ip,output,trust_region=trust)
        self.assertTrue(torch.equal(result['au_embedding'][...,:12],base_au[...,:12]))
        self.assertTrue(torch.all(effective['image_delta'].square().mean(-1).sqrt() <= .025*base_ip.square().mean((1,2)).sqrt()[:,None]*(1+1e-6)))

    def noedit_gradient_probe(self,device):
        config=configuration(readout='slot',patch_scale='unit_mean')
        model=make_model('bidirectional',config).to(device)
        image,text,au=inputs(config,device)
        output=forward_source_centered(model,image,text,au,text,au)
        # Nonzero upstream loss derivative: zero parameter gradients must arise
        # from actual shared-branch cancellation, not a zero squared loss.
        loss=(output.au_delta*torch.randn_like(output.au_delta)).sum()+(output.image_delta*torch.randn_like(output.image_delta)).sum()
        loss.backward()
        nonnull=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(nonnull)
        self.assertTrue(all(torch.count_nonzero(g).item()==0 for g in nonnull))
        OBSERVATIONS['noedit_'+device]={'nonnull_parameter_grads':len(nonnull),'nonzero_parameter_grads':0}

    def edited_gradient_probe(self,device):
        config=configuration(readout='global',patch_scale='unit_mean')
        model=make_model('bidirectional',config).to(device)
        image,source_text,source_au=inputs(config,device)
        target_text=torch.randn_like(source_text);target_au=torch.zeros_like(source_au)
        for value in (image,source_text,source_au,target_text,target_au):value.requires_grad_(True)
        output=forward_source_centered(model,image,target_text,target_au,source_text,source_au)
        for branch in (output.target_output,output.source_output):
            branch.au_delta.retain_grad();branch.image_delta.retain_grad()
        loss=(output.au_delta*torch.randn_like(output.au_delta)).sum()+(output.image_delta*torch.randn_like(output.image_delta)).sum()
        loss.backward()
        self.assertTrue(torch.equal(output.target_output.au_delta.grad,-output.source_output.au_delta.grad))
        self.assertTrue(torch.equal(output.target_output.image_delta.grad,-output.source_output.image_delta.grad))
        gradients={name:float(value.grad.norm().detach().cpu()) for name,value in [('patches',image),('source_text',source_text),('source_au',source_au),('target_text',target_text),('target_au',target_au)]}
        self.assertTrue(all(value>0 and torch.isfinite(torch.tensor(value)) for value in gradients.values()))
        nonzero=sum(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in model.parameters())
        self.assertGreater(nonzero,2)
        OBSERVATIONS['edited_'+device]={'input_gradient_norms':gradients,'nonzero_parameter_tensors':nonzero,'both_branch_grads_exact_opposites':True}

    def test_cpu_noedit_parameter_gradient_cancels(self):self.noedit_gradient_probe('cpu')
    def test_cpu_edited_both_branches_and_inputs_receive_gradients(self):self.edited_gradient_probe('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_noedit_parameter_gradient_cancels(self):self.noedit_gradient_probe('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_edited_both_branches_and_inputs_receive_gradients(self):self.edited_gradient_probe('mps')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceCenteredTests))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x') as handle:
            json.dump({'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'skipped':result.skipped,'passed':result.wasSuccessful(),'observations':OBSERVATIONS,'new_parameters':0,'training_run':False,'generated_images':0,'real_nonzero_source_validation':False,'helper_sha256':hashlib.sha256((HERE/'source_centered_adapter.py').read_bytes()).hexdigest()},handle,indent=2)
    raise SystemExit(not result.wasSuccessful())
