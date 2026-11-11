#!/usr/bin/env python3
"""Independent CPU contract checks; fake 2x2 pipeline images are NOT model results."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock
from align_real_frames import HERE,EXPERIMENT,environment,sha,write_json,copy_exact


def main():
    environment();sys.path.insert(0,str(EXPERIMENT))
    import torch
    from PIL import Image
    import diffusers
    import train_adapter as train
    import modeling.frozen_backbone as frozen
    from modeling.safe_token_residual_adapter import TokenAdapterConfig,SafeTokenCrossModalResidualAdapter,RelativeRMSConfig,augment_condition_embeddings
    torch.set_num_threads(2)
    out=HERE/'adapter_contract_review';out.mkdir(exist_ok=False)
    generator_source=EXPERIMENT/'generate_adapter_development.py'
    copy_exact(generator_source,out/'generation_source_snapshot.py',sha(generator_source))
    checkpoint=EXPERIMENT/'runs/v3_global_raw_smoke_202609141727/checkpoint_00000008.pt'
    cfg=train.TrainConfig(**json.loads((EXPERIMENT/'configs/v3_global_raw_dev64.json').read_text()))
    state=train.load_checkpoint(checkpoint)
    training,dev,groups,data_bindings=train.load_data(cfg)
    assert len(training)==343 and len(dev)==12 and len({x['subject'] for x in dev})==4
    train.validate_resume(state,cfg,state['bindings'])
    rejects={}
    for key,value in [('readout','slot'),('patch_scale','unit_mean'),('rho_ip',.1),('variant','condition_reads_image')]:
        try:train.validate_resume(state,dataclasses.replace(cfg,**{key:value}),state['bindings'])
        except ValueError:rejects[key]=True
        else:raise AssertionError('Resume silently accepted changed '+key)
    qualified,info=train.validate_au_qualification(cfg,groups,data_bindings['split'])
    assert info['enabled'] and qualified.is_file()
    mc=TokenAdapterConfig(**state['model_config'])
    assert mc.direction==state['variant']==cfg.variant and mc.readout==cfg.readout and mc.patch_scale==cfg.patch_scale
    adapter=SafeTokenCrossModalResidualAdapter(mc).eval();adapter.load_state_dict(state['adapter'],strict=True)
    e=state['effective_config'];trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
    assert trust.enabled and trust.rho_au==.1 and trust.rho_ip==.025
    calls=[]
    class FakeBackbone:
        def __init__(self,*args,**kwargs):self.unet=object()
        def baseline_conditions(self,aus,identity):
            au=torch.ones(len(aus),1,768);au[..., :12]=aus[:,None]
            ip=torch.nn.functional.pad(identity,(0,256))[:,None].expand(-1,16,-1).clone()
            return au,ip
    class FakeEncoders:
        def __init__(self,*args,**kwargs):self.vae=self.text_encoder=self.tokenizer=object()
    class FakePipe:
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return cls()
        def to(self,*args,**kwargs):return self
        def set_progress_bar_config(self,*args,**kwargs):pass
        def __call__(self,prompt,**kwargs):
            cross=kwargs['cross_attention_kwargs']
            assert kwargs['guidance_scale']>1
            assert cross['au_embedding'].shape==(2,1,768) and cross['image_embedding'].shape==(2,16,768)
            generator=kwargs['generator']
            calls.append({'prompt':prompt,'seed':generator.initial_seed(),'noise':torch.randn(16,generator=generator),
                          'au':cross['au_embedding'].clone(),'ip':cross['image_embedding'].clone(),
                          'au_scale':cross['au_scale'],'ip_scale':cross['ip_scale']})
            return type('FakeResult',(),{'images':[Image.new('RGB',(2,2),(0,0,0))]})()
    original_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kwargs):
        if args and str(args[0])=='mps':args=('cpu',)+args[1:]
        if str(kwargs.get('device'))=='mps':kwargs['device']='cpu'
        return original_to(tensor,*args,**kwargs)
    spec=importlib.util.spec_from_file_location('review_real_generation_entry',generator_source)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    fake_output=out/'synthetic_pipeline_only_NOT_MODEL_OUTPUT'
    argv=['generate_adapter_development.py','--cache',cfg.cache_dir,'--cases',cfg.dev_case_manifest,'--checkpoint',str(checkpoint),
          '--output',str(fake_output),'--steps','20','--guidance-scale','7.5']
    with mock.patch.object(sys,'argv',argv),mock.patch.object(torch.Tensor,'to',cpu_to),mock.patch.object(frozen,'FrozenConditionedDenoiser',FakeBackbone),mock.patch.object(frozen,'FrozenImageTextEncoders',FakeEncoders),mock.patch.object(diffusers,'StableDiffusionPipeline',FakePipe):
        runner.main()
    cases=json.loads(Path(cfg.dev_case_manifest).read_text())
    rows={r['case_id']:r for r in dev};seeds=cases['sampling_seeds'];expected_count=12*2*len(seeds)
    assert len(calls)==expected_count
    records=[json.loads(x) for x in (fake_output/'generation_records.jsonl').read_text().splitlines()]
    assert len(records)==expected_count and all(r['status']=='complete' and r['split']=='dev' for r in records)
    selected_groups={s:[] for s in groups['dev']}
    for case_index,case in enumerate(cases['cases']):
        s=rows[case['case_id']];selected_groups[case['subject']].append(case['group'])
        patch,text,aus,identity=[s[k][None] for k in ('image_patches','text_pooled','target_aus','base_faceid')]
        base_au,base_ip=FakeBackbone().baseline_conditions(aus,identity)
        with torch.no_grad():expected=augment_condition_embeddings(base_au,base_ip,adapter(patch,text,aus),trust_region=trust)
        au_features=adapter.au_encoder(aus)
        assert torch.equal(au_features[...,:12],aus/5)
        for seed_index,seed in enumerate(seeds):
            baseline=calls[(case_index*2)*len(seeds)+seed_index];adapted=calls[(case_index*2+1)*len(seeds)+seed_index]
            assert baseline['seed']==adapted['seed']==seed and torch.equal(baseline['noise'],adapted['noise'])
            assert baseline['prompt']==adapted['prompt']==s['prompt']
            assert baseline['au_scale']==adapted['au_scale']==1 and baseline['ip_scale']==adapted['ip_scale']==.9
            assert torch.equal(adapted['au'][1:],expected['au_embedding']) and torch.equal(adapted['ip'][1:],expected['image_embedding'])
            assert torch.equal(baseline['au'][0],adapted['au'][0]) and torch.equal(baseline['ip'][0],adapted['ip'][0])
            assert torch.equal(adapted['au'][1:,:,:12],base_au[:,:,:12])
    assert all(sorted(v)==['lowest_available','no_edit','second_edited'] for v in selected_groups.values())
    # Independently demonstrate why a 2B condition with CFG off is wrong:
    # the author's processor reshapes it into extra tokens, without an error.
    spec=importlib.util.spec_from_file_location('review_upstream_au_attention',Path.home()/'Projects/github/tvaranka/fineface/fineface/au_attention.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    from diffusers.models.attention_processor import Attention
    attention=Attention(query_dim=8,cross_attention_dim=12,heads=2,dim_head=4)
    processor=upstream.AUIPAttnProcessor2_0(8,12)
    h=torch.randn(1,3,8);text=torch.randn(1,5,12);au=torch.randn(1,1,12);ip=torch.randn(1,16,12)
    with torch.no_grad():
        correct=processor(attention,h,text,au_embedding=au,image_embedding=ip)
        mixed=processor(attention,h,text,au_embedding=torch.cat([torch.zeros_like(au),au]),image_embedding=torch.cat([torch.zeros_like(ip),ip]))
    assert mixed.shape==correct.shape and not torch.allclose(mixed,correct)
    report={'status':'passed','scope':'CPU data/restore/condition/seed contract tests; synthetic pipeline is NOT a generation result',
            'review_script_sha256':sha(__file__),'training_source_sha256':sha(EXPERIMENT/'train_adapter.py'),
            'generation_source_sha256':sha(out/'generation_source_snapshot.py'),'checkpoint_sha256':sha(checkpoint),
            'train_count':343,'dev_count':12,'dev_subject_count':4,'test_images_used':False,'mps_used':False,
            'resume_rejects_configuration_changes':rejects,'qualified_au_entry_accepted':True,
            'restored_variant':mc.direction,'restored_readout':mc.readout,'restored_patch_scale':mc.patch_scale,
            'restored_trust_region':dataclasses.asdict(trust),'method_case_seed_records_checked':len(records),
            'all_methods_share_cases_prompts_noise_seeds_and_null_conditions':True,'protected_raw_au_prefix_unchanged':True,
            'adapter_internal_au_scale_exactly_divide_5':True,'current_guidance_7_5_contract_valid':True,
            'cfg_off_double_batch_upstream_silent_mix_reproduced':True,'cfg_off_max_output_error_synthetic':float((mixed-correct).abs().max())}
    write_json(out/'report.json',report);print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
