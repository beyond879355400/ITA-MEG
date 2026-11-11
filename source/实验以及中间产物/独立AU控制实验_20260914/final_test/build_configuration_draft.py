"""Assemble existing verified fragments into non-executable final-test DRAFTs."""
import argparse
import collections
import copy
import json
from pathlib import Path
import re
from evaluation_contract import HERE,EXP,PROJECT,POLICY_SHA,SOURCE_METRICS,sha,read,save,analysis_contract,frozen_header,EvaluationGateError

EXPECTED=['bidirectional','condition_reads_image','image_reads_condition','mlp','public_baseline','magicface','faceadapter','liveportrait']
CHECKPOINTS={'bidirectional':'7fcc25b704fca8c0a4e46a171096df8fea940b40a246d4939193e64f533b4e30',
 'condition_reads_image':'969def9470fb6a8ba4ff5b37e11b601144101e563e25ff3c26e09e4fdb71396b',
 'image_reads_condition':'99b565be708493e23a5a5d11829353cbc88cba3c2d78a388be49534703bd695f',
 'mlp':'f58336567246ecd2f6818ea4ae30f947914af4e9c2296177b86169b95d6aa6f5'}
DIAGNOSTIC=['delta_mae_changed','identity_cosine','unchanged_source_relative_drift','extra5_proxy_mae_to_real_target']

def merged_sources(*sources):
    result={}
    for mapping in sources:
        for key,ref in mapping.items():
            if key in result:assert result[key]==ref,'Conflicting execution source'
            path=Path(ref['path']);path=path if path.is_absolute() else PROJECT/path
            assert path.is_file() and sha(path)==ref['sha256']
            result[key]={'path':str(path.resolve()),'sha256':ref['sha256']}
    return result

def build(output):
    output=Path(output).resolve();assert output.is_relative_to(HERE) and not output.exists()
    assert not(HERE/'frozen_protocol.json').exists(),'Do not alter an existing frozen protocol'
    ours_path=HERE/'draft_fragments/ours.json';external_path=HERE/'draft_fragments/external.json'
    ours=read(ours_path);external=read(external_path);validation=read(HERE/'draft_fragments/validation.json')
    assert ours['status']==external['status']=='not_frozen' and validation['ours_sha256']==sha(ours_path) and validation['external_sha256']==sha(external_path)
    assert not ours['models_loaded'] and not external['models_loaded'] and not ours['test_pixels_read'] and not external['test_pixels_read']
    models=copy.deepcopy(ours['models']+external['models']);assert [m['method'] for m in models]==EXPECTED
    sampling=copy.deepcopy(ours['sampling']);sampling['method_settings'].update(copy.deepcopy(external['sampling']['method_settings']))
    sampling.update(status='not_frozen',kind='final_test_sampling_settings_DRAFT',draft_pending={'num_inference_steps':'root_to_choose_common20_or50_after_dev_check'},selection_context='all four learned structures use early200 checkpoints andIP0.9; common inference steps not yet selected')
    assert list(sampling['method_settings'])==EXPECTED and sampling['sampling_seeds']==[101,202,303]
    for m in EXPECTED[:5]:
        params=sampling['method_settings'][m]['native_parameters'];assert params['num_inference_steps'] is None and params['ip_scale']==.9 and params['guidance_scale']==7.5
    for m,digest in CHECKPOINTS.items():
        ref=sampling['method_settings'][m]['runtime']['adapter_checkpoint'];assert ref['sha256']==digest and sha(ref['path'])==digest
    assert sampling['method_settings']['magicface']['native_parameters']['steps']==50
    assert sampling['method_settings']['faceadapter']['native_parameters']['steps']==25
    lp=sampling['method_settings']['liveportrait'];assert lp['native_parameters']['flag_relative_motion'] is False and lp['native_parameters']['animation_region']=='all'
    assert lp['seed_execution']=={'mode':'deterministic_alias','actual_seed':101}
    implementation=read(HERE/'evaluation_implementation_sources.json')['execution_sources']
    sources=merged_sources(ours['execution_sources'],external['execution_sources'],implementation)
    metrics={f'{name}__{scope}':{'source_metric':name,'scope':scope,'direction':direction} for name,direction in SOURCE_METRICS.items() for scope in ['all','edited','no_edit']}
    assert len(metrics)==42
    external_refs=['magicface','faceadapter','liveportrait'];internal_refs=['public_baseline','condition_reads_image','image_reads_condition','mlp'];all_refs=external_refs+internal_refs
    comparisons=[]
    for family,refs,role in [('external_target_au',external_refs,'primary'),('internal_target_au',internal_refs,'secondary_structure')]:
        for ref in refs:comparisons.append({'id':f'bidirectional_vs_{ref}__target_mae_changed__edited','method':'bidirectional','reference':ref,'metric':'target_mae_changed__edited','family':family,'analysis_role':role})
    for metric in DIAGNOSTIC:
        for ref in all_refs:comparisons.append({'id':f'bidirectional_vs_{ref}__{metric}__edited','method':'bidirectional','reference':ref,'metric':metric+'__edited','family':'diagnostic_'+metric,'analysis_role':'descriptive'})
    families=dict(collections.Counter(c['family'] for c in comparisons))
    assert families['external_target_au']==3 and families['internal_target_au']==4 and len(comparisons)==35
    assert all(families['diagnostic_'+m]==7 for m in DIAGNOSTIC)
    evaluation={'kind':'final_test_evaluation_policy_DRAFT','status':'not_frozen','methods':EXPECTED,'measurement_policy':{'path':str(EXP/'development_evaluation/evaluation_policy_v2/policy.json'),'sha256':POLICY_SHA},
       'reference_image_space':'source_cache_canonical512_and_independent_target_canonical512','metrics':metrics,'comparisons':comparisons,
       'primary_family':'external_target_au','primary_families':['external_target_au'],'secondary_families':['internal_target_au'],
       'descriptive_families':['diagnostic_'+m for m in DIAGNOSTIC],'statistics':{'bootstrap_samples':50000,'bootstrap_seed':20260914,'confidence_level':.95},
       'reporting':{'primary_comparison_count':3,'secondary_structure_comparison_count':4,'descriptive_comparison_count':28,'raw_reports':'all14 existingmetrics acrossall/edited/no_edit',
           'descriptive_p_values':'computed transparently by existingtool but cannot support an additional primary claim, whether unadjusted orHolm-adjusted',
           'family_separation':'Diagnostic families do not alter external3/internal4 denominators','no_composite_score':True,'no_identity_threshold_selected':True,
           'input_groups':{'source_text_au':EXPECTED[:5],'source_manual_au':['magicface'],'source_real_driver':['faceadapter','liveportrait']}},
       'analysis_plan_source':{'path':str(HERE/'ANALYSIS_PLAN_CANDIDATE.md'),'sha256':sha(HERE/'ANALYSIS_PLAN_CANDIDATE.md')}}
    protocol=copy.deepcopy(read(HERE/'protocol.template.json'))
    protocol.update(status='not_frozen',extraction_authorized=False,models=models,execution_sources=sources,
       note='DRAFT ONLY. Candidate/early checkpoints and analysis families are chosen ondev; common our inference steps remain pending20/50. Root alone freezes after that check.',
       draft_pending={'our_inference_steps':'root choice after20/50 dev comparison','test_video_content_hashes':'still null; no test-video full read performed by draft assembly','authorization':'not frozen or authorized'})
    assert all(video['sha256'] is None for video in protocol['videos'])
    output.mkdir();save(output/'sampling.DRAFT.json',sampling);save(output/'evaluation.DRAFT.json',evaluation)
    protocol['inputs']['sampling_settings']={'path':str(output/'sampling.DRAFT.json'),'sha256':sha(output/'sampling.DRAFT.json')}
    protocol['inputs']['evaluation_policy']={'path':str(output/'evaluation.DRAFT.json'),'sha256':sha(output/'evaluation.DRAFT.json')}
    save(output/'protocol.DRAFT.json',protocol)
    # Pure schema checks only: never pass this in-memory copy to image/model stages.
    p=copy.deepcopy(protocol);s=copy.deepcopy(sampling);e=copy.deepcopy(evaluation);p['status']='frozen_final_test';p['extraction_authorized']=True;s['status']=e['status']='frozen_for_final_test'
    analysis_contract(p,s,e)
    blocked=False
    try:frozen_header(protocol)
    except EvaluationGateError:blocked=True
    assert blocked and not(HERE/'frozen_protocol.json').exists()
    artifacts={}
    for model in models:
        for ref in model['artifacts']:
            assert re.fullmatch('[0-9a-f]{64}',ref['sha256']);path=Path(ref['path']);assert path.is_file()
            if str(path) in artifacts:assert artifacts[str(path)]['sha256']==ref['sha256']
            artifacts[str(path)]={'sha256':ref['sha256'],'bytes':path.stat().st_size}
    receipt={'status':'not_frozen','schema_validation':'PASS_in_memory_only','saved_draft_execution_rejected':blocked,'model_methods':EXPECTED,'learned_checkpoint_budget':200,'our_ip_scale':.9,
       'our_inference_steps':None,'families':families,'metrics':len(metrics),'comparisons':len(comparisons),'unique_artifact_paths_checked':len(artifacts),
       'public_content_hash_provenance':{'fragment_validation_path':str(HERE/'draft_fragments/validation.json'),'sha256':sha(HERE/'draft_fragments/validation.json'),'producer_unique_files_content_hashed':validation['unique_files_content_hashed']},
       'our4checkpoint_content_hashes_rechecked':CHECKPOINTS,'execution_source_hashes_rechecked':sources,'test_video_full_content_read':False,'test_pixels_read':False,'models_loaded':False,
       'frozen_protocol_created':False,'builder_sha256':sha(__file__),'draft_files':{f:sha(output/f) for f in ['protocol.DRAFT.json','sampling.DRAFT.json','evaluation.DRAFT.json']}}
    save(output/'validation.json',receipt);save(output/'artifact_index.json',artifacts);(output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    text='''# 最终测试配置草稿（未冻结）

已经填入8方法、4个真实早期200步checkpoint及其模型/源码SHA、TRAIN source-cache锚点和公开权重来源。候选为bidirectional/200/IP0.9，内部对照同预算同配置；MF50、FA25、LP绝对驱动all已按既有原生验证配置填写。

本方5条通路的num_inference_steps仍为null，明确等待root完成20/50步检查后统一选择。所有文件status均not_frozen，extraction_authorized=false，没有创建frozen_protocol.json；8段test视频的全文SHA仍为空，没有读取视频全文或像素，没有运行模型。

统计配置：external_target_au只有3个主要C编辑比较；internal_target_au有4个次要结构C编辑比较。另4个诊断家族每族7比较，仅编辑范围的ΔC、身份、未改AU漂移和extra5差值CI；无论其p是否校正，都不作为另一个主要结论。诊断项不改变主3/次4的家族分母。

全部14指标×3scope共42项原均值/n报告保留；三seed先case内均值，再case→subject→8主体等权。LP101实体+202/303别名的实际运行数另报，不当3次独立生成。没有新增综合分数或身份通过阈值。

文件：protocol.DRAFT.json（总草稿）、sampling.DRAFT.json（采样/模型）、evaluation.DRAFT.json（指标/家族）、validation.json（只读核验）。这些DRAFT路径与SHA指向真实已创建的草稿，不是假装未来正式协议已经存在。

Root在50步开发检查后亲自选steps、补真实视频内容SHA并另行冻结；不要通过只改本文件status来跳过剩余核准。
'''
    (output/'README.md').write_text(text)
    print(json.dumps({k:receipt[k] for k in ['status','model_methods','metrics','comparisons','families','our_inference_steps','frozen_protocol_created']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
