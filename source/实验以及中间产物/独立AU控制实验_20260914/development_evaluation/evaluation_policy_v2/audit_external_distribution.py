#!/usr/bin/env python3
"""Read-only derivation of existing subject and all36 AU-level differences."""
import json
from openface_v2 import HERE,sha,save,POLICY_SHA
METHODS=['liveportrait','faceadapter']
def main():
    file=HERE/'comparison_seven/results.json';r=json.loads(file.read_text());assert r['policy_sha256']==POLICY_SHA and not r['test_used']
    subjects={};cells=[];counts={}
    for metric in ['target_mae_changed','delta_mae_changed']:
        part=r['all_methods_common_valid']['all']['edited'][metric];assert part['common_cases']==15 and len(part['subjects'])==4
        subjects[metric]={s:{'existing_means':{m:v[m] for m in ['current_model']+METHODS},'bidirectional_minus_external':{m:v['current_model']-v[m] for m in METHODS}} for s,v in part['per_subject'].items()}
    for au in [1,2,4,5,6,9,12,15,17,20,25,26]:
        for level in [1,2,3]:
            stat=r['au_level_all_methods'][f'AU{au}_level{level}'];assert stat['planned_n']>0
            row={'au':au,'target_level':level,'planned_n':stat['planned_n'],'case_ids':stat['planned_case_ids'],'metrics':{}}
            for key in ['target_error','delta_error']:
                v=stat['metrics'][key];assert v['n']==stat['planned_n']
                row['metrics'][key]={'n':v['n'],'subject_n':v['subject_n'],'per_subject':{s:{m:a[m] for m in ['current_model']+METHODS} for s,a in v['per_subject'].items()},
                    'existing_subject_equal_means':{m:v['subject_equal_means'][m] for m in ['current_model']+METHODS},
                    'bidirectional_minus_external':{m:v['subject_equal_means']['current_model']-v['subject_equal_means'][m] for m in METHODS}}
            cells.append(row)
    assert len(cells)==36
    for key in ['target_error','delta_error']:
        counts[key]={}
        for method in METHODS:
            delta=[row['metrics'][key]['bidirectional_minus_external'][method] for row in cells]
            counts[key][method]={'smaller':sum(x<0 for x in delta),'equal':sum(x==0 for x in delta),'larger':sum(x>0 for x in delta)}
    out=HERE/'comparison_seven/distribution_audit';out.mkdir(exist_ok=False)
    save(out/'results.json',{'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'source_sha256':sha(file),'new_metric_or_threshold':False,'subjects':subjects,'all36_cells':cells,
          'descriptive_cell_sign_counts':counts,'cell_count_not_independent_sample_size':True,'script_sha256':sha(__file__)})
    lines=['# 双向相对两种真实驱动方法：优势分布核查','',
      '使用已完成七方法报告的同15个编辑请求、4个开发主体和全部36个AU×1/2/3级格。不读新数据，不改变指标，不挑子集。差值统一为“双向−外部方法”，误差为负表示双向较低。','',
      '## 每个主体的C与ΔC差','', '| 主体 | 编辑请求数 | C：对LP | C：对FA | ΔC：对LP | ΔC：对FA |','|---|---:|---:|---:|---:|---:|']
    expected_n={'SN004':3,'SN012':2,'SN017':4,'SN018':6}
    for subject in ['SN004','SN012','SN017','SN018']:
        values=[subjects[key][subject]['bidirectional_minus_external'][m] for key in ['target_mae_changed','delta_mae_changed'] for m in METHODS]
        lines+=['| '+subject+' | '+str(expected_n[subject])+' | '+' | '.join(f'{v:.5f}' for v in values)+' |']
    lines+=['','四个主体的两项误差差值均为负：不是只有某一个主体拉低均值。SN018对两个外部驱动方法仍更低，但在内部基底比较中SN018的C并未改善；两件事不是同一个参照。','',
      '## 全部36格（不省略不利格）','', '| AU | 目标级别 | 请求 n / 主体 n | 目标误差：对LP | 目标误差：对FA | 变化误差：对LP | 变化误差：对FA |','|---|---:|---:|---:|---:|---:|---:|']
    for row in cells:
        values=[row['metrics'][key]['bidirectional_minus_external'][m] for key in ['target_error','delta_error'] for m in METHODS]
        lines+=['| '+str(row['au'])+' | '+str(row['target_level'])+' | '+str(row['planned_n'])+' / '+str(row['metrics']['target_error']['subject_n'])+' | '+' | '.join(f'{v:.5f}' for v in values)+' |']
    lines+=['','## 分布判断','',
      '- C目标误差：对LivePortrait，36格中21格更低、4格相等、11格更高；对Face-Adapter，22格更低、4格相等、10格更高。改善分布在多个AU，不是单一AU贡献，但不是每个AU/级别都改善。',
      '- Δ变化误差：对LivePortrait，18格更低、4格相等、14格更高；对Face-Adapter，24格更低、4格相等、8格更高。对LP的变化误差分布更混合，不能只用总体均值说所有动作更准。',
      '- AU4的1/2/3级目标误差对两方法都更低。AU15、AU17及其他若干格也有改善；AU26对LP的三个级别均更高，AU20三级对两方法均更高，AU6二/三级对两方法均更高。这些缺口完整保留。',
      '- 每格只有1–3个请求，若干格共享同一个目标帧，36格不是36个独立统计样本。这里只是既有分层结果的方向核查，没有新增综合分数或显著性判断。',
      '- 对外部方法的目标C/Δ优势伴随身份和非目标稳定性差距；应与原报告的完整指标共同解释。LivePortrait与Face-Adapter获真实目标驱动图，不能将此表称为相同输入条件或相同训练设置排名。','']
    with (out/'report.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'complete','subjects':4,'cells':36,'counts':counts},ensure_ascii=False))
if __name__=='__main__':main()
