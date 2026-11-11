#!/usr/bin/env python3
"""All36 existing AU-level comparisons and every subject; no new scoring rule."""
import json
from openface_v2 import HERE,POLICY_SHA,sha,save

def main():
    file=HERE/'comparison_eight/results.json';r=json.loads(file.read_text());assert r['policy_sha256']==POLICY_SHA and not r['test_used']
    scopes=r['all_methods_common_valid']['all'];subjects={};cells=[];counts={}
    for key in ['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','identity_cosine']:
        p=scopes['edited'][key];assert p['common_cases']==15 and len(p['subjects'])==4
        subjects[key]={s:{'bidirectional':v['current_model'],'magicface':v['magicface'],'difference':v['current_model']-v['magicface']} for s,v in p['per_subject'].items()}
    for au in [1,2,4,5,6,9,12,15,17,20,25,26]:
        for level in [1,2,3]:
            p=r['au_level_all_methods'][f'AU{au}_level{level}'];v={'au':au,'level':level,'planned_n':p['planned_n'],'case_ids':p['planned_case_ids'],'metrics':{}}
            for key in ['target_error','delta_error']:
                q=p['metrics'][key];assert q['n']==p['planned_n']>0
                v['metrics'][key]={'n':q['n'],'subject_n':q['subject_n'],'bidirectional':q['subject_equal_means']['current_model'],'magicface':q['subject_equal_means']['magicface'],
                    'difference':q['subject_equal_means']['current_model']-q['subject_equal_means']['magicface'],'per_subject':{s:{m:a[m] for m in ['current_model','magicface']} for s,a in q['per_subject'].items()}}
            cells.append(v)
    for key in ['target_error','delta_error']:
        values=[v['metrics'][key]['difference'] for v in cells];counts[key]={'smaller':sum(v<0 for v in values),'equal':sum(v==0 for v in values),'larger':sum(v>0 for v in values)}
    out=HERE/'comparison_eight/magicface_distribution';out.mkdir(exist_ok=False)
    save(out/'results.json',{'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'source_sha256':sha(file),'new_metric_or_threshold':False,'subjects':subjects,'all36_cells':cells,
         'cell_direction_counts':counts,'cells_are_not_independent_samples':True,'script_sha256':sha(__file__)})
    lines=['# 双向与MagicFace：全部主体和36格检查','',
      '同15个编辑请求、4个开发主体。全部差值为双向−MagicFace，误差越低越好，身份越高越好。使用原先定义的指标，不增加评分、不选阈值、不改样本。','',
      '## 主体差异','', '| 主体 | C误差差 | ΔC误差差 | 未改12AU漂移差 | FaceNet身份差 |','|---|---:|---:|---:|---:|']
    for s in ['SN004','SN012','SN017','SN018']:
        lines+=['| '+s+' | '+' | '.join(f"{subjects[key][s]['difference']:.5f}" for key in ['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','identity_cosine'])+' |']
    lines+=['','C在3个主体更低，SN018略高0.01097；ΔC在4个主体更低。身份只有SN004更高，其他3个主体更低。不是对所有主体和指标都占优。','',
      '## 全部AU×等级','', '| AU | 目标等级 | 有效请求 n/主体 n | 目标误差差 | 变化误差差 |','|---|---:|---:|---:|---:|']
    for c in cells:lines+=['| '+str(c['au'])+' | '+str(c['level'])+' | '+str(c['planned_n'])+'/'+str(c['metrics']['target_error']['subject_n'])+' | '+f"{c['metrics']['target_error']['difference']:.5f} | {c['metrics']['delta_error']['difference']:.5f} |"]
    lines+=['','目标误差36格：22格更低、3格相等、11格更高。变化误差36格：22格更低、2格相等、12格更高。每格1–3例，且多个格共享同一帧；方向计数不是独立样本统计检验。','',
      '额外5AU平均目标测量误差：双向0.48446，MagicFace0.50204；这一项平均略低，但仍必须连同全5项分别列出的误差、未改12AU漂移和身份一起看。完整数值见上一层八方法报告。','',
      'MF是原生AU变化输入，不接收真实目标驱动图；LP/FA的分布检查保存在七方法distribution_audit中，八方法报告仍将真实驱动组单列。全部报告只用于开发判断，最终测试尚未进入。','']
    with (out/'report.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'complete','subjects':4,'cells':36,'counts':counts}))
if __name__=='__main__':main()
