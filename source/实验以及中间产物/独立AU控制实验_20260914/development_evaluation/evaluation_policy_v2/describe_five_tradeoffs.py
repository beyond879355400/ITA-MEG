#!/usr/bin/env python3
"""Existing metrics only: all cases/subjects and non-target costs, no new score."""
import json
from pathlib import Path
from statistics import mean
from openface_v2 import HERE,POLICY_SHA,sha,save
METHODS=['baseline','current_model','condition_reads_image','image_reads_condition','mlp_model']
NAMES=['公开基底','双向','条件读图像','图像读条件','MLP']
KEYS={'target_mae_changed':'C-MAE↓','delta_mae_changed':'ΔC-MAE↓','unchanged_source_relative_drift':'未改12AU漂移↓','identity_cosine':'FaceNet身份↑',
      'target_mae_all12':'全12AU目标MAE↓','delta_mae_all12':'全12AU变化MAE↓','direction_correct_fraction':'方向正确比例↑',
      'extra5_proxy_mae_to_real_target':'额外5AU平均目标测量误差↓'}
KEYS.update({f'extra_AU{a}_error_to_real_target':f'额外AU{a}目标测量误差↓' for a in [7,10,14,23,45]})
def average(values):return mean(values) if values else None
def fmt(value):return 'NA' if value is None else f'{value:.5f}'
def main():
    source=HERE/'comparison_five/results.json';result=json.loads(source.read_text());assert result['policy_sha256']==POLICY_SHA and not result['test_used']
    rows=[r for r in result['records'] if r['method'] in METHODS];mapping={(r['case_id'],r['method']):r for r in rows}
    scopes={};subjects={};individual=[]
    for scope in ['all','edited','no_edit']:
        cases=sorted({r['case_id'] for r in rows if scope=='all' or (r['group']=='no_edit')==(scope=='no_edit')});scopes[scope]={};subjects[scope]={}
        for key in KEYS:
            valid=[c for c in cases if all(mapping[(c,m)][key] is not None for m in METHODS)];ss=sorted({mapping[(c,METHODS[0])]['subject'] for c in valid})
            per={s:{m:average([mapping[(c,m)][key] for c in valid if mapping[(c,m)]['subject']==s]) for m in METHODS} for s in ss}
            counts={s:sum(mapping[(c,METHODS[0])]['subject']==s for c in valid) for s in ss}
            scopes[scope][key]={'planned':len(cases),'common_n':len(valid),'subject_n':len(ss),'means':{m:average([per[s][m] for s in ss]) for m in METHODS},'case_ids':valid,'per_subject':per,'per_subject_n':counts}
        for c in cases:
            if scope!='all':continue
            first=mapping[(c,METHODS[0])];individual.append({'case_id':c,'subject':first['subject'],'group':first['group'],'cohort':first['cohort'],
                       'changed_aus':first['changed_aus'],'unchanged_aus':first['unchanged_aus'],'source_labels_12':first['source_labels_12'],'target_labels_12':first['target_labels_12'],
                       'methods':{m:{k:mapping[(c,m)][k] for k in KEYS} for m in METHODS}})
    out=HERE/'comparison_five/tradeoffs';out.mkdir(exist_ok=False)
    save(out/'metrics_by_scope_subject_case.json',{'status':'complete','source_sha256':sha(source),'policy_sha256':POLICY_SHA,'test_used':False,
              'new_metric_or_threshold_introduced':False,'aggregation':'common cases for all5; mean within subject, then mean across subjects','scopes':scopes,'cases':individual,'script_sha256':sha(__file__)})
    lines=['# 五方法开发集完整指标与非目标代价','',
      '这份核查只重排已经冻结的指标，不新增评分、阈值或样本。五方法使用同19请求/种子，四个可训练结构同200步、同训练案例顺序/时刻、近等参数量。这里只看开发集，没有读取测试数据。','',
      '主体等权均值先在共同有效请求内按主体平均，再在主体间平均。C/ΔC仅在人工源/目标12AU不同的维度计算；不编辑请求的C为空，报告NA，不补0。额外5AU与真实目标图的OpenFace测量比较，没有人工零标签。', '']
    for scope,title in [('edited','15个编辑请求（主比较）'),('all','全部19请求（含4个不编辑）'),('no_edit','4个不编辑请求（单列）')]:
        lines+=['## '+title,'','| 既有指标 | 共同有效 n/主体数 | '+' | '.join(NAMES)+' |','|---|---:|'+'---:|'*5]
        for key,label in KEYS.items():
            stat=scopes[scope][key];lines+=['| '+label+' | '+str(stat['common_n'])+'/'+str(stat['subject_n'])+' | '+' | '.join(fmt(stat['means'][m]) for m in METHODS)+' |']
        lines+=['']
    lines+=['## 编辑请求的主体分布','','各主体编辑请求数不相同，因此没有把所有帧直接池化排名。','']
    for key in ['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','extra5_proxy_mae_to_real_target','identity_cosine']:
        stat=scopes['edited'][key];lines+=['### '+KEYS[key],'','| 主体 | 有效 n | '+' | '.join(NAMES)+' |','|---|---:|'+'---:|'*5]
        for s,values in stat['per_subject'].items():lines+=['| '+s+' | '+str(stat['per_subject_n'][s])+' | '+' | '.join(fmt(values[m]) for m in METHODS)+' |']
        lines+=['']
    lines+=['## 用于后续训练验收的结论','',
      '1. 双向的15例C-MAE与ΔC-MAE均为当前五方法最低；相对图像读条件单向，差值仅约0.01057和0.00182，不能据此把两个方向都必要当成已证事实。',
      '2. 双向未改12AU漂移0.43468，略低于基底0.44054和两单向，但高于MLP0.42004。当前C/Δ改善不伴随该12AU平均漂移的整体上升，仍有主体差异。',
      '3. 额外5AU的平均真实目标测量误差：基底0.43904，双向0.48446，两单向0.47417/0.48738，MLP0.47021。四种训练结构都比基底高。AU45单项更好，不能据此概括非目标动作整体更稳。',
      '4. SN018的双向C-MAE1.45236比基底1.41847更高；SN012的双向FaceNet身份0.73801低于基底0.77627。其余主体也全部保留，未按正向结果筛人。',
      '5. 若继续200→2000步，沿用这套C、ΔC、未改12AU漂移、独立FaceNet、全部5额外AU和覆盖率逐项验收，报告相同19/15/4口径、逐主体和36个AU×等级。不得仅凭C下降切换更有利的身份或非目标指标，也不能用4个零请求稀释编辑代价。',
      '6. 四个不编辑请求的全部五方法PNG SHA完全相同。这验证零请求的构造性质，不代表非零请求没有副作用。','',
      '逐请求全部数值保存在同目录`metrics_by_scope_subject_case.json`的19个cases中；完整原图/检测候选/原始CSV/逐AU等级仍由上一层`results.json`链接。没有重新生成图片，也没有调整测量政策。','']
    with (out/'report.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'complete','cases':len(individual),'subjects':{s:scopes[s]['target_mae_all12']['per_subject_n'] for s in scopes},'path':str(out/'report.md')},ensure_ascii=False))
if __name__=='__main__':main()
