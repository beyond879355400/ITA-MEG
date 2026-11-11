#!/usr/bin/env python3
"""Present all existing external-comparison metrics; do not manufacture a score."""
import argparse
import json
from openface_v2 import HERE,POLICY_SHA,sha,save
NAMES={'baseline':'基底','current_model':'双向','condition_reads_image':'条件读图像','image_reads_condition':'图像读条件','mlp_model':'MLP','magicface':'MagicFace','liveportrait':'LivePortrait†','faceadapter':'Face-Adapter†','liveportrait_absolute':'LP绝对驱动†'}
KEYS={'target_mae_changed':'C-MAE↓','delta_mae_changed':'ΔC-MAE↓','unchanged_source_relative_drift':'未改12AU漂移↓','target_mae_all12':'全12AU目标MAE↓',
      'delta_mae_all12':'全12AU变化MAE↓','direction_correct_fraction':'方向正确比例↑','identity_cosine':'FaceNet身份↑'}
EXTRA={'buffalo_identity_cosine':'buffalo辅助身份↑','extra5_proxy_mae_to_real_target':'额外5AU平均目标测量误差↓'}
EXTRA.update({f'extra_AU{a}_error_to_real_target':f'额外AU{a}目标测量误差↓' for a in [7,10,14,23,45]})
def fmt(x):return 'NA' if x is None else f'{x:.5f}'
def main():
    p=argparse.ArgumentParser();p.add_argument('folder',choices=('comparison_nine',));args=p.parse_args();out=HERE/args.folder
    r=json.loads((out/'results.json').read_text());check=json.loads((out/'independent_verification.json').read_text());assert r['policy_sha256']==POLICY_SHA and check['status']=='pass' and check['input_sha256']==sha(out/'results.json')
    methods=r['methods'];lines=['# 原生外部方法：同开发请求、同评价器对照','',
       '已完成方法：'+ '、'.join(NAMES[m] for m in methods)+'。每种方法同19个冻结开发请求、相同源身份和人工12AU目标；完整保留15编辑与4不编辑。所有统计按共同有效案例、先主体内平均再4主体等权。', '',
       '九组设置不是九个独立SOTA模型：LivePortrait的原生相对驱动与绝对驱动分别保留。绝对驱动是开发阶段追加的单因素候选，只将flag_relative_motion从True改False，animation_region=all和其他参数、19输入、模型/源码均相同；没有逐例选更优输出。', '',
       '† LivePortrait和Face-Adapter收到额外的真实目标驱动图；其他方法通过源身份及人工AU请求生成。驱动组单列，不冒充原生AU控制组。各作者的原生预处理、预训练数据、采样器与步数不同，因此这是同案例、同指标的实际调用对照，不是相同训练设置的SOTA排名。', '',
       '评价读取封存的原始生成PNG：LivePortrait1024×768，Face-Adapter512×512，MagicFace为512×512，本方512×512。不再额外统一裁图、美化或按检测结果调图；真实源/目标参照使用原固定canonical512。OpenFace、FaceNet只按几何面积选主脸，全部候选和原始CSV保留。','',
       '## 覆盖率','', '| 方法 | 输入条件组 | AU有效/19 | FaceNet有效/19 | buffalo有效/19 |','|---|---|---:|---:|---:|']
    for m in methods:
        a=r['coverage'][m]['all'];group='源图+真实目标驱动图' if m in ('liveportrait','faceadapter','liveportrait_absolute') else '源图+原生AU变化' if m=='magicface' else '源图+AU/派生文本条件'
        lines+=['| '+NAMES[m]+' | '+group+' | '+str(a['AU_valid'])+' | '+str(a['FaceNet_valid'])+' | '+str(a['buffalo_valid'])+' |']
    for scope,title in [('edited','15个编辑请求'),('all','全部19个请求'),('no_edit','4个不编辑请求')]:
        lines+=['','## '+title,'','| 指标 | 共同有效 n/主体数 | '+' | '.join(NAMES[m] for m in methods)+' |','|---|---:|'+'---:|'*len(methods)]
        stats={**r['all_methods_common_valid']['all'][scope],**r['extra_diagnostics']['all_methods'][scope]}
        for key,label in {**KEYS,**EXTRA}.items():
            v=stats[key];n=v.get('common_cases',v.get('n'));lines+=['| '+label+' | '+str(n)+'/'+str(len(v['subjects']))+' | '+' | '.join(fmt(v['subject_equal_means'][m]) for m in methods)+' |']
    lines+=['','## 逐主体检查','']
    for key in ['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','identity_cosine','extra5_proxy_mae_to_real_target']:
        stats=r['all_methods_common_valid']['all']['edited'] if key in KEYS else r['extra_diagnostics']['all_methods']['edited'];v=stats[key]
        lines+=['### '+({**KEYS,**EXTRA}[key]),'','| 主体 | '+' | '.join(NAMES[m] for m in methods)+' |','|---|'+'---:|'*len(methods)]
        for subject,values in v['per_subject'].items():lines+=['| '+subject+' | '+' | '.join(fmt(values[m]) for m in methods)+' |']
        lines+=['']
    lines+=['## 当前可据实作出的判断','',
       '- 对原生AU编辑方法MagicFace，双向C-MAE为0.90517（MagicFace1.17335），ΔC-MAE为1.19833（MagicFace1.51144）；身份为0.71771（MagicFace0.76857）。MF的未改12AU漂移和全12AU平均误差也更低，因此优势是目标动作响应，不是全部质量指标。',
       '- 对MagicFace的C误差在3/4主体更低，SN018双向比MF高0.01097；ΔC在4/4主体更低。FaceNet身份只在SN004更高，另外3个主体低于MF。这些主体差异全部列出，不按方向筛人。',
       '- LP绝对驱动比原相对驱动更强：C-MAE1.05010（原1.10428），ΔC1.39652（原1.40182），方向正确比例0.62569（原0.56354）。双向的C0.90517/ΔC1.19833/方向0.67778仍较好，但只在3/4主体的C/Δ更低；SN004绝对驱动更好。不能沿用原对LP四主体都更低的表述。',
       '- LP绝对驱动身份0.95927、未改12AU漂移0.20458、额外5AU平均误差0.22417，均优于双向0.71771/0.43468/0.48446。整体身份/非目标稳定性的差距没有消失。',
       '- 本方双向200步相对两个真实驱动方法，C-MAE和ΔC-MAE更低、变化方向正确比例更高；目标动作响应是当前开发对照表现较好的维度。',
       '- LivePortrait和Face-Adapter的身份保持、未改AU漂移、全12AU平均误差和额外5AU目标测量误差较好。本方不能把“身份保持超越所有近期方法”作为这组数据的结论。',
       '- 原内部五方法比较仍单独保留：双向的目标变化误差最低，但对图像读条件单向的差很小；不能从开发集点估计推成双向机制必不可少。',
       '- 12AU的C集合由人工源/目标标签不同的维度定义；不编辑C为空，返回NA。额外5AU全部与真实目标的测量值比较，没有伪造人工零值。',
       '- 本方四个不编辑请求的五结构PNG相同；外部方法的零请求按真实生成结果评价，不要求或假定它们也逐字节相同。',
       '- 目前只有4个开发主体，每个AU×强度格1–3例，报告逐项n和所有主体，不据此作最终测试显著性或总体SOTA结论。','',
       '完整逐图记录、所有两两比较、共同有效n、AU×1/2/3分层和输入来源见`results.json`；原八组190记录原样保留。独立检查'+str(check['independent_metric_checks'])+'个逐图指标、'+str(check['subject_equal_mean_checks'])+'个汇总、'+str(check['geometry_checks'])+'个主脸选择与'+str(check['input_images_SHA_verified'])+'个输入SHA全部通过。','']
    if 'magicface' not in methods:lines+=['MagicFace19原生调用仍在运行，当前表没有用smoke或旧图代替；完成后另写八方法报告，当前七方法文件保留。','']
    with (out/'report.md').open('x') as f:f.write('\n'.join(lines))
    save(out/'report_receipt.json',{'source_sha256':sha(out/'results.json'),'report_sha256':sha(out/'report.md'),'script_sha256':sha(__file__),'policy_sha256':POLICY_SHA,'test_used':False})
    print(str(out/'report.md'))
if __name__=='__main__':main()
