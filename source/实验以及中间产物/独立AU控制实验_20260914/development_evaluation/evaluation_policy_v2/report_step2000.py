"""Actual completed step2000 results: independent CSV audit and full200->2000 report."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from step2000_cpu_eval import checked
from step2000_gate import HERE,POLICY_SHA,sha,save,require

AUS12=[1,2,4,5,6,9,12,15,17,20,25,26];AUS17=[1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]
PAIRS={'bidirectional':('current_model','bidirectional_step2000'),'condition_reads_image':('condition_reads_image','condition_reads_image_step2000'),
       'image_reads_condition':('image_reads_condition','image_reads_condition_step2000'),'mlp':('mlp_model','mlp_step2000')}
KEYS=['target_mae_changed','delta_mae_changed','identity_cosine','unchanged_source_relative_drift','extra5_proxy_mae_to_real_target',
      'target_mae_all12','delta_mae_all12','direction_correct_fraction','buffalo_identity_cosine']+[f'extra_AU{a}_error_to_real_target' for a in (7,10,14,23,45)]

def direct_csv(row):
    files=row.get('csv_files',[])
    if len(files)!=1:
        require(row['vector'] is None,'MissingCSV but finiteAU claimed');return {'status':'no_single_csv','input_sha256':row['sha256']}
    path=Path(files[0]);require(path.is_file(),'RawCSV missing from archived result')
    with path.open() as f:raw=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
    candidates=[]
    for index,r in enumerate(raw):
        try:
            xs=[float(r[f'x_{i}']) for i in range(68)];ys=[float(r[f'y_{i}']) for i in range(68)]
            if all(math.isfinite(x) for x in xs+ys):
                area=(max(xs)-min(xs))*(max(ys)-min(ys))
                if area>0:candidates.append((area,index))
        except (KeyError,ValueError):pass
    selected=max(candidates,key=lambda x:(x[0],-x[1]))[1] if candidates else None;values=None;confidence=None;vector=None
    if selected is not None:
        r=raw[selected];values={}
        for a in AUS17:
            try:v=float(r[f'AU{a:02d}_r']);v=v if math.isfinite(v) and 0<=v<=5 else None
            except (KeyError,ValueError):v=None
            values[str(a)]=v
        try:confidence=float(r['confidence']);confidence=confidence if math.isfinite(confidence) else None
        except (KeyError,ValueError):pass
        if confidence is not None and all(values[str(a)] is not None for a in AUS12):vector=[values[str(a)] for a in AUS12]
    if row.get('returncode')!=0 or row.get('input_unchanged') is False:vector=None;values=None
    require(row['raw_rows']==raw and row['face_count']==len(raw),'Archived raw rows/count changed')
    require(row['selected_row']==selected and row['confidence']==confidence,'Archived primary face/confidence changed')
    require(row['vector']==vector and row['selected_au17']==values,'NewAU differs from actual originalCSV (possible input pollution)')
    require(sha(row['path'])==row['sha256'],'Input image changed since measurement')
    return {'status':'pass','csv':str(path),'csv_sha256':sha(path),'case_id':row['case_id'],'method':row['method'],'selected_row':selected,'AU_valid':vector is not None}

def old_new_comparisons(old,new):
    a={(r['case_id'],r['method']):r for r in old};b={(r['case_id'],r['method']):r for r in new};result={}
    for variant,(previous,current) in PAIRS.items():
        result[variant]={}
        for scope in ('all','edited','no_edit'):
            ids=sorted(r['case_id'] for r in new if r['method']==current and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit')));stats={}
            for key in KEYS:
                valid=[cid for cid in ids if a[(cid,previous)].get(key) is not None and b[(cid,current)].get(key) is not None]
                subjects=sorted({b[(cid,current)]['subject'] for cid in valid});per={}
                for s in subjects:
                    cases=[cid for cid in valid if b[(cid,current)]['subject']==s];av=mean(a[(cid,previous)][key] for cid in cases);bv=mean(b[(cid,current)][key] for cid in cases)
                    per[s]={'case_count':len(cases),'step200':av,'step2000':bv,'step2000_minus200':bv-av}
                stats[key]={'planned_cases':len(ids),'common_cases':len(valid),'case_ids':valid,'subject_n':len(subjects),'per_subject':per,
                    'step200':mean(v['step200'] for v in per.values()) if per else None,'step2000':mean(v['step2000'] for v in per.values()) if per else None,
                    'step2000_minus200':mean(v['step2000_minus200'] for v in per.values()) if per else None}
            result[variant][scope]=stats
    return result

def fmt(x):return 'NA' if x is None else f'{x:.5f}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);args=p.parse_args();out,doc=checked(args.group);folder=out/'comparison'
    result=json.loads((folder/'results.json').read_text());verification=json.loads((folder/'independent_verification.json').read_text())
    require(result['status']=='complete' and result['policy_sha256']==POLICY_SHA and verification['status']=='pass' and verification['input_sha256']==sha(folder/'results.json'),'Actual complete independently verified result required')
    audit=[direct_csv(r) for r in result['records']]
    save(folder/'independent_raw_csv_audit.json',{'status':'pass','record_count':len(audit),'CSV_verified_count':sum(r['status']=='pass' for r in audit),'records':audit,
          'results_sha256':sha(folder/'results.json'),'script_sha256':sha(__file__),'test_used':False,'no_evaluator_called':True})
    old=json.loads((HERE/'comparison_nine/results.json').read_text());changes=old_new_comparisons(old['records'],result['records'])
    save(folder/'step200_to2000.json',{'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'old_results_sha256':sha(HERE/'comparison_nine/results.json'),
          'new_results_sha256':sha(folder/'results.json'),'differences':changes,'difference_definition':'2000 minus200 on same valid cases/subjects; no direction sign flip'})
    methods=result['methods'];lines=['# IP0.9：四结构2000步开发评价','',
      '本报告来自真实完成的76张新模型生成图，保留封存基底与MagicFace、Face-Adapter、LP相对/绝对设置。原200步结果不覆盖。所有方法同dev19与seed101；这里只是开发评价，不是最终测试。', '',
      'LP和Face-Adapter获得真实目标驱动图，属于额外driver组；LP相对/绝对是同方法两种设置。参数预算与输入信息不能混为同训练SOTA排名。', '',
      '## 共同有效九组比较','']
    for scope,title in [('edited','编辑请求'),('all','全部请求'),('no_edit','不编辑请求')]:
        primary=result['all_methods_common_valid']['all'][scope];extra=result['extra_diagnostics']['all_methods'][scope]
        lines+=['### '+title,'','| 指标 | 共同n/主体 | '+' | '.join(methods)+' |','|---|---:|'+'---:|'*len(methods)]
        for key in KEYS:
            stat=primary.get(key) or extra[key];n=stat.get('common_cases',stat.get('n'));means=stat['subject_equal_means']
            lines+=['| '+key+' | '+str(n)+'/'+str(len(stat['subjects']))+' | '+' | '.join(fmt(means[m]) for m in methods)+' |']
        lines+=['']
    lines+=['## 同结构200→2000（完整指标，不只报改善）','']
    for variant,parts in changes.items():
        lines+=['### '+variant,'','| 指标 | 共同n/主体 | 200步 | 2000步 | 差值2000−200 |','|---|---:|---:|---:|---:|']
        for key in KEYS:
            s=parts['edited'][key];lines+=['| '+key+' | '+str(s['common_cases'])+'/'+str(s['subject_n'])+' | '+fmt(s['step200'])+' | '+fmt(s['step2000'])+' | '+fmt(s['step2000_minus200'])+' |']
        lines+=['','逐主体（编辑请求）：','', '| 主体 | C 200→2000 | ΔC 200→2000 | ID 200→2000 | 未改AU漂移200→2000 | extra5 200→2000 |','|---|---|---|---|---|---|']
        subjects=sorted({s for k in KEYS for s in parts['edited'][k]['per_subject']})
        for subject in subjects:
            values=[]
            for key in ['target_mae_changed','delta_mae_changed','identity_cosine','unchanged_source_relative_drift','extra5_proxy_mae_to_real_target']:
                x=parts['edited'][key]['per_subject'].get(subject);values.append('NA' if x is None else fmt(x['step200'])+' → '+fmt(x['step2000']))
            lines+=['| '+subject+' | '+' | '.join(values)+' |']
        lines+=['']
    lines+=['## 验证与范围','',
      '- 全部原始CSV独立读取、重新按68点框最大面积选主脸，再核12/17AU、confidence、行数与实际PNG SHA；不是只对同一份汇总数字再次相加。',
      '- 共同有效案例按指标明确给n，先主体内均值、再主体等权；失败/无脸/NA保留在coverage中。相同seed/同案例的200与2000才配对。',
      '- 来源、训练顺序/时刻、检查点/父检查点、白名单和零请求与基底一致性均由prepare门禁核验。所有新评价CPU，未访问最终test。',
      '- 全部原始记录和36AU×等级在results.json；200→2000的all/edited/no_edit与每主体数值在step200_to2000.json。本程序不挑最优模型或更换指标。','']
    with (folder/'report.md').open('x') as f:f.write('\n'.join(lines))
    summary={'status':'complete','group':args.group,'policy_sha256':POLICY_SHA,'results_sha256':sha(folder/'results.json'),'raw_CSV_audit_pass':True,
      'edited_nine':{k:{'n':v['common_cases'],'means':v['subject_equal_means']} for k,v in result['all_methods_common_valid']['all']['edited'].items()},
      'edited200_to2000':{v:{k:{name:s[name] for name in ['common_cases','subject_n','step200','step2000','step2000_minus200']} for k,s in parts['edited'].items()} for v,parts in changes.items()}}
    save(folder/'compact_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
