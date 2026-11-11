"""Chinese report and fixed diagnostic galleries; display only, no detectors.

Reuses the verified old contact-sheet font/text/contain geometry helpers.
Metric values and Top3 selection are read from completed frozen exports.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import types
from PIL import Image,ImageDraw,ImageFont
import protocol as p
import extent_io as extent

HERE=Path(__file__).resolve().parent
OLD_RENDER=p.EXP/'completion_20260915/visual_review_tools/render_contacts.py'
OLD_RENDER_SHA='38b8dc1707080c73b0de342a985c442651df5ee034059d3411b292c08e97ef3a'
NAMES={p.CHOSEN:'本方（所选双向）','base_local_lp_s050':'公开基底+固定控制器',
       'condition_reads_image_local_lp_s050':'条件读取图像（单向）','image_reads_condition_local_lp_s050':'图像读取条件（单向）',
       'mlp_local_lp_s050':'MLP+相同控制器','magicface':'MagicFace','faceadapter':'Face-Adapter',
       'liveportrait':'LivePortrait','swiftedit':'SwiftEdit','performrecast':'PerformRecast',
       'source_real':'源图（评价参照）','target_real':'目标图（评价参照）'}
CORE=[('target_mae_changed','目标AU误差↓'),('delta_mae_changed','AU变化误差↓'),
      ('unchanged_source_relative_drift','非目标AU漂移↓'),('target_mae_all12','全部12AU误差↓'),
      ('direction_correct_fraction','变化方向正确率↑'),('identity_cosine','身份相似度↑')]


def helpers():
    source=extent.read_bytes(OLD_RENDER,OLD_RENDER_SHA)
    module=types.ModuleType('report_frozen_contact_helpers');module.__file__=str(OLD_RENDER)
    exec(compile(source,str(OLD_RENDER),'exec'),module.__dict__)
    return module


def number(value,precision=4):return 'NA'if value is None else f'{value:.{precision}f}'


def image_bytes(reference):
    p.need(isinstance(reference,dict),'No image reference')
    path=Path(reference['path']).resolve();p.need(path.is_relative_to(p.EXP),'Image outside experiment')
    data=extent.read_bytes(path,reference['sha256'])
    with Image.open(io.BytesIO(data))as image:
        image.load();p.need(image.mode=='RGB','Native image must remain RGB');result=image.copy()
    return result,{'path':str(path),'sha256':reference['sha256'],'image_size':list(result.size)}


def measured_face_box(measurement,reference,size):
    """Only largest-face OpenFace bbox already measured on these exact bytes."""
    if not measurement or not reference or measurement.get('sha256')!=reference.get('sha256'):return None
    if measurement.get('policy_sha256')!=p.POLICY_SHA:return None
    if measurement.get('image_size')is not None and list(size)!=measurement['image_size']:return None
    chosen=measurement.get('selected_row')
    rows=[r for r in measurement.get('all_candidates',[])if r.get('row_index')==chosen]
    if len(rows)!=1:return None
    box=rows[0].get('bbox68')
    if not isinstance(box,list)or len(box)!=4 or not all(type(v)in (float,int)and math.isfinite(v)for v in box):return None
    x0,y0,x1,y1=box
    if x1<=x0 or y1<=y0:return None
    # One predeclared rule for every column: square at1.15 x maximum side,
    # clipped to real pixels. No hand-selected brow/chin/face regions.
    side=max(x1-x0,y1-y0)*1.15;cx=(x0+x1)/2;cy=(y0+y1)/2
    crop=(max(0,math.floor(cx-side/2)),max(0,math.floor(cy-side/2)),
          min(size[0],math.ceil(cx+side/2)),min(size[1],math.ceil(cy+side/2)))
    return list(crop)if crop[2]>crop[0]and crop[3]>crop[1]else None


def draw_cell(canvas,box,image,helper,font,missing=None):
    draw=ImageDraw.Draw(canvas);x,y,w,h=box
    draw.rectangle((x,y,x+w-1,y+h-1),fill='#edf0f2',outline='#bac1c8')
    if image is None:
        helper.text(draw,(x+12,y+h//2-35),missing or '无可用图像',font,w-24,'#9b3333')
        return {'status':'placeholder','reason':missing}
    display,offset,scale=helper.contain_geometry(image.size,box)
    canvas.paste(image.resize(display,Image.Resampling.LANCZOS),offset)
    return {'status':'displayed','source_size':list(image.size),'display_size':list(display),'display_offset':list(offset),'uniform_scale':scale}


def render_case(baseline,cid,subject,rank,case_mean,cells,output,fixture=False):
    helper=helpers();font_path=helper.choose_font()
    fonts={s:ImageFont.truetype(str(font_path),s)for s in [18,20,24,30]}
    loaded=[]
    for cell in cells:
        item={**cell,'loaded':None,'readback':None,'display_error':None}
        if cell.get('status')=='complete':
            try:item['loaded'],item['readback']=image_bytes(cell['image'])
            except (ValueError,OSError,KeyError,TypeError)as error:item['display_error']=str(error)
        else:item['display_error']=cell.get('error')or cell.get('status','缺失')
        item['crop_box']=measured_face_box(cell.get('measurement'),cell.get('image'),item['loaded'].size)if item['loaded']is not None else None
        loaded.append(item)
    can_zoom=all(x['crop_box']is not None for x in loaded)
    width=1490;margin=25;gap=10;cw=(width-2*margin-3*gap)//4
    height=1060 if can_zoom else 710
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    title=('合成版式测试（非实验结果） | 'if fixture else '')+NAMES[baseline]+'：误差较大案例（非整体表现）'
    helper.text(draw,(margin,12),title,fonts[30],width-2*margin)
    helper.text(draw,(margin,60),f'Top {rank} | {subject} | {cid}',fonts[24],width-2*margin)
    helper.text(draw,(margin,98),'按案例跨有效种子平均目标AU误差排序：'+number(case_mean)+'；随机方法固定展示101，不挑最差种子。',fonts[20],width-2*margin)
    receipts=[]
    for i,item in enumerate(loaded):
        x=margin+i*(cw+gap);method=item['method'];label=NAMES[method]
        helper.text(draw,(x,138),label,fonts[24],cw)
        seed='确定性输出'if method=='performrecast'else '固定种子101'if method not in ['source_real','target_real']else '固定评价参照'
        helper.text(draw,(x,174),seed,fonts[18],cw)
        shown=draw_cell(canvas,(x,205,cw,cw),item['loaded'],helper,fonts[20],item['display_error'])
        size='×'.join(map(str,item['readback']['image_size']))if item['readback']else '无图'
        metric=(' | 当前图AU误差 '+number(item.get('display_C_MAE')))if method not in ['source_real','target_real']else ''
        helper.text(draw,(x,205+cw+8),'原生尺寸 '+size+metric,fonts[18],cw)
        rec={k:v for k,v in item.items()if k not in ['loaded','measurement']}
        rec['native_display']=shown
        if can_zoom:
            crop=item['loaded'].crop(item['crop_box'])
            rec['face_zoom']=draw_cell(canvas,(x,660,cw,cw),crop,helper,fonts[18])
            rec['face_zoom']['measured_bbox_policy']='OpenFace selected bbox68; square1.15maxside; clip to real pixels'
        receipts.append(rec)
    if can_zoom:
        helper.text(draw,(margin,620),'面部统一放大：仅使用既有评价框，四列规则相同；只用于展示，不参与任何指标。',fonts[20],width-2*margin)
    else:
        helper.text(draw,(margin,620),'本案例缺少四列一致的既有面部框，仅展示原生图；未新增检测、手工择区或修图。',fonts[20],width-2*margin)
    helper.text(draw,(margin,height-28),'完整总体结果、全32案例与全部失败见同目录中文报告和索引；此页不能代表方法整体表现。',fonts[18],width-2*margin)
    canvas.save(output,format='PNG')
    return {'path':str(Path(output).resolve()),'sha256':extent.sha(output),'width':width,'height':height,'case_id':cid,
            'baseline':baseline,'rank':rank,'cells':receipts,'face_zoom_all_four_columns':can_zoom,
            'source_images_modified':False,'display_only':True,'visual_review_performed':False,'fixture':fixture}


def failure_reasons(row):
    reasons=[]
    if row.get('generation_status')!='complete':return ['生成状态：'+str(row.get('generation_status'))+'；'+str(row.get('failure_reason')or '')]
    if row.get('input_availability')!='available':reasons.append('输入不可测：'+str(row.get('input_availability')))
    if row.get('au_parse_status')!='primary_face_finite':reasons.append('AU评价：'+str(row.get('au_parse_status')))
    if row.get('facenet_status')!='valid':reasons.append('FaceNet评价：'+str(row.get('facenet_status')))
    if row.get('buffalo_status')!='valid_primary_face':reasons.append('辅助身份评价：'+str(row.get('buffalo_status')))
    if row.get('extra5_proxy_mae_to_real_target')is None:reasons.append('附加5AU或其真实目标参照缺测')
    return reasons


def link(reference,label='图像'):
    if not reference:return '无图'
    return f'[{label}](<{Path(reference["path"]).resolve()}>)'


def load_inputs(job):
    job=Path(job).resolve();p.need(job.is_relative_to(HERE),'Job outside versioned evaluation')
    receipt=extent.read_json(job/'statistics/export_receipt.json');p.need(receipt['status']=='complete','Statistics not terminal')
    for name,digest in receipt['outputs'].items():p.need(extent.sha(job/'statistics'/name)==digest,'Completed export changed: '+name)
    doc=extent.read_json(job/'items.json');seal=extent.read_json(job/'items_seal.json')
    p.need(extent.sha(job/'items.json')==seal['sha256'],'Prepared input changed')
    p.need(doc['protocol']['sha256']=='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f','Wrong frozen report protocol')
    extent.read_json(doc['protocol']['path'],doc['protocol']['sha256'])
    combined=extent.read_json(job/'statistics/all_case_seed_metrics.json');case_input=extent.read_json(job/'statistics/case_metric_input.json')
    stats=extent.read_json(job/'statistics/paired_subject_results.json');gallery=extent.read_json(job/'statistics/gallery_selection.json')
    p.need(len(combined['records'])==896 and len(case_input['records'])==320,'Full denominator missing')
    expected=p.gallery_plan(case_input,doc['case_metadata'],combined['per_image_table'])
    p.need(gallery==expected,'Frozen Top3/full-index selection differs')
    old_binding=extent.read_json(p.OLD/'all_methods_v1/openface/binding.json')
    new_binding=extent.read_json(job/'openface/binding.json')
    p.need(old_binding['measurement_policy_sha256']==new_binding['measurement_policy_sha256']==p.POLICY_SHA,'Display geometry policy changed')
    p.need(new_binding['protocol_sha256']==doc['protocol_sha256'],'New geometry belongs to another protocol')
    old_au=extent.read_json(p.OLD/'all_methods_v1/openface/results.json',old_binding['results_sha256'])
    new_au=extent.read_json(job/'openface/results.json',new_binding['results_sha256'])
    refs=extent.read_json(doc['references']['path'],doc['references']['sha256'])
    return doc,combined,case_input,stats,gallery,old_au,new_au,refs,receipt


def report_text(stats,per_image,meta,links):
    lines=['# 近期方法对比汇报','','本轮比较使用固定32个请求、8个主体；其中24个编辑请求、8个零编辑请求。随机方法先在案例内平均种子，再按主体等权汇总；Perform每案例只有一个确定性输出。','',
      '## 编辑请求总体结果','','主终点是目标AU控制误差。其余指标一起保留，用来观察动作、身份与非目标变化之间的权衡。','',
      '| 方法 | '+' | '.join(label for _,label in CORE)+' | 有效案例/主体（主终点） |','| --- | '+' | '.join(['---:']*len(CORE))+' | ---: |']
    raw=stats['raw_method_statistics']
    for method in p.METHODS:
        row=raw[method];main=row['target_mae_changed__edited']
        lines.append('| '+NAMES[method]+' | '+' | '.join(number(row[m+'__edited']['mean'])for m,_ in CORE)+f' | {main["n_cases"]}/{main["n_subjects"]} |')
    lines+=['','各指标自己的有效案例数、种子数和缺测原因以完整CSV/JSON为准，不将缺测填为0。',
      '','## 五个外部方法的同案例配对结果','','| 对照 | 本方AU误差 | 对照AU误差 | 本方减对照 | 95%区间 | 原始p | Holm校正p | 配对案例/主体 |',
      '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    favorable=[]
    for row in stats['comparisons']:
        ci=row.get('difference_ci95');ci_text='NA'if ci is None else '['+number(ci[0])+', '+number(ci[1])+']'
        lines.append('| '+NAMES[row['reference']]+' | '+' | '.join([number(row.get('paired_method_mean')),number(row.get('paired_reference_mean')),number(row.get('mean_difference')),ci_text,number(row.get('two_sided_sign_flip_p')),number(row.get('holm_adjusted_p')),f'{row["n_cases"]}/{row["n_subjects"]}'])+' |')
        if row.get('mean_difference')is not None and row['mean_difference']<0:
            favorable.append(NAMES[row['reference']]+f'（{number(row["paired_method_mean"])} 对 {number(row["paired_reference_mean"])}）')
    if favorable:lines+=['','按共同有效案例配对，本方目标AU误差均值较低的对照有：'+'、'.join(favorable)+'。是否达到统计标准直接看上表Holm校正结果，不把均值优势写成全面领先。']
    else:lines+=['','这轮主终点未出现本方配对均值更低的对照；表内保留实际结果，不改数或改比较口径。']
    lines+=['','## 零编辑请求','','| 方法 | 非目标AU漂移↓ | 全部12AU误差↓ | 身份相似度↑ |','| --- | ---: | ---: | ---: |']
    for method in p.METHODS:
        lines.append('| '+NAMES[method]+' | '+' | '.join(number(raw[method][m+'__no_edit']['mean'])for m in ['unchanged_source_relative_drift','target_mae_all12','identity_cosine'])+' |')
    failures=[r for r in per_image if failure_reasons(r)]
    lines+=['','## 完整性与展示','','- 逐图表共896行：旧8方法768行、新Swift96行、Perform32行；旧指标直接复用。',
            f'- 无效/缺测条目：{len(failures)}条，全部列在[失败与缺测清单](all_failures.md)，不从分母删除。',
            '- [全32案例索引](all_32_cases.md)包含源/目标、本方与两新方法全部种子图链接；完整指标与逐主体表保留。',
            '- Top3按预定的案例目标AU误差排序，随机方法固定显示seed101；每页标注“误差较大案例（非整体表现）”。',
            '- 原生图标注实际尺寸；四列都有同一OpenFace评价框时才统一放大15%，仅为显示，不重新检测或改变评分。',
            '','## 实验设置说明','','本方使用源图、文本与数值AU；Perform及部分旧对照接收真实目标驱动图，Swift使用文本条件。这里是同请求、同评价器的比较，不是同训练输入排名。Swift保留“官方专用检查点＋固定SD2.1社区辅助归档”来源记录，本轮报告不自动进入论文主表。','',
            '## 诊断图片','']
    for row in links:lines.append('- '+link(row,NAMES[row['baseline']]+' / '+row['case_id']))
    return '\n'.join(lines)+'\n'


def generate(job,output):
    doc,combined,case_input,stats,gallery,old_au,new_au,refs,receipt=load_inputs(job)
    output=Path(output).resolve();p.need(output.is_relative_to(HERE)and not output.exists(),'Append-only report directory required');output.mkdir(parents=True)
    metrics={(r['case_id'],r['method'],r['seed']):r for r in combined['records']}
    images={(r['case_id'],r['method'],r['seed']):r for r in combined['per_image_table']}
    reference_rows={r['case_id']:r for r in refs['cases']}
    geometry={(r['case_id'],r['method'],r['seed']):r for r in old_au['records']}
    geometry.update({(r['case_id'],r['method'],r['seed']):r for r in new_au['records']})
    pages=[]
    for baseline in p.NEW_METHODS:
        for rank,choice in enumerate(gallery['top3_by_method'][baseline]['top3'],1):
            cid=choice['case_id'];cells=[]
            for method in ['source_real','target_real',p.CHOSEN,baseline]:
                seed=0 if method.endswith('_real')else -1 if method=='performrecast'else 101
                if method.endswith('_real'):
                    value=reference_rows[cid][method[:-5]+'_reference'];status=value['status'];reference=value.get('image');error=value.get('error');score=None
                else:
                    row=images[cid,method,None if seed==-1 else seed];status=row['generation_status'];reference=row.get('output');error=row.get('failure_reason');score=row.get('target_mae_changed')
                cells.append({'method':method,'status':status,'image':reference,'error':error,'display_C_MAE':score,
                              'measurement':geometry.get((cid,method,seed))})
            pages.append(render_case(baseline,cid,choice['subject'],rank,choice['case_target_C_MAE'],cells,
                                     output/f'{baseline}_Top{rank}_{cid}.png'))
    meta=doc['case_metadata'];case_lines=['# 全32案例索引','','按冻结清单顺序；随机种子全部保留，Perform没有伪造三种子。','',
       '| 案例 | 组别 | 源图 | 目标图 | 本方101/202/303 | Swift101/202/303 | Perform确定性输出 |',
       '| --- | --- | --- | --- | --- | --- | --- |']
    def image_links(cid,method):
        result=[]
        for seed in ([None]if method=='performrecast'else p.SEEDS):
            r=images[cid,method,seed];label='无seed'if seed is None else str(seed)
            result.append(link(r.get('output'),label)if r['generation_status']=='complete'else label+':'+r['generation_status'])
        return ' '.join(result)
    for cid,m in meta.items():
        rr=reference_rows[cid];case_lines.append('| '+cid+' | '+('零编辑'if m['group']=='no_edit'else '编辑')+' | '+
            ' | '.join([link(rr['source_reference'].get('image')),link(rr['target_reference'].get('image')),image_links(cid,p.CHOSEN),image_links(cid,'swiftedit'),image_links(cid,'performrecast')])+' |')
    failure_rows=[{**r,'report_reasons':failure_reasons(r)}for r in combined['per_image_table']if failure_reasons(r)]
    failure_lines=['# 全部失败与缺测','','包含生成/输入/评价/附加指标缺测，不能把技术未开始等同于模型质量失败；零编辑的C不适用不计为失败。','',
                   '| 方法 | 案例 | seed | 状态/原因 | 原生图 |','| --- | --- | --- | --- | --- |']
    for r in failure_rows:failure_lines.append('| '+NAMES[r['method']]+' | '+r['case_id']+' | '+('不适用'if r['seed']is None else str(r['seed']))+' | '+'；'.join(r['report_reasons']).replace('|','/')+' | '+link(r.get('output'))+' |')
    if not failure_rows:failure_lines.append('| — | — | — | 本次没有失败或缺测条目 | — |')
    for name,text in [('summary_zh.md',report_text(stats,combined['per_image_table'],meta,pages)),('all_32_cases.md','\n'.join(case_lines)+'\n'),('all_failures.md','\n'.join(failure_lines)+'\n')]:
        with (output/name).open('x')as f:f.write(text)
    p.save(output/'all_failures.json',failure_rows)
    p.save(output/'display_manifest.json',{'kind':'recent_external_chinese_report_v1','status':'rendered_pending_visual_review',
      'statistics_export_receipt':extent.ref(Path(job)/'statistics/export_receipt.json'),'gallery_selection':extent.ref(Path(job)/'statistics/gallery_selection.json'),
      'renderer':extent.ref(__file__),'extent_io':extent.ref(HERE/'extent_io.py'),'reused_helper':{'path':str(OLD_RENDER),'sha256':OLD_RENDER_SHA},'cases':len(meta),
      'per_image_rows':len(combined['per_image_table']),'failures_and_NA_rows':len(failure_rows),'pages':pages,
      'source_images_modified':False,'new_detectors_or_models_called':False,'metrics_recomputed':False,'visual_review_performed':False,
      'native_plus_face_zoom':'face zoom only if allfour have same frozen OpenFace geometry, exactimageSHA; square1.15maxside clipped',
      'all_output_contact_sheets':'root-owned NEW/visual_review.py remains the full128-output6-perpage overview; not duplicated here'})
    return {'report':str(output/'summary_zh.md'),'pages':len(pages),'full_cases':len(meta),'failure_rows':len(failure_rows)}


def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--output',type=Path)
    a.add_argument('--execute',action='store_true');args=a.parse_args()
    if not args.execute:print(json.dumps({'status':'report_generator_prepared','needs_completed_statistics':True,'models_called':False}));return
    print(json.dumps(generate(args.job,args.output),ensure_ascii=False))


if __name__=='__main__':main()
