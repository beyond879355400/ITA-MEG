from pathlib import Path
from PIL import Image
import hashlib,json,re
Q=Path(__file__).resolve().parent;ROOT=Q.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((Q/'release_v3/receipt.json').read_text());idx=json.loads((Q/'visual_v3/index.json').read_text())
assert r['status']=='compiled_and_auto_verified_pending_visual' and r['prior_pdf_stats_unchanged']
assert all(sha(ROOT/p)==v for p,v in r['inputs_after'].items())
assert idx['sha256']==r['pdf']['sha256'] and not any(p['out_of_page'] for p in idx['metadata'])
def digest(p):
    with Image.open(p) as im:return hashlib.sha256(im.convert('RGB').crop((0,0,im.width,im.height-110)).tobytes()).hexdigest()
old=ROOT/'实验以及中间产物/题目聚焦与逐段审查_20260917/visual_v3/pages'
previous={digest(p):int(p.stem.split('-')[-1]) for p in old.glob('*.png')};same=[];changed=[]
for p in sorted((Q/'visual_v3/pages').glob('*.png')):
    number=int(p.stem.split('-')[-1]);match=previous.get(digest(p))
    if match is None:changed.append(number)
    else:same.append({'new':number,'reviewed':match})
figs=[]
for p in idx['metadata']:
    t=re.sub(r'\s+','',p['text'])
    if p['physical']>17 and '同案例下ITA-MEG与五种外部方法的定性展示' in t:
        assert all(m in p['text'] for m in ['MagicFace','Face-Adapter','LivePortrait','SwiftEdit','PerformRecast'])
        assert '占位' not in t
        figs.append({'physical':p['physical'],'printed':p['physical']-17})
assert len(figs)==2
data={'status':'SOURCE_PDF_PASS_PENDING_VISUAL','pdf':r['pdf'],'pages':idx['pages'],'changed_pages':changed,
 'unchanged_page_bodies':len(same),'matches':same,'comparison_figure_pages':figs,
 'five_baselines_visible_on_both_figure_pages':True,'source_inputs_bound':True,'old_archives_unchanged':True,'out_of_page':0}
with (Q/'pdf_qa_delivery.json').open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in data.items() if k!='matches'},ensure_ascii=False,indent=2))
