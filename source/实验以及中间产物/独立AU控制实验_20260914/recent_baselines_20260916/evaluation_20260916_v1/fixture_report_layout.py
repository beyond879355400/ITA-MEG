"""One tiny synthetic layout only. Never uses/creates experimental results."""
from pathlib import Path
import json
from PIL import Image,ImageDraw
import render_report as r
import extent_io as e

out=r.HERE/'synthetic_report_layout_v1';out.mkdir(exist_ok=False)
cells=[]
for i,(method,size)in enumerate(zip(['source_real','target_real',r.p.CHOSEN,'swiftedit'],[(320,240),(256,256),(512,512),(512,512)])):
    image=Image.new('RGB',size,['#ddeaf5','#eee2d2','#dcece1','#eddfeb'][i]);draw=ImageDraw.Draw(image)
    box=[size[0]*.30,size[1]*.20,size[0]*.70,size[1]*.70]
    if method=='swiftedit':box=[size[0]*.4,size[1]*.12,size[0]*.6,size[1]*.32]
    draw.rectangle(box,fill=['#73889a','#a39179','#759887','#a480a0'][i]);draw.text((10,10),'SYNTHETIC LAYOUT ONLY',fill='black')
    p=out/(method+'.png');image.save(p);ref=e.ref(p)
    cells.append({'method':method,'status':'complete','image':ref,'display_C_MAE':.1234 if not method.endswith('_real')else None,
                  'measurement':{'sha256':ref['sha256'],'policy_sha256':r.p.POLICY_SHA,'image_size':list(size),'selected_row':0,
                                 'all_candidates':[{'row_index':0,'bbox68':box}]}})
result=r.render_case('swiftedit','SYNTHETIC_CASE_NOT_EXPERIMENT','TEST',1,.4567,cells,out/'SYNTHETIC_LAYOUT.png',fixture=True)
with (out/'fixture_manifest.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'synthetic_only':True,'path':str(out/'SYNTHETIC_LAYOUT.png'),'models_called':False}))
