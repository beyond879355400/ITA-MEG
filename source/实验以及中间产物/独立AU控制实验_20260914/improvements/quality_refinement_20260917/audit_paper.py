from pathlib import Path
import argparse
import hashlib,importlib.util,json,os,re,subprocess
Q=Path(__file__).resolve().parent;EXP=Q.parents[1];ROOT=EXP.parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Q/'paper_audit.json');args=p.parse_args()
os.environ['PATH']='/Library/TeX/texbin:'+os.environ.get('PATH','')
s=importlib.util.spec_from_file_location('counter',EXP/'completion_20260915/manuscript_audit.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c.counter.WORK=Q/'count_tmp'
before=subprocess.check_output(['git','show','75b8b72:chapters/body.tex'],cwd=ROOT,text=True);after=(ROOT/'chapters/body.tex').read_text()
old,new=c.blocks(before),c.blocks(after)
assert old['tables']==new['tables'];assert old['equations']==new['equations'];assert old['displays']==new['displays']
assert before.split('\\mainmatter')[0]==after.split('\\mainmatter')[0]
assert before.split('\\backmatter')[1]==after.split('\\backmatter')[1]
labels=re.findall(r'\\label\{([^}]+)\}',after);assert len(labels)==len(set(labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',after))<=set(labels)
assert '\\appendix' not in after
for path in ('main.tex','assets/frontmatter.pdf'):
    oldbytes=subprocess.check_output(['git','show','75b8b72:'+path],cwd=ROOT);assert oldbytes==(ROOT/path).read_bytes()
for name in ('quality_comparison_expansion.tex','quality_comparison_diagnostics.tex'):
    t=(ROOT/'figures'/name).read_text();assert '占位' not in t and '../' not in t
    assert all(t.count('{'+m+'}')==2 for m in ('MagicFace','Face-Adapter','LivePortrait','SwiftEdit','PerformRecast'))
    assert t.count('CFG=3.5')>=2
counts0,counts1=c.count(before),c.count(after);assert counts1['body_chapters_1_to_6']<30000
result={'status':'PASS','before':counts0,'after':counts1,'tables_unchanged':len(new['tables']),
 'equations_unchanged':len(new['equations']),'display_math_unchanged':len(new['displays']),
 'abstracts_references_dates_unchanged':True,'five_baselines_in_each_case':True,'all_labels_resolve':True,
 'source_sha256':hashlib.sha256(after.encode()).hexdigest()}
with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
