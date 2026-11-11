"""Count and compare a released thesis without changing source or PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import manuscript_audit as audit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE='983190b'
BASE_PDF=ROOT/'build/王超悦-硕士学位论文-202609092025.pdf'
BASE_PDF_SHA='498e110c13b45856c267ebd000d32eec41a4b51d196b601f64c2292de9d4c3a4'
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def pdf_pages(path):
    text=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)
    pages=text.split('\f')
    if not pages[-1].strip():pages.pop()
    info=subprocess.check_output(['pdfinfo',str(path)],text=True)
    n=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
    need(n==len(pages),'PDF page extraction count mismatch')
    chapters={};references=None;blank=[]
    for i,page in enumerate(pages,1):
        lines=[line.strip() for line in page.splitlines() if line.strip()]
        if not lines:blank.append(i)
        # Body pages have Arabic footers; front matter uses Roman numerals.
        if not lines or not re.fullmatch(r'\d+',lines[-1]):continue
        for m in re.finditer(r'(?m)^\s*第\s*([1-6])\s*章\s+([^\n]+)',page):
            chapter=int(m[1])
            need(chapter not in chapters,'Repeated chapter heading in body')
            chapters[chapter]={'physical_start':i,'printed_start':int(lines[-1]),'title':m[2].strip()}
        if re.search(r'(?m)^\s*参\s*考\s*文\s*献\s*$',page):references=i
    need(set(chapters)==set(range(1,7)) and references is not None,'Chapter/reference boundaries unresolved')
    for chapter in range(1,7):
        stop=chapters[chapter+1]['physical_start'] if chapter<6 else references
        chapters[chapter]['pages']=stop-chapters[chapter]['physical_start']
    return {'path':str(path),'sha256':sha(path),'pages':n,'a4':'A4' in info,'chapters':chapters,
            'frontmatter_pages':chapters[1]['physical_start']-1,'references_start':references,'empty_text_pages':blank}
def table_rows(block):
    return [audit.normalized(line) for line in block.splitlines() if '&' in line and line.rstrip().endswith(r'\\')]
def bibliography(text):
    back=text.split(r'\ThesisBackTitle{参考文献}',1)[1].split(r'\ThesisBackTitle{致谢}',1)[0]
    blocks={}
    for m in re.finditer(r'(?m)^\[(\d+)\].*?(?=\n\n|\Z)',back,re.S):
        blocks[int(m[1])]=m[0].replace(r'\endgroup','').strip()
    return blocks
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compile-receipt',type=Path)
    a=p.parse_args();pdf=a.pdf.resolve();out=a.output.resolve()
    need(pdf.parent==ROOT/'build' and pdf.is_file(),'Released root build PDF required')
    need(HERE in out.parents and not out.exists(),'Append-only audit output inside completion required')
    need(sha(BASE_PDF)==BASE_PDF_SHA,'Baseline PDF was changed')
    old=subprocess.check_output(['git','show',BASE+':chapters/body.tex'],cwd=ROOT,text=True)
    new=(ROOT/'chapters/body.tex').read_text()
    ob,nb=audit.blocks(old),audit.blocks(new)
    equations={k:[i+1 for i,v in enumerate(ob[k]) if audit.normalized(v) not in {audit.normalized(n) for n in nb[k]}] for k in ('equations','displays')}
    need(not any(equations.values()),'Original equation mathematics missing')
    tables=[]
    for i,block in enumerate(ob['tables'],1):
        if audit.normalized(block) in {audit.normalized(n) for n in nb['tables']}:
            tables.append({'baseline_index':i,'status':'unchanged_content'});continue
        need(i in (1,23,24),'Unexpected original experiment table alteration')
        rows=table_rows(block)
        if i==1:rows=[r.replace('AU级静态微表情图像','AU级静态面部状态图像') for r in rows]
        sets=[set(table_rows(n)) for n in nb['tables']]
        need(any(set(rows)<=s for s in sets),'Original summary table data row missing')
        tables.append({'baseline_index':i,'status':'extended_summary_original_rows_retained',
            'declared_text_change':'static facial-state output label' if i==1 else None})
    images=[]
    for rel in ob['figure_paths']:
        need(rel in nb['figure_paths'],'Original image source removed')
        before=git('rev-parse',BASE+':'+rel);after=git('hash-object',rel)
        need(before==after,'Original image bytes changed: '+rel)
        images.append({'path':rel,'git_blob':after})
    front_same=git('rev-parse',BASE+':assets/frontmatter.pdf')==git('hash-object','assets/frontmatter.pdf')
    need(front_same,'Protected front matter resource changed')
    abstract_same=old.split(r'\ThesisTableOfContents',1)[0]==new.split(r'\ThesisTableOfContents',1)[0]
    need(abstract_same,'Abstract changed without current authorization')
    oldrefs,newrefs=bibliography(old),bibliography(new)
    need(len(oldrefs)==83 and all(newrefs.get(k)==v for k,v in oldrefs.items()),'Original bibliography changed')
    need(set(newrefs)==set(range(1,len(newrefs)+1)),'Bibliography numbering gap')
    before,after=audit.count(old),audit.count(new)
    binding=None
    if a.compile_receipt:
        binding=json.loads(a.compile_receipt.read_text())
        need(binding['status']=='compiled_and_auto_verified_pending_visual' and binding['pdf']['sha256']==sha(pdf),'PDF/compile receipt mismatch')
        for rel,expected in binding['inputs_after'].items():need(sha(ROOT/rel)==expected,'Source differs from compiled release: '+rel)
    result={'status':'PASS' if binding else 'COMPONENT_CHECKS_PASS_UNBOUND_PDF','baseline_commit':BASE,'current_head_before_commit':git('rev-parse','HEAD'),
        'before_counts':before,'after_counts':after,'before_pdf':pdf_pages(BASE_PDF),'after_pdf':pdf_pages(pdf),
        'equation_preservation':equations,'tables':tables,'original_image_hashes':images,
        'protected_frontmatter_unchanged':front_same,'abstracts_unchanged':abstract_same,
        'original_references_unchanged':len(oldrefs),'current_reference_count':len(newrefs),
        'new_reference_numbers':sorted(set(newrefs)-set(oldrefs)),
        'body_sha256':sha(ROOT/'chapters/body.tex'),
        'source_pdf_binding_verified':binding is not None,
        'tracked_changed_files':git('diff','--name-only',BASE).splitlines(),
        'untracked_files':git('ls-files','--others','--exclude-standard').splitlines(),
        'scope':'Source/data preservation and pagination/count checks; visual/experiment verification reported separately'}
    need(result['after_pdf']['a4'] and not result['after_pdf']['empty_text_pages'],'PDF A4/blank-page check failed')
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'body_count':after['body_chapters_1_to_6'],'total_count':after['total'],
        'pages':result['after_pdf']['pages'],'chapters':result['after_pdf']['chapters'],'output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
