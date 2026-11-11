"""Versioned thesis counts and preservation inventory; no source modification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
THESIS = HERE.parents[2]
BASELINE = '983190b'
HELPER = THESIS / '实验以及中间产物/去除工程化附录_20260905/audit.py'
spec = importlib.util.spec_from_file_location('existing_thesis_count', HELPER)
counter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counter)
counter.WORK = HERE / 'manuscript_audit_tmp'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def count(text):
    counter.WORK.mkdir(exist_ok=True)
    total, parts = counter.count(text)
    back, _ = counter.count(r'\backmatter' + text.split(r'\backmatter', 1)[1])
    parts[-1]['count'] -= back
    parts.append({'part': '参考文献及致谢', 'count': back})
    assert sum(row['count'] for row in parts) == total
    return {'total': total, 'body_chapters_1_to_6': sum(row['count'] for row in parts[1:7]), 'parts': parts,
            'method': 'Existing project TeXcount -ch -sum -sub=chapter; not CNKI billing characters'}

def blocks(text):
    tables = [m[0] for m in re.finditer(r'\\begin\{(tabular(?:x|\*)?|longtable)\}.*?\\end\{\1\}', text, re.S)]
    equations = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', text, re.S)
    displays = re.findall(r'(?m)^[ \t]*\\\[[ \t]*\n.*?^[ \t]*\\\][ \t]*$', text, re.S)
    images = sorted(set(re.findall(r'figures/[^}\s]+', text)))
    return {'tables': tables, 'equations': equations, 'displays': displays, 'figure_paths': images}

def normalized(value):
    # Original mathematics must remain; equation tags may shift after inserted formulas.
    value = re.sub(r'\\tag\{[^}]+\}', '', value)
    value = re.sub(r'\\ref\{tab:legacy5-(\d+)\}', r'5.\1', value)
    value = value.replace(r'\ref{sec:legacy_text}', '5.4.4')
    return re.sub(r'\s+', '', value)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-only', action='store_true')
    a = p.parse_args()
    old = subprocess.check_output(['git', 'show', BASELINE + ':chapters/body.tex'], cwd=THESIS, text=True)
    new = (THESIS / 'chapters/body.tex').read_text()
    before, after = count(old), count(new)
    inventory, current = blocks(old), blocks(new)
    preserved = {}
    for key in ('tables', 'equations', 'displays'):
        available = [normalized(v) for v in current[key]]
        missing = [i + 1 for i, v in enumerate(inventory[key]) if normalized(v) not in available]
        preserved[key] = {'baseline_count': len(inventory[key]), 'current_count': len(current[key]), 'missing_original_indices': missing}
    preserved['figures'] = {'baseline_paths': inventory['figure_paths'],
                            'missing_original_paths': sorted(set(inventory['figure_paths']) - set(current['figure_paths']))}
    result = {'baseline': BASELINE, 'before': before, 'after': after,
              'body_delta': after['body_chapters_1_to_6'] - before['body_chapters_1_to_6'],
              'total_delta': after['total'] - before['total'], 'preservation': preserved,
              'original_body_sha256': sha(old.encode()), 'current_body_sha256': sha(new.encode()),
              'date_resource_sha256': sha((THESIS / 'assets/frontmatter.pdf').read_bytes())}
    if a.baseline_only:
        assert old == new, 'Formal text was edited before baseline capture'
    out = a.output.resolve()
    assert HERE in out.parents and not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'before': before, 'after': after, 'preservation': preserved, 'report': str(out)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
