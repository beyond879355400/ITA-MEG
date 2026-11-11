#!/usr/bin/env python3
"""Two-method50-step CPU check; no new general experiment framework."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent;CORE=HERE/'step2000_cpu_eval.py'
CORE_SHA='be53429a850d70a1ee09428ede5c0af7ff134b4695326cd608c64a5ccd275035'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def derived_source():
    if sha(CORE)!=CORE_SHA:raise ValueError('Frozen CPU core changed')
    source=CORE.read_text()
    replacements={
      'from step2000_gate import':'from steps50_gate import',
      "['baseline']+list(LABELS.values())":"list(LABELS.values())",
      ",'source_plus_manual_AU_native':['magicface'],'source_plus_real_driver':['liveportrait','faceadapter','liveportrait_absolute']":"",
      'and95 logical records':'and57 logical records (source19 plus fresh50-step baseline19 and candidate19)',
      'Expected76 new output measurements':'Expected38 fresh50-step baseline+candidate measurements'}
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Ambiguous steps50 CPU binding: '+old)
        source=source.replace(old,new)
    class Adapt(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int:
                if node.value==76:node.value=38
                elif node.value==95:node.value=57
                elif node.value==209:node.value=76
                elif node.value==171:node.value=38
            elif node.value=='step2000_[a-z0-9_]+':node.value='steps50_[a-z0-9_]+'
            elif node.value=='len(items)==95':node.value='len(items)==57'
            return node
    return ast.unparse(ast.fix_missing_locations(Adapt().visit(ast.parse(source))))+'\n'


def main():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--group');args,_=pre.parse_known_args()
    source=derived_source();scope={'__name__':'steps50_fixed_cpu','__file__':str(Path(__file__).resolve())}
    exec(compile(source,str(CORE),'exec'),scope);scope['main']()
    if args.group:
        if re.fullmatch('steps50_[a-z0-9_]+',args.group) is None:raise ValueError('Invalid steps50 group')
        output=HERE/args.group
        if output.exists():
            root=output/'cpu_entry_bindings';root.mkdir(exist_ok=True)
            for name,content in [('core_snapshot.py',CORE.read_bytes()),('derived_snapshot.py',source.encode()),('entry_snapshot.py',Path(__file__).read_bytes())]:
                p=root/name
                if p.exists() and p.read_bytes()!=content:raise ValueError('CPU phase code changed')
                if not p.exists():p.write_bytes(content)
            with (root/(sys.argv[1]+'.json')).open('x') as f:json.dump({'core_sha256':CORE_SHA,'entry_sha256':sha(__file__),
               'gate_sha256':sha(HERE/'steps50_gate.py'),'derived_sha256':sha(root/'derived_snapshot.py'),
               'changes':'only strict38-image steps50 gate, source19+new38 count57, two-method aggregation, output namespace',
               'metric_algorithms_changed':False,'test_used':False},f,indent=2)


if __name__=='__main__':main()
