#!/usr/bin/env python3
"""IP1.0 dev-only CPU entry. Reuses frozen metric code, never the IP0.9 baseline."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
CORE=HERE/'step2000_cpu_eval.py'
CORE_SHA='be53429a850d70a1ee09428ede5c0af7ff134b4695326cd608c64a5ccd275035'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_source():
    if sha(CORE)!=CORE_SHA:raise ValueError('Frozen IP0.9 CPU core changed')
    source=CORE.read_text()
    replacements={
      'from step2000_gate import':'from ip10_gate import',
      "['baseline']+list(LABELS.values())":"list(LABELS.values())",
      'and95 logical records':'and114 logical records (source19 plus fresh IP1 baseline19 and four models76)',
      'Expected76 new output measurements':'Expected95 fresh IP1 output measurements including baseline',
      "actual=validate(doc['run']);":"actual=validate(doc['run'],doc['reference_ip09_group']);",
      "p.add_argument('--group');a=p.parse_args()":"p.add_argument('--group');p.add_argument('--reference-group');a=p.parse_args()",
      'doc=validate(a.run);':'doc=validate(a.run,a.reference_group);',
      'print(prepare(a.run,a.group))':'print(prepare(a.run,a.group,a.reference_group))',
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Ambiguous fixed-core adaptation: '+old)
        source=source.replace(old,new)
    class Adapt(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int:
                if node.value==76:node.value=95
                elif node.value==95:node.value=114
            elif node.value=='step2000_[a-z0-9_]+':node.value='ip10_[a-z0-9_]+'
            elif node.value=='len(items)==95':node.value='len(items)==114'
            return node
    tree=ast.fix_missing_locations(Adapt().visit(ast.parse(source)))
    return ast.unparse(tree)+'\n'


def main():
    # Original CLI phases and checks are retained; no implicit run/output selection.
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--group');args,_=pre.parse_known_args()
    source=derived_source()
    namespace={'__name__':'ip10_frozen_cpu_entry','__file__':str(Path(__file__).resolve())}
    exec(compile(source,str(CORE),'exec'),namespace)
    namespace['main']()
    if args.group:
        if re.fullmatch('ip10_[a-z0-9_]+',args.group) is None:raise ValueError('Invalid IP1 group')
        output=HERE/args.group
        if output.exists():
            binding=output/'cpu_entry_bindings';binding.mkdir(exist_ok=True)
            phase=sys.argv[1]
            target=binding/(phase+'.json')
            if target.exists():raise ValueError('Never overwrite an IP1 phase binding')
            core_copy=binding/'ip09_core_snapshot.py';derived_copy=binding/'ip10_derived_snapshot.py'
            if not core_copy.exists():core_copy.write_bytes(CORE.read_bytes())
            if not derived_copy.exists():derived_copy.write_text(source)
            if sha(core_copy)!=CORE_SHA or derived_copy.read_text()!=source:raise ValueError('Phase used a different CPU entry')
            with target.open('x') as f:json.dump({'core_sha256':CORE_SHA,'entry_sha256':sha(__file__),
                'gate_sha256':sha(HERE/'ip10_gate.py'),'derived_sha256':sha(derived_copy),'test_used':False,
                'reference_ip09_group':json.loads((output/'items.json').read_text())['reference_ip09_group'],
                'changes':['new strict IP1.0 gate','ip10 output namespace','fresh baseline included in5 measured methods',
                           '95 generated AU measurements and114 identity inputs','old baseline removed from input-group description'],
                'metric_algorithms_changed':False,'ip09_results_overwritten':False},f,indent=2)


if __name__=='__main__':main()
