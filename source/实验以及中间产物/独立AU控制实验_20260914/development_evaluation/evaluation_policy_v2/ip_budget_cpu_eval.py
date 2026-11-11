#!/usr/bin/env python3
"""Explicit-budget IP1.0 CPU entry; the existing IP2000 and IP0.9 entries stay frozen."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;CORE=HERE/'step2000_cpu_eval.py'
CORE_SHA='be53429a850d70a1ee09428ede5c0af7ff134b4695326cd608c64a5ccd275035'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_source():
    if sha(CORE)!=CORE_SHA:raise ValueError('Frozen CPU computation core changed')
    source=CORE.read_text()
    old_import='from step2000_gate import HERE,ROOT,POLICY_SHA,LABELS,REFERENCES,require,read,sha,save,validate,prepare,check_input_metadata'
    replacements={old_import:'from ip_budget_gate import HERE,ROOT,POLICY_SHA,REFERENCES,require,read,sha,save,check_input_metadata',
      "actual=validate(doc['run']);":"actual=validate(doc['run'],doc.get('reference_ip09_group'));",
      "['baseline']+list(LABELS.values())":"list(LABELS.values())",
      'and95 logical records':'and114 logical records (source19 plus fresh baseline19 and four models76)',
      'Expected76 new output measurements':'Expected95 fresh IP1 baseline+model measurements'}
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Ambiguous fixed-core adaptation: '+old)
        source=source.replace(old,new)
    class Adapt(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int:
                if node.value==76:node.value=95
                elif node.value==95:node.value=114
            elif node.value=='step2000_[a-z0-9_]+':node.value='ipbudget_[a-z0-9_]+'
            elif node.value=='len(items)==95':node.value='len(items)==114'
            return node
    tree=ast.fix_missing_locations(Adapt().visit(ast.parse(source)))
    return ast.unparse(tree)+'\n'


def bind_namespace(budget):
    if type(budget) is not int or budget not in (200,2000):raise ValueError('Explicit supported budget required')
    import ip_budget_gate as gate
    source=derived_source();scope={'__name__':'explicit_budget_cpu_core','__file__':str(Path(__file__).resolve())}
    exec(compile(source,str(CORE),'exec'),scope)
    scope['LABELS']=gate.labels_for_budget(budget)
    def validate(run,reference_group=None):
        doc=gate.validate(run,budget,reference_group)
        if doc.get('expected_budget')!=budget or doc.get('method_labels')!=scope['LABELS']:raise ValueError('Gate returned another budget/label set')
        return doc
    scope['validate']=validate
    return gate,scope,source


def save_binding(group,phase,budget,source):
    if re.fullmatch('ipbudget_[a-z0-9_]+',group) is None:raise ValueError('Invalid budget group')
    output=HERE/group;doc=json.loads((output/'items.json').read_text());binding=output/'cpu_entry_bindings';binding.mkdir(exist_ok=True)
    if doc['expected_budget']!=budget:raise ValueError('Prepared group uses another budget')
    for name,content in [('core_snapshot.py',CORE.read_bytes()),('derived_snapshot.py',source.encode()),('entry_snapshot.py',Path(__file__).read_bytes())]:
        path=binding/name
        if path.exists() and path.read_bytes()!=content:raise ValueError('Phase entry changed')
        if not path.exists():path.write_bytes(content)
    with (binding/(phase+'.json')).open('x') as f:json.dump({'core_sha256':CORE_SHA,'entry_sha256':sha(__file__),'gate_sha256':sha(HERE/'ip_budget_gate.py'),
          'derived_sha256':sha(binding/'derived_snapshot.py'),'expected_budget':budget,'method_labels':doc['method_labels'],
          'test_used':False,'metric_algorithms_changed':False,'old_entries_or_results_modified':False},f,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('validate','prepare','openface','facenet','buffalo','merge'))
    parser.add_argument('--budget',type=int,choices=(200,2000),required=True);parser.add_argument('--run',type=Path)
    parser.add_argument('--group');parser.add_argument('--reference-group');args=parser.parse_args()
    gate,scope,source=bind_namespace(args.budget)
    if args.phase in ('validate','prepare'):
        if args.run is None:raise ValueError('Actual complete run required; no inferred run path')
        if args.phase=='validate':
            doc=scope['validate'](args.run,args.reference_group)
            print(json.dumps({'status':doc['status'],'expected_budget':args.budget,'logical_count':doc['logical_count']}));return
        if args.group is None:raise ValueError('Explicit new group required')
        gate.prepare(args.run,args.group,args.budget,args.reference_group)
    else:
        if args.group is None:raise ValueError('Prepared group required')
        if args.phase=='openface':scope['openface'](args.group)
        elif args.phase in ('facenet','buffalo'):scope['identity'](args.phase,args.group)
        else:scope['merge'](args.group)
    save_binding(args.group,args.phase,args.budget,source)


if __name__=='__main__':main()
