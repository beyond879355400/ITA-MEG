#!/usr/bin/env python3
"""Thin pinned-core adapter for a separately frozen v2 input manifest."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

CORE_SHA = '5d8e1916bb5bcc9afc307980438b5f2f3fb0610f5dde12141f90e85efa7ebc2d'
CORE_INPUT_SHA = 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', required=True)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--logical-count', type=int, required=True)
    args = p.parse_args()
    assert re.fullmatch('[a-z][a-z0-9_]+', args.group)
    assert re.fullmatch('[0-9a-f]{64}', args.input_sha256) and args.logical_count in (38, 57)
    here = Path(__file__).resolve().parent
    core = here / 'run_facenet_single_directions.py'
    source = core.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == CORE_SHA
    old_root = "root = HERE / 'single_directions'"
    old_hash = f"INPUT_SHA = '{CORE_INPUT_SHA}'"
    assert source.count(old_root) == source.count(old_hash) == 1
    source = source.replace(old_root, f"root = HERE / '{args.group}'")
    source = source.replace(old_hash, f"INPUT_SHA = '{args.input_sha256}'")
    class ChangeCount(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 57:
                node.value = args.logical_count
            return node
    tree = ChangeCount().visit(ast.parse(source))
    namespace = {'__name__': 'facenet_extension_pinned_core', '__file__': str(Path(__file__).resolve())}
    exec(compile(ast.fix_missing_locations(tree), str(core), 'exec'), namespace)
    namespace['main']()
    output = here / args.group / 'facenet'
    (output / 'core_evaluator_snapshot.py').write_bytes(core.read_bytes())
    (output / 'derived_evaluator_snapshot.py').write_text(ast.unparse(tree) + '\n')
    with (output / 'extension_binding.json').open('x') as f:
        json.dump({'core_sha256': CORE_SHA, 'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'input_sha256': args.input_sha256, 'logical_count': args.logical_count,
                   'changes': 'only separately frozen input directory, input SHA and logical record count; no model/preprocess/selection change',
                   'group': args.group}, f, indent=2)


if __name__ == '__main__':
    main()
