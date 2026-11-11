"""Static tests; no future image directory, image decode, evaluator or model."""
import ast
import sys
import unittest

import ip10_cpu_eval as entry


class CpuEntryTests(unittest.TestCase):
    def test_computations_unchanged_after_reversing_declared_adapter(self):
        derived=entry.derived_source()
        # Reverse only the reviewed input/count/namespace changes.
        class Reverse(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='add_argument' and node.value.args and isinstance(node.value.args[0],ast.Constant) and node.value.args[0].value=='--reference-group':return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id in ('validate','prepare') and len(node.args)>1 and (ast.unparse(node.args[-1]) in ("doc['reference_ip09_group']",'a.reference_group')):node.args.pop()
                return node
            def visit_Constant(self,node):
                if type(node.value) is int:
                    if node.value==95:node.value=76
                    elif node.value==114:node.value=95
                elif node.value=='ip10_[a-z0-9_]+':node.value='step2000_[a-z0-9_]+'
                elif node.value=='len(items)==114':node.value='len(items)==95'
                return node
        text=ast.unparse(Reverse().visit(ast.parse(derived)))
        text=text.replace('from ip10_gate import','from step2000_gate import')
        text=text.replace('and114 logical records (source19 plus fresh IP1 baseline19 and four models76)','and95 logical records')
        text=text.replace('Expected95 fresh IP1 output measurements including baseline','Expected76 new output measurements')
        # The sole generated-input-group expression no longer prepends old baseline.
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                for i,key in enumerate(node.keys):
                    if isinstance(key,ast.Constant) and key.value=='source_plus_text_manual_AU':
                        node.values[i]=ast.BinOp(left=ast.List(elts=[ast.Constant('baseline')],ctx=ast.Load()),op=ast.Add(),right=node.values[i])
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(entry.CORE.read_text()),include_attributes=False))

    def test_new_counts_and_baseline_measurement_not_reuse(self):
        source=entry.derived_source();tree=ast.parse(source)
        funcs={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
        openface=ast.unparse(funcs['openface']);merge=ast.unparse(funcs['merge']);identity=ast.unparse(funcs['derived_identity_source'])
        self.assertIn('len(items) == 95',openface)
        self.assertIn('len(au[\'records\']) == 95',merge)
        self.assertIn("len(bf['records']) == 114",merge)
        self.assertIn("node.value = 114",identity)
        self.assertIn('len(items)==114',identity)
        self.assertNotIn("['baseline'] + list(LABELS.values())",merge)
        self.assertIn('reference_ip09_group',source)
        self.assertIn('--reference-group',source)

    def test_import_does_not_load_model_or_image_libraries(self):
        self.assertFalse(any(name in sys.modules for name in ('torch','cv2','PIL','onnxruntime')))


if __name__=='__main__':unittest.main(verbosity=2)
