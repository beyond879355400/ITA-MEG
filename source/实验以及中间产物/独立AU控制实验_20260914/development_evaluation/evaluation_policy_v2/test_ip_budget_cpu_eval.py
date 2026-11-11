"""Static adapter checks only; no actual image/model/CPU evaluation."""
import ast
import sys
import unittest
import ip_budget_cpu_eval as entry


class BudgetCpuTests(unittest.TestCase):
    def test_undeclared_budget_rejected_before_gate_import(self):
        for budget in (None,True,199,201,200.0):
            with self.assertRaises(ValueError):entry.bind_namespace(budget)

    def test_cpu_computations_match_old_core_after_binding_reversal(self):
        tree=ast.parse(entry.derived_source())
        class Reverse(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) is int:
                    if node.value==95:node.value=76
                    elif node.value==114:node.value=95
                elif node.value=='ipbudget_[a-z0-9_]+':node.value='step2000_[a-z0-9_]+'
                elif node.value=='len(items)==114':node.value='len(items)==95'
                elif node.value=='Expected95 fresh IP1 baseline+model measurements':node.value='Expected76 new output measurements'
                elif node.value=='only input directory, frozen manifest SHA and114 logical records (source19 plus fresh baseline19 and four models76); face-selection/model/preprocessing unchanged':node.value='only input directory, frozen manifest SHA and95 logical records; face-selection/model/preprocessing unchanged'
                return node
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='validate' and len(node.args)==2 and ast.unparse(node.args[-1])=="doc.get('reference_ip09_group')":node.args.pop()
                return node
            def visit_ImportFrom(self,node):
                if node.module=='ip_budget_gate':return ast.parse('from step2000_gate import HERE,ROOT,POLICY_SHA,LABELS,REFERENCES,require,read,sha,save,validate,prepare,check_input_metadata').body[0]
                return node
            def visit_Dict(self,node):
                self.generic_visit(node)
                for i,key in enumerate(node.keys):
                    if isinstance(key,ast.Constant) and key.value=='source_plus_text_manual_AU':node.values[i]=ast.BinOp(ast.List([ast.Constant('baseline')],ast.Load()),ast.Add(),node.values[i])
                return node
        restored=Reverse().visit(tree)
        self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(ast.parse(entry.CORE.read_text()),include_attributes=False))

    def test_new_counts_are_95_and114(self):
        source=entry.derived_source();self.assertIn('len(items) == 95',source);self.assertIn("len(bf['records']) == 114",source)
        self.assertIn('node.value = 114',source);self.assertIn('len(items)==114',source)

    def test_no_image_model_library_on_import(self):
        self.assertFalse(any(n in sys.modules for n in ('torch','cv2','PIL','onnxruntime')))


if __name__=='__main__':unittest.main(verbosity=2)
