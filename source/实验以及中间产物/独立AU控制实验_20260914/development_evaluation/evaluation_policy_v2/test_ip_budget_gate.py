"""Explicit-budget metadata fixtures only; no realIP1.0 image/model invocation."""
import copy
import unittest
import ip_budget_gate as g
from test_ip10_gate import ip_fixture

def fixture(budget):
    r,s,c,o,refs,prompts,records=ip_fixture()
    if budget==200:
        for m in s['methods']:m['name']=m['name'].replace('_step2000_','_step200_')
        for row in records:row['method']=row['method'].replace('_step2000_','_step200_')
    return r,s,c,o,refs,prompts,records

class BudgetGates(unittest.TestCase):
    def test_both_explicit_budgets(self):
        for budget in (200,2000):
            r,s,c,o,refs,prompts,records=fixture(budget);_,ns=g.gate_source(budget);ns['document_gate'](r,s,c,c,o)
            mapping=ns['record_gate'](records,s,c,refs,prompts);self.assertEqual(mapping['baseline'],f'baseline_ip10_b{budget}');self.assertEqual(len(mapping),5)
    def test_missing_or_invalid_budget_rejected(self):
        for budget in (None,True,0,201,1999,200.0,'200'):
            with self.assertRaises(g.GateError):g.valid_budget(budget)
    def test_wrong_budget_model_names_rejected(self):
        r,s,c,o,*_=fixture(200);_,ns=g.gate_source(2000)
        with self.assertRaises(ValueError):ns['document_gate'](r,s,c,c,o)
        r,s,c,o,*_=fixture(2000);_,ns=g.gate_source(200)
        with self.assertRaises(ValueError):ns['document_gate'](r,s,c,c,o)
    def test_different_checkpoint_sha_rejected(self):
        original=[{'variant':v,'checkpoint_sha256':str(i)*64} for i,v in enumerate(g.VARIANTS)]
        for budget in (200,2000):
            g.checkpoint_budget_match(budget,original,copy.deepcopy(original));bad=copy.deepcopy(original);bad[0]['checkpoint_sha256']='f'*64
            with self.assertRaises(g.GateError):g.checkpoint_budget_match(budget,bad,original)
    def test_wrong_reference_budget_rejected_before_files(self):
        with self.assertRaises(g.GateError):g.reference_for_budget(200,'step2000_other_group')
        with self.assertRaises(g.GateError):g.reference_for_budget(2000,'sealed_step200')
    def test200_does_not_use2000_parent_trace_rule(self):
        text,_=g.gate_source(200);self.assertIn('range(1,201)',text);self.assertNotIn('range(201,2001)',text)
        self.assertIn('trace=_trace200(checkpoint)',text);self.assertIn('Not the exact sealed corresponding200 checkpoint',text)
    def test2000_keeps_continuation_rule(self):
        text,_=g.gate_source(2000);self.assertIn('range(201,2001)',text);self.assertIn('Not resumed from sealed corresponding step200',text)
    def test_new_baseline_names_separate_budgets(self):
        self.assertNotEqual(g.labels_for_budget(200)['baseline'],g.labels_for_budget(2000)['baseline']);self.assertNotIn('baseline',g.REFERENCES)
    def test_no_missing_baseline_for_either_budget(self):
        for budget in (200,2000):
            r,s,c,o,refs,prompts,records=fixture(budget);_,ns=g.gate_source(budget);records.pop(0)
            with self.assertRaises(ValueError):ns['record_gate'](records,s,c,refs,prompts)
    def test_wrong_scale_for_either_budget(self):
        for budget in (200,2000):
            r,s,c,o,*_=fixture(budget);s['ip_scale']=.9;_,ns=g.gate_source(budget)
            with self.assertRaises(ValueError):ns['document_gate'](r,s,c,c,o)
    def test_original_sources_unchanged(self):
        self.assertEqual(g.sha(g._ip.__file__),g.IP_GATE_SHA);self.assertEqual(g.sha(g._old.__file__),g._ip.CORE_SHA)

if __name__=='__main__':unittest.main()
