"""Pure selected-candidate steps50 metadata fixtures, no inference/images."""
import copy
import unittest
import steps50_gate as g
from test_ip_budget_gate import fixture

def data():
    r,s,c,o,refs,prompts,records=fixture(200);s['methods']=[m for m in s['methods'] if m['effective_config']['variant']=='bidirectional']
    s.update(ip_scale=.9,steps=50);r.update(methods=2,images=38)
    wanted={'baseline',s['methods'][0]['name']};records=[{**x,'ip_scale':.9} for x in records if x['method'] in wanted]
    return r,s,c,o,refs,prompts,records

class Steps50Tests(unittest.TestCase):
    def test_only_candidate_and_fresh_baseline(self):
        r,s,c,o,refs,prompts,records=data();_,n=g.derive_source();n['document_gate'](r,s,c,c,o);m=n['record_gate'](records,s,c,refs,prompts)
        self.assertEqual(len(m),2);self.assertEqual(m['baseline'],'baseline_steps50');self.assertEqual(n['REFERENCES'],())
    def test_wrong_step_rejected(self):
        r,s,c,o,*_=data();s['steps']=20;_,n=g.derive_source()
        with self.assertRaises(ValueError):n['document_gate'](r,s,c,c,o)
    def test_wrong_ip_rejected(self):
        r,s,c,o,*_=data();s['ip_scale']=1.;_,n=g.derive_source()
        with self.assertRaises(ValueError):n['document_gate'](r,s,c,c,o)
    def test_other_variant_rejected(self):
        r,s,c,o,*_=data();s['methods'][0]['effective_config']['variant']='mlp';_,n=g.derive_source()
        with self.assertRaises(ValueError):n['document_gate'](r,s,c,c,o)
    def test_missing_baseline_rejected(self):
        r,s,c,o,refs,prompts,records=data();records.pop(0);_,n=g.derive_source()
        with self.assertRaises(ValueError):n['record_gate'](records,s,c,refs,prompts)
    def test_scheduler_only_step_can_change(self):
        old={'steps':20,'config':{'a':1,'_use_default_values':['b','a']}};new=copy.deepcopy(old);new['steps']=50;g.scheduler50(new,old)
        new['config']['a']=2
        with self.assertRaises(ValueError):g.scheduler50(new,old)
    def test_fresh_counts_in_derived_gate(self):
        source,n=g.derive_source();self.assertIn("'generated_count':38",source);self.assertIn("'logical_count':57",source);self.assertIn('len(methods)==1',source)

if __name__=='__main__':unittest.main()
