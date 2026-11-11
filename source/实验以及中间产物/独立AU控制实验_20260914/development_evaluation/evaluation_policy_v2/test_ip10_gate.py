"""Pure gate fixtures; no generated image/model/final-test evaluation."""
import copy
import unittest
import ip10_gate as g
import step2000_gate as old
from test_step2000_gate import fixture

def ip_fixture():
    result,settings,cases,reference,sources,prompts,old_records=fixture()
    settings.update(ip_scale=1.0,skip_baseline=False,setting_status='development sensitivity only; not the final parameter choice')
    result.update(methods=5,images=95)
    records=[]
    for case in cases['cases']:
        cid=case['case_id'];base=copy.deepcopy(next(r for r in old_records if r['case_id']==cid));base.update(method='baseline',ip_scale=1.0)
        records.append(base)
        for row in old_records:
            if row['case_id']==cid:records.append({**row,'ip_scale':1.0})
    return result,settings,cases,reference,sources,prompts,records

class IP10Gates(unittest.TestCase):
    def test_complete95_schema(self):
        r,s,c,o,refs,prompts,records=ip_fixture();g.document_gate(r,s,c,c,o)
        mapping=g.record_gate(records,s,c,refs,prompts);self.assertEqual(len(mapping),5);self.assertEqual(mapping['baseline'],'baseline_ip10')
        self.assertEqual(len(g.LABELS),5);self.assertNotIn('baseline',g.REFERENCES)
    def test_old_gate_not_modified(self):
        self.assertEqual(old.sha(old.__file__),g.CORE_SHA);self.assertEqual(old.GENERATOR_SHA,'fbba652deaf7f1dbdb3972799b00a010e5782eaedddab40f85d4154941e9a83f')
    def test_ip09_rejected(self):
        r,s,c,o,*_=ip_fixture();s['ip_scale']=.9
        with self.assertRaises(g.GateError):g.document_gate(r,s,c,c,o)
    def test_skip_baseline_rejected(self):
        r,s,c,o,*_=ip_fixture();s['skip_baseline']=True;r.update(methods=4,images=76)
        with self.assertRaises(g.GateError):g.document_gate(r,s,c,c,o)
    def test_missing_baseline_record_rejected(self):
        r,s,c,o,refs,prompts,records=ip_fixture();records.pop(0)
        with self.assertRaises(g.GateError):g.record_gate(records,s,c,refs,prompts)
    def test_row_scale_must_be_explicit(self):
        r,s,c,o,refs,prompts,records=ip_fixture();records[2].pop('ip_scale')
        with self.assertRaises(g.GateError):g.record_gate(records,s,c,refs,prompts)
    def test_mixed_scale_rejected(self):
        r,s,c,o,refs,prompts,records=ip_fixture();records[3]['ip_scale']=.9
        with self.assertRaises(g.GateError):g.record_gate(records,s,c,refs,prompts)
    def test_boolean_scale_rejected(self):
        r,s,c,o,refs,prompts,records=ip_fixture();records[1]['ip_scale']=True
        with self.assertRaises(g.GateError):g.record_gate(records,s,c,refs,prompts)
    def test_other_sampling_change_rejected(self):
        r,s,c,o,*_=ip_fixture();s['guidance_scale']=6
        with self.assertRaises(g.GateError):g.document_gate(r,s,c,c,o)
    def test_exact_checkpoint_sets_required(self):
        row={'checkpoints':[{'variant':v,'checkpoint_sha256':str(n)*64} for n,v in enumerate(g.VARIANTS)]};g.checkpoint_match(row,copy.deepcopy(row))
        changed=copy.deepcopy(row);changed['checkpoints'][0]['checkpoint_sha256']='a'*64
        with self.assertRaises(g.GateError):g.checkpoint_match(changed,row)
    def test_reference_group_not_guessed(self):
        with self.assertRaises(g.GateError):g.validate('SYNTHETIC_NOT_A_RUN','ip10_not_an_ip09_group')
        with self.assertRaises(g.GateError):g.validate('SYNTHETIC_NOT_A_RUN',None)
    def test_baseline_metadata_whitelist(self):
        _,_,_,_,refs,_,_=ip_fixture();source=copy.deepcopy(next(iter(refs.values())));source.update(vector=[987]*12,identity_cosine=1)
        item={**g.input_metadata(source),'method':'baseline_ip10','original_method':'baseline'};g.check_input_metadata(item);self.assertNotIn('vector',item)

if __name__=='__main__':unittest.main()
