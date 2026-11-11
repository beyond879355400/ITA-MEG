import copy,unittest
import run_formal as formal
from run_formal_all import normalized
from run_resume import completed_map,pending_cases


def fixture():
    cases=formal.stage_manifest()['cases']
    rows=[normalized(c,s,'synthetic',reason='notrun') for s in formal.SEEDS for c in cases]
    for row in rows[:17]:
        row.update(status='complete',execution_started=True,output={'path':'unchanged_prior_path','sha256':'unchanged_prior_sha'})
        row.pop('failure_reason')
    return cases,{'kind':'recent_baseline_generation_index_v1','method':'swiftedit','case_manifest_sha256':formal.MANIFEST_SHA,'sampling_seeds':formal.SEEDS,'records':rows}


class ResumeTests(unittest.TestCase):
    def test_exact_missing_counts_and_original_completed_values(self):
        cases,index=fixture();before=copy.deepcopy(index);done=completed_map(index,False)
        self.assertEqual([len(pending_cases(cases,s,done)) for s in formal.SEEDS],[15,32,32])
        self.assertEqual(len(done),17);self.assertEqual(index,before)
        self.assertEqual(sum(len(pending_cases(cases,s,done)) for s in formal.SEEDS)+len(done),96)
        self.assertEqual(next(iter(done.values()))['output']['path'],'unchanged_prior_path')

    def test_duplicate_key_is_rejected_not_selected(self):
        _,index=fixture();index['records'].append(copy.deepcopy(index['records'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):completed_map(index,False)

    def test_missing_terminal_key_is_rejected(self):
        _,index=fixture();index['records'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete'):completed_map(index,False)

    def test_noncompleted_rows_are_never_silently_dropped(self):
        cases,index=fixture();index['records'][17]['status']='failed';index['records'][17]['execution_started']=True
        done=completed_map(index,False)
        self.assertIn(cases[17],pending_cases(cases,101,done))


if __name__=='__main__':unittest.main(verbosity=2)
