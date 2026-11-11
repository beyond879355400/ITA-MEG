"""CPU-only development-manifest positive/negative gates; no model loading."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import generate_centered_development as generation


class ManifestGateTests(unittest.TestCase):
    def setUp(self):
        parent=generation.ROOT/'development_diagnostics/generation_manifest_gate'
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='test_',dir=parent)
        self.root=Path(self.temporary.name)
        def case(number):
            return {'case_id':'dev'+str(number),'split':'dev','subject':'S'+str(number%4),'group':'no_edit' if number==0 else 'edited',
                    'source_au_vector_0_5':[0]*12,'target_au_vector_0_5':[number%3]+[0]*11}
        self.cases=[case(i) for i in range(14)]
        self.samples={c['case_id']:{'split':'dev','subject':c['subject'],'source_aus':c['source_au_vector_0_5'],'target_aus':c['target_au_vector_0_5']} for c in self.cases}
        self.parent={'status':'frozen_for_development','test_data_used':False,'cache_sha256':'cache','subject_split_sha256':'split','sampling_seeds':[101],'selection':'fixed original','cases':self.cases[:12]}
        self.parent_path=self.root/'parent.json';self.write(self.parent_path,self.parent)
        self.parent_sha=self.sha(self.parent_path)
        self.extension={**copy.deepcopy(self.parent),'status':'frozen_additional_development','source_manifest_path':str(self.parent_path),'source_manifest_sha256':self.parent_sha,'cases':copy.deepcopy(self.cases)}
        self.bindings={'cache':'cache','split':'split','development_subset':{'manifest_sha256':self.parent_sha},'dev_case_ids':[c['case_id'] for c in self.cases[:12]],'all_dev_case_ids':[c['case_id'] for c in self.cases]}

    def tearDown(self):self.temporary.cleanup()
    def write(self,path,doc):path.write_text(json.dumps(doc,sort_keys=True))
    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def validate(self,doc,bindings=True):
        path=self.root/'input.json';self.write(path,doc)
        return generation.validate_development_manifest(path,doc,self.samples,'cache','split',self.bindings if bindings else None)

    def test_exact_checkpoint_and_legacy_baseline_only(self):
        # Identical serialization guarantees this copy has the pinned SHA.
        for with_checkpoint in (True,False):
            result=self.validate(copy.deepcopy(self.parent),with_checkpoint)
            self.assertEqual(result['case_count'],12);self.assertIsNone(result['parent_manifest_sha256'])

    def test_explicit_superset_keeps_prefix_and_reports_chain(self):
        result=self.validate(self.extension)
        self.assertEqual(result['parent_manifest_sha256'],self.parent_sha)
        self.assertEqual(result['original_case_count'],12)
        self.assertEqual(result['additional_case_ids'],['dev12','dev13'])

    def test_superset_baseline_only_needs_checkpoint(self):
        with self.assertRaisesRegex(ValueError,'protocol checkpoint'):
            self.validate(self.extension,False)

    def test_parent_sha_or_actual_parent_changed_rejected(self):
        doc=copy.deepcopy(self.extension);doc['source_manifest_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'Parent manifest SHA'):self.validate(doc)
        changed=copy.deepcopy(self.parent);changed['selection']='changed'
        self.write(self.parent_path,changed)
        with self.assertRaisesRegex(ValueError,'Parent manifest SHA'):self.validate(self.extension)

    def test_prefix_field_order_missing_and_duplicate_rejected(self):
        changes=[]
        doc=copy.deepcopy(self.extension);doc['cases'][0]['group']='changed';changes.append(doc)
        doc=copy.deepcopy(self.extension);doc['cases'][0],doc['cases'][1]=doc['cases'][1],doc['cases'][0];changes.append(doc)
        doc=copy.deepcopy(self.extension);doc['cases'].pop(0);changes.append(doc)
        doc=copy.deepcopy(self.extension);doc['cases'].append(copy.deepcopy(doc['cases'][0]));changes.append(doc)
        for doc in changes:
            with self.subTest(doc=doc),self.assertRaises(ValueError):self.validate(doc)

    def test_changed_or_duplicate_seeds_rejected(self):
        for seeds in ([102],[101,102],[101,101]):
            doc=copy.deepcopy(self.extension);doc['sampling_seeds']=seeds
            with self.subTest(seeds=seeds),self.assertRaises(ValueError):self.validate(doc)
        changed=copy.deepcopy(self.parent);changed['sampling_seeds']=[102]
        with self.assertRaisesRegex(ValueError,'differs from checkpoint'):self.validate(changed)

    def test_extra_labels_source_target_and_test_rejected(self):
        for key in ('source_au_vector_0_5','target_au_vector_0_5'):
            doc=copy.deepcopy(self.extension);doc['cases'][-1][key][0]+=1
            with self.assertRaisesRegex(ValueError,'AU labels changed'):self.validate(doc)
        doc=copy.deepcopy(self.extension);doc['cases'][-1]['split']='test'
        with self.assertRaises(ValueError):self.validate(doc)
        doc=copy.deepcopy(self.extension);doc['test_data_used']=True
        with self.assertRaises(ValueError):self.validate(doc)
        self.samples['dev13']['split']='test'
        with self.assertRaises(ValueError):self.validate(self.extension)

    def test_unbound_extra_and_unapproved_status_rejected(self):
        self.bindings['all_dev_case_ids'].remove('dev13')
        with self.assertRaisesRegex(ValueError,'outside checkpoint'):self.validate(self.extension)
        self.bindings['all_dev_case_ids'].append('dev13')
        doc=copy.deepcopy(self.extension);doc['status']='draft'
        with self.assertRaisesRegex(ValueError,'Unsupported'):self.validate(doc)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManifestGateTests))
    if args.report:
        with args.report.open('x') as handle:json.dump({'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'passed':result.wasSuccessful(),
            'device':'cpu','model_loaded':False,'generator_sha256':hashlib.sha256(Path(generation.__file__).read_bytes()).hexdigest()},handle,indent=2)
    raise SystemExit(not result.wasSuccessful())
