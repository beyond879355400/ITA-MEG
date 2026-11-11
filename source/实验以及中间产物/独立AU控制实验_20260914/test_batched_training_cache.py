"""CPU-only cache reuse and batch-equivalence gate checks; no pretrained models."""
import argparse
import copy
from pathlib import Path
import unittest
import torch
import prepare_batched_training_cache as cache


def image_record(sample_id,role,subject,aus):
    return {'sample_id':sample_id,'split':role,'subject':subject,'status':'aligned','raw_path':sample_id+'.png','raw_sha256':'raw-'+sample_id,
            'aligned_path':sample_id+'-aligned.png','aligned_sha256':'aligned-'+sample_id,'input':{'labels':{'au'+str(a):v for a,v in zip(cache.AU_ORDER,aus)},'au_vector_0_5':aus,'au_order':cache.AU_ORDER}}


def pair(case,role,subject,source_id,target_id,value=0):
    source=[0]*12;target=[value]+[0]*11
    return {'case_id':case,'split':role,'subject':subject,'au_order':cache.AU_ORDER,'source':image_record(source_id,role,subject,source),
            'target':image_record(target_id,role,subject,target),'source_au_vector_0_5':source,'target_au_vector_0_5':target,'is_no_edit':value==0}


def sample(p):
    row={key:torch.zeros(shape,dtype=torch.float32) for key,shape in cache.SHAPES.items()}
    row.update(case_id=p['case_id'],split=p['split'],subject=p['subject'],prompt=cache.make_prompt(p['target_au_vector_0_5']))
    row['target_aus']=torch.tensor(p['target_au_vector_0_5'],dtype=torch.float32)
    return row


def metadata(p):
    return {**{k:v for k,v in p.items() if k not in ('source','target')},'prompt':cache.make_prompt(p['target_au_vector_0_5']),'aligned_source':p['source'],'aligned_target':p['target']}


class BatchedCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(20260914)
        self.old_pairs=[pair('a','train','A','src_A','target_a'),pair('b','train','A','src_A','target_b',1),pair('d','dev','D','src_D','target_d')]
        self.old_samples=[sample(p) for p in self.old_pairs];self.old_metadata=[metadata(p) for p in self.old_pairs]
        self.new_pairs=[pair('new','train','A','src_A','target_new',1),copy.deepcopy(self.old_pairs[2])]

    def build(self):return cache.make_reuse_index(self.new_pairs,self.old_pairs,self.old_metadata,self.old_samples)

    def test_dev_and_source_reuse_exact_without_new_fields(self):
        index=self.build();latent=(torch.ones(4,64,64),torch.full((4,64,64),.1))
        rows,meta=cache.assemble_samples({'pairs':self.new_pairs},index,{'target_new':latent})
        cache.compare_samples(rows[1],self.old_samples[2])
        self.assertEqual(meta[1],self.old_metadata[2])
        self.assertIs(rows[1],self.old_samples[2])
        self.assertEqual(set(rows[1]),set(self.old_samples[2]))
        for field in cache.SOURCE_FIELDS:self.assertTrue(torch.equal(rows[0][field],self.old_samples[0][field]))

    def test_source_metadata_change_rejected_not_recomputed(self):
        self.new_pairs[0]['source']['raw_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'anchor changed'):self.build()

    def test_dev_pair_change_or_drop_rejected(self):
        self.new_pairs[1]['extra_reused']=True
        with self.assertRaisesRegex(ValueError,'Development pair changed'):self.build()
        self.new_pairs=self.new_pairs[:1]
        with self.assertRaisesRegex(ValueError,'coverage changed'):self.build()

    def test_same_source_inconsistent_tensor_rejected(self):
        self.old_samples[1]['base_faceid']=torch.ones(512)
        with self.assertRaisesRegex(ValueError,'inconsistent'):self.build()

    def test_tensor_readback_checks_all_fields(self):
        row=sample(self.old_pairs[0]);changed=copy.deepcopy(row);changed['latent_mean'][0,0,0]=1
        with self.assertRaisesRegex(ValueError,'tensor changed'):cache.compare_samples(row,changed)
        changed=copy.deepcopy(row);changed['prompt']='other'
        with self.assertRaisesRegex(ValueError,'metadata changed'):cache.compare_samples(row,changed)

    def test_batch_gate_fixed_threshold_and_failure(self):
        mean,std=torch.zeros(4,4,64,64),torch.ones(4,4,64,64)
        singles=[(mean[i:i+1].clone(),std[i:i+1].clone()) for i in range(4)]
        self.assertTrue(cache.check_batch_outputs((mean,std),singles)['passed'])
        singles[2][0][0,0,0,0]=1e-4
        self.assertTrue(cache.check_batch_outputs((mean,std),singles)['passed'])
        singles[2][0][0,0,0,0]=3e-4
        report=cache.check_batch_outputs((mean,std),singles)
        self.assertFalse(report['passed']);self.assertEqual(report['threshold_max_abs'],2e-4)
        with self.assertRaisesRegex(ValueError,'cannot be relaxed'):cache.check_batch_outputs((mean,std),singles,threshold=1e-3)
        singles[2][0][0,0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):cache.check_batch_outputs((mean,std),singles)

    def test_subject_and_labels_checked(self):
        groups={'train':{'A'},'dev':{'D'},'test':{'T'}}
        cache.validate_pair(self.old_pairs[0],groups)
        changed=copy.deepcopy(self.old_pairs[0]);changed['target_au_vector_0_5'][0]=2
        with self.assertRaisesRegex(ValueError,'Manual AU'):cache.validate_pair(changed,groups)
        changed=pair('test','test','T','srcT','targetT')
        with self.assertRaisesRegex(ValueError,'frozen train/dev'):cache.validate_pair(changed,groups)

    def test_image_counts_distinguish_records_raw_and_aligned_content(self):
        p=pair('same','train','A','source_id','target_id')
        p['target']['raw_sha256']=p['source']['raw_sha256']
        p['target']['aligned_sha256']=p['source']['aligned_sha256']
        self.assertEqual(cache.image_inventory([p]),{'unique_images':1,'unique_image_records':2,'unique_aligned_images':1})
        p['target']['aligned_sha256']='different-alignment'
        self.assertEqual(cache.image_inventory([p]),{'unique_images':1,'unique_image_records':2,'unique_aligned_images':2})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BatchedCacheTests))
    if args.report:cache.json_new(args.report,{'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'skipped':result.skipped,'passed':result.wasSuccessful(),
        'device':'cpu','pretrained_models_loaded':False,'batch_equivalence_on_real_vae_run':False,'source_sha256':cache.sha256(cache.ROOT/'prepare_batched_training_cache.py')})
    raise SystemExit(not result.wasSuccessful())
