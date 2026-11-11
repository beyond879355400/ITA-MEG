"""Synthetic tiny AVI/CSV fixtures only. Never reads production test pixels."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import extract_final_test_frames as extraction
from prepare_disfa_pilot import labels_for_subject
from plan_final_test_labels import avi_header_only,source_interval


class FinalExtractionTests(unittest.TestCase):
    def setUp(self):
        base=extraction.ROOT/'data_prepare/final_test_extractor_tests';base.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='synthetic_',dir=base)
        self.root=Path(self.temporary.name);raw=self.root/'raw';raw.mkdir();self.video=raw/'synthetic_original.avi'
        command=[shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-filter_threads','1','-f','lavfi','-i',
                 "nullsrc=s=16x16:r=5:d=2,format=rgb24,geq=r='N*20':g='N*10':b='N*5'",'-frames:v','10','-c:v','rawvideo','-pix_fmt','bgr24','-threads','1','-n',str(self.video)]
        subprocess.run(command,check=True,capture_output=True)
        subject='SYN001';label_root=raw/'labels';(label_root/subject).mkdir(parents=True)
        for au in extraction.AUS:
            values={4:1,6:2,8:3} if au==1 else {}
            (label_root/subject/f'{subject}_au{au}.txt').write_text(''.join(f'{frame},{values.get(frame,0)}\n' for frame in range(1,11)))
        labels,_,records=labels_for_subject(label_root,subject);header=avi_header_only(self.video);header.update(subject=subject,camera='right')
        interval=source_interval(labels,subject);source=interval['source_frame_1based'];cases=[]
        for frame in (source,4,6,8):
            cases.append({'case_id':'synthetic_case_'+str(frame),'subject':subject,'split':'test','camera':'right','video_path':str(self.video),
                'video_metadata_sha256':header['video_metadata_sha256'],'kind':'source_noedit' if frame==source else 'nonzero_target',
                'source_frame_1based':source,'source_frame_zero_based':source-1,'target_frame_1based':frame,'target_frame_zero_based':frame-1,
                'source_au_vector_0_5':[0]*12,'target_au_vector_0_5':[labels[frame]['au'+str(a)] for a in extraction.AUS]})
        self.plan={'test_subjects':[subject],'cases':cases,'sampling_seeds_proposed':[101,202,303],'source_intervals':{subject:interval},'fixture':True}
        self.plan_path=self.write('plan.json',self.plan)
        inv=self.write('inventory.json',{'label_records':records,'video_headers':[header]})
        split=self.write('split.json',{'groups':{'test':[subject]},'test_locked':True})
        self.contract=extraction.Contract(self.root,label_root,raw,(subject,),4,extraction.digest(self.plan_path),extraction.digest(inv),extraction.digest(split))
        accepted=self.write('acceptance.json',{'status':'label_only_plan_selected','plan_sha256':self.contract.plan_sha256,'case_ids':[c['case_id'] for c in cases]})
        sampling=self.write('sampling.json',{'status':'frozen_for_final_test','sampling_seeds':[101,202,303],
            'method_settings':{'fixture_A':{'mode':'synthetic_not_executed'},'fixture_B':{'mode':'synthetic_not_executed'}},'our_generator_uses_target_pixels':False})
        policy=self.write('evaluation.json',{'status':'frozen_for_final_test','methods':['fixture_A','fixture_B'],'metrics':['synthetic_metric'],
            'primary_family':'synthetic_family','comparisons':[{'family':'synthetic_family'}]})
        self.artifact=self.root/'inert_fixture_checkpoint.pt';self.artifact.write_bytes(b'not-a-neural-model; hash-only fixture')
        def ref(path):return {'path':str(path),'sha256':extraction.digest(path)}
        self.protocol={'kind':'final_test_extraction_protocol','status':'frozen_final_test','extraction_authorized':True,
            'extractor_sha256':extraction.digest(Path(extraction.__file__)),'legacy_helper_sha256':extraction.digest(extraction.HERE/'prepare_disfa_pilot.py'),
            'inputs':{'label_plan':ref(self.plan_path),'label_inventory':ref(inv),'subject_split':ref(split),'label_acceptance':ref(accepted),
                      'sampling_settings':ref(sampling),'evaluation_policy':ref(policy)},'case_ids':[c['case_id'] for c in cases],
            'sampling_seeds':[101,202,303],'case_seed_policy':'full_cartesian',
            'models':[{'method':'fixture_A','artifacts':[ref(self.artifact)]},{'method':'fixture_B','artifacts':[ref(self.artifact)]}],
            'videos':[{'subject':subject,'camera':'right',**ref(self.video),'video_metadata_sha256':header['video_metadata_sha256']}],
            'fixture':True}
        self.protocol_path=self.write('protocol.json',self.protocol)

    def tearDown(self):self.temporary.cleanup()
    def write(self,name,value):path=self.root/name;path.write_text(json.dumps(value));return path

    def test_missing_or_unfrozen_protocol_blocks_before_extraction(self):
        with self.assertRaisesRegex(ValueError,'does not exist'):
            extraction.execute(self.root/'missing.json',contract=self.contract)
        self.protocol['status']='not_frozen';self.write('protocol.json',self.protocol)
        with self.assertRaisesRegex(ValueError,'not frozen'):
            extraction.execute(self.protocol_path,contract=self.contract)
        self.assertFalse((self.root/'data_prepare/final_test_extractions').exists())

    def test_complete_sha_and_case_seed_gates(self):
        self.artifact.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):extraction.preflight(self.protocol_path,self.contract)
        self.artifact.write_bytes(b'not-a-neural-model; hash-only fixture')
        for changes in ({'sampling_seeds':[101]},{'case_ids':self.protocol['case_ids'][:-1]},{'models':[]}):
            changed={**copy.deepcopy(self.protocol),**changes};self.write('protocol.json',changed)
            with self.assertRaises(ValueError):extraction.preflight(self.protocol_path,self.contract)

    def test_camera_and_label_tampering_rejected(self):
        changed=copy.deepcopy(self.protocol);changed['videos'][0]['camera']='left';self.write('protocol.json',changed)
        with self.assertRaisesRegex(ValueError,'same accepted camera'):extraction.preflight(self.protocol_path,self.contract)
        self.write('protocol.json',self.protocol)
        path=self.contract.label_root/'SYN001/SYN001_au1.txt';path.write_text(path.read_text().replace('4,1','4,2'))
        with self.assertRaisesRegex(ValueError,'manual labels changed'):extraction.preflight(self.protocol_path,self.contract)

    def test_synthetic_exact_frames_full_size_and_target_separation(self):
        from PIL import Image
        result=extraction.execute(self.protocol_path,contract=self.contract)
        self.assertEqual(result['status'],'raw_frames_extracted');self.assertEqual(result['successful_unique_frames'],4)
        for case in result['cases']:
            self.assertEqual(set(case['generation_input']),{'source_image','source_au_vector_0_5','target_au_vector_0_5'})
            self.assertFalse(case['target_pixels_passed_to_our_generator'])
            for record in (case['generation_input']['source_image'],case['target_reference_image']):
                frame=record['label_frame_1based']
                with Image.open(record['path']) as image:
                    self.assertEqual(image.size,(16,16));self.assertEqual(image.mode,'RGB')
                    self.assertEqual(image.getpixel((0,0)),(20*(frame-1),10*(frame-1),5*(frame-1)))
        again=extraction.execute(self.protocol_path,contract=self.contract)
        self.assertNotEqual(result['output_directory'],again['output_directory'])
        self.assertTrue(Path(result['output_directory']).exists())

    def test_failure_preserves_cases_and_does_not_substitute(self):
        def fail_after_source(video,frame,target):
            if frame==4:raise RuntimeError('synthetic intentional failure')
            extraction.extract_frame(video,frame,target)
        result=extraction.execute(self.protocol_path,contract=self.contract,decoder=fail_after_source)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['planned_cases'],4);self.assertEqual(result['successful_unique_frames'],1)
        self.assertFalse(result['replacement_cases_selected']);self.assertEqual(len(result['cases']),4)
        self.assertEqual([c['target_reference_image']['status'] for c in result['cases']],['complete','failed','not_started','not_started'])

    def test_source_disappearing_still_writes_failed_full_case_manifest(self):
        def remove_fixture_after_first_frame(video,frame,target):
            extraction.extract_frame(video,frame,target)
            video.unlink()
        result=extraction.execute(self.protocol_path,contract=self.contract,decoder=remove_fixture_after_first_frame)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(result['cases']),4)
        self.assertTrue(result['post_extraction_integrity_errors'])
        self.assertTrue((Path(result['output_directory'])/'raw_frame_manifest.json').is_file())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalExtractionTests))
    if args.report:extraction.save_new(args.report,{'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'passed':result.wasSuccessful(),
        'pixels_used':'synthetic 16x16 color-coded AVI fixtures only','production_test_pixels_read':False,'models_called':False,
        'extractor_sha256':extraction.digest(Path(extraction.__file__))})
    raise SystemExit(not result.wasSuccessful())
