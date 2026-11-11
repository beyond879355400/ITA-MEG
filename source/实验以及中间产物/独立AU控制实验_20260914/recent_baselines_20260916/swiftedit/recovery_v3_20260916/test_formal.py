import hashlib,io,json,tempfile,unittest
from pathlib import Path
import run_formal as formal
from run_formal_all import normalized


class FormalTests(unittest.TestCase):
    def test_exact_manifest_and_matrix(self):
        doc=formal.stage_manifest()
        self.assertEqual(len(doc['cases']),32)
        self.assertEqual(len({(r['case_id'],s) for r in doc['cases'] for s in formal.SEEDS}),96)
        self.assertTrue(all(r['sampling_seeds']==[101,202,303] for r in doc['cases']))

    def test_empty_or_modified_manifest_fails(self):
        for raw in [b'',b'{}',b' '*50750]:
            with self.assertRaises(ValueError):formal.validate_manifest(raw)

    def test_no_unapproved_seed(self):
        with self.assertRaises(ValueError):formal.configure(42)

    def test_failure_and_not_started_both_retained(self):
        case=formal.stage_manifest()['cases'][0]
        failed=normalized(case,101,'synthetic',started=True,reason='synthetic failure')
        absent=normalized(case,202,'synthetic',reason='upstream stopped')
        self.assertEqual(failed['status'],'failed');self.assertEqual(absent['status'],'not_started')
        self.assertIsNone(failed['output']);self.assertIsNone(absent['output'])
        self.assertTrue(failed['execution_started']);self.assertFalse(absent['execution_started'])
        self.assertFalse(failed['target_image_used'])

    def test_target_edit_is_always_primary(self):
        case=formal.stage_manifest()['cases'][0]
        with tempfile.TemporaryDirectory(dir=formal.TMP) as temp:
            path=Path(temp)/'not_real_model_output';path.write_bytes(b'synthetic')
            record={'files':[{'role':'source_reconstruction','path':'must-not-read'},
                             {'role':'target_edit','path':str(path),'sha256':hashlib.sha256(b'synthetic').hexdigest()}]}
            row=normalized(case,303,'synthetic',record)
            self.assertEqual(row['output']['path'],str(path));self.assertEqual(row['primary_output_role'],'target_edit')

    def test_source_bytes_transport_preserves_pixels(self):
        from PIL import Image
        image=Image.new('RGB',(3,2),(12,37,86));buffer=io.BytesIO();image.save(buffer,format='PNG');raw=buffer.getvalue()
        with tempfile.TemporaryDirectory(dir=formal.TMP) as temp:
            path=Path(temp)/'synthetic.png';path.write_bytes(raw)
            self.assertEqual(Image.open(path).resize((512,512)).tobytes(),Image.open(io.BytesIO(raw)).resize((512,512)).tobytes())

    def test_hot_vae_redirects_only_approved_component(self):
        from hot_vae_route import factory_route
        calls=[]
        def fake(*args,**kwargs):calls.append((args,kwargs));return 'same_factory'
        route=factory_route(fake,Path('/old/sd21'),Path('/hot'))
        self.assertEqual(route('/old/sd21',subfolder='vae'),'same_factory')
        self.assertEqual(calls[-1],(('/hot',),{'subfolder':'vae'}))
        route('/old/turbo',subfolder='vae');self.assertEqual(calls[-1][0][0],'/old/turbo')
        route('/old/sd21',subfolder='text_encoder');self.assertEqual(calls[-1][0][0],'/old/sd21')


if __name__=='__main__':unittest.main(verbosity=2)
