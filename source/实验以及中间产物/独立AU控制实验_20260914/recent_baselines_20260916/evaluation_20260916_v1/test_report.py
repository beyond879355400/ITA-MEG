"""Small synthetic display/IO tests only; no real scores or model calls."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import extent_io
import render_report as r


class ExtentTests(unittest.TestCase):
    def test_never_extra_eof_read(self):
        class Stream:
            def __init__(self):self.left=5;self.closed=False
            def __enter__(self):return self
            def __exit__(self,*a):self.closed=True
            def fileno(self):return 42
            def read(self,n):
                if not self.left:raise AssertionError('Extra EOF read')
                take=min(n,self.left);self.left-=take;return b'x'*take
        class S:
            st_dev=1;st_ino=2;st_size=5;st_mtime_ns=3;st_mode=0o100644
        stream=Stream()
        with patch.object(Path,'stat',return_value=S()),patch.object(Path,'open',return_value=stream),patch('extent_io.os.fstat',return_value=S()):
            self.assertEqual(extent_io.read_bytes('synthetic'),b'xxxxx')
        self.assertTrue(stream.closed)

    def test_short_read_rejected(self):
        class S:
            st_dev=1;st_ino=2;st_size=5;st_mtime_ns=3;st_mode=0o100644
        stream=io.BytesIO(b'xx')
        with patch.object(Path,'stat',return_value=S()),patch.object(Path,'open',return_value=stream),patch('extent_io.os.fstat',return_value=S()),patch.object(stream,'fileno',return_value=42):
            with self.assertRaisesRegex(ValueError,'short'):extent_io.read_bytes('synthetic')

    def test_actual_hash_and_pin(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_report_',dir=r.HERE)as tmp:
            p=Path(tmp)/'tiny';p.write_bytes(b'abc')
            self.assertEqual(extent_io.sha(p),hashlib.sha256(b'abc').hexdigest())
            with self.assertRaises(ValueError):extent_io.read_bytes(p,'0'*64)


class GalleryTests(unittest.TestCase):
    def measurement(self,box=[20.,30.,80.,90.]):
        return {'sha256':'a'*64,'policy_sha256':r.p.POLICY_SHA,'selected_row':0,
                'image_size':[100,120],'all_candidates':[{'row_index':0,'bbox68':box}]}

    def test_fixed_square_pad_clips_to_pixels(self):
        box=r.measured_face_box(self.measurement(),{'sha256':'a'*64},(100,120))
        self.assertEqual(box,[15,25,85,95])

    def test_no_cross_image_bbox(self):
        self.assertIsNone(r.measured_face_box(self.measurement(),{'sha256':'b'*64},(100,120)))

    def test_no_dimension_mismatch_bbox(self):
        self.assertIsNone(r.measured_face_box(self.measurement(),{'sha256':'a'*64},(512,512)))

    def test_no_face_fallback_detector(self):
        m=self.measurement();m['selected_row']=None
        self.assertIsNone(r.measured_face_box(m,{'sha256':'a'*64},(100,120)))

    def test_invalid_bbox_no_zoom(self):
        self.assertIsNone(r.measured_face_box(self.measurement([80,30,20,90]),{'sha256':'a'*64},(100,120)))

    def test_zeroedit_C_NA_not_failure(self):
        row={'generation_status':'complete','input_availability':'available','au_parse_status':'primary_face_finite',
             'facenet_status':'valid','buffalo_status':'valid_primary_face','extra5_proxy_mae_to_real_target':0.1,
             'group':'no_edit','target_mae_changed':None}
        self.assertEqual(r.failure_reasons(row),[])

    def test_nonprimary_failure_not_hidden(self):
        row={'generation_status':'complete','input_availability':'available','au_parse_status':'primary_face_finite',
             'facenet_status':'no_face','buffalo_status':'valid_primary_face','extra5_proxy_mae_to_real_target':0.1,
             'group':'no_edit','target_mae_changed':None}
        self.assertTrue(any('FaceNet'in x for x in r.failure_reasons(row)))


if __name__=='__main__':unittest.main(verbosity=2)
