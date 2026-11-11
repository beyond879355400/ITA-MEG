import csv
from pathlib import Path
import tempfile
import unittest
from openface_v2 import parse,AUS17

class PrimaryFaceTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent/'synthetic_tests';self.root.mkdir(exist_ok=True)
    def make(self,scale,value,confidence='1'):
        d={'face':'0','confidence':confidence}
        for i in range(68):d[f'x_{i}']=str((i%10)*scale);d[f'y_{i}']=str((i//10)*scale)
        for au in AUS17:d[f'AU{au:02d}_r']=str(value)
        return d
    def run_rows(self,rows):
        with tempfile.TemporaryDirectory(dir=self.root) as d:
            p=Path(d)/'x.csv'
            with p.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            return parse(p)
    def test_largest_even_if_AU_worse_or_confidence_lower(self):
        r=self.run_rows([self.make(1,0,'1'),self.make(10,4,'.1')])
        self.assertEqual(r['selected_row'],1);self.assertEqual(r['vector'],[4]*12);self.assertEqual(r['face_count'],2)
    def test_invalid_selected_AU_does_not_fall_back(self):
        a=self.make(10,2);a['AU01_r']='nan';r=self.run_rows([self.make(1,0),a])
        self.assertEqual(r['selected_row'],1);self.assertIsNone(r['vector'])
    def test_tie_is_first_returned_row(self):
        r=self.run_rows([self.make(10,1),self.make(10,2)]);self.assertEqual(r['selected_row'],0)
    def test_invalid_landmarks_not_rankable(self):
        a=self.make(10,1);a['x_0']='nan';r=self.run_rows([a]);self.assertIsNone(r['vector']);self.assertIsNone(r['selected_row'])

if __name__=='__main__':unittest.main()
