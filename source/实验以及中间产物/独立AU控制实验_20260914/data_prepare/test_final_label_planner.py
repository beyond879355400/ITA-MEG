"""Synthetic label/AVI metadata tests; no real pixel data or model."""
from pathlib import Path
import struct
import tempfile
import unittest
import plan_final_test_labels as planner


def chunk(kind,payload):return kind+struct.pack('<I',len(payload))+payload+(b'\0' if len(payload)%2 else b'')


class HeaderAndLabelTests(unittest.TestCase):
    def test_header_reader_does_not_read_movie_payload(self):
        avih=struct.pack('<14I',50000,0,0,0,4845,0,1,0,1024,768,0,0,0,0)
        strh=bytearray(56);strh[:4]=b'vids';strh[4:8]=b'MJPG';struct.pack_into('<4I',strh,20,1,20,0,4845)
        hdrl=chunk(b'LIST',b'hdrl'+chunk(b'avih',avih)+chunk(b'LIST',b'strl'+chunk(b'strh',bytes(strh))))
        body=b'AVI '+hdrl+chunk(b'LIST',b'movi'+b'NO_PIXEL_READ'*100)
        with tempfile.TemporaryDirectory(prefix='header_test_',dir=planner.HERE/'final_test_label_plan') as directory:
            path=Path(directory)/'synthetic.avi';path.write_bytes(b'RIFF'+struct.pack('<I',len(body))+body)
            result=planner.avi_header_only(path)
        self.assertEqual(result['frames'],4845)
        self.assertEqual(result['header']['width'],1024)
        self.assertFalse(result['decoder_used']);self.assertFalse(result['movie_packet_bodies_read'])
        self.assertTrue(all(r['offset']+r['bytes']<=12+len(hdrl) for r in result['read_ranges']))

    def test_longest_zero_interval_and_lower_midpoint(self):
        vectors={i:{'au'+str(a):(0 if i in (1,2,4,5,6,7) else 1) for a in planner.AUS} for i in range(1,9)}
        result=planner.source_interval(vectors,'S')
        self.assertEqual((result['start_frame_1based'],result['end_frame_1based'],result['source_frame_1based']),(4,7,5))
        vectors={i:{'au'+str(a):1 for a in planner.AUS} for i in range(1,3)}
        with self.assertRaises(ValueError):planner.source_interval(vectors,'S')

    def test_coverage_cell_mapping(self):
        vector=[1,2,3]+[0]*9
        self.assertEqual(planner.cell_indices(vector),[0,4,8])
        self.assertEqual([planner.CELLS[i] for i in planner.cell_indices(vector)],[(1,1),(2,2),(4,3)])


if __name__=='__main__':unittest.main(verbosity=2)
