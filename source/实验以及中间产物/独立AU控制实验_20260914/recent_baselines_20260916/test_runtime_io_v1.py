import hashlib,tempfile,unittest
from pathlib import Path
from runtime_io_v1 import hash_exact


class ExactReadTests(unittest.TestCase):
    def test_exact_sizes_and_pin(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            for size in [0,1,4097,4*1024*1024+3]:
                raw=b'x'*size;path=Path(temp)/str(size);path.write_bytes(raw)
                row=hash_exact(path,size,hashlib.sha256(raw).hexdigest())
                self.assertEqual(row['bytes'],size);self.assertFalse(row['read_past_expected_size']);self.assertTrue(row['pin_verified'])

    def test_wrong_size_or_pin_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/'one';path.write_bytes(b'a')
            with self.assertRaises(ValueError):hash_exact(path,2)
            with self.assertRaises(ValueError):hash_exact(path,1,'0'*64)


if __name__=='__main__':unittest.main(verbosity=2)
