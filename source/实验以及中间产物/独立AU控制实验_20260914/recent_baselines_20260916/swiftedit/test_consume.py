import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import consume_extract as consume
import stage_weights as stage


def fixture(root):
    downloads=root/'downloads';downloads.mkdir()
    archive=io.BytesIO();rng=random.Random(101)
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        for i in range(3):
            data=rng.randbytes(140000)
            member=tarfile.TarInfo('weights/file%d.bin'%i);member.size=len(data)
            tar.addfile(member,io.BytesIO(data))
    raw=archive.getvalue();size=(len(raw)+4)//5;entries=[];paths=[]
    for i in range(5):
        data=raw[i*size:(i+1)*size];name='part-'+str(i);path=downloads/name;path.write_bytes(data);paths.append(path)
        entries.append({'name':name,'size':len(data),'digest':'sha256:'+hashlib.sha256(data).hexdigest()})
    return downloads,entries,stage.inspect(paths)


class ConsumeTests(unittest.TestCase):
    def test_completed_outputs_before_ordered_part_deletion(self):
        with tempfile.TemporaryDirectory(dir=stage.HERE,prefix='synthetic_consume_') as temp:
            root=Path(temp);downloads,entries,index=fixture(root);out=root/'out'
            receipt=consume.extract_consume(downloads,out,entries,index,reserve=0)
            self.assertEqual(receipt['status'],'complete')
            self.assertEqual(len(receipt['deleted_parts']),5)
            self.assertFalse(list(downloads.iterdir()))
            rows=[json.loads(x) for x in (out/'consumption_events.jsonl').read_text().splitlines()]
            completed=0;names=[]
            for row in rows:
                if row['event']=='member_durable_verified':completed+=1
                if row['event']=='delete_intent':
                    self.assertGreater(completed,0);self.assertTrue(row['reader_closed'])
                    names.append(Path(row['path']).name)
            self.assertEqual(names,[r['name'] for r in entries])
            self.assertEqual(len(receipt['files']),3)

    def test_failure_retains_partial_completed_and_remaining_parts(self):
        with tempfile.TemporaryDirectory(dir=stage.HERE,prefix='synthetic_consume_') as temp:
            root=Path(temp);downloads,entries,index=fixture(root);out=root/'out'
            original=stage.safe_member
            def fail_second(member):
                if member.name=='weights/file1.bin':raise ValueError('injected failure')
                return original(member)
            with patch.object(stage,'safe_member',side_effect=fail_second):
                with self.assertRaisesRegex(ValueError,'injected'):
                    consume.extract_consume(downloads,out,entries,index,reserve=0)
            doc=json.loads((out/'extraction_failure.json').read_text())
            self.assertEqual(len(doc['completed_files']),1)
            self.assertTrue((out/'weights/file0.bin').is_file())
            self.assertGreater(len(doc['remaining_parts']),0)
            self.assertFalse((out/'extraction_receipt.json').exists())

    def test_symlink_part_rejected_without_deletion(self):
        with tempfile.TemporaryDirectory(dir=stage.HERE,prefix='synthetic_consume_') as temp:
            root=Path(temp);downloads,entries,index=fixture(root)
            original=downloads/entries[0]['name'];saved=root/'preserved';original.rename(saved);original.symlink_to(saved)
            with self.assertRaisesRegex(ValueError,'symlink'):
                consume.extract_consume(downloads,root/'out',entries,index,reserve=0)
            self.assertTrue(saved.exists());self.assertTrue(original.is_symlink())
            self.assertEqual(len(list(downloads.iterdir())),5)

    def test_escaped_part_rejected_without_deletion(self):
        with tempfile.TemporaryDirectory(dir=stage.HERE,prefix='synthetic_consume_') as temp:
            root=Path(temp);downloads,entries,index=fixture(root);entries[0]['name']='../escape'
            with self.assertRaises(ValueError):consume.extract_consume(downloads,root/'out',entries,index,reserve=0)
            self.assertEqual(len(list(downloads.iterdir())),5)

    def test_disk_failure_preserves_all_parts(self):
        with tempfile.TemporaryDirectory(dir=stage.HERE,prefix='synthetic_consume_') as temp:
            root=Path(temp);downloads,entries,index=fixture(root)
            with self.assertRaisesRegex(ValueError,'reserve'):
                consume.extract_consume(downloads,root/'out',entries,index,reserve=10**20)
            self.assertEqual(len(list(downloads.iterdir())),5)


if __name__=='__main__':unittest.main(verbosity=2)
