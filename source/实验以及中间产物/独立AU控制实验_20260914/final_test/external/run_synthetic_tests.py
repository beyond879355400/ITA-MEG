#!/usr/bin/env python3
"""Persist only synthetic test evidence; refuses accidental model-library imports."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent


def main():
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_external_final.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    forbidden=[x for x in ('torch','cv2','onnxruntime','PIL') if x in sys.modules]
    assert not forbidden,('Unexpected model/image library import',forbidden)
    payload={'kind':'synthetic_external_final_gate_tests','status':'PASS' if result.wasSuccessful() else 'FAIL',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'real_test_pixels_read':False,'native_inference_executed':False,'model_libraries_imported':forbidden,
             'fixture_directory':str(HERE/'synthetic_tmp'),
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'contracts.py',HERE/'native_runtime.py',HERE/'run_external_final.py',HERE/'test_external_final.py',HERE.parent/'protocol_gate.py']}}
    out=HERE/'tests';out.mkdir(exist_ok=True);base=out/('synthetic_tests_'+datetime.now().strftime('%Y%m%d%H%M'))
    target=base.with_suffix('.json');index=2
    while target.exists():target=base.with_name(base.name+f'-{index:02d}').with_suffix('.json');index+=1
    with target.open('x') as file:json.dump(payload,file,indent=2)
    print(json.dumps({'artifact':str(target),**payload},ensure_ascii=False),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
