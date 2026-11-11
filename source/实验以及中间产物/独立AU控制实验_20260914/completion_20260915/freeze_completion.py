"""Main-owned single freeze after code tests and selected-V11 DEV parity."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

def load(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    gen = load('completion_freeze_generation', HERE / 'generation/common.py')
    ev = load('completion_freeze_evaluation', HERE / 'evaluation/contract.py')
    ext = load('completion_freeze_external', HERE / 'external/bridge.py')
    extraction = load('completion_freeze_extraction', HERE / 'protocol/extract_frames.py')
    labels = HERE / 'protocol/proposal_v1/cases_label_only.json'
    parity = HERE / 'generation/dev_parity_202609152317/result.json'
    gen.need(gen.sha(labels) == '93ca535e58fed243b9002369b647eb4d46887fcceebfbd0cd01db0314ff3d0b7', 'Accepted label proposal changed')
    gen.need(gen.sha(parity) == '5f415cb6a70e64ef0277fa20287b621d3e0843f08f91469b3c6a2d88f02c27d9' and gen.read(parity)['passed'] is True,
             'Selected V11 full-constructor parity must pass')
    generation = gen.execution_spec(); evaluation = ev.execution_spec(); external = ext.execution_spec()
    gen.need(generation == gen.read(HERE / 'generation/generation_spec_candidate_v3.json'), 'Reviewed generation spec differs')
    gen.need(evaluation == gen.read(HERE / 'evaluation/spec_v2.json'), 'Reviewed evaluation spec differs')
    old_protocol = gen.read(EXP / 'final_test/frozen_protocol.json')
    assets = {}
    for model in old_protocol['models']:
        for asset in model['artifacts']:
            key = str(Path(asset['path']).resolve())
            gen.need(key not in assets or assets[key]['sha256'] == asset['sha256'], 'Conflicting public/checkpoint artifact binding')
            assets[key] = {'path': key, 'sha256': asset['sha256']}
    case_ids = [r['case_id'] for r in gen.read(labels)['cases']]
    doc = {'kind': 'completion_v11_frozen_protocol', 'status': 'frozen', 'extraction_authorized': True,
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'label_plan': gen.ref(labels), 'case_ids': case_ids,
        'data_role': 'postfreeze_case_validation', 'independent_subject_test': False,
        'exposure_scope': '8 previously exposed held-out subjects; exact32 frames not reused, nearest prior cached frame2 steps; not temporally independent',
        'no_further_parameter_selection': True, 'no_result_based_case_replacement': True,
        'selection': gen.ref(EXP / 'improvements/selected_a_review_20260915/USER_SELECTION.md'),
        'development_constructor_parity': gen.ref(parity), 'development_parity_checker': gen.ref(HERE / 'generation/check_dev_parity.py'),
        'generation': generation, 'evaluation': evaluation, 'external': external,
        'extraction': extraction.execution_spec(), 'public_and_checkpoint_artifacts': list(assets.values()),
        'builder': gen.ref(__file__), 'experiment_scope': '32cases; 5 internal methods x3seeds; MF/FA96each, LPabsolute32actual+64aliases; no training'}
    # Hash every declared source/model/label/video before publishing an authorized freeze.
    refs = extraction.collect_refs(doc)
    refs += extraction.collect_refs(gen.read(doc['extraction']['inputrefs']['path']))
    checks = extraction.verify_refs(refs)
    for method, spec in external['methods'].items():
        ext.NATIVE.validate_runtime(spec['settings'], spec['artifacts'])
    if not a.execute:
        print(json.dumps({'status': 'preflight_no_freeze_no_pixels', 'cases': len(case_ids), 'verified_files': len(checks)}))
        return
    target = HERE / 'frozen_protocol.json'
    gen.need(not target.exists(), 'Never overwrite the single formal completion freeze')
    gen.save(target, doc)
    verified = extraction.preflight(target, gen.sha(target))
    gen.save(HERE / 'freeze_receipt.json', {'status': 'frozen_verified_before_extraction', 'protocol': gen.ref(target),
        'verified_file_count': len(checks), 'pixels_decoded': verified['pixels_decoded'], 'models_loaded': False,
        'case_count': len(case_ids), 'checks': checks})
    print(json.dumps({'status': 'frozen_verified_before_extraction', 'protocol': gen.ref(target), 'verified_files': len(checks)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
