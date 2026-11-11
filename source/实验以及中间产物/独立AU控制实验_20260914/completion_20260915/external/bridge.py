"""Reuse pinned native computations with an explicit post-freeze data-role adapter."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
EXP = COMPLETION.parent
OLD = EXP / 'final_test/external'
ROLE = 'postfreeze_case_validation'
METHODS = ('magicface', 'faceadapter', 'liveportrait')
OLD_PROTOCOL_SHA = '41d0f77e1307fd7da22e212a2485101fa6ab9662f11b02ad63fdb94d0bceda55'

def need(ok, text):
    if not ok:
        raise ValueError(text)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

def load():
    protocol_path = EXP / 'final_test/frozen_protocol.json'
    need(sha(protocol_path) == OLD_PROTOCOL_SHA, 'Original protocol must be pinned before loading old source definitions')
    old_protocol = read(protocol_path)
    roles = old_protocol['execution_sources']
    for name, file in (('external_contracts', 'contracts.py'), ('external_native', 'native_runtime.py')):
        expected = roles[name]
        need(Path(expected['path']).resolve() == OLD / file and sha(OLD / file) == expected['sha256'], 'Pinned old external source changed')
    rename = {'test': ROLE, 'final_test_native_case': 'completion_native_case',
              'final_test_external_projection': 'completion_external_projection',
              'final_test_generation_index': 'completion_external_generation_index'}
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in rename:
                node.value = rename[node.value]
            return node
        def visit_ImportFrom(self, node):
            if node.module == 'contracts':
                node.module = 'completion_external_contracts'
            return node
    modules = []
    for filename, name in (('contracts.py', 'completion_external_contracts'), ('native_runtime.py', 'completion_external_native')):
        original = (OLD / filename).read_text()
        tree = ast.fix_missing_locations(Adapter().visit(ast.parse(original)))
        module = types.ModuleType(name)
        # Preserve original asset-root resolution; all runtime output paths are explicit new paths.
        module.__file__ = str(OLD / filename)
        sys.modules[name] = module
        exec(compile(tree, str(OLD / filename), 'exec'), module.__dict__)
        modules.append(module)
    return tuple(modules)

CONTRACTS, NATIVE = load()

def execution_spec():
    protocol_path = EXP / 'final_test/frozen_protocol.json'
    need(sha(protocol_path) == OLD_PROTOCOL_SHA, 'Original protocol reference changed')
    old = read(protocol_path)
    sampling_ref = old['inputs']['sampling_settings']
    need(sha(sampling_ref['path']) == sampling_ref['sha256'], 'Original native parameters changed')
    sampling = read(sampling_ref['path'])
    models = {r['method']: r for r in old['models']}
    methods = {m: {'settings': copy.deepcopy(sampling['method_settings'][m]),
                   'artifacts': copy.deepcopy(models[m]['artifacts'])} for m in METHODS}
    for spec in methods.values():
        CONTRACTS.validate_parameters(spec['settings'])
    need(methods['liveportrait']['settings']['native_parameters']['flag_relative_motion'] is False, 'Use the already selected stronger absolute LP setting')
    return {'kind': 'completion_native_external_spec', 'data_role': ROLE, 'methods': methods,
            'sampling_seeds': [101, 202, 303], 'physical_calls': {'magicface': 96, 'faceadapter': 96, 'liveportrait': 32},
            'seed_aliases': {'liveportrait': 64}, 'old_protocol': ref(protocol_path),
            'old_native_sources': {name: ref(OLD / name) for name in ('contracts.py', 'native_runtime.py')},
            'sources': {name: ref(HERE / name) for name in ('bridge.py', 'run_external.py')},
            'algorithm_changes': False, 'metadata_adapter': 'AST replaces exact data-role/kind literals and contracts import only',
            'projection_builder': 'new label/video-bound metadata projection; raw failures retained without native calls',
            'prior_subject_exposure': True, 'independent_subject_test': False}

def frozen(path, expected):
    need(sha(path) == expected, 'Explicit completion freeze SHA changed')
    doc = read(path)
    need(doc['kind'] == 'completion_v11_frozen_protocol' and doc['status'] == 'frozen' and doc['extraction_authorized'] is True, 'Completion freeze absent')
    need(doc['external'] == execution_spec(), 'Frozen external settings/source changed')
    label = doc['label_plan']; need(sha(label['path']) == label['sha256'], 'Frozen label plan changed')
    need(doc['case_ids'] == [r['case_id'] for r in read(label['path'])['cases']], 'Frozen case order changed')
    return doc

def projection(freeze, freeze_sha, raw_path, raw_sha, method):
    need(method in METHODS and sha(raw_path) == raw_sha, 'Method/raw manifest mismatch')
    raw = read(raw_path)
    label_ref = freeze['label_plan']
    need(sha(label_ref['path']) == label_ref['sha256'], 'Frozen label plan changed')
    labels = read(label_ref['path'])['cases']
    need(raw.get('status') in ('raw_frames_extracted', 'completed_with_failures') and not raw.get('post_extraction_integrity_errors'), 'Raw extraction incomplete/integrity failed')
    need(raw.get('protocol_sha256') == freeze_sha and raw.get('label_plan_sha256') == label_ref['sha256'], 'Raw freeze/label binding changed')
    need(raw.get('sampling_seeds') == [101, 202, 303] and raw.get('case_count_unchanged') is True
         and raw.get('replacement_cases_selected') is False, 'Raw case/seed policy changed')
    need(len(raw.get('cases', [])) == len(labels) == 32 and len(set(freeze['case_ids'])) == 32
         and [r['case_id'] for r in labels] == [r['case_id'] for r in raw['cases']] == freeze['case_ids'], 'Full32 case order changed')
    video_ref = freeze['extraction']['inputrefs']
    need(sha(video_ref['path']) == video_ref['sha256'], 'Frozen original video bindings changed')
    video_doc = read(video_ref['path'])
    need(video_doc['proposal']['sha256'] == label_ref['sha256'] and video_doc['status'] == 'verified', 'Video/label binding mismatch')
    videos = {v['subject']: v for v in video_doc['videos']}
    settings = freeze['external']['methods'][method]['settings']
    impl = CONTRACTS.validate_parameters(settings)
    physical = []; logical = []; prefailed = {}

    def frame_for_case(frame, label, side):
        video = videos[label['subject']]
        need(all(frame.get(k) == label[k] for k in ('subject', 'split', 'camera')), 'Raw frame subject/role/camera changed')
        need(frame.get('label_frame_1based') == label[side + '_frame_1based']
             and frame.get('video_frame_zero_based') == label[side + '_frame_zero_based'], 'Raw frame index differs from label plan')
        need(frame.get('original_video_path') == label['video_path'] == video['path']
             and frame.get('original_video_sha256') == video['sha256'] and frame['camera'] == video['camera'], 'Raw original video provenance changed')
        need(frame.get('mode') == 'RGB' and frame.get('width') == video['width'] and frame.get('height') == video['height'], 'Raw full-frame dimensions changed')
        need(frame.get('status') in ('complete', 'failed', 'not_started'), 'Unexpected raw frame state')
        if frame['status'] == 'complete':
            return CONTRACTS.frame_metadata(frame), None
        # Retain the real missing/failed metadata; never forge a valid SHA.
        need(frame.get('sha256') is None, 'Failed raw frame must not advertise verified image SHA')
        return copy.deepcopy(frame), {'role': side, 'status': frame['status'], 'error': frame.get('error', 'raw frame unavailable')}

    for label, raw_case in zip(labels, raw['cases']):
        need(label['split'] == ROLE and label['au_order'] == CONTRACTS.AUS, 'Wrong postfreeze label role/AU order')
        need(all(raw_case.get(k) == label[k] for k in ('subject', 'split', 'camera')), 'Raw case identity/role/camera changed')
        need(raw_case.get('target_pixels_passed_to_our_generator') is False, 'Raw generation input role violation')
        inputs = raw_case['generation_input']
        for key in ('source_au_vector_0_5', 'target_au_vector_0_5'):
            need(inputs[key] == label[key], 'Raw human AU values differ from frozen label plan')
        source, source_error = frame_for_case(inputs['source_image'], label, 'source')
        target = None; target_error = None
        if impl != 'magicface_native':
            target, target_error = frame_for_case(raw_case['target_reference_image'], label, 'target')
        errors = [e for e in (source_error, target_error) if e is not None]
        cid = label['case_id']; seeds = [101] if impl == 'liveportrait_native' else [101, 202, 303]
        for seed in seeds:
            job_id = cid + '_seed' + str(seed)
            row = {'case_id': job_id, 'request_case_id': cid, 'subject': label['subject'], 'split': ROLE, 'seed': seed,
                'source': source['path'], 'source_sha256': source['sha256'], 'source_width': source['width'], 'source_height': source['height'],
                'source_frame_1based': source['label_frame_1based'], 'camera': source['camera'], 'source_video_sha256': source['original_video_sha256'],
                'source_au_vector_0_5': copy.deepcopy(label['source_au_vector_0_5']), 'target_au_vector_0_5': copy.deepcopy(label['target_au_vector_0_5']),
                'au_order': CONTRACTS.AUS, 'input_group': settings['input_group'], 'test_used': True, 'method': method,
                'execution_id': method + '/' + job_id, 'protocol_sha256': freeze_sha, 'raw_manifest_sha256': raw_sha}
            if impl == 'magicface_native':
                row.update(au_delta_native=[t - s for s, t in zip(label['source_au_vector_0_5'], label['target_au_vector_0_5'])],
                    native_inference_steps=settings['native_parameters']['steps'], prompt=settings['native_parameters']['prompt'])
            else:
                row.update(target=target['path'], target_sha256=target['sha256'], target_width=target['width'], target_height=target['height'],
                    target_frame_1based=target['label_frame_1based'], target_role='real_observed_driver')
            if errors:
                row['preflight_error'] = {'error_type': 'RawFrameUnavailable', 'frames': copy.deepcopy(errors)}
                prefailed[row['execution_id']] = {'execution_id': row['execution_id'], 'status': 'failed', 'output': None,
                    'generation_started': False, 'worker_dispatched': False, 'preflight_failure': True, 'error': row['preflight_error']}
            physical.append(row)
        for seed in (101, 202, 303):
            actual = 101 if impl == 'liveportrait_native' else seed
            execution_id = method + '/' + cid + '_seed' + str(actual)
            logical.append({'method': method, 'subject': label['subject'], 'case_id': cid, 'seed': seed, 'split': ROLE,
                'status': 'failed' if execution_id in prefailed else 'not_started', 'output': None, 'execution_started': False,
                'source_aus_0_5': copy.deepcopy(label['source_au_vector_0_5']),
                'target_aus_0_5': copy.deepcopy(label['target_au_vector_0_5']), 'au_order': copy.deepcopy(CONTRACTS.AUS),
                'execution_id': execution_id, 'alias_of': None if seed == actual else {'method': method, 'case_id': cid, 'seed': actual}})
    return {'kind': 'completion_external_projection', 'status': 'frozen_projection', 'method': method, 'implementation': impl,
        'protocol_sha256': freeze_sha, 'raw_manifest_sha256': raw_sha, 'settings_sha256': CONTRACTS.signature(settings),
        'test_used': True, 'split': ROLE, 'case_ids': freeze['case_ids'], 'sampling_seeds': [101, 202, 303],
        'cases': physical, 'logical_records': logical, 'preflight_receipts': prefailed,
        'planned_physical_executions': len(physical), 'planned_logical_records': len(logical), 'independent_subject_test': False}
