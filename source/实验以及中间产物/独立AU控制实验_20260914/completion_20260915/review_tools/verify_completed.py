#!/usr/bin/env python3
"""Independent terminal audit. No image decoding, ML imports or model calls.

CSV selection/AU arithmetic use the pinned independent prior verifier, never the
new evaluator. Statistics/50k bootstrap are independently recomputed locally.
Identity cosines are JSON+input-SHA readbacks: embeddings are not recomputed.
"""
import argparse
import collections
import copy
import csv
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
EXP = COMPLETION.parent
ROLE = 'postfreeze_case_validation'
FREEZE_SHA = '498298977f3589248084e37d053ec7cd8f37f71b35e914849465081a3df9af33'
LABEL_SHA = '93ca535e58fed243b9002369b647eb4d46887fcceebfbd0cd01db0314ff3d0b7'
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
OLD_VERIFIER = EXP / 'final_test/review_tools/verify_final_measurements.py'
OLD_VERIFIER_SHA = 'ba6bcd973a72e593b14e3b8915a83837e42f6137ddd0da077797091a9f3d6c1d'
AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
EXTRA = (7, 10, 14, 23, 45)
SUBJECTS = ('SN003', 'SN005', 'SN008', 'SN009', 'SN021', 'SN024', 'SN031', 'SN032')
SEEDS = [101, 202, 303]
OURS = [v + '_local_lp_s050' for v in ('base', 'bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp')]
METHODS = OURS + ['magicface', 'faceadapter', 'liveportrait']
from statistics_check import aggregate, close, key, mapped, mean, verify_statistics


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class HashAudit:
    def __init__(self):
        self.checked = {}; self.kinds = collections.defaultdict(set)

    def ref(self, value, kind='artifact'):
        need(isinstance(value, dict) and isinstance(value.get('path'), str)
             and re.fullmatch(r'[0-9a-f]{64}', str(value.get('sha256', ''))), 'Malformed ' + kind + ' reference')
        path = Path(value['path']).resolve(); expected = value['sha256']
        need(str(path) not in self.checked or self.checked[str(path)] == expected, 'Conflicting hashes for ' + str(path))
        if str(path) not in self.checked:
            need(path.is_file() and sha(path) == expected, kind + ' SHA mismatch: ' + str(path))
            self.checked[str(path)] = expected
        self.kinds[kind].add(str(path))
        return path

    def nested(self, value, kind='artifact'):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and value['sha256'] is not None:
                self.ref(value, kind)
            for child in value.values():
                self.nested(child, kind)
        elif isinstance(value, list):
            for child in value:
                self.nested(child, kind)


def independent_core():
    need(sha(OLD_VERIFIER) == OLD_VERIFIER_SHA, 'Independent prior verifier source changed')
    spec = importlib.util.spec_from_file_location('prior_independent_arithmetic_only', OLD_VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manual_cases(plan, hashes):
    columns = {}
    for item in plan['labels']:
        pair = item['subject'], item['au']
        need(pair not in columns, 'Duplicate manual label column')
        values = {}
        with hashes.ref(item, 'manual_label').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.reader(stream):
                if row:
                    need(len(row) == 2, 'Malformed human label row')
                    frame, value = map(int, row)
                    need(frame > 0 and frame not in values and 0 <= value <= 5, 'Invalid human label')
                    values[frame] = value
        need(sorted(values) == list(range(1, 4846)), 'Expected full4845-frame human label sequence')
        columns[pair] = values
    close(set(columns), set(itertools.product(SUBJECTS, AUS)), '8x12 human labels')
    metadata = {}
    for row in plan['cases']:
        need(row['split'] == ROLE and row['au_order'] == list(AUS), 'Label role/AU order mismatch')
        vectors = {}
        for side in ('source', 'target'):
            frame = row[side + '_frame_1based']
            need(row[side + '_frame_zero_based'] == frame - 1, 'Manual/video frame convention changed')
            vectors[side] = [columns[row['subject'], au][frame] for au in AUS]
            close(row[side + '_au_vector_0_5'], vectors[side], 'Original manual vector')
        metadata[row['case_id']] = {'case_id': row['case_id'], 'subject': row['subject'], 'split': ROLE,
            'group': 'no_edit' if vectors['source'] == vectors['target'] else 'edited',
            'source_labels_12': vectors['source'], 'target_labels_12': vectors['target']}
    need(len(metadata) == len(plan['cases']) == 32, '32 unique cases required')
    close(collections.Counter(r['group'] for r in metadata.values()), {'no_edit': 8, 'edited': 24}, 'Case scopes')
    return metadata


def normalize_generation(indexes, cases):
    records = []; names_seen = set()
    expected = {(m, row['subject'], cid, seed) for m in METHODS for cid, row in cases.items() for seed in SEEDS}
    for index in indexes:
        need(index['status'] in ('complete', 'completed_with_failures') and index.get('usable_for_evaluation', True) is True,
             'Nonterminal or quarantined generation index')
        names = index['method_names']
        need(names and not names_seen.intersection(names) and len(names) == len(set(names)), 'Duplicate method indexes')
        names_seen.update(names)
        close(index['case_ids'], list(cases), 'Index cases')
        close(index['sampling_seeds'], SEEDS, 'Index seed grid')
        local = []
        for original in index['records']:
            row = copy.deepcopy(original); k = key(row)
            need(k in expected and row['method'] in names, 'Unplanned generation key')
            need(row['split'] == ROLE and row['status'] in ('complete', 'failed', 'not_started'), 'Wrong generation state/role')
            need(type(row.get('execution_started')) is bool and isinstance(row.get('execution_id'), str) and row['execution_id'], 'Execution identity missing')
            for field, target in [('source_aus_0_5', 'source_labels_12'), ('target_aus_0_5', 'target_labels_12')]:
                close(row.get(field), cases[row['case_id']][target], 'Generation label binding')
            if row['status'] == 'complete':
                need(row['execution_started'], 'Complete output without execution')
                if isinstance(row.get('output'), str):
                    row['output'] = {'path': row['output'], 'sha256': row.get('sha256')}
                need(isinstance(row.get('output'), dict), 'Complete output missing reference')
            else:
                need(row['status'] != 'not_started' or not row['execution_started'], 'Not-started row claims execution')
                if row.get('output') is not None or row.get('sha256') is not None:
                    artifact = {'output': copy.deepcopy(row.get('output')), 'sha256': row.get('sha256'),
                        'reason': 'retained unaccepted artifact from failed generation; not measured or counted as success'}
                    need('failed_artifact' not in row or row['failed_artifact'] == artifact, 'Conflicting failure artifact')
                    row.update(failed_artifact=artifact, output=None, sha256=None)
            local.append(row); records.append(row)
        if 'physical_executions' in index:
            close(index['physical_executions'], sum(r['execution_started'] for r in local if r.get('alias_of') is None), 'Declared executions')
        if 'unique_output_images' in index:
            close(index['unique_output_images'], len({r['output']['sha256'] for r in local if r['status'] == 'complete'}), 'Declared unique outputs')
    close(names_seen, set(METHODS), 'All8 methods present')
    indexed = mapped(records); physical_ids = set()
    for k, row in indexed.items():
        if row['method'] == 'liveportrait' and row['seed'] != 101:
            close(row.get('alias_of'), {'method': 'liveportrait', 'case_id': row['case_id'], 'seed': 101}, 'LP alias')
            base = indexed.get(k[:3] + (101,)); need(base is not None, 'Alias has no seed101 base')
            for field in ('execution_id', 'execution_started', 'status', 'output'):
                close(row.get(field), base.get(field), 'Alias ' + field)
        else:
            need(row.get('alias_of') is None and row['execution_id'] not in physical_ids, 'Repeated actual execution or stochastic alias')
            physical_ids.add(row['execution_id'])
    accounting = {}
    for method in METHODS:
        own = [r for r in records if r['method'] == method]; physical = [r for r in own if r.get('alias_of') is None]
        accounting[method] = {'planned_logical_seed_outputs': 96, 'observed_logical_rows': len(own), 'missing_logical_rows': 96 - len(own),
            'planned_physical_executions': 32 if method == 'liveportrait' else 96,
            'executed_physical_calls': sum(r['execution_started'] for r in physical),
            'successful_physical_calls': sum(r['status'] == 'complete' for r in physical),
            'alias_rows': len(own) - len(physical),
            'unique_successful_image_SHAs': len({r['output']['sha256'] for r in own if r['status'] == 'complete'})}
    close(sum(v['planned_physical_executions'] for v in accounting.values()), 704, '704 planned physical jobs')
    return records, accounting


def verify_raw_references(freeze, plan, references, hashes):
    raw = read(hashes.ref(references['raw_manifest'], 'raw_manifest'))
    need(raw['protocol_sha256'] == FREEZE_SHA and raw['label_plan_sha256'] == LABEL_SHA
         and raw['status'] in ('raw_frames_extracted', 'completed_with_failures') and not raw.get('post_extraction_integrity_errors'), 'Raw extraction not accepted')
    close([r['case_id'] for r in raw['cases']], freeze['case_ids'], 'Raw full case order')
    cache = read(hashes.ref(references['source_cache_manifest'], 'source_cache'))
    need(cache['protocol_sha256'] == FREEZE_SHA and cache['status'] in ('complete', 'completed_with_failures')
         and cache['target_pixels_read'] is False and cache['target_latents_computed'] is False, 'Source cache role/freeze mismatch')
    hashes.ref(cache['features_file'], 'source_feature_tensor_hash_only')
    sources = {r['source_id']: r for r in cache['sources']}; cache_cases = {r['case_id']: r for r in cache['cases']}
    close(len(sources), 8, 'Source cache count')
    labels = {r['case_id']: r for r in plan['cases']}
    video_doc = read(hashes.ref(freeze['extraction']['inputrefs'], 'video_manifest'))
    hashes.nested(video_doc, 'original_video_and_input_binding')
    videos = {r['subject']: r for r in video_doc['videos']}
    refs = {r['case_id']: r for r in references['cases']}
    close(list(refs), freeze['case_ids'], 'Evaluation reference order')
    for row in raw['cases']:
        cid = row['case_id']; label = labels[cid]
        close((row['subject'], row['split'], row['camera']), (label['subject'], ROLE, label['camera']), 'Raw case owner/role')
        for side in ('source', 'target'):
            frame = row['generation_input']['source_image'] if side == 'source' else row['target_reference_image']
            close(row['generation_input'][side + '_au_vector_0_5'], label[side + '_au_vector_0_5'], 'Raw AU labels')
            close(frame['label_frame_1based'], label[side + '_frame_1based'], 'Raw frame number')
            close(frame['video_frame_zero_based'], label[side + '_frame_zero_based'], 'Raw video frame number')
            close(frame['original_video_sha256'], videos[label['subject']]['sha256'], 'Raw video SHA')
            close(frame['original_video_path'], label['video_path'], 'Raw video path')
            close((frame['subject'], frame['split'], frame['camera']), (label['subject'], ROLE, label['camera']), 'Raw frame identity')
            if frame['status'] == 'complete':
                hashes.ref(frame, 'raw_image_hash_only')
            else:
                need(frame['sha256'] is None, 'Failed raw frame carries accepted SHA')
            close(refs[cid][side + '_reference']['raw_image'], frame, 'Canonical/raw reference binding')
            reference = refs[cid][side + '_reference']
            if reference['status'] == 'complete':
                hashes.ref(reference['image'], 'canonical_reference_hash_only')
        source = sources[cache_cases[cid]['source_id']]
        close(source['raw_source'], row['generation_input']['source_image'], 'Source cache raw binding')
        close(source['subject'], label['subject'], 'Source cache subject')
        close(source['split'], ROLE, 'Source cache role')
        if refs[cid]['source_reference']['status'] == 'complete':
            close(refs[cid]['source_reference']['image'], source['canonical_source'], 'No source recrop')
    return raw, cache, refs


def verify_openface(job, payload, available, hashes, core):
    need(payload['status'] == 'complete' and payload['protocol_sha256'] == FREEZE_SHA
         and payload['measurement_policy_sha256'] == POLICY_SHA, 'OpenFace run not terminal/bound')
    logical = mapped(payload['records']); close(set(logical), set(available), 'OpenFace logical availability')
    reader = core.load_csv_reader(); parsed = {}; audit = []; failures = []
    for row in payload['unique_measurements']:
        digest = row['sha256']; need(digest not in parsed, 'Duplicate OpenFace physical SHA')
        folder = job / 'openface/raw' / ('sha_' + digest)
        files = sorted(folder.glob('*.csv'))
        if 'csv_files' in row:
            close(sorted(Path(p).resolve() for p in row['csv_files']), files, 'CSV disk inventory')
        vector = values = None
        if len(files) == 1 and row['parse_status'] != 'error':
            raw, selected, vector, values, confidence = reader(files[0])
            for field, expected in [('raw_rows', raw), ('selected_row', selected), ('face_count', len(raw)), ('confidence', confidence)]:
                close(row[field], expected, 'Independent68-area CSV ' + field)
            if row.get('returncode') not in (None, 0) or row.get('input_unchanged') is False:
                vector = values = None
            csv_sha = sha(files[0]); hashes.ref({'path': str(files[0]), 'sha256': csv_sha}, 'openface_csv')
            audit.append({'image_sha256': digest, 'path': str(files[0]), 'csv_sha256': csv_sha, 'selected_row': selected,
                'raw_row_count': len(raw), 'selection': 'largest68landmark_bbox_area_then_first_row'})
        else:
            need(row['parse_status'] in ('error', 'missing_or_multiple_csv', 'process_failed', 'input_changed'), 'Unexplained CSV failure')
        close(row['vector'], vector, 'CSV-derived AU12')
        close(row['selected_au17'], values, 'CSV-derived AU17')
        if vector is None:
            failures.append({'sha256': digest, 'parse_status': row['parse_status'], 'csv_files': [str(p) for p in files]})
        parsed[digest] = {'vector': vector, 'selected_au17': values}
    close(set(parsed), {r['sha256'] for r in available.values()}, 'Physical image SHA coverage')
    for k, row in logical.items():
        close(row['sha256'], available[k]['sha256'], 'AU logical image hash')
        close(row['split'], ROLE, 'AU logical role')
        close(row['vector'], parsed[row['sha256']]['vector'], 'Logical AU12')
        close(row['selected_au17'], parsed[row['sha256']]['selected_au17'], 'Logical AU17')
    return logical, parsed, audit, failures


def independent_metric_rows(doc, exported, generations, am, parsed, identities, cases, spec, core):
    rows = []; generation_map = mapped(generations); exported_map = mapped(exported)
    close(set(exported_map), set(mapped(doc['planned'])), 'Export retains768 planned rows')
    for planned in doc['planned']:
        k = key(planned); meta = cases[planned['case_id']]; row = exported_map[k]
        for field, expected in meta.items():
            close(row[field], expected, 'Metric metadata')
        output = parsed[am[k]['sha256']] if k in am else {}
        sk = ('source_real', planned['subject'], planned['case_id'], 0)
        tk = ('target_real', planned['subject'], planned['case_id'], 0)
        source = parsed[am[sk]['sha256']] if sk in am else {}
        target = parsed[am[tk]['sha256']] if tk in am else {}
        values = core.au_metrics(output.get('vector'), source.get('vector'), meta['source_labels_12'], meta['target_labels_12'],
                                 output.get('selected_au17'), target.get('selected_au17'))
        for kind, metric, field in [('facenet', 'identity_cosine', 'FaceNet_measurement'), ('buffalo', 'buffalo_identity_cosine', 'buffalo_measurement')]:
            record = identities[kind].get(k); values[metric] = record.get('identity_cosine') if record else None
            close(row[field], record, 'Bound identity JSON readback')
        close(row['AU_measurement'], am.get(k), 'AU export readback')
        for metric, value in values.items():
            close(row[metric], value, 'Recomputed ' + metric)
        for au in EXTRA:
            p = (output.get('selected_au17') or {}).get(str(au)); t = (target.get('selected_au17') or {}).get(str(au)); s = (source.get('selected_au17') or {}).get(str(au))
            expected = {'output': p, 'real_target_measured': t, 'real_source_measured': s,
                'absolute_error_to_target': None if p is None or t is None else abs(p - t),
                'output_minus_source': None if p is None or s is None else p - s,
                'target_minus_source': None if t is None or s is None else t - s}
            close(row['extra5'][str(au)], expected, 'Extra AU diagnostic fields')
        generation = generation_map.get(k)
        if generation is None:
            continue
        metrics = {}; reasons = {}
        for name, definition in spec['metrics'].items():
            value = None
            if generation['status'] != 'complete': reason = 'generation_' + generation['status']
            elif definition['scope'] != 'all' and meta['group'] != definition['scope']: reason = 'out_of_predeclared_scope'
            else:
                value = values[definition['source_metric']]
                reason = None if value is not None else 'measurement_or_reference_NA'
            metrics[name] = value; reasons[name] = reason
        rows.append({**planned, 'status': generation['status'], 'metrics': metrics, 'metric_reasons': reasons,
            'execution_id': generation['execution_id'], 'alias_of': generation.get('alias_of'), 'output': generation.get('output'),
            'failed_artifact': generation.get('failed_artifact')})
    return rows


def verify(job, freeze_path, expected_sha):
    job = Path(job).resolve(); freeze_path = Path(freeze_path).resolve()
    need(job.is_relative_to(COMPLETION / 'evaluation'), 'Only completion evaluation jobs may be verified')
    # Check terminal export files before touching any real image or model bytes.
    for path in [job / 'statistics/paired_subject_results.json', job / 'statistics/all_case_seed_metrics.json',
                 *[job / kind / 'binding.json' for kind in ('openface', 'facenet', 'buffalo')]]:
        need(path.is_file(), 'Terminal export absent; no ongoing image inspection: ' + str(path))
    need(expected_sha == FREEZE_SHA and sha(freeze_path) == expected_sha, 'Exact completion freeze SHA required')
    freeze = read(freeze_path)
    need(freeze['kind'] == 'completion_v11_frozen_protocol' and freeze['status'] == 'frozen' and freeze['data_role'] == ROLE
         and freeze['independent_subject_test'] is False, 'Explicit postfreeze/nonblind scope missing')
    evaluation = freeze['evaluation']; spec = evaluation['analysis_spec']
    close(evaluation['methods'], METHODS, 'Frozen8 methods')
    close(evaluation['sampling_seeds'], SEEDS, 'Frozen3 seeds')
    close((spec['bootstrap_samples'], spec['bootstrap_seed'], spec['confidence_level']), (50000, 20260915, .95), 'Inference protocol')
    close(evaluation['families'], {'external_primary_C': 3, 'internal_primary_C': 4}, 'Holm fixed families')
    hashes = HashAudit(); hashes.nested(freeze, 'frozen_code_model_artifact')
    plan = read(hashes.ref(freeze['label_plan'], 'label_plan')); need(freeze['label_plan']['sha256'] == LABEL_SHA, 'Frozen case selection changed')
    cases = manual_cases(plan, hashes); close(list(cases), freeze['case_ids'], 'Frozen case order')
    doc = read(job / 'items.json'); seal = read(job / 'items_seal.json')
    hashes.ref({'path': str(job / 'items.json'), 'sha256': seal['sha256']}, 'measurement_items')
    need(doc['kind'] == 'completion_measurement_inputs' and doc['data_role'] == ROLE and doc['independent_final_test'] is False
         and doc['previously_exposed_subjects'] is True, 'Measurement job incorrectly claims blind/final scope')
    close(doc['freeze'], {'path': str(freeze_path), 'sha256': FREEZE_SHA}, 'Job freeze')
    close((doc['protocol_sha256'], seal['protocol_sha256']), (FREEZE_SHA, FREEZE_SHA), 'Job seal freeze')
    close((doc['measurement_policy_sha256'], seal['policy_sha256']), (POLICY_SHA, POLICY_SHA), 'Job policy')
    close(doc['case_metadata'], cases, 'Case metadata from human files')
    planned = [{'method': method, 'subject': row['subject'], 'case_id': cid, 'seed': seed}
               for method in METHODS for cid, row in cases.items() for seed in SEEDS]
    close(mapped(doc['planned']), mapped(planned), 'All768 planned logical rows')
    references = read(hashes.ref(doc['reference_manifest'], 'reference_manifest'))
    need(references['kind'] == 'completion_evaluation_references' and references['status'] == 'complete' and references['data_role'] == ROLE
         and references['protocol_sha256'] == FREEZE_SHA and references['measurement_policy_sha256'] == POLICY_SHA, 'Reference manifest not accepted')
    raw, cache, ref_cases = verify_raw_references(freeze, plan, references, hashes)
    indexes = []
    for index_ref in doc['generation_indexes']:
        index = read(hashes.ref(index_ref, 'generation_index'))
        if index['kind'] == 'completion_v11_generation':
            config = read(hashes.ref(index['configuration'], 'generation_configuration'))
            hashes.nested(config, 'generation_source_cache_and_geometry_binding')
            if 'runtime' in index:
                runtime = read(hashes.ref(index['runtime'], 'generation_runtime'))
                hashes.nested(runtime, 'generation_runtime_source_artifact')
            close(config['freeze'], doc['freeze'], 'Our generator freeze')
            close(config['source_cache'], references['source_cache_manifest'], 'Our canonical source cache')
            close(config['policy'], freeze['generation']['policy'], 'Our frozen V11 settings')
            need(index['target_pixels_used'] is False and index['target_latents_computed'] is False, 'Our target pixel leakage')
            for row in index['records']:
                expected = ref_cases[row['case_id']]['source_reference']
                if expected['status'] == 'complete':
                    close(row['source_real'], expected['image'], 'Our per-output source identity')
            index = {**index, 'method_names': OURS, 'case_ids': list(cases), 'sampling_seeds': SEEDS}
        else:
            close(index['protocol_sha256'], FREEZE_SHA, 'External freeze')
            close(index['raw_manifest'], references['raw_manifest'], 'External native source/driver raw binding')
        indexes.append(index)
    generations, accounting = normalize_generation(indexes, cases)
    close(mapped(doc['normalized_generation_records']), mapped(generations), 'Original to normalized generation records')
    close(doc['execution_accounting'], accounting, 'Physical/alias accounting')
    available = mapped(doc['items']); gen = mapped(generations)
    allkeys = set(mapped(planned)) | {(role, c['subject'], cid, 0) for role in ('source_real', 'target_real') for cid, c in cases.items()}
    availability = {tuple(r['key']): r for r in doc['availability']}
    need(len(availability) == len(doc['availability']), 'Duplicate availability key')
    close(set(availability), allkeys, 'All832 generation/reference availability keys')
    close(set(available), {k for k, r in availability.items() if r['status'] == 'available'}, 'Available item set')
    for k in allkeys:
        if k[0] in METHODS:
            generation = gen.get(k)
            state = 'missing_output' if generation is None else generation['status']
            if state != 'complete':
                close(availability[k]['status'], state, 'Failed/missing output availability')
            elif k not in available:
                close(availability[k]['status'], 'unavailable', 'Invalid completed image availability')
    for k, item in available.items():
        for field, value in cases[item['case_id']].items():
            close(item[field], value, 'Available metadata')
        expected = ref_cases[item['case_id']][item['method'].replace('_real', '_reference')]['image'] if item['method'].endswith('_real') else gen[k]['output']
        close(Path(item['path']).resolve(), Path(expected['path']).resolve(), 'Measurement image source')
        close(item['sha256'], expected['sha256'], 'Measurement image hash')
        hashes.ref(item, 'evaluated_image_hash_only')
    for row in generations:
        if row['status'] == 'complete':
            hashes.ref(row['output'], 'generated_output_hash_only')
    payloads = {}
    for kind in ('openface', 'facenet', 'buffalo'):
        binding = read(job / kind / 'binding.json')
        close(binding['protocol_sha256'], FREEZE_SHA, 'Measurement binding freeze')
        close(binding['measurement_policy_sha256'], POLICY_SHA, 'Measurement binding policy')
        close(binding['input_sha256'], seal['sha256'], 'Measurement items binding')
        payloads[kind] = read(hashes.ref({'path': str(job / kind / 'results.json'), 'sha256': binding['results_sha256']}, 'measurement_results'))
        close(payloads[kind]['policy_sha256'], POLICY_SHA, 'Measurement policy')
    core = independent_core()
    am, parsed, csv_audit, csv_failures = verify_openface(job, payloads['openface'], available, hashes, core)
    identities = {}
    for kind in ('facenet', 'buffalo'):
        payload = payloads[kind]
        need(payload.get('status', payload.get('summary', {}).get('status')) == 'complete', 'Identity run not terminal')
        identities[kind] = mapped(payload['records']); close(set(identities[kind]), set(available), 'Identity availability')
        for k, row in identities[kind].items():
            close(row['sha256'], available[k]['sha256'], 'Identity image SHA')
            close(row['split'], ROLE, 'Identity data role')
            value = row.get('identity_cosine')
            need(value is None or type(value) in (int, float) and math.isfinite(value) and -1.000001 <= value <= 1.000001, 'Invalid identity readback scalar')
    allpath = job / 'statistics/all_case_seed_metrics.json'; all_metrics = read(allpath)
    need(all_metrics['status'] == 'complete' and all_metrics['data_role'] == ROLE and all_metrics['protocol_sha256'] == FREEZE_SHA, 'Metric export not accepted')
    close(all_metrics['availability'], doc['availability'], 'Exported availability')
    close(all_metrics['execution_accounting'], accounting, 'Exported accounting')
    independent = independent_metric_rows(doc, all_metrics['records'], generations, am, parsed, identities, cases, spec, core)
    input_path = job / 'statistics/paired_subject_input.json'; result_path = job / 'statistics/paired_subject_results.json'
    stat_input = read(input_path); stat_result = read(result_path)
    close(stat_result['input_sha256'], sha(input_path), 'Statistics input SHA')
    close(stat_result['statistics_source_sha256'], evaluation['algorithm_sources']['statistics']['sha256'], 'Statistics algorithm binding')
    for metadata in (stat_input['metadata'], stat_result['metadata']):
        need(metadata['data_role'] == ROLE and metadata['independent_final_test'] is False and metadata['previously_exposed_subjects'] is True, 'Statistics incorrectly claims independence')
        close(metadata['protocol_sha256'], FREEZE_SHA, 'Statistics freeze')
        close(metadata['measurement_policy_sha256'], POLICY_SHA, 'Statistics measurement policy')
        close(metadata['physical_execution_accounting'], accounting, 'Statistics physical denominator')
        close(metadata['primary_metric'], evaluation['primary_metric'], 'Primary endpoint binding')
        close(metadata['primary_family'], evaluation['primary_family'], 'Primary family binding')
        need(metadata['alias_seed_rows_are_not_independent_generations'] is True, 'Aliases treated as independent generations')
    statistics = verify_statistics(stat_input, stat_result, independent, planned, cases, spec)
    # Rehash all input records once at the end to detect changes during this audit.
    for path, expected in hashes.checked.items():
        need(sha(path) == expected, 'Artifact changed during independent audit: ' + path)
    return {'status': 'PASS', 'data_role': ROLE, 'independent_final_test': False, 'previously_exposed_subjects': True,
        'job': str(job), 'protocol_sha256': FREEZE_SHA, 'items_sha256': seal['sha256'], 'statistics_input_sha256': sha(input_path),
        'statistics_results_sha256': sha(result_path), 'all_case_seed_metrics_sha256': sha(allpath),
        'planned_logical_outputs': 768, 'planned_physical_jobs': 704,
        'actual_started_physical_jobs': sum(v['executed_physical_calls'] for v in accounting.values()),
        'successful_physical_jobs': sum(v['successful_physical_calls'] for v in accounting.values()),
        'case_scope_counts': {'all': 32, 'edited': 24, 'no_edit': 8}, 'execution_accounting': accounting,
        'manual_label_files_verified': 96, 'metric_sources_recomputed_or_bound': 14, 'scoped_metrics': len(spec['metrics']),
        'CSV_audit': csv_audit, 'CSV_measurement_failures_retained': csv_failures,
        'verified_reference_counts': {kind: len(paths) for kind, paths in sorted(hashes.kinds.items())},
        **statistics, 'script_sha256': sha(__file__), 'statistics_check_sha256': sha(HERE / 'statistics_check.py'),
        'independent_prior_arithmetic_sha256': OLD_VERIFIER_SHA, 'independent_csv_reader_sha256': core.READER_SHA,
        'identity_verification_scope': {kind: {'embeddings_saved': payloads[kind].get('embeddings_saved', False),
            'verified': 'bound results JSON, image/source-case SHA provenance and scalar export only', 'cosine_independently_recomputed': False} for kind in identities},
        'not_independently_recomputed': ['FaceNet/buffalo feature embeddings and cosine arithmetic (embeddings not read/recomputed)',
            'visual quality or actual image dimensions (hash verification only, no image decoding)'],
        'noedit_interpretation': 'source VAE reconstruction by construction; not learned identity evidence',
        'models_loaded_or_called': 0, 'images_decoded_or_viewed': 0, 'source_or_evaluator_files_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=COMPLETION / 'evaluation/all_methods_v1')
    parser.add_argument('--freeze', type=Path, default=COMPLETION / 'frozen_protocol.json')
    parser.add_argument('--freeze-sha', default=FREEZE_SHA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true', help='Run only after main explicitly authorizes terminal verification')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'ready_no_artifact_inspection', 'job': str(args.job), 'expected_logical': 768,
            'expected_physical_plan': 704, 'models_loaded': 0, 'images_read': 0, 'execution_requires_terminal_authorization': True})); return
    output = (args.output or HERE / ('verification_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.json')).resolve()
    need(HERE in output.parents and not output.exists(), 'New report under review_tools required')
    result = verify(args.job, args.freeze, args.freeze_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': 'PASS', 'output': str(output), 'planned_logical': 768, 'planned_physical': 704,
        'actual_started': result['actual_started_physical_jobs'], 'bootstrap_recomputed': True, 'identity_cosines_recomputed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
