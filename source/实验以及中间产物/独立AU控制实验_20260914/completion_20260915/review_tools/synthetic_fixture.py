"""Build a complete fake terminal artifact graph inside a temporary review folder.

All image/video/feature files contain explicit non-media placeholder bytes.
Nothing in this fixture can be mistaken for real experimental evidence.
"""
import copy
import csv
import json
from pathlib import Path
import verify_completed as verifier


def build(root, original_labels, indexes, contract, engine):
    def blob(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
        return {'path': str(path), 'sha256': verifier.sha(path)}
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return {'path': str(path), 'sha256': verifier.sha(path)}
    def image(path):
        return {**blob(path, b'SYNTHETIC NON-IMAGE PLACEHOLDER'), 'width': 512, 'height': 512, 'mode': 'RGB'}
    labels = copy.deepcopy(original_labels); videos = []
    for subject in verifier.SUBJECTS:
        group = [c for c in labels if c['subject'] == subject]
        video = {**blob(root / 'videos' / (subject + '.avi'), ('SYNTHETIC NON-VIDEO ' + subject).encode()),
            'subject': subject, 'camera': group[0]['camera'], 'width': 1024, 'height': 768}
        videos.append(video)
        for case in group: case['video_path'] = video['path']
    label_files = []
    for subject in verifier.SUBJECTS:
        group = [c for c in labels if c['subject'] == subject]
        for index, au in enumerate(verifier.AUS):
            lookup = {c['target_frame_1based']: c['target_au_vector_0_5'][index] for c in group}
            values = ''.join(f'{frame},{lookup.get(frame, 0)}\n' for frame in range(1, 4846))
            label_files.append({**blob(root / 'labels' / subject / f'au{au}.txt', values.encode()), 'subject': subject, 'au': au})
    plan_ref = save(root / 'label_plan.json', {'cases': labels, 'labels': label_files})
    video_ref = save(root / 'video_refs.json', {'status': 'verified', 'videos': videos, 'proposal': plan_ref})
    cases = {c['case_id']: {'case_id': c['case_id'], 'subject': c['subject'], 'split': verifier.ROLE,
        'group': 'no_edit' if c['source_au_vector_0_5'] == c['target_au_vector_0_5'] else 'edited',
        'source_labels_12': c['source_au_vector_0_5'], 'target_labels_12': c['target_au_vector_0_5']} for c in labels}
    policy = {'synthetic_fixed_policy': True}
    stats_ref = {'path': str(verifier.EXP / 'statistics/paired_subject_statistics.py'),
        'sha256': verifier.sha(verifier.EXP / 'statistics/paired_subject_statistics.py')}
    freeze_ref = save(root / 'frozen_protocol.json', {'kind': 'completion_v11_frozen_protocol', 'status': 'frozen',
        'data_role': verifier.ROLE, 'independent_subject_test': False, 'label_plan': plan_ref,
        'case_ids': list(cases), 'extraction': {'inputrefs': video_ref}, 'generation': {'policy': policy},
        'evaluation': {'methods': verifier.METHODS, 'sampling_seeds': verifier.SEEDS, 'analysis_spec': contract.analysis_spec(),
            'primary_metric': contract.PRIMARY, 'primary_family': 'external_primary_C',
            'families': {'external_primary_C': 3, 'internal_primary_C': 4}, 'algorithm_sources': {'statistics': stats_ref}}})
    raw_rows = []; sources = {}; cache_cases = []; raw_frames = {}; canonical = {}
    for label in labels:
        video = next(v for v in videos if v['subject'] == label['subject'])
        for side in ('source', 'target'):
            frame = label[side + '_frame_1based']; k = label['subject'], frame
            if k not in raw_frames:
                raw_frames[k] = {**blob(root / 'raw' / label['subject'] / f'{frame}.png', b'SYNTHETIC NON-IMAGE RAW'),
                    'status': 'complete', 'subject': label['subject'], 'split': verifier.ROLE, 'camera': label['camera'],
                    'width': 1024, 'height': 768, 'mode': 'RGB', 'label_frame_1based': frame, 'video_frame_zero_based': frame - 1,
                    'original_video_path': video['path'], 'original_video_sha256': video['sha256']}
                canonical[k] = image(root / 'canonical' / label['subject'] / f'{frame}.png')
        source = raw_frames[label['subject'], label['source_frame_1based']]
        target = raw_frames[label['subject'], label['target_frame_1based']]
        source_id = label['subject'] + '_source'
        sources[source_id] = {'source_id': source_id, 'subject': label['subject'], 'split': verifier.ROLE, 'status': 'complete',
            'raw_source': source, 'canonical_source': canonical[label['subject'], label['source_frame_1based']]}
        cache_cases.append({**label, 'source_id': source_id})
        raw_rows.append({'case_id': label['case_id'], 'subject': label['subject'], 'split': verifier.ROLE, 'camera': label['camera'],
            'generation_input': {'source_image': source, 'source_au_vector_0_5': label['source_au_vector_0_5'], 'target_au_vector_0_5': label['target_au_vector_0_5']},
            'target_reference_image': target})
    raw_ref = save(root / 'raw_manifest.json', {'protocol_sha256': freeze_ref['sha256'], 'label_plan_sha256': plan_ref['sha256'],
        'status': 'raw_frames_extracted', 'cases': raw_rows, 'post_extraction_integrity_errors': []})
    cache_ref = save(root / 'source_cache.json', {'protocol_sha256': freeze_ref['sha256'], 'status': 'complete',
        'target_pixels_read': False, 'target_latents_computed': False, 'sources': list(sources.values()), 'cases': cache_cases,
        'features_file': blob(root / 'features.pt', b'SYNTHETIC NON-TENSOR BYTES')})
    reference_rows = []
    for label, raw in zip(labels, raw_rows):
        reference_rows.append({**cases[label['case_id']],
            'source_reference': {'status': 'complete', 'raw_image': raw['generation_input']['source_image'], 'image': canonical[label['subject'], label['source_frame_1based']]},
            'target_reference': {'status': 'complete', 'raw_image': raw['target_reference_image'], 'image': canonical[label['subject'], label['target_frame_1based']]}})
    reference_ref = save(root / 'references.json', {'kind': 'completion_evaluation_references', 'status': 'complete', 'data_role': verifier.ROLE,
        'protocol_sha256': freeze_ref['sha256'], 'measurement_policy_sha256': verifier.POLICY_SHA,
        'source_cache_manifest': cache_ref, 'raw_manifest': raw_ref, 'cases': reference_rows})
    configuration_ref = save(root / 'configuration.json', {'freeze': freeze_ref, 'source_cache': cache_ref, 'policy': policy})
    index_refs = []; physical = {}
    for number, index in enumerate(indexes):
        for row in index['records']:
            if row['execution_id'] not in physical:
                physical[row['execution_id']] = image(root / 'generated' / (row['execution_id'] + '.png'))
            row['output'] = {k: physical[row['execution_id']][k] for k in ('path', 'sha256')}
            if number == 0:
                label = next(c for c in labels if c['case_id'] == row['case_id'])
                row['source_real'] = canonical[label['subject'], label['source_frame_1based']]
        if number == 0:
            index.update(kind='completion_v11_generation', configuration=configuration_ref,
                target_pixels_used=False, target_latents_computed=False)
        else:
            index.update(kind='completion_external_generation_index', protocol_sha256=freeze_ref['sha256'], raw_manifest=raw_ref)
        index_refs.append(save(root / f'index_{number}.json', index))
    planned, normalized, accounting = contract.normalize_records(cases, indexes)
    items = []; availability = []
    for label in labels:
        for side in ('source', 'target'):
            img = canonical[label['subject'], label[side + '_frame_1based']]
            items.append({**cases[label['case_id']], 'method': side + '_real', 'seed': 0,
                'path': img['path'], 'sha256': img['sha256'], 'image_size': [512, 512]})
    for p in planned:
        g = normalized[verifier.key(p)]
        items.append({**cases[p['case_id']], **p, **g['output'], 'image_size': [512, 512]})
    for item in items:
        availability.append({'key': list(verifier.key(item)), 'status': 'available',
            'role': 'generated' if item['method'] in verifier.METHODS else 'real_' + item['method'].replace('_real', ''), 'image_size': [512, 512]})
    job = root / 'evaluation/all_methods_v1'
    doc = {'kind': 'completion_measurement_inputs', 'data_role': verifier.ROLE, 'independent_final_test': False,
        'previously_exposed_subjects': True, 'protocol_sha256': freeze_ref['sha256'], 'measurement_policy_sha256': verifier.POLICY_SHA,
        'freeze': freeze_ref, 'reference_manifest': reference_ref, 'generation_indexes': index_refs, 'case_metadata': cases,
        'planned': planned, 'normalized_generation_records': list(normalized.values()), 'execution_accounting': accounting,
        'items': items, 'availability': availability}
    items_ref = save(job / 'items.json', doc)
    save(job / 'items_seal.json', {'sha256': items_ref['sha256'], 'protocol_sha256': freeze_ref['sha256'], 'policy_sha256': verifier.POLICY_SHA})
    digest = items[0]['sha256']; csv_path = job / 'openface/raw' / ('sha_' + digest) / 'synthetic.csv'
    csv_path.parent.mkdir(parents=True)
    csv_row = {'confidence': .9, **{f'x_{i}': i for i in range(68)}, **{f'y_{i}': i % 2 for i in range(68)},
        **{f'AU{au:02d}_r': 0 for au in (*verifier.AUS, *verifier.EXTRA)}}
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_row)); writer.writeheader(); writer.writerow(csv_row)
    core = verifier.independent_core(); raw, selected, vector, values, confidence = core.load_csv_reader()(csv_path)
    measurement = {**items[0], 'sha256': digest, 'raw_rows': raw, 'selected_row': selected, 'face_count': len(raw),
        'vector': vector, 'selected_au17': values, 'confidence': confidence, 'parse_status': 'primary_face_finite',
        'returncode': 0, 'input_unchanged': True, 'csv_files': [str(csv_path)], 'policy_sha256': verifier.POLICY_SHA}
    au_rows = [{**measurement, **item} for item in items]
    payloads = {'openface': {'status': 'complete', 'protocol_sha256': freeze_ref['sha256'], 'measurement_policy_sha256': verifier.POLICY_SHA,
        'policy_sha256': verifier.POLICY_SHA, 'records': au_rows, 'unique_measurements': [measurement]}}
    for kind in ('facenet', 'buffalo'):
        payloads[kind] = {'status': 'complete', 'policy_sha256': verifier.POLICY_SHA, 'embeddings_saved': False,
            'records': [{**item, 'identity_cosine': 1.0} for item in items]}
    for kind, payload in payloads.items():
        result_ref = save(job / kind / 'results.json', payload)
        save(job / kind / 'binding.json', {'protocol_sha256': freeze_ref['sha256'], 'measurement_policy_sha256': verifier.POLICY_SHA,
            'input_sha256': items_ref['sha256'], 'results_sha256': result_ref['sha256']})
    maps = {kind: verifier.mapped(value['records']) for kind, value in payloads.items()}
    metric_rows = []
    for p in planned:
        k = verifier.key(p); case = cases[p['case_id']]
        metrics = core.au_metrics(vector, vector, case['source_labels_12'], case['target_labels_12'], values, values)
        extra = {str(au): {'output': 0., 'real_target_measured': 0., 'real_source_measured': 0.,
            'absolute_error_to_target': 0., 'output_minus_source': 0., 'target_minus_source': 0.} for au in verifier.EXTRA}
        metric_rows.append({**case, **p, **metrics, 'identity_cosine': 1., 'buffalo_identity_cosine': 1., 'extra5': extra,
            'AU_measurement': maps['openface'][k], 'FaceNet_measurement': maps['facenet'][k], 'buffalo_measurement': maps['buffalo'][k]})
    save(job / 'statistics/all_case_seed_metrics.json', {'status': 'complete', 'data_role': verifier.ROLE,
        'protocol_sha256': freeze_ref['sha256'], 'records': metric_rows, 'availability': availability, 'execution_accounting': accounting})
    inputs = contract.statistics_document(doc, metric_rows)
    input_ref = save(job / 'statistics/paired_subject_input.json', inputs)
    results = contract.enrich_coverage(engine.analyze(inputs), inputs, cases)
    results.update(input_sha256=input_ref['sha256'], statistics_source_sha256=stats_ref['sha256'])
    save(job / 'statistics/paired_subject_results.json', results)
    return job, freeze_ref, plan_ref
