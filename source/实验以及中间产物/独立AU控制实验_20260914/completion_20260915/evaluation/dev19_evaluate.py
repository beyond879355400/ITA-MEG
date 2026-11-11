"""Existing19 DEV-only evaluation. Main frozen postfreeze evaluator is read-only."""
import argparse
import ast
import copy
import importlib
import itertools
import math
from pathlib import Path
import sys
import contract as c
import evaluate as main_eval

HERE = Path(__file__).resolve().parent
TOOLS = HERE / 'dev19_tools'
EXTERNAL = ['magicface', 'faceadapter', 'liveportrait', 'liveportrait_absolute']
METHODS = c.OURS + EXTERNAL
PRIOR_PINS = {
    'labels': ('development_evaluation/dev_coverage_audit/dev19_manifest_2k.json', '90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28'),
    'manifest': ('development_evaluation/evaluation_policy_v2/comparison_nine/case_output_manifest.json', '9fa2105acb99e319f6adac2a329f7b6a3ca4c324e8920f7d5f5f2c64261ca4d8'),
    'results': ('development_evaluation/evaluation_policy_v2/comparison_nine/results.json', '352836101ebdaa9fb5cf8a95990ddb951c9677313d0b24434a2a4866b2e64e04'),
}
MAIN_PINS = {'contract.py': 'babf8b3075eb905e8d2f748ca39cb4b5075971abcfc5a7b9d093983fee633cd7',
    'evaluate.py': 'c7691fa1cbd504d12cadb0b133ebc32578a694bae354c55ac89dcebae0fc28b7',
    'spec_v2.json': '068c437b912daefcecfaf0dc93cc8dc629bb33751c7c1f4ae41e6acf261d2e4f'}


def check_main():
    for name, expected in MAIN_PINS.items():
        c.need(c.sha(HERE / name) == expected, 'Main frozen evaluator changed: ' + name)
    c.need(c.read(HERE / 'spec_v2.json') == c.execution_spec(), 'Frozen main dependency changed')


def helpers():
    check_main()
    c.legacy('evaluation_export')  # Import the hash-pinned pure metric helper chain.
    return importlib.import_module('summarize_v2')


def same_number(a, b, tolerance=1e-12):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def validate_prior(labels_doc, manifest, report):
    """Pure metadata checks: exact19 cases, human labels, native inputs and scores."""
    c.need(labels_doc.get('test_data_used') is False and report.get('test_used') is False and manifest.get('test_used') is False, 'Only existing DEV evidence permitted')
    c.need(report.get('status') == 'complete' and manifest.get('status') == 'frozen_complete', 'Prior comparison incomplete')
    c.need(report['policy_sha256'] == manifest['policy_sha256'] == c.POLICY_SHA, 'Prior measurement policy differs')
    labels = labels_doc['cases']
    ids = [r['case_id'] for r in labels]
    c.need(len(ids) == len(set(ids)) == 19 and ids == labels_doc['original_case_ids'] + labels_doc['additional_case_ids'], 'Full original19-case order required')
    c.need({r['subject'] for r in labels} == {'SN004', 'SN012', 'SN017', 'SN018'}, 'Original4 DEV subjects required')
    declared = {r['case_id']: r for r in manifest['cases']}
    c.need(len(declared) == 19 and set(declared) == set(ids), 'Prior manifest case coverage changed')
    rows = {(r['case_id'], r['method']): r for r in report['records']}
    c.need(len(rows) == len(report['records']) == 209, 'Prior9 settings plus2 references must cover all19')
    result, reused, refs, cells = {}, [], [], set()
    metric = helpers()
    for label in labels:
        cid = label['case_id']
        case = declared[cid]
        c.need(label['split'] == case['split'] == 'dev' and label['subject'] == case['subject'], 'Prior role/subject changed')
        source, target = label['source_au_vector_0_5'], label['target_au_vector_0_5']
        c.need(len(source) == len(target) == 12 and not any(source) and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 5 for v in source + target), 'Invalid12AU labels')
        for i, au in enumerate(c.AUS):
            if source[i] != target[i] and target[i] in (1, 2, 3): cells.add((au, target[i]))
        for role, expected_labels in (('source', source), ('target', target)):
            frame = case['reference_frames'][role]
            r = rows[cid, role + '_real']
            c.need(frame['manual_AU_0_5'] == expected_labels and r['source_labels_12'] == source and r['target_labels_12'] == target, 'Prior reference manual labels changed')
            c.need(r['sha256'] == frame['evaluation_sha256'] and Path(r['path']).resolve() == Path(frame['evaluation_path']).resolve(), 'Prior real-reference image binding mismatch')
            refs += [{'path': frame['evaluation_path'], 'sha256': frame['evaluation_sha256']},
                     {'path': frame['raw_frame_path'], 'sha256': frame['raw_frame_sha256']}]
        original = rows[cid, 'source_real']
        meta = {k: copy.deepcopy(original[k]) for k in ('case_id', 'subject', 'split', 'group', 'cohort', 'source_labels_12', 'target_labels_12')}
        c.need((meta['group'] == 'no_edit') == (source == target), 'Prior group/labels mismatch')
        outputs = {r['method']: r for r in case['outputs']}
        c.need(len(outputs) == 9, 'Original9 setting manifest required')
        for method in ['source_real', 'target_real'] + EXTERNAL:
            row = copy.deepcopy(rows[cid, method])
            c.need(row['policy_sha256'] == c.POLICY_SHA and all(row[k] == meta[k] for k in meta), 'Reused row role/labels/subject differs')
            fn = row['facenet_primary']
            c.need(fn['sha256'] == row['sha256'] and fn['policy_sha256'] == c.POLICY_SHA and same_number(row['identity_cosine'], fn['identity_cosine']), 'Reused FaceNet output binding differs')
            if method in EXTERNAL:
                output = outputs[method]
                request = output['original_inference_input']
                c.need(row['sha256'] == output['output_sha256'] and Path(row['path']).resolve() == Path(output['output_path']).resolve(), 'External output binding changed')
                c.need(output['request_seed_key'] == request['seed'] == 101 and request['request_case_id'] == cid and request['subject'] == meta['subject'] and request['split'] == 'dev', 'External request case/subject/seed differs')
                c.need(request['source_au_vector_0_5'] == source and request['target_au_vector_0_5'] == target and request['au_order'] == c.AUS, 'External requested labels/order changed')
                c.need(request['source_sha256'] == case['reference_frames']['source']['raw_frame_sha256'], 'External raw source differs')
                if method == 'magicface':
                    c.need(output['extra_real_target_driver'] is False and request['au_delta_native'] == [b-a for a, b in zip(source, target)], 'MagicFace native AU request changed')
                else:
                    c.need(output['extra_real_target_driver'] is True and request['target_sha256'] == case['reference_frames']['target']['raw_frame_sha256'], 'External raw target driver differs')
                refs.append({'path': output['output_path'], 'sha256': output['output_sha256']})
                row['seed'] = 101
                row['inference_binding'] = output
            else:
                row['seed'] = 0
            recomputed = metric.e.au_metrics(row['vector'], original['vector'], source, target)
            for name in ('target_mae_changed', 'delta_mae_changed', 'unchanged_source_relative_drift', 'target_mae_all12', 'delta_mae_all12', 'direction_correct_fraction'):
                c.need(same_number(row.get(name), recomputed[name]), 'Stored prior metric differs from frozen formula: ' + name)
            extra, mean = metric.extra_errors(row['selected_au17'], rows[cid, 'target_real']['selected_au17'], original['selected_au17'])
            c.need(same_number(row['extra5_proxy_mae_to_real_target'], mean) and row['extra5'] == extra, 'Stored prior extra5 differs')
            row['measurement_reused'] = True
            reused.append(row)
        result[cid] = {**meta, 'reference_frames': case['reference_frames']}
    c.need(sum(r['group'] == 'no_edit' for r in result.values()) == 4 and len(cells) == 36, 'Expected4 noedit/15edited/36AU-level cells')
    return result, reused, refs, sorted(cells)


def load_prior(verify_images=False):
    check_main()
    documents, refs = {}, {}
    for name, (relative, expected) in PRIOR_PINS.items():
        path = c.EXP / relative
        c.need(c.sha(path) == expected, 'Frozen prior DEV artifact changed: ' + name)
        documents[name], refs[name] = c.read(path), c.ref(path)
    c.need(documents['manifest']['evaluation_results_sha256'] == refs['results']['sha256'], 'Manifest binds another prior comparison')
    meta, reused, image_refs, cells = validate_prior(documents['labels'], documents['manifest'], documents['results'])
    if verify_images:
        for image_ref in {r['path']: r for r in image_refs}.values(): c.verified(image_ref)
    return {'metadata': meta, 'reused_rows': reused, 'refs': refs, 'cells': cells, 'images_sha_verified': verify_images}


def validate_new_generation(result, cases_doc, config, prior):
    meta = prior['metadata']
    c.need(result.get('kind') == config.get('kind') == 'selected_v11_existing_dev19_completion' and result['split'] == config['split'] == 'dev', 'Wrong generated run role')
    c.need(result['status'] in ('complete', 'completed_with_failures') and result['sampling_seeds'] == config['sampling_seeds'] == [101], 'Terminal single-seed DEV run required')
    c.need(result.get('training_performed') is False and result.get('target_pixels_used') is False and result.get('target_latents_computed') is False, 'DEV run target/training boundary differs')
    c.need(config['case_ids'] == list(meta) and config['variants'] == c.VARIANTS, 'DEV cases/variants changed')
    for name, wanted in (('cases', 'labels'), ('comparison', 'manifest')):
        c.need(config['inputs'][name] == prior['refs'][wanted], 'New run binds another original DEV plan')
    c.need(cases_doc['kind'] == 'existing_dev19_source_only_cases' and cases_doc['split'] == 'dev' and cases_doc['case_count'] == 19, 'DEV source manifest mismatch')
    sources = {r['source_id']: r for r in cases_doc['sources']}
    c.need(len(sources) == 4 and [r['case_id'] for r in cases_doc['cases']] == list(meta), 'DEV source/case coverage changed')
    for case in cases_doc['cases']:
        truth = meta[case['case_id']]
        c.need(case['subject'] == truth['subject'] and case['split'] == 'dev' and case['source_au_vector_0_5'] == truth['source_labels_12'] and case['target_au_vector_0_5'] == truth['target_labels_12'], 'New case labels differ')
        source = sources[case['source_id']]
        c.need(source['subject'] == truth['subject'] and source['canonical_source']['sha256'] == truth['reference_frames']['source']['evaluation_sha256'], 'New shared source differs from original case canonical image')
    plan = [{'method': method, 'subject': truth['subject'], 'case_id': cid, 'seed': 101} for method in c.OURS for cid, truth in meta.items()]
    expected = {c.row_key(r) for r in plan}
    records = {}
    for original in result['records']:
        row = copy.deepcopy(original)
        key = c.row_key(row)
        c.need(key in expected and key not in records and row['split'] == 'dev' and row['status'] in ('complete', 'failed', 'not_started'), 'Duplicate/unplanned DEV output')
        truth = meta[row['case_id']]
        c.need(row['source_aus_0_5'] == truth['source_labels_12'] and row['target_aus_0_5'] == truth['target_labels_12'], 'DEV generated labels changed')
        c.need(row['source_real']['sha256'] == truth['reference_frames']['source']['evaluation_sha256'], 'DEV output source image differs')
        c.need(type(row.get('execution_started')) is bool and isinstance(row.get('execution_id'), str) and row['execution_id'], 'Missing execution accounting')
        c.need(row.get('alias_of') is None and (row['status'] != 'not_started' or row['execution_started'] is False), 'DEV stochastic execution/alias status invalid')
        if row['status'] == 'complete':
            c.need(row['execution_started'] and isinstance(row.get('output'), str) and isinstance(row.get('sha256'), str), 'Completed DEV output requires image and actual call')
        else:
            if row.get('output') is not None or row.get('sha256') is not None:
                row['failed_artifact'] = {'path': row.get('output'), 'sha256': row.get('sha256'), 'eligible_for_evaluation': False}
                row['output'], row['sha256'] = None, None
        records[key] = row
    c.need(result.get('planned') == 95 and len(plan) == 95, 'Complete95 output denominator required')
    if 'completed' in result:
        c.need(result['completed'] == sum(r['status'] == 'complete' for r in records.values()), 'Declared completed count differs')
    c.need(len({r['execution_id'] for r in records.values()}) == len(records), 'DEV physical execution ID reused')
    return plan, records


def prepare(run, result_sha, output):
    run = Path(run).resolve()
    c.need(run.parent == c.COMPLETION / 'dev19_runs', 'Use completion/dev19_runs native result')
    result_path = run / 'result.json'
    c.need(c.sha(result_path) == result_sha, 'Explicit finished generation resultSHA required')
    result = c.read(result_path)
    config = c.read(c.verified(result['configuration']))
    cases_doc = c.read(run / 'cases.json')
    prior = load_prior(verify_images=True)
    plan, generated = validate_new_generation(result, cases_doc, config, prior)
    c.need(Path(output).resolve().is_relative_to(TOOLS), 'DEV job must be inside dev19_tools')
    out = c.new_directory(output)
    items, availability = [], []
    for row in prior['reused_rows']:
        if row['method'] not in ('source_real', 'target_real'): continue
        meta = {k: row[k] for k in ('case_id', 'subject', 'split', 'group', 'cohort', 'source_labels_12', 'target_labels_12', 'method', 'seed')}
        role = 'real_source' if row['method'] == 'source_real' else 'real_target'
        items.append(main_eval.inspect_image_item(meta, {'path': row['path'], 'sha256': row['sha256']}, role))
    for planned in plan:
        row = generated.get(c.row_key(planned))
        if row is None or row['status'] != 'complete':
            availability.append({**planned, 'status': row['status'] if row else 'missing_output'})
            continue
        try:
            c.verified(row['source_real'])
            truth = prior['metadata'][row['case_id']]
            meta = {k: truth[k] for k in ('case_id', 'subject', 'split', 'group', 'cohort', 'source_labels_12', 'target_labels_12')}
            item = main_eval.inspect_image_item({**meta, **planned}, {'path': row['output'], 'sha256': row['sha256']}, 'generated')
            items.append(item)
            availability.append({**planned, 'status': 'available', 'image_size': item['image_size']})
        except Exception as error:
            availability.append({**planned, 'status': 'unavailable', 'error': str(error)})
    document = {'kind': 'selected_v11_dev19_evaluation_inputs', 'status': 'prepared', 'data_role': 'dev', 'test_used': False,
        'independent_final_test': False, 'case_count': 19, 'noedit_cases': 4, 'edited_cases': 15,
        'methods': METHODS, 'planned_new_generations': 95, 'seed': 101, 'items': items,
        'planned': plan, 'normalized_generation_records': list(generated.values()), 'availability': availability,
        'case_metadata': {cid: {k: v for k, v in r.items() if k != 'reference_frames'} for cid, r in prior['metadata'].items()},
        'reused_rows': prior['reused_rows'], 'prior_refs': prior['refs'], 'coverage_cells': prior['cells'],
        'generation_result': c.ref(result_path), 'generation_configuration': result['configuration'], 'generation_cases': c.ref(run / 'cases.json'),
        'policy_sha256': c.POLICY_SHA, 'script_sha256': c.sha(__file__), 'main_frozen_files': MAIN_PINS,
        'buffalo_new_measurement': False, 'source_target_AU_reuse': True,
        'FaceNet_real_references_reencoded_for_new_embedding_similarity': True}
    c.save(out / 'items.json', document)
    c.save(out / 'items_seal.json', {'sha256': c.sha(out / 'items.json'), 'policy_sha256': c.POLICY_SHA})
    return out


def job_control(job):
    check_main()
    job = Path(job).resolve()
    c.need(job.is_relative_to(TOOLS), 'DEV job outside dev19_tools')
    doc, seal = c.read(job / 'items.json'), c.read(job / 'items_seal.json')
    c.need(c.sha(job / 'items.json') == seal['sha256'] and doc['policy_sha256'] == seal['policy_sha256'] == c.POLICY_SHA, 'DEV job binding changed')
    c.need(doc['kind'] == 'selected_v11_dev19_evaluation_inputs' and doc['data_role'] == 'dev' and doc['test_used'] is False, 'DEV job role changed')
    c.need(doc['script_sha256'] == c.sha(__file__), 'DEV evaluation code changed after prepare')
    for reference in [doc['generation_result'], doc['generation_configuration'], doc['generation_cases'], *doc['prior_refs'].values()]: c.verified(reference)
    return job, doc


def derive_facenet(job, input_sha, count, source_count):
    core, text = main_eval.derive_identity('facenet', job, input_sha, count, source_count)
    class DevelopmentMetadata(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == c.ROLE: node.value = 'dev'
            return node
        def visit_Dict(self, node):
            self.generic_visit(node)
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'test_used': node.values[i] = ast.Constant(False)
            return node
    return core, ast.unparse(ast.fix_missing_locations(DevelopmentMetadata().visit(ast.parse(text)))) + '\n'


def measure(job, kind):
    job, doc = job_control(job)
    out = job / kind
    c.need(not out.exists(), 'Never overwrite DEV measurements')
    for item in doc['items']: c.verified({'path': item['path'], 'sha256': item['sha256']})
    if kind == 'openface':
        out.mkdir()
        generated = [r for r in doc['items'] if r['method'] in c.OURS]
        records, physical = main_eval.openface_items(generated, out)
        # The reusable main algorithm attaches its own role; restore the actual DEV role explicitly.
        for row in records: row.update(data_role='dev', test_used=False, independent_final_test=False)
        refs = [copy.deepcopy(r) for r in doc['reused_rows'] if r['method'] in ('source_real', 'target_real')]
        c.save(out / 'results.json', {'status': 'complete', 'data_role': 'dev', 'test_used': False,
            'policy_sha256': c.POLICY_SHA, 'records': records + refs, 'new_unique_measurements': physical,
            'reused_real_reference_rows': len(refs), 'new_generated_logical_rows': len(records)})
    else:
        c.need(kind == 'facenet', 'No new buffalo or other model is authorized for DEV diagnostics')
        source_count = sum(r['method'] == 'source_real' for r in doc['items'])
        core, text = derive_facenet(job, c.sha(job / 'items.json'), len(doc['items']), source_count)
        namespace = {'__name__': 'selected_dev19_native_facenet', '__file__': str(Path(__file__).resolve())}
        exec(compile(text, str(core), 'exec'), namespace)
        namespace['main']()
        with (out / 'derived_snapshot.py').open('x') as stream: stream.write(text)
    c.save(out / 'binding.json', {'data_role': 'dev', 'test_used': False, 'input_sha256': c.sha(job / 'items.json'),
        'results_sha256': c.sha(out / 'results.json'), 'policy_sha256': c.POLICY_SHA})
    return out / 'results.json'


def comparison_report(rows, availability):
    helper = helpers()
    mapping = {(r['case_id'], r['method']): r for r in rows}
    c.need(len(mapping) == len(rows) == 209, 'Full19×(new5+external4+refs2) rows required, including NA')
    pairs = {a + '_vs_' + b: [a, b] for a, b in itertools.combinations(METHODS, 2) if a in c.OURS or b in c.OURS}
    cohorts = {scope: [r for r in rows if scope == 'all' or r['cohort'] == scope] for scope in ('all', 'original', 'additional')}
    extra = ['buffalo_identity_cosine', 'extra5_proxy_mae_to_real_target'] + [f'extra_AU{a}_error_to_real_target' for a in helper.EXTRA]
    coverage = {}
    for method in ['source_real', 'target_real'] + METHODS:
        coverage[method] = {}
        for scope in ('all', 'edited', 'no_edit'):
            part = [r for r in rows if r['method'] == method and (scope == 'all' or (r['group'] == 'no_edit') == (scope == 'no_edit'))]
            coverage[method][scope] = {'planned': len(part), 'AU_valid': sum(r['vector'] is not None for r in part),
                'FaceNet_valid': sum(r['identity_cosine'] is not None for r in part),
                'buffalo_valid': sum(r.get('buffalo_identity_cosine') is not None for r in part)}
    return {'status': 'complete', 'data_role': 'dev', 'test_used': False, 'independent_final_test': False,
        'same_training_protocol_sota_ranking': False, 'case_count': 19, 'edited_cases': 15, 'noedit_cases': 4,
        'setting_count': 9, 'LP_is_one_method_two_settings': True, 'methods': METHODS, 'records': rows,
        'coverage': coverage, 'availability': availability,
        'aggregation': 'per-metric common valid cases; case mean then equal subject means; seed101 only',
        'pairwise': {name: {cohort: helper.compare(part, methods) for cohort, part in cohorts.items()} for name, methods in pairs.items()},
        'all_methods_common_valid': {cohort: helper.compare(part, METHODS) for cohort, part in cohorts.items()},
        'au_level_pairwise': {name: helper.strata_helper.stratify(rows, methods) for name, methods in pairs.items()},
        'au_level_all_methods': helper.strata_helper.stratify(rows, METHODS),
        'extra_diagnostics': {'pairwise': {name: helper.other_compare(rows, methods, extra) for name, methods in pairs.items()},
            'all_methods': helper.other_compare(rows, METHODS, extra)},
        'no_new_buffalo': True, 'no_hypothesis_selection_or_p_values': True,
        'comparison_scope': 'Full19 already-exposed development cases; new5 selected-controller variants vs matched external4 settings, not the prior4case mean or independent testing',
        'input_groups': {'source_plus_text_manual_AU': c.OURS, 'source_plus_manual_AU_native': ['magicface'],
            'source_plus_real_driver': ['faceadapter', 'liveportrait', 'liveportrait_absolute']}}


def export(job):
    job, doc = job_control(job)
    measurements = {}
    available = {(r['case_id'], r['method'], r['seed']): r for r in doc['items']}
    for kind in ('openface', 'facenet'):
        data, binding = c.read(job / kind / 'results.json'), c.read(job / kind / 'binding.json')
        c.need(binding['input_sha256'] == c.sha(job / 'items.json') and binding['results_sha256'] == c.sha(job / kind / 'results.json') and binding['policy_sha256'] == c.POLICY_SHA, 'DEV measurement seal mismatch')
        seen = set()
        for row in data['records']:
            key = row['case_id'], row['method'], row['seed']
            c.need(key in available and key not in seen and row['sha256'] == available[key]['sha256'], 'DEV measurement logical key/SHA differs')
            seen.add(key)
        c.need(seen == set(available), 'Every available DEV item needs a measurement row, including invalid faces')
        measurements[kind] = data
    merge = c.legacy('evaluation_export').merge_measurements
    new_rows = merge(doc, measurements['openface'], measurements['facenet'], {'records': []})
    generated = {c.row_key(r): r for r in doc['normalized_generation_records']}
    fn = {(r['case_id'], r['method']): r for r in measurements['facenet']['records']}
    for row in new_rows:
        g = generated.get(c.row_key(row))
        row['generation_status'] = g['status'] if g else 'missing_output'
        row['generation_record'] = g
        row['vector'] = row['AU_measurement']['vector'] if row['AU_measurement'] else None
        row['selected_au17'] = row['AU_measurement'].get('selected_au17') if row['AU_measurement'] else None
        row['facenet_primary'] = row['FaceNet_measurement']
        au = row['AU_measurement']
        row['path'] = au['path'] if au else g.get('output') if g else None
        row['sha256'] = au['sha256'] if au else g.get('sha256') if g else None
        for field in ('raw_rows', 'selected_row', 'all_candidates', 'confidence', 'parse_status', 'image_size', 'returncode'):
            row[field] = au.get(field) if au else None
        row['buffalo_auxiliary'] = None
        row['buffalo_identity_cosine'] = None
        if row['generation_status'] != 'complete':
            for key in c.SOURCE_METRICS: row[key] = None
            row['vector'], row['selected_au17'] = None, None
    reused = copy.deepcopy(doc['reused_rows'])
    parity = []
    for row in reused:
        if row['method'] not in ('source_real', 'target_real'): continue
        actual = fn[row['case_id'], row['method']]
        parity.append({'case_id': row['case_id'], 'method': row['method'], 'old': row['identity_cosine'],
            'new': actual['identity_cosine'], 'match': same_number(row['identity_cosine'], actual['identity_cosine'], 1e-6)})
        row['previous_facenet_primary'] = row['facenet_primary']
        row['facenet_primary'], row['identity_cosine'] = actual, actual['identity_cosine']
    rows = reused + new_rows
    report = comparison_report(rows, doc['availability'])
    report.update(policy_sha256=c.POLICY_SHA, input_sha256=c.sha(job / 'items.json'), prior_refs=doc['prior_refs'],
        generated_result=doc['generation_result'], reference_facenet_parity=parity,
        comparison_usable=all(r['match'] for r in parity), new_generation_denominator=95,
        reused_reference_rows=38, reused_external_rows=76, no_old_internal_outputs_used=True)
    out = job / 'comparison'
    out.mkdir(exist_ok=False)
    c.save(out / 'results.json', report)
    return out / 'results.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit-prior', 'prepare', 'openface', 'facenet', 'export'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--result-sha')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.phase == 'audit-prior':
        prior = load_prior(verify_images=False)
        print(c.json.dumps({'status': 'prior_metadata_verified', 'cases': len(prior['metadata']), 'cells': len(prior['cells']), 'reused_rows': len(prior['reused_rows']), 'models_called': False}))
        return
    if not args.execute:
        check_main()
        print(c.json.dumps({'status': 'not_executed', 'phase': args.phase, 'models_called': False, 'new_images_read': False}))
        return
    if args.phase == 'prepare': result = prepare(args.run, args.result_sha, args.output)
    elif args.phase == 'export': result = export(args.job)
    else: result = measure(args.job, args.phase)
    print(c.json.dumps({'status': 'complete', 'output': str(result)}))


if __name__ == '__main__':
    main()
