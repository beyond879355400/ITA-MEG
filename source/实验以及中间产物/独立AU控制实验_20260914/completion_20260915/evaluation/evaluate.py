"""Frozen-policy evaluation for postfreeze cases. Models require explicit --execute."""
import argparse
import ast
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from contract import *


def inspect_image_item(meta, image_ref, role):
    """Read-only native RGB inspection; only real canonical references are512."""
    need(role in ('real_source', 'real_target', 'generated'), 'Unknown image role')
    from PIL import Image
    path = verified(image_ref)
    with Image.open(path) as image:
        image.load()
        need(image.mode == 'RGB' and min(image.size) > 0, 'Evaluation requires native decodable RGB, without conversion')
        if role != 'generated':
            need(image.size == (512, 512), 'Real reference must be canonical512RGB')
        size = list(image.size)
    need(sha(path) == image_ref['sha256'], 'Input changed during read-only image inspection')
    return {**meta, 'path': str(path), 'sha256': image_ref['sha256'], 'image_size': size}


def source_items(control, cache, sources):
    items, unavailable = [], []
    for row in cache['cases']:
        source = sources[row['source_id']]
        meta = {**control['cases'][row['case_id']], 'method': 'source_real', 'seed': 0}
        if source['status'] == 'complete':
            image = source['canonical_source']
            items.append({**meta, 'path': image['path'], 'sha256': image['sha256'], 'image_size': [512, 512]})
        else:
            unavailable.append({**meta, 'path': None, 'sha256': None, 'vector': None,
                'selected_au17': None, 'raw_rows': [], 'selected_row': None, 'confidence': None,
                'returncode': None, 'parse_status': 'source_unavailable', 'error': source.get('error')})
    return items, unavailable


def expand_openface_records(items, physical):
    bysha = {r['sha256']: r for r in physical}
    return [{**bysha[item['sha256']], **item, 'measurement_source_path': bysha[item['sha256']]['path'],
        'data_role': ROLE, 'independent_final_test': False} for item in items]


def openface_items(items, output):
    core = legacy('openface')
    policy = read(V2 / 'policy.json')
    seal = read(core.TOOLS / 'tool_seal.json')
    need(sha(core.TOOLS / 'tool_seal.json') == policy['openface']['tool_seal_sha256'], 'OpenFace tool seal changed')
    for asset in [seal['binary'], seal['launcher'], *seal['model_files']]:
        need(sha(core.TOOLS / asset['path']) == asset['sha256'], 'OpenFace asset changed')
    unique = {}
    for item in items:
        role = 'real_source' if item['method'] == 'source_real' else 'real_target' if item['method'] == 'target_real' else 'generated'
        inspected = inspect_image_item(item, {'path': item['path'], 'sha256': item['sha256']}, role)
        need(item.get('image_size') == inspected['image_size'], 'Prepared image dimensions changed')
        unique.setdefault(item['sha256'], {**item, 'sample_id': 'sha_' + item['sha256']})
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    physical = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        for future in as_completed([workers.submit(core.measure, item, output) for item in unique.values()]):
            physical.append(future.result())
    return expand_openface_records(items, physical), physical


def source_openface(control, cache, sources, cache_ref, output):
    out = new_directory(output)
    items, unavailable = source_items(control, cache, sources)
    checked = []
    for item in items:
        try:
            checked.append(inspect_image_item(item, {'path': item['path'], 'sha256': item['sha256']}, 'real_source'))
        except Exception as error:
            unavailable.append({**item, 'vector': None, 'selected_au17': None, 'raw_rows': [],
                'selected_row': None, 'confidence': None, 'returncode': None,
                'parse_status': 'source_unavailable', 'error': str(error)})
    records, physical = openface_items(checked, out)
    records += [{**r, 'image_size': [512, 512], 'policy_sha256': POLICY_SHA} for r in unavailable]
    order = {cid: i for i, cid in enumerate(control['cases'])}
    records.sort(key=lambda r: order[r['case_id']])
    document = {'kind': 'completion_source_openface', 'status': 'complete', 'data_role': ROLE,
        'independent_final_test': False, 'previously_exposed_subjects': True, 'source_only': True,
        'protocol_sha256': control['protocol_sha256'], 'source_cache_sha256': cache_ref['sha256'],
        'source_cache': cache_ref, 'freeze': control['freeze'], 'policy_sha256': POLICY_SHA,
        'measurement_policy_sha256': POLICY_SHA, 'records': records, 'unique_measurements': physical,
        'logical_source_cases': len(records), 'unique_images_measured': len(physical),
        'source_canonical_reencoded': False, 'target_images_read': False, 'algorithm': ref(EXP / PINS['openface'][0])}
    save(out / 'results.json', document)
    return out / 'results.json'


def resolve_target_reference(case_id, target_raw, source_by_raw_sha, cache, align):
    """Deduplicate only successful byte identities, never unrelated missingSHAs."""
    digest = target_raw.get('sha256')
    valid_raw = target_raw.get('status') == 'complete' and re.fullmatch(r'[0-9a-f]{64}', str(digest)) is not None
    key = ('sha256', digest) if valid_raw else ('failed_case', case_id, str(target_raw.get('path')))
    if key not in cache:
        if not valid_raw:
            cache[key] = {'status': 'failed', 'raw_image': copy.deepcopy(target_raw),
                'error': target_raw.get('error') or 'Raw target unavailable or lacks validSHA',
                'error_type': target_raw.get('error_type', 'RawTargetUnavailable'), 'target_alignment_attempted': False}
        else:
            try:
                cache[key] = {**source_by_raw_sha[digest], 'evaluation_target_same_raw_as_source': True} if digest in source_by_raw_sha else align(target_raw)
            except Exception as error:
                cache[key] = {'status': 'failed', 'raw_image': copy.deepcopy(target_raw),
                    'error': str(error), 'error_type': type(error).__name__}
    result = copy.deepcopy(cache[key])
    result['measurement_raw_image'] = result.get('raw_image')
    result['raw_image'] = copy.deepcopy(target_raw)
    return result


def target_references(control, cache, sources, cache_ref, raw_ref, output):
    raw = read(verified(raw_ref))
    need(raw['protocol_sha256'] == control['protocol_sha256'], 'Raw manifest freeze mismatch')
    need(raw.get('status') != 'integrity_failed' and not raw.get('post_extraction_integrity_errors'), 'Raw extraction integrity failure cannot proceed')
    raw_rows = {r['case_id']: r for r in raw['cases']}
    need(len(raw_rows) == 32 and set(raw_rows) == set(control['cases']), 'Raw case set changed')
    labels = {r['case_id']: r for r in control['labels']}
    for case in cache['cases']:
        r, label = raw_rows[case['case_id']], labels[case['case_id']]
        g = r['generation_input']
        need(r['split'] == ROLE and r['subject'] == label['subject'], 'Raw case role/subject mismatch')
        need(g['source_au_vector_0_5'] == label['source_au_vector_0_5'] and g['target_au_vector_0_5'] == label['target_au_vector_0_5'], 'Raw labels changed')
        source_raw = sources[case['source_id']]['raw_source']
        need(g['source_image'] == source_raw and source_raw['label_frame_1based'] == label['source_frame_1based'], 'Source cache belongs to another raw image')
        need(r['target_reference_image']['label_frame_1based'] == label['target_frame_1based'], 'Target raw frame changed')
    out = new_directory(output)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TMPDIR'):
        folder = out / 'cache' / key.lower()
        folder.mkdir(parents=True)
        os.environ[key] = str(folder)
    source_refs, byraw = {}, {}
    for sid, source in sources.items():
        result = {'status': 'failed', 'source_id': sid, 'raw_image': source['raw_source'], 'error': source.get('error')}
        try:
            need(source['status'] == 'complete', 'Source canonical unavailable')
            image_ref = source['canonical_source']
            inspect_image_item({}, image_ref, 'real_source')
            result = {'status': 'complete', 'source_id': sid, 'raw_image': source['raw_source'], 'image': image_ref, 'reused_source_cache': True}
            byraw[source['raw_source']['sha256']] = result
        except Exception as error:
            result.update(error=str(error), error_type=type(error).__name__)
        source_refs[sid] = result
    # Reuse the exact original target-alignment implementation, no source recrop.
    align = legacy('evaluation_references').make_target_aligner(out)
    targets, rows = {}, []
    for c in cache['cases']:
        cid = c['case_id']
        target_raw = raw_rows[cid]['target_reference_image']
        target = resolve_target_reference(cid, target_raw, byraw, targets, align)
        rows.append({**control['cases'][cid], 'source_reference': source_refs[c['source_id']],
            'target_reference': target, 'target_passed_to_our_generator': False})
    result = {'kind': 'completion_evaluation_references', 'status': 'complete', 'data_role': ROLE,
        'protocol_sha256': control['protocol_sha256'], 'measurement_policy_sha256': POLICY_SHA,
        'source_cache_manifest': cache_ref, 'raw_manifest': raw_ref, 'freeze': control['freeze'],
        'cases': rows, 'source_references_reencoded': False, 'targets_only_used_for_evaluation': True,
        'algorithm': ref(EXP / PINS['evaluation_references'][0])}
    save(out / 'references.json', result)
    return out / 'references.json'


def load_indexes(index_refs, control, references):
    documents = []
    for index_ref in index_refs:
        index = read(verified(index_ref))
        if index.get('kind') == 'completion_v11_generation':
            config = read(verified(index['configuration']))
            need(config['freeze'] == control['freeze'] and config['source_cache'] == references['source_cache_manifest'], 'Our generation uses different freeze/canonical source cache')
            need(index.get('target_pixels_used') is False and index.get('target_latents_computed') is False, 'Our generation target boundary changed')
            index = {**index, 'method_names': OURS, 'case_ids': list(control['cases']), 'sampling_seeds': SEEDS}
            source_by_case = {r['case_id']: r['source_reference'] for r in references['cases']}
            for row in index['records']:
                source = source_by_case[row['case_id']]
                if source['status'] == 'complete':
                    need(row['source_real'] == source['image'], 'Our generated row canonical source differs')
        else:
            need(index.get('protocol_sha256') == control['protocol_sha256'] and index.get('raw_manifest') == references['raw_manifest'], 'External generation freeze/raw binding mismatch')
            need(set(index['method_names']) <= set(EXTERNAL), 'External index declared internal methods')
        documents.append(index)
    return documents


def prepare_job(control, reference_ref, index_refs, output):
    references = read(verified(reference_ref))
    need(references.get('kind') == 'completion_evaluation_references' and references.get('status') == 'complete' and references.get('data_role') == ROLE, 'Wrong/incomplete evaluation references')
    need(references['protocol_sha256'] == control['protocol_sha256'] and references['measurement_policy_sha256'] == POLICY_SHA, 'Reference freeze/policy mismatch')
    verified(references['source_cache_manifest'])
    verified(references['raw_manifest'])
    docs = load_indexes(index_refs, control, references)
    plan, normalized, accounting = normalize_records(control['cases'], docs)
    out = new_directory(output)
    items, availability = [], []
    def add(meta, image_ref, role):
        try:
            item = inspect_image_item(meta, image_ref, role)
            items.append(item)
            availability.append({'key': [meta['method'], meta['subject'], meta['case_id'], meta['seed']], 'status': 'available', 'role': role, 'image_size': item['image_size']})
        except Exception as error:
            availability.append({'key': [meta['method'], meta['subject'], meta['case_id'], meta['seed']], 'status': 'unavailable', 'role': role, 'error': str(error)})
    need([r['case_id'] for r in references['cases']] == list(control['cases']), 'Reference case order changed')
    for r in references['cases']:
        meta = control['cases'][r['case_id']]
        need(all(r.get(k) == v for k, v in meta.items()), 'Reference labels/role changed')
        for role in ('source', 'target'):
            reference = r[role + '_reference']
            item = {**meta, 'method': role + '_real', 'seed': 0}
            if reference['status'] == 'complete': add(item, reference['image'], 'real_' + role)
            else: availability.append({'key': list(row_key(item)), 'status': 'reference_failed', 'role': 'real_' + role, 'error': reference.get('error')})
    for p in plan:
        row = normalized.get(row_key(p))
        if row and row['status'] == 'complete':
            add({**control['cases'][p['case_id']], **p, 'execution_id': row['execution_id'],
                 'execution_started': row['execution_started'], 'alias_of': row.get('alias_of')}, row['output'], 'generated')
        else:
            availability.append({'key': list(row_key(p)), 'status': row['status'] if row else 'missing_output', 'role': 'generated'})
    need(len({row_key(r) for r in items}) == len(items), 'Duplicate measurement item')
    document = {'kind': 'completion_measurement_inputs', 'status': 'prepared', 'data_role': ROLE,
        'independent_final_test': False, 'previously_exposed_subjects': True, 'test_used': True,
        'test_used_semantics': 'historically held-out but already exposed subjects; not a blind final test',
        'protocol_sha256': control['protocol_sha256'], 'measurement_policy_sha256': POLICY_SHA,
        'freeze': control['freeze'], 'reference_manifest': reference_ref, 'generation_indexes': index_refs,
        'case_metadata': control['cases'], 'planned': plan, 'normalized_generation_records': list(normalized.values()),
        'execution_accounting': accounting, 'items': items, 'availability': availability}
    save(out / 'items.json', document)
    save(out / 'items_seal.json', {'sha256': sha(out / 'items.json'), 'policy_sha256': POLICY_SHA, 'protocol_sha256': control['protocol_sha256']})
    return out


def job_control(job):
    job = Path(job).resolve()
    need(job.is_relative_to(HERE), 'Job outside evaluation directory')
    doc, seal = read(job / 'items.json'), read(job / 'items_seal.json')
    control = load_control(doc['freeze']['path'], doc['freeze']['sha256'])
    need(sha(job / 'items.json') == seal['sha256'] and doc['protocol_sha256'] == seal['protocol_sha256'] == control['protocol_sha256'], 'Job seal changed')
    need(doc['data_role'] == ROLE and doc['measurement_policy_sha256'] == seal['policy_sha256'] == POLICY_SHA, 'Job role/policy mismatch')
    verified(doc['reference_manifest'])
    for reference in doc['generation_indexes']: verified(reference)
    return job, doc, control


def derive_identity(kind, job, input_sha, count, source_count):
    core, text = legacy('evaluation_measurements').derive_identity(kind, job, input_sha, count, source_count)
    class RoleAdapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value in ('test', 'final_test'): node.value = ROLE
            return node
        def visit_Dict(self, node):
            self.generic_visit(node)
            if any(isinstance(k, ast.Constant) and k.value == 'test_used' for k in node.keys):
                node.keys += [ast.Constant('data_role'), ast.Constant('independent_final_test')]
                node.values += [ast.Constant(ROLE), ast.Constant(False)]
            return node
    return core, ast.unparse(ast.fix_missing_locations(RoleAdapter().visit(ast.parse(text)))) + '\n'


def measure(job, kind):
    job, doc, control = job_control(job)
    out = job / kind
    need(not out.exists(), 'Never overwrite measurements')
    for item in doc['items']: verified({'path': item['path'], 'sha256': item['sha256']})
    if kind == 'openface':
        out.mkdir()
        records, physical = openface_items(doc['items'], out)
        save(out / 'results.json', {'status': 'complete', 'records': records, 'unique_measurements': physical,
            'policy_sha256': POLICY_SHA, 'data_role': ROLE, 'protocol_sha256': control['protocol_sha256'],
            'measurement_policy_sha256': POLICY_SHA, 'input_sha256': sha(job / 'items.json')})
    elif not doc['items']:
        out.mkdir()
        save(out / 'results.json', {'records': [], 'policy_sha256': POLICY_SHA, 'summary': {'status': 'complete'}, 'no_models_called': True, 'data_role': ROLE})
    else:
        sources = len({r['case_id'] for r in doc['items'] if r['method'] == 'source_real'})
        core, text = derive_identity(kind, job, sha(job / 'items.json'), len(doc['items']), sources)
        namespace = {'__name__': 'completion_identity_native', '__file__': str(Path(__file__).resolve())}
        exec(compile(text, str(core), 'exec'), namespace)
        namespace['main']()
        # Native scalar/geometry code unchanged; snapshots are reproducibility outputs.
        with (out / 'derived_snapshot.py').open('x') as stream: stream.write(text)
    save(out / 'binding.json', {'data_role': ROLE, 'independent_final_test': False,
        'protocol_sha256': control['protocol_sha256'], 'measurement_policy_sha256': POLICY_SHA,
        'input_sha256': sha(job / 'items.json'), 'results_sha256': sha(out / 'results.json')})
    return out / 'results.json'


def export(job):
    job, doc, control = job_control(job)
    data = []
    available = {row_key(r): r for r in doc['items']}
    for kind in ('openface', 'facenet', 'buffalo'):
        measurement = read(job / kind / 'results.json')
        binding = read(job / kind / 'binding.json')
        need(binding['protocol_sha256'] == control['protocol_sha256'] and binding['measurement_policy_sha256'] == POLICY_SHA and binding['input_sha256'] == sha(job / 'items.json'), 'Measurement binding mismatch')
        need(binding['results_sha256'] == sha(job / kind / 'results.json'), 'Measurement result changed')
        keys = [row_key(r) for r in measurement['records']]
        need(len(keys) == len(set(keys)) and set(keys) == set(available), 'Every available image must keep evaluator row, including failed detections')
        for row in measurement['records']:
            need(row['sha256'] == available[row_key(row)]['sha256'], 'Evaluator image SHA changed')
        data.append(measurement)
    rows = legacy('evaluation_export').merge_measurements(doc, *data)
    out = new_directory(job / 'statistics')
    input_doc = statistics_document(doc, rows)
    save(out / 'all_case_seed_metrics.json', {'status': 'complete', 'data_role': ROLE,
        'protocol_sha256': control['protocol_sha256'], 'records': rows, 'availability': doc['availability'],
        'execution_accounting': doc['execution_accounting']})
    save(out / 'paired_subject_input.json', input_doc)
    stats = enrich_coverage(legacy('statistics').analyze(input_doc), input_doc, doc['case_metadata'])
    stats['input_sha256'] = sha(out / 'paired_subject_input.json')
    stats['statistics_source_sha256'] = PINS['statistics'][1]
    save(out / 'paired_subject_results.json', stats)
    return out / 'paired_subject_results.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=('source-openface', 'references', 'prepare', 'openface', 'facenet', 'buffalo', 'export'))
    parser.add_argument('--emit-spec', type=Path)
    for name in ('freeze', 'source-cache', 'raw-manifest', 'references', 'job', 'output'):
        parser.add_argument('--' + name, type=Path)
    for name in ('freeze', 'source-cache', 'raw', 'references'):
        parser.add_argument('--' + name + '-sha')
    parser.add_argument('--generation-index', action='append', type=Path, default=[])
    parser.add_argument('--generation-sha', action='append', default=[])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.emit_spec:
        save(args.emit_spec, execution_spec())
        print(json.dumps({'status': 'spec_only_no_models', 'spec': ref(args.emit_spec)}))
        return
    if args.phase in ('openface', 'facenet', 'buffalo', 'export'):
        job, doc, _ = job_control(args.job)
        if not args.execute:
            print(json.dumps({'status': 'preflight_no_models', 'phase': args.phase, 'logical_images': len(doc['items'])}))
            return
        result = export(job) if args.phase == 'export' else measure(job, args.phase)
    else:
        control = load_control(args.freeze, args.freeze_sha)
        if args.phase in ('source-openface', 'references'):
            cache, sources = load_cache(args.source_cache, args.source_cache_sha, control)
            cache_ref = {'path': str(args.source_cache.resolve()), 'sha256': args.source_cache_sha}
        if not args.execute:
            print(json.dumps({'status': 'preflight_no_models', 'phase': args.phase, 'case_count': len(control['cases'])}))
            return
        if args.phase == 'source-openface': result = source_openface(control, cache, sources, cache_ref, args.output)
        elif args.phase == 'references': result = target_references(control, cache, sources, cache_ref,
            {'path': str(args.raw_manifest.resolve()), 'sha256': args.raw_sha}, args.output)
        elif args.phase == 'prepare':
            need(len(args.generation_index) == len(args.generation_sha) > 0, 'Explicit generation indexes and matching SHAs required')
            result = prepare_job(control, {'path': str(args.references.resolve()), 'sha256': args.references_sha},
                [{'path': str(p.resolve()), 'sha256': s} for p, s in zip(args.generation_index, args.generation_sha)], args.output)
        else: raise ValueError('A phase is required')
    print(json.dumps({'status': 'complete', 'output': str(result)}))


if __name__ == '__main__':
    main()
