"""Read-only independent arithmetic/receipt verification of the completed19DEV run.

Uses only stdlib. Does not import evaluator, torch, image/model runtimes or the
in-progress32-case pipeline. Reports are the only files this program can write.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
EXP = COMPLETION.parent
RUN = COMPLETION / 'dev19_runs/dev19_202609152322'
JOB = COMPLETION / 'evaluation/dev19_tools/dev19_202609152322'
GEN = COMPLETION / 'generation'
RUN_SHA = '1abc9e2f647c3fa6ac60294aa09dd8d950ddc19ecbb8cec5df1df6dd83403547'
REPORT_SHA = '7ecb0fb78039d86bc13d19d6b3feb91710e3da1bf4d8b680c3565098429d6b12'
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
EXTRA = [7, 10, 14, 23, 45]
AUS17 = sorted(AUS + EXTRA)
VARIANTS = ['base', 'bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp']
OURS = [v + '_local_lp_s050' for v in VARIANTS]
EXTERNAL = ['magicface', 'faceadapter', 'liveportrait', 'liveportrait_absolute']
METHODS = OURS + EXTERNAL
TIMES = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
METRICS = ['target_mae_changed', 'delta_mae_changed', 'unchanged_source_relative_drift', 'target_mae_all12',
           'delta_mae_all12', 'direction_correct_fraction', 'identity_cosine']
FILE_RECEIPTS = {}
CSV_CACHE = {}


def need(ok, label):
    if not ok:
        raise ValueError(label)


def allowed(path):
    path = Path(path).resolve()
    need(EXP in path.parents, 'Outside experiment: ' + str(path))
    if COMPLETION in path.parents:
        need(any(root == path or root in path.parents for root in (RUN, JOB, GEN, HERE)), 'No access to in-progress32 artifacts: ' + str(path))
    return path


def sha(path):
    path = allowed(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    value = h.hexdigest()
    FILE_RECEIPTS[str(path)] = value
    return value


def read(path):
    return json.loads(allowed(path).read_text())


def checked(reference, label):
    path = allowed(reference['path'])
    need(sha(path) == reference['sha256'], label + ' SHA mismatch')
    return path


def same(left, right, label='value'):
    if type(right) in (int, float):
        need(type(left) in (int, float) and math.isfinite(left) and math.isfinite(right) and abs(left - right) <= 1e-12, label + ': numeric mismatch ' + repr((left, right)))
    elif isinstance(right, dict):
        need(isinstance(left, dict) and set(left) == set(right), label + ': field mismatch')
        for key in right:
            same(left[key], right[key], label + '.' + str(key))
    elif isinstance(right, (list, tuple)):
        need(isinstance(left, (list, tuple)) and len(left) == len(right), label + ': list mismatch')
        for i, (a, b) in enumerate(zip(left, right)):
            same(a, b, label + '[' + str(i) + ']')
    else:
        need(left == right, label + ': mismatch')


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def mapping(rows):
    result = {(r['case_id'], r['method']): r for r in rows}
    need(len(result) == len(rows), 'Duplicate logical row')
    return result


def independent_openface(path):
    path = allowed(path)
    if str(path) in CSV_CACHE:
        return CSV_CACHE[str(path)]
    sha(path)
    with path.open() as stream:
        raw = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]
    candidates = []
    for i, row in enumerate(raw):
        xs = [float(row['x_' + str(j)]) for j in range(68)]
        ys = [float(row['y_' + str(j)]) for j in range(68)]
        need(all(math.isfinite(v) for v in xs + ys), 'Nonfinite raw landmarks')
        box = [min(xs), min(ys), max(xs), max(ys)]
        area = (box[2] - box[0]) * (box[3] - box[1])
        candidates.append({'row_index': i, 'face_id': row.get('face'), 'bbox68': box,
            'area68': area if area > 0 else None, 'confidence_raw': row.get('confidence')})
    valid = [r for r in candidates if r['area68'] is not None]
    need(valid, 'No rankable raw face')
    winner = sorted(valid, key=lambda r: (-r['area68'], r['row_index']))[0]
    selected = raw[winner['row_index']]
    values = {str(au): float(selected[f'AU{au:02d}_r']) for au in AUS17}
    confidence = float(selected['confidence'])
    need(math.isfinite(confidence) and all(math.isfinite(v) and 0 <= v <= 5 for v in values.values()), 'Selected raw AU invalid')
    areas = sorted((r['area68'] for r in valid), reverse=True)
    result = {'raw_rows': raw, 'face_count': len(raw), 'all_candidates': candidates, 'selected_row': winner['row_index'],
        'selected_au17': values, 'vector': [values[str(au)] for au in AUS], 'confidence': confidence,
        'parse_status': 'primary_face_finite', 'second_to_first_area_ratio': areas[1] / areas[0] if len(areas) > 1 else None}
    CSV_CACHE[str(path)] = result
    return result


def validate_measurement(row):
    measured = row if 'csv_files' in row else row['AU_measurement']
    need(row['returncode'] == 0 and row['parse_status'] == 'primary_face_finite' and len(measured['csv_files']) == 1, 'Invalid measured OpenFace row')
    need(measured['sha256'] == row['sha256'], 'Nested raw measurement image binding mismatch')
    raw = independent_openface(measured['csv_files'][0])
    for key in raw:
        if key in row:
            same(row[key], raw[key], 'rawCSV ' + row['case_id'] + '/' + row['method'] + '/' + key)
    return raw


def metrics(predicted, source, real_target, before, after):
    changed = [i for i in range(12) if before[i] != after[i]]
    unchanged = [i for i in range(12) if i not in changed]
    delta = [a - b for a, b in zip(predicted['vector'], source['vector'])]
    request = [b - a for a, b in zip(before, after)]
    result = {'changed_aus': [AUS[i] for i in changed], 'unchanged_aus': [AUS[i] for i in unchanged],
        'target_mae_changed': mean(abs(predicted['vector'][i] - after[i]) for i in changed),
        'delta_mae_changed': mean(abs(delta[i] - request[i]) for i in changed),
        'unchanged_source_relative_drift': mean(abs(delta[i]) for i in unchanged),
        'target_mae_all12': mean(abs(predicted['vector'][i] - after[i]) for i in range(12)),
        'delta_mae_all12': mean(abs(delta[i] - request[i]) for i in range(12)), 'measured_delta_12': delta,
        'direction_correct_fraction': mean(int(delta[i] * request[i] > 0) for i in changed),
        'response_gain_changed': [delta[i] / request[i] for i in changed]}
    extra = {}
    for au in EXTRA:
        a, b, s = (r['selected_au17'][str(au)] for r in (predicted, real_target, source))
        extra[str(au)] = {'output': a, 'real_target_measured': b, 'real_source_measured': s,
            'absolute_error_to_target': abs(a - b), 'output_minus_source': a - s, 'target_minus_source': b - s}
        result[f'extra_AU{au}_error_to_real_target'] = abs(a - b)
    result['extra5'] = extra
    result['extra5_proxy_mae_to_real_target'] = mean(v['absolute_error_to_target'] for v in extra.values())
    return result


def scope_rows(rows, scope):
    return [r for r in rows if scope == 'all' or (r['group'] == 'no_edit') == (scope == 'no_edit')]


def aggregate(part, methods, metric, extra=False):
    rows = mapping(part)
    candidates = sorted(cid for cid, method in rows if method == methods[0])
    common = [cid for cid in candidates if all(rows[cid, method].get(metric) is not None for method in methods)]
    subjects = sorted({rows[cid, methods[0]]['subject'] for cid in common})
    by_subject = {s: {m: mean(rows[cid, m][metric] for cid in common if rows[cid, m]['subject'] == s) for m in methods} for s in subjects}
    result = {'case_ids': common, 'subjects': subjects, 'per_subject': by_subject,
        'subject_equal_means': {m: mean(v[m] for v in by_subject.values()) for m in methods}}
    if extra:
        result['n'] = len(common)
    else:
        result.update(planned_cases=len(candidates), common_cases=len(common))
        if len(methods) == 2:
            result['new_minus_reference'] = result['subject_equal_means'][methods[1]] - result['subject_equal_means'][methods[0]] if common else None
    return result


def compare(rows, methods, keys=METRICS, extra=False):
    return {scope: {key: aggregate(scope_rows(rows, scope), methods, key, extra) for key in keys} for scope in ('all', 'edited', 'no_edit')}


def strata(rows, methods):
    mapped = mapping(rows)
    sources = {cid: r for (cid, method), r in mapped.items() if method == 'source_real'}
    result = {}
    for i, au in enumerate(AUS):
        for level in (1, 2, 3):
            cases = sorted(cid for cid, r in sources.items() if r['source_labels_12'][i] != r['target_labels_12'][i] and r['target_labels_12'][i] == level)
            values = {}
            for cid in cases:
                for method in methods:
                    row = mapped[cid, method]
                    delta = row['vector'][i] - sources[cid]['vector'][i]
                    desired = row['target_labels_12'][i] - row['source_labels_12'][i]
                    values[cid, method] = {'target_error': abs(row['vector'][i] - level), 'delta_error': abs(delta - desired),
                        'direction_correct': int(delta * desired > 0), 'identity_cosine': row['identity_cosine']}
            calculated = {}
            for metric in ('target_error', 'delta_error', 'direction_correct', 'identity_cosine'):
                common = [cid for cid in cases if all(values[cid, m][metric] is not None for m in methods)]
                subjects = sorted({sources[cid]['subject'] for cid in common})
                by_subject = {s: {m: mean(values[cid, m][metric] for cid in common if sources[cid]['subject'] == s) for m in methods} for s in subjects}
                calculated[metric] = {'n': len(common), 'case_ids': common, 'subject_n': len(subjects), 'per_subject': by_subject,
                    'subject_equal_means': {m: mean(v[m] for v in by_subject.values()) for m in methods}}
            result[f'AU{au}_level{level}'] = {'au': au, 'level': level, 'planned_n': len(cases), 'planned_case_ids': cases,
                'valid_counts': {m: {key: sum(values[cid, m][key] is not None for cid in cases) for key in ('target_error', 'delta_error', 'identity_cosine')} for m in methods},
                'metrics': calculated}
    return result


def verify_generation(labels):
    need(sha(RUN / 'result.json') == RUN_SHA, 'Finished dev95 result SHA changed')
    result = read(RUN / 'result.json')
    need(result['status'] == 'complete' and result['planned'] == result['completed'] == 95 and result['split'] == 'dev', 'Dev95 not complete')
    config = read(checked(result['configuration'], 'generation config'))
    runtime = read(checked(result['runtime'], 'generation runtime'))
    checked(runtime['source_latents'], 'source latents entity')
    records = mapping(result['records'])
    need(set(records) == {(cid, method) for cid in labels for method in OURS}, '95-case/method denominator differs')
    same(read(GEN / 'generation_spec_candidate_v3.json'), config['core_generation'], 'fixed generation spec')
    four = read(config['inputs']['full_constructor_parity']['path'])
    checked(config['inputs']['full_constructor_parity'], 'actual4 constructor gate')
    need(four['passed'] and len(four['records']) == 4 and all(r['passed'] and all(r['checks'].values()) for r in four['records']), 'Actual4 gate failed')
    for r in four['records']:
        checked(r['output'], 'actual4 gate output')
    need(len(runtime['checkpoints']) == 4, 'Four early200 checkpoints required')
    for variant, checkpoint in runtime['checkpoints'].items():
        path = checked(checkpoint, variant + ' weights')
        need(checkpoint['global_step'] == 200 and read(path.with_suffix('.json'))['checkpoint_sha256'] == checkpoint['sha256'], 'Checkpoint seal/step mismatch')
    selected = read(checked(config['inputs']['selected_result'], 'selected oldV11'))
    old = mapping([r for r in selected['records'] if r['method'] in ('baseline_local_lp_s050', 'bidirectional_local_lp_s050')])
    sources = {r['source']['sha256']: r for r in runtime['source_vae_records']}
    need(len(sources) == 4 and all(r['status'] == 'complete' for r in sources.values()), 'Four source VAE records failed')
    for r in sources.values():
        checked(r['source'], 'VAE source'); checked(r['reconstruction'], 'source reconstruction')
    zero, old_checks, outputs = collections.defaultdict(list), [], []
    need([json.loads(line) for line in (RUN / 'generation_records.jsonl').read_text().splitlines()] == result['records'], 'JSONL/result mismatch')
    for (cid, method), r in records.items():
        need(r['status'] == 'complete' and r['execution_started'] and r['seed'] == 101 and r['split'] == 'dev', 'Unexpected generation execution state')
        label = labels[cid]
        need(r['subject'] == label['subject'] and r['source_aus_0_5'] == label['source_au_vector_0_5'] and r['target_aus_0_5'] == label['target_au_vector_0_5'], 'Generation label binding failed')
        checked({'path': r['output'], 'sha256': r['sha256']}, 'new DEV output')
        need(RUN in Path(r['output']).resolve().parents, 'New95 generation cannot point to any old output directory')
        checked(r['source_real'], 'generation source'); checked(r['mask_receipt'], 'mask receipt')
        need([r[k] for k in ('denoising_steps', 'deis_unet_calls', 'terminal_unet_calls', 'total_unet_calls')] == [15, 15, 0, 15], 'Wrong actual UNet counts')
        need([c['t'] for c in r['unet_call_trace']] == TIMES and all(c['phase'] == 'deis' and c['au_scale'] == 1.5 and c['cfg_batch'] == 2 for c in r['unet_call_trace']), 'UNet trace mismatch')
        need([c['t'] for c in r['projection_trace']] == TIMES and [c['next_t'] for c in r['projection_trace']] == TIMES[1:] + [None], 'Projection trajectory mismatch')
        t = r['terminal_trace']
        need(t['mode'] == 'local_lp_s050' and t['epsilon_source'] == 'last_t50' and t['last_deis_timestep'] == 50, 'V11 terminal mode changed')
        need(t['projected_latent_sha256'] == r['projection_trace'][-1]['latent_sha256'] and t['filter']['sigma_latent'] == .5, 'Final projection/filter mismatch')
        need(r['source_latent_sha256'] == sources[r['source_real']['sha256']]['latent_sha256'], 'Source latent lineage mismatch')
        if not any(r['target_aus_0_5']):
            need(r['sha256'] == r['source_vae_reconstruction']['sha256'], 'Zeroedit not exact VAE reconstruction')
            zero[cid].append(r['sha256'])
        old_method = 'baseline_local_lp_s050' if method == OURS[0] else method
        if (cid, old_method) in old:
            original = old[cid, old_method]
            checked({'path': original['output'], 'sha256': original['sha256']}, 'old V11 expected file')
            for key in ('sha256', 'condition_hashes', 'projection_trace', 'source_latent_sha256', 'noise_sha256'):
                same(r[key], original[key], 'oldV11 exact ' + cid + '/' + method + '/' + key)
            need(r['old_v11_parity']['passed'] and all(r['old_v11_parity']['checks'].values()), 'Stored parity flags inconsistent')
            old_checks.append({'case_id': cid, 'method': method, 'sha256': r['sha256'], 'exact': True})
        outputs.append(r['output'])
    need(len(set(outputs)) == 95 and len({r['execution_id'] for r in records.values()}) == 95, 'Physical file/execution denominator differs')
    need(len(zero) == 4 and all(len(v) == 5 and len(set(v)) == 1 for v in zero.values()), 'Four×five zeroedit invariance failed')
    need(len(old_checks) == 16 and len({r['noise_sha256'] for r in records.values()}) == 1, 'Old16/noise parity count failed')
    return result, {'new_outputs_checked': 95, 'unet_forward_calls_checked': 1425, 'four_early200_checkpoints_verified': True,
        'four_source_vae_checks': True, 'four_noedit_cases_all_five_equal': True, 'four_real_constructor_gate_passed': True,
        'old_v11_exact_checks': old_checks, 'old_v11_exact_count': 16}


def verify_facenet(document, items, comparison, old_rows):
    item_map = mapping(items['items'])
    rows = mapping(document['records'])
    need(len(rows) == 133 and set(rows) == set(item_map), 'FaceNet133 logical denominator changed')
    unique = {r['sha256']: r for r in document['unique_measurements']}
    need(len(unique) == len(document['unique_measurements']) == 98 and set(unique) == {r['sha256'] for r in rows.values()}, 'FaceNet98 unique denominator changed')
    for digest, row in unique.items():
        need(row['status'] == 'valid' and math.isfinite(row['embedding_norm']) and abs(row['embedding_norm'] - 1) < 1e-5, 'Invalid unique FaceNet measurement')
        checked({'path': row['measurement_source_path'], 'sha256': digest}, 'FaceNet actual measured file')
        rank = []
        for candidate in row['candidates']:
            x0, y0, x1, y1 = candidate['bbox']
            area = (x1 - x0) * (y1 - y0)
            same(candidate['area'], area, 'FaceNet bbox area')
            if candidate['geometry_rankable']:
                rank.append((area, -candidate['index'], candidate['index']))
        need(rank and max(rank)[2] == row['selected_index'], 'FaceNet largest bbox selection changed')
    parity = []
    for key, row in rows.items():
        need(row['sha256'] == item_map[key]['sha256'] and row['status'] == 'valid' and -1 - 1e-12 <= row['identity_cosine'] <= 1 + 1e-12, 'FaceNet logical binding/validity failed')
        unique_row = unique[row['sha256']]
        for field in ('status', 'selected_index', 'embedding_norm', 'candidates'):
            same(row[field], unique_row[field], 'FaceNet logical unique match')
        same(comparison[key]['identity_cosine'], row['identity_cosine'], 'FaceNet comparison value')
        if key[1] in ('source_real', 'target_real'):
            old = old_rows[key]['identity_cosine']
            need(abs(old - row['identity_cosine']) <= 1e-6, 'Real reference old/new FaceNet mismatch')
            parity.append({'case_id': key[0], 'method': key[1], 'old': old, 'new': row['identity_cosine'], 'match': True})
    need(len(parity) == 38 and document['summary']['logical_records'] == 133 and document['summary']['valid_unique_images'] == 98, 'FaceNet summary mismatch')
    return parity, {'logical_records': 133, 'unique_images': 98, 'valid_unique_images': 98, 'largest_bbox_selection_recomputed': True,
        'reference_parity_count': 38, 'all_reference_parity_passed': True, 'embeddings_saved': document['embeddings_saved'],
        'limitation': 'Embeddings were not saved; independent cosine-from-vector recalculation is unavailable without rerunning models. Validity/selection, source binding, stored cosine propagation and38old-new real-reference checks were verified.'}


def verify():
    need(sha(JOB / 'comparison/results.json') == REPORT_SHA, 'New completed comparison SHA changed')
    report = read(JOB / 'comparison/results.json')
    items = read(JOB / 'items.json')
    need(sha(JOB / 'items.json') == read(JOB / 'items_seal.json')['sha256'] == report['input_sha256'], 'Evaluation item seal mismatch')
    need(report['policy_sha256'] == items['policy_sha256'] == POLICY_SHA, 'Frozen measurement policy changed')
    prior = {k: read(checked(v, 'prior ' + k)) for k, v in items['prior_refs'].items()}
    labels = {r['case_id']: r for r in prior['labels']['cases']}
    need(len(labels) == 19 and {r['subject'] for r in labels.values()} == {'SN004', 'SN012', 'SN017', 'SN018'}, 'Wrong19/4dev labels')
    generated, generation_check = verify_generation(labels)
    need(items['generation_result']['sha256'] == RUN_SHA and report['generated_result']['sha256'] == RUN_SHA, 'Comparison does not use new95')
    rows = mapping(report['records'])
    need(report['status'] == 'complete' and len(rows) == 209 and set(rows) == {(cid, m) for cid in labels for m in METHODS + ['source_real', 'target_real']}, 'Complete209 denominator changed')
    need(report['test_used'] is False and report['independent_final_test'] is False and report['no_old_internal_outputs_used'] is True, 'Dev scope changed')
    generated_map = mapping(generated['records'])
    old_rows = mapping(prior['results']['records'])
    prior_cases = {r['case_id']: r for r in prior['manifest']['cases']}
    raw_values, source_values, target_values, external_checks = {}, {}, {}, []
    for key, row in rows.items():
        cid, method = key
        label = labels[cid]
        need(row['split'] == 'dev' and row['subject'] == label['subject'] and row['source_labels_12'] == label['source_au_vector_0_5'] and row['target_labels_12'] == label['target_au_vector_0_5'], 'Comparison labels/role mismatch')
        need(row.get('policy_sha256', row.get('AU_measurement', {}).get('policy_sha256')) == POLICY_SHA, 'Per-row evaluation policy changed')
        checked({'path': row['path'], 'sha256': row['sha256']}, 'comparison physical image')
        raw_values[key] = validate_measurement(row)
        if method in OURS:
            need(row['generation_status'] == 'complete' and row['path'] == generated_map[key]['output'] and row['sha256'] == generated_map[key]['sha256'], 'New row accidentally uses old internal output')
        if method in ('source_real', 'target_real'):
            role = 'source' if method == 'source_real' else 'target'
            original = prior_cases[cid]['reference_frames'][role]
            checked({'path': original['raw_frame_path'], 'sha256': original['raw_frame_sha256']}, 'real raw frame')
            need(original['manual_AU_0_5'] == label[role + '_au_vector_0_5'] and row['sha256'] == original['evaluation_sha256'], 'Real reference binding differs')
            (source_values if role == 'source' else target_values)[cid] = raw_values[key]
        if method in EXTERNAL:
            old = old_rows[key]
            for field in ('path', 'sha256', 'raw_rows', 'selected_row', 'vector', 'selected_au17', 'identity_cosine'):
                same(row[field], old[field], 'External exact reused ' + field)
            output = next(r for r in prior_cases[cid]['outputs'] if r['method'] == method)
            request = output['original_inference_input']
            source_ref, target_ref = (prior_cases[cid]['reference_frames'][r] for r in ('source', 'target'))
            need(output['output_sha256'] == row['sha256'] and output['output_path'] == row['path'], 'External original output mapping changed')
            need(request['request_case_id'] == cid and request['subject'] == label['subject'] and request['split'] == 'dev' and request['seed'] == output['request_seed_key'] == 101, 'External original case/seed/role mismatch')
            need(request['source_au_vector_0_5'] == label['source_au_vector_0_5'] and request['target_au_vector_0_5'] == label['target_au_vector_0_5'] and request['au_order'] == AUS, 'External original request labels changed')
            need(request['source_sha256'] == source_ref['raw_frame_sha256'], 'External raw source mismatch')
            if method == 'magicface':
                need(not output['extra_real_target_driver'] and request['au_delta_native'] == [b - a for a, b in zip(label['source_au_vector_0_5'], label['target_au_vector_0_5'])], 'MagicFace native request mismatch')
            else:
                need(output['extra_real_target_driver'] and request['target_sha256'] == target_ref['raw_frame_sha256'], 'External raw target driver mismatch')
            external_checks.append({'case_id': cid, 'method': method, 'output_sha256': row['sha256'], 'matched': True})
    need(len(external_checks) == 76, 'External76 reused denominator mismatch')
    # New physical raw measurement records independently back every95 logical generated row.
    openface = read(JOB / 'openface/results.json')
    fn = read(JOB / 'facenet/results.json')
    for name in ('openface', 'facenet'):
        binding = read(JOB / name / 'binding.json')
        need(binding['results_sha256'] == sha(JOB / name / 'results.json') and binding['input_sha256'] == sha(JOB / 'items.json'), name + ' seal mismatch')
        need(binding['policy_sha256'] == POLICY_SHA, name + ' policy binding mismatch')
    physical = {r['sha256']: r for r in openface['new_unique_measurements']}
    need(len(physical) == 79 and len(openface['records']) == 133 and openface['new_generated_logical_rows'] == 95 and openface['reused_real_reference_rows'] == 38, 'OpenFace79/133/95/38 denominator mismatch')
    need(set(physical) == {r['sha256'] for r in generated['records']}, 'New uniqueCSV image set differs from new95 outputs')
    availability = mapping(items['availability'])
    need(len(availability) == 95 and set(availability) == set(generated_map) and all(r['status'] == 'available' for r in availability.values()), 'Availability denominator differs from all95 outputs')
    for digest, row in physical.items():
        need(JOB in Path(row['csv_files'][0]).parents, 'New measurement raw CSV is not from this new job')
        need(row['input_unchanged'] and sha(row.get('measurement_source_path', row['path'])) == digest, 'New rawCSV measured image mismatch')
        validate_measurement(row)
    for key, row in mapping(openface['records']).items():
        same(row['vector'], raw_values[key]['vector'], 'logical OpenFace versus independently read CSV')
        if key[1] in OURS:
            need(row['sha256'] in physical and row['csv_files'] == physical[row['sha256']]['csv_files'], 'Logical generated rawCSV mapping changed')
    ref_parity, facenet_check = verify_facenet(fn, items, rows, old_rows)
    same(report['reference_facenet_parity'], ref_parity, '38reported parity rows')
    recomputed = []
    for key, row in rows.items():
        cid, method = key
        values = metrics(raw_values[key], source_values[cid], target_values[cid], row['source_labels_12'], row['target_labels_12'])
        for field, value in values.items():
            if field in row:
                same(row[field], value, cid + '/' + method + '/' + field)
        recomputed.append({**row, **values, 'vector': raw_values[key]['vector'], 'selected_au17': raw_values[key]['selected_au17']})
    # Recompute all reported pairwise/cohort/scope aggregates and36AU-level strata.
    pairs = {a + '_vs_' + b: [a, b] for a, b in itertools.combinations(METHODS, 2) if a in OURS or b in OURS}
    need(set(report['pairwise']) == set(pairs), 'Pairwise comparison roster changed')
    for cohort in ('all', 'original', 'additional'):
        part = [r for r in recomputed if cohort == 'all' or r['cohort'] == cohort]
        same(report['all_methods_common_valid'][cohort], compare(part, METHODS), 'All-method aggregate ' + cohort)
        for name, methods in pairs.items():
            same(report['pairwise'][name][cohort], compare(part, methods), 'Pairwise aggregate ' + name + '/' + cohort)
    extra_metrics = ['buffalo_identity_cosine', 'extra5_proxy_mae_to_real_target'] + [f'extra_AU{au}_error_to_real_target' for au in EXTRA]
    same(report['extra_diagnostics']['all_methods'], compare(recomputed, METHODS, extra_metrics, True), 'Extra all-method aggregate')
    for name, methods in pairs.items():
        same(report['extra_diagnostics']['pairwise'][name], compare(recomputed, methods, extra_metrics, True), 'Extra paired ' + name)
    level_stats = strata(recomputed, METHODS)
    same(report['au_level_all_methods'], level_stats, '36AUlevel all-method strata')
    for name, methods in pairs.items():
        same(report['au_level_pairwise'][name], strata(recomputed, methods), '36AUlevel paired strata ' + name)
    need(len(level_stats) == 36 and all(r['planned_n'] > 0 for r in level_stats.values()), '36AUlevel coverage lost')
    for method in METHODS + ['source_real', 'target_real']:
        for scope in ('all', 'edited', 'no_edit'):
            part = scope_rows([r for r in recomputed if r['method'] == method], scope)
            expected = {'planned': len(part), 'AU_valid': sum(r['vector'] is not None for r in part), 'FaceNet_valid': sum(r['identity_cosine'] is not None for r in part), 'buffalo_valid': sum(r.get('buffalo_identity_cosine') is not None for r in part)}
            same(report['coverage'][method][scope], expected, 'Coverage ' + method + '/' + scope)
    edited = compare(recomputed, METHODS)['edited']
    noedit = compare(recomputed, METHODS)['no_edit']
    table = {method: {metric: edited[metric]['subject_equal_means'][method] for metric in METRICS} for method in METHODS}
    for method in METHODS:
        table[method]['extra5_proxy_mae_to_real_target'] = compare(recomputed, METHODS, extra_metrics, True)['edited']['extra5_proxy_mae_to_real_target']['subject_equal_means'][method]
    ours = 'bidirectional_local_lp_s050'
    reductions = {m: 1 - table[ours]['target_mae_changed'] / table[m]['target_mae_changed'] for m in EXTERNAL}
    return {'status': 'verified', 'created_utc': datetime.now(timezone.utc).isoformat(), 'verification_scope': 'completed19DEV only; source files and new79rawCSV independently read; no ML rerun; no in-progress32access',
        'generation_result_sha256': RUN_SHA, 'comparison_result_sha256': REPORT_SHA, 'generation': generation_check,
        'openface': {'logical_rows_in_new_measurement_file': 133, 'new_generated_logical_rows': 95, 'new_unique_raw_csv_measurements': 79,
            'all_new_csv_largest68bbox_selection_recomputed': True, 'all209rows_raw12AU_and_extra5_recomputed': True,
            'total_distinct_csv_files_reread': len(CSV_CACHE)}, 'facenet': facenet_check,
        'external_reuse': {'matched_rows': 76, 'records': external_checks, 'raw_source_target_frame_hashes_and_original_labels_verified': True},
        'denominators': {'cases': 19, 'edited': 15, 'noedit': 4, 'subjects': 4, 'settings': 9, 'combined_rows': 209,
            'generated_new': 95, 'external_reused': 76, 'real_reference_rows': 38, 'AU_level_cells': 36},
        'statistics': {'all_casewise_AU_and_extra_values_recomputed': True, 'all_subject_equal_aggregates_recomputed': True,
            'pairwise_comparisons': len(pairs), 'cohorts': ['all', 'original', 'additional'], 'scopes': ['all', 'edited', 'no_edit'],
            'all36AUlevel_strata_recomputed': True, 'edited_metric_common_cases': {k: v['common_cases'] for k, v in edited.items()},
            'edited_subjects': edited['target_mae_changed']['subjects'], 'edited_case_count_by_subject': dict(collections.Counter(labels[cid]['subject'] for cid in edited['target_mae_changed']['case_ids']))},
        'edited_subject_equal_means': table, 'noedit_subject_equal_means': {m: {k: v['subject_equal_means'][m] for k, v in noedit.items()} for m in METHODS},
        'target_mae_relative_reduction_against_external': reductions,
        'limits': ['All19cases are development evidence, not independent final testing or same-training-protocol SOTA ranking.',
            'All-four-subject equal weighting; C metrics cover15edited cases. Noedit cases retain NA for changed-AU errors.',
            'No-edit reconstruction is by construction. External FaceAdapter/LivePortrait receive real target drivers.',
            'FaceNet embeddings were not persisted; cosine-vector recomputation is not possible without an unauthorized model rerun.'],
        'verified_file_sha256': FILE_RECEIPTS, 'verifier_sha256': sha(__file__)}


def markdown(result):
    lines = ['# 完整19例开发集独立复核', '', '**复核通过。** 本报告核验新95图，不以旧内部15例结果替代；未运行模型，未读取进行中新32例图像。', '',
        '- 95/95生成文件SHA正确，每图15 DEIS、0额外UNet；4检查点均早期200，4源VAE/4零编辑组通过。',
        '- 16条旧V11子集的PNG、条件和完整潜变量轨迹严格一致；原4图构造器gate有效。',
        '- 新79unique OpenFace原CSV独立按68点bbox面积重新选行；209行12AU、delta、漂移和extra5逐项重算一致。',
        '- FaceNet133逻辑行、98unique均valid，最大bbox选行及绑定一致；38真实参考的旧新值一致。',
        '- 19案例×(5新结构+4外部设置+2参考)=209行，无遗漏；15编辑/4零编辑、4主体等权和36AU×level均已独立复算。',
        '- 外部76行逐项对应原case、原输出、原raw源/驱动帧及人工AU标签SHA。', '',
        '## 15个编辑案例，4主体等权均值', '', '| 方法 | 目标AU误差↓ | AU变化误差↓ | 非目标漂移↓ | 身份相似度↑ | extra5误差↓ |', '|---|---:|---:|---:|---:|---:|']
    for method, values in result['edited_subject_equal_means'].items():
        lines.append('| ' + method + ' | ' + ' | '.join(f'{values[key]:.6f}' for key in ('target_mae_changed', 'delta_mae_changed', 'unchanged_source_relative_drift', 'identity_cosine', 'extra5_proxy_mae_to_real_target')) + ' |')
    reductions = result['target_mae_relative_reduction_against_external']
    lines += ['', '选定双向版目标AU误差相对MagicFace/FaceAdapter/LivePortrait默认/absolute分别降低 ' + ' / '.join(f'{reductions[m] * 100:.2f}%' for m in EXTERNAL) + '。这是完整开发覆盖上的特定指标优势，不能写成全面超过SOTA。', '',
        '## 复核边界', '', 'FaceNet未保存embedding，因而未从向量重新计算余弦；本次核验的是测量有效性、几何选行、绑定、数值传递及38真实参考前后对应，不把这些冒充模型重推理。', '',
        '生成result SHA：`' + result['generation_result_sha256'] + '`。比较result SHA：`' + result['comparison_result_sha256'] + '`。']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = verify()
    if args.write:
        with (HERE / 'dev19_verification.json').open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        with (HERE / 'dev19_verification.md').open('x') as stream:
            stream.write(markdown(result))
    print(json.dumps({'status': result['status'], 'denominators': result['denominators'], 'edited_subject_equal_means': result['edited_subject_equal_means'], 'reports_written': args.write}, ensure_ascii=False))
