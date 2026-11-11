"""One-shot final-test freeze. Default: verify only; never open test videos.

Only --execute hashes the eight accepted videos (no decoding), then exclusively
writes sampling/evaluation and, LAST, frozen_protocol.json. No model is loaded.
Python-only Contract injection exists for synthetic tests; CLI has no bypass.
"""
import argparse
import ast
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROJECT = EXP.parent.parent
METHODS = ('bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp',
           'public_baseline', 'magicface', 'faceadapter', 'liveportrait')
SUBJECTS = ('SN003', 'SN005', 'SN008', 'SN009', 'SN021', 'SN024', 'SN031', 'SN032')
SEEDS = [101, 202, 303]
CHECKPOINTS = {
    'bidirectional': '7fcc25b704fca8c0a4e46a171096df8fea940b40a246d4939193e64f533b4e30',
    'condition_reads_image': '969def9470fb6a8ba4ff5b37e11b601144101e563e25ff3c26e09e4fdb71396b',
    'image_reads_condition': '99b565be708493e23a5a5d11829353cbc88cba3c2d78a388be49534703bd695f',
    'mlp': 'f58336567246ecd2f6818ea4ae30f947914af4e9c2296177b86169b95d6aa6f5',
}
DRAFT_SHAS = {
    'protocol.DRAFT.json': '38058b10b4bcf025c2f768be793084aaa79ed0df15c9c2cd55c6740f4be0adef',
    'sampling.DRAFT.json': '4b3236f0670c8a7728f1b596cefee88068b59fef092710e1e16eb074bf368fef',
    'evaluation.DRAFT.json': '375340dd0e2b66498f35b8144c5f825aadde3d6edb46f8fe85f20be95e3a5c85',
}
INPUT_SHAS = {
    'label_plan': 'b2052f261a5caf81b444e5a9aaa2620c917b399904f7f404c7cf6fe2a04dc684',
    'label_inventory': '2475a2e971922c3983321f74d7de18edffef8157cb4b0ea15b16685f53a3c8ab',
    'subject_split': 'b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d',
}
DIAGNOSTICS = ('delta_mae_changed', 'identity_cosine', 'unchanged_source_relative_drift',
               'extra5_proxy_mae_to_real_target')


@dataclass(frozen=True)
class Contract:
    root: Path
    project: Path
    video_root: Path
    draft_shas: dict
    input_shas: dict
    checkpoints: dict


PRODUCTION = Contract(EXP, PROJECT, Path.home() / 'Desktop/Cursor/geme-main/DISFA',
                      DRAFT_SHAS, INPUT_SHAS, CHECKPOINTS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def path_for(value, contract):
    require(isinstance(value, str) and bool(value), 'Missing reference path')
    path = Path(value)
    return (path if path.is_absolute() else contract.project / path).resolve()


def refs(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        for child in value.values():
            yield from refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs(child)


class Verifier:
    def __init__(self, contract):
        self.contract = contract
        self.files = {}
        self.signatures = {}
        self.video_reads = 0

    @staticmethod
    def signature(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def check(self, reference, *, video=False):
        path = path_for(reference.get('path'), self.contract)
        expected = reference.get('sha256')
        require(expected is None and video or isinstance(expected, str) and
                re.fullmatch('[0-9a-f]{64}', expected), 'Missing SHA: ' + str(path))
        raw = path.is_relative_to(self.contract.video_root.resolve())
        require(raw == video, 'Raw dataset file may only be read by explicit video-hash phase')
        require(path.is_file(), 'Missing artifact: ' + str(path))
        if str(path) not in self.files:
            before = self.signature(path)
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    h.update(block)
            require(before == self.signature(path), 'File changed while hashing: ' + str(path))
            self.files[str(path)] = h.hexdigest()
            self.signatures[str(path)] = before
            self.video_reads += int(video)
        require(expected is None or self.files[str(path)] == expected, 'SHA mismatch: ' + str(path))
        return path

    def unchanged(self):
        for name, signature in self.signatures.items():
            require(self.signature(Path(name)) == signature, 'Verified file changed: ' + name)


def literal_constants(path):
    """Read pinned source constants without importing model/evaluator runtimes."""
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    try:
                        result[name.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return result


def verify_transitive(v, p, s, e):
    """Verify code/model seals already used by unchanged production consumers."""
    root = v.contract.root
    for method in METHODS[:4]:
        ckpt = path_for(s['method_settings'][method]['runtime']['adapter_checkpoint']['path'], v.contract)
        side = read(ckpt.with_suffix('.json'))
        require(side.get('step') == 200 and side.get('variant') == method and
                side.get('checkpoint_sha256') == v.contract.checkpoints[method], 'Wrong early checkpoint sidecar')
        binding_path = ckpt.parent / 'bindings.json'
        binding = read(binding_path)
        require(fingerprint(binding) == side['bindings_fingerprint'], 'Training binding fingerprint mismatch')
        v.check({'path': str(binding_path), 'sha256': hashlib.sha256(binding_path.read_bytes()).hexdigest()})
        for name, sha in binding['sources'].items():
            v.check({'path': str(root / name), 'sha256': sha})
    policy = read(path_for(e['measurement_policy']['path'], v.contract))
    for tool, key in (('openface', 'openface'), ('facenet', 'identity_primary')):
        folder = root / 'tools' / tool
        seal = read(v.check({'path': str(folder / 'tool_seal.json'),
                             'sha256': policy[key]['tool_seal_sha256']}))
        for ref in refs(seal):
            # tool seal paths are relative to the individual tool root.
            v.check({**ref, 'path': str(folder / ref['path'])})
    evaluation = literal_constants(path_for(p['execution_sources']['final_evaluation_measurements']['path'], v.contract))
    for name, sha in evaluation['CORES'].values():
        v.check({'path': str(root / 'development_evaluation/evaluation_policy_v2' / name), 'sha256': sha})
    native = literal_constants(path_for(p['execution_sources']['external_native']['path'], v.contract))
    for name, sha in native['CORE_HASHES'].items():
        v.check({'path': str(root / 'external_baselines/runtime' / name), 'sha256': sha})
    for name, key in (('run_magicface_smoke.py', 'MF_CORE_SHA'), ('magicface_contract.py', 'MF_CONTRACT_SHA')):
        v.check({'path': str(root / 'external_baselines' / name), 'sha256': native[key]})
    seal_ref = s['method_settings']['magicface']['runtime']['model_layout_seal']
    seal = read(path_for(seal_ref['path'], v.contract))
    for ref in seal['models'] + seal['auxiliary_sources']:
        v.check({'path': ref['target'], 'sha256': ref['sha256']})


def validate_shape(p, s, e, plan, inventory, split, accepted, contract):
    require(p['status'] == s['status'] == e['status'] == 'not_frozen' and
            p['extraction_authorized'] is False, 'DRAFT status/authorization changed')
    require(p['kind'] == 'final_test_extraction_protocol' and
            s['kind'] == 'final_test_sampling_settings_DRAFT' and
            e['kind'] == 'final_test_evaluation_policy_DRAFT', 'Unexpected DRAFT kinds')
    require([m['method'] for m in p['models']] == list(METHODS) and
            list(s['method_settings']) == list(METHODS) and e['methods'] == list(METHODS), 'Exact eight methods required')
    require(p['sampling_seeds'] == s['sampling_seeds'] == plan['sampling_seeds_proposed'] == SEEDS and
            p['case_seed_policy'] == 'full_cartesian', 'Exact three-seed Cartesian grid required')
    ids = [c['case_id'] for c in plan['cases']]
    require(len(ids) == len(set(ids)) == 32 and p['case_ids'] == ids, 'Accepted 32 cases/order changed')
    require(tuple(plan['test_subjects']) == tuple(split['groups']['test']) == SUBJECTS and
            split['test_locked'] is True, 'Accepted eight subjects changed')
    require(accepted['status'] == 'label_only_plan_selected' and accepted['case_ids'] == ids and
            accepted['plan_sha256'] == contract.input_shas['label_plan'], 'Label acceptance changed')
    require(s['our_generator_uses_target_pixels'] is False, 'Target-pixel boundary changed')
    for method in METHODS[:5]:
        settings = s['method_settings'][method]
        require(settings['native_parameters'] == {'height': 512, 'width': 512, 'num_inference_steps': None,
                'guidance_scale': 7.5, 'au_scale': 1, 'ip_scale': .9, 'scheduler': 'DEISMultistepScheduler',
                'negative_prompt': ''}, 'Our fixed sampling parameters changed')
        if method in contract.checkpoints:
            ref = settings['runtime']['adapter_checkpoint']
            require(ref['sha256'] == contract.checkpoints[method] and
                    any(a['path'] == ref['path'] and a['sha256'] == ref['sha256'] for
                        a in next(m['artifacts'] for m in p['models'] if m['method'] == method)), 'Early200 checkpoint changed')
    require(s['source_features']['anchor_checkpoint'] == s['method_settings']['bidirectional']['runtime']['adapter_checkpoint'], 'Source anchor changed')
    require(s['method_settings']['magicface']['native_parameters']['steps'] == 50 and
            s['method_settings']['faceadapter']['native_parameters']['steps'] == 25, 'Native baseline steps changed')
    lp = s['method_settings']['liveportrait']
    require(lp['native_parameters']['flag_relative_motion'] is False and
            lp['native_parameters']['animation_region'] == 'all' and
            lp['seed_execution'] == {'mode': 'deterministic_alias', 'actual_seed': 101}, 'LP absolute/all/alias changed')
    expected = []
    for family, references, role in (('external_target_au', METHODS[5:], 'primary'),
                                     ('internal_target_au', ('public_baseline', *METHODS[1:4]), 'secondary_structure')):
        for ref in references:
            expected.append({'id': f'bidirectional_vs_{ref}__target_mae_changed__edited', 'method': 'bidirectional',
                             'reference': ref, 'metric': 'target_mae_changed__edited', 'family': family, 'analysis_role': role})
    for metric in DIAGNOSTICS:
        for ref in (*METHODS[5:], 'public_baseline', *METHODS[1:4]):
            expected.append({'id': f'bidirectional_vs_{ref}__{metric}__edited', 'method': 'bidirectional',
                             'reference': ref, 'metric': metric + '__edited', 'family': 'diagnostic_' + metric, 'analysis_role': 'descriptive'})
    require(e['comparisons'] == expected and e['primary_families'] == ['external_target_au'] and
            e['secondary_families'] == ['internal_target_au'] and
            e['descriptive_families'] == ['diagnostic_' + m for m in DIAGNOSTICS], 'Exact primary3/secondary4/diagnostic28 families changed')
    require(len(e['metrics']) == 42, 'All14 metrics x three scopes must remain')
    require(set(p['draft_pending']) == {'our_inference_steps', 'test_video_content_hashes', 'authorization'} and
            set(s['draft_pending']) == {'num_inference_steps'}, 'Unexpected pending fields')
    headers = {(r['subject'], r['camera']): r for r in inventory['video_headers']}
    require([r['subject'] for r in p['videos']] == list(SUBJECTS), 'Exact eight videos/order required')
    for video in p['videos']:
        cases = [c for c in plan['cases'] if c['subject'] == video['subject']]
        require(len(cases) == 4 and all(c['split'] == 'test' for c in cases), 'Four accepted cases per subject required')
        require(video['sha256'] is None and {c['camera'] for c in cases} == {video['camera']} and
                {str(path_for(c['video_path'], contract)) for c in cases} == {str(path_for(video['path'], contract))}, 'Video/camera selection changed')
        require(path_for(video['path'], contract).is_relative_to(contract.video_root.resolve()), 'Video outside raw dataset')
        header = headers[(video['subject'], video['camera'])]
        require(video['video_metadata_sha256'] == header['video_metadata_sha256'] and
                all(c['video_metadata_sha256'] == video['video_metadata_sha256'] for c in cases), 'Accepted video header binding changed')


def prepare(steps, decision_path, contract=PRODUCTION):
    require(type(steps) is int and steps in (20, 50), 'Explicit --steps20 or --steps50 required')
    final = contract.root / 'final_test'
    outputs = [final / name for name in ('frozen_sampling.json', 'frozen_evaluation.json', 'frozen_protocol.json')]
    require(not any(p.exists() or p.is_symlink() for p in outputs), 'Formal output already exists; never overwrite or refreeze')
    decision_path = Path(decision_path).resolve()
    require(decision_path.is_relative_to(contract.root.resolve()) and decision_path.suffix == '.json' and decision_path.is_file(), 'Existing selection_decision within experiment required')
    decision = read(decision_path)
    require(decision.get('kind') == 'final_test_selection_decision' and
            decision.get('status') == 'selected_on_development_only' and
            decision.get('selected_num_inference_steps') == steps and
            decision.get('selected_method') == 'bidirectional' and decision.get('selected_checkpoint_step') == 200 and
            decision.get('selected_ip_scale') == .9 and decision.get('test_pixels_read') is False and
            decision.get('test_video_full_content_read') is False and isinstance(decision.get('reason'), str) and
            decision['reason'].strip() and isinstance(decision.get('evidence'), list) and decision['evidence'], 'Selection decision is incomplete or differs from --steps')
    v = Verifier(contract)
    v.check({'path': str(decision_path), 'sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest()})
    for ref in decision['evidence']:
        path = path_for(ref['path'], contract)
        require(path.is_relative_to(contract.root / 'development_evaluation') and path.suffix in ('.json', '.md'), 'Selection evidence must be development metadata/report')
        v.check(ref)
    draft = final / 'draft_configuration_20260915'
    docs = [read(v.check({'path': str(draft / name), 'sha256': contract.draft_shas[name]}))
            for name in ('protocol.DRAFT.json', 'sampling.DRAFT.json', 'evaluation.DRAFT.json')]
    p, s, e = docs
    for role, expected in contract.input_shas.items():
        require(p['inputs'][role]['sha256'] == expected, 'Accepted input hash changed: ' + role)
    require(path_for(p['inputs']['sampling_settings']['path'], contract) == draft / 'sampling.DRAFT.json' and
            path_for(p['inputs']['evaluation_policy']['path'], contract) == draft / 'evaluation.DRAFT.json', 'DRAFT cross-reference changed')
    # Metadata/model/code references only. The video list is intentionally excluded.
    for value in (p['inputs'], p['models'], p['execution_sources'], s, e):
        for reference in refs(value):
            v.check(reference)
    for name, key in (('extract_final_test_frames.py', 'extractor_sha256'), ('prepare_disfa_pilot.py', 'legacy_helper_sha256')):
        v.check({'path': str(contract.root / 'data_prepare' / name), 'sha256': p[key]})
    meta = {key: read(path_for(p['inputs'][key]['path'], contract)) for key in
            ('label_plan', 'label_inventory', 'subject_split', 'label_acceptance')}
    validate_shape(p, s, e, *[meta[k] for k in ('label_plan', 'label_inventory', 'subject_split', 'label_acceptance')], contract)
    verify_transitive(v, p, s, e)
    p, s, e = copy.deepcopy(docs)
    for method in METHODS[:5]:
        s['method_settings'][method]['native_parameters']['num_inference_steps'] = steps
    s.update(kind='final_test_sampling_settings', status='frozen_for_final_test',
             selection_context=f'Development-only choice: common early200/IP0.9/DEIS{steps}; see protocol freeze_receipt.selection_decision')
    e.update(kind='final_test_evaluation_policy', status='frozen_for_final_test')
    p.update(status='frozen_final_test', extraction_authorized=True,
             note='Frozen after development-only selection; case/model/evaluation data retained from pinned DRAFTs')
    del p['draft_pending']
    del s['draft_pending']
    # Existing analysis gate is pure schema validation: unlike extraction preflight,
    # it does not read any label, video, pixel, or model files.
    from evaluation_contract import analysis_contract
    analysis_contract(p, s, e)
    v.unchanged()
    return v, p, s, e, outputs, decision_path


def exclusive_write(path, content):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def run(steps, decision_path, *, execute=False, contract=PRODUCTION):
    v, p, s, e, outputs, decision_path = prepare(steps, decision_path, contract)
    summary = {'status': 'ready_not_frozen', 'steps': steps, 'methods': list(METHODS), 'cases': 32,
               'seeds': SEEDS, 'comparisons': {'primary': 3, 'secondary': 4, 'diagnostic': 28},
               'verified_nonvideo_files': len(v.files), 'test_video_full_content_read': False,
               'test_pixels_read': False, 'models_loaded': False, 'frozen_protocol_created': False}
    if not execute:
        return summary
    lock = contract.root / 'final_test/.freeze_protocol.lock'
    exclusive_write(lock, encode({'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}))
    try:
        require(not any(path.exists() or path.is_symlink() for path in outputs), 'Formal output appeared; refuse overwrite')
        for video in p['videos']:
            path = v.check(video, video=True)
            video['sha256'] = v.files[str(path)]
        v.unchanged()
        p['freeze_receipt'] = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection_decision': {'path': str(decision_path), 'sha256': v.files[str(decision_path)]},
            'source_drafts': contract.draft_shas, 'selected_steps': steps,
            'freezer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'verified_file_shas': dict(v.files), 'video_full_hash_count': v.video_reads,
            'test_pixels_decoded': False, 'models_loaded': False}
        require(v.video_reads == 8, 'Exactly eight original video hashes required')
        # Compute output hashes from exact bytes before publication. All validation
        # is complete before the first file; formal protocol is committed last.
        content = [encode(s), encode(e)]
        for key, path, data in zip(('sampling_settings', 'evaluation_policy'), outputs[:2], content):
            p['inputs'][key] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
        for path, data in zip(outputs[:2], content):
            exclusive_write(path, data)
        v.unchanged()
        for path, data in zip(outputs[:2], content):
            require(path.read_bytes() == data, 'Published config readback mismatch')
        exclusive_write(outputs[2], encode(p))
        return {**summary, 'status': 'frozen_final_test', 'test_video_full_content_read': True,
                'frozen_protocol_created': True, 'outputs': {path.name: {'path': str(path),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in outputs}}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', required=True, type=int, choices=(20, 50))
    parser.add_argument('--selection-decision', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Root-only decision: hash accepted test videos and freeze once')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.steps, args.selection_decision, execute=args.execute), ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'blocked', 'error': str(error), 'error_type': type(error).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
