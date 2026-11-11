#!/usr/bin/env python3
"""CPU-only original Face-Adapter conditioning; stop before the diffusion call."""
import argparse
import ast
import contextlib
import copy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from prepare_dev_contracts import HERE, OLD, REPOS, sha, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    document = json.loads(args.manifest.read_text())
    assert document['schema_version'] == 3 and not document['test_used'] and len(document['cases']) == 19
    case = next(row for row in document['cases'] if row['group'] != 'no_edit')
    assert case['seed'] == 101
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
                      HF_HOME=str(HERE / 'cache/huggingface'), TORCH_HOME=str(HERE / 'cache/torch'),
                      MPLCONFIGDIR=str(HERE / 'cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'cache/xdg'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    runtime = HERE / 'runtime'
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location('faceadapter_preprocess_wrapper', runtime / 'run_faceadapter.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    repo = REPOS['faceadapter']
    wrapper.git_state(repo)
    loaded = wrapper.load_cases(args.manifest, {case['case_id']})
    assert len(loaded) == 1
    case = loaded[0]
    tree = ast.parse((repo / 'infer.py').read_text())
    infer = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'infer')
    src_for = next(x for x in infer.body if isinstance(x, ast.For) and getattr(x.target, 'id', None) == 'src_im_file')
    drive_for = next(x for x in src_for.body if isinstance(x, ast.For))
    initialization = []
    collect = False
    for node in infer.body:
        if wrapper.assigned(node, 'src_img_list'):
            break
        if wrapper.assigned(node, 'net_d3dfr'):
            collect = True
        if collect:
            initialization.append(node)
    before_drive = src_for.body[:src_for.body.index(drive_for)]
    stop = next(i for i, node in enumerate(drive_for.body) if wrapper.assigned(node, 'image') and
                any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'pipe' for child in ast.walk(node.value)))
    conditioning = before_drive + drive_for.body[:stop]
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'pipe' for node in conditioning for n in ast.walk(node))
    def compiled(nodes):
        module = wrapper.Compatibility().visit(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
        return compile(ast.fix_missing_locations(module), '<author-FaceAdapter-CPU-conditioning>', 'exec')
    module_code, _, _ = wrapper.split_upstream((repo / 'infer.py').read_text())
    base = HERE / ('runs/faceadapter_cpu_conditioning_' + datetime.now().strftime('%Y%m%d%H%M'))
    out = base
    i = 2
    while out.exists():
        out = base.with_name(base.name + f'-{i:02d}')
        i += 1
    out.mkdir(parents=True)
    sys.path.insert(0, str(repo))
    namespace = {'__name__': 'author_faceadapter_cpu_conditioning', '__file__': str(repo / 'infer.py')}
    start = time.monotonic()
    with (out / 'runtime.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        wrapper.load_compatible_pipeline(repo)
        exec(module_code, namespace)
        torch = namespace['torch']
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        proxy = wrapper.TorchLoadProxy(torch)
        namespace['torch'] = proxy
        namespace['model_insightface_backbone'].torch = proxy
        namespace.update(device='cpu', weight_dtype=torch.float32, onnx_providers=['CPUExecutionProvider'], onnx_ctx_id=-1)
        options = SimpleNamespace(checkpoint=str(OLD / 'models/face_adapter'), source=str(Path(case['source']).parent),
                    target=str(Path(case['target']).parent), crop_ratio=wrapper.POLICY['crop_ratio'],
                    empty_prompt=str(repo / 'empty_prompt_embedding.pth'))
        namespace['args'] = options
        namespace['set_seed'](wrapper.POLICY['condition_seed'])
        exec(compiled(initialization), namespace)
        namespace['app'] = wrapper.DetectionPolicy(namespace['app'], 'author_default')
        namespace['app'].begin_case(case['case_id'])
        namespace['set_seed'](wrapper.POLICY['condition_seed'])
        namespace.update(src_im_file=Path(case['source']).name, drive_im_file=Path(case['target']).name, case_seed=case['seed'])
        exec(compiled(conditioning), namespace)
        from PIL import Image
        Image.fromarray(namespace['src_im_crop512']).save(out / 'source_author_crop512.png')
        Image.fromarray(namespace['drive_im_crop512']).save(out / 'driver_author_crop512.png')
        names = ('images_src', 'images_tar', 'image_src_crop256', 'image_tar_crop256', 'im_pts70', 'face_masks_tar',
                 'controlnet_image', 'blend_mask', 'encoder_hidden_states_src', 'controlnet_encoder_hidden_states_src',
                 'controlnet_encoder_hidden_states_tar', 'empty_prompt_token')
        payload = {name: namespace[name].detach().cpu() for name in names}
        assert all(t.device.type == 'cpu' and torch.isfinite(t).all() for t in payload.values())
        torch.save(payload, out / 'conditioning_cpu.pt')
        tensors = {name: {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device),
                          'sha256': __import__('hashlib').sha256(t.numpy().tobytes()).hexdigest()} for name, t in payload.items()}
    executable = [str(Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'), '-B',
                  str(runtime / 'run_faceadapter.py'), '--repo', str(repo), '--checkpoint', str(OLD / 'models/face_adapter'),
                  '--base-model', str(OLD / 'models/stable-diffusion-v1-5'), '--vae-model', str(OLD / 'models/sd-vae-ft-mse'),
                  '--manifest', str(args.manifest.resolve()), '--case-id', case['case_id'], '--output',
                  str(runtime / 'runs' / ('faceadapter_smoke_raw_native_' + datetime.now().strftime('%Y%m%d%H%M'))),
                  '--device', 'mps', '--preload-final-components', '--detection-policy', 'author_default', '--execute']
    save(out / 'prepared.json', {'status': 'complete', 'case': case, 'tensors': tensors, 'conditioning_sha256': sha(out / 'conditioning_cpu.pt'),
            'source_crop_sha256': sha(out / 'source_author_crop512.png'), 'driver_crop_sha256': sha(out / 'driver_author_crop512.png'),
            'detections': namespace['app'].events, 'device': 'cpu', 'diffusion_called': False, 'test_used': False,
            'source_sha256': sha(__file__), 'wrapper_sha256': sha(runtime / 'run_faceadapter.py'), 'upstream_sha256': sha(repo / 'infer.py'),
            'policy': wrapper.POLICY, 'token_encoder_training': {k: namespace[k].training for k in ('net_image2token', 'net_id2token')},
            'seconds': time.monotonic() - start, 'note': 'CPU conditioning checks are not reused to bypass native MPS recomputation; no pipeline/UNet loaded here'})
    save(out / 'mps_command_not_executed.json', {'command': executable, 'requires_root_mps_window': True, 'executed': False})
    (out / 'preparer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(out), 'case_id': case['case_id'], 'status': 'complete', 'device': 'cpu', 'diffusion_called': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
