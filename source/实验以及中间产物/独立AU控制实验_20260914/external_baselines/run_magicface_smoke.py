#!/usr/bin/env python3
"""Actual first-edited raw-v2 MagicFace smoke, separately gated CPU/MPS phases."""
import argparse
import ast
import contextlib
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from prepare_dev_contracts import HERE, REPOS, COMMITS, sha, save
from magicface_contract import validate_case, native_inference_kwargs, PROMPT


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CPUOnly(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            node.func.attr = 'to'
            node.args = [ast.Constant('cpu')]
            node.keywords = []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
            node.args = [ast.Constant('cpu') if isinstance(arg, ast.Constant) and arg.value == 'cuda' else arg for arg in node.args]
        return node


def official_functions(path, namespace, cpu=False):
    source = ast.parse(path.read_text())
    selected = ast.Module(body=[x for x in source.body if isinstance(x, ast.FunctionDef)], type_ignores=[])
    if cpu:
        selected = CPUOnly().visit(selected)
    exec(compile(ast.fix_missing_locations(selected), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('preprocess', 'generate'), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--case-id', help='Explicit frozen case for an isolated full-batch child process')
    args = parser.parse_args()
    out = args.output.resolve()
    if HERE not in out.parents:
        raise ValueError('Output must stay in this experiment')
    contract = json.loads(args.manifest.read_text())
    assert contract['schema_version'] in (2, 3) and not contract['test_used']
    case = (next(row for row in contract['cases'] if row['case_id'] == args.case_id) if args.case_id
            else next(row for row in contract['cases'] if row['group'] != 'no_edit'))
    validate_case(case)
    assert case['seed'] == 101 and sha(case['source']) == case['source_sha256']
    repo = REPOS['magicface']
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMITS['magicface']
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    runtime = HERE / 'magicface_runtime'
    seal = json.loads((runtime / 'layout_seal.json').read_text())
    for row in seal['models'] + seal['auxiliary_sources']:
        assert sha(row['target']) == row['sha256']
    if args.phase == 'preprocess':
        out.mkdir(parents=True, exist_ok=False)
        save(out / 'case.json', {'case': case, 'manifest_sha256': sha(args.manifest), 'repo_commit': COMMITS['magicface'],
                               'layout_seal_sha256': sha(runtime / 'layout_seal.json'), 'test_used': False,
                               'target_image_read': False,
                               'selection': 'explicit frozen case_id' if args.case_id else 'first non-noedit case in frozen contract order'})
    else:
        prior = json.loads((out / 'case.json').read_text())
        assert prior['case'] == case and prior['manifest_sha256'] == sha(args.manifest)
        assert not (out / 'generated.png').exists()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='1',
                      NO_ALBUMENTATIONS_UPDATE='1', MPLCONFIGDIR=str(HERE / 'cache/matplotlib'),
                      HF_HOME=str(HERE / 'cache/huggingface'), TORCH_HOME=str(HERE / 'cache/torch'),
                      XDG_CACHE_HOME=str(HERE / 'cache/xdg'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    try:
        if args.phase == 'preprocess':
            import cv2
            import matplotlib.pyplot as plt
            import onnxruntime as ort
            from insightface.model_zoo import get_model
            from torchvision.utils import save_image
            dataset = module_from_file('magicface_author_dataset', repo / 'utils/data/datasets_faceswap.py')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            detector = get_model(str(runtime / 'models/preprocess/scrfd_10g_bnkps.onnx'), providers=['CPUExecutionProvider'], sess_options=opts)
            detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=.5)
            events = []
            class App:
                def get(self, bgr):
                    boxes, points = detector.detect(bgr, max_num=0, metric='default')
                    events.append({'shape': list(bgr.shape), 'face_count': len(boxes), 'boxes': boxes.tolist()})
                    return [{'bbox': box[:4], 'det_score': float(box[4]), 'kps': point} for box, point in zip(boxes, points)]
            base_namespace = dict(np=np, cv2=cv2, torch=torch, Image=Image, transforms=transforms,
                                  plt=plt, save_image=save_image, datasets_faceswap=dataset, app=App())
            crop_namespace = dict(base_namespace, pil2tensor=transforms.ToTensor())
            official_functions(repo / 'utils/preprocess.py', crop_namespace)
            crop_path, bg_path = out / 'source_author_crop.png', out / 'source_background_contour.png'
            crop_namespace['crop_one_image'](SimpleNamespace(img_path=case['source'], save_path=str(crop_path)))
            sys.path.insert(0, str(repo / 'utils'))
            import torch.utils.model_zoo as modelzoo
            from model import BiSeNet
            # The full released BiSeNet state immediately overwrites this initialization.
            with contextlib.ExitStack() as stack:
                from unittest.mock import patch
                stack.enter_context(patch.object(modelzoo, 'load_url', return_value={}))
                segmenter = BiSeNet(n_classes=19)
            segmenter.load_state_dict(torch.load(runtime / 'models/preprocess/79999_iter.pth', map_location='cpu', weights_only=True), strict=True)
            segmenter.eval()
            d3 = module_from_file('magicface_author_d3dfr', runtime / 'auxiliary/third_party/model_resnet_d3dfr.py')
            bfm = module_from_file('magicface_author_bfm', runtime / 'auxiliary/third_party/d3dfr/bfm.py')
            d3model = d3.getd3dfr_res50(str(runtime / 'models/preprocess/d3dfr_res50_nofc.pth')).eval()
            bfmmodel = bfm.BFM(focal=1015 * 256 / 224, image_size=256,
                              bfm_model_path=str(runtime / 'models/preprocess/BFM_model_front.mat')).to('cpu')
            bg_namespace = dict(base_namespace, net=segmenter, net_d3dfr=d3model, bfm_facemodel=bfmmodel,
                                pil2tensor=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=.5, std=.5)]))
            official_functions(repo / 'utils/retrieve_bg.py', bg_namespace, cpu=True)
            bg_namespace['make_bg_for_one_image'](SimpleNamespace(img_path=str(crop_path), save_path=str(bg_path)))
            if not bg_path.exists():
                raise RuntimeError('Official source-side background/contour did not produce an image')
            for path in (crop_path, bg_path):
                with Image.open(path) as image:
                    image.load()
                    assert image.size == (512, 512) and image.mode == 'RGB'
            save(out / 'preprocess.json', {'status': 'complete', 'source_raw_sha256': case['source_sha256'],
                   'crop_sha256': sha(crop_path), 'background_sha256': sha(bg_path), 'detector_events': events,
                   'native_crop_source_sha256': sha(repo / 'utils/preprocess.py'),
                   'native_background_source_sha256': sha(repo / 'utils/retrieve_bg.py'),
                   'auxiliary_sources': seal['auxiliary_sources'], 'device': 'cpu', 'target_image_read': False,
                   'changes': ['CPU device placement only inside original preprocessing function bodies',
                               'skip unused ImageNet initialization before strict complete BiSeNet weight load'],
                   'native_bbox_formula_preserved': True, 'seconds': time.monotonic() - started, 'runner_sha256': sha(__file__)})
            print(json.dumps({'phase': 'preprocess', 'status': 'complete', 'case_id': case['case_id'], 'output': str(out)}, ensure_ascii=False), flush=True)
            return
        preprocessing = json.loads((out / 'preprocess.json').read_text())
        assert preprocessing['status'] == 'complete' and not preprocessing['target_image_read']
        crop_path, bg_path = out / 'source_author_crop.png', out / 'source_background_contour.png'
        assert sha(crop_path) == preprocessing['crop_sha256'] and sha(bg_path) == preprocessing['background_sha256']
        sys.path.insert(0, str(repo))
        from diffusers import AutoencoderKL, PNDMScheduler, UniPCMultistepScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from mgface.pipelines_mgface.pipeline_mgface import MgPipeline
        from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
        from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel
        print('[MagicFace] loading pinned local components on ' + args.device, flush=True)
        modelroot, baseroot = runtime / 'models/generator', runtime / 'models/base'
        vae = AutoencoderKL.from_pretrained(str(baseroot), subfolder='vae', local_files_only=True).eval()
        text_encoder = CLIPTextModel.from_pretrained(str(modelroot), subfolder='text_encoder', local_files_only=True).eval()
        tokenizer = CLIPTokenizer.from_pretrained(str(baseroot), subfolder='tokenizer', local_files_only=True)
        id_model, id_info = UNetID2DConditionModel.from_pretrained(str(modelroot), subfolder='ID_enc', use_safetensors=True,
                           low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True, output_loading_info=True)
        deno_model, deno_info = UNetDeno2DConditionModel.from_pretrained(str(modelroot), subfolder='denoising_unet', use_safetensors=True,
                               low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True, output_loading_info=True)
        for info in (id_info, deno_info):
            if info.get('mismatched_keys') or info.get('missing_keys') or info.get('unexpected_keys') or info.get('error_msgs'):
                raise RuntimeError('Official UNet checkpoint does not load exactly: ' + str(info))
        scheduler = PNDMScheduler.from_pretrained(str(baseroot), subfolder='scheduler', local_files_only=True)
        pipeline = MgPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet_ID=id_model, unet_deno=deno_model,
                              scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        for model in (vae, text_encoder, id_model, deno_model):
            model.requires_grad_(False).eval()
        pipeline.to(args.device)
        pipeline.set_progress_bar_config(disable=True)
        text_input = tokenizer([PROMPT], max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')
        prompt_embeds = text_encoder(text_input.input_ids.to(args.device))[0]
        transform = transforms.ToTensor()
        source = transform(Image.open(crop_path).convert('RGB')).unsqueeze(0)
        background = transform(Image.open(bg_path).convert('RGB')).unsqueeze(0)
        generator = torch.Generator(device='cpu').manual_seed(case['seed'])
        kwargs = native_inference_kwargs(case, source, background, prompt_embeds,
                                        background_derived_from_source_sha256=case['source_sha256'], generator=generator)
        print('[MagicFace] starting official 50-step sampling', flush=True)
        # Keep the author's default callback=None. Its optional callback branch
        # refers to an undefined vton_latents local and is not used by inference.py.
        sampled = pipeline(**kwargs).images[0]
        sampled.save(out / 'generated.png')
        array = np.asarray(sampled.convert('RGB'))
        assert sampled.size == (512, 512) and np.isfinite(array).all()
        if args.device == 'mps':
            torch.mps.synchronize()
        save(out / 'generation.json', {'status': 'complete', 'case_id': case['case_id'], 'request_case_id': case['request_case_id'],
               'subject': case['subject'], 'seed': case['seed'], 'device': args.device, 'dtype': str(prompt_embeds.dtype),
               'generator_device': 'cpu', 'generator_compatibility': 'MPS generator uses CPU random stream, transferred by author diffusers utilities',
               'pytorch_mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
               'inference_steps': 50, 'scheduler': type(pipeline.scheduler).__name__, 'guidance_scale': 7.5, 'image_guidance_scale': 1.5,
               'au_order': case['au_order'], 'source_aus': case['source_au_vector_0_5'], 'target_aus': case['target_au_vector_0_5'],
               'au_variation_sent': case['au_delta_native'], 'source_raw_sha256': case['source_sha256'],
               'source_crop_sha256': sha(crop_path), 'background_sha256': sha(bg_path), 'output_sha256': sha(out / 'generated.png'),
               'output': str(out / 'generated.png'), 'output_size': list(sampled.size), 'pixel_std': float(array.std()),
               'model_layout_seal_sha256': sha(runtime / 'layout_seal.json'), 'repo_commit': COMMITS['magicface'],
               'native_pipeline_sha256': sha(repo / 'mgface/pipelines_mgface/pipeline_mgface.py'),
               'id_loading': id_info, 'denoising_loading': deno_info, 'runner_sha256': sha(__file__),
               'target_image_read': False, 'test_used': False, 'seconds_including_load': time.monotonic() - started})
        del pipeline, vae, text_encoder, id_model, deno_model, prompt_embeds, sampled
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
        print(json.dumps({'phase': 'generate', 'status': 'complete', 'output': str(out / 'generated.png'), 'mps_released_on_process_exit': True}, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = out / (args.phase + '_failure.json')
        suffix = 2
        while failure.exists():
            failure = out / (args.phase + f'_failure_{suffix:02d}.json')
            suffix += 1
        save(failure, {'status': 'failed', 'phase': args.phase, 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'case_id': case['case_id'], 'target_image_read': False, 'test_used': False})
        raise


if __name__ == '__main__':
    main()
