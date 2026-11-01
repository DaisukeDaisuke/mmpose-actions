"""Export official learned RTMW3D L/X weights, validate, and pack browser outputs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto
from onnxconverter_common import float16
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn

from onnx_graph import append_simcc_decoder, widen_float16_for_cpu_reference
from quality import drift, passes_fullbody_gate

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 1000:
        return target
    temporary = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'mmpose-actions/0.1'})
            with urllib.request.urlopen(request, timeout=180) as src, temporary.open('wb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            temporary.replace(target)
            return target
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()


def samples(upstream, image_ids=(40083, 197388), all_people=False):
    # Real person crops from the pinned upstream test corpus, with supplied COCO boxes.
    annotations = json.loads((upstream / 'tests/data/coco/test_coco.json').read_text(encoding='utf-8'))
    images = {item['id']: item for item in annotations['images']}
    chosen = [a for a in annotations['annotations'] if a.get('num_keypoints', 0) >= 14]
    chosen = ([a for a in chosen if a['image_id'] in image_ids] if all_people else
              [next(a for a in chosen if a['image_id'] == image_id) for image_id in image_ids])
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    result = []
    for ann in chosen:
        info = images[ann['image_id']]
        image = Image.open(upstream / 'tests/data/coco' / info['file_name']).convert('RGB')
        x, y, w, h = ann['bbox']
        cx, cy = x + w / 2, y + h / 2
        w, h = w * 1.25, h * 1.25
        if w / h > .75:
            h = w / .75
        else:
            w = h * .75
        patch = image.transform((288, 384), Image.Transform.AFFINE, (w / 288, 0, cx - w / 2, 0, h / 384, cy - h / 2), Image.Resampling.BILINEAR)
        data = ((np.asarray(patch, dtype=np.float32) - mean) / std).transpose(2, 0, 1)[None].copy()
        result.append((info['file_name'], data))
    return result


class RawSimCC(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone, self.neck, self.head = model.backbone, model.neck, model.head

    def forward(self, image):
        return self.head(self.neck(self.backbone(image)))


def session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model if isinstance(model, bytes) else str(model), opts, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=CATALOG['models'], required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('release'))
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != CATALOG['upstream_revision']:
        raise RuntimeError(f'Upstream revision mismatch: {revision}')
    spec = CATALOG['models'][args.model]
    sys.path.insert(0, str(upstream / 'projects/rtmpose3d'))
    from mmengine.config import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS
    import mmpose.models  # register standard modules
    import mmpose.codecs
    import rtmpose3d  # register project modules from the pinned checkout

    torch.set_num_threads(2)
    torch.manual_seed(0)
    init_default_scope('mmpose')
    config = Config.fromfile(str(upstream / spec['config']))
    config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = revert_sync_batchnorm(MODELS.build(config.model)).cpu().eval()
    checkpoint = download(spec['checkpoint'], ROOT / '.cache/checkpoints' / Path(spec['checkpoint']).name)
    ckpt = torch.load(checkpoint, map_location='cpu')
    weights = ckpt.get('state_dict', ckpt)
    weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    loaded = model.load_state_dict(weights, strict=False)
    missing = [k for k in loaded.missing_keys if not k.endswith('num_batches_tracked') and not k.startswith('data_preprocessor.')]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(f'Checkpoint contract mismatch: missing={missing}; unexpected={loaded.unexpected_keys}')
    parameter_count = sum(p.numel() for p in model.parameters())
    network = RawSimCC(model).eval()
    cases = samples(upstream)
    reference = []
    with torch.inference_mode():
        for _, data in cases:
            reference.append([x.numpy() for x in network(torch.from_numpy(data))])
    del ckpt, weights, model
    gc.collect()

    work = ROOT / 'build' / args.model
    out = args.output.resolve()
    (out / 'onnx').mkdir(parents=True, exist_ok=True)
    (out / 'reports').mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    raw_path = work / 'simcc-fp32.onnx'
    with torch.inference_mode():
        torch.onnx.export(network, torch.from_numpy(cases[0][1]), str(raw_path), opset_version=17,
                          input_names=['input'], output_names=['simcc_x', 'simcc_y', 'simcc_z'],
                          do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
    del network
    gc.collect()
    raw = onnx.load(str(raw_path))
    onnx.checker.check_model(raw)
    raw_session = session(raw_path)
    comparison = []
    official_decoder = rtmpose3d.SimCC3DLabel(input_size=(288, 384, 288), simcc_split_ratio=2, normalize=False, use_dark=False, root_index=(11, 12))
    decoded_references = []
    for (name, data), expected in zip(cases, reference):
        observed = raw_session.run(None, {'input': data})
        for a, b in zip(observed, expected):
            np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-4)
        coordinates, _, scores = official_decoder.decode(*observed)
        decoded_references.append((coordinates.copy(), scores.copy()))
        comparison.append({'image': name, 'maxAbsoluteSimCCError': max(float(np.max(np.abs(a - b))) for a, b in zip(observed, expected)),
                           'observedJointsAt0_3': int((scores > .3).sum())})
    del raw_session, reference
    gc.collect()

    # Export a base artifact once per architecture. Quantization policies run in
    # separate parallel jobs and consume exactly the same weights and inputs.
    base = out / 'base' / args.model
    base.mkdir(parents=True, exist_ok=True)
    onnx.save_model(raw, str(base / 'simcc-fp32.onnx'))
    np.savez_compressed(base / 'validation.npz', **{f'input_{i}': data for i, (_, data) in enumerate(cases)},
                        **{f'keypoints_{i}': ref[0] for i, ref in enumerate(decoded_references)},
                        **{f'scores_{i}': ref[1] for i, ref in enumerate(decoded_references)})
    calibration_cases = samples(upstream, image_ids=(785, 196141), all_people=True)
    calibration_arrays = []
    mean = np.asarray([123.675, 116.28, 103.53], np.float32)[None, :, None, None]
    std = np.asarray([58.395, 57.12, 57.375], np.float32)[None, :, None, None]
    for _, data in calibration_cases:
        for factor in [.8, 1., 1.2]:
            rgb = np.clip((data * std + mean) * factor, 0, 255)
            calibration_arrays.append(((rgb - mean) / std).astype(np.float32))
    np.savez_compressed(base / 'calibration.npz', **{f'input_{i}': x for i, x in enumerate(calibration_arrays)})
    (base / 'provenance.json').write_text(json.dumps({'model': args.model, 'upstreamRevision': revision,
        'checkpointSha256': digest(checkpoint), 'validationImages': [name for name, _ in cases],
        'calibrationImages': sorted({name for name, _ in calibration_cases}), 'calibrationCrops': len(calibration_arrays)}), encoding='utf-8')
    fixtures = out / 'fixtures'
    fixtures.mkdir(parents=True, exist_ok=True)
    with (fixtures / 'inputs.bin').open('wb') as f:
        for _, data in cases:
            f.write(data.astype('<f4').tobytes())
    (fixtures / f'{args.model}-reference.json').write_text(json.dumps({'model': args.model, 'names': [n for n, _ in cases],
        'shape': [1, 3, 384, 288], 'keypoints': [r[0].tolist() for r in decoded_references], 'scores': [r[1].tolist() for r in decoded_references]}), encoding='utf-8')

    report = {'model': args.model, 'upstreamRepository': 'https://github.com/open-mmlab/mmpose', 'upstreamRevision': revision,
              'config': spec['config'], 'checkpointUrl': spec['checkpoint'], 'checkpointSha256': digest(checkpoint),
              'parameters': parameter_count, 'officialMetrics': {'cocoWholebodyAP': spec['official_coco_wholebody_ap'], 'h3wbMPJPE_m': spec['official_h3wb_mpjpe_m']},
              'pytorchVsOnnx': comparison, 'variants': [], 'browserGPUValidated': False, 'targetGPU60FPSValidated': False}
    for precision in ['fp32', 'fp16']:
        if precision == 'fp32':
            raw_variant = raw
        else:
            blocked = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | {'ReduceL2', 'ReduceSum', 'Sqrt', 'Pow'})
            raw_variant = float16.convert_float_to_float16(raw, keep_io_types=True, op_block_list=blocked, disable_shape_infer=False)
        compact = append_simcc_decoder(raw_variant)
        path = out / 'onnx' / f'{args.model}-{precision}.onnx'
        onnx.save_model(compact, str(path))
        onnx.checker.check_model(str(path))
        # CPU fp16 support differs by runner: always explicitly disclose this reference mode.
        validation_model = compact if precision == 'fp32' else widen_float16_for_cpu_reference(compact)
        evaluator = session(validation_model.SerializeToString())
        comparisons, latency = [], []
        for (name, data), (expected_coords, expected_scores) in zip(cases, decoded_references):
            started = time.perf_counter()
            coords, scores = evaluator.run(None, {'input': data})
            latency.append((time.perf_counter() - started) * 1000)
            if not np.isfinite(coords).all() or not np.isfinite(scores).all():
                raise RuntimeError(f'Nonfinite {precision} output')
            assert coords.shape == (1, 133, 3) and scores.shape == (1, 133)
            confident = expected_scores > .3
            xy_error = np.linalg.norm((coords - expected_coords)[..., :2], axis=-1)[confident]
            z_error = np.abs(coords[..., 2] - expected_coords[..., 2])[confident]
            score_error = float(np.max(np.abs(scores - expected_scores)))
            if precision == 'fp32':
                np.testing.assert_allclose(coords, expected_coords, atol=2e-5, rtol=1e-6)
                np.testing.assert_allclose(scores, expected_scores, atol=2e-5, rtol=1e-6)
            comparisons.append({'image': name, 'xyErrorP95_px': float(np.quantile(xy_error, .95)), 'zErrorP95_m': float(np.quantile(z_error, .95)), 'scoreMaxAbsoluteError': score_error,
                                'parts': drift(coords, scores, expected_coords, expected_scores)})
        del evaluator, validation_model
        gc.collect()
        half_initializers = sum(t.data_type == TensorProto.FLOAT16 for t in compact.graph.initializer)
        if precision == 'fp16' and half_initializers == 0:
            raise RuntimeError('FP16 conversion produced no FP16 weights')
        report['variants'].append({'id': f'{args.model}-{precision}', 'file': f'onnx/{path.name}', 'sizeBytes': path.stat().st_size, 'sha256': digest(path),
            'precision': precision, 'fp16Initializers': half_initializers, 'input': {'name': 'input', 'type': 'float32', 'shape': [1, 3, 384, 288]},
            'outputs': [{'name': 'keypoints', 'type': 'float32', 'shape': [1, 133, 3], 'units': ['crop-pixels', 'crop-pixels', 'root-relative-metres']}, {'name': 'scores', 'type': 'float32', 'shape': [1, 133]}],
            'outputBytesPerFrame': 2128, 'oldSimCCOutputBytesPerFrame': 1021440,
            'validationMode': 'CPU-fp32' if precision == 'fp32' else 'CPU-fp32-compute-with-fp16-rounded-weights-and-constants',
            'policy': 'fp32' if precision == 'fp32' else 'fp16-all',
            'qualityPassed': passes_fullbody_gate(comparisons),
            'nativeFP16GPUValidated': False, 'referenceLatencyMs': latency, 'comparisons': comparisons})
        print(json.dumps(report['variants'][-1], indent=2), flush=True)
        del compact
        if precision == 'fp16':
            del raw_variant
        gc.collect()
    destination = out / 'reports' / f'{args.model}.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Validated: {destination}', flush=True)


if __name__ == '__main__':
    main()
