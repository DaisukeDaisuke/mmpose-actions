"""Parallel, layer-selected FP16 / calibrated QDQ INT8 experiments."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

from onnx_graph import append_simcc_decoder, widen_float16_for_cpu_reference
from quality import drift, passes_fullbody_gate

POLICIES = ['fp16-backbone', 'fp16-neck-head', 'int8-backbone', 'int8-neck', 'int8-head', 'int8-all']


class CalibrationInputs(CalibrationDataReader):
    def __init__(self, path):
        data = np.load(path)
        self.items = [{'input': data[k].astype(np.float32)} for k in sorted(data.files)]
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.items)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rtmw3d-l', 'rtmw3d-x'], required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('release'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'onnx').mkdir(exist_ok=True)
    (args.output / 'reports').mkdir(exist_ok=True)
    original = onnx.load(str(args.base / 'simcc-fp32.onnx'))
    provenance = json.loads((args.base / 'provenance.json').read_text(encoding='utf-8'))
    policy = args.policy
    groups = ['backbone'] if policy.endswith('-backbone') else ['neck', 'head'] if policy.endswith('-neck-head') else ['neck'] if policy.endswith('-neck') else ['head'] if policy.endswith('-head') else ['backbone', 'neck', 'head']
    selected = [n for n in original.graph.node if any(f'/{group}/' in n.name for group in groups)]
    if not selected:
        raise RuntimeError(f'No nodes matched the requested layer groups: {groups}')
    target_nodes = selected
    if policy.startswith('fp16'):
        selected_names = {n.name for n in selected}
        blocked = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | {'ReduceL2', 'ReduceSum', 'Sqrt', 'Pow'})
        variant = float16.convert_float_to_float16(original, keep_io_types=True, op_block_list=blocked,
            node_block_list=[n.name for n in original.graph.node if n.name not in selected_names])
        precision = 'mixed-fp16'
    else:
        target_nodes = [n for n in selected if n.op_type in ['Conv', 'MatMul', 'Gemm']]
        if not target_nodes:
            raise RuntimeError(f'No quantizable Conv/MatMul/Gemm matched {groups}')
        work = Path('build') / args.model / policy
        work.mkdir(parents=True, exist_ok=True)
        quantized_path = work / 'simcc-qdq.onnx'
        quantize_static(str(args.base / 'simcc-fp32.onnx'), str(quantized_path), CalibrationInputs(args.base / 'calibration.npz'),
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            per_channel=True, calibrate_method=CalibrationMethod.MinMax, nodes_to_quantize=[n.name for n in target_nodes],
            extra_options={'WeightSymmetric': True, 'ActivationSymmetric': False})
        variant = onnx.load(str(quantized_path))
        precision = 'qdq-int8'
    compact = append_simcc_decoder(variant)
    destination = args.output / 'onnx' / f'{args.model}-{policy}.onnx'
    onnx.save_model(compact, str(destination))
    onnx.checker.check_model(str(destination))
    evaluation = widen_float16_for_cpu_reference(compact) if policy.startswith('fp16') else compact
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(evaluation.SerializeToString(), options, providers=['CPUExecutionProvider'])
    data = np.load(args.base / 'validation.npz')
    comparisons, times = [], []
    index = 0
    while f'input_{index}' in data:
        source, expected, expected_score = data[f'input_{index}'], data[f'keypoints_{index}'], data[f'scores_{index}']
        session.run(None, {'input': source})  # warm-up excluded
        for _ in range(5):
            start = time.perf_counter()
            points, scores = session.run(None, {'input': source})
            times.append((time.perf_counter() - start) * 1000)
        if not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite quantized output')
        parts = drift(points, scores, expected, expected_score)
        comparisons.append({'image': provenance['validationImages'][index], 'parts': parts,
            'scoreMaxAbsoluteError': float(np.max(np.abs(scores - expected_score)))})
        index += 1
    del session, evaluation
    gc.collect()
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    quality = passes_fullbody_gate(comparisons)
    result = {'id': f'{args.model}-{policy}', 'model': args.model, 'file': f'onnx/{destination.name}', 'policy': policy,
        'precision': precision, 'layerGroups': groups, 'selectedNodeCount': len(target_nodes), 'selectedNodes': [n.name for n in target_nodes],
        'sizeBytes': destination.stat().st_size, 'sha256': digest, 'outputBytesPerFrame': 2128,
        'input': {'name': 'input', 'type': 'float32', 'shape': [1, 3, 384, 288]},
        'outputs': [{'name': 'keypoints', 'shape': [1, 133, 3], 'type': 'float32'}, {'name': 'scores', 'shape': [1, 133], 'type': 'float32'}],
        'qualityPassed': quality, 'qualityGate': 'Independent body/hip/knee/ankle/foot gates: p95 <= 3 crop px, p95 <= 0.05 relative-Z m, no confidence losses. Small fixture only.',
        'comparisons': comparisons, 'referenceLatencyMs': times, 'cpuP50Ms': float(np.quantile(times, .5)), 'cpuP95Ms': float(np.quantile(times, .95)),
        'validationMode': 'CPU-fp32-compute-with-fp16-rounded-weights-and-constants' if policy.startswith('fp16') else 'CPU-ONNXRuntime-QDQ',
        'browserGPUValidated': False, 'nativeFP16GPUValidated': False, 'webGPUOperatorSupportVerified': False,
        'provenance': provenance,
        'limitations': ['CPU timings from different runners are not a GPU speed ranking.', 'A small held-out fixture is not a COCO/H3WB accuracy evaluation.',
                       'FP16 and INT8 can be slower; browser measurements determine which variant is useful.']}
    (args.output / 'reports' / f'{args.model}-{policy}.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['id', 'sizeBytes', 'qualityPassed', 'cpuP50Ms', 'layerGroups', 'selectedNodeCount']}, indent=2))


if __name__ == '__main__':
    run()
