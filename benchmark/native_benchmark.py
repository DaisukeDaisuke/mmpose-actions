"""GPU diagnosis only: reuse the browser's exact FP32 model and normalized inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Browser-exported little-endian float32 .f32 file')
    parser.add_argument('--provider', choices=['cuda', 'tensorrt'], default='cuda')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--views', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--output', type=Path, default=Path('rtmw3d-native-comparison.json'))
    args = parser.parse_args()
    provider = {'cuda': 'CUDAExecutionProvider', 'tensorrt': 'TensorrtExecutionProvider'}[args.provider]
    if provider not in ort.get_available_providers():
        raise RuntimeError(f'{provider} is unavailable; no CPU performance substitute is allowed')
    if not 12 <= args.iterations <= 1000:
        raise ValueError('iterations must be between 12 and 1000')
    data = np.fromfile(args.input, dtype='<f4')
    elements = 3 * 384 * 288
    if data.size != 3 * elements or not np.isfinite(data).all():
        raise ValueError('Expected the exact three finite [1,3,384,288] browser inputs')
    samples = data.reshape(3, 1, 3, 384, 288).astype(np.float32)
    providers = [(provider, {'trt_fp16_enable': False})] if args.provider == 'tensorrt' else [provider]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.model), options, providers=providers)
    if session.get_inputs()[0].shape != [1, 3, 384, 288]:
        raise ValueError('Model input does not match the browser contract')

    def run(target, sample):
        outputs = dict(zip([o.name for o in target.get_outputs()], target.run(None, {'input': sample})))
        points, scores = outputs['keypoints'], outputs['scores']
        if points.shape != (1, 133, 3) or scores.shape != (1, 133) or not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise RuntimeError('Invalid full-body output')

    for i in range(5):
        run(session, samples[i % 3])
    times = []
    begin = time.perf_counter()
    for i in range(args.iterations):
        start = time.perf_counter()
        run(session, samples[i % 3])
        times.append((time.perf_counter() - start) * 1000)
    wall = (time.perf_counter() - begin) * 1000
    del session
    # Provider placement and node timing use a separate instrumented session.
    provider_times, nodes = defaultdict(float), []
    with tempfile.TemporaryDirectory(prefix='rtmw-native-profile-') as directory:
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / 'profile')
        diagnostic = ort.InferenceSession(str(args.model), options, providers=providers)
        run(diagnostic, samples[0])
        path = diagnostic.end_profiling()
        events = json.loads(Path(path).read_text(encoding='utf-8'))
        del diagnostic
        for event in events:
            selected = event.get('args', {}).get('provider')
            if selected and event.get('cat') == 'Node' and event.get('name', '').endswith('_kernel_time'):
                milliseconds = event.get('dur', 0) / 1000
                provider_times[selected] += milliseconds
                nodes.append({'name': event['name'], 'provider': selected, 'milliseconds': milliseconds})
    if not any(name != 'CPUExecutionProvider' and total > 0 for name, total in provider_times.items()):
        raise RuntimeError('No GPU provider nodes were observed in the native profile')
    ordered = sorted(times)
    with args.model.open('rb') as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    result = {'schema': 'rtmw3d-native-comparison/v1', 'onnxruntime': ort.__version__,
        'modelSha256': digest.hexdigest(), 'requestedProvider': provider, 'views': args.views,
        'metrics': {'p50Ms': ordered[int((len(ordered) - 1) * .5)], 'p95Ms': ordered[int((len(ordered) - 1) * .95)],
                    'aggregateFPS': len(times) / wall * 1000, 'perInputFPS': len(times) / wall * 1000 / args.views,
                    'actualInferences': len(times), 'wallMs': wall, 'sessionTimesMs': times},
        'separateProfile': {'providerMilliseconds': dict(provider_times), 'topNodes': sorted(nodes, key=lambda n: -n['milliseconds'])[:30]},
        'notes': ['Diagnostic path only; production remains browser-based', 'Same three normalized browser inputs; upload, inference and CPU output retrieval included',
                  'CPU fallback nodes are disclosed in the separate profile', 'No video, tensor contents, landmark coordinates or device identity are included in this report']}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'metrics': result['metrics'] | {'sessionTimesMs': None}}, indent=2))


if __name__ == '__main__':
    main()
