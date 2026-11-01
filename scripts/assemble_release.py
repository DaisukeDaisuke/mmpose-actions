"""Assemble only complete, hash-verified experiment artifacts for publication."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
release = Path(sys.argv[1] if len(sys.argv) > 1 else 'release')
settings = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
variants = []
for path in sorted((release / 'reports').glob('*.json')):
    report = json.loads(path.read_text(encoding='utf-8'))
    if 'variants' in report:
        for variant in report['variants']:
            variants.append({**variant, 'model': report['model'], 'report': f'reports/{path.name}'})
    else:
        variants.append({**report, 'report': f'reports/{path.name}'})
if len(variants) != 16 or len({v['id'] for v in variants}) != 16:
    raise RuntimeError(f'Expected 2 full-body architectures × 8 policies, got {len(variants)}')
for variant in variants:
    path = release / variant['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != variant['sha256']:
        raise RuntimeError(f'Artifact hash mismatch: {path}')
    if variant['input']['shape'] != [1, 3, 384, 288] or variant['outputs'][0]['shape'] != [1, 133, 3]:
        raise RuntimeError('All 133 full-body joints must remain')
runtime = json.loads((release / 'runtime/manifest.json').read_text(encoding='utf-8'))
if runtime['version'] != '1.27.0' or not runtime.get('patchSha256'):
    raise RuntimeError('Expected the version-matched MatMul f32 accumulation patch')
for entry in runtime['files']:
    if hashlib.sha256((release / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError(f'Runtime checksum mismatch: {entry["path"]}')
catalog = {'schema': 'rtmw3d-webgpu-onnx/v1', 'producer': 'DaisukeDaisuke/mmpose-actions',
    'sourceCommit': os.environ.get('GITHUB_SHA'), 'sourceRunId': os.environ.get('GITHUB_RUN_ID'),
    'upstream': 'https://github.com/open-mmlab/mmpose', 'upstreamRevision': settings['upstream_revision'],
    'huggingfaceRepo': settings['huggingface_repo_id'], 'keypointCount': 133,
    'keypointGroups': {'body': [0, 16], 'ankles': [15, 16], 'feet': [17, 22], 'face': [23, 90], 'hands': [91, 132]},
    'cameraInputs': {'min': 1, 'max': 4, 'modelBatch': 1, 'method': 'reuse one session; distinguish per-source and aggregate inference rates'},
    'coordinateContract': 'XY are crop pixels; Z is root-relative metres. Absolute root/world position requires calibration/multi-view reconstruction.',
    'runtime': runtime, 'variants': sorted(variants, key=lambda v: v['id']), 'browserGPUValidated': False, 'GPU60FPSClaimed': False}
(release / 'models.json').write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
shutil.copyfile(ROOT / 'MODEL_CARD.md', release / 'README.md')
shutil.copyfile(ROOT / 'UPSTREAM_LICENSE', release / 'LICENSE')
shutil.copytree(ROOT / 'benchmark', release / 'benchmark', dirs_exist_ok=True)
lines = ['| Full-body variant | Size MB | Body/feet smoke gate | CPU reference p50 ms |', '|---|---:|---|---:|']
for variant in catalog['variants']:
    samples = sorted(variant['referenceLatencyMs'])
    lines.append(f'| {variant["id"]} | {variant["sizeBytes"] / 1e6:.1f} | {"pass" if variant["qualityPassed"] else "inspect report"} | {samples[len(samples)//2]:.1f} |')
summary = '\n'.join(lines) + '\n\nCPU measurements are not a WebGPU ranking; use the bundled benchmark on the target GPU.\n'
(release / 'COMPARISON.md').write_text(summary, encoding='utf-8')
with (release / 'README.md').open('a', encoding='utf-8') as file:
    file.write('\n## Build comparison\n' + summary)
if os.environ.get('GITHUB_STEP_SUMMARY'):
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as file:
        file.write(summary)
print(json.dumps({'variants': len(variants), 'fullBodyJoints': 133, 'totalModelBytes': sum(v['sizeBytes'] for v in variants)}, indent=2))
