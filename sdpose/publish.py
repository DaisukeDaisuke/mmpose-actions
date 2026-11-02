import hashlib
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

root = Path('release')
reports = []
for path in sorted(root.glob('*/*/manifest.json')):
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['base_model'] == 'teemosliang/SDPose-Wholebody'
    assert report['keypoints'] == 133 and report['dimensions'] == 2
    for entry in report['files']:
        file = path.parent / entry['file']
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == entry['sha256'] and file.stat().st_size == entry['bytes']
    reports.append({'path': str(path.relative_to(root)), **report})
assert len(reports) == 8, len(reports)
catalog = {'base_model': 'teemosliang/SDPose-Wholebody', 'variants': reports,
           'sourceCommit': os.environ['GITHUB_SHA'], 'pipelinePairs': {
               'fp32': ['fp32/vae', 'fp32/pose'],
               'fp16': ['fp16/vae', 'fp16/pose'],
               'int8-weight-all': ['int8-weight-all/vae', 'int8-weight-all/pose'],
               'int8-weight-unet': ['fp32/vae', 'int8-weight-unet/pose'],
               'int8-weight-head': ['fp32/vae', 'int8-weight-head/pose']}}
(root / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
for name in ['README.md', 'LICENSE-SDPose', 'requirements.txt', 'export.py']:
    shutil.copyfile(Path('sdpose') / name, root / name)
api = HfApi(token=os.environ['HF_TOKEN'])
destination = os.environ['HF_REPO_ID']
assert destination == 'RabbitDaisuke/SDPose-Wholebody-onnx'
api.upload_folder(repo_id=destination, folder_path=str(root),
                  delete_patterns=None, commit_message='Export SDPose WholeBody133 ONNX matrix ' + os.environ['GITHUB_SHA'])
print('Published SDPose ONNX with base_model metadata')
