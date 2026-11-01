import hashlib
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

raise RuntimeError('RTMPose publication disabled: checkpoint redistribution terms need review; do not mix into existing RTMW distribution')

folder = Path('release')
variants = []
for path in sorted(folder.glob('rtmpose-*.json')):
    report = json.loads(path.read_text(encoding='utf-8'))
    model = folder / report['file']
    assert model.stat().st_size == report['bytes']
    assert hashlib.sha256(model.read_bytes()).hexdigest() == report['sha256']
    variants.append(report)
assert len(variants) == 14, len(variants)
catalog = {'schema': 'rtmpose-wholebody133/v1', 'keypointCount': 133, 'dimensions': 2,
           'sourceCommit': os.environ['GITHUB_SHA'], 'upstream': 'https://github.com/open-mmlab/mmpose',
           'upstreamRevision': '759b39c13fea6ba094afc1fa932f51dc1b11cbf9', 'variants': variants,
           'browserGPUValidated': False, 'speedRanking': None}
(folder / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
shutil.copyfile('lightpose/README.md', folder / 'README.md')
shutil.copyfile('UPSTREAM_LICENSE', folder / 'LICENSE-MMPose')
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(repo_id='RabbitDaisuke/rtmw3d-webgpu-onnx', folder_path=str(folder),
                  path_in_repo='rtmpose2d', delete_patterns=None,
                  commit_message='Append RTMPose-t/s WholeBody133 precision matrix ' + os.environ['GITHUB_SHA'])
print('Published 14 wholebody variants; existing RTMW/RTMW3D files preserved')
