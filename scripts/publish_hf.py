"""Publish to the exact user-designated repository, without deleting Hub files."""
import hashlib
import json
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
release = Path(sys.argv[1] if len(sys.argv) > 1 else 'release')
settings = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
repo = settings['huggingface_repo_id']
if repo != 'RabbitDaisuke/rtmw3d-webgpu-onnx':
    raise RuntimeError('Destination does not match the user-designated repository')
token = os.environ.get('HF_TOKEN')
if not token:
    raise RuntimeError('The repository Actions secret HF_TOKEN is required')
catalog = json.loads((release / 'models.json').read_text(encoding='utf-8'))
if catalog.get('producer') != 'DaisukeDaisuke/mmpose-actions' or len(catalog.get('variants', [])) != 16:
    raise RuntimeError('Incomplete or unrelated release artifact')
for variant in catalog['variants']:
    if hashlib.sha256((release / variant['file']).read_bytes()).hexdigest() != variant['sha256']:
        raise RuntimeError(f'Checksum mismatch: {variant["file"]}')
api = HfApi(token=token)
api.whoami()  # Validate configured credential; never log token or account data.
api.create_repo(repo_id=repo, repo_type='model', private=False, exist_ok=True)
result = api.upload_folder(repo_id=repo, repo_type='model', folder_path=release,
    allow_patterns=['onnx/*.onnx', 'reports/*.json', 'runtime/*', 'benchmark/*', 'models.json', 'README.md', 'COMPARISON.md', 'LICENSE'],
    commit_message=f'Build full-body RTMW3D experiment matrix from {os.environ.get("GITHUB_SHA", "local")}',
    delete_patterns=None)
print(f'Published: https://huggingface.co/{repo}')
print(f'Commit: {result.oid}')
if os.environ.get('GITHUB_STEP_SUMMARY'):
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as file:
        file.write(f'Published 16 whole-body candidates: https://huggingface.co/{repo}\nHF commit: `{result.oid}`\n')
