"""Download official Chronos-2 weights with a pinned revision and checksums."""
import argparse
import hashlib
import json
from pathlib import Path


def download(destination, revision='main'):
    from huggingface_hub import HfApi, snapshot_download
    repo = 'amazon/chronos-2'
    resolved = HfApi().model_info(repo, revision=revision).sha
    root = Path(destination).resolve()
    snapshot_download(repo_id=repo, revision=resolved, local_dir=str(root),
                      allow_patterns=['*.json', '*.safetensors', 'README.md', 'LICENSE*'])
    if not list(root.glob('*.safetensors')) or not (root / 'config.json').is_file():
        raise RuntimeError('incomplete_model_download')
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '.cache' in path.parts or path.name == 'download_manifest.json':
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        files[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    manifest = {'repo_id': repo, 'revision': resolved, 'status': 'downloaded', 'files': files}
    temporary = root / 'download_manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.replace(root / 'download_manifest.json')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='data/pretrained/chronos-2')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    print(json.dumps(download(args.destination, args.revision), indent=2))
