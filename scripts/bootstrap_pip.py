"""Bootstrap pip using verified PyPI metadata when old pip TLS fails."""
import hashlib
from pathlib import Path
import subprocess
import sys
import requests


def main():
    response = requests.get('https://pypi.org/pypi/pip/json', timeout=30)
    response.raise_for_status()
    wheel = next(f for f in response.json()['urls'] if f['filename'].endswith('py3-none-any.whl'))
    response = requests.get(wheel['url'], timeout=60)
    response.raise_for_status()
    payload = response.content
    if hashlib.sha256(payload).hexdigest() != wheel['digests']['sha256']:
        raise RuntimeError('wheel_checksum_mismatch')
    root = Path(__file__).resolve().parents[1] / 'data' / 'wheels'
    root.mkdir(parents=True, exist_ok=True)
    path = root / wheel['filename']
    path.write_bytes(payload)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--upgrade', str(path)], check=True)


if __name__ == '__main__':
    main()
