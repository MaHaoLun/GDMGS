"""Resolve bundled source and build caches without historical experiment paths."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream' / 'ProxyGS'

def configure():
    for path in (UPSTREAM / 'tools', UPSTREAM, ROOT / 'system'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    cache = Path(os.environ.get('GDMGS_BUILD_DIR', Path.home() / '.cache' / 'gdmgs'))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(cache / 'torch'))
    Path(os.environ['TORCH_EXTENSIONS_DIR']).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('ANCHOR_GPU_BUILD', str(cache / 'anchor'))
    os.environ.setdefault('MAX_JOBS', '2')
    return cache
