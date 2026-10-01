"""Select one source family per process; never mix the two loader namespaces."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {'cachegs': ROOT/'upstream/CacheGS', 'proxygs': ROOT/'upstream/ProxyGS'}
UPSTREAM = BACKENDS['cachegs']
_ACTIVE = None


def configure(backend=None):
    global _ACTIVE
    backend = backend or _ACTIVE or 'cachegs'
    if backend not in BACKENDS:
        raise ValueError(f'unknown model backend: {backend}')
    if _ACTIVE is not None and _ACTIVE != backend:
        raise RuntimeError('use a separate process for each model backend')
    upstream = BACKENDS[backend]
    for name in ('scene', 'gaussian_renderer', 'utils', 'gdmgs'):
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        if path and not Path(path).resolve().is_relative_to(upstream.resolve()):
            raise RuntimeError(f'{name} was already imported from another source tree')
    _ACTIVE = backend
    excluded = {str(p) for root in BACKENDS.values() for p in (root, root/'tools')}
    sys.path[:] = [value for value in sys.path if value not in excluded]
    sys.path[:0] = [str(ROOT/'system'), str(upstream), str(upstream/'tools')]
    cache = Path(os.environ.get('GDMGS_BUILD_DIR', Path.home()/'.cache/gdmgs'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(cache/'torch'))
    os.environ.setdefault('ANCHOR_GPU_BUILD', str(cache/'anchor'))
    os.environ.setdefault('MAX_JOBS', '2')
    return cache
