"""Run the preserved CacheGS fresh renderer independently of shared scheduling."""
import sys
from system.bootstrap import configure, BACKENDS
import runpy

if __name__ == '__main__':
    configure('cachegs')
    path = BACKENDS['cachegs']/'render.py'
    if '--pipeline' not in sys.argv:
        sys.argv.extend(['--pipeline', 'gdmgs'])
    runpy.run_path(str(path), run_name='__main__')
