"""Run the complete bundled ProxyGS proxy-aware training entrypoint."""
import os
import runpy
from system.bootstrap import configure, UPSTREAM

if __name__ == '__main__':
    configure()
    os.chdir(UPSTREAM)
    runpy.run_path(str(UPSTREAM / 'train.py'), run_name='__main__')
