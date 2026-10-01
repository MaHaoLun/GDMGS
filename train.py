"""Launch original CacheGS training; ProxyGS is an explicit compatibility option."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from system.bootstrap import BACKENDS, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--backend', choices=tuple(BACKENDS), default='cachegs')
    selected, rest = parser.parse_known_args()
    if rest == ['--help'] or rest == ['-h']:
        print('Default: CacheGS/GDMGS_Codebase training with --config CONFIG.yaml.\n'
              'Example: python train.py --config /absolute/path/to/training.yaml\n'
              'Historical ProxyGS: python train.py --backend proxygs [original arguments]\n'
              'Full upstream options are documented in upstream/<backend>/train.py.')
        return
    # The unchanged training script reads .gitignore from its working directory.
    # Resolve caller-provided paths before entering that directory.
    path_options = {'--config', '--source_path', '-s', '--model_path', '-m',
                    '--ply_path', '--ply_mesh', '--depth_npy_dir', '--start_checkpoint'}
    for i, value in enumerate(rest):
        if i and rest[i-1] in path_options:
            rest[i] = str(Path(value).expanduser().resolve())
        elif '=' in value and value.split('=',1)[0] in path_options:
            key, path = value.split('=',1)
            rest[i] = key+'='+str(Path(path).expanduser().resolve())
    configure(selected.backend)
    upstream = BACKENDS[selected.backend]
    sys.argv = [str(upstream/'train.py'), *rest]
    os.chdir(upstream)
    runpy.run_path(str(upstream/'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
