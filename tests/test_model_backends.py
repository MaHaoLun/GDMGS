"""Model-family boundary tests; no CUDA or fVDB execution is simulated."""
import ast
from pathlib import Path
import subprocess
import sys
import types
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from batch import BundleMetadata, NeuralGaussianBatch
from model_bridge import from_cachegs_batch, source_state, DecoderInterface, family


def original_cachegs_batch_class():
    # Execute the unchanged tensor container classes; their jagged/fVDB methods
    # are not exercised. This isolates the real storage policy from GPU setup.
    path = ROOT/'upstream/CacheGS/gaussian_renderer/neural_gaussians.py'
    tree = ast.parse(path.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    prefix = ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass, field\nfrom typing import *\nimport torch\n').body
    module = types.ModuleType('cachegs_original_container_test')
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix+classes, type_ignores=[])), str(path), 'exec'), module.__dict__)
    return module


def test_preserves_original_cachegs_storage_and_ownership():
    original = original_cachegs_batch_class()
    ids = torch.tensor([7, 13]); levels = torch.tensor([1, 2])
    mask = torch.tensor([True, False, False, True])
    meta = original.BundleMetadata.from_selection(ids, levels, mask, 2)
    scaling = torch.tensor([[.123456, .234567, .345678], [.456789, .567891, .678912]])
    rotation = torch.tensor([[.123456, .234567, .345678, .456789]]).repeat(2, 1)
    source = original.NeuralGaussianBatch(None, ids, torch.ones(2, 3), torch.ones(2, 3),
        torch.ones(2, 1), scaling, rotation, mask, None, {}, meta)
    assert source.scaling.dtype == torch.float16
    result = from_cachegs_batch(source)
    result.validate_contract()
    assert torch.equal(result.scaling, source.materialize()[3])
    assert torch.equal(result.rotation, source.materialize()[4])
    assert not torch.equal(result.scaling, scaling)  # Rounding must not be erased.
    assert result.bundle_metadata.row_owner_ids.tolist() == [7, 13]
    assert result.bundle_metadata.row_offset_slots.tolist() == [0, 1]
    assert result.xyz is source.xyz


def test_cachegs_source_does_not_mutate_shared_fvdb_state():
    class Model:
        _gdmgs_model_backend = 'cachegs'
        def set_anchor_mask(self, *args):
            raise AssertionError('shared fVDB state must not be mutated')
    source_state(object(), Model())
    interface = DecoderInterface(Model())
    assert not interface.add_level and not interface.add_color_dist
    with pytest.raises(ValueError):
        family(object())


def test_backend_selection_is_explicit_and_process_local(tmp_path):
    code = '''
import os, importlib.util
os.environ['GDMGS_BUILD_DIR'] = {cache!r}
from system.bootstrap import configure, BACKENDS
configure()
assert 'CacheGS' in importlib.util.find_spec('scene').origin
try:
    configure('proxygs')
except RuntimeError:
    pass
else:
    raise AssertionError('namespace mixing accepted')
'''.format(cache=str(tmp_path/'build'))
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
    code = code[:code.index('configure()')] + "configure('proxygs')\nassert 'ProxyGS' in importlib.util.find_spec('scene').origin\n"
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_training_default_is_cachegs():
    result = subprocess.run([sys.executable, 'train.py', '--help'], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    assert 'Default: CacheGS' in result.stdout
    assert '--backend proxygs' in result.stdout
