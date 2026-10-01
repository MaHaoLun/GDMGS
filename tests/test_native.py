"""Exercise the retained C++ kernels, not a Python stand-in."""
import ctypes
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def native(tmp_path_factory):
    spec = importlib.util.spec_from_file_location('build_native', ROOT/'scripts/build_native.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    lib = ctypes.CDLL(str(module.build_cpu(tmp_path_factory.mktemp('native')/'cpu_select.so')))
    lib.cpu_select.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int64]+[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.cpu_select.restype = ctypes.c_int
    lib.cpu_build_holes.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]*2
    lib.cpu_build_holes.restype = ctypes.c_int
    return lib

def camera():
    return np.r_[np.eye(4).ravel(), 100., 100., 50., 50., 100., 100., 1.0001].astype(np.float64)

def query(native, bounds, planes=None):
    bounds = np.ascontiguousarray(bounds, np.float64)
    cp = camera(); candidates = np.ones(len(bounds), np.uint8)
    holes = 0 if planes is None else 1
    planes = np.zeros((1,4), np.float64) if planes is None else np.ascontiguousarray(planes, np.float64)
    starts = np.array([0, len(planes)], np.int32)
    output = np.empty(len(bounds), np.uint8)
    rc = native.cpu_select(bounds.ctypes.data, cp.ctypes.data, candidates.ctypes.data,
                            len(bounds), planes.ctypes.data, starts.ctypes.data, holes, 2, output.ctypes.data)
    assert rc == 0
    return output

def test_dense_real_kernel(native):
    out = query(native, [[0,0,2,.1,.1,3,.01], [100,100,2,101,101,3,.01], [0,0,-3,.1,.1,-2,.01]])
    assert out.tolist() == [1,0,0]

def test_hole_uses_gaussian_support_not_only_centers(native):
    # Hole x>0: centers are inside, but the first anchor's support reaches x<0.
    bounds = [[.1,0,2,.2,.1,3,.1], [.5,0,2,.6,.1,3,.1]]
    assert query(native, bounds, [[1,0,0,0]]).tolist() == [1,0]

def test_native_hole_planes_match_python(native):
    spec = importlib.util.spec_from_file_location('geometry', ROOT/'system/fast_geometry.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    boxes = np.array([[0,0,1,1,1,2], [2,1,3,3,2,4]], np.float64)
    eye = np.array([-2,.5,0], np.float64)
    planes = np.empty((64,4), np.float64); starts=np.zeros(3,np.int32)
    assert native.cpu_build_holes(boxes.ctypes.data, eye.ctypes.data, 2, 2, planes.ctypes.data, starts.ctypes.data)==0
    for i, box in enumerate(boxes):
        np.testing.assert_allclose(planes[starts[i]:starts[i+1]], module.fast_hole_planes(box, eye), atol=1e-12)

@pytest.mark.parametrize('threads', [1, 2, 4])
def test_native_index_matches_dense_on_all_rows(native, threads):
    rng = np.random.default_rng(20261001)
    n = 513
    lower = rng.uniform([-12,-12,.05], [12,12,16], (n,3))
    bounds = np.ascontiguousarray(np.c_[lower, lower+rng.uniform(.01,1,(n,3)), rng.uniform(.001,.3,n)])
    order = np.argsort(lower[:,0], kind='stable').astype(np.int64)
    leaves = (n+31)//32; internal=leaves-1
    nodes=np.zeros((2*leaves-1,7),np.float64)
    left=np.full(len(nodes),-1,np.int32); right=left.copy()
    begin=np.zeros(len(nodes),np.int64); end=begin.copy()
    next_internal=[0]
    def build(a,b):
        if b-a==1:
            slot=internal+a; rows=bounds[order[a*32:min(b*32,n)]]
            nodes[slot]=np.r_[rows[:,:3].min(0),rows[:,3:6].max(0),rows[:,6].max()]
        else:
            slot=next_internal[0];next_internal[0]+=1
            mid=(a+b)//2; l,r=build(a,mid),build(mid,b)
            left[slot],right[slot]=l,r
            nodes[slot]=np.r_[np.minimum(nodes[l,:3],nodes[r,:3]),np.maximum(nodes[l,3:6],nodes[r,3:6]),max(nodes[l,6],nodes[r,6])]
        begin[slot],end[slot]=a*32,min(b*32,n)
        return slot
    assert build(0,leaves)==0
    planes=np.zeros((6,8),np.float64)
    planes[:,:4]=[[1,0,.5,0],[-1,0,.5,0],[0,1,.5,0],[0,-1,.5,0],[0,0,1,-.01],[0,0,-1,1e10]]
    camera_input=camera(); candidates=rng.integers(0,2,n,dtype=np.uint8)
    holes=np.array([[1,0,0,0],[0,0,1,-3]],np.float64);starts=np.array([0,2],np.int32)
    dense=np.zeros(n,np.uint8); indexed=dense.copy();counters=np.zeros(3,np.int64)
    assert native.cpu_select(bounds.ctypes.data,camera_input.ctypes.data,candidates.ctypes.data,n,
        holes.ctypes.data,starts.ctypes.data,1,threads,dense.ctypes.data)==0
    native.cpu_select_tree_keep.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int64]*2+[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]*2
    assert native.cpu_select_tree_keep(*[a.ctypes.data for a in (bounds,nodes,left,right,order,begin,end,camera_input,planes,candidates)],
        n,leaves,holes.ctypes.data,starts.ctypes.data,1,threads,indexed.ctypes.data,counters.ctypes.data)==0
    np.testing.assert_array_equal(indexed,dense)
    assert indexed.any() and not indexed.all()


def test_allocation_uses_floor_ceil_optimum():
    spec=importlib.util.spec_from_file_location('planning',ROOT/'system/planning.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert sum(mod.allocate(120,1/60,1/40))==72
    for n in range(1,70):
        for cc,cg in [(1.,1.),(.1,7.),(3.2,.9)]:
            q=sum(mod.allocate(n,cc,cg))
            assert max(q*cc,(n-q)*cg)==min(max(j*cc,(n-j)*cg) for j in range(n+1))

def test_canonical_lod_half_integer_ties(native):
    # Threshold equality follows round-to-even: odd target level excludes, even includes.
    position=np.array([[1.,0,0],[1.,0,0],[100.,0,0],[.5,0,0]],np.float64)
    radius2=np.ones(4,np.float64);levels=np.array([1,2,0,3],np.int32)
    eye=np.zeros(3,np.float64);output=np.zeros(4,np.uint8)
    native.cpu_lod_threshold.argtypes=[ctypes.c_void_p]*4+[ctypes.c_double,ctypes.c_int64,ctypes.c_int,ctypes.c_void_p]
    assert native.cpu_lod_threshold(position.ctypes.data,radius2.ctypes.data,levels.ctypes.data,
        eye.ctypes.data,1.,4,2,output.ctypes.data)==0
    assert output.tolist()==[0,1,1,1]


def test_device_geometry_matches_scalar_predicates():
    import sys
    import torch
    sys.path.insert(0,str(ROOT/'system'))
    from gpu_occluders import GPUOccluders
    from holed_index import OccluderIndex
    from hole_geometry import inside_hole
    from fast_geometry import fast_hole_planes
    lo,hi=[0,0,0],[2,2,2];nodes=[(1,0,0,0)]
    planes=np.array([[1,0,0,1],[-1,0,0,5]],np.float64)
    w2c=np.eye(4);w2c[2,:3]=[1,0,0]
    cpu=OccluderIndex(lo,hi,2,nodes)
    tensor=GPUOccluders(lo,hi,2,nodes,device='cpu')
    a=cpu.query(planes,w2c,.4);b=tensor.query(planes,w2c,.4).numpy()
    assert sorted(map(tuple,a))==sorted(map(tuple,b))
    assert len(a)==4
    boxes=torch.tensor([[0.,0,1,1,1,2],[2,1,3,3,2,4]],dtype=torch.float64)
    eye=np.array([-2,.5,0.])
    hp,counts=tensor.holes(boxes,eye)
    rng=np.random.default_rng(2)
    points=rng.uniform(-5,5,(100,3));bounds=np.c_[points,points+.03]
    for i,box in enumerate(boxes.numpy()):
        expected=inside_hole(bounds,fast_hole_planes(box,eye))
        actual=inside_hole(bounds,hp[i,:counts[i]].numpy())
        np.testing.assert_array_equal(actual,expected)
