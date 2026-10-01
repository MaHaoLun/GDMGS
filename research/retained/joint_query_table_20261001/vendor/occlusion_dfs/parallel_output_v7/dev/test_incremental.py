import numpy as np
import pytest
import torch
from gdmgs.mesh_index import MeshIndex,CameraDomain
from online_proxy_depth import OnlineProxyDepthRasterizer
from incremental_depth import IncrementalDepth
from hybrid import partial_upper


def fixture():
    triangles=np.array([
        [[-2,-2,2],[2,-2,2],[0,2,2]],
        [[-5,-5,5],[5,-5,5],[0,5,5]],
        [[-.3,-.3,1],[.3,-.3,1],[0,.3,1]],
        [[1.1,-.4,2],[1.9,-.4,2],[1.5,.4,2]],
    ],dtype=np.float64)
    cpu=MeshIndex(triangles.reshape(-1,3),np.arange(12,dtype=np.int64).reshape(-1,3),leaf_size=8)
    raster=OnlineProxyDepthRasterizer(cpu);inc=IncrementalDepth(raster)
    camera=CameraDomain.parse(dict(w2c=np.eye(4),angular_domain=(-1.,1.,-1.,1.),near=.01,far=100.,camera_id='incremental'))
    return raster,inc,camera


def equal_with_tolerance(actual,expected,camera):
    finite=torch.isfinite(expected);assert torch.equal(torch.isfinite(actual),finite)
    budget=torch.maximum(partial_upper(actual,camera.near,camera.far)-actual,partial_upper(expected,camera.near,camera.far)-expected)
    assert ((actual[finite]-expected[finite]).abs()<=budget[finite]).all()


@pytest.mark.parametrize('order',[(0,1,2,3),(3,2,1,0),(1,3,0,2)])
def test_append_preserves_native_buffers_and_depth(order):
    raster,inc,camera=fixture();inc.begin(camera);seen=[]
    for face in order:
        seen.append(face);actual=inc.append(torch.tensor([face],device='cuda',dtype=torch.int64))
        reference=raster.render(torch.tensor(sorted(seen),device='cuda',dtype=torch.int64),camera,(1600,900),copy_to_cpu=False).depth_gpu
        equal_with_tolerance(actual,reference,camera)
    d=inc.diagnostics();assert d['native_depth_buffer_reused'] and d['clears']==1 and d['triangle_submissions']==4
    assert d['nonempty_draws']==4 and d['final_full_redraws']==0
    assert all(x>0 for row in d['buffer_pointers'] for x in row)


def test_empty_batches_and_frame_reset():
    raster,inc,camera=fixture();all_ids=torch.arange(4,device='cuda')
    inc.begin(camera);inc.append(all_ids)
    inc.begin(camera);empty=torch.empty(0,device='cuda',dtype=torch.int64)
    assert torch.isinf(inc.append(empty)).all()
    actual=inc.append(torch.tensor([1],device='cuda',dtype=torch.int64))
    reference=raster.render(torch.tensor([1],device='cuda',dtype=torch.int64),camera,(1600,900),copy_to_cpu=False).depth_gpu
    equal_with_tolerance(actual,reference,camera)
    assert inc.diagnostics()['clears']==1
    moved=np.eye(4);moved[0,3]=100
    camera2=CameraDomain.parse(dict(w2c=moved,angular_domain=(-1.,1.,-1.,1.),near=.01,far=100.,camera_id='moved'))
    inc.begin(camera2);assert torch.isinf(inc.append(all_ids)).all()


def test_incremental_nondefault_stream():
    with torch.cuda.stream(torch.cuda.Stream()):
        raster,inc,camera=fixture();inc.begin(camera)
        inc.append(torch.tensor([1],device='cuda',dtype=torch.int64))
        actual=inc.append(torch.tensor([0,2,3],device='cuda',dtype=torch.int64))
        reference=raster.render(torch.arange(4,device='cuda'),camera,(1600,900),copy_to_cpu=False).depth_gpu
        equal_with_tolerance(actual,reference,camera)
