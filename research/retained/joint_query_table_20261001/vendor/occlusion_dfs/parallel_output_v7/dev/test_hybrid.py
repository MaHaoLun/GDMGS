import numpy as np
import pytest
import torch
import test_joint as original
from hybrid import Hybrid,pyramid,native,partial_upper
from serial_reference import SerialHybrid
from gdmgs.mesh_index.gpu_index import camera_planes


def verify(ab,v,far=100.,shift=0.):
    j=original.verify(ab,v,far,shift);w=np.eye(4);w[0,3]=shift
    cp=torch.tensor(list(w.flatten())+[100.,100.,100.,100.,200.,200.,1.0001],device='cuda',dtype=torch.float64)
    planes=torch.tensor(camera_planes(dict(w2c=w,angular_domain=(-1.,1.,-1.,1.),near=.01,far=far,camera_id='test')),device='cuda')
    eye=torch.tensor([-shift,0.,0.],device='cuda',dtype=torch.float64);ids=torch.arange(j.na,device='cuda')[::2].contiguous()
    ref=j.query(cp,planes,eye,ids)
    for cut in (0,1,6,8,10,12,16,20):
        h=Hybrid(j,cut);r=h.query(cp,planes,eye,ids)
        assert torch.equal(r[0],ref[0]) and torch.equal(r[1],ref[1]),cut
        assert int(r[3]['counts'][:,5].sum())==0
        sr=SerialHybrid(j,cut).query(cp,planes,eye,ids)
        assert torch.equal(r[0],sr[0]) and torch.equal(r[1],sr[1])
        assert torch.equal(r[3]['counts'][:,:6],sr[3]['counts'])
    return j


@pytest.mark.parametrize('na,nm',[(0,1),(1,0),(1,1),(31,1),(32,1),(33,33),(65,97)])
def test_types_and_leaf_sizes(na,nm):
    verify(np.tile([-.1,-.1,2.,.1,.1,2.,.001],(na,1)),np.tile([[-.1,-.1,2.],[.1,-.1,2.],[0.,.1,2.]],(nm,1)))


@pytest.mark.parametrize('shift',[-10.,0.,8.])
def test_random(shift):
    rng=np.random.default_rng(24);lo=rng.uniform(-20,20,(301,3));hi=lo+rng.uniform(0,4,(301,3))
    verify(np.c_[lo,hi,rng.uniform(.001,.1,301)],rng.uniform(-20,20,(903,3)),shift=shift)


def test_cross_split_and_near():
    p=np.array([[0.,0.,.01],[0.,0.,100.],[2.,0.,2.],[-2.,0.,2.],[0.,2.,2.],[0.,-2.,2.],[0.,0.,-1.],[0.,0.,1e6]])
    verify(np.c_[p,p,np.zeros(len(p))],np.repeat(p,3,axis=0))
    ab=np.tile([100.,100.,2.,101.,101.,2.,.001],(256,1));ab[0]=[-100.,-100.,2.,100.,100.,2.,10.]
    v=np.tile([[100.,0.,2.],[101.,0.,2.],[100.,1.,2.]],(257,1));v[:3]=[[-100.,-1.,2.],[100.,-1.,2.],[100.,100.,2.]]
    verify(ab,v)


def test_depth_proof_holes_and_uncertainty():
    ext=native();cp=torch.tensor(list(np.eye(4).flatten())+[100.,100.,100.,100.,200.,200.,1.0001],device='cuda',dtype=torch.float64)
    ab=torch.tensor([[0,0,5,0,0,5,.001],[0,0,1,0,0,1,.001],[0,0,-1,0,0,1,.001],[0,0,2,0,0,2,.001]],device='cuda',dtype=torch.float64)
    ids=torch.arange(4,device='cuda');d=torch.full((200,200),2.,device='cuda');hz,m=pyramid(d)
    assert ext.filter(ab,cp,hz,m,ids).tolist()==[False,True,True,True]
    d[100,100]=float('inf');hz,m=pyramid(d)
    assert ext.filter(ab,cp,hz,m,ids).all()
    d.fill_(float('nan'));hz,m=pyramid(d)
    assert ext.filter(ab,cp,hz,m,ids).all()


def test_stream():
    with torch.cuda.stream(torch.cuda.Stream()):
        verify(np.tile([0.,0.,2.,0.,0.,2.,.01],(65,1)),np.tile([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]],(65,1)))


def test_progressive_matches_complete_snapshot():
    from types import SimpleNamespace
    rng=np.random.default_rng(42)
    centers=rng.uniform(-.3,.3,(513,3));centers[:,2]+=5
    ab=np.c_[centers,centers,np.full(513,.001)]
    verts=rng.uniform(-.3,.3,(1539,3));verts[:,2]+=2
    j=original.verify(ab,verts)
    cp=torch.tensor(list(np.eye(4).flatten())+[100.,100.,100.,100.,200.,200.,1.0001],device='cuda',dtype=torch.float64)
    planes=torch.tensor(camera_planes(dict(w2c=np.eye(4),angular_domain=(-1.,1.,-1.,1.),near=.01,far=100.,camera_id='test')),device='cuda')
    eye=torch.zeros(3,device='cuda',dtype=torch.float64);ids=torch.arange(j.na,device='cuda')
    # Synthetic consistent snapshots isolate traversal from rasterization.
    class Raster:
        def render(self,ids,*args,**kwargs):
            return SimpleNamespace(depth_gpu=torch.full((200,200),2. if len(ids) else float('inf'),device='cuda'))
    for cut in (0,1,6,8,12,16,20):
        h=Hybrid(j,cut);raster=Raster()
        ref=h.query(cp,planes,eye,ids,raster=raster,epochs=1,occlusion=True)
        for epochs in (2,4):
            actual=h.query(cp,planes,eye,ids,raster=raster,epochs=epochs,occlusion=True)
            assert torch.equal(actual[0],ref[0]) and torch.equal(actual[1],ref[1])
            assert int(actual[3]['counts'][:,5].sum())==0



def test_registered_depth_tolerance_and_unknown():
    d=torch.tensor([[1.72841989994,1.93998241425,10.,100.,float('inf')]],device='cuda')
    u=partial_upper(d,.01,100.)
    assert float(u[0,0])>=1.72853922844 and float(u[0,1])>=1.94009125233
    assert torch.all(u>=d) and torch.isinf(u[0,-1])
    assert torch.isinf(partial_upper(torch.tensor([[1e8]],device='cuda'),.01,100.)).all()


@pytest.mark.parametrize('na,nm',[(65539,196613),(257,258),(1,0),(0,1)])
def test_single_keep_subtree_expands_to_many_blocks(na,nm):
    from joint import Joint
    ab=torch.tensor([[-.1,-.1,2.,.1,.1,2.,.001]],device='cuda',dtype=torch.float64).repeat(na,1)
    v=torch.tensor([[-.1,-.1,2.],[.1,-.1,2.],[0.,.1,2.]],device='cuda',dtype=torch.float64).repeat(nm,1)
    faces=torch.arange(nm*3,device='cuda',dtype=torch.int64).reshape(-1,3)
    mb=torch.tensor([[-.1,-.1,2.,.1,.1,2.,0.]],device='cuda',dtype=torch.float64).repeat(nm,1)
    j=Joint(ab,v,faces,mb);cp=torch.tensor(list(np.eye(4).flatten())+[100.,100.,100.,100.,200.,200.,1.0001],device='cuda',dtype=torch.float64)
    planes=torch.tensor(camera_planes(dict(w2c=np.eye(4),angular_domain=(-1.,1.,-1.,1.),near=.01,far=100.,camera_id='large')),device='cuda')
    eye=torch.zeros(3,device='cuda',dtype=torch.float64);ids=torch.arange(na,device='cuda')
    h=Hybrid(j,0);r=h.query(cp,planes,eye,ids)
    sr=SerialHybrid(j,0).query(cp,planes,eye,ids)
    assert torch.equal(r[0],ids) and torch.equal(r[1],torch.arange(nm,device='cuda'))
    assert torch.equal(r[0],sr[0]) and torch.equal(r[1],sr[1])
    assert int(r[3]['counts'][0,0])==1
    assert int(r[3]['counts'][0,6])==((na+nm+255)//256 if na+nm>256 else 0)
    assert torch.equal(r[3]['counts'][:,:6],sr[3]['counts'])
    if na+nm>262144:assert int(r[3]['counts'][0,7])>100
    # Entirely rejected tree emits zero tasks and does not read uninitialized range states.
    moved=cp.clone();moved[3]=1e6
    w=np.eye(4);w[0,3]=1e6
    far_planes=torch.tensor(camera_planes(dict(w2c=w,angular_domain=(-1.,1.,-1.,1.),near=.01,far=100.,camera_id='reject')),device='cuda')
    rejected=Hybrid(j,12).query(moved,far_planes,eye,ids)
    assert rejected[0].numel()==0 and rejected[1].numel()==0
    assert int(rejected[3]['counts'][:,6].sum())==0


def test_mixed_inline_and_large_ranges():
    rng=np.random.default_rng(73);lo=rng.uniform(-5,5,(257,3));lo[:,2]=rng.uniform(1.,3.,257)
    ab=np.c_[lo,lo+.03,np.full(257,.001)]
    v=np.tile([[-.1,-.1,2.],[.1,-.1,2.],[0.,.1,2.]],(16385,1))
    verify(ab,v)
