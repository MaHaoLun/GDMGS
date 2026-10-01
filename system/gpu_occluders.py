"""Device-resident solid-cell retrieval and batched convex hole construction.

Works on tensor devices for predicate tests; production uses CUDA tensors.
No triangle IDs, occluder cells, hole planes, or selected IDs return to the CPU.
"""
import itertools
import torch


class GPUOccluders:
    def __init__(self, lo, hi, level, nodes, device='cuda'):
        self.device=torch.device(device);self.level=int(level)
        self.lo=torch.as_tensor(lo,dtype=torch.float64,device=self.device)
        self.hi=torch.as_tensor(hi,dtype=torch.float64,device=self.device)
        self.nodes=torch.as_tensor(nodes,dtype=torch.int64,device=self.device).reshape(-1,4)
        self.bits=torch.tensor(list(itertools.product((0,1),repeat=3)),dtype=torch.int64,device=self.device)
        self.pairs=torch.triu_indices(8,8,offset=1,device=self.device)
        from holed_index import OccluderIndex
        topology=OccluderIndex(lo,hi,self.level,nodes)
        records=[];children=[];full=[]
        def pack(node,depth,xyz):
            index=len(records);records.append([index,depth,*xyz]);children.append([-1]*8);full.append(node.full)
            for child,value in node.children.items():
                bits=((child>>2)&1,(child>>1)&1,child&1)
                children[index][child]=pack(value,depth+1,tuple(2*xyz[a]+bits[a] for a in range(3)))
            return index
        pack(topology.root,0,(0,0,0))
        self.records=torch.tensor(records,dtype=torch.int64,device=self.device)
        self.children=torch.tensor(children,dtype=torch.int64,device=self.device)
        self.full=torch.tensor(full,dtype=torch.bool,device=self.device)

    def query(self, planes, w2c, near):
        """Report visible full cells; split near-plane-crossing full cells."""
        planes=torch.as_tensor(planes,dtype=torch.float64,device=self.device)[:,:4]
        w2c=torch.as_tensor(w2c,dtype=torch.float64,device=self.device)
        near_plane=w2c[2].clone();near_plane[3]-=near
        current=self.records[:1];accepted=[]
        for _ in range(self.level+1):
            if not len(current):break
            full=(current[:,0]<0)|self.full[current[:,0].clamp(min=0)]
            scale=(self.hi-self.lo)/torch.pow(2.,current[:,1:2])
            lower=self.lo+current[:,2:]*scale;upper=lower+scale
            corners=torch.where(planes[None,:,:3]>=0,upper[:,None],lower[:,None])
            meet=((corners*planes[None,:,:3]).sum(-1)+planes[None,:,3]>=-1e-9).all(1)
            near_min=(torch.where(near_plane[:3]>=0,lower,upper)*near_plane[:3]).sum(1)+near_plane[3]
            keep=meet & full & (near_min>1e-9)
            accepted.append(torch.cat((lower[keep],upper[keep]),1))
            descend=meet & ~keep & (current[:,1]<self.level)
            partial=current[descend & ~full]
            child_ids=self.children[partial[:,0]].reshape(-1)
            real=self.records[child_ids[child_ids>=0]]
            parents=current[descend & full]
            depth=(parents[:,None,1:2]+1).expand(-1,8,-1)
            xyz=parents[:,None,2:]*2+self.bits[None]
            virtual_id=torch.full_like(depth,-1)
            virtual=torch.cat((virtual_id,depth,xyz),2).reshape(-1,5)
            current=torch.cat((real,virtual))
        return torch.cat(accepted) if accepted else self.lo.new_empty((0,6))

    def holes(self, boxes, eye):
        """At most 28 supporting cone planes plus three entry planes per cell.

        Redundant supporting planes are harmless. Inactive slots use the
        tautology 1>0, avoiding variable-length host packing and synchronization.
        """
        eye=torch.as_tensor(eye,dtype=torch.float64,device=self.device)
        if not len(boxes):
            return boxes.new_empty((0,31,4)),torch.empty(0,dtype=torch.int32,device=self.device)
        corners=torch.where(self.bits[None].bool(),boxes[:,None,3:],boxes[:,None,:3])
        rays=corners-eye
        normals=torch.linalg.cross(rays[:,self.pairs[0]],rays[:,self.pairs[1]],dim=-1)
        lengths=normals.norm(dim=-1)
        normals=normals/lengths.clamp(min=1e-12)[...,None]
        sides=(normals[:,:,None,:]*rays[:,None,:,:]).sum(-1)
        minimum,maximum=sides.amin(-1),sides.amax(-1)
        valid=(lengths>=1e-12)&((minimum>=-1e-12)|(maximum<=1e-12))
        normals=torch.where((maximum<=1e-12)[...,None],-normals,normals)
        planes=boxes.new_zeros((len(boxes),31,4));planes[:,:,3]=1
        candidate=torch.cat((normals,-(normals*eye).sum(-1,keepdim=True)),2)
        planes[:,:28]=torch.where(valid[...,None],candidate,planes[:,:28])
        for axis in range(3):
            low=eye[axis]<boxes[:,axis];high=eye[axis]>boxes[:,axis+3]
            planes[:,28+axis,axis]=low.to(boxes.dtype)-high.to(boxes.dtype)
            planes[:,28+axis,3]=torch.where(low,-boxes[:,axis],torch.where(high,boxes[:,axis+3],torch.ones_like(boxes[:,axis])))
        counts=torch.full((len(boxes),),31,dtype=torch.int32,device=self.device)
        return planes.contiguous(),counts
