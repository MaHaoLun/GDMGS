"""Vectorized view query for the 31 leaf-only experimental occluder cells."""
import numpy as np


_PAIR_I,_PAIR_J=np.triu_indices(8,k=1)
_CORNER_BITS=np.array(list(np.ndindex(2,2,2)),dtype=bool)


def visible_leaf_boxes(boxes,planes,w2c,near,tolerance=1e-9):
    if not len(boxes):
        return boxes
    normals=np.asarray(planes)[:,:3]
    constants=np.asarray(planes)[:,3]
    farthest=np.where(normals[None]>=0,boxes[:,None,3:],boxes[:,None,:3])
    maximum=np.sum(farthest*normals[None],axis=2)+constants[None]
    selected=np.all(maximum>=-tolerance,axis=1)
    near_plane=np.asarray(w2c)[2].copy()
    near_plane[3]-=near
    closest=np.where(near_plane[None,:3]>=0,boxes[:,:3],boxes[:,3:])
    near_minimum=np.sum(closest*near_plane[:3],axis=1)+near_plane[3]
    selected &= near_minimum>tolerance
    return boxes[selected]


def fast_hole_planes(box,eye,tolerance=1e-12):
    box=np.asarray(box,np.float64)
    eye=np.asarray(eye,np.float64)
    corners=np.where(_CORNER_BITS,box[3:],box[:3])
    rays=corners-eye
    normals=np.cross(rays[_PAIR_I],rays[_PAIR_J])
    lengths=np.linalg.norm(normals,axis=1)
    valid=lengths>tolerance
    normals=normals[valid]/lengths[valid,None]
    side=normals@rays.T
    minima=side.min(1)
    maxima=side.max(1)
    support=(minima>=-tolerance)|(maxima<=tolerance)
    normals=normals[support]
    maxima=maxima[support]
    normals[maxima<=tolerance]*=-1
    unique=[]
    for n in normals:
        if not any(np.allclose(n,p,atol=1e-10,rtol=0) for p in unique):
            unique.append(n)
    planes=[np.r_[n,-np.dot(n,eye)] for n in unique]
    for axis in range(3):
        direction=np.zeros(3)
        if eye[axis]<box[axis]:
            direction[axis]=1
            planes.append(np.r_[direction,-box[axis]])
        elif eye[axis]>box[axis+3]:
            direction[axis]=-1
            planes.append(np.r_[direction,box[axis+3]])
    return np.asarray(planes,np.float64).reshape(-1,4)
