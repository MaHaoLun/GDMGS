"""Split ProxyGS decoder extracted from the retained full-block implementation."""
import torch
from .raster_batch import NeuralGaussianBatch, BundleMetadata

def prepare_decode(view,model,ids):
    if any((model.use_feat_bank,model.add_level,model.appearance_dim,model.add_opacity_dist,model.add_color_dist,model.add_cov_dist)) or model.dist2level!='round':
        raise ValueError('only the frozen model configuration is qualified')
    anchor=model.get_anchor[ids];feat=model.get_anchor_feat[ids];level=model.get_level[ids]
    grid_offsets=model._offset[ids];grid_scaling=model.get_scaling[ids]
    ob_view=anchor-view.camera_center;ob_dist=ob_view.norm(dim=1,keepdim=True);ob_view=ob_view/ob_dist
    cat_local_view=torch.cat([feat,ob_view,ob_dist],dim=1)
    cat_local_view_wodist=torch.cat([feat,ob_view],dim=1)
    neural_opacity=model.get_opacity_mlp(cat_local_view_wodist).reshape(-1,1)
    mask=(neural_opacity>0.).view(-1)
    rows=int(mask.sum(dtype=torch.int64).item())
    return dict(anchor=anchor,feat=feat,level=level,grid_offsets=grid_offsets,grid_scaling=grid_scaling,
                cat_local_view=cat_local_view,cat_local_view_wodist=cat_local_view_wodist,
                neural_opacity=neural_opacity,mask=mask,rows=rows)


def finish_decode(st,model,ids,levels):
    anchor=st['anchor'];mask=st['mask'];n_offsets=model.n_offsets
    opacity=st['neural_opacity'][mask]
    color=model.get_color_mlp(st['cat_local_view_wodist']).reshape(len(anchor)*n_offsets,3)
    scale_rot=model.get_cov_mlp(st['cat_local_view_wodist']).reshape(len(anchor)*n_offsets,7)
    offsets=st['grid_offsets'].view(-1,3)
    concatenated=torch.cat([st['grid_scaling'],anchor],dim=-1)
    repeated=concatenated.repeat_interleave(n_offsets,dim=0)
    masked=torch.cat([repeated,color,scale_rot,offsets],dim=-1)[mask]
    scaling_repeat,repeat_anchor,color,scale_rot,offsets=masked.split([6,3,3,7,3],dim=-1)
    scaling=scaling_repeat[:,3:]*torch.sigmoid(scale_rot[:,:3]);rotation=model.rotation_activation(scale_rot[:,3:7])
    xyz=repeat_anchor+offsets*scaling_repeat[:,:3]
    batch=fused_batch(ids,(xyz,color,opacity,scaling,rotation,mask),n_offsets,levels)
    assert len(batch.xyz)==st['rows']
    return batch


def fused_batch(ids, tensors, n_offsets, levels):
    """Preserve full row ownership, including anchors with no retained offsets."""
    xyz, color, opacity, scaling, rotation, mask = tensors
    counts = mask.reshape(len(ids), n_offsets).sum(1, dtype=torch.long)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    owners = ids.repeat_interleave(n_offsets)[mask]
    slots = torch.arange(n_offsets, device=ids.device).repeat(len(ids))[mask]
    metadata = BundleMetadata(ids, owners, slots, counts, offsets, levels,
                              levels.repeat_interleave(n_offsets)[mask])
    return NeuralGaussianBatch(ids, xyz, color, opacity, scaling, rotation, mask, None, metadata)


class ProxyGSDecoder:
    """Read-only, inference-only adapter for the historically qualified model.

    The caller loads the original ProxyGS model/checkpoint. No source-camera
    LoD mask is applied: target selections already define the explicit union.
    """
    def __init__(self, model):
        self.model = model

    def prepare(self, source, ids):
        from types import SimpleNamespace
        view = SimpleNamespace(camera_center=torch.tensor(source.center, dtype=self.model.get_anchor.dtype,
                                                          device=ids.device))
        if len(ids) == 0:
            return dict(rows=0, empty=True)
        if bool((ids >= len(self.model.get_anchor)).any()):
            raise ValueError('anchor ID exceeds model row count')
        if bool(((self.model.get_anchor[ids] - view.camera_center).norm(dim=1) == 0).any()):
            raise ValueError('decoder view direction undefined at anchor center')
        return prepare_decode(view, self.model, ids)

    def row_count(self, state):
        return state['rows']

    def finish(self, state, ids):
        levels = self.model.get_level[ids].reshape(-1).long()
        if state.get('empty'):
            z = lambda width: torch.empty((0, width), device=ids.device, dtype=self.model.get_anchor.dtype)
            mask = torch.empty(0, dtype=torch.bool, device=ids.device)
            return fused_batch(ids, (z(3), z(3), z(1), z(3), z(4), mask), self.model.n_offsets, levels)
        return finish_decode(state, self.model, ids, levels)
