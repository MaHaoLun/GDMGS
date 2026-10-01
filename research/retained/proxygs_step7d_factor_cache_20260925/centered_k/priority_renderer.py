"""Demand-bit cache, shared-union streams and grouped asynchronous intersections."""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from parallel_renderer import ParallelRenderer
from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode,rasterize_to_pixels

@triton.jit
def _sanitize(RAD,MEANS,Z,CONICS,OP,FLAGS,OUT_OP,N,BIT:tl.constexpr,OPSTRIDE:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);inside=i<N
 r=tl.load(RAD+i,inside,0);f=tl.load(FLAGS+i,inside,0)
 valid=inside & (r>0) & ((f & BIT)!=0)
 tl.store(RAD+i,tl.where(valid,r,0),inside)
 for j in tl.static_range(2):
  x=tl.load(MEANS+2*i+j,valid,0.);tl.store(MEANS+2*i+j,x,inside)
 for j in tl.static_range(3):
  x=tl.load(CONICS+3*i+j,valid,0.);tl.store(CONICS+3*i+j,x,inside)
 z=tl.load(Z+i,valid,0.);tl.store(Z+i,z,inside)
 op=tl.load(OP+i*OPSTRIDE,valid,0.);tl.store(OUT_OP+i,op,inside)

@dataclass(frozen=True)
class EpochState:
 arena: object
 requests: tuple
 frames: tuple
 row_flags: object
 flag_bytes: int

class PriorityRenderer(ParallelRenderer):
 def prepare_epoch(self,arena,requests,frames,mode):
  if len(requests)!=len(frames) or len(frames)>4 or len(frames)==0:raise ValueError('invalid epoch')
  flags=None;size=0
  if mode.startswith('mask') or mode.startswith('staged'):
   anchor_flags=torch.zeros(len(arena.batch.anchor_indices),dtype=torch.uint8,device=self.device)
   for j,pos in enumerate(arena.maps):anchor_flags[pos]=anchor_flags[pos] | (1<<j)
   flags=torch.repeat_interleave(anchor_flags,arena.batch.bundle_metadata.counts,output_size=len(arena.batch.xyz))
   size=flags.numel()*flags.element_size()
  return EpochState(arena,tuple(requests),tuple(frames),flags,size)

 def _project_target(self,state,j,fused):
  batch=state.arena.batch;f=state.frames[j];n=len(batch.xyz)
  rad,means,z,conics,_=fully_fused_projection(batch.xyz,None,batch.rotation,batch.scaling,
   self.viewmats[f][None],self.intrinsics[f][None],self.width,self.height,packed=False)
  if fused:
   assert state.row_flags is not None and all(t.is_contiguous() for t in (rad,means,z,conics))
   opacity=torch.empty((1,n),dtype=torch.float32,device=self.device)
   op=batch.opacity.reshape(-1)
   if n:_sanitize[(triton.cdiv(n,256),)](rad,means,z,conics,op,state.row_flags,opacity,n,1<<j,op.stride(0),256)
  else:
   selected=((state.row_flags & (1<<j))!=0) if state.row_flags is not None else self._membership(batch.bundle_metadata.row_owner_ids,state.requests[j])
   valid=(rad>0)&selected[None]
   rad=torch.where(valid,rad,0).contiguous();means=torch.where(valid[...,None],means,0).contiguous()
   z=torch.where(valid,z,0).contiguous();conics=torch.where(valid[...,None],conics,0).contiguous()
   opacity=torch.where(valid,batch.opacity.reshape(1,-1),0).contiguous()
  return dict(rad=rad,means=means,z=z,conics=conics,opacity=opacity)

 def _raster_target(self,state,j,pr,ix,depth):
  _,keys,flat=ix
  off=isect_offset_encode(keys,1,self.tile_width,self.tile_height)
  colors=state.arena.batch.color[None];bg=self.background[None]
  if depth:
   colors=torch.cat((colors,pr['z'][...,None]),-1);bg=torch.cat((bg,bg.new_zeros((1,1))),-1)
  rgb,alpha=rasterize_to_pixels(pr['means'],pr['conics'],colors.contiguous(),pr['opacity'],self.width,self.height,self.tile_size,off,flat,
   backgrounds=bg.contiguous(),packed=False)
  ed=rgb[...,-1:]/alpha.clamp(min=1e-10) if depth else None
  out=dict(render=rgb[0,...,:3].permute(2,0,1),render_alpha=alpha[0].permute(2,0,1),render_depth=ed[0].permute(2,0,1) if depth else None)
  return out,dict(frame_id=state.frames[j],intersection_entries=flat.numel(),projected_rows=len(state.arena.batch.xyz))

 def _one(self,state,j,depth,fused):
  pr=self._project_target(state,j,fused)
  ix=isect_tiles(pr['means'],pr['rad'],pr['z'],self.tile_size,self.tile_width,self.tile_height,sort=True,packed=False,n_cameras=1)
  return self._raster_target(state,j,pr,ix,depth)

 def _record_inputs(self,state,j,stream):
  batch=state.arena.batch
  tensors=self._batch_tensors(batch)+[batch.bundle_metadata.row_owner_ids,state.requests[j],self.background,self.viewmats[state.frames[j]],self.intrinsics[state.frames[j]]]
  if state.row_flags is not None:tensors.append(state.row_flags)
  for t in tensors:t.record_stream(stream)

 @torch.no_grad()
 def render_epoch(self,state,depth,mode):
  self._reap()
  batch=state.arena.batch;count=len(state.frames)
  if not len(batch.xyz):
   out=[dict(render=self.background[:,None,None].expand(3,self.height,self.width).clone(),render_alpha=torch.zeros((1,self.height,self.width),device=self.device),
    render_depth=torch.zeros((1,self.height,self.width),device=self.device) if depth else None) for _ in state.frames]
   return out,dict(mode=mode,frames=list(state.frames),targets=[],flag_bytes=state.flag_bytes,count_host_readbacks=0)
  streams=mode.endswith('stream2') or mode=='staged_count2'
  fused='fused' in mode or mode=='staged_count2'
  caller=torch.cuda.current_stream(self.device)
  if not streams:
   values=[self._one(state,j,depth,fused) for j in range(count)]
   return [x[0] for x in values],dict(mode=mode,frames=list(state.frames),targets=[x[1] for x in values],flag_bytes=state.flag_bytes,count_host_readbacks=count)
  publication=torch.cuda.Event();publication.record(caller)
  out=[];stats=[];done=[];held=[]
  if mode=='staged_count2':
   import staged_isect
   projected=[];counts=[];ready=[]
   for j in range(count):
    worker=self.workers[j%2];worker.wait_event(publication)
    with torch.cuda.stream(worker):
     self._record_inputs(state,j,worker);pr=self._project_target(state,j,True)
     co=staged_isect.count_async(pr['means'],pr['rad'],pr['z'],self.tile_size,self.tile_width,self.tile_height)
     ev=torch.cuda.Event();ev.record(worker)
    projected.append(pr);counts.append(co);ready.append(ev)
   for ev in ready:caller.wait_event(ev)
   totals=staged_isect.read_totals(counts)
   for j in range(count):
    worker=self.workers[j%2]
    with torch.cuda.stream(worker):
     ix=staged_isect.finish(counts[j],totals[j],sort=True)
     result,stat=self._raster_target(state,j,projected[j],ix,depth)
     ev=torch.cuda.Event();ev.record(worker)
    out.append(result);stat['worker_index']=j%2;stats.append(stat);done.append(ev)
   held=[projected,counts,ready]
  else:
   for j in range(count):
    worker=self.workers[j%2];worker.wait_event(publication)
    with torch.cuda.stream(worker):
     self._record_inputs(state,j,worker);result,stat=self._one(state,j,depth,fused)
     ev=torch.cuda.Event();ev.record(worker)
    out.append(result);stat['worker_index']=j%2;stats.append(stat);done.append(ev)
  for ev in done:caller.wait_event(ev)
  for result in out:
   for t in result.values():
    if t is not None:t.record_stream(caller)
  self._pending.append(dict(done=done,publication=publication,inputs=[state,held],outputs=out))
  return out,dict(mode=mode,frames=list(state.frames),targets=stats,flag_bytes=state.flag_bytes,
   count_host_readbacks=1 if mode=='staged_count2' else count,worker_streams=[int(s.cuda_stream) for s in self.workers[:2]])
