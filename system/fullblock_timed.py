"""Private full-block lanes: online selection, split exact decode, target render."""
import ast,copy,inspect,textwrap,threading,time,types
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import torch
from einops import repeat
from cache_build_optim import build_union_plan,PlannedArena,fused_batch
from epoch_cache import epoch_spec,source_view
from priority_renderer import PriorityRenderer


@contextmanager
def mark(enabled,name):
    if enabled:torch.cuda.nvtx.range_push(name)
    try:yield
    finally:
        if enabled:torch.cuda.nvtx.range_pop()


def clone_method(instance,name,barriers,remove_guard=False):
    fn=getattr(instance,name).__func__
    if fn.__name__=='guarded':fn=inspect.getclosurevars(fn).nonlocals['method']
    source=textwrap.dedent(inspect.getsource(fn));tree=ast.parse(source)
    if remove_guard:tree.body[0].decorator_list=[]
    class Narrow(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='torch.cuda.synchronize':
                self.count+=1
                stream=ast.Call(func=ast.parse('torch.cuda.current_stream',mode='eval').body,args=node.args,keywords=node.keywords)
                return ast.copy_location(ast.Call(func=ast.Attribute(value=stream,attr='synchronize',ctx=ast.Load()),args=[],keywords=[]),node)
            return node
    tr=Narrow();tree=tr.visit(tree);ast.fix_missing_locations(tree)
    assert tr.count==barriers,(name,tr.count)
    ns=dict(fn.__globals__);exec(compile(tree,'<private-lane:'+name+'>','exec'),ns)
    setattr(instance,name,types.MethodType(ns[name],instance))
    return dict(method=name,barriers=barriers,guard_removed=remove_guard,source_file=inspect.getsourcefile(fn),transformed=ast.unparse(tree))


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
    repeated=repeat(concatenated,'n (c) -> (n k) (c)',k=n_offsets)
    masked=torch.cat([repeated,color,scale_rot,offsets],dim=-1)[mask]
    scaling_repeat,repeat_anchor,color,scale_rot,offsets=masked.split([6,3,3,7,3],dim=-1)
    scaling=scaling_repeat[:,3:]*torch.sigmoid(scale_rot[:,:3]);rotation=model.rotation_activation(scale_rot[:,3:7])
    xyz=repeat_anchor+offsets*scaling_repeat[:,:3]
    batch=fused_batch(ids,(xyz,color,opacity,scaling,rotation,mask),n_offsets,levels)
    assert len(batch.xyz)==st['rows']
    return batch


class Admission:
    """FIFO exact row reservations; budget released only by the ordered consumer."""
    def __init__(self,capacity):
        self.capacity=capacity;self.cv=threading.Condition();self.next=0;self.used={};self.actual={};self.failed=False
        self.max_reserved=0;self.max_actual=0;self.max_generations=0;self.waits=[]
    def acquire(self,index,rows):
        if rows>self.capacity:raise RuntimeError('single exact generation exceeds capacity')
        before=time.perf_counter();waited=False
        with self.cv:
            while not self.failed and (index!=self.next or sum(self.used.values())+rows>self.capacity):
                waited=True;self.cv.wait()
            if self.failed:raise RuntimeError('admission cancelled')
            self.used[index]=rows;self.next+=1;self.max_reserved=max(self.max_reserved,sum(self.used.values()))
            self.max_generations=max(self.max_generations,len(self.used));self.cv.notify_all()
            self.waits.append(dict(index=index,rows=rows,waited=waited,wall_ms=(time.perf_counter()-before)*1000))
    def materialized(self,index,rows):
        with self.cv:
            assert self.used[index]==rows
            self.actual[index]=rows;self.max_actual=max(self.max_actual,sum(self.actual.values()))
    def release(self,index):
        with self.cv:del self.used[index];self.actual.pop(index,None);self.cv.notify_all()
    def cancel(self):
        with self.cv:self.failed=True;self.cv.notify_all()
    def summary(self):
        return dict(capacity_rows=self.capacity,max_reserved_plus_live_rows=self.max_reserved,
                    max_materialized_live_rows=self.max_actual,max_generations=self.max_generations,waits=self.waits,
                    final_reserved_rows=sum(self.used.values()))


class LaneRenderer(PriorityRenderer):
    profile=False
    lane_id=0
    def _project_target(self,state,j,fused):
        with mark(self.profile,f'FULL|{state.arena.spec["start"]}|{self.lane_id}|project|{state.frames[j]}'):
            return super()._project_target(state,j,fused)
    def _raster_target(self,state,j,pr,ix,depth):
        with mark(self.profile,f'FULL|{state.arena.spec["start"]}|{self.lane_id}|raster|{state.frames[j]}'):
            return super()._raster_target(state,j,pr,ix,depth)


class Lane:
    def __init__(self,e,index):
        self.index=index;self.stream=torch.cuda.Stream(device=e.rt.background.device)
        rt=copy.copy(e.rt);rt.model=copy.copy(e.rt.model);rt.gpu_lock=threading.RLock()
        rt.pending_checks=[];rt.pending_mesh_checks=[]
        for name in ('_anchor_mask','_prog_ratio','transition_mask'):
            value=getattr(rt.model,name,None)
            if isinstance(value,torch.Tensor):setattr(rt.model,name,value.clone())
        rt.gpu_mesh_index=copy.copy(e.rt.gpu_mesh_index);rt.gpu_anchor_index=copy.copy(e.rt.gpu_anchor_index)
        rt.rasterizer=copy.copy(e.rt.rasterizer)
        rt.rasterizer._context=rt.rasterizer._dr.RasterizeCudaContext(device=rt.rasterizer.device)
        self.audit=[clone_method(rt,'_mesh',2),clone_method(rt,'_gpu_finish_selection',1,True),clone_method(rt.rasterizer,'render',1)]
        render=rt.rasterizer.render;self.capture=False;self.last_depth=None
        def capture(*args,**kwargs):
            value=render(*args,**kwargs)
            if self.capture:self.last_depth=value.depth_gpu
            return value
        rt.rasterizer.render=capture
        self.rt=rt;self.renderer=LaneRenderer(e.views,e.rt.background,requests_sorted=True);self.renderer.lane_id=index
        self.original_workers=list(self.renderer.workers)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix=f'fullblock-{index}')
    def configure(self,p):
        self.renderer.workers=list(self.original_workers)
        if p==1:self.renderer.workers[1]=self.renderer.workers[0]
        assert p in (1,2)
    def close(self):self.executor.shutdown(wait=True,cancel_futures=True)


def execute_block(e,lane,index,spec,pool,quality,profile,begin,capture=False,delay=0):
    torch.set_grad_enabled(False);rt=lane.rt;frames=list(range(spec['start'],spec['end']));requests=[];snapshots=[]
    started_wall=(time.perf_counter()-begin)*1000
    def tag(phase):return mark(profile,f'FULL|{spec["start"]}|{lane.index}|{phase}')
    with torch.cuda.device(rt.background.device),torch.cuda.stream(lane.stream):
        lane.capture=capture;lane.renderer.profile=profile
        with tag('selection'):
            for f in frames:
                tl=e.base.Timeline();mesh,ms=rt._mesh(f,tl);ids,_=rt._gpu_finish_selection(f,mesh,ms,tl)
                requests.append(ids)
                if capture:snapshots.append(dict(frame=f,mesh=mesh.triangle_ids.clone(),ids=ids.clone(),depth=lane.last_depth.clone()))
                rt.pending_checks.clear();rt.pending_mesh_checks.clear()
        selected_wall=(time.perf_counter()-begin)*1000
        with tag('decode_prepare'):
            plan=build_union_plan(requests);view=source_view(e,spec)
            rt.model.set_anchor_mask(view.camera_center,40000,view.resolution_scale)
            st=prepare_decode(view,rt.model,plan.ordered)
        exact_rows=st['rows'];prepared_wall=(time.perf_counter()-begin)*1000
        with tag('admission'):pool.acquire(index,exact_rows)
        admitted_wall=(time.perf_counter()-begin)*1000
        with tag('decode_finish'):
            def decode(ids,levels):return finish_decode(st,rt.model,ids,levels)
            arena=PlannedArena(spec,requests,rt.levels,decode,pool.capacity,plan)
            assert not arena.overflow and arena.decoded_rows==exact_rows
            pool.materialized(index,exact_rows)
            del st,plan,decode,view
        decoded_wall=(time.perf_counter()-begin)*1000
        with tag('render'):
            if delay:torch.cuda._sleep(delay)
            state=lane.renderer.prepare_epoch(arena,requests,frames,'staged_count2')
            outputs,stats=lane.renderer.render_epoch(state,quality,'staged_count2')
            del state
            done=torch.cuda.Event();done.record(lane.stream)
        submitted_wall=(time.perf_counter()-begin)*1000
        with tag('completion'):done.synchronize()
        lane.renderer._reap();lane.last_depth=None;lane.capture=False
        return dict(arena=arena,outputs=outputs,requests=requests,snapshots=snapshots,
                    record=dict(spec,union_anchors=arena.decoded_anchors,cache_rows=arena.decoded_rows,cache_bytes=arena.bytes,
                                decoder_calls=arena.decoder_calls,renderer_stats=stats,lane=lane.index,exact_admission_rows=exact_rows,
                                started_wall_ms=started_wall,selection_ready_wall_ms=selected_wall,opacity_ready_wall_ms=prepared_wall,
                                admitted_wall_ms=admitted_wall,
                                decode_ready_wall_ms=decoded_wall,render_submitted_wall_ms=submitted_wall,
                                gpu_done_observed_wall_ms=(time.perf_counter()-begin)*1000))


def full_sequence(e,lanes,mode,quality,*,on_epoch=None,on_producer=None,starts=None,profile=False,delay=0):
    parallel=mode.startswith('full_parallel');p=int(mode[-1]);active=2 if parallel else 1
    for lane in lanes:lane.configure(p)
    specs=[epoch_spec(s,4,len(e.views),True) for s in (range(0,len(e.views),4) if starts is None else starts)]
    expected=[f for s in specs for f in range(s['start'],s['end'])]
    assert expected==sorted(set(expected))
    pool=Admission(e.base.CAPACITY_ROWS);pending=deque();metrics=[];epochs=[];checks=[];delivered=[];next_index=0
    e.guard();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();first=None
    def submit(index,lane_index):
        f=lanes[lane_index].executor.submit(execute_block,e,lanes[lane_index],index,specs[index],pool,quality,profile,begin,on_producer is not None,delay)
        pending.append((index,lane_index,f))
    try:
        while next_index<min(active,len(specs)):submit(next_index,next_index);next_index+=1
        while pending:
            index,lane_index,future=pending.popleft();result=future.result()
            if first is None:first=(time.perf_counter()-begin)*1000
            arena=result.pop('arena');outputs=result.pop('outputs');requests=result.pop('requests');snapshots=result.pop('snapshots')
            frames=list(range(arena.spec['start'],arena.spec['end']))
            for output in outputs:
                for tensor in output.values():
                    if isinstance(tensor,torch.Tensor):tensor.record_stream(torch.cuda.default_stream())
            if quality:metrics.extend(e.metrics(f,o) for f,o in zip(frames,outputs))
            if on_epoch is not None:on_epoch(arena,outputs,result['record']['renderer_stats'])
            if on_producer is not None:on_producer(snapshots)
            checks.extend(zip(frames,requests));delivered.extend(frames)
            record=result.pop('record');record['host_delivery_wall_ms']=(time.perf_counter()-begin)*1000;epochs.append(record)
            # Future keeps result: clear the dictionary and all arena/output refs
            # BEFORE returning the row reservation to another decoder.
            result.clear();del arena,outputs,requests,snapshots,result,future,output,tensor
            pool.release(index)
            epochs[-1]['reclaimed_wall_ms']=(time.perf_counter()-begin)*1000
            if next_index<len(specs):submit(next_index,lane_index);next_index+=1
        torch.cuda.synchronize()
    except BaseException:
        pool.cancel()
        for _,_,f in pending:
            try:f.result()
            except BaseException:pass
        torch.cuda.synchronize();raise
    wall=(time.perf_counter()-begin)*1000
    assert delivered==expected and all(torch.equal(ids,e.oracle_gpu[f]) for f,ids in checks)
    assert not pool.used;e.guard()
    return dict(mode=mode,frames=len(delivered),wall_ms=None if quality else wall,fps=None if quality else len(delivered)*1000/wall,
                metrics=metrics,epochs=epochs,quality_pass=all(q['safe'] for q in metrics) if quality else None,
                decoder_calls=sum(x['decoder_calls'] for x in epochs),decoded_anchors=sum(x['union_anchors'] for x in epochs),
                first_group_ready_wall_ms=first,max_allocated_bytes=torch.cuda.max_memory_allocated(),admission=pool.summary(),
                oracle_exact=True,gpu=e.guard_records[-2:],artificial_delay_cycles=delay,
                timeline_origin_s=begin)
