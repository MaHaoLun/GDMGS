"""Full 161-camera qualification; unmeasured until a fresh remote run succeeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from common import load_model,candidates,measured,render,CAM,ART,save,decode_batch
from gdmgs.mesh_index import MeshIndex,CameraDomain
from gdmgs.mesh_index.gpu_index import GPUMeshIndex,camera_planes
from gdmgs.anchor_frustum.gpu_construction import GPUThreeTrees,camera_tensor
from morton_kd import MortonBitKD
from joint import Joint,audit
from hybrid import Hybrid,partial_upper
from serial_reference import SerialHybrid
from range_reference import RangeHybrid
from incremental_depth import IncrementalDepth
from online_proxy_depth import OnlineProxyDepthRasterizer

ROOT=Path(__file__).resolve().parents[1]
QUERY=('joint_one_call','dense_both','serial_no_depth','range_no_depth','parallel_no_depth')
PIPE=('joint_one_call','dense_both','serial_final','range_final','parallel_final','serial_progressive','range_progressive','parallel_progressive','parallel_incremental_single','parallel_incremental')


def stats(v):
    return dict(mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),max=float(np.max(v)))


def diagnostics(d):
    c=d['counts'].cpu().numpy()
    assert int(c[:,5].sum())==0,'DFS stack overflow'
    return dict(frontier=d['frontier'],active=d['active'].cpu().tolist(),counts=c.tolist())


@torch.no_grad()
def main(out,cut,epochs):
    out.mkdir(parents=True,exist_ok=False)
    save(out/'status.json',dict(state='loading',completed_views=0,expected_views=161))
    model,views=load_model();ai=GPUThreeTrees(model)
    meshfile=ART/'proxygs_step4_cpu_mesh_index_g1_v2_20260914/indices/amsterdam/mesh_bvh.npz'
    before=meshfile.stat();cpu=MeshIndex.load(meshfile);gm=GPUMeshIndex(cpu);mi=MortonBitKD(gm)
    j=Joint(ai.bounds,mi.vertices,mi.faces,mi.bounds);h=Hybrid(j,cut);old=SerialHybrid(j,cut);range_only=RangeHybrid(j,cut)
    tree_audit=audit(j.tree,(j.na,j.an,j.mn));assert tree_audit['max_depth']<63
    raster=OnlineProxyDepthRasterizer(cpu);incremental=IncrementalDepth(raster)
    records=json.loads((CAM/'camera_domain_records.json').read_text())
    assert len(records)==161 and [v.image_name for v in views]==[r['camera'] for r in records]
    sm=torch.cuda.get_device_properties(0).multi_processor_count;assert sm==108,(sm,'unexpected resource cap')
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('dev','serial','range_only','nvdiffrast_incremental') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    dep=ROOT.parent/'parallel_output_v5'
    for path in [dep/'dev/incremental_depth.py',*dep.joinpath('nvdiffrast_incremental').rglob('*')]:
        if path.is_file() and '__pycache__' not in str(path):source['../parallel_output_v5/'+str(path.relative_to(dep))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in ('serial','range_only'):
        for path in (ROOT.parent/'parallel_output_v6'/folder).glob('*'):
            if path.is_file():source['../parallel_output_v6/'+str(path.relative_to(ROOT.parent/'parallel_output_v6'))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(out/'contract.json',dict(cameras=[v.image_name for v in views],views=161,repeats=3,cut=cut,epochs=epochs,
        query_modes=QUERY,pipeline_modes=PIPE,source=source,primary_baseline='joint query -> depth generation -> anchor occlusion filter (selection timing)',incremental_depth='actual CudaRaster depth/color buffers retained; project once; append new triangle ranges once; no final full redraw',incremental_depth_comparison='same finite mask, registered camera-z tolerance; final IDs and RGB exact, checks outside timing',early_depth_policy='user-authorized tolerance; no deferred confirmation/recovery',partial_tolerance='max(1e-3 absolute,1e-4 relative,64 float32 eps NDC propagated through inverse projection); invalid -> inf',dfs_warp_per_block=1,output_threads_per_block=256,output_grid='min(leaves,4*reported_SM), GPU grid-stride chunks',implementation_change='deeper BFS/DFS switch, bitmap root compaction, skip single-epoch root sorting, reuse prefix terminal state, direct output chunk owner map, <=256-object inline terminals with larger ranges parallel, local diagnostic reduction; geometry and depth predicates unchanged',tree_audit=tree_audit,sm=sm,device=torch.cuda.get_device_name(),
        build_ms=j.build_ms,scope='same-frame positive-z support-based proxy occlusion; not historical center predicate',
        timing='synchronized resident-camera query; selection includes raster/HiZ/filter; pipeline includes fresh decode/RGB; no outlier removal',
        caveats=['frontier count host read is included','exact diagnostic counts locally reduced; reduced global atomics included','progressive uses partial rasters plus final full raster','not an optimized production kernel']))
    bg=torch.zeros(3,device='cuda');rows=[]
    for vi,(v,c) in enumerate(zip(views,records)):
        ids=candidates(model,v);cp=camera_tensor(v)
        domain=CameraDomain.parse(dict(w2c=np.array(c['w2c'],np.float64),angular_domain=c['angular_domain'],near=c['near'],far=c['far'],camera_id=c['camera']))
        assert (v.image_width,v.image_height)==(1600,900)
        assert np.array_equal(v.world_view_transform.T.cpu().numpy(),np.asarray(c['w2c']))
        xmin,xmax,ymin,ymax=domain.angular_domain
        expected=np.array([1600/(xmax-xmin),900/(ymax-ymin),-xmin*1600/(xmax-xmin),-ymin*900/(ymax-ymin)])
        assert np.allclose(cp[16:20].cpu().numpy(),expected,rtol=1e-6,atol=1e-5),'depth/support camera mismatch'
        planes=torch.tensor(camera_planes(domain),device='cuda',dtype=torch.float64)
        eye=torch.tensor(np.linalg.solve(domain.w2c[:3,:3],-domain.w2c[:3,3]),device='cuda',dtype=torch.float64)
        ae=ids[ai.ext.gpu_dense_mask(ai.bounds,cp)[ids]]
        me=torch.tensor(cpu.query(domain,backend='brute_force').triangle_ids,device='cuda')
        def aq():
            mask,_,_=ai.ext.query_radix(ai.bounds,ai.nodes,ai.left,ai.right,ai.parent,ai.sorted_ids,cp,ai.axis,ai.plane,32,True,eye)
            return ids[mask[ids]]
        def mq():
            mask,_,_=mi.ext.query_mesh(mi.vertices,mi.faces,mi.nodes,mi.left,mi.right,mi.parent,mi.order,planes,mi.axis,mi.split,eye,32,True)
            return torch.nonzero(mask).flatten()
        def query(mode):
            if mode=='independent_anchor':return aq(),None
            if mode=='independent_mesh':return None,mq()
            if mode=='independent_both':return aq(),mq()
            if mode=='shared_two_calls':return j.query(cp,planes,eye,ids,1)[0],j.query(cp,planes,eye,ids,2)[1]
            if mode=='joint_one_call':return j.query(cp,planes,eye,ids)[:2]
            if mode=='parallel_no_depth':return h.query(cp,planes,eye,ids)[:2]
            if mode=='range_no_depth':return range_only.query(cp,planes,eye,ids)[:2]
            if mode=='serial_no_depth':return old.query(cp,planes,eye,ids)[:2]
            if mode=='dense_both':
                a=ids[ai.ext.gpu_dense_mask(ai.bounds,cp)[ids]]
                mask=gm._native.triangle_mask(gm._vertices,gm._triangles,planes,gm._all_leaves,gm._face_leaf,False)
                return a,torch.nonzero(mask).flatten()
            raise ValueError(mode)
        def select(mode):
            if mode in ('parallel_incremental_single','parallel_incremental'):
                return h.query(cp,planes,eye,ids,raster=raster,domain=domain,epochs=1 if mode.endswith('_single') else epochs,occlusion=True,incremental=incremental)
            if mode.startswith(('serial_','range_','parallel_')):
                impl=old if mode.startswith('serial_') else (range_only if mode.startswith('range_') else h)
                return impl.query(cp,planes,eye,ids,raster=raster,domain=domain,epochs=epochs if mode.endswith('_progressive') else 1,occlusion=True)
            a,m=query(mode);d=raster.render(m,domain,(1600,900),copy_to_cpu=False).depth_gpu
            return h.filter(a,cp,d),m,d,None
        def pipeline(mode):
            a,m,d,diag=select(mode)
            batch=decode_batch(v,model,a,model.get_level[a].reshape(-1).long())
            return a,m,d,render(v,batch,bg),diag
        depth_ref=raster.render(me,domain,(1600,900),copy_to_cpu=False).depth_gpu
        all_depth=raster.render(torch.arange(mi.N,device='cuda'),domain,(1600,900),copy_to_cpu=False).depth_gpu
        assert torch.equal(depth_ref,all_depth),'full mesh depth mismatch'
        del all_depth
        selected_ref=h.filter(ae,cp,depth_ref)
        batch=decode_batch(v,model,selected_ref,model.get_level[selected_ref].reshape(-1).long())
        rgb_ref=render(v,batch,bg);selected_gaussians=len(batch.xyz);del batch
        full_batch=decode_batch(v,model,ae,model.get_level[ae].reshape(-1).long())
        full_rgb=render(v,full_batch,bg);delta=(rgb_ref-full_rgb).abs()
        row=dict(camera=v.image_name,index=vi,candidates=len(ids),frustum_anchors=len(ae),selected_anchors=len(selected_ref),
            triangles=len(me),selected_gaussians=selected_gaussians,frustum_gaussians=len(full_batch.xyz),
            ids_exact=True,depth_exact=True,rgb_exact_across_modes=True,
            quality=dict(pixel_exact_no_occlusion=torch.equal(rgb_ref,full_rgb),max_rgb_delta=float(delta.max()),mae=float(delta.mean()),mse=float((delta**2).mean())),
            query={m:[] for m in QUERY},selection={m:[] for m in PIPE},pipeline={m:[] for m in PIPE},unfiltered_pipeline={m:[] for m in PIPE[:2]},diagnostics={})
        del full_batch,delta
        def unfiltered(mode):
            a,m=query(mode);d=raster.render(m,domain,(1600,900),copy_to_cpu=False).depth_gpu
            batch=decode_batch(v,model,a,model.get_level[a].reshape(-1).long())
            return a,m,d,render(v,batch,bg)
        def check_selection(result,mode,with_rgb=False):
            assert torch.equal(result[0],selected_ref),(vi,mode,'selected anchor mismatch')
            assert torch.equal(result[1],me),(vi,mode,'mesh IDs mismatch')
            if mode in ('parallel_incremental_single','parallel_incremental'):
                finite=torch.isfinite(depth_ref);assert torch.equal(torch.isfinite(result[2]),finite),(vi,mode,'incremental coverage')
                actual=result[2];allow=torch.maximum(partial_upper(depth_ref,domain.near,domain.far)-depth_ref,partial_upper(actual,domain.near,domain.far)-actual)
                error=(actual[finite]-depth_ref[finite]).abs();assert (error<=allow[finite]).all(),(vi,mode,'incremental depth tolerance')
                row.setdefault('incremental_depth',{})[mode]=dict(bitwise_exact=torch.equal(actual,depth_ref),max_abs_error=float(error.max()) if error.numel() else 0.,within_tolerance=True)
            else:assert torch.equal(result[2],depth_ref),(vi,mode,'depth mismatch')
            if with_rgb:assert torch.equal(result[3],rgb_ref),(vi,mode,'RGB mismatch')
            diag=result[4] if with_rgb else result[3]
            if diag is not None:
                row['diagnostics'][mode]=diagnostics(diag)
                if diag.get('incremental') is not None:
                    inc=diag['incremental'];assert inc['native_depth_buffer_reused'] and inc['triangle_submissions']==len(me) and inc['final_full_redraws']==0
                    assert inc['clears']==(1 if len(me) else 0)
                    row['diagnostics'][mode]['incremental']=inc
                if diag['partial'] is not None:
                    partial=diag['partial'];finite=torch.isfinite(partial)
                    assert torch.isfinite(depth_ref[finite]).all(),'partial coverage missing from full mesh'
                    margin=32*torch.finfo(torch.float32).eps*(partial[finite].abs()+1)
                    row['diagnostics'][mode]['partial_nonmonotonic_pixels']=int((depth_ref[finite]>partial[finite]+margin).sum())
                    upper=partial_upper(partial,domain.near,domain.far)
                    assert (depth_ref[finite]<=upper[finite]).all(),'partial depth exceeds registered tolerance'
                    row['diagnostics'][mode]['max_full_minus_partial']=float((depth_ref[finite]-partial[finite]).clamp_min(0).max()) if finite.any() else 0.
                    row['diagnostics'][mode]['partial_coverage']=float(finite.float().mean())
        # Every view gets paired repetitions; first view also gets unrecorded warmups.
        for rep in range(-2 if vi==0 else 0,3):
            shift=(vi+rep)%len(QUERY)
            for mode in QUERY[shift:]+QUERY[:shift]:
                result,timing=measured(lambda:query(mode))
                if result[0] is not None:assert torch.equal(result[0],ae),(vi,mode,'frustum anchor mismatch')
                if result[1] is not None:assert torch.equal(result[1],me),(vi,mode,'frustum mesh mismatch')
                if rep>=0:row['query'][mode].append(timing)
            shift=(vi+rep)%len(PIPE)
            for mode in PIPE[shift:]+PIPE[:shift]:
                result,timing=measured(lambda:select(mode));check_selection(result,mode)
                if rep>=0:row['selection'][mode].append(timing)
                result,timing=measured(lambda:pipeline(mode));check_selection(result,mode,True)
                if rep>=0:row['pipeline'][mode].append(timing)
                if mode in PIPE[:2]:
                    raw,rt=measured(lambda:unfiltered(mode))
                    assert torch.equal(raw[0],ae) and torch.equal(raw[1],me) and torch.equal(raw[2],depth_ref) and torch.equal(raw[3],full_rgb),(vi,mode,'unfiltered pipeline parity')
                    if rep>=0:row['unfiltered_pipeline'][mode].append(rt)
        for kind in ('final','progressive'):
            aa=np.asarray(row['diagnostics']['serial_'+kind]['counts']);bb=np.asarray(row['diagnostics']['parallel_'+kind]['counts'])
            assert np.array_equal(aa,bb[:,:6]),(vi,kind,'traversal or predicate work changed')
            cc=np.asarray(row['diagnostics']['range_'+kind]['counts']);assert np.array_equal(aa,cc[:,:6])
        rows.append(row);save(out/'per_view.json',rows)
        save(out/'status.json',dict(state='running',completed_views=len(rows),expected_views=161))
        print(f'{vi+1}/161 serial/parallel IDs, work counts, depth and RGB exact',flush=True)
    after=meshfile.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    summary=dict(status='PASS_ALGORITHM',views=len(rows),repeats=3,ids_rgb_parity=True,nonincremental_depth_exact=True,incremental_depth_within_tolerance=True,input_unchanged=True,
        no_occlusion_rgb_exact_views=sum(r['quality']['pixel_exact_no_occlusion'] for r in rows),
        quality=dict(max_rgb_delta=max(r['quality']['max_rgb_delta'] for r in rows),mean_mae=float(np.mean([r['quality']['mae'] for r in rows]))),
        removed_anchors=sum(r['frustum_anchors']-r['selected_anchors'] for r in rows),
        timings={scenario:{m:{k:stats([t[k] for r in rows for t in r[scenario][m]]) for k in ('wall_ms','event_ms')} for m in modes} for scenario,modes in [('query',QUERY),('selection',PIPE),('pipeline',PIPE),('unfiltered_pipeline',PIPE[:2])]})
    save(out/'summary.json',summary);save(out/'status.json',dict(state='complete',status=summary['status'],completed_views=161))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cut',type=int,default=20);parser.add_argument('--epochs',type=int,default=4);a=parser.parse_args()
    try:main(a.output,a.cut,a.epochs)
    except Exception:
        # Never overwrite a prior completed output as a new failure.
        failure=ROOT/'last_failure.txt';failure.write_text(traceback.format_exc());raise
