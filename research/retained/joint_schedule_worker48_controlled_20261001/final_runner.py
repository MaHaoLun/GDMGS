"""Reviewed batch-barrier CPU/GPU joint-query scheduling experiment."""
import hashlib,inspect,json,math,os,statistics,subprocess,sys,time,types
from pathlib import Path
from collections import deque
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
ART=Path('/ssddata/lun/gdmgs_artifacts')
VENDOR=ROOT/'vendor'
for p in (VENDOR/'base',VENDOR/'occlusion_dfs/parallel_output_v7/dev'):sys.path.insert(0,str(p))
from selection_generalized import AssetsOpt as Assets,CPU_CORES,RESOURCES,PROFILE
CPU_WORKERS=len(CPU_CORES)
# Existing materialization implementation; queries are replaced by this run's results.
sys.path.insert(0,str(ART/'experiment_correction_20260930'))
import run_overlap_timed as existing
import fullblock_timed as blocks
from run_controls import guard
sys.path.insert(0,str(ROOT))

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def allocate(frames,mode,cc,cg):
    n=len(frames)
    if mode=='GPU_only':nc=0
    elif mode=='CPU_only':nc=n
    elif mode=='equal':nc=n//2
    else:
        x=n*cg/(cc+cg);choices={max(0,min(n,math.floor(x))),max(0,min(n,math.ceil(x)))}
        nc=min(choices,key=lambda c:(max(cc*c,cg*(n-c)),c))
    return {f:('cpu' if ((i+1)*nc)//n>(i*nc)//n else 'gpu') for i,f in enumerate(frames)}


def materialize(e,lanes,selected,starts,B,P,quality):
    # S1 is already complete. These adapters consume live query results only.
    bridge={'ids':selected}
    for lane in lanes:
        lane.configure(P)
        def mesh(rt,f,timeline):return f,0.0
        def finish(rt,f,token,ms,timeline,_lane=lane):
            ids=bridge['ids'][f];ids.record_stream(_lane.stream);return ids,{'preselected':True}
        lane.rt._mesh=types.MethodType(mesh,lane.rt);lane.rt._gpu_finish_selection=types.MethodType(finish,lane.rt)
    pool=blocks.Admission(e.base.CAPACITY_ROWS);pending=deque();epochs=[];digests={};unions=[];idx=0
    specs=[blocks.epoch_spec(s,4,len(e.views),True) for s in starts];begin=time.perf_counter()
    def submit(i,lane):
        fut=lanes[lane].executor.submit(blocks.execute_block,e,lanes[lane],i,specs[i],pool,quality,False,begin)
        pending.append((i,lane,fut))
    try:
        while idx<min(B,len(specs)):submit(idx,idx);idx+=1
        while pending:
            i,li,future=pending.popleft();result=future.result();arena=result['arena'];outputs=result['outputs'];record=result['record']
            frames=list(range(arena.spec['start'],arena.spec['end']))
            if quality:
                for f,o in zip(frames,outputs):
                    h=hashlib.sha256()
                    for name in ('render','render_alpha','render_depth'):
                        t=o[name];assert bool(torch.isfinite(t).all());h.update(t.detach().cpu().contiguous().numpy().tobytes())
                    digests[f]=h.hexdigest()
                unions.append(dict(start=arena.spec['start'],union_ids_sha256=hashlib.sha256(arena.batch.anchor_indices.detach().cpu().numpy().tobytes()).hexdigest()))
            record['host_delivery_wall_ms']=(time.perf_counter()-begin)*1000
            epochs.append(record)
            for f in frames:bridge['ids'].pop(f)
            result.clear();del result,arena,outputs,future
            pool.release(i);record['reclaimed_wall_ms']=(time.perf_counter()-begin)*1000
            if idx<len(specs):submit(idx,li);idx+=1
        torch.cuda.synchronize()
    except BaseException:
        pool.cancel()
        for _,_,f in pending:
            try:f.result()
            except BaseException:pass
        raise
    assert not selected and not pool.used
    end=time.perf_counter()
    return dict(begin=begin,end=end,wall_ms=(end-begin)*1000,epochs=epochs,admission=pool.summary(),digests=digests,unions=unions,decoder_calls=sum(r['decoder_calls'] for r in epochs))

def main(scene,stage='all'):
    out=ROOT/('final_qualification' if stage=='parity' else 'final_runs')/PROFILE/scene;out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,set(RESOURCES['main_cores']))
    torch.set_grad_enabled(False);torch.set_num_threads(1)
    uuid=guard();assert uuid==RESOURCES['expected_gpu_uuid'],(uuid,RESOURCES['expected_gpu_uuid']);save(out/'status.json',dict(state='loading'))
    existing.old.OUT=out;existing.old.c.p.ROOT=out
    e=existing.old.c.p.experiment(scene,'joint_barrier_'+str(time.time_ns()));assert len(e.views)==125
    assert e.base.CAPACITY_ROWS==6826846
    save(out/'materialization_sources.json',{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (blocks.__file__,inspect.getfile(blocks.PlannedArena),inspect.getfile(blocks.LaneRenderer))})
    # Remove the old selection oracle from GPU memory and never use its IDs.
    e.oracle_gpu=[]
    a=Assets(e,ROOT,scene);a.configure(1,1)
    save(out/'hardware.json',dict(initial_visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),main_affinity=sorted(os.sched_getaffinity(0)),gpu_uuid=uuid,resource_assignment=RESOURCES,profile=PROFILE,CPU_index_bytes=a.cpu_index_bytes,topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),native_sha256=hashlib.sha256((ROOT/'cpu/joint_cpu_v2.so').read_bytes()).hexdigest(),dataset_manifest_sha256=hashlib.sha256((e.folder/'manifest.json').read_bytes()).hexdigest(),model_files={str(p):existing.old.c.p.sha(p) for p in e.args_model_files()}))
    save(out/'status.json',dict(state='parity'))
    oracle={};differences=[];all_counts=[]
    for f in range(125):
        cpu,cr,cm=a.cpu(f,False);gpu,gr,gm=a.gpu_workers[0].one(f,True)
        g=gpu.cpu().numpy();mesh=gm.cpu().numpy();delta=np.setxor1d(cpu,g)
        assert np.array_equal(cm,mesh),(scene,f,'mesh predicate mismatch')
        assert len(delta)<=1,(scene,f,'more than isolated one-ID boundary difference',delta.tolist())
        if len(delta):differences.append(dict(frame=f,ids=delta.tolist(),cpu_only=np.setdiff1d(cpu,g).tolist(),gpu_only=np.setdiff1d(g,cpu).tolist(),reason='pending LoD-specific verification'))
        oracle[(f,'cpu')]=hashlib.sha256(cpu.tobytes()).hexdigest();oracle[(f,'gpu')]=hashlib.sha256(g.tobytes()).hexdigest()
        all_counts.append(dict(frame=f,cpu_anchors=len(cpu),gpu_anchors=len(g),mesh=len(mesh),cpu_nodes=cr['counters']))
        if f%25==0:print(scene,'parity',f,flush=True)
    save(out/'parity.json',dict(frames=125,mesh_exact=True,anchor_exceptions=differences,counts=all_counts))
    if differences:
        # Attribute exceptions using full GPU LoD membership and CPU scalar expression.
        for d in differences:
            f=d['frame'];v=e.views[f];m=e.rt.model;m.set_anchor_mask(v.camera_center,40000,v.resolution_scale)
            for id in d['ids']:
                import ctypes
                c=a.cameras[f];pred=np.zeros(1,np.float32)
                keep=bool(a.native.cpu_lod_keep(ctypes.byref(a.view),id,c['eye'].ctypes.data,a.standard,a.fork,c['resolution'],a.maxlevel,pred.ctypes.data))
                gpu_keep=bool(m._anchor_mask[id]);assert keep!=gpu_keep
                assert keep==(id in d['cpu_only']) and gpu_keep==(id in d['gpu_only'])
                cpuspatial,_,_=a.cpu(f,False,True)
                full=torch.arange(a.view.na,device='cuda',dtype=torch.int64)
                gpuSpatial=a.gpu_workers[0].hybrid.query(c['cp'],c['planes_gpu'],c['eye_gpu'],full,epochs=1,occlusion=False)[0].cpu().numpy()
                assert np.array_equal(cpuspatial,gpuSpatial)
                d.update(reason='CPU/GPU LoD keep differs; common all-anchor spatial query exact',cpu_predicted_level=float(pred[0]),cpu_lod_keep=keep,gpu_lod_keep=gpu_keep,spatial_exact_with_common_lod=True)
        save(out/'parity.json',dict(frames=125,mesh_exact=True,anchor_exceptions=differences,counts=all_counts))
    if stage=='parity':save(out/'status.json',dict(state='parity_complete'));a.close();return
    save(out/'status.json',dict(state='calibration'));cal_start=time.perf_counter();cal=[]
    for side,values in [('cpu',[CPU_WORKERS]),('gpu',[RESOURCES['frozen_gpu_concurrency'][scene]])]:
        for workers in values:
            point_begin=time.perf_counter()
            a.configure(workers if side=='cpu' else 1,workers if side=='gpu' else 1)
            assignment={f:side for f in range(125)}
            selected,_=a.batch(list(range(125)),assignment);del selected
            torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            selected,r=a.batch(list(range(125)),assignment);del selected
            r['gpu_resident_before_bytes']=resident;r['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            cal.append(dict(side=side,workers=workers,effective_ms=r['wall_ms']/125,calibration_point_ms=(time.perf_counter()-point_begin)*1000,measurement=r))
            save(out/'calibration.json',cal);print(scene,'calibration',side,workers,round(r['wall_ms'],3),flush=True)
    scaling_study_ms=(time.perf_counter()-cal_start)*1000
    bestc=next(r for r in cal if r['side']=='cpu' and r['workers']==CPU_WORKERS)
    bestg=next(r for r in cal if r['side']=='gpu')
    cc,cg=bestc['effective_ms'],bestg['effective_ms'];controller_begin=time.perf_counter();a.configure(CPU_WORKERS,bestg['workers'])
    def quota_assignment(n):return {f:('cpu' if ((f+1)*n)//125>(f*n)//125 else 'gpu') for f in range(125)}
    quota_rows=[]
    for n in (1,2,4,6,8,10,12,14,16,24,32,48):
        for q in (0,n):
            ids,_=a.batch(list(range(125)),quota_assignment(q));del ids
        pairs=[]
        for repeat in range(3):
            pair={}
            for q in ((0,n) if repeat%2==0 else (n,0)):
                guard();torch.cuda.synchronize();ids,record=a.batch(list(range(125)),quota_assignment(q));del ids
                pair['gpu_only' if q==0 else 'hybrid']=record
            pairs.append(pair)
        ratios=[r['gpu_only']['wall_ms']/r['hybrid']['wall_ms'] for r in pairs]
        quota_rows.append(dict(cpu_count=n,ratios=ratios,median_ratio=statistics.median(ratios),pairs=pairs))
        save(out/'quota_calibration.json',quota_rows);print(scene,'quota',n,round(statistics.median(ratios),4),flush=True)
    controller_calibration_ms=(time.perf_counter()-controller_begin)*1000
    positive=[r for r in quota_rows if r['median_ratio']>1]
    chosen=8 if scene=='amsterdam' else (max(positive,key=lambda r:r['median_ratio'])['cpu_count'] if positive else 0)
    global_assignment=quota_assignment(chosen)
    calibration_ms=controller_calibration_ms+sum(r['calibration_point_ms'] for r in cal if r['side']=='gpu')
    arms=[dict(name='GPU_only',allocation='GPU_only',parallel=True,batch=125,B=2,P=1),
          dict(name='CPU_only',allocation='CPU_only',parallel=True,batch=125,B=2,P=1),
          dict(name='hybrid_equal',allocation='equal',parallel=True,batch=125,B=2,P=1),
          dict(name='ours',allocation='calibrated',parallel=True,batch=125,B=2,P=1)]
    if chosen:
        selected=next(x for x in quota_rows if x['cpu_count']==chosen)
        actual_cc=statistics.median(x['hybrid']['cpu_finish_ms'] for x in selected['pairs'])/chosen
        actual_cg=statistics.median(x['hybrid']['gpu_finish_ms'] for x in selected['pairs'])/(125-chosen)
    else:actual_cc=cc;actual_cg=cg
    save(out/'config.json',dict(scene=scene,gpu_uuid=uuid,cpu_workers=CPU_WORKERS,gpu_workers=bestg['workers'],cpu_count=chosen,gpu_count=125-chosen,c_C_solo_ms=cc,c_G_solo_ms=cg,c_C_mixed_ms=actual_cc,c_G_mixed_ms=actual_cg,profiling_total_wall_ms=scaling_study_ms,cpu_scaling_study_ms=sum(r['calibration_point_ms'] for r in cal if r['side']=='cpu'),gpu_concurrency_calibration_ms=sum(r['calibration_point_ms'] for r in cal if r['side']=='gpu'),profile=PROFILE,resource_assignment=RESOURCES,amsterdam_calibration_note='OriginalCPU8exceptionretainedforbothprofiles;commonquotagridstillmeasuredandcharged',controller_calibration_ms=controller_calibration_ms,calibration_ms=calibration_ms,selection_rule='OriginalGPUconcurrencyfrozenperSceneandsharedbyprofiles;AmsterdamCPU8frozeninboth;othersuseoriginalquotagridandmaxpositivemedianS1ratioelse0;allfixedbeforeE2E',batch_assignment='full125only;noB/P/batchsweeps;onlyfourtablearms',arms=arms,K=4,capacity=6826846,formal_repeats=1,render_P_semantics='streams per group lane',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}))
    lanes=[blocks.Lane(e,i) for i in range(2)]
    def run(arm,quality=False):
        guard();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();batch_rows=[];digests={};unions=[];selection_hashes={}
        for start in range(0,125,arm['batch']):
            end=min(125,start+arm['batch']);frames=list(range(start,end));assignment={f:global_assignment[f] for f in frames} if arm['allocation']=='calibrated' else allocate(frames,arm['allocation'],cc,cg)
            ids,s1=a.batch(frames,assignment,arm['parallel']);barrier=time.perf_counter()
            if quality:
                for f,t in ids.items():
                    h=hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest();assert h==oracle[(f,assignment[f])]
                    selection_hashes[f]=h
            m=materialize(e,lanes,ids,list(range(start,end,4)),arm['B'],arm['P'],quality)
            assert m['begin']>=s1['end'] and all(s1['end']>=r['end'] for r in s1['records'])
            assert all(m['begin']+r['started_wall_ms']/1000>=s1['end'] for r in m['epochs'])
            digests.update(m.pop('digests'));unions.extend(m.pop('unions'))
            batch_rows.append(dict(selection=s1,materialization_render=m,predicted_selection_ms=(max((actual_cc if arm['allocation']=='calibrated' else cc)*s1['x_C'],(actual_cg if arm['allocation']=='calibrated' else cg)*s1['x_G']) if arm['parallel'] else (actual_cc*s1['x_C']+actual_cg*s1['x_G'])),predicted_imbalance_ms=abs((actual_cc if arm['allocation']=='calibrated' else cc)*s1['x_C']-(actual_cg if arm['allocation']=='calibrated' else cg)*s1['x_G']),actual_imbalance_ms=abs(s1['cpu_finish_ms']-s1['gpu_finish_ms'])))
        torch.cuda.synchronize();end=time.perf_counter();wall=(end-begin)*1000
        s1=sum(b['selection']['wall_ms'] for b in batch_rows);s23=sum(b['materialization_render']['wall_ms'] for b in batch_rows);gaps=[];previous=begin
        for b in batch_rows:
            gaps.extend([(b['selection']['begin']-previous)*1000,(b['materialization_render']['begin']-b['selection']['end'])*1000]);previous=b['materialization_render']['end']
        gaps.append((end-previous)*1000);assert all(x>=-1e-5 for x in gaps)
        overhead=sum(gaps)
        assert abs(wall-s1-s23-overhead)<1e-4
        assert overhead>=-1e-5 and sum(b['materialization_render']['decoder_calls'] for b in batch_rows)==32
        result=dict(arm=arm,frames=125,wall_ms=wall,fps=125000/wall,selection_ms=s1,S2_S3_ms=s23,overhead_ms=overhead,overhead_gaps_ms=gaps,timeline_begin=begin,timeline_end=end,closure_error_ms=wall-(s1+s23+overhead),first_group_delivery_ms=(batch_rows[0]['materialization_render']['begin']-begin)*1000+batch_rows[0]['materialization_render']['epochs'][0]['host_delivery_wall_ms'],peak_selected_id_bytes=max(b['selection']['selected_id_bytes'] for b in batch_rows),cpu_scratch_peak_estimated_bytes=max(b['selection']['cpu_scratch_peak_estimated_bytes'] for b in batch_rows),cpu_owned_persistent_peak_bytes=max(b['selection']['cpu_owned_persistent_bytes'] for b in batch_rows),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),calibration_inclusive_resident_ms=calibration_ms+wall,batches=batch_rows)
        if quality:result.update(digests=digests,unions=unions,selection_hashes=selection_hashes)
        guard();return result
    try:
        quality={}
        for arm in arms:
            save(out/'status.json',dict(state='quality',arm=arm['name']));quality[arm['name']]=run(arm,True);save(out/'quality.json',quality)
            print(scene,'quality',arm['name'],flush=True)
        quality_comparison={name:sum(v['digests'][f]!=quality['GPU_only']['digests'][f] for f in range(125)) for name,v in quality.items()}
        save(out/'quality_comparison.json',quality_comparison)
        order=arms['amsterdam barcelona bilbao chicago hollywood pompidou quebec rome'.split().index(scene)%len(arms):]+arms[:'amsterdam barcelona bilbao chicago hollywood pompidou quebec rome'.split().index(scene)%len(arms)]
        timed=[]
        for arm in order:
            save(out/'status.json',dict(state='warmup',arm=arm['name']))
            for _ in range(2):run(arm)
            save(out/'status.json',dict(state='timing',arm=arm['name']));result=run(arm);timed.append(result);save(out/'performance.json',timed)
            print(scene,'TIMED',arm['name'],round(result['fps'],3),flush=True)
        save(out/'summary.json',dict(scene=scene,frames=125,results={r['arm']['name']:{k:r[k] for k in ('fps','wall_ms','selection_ms','S2_S3_ms','first_group_delivery_ms','peak_selected_id_bytes','calibration_inclusive_resident_ms')} for r in timed},cpu_workers=CPU_WORKERS,gpu_workers=bestg['workers'],cpu_count=chosen))
        save(out/'status.json',dict(state='complete'))
    finally:
        a.close()
        for lane in lanes:lane.close()
if __name__=='__main__':
    try:main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'all')
    except BaseException as exc:save(ROOT/('final_qualification' if len(sys.argv)>2 and sys.argv[2]=='parity' else 'final_runs')/PROFILE/sys.argv[1]/'status.json',dict(state='failed',error=repr(exc)));raise
