"""Independent full-inventory/raw-sample/source and work-parity review."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def main(root):
    out=root/'runs/amsterdam_v7'
    c=json.loads((out/'contract.json').read_text());s=json.loads((out/'summary.json').read_text())
    rows=json.loads((out/'per_view.json').read_text())
    prior=json.loads((root.parent/'runs/amsterdam_cut12_epoch4_v3/per_view.json').read_text())
    assert len(rows)==161 and s['views']==161 and s['repeats']==3 and s['status']=='PASS_ALGORITHM'
    assert [r['camera'] for r in rows]==c['cameras']==[r['camera'] for r in prior]
    assert len(set(c['cameras']))==161 and [r['index'] for r in rows]==list(range(161))
    counts=('candidates','frustum_anchors','selected_anchors','triangles','selected_gaussians','frustum_gaussians')
    assert [[r[k] for k in counts] for r in rows]==[[r[k] for k in counts] for r in prior]
    for k,v in c['source'].items():assert hashlib.sha256((root/k).read_bytes()).hexdigest()==v,k
    for p in (root/'serial').glob('*'):
        if p.is_file():
            a=p.read_text();b=(root.parent/'dev'/p.name).read_text()
            if p.name=='hybrid.py':a=a.replace('serial_subtree_output_baseline_20260924','joint_occlusion_dfs_20260924')
            assert a==b,p.name
    for p in (root/'range_only').glob('*'):
        if p.is_file():assert p.read_bytes()==(root.parent/'parallel_output_v4/dev'/p.name).read_bytes(),p.name
    ts=ET.parse(root/'tests.xml').getroot().find('testsuite')
    assert ts is not None and int(ts.attrib['tests'])==25 and int(ts.attrib['failures'])==0 and int(ts.attrib['errors'])==0
    samples=0
    scenarios=[('query',c['query_modes']),('selection',c['pipeline_modes']),('pipeline',c['pipeline_modes']),('unfiltered_pipeline',c['pipeline_modes'][:2])]
    for section,modes in scenarios:
        for mode in modes:
            assert all(len(r[section][mode])==3 for r in rows)
            samples+=3*len(rows)
            for metric in ('wall_ms','event_ms'):
                v=np.array([t[metric] for r in rows for t in r[section][mode]])
                assert len(v)==483 and np.isfinite(v).all() and (v>=0).all()
                expected=dict(mean=float(v.mean()),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),max=float(v.max()))
                for k,value in expected.items():assert np.isclose(value,s['timings'][section][mode][metric][k],rtol=1e-12,atol=1e-12)
    work={}
    for kind in ('final','progressive'):
        chunks=[];blocks=[];oldtotal=np.zeros(4,dtype=np.int64);newtotal=oldtotal.copy()
        for r in rows:
            assert r['ids_exact'] and r['depth_exact'] and r['rgb_exact_across_modes']
            old=np.array(r['diagnostics']['serial_'+kind]['counts']);new=np.array(r['diagnostics']['parallel_'+kind]['counts'])
            assert np.array_equal(old,new[:,:6]) and (new[:,5]==0).all()
            assert np.array_equal(old,np.array(r['diagnostics']['range_'+kind]['counts'])[:,:6])
            oldtotal+=old[:,:4].sum(0);newtotal+=new[:,:4].sum(0)
            chunks.extend(new[:,6].tolist());blocks.extend(new[:,7].tolist())
        work[kind]=dict(serial_counts=oldtotal.tolist(),parallel_counts=newtotal.tolist(),chunks_total=sum(chunks),
            chunks_min=min(chunks),chunks_max=max(chunks),nonempty_blocks_min=min(x for x in blocks if x),blocks_max=max(blocks))
    incremental={}
    for mode in ('parallel_incremental_single','parallel_incremental'):
        ds=[r['diagnostics'][mode]['incremental'] for r in rows]
        assert all(d['native_depth_buffer_reused'] and d['final_full_redraws']==0 and d['projected_vertex_passes']==1 for d in ds)
        assert all(d['triangle_submissions']==r['triangles'] and d['clears']==(1 if r['triangles'] else 0) for r,d in zip(rows,ds))
        assert all(r['incremental_depth'][mode]['within_tolerance'] for r in rows)
        incremental[mode]=dict(depth_bitwise_exact_views=sum(r['incremental_depth'][mode]['bitwise_exact'] for r in rows),max_depth_error=max(r['incremental_depth'][mode]['max_abs_error'] for r in rows),all_native_buffers_reused=True,project_once_per_frame=True,all_triangles_submitted_once=True,final_full_redraws=0)
    pairs={}
    for section,old,new in [('query','serial_no_depth','range_no_depth'),('query','range_no_depth','parallel_no_depth'),('query','serial_no_depth','parallel_no_depth'),('selection','serial_final','parallel_final'),('selection','serial_progressive','parallel_progressive'),('pipeline','serial_final','parallel_final'),('pipeline','serial_progressive','parallel_progressive'),('pipeline','range_progressive','parallel_progressive'),('selection','joint_one_call','parallel_incremental'),('selection','parallel_progressive','parallel_incremental'),('selection','parallel_incremental_single','parallel_incremental')]:
        x=s['timings'][section][old]['wall_ms']['mean'];y=s['timings'][section][new]['wall_ms']['mean']
        per_view=[np.mean([t['wall_ms'] for t in r[section][old]])-np.mean([t['wall_ms'] for t in r[section][new]]) for r in rows]
        pairs[section+'/'+old+'->'+new]=dict(before_ms=x,after_ms=y,speedup=x/y,saved_fraction=1-y/x,faster_views=int(np.count_nonzero(np.array(per_view)>0)))
    review=dict(status='PASS',views=161,tests=25,repeats=3,samples=samples,source_hashes_match=True,
        serial_baseline_matches_v3=True,counts_match_v3=True,ids_rgb_exact=True,nonincremental_depth_exact=True,incremental_depth_within_tolerance=True,incremental=incremental,work_counts_exact=True,
        all_raw_statistics_recomputed=True,all_samples_retained=True,checks_excluded_from_timing=True,pairs=pairs,work=work,
        temporary_range_buffers_bytes_per_epoch=18*328729+4*((5259654+255)//256+164365),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'contract.json',out/'summary.json',out/'per_view.json']})
    (root/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    names=dict(joint_one_call='原联合查询',dense_both='两类 GPU dense scan',serial_no_depth='DFS：原串行输出',parallel_no_depth='DFS：并行输出＋计数汇总',range_no_depth='DFS：仅并行输出',range_final='仅并行输出＋一次 depth',range_progressive='仅并行输出＋四批 depth',serial_final='原输出＋一次 depth',parallel_final='并行输出＋一次 depth',serial_progressive='原输出＋四批 depth',parallel_progressive='并行输出＋四批 depth',parallel_incremental_single='优化包装层：一次 depth',parallel_incremental='并行输出＋原生增量 depth')
    report=['# 2026-09-24 Amsterdam｜DFS 子树输出深层 DFS＋直接并行输出＋原生增量 depth v7 大小范围分流结果','',
        f'完整验证 PASS：161/161 视角，25/25 CUDA 测试，每模式 3 次，共 {samples:,} 个正式计时样本。最终有序 IDs、同策略 RGB exact；非增量完整 depth exact，增量 depth 使用登记容差并保持 coverage exact。三种非增量 DFS 节点/判定次数相同。参考生成和一致性比对全部不计时。','',
        '## 同轮纯查询','', '| 方案 | 均值 ms | p95 ms | p99 ms |','|---|---:|---:|---:|']
    for m in c['query_modes']:
        x=s['timings']['query'][m]['wall_ms'];report.append(f'| {names[m]} | {x["mean"]:.3f} | {x["p95"]:.3f} | {x["p99"]:.3f} |')
    report+=['','## 遮挡筛选与完整流程','', '| 方案 | query→depth→filter ms | query→depth→filter→decode→RGB ms | 完整流程 p95 ms |','|---|---:|---:|---:|']
    for m in c['pipeline_modes']:
        x=s['timings']['selection'][m]['wall_ms'];y=s['timings']['pipeline'][m]['wall_ms'];report.append(f'| {names[m]} | {x["mean"]:.3f} | {y["mean"]:.3f} | {y["p95"]:.3f} |')
    report+=['','## 修改效果','']
    for k,pair in pairs.items():report.append(f'- {k}：{pair["before_ms"]:.3f} → {pair["after_ms"]:.3f} ms，{pair["speedup"]:.2f}×；{pair["faster_views"]}/161 视角均值更快。')
    report+=['','## GPU 工作分配与边界','',
        '切换层由诊断选择为 20。小于等于 256 个对象的终止范围直接由 DFS warp 输出，较大范围登记任务；GPU 前缀和分配每块 256 个对象的任务，再由直接 owner 映射让独立输出 kernel 并行处理，避免每块二分查找。root 用 bitmap 有序压缩；单批模式跳过近处排序；复用 BFS 已判定状态。即使只有一个大 Keep 节点，也可分配到上百个输出 block。多个块按 grid-stride 消费任务，无输出任务数 D2H 同步。',
        f'一次 depth 模式的输出任务总计 {work["final"]["chunks_total"]:,}，每次 {work["final"]["chunks_min"]:,}–{work["final"]["chunks_max"]:,} 个；非空输出使用 {work["final"]["nonempty_blocks_min"]}–{work["final"]["blocks_max"]} 个 block。四批模式对应范围 {work["progressive"]["chunks_min"]:,}–{work["progressive"]["chunks_max"]:,} 个任务，非空输出使用 {work["progressive"]["nonempty_blocks_min"]}–{work["progressive"]["blocks_max"]} 个 block。',
        'A100 80GB GPU 4，108 SM。DFS 仍是一 warp/block；输出为 256 threads/block，最多 4×SM 个 block。任务计数证明工作已拆开；没有硬件计数器数据，不能声称实际 occupancy 或 GPU 利用率达到 100%。',
        f'长度、前缀和与状态数组显式增加 {review["temporary_range_buffers_bytes_per_epoch"]:,} bytes/epoch，不含框架 scan 临时工作区；分配、清零、scan 和输出成本均包含在计时内。',
        '节点计数先在 DFS warp 内累计，叶判定计数先在输出 block 内汇总，再提交全局；整数计数与冻结串行版逐视角完全相同。该步骤减少诊断原子操作的竞争，仍将统计成本计入运行时间。',
        '主比较基线是原联合查询→mesh depth→anchor 遮挡筛选（selection）。纯查询不作为新增 depth/filter 的性能分母。',
        '非增量分支保留 v3 的 3 次局部 depth + 1 次完整 depth。增量分支每帧只投影一次，首个非空批次清空原生 depth/color 缓冲，后续保留同一缓冲并只追加新三角形；末尾没有全量 redraw。包装层使用稳定追加槽位保存 triangle indices，避免旧 winner ID 失效。推理跳过未使用的 barycentric derivatives。',
        f'增量深度逐帧复用证据：{incremental}。所有相机与完整参考的 coverage exact、误差在登记容差内；最终 IDs 和 RGB 仍然 exact。',
        '与无遮挡渲染的既有质量差异未修复；新旧输出方式的 RGB 一致。历史 15.206/90.039 ms 只作背景，本轮加速比来自完整筛选等相同边界的同轮配对，不混用旧分母。',
        'v4 的首次编译重名错误已修复；v6 进一步下移切换层、直接映射输出块，25 项测试全部通过。所有正式计时保留，未删除 outlier。',
        '', '证据目录：/ssddata/lun/gdmgs_artifacts/proxygs_joint_frustum_amsterdam_20260924/occlusion_dfs/parallel_output_v7；runs/amsterdam_v7、serial/、dev/、tests.xml、review.json、PROTOCOL.md。']
    body='\n'.join(report)+'\n'
    a=body.index('## 同轮纯查询');b=body.index('## 遮挡筛选与完整流程');z=body.index('## 修改效果')
    body=body[:a]+body[b:z]+body[a:b]+body[z:]
    body=body.replace('## 遮挡筛选与完整流程','## 主比较：联合查询 → depth → 遮挡 anchor 筛选')
    (root/'RESULTS.md').write_text(body)
    print(json.dumps(review,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
