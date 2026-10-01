"""Build a human-readable final decision from persisted independent reviews."""
import argparse,json,statistics
from pathlib import Path

LABELS={'pair2_control':'两帧原方案','pair2_fast':'命中 batch 直接交付','pair2_arena':'保留 union arena',
        'pair2_prefix':'当前前段免拷贝 + 命中直接交付','epoch3':'三帧刷新','epoch4':'四帧刷新',
        'pair2_fp16':'FP16 缓存','pair2_c50':'50% 容量','pair2_c75':'75% 容量'}

def load(p): return json.loads(p.read_text())

def main(root):
    a=load(root/'review/final_ablation_v1.json')
    c=load(root/'review/final_confirmation_v1.json')
    if c['status']!='pass' or c['failures']:
        raise ValueError('final confirmation not passing')
    selected=c['selected_policy'];v=c['policies'][selected]
    q=v['quality']; h=v['cache_metrics']; depth=v['depth']
    lines=['# Step 7ABC Cache 最终方案与消融结果','',
           f'最终选择：**{selected} — {LABELS[selected]}**。8/8 场景、256/256 帧、74 项测试通过，独立审查无失败。',
           '','## 方法与合同变化','',
           '旧 Step 7B unconditional residency 的 852/904 失败结论保持不变。经用户授权，ADR 0014/0015 引入一帧期限、当前/下一帧联合解码及短期空结果记录。',
           'Step 7A 的完整 bundle、LoD/owner/offset 对齐保持；Step 7B 冻结质量/年龄/精度；Step 7C 负责 next-demand 准入、整 bundle 淘汰、packed generation 和原子 publication。',
           '默认只复用一帧。刷新帧解码当前请求加下一帧新增请求；当前帧直接使用连续前段，下一帧直接消费已封装 batch。未覆盖请求、过期、容量不足按当前 pose 一次 batch 补解码。',
           '零行结果仅存短期 negative descriptor，不占 Gaussian rows；正、负 descriptor 均受固定 anchor universe 上限约束。场景、模型或 anchor table 改变时新建 cache。',
           '旧 R0–R4/通用 LRU/可变 free-segment allocator 设计由本次明确版本化的两帧 next-demand 策略和容量/布局消融替代；不宣称已实现那些旧计划。',
           '','## 最终五轮确认（相同 fresh baseline）','',
           '| 方案 | 加速比 | decoder 调用 | RGB/深度门槛 |',
           '|---|---:|---:|---|']
    for name,val in c['policies'].items():
        lines.append(f"| {LABELS[name]} | {val['speedup']:.4f}× | {val['decoder_calls']}/256 | {'通过' if val['quality_pass'] else '失败'} |")
    base=c['policies']['pair2_control']['cache_ms']
    lines += ['',f"选定方案阶段时间 {v['cache_ms']:.2f} ms，fresh 为 {v['fresh_ms']:.2f} ms；相对同轮两帧原方案再减少 {100*(1-v['cache_ms']/base):.2f}% 时间。",
              f"解码 anchor 总数 {v['baseline_decoded_anchors']:,} → {v['decoded_anchors']:,}，含预取，减少 {100*v['decoded_anchor_reduction']:.2f}%。",
              '','## 质量、命中、深度与内存','',
              '| 指标 | 结果 |','|---|---:|',
              f"| GT PSNR 损失（全帧平均 / 最差帧） | {q['psnr']['mean_loss']:.6f} / {q['psnr']['worst_loss']:.6f} dB |",
              f"| GT SSIM 损失（平均 / 最差） | {q['ssim']['mean_loss']:.6f} / {q['ssim']['worst_loss']:.6f} |",
              f"| GT LPIPS 增加（平均 / 最差） | {q['lpips']['mean_loss']:.6f} / {q['lpips']['worst_loss']:.6f} |",
              f"| Anchor 命中率（含刷新帧） | {100*h['anchor_hit_rate']:.3f}% |",
              f"| 非空 / 空结果命中占全部请求 | {100*h['positive_hit_rate']:.3f}% / {100*h['empty_hit_rate']:.3f}% |",
              f"| 非空输出请求中的非空命中率 | {100*h['positive_hit_rate_among_nonempty_output_requests']:.3f}% |",
              f"| 复用帧命中率 | {100*h['reuse_frame_hit_rate']:.3f}% |",
              f"| 输出 Gaussian rows 命中占比 | {100*h['row_hit_fraction']:.3f}% |",
              f"| 新增预取 anchor 实际首次使用率 | {100*h['extra_prefetch_utilization']:.3f}% |",
              f"| 整 bundle 淘汰率 | {100*h['whole_bundle_eviction_rate']:.3f}% |",
              f"| 平均 / 最差帧相对深度 MAE | {100*depth['relative_mae_mean']:.5f}% / {100*depth['relative_mae_worst']:.5f}% |",
              f"| 前景 mask XOR（平均 / 最差） | {100*depth['foreground_xor_mean']:.5f}% / {100*depth['foreground_xor_worst']:.5f}% |",
              f"| 最大驻留 rows / 配置容量 | {v['max_resident_rows']:,} / 6,826,846 |",
              f"| 最大 generation 驻留内存 | {v['max_resident_bytes']/1e6:.2f} MB |",
              f"| 最大帧临时内存（包含 decode/render） | {v['max_scratch_bytes']/1e9:.3f} GB |",
              '', '深度为 candidate 对 fresh expected depth 的比较，并非对 GT depth。共享前景 alpha≥0.5；同时报告前景覆盖变化，未掩盖非重叠区域。上述命中率按请求总数加权；质量为完整 256 帧平均，最差帧和逐场景均单独检查。',
              '','## 八候选消融（同轮三次重复）','',
              '| 方案 | 加速比 | 调用数 | 命中率 | PSNR 平均/最差损失 | 资格 |',
              '|---|---:|---:|---:|---:|---|']
    for name,val in a['policies'].items():
        pq=val['quality']['psnr']
        lines.append(f"| {LABELS[name]} | {val['speedup']:.3f}× | {val['decoder_calls']} | {100*val['cache_metrics']['anchor_hit_rate']:.2f}% | {pq['mean_loss']:.3f}/{pq['worst_loss']:.3f} dB | {'通过' if val['qualified'] else '失败'} |")
    lines += ['', '三/四帧提高命中率与速度但不能满足全部场景的画质门槛；FP16 的内存收益伴随不可接受的图像损失。50%/75% 容量会触发淘汰及补解码，可作内存受限配置，但不是默认最快方案。默认使用 float32 和原 p100 row budget。',
              '','## 每场景最终结果','',
              '| 场景 | 加速比 | PSNR 平均损失 | 相对深度 MAE |',
              '|---|---:|---:|---:|']
    for scene,val in v['per_scene'].items():
        lines.append(f"| {scene} | {val['speedup']:.3f}× | {q['psnr']['scene_means'][scene]:.4f} dB | {100*depth['scene_relative_mae'][scene]:.4f}% |")
    reps=[]
    for repeat in range(5):
        fresh=chosen=control=0
        for scene in v['per_scene']:
            perf=load(root/'runs'/scene/'confirmation_v1/performance.json')
            fresh+=perf['fresh'][repeat]['total_ms'];chosen+=perf[selected][repeat]['total_ms']
            control+=perf['pair2_control'][repeat]['total_ms']
        reps.append(dict(fresh_ms=fresh,selected_ms=chosen,control_ms=control,speedup=fresh/chosen,
                         improvement_over_control=control/chosen))
    lines += ['', '## 边界与 Step 8 接口','',
              '这是 available-lookahead cache-stage 结果。计时包含 ID H2D、decode、gather/view、generation 更新和 RGB render；模型/index 一次性加载及 cache 初始化不在逐帧计时中。',
              'Step 6 offline S(t+1) 是显式前置输入。尚未证明 CPU query、online proxy depth、Anchor query 能在刷新帧解码前提供 next demand；不能将上述数值当作端到端 FPS。',
              'Step 8 必须保留 Retained v2：Barcelona 24-worker optimized BVH，其余场景 32-worker exact brute。刷新帧必须同时拿到当前/下一帧 S；因此需要明确 pair warm-up、lookahead deadline、CPU/GPU 依赖和未就绪 fallback，不能使用免费的 oracle demand 作为在线性能。',
              '需要真实连续 wall-clock、stage timeline、exposed CPU/depth/cache time、GPU contention、hit/miss/age、deadline miss 和质量对照；若效果不好，再以瓶颈驱动消融。',
              '','## 证据','',
              '- `review/final_formal_v1.json`：初始两帧完整资格审查。',
              '- `review/final_ablation_v1.json`：八候选完整消融。',
              '- `review/final_confirmation_v1.json`：四候选五轮完整确认。',
              '- `runs/<scene>/<run>/`：逐帧 RGB/depth/cache 指标与逐轮时间。',
              '- `manifests/*source_sha256.json`：独立版本源码绑定；原模型与输入继续 path/size/mtime 绑定。',
              '- `tests/confirmation_contract_tests.txt`：74 tests。',
              '- `docs/adr/0014-...md`、`0015-...md`：方法授权、事前门槛及消融设计。','']
    (root/'review/final_cache_report.md').write_text('\n'.join(lines))
    contract=dict(status='qualified',selected_policy=selected,refresh_period=2,max_source_age=1,
        payload_dtype='float32',capacity_rows=6826846,fast_handoff=True,
        prefix_decode=selected=='pair2_prefix',union_arena=selected=='pair2_arena',
        negative_entries='bounded by anchor universe; expire with source generation',
        lookahead='next selected request set must be ready BEFORE refresh decode',
        missing_or_late_lookahead='fresh current-pose decode; no dropped requests',
        scope=c['scope'],quality=q,depth=depth,cache_metrics=h,
        performance=dict(speedup=v['speedup'],decoder_calls=v['decoder_calls'],
                         decoded_anchor_reduction=v['decoded_anchor_reduction'],paired_repetitions=reps),
        step8='authorized for design/implementation after Notion synchronization; not validated here')
    (root/'review/selected_cache_contract.json').write_text(json.dumps(contract,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
