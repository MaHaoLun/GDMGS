# GDM-GS A/B 实现与验收报告

A/B 代码、子代理交叉审查和清理已经交付；整体验收仍受原 MatrixCity checkpoint 的非有限参数阻塞。13 个模型均实际加载成功，其中 12 个数值有效。本次没有把它写成 13/13 可渲染，也没有推进 C–G。

依据：[Notion 实施计划](https://app.notion.com/p/3ceefdb220d0815fa9f9d16c20bc4da4)、[ADR 0007](../../docs/adr/0007-gdmgs-ab-isolation-and-fresh-interface.md) 与 [分工和验收计划](../../docs/gdmgs_ab_implementation_plan.md)。

## 实现结果

- A：建立本地、zxcpu2 隔离副本；共享模型加载和相机枚举；保留旧 CLI/default/cache；增加独立输出、显式设备及路径选择；metrics 改为逐帧严格配对。
- B：固定 PLY row ID；pose-local LoD/FoV；显式选集直接进入 fresh decoder/raster；完整 bundle 的 owner、offset slot、counts、offsets；保留七项 tuple、旧 dtype 和训练梯度；严格形状与 session/camera 校验，提供可选 depth/alpha 返回。
- 单列修复了旧 precompute 的 grid 数量错误、City precompute 配置不一致、未解析的 iteration=-1、短轨迹 FPS、单 anchor mask、appearance override、静默截断及跨 scene 状态残留。
- 新代码在 `GDMGS_Codebase/`；远端对应 `/home/zyl/lun/GDMGS_Codebase`。原 CacheGS、checkpoint 和上下文档案未作源码改写。`train.py` 与指定 clean archive 逐字节一致。没有计算内容哈希或校验和。

## 实测

最终测试为 **110 passed、122 subtests passed、退出码 0，27.40 秒**，包括真实 CUDA 渲染和临时训练。8 条 warning 是原 torch.load 默认行为的 FutureWarning。[完整测试输出](../../output/gdmgs_ab_validation_20260905/final_tests.log)

渲染对照明确限定为 **Tanks and Temples / Truck 的 7 个固定帧 `[0,1,2,3,4,5,84]`**，保留全部 251 帧的相机清单。fresh、旧 cache、precompute、组合模式、render2 和新 GDM-GS fresh 各 7 帧，原始 tensor 与对应独立 clean baseline 全部 `torch.equal`。render2 使用真实 dispatch 捕获，并检查原 helper 写出的 PNG；没有用第二次普通渲染代替它的输出。precompute/组合在旧基线中有 grid header 故障，修复后的输出分别对照原 fresh/cache。[模式与来源汇总](../../output/gdmgs_ab_validation_20260905/validation_summary.json)

metrics 的逐帧与汇总 JSON 完全一致：PSNR 27.54470253、SSIM 0.91659313、LPIPS 0.12341491。临时训练使用独立模型，真实运行 forward/backward、MLP optimizer 更新、训练统计和 PLY+MLP 保存重载，loss 与基线相同；未声称验证 `train.py --start_checkpoint` 的完整训练断点恢复。

全部 13 个 iteration-40000 checkpoint 实际完成 PLY、MLP、fVDB 加载；完整相机 UID/顺序/尺度/尺寸与原基线一致。直接比较 PLY 位模式和模型行，区分 NaN 不相等与真实行重排；输入 checkpoint 的 size/mtime 保持不变。[完整输入清单](../../output/gdmgs_ab_validation_20260905/new_survey.json)

| 场景 | Anchor 数 | 相机数 | 数值就绪 |
| --- | ---: | ---: | --- |
| amsterdam | 1,127,842 | 161 | 通过 |
| barcelona | 1,333,612 | 160 | 通过 |
| bilbao | 1,023,244 | 129 | 通过 |
| chicago | 1,091,705 | 160 | 通过 |
| hollywood | 955,652 | 125 | 通过 |
| pompidou | 1,115,493 | 161 | 通过 |
| quebec | 1,045,629 | 160 | 通过 |
| rome | 1,303,845 | 158 | 通过 |
| drjohnson | 226,358 | 263 | 通过 |
| playroom | 188,110 | 225 | 通过 |
| small_city | 5,770,852 | 355 | 原 checkpoint 无效 |
| train | 283,287 | 301 | 通过 |
| truck | 263,400 | 251 | 通过 |

MatrixCity 有 **5,770,202 / 5,770,852 行 anchor 坐标为 NaN**，features 和 covariance MLP 也包含 NaN。新 fresh session 明确拒绝该输入，未删除坏行、改写参数或重新训练。另一个独立旧问题是该配置采用 Blender 布局，而原 merged render CLI 不支持此布局；实际模型加载和相机清单不等于该旧 CLI 可运行。

原始 `_scaling=-inf` 是 `exp(-inf)=0` 的零尺度表示，不能当作 NaN。验证保留这种表示，同时拒绝 NaN、正无穷和激活后溢出。

## 跨 pose 预检

每一对都使用目标帧实际的 `S_t`，并与正常 fresh 渲染选集核对。将源帧 0 解码的完整 bundle 用在目标视角，与目标视角 fresh decode 对比：

| 目标帧 | 目标实际选集数 | PSNR | MAE |
| --- | ---: | ---: | ---: |
| 1 | 131,359 | 55.9918 | 0.000884 |
| 84 | 105,197 | 19.6553 | 0.065347 |

大转向时直接复用有明显误差；这些数值只用于后续缓存方案预检，C2 尚未冻结，没有质量 PASS 结论。当前新 pipeline 始终 fresh decode。计时仅为诊断，没有完整轨迹或端到端加速结论。

## 审查、清理和剩余条件

grill-with-docs 子代理先质询并记录决策；CLI、model/FoV、bundle/raster、runtime 分工实现；非作者交叉审查，独立验证代理运行实际输入与 baseline。已关闭输出路径绕回 checkpoint、可变 pose/payload、session 清理、ID 溢出、precompute header 以及验证工具自身的两个归属问题。

删除 17 个无调用的旧原型/历史计划文件，并移除无调用的迁移与哈希 helper；保留原训练、cache、CLI、配置、LICENSE、实验脚本和论文文件。生成的字节码、pytest 缓存另做最终清理。[精确清理记录](cleanup.md)

完整原始图像、tensor、逐帧记录及基线源码保存在 zxcpu2 `/ssddata/lun/gdmgs_artifacts/ab_validation_20260905/`；主要 JSON 与最终测试日志已同步到本地 `output/gdmgs_ab_validation_20260905/`。没有未修的已确认实现/review 问题；要解除完整输入验收阻塞，需要有效的 MatrixCity checkpoint。后续 mesh/BVH、缓存与调度仍按 C–G 独立开展。
