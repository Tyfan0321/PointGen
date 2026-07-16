# 3DMatch / 3DLoMatch 数据增强开发规划

本文基于 `docs/3dmatch_lomatch_data_augmentation_analysis.md`，目标是把 LoMatch 性能差距定位出的分布缺口转化为可实现、可评估、可回滚的开发任务。

## 目标

主要目标：

1. 增强训练分布对 3DLoMatch 的长基线、大旋转、局部可见变化的覆盖。
2. 每类增强独立开关，支持 ablation。
3. 保持现有训练/评估路径兼容，默认行为尽量可控。
4. 输出能直接比较 3DMatch / 3DLoMatch 全量指标的实验脚本和记录。

非目标：

1. 不改模型结构。
2. 不改 Flow Matching sign convention。
3. 不重写 Sonata/KPConv processor。
4. 不默认替换用户本地 dataset/checkpoint 路径。

## 开发阶段

### Phase 0: 基线复验

目的：确认当前样本数为 16 的观察能在全量评估上成立。

任务：

1. 使用 `scripts/eval_grid_3dmatch.sh` 跑全量 3DMatch / 3DLoMatch。
2. 至少保留 `epoch-119`、`epoch-124`、`epoch-134`，steps 取 `5` 和 `10`。
3. 额外跑一组 `eval.use_overlap_metrics=true`。
4. 汇总 `metrics.jsonl` 到表格，确认 LoMatch 的主要失败是否仍存在。

验收：

1. 有全量 `RR/RRE/RTE/dist_error`。
2. 有 overlap on/off 的对照。
3. 明确当前 baseline checkpoint。

建议命令：

```bash
RUN_DIR=outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12 \
CKPTS="epoch-119 epoch-124 epoch-134" \
STEPS="5 10" \
NUM_GEN_SAMPLES=null \
USE_OVERLAP_METRICS=false \
bash scripts/eval_grid_3dmatch.sh
```

```bash
RUN_DIR=outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12 \
CKPTS="epoch-124" \
STEPS="5" \
NUM_GEN_SAMPLES=null \
USE_OVERLAP_METRICS=true \
bash scripts/eval_grid_3dmatch.sh
```

### Phase 1: 增强配置与 Pose Augmentation

目的：先补最明确的分布缺口，即 source-side yaw 与大旋转长尾。

改动文件：

1. `config/config_v0.yaml`
2. `src/data/dataset_factory.py`
3. `src/data/threedmatch_data.py`
4. 新增 `src/data/augmentation_3dmatch.py`

建议配置：

```yaml
data:
  augment: 1.0
  augmentation:
    enabled: true
    pose:
      enabled: true
      source_prob: 1.0
      global_prob: 0.0
      modes:
        - name: small
          prob: 0.5
          roll: 30
          pitch: 30
          yaw: 30
        - name: medium
          prob: 0.3
          roll: 75
          pitch: 75
          yaw: 90
        - name: hard
          prob: 0.2
          roll: 90
          pitch: 90
          yaw: 180
```

实现要点：

1. 保留 `data.augment` 作为总开关/概率，兼容旧配置。
2. 新增 `ThreeDMatchAugmentor`，由 `IndoorDataset` 初始化。
3. 将当前 `sample_random_rotation()` 替换为配置驱动的 rotation sampler。
4. source-side augmentation 的变换约定必须保持当前代码逻辑：

```python
src_points = src_points @ A
Tr[:3, :3] = Tr[:3, :3] @ A
```

其中 `Tr` 仍满足：

```python
ref_points ~= src_points @ Tr[:3, :3].T + Tr[:3, 3]
```

验收：

1. 默认旧配置仍能训练。
2. 新配置下 yaw 不再固定为 0。
3. `python -m py_compile src/data/threedmatch_data.py src/data/dataset_factory.py src/data/augmentation_3dmatch.py` 通过。
4. 对一个样本做 transform consistency 检查，增强前后 `Tr` 映射误差在浮点误差范围内。

建议提交：

```text
feat: add configurable 3dmatch pose augmentation
```

### Phase 2: Global Rotation Augmentation

目的：让模型减少对训练集全局房间朝向的依赖。

建议配置：

```yaml
data:
  augmentation:
    pose:
      global_prob: 0.5
      global:
        roll: 180
        pitch: 180
        yaw: 180
```

实现要点：

如果同时对 ref/src 做右乘旋转：

```python
ref_points = ref_points @ G
src_points = src_points @ G
```

则需要更新：

```python
R = Tr[:3, :3]
t = Tr[:3, 3]
Tr[:3, :3] = G.T @ R @ G
Tr[:3, 3] = t @ G
```

原因：当前代码使用 row-vector 点坐标，但 `Tr` 存储的是列向量形式的 `R, t`，应用时用 `points @ R.T + t`。

验收：

1. 随机生成点、随机 `Tr`、随机 `G`，验证增强前后映射一致。
2. 打开 `global_prob=1.0` 后训练数据能正常进入 processor。

建议提交：

```text
feat: add global rotation augmentation for 3dmatch
```

### Phase 3: Partial View / Crop Augmentation

目的：模拟 LoMatch 长基线下可见区域形态变化。

建议配置：

```yaml
data:
  augmentation:
    crop:
      enabled: true
      prob: 0.5
      min_points: 2048
      modes:
        sphere:
          prob: 0.4
          keep_ratio: [0.5, 0.9]
        halfspace:
          prob: 0.4
          keep_ratio: [0.5, 0.9]
        block_dropout:
          prob: 0.2
          keep_ratio: [0.6, 0.95]
```

实现要点：

1. crop 可独立作用于 `ref_points` 和 `src_points`。
2. crop 不改变 `Tr`。
3. crop 后必须重新计算 `ref_overlap` 和 `src_overlap`。
4. 如果 crop 后点数低于 `min_points`，跳过该次 crop，避免 processor 输入过小。
5. crop 先做在原始 voxelized 点上，之后再交给 Sonata/KPConv processor。

推荐函数：

```python
random_sphere_crop(points, keep_ratio, min_points)
random_halfspace_crop(points, keep_ratio, min_points)
random_block_dropout(points, keep_ratio, min_points)
```

验收：

1. crop 后点数不低于 `min_points`。
2. overlap mask 长度与增强后点数一致。
3. 开启 crop 后 `IndoorDataset.__getitem__` 返回的 key 和 tensor shape 与现有 contract 一致。

建议提交：

```text
feat: add partial-view crop augmentation for 3dmatch
```

### Phase 4: Density / Noise Augmentation

目的：补充点密度、传感器噪声和局部缺失的鲁棒性。

建议配置：

```yaml
data:
  augmentation:
    density:
      enabled: true
      dropout_prob: 0.3
      dropout_ratio: [0.1, 0.3]
      jitter_prob: 0.3
      jitter_sigma: [0.002, 0.006]
      jitter_clip: 0.02
```

实现要点：

1. jitter 不改变 `Tr`。
2. dropout 不改变 `Tr`，但需要重新计算 overlap。
3. jitter 后是否重算 overlap建议作为配置项；默认重算更严谨，但耗时更高。
4. 不建议第一轮引入 random voxel size，因为当前读取函数直接用 `voxel_size` 读点，随机化会改变读取成本和点数分布，先保留为后续实验。

验收：

1. 点数变化符合配置。
2. jitter 幅度小于 voxel size 量级。
3. 无 NaN/Inf 坐标。

建议提交：

```text
feat: add density and jitter augmentation for 3dmatch
```

### Phase 5: Hard Pair Re-sampling

目的：提高训练中 hard pair 出现概率，使训练分布更接近 LoMatch。

建议先用 dataset-level index expansion，不直接使用 `WeightedRandomSampler`。原因是当前训练 DataLoader 用 `shuffle=True`，并且需要与 Accelerate/DDP 保持简单兼容。

建议配置：

```yaml
data:
  hard_sampler:
    enabled: true
    max_repeat: 4
    gap_threshold: 15
    trans_threshold: 1.5
    rot_threshold: 60
    weights:
      gap: 1
      trans: 1
      rot: 2
```

实现要点：

1. 在 `IndoorDataset.make_dataset()` 中保存 `ref_id/src_id/seq` 和 hard stats。
2. `gap = abs(src_id - ref_id)`。
3. `trans_norm = np.linalg.norm(Tr[:3, 3])`。
4. `rot_deg = arccos((trace(R)-1)/2)`。
5. 构造 `sample_indices`，`__len__` 返回扩展后的长度，`__getitem__` 映射回原始样本。
6. 保留原 `self.dataset`，方便测试集和调试。

验收：

1. hard_sampler 关闭时 `len(dataset)` 不变。
2. hard_sampler 开启时 hard pair 占比上升。
3. DataLoader 仍使用 `shuffle=True`。
4. 训练日志记录 hard_sampler 开启后的有效样本数。

建议提交：

```text
feat: add hard pair resampling for 3dmatch
```

## 实验矩阵

建议按以下顺序训练，避免一次性叠加过多变量：

| 实验 | pose | global | crop | density | hard sampler |
| --- | --- | --- | --- | --- | --- |
| baseline | off/current | off | off | off | off |
| A | on | off | off | off | off |
| B | on | on | off | off | off |
| C | on | on | on | off | off |
| D | on | on | on | on | off |
| E | on | on | on | on | on |

每个实验至少保存并评估：

1. best validation loss checkpoint。
2. 中期 checkpoint。
3. 最后 checkpoint。

核心指标：

1. 3DMatch RR/RRE/RTE/dist_error。
2. 3DLoMatch RR/RRE/RTE/dist_error。
3. `eval.use_overlap_metrics=true` 下的 `RR_ov/RR_w/overlap_ratio`。
4. 按 `rot_deg/trans_norm/gap` 分 bucket 的 RR。

## 需要补的分析工具

建议新增：

```text
scripts/analyze_3dmatch_pairs.py
scripts/summarize_eval_metrics.py
```

`analyze_3dmatch_pairs.py`：

1. 统计 train/val/3DMatch/3DLoMatch 的 gap/trans/rot。
2. 输出 hard_sampler 阈值命中率。
3. 保存 CSV/JSON 摘要。

`summarize_eval_metrics.py`：

1. 读取 eval 目录下的 `metrics.jsonl`。
2. 按 checkpoint、steps、overlap on/off 汇总。
3. 输出 Markdown 表格，方便写实验记录。

这两个脚本不影响训练路径，可以单独提交。

## 验证计划

轻量验证：

```bash
python -m py_compile src/data/threedmatch_data.py src/data/dataset_factory.py src/data/augmentation_3dmatch.py
```

数据一致性验证：

```bash
python scripts/check_3dmatch_augmentation.py data.augmentation.pose.enabled=true data.augmentation.crop.enabled=true
```

该脚本建议检查：

1. 返回 keys 完整。
2. `ref_points/src_points` shape 合法。
3. `ref_overlap/src_overlap` 长度匹配。
4. source-side/global rotation 后 `Tr` 映射一致。
5. crop/dropout 后没有空点云。

训练 smoke test：

```bash
python train.py num_train_epochs=1 val_epochs=1 ckpt_epochs=1 num_workers=0 num_gen_samples=2
```

如当前环境缺 CUDA/Open3D/torch_scatter，则记录无法运行原因，只做 py_compile 和脚本级检查。

## 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 增强过强导致 3DMatch 下降 | 分阶段打开，保留 small/medium/hard 概率配置 |
| crop 后点数过少 | `min_points` 保护，失败则跳过 crop |
| overlap mask 与点云不一致 | 所有改变点集合的增强后统一重算 overlap |
| hard sampler 破坏 easy pair 覆盖 | 设置 `max_repeat`，保留原始样本全量覆盖 |
| DDP sampler 复杂度增加 | 第一版使用 dataset-level index expansion |
| 变换公式出错 | 增加 transform consistency 单元/脚本检查 |

## 推荐提交拆分

1. `feat: add configurable 3dmatch pose augmentation`
2. `feat: add global rotation augmentation for 3dmatch`
3. `feat: add partial-view crop augmentation for 3dmatch`
4. `feat: add density and jitter augmentation for 3dmatch`
5. `feat: add hard pair resampling for 3dmatch`
6. `test: add 3dmatch augmentation consistency checks`
7. `scripts: add 3dmatch pair and eval summary utilities`

## 推荐优先级

第一轮只做：

1. Phase 0 baseline full eval。
2. Phase 1 source-side pose augmentation。
3. Phase 2 global rotation augmentation。
4. `check_3dmatch_augmentation.py` 一致性检查。

原因：这部分直接对应 LoMatch 的大旋转分布缺口，代码风险最低，且能最快判断方向是否有效。

第二轮再做：

1. crop augmentation。
2. hard pair re-sampling。

原因：这两项更可能提升 LoMatch，但也更容易影响点数分布和训练稳定性，需要在 pose augmentation 的收益确认后再叠加。
