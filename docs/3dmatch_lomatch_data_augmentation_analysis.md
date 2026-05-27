# 3DMatch / 3DLoMatch 数据增强分析

## 背景

实验目录：

```text
outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12
```

关注 checkpoint：

```text
ckpt/epoch-124
```

当前结构化评估结果位于：

```text
outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12/eval/epoch-124/steps-5_samples-16_scheduler-fm-euler_seed-1/
```

注意：该目录中的 JSON 记录显示 `evaluated_samples=16`，所以以下数值主要用于定位问题和提出增强方向；最终结论建议用 `num_gen_samples=null` 做全量复验。

## 现象总结

epoch-124、5-step FM Euler、seed=1、samples=16 的结果：

| 测试集 | dist_error | RRE | RTE | RR |
| --- | ---: | ---: | ---: | ---: |
| 3DMatch | 0.2668 | 10.0479 | 0.3154 | 0.8750 |
| 3DLoMatch | 1.5630 | 55.2784 | 1.0411 | 0.0625 |

打开 overlap 相关评估后：

| 测试集 | RR | RR_ov | RR_w | overlap_ratio |
| --- | ---: | ---: | ---: | ---: |
| 3DMatch | 0.8750 | 0.8750 | 0.8750 | 0.3514 |
| 3DLoMatch | 0.0625 | 0.0625 | 0.0625 | 0.1720 |

这说明：只在 SVD 阶段使用 overlap 点或 overlap 权重，并不能改善 LoMatch 的失败样本。问题更可能发生在生成点 `pred_points` 本身，即模型没有学到足够鲁棒的长基线、低重叠、强姿态变化条件下的目标点分布。

## 数据分布证据

基于 `data/3dmatch_list` 中的 train/benchmark pair list 统计：

| split | pair 数 | gap median | gap mean | trans median | trans mean | rot median | rot mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 20642 | 8 | 12.66 | 0.854 | 1.067 | 33.70 deg | 42.27 deg |
| 3DMatch | 1623 | 7 | 13.97 | 0.745 | 1.055 | 29.30 deg | 35.16 deg |
| 3DLoMatch | 1781 | 16 | 19.68 | 1.638 | 1.736 | 51.98 deg | 61.35 deg |

LoMatch 相比 train/3DMatch 有更大的 fragment 间隔、更大的平移幅度和更强的相对旋转。进一步看 Euler 角绝对值分布：

| split | roll mean / q90 | pitch mean / q90 | yaw mean / q90 |
| --- | ---: | ---: | ---: |
| train | 22.2 / 52.2 deg | 26.7 / 55.1 deg | 27.3 / 74.1 deg |
| 3DMatch | 15.7 / 34.2 deg | 25.5 / 54.4 deg | 18.8 / 45.0 deg |
| 3DLoMatch | 37.3 / 107.7 deg | 39.4 / 68.5 deg | 47.1 / 145.2 deg |

LoMatch 的 yaw 和 roll 长尾明显更重；其中 yaw 的 q90 达到 145.2 deg。

## 当前增强实现的问题

配置中开启了数据增强：

```yaml
data:
  augment: 1.0
```

但 `src/data/threedmatch_data.py` 中的训练增强非常窄：

```python
if np.random.rand() < self.augment:
    aug_T = np.eye(4, dtype=np.float32)
    aug_T[:3,:3] = self.sample_random_rotation()
    src_points = src_points @ aug_T[:3,:3]
    Tr = Tr @ aug_T
```

`sample_random_rotation()` 只采样 roll/pitch，yaw 固定为 0：

```python
def sample_random_rotation(self, pitch_scale=np.pi/3., roll_scale=np.pi/4.):
    roll = np.random.uniform(-roll_scale, roll_scale)
    pitch = np.random.uniform(-pitch_scale, pitch_scale)
    r = R.from_euler('xyz', [roll, pitch, 0.], degrees=False)
    return r.as_matrix()
```

这会带来几个分布缺口：

1. 没有 source-side yaw 增强，难以覆盖 LoMatch 的大 yaw 长尾。
2. 只改变 source 相对姿态，没有 ref/src 同时旋转的 global orientation 增强。
3. 没有 partial-view crop / block dropout，无法模拟长基线下可见区域形态变化。
4. 没有 density / voxel / sensor noise 增强，点密度退化不够。
5. DataLoader 仍从原 pair list 均匀采样，没有提高 hard pair 的训练频率。

此外 Sonata processor 默认只做 `CenterShift + GridSample + NormalizeColor + ToTensor + Collect`，没有额外几何增强。

## 主要判断

当前 3DMatch 与 3DLoMatch 的巨大差距，更像是训练分布过于偏向 easy / medium pair，模型没有充分学习 LoMatch 的长基线、大旋转和局部可见变化。

overlap 确实更低，但从当前 overlap-only / overlap-weighted 评估看，失败不能只归因于 SVD 阶段是否使用 overlap 点。更关键的是模型生成的 coarse target points 在 LoMatch 条件下已经偏离太多。

## 增强优化方案

### 1. Pose Augmentation：优先级最高

将当前只含 roll/pitch 的 source-side 增强改成混合姿态增强：

| 类型 | 概率 | 建议范围 |
| --- | ---: | --- |
| small | 0.50 | roll/pitch/yaw: +-15 到 +-30 deg |
| medium | 0.30 | roll/pitch/yaw: +-60 到 +-90 deg |
| hard | 0.20 | yaw: +-180 deg，roll/pitch: +-90 deg |

建议使用 curriculum：

1. 前 20-30 epoch：以 small/medium 为主。
2. 中期：逐步提高 hard 比例。
3. 后期：hard 保持 15%-25%，避免 3DMatch 性能明显下降。

### 2. Global Rotation Augmentation：提升全局朝向鲁棒性

增加 ref/src 同时旋转的增强：

```text
ref' = ref @ G
src' = src @ G
Tr' 需要与右乘/左乘约定保持一致
```

作用：

1. 让 encoder feature 和 transformer 不过度依赖训练集中的固定房间朝向。
2. 补充测试场景 global orientation 的分布变化。

实现时要仔细验证 `Tr` 仍然满足 source -> reference 的映射约定。

### 3. Partial View / Crop Augmentation：非常关键

LoMatch 的困难不只是 overlap 数值低，而是可见区域形态更不稳定。建议在 dataset 层加入：

1. random sphere crop
2. random box crop
3. random half-space crop
4. block dropout
5. ref/src 独立 crop，但保留最少点数

增强后必须重新计算：

```text
ref_overlap, src_overlap
```

否则 overlap label 会与增强后的点云不一致。

### 4. Hard Pair Re-sampling：提高训练分布匹配度

基于 pair metadata 预先计算：

```text
gap = abs(src_id - ref_id)
trans_norm = ||t||
rot_deg = angle(R)
```

然后提高 hard pair 的采样概率：

```text
weight = 1 + a * I(gap > threshold)
           + b * I(trans_norm > threshold)
           + c * I(rot_deg > threshold)
```

建议先试：

1. `rot_deg > 60 deg`
2. `trans_norm > 1.5 m`
3. `gap > 15`

保留 easy pair 的原因是防止模型只适配 LoMatch，导致 3DMatch 上的稳定性下降。

### 5. Density / Noise Augmentation：辅助增强

建议作为第二阶段补充，而不是第一优先级：

1. coordinate jitter: `sigma=0.002-0.006 m`
2. random point dropout: `10%-30%`
3. random voxel size: `0.020-0.035`
4. local density dropout

这些增强主要提升鲁棒性，但预计不如 pose/crop/hard sampling 对 LoMatch 的收益直接。

## 推荐实验顺序

### Experiment A: pose_aug

只加 source-side yaw + 大旋转混合增强。

观察：

1. 3DLoMatch RR 是否明显提升。
2. 3DMatch RR 是否保持在合理范围。
3. RRE 是否显著下降。

### Experiment B: pose_aug + crop_aug

在 A 的基础上加入 partial view crop / block dropout。

观察：

1. LoMatch 的 dist_error 是否下降。
2. overlap-weighted 和 full SVD 的差距是否变小。
3. 低 overlap 样本是否开始恢复。

### Experiment C: pose_aug + crop_aug + hard_sampler

在 B 的基础上加入 hard pair 重采样。

观察：

1. LoMatch RR 是否继续提升。
2. 3DMatch 是否出现明显回退。
3. 训练 loss 是否更不稳定，必要时降低 hard ratio 或使用 curriculum。

## 评估建议

后续对比时建议统一使用：

```bash
python eval.py resume=outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12/ckpt/epoch-124 num_gen_samples=null num_inference_steps=5
```

并额外打开 overlap 指标：

```bash
python eval.py resume=outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12/ckpt/epoch-124 num_gen_samples=null num_inference_steps=5 eval.use_overlap_metrics=true
```

建议固定报告：

1. 3DMatch / 3DLoMatch 全量 RR、RRE、RTE、dist_error。
2. overlap_ratio、RR_ov、RR_w。
3. 按 `rot_deg`、`trans_norm`、`gap` 分 bucket 的 RR。
4. 每个增强实验至少保留 3 个 checkpoint 的 eval，避免单 checkpoint 偶然性。

## 最终建议

优先做：

1. source-side yaw + 大旋转混合增强。
2. ref/src 同时旋转的 global augmentation。
3. partial-view crop / block dropout，并重新计算 overlap。
4. hard pair re-sampling 或 hard pair curriculum。

不建议只加普通 jitter/dropout 后直接期待 LoMatch 大幅提升；当前差距的主因更像是长基线和姿态分布缺口，而不是简单的点坐标噪声不足。
