# Control-Aligned Predictive Gradient Routing：实验数据优先汇报

## 1. 当前最重要的实验结论

Control-Aligned Predictive Gradient Routing（以下简称 aligned）在 gridfire seed 0 上同时表现出三种变化：

1. 下游规划明显优于直接对照 pred03；
2. latent pair geometry 从 watermark 主导翻转为 physical content 主导；
3. frozen probe 显示 block angle、block position 和 pusher position 信息基本保留，而 watermark RGB 的线性可解码性显著下降。

因此，aligned 当前最值得关注的结果不是单个 `0.88` 峰值，而是以下联合证据：

> 在物理状态可解码性基本不变的情况下，watermark 的表示显著性降低，content/tag 几何发生逆转，并伴随规划改善。

这支持 Enigma 类失效是一个 representational allocation / geometric prominence 问题，而不只是任务信息是否存在的问题。

## 2. Gridfire 规划成功率

评估协议：seed 0；每 2k step 评估一次；每个 checkpoint 使用 50 个 episode。因此单次 `0.02` 的变化对应一个 episode。

### 2.1 Aligned 与 pred03

| 方法 | 最佳 SR | 最终 SR | 最后 10 次均值 |
|---|---:|---:|---:|
| pred03 | 0.72 | 0.64 | 0.624 |
| aligned | **0.88** | **0.80** | **0.774** |

Aligned 相比 pred03：

- 最佳 SR：`+0.16`；
- 最终 SR：`+0.16`；
- 最后 10 次均值：`+0.150`。

Aligned 在 96k 和 118k 附近达到 `0.88`。但 `0.88` 是 50-episode 单次峰值，当前不能称为稳定成功率。最稳妥的主数字是 aligned 后期均值 `0.774`，对比 pred03 的 `0.624`。

### 2.2 其他 gridfire 完整训练参照

| 方法 | 最佳 SR | 最终 SR | 后期概况 |
|---|---:|---:|---:|
| conditional | 0.64 | 0.54 | recent mean 约 0.56 |
| masked reachability | 0.74 | 0.66 | recent mean 约 0.66 |
| pred03 | 0.72 | 0.64 | last-10 0.624 |
| aligned | **0.88** | **0.80** | last-10 **0.774** |

这些结果均主要是 seed 0，尚不能替代多种子统计。

### 2.3 旧机 IDM/MSID 成功率参照

旧机结果与 gridfire 结果不是完全相同的运行环境和统计协议，因此不能将小数点后的差异解释成严格胜负。它们目前用于判断性能带和训练曲线形态。

Seed 0 摘要：

| 方法 | Last-10 SR | Peak SR | Final SR |
|---|---:|---:|---:|
| focus IDM | **0.82** | 0.88 | 0.78 |
| focus MSID | 0.81 | 0.88 | **0.82** |
| aligned（gridfire） | 0.774 | 0.88 | 0.80 |

旧机 seed 0 每 10k step 抽样曲线：

| Step | Focus IDM | Focus MSID |
|---:|---:|---:|
| 10k | 0.14 | 0.18 |
| 20k | 0.50 | 0.44 |
| 30k | 0.58 | 0.72 |
| 40k | 0.66 | 0.66 |
| 50k | 0.62 | 0.72 |
| 60k | 0.68 | 0.74 |
| 70k | 0.70 | 0.84 |
| 80k | 0.74 | 0.80 |
| 90k | 0.78 | 0.76 |
| 100k | 0.72 | 0.84 |
| 110k | 0.78 | 0.84 |
| 120k | 0.86 | 0.78 |
| 130k | 0.82 | 0.86 |
| 138k | 0.78 | 0.82 |

两条旧机曲线均在前 20k 快速爬升，30–60k 进入 `0.6–0.7`，后期主要在 `0.70–0.88` 区间波动，并非依赖单个尖峰。MSID 前期整体稍快，最终与 IDM 处于相同水平。

Seed 1 摘要：

- focus IDM：last-10 `0.77`，peak `0.88`，final `0.78`；
- focus MSID：截至 138k 的后期均值约 `0.77`；原始 JSONL 末尾存在两条疑似重跑开头的 `0.04/0.00`，计算 last-10 时必须排除。

因此，目前关于 SR 的准确结论是：

> Aligned seed 0 已进入旧机 IDM/MSID 的约 `0.8` 性能带，并显著超过 gridfire 上的 pred03、masked reachability 和 conditional；但由于协议差异和 aligned 尚无多种子结果，不能宣称 aligned 已在 SR 上稳定超过 IDM/MSID。

Aligned 当前相对 IDM/MSID 更有力的证据是表示质量：历史 IDM/MSID 的 angle probe 较弱且 tag probe 仍高，而 aligned 同时表现出高 angle、低 tag 和 content-dominant pair geometry。该优势仍需同协议 checkpoint 重测确认。

## 3. Content/tag pair geometry

Pair metric 比较两类输入对：

- same-content：物理画面相同、watermark 不同；
- same-tag：watermark 相同、物理内容不同。

如果 same-tag similarity 高于 same-content，说明 watermark 更可能主导 latent distance。反之则说明物理内容更占主导。

Aligned 最终 backbone：

- same-content cosine：约 `0.807`；
- same-tag cosine：约 `0.176`；
- content/tag margin：`+0.631`。

完整训练参照：

| 方法 | Same-content cosine | Same-tag cosine | Content/tag margin |
|---|---:|---:|---:|
| conditional | 0.327 | 0.639 | **−0.312** |
| masked reachability | 0.470 | 0.517 | **−0.047** |
| aligned | **0.807** | **0.176** | **+0.631** |

Aligned 不只是减小了 same-tag similarity，而是将表示从 tag-dominant 明显翻转为 content-dominant。

## 4. Frozen representation probe

### 4.1 Gridfire 统一协议

- dataset：`pusht_expert_train.h5`；
- 512 clips，每个 clip 4 帧；
- frameskip 5；
- 按 sampled clip 做 80/20 train/test split；
- frozen encoder；
- StandardScaler + Ridge，`alpha=1`；
- seed 73。

`tag RGB R²` 测量从 latent 中线性恢复 watermark RGB 的能力。Watermark 与 PushT 物理控制无关，因此只有在物理 probe 和规划不受损时，tag R² 下降才是正面结果。

### 4.2 Projection-space probe

Projection 是 planner 使用的主要 latent 空间。

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| masked reachability final | 0.056 | 0.869 | 0.999 | 0.928 |
| pred03 near 54k | 0.745 | 0.937 | 0.997 | 0.916 |
| pred03 final | **0.800** | **0.952** | 0.997 | 0.889 |
| aligned near 96k | 0.793 | 0.929 | 0.996 | 0.522 |
| aligned final | 0.795 | 0.926 | 0.996 | **0.502** |

Aligned final 相比 pred03 final：

- block angle：`0.800 → 0.795`，变化 `−0.005`；
- block XY：`0.952 → 0.926`，变化 `−0.026`；
- pusher XY：`0.997 → 0.996`，基本不变；
- tag RGB：`0.889 → 0.502`，变化 `−0.387`。

因此 aligned 在 projection 中保留了很强的物理状态信息，同时大幅降低 watermark 线性显著性。

### 4.3 Backbone-space probe

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| masked reachability final | 0.202 | 0.884 | 0.998 | 0.960 |
| pred03 near 54k | 0.852 | 0.968 | 0.997 | 0.934 |
| pred03 final | 0.862 | **0.971** | 0.997 | 0.890 |
| aligned near 96k | 0.867 | 0.966 | 0.996 | 0.610 |
| aligned final | **0.870** | 0.963 | 0.995 | **0.566** |

Aligned final 相比 pred03 final：

- block angle：`0.862 → 0.870`，变化 `+0.008`；
- block XY：`0.971 → 0.963`，变化 `−0.008`；
- pusher XY：始终接近饱和；
- tag RGB：`0.890 → 0.566`，变化 `−0.324`。

### 4.4 Probe 的关键含义

Pred03 和 aligned 的 block-angle R² 几乎相同，但 aligned 的规划与 pair geometry 明显更好。因此 aligned 的收益不能简单归因于“重新加入了缺失的 angle 信息”。

更符合数据的解释是：

> Pred03 已经包含可线性解码的物理信息，但 watermark 仍具有很强的表示显著性；aligned 保留物理信息，同时降低 watermark 对表示及距离结构的相对支配。

即：

\[
\text{decodability}
\neq
\text{correct allocation}
\neq
\text{planner influence}.
\]

## 5. 旧机 IDM/MSID 历史 probe

旧机 `results/probe_linear.json` 使用 6000 帧、Ridge `alpha=10`、按 episode 切分；gridfire 使用 512 clips、Ridge `alpha=1`、按 clip 切分。以下数据只能作为跨协议量级参照，不能作为严格排名。

旧机 projection：

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| focus IDM | 0.24 | 0.90 | 0.98 | 0.87 |
| focus MSID | 0.43 | 0.92 | 0.98 | 0.91 |
| LDAD, lambda=1 | 0.15 | 0.89 | 0.98 | 0.90 |
| LDAD, lambda=10 | 0.21 | 0.91 | 0.99 | 0.86 |
| clean baseline | 0.92 | 0.98 | 0.95 | — |
| watermarked baseline | 0.06 | 0.81 | 0.53 | 0.95 |
| aligned，gridfire 协议 | **0.795** | 0.926 | 0.996 | **0.502** |

旧机 backbone：

- focus IDM：angle `0.46`，block XY `0.93`，tag `0.92`；
- focus MSID：angle `0.56`，block XY `0.94`，tag `0.90`；
- aligned（gridfire 协议）：angle `0.870`，block XY `0.963`，tag `0.566`。

跨协议趋势显示，aligned 的 angle decodability 进入接近 clean baseline 的量级，同时 tag decodability 显著低于 IDM/MSID。但必须用同一 probe 重测 IDM/MSID checkpoint，才能形成严格结论。

## 6. 梯度诊断

Aligned 完整训练末期：

- prediction/control gradient cosine：约 `0.0715`；
- orthogonal gate：约 `0.0752`；
- retained prediction-gradient norm：约 `0.1057`；
- reversed/negative fraction：约 `0.125`。

Retention 不是人工预设的 schedule，而是由每个样本、每个 batch 中 prediction/control 梯度关系动态产生。训练阶段均值大致为：

- 0–20k：`0.160`；
- 20–60k：`0.139`；
- 60–90k：`0.121`；
- 90–108k：`0.110`；
- 最终：约 `0.106`。

它在 warm-up 后总体下降，但不是严格单调。随着训练进行，prediction/control cosine 从约 `0.11` 降至约 `0.07`，路由自然收紧。

最重要的诊断是：平均 cosine 仍为正，负冲突比例并不高。Aligned 主要处理的不是少量负冲突，而是大量 near-orthogonal prediction updates。

## 7. 10k 机制消融

所有数字均为 seed 0、每次 50 episode 的短程结果。

| 方法 | 10k SR | 10k peak | Retention | Content/tag margin |
|---|---:|---:|---:|---:|
| aligned | 0.36 | 0.36 | 0.161 | 0.754 |
| parallel-only | 0.24 | 0.34 | 0.085 | **0.949** |
| shuffled guide | 0.18 | 0.18 | 0.031 | 0.882 |
| norm-matched scalar | **0.42** | **0.42** | 0.153 | 0.682 |
| retention-matched shuffled | 0.28 | 0.30 | 0.149 | 0.716 |

主要观察：

1. Parallel-only 获得最强的 nuisance invariance，却没有获得最好规划，说明“删除越多越好”不成立；正交 prediction 中仍包含 control head 没有完整描述的有用 world structure。
2. Norm-matched scalar 在 10k 达到 `0.42`，暂时高于 aligned 的 `0.36`。差异仅为 3/50 个 episode，不能定论，但说明 aligned 的收益可能主要来自 control-conditioned dynamic budgeting，而不完全来自方向旋转。
3. Retention-matched shuffled 低于 aligned，说明正确的逐样本 control correspondence 可能仍有价值。
4. 必须完成 norm-matched scalar 全程训练，才能区分方向路由与动态降权。

## 8. 从数据导出的机制判断

当前数据支持：

1. 任务物理信息“存在”不是充分条件。Pred03 已经能很好解码 angle，却没有 aligned 的几何和规划表现。
2. Watermark 不一定需要与 control gradient 负冲突就能造成伤害。大量 near-orthogonal prediction updates 可以长期改变表示。
3. Aligned 的结果更像 selective reallocation，而不是整体 collapse：物理 R² 保留、tag R² 下降、content geometry 上升、planning 改善。
4. 极端删除 orthogonal prediction 并非最优，world model 仍需要 control auxiliary 未完整覆盖的结构。
5. 当前尚不能确定最终最佳实现是方向性 routing，还是更简单的 control-conditioned scalar admission。

## 9. Method：Aligned 如何实现

设 prediction loss 和 control loss 对共享 latent 的逐样本梯度为：

\[
g_p=\nabla_z L_{\mathrm{pred}},
\qquad
g_c=\nabla_z L_{\mathrm{control}}.
\]

分解 prediction gradient：

\[
g_p=\alpha g_c+g_\perp,
\qquad
\alpha=\frac{g_p^\top g_c}{\|g_c\|^2}.
\]

进入 encoder 的 prediction gradient 被替换为：

\[
\widetilde g_p
=
[\alpha]_+g_c
+
[\cos(g_p,g_c)]_+g_\perp.
\]

其中：

- 负向平行分量被删除；
- 正向平行分量完整保留；
- 近正交分量按非负 cosine 衰减；
- predictor 仍接收完整 prediction gradient；
- control head 仍接收完整 control gradient；
- 只有 prediction 对 encoder 的塑造被管理。

局部表示空间中：

\[
g_c^\top\widetilde g_p=[g_c^\top g_p]_+\ge 0.
\]

该性质不构成 Adam 参数空间中的全局性能保证。

## 10. 与 IDM/MSID 的区别

IDM/MSID 主要要求动作可以从 latent transition 中恢复：

\[
G(z_t,z_{t+H})\rightarrow a_{t:t+H-1}.
\]

它回答“动作信息是否进入表示”。Aligned 进一步管理：

> 即使动作和物理状态已经可解码，prediction loss 的哪些更新仍有资格改变整个规划空间？

因此，aligned 的目标不是替代动作监督，而是利用动作监督提供的控制方向，管理 forward prediction 对 encoder 的长期影响。

## 11. 新颖性边界

Gradient cosine、辅助任务门控和正交分解有既有工作，不能声称为通用数学首创。最相关的工作包括 Du et al. 2018 的 gradient-similarity auxiliary gating、Dery et al. 2021 的 helpful/harmful/neutral auxiliary update decomposition、PCGrad、Bloop，以及使用预设正交 latent 子空间的 SD-JEPA。

当前候选贡献是：

> 在 JEPA predictable-nuisance failure 中识别 near-orthogonal prediction updates 这一长期表示塑造通道；在逐样本 latent interface 上对 predictor 与 encoder 实施非对称 prediction-gradient admission；并通过 planning、pair geometry 与 frozen physical/nuisance probes 证明它改变的是表示因素的相对几何显著性。

## 12. 当前不能过度主张的内容

1. Aligned 规划结果仍是 seed 0；
2. `0.88` 是单次峰值，不是稳定成功率；
3. 旧机 IDM/MSID 与 gridfire probe 协议不同；
4. 约 10% gradient retention 不能排除强降权解释；
5. Frozen probe 证明 decodability，不证明 planner causal use；
6. 尚未完成 norm-matched scalar full run；
7. 尚未完成 future-latent intervention。

## 13. 下一步实验优先级

1. 完成 norm-matched scalar seed 0 全程训练；
2. 运行 aligned seed 1、2；
3. 在同一协议下重测 IDM/MSID checkpoint；
4. 增加 Du-style cosine-gated prediction baseline；
5. 进行 future-latent permutation intervention，测量 CEM cost rank、elite overlap、动作变化、真实物理 progress 和 success rate。

## 14. 当前最稳妥的汇报结论

> Aligned routing 在 seed 0 上将 planning last-10 success 从 pred03 的 0.624 提高到 0.774，并进入旧机 IDM/MSID 的约 0.8 性能带，同时把 content/tag geometry margin 翻转到 +0.631。Frozen probes 显示 aligned 与 pred03 具有近乎相同的 block-angle 和 block-position decodability，但 projection-space watermark RGB R² 从 0.889 降至 0.502。因此，当前结果更支持“重新分配任务因素与 nuisance 的几何显著性”，而不是“简单恢复缺失任务信息”。该现象直接对应 Enigma 暴露的 predictability–utility mismatch；但 aligned 尚未在同协议、多种子 SR 上证明超过 IDM/MSID，仍需要多种子、同协议 IDM、norm-matched scalar 和 planner intervention 完成验证。
