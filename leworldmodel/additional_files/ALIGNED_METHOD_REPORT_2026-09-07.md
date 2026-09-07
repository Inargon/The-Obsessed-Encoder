# Control-Aligned Predictive Gradient Routing

## 1. 一句话结论

我们发现，Enigma 类失效不只是任务信息有没有进入 embedding，而是易预测的 nuisance 是否获得了不成比例的几何支配力。Control-Aligned Predictive Gradient Routing 在保留 block angle、block position 和 pusher position 信息的同时，显著削弱 watermark 对表示的影响，逆转 content/tag 几何，并改善下游规划。

核心思想是：

> Predictor 可以学习所有可预测内容，但 prediction 是否有资格塑造 encoder，应由它与控制学习方向的关系决定。

## 2. Enigma 暴露的问题

在带彩色 watermark 的 PushT 中，watermark 与任务物理状态无关，却非常容易预测。普通 prediction loss 同时承担两个职责：

1. 训练 predictor 拟合未来表示；
2. 决定 encoder 应该强化哪些视觉因素。

这两个职责并不天然一致。一个特征容易预测，不代表它值得主导规划表示。训练可能因此形成反馈循环：watermark 容易预测，encoder 优先编码 watermark，predictor 更依赖 watermark，最终 watermark 主导 latent distance，即使任务信息仍然可以从 embedding 中解码。

这提示我们区分三个概念：

\[
\text{information decodability}
\neq
\text{correct geometric allocation}
\neq
\text{planner influence}.
\]

## 3. 方法

设 prediction loss 对表示的梯度为 \(g_p\)，控制辅助目标的梯度为 \(g_c\)。将 prediction gradient 分解为：

\[
g_p=\alpha g_c+g_\perp,
\qquad
\alpha=\frac{g_p^\top g_c}{\lVert g_c\rVert^2}.
\]

我们对进入 encoder 的 prediction gradient 使用：

\[
\widetilde g_p
=
[\alpha]_+g_c
+
[\cos(g_p,g_c)]_+g_\perp.
\]

其中：

- 与控制梯度冲突的 prediction 分量被删除；
- 与控制近乎正交的 prediction 分量被衰减；
- predictor 仍接收完整 prediction gradient；
- encoder 接收控制梯度与路由后的 prediction gradient。

因此，这不是简单增加一个 auxiliary head，而是直接干预 prediction objective 如何塑造 encoder。

路由满足一个局部性质：

\[
g_c^\top\widetilde g_p=[g_c^\top g_p]_+\ge 0.
\]

但该性质只描述表示空间中的局部梯度变换，不等价于 Adam 参数更新下的全局性能保证。

## 4. Gridfire 规划结果

当前结果来自 seed 0、每个 checkpoint 50 个评估 episode：

- aligned 最佳成功率：**0.88**（96k 和 118k 附近）
- aligned 最后十次评估均值：**0.774**
- aligned 最终成功率：**0.80**
- pred03 最佳成功率：**0.72**
- pred03 最后十次评估均值：**0.624**
- pred03 最终成功率：**0.64**

`0.88` 是单次 50-episode 评估峰值，不能表述为稳定成功率。当前更稳妥的结果是 aligned 后期均值 `0.774`，高于 pred03 的 `0.624`。

## 5. Content/tag 几何逆转

aligned 最终 backbone pair geometry：

- same-content cosine：约 **0.807**
- same-tag cosine：约 **0.176**
- content/tag margin：**+0.631**

历史 gridfire 对照：

- masked reachability margin：**−0.047**
- conditional margin：**−0.312**

因此，aligned 将 latent geometry 从“相同 watermark 更相似”翻转为“相同物理内容更相似”。这比单独降低某个训练 loss 更直接地对应 Enigma 的 watermark dominance。

## 6. Frozen representation probe

### 6.1 协议

Gridfire 诊断使用：

- PushT expert dataset；
- 512 个 clips，每个 clip 4 帧；
- frameskip 5；
- 按 sampled clip 做 80/20 划分，避免相邻帧泄漏；
- frozen encoder；
- StandardScaler + Ridge，\(\alpha=1\)；
- seed 73。

`tag RGB R²` 表示从冻结 latent 中线性恢复彩色 watermark RGB 的能力。watermark 与 PushT 物理控制无关，因此在物理信息和规划能力不受损的前提下，tag R² 降低是正面结果。但 tag 越低并不天然代表模型越好，必须与物理 probe 和规划结果联合判断。

### 6.2 Gridfire 同协议结果

Projection space 是 planner 直接使用的主要空间。

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| masked reachability final | 0.056 | 0.869 | 0.999 | 0.928 |
| pred03 final | **0.800** | 0.952 | 0.997 | 0.889 |
| aligned near 96k | 0.793 | 0.929 | 0.996 | 0.522 |
| aligned final | 0.795 | 0.926 | 0.996 | **0.502** |

Backbone space：

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| masked reachability final | 0.202 | 0.884 | 0.998 | 0.960 |
| pred03 final | 0.862 | **0.971** | 0.997 | 0.890 |
| aligned near 96k | 0.867 | 0.966 | 0.996 | 0.610 |
| aligned final | **0.870** | 0.963 | 0.995 | **0.566** |

aligned 相比 pred03 final：

- backbone angle：`0.862 → 0.870`，基本持平；
- backbone block XY：`0.971 → 0.963`，基本持平；
- backbone tag：`0.890 → 0.566`，下降 `0.324`；
- projection angle：`0.800 → 0.795`，基本持平；
- projection tag：`0.889 → 0.502`，下降 `0.387`。

因此，aligned 并没有通过丢弃物理状态信息来降低 watermark。更关键的是，pred03 本来已经可以很好地线性解码 block angle，但其规划和 content/tag geometry 仍然较差。这说明单纯 decodability 不能解释 aligned 的收益。

## 7. 旧机 IDM/MSID 历史参照

旧机 `probe_linear.json` 使用 6000 帧、Ridge \(\alpha=10\)、按 episode 划分；Gridfire 使用 512 clips、Ridge \(\alpha=1\)、按 clip 划分。因此下列数字只能作为跨协议量级参照，不能作为严格排行榜。

旧机 projection probe：

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| focus IDM | 0.24 | 0.90 | 0.98 | 0.87 |
| focus MSID | 0.43 | 0.92 | 0.98 | 0.91 |
| clean baseline | 0.92 | 0.98 | 0.95 | — |
| watermarked baseline | 0.06 | 0.81 | 0.53 | 0.95 |
| aligned（Gridfire 协议） | **0.795** | 0.926 | 0.996 | **0.502** |

旧机 backbone probe：

- focus IDM：angle `0.46`，block XY `0.93`，tag `0.92`；
- focus MSID：angle `0.56`，block XY `0.94`，tag `0.90`；
- aligned（Gridfire 协议）：angle `0.87`，block XY `0.963`，tag `0.566`。

跨协议结果显示一个很强的趋势：aligned 的 orientation decodability 进入接近 clean baseline 的量级，同时 watermark decodability 明显低于 IDM/MSID。但在用同一 probe 重测 IDM/MSID checkpoint 之前，不能据此宣称严格超过 IDM/MSID。

## 8. 当前机制解释

现有证据支持以下解释：

1. Masked reachability 中 watermark 高度可解码，block angle 很弱。
2. Pred03 恢复了很强的 block angle 信息，但 watermark 仍高度可解码，规划提升有限。
3. Aligned 保留了与 pred03 几乎相同的物理状态 decodability，却显著降低 watermark decodability。
4. Aligned 同时将 pair geometry 翻转为 content-dominant，并获得更好的规划表现。

因此，aligned 的收益不能简单解释为“向 embedding 注入了更多 block angle 信息”。更符合数据的解释是：

> Aligned routing 改变了任务因素与 nuisance 的相对几何显著性，使已有物理信息更可能支配 planner 使用的表示空间。

这直接对应 Enigma 的核心漏洞：predictability 与 representation utility 发生错位。

## 9. 与 IDM/MSID 的概念区别

IDM/MSID 要求从相邻或多步 latent 中恢复动作：

\[
G(z_t,z_{t+H})\rightarrow a_{t:t+H-1}.
\]

它们主要回答“动作相关信息能否被恢复”。Aligned 进一步回答：

> 即使动作或物理状态已经可解码，哪些 prediction updates 仍被允许改变整个表示空间？

因此：

- IDM/MSID 增加控制信息约束；
- aligned 管理 prediction 对 encoder 的优化权限；
- IDM/MSID 关注 action decodability；
- aligned 关注 predictive gradients 对规划几何的塑造。

## 10. 必须保留的边界

当前不能过度主张：

1. 规划主结果仍是单 seed；
2. 每次评估只有 50 episodes，单点波动为 0.02；
3. 尚未完成同协议 IDM/MSID 严格对照；
4. 保留约 10% prediction gradient 不能单独排除“强降权”解释；
5. Frozen linear probe 证明信息可解码，不证明该信息被 CEM 因果使用；
6. 表示空间的局部梯度性质不等价于 Adam 参数空间中的全局优化保证。

## 11. 下一步关键实验

1. **Norm-matched scalar full run**：区分方向路由和动态 prediction 降权。
2. **Aligned seed 1、2**：确认规划和几何结果的稳定性。
3. **同协议 IDM/MSID probe 与规划评估**：建立严格基线。
4. **Future-latent permutation intervention**：打乱候选动作与 predicted future latent 的对应关系，测量 CEM cost ranking、top-k overlap、所选动作、真实物理进展和成功率变化。

## 12. 暂定贡献表述

> Enigma-style failure is not solely an information-absence problem. Task-relevant physical factors may already be linearly decodable while predictable nuisances still dominate the geometry used for planning. Control-Aligned Predictive Gradient Routing preserves full predictive supervision for the predictor but conditionally admits prediction gradients into the encoder according to their alignment with control learning. The resulting representation preserves physical-state decodability, suppresses watermark prominence, reverses content/tag geometry, and improves downstream planning.

中文版本：

> Enigma 类失效不只是任务信息缺失。物理状态可能已经可以从 embedding 中解码，但易预测 nuisance 仍然能够主导规划几何。Control-Aligned Predictive Gradient Routing 保留 predictor 的完整预测监督，同时根据 prediction 与控制梯度的对齐关系，决定预测更新进入 encoder 的程度。结果显示，该方法在保留物理状态信息的同时，削弱 watermark 的表示显著性，逆转 content/tag 几何，并改善下游规划。

## 13. 推荐名称

当前实现名称：

> **Control-Aligned Predictive Gradient Routing**

更抽象、能够兼容后续 scalar 版本的核心概念：

> **Control-Conditioned Predictive Gradient Admission**

后者更接近可能最终保留下来的论文贡献：prediction 是否以及多大程度上能够塑造 encoder，应由其与任务学习方向的关系决定。
