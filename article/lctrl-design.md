# $L_{\mathrm{ctrl}}$ 的设计逻辑

本文档单独说明当前主方法中控制目标 $L_{\mathrm{ctrl}}$ 的定义、设计动机与理论边界。这里讨论的是代码中的 `masked_reachability` 配置，不把它与梯度路由规则混为一谈。

## 1. 我们希望 $L_{\mathrm{ctrl}}$ 提供什么

标准 JEPA 的预测损失要求预测 latent 与未来观察的 latent 接近：

\[
L_{\mathrm{pred}}=\lVert \hat z_{t+h}-z_{t+h}\rVert^2.
\]

这个目标奖励一切容易预测的视觉因素。若一个彩色 tag 在整条轨迹中保持不变，那么只编码 tag 也能使预测损失很低，但这种表示不一定能区分不同动作造成的物理后果。

因此，$L_{\mathrm{ctrl}}$ 的职责不是直接提高图像预测精度，而是向共享 encoder 提供另一种训练证据：

> 表示应当保留足以解释“什么动作连接了两个状态，以及该动作会到达哪个后续状态”的信息。

这一目标只使用观察、动作与轨迹顺序，不使用模拟器状态、任务成功标签或候选动作的真实执行结果。

## 2. 精确定义

设

- $z_t=E(x_t)$ 为真实观察的 latent；
- \(\hat z_{t+1}=P(z_t,a_t)\) 为 JEPA predictor 产生的下一时刻 latent；
- \(A_{t:t+h-1}=(a_t,\ldots,a_{t+h-1})\) 为长度为 $h$ 的动作序列；
- 当前最大时间跨度为 $H=3$。

主实验使用：

\[
\boxed{
L_{\mathrm{ctrl}}
=L_{\mathrm{inv}}
+0.5L_{\mathrm{cycle}}
+0.1L_{\mathrm{reach}}
}
\]

对应配置为：

```text
mode = masked_reachability
max_horizon = 3
inverse_target = sequence
inverse_weight = 1.0
cycle_weight = 0.5
reachability_weight = 0.1
mask_keep_prob = 0.5
temperature = 0.1
```

### 2.1 多时间跨度逆动力学

对于 $h\in\{1,2,3\}$，使用独立的 inverse head $D_h$，根据真实起点和终点 latent 恢复完整动作序列：

\[
\hat A_{t:t+h-1}=D_h(\widetilde z_t,\widetilde z_{t+h}),
\]

\[
L_{\mathrm{inv}}
=\frac{1}{H}\sum_{h=1}^{H}
\operatorname{SmoothL1}
\bigl(\hat A_{t:t+h-1},A_{t:t+h-1}\bigr).
\]

这里预测的是完整的 $h$-步动作序列，而不是动作均值。

它提供的证据是：如果两段物理变化由不同动作造成，表示应当保留能够区分这些变化的信息。轨迹内恒定的 tag 无法单独确定连接两个端点的动作序列。

采用多个时间跨度，是为了避免控制信号只描述最短期的相邻帧变化，并让表示同时接触一至三步动作后果。

### 2.2 预测动作闭环

逆动力学仅作用于真实的 endpoint pair 时，encoder 可能具有动作信息，但 predictor 产生的未来 latent 未必保留同样的信息。因此，我们再把 predictor 的输出送入一步 inverse head：

\[
\hat a_t^{\mathrm{cycle}}=D_1(\widetilde z_t,\widetilde{\hat z}_{t+1}),
\]

\[
L_{\mathrm{cycle}}
=\operatorname{SmoothL1}(\hat a_t^{\mathrm{cycle}},a_t).
\]

这一项要求预测出来的 latent 变化仍然能够解释输入动作。它把动作监督从真实表示延伸到预测表示，避免 control head 与 JEPA predictor 各自解决互不相干的问题。

这里的“cycle”是动作重构闭环，不是像素重建，也不是要求 latent 动力学严格可逆。

### 2.3 动作条件终点辨识

给定起点 $z_t$ 和动作序列 $A_{t:t+h-1}$，构造 query：

\[
q_{t,h}=Q_h(\widetilde z_t,A_{t:t+h-1}).
\]

同一轨迹中的所有 latent \(\{z_\tau\}\) 经 key head 得到候选：

\[
k_\tau=K(\widetilde z_\tau).
\]

模型需要从同一轨迹的候选中识别真实终点 $z_{t+h}$：

\[
L_{\mathrm{reach}}
=-\log
\frac{\exp(q_{t,h}^{\top}k_{t+h}/\tau)}
{\sum_{\tau'}\exp(q_{t,h}^{\top}k_{\tau'}/\tau)},
\qquad \tau=0.1.
\]

关键设计是：负样本来自同一条轨迹。对 tagged PushT 而言，同一轨迹中的候选具有相同的 episode-level tag，因此仅凭 tag 不能判断哪一个候选是动作执行后的正确终点。模型必须利用轨迹内部随物理状态变化的信息。

该项补充了逆动力学的方向：

- inverse：给定两个端点，恢复连接它们的动作；
- reachability：给定起点和动作，从候选中找出正确终点。

二者共同要求 latent 中的动作与状态变化具有双向可读的关系。

## 3. 为什么使用共享随机子空间掩码

训练期间，对一对起点与终点使用同一个随机 feature mask，保留概率为 $0.5$，并使用 inverted-dropout 缩放：

\[
\widetilde z=m\odot z/p,
\qquad m_j\sim\operatorname{Bernoulli}(p),\quad p=0.5.
\]

reachability 中，同一个样本的 query 起点与全部候选终点也共享该 mask。

这样做的目的有两个：

1. 防止辅助任务长期依赖极少数固定 latent 坐标；
2. 迫使动作相关信息在随机部分特征缺失时仍可被读取。

共享 mask 很重要。如果起点、正确终点和负样本分别使用独立 mask，分类器可能利用 mask 差异完成任务，而不是学习动作条件关系。

这一机制只能被表述为正则化和抗局部捷径设计，不能据此证明表示已经恢复了完整物理状态。

## 4. 三个分量为什么都有动机，但尚未证明缺一不可

| 分量 | 直接约束 | 主要防止的问题 |
|---|---|---|
| $L_{\mathrm{inv}}$ | 真实 endpoint pair 必须支持动作序列恢复 | 表示只保留静态、可预测但与动作无关的信息 |
| $L_{\mathrm{cycle}}$ | predictor 输出必须保留输入动作造成的变化 | control head 只在真实 latent 上有效，预测未来仍与动作脱节 |
| $L_{\mathrm{reach}}$ | 动作条件 query 必须识别同轨迹真实终点 | 利用 episode-level 常量或粗粒度时间无关特征完成辅助任务 |

这张表给出的是设计动机，不是必要性定理。目前不能声称删掉任一项一定失败；若要建立这种结论，需要逐项移除的严格训练消融。

## 5. $L_{\mathrm{ctrl}}$ 与梯度路由的分工

两者解决不同问题：

- $L_{\mathrm{ctrl}}$ 定义“当前动作监督希望表示往哪里改变”；
- gradient router 决定 prediction gradient 中哪些分量进入共享 encoder。

令

\[
g_c=\nabla_z L_{\mathrm{ctrl}},
\qquad
g_p=\nabla_z L_{\mathrm{pred}}.
\]

路由器把 $g_c$ 当作局部 guide，但不会修改 $L_{\mathrm{ctrl}}$ 本身；控制损失仍然完整地反向传播。与此同时，predictor-only 参数仍接收完整预测梯度。

因此，方法的逻辑顺序是：

\[
\text{动作与轨迹}
\xrightarrow{L_{\mathrm{ctrl}}}
\text{局部动作相关学习信号 }g_c
\xrightarrow{\text{router}}
\text{共享 encoder 的预测更新分配}.
\]

## 6. 为什么不只使用普通 IDM

一步 IDM 已经能迫使表示保留部分动作相关变化，但存在三个缺口：

1. 它只约束真实相邻帧，不直接约束 predictor 产生的未来 latent；
2. 它不要求模型在同一轨迹的多个相似状态中找出动作对应的真实终点；
3. 一步动作可能不足以描述跨多个采样间隔的物理变化。

当前 $L_{\mathrm{ctrl}}$ 因而是在 IDM 基础上的关系增强，而不是简单增加一个更大的动作回归头。

## 7. 它能支持什么主张

可以支持的表述是：

> $L_{\mathrm{ctrl}}$ 利用多步动作反演、预测动作闭环和同轨迹终点辨识，为共享表示提供局部的动作相关训练证据；同轨迹负样本使 episode-level 恒定 tag 不能单独解决 reachability 任务。

结合 tagged PushT 的实验，我们还可以说：

> 使用该 control guide 进行 encoder 优化时，模型恢复了更强的物理状态可解码性，并显著降低了 tag 对候选代价和动作选择的影响。

## 8. 它不能单独证明什么

以下说法超出了当前目标和证据：

- $L_{\mathrm{ctrl}}$ 识别了完整的 controllable state；
- 能从动作反演的信息就是全部 planning-relevant information；
- 与 $g_c$ 正交的方向一定是 nuisance；
- control head 无法使用任何其他 shortcut；
- 更低的 control loss 必然带来更高闭环成功率；
- 三个分量以及当前权重 $1.0/0.5/0.1$ 是理论最优的。

逆动力学还可能受到专家策略偏差和动作多解性的影响；reachability 也可能利用轨迹内时间线索。随机 masking 和 action-shuffle diagnostics 能帮助发现部分捷径，但不能消除所有不可辨识性。

## 9. 最简洁的汇报版本

> 我们的 $L_{\mathrm{ctrl}}$ 不是特权状态监督，而是由轨迹自身构造的动作关系监督。它包含三项：从真实起终点恢复一至三步动作序列、从预测终点恢复输入动作，以及给定起点和动作后从同一轨迹中识别真实终点。同轨迹候选共享 episode-level tag，因此 tag 无法单独完成终点辨识。这个目标为 encoder 提供局部动作相关梯度，但我们不把它解释为完整可控状态的标注；它只是比单一步 IDM 更稳定、更贴近规划所需关系的 guide。
