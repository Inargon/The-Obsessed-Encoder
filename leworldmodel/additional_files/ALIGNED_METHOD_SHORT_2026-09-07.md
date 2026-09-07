# Control-Aligned Predictive Gradient Routing

## Motivation

在 JEPA 中，prediction loss 不仅训练 predictor，也通过反向传播决定 encoder 强化哪些视觉因素。Enigma 表明，watermark 等易预测但与控制无关的因素可能因此主导 latent geometry。

我们的核心原则是：

> Predictor 可以学习全部可预测信息，但 prediction gradient 只有在与控制学习一致时，才能完整地塑造 encoder。

## Method

对每个样本，记 prediction loss 与 control loss 关于共享表示 (z) 的梯度为

\[
g_p=\nabla_z\mathcal L_{pred},
\qquad
g_c=\nabla_z\mathcal L_{ctrl}.
\]

将 prediction gradient 分解为相对 (g_c) 的平行分量和正交分量：

\[
g_p=\alpha g_c+g_\perp,
\qquad
\alpha=\frac{g_p^\top g_c}{\lVert g_c\rVert^2}.
\]

定义非负对齐门控

\[
\gamma=[\cos(g_p,g_c)]_+,
\]

并将进入 encoder 的 prediction gradient 替换为

\[
\widetilde g_p
=
[\alpha]_+g_c+\gamma g_\perp.
\]

因此：

- 与控制方向正向平行的 prediction 分量被完整保留；
- 与控制冲突的平行分量被删除；
- 与控制近乎正交的 prediction 分量按对齐程度衰减。

## Asymmetric optimization

路由只发生在 prediction loss 进入共享 encoder 的梯度接口：

\[
g_{encoder}=g_c+\widetilde g_p.
\]

Predictor 本身仍接收完整的 (g_p)，继续学习预测未来表示。实现中使用一个前向值为零的 gradient surrogate，在反向传播时将 encoder 处的 (g_p) 精确替换为 (\widetilde g_p)，不改变 loss 数值或 predictor 的训练信号。

该路由具有局部性质：

\[
g_c^\top\widetilde g_p=[g_c^\top g_p]_+\ge0.
\]

也就是说，路由后的 prediction gradient 不会在表示空间中保留沿当前控制梯度的负投影；同时，它还管理传统冲突消除方法通常完整保留的近正交 prediction updates。
