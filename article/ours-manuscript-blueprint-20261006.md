# Ours 文章重构总纲（2026-10-06）

本文档是重写论文的单一叙事入口。方法在正式命名前统一写作 **Ours**；
“Bloop”“EMA orthogonal repair”和“修复方法”只作为内部别名，不进入主标题。

## 1. 一句话故事

视觉 JEPA 会优先组织容易预测、但未必影响决策的因素；Ours 将控制视为主任务，
用控制梯度的指数滑动平均构造稳定保护方向，并且只在预测梯度与该方向冲突时
删除有害分量，从而在保留预测学习的同时减少 nuisance 对规划几何的支配。

## 2. 论文主张的强弱边界

### 可以主张

1. 一个可预测且与控制无关的 episode-constant tag 可以主导 JEPA latent，
   即使表征没有 collapse。
2. 高线性可解码性不等价于 planner 真正使用该信息；必须同时检查 latent
   allocation、反事实干预和闭环成功率。
3. Ours 在局部一阶意义上消除预测更新对 EMA 控制方向的负内积，同时不修改
   无冲突预测梯度。
4. 在目前的 seed-0、50-episode pilot 中，Ours 在五个 CEM 任务上均取得较高
   成功率，并显著降低 tagged PushT 的反事实 tag 敏感性。
5. 冻结 Ours encoder 后，GC-IDM 在 Cube、Reacher 和 TwoRoom 上达到 100%，
   说明这些 latent displacement 可被轻量策略直接摊销为动作；PushT 的接触多解性
   仍使在线 CEM 更稳。

### 暂时不能主张

1. Ours 找到了完整的任务相关子空间，或所有正交方向都是 nuisance。
2. 单训练种子、单评测种子的百分比构成最终统计结论。
3. GC-IDM 与 CEM 的成功率差异全部来自 planner；训练目标和推理协议也不同。
4. 当前跨来源 clean baseline 已经全部严格 matched。
5. tagged PushT 的 `88%` 与 rollout showcase 中的 `90%` 可以混用。正文暂以保存的
   final-eval JSON 中 `88%` 为主；在协议审计前，`90%` 只用于对应可视化组说明。

## 3. 方法

设共享 encoder 参数为 $\theta$，控制损失和预测损失对其梯度分别为

$$
g_c=\nabla_\theta\mathcal L_{\mathrm{ctrl}},\qquad
g_p=\nabla_\theta\mathcal L_{\mathrm{pred}}.
$$

维护控制梯度的指数滑动平均（EMA）：

$$
m_t=\beta m_{t-1}+(1-\beta)\operatorname{stopgrad}(g_c).
$$

仅在 $g_p^\top m_t<0$ 时删除预测梯度沿控制保护方向的冲突分量：

$$
\widetilde g_p=
\begin{cases}
g_p-\dfrac{g_p^\top m_t}{\lVert m_t\rVert_2^2+\epsilon}m_t,
&g_p^\top m_t<0,\\[6pt]
g_p,&g_p^\top m_t\geq0.
\end{cases}
$$

共享 encoder 使用

$$
g_{\mathrm{shared}}=g_c+\lambda\widetilde g_p.
$$

预测器和控制头等任务专属参数仍由各自原始损失更新。该规则与“始终完全正交化”
不同：后者无论冲突与否都删除平行分量；Ours 只修复负冲突，因此保留协同分量和
全部无冲突预测信息。EMA 的作用是降低单 minibatch 控制梯度噪声造成的错误投影。

局部性质为：在 $m_t\neq0$ 且发生冲突时，
$\widetilde g_p^\top m_t\approx0$；不冲突时保持
$\widetilde g_p=g_p$。这是局部优化性质，不是全局语义识别定理。

## 4. 主实验表：在线 CEM 规划

统一表述为 seed-0 checkpoint、evaluation seed 42、每项 50 episodes 的 pilot。
不同任务的环境/runtime 细节必须在附录逐项列出。

| Task | Condition | Planner | Episodes | Success rate |
|---|---|---:|---:|---:|
| PushT | clean | CEM | 50 | 92% |
| PushT | tagged | CEM | 50 | 88% |
| Reacher | clean | CEM | 50 | 88% |
| TwoRoom | clean | CEM | 50 | 96% |
| Cube | clean | CEM | 50 | 80% |

这张表证明方法没有只在水印环境中工作，但在补齐多训练种子与置信区间以前，应写成
“strong pilot performance”，不能写成最终 SOTA。

## 5. 第二实验表：冻结 encoder 的 GC-IDM

GC-IDM 根据当前 latent 与目标 latent 直接预测动作序列，用来检验 latent geometry
是否可被摊销为控制，而不依赖昂贵的在线候选搜索。

| Task | Frozen representation | GC-IDM SR | CEM reference SR | Observation |
|---|---|---:|---:|---|
| Cube | Ours | 100% | 80% | latent displacement 可直接控制 |
| Reacher | Ours | 100% | 88% | 目标姿态几何清晰 |
| TwoRoom | Ours | 100% | 96% | 路径目标可直接解码 |
| clean PushT | Ours | 86% | 92% | 接触与动作多解使直接回归受限 |

协议：50 episodes、evaluation seed 42、goal offset 25、evaluation budget 50、
GC-IDM 训练 200 epochs。该表不应写成 GC-IDM 全面优于 CEM；更准确的结论是，
简单或低歧义任务可以摊销规划，而 PushT 仍受益于在线搜索。自然的后续方法是
GC-IDM warm start 加少量 CEM refinement。

## 6. Tagged PushT 的核心机制证据

### 6.1 表征中有什么

Projection-space linear probe：

| Model | Pusher $R^2$ | Block XY $R^2$ | Angle $R^2$ | Tag RGB $R^2$ |
|---|---:|---:|---:|---:|
| JEPA | 0.2341 | 0.3773 | -0.1678 | 0.9455 |
| Ours | 0.9963 | 0.9534 | 0.6894 | 0.9191 |

这里不能声称 Ours 删除了 tag；tag 仍然高度可解码。正确结论是 Ours 恢复了物理状态
的简单可访问性，并改变了物理内容相对 tag 的特征分配。

### 6.2 表征把多少几何变化分给 tag

Matched trajectory feature-allocation ratio
（tag-only latent path length / physical-content latent path length）：

- JEPA geometric mean：20.8712，95% CI [18.8516, 23.1068]；
- Ours geometric mean：1.4200，95% CI [1.3357, 1.5093]；
- ratio 小于 1 的比例：JEPA 0%，Ours 13.28%。

这是文章最直观的 representation-allocation 结果：JEPA 的 latent path 对 tag 变化约
为物理变化的 21 倍，而 Ours 将两者拉回接近同一量级。

### 6.3 Planner 是否真的被 tag 改变

固定物理帧、目标与候选动作，只交换 tag：

| Metric | JEPA | Full | Cycle | Ours |
|---|---:|---:|---:|---:|
| Pair-order reversal | 9.84% | 3.47% | 3.62% | **2.27%** |
| Selected candidate changed | 69.53% | 8.59% | 10.16% | **7.03%** |
| Cost RMSE / reference std | 58.36% | 17.58% | 16.78% | **10.78%** |

配对 bootstrap 中，Ours 相对 JEPA 的三个差值置信区间均严格低于零。相对 Full 与
Cycle，pair-order reversal 与 cost RMSE 的置信区间严格低于零；candidate-selection
change 的区间跨零。因此正文应突出“排序翻转和 cost distortion 显著降低”，不要
声称每一个 intervention metric 都显著优于所有对照。

## 7. 建议的正文结构

### 1. Introduction

- JEPA 的失败不是 collapse，而是有限表征容量与几何被容易预测的偏差占据。
- Probe 可解码性不足以保证 planner 使用；需要 allocation 与 intervention。
- 控制目标提供局部保护信号，但瞬时梯度噪声大，完全正交又会误删协同信息。
- 提出 EMA conflict repair，并给出跨任务及 amortized-control 证据。

### 2. Related Work

- JEPA / LeWorldModel 与 predictable-bias failure；
- multi-task gradient surgery（PCGrad 等），明确 Ours 的新意不在“首次投影”；
- action-conditioned representation learning / inverse dynamics；
- latent planning 与 amortized planning（Latent Geometry Beyond Search）；
- identifiability 与 latent straight-line diagnostics（When Does JEPA Learn）。

### 3. Method

1. predictive encoder allocation 问题；
2. control auxiliary objective；
3. instantaneous orthogonalization 的两个缺陷；
4. EMA conflict repair；
5. 局部性质、算法框与计算开销；
6. 与 Always-Orthogonal、Joint、Full/Cycle 的区别。

### 4. Experimental Setup

- 五个任务与 tagged PushT 构造；
- matched train/eval protocol；
- CEM 与 GC-IDM；
- 四层指标：probe、allocation、counterfactual intervention、closed-loop SR；
- pilot 与正式多种子结果的标注规则。

### 5. Results

1. tagged PushT：JEPA 的偏差与 Ours 的修复；
2. 跨任务 CEM 主表；
3. frozen-representation GC-IDM；
4. gradient mechanism diagnostics；
5. failure boundary：clean PushT 的 GC-IDM 低于 CEM。

### 6. Limitations

- 单训练种子和 50-episode pilot；
- EMA 方向不是完整控制子空间；
- tag 是可控合成 nuisance；
- historical Reacher runtime 与现代任务环境需逐项审计；
- GC-IDM 与 CEM 不是完全相同的优化器预算。

## 8. 主文图表顺序

1. **Figure 1：问题图。** Matched tag intervention：固定物理内容，只改变 tag，
   对比 JEPA 与 Ours 的 latent/cost 改变。
2. **Figure 2：方法图。** `bloop-training-mechanism.png`，重标为 Ours，展示
   EMA protection direction 与 conflict-only projection。
3. **Figure 3：feature allocation。** 只放 JEPA 与 Ours；主图用配对点/分布，
   图注明 ratio=1，而配对关系必须用真正逐样本连接或干脆移除连接线。
4. **Figure 4：matched rollout。** 同一 episode 的 JEPA failure 与 Ours success。
5. **Figure 5：跨任务成功案例。** clean/tagged PushT、Cube、TwoRoom 各选一例；
   Reacher 单独用黄色当前姿态与洋红真实 goal pose 的重叠图。
6. **Table 1：CEM 跨任务 SR。**
7. **Table 2：GC-IDM 与 CEM reference。**
8. **Appendix：latent straight-line gallery。** 该诊断中 Ours 并非最优，不应放在
   主文中作为优势证据；可以作为“SR 与全局线性 identifiability 不完全等价”的边界。

## 9. 近期写作顺序

1. 先重写 abstract、introduction 和 method，冻结符号与主张；
2. 建立新的 machine-readable result snapshot，逐项链接原始 JSON；
3. 重写 experiments/results，任何数字必须从 snapshot 生成或核对；
4. 将图中所有内部名字统一改为 Ours；
5. 最后写 related work 与 limitations，避免用 related work 反向决定主叙事。

## 10. 当前必须审计的三个不一致

1. tagged PushT final evaluation 为 88%，可视化 matched group 记录为 90%；
2. 历史 `results_snapshot.json` 仍描述旧 gated-orthogonal/CGPA，不可直接给新方法引用；
3. `paper/` 当前正文仍以 cosine admission 为主，必须整体迁移，不能只替换方法名。

