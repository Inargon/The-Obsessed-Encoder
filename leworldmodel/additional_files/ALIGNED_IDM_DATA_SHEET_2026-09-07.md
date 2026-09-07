# Aligned vs. IDM / MSID：实验数据表

## 1. 规划成功率

| 方法 | Seed | Peak SR | Final SR | Last-10 SR |
|---|---:|---:|---:|---:|
| **Aligned** | 0 | **0.88** | 0.80 | 0.774 |
| Focus IDM | 0 | **0.88** | 0.78 | **0.82** |
| Focus MSID | 0 | **0.88** | **0.82** | 0.81 |
| Focus IDM | 1 | **0.88** | 0.78 | 0.77 |
| Focus MSID | 1 | — | 约 0.77（138k） | 约 0.77 |

Aligned 的 peak 出现在 96k 和 118k。Focus MSID seed 1 JSONL 末尾两条 `0.04/0.00` 疑似重跑开头记录，未计入后期统计。

## 2. 成功率曲线

### 2.1 Aligned seed 0

| Step | SR | Step | SR |
|---:|---:|---:|---:|
| 76k | 0.70 | 108k | 0.86 |
| 78k | 0.80 | 110k | 0.74 |
| 80k | 0.74 | 112k | 0.82 |
| 82k | 0.74 | 114k | 0.78 |
| 84k | 0.76 | 116k | 0.84 |
| 86k | 0.78 | 118k | **0.88** |
| 88k | 0.72 | 120k | 0.82 |
| 90k | 0.84 | 122k | 0.78 |
| 92k | 0.82 | 124k | 0.80 |
| 94k | 0.76 | 126k | 0.76 |
| 96k | **0.88** | 128k | 0.76 |
| 98k | 0.82 | 130k | 0.74 |
| 100k | 0.74 | 132k | 0.78 |
| 102k | 0.66 | 134k | 0.74 |
| 104k | 0.78 | 136k | 0.76 |
| 106k | 0.76 | 138k | 0.80 |

Aligned last 10：

```text
0.82, 0.78, 0.80, 0.76, 0.76, 0.74, 0.78, 0.74, 0.76, 0.80
mean = 0.774
```

### 2.2 IDM / MSID seed 0

| Step | Focus IDM SR | Focus MSID SR |
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

## 3. Frozen linear probe：projection 空间

这是 planner 实际使用的表示空间。

| 方法 / checkpoint | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| **Aligned near 96k** | **0.793** | **0.929** | **0.996** | **0.522** |
| **Aligned final** | **0.795** | **0.926** | **0.996** | **0.502** |
| Focus IDM | 0.24 | 0.90 | 0.98 | 0.87 |
| Focus MSID | 0.43 | 0.92 | 0.98 | 0.91 |

## 4. Frozen linear probe：backbone 空间

| 方法 / checkpoint | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| **Aligned near 96k** | **0.867** | **0.966** | 0.996 | **0.610** |
| **Aligned final** | **0.870** | **0.963** | 0.995 | **0.566** |
| Focus IDM | 0.46 | 0.93 | — | 0.92 |
| Focus MSID | 0.56 | 0.94 | — | 0.90 |

## 5. Aligned content/tag 几何

Aligned final backbone：

| Same-content cosine | Same-tag cosine | Content − tag margin |
|---:|---:|---:|
| **0.807** | **0.176** | **+0.631** |

```text
Content − tag margin
= same-content cosine − same-tag cosine
= 0.807 − 0.176
= +0.631
```

正数表示表示空间更按任务内容组织；负数表示更按 watermark/tag 组织。当前没有 IDM/MSID 的同项 pair-geometry 数据。

## 6. Aligned 梯度数据

### 6.1 训练终点

| Metric | Value |
|---|---:|
| Prediction/control gradient cosine | 0.0715 |
| Orthogonal gate | 0.0752 |
| Retained prediction-gradient norm | 0.1057 |
| Reversed / negative fraction | 0.125 |

### 6.2 Retention 随训练变化

| Step range | Mean retained prediction-gradient norm |
|---|---:|
| 0–20k | 0.160 |
| 20–60k | 0.139 |
| 60–90k | 0.121 |
| 90–108k | 0.110 |
| Final | 0.106 |

## 7. 数据口径备注

| 数据 | 评估口径 |
|---|---|
| Aligned planning | Seed 0；50 episodes/checkpoint |
| Aligned probe | 512 clips；80/20 sampled-clip split；Ridge α=1；probe seed 73 |
| IDM/MSID planning | Seed 0/1；旧训练内评估 |
| IDM/MSID probe | 6000 frames；episode split；Ridge α=10 |

当前先并表用于直观比较；后续用统一训练与 probe 协议重跑后替换对应数字。
