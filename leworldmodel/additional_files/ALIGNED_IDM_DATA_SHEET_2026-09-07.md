# Aligned / IDM / MSID：实验数据表

## 1. 规划成功率摘要

### 1.1 Gridfire

Seed 0；每次评估 50 episodes；每 `0.02` 对应 1 个成功 episode。

| 方法 | 训练步数 | Peak SR | Peak step | Final SR | Last-10 / recent mean |
|---|---:|---:|---:|---:|---:|
| conditional | 139,329 | 0.64 | 64k | 0.54 | 0.56 |
| masked reachability | 139,329 | 0.74 | 110k | 0.66 | 0.66 |
| pred03 | 139,329 | 0.72 | 54k | 0.64 | 0.624 |
| aligned | 139,329 | **0.88** | 96k / 118k | **0.80** | **0.774** |

### 1.2 旧机

Seed 0；旧机训练内评估协议，与 gridfire 不作严格小数位横比。

| 方法 | Peak SR | Final SR | Last-10 SR |
|---|---:|---:|---:|
| focus IDM | 0.88 | 0.78 | **0.82** |
| focus MSID | 0.88 | **0.82** | 0.81 |

Seed 1：

| 方法 | Peak SR | Final SR | Last-10 SR |
|---|---:|---:|---:|
| focus IDM | 0.88 | 0.78 | 0.77 |
| focus MSID | — | 约 0.77（138k） | 约 0.77 |

注：focus MSID seed1 JSONL 末尾两条 `0.04/0.00` 疑似重跑开头记录，未计入后期统计。

## 2. 旧机 IDM/MSID seed0 曲线

每 10k step 抽样；最后一行为 138k。

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

## 3. Aligned seed0 后期曲线

Gridfire；每次 50 episodes。

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

Last-10（120k–138k）：

```text
0.82, 0.78, 0.80, 0.76, 0.76, 0.74, 0.78, 0.74, 0.76, 0.80
mean = 0.774
```

## 4. Gridfire frozen linear probe

协议：512 clips；4 frames/clip；frameskip 5；80/20 split by sampled clip；StandardScaler + Ridge `alpha=1`；seed 73。

### 4.1 Projection

| 方法 / checkpoint | Block angle sin-cos R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| masked reachability final | 0.0561 | 0.8695 | **0.9988** | **0.9282** |
| pred03 near 54k | 0.7447 | 0.9375 | 0.9972 | 0.9157 |
| pred03 final | **0.7999** | **0.9516** | 0.9966 | 0.8889 |
| aligned near 96k | 0.7934 | 0.9291 | 0.9961 | 0.5218 |
| aligned final | 0.7954 | 0.9256 | 0.9961 | **0.5017** |

### 4.2 Backbone

| 方法 / checkpoint | Block angle sin-cos R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| masked reachability final | 0.2016 | 0.8844 | **0.9982** | **0.9599** |
| pred03 near 54k | 0.8520 | 0.9685 | 0.9974 | 0.9339 |
| pred03 final | 0.8617 | **0.9713** | 0.9972 | 0.8903 |
| aligned near 96k | 0.8675 | 0.9661 | 0.9956 | 0.6098 |
| aligned final | **0.8705** | 0.9630 | 0.9955 | **0.5659** |

### 4.3 Aligned final − pred03 final

| Space | Block angle ΔR² | Block XY ΔR² | Pusher XY ΔR² | Tag RGB ΔR² |
|---|---:|---:|---:|---:|
| projection | −0.0046 | −0.0261 | −0.0005 | **−0.3872** |
| backbone | +0.0088 | −0.0082 | −0.0018 | **−0.3245** |

## 5. 旧机 frozen linear probe

旧机协议：6000 frames；StandardScaler + Ridge `alpha=10`；split by episode。与第 4 节协议不同。

### 5.1 Projection

| 方法 | Block angle R² | Block XY R² | Pusher XY R² | Tag RGB R² |
|---|---:|---:|---:|---:|
| focus IDM | 0.24 | 0.90 | 0.98 | 0.87 |
| focus MSID | **0.43** | **0.92** | 0.98 | 0.91 |
| LDAD, lambda=1 | 0.15 | 0.89 | 0.98 | 0.90 |
| LDAD, lambda=10 | 0.21 | 0.91 | **0.99** | **0.86** |
| clean baseline | **0.92** | **0.98** | 0.95 | — |
| watermarked baseline | 0.06 | 0.81 | 0.53 | 0.95 |

Gridfire aligned final 参照（不同协议）：angle `0.795`，block XY `0.926`，pusher XY `0.996`，tag RGB `0.502`。

### 5.2 Backbone

| 方法 | Block angle R² | Block XY R² | Tag RGB R² |
|---|---:|---:|---:|
| focus IDM | 0.46 | 0.93 | 0.92 |
| focus MSID | 0.56 | 0.94 | 0.90 |

Gridfire aligned final 参照（不同协议）：angle `0.870`，block XY `0.963`，tag RGB `0.566`。

## 6. Gridfire content/tag pair geometry

Backbone final checkpoint。

| 方法 | Same-content cosine | Same-tag cosine | Content − tag margin |
|---|---:|---:|---:|
| conditional | 0.3268 | 0.6388 | −0.3120 |
| masked reachability | 0.4702 | 0.5175 | −0.0473 |
| aligned | **0.807** | **0.176** | **+0.631** |

## 7. Aligned 梯度统计

### 7.1 Final

| Metric | Value |
|---|---:|
| prediction/control gradient cosine | 0.0715 |
| orthogonal gate | 0.0752 |
| retained prediction-gradient norm | 0.1057 |
| reversed / negative fraction | 0.125 |

### 7.2 Retention 随训练阶段

| Step range | Mean retained prediction-gradient norm |
|---|---:|
| 0–20k | 0.160 |
| 20–60k | 0.139 |
| 60–90k | 0.121 |
| 90–108k | 0.110 |
| final | 0.106 |

## 8. 10k matched ablations

Seed 0；50 evaluation episodes。

| 方法 | 2k | 4k | 6k | 8k | 10k | Peak | Retention | Content/tag margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | 0.02 | 0.04 | 0.22 | 0.30 | 0.36 | 0.36 | 0.161 | 0.754 |
| parallel-only | 0.02 | 0.06 | 0.02 | 0.34 | 0.24 | 0.34 | 0.085 | **0.949** |
| shuffled guide | 0.04 | 0.10 | 0.16 | 0.12 | 0.18 | 0.18 | 0.031 | 0.882 |
| norm-matched scalar | 0.08 | 0.14 | 0.20 | 0.40 | **0.42** | **0.42** | 0.153 | 0.682 |
| retention-matched shuffled | 0.02 | 0.14 | 0.20 | 0.30 | 0.28 | 0.30 | 0.149 | 0.716 |

### 8.1 10k final losses / routing diagnostics

| 方法 | Pred loss | Control loss | Reference retention | Norm-match error | Direction cosine to aligned |
|---|---:|---:|---:|---:|---:|
| aligned | 0.0899 | 0.2196 | — | — | 1.000 |
| parallel-only | 0.3323 | 0.2718 | — | — | — |
| shuffled guide | 0.2621 | 0.2753 | — | — | — |
| norm-matched scalar | 0.1096 | 0.2582 | 0.1532 | 0.0020 | 0.7536 |
| retention-matched shuffled | 0.1377 | 0.2646 | 0.1485 | 0.0017 | 0.5388 |

## 9. 数据口径

| 数据块 | Seed | Eval episodes | Probe split | Ridge alpha | 可否严格横比 |
|---|---:|---:|---|---:|---|
| Gridfire planning | 0 | 50/checkpoint | — | — | Gridfire 内可比 |
| Gridfire frozen probe | probe seed 73 | — | by sampled clip | 1 | Gridfire probe 内可比 |
| 旧机 planning | 0 / 1 | 旧训练内评估 | — | — | 旧机内部可比 |
| 旧机 frozen probe | — | — | by episode | 10 | 旧机 probe 内可比 |

跨旧机与 gridfire 的 SR、probe 数字只作为量级参照，不用于小数位严格排名。
