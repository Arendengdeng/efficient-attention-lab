# Efficient Attention Lab — Notebook Series Outline

本系列 Notebook 以 **FlashAttention** 论文（Dao et al., NeurIPS 2022）为核心，
从标准注意力的原理与瓶颈出发，逐步深入到 IO 感知的高效实现与量化压缩。

---

## 01 · Standard Attention（标准注意力）

**目标**：透彻理解 Scaled Dot-Product Attention 的数学原理、朴素实现与性能特征，为后续优化建立基准。

| 节 | 内容 |
|----|------|
| 1. 背景与动机 | Transformer 架构回顾；注意力在其中的角色 |
| 2. 数学定义 | $\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$；各符号含义 |
| 3. 朴素 PyTorch 实现 | 逐步手写 `naive_attention`；对比 `F.scaled_dot_product_attention` |
| 4. 多头注意力（MHA） | 拆分多头、拼接输出；`MultiHeadAttention` 封装 |
| 5. 复杂度分析 | 时间 $O(N^2 d)$、空间 $O(N^2)$；$N$ 变大时的瓶颈 |
| 6. 内存访问剖析 | HBM 读写次数分析；以此引出 IO 感知优化的必要性 |
| 7. 基准测试 | 用 `torch.utils.benchmark` 测量不同 $N$、$d$ 下的延迟与显存 |
| 8. 小结 | 总结瓶颈，预告 FlashAttention 的改进思路 |

---

## 02 · Flash Attention（IO 感知的精确注意力）

**目标**：理解 FlashAttention 的 Tiling 算法与 Online Softmax，通过实验验证其速度和显存优势。

| 节 | 内容 |
|----|------|
| 1. 问题回顾 | 标准实现的 HBM 瓶颈 |
| 2. Online Softmax | 数值稳定的递推 softmax；$m$、$\ell$ 累积量的更新规则 |
| 3. Tiling 算法 | 将 $Q/K/V$ 分块装入 SRAM；前向伪代码逐行解析 |
| 4. IO 复杂度分析 | HBM 读写 $O(N^2 d / M)$ vs 标准 $O(Nd + N^2)$ |
| 5. PyTorch 模拟实现 | 用纯 Python/PyTorch 复现 tiling 逻辑（非 CUDA） |
| 6. `torch.nn.functional.scaled_dot_product_attention` | PyTorch 2.x 内置 Flash Attention 使用方法 |
| 7. 基准测试 | 与标准注意力对比延迟、显存、序列长度可扩展性 |
| 8. 反向传播简述 | 重计算策略；不存储大中间矩阵的技巧 |
| 9. 小结 | FlashAttention v1 vs v2 改进点预览 |

---

## 03 · Quantization（注意力量化）

**目标**：掌握 Post-Training Quantization（PTQ）在注意力层中的应用，评估精度与速度的权衡。

| 节 | 内容 |
|----|------|
| 1. 量化基础 | 对称/非对称量化；scale & zero-point；INT8 / FP8 |
| 2. 注意力中的量化目标 | $Q$、$K$、$V$、$A$（注意力权重）分别量化的挑战 |
| 3. 逐张量 vs 逐通道量化 | 粒度对精度的影响 |
| 4. `torch.quantization` / `bitsandbytes` | 工具链实操 |
| 5. 量化误差分析 | 与 FP32 基准的余弦相似度、MSE |
| 6. 量化感知训练（QAT）简介 | Straight-Through Estimator 原理 |
| 7. 基准测试 | INT8 量化前后的延迟与显存对比 |
| 8. 小结 | 精度-效率 Pareto 前沿讨论 |

---

## 04 · Results Analysis（综合结果分析）

**目标**：汇总前三个 Notebook 的实验数据，绘制综合对比图表，得出结论性洞察。

| 节 | 内容 |
|----|------|
| 1. 实验数据加载 | 读取 `results/` 目录下的 CSV / JSON 文件 |
| 2. 延迟对比 | 标准注意力 vs Flash Attention vs 量化方案；折线图 |
| 3. 显存对比 | 不同序列长度下的峰值显存柱状图 |
| 4. 精度对比 | 量化误差随位宽的变化曲线 |
| 5. 序列长度可扩展性 | $N=128$ 至 $N=8192$ 的延迟增长率 |
| 6. 硬件利用率 | FLOPS 利用率、带宽利用率（如有 Nsight 数据） |
| 7. 结论与展望 | 各方法适用场景总结；FlashAttention v2、Paged Attention 等延伸方向 |

---

## 依赖与环境

```
torch >= 2.0
numpy
matplotlib
pandas
triton          # 可选，用于 CUDA kernel 实验
bitsandbytes    # 可选，用于量化实验
```

运行顺序：`01` → `02` → `03` → `04`（04 依赖前三个 Notebook 产出的结果文件）。
