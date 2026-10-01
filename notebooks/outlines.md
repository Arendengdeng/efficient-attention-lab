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

## 02 · Online Softmax

**目标**：从 Safe Softmax 出发推导 Online Softmax 的递推公式，实现可增量维护的单趟扫描注意力，为 FlashAttention 的 Tiling 算法奠定算法基础。

| 节 | 内容 |
|----|------|
| 1. 为什么需要 Online Softmax？ | 标准 Softmax 需要两趟扫描；Tiling 要求一趟完成 |
| 2. Safe Softmax | 减去行最大值防止溢出；仍是两趟扫描 |
| 3. Online Softmax 推导 | 累积量 $(m^{(t)}, \ell^{(t)})$ 的递推公式与正确性证明；标量版与块级版 |
| 4. 带输出累积的 Online Attention | 同步维护 $O^{(t)}$；归一化版更新公式 |
| 5. 分块（Tiling）Online Attention | 批量多头版本；FlashAttention Algorithm 1 对应实现 |
| 6. 数值精度分析 | 块大小对精度的影响；FP16 vs FP32 误差 |
| 7. 性能对比 | 与标准注意力及 `F.sdpa` 对比延迟与显存 |
| 8. 小结与展望 | $(m,\ell)$ 可结合性；与 FlashAttention 的关系；预告 Notebook 03 |

---

## 03 · Flash Attention（IO 感知的精确注意力）

**目标**：在 Online Softmax 的基础上，理解 FlashAttention 的 IO 复杂度分析、块大小选择与实际加速效果。

| 节 | 内容 |
|----|------|
| 1. IO 复杂度分析 | HBM 读写 $O(N^2 d / M)$ vs 标准 $O(Nd + N^2)$；最优块大小推导 |
| 2. Tiling 算法完整伪代码 | Algorithm 1 逐行解析；与 Notebook 02 实现对应 |
| 3. `F.scaled_dot_product_attention` | PyTorch 2.x 内置 backend 选择策略；`sdp_kernel` 控制 |
| 4. 基准测试 | GPU 上 Flash vs 标准注意力：延迟、显存、序列长度可扩展性 |
| 5. 反向传播：重计算策略 | 不存储 $S$/$A$；反向重算；总显存 $O(N)$ 证明 |
| 6. FlashAttention v2 改进 | 外循环换成 $Q$ 块；FLOPs 减半；因果掩码优化；序列并行 |
| 7. 小结 | 精度等价性确认；适用场景总结；预告量化 Notebook |

---

## 04 · Quantization（注意力量化）

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

## 05 · Results Analysis（综合结果分析）

**目标**：汇总前四个 Notebook 的实验数据，绘制综合对比图表，得出结论性洞察。

| 节 | 内容 |
|----|------|
| 1. 实验数据加载 | 读取 `results/` 目录下的 JSON 文件 |
| 2. 延迟对比 | 标准注意力 vs Flash Attention vs 量化方案；折线图 |
| 3. 显存对比 | 不同序列长度下的峰值显存柱状图 |
| 4. 精度对比 | 量化误差随位宽的变化曲线 |
| 5. 序列长度可扩展性 | $N=128$ 至 $N=8192$ 的延迟增长率 |
| 6. 结论与展望 | 各方法适用场景总结；FlashAttention v2、Paged Attention 等延伸方向 |

---

## 依赖与环境

```
torch >= 2.0
numpy
matplotlib
pandas
scipy          # 用于 linregress 拟合
triton         # 可选，用于 CUDA kernel 实验
bitsandbytes   # 可选，用于量化实验
```

运行顺序：`01` → `02` → `03` → `04` → `05`（05 依赖前四个 Notebook 产出的结果文件）。
