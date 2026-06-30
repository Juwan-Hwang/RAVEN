# RAVEN 竞品分析与战略报告

> 评估时间：2026-06-28
> 评估范围：RAVEN v9.1 (commit aa314b8) + 四个参考仓库 (Glass, DiskANN, NGT, PatANN)
> 评估方式：源码审查 + 实测基准对比 + 架构差异分析
> 核心结论：RAVEN 在 SIFT1M 上已达到/超过 Glass HNSW 同配置水平，但距离"世界第一"还需在 ann-benchmarks 官方榜单上证明

---

## 一、执行摘要

### 1.1 当前状态

H20 实测推翻了 "avg_visited 膨胀 9 倍" 的虚假叙事后，RAVEN 的核心定位发生了根本性变化：

| 指标 (ef=50, R=32, L=200, FP32, 单线程) | RAVEN v9.1 | Glass HNSW (H20 实测) | 差距 |
|:--|:--|:--|:--|
| recall@10 | **97.05%** | 94.65% | RAVEN +2.4pp |
| QPS | **9,195** | 7,678 | RAVEN +20% |
| avg_visited | 1,227 | 1,041 | Glass -15% |

**RAVEN 在 recall 和 QPS 上均优于 Glass**，avg_visited 略高 15% 属于正常范围。不存在需要修复的 "膨胀" 问题。

### 1.2 核心判断

RAVEN 当前不是"有 bug 需要修"的状态，而是"性能已经很好但缺少官方背书"的状态。要实现"世界第一"，重心应从**内省式调试**转向**外向式竞争**——在 ann-benchmarks 官方框架上跑出可验证的数字。

---

## 二、竞品深度分析

### 2.1 PatANN — Pattern-Aware Vector Search

**来源**: `patann-main` (闭源 C++ SDK + Python/Swift/Kotlin 绑定)
**官网**: patann.dev
**状态**: Beta，闭源

#### 核心宣称

PatANN 自称 "pattern-aware" 向量搜索框架，利用向量中的 "macro 和 micro patterns" 在距离计算前大幅缩减搜索空间。官网宣称在 SIFT-128-euclidean 上超越 HNSW (hnswlib)、ScaNN、FAISS 等。

#### 技术分析（基于示例代码和 ann-benchmarks 配置）

从 `patann_utils.py` 和 API 接口可推断：

| 概念 | 推测含义 | 技术对标 |
|:--|:--|:--|
| `setConstellationSize(16)` | 类似 HNSW 的 M 参数，控制图度数 | HNSW M |
| `setRadius(100)` | 搜索半径，控制候选集大小 | ef_search 的变体 |
| `createInstance(dim)` | 按维度创建引擎实例 | 维度特化编译 |
| `addVector()` + `waitForIndexReady()` | 异步增量插入 | HNSW incremental |
| `createQuerySession()` + async callback | 异步查询 + 回调 | 事件驱动搜索 |
| Pattern-Aware | **利用向量分布的模式先验**做预过滤 | 独特创新点 |

从 ann-benchmarks 的 `expann` (PatANN 的 ann-benchmarks 名称) 配置看：
- 参数: `M`, `ef_construction`, `ortho_count`, `prune_overflow`, `use_compression`
- `ortho_count` 暗示正交化处理（可能类似 ScaNN 的各向异性量化）
- `prune_overflow` 暗示图剪枝溢出控制
- 维度特化编译 (64/128/256/832/960 多个二进制)

#### 创新点评估

| 创新点 | 创新程度 | RAVEN 是否具备 |
|:--|:--|:--|
| Pattern-Aware 预过滤 | **高** — 在距离计算前用模式匹配缩减搜索空间 | ❌ 无 |
| 异步事件驱动搜索 | 中 — 工程创新，非算法创新 | ❌ 无（同步搜索） |
| 多平台绑定 (Python/iOS/Android) | 低 — 工程封装 | ❌ 仅 Rust |
| 维度特化编译 | 低 — 常见优化 | ⚠️ 有 dispatch 宏但未编译期特化 |

**关键风险**: PatANN 闭源，无法审计其 "pattern-aware" 声明的真实性。ann-benchmarks 上的结果可能是特定配置下的 cherry-pick。但其异步搜索 + 多平台覆盖在工程层面有明显优势。

### 2.2 Glass — HNSW + 多量化器

**来源**: `glass-ref/pyglass-main` (开源 C++ + Python 绑定)
**状态**: 活跃开发，ann-benchmarks 榜单常客

#### 架构

| 层 | 实现 | 质量 |
|:--|:--|:--|
| 图算法 | HNSW (增量插入 + Heuristic2 剪枝) | 生产级 |
| 量化器 | FP32/FP16/BF16/SQ8/SQ4/PQ8 全覆盖 | 业界最全 |
| 距离核 | helpa 库 (AVX2/AVX-512/NEON) | 生产级 |
| 搜索池 | LinearPool (排序数组 + 游标弹出) | 高效 |
| 预取优化 | po/pl 双参数自动调优 | 工程亮点 |
| Refiner | 量化搜索 + FP32/FP16 rerank | 灵活 |

#### Glass 的核心优势

1. **SearchImpl2 + 预取**: Glass 的 `SearchImpl2` 先收集所有未访问邻居到 `edge_buf`，然后批量预取向量数据再计算距离。这种 "collect-prefetch-compute" 三段式比传统的 "逐个 visit-compute" 模式对 CPU cache 更友好。RAVEN 已经借鉴了这个模式。
2. **Optimize()**: 自动扫描 (po, pl) 参数空间找最佳预取配置。SIFT1M 上获得 30-55% 的性能提升。
3. **量化器全覆盖**: 一套搜索框架支持 7 种量化器，Refiner 可以任意组合粗量化 + 精量化。

### 2.3 DiskANN — Vamana + SSD

**来源**: `DiskANN-ref/DiskANN-main` (Rust 实现，开源)
**状态**: 微软研究院项目，面向 SSD 大规模

#### 架构

DiskANN 是 RAVEN 的算法祖先——Vamana 图构建、RobustPrune 剪枝、medoid 入口点均来自 DiskANN 论文。RAVEN 在此基础上增加了：
- AVQ 检索感知量化（vs DiskANN 的 PQ）
- RP-Tuning 后验调优（DiskANN 无）
- 分层导航（DiskANN 无，用 SSD 顺序读替代）
- LinearPool + Two-Pass Prefetch（RAVEN 独立实现，与 Glass 类似）

DiskANN 的核心创新在于 SSD 场景的 Vamana 图布局优化（compressed graph + disk-resident layout），RAVEN 目前不做 SSD。

### 2.4 NGT — 多图类型 + QG 优化

**来源**: `NGT-ref/NGT-main` (C++ + Python，开源)
**状态**: Yahoo/NTT 维护，ann-benchmarks 参与者

#### 架构

NGT 提供多种图类型：
- **ANNG** (Approximate Nearest Neighbor Graph): 增量构建
- **ONNG** (Optimized ANNG): 在 ANNG 基础上优化度数
- **PANNG** (Path-Adjusted ANNG): 路径优化
- **QG** (Quantized Graph): 量化优化

NGT 的差异化在于图优化流水线（ANNG → ONNG → PANNG → QG），每一步都有明确的优化目标。RAVEN 的 Vamana two-pass 是类似的思路但更简洁。

---

## 三、横向对比矩阵

### 3.1 算法层

| 维度 | RAVEN | Glass | DiskANN | NGT | PatANN |
|:--|:--|:--|:--|:--|:--|
| 图构建 | Vamana two-pass | HNSW incremental | Vamana two-pass | ANNG+ONNG | 未知 (pattern-aware) |
| 剪枝 | RobustPrune + DirectionalPrune | Heuristic2 | RobustPrune | 度数优化 | ortho_count + prune_overflow |
| 导航 | Layered Nav (独立分层图) | HNSW 多层 | Medoid 单入口 | 多入口 | Constellation |
| 入口点 | Medoid + k-means centroid | HNSW 层级入口 | Medoid | 多入口 | 未知 |
| 量化 | AVQ (retrieval-aware) | FP16/SQ8/PQ8 | PQ | QG | use_compression |
| 后验调优 | RP-Tuning (Scheme A) | Optimize(po,pl) | 无 | 无 | 无 |

### 3.2 工程层

| 维度 | RAVEN | Glass | DiskANN | NGT | PatANN |
|:--|:--|:--|:--|:--|:--|
| 语言 | Rust | C++ | Rust | C++ | C++ |
| SIMD | AVX-512 + AVX2 + f16 | helpa (AVX2/512/NEON) | AVX2 | 标量为主 | 未知 |
| 内存布局 | Hybrid Blocked-CSR | 连续数组 | Blocked-CSR | 邻接表 | 未知 |
| 搜索池 | LinearPool (自研) | LinearPool | NeighborPriorityQueue | 堆 | 未知 |
| 预取 | Two-Pass + graph prefetch | po/pl 自动调优 | 无 | 无 | 未知 |
| 异步搜索 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 多平台 | ❌ | Python | Rust | Python | Python/iOS/Android |
| SSD 支持 | 预留接口 | ❌ | ✅ | ❌ | ✅ (on-disk) |
| ann-benchmarks | ⚠️ 基本可用 | ✅ | ✅ | ✅ | ✅ |

### 3.3 性能对比 (SIFT1M, FP32, 单线程)

| 系统 | ef/配置 | recall@10 | QPS | avg_visited | 来源 |
|:--|:--|:--|:--|:--|:--|
| **RAVEN v9.1** | ef=50, R=32 | **97.05%** | **9,195** | 1,227 | H20 实测 |
| **Glass HNSW** | ef=50, R=32 | 94.65% | 7,678 | 1,041 | H20 实测 |
| Glass HNSW | ef=80, R=32 | 97.45% | 5,643 | 1,487 | H20 实测 |
| hnswlib (ann-benchmarks) | ef=50-70 | ~95% | ~8,000-10,000 | 未报告 | 榜单估计 |
| ScaNN (ann-benchmarks) | — | ~95% | ~12,000-15,000 | 未报告 | 榜单估计 |
| PatANN (官网宣称) | — | ~95% | >15,000 | 未报告 | 官网 (未验证) |

> **注**: ScaNN 和 PatANN 的数字来自 ann-benchmarks 官网/官网，配置可能不完全可比（多线程、不同 R 值等）。RAVEN 和 Glass 的数字是同配置单线程严格对比。

---

## 四、RAVEN 优势与劣势分析

### 4.1 核心优势

1. **recall-QPS Pareto 已超越 Glass**
   - ef=50 同配置下 RAVEN recall 高 2.4pp，QPS 高 20%
   - 这不是微弱优势——在 ANN 领域 2pp recall 差距在相同 QPS 下是显著差异

2. **Vamana 图质量优于 HNSW**
   - Two-pass (α=1.0 → α=1.2) 产出比 HNSW 单轮增量插入更优的图结构
   - RobustPrune 的 α 遮挡判定比 Heuristic2 更严谨（角度 vs 距离比值）

3. **Two-Pass Prefetch 热路径优化**
   - 搜索热路径借鉴 Glass SearchImpl2，先收集 edge_buf 再批量预取
   - Graph prefetch 预取下一轮 pop 节点的邻居列表 (4 cache lines)
   - Vector prefetch 前瞻 po=8 个邻居的向量数据

4. **AVQ 检索感知量化**
   - 唯一优化 retrieval loss 而非 reconstruction loss 的量化方案
   - 混合 loss: α×recon + (1-α)×retrieval，可调权重
   - 对 SIFT 数据集 ADC+rerank recall=0.9213（vs f32 0.9705，仅降 5%）

5. **Rust 内存安全 + 零开销抽象**
   - 无 GC 停顿，无 undefined behavior
   - `unsafe` 仅在热路径 (LinearPool memmove, SIMD intrinsics) 中使用，每处都有 SAFETY 注释

### 4.2 核心劣势

1. **无 ann-benchmarks 官方成绩**
   - 这是最大的短板。当前所有性能数据都是本地测量，未经第三方验证
   - ann-benchmarks 的 Docker 环境和评测协议与本地不同
   - 不上榜单 = 不被社区认可 = 不能宣称"世界第一"

2. **无多线程搜索**
   - Glass 的 `batch_search` 支持 OpenMP 并行
   - RAVEN 当前仅单线程搜索
   - ann-benchmarks 默认用多线程测 QPS，单线程成绩天然吃亏

3. **量化器覆盖不足**
   - Glass 支持 7 种量化器 (FP32/FP16/BF16/SQ8/SQ4/PQ8)
   - RAVEN 仅支持 PQ/OPQ/AVQ，无 FP16/SQ8 搜索路径
   - 在低内存场景 (SQ8/SQ4) 缺乏竞争力

4. **无 Refiner 机制**
   - Glass 的 Refiner 允许粗量化搜索 + 精量化 rerank
   - RAVEN 的 rerank 是手动拼接 (ADC top-100 → f32 rerank)，非框架级支持
   - 缺少 `reorder_mul` 自适应倍率

5. **无自动预取调优**
   - Glass 的 `Optimize()` 自动扫描 (po, pl) 找最佳配置
   - RAVEN 的 po 是硬编码 (默认 8)，无自动调优

6. **数据集覆盖单一**
   - 仅在 SIFT1M 上验证
   - 缺少 GIST1M、Glove、Cohere、Music-100 等常见 benchmark 数据集
   - 不同数据集维度/分布差异大，泛化能力未验证

---

## 五、通往"世界第一"的战略路线图

### 5.1 阶段划分

| 阶段 | 目标 | 时间估计 | 优先级 |
|:--|:--|:--|:--|
| Phase 1 | ann-benchmarks 官方上榜 | 1-2 周 | 🔴 P0 |
| Phase 2 | 多数据集 + 多量化器 | 2-3 周 | 🟡 P1 |
| Phase 3 | 多线程搜索 + 自动调优 | 2-3 周 | 🟡 P1 |
| Phase 4 | 差异化创新 (AVQ 论文 + ScaNN 级 PQ) | 4-6 周 | 🟢 P2 |

### 5.2 Phase 1: ann-benchmarks 官方上榜 (P0)

**为什么这是第一优先级**: 不上榜单，所有性能宣称都是空话。ann-benchmarks 是 ANN 领域的 "ImageNet"——社区认可的唯二标准之一 (另一个是 big-ann-benchmarks)。

**具体步骤**:

1. **修复 ann-benchmarks 集成**
   - 当前 `query()` 单查询返回空列表
   - 确保 `query_batch()` 在 Docker 环境中正确运行
   - 适配 ann-benchmarks 的 HDF5 数据格式 (当前只读 fvecs/ivecs)

2. **提交 SIFT-128-euclidean**
   - 这是 ann-benchmarks 最经典的数据集
   - RAVEN 已有 SIFT1M 完整数据，转换格式即可
   - 目标: recall@10=95% 时 QPS > 8,000 (超越 Glass 的 7,678)

3. **提交 GIST-960-euclidean**
   - 第二经典数据集，维度 960
   - 验证 RAVEN 在高维下的泛化能力
   - 可能需要调整 R/L 参数

4. **确保 Docker 可复现**
   - ann-benchmarks 要求 Dockerfile 可构建
   - 所有依赖 (Rust toolchain, SIMD flags) 需在 Docker 中正确配置

### 5.3 Phase 2: 多数据集 + 多量化器 (P1)

**目标**: 在 5+ 数据集上上榜，覆盖不同维度/分布

**数据集清单**:
- SIFT-128-euclidean (已有)
- GIST-960-euclidean (需转换)
- Glove-100-angular (内积距离，需实现 IP metric)
- Cohere-768-angular (现代 embedding 数据集)
- Music-100 (音频特征)

**量化器扩展**:
1. **FP16 搜索路径**: 已有 f16 距离核，需接入搜索框架
2. **SQ8 搜索路径**: 标量量化 8-bit，低内存场景必备
3. **Refiner 框架**: 粗量化搜索 + 精量化 rerank 的通用管道

### 5.4 Phase 3: 多线程搜索 + 自动调优 (P1)

**多线程 batch search**:
- 用 Rayon 并行化查询循环
- 每个 worker 独立 VisitedTracker + LinearPool
- 目标: 多核 QPS 线性扩展 (8 核 → 8x QPS)

**自动预取调优** (借鉴 Glass Optimize):
```rust
// 伪代码
fn optimize_prefetch(&mut self, sample_queries: &[f32]) {
    let mut best_qps = 0.0;
    for po in 0..=32 {
        for pl in 0..=16 {
            self.po = po;
            self.pl = pl;
            let qps = self.benchmark(sample_queries);
            if qps > best_qps {
                best_qps = qps;
                self.best_po = po;
                self.best_pl = pl;
            }
        }
    }
}
```

### 5.5 Phase 4: 差异化创新 (P2)

**方向 A: AVQ 论文级打磨**
- AVQ 是 RAVEN 独有的 retrieval-aware 量化，有学术发表价值
- 需要在更多数据集上验证 (当前仅 SIFT)
- 需要对比 ScaNN 的 anisotropic quantization 并论证差异

**方向 B: ScaNN 级 PQ (FastScan)**
- ScaNN 的核心优势在于 PQ + SIMD lookup table (FastScan)
- 用 4-bit PQ + AVX-512 VPDPBUSD 指令实现超高速距离近似
- 这是在 recall~90% 低 recall 区间拉开 QPS 差距的关键技术

**方向 C: PatANN 式预过滤**
- PatANN 宣称的 "pattern-aware" 预过滤可能是真正的创新点
- 如果能在距离计算前用廉价的特征匹配（如 LSH hash 碰撞）排除 80% 的候选
- 则 avg_visited 可降到 200 以下，QPS 翻倍

**方向 D: GPU 加速**
- ann-benchmarks 有 faiss-gpu 赛道
- 如果 RAVEN 能提供 CUDA 距离核，将在 QPS 上碾压所有 CPU 方案

---

## 六、技术债务清理 (与性能无关但必须做)

| 项目 | 严重度 | 工作量 | 说明 |
|:--|:--|:--|:--|
| pipeline.rs final_prune 用 truncate | 🔴 | 1h | 违反硬约束，但 Pipeline 路径非主路径 |
| pipeline.rs max_iterations=1 | 🔴 | 5min | 改为 2 |
| pipeline.rs quant_aware_prune no-op | 🟡 | 4h | 接通 QuantAwareRobustPrune |
| rp_tuning.rs SchemeB/C 静默回退 | 🟡 | 2h | 删除或实现 |
| ann-benchmarks query() 返回空 | 🔴 | 4h | P0 依赖项 |
| GEMM 路径标量回退 | 🟢 | 8h | 批量模式辅助路径 |
| centroid 均匀采样 → k-means | 🟢 | 已修复 | navigation.rs 已用 k-means |

---

## 七、竞争态势总结

### 7.1 RAVEN 的准确定位

RAVEN 当前处于 **"技术上领先但生态上边缘"** 的状态:
- 技术上: SIFT1M 同配置实测超越 Glass HNSW
- 生态上: 无 ann-benchmarks 成绩、无多数据集验证、无多量化器搜索路径
- 学术上: AVQ + RP-Tuning 有创新性但未发表

### 7.2 "世界第一"的定义

"世界第一"在 ANN 领域有三种可能的定义:

| 定义 | 含义 | RAVEN 当前距离 |
|:--|:--|:--|
| ann-benchmarks SIFT-128-euclidean 榜首 | QPS 最高 @ recall≥95% | **近** — 本地数据已超 Glass，需上榜单验证 |
| 全数据集平均榜首 | 在 10+ 数据集上平均 QPS 最高 | **远** — 仅 SIFT1M 验证 |
| 学术界认可的创新 | 论文发表 + 引用 | **中** — AVQ 有创新性但未发表 |

### 7.3 最短路径建议

**如果目标是 ann-benchmarks SIFT-128-euclidean 榜首**:

1. 修复 ann-benchmarks 集成 (1 周)
2. 在 Docker 环境跑 SIFT1M (2 天)
3. 提交结果 (1 天)
4. 如果 QPS 不够高，加多线程 batch search (1 周)
5. 如果 recall 不够高，调参 R/L/alpha (3 天)

**总计 2-3 周可达成本地已验证的 ann-benchmarks 上榜**。

---

## 八、PatANN 特别评估

### 8.1 威胁等级: 中

PatANN 闭源，无法审计核心算法。其官网宣称的 benchmark 结果无法独立验证。但其多平台覆盖 (Python/iOS/Android) 和异步搜索 API 在工程层面有明显优势——如果其 "pattern-aware" 声明属实，则可能在某些数据集上有颠覆性表现。

### 8.2 可借鉴之处

| PatANN 特性 | RAVEN 可借鉴方式 | 难度 |
|:--|:--|:--|
| 异步事件驱动搜索 | 用 Rust async/tokio 实现 async search | 中 |
| 维度特化编译 | 用 const generics 真正编译期特化高频维度 | 低 |
| Constellation (多入口) | 已有 centroid overlay，可扩展为多入口搜索 | 低 |
| Pattern-Aware 预过滤 | 研究 LSH/sketch 预过滤可行性 | 高 (需研究) |

### 8.3 不可借鉴之处

- PatANN 的闭源 SDK 模式不适合 RAVEN 的开源路线
- 多平台绑定 (iOS/Android) 是工程投入，不影响算法竞争力
- 其 ann-benchmarks 集成使用维度特化二进制 (expann_py_64/128/256...)，编译复杂度高

---

## 九、最终建议

### 9.1 立即行动 (本周)

1. **修复 ann-benchmarks `query()` 接口** — 这是上榜单的阻塞项
2. **跑通 Docker 环境** — 确保 `ann-benchmarks run --algorithm raven --dataset sift-128-euclidean` 能完成
3. **提交 SIFT-128-euclidean 结果** — 哪怕初始成绩不理想，先上榜

### 9.2 近期目标 (2-3 周)

1. 多线程 batch search (Rayon 并行)
2. FP16 搜索路径接入
3. GIST-960-euclidean 上榜

### 9.3 中期目标 (1-2 月)

1. AVQ 论文撰写 (retrieval-aware quantization 的理论贡献)
2. ScaNN 级 FastScan PQ (4-bit + SIMD lookup table)
3. 自动预取调优 (Optimize 机制)

### 9.4 长期愿景

RAVEN 的终极差异化在于 **AVQ (检索感知量化) + RP-Tuning (后验调优) + Vamana 图质量** 的三位一体。没有任何竞品同时具备这三个能力。如果能：
1. 在 ann-benchmarks 上证明 Vamana 图质量 (Phase 1)
2. 在论文中论证 AVQ 的理论优势 (Phase 4)
3. 在工业场景中展示 RP-Tuning 的实用价值

则 RAVEN 有可能成为 **学术界有贡献 + 工业界有用 + 社区有认可** 的三栖项目。

---

*报告结束*
