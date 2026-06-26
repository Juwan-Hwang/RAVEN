# RAVEN 冲击 ann-benchmarks 世界第一：战略路线图

> 创建时间：2026-06-25
> 目标：在 ann-benchmarks SIFT1M (sift-128-euclidean) 上达到 recall-QPS Pareto 前沿世界第一

---

## 〇、项目最高规则（不可违反）

> 以下规则为项目级别最高约束，所有优化工作必须严格遵守，不得有任何例外。

1. **每个修复必须有 before/after 实验数据**：任何优化在提交前，必须记录修改前的基线数据和修改后的对比数据，确认优化有效且无性能回退。
2. **不接受假优化和性能回退**：如果优化后 recall 下降或 QPS 降低，该优化必须回退，不得保留。
3. **一个一个来，每个优化做到极致**：不批量修改，逐项推进，每项做到最好再进入下一项。
4. **实验结果补充到本文档**：每项优化的 before/after 数据、结论必须记录在本文档对应章节中。
5. **达不到目的就舍弃并记录**：如果某项优化实验后未达到预期效果，必须回退代码，并在本文档中记录"已舍弃"及原因。
6. **Git 提交纪律**：每项优化确认有效后，单独 commit，commit message 包含 before/after 数据摘要。
7. **QPS 口径必须与 leaderboard 一致**：所有对外声称的 QPS 必须与 ann-benchmarks leaderboard 的硬件和线程口径严格一致。单线程 QPS 不得与多线程 QPS 混用，否则不计入"打榜成绩"。
8. **环境干净**：每次跑 benchmark 前，必须确认没有后台进程（旧 exe、cargo、rustc）残留。CPU 被抢占会导致双向 2x 劣化（QPS 减半 + 建图翻倍），这是环境污染的经典签名，不是代码回归。

---

## 实验记录

### 基线数据（BEFORE）

> 所有优化的对比基准。在开始任何修改前先建立。
> **必须在干净环境下重新测量**——之前的 QPS=1,287 / 建图=1,054.8s 可能被后台进程污染。

| 指标 | 数值 | 备注 |
|:--|:--|:--|
| cargo test | **155 passed, 0 failed** | 全部通过 |
| cargo clippy | **34 errors** | 已存在问题（needless_range_loop, module_inception 等） |
| recall@10 | **待测（干净环境）** | SIFT1M quick_recall_check (α=1.2, l_build=100, r_max=32, ef=100) |
| QPS | **待测（干净环境）** | SIFT1M quick_recall_check |
| 建图时间 | **待测（干净环境）** | SIFT1M quick_recall_check |

**额外修复（FIX-0）**：`init_random_graph` 死循环 bug
- **问题**：`neighbor_count = config.r_max`（默认 64），当 n < r_max+1 时，while 循环永远无法凑够邻居数，导致死循环。
- **修复**：`let neighbor_count = config.r_max.min(n.saturating_sub(1));`
- **验证**：修复后 155 个测试全部通过，无卡死。

---

### FIX-1: 修配置默认冲突 (avq+L2)

- **问题**：`Config::default()` 中 `avq=true` + `distance="l2"` 违反规则 `avq_l2_conflict`，导致 `merge_config()` 必然失败。
- **修复**：`config.rs` 第 115 行 `avq: true` → `avq: false`；更新 3 个相关测试断言。
- **Before**：`merge_config(None, None, false)` → `Err`（默认配置无法通过校验）
- **After**：`merge_config(None, None, false)` → `Ok`（默认配置通过校验）
- **cargo test**：155 passed, 0 failed（与基线一致，无回归）
- **性能影响**：无（纯配置默认值修改，不影响搜索路径）
- **结论**：✅ 已采纳

---

## 一、现状 vs 世界第一的差距分析

### 1.1 当前 RAVEN 性能（SIFT1M, 128-dim, L2）

> 以下数据需在干净环境下重新测量确认。

| 工作点 | recall@10 | QPS | 路径 |
|:--|:--|:--|:--|
| α=1.2, r_max=64, ef=100 | 0.9961 | 2,434 | f32 全精度 |
| α=1.0, r_max=64, ef=50 | 0.9275 | 7,611 | f32 全精度 |
| α=1.2, ef=100, ADC+rerank | 0.9676 | 2,025 | AVQ 量化 |

### 1.2 竞争目标：ann-benchmarks 真实 Pareto 前沿

> **不可使用估算数据。** 必须从 ann-benchmarks 官方 results 仓库拉取真实的、逐工作点的 Pareto 数据。
> SIFT1M (sift-128-euclidean) 上常年榜首未必是 ScaNN——glass、NGT-qg、hnswlib fork 经常压过 ScaNN。
> 目标是超越当前 Pareto 前沿的**外包络线**（所有库取最优），不是某一个特定库。

**TODO**：从 https://github.com/erikbern/ann-benchmarks 的 `results/` 目录拉取 sift-128-euclidean 的最新 JSON 数据，提取 Pareto 前沿外包络线，填入下表。

| 工作点 | 榜首算法 | recall@10 | QPS | 备注 |
|:--|:--|:--|:--|:--|
| 高 recall 区 | 待填（真实数据） | ~0.99 | 待填 | |
| 中 recall 区 | 待填（真实数据） | ~0.95 | 待填 | |
| 低 recall 区 | 待填（真实数据） | ~0.90 | 待填 | |

### 1.3 差距诊断

> 以下"差距倍数"在拉取真实排行榜数据前为**暂估**，确认后更新。

| 维度 | RAVEN（当前基线） | 榜首外包络（待填） | 差距倍数 |
|:--|:--|:--|:--|
| recall 0.95 处 QPS | 待测（干净环境） | 待填 | 待计算 |
| recall 0.99 处 QPS | 待测（干净环境） | 待填 | 待计算 |
| ADC 路径 vs f32 | ADC **更慢** | 榜首 ADC 快 3-5x | **根本性缺陷** |
| 建图时间 | ~790s (r_max=32) | ~200-500s | 2-4x |

**核心发现：RAVEN 的 ADC 路径比 f32 还慢，这是反直觉的。** 世界顶尖库（ScaNN 等）之所以快，正是因为 PQ-ADC 路径比 f32 快 3-5 倍（1 字节 PQ code vs 4 字节 f32，内存带宽降低 4 倍）。RAVEN 有量化器但搜索时没有用 SIMD lookup table 加速 ADC，而是逐子空间标量计算——这等于浪费了量化的全部优势。

---

## 二、代码审计：已验证的缺陷清单

> 以下每一条均经源码逐行验证，**全部属实**。

### 2.1 严重问题（S 级）：未完整实现 / 不算生产级

#### S1. ann-benchmarks wrapper 每次查询重建索引 ✅ 已验证

**文件**：`ann_benchmarks/algorithms/raven/__init__.py:86` + `src/bin/raven_ann_bench.rs:99`

**证据**：
- `__init__.py` 的 `query()` 方法调用二进制时传递了 `--train` 参数，但**未传递 `--load`**。
- `query_batch()` 方法同样如此。
- `fit()` 方法构建索引后**未传递 `--save`** 保存索引。
- `raven_ann_bench.rs`：`if !load_path.is_empty()` → 由于 `query()` 不传 `--load`，`load_path` 为空，必然走 else 分支**重新构建索引**。
- 结论：**每次 `query()` / `query_batch()` 调用都会从训练集重新建图**，完全浪费了 `fit()` 阶段的构建工作。评测 QPS 被建图开销严重污染。

**影响**：benchmark 结果完全不可信，QPS 被建图时间淹没。

---

#### S2. 批量查询 GEMM 是标量回退 ✅ 已验证

**文件**：`src/memory/query_ctx.rs`

**证据**：
- 注释明确写道：`当前实现为标量回退，GEMM 路径在 Week 3-4 接入`
- `gemm_path()` 实际实现是逐候选调用 `l2_simd`，与 `scalar_simd_path()` 完全相同——**没有任何 GEMM 矩阵乘法**。

**影响**：批量吞吐不算完整实现。

---

#### S3. big-ann / SSD 路径只是预留接口 ✅ 已验证

**文件**：`Cargo.toml`

**证据**：`big_ann = []` 空特征，无任何实现代码。

**影响**：设计文档暗示支持 big-ann，实际无任何实现。

---

#### S4. RP-Tuning B/C 方案未实现，静默回退 ✅ 已验证

**文件**：`src/graph/rp_tuning.rs`

**证据**：SchemeB/C 的行为与 SchemeA 完全相同（都是 `neighbors.clone()`），仅多了一条 `eprintln!` 警告。

**影响**：API 暴露了 SchemeB/C 选项但实际无效。

---

#### S5. 维度分发是统一 SIMD 分发，非 const-generic 特化 ✅ 已验证

**文件**：`src/distance/dispatch.rs`

**证据**：`select_l2()` 所有分支返回同一个 `l2_simd` 函数指针，match 表达式完全是 no-op。

**影响**：运行时分发本身已足够，但与文档描述有差距。

---

#### S6. 配置默认值自相矛盾 ✅ 已验证

**文件**：`src/config/config.rs` + `src/config/rules.rs`

**证据**：`Config::default()` 中 `avq=true` + `distance="l2"` 违反规则 `avq_l2_conflict`，默认配置无法通过校验。

**影响**：开箱即用的默认配置是坏的。

---

### 2.2 代码质量问题（M/L 级）

#### M1. `eprintln!` 遍布热路径 ✅ 已验证

**证据**（grep 结果，共 40+ 处）：分布在 `avq.rs`, `opq.rs`, `vamana.rs`, `rp_tuning.rs`, `raven_ann_bench.rs` 等。

**影响**：生产构建中 `eprintln!` 会锁 stderr、格式化字符串、I/O 系统调用，在热路径中造成微秒级延迟。应替换为 `tracing`。

---

#### M2. `unwrap()` 散布于核心模块 ✅ 已验证

**证据**：`src/` 目录下 18+ 处 `unwrap()`。

**影响**：核心库中的 `unwrap()` 在异常输入时会 panic，不符合生产级健壮性要求。

---

### 2.3 质量评价总结

| 维度 | 评分 | 说明 |
|:--|:--|:--|
| 研究原型 | 8/10 | 算法实现正确，实验有数据支撑，模块边界清晰 |
| 工程完整度 | 6.5/10 | 多处"有原型/有回退/有预留接口"而非完整实现 |
| 生产可用度 | 5.5/10 | 缺稳定 API、索引生命周期管理、错误处理、并发查询 |
| 文档遵照度 | 65-75% | 核心算法已实现，但 big-ann/GEMM/RP-Tuning B-C/维度特化/默认配置均有差距 |

---

## 三、六阶段冲刺计划

### Phase 0：修复地基（1-2 天）

**目标：消除所有已知缺陷，让代码库处于完美状态**

> **执行纪律**：Phase 0 的每一项修复都是独立 commit，逐项推进。每项修复前后跑 `cargo test` 确认无回归。

| 编号 | 任务 | 文件 | 原因 |
|:--|:--|:--|:--|
| 0.1 | **修复 ann-benchmarks wrapper**：`fit()` 传 `--save`，`query()`/`query_batch()` 传 `--load` | `__init__.py` + `raven_ann_bench.rs` | S1：每次查询重建索引 |
| 0.2 | `pipeline.rs` final_prune 改用 RobustPrune | `pipeline.rs` | 违反硬约束 |
| 0.3 | `pipeline.rs` max_iterations 改为 2 | `pipeline.rs` | 违反硬约束 |
| 0.4 | `pipeline.rs` quant_aware_prune 接通真实实现 | `pipeline.rs` | 当前是 no-op |
| 0.5 | `delayed_prune.rs` final_prune 改用 RobustPrune | `delayed_prune.rs` | 违反硬约束 |
| 0.6 | `rp_tuning.rs` SchemeB/C 实现或标注为 `unimplemented!()` | `rp_tuning.rs` | S4：静默回退 |
| 0.7 | **修配置默认冲突**：`default()` 中 `avq=false` | `config.rs` | S6：默认配置校验失败 |
| 0.8 | 全代码库 `eprintln!` → `tracing` | `avq.rs`, `opq.rs`, `vamana.rs` 等 | M1：热路径 I/O |
| 0.9 | 魔法数字提取为具名常量 | `avq.rs` | 可维护性 |
| 0.10 | `try_into().unwrap()` → `?` 错误传播 | `graph.rs`, `vamana.rs` | M2：健壮性 |
| 0.11 | 清理过时 `#[allow(dead_code)]` | `kernel.rs` | 代码整洁 |
| 0.12 | 明确标注 big-ann / GEMM 为未实现 | `Cargo.toml`, README | S2/S3：文档诚实 |

---

### Phase 1：LUT16 SIMD PQ-ADC 距离计算（3-5 天）🔴 最高优先级

**这是决定成败的关键。没有这个，不可能世界第一。**

#### 1.1 原理

当前 ADC 路径对每个候选节点：
1. 取 M 个 PQ codes（M 字节）
2. 对每个子空间，遍历 K=256 个聚类中心做标量 L2 → 极慢

顶尖库的做法：
1. **查询时预计算** distance lookup table：对每个子空间 m，计算 query 到 K 个中心的 L2 距离
2. **候选计算**只需 M 次 table lookup + 求和

#### 1.2 实现方案

```
adc_distance(query, pq_codes[M]):
  // 预计算（每查询一次）
  for m in 0..M:
    for k in 0..K:
      lut[m][k] = l2_sq(query_sub[m], centroid[m][k])

  // 候选距离（每候选一次）
  dist = 0
  for m in 0..M:
    dist += lut[m][pq_codes[m]]   // 单次 lookup
  return dist
```

#### 1.3 SIMD 加速：LUT16-shuffle 路线（放弃 gather）

> **关键决策：放弃 gather 指令，采用 pshufb + maddubs 的 LUT16 路线。**
>
> gather 指令（`_mm512_i32gather_ps` / `_mm256_i32gather_ps`）在绝大多数 x86 微架构上并不快：
> 内部仍是串行访存，延迟很高，很多场景下还不如标量循环。gather 达不到预期是**大概率事件**。
>
> ScaNN 真正的快路径（LUT16）走的是完全不同的机制：
> 1. 把 PQ 量化到 **4-bit**（每子空间 16 个中心，而不是 256）
> 2. 把 16 个中心的距离量化成 u8 放进一个寄存器
> 3. 用 `_mm256_shuffle_epi8`（pshufb）做**寄存器内查表**——单周期，不碰内存
> 4. 用 `_mm256_maddubs_epi16` 做定点累加
>
> 关键差别：shuffle 是寄存器内操作，不碰内存，这才是 4-5x 加速的来源；gather 碰内存，基本拿不到收益。

**LUT16 实现方案**：

```
// 预计算（每查询一次）
for m in 0..M:
    for k in 0..16:  // 注意：K=16，不是 256
        lut_u8[m][k] = quantize_to_u8(l2_sq(query_sub[m], centroid[m][k]))
        // 16 个 u8 = 正好一个 __m128i lane

// 候选距离（SIMD，一次处理 32 个子空间）
// PQ codes 是 4-bit，两个 code 打包进一个 byte
// pshufb 做寄存器内查表，maddubs 做定点累加
dist_u16 = _mm256_maddubs_epi16(
    _mm256_shuffle_epi8(lut_packed, codes_packed),
    ones
);
// 水平求和得到 u16 距离，再转 f32
```

**连锁影响：4-bit 量化会降低 recall**

4-bit PQ（K=16）比 8-bit PQ（K=256）的量化误差更大，recall 会下降。
**不能把 4-bit 化和 rerank 当成两个独立 Phase。**
Phase 1 必须包含 4-bit PQ 训练 + LUT16 SIMD ADC + 基本 rerank，作为一个整体验证 recall。
Phase 2 再做 rerank 策略的精细调优（ef_search 自适应、增量 rerank 等）。

#### 1.4 预期收益

> **注意：以下收益为暂估，不是连乘关系。**
> Phase 1（ADC SIMD）会把距离计算从 compute-bound 拉到 memory-bound。
> 一旦瓶颈变为 memory-bound，Phase 3 的 cache 优化收益会大幅衰减。
> Phase 1 完成后必须用 `perf stat` 测 LLC miss 和带宽利用率，再决定 Phase 3/4 是否值得投入。

| 路径 | 当前 QPS | 预期 QPS | 加速比 | 备注 |
|:--|:--|:--|:--|:--|
| f32 全精度 (ef=100) | 待测 | 不变 | 1x | 基线 |
| ADC 标量 (当前) | 待测 | - | - | 比全精度还慢 |
| **LUT16 SIMD ADC (目标)** | - | 待验证 | **预计 3-5x vs f32** | shuffle 不碰内存 |
| LUT16 SIMD + rerank | - | 待验证 | **预计 2-3x vs f32** | rerank 补偿 recall |

---

### Phase 2：两阶段搜索管道优化（2-3 天）

**目标：LUT16 ADC 快速粗筛 → f32 精确 rerank，在 recall 不变前提下最大化 QPS**

> **注意**：Phase 1 已包含基本 rerank（因为 4-bit 量化必然降低 recall）。
> Phase 2 的重点是 rerank 策略的**精细调优**，不是从零开始加 rerank。

#### 2.1 搜索流程

```
greedy_search(query, ef_search):
  // Phase 1: 用 LUT16-ADC 距离做图导航（快但粗略）
  candidates = graph_walk(entry_point, ef_search, adc_distance)

  // Phase 2: 对 top-N 候选用 f32 精确距离 rerank
  reranked = top_n_candidates.sort_by(|a, b| l2_f32(query, a).cmp(l2_f32(query, b)))

  return reranked[0..k]
```

#### 2.2 关键参数

- `ef_search`：图搜索宽度，ADC 路径下可以加大（因为 ADC 快 3-5x）
- `top_n`：rerank 候选数，设计文档已有 `top_n >= k` 规则校验
- `rerank_strategy`：全量 rerank vs 增量 rerank

#### 2.3 图导航用 ADC 还是 f32？

这是一个需要实验验证的关键决策点：
- **方案 A**：图导航用 ADC 距离（更快但可能走错路 → recall 下降）
- **方案 B**：图导航用 f32，仅最终 rerank 用 ADC（无意义，更慢）
- **方案 C**：图导航用 ADC，但 ef_search 加大到补偿 recall 损失

顶尖库用的是方案 C——ADC 导航 + 大 ef + f32 rerank。

---

### Phase 3：图质量与内存布局优化（3-5 天）

> **前提条件**：Phase 1 完成后，必须先用 `perf stat` 测 LLC miss 和带宽利用率。
> 如果 Phase 1 已经把距离计算从 compute-bound 拉到 memory-bound，Phase 3 的 cache 优化收益可能大幅衰减。
> 不要预设 +30-50% 还在——根据实测数据决定是否投入。

#### 3.1 图节点重排序（Cache Locality Optimization）

**原理**：当前节点 ID 是随机的，图遍历时访问 `vectors[node_id * dim]` 会在 493MB 的向量数据中随机跳跃，cache miss 率极高。

**方案**：按 BFS 遍历顺序重排节点 ID，使图遍历的内存访问模式变为顺序访问。

**预期收益**：待 Phase 1 后用 perf stat 实测决定。

#### 3.2 PQ codes 连续存储

将所有节点的 PQ codes 存储为一个连续的 `Vec<u8>`（N × M 字节），而非 `Vec<Vec<u8>>`。

#### 3.3 图质量提升

| 优化 | 当前 | 目标 | 对 QPS 的影响 |
|:--|:--|:--|:--|
| 初始图用 NN-guided | 随机图 | 用少量近邻引导 | 更少迭代收敛 |
| r_max 自适应 | 固定 64 | 按数据集自动选择 | 减少无效距离计算 |
| ef_build 加大 | 200 | 400 | 图质量↑ → ef_search↓ → QPS↑ |

---

### Phase 4：搜索热路径微优化（2-3 天）

> **收益可能衰减**——如果 Phase 1 后瓶颈已变为 memory-bound，微优化的边际收益有限。

#### 4.1 BinaryHeap 优化

- 用 `BinaryHeap<u64>` 打包 `(distance_bits << 32 | node_id)`，减少比较开销

#### 4.2 VisitedTracker 优化

- 可考虑 `Vec<u64>` + bitmap

#### 4.3 预取策略调优

ADC 路径下预取 PQ codes（4-bit vs 512 字节 f32）的 cache 影响完全不同。

---

### Phase 5：多数据集适配与参数自动调优（2-3 天）

#### 5.1 ann-benchmarks 全数据集支持

| 数据集 | dim | N | 距离 |
|:--|:--|:--|:--|
| SIFT-128 | 128 | 1M | L2 |
| GIST-960 | 960 | 1M | L2 |
| GloVe-100 | 100 | 1.2M | L2 |

#### 5.2 自动参数选择

每个数据集需要不同的 `(r_max, α, ef_build, M, ef_search, top_n)` 组合。

#### 5.3 ann-benchmarks wrapper 修复（与 Phase 0.1 关联）

- 确保单线程口径与 leaderboard 一致
- 多线程查询需单独标注，不得与单线程混用

---

### Phase 6：极致工程化（2-3 天）

#### 6.1 编译优化

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
opt-level = 3
target-cpu = native  # 启用 AVX-512
```

#### 6.2 Profile-guided Optimization (PGO)

**预期收益**：QPS +5-10%

#### 6.3 NUMA 亲和性

---

## 四、预期最终性能

> **注意**：以下目标建立在"干净环境真实基线"之上。如果当前 QPS 本身是被污染的假数字，"提升 N 倍"的叙事不成立。
> 必须先建立干净基线，再设目标。

### 4.1 SIFT1M 预期 Pareto 前沿

| recall@10 | 当前 QPS（待测） | 目标 QPS | vs 榜首外包络 |
|:--|:--|:--|:--|
| 0.90 | 待测 | 待定 | 目标：持平或超越 |
| 0.95 | 待测 | 待定 | 目标：持平或超越 |
| 0.99 | 待测 | 待定 | 目标：持平或超越 |

### 4.2 打榜 vs 论文：明确拆分

**打榜只认 recall-QPS 曲线那一根线。** 以下拆分避免"差异化优势"模糊"这些东西能不能帮你上榜"的问题。

#### 打榜要靠的（纯吞吐优化）

- Phase 1: LUT16 SIMD PQ-ADC（核心突破点）
- Phase 2: 两阶段搜索 rerank 精细调优
- Phase 3: 内存布局优化（如果 perf stat 证明还有 cache 空间）
- Phase 4: 热路径微优化
- Phase 6: 编译优化 / PGO

#### 论文要靠的（机制证据，不计入打榜分数）

- **RP-Tuning**：一次构建覆盖整条 Pareto 曲线。打榜不计分，但论文价值高。
- **AVQ 检索感知量化**：比标准 PQ 的 recall 更高。需要消融数据支撑，打榜不认。
- **量化感知剪枝**：β/α 协同调参。需要消融数据，打榜不认。
- **Rust 安全性 + 确定性构建**：工程价值，打榜不认，审稿人认。

---

## 五、优先级排序与时间线

> **收益不可简单连乘。** Phase 1（ADC SIMD）、Phase 3（内存布局）、Phase 3.2（PQ codes 连续存储）针对的都是同一个瓶颈——访存带宽和 cache miss。
> 一旦 Phase 1 把距离计算从 compute-bound 拉到 memory-bound，Phase 3/4 的边际收益会快速衰减。
> 真实情况更可能：Phase 1 吃掉大部分收益，Phase 3/4 的边际收益递减。
> **Phase 1 完成后立刻用 perf stat 测 LLC miss 和带宽利用率，再决定 Phase 3/4 是否还值得投入。**

| 优先级 | Phase | 预期 QPS 提升 | 时间 | 风险 |
|:--|:--|:--|:--|:--|
| 🔴 P0 | Phase 0: 修复地基 | 0% (质量) | 1-2天 | 🟢 低 |
| 🔴 P0 | **Phase 1: LUT16 SIMD PQ-ADC** | **主要收益（待验证）** | 3-5天 | 🟡 中 |
| 🟡 P1 | Phase 2: 两阶段 rerank 精调 | 边际收益（待验证） | 2-3天 | 🟡 中 |
| 🟡 P1 | Phase 3: 内存布局 | **可能衰减**（取决于 Phase 1 后的瓶颈分析） | 3-5天 | 🟢 低 |
| 🟢 P2 | Phase 4: 热路径微优化 | **可能衰减** | 2-3天 | 🟢 低 |
| 🟢 P2 | Phase 5: 多数据集 | 扩展覆盖 | 2-3天 | 🟡 中 |
| 🟢 P3 | Phase 6: PGO/NUMA | +5-10% | 2-3天 | 🟢 低 |

**总计**：约 15-24 天。具体 QPS 提升待 Phase 1 完成后根据实测数据重新估算。

---

## 六、核心技术风险与对策

| 风险 | 概率 | 影响 | 对策 |
|:--|:--|:--|:--|
| 4-bit 量化导致 recall 下降 | **高** | 高 | Phase 1 内含 rerank 补偿；加大 ef_search；AVQ 优化 codebook 质量 |
| ~~AVX-512 gather 不如预期~~ | ~~低~~ → **已放弃 gather** | - | 改用 LUT16-shuffle（pshufb + maddubs），不碰内存 |
| Phase 3 cache 优化收益衰减 | **高** | 中 | Phase 1 后用 perf stat 测 LLC miss，根据数据决定是否投入 |
| ann-benchmarks Docker 环境差异 | 中 | 中 | 在与 leaderboard 相同的硬件和线程口径上验证 |
| 榜首算法不是 ScaNN（瞄错靶子） | 中 | 中 | 拉取真实排行榜，目标设为所有库的外包络线 |

---

## 七、附录：审计发现优先修复清单

> 按"打榜影响 × 修复成本"排序

### 第一优先：直接影响 benchmark 结果

1. **建立干净基线** → 确认无后台进程，重测 QPS/recall/建图时间
2. **S1 修复 wrapper** → `fit()` 用 `--save`，`query()` 用 `--load`（否则 QPS 数据全错）
3. **拉取真实排行榜** → 从 ann-benchmarks results 仓库获取 SIFT1M Pareto 数据
4. **Phase 1 LUT16 SIMD PQ-ADC** → 核心性能突破点

### 第二优先：代码质量与可维护性

5. **S4 RP-Tuning B/C** → 标注 `unimplemented!()` 或实现
6. **M1 `eprintln!` → `tracing`** → 热路径零 I/O
7. **M2 `unwrap()` → `?`** → 核心库健壮性

### 第三优先：文档诚实度

8. **S2 GEMM 标为实验性** → 避免暗示已完整支持
9. **S3 big-ann 标为未实现** → 避免暗示已完整支持
10. **S5 维度分发说明** → 文档注明"运行时分发，非编译期特化"

---

## 八、一句话总结

**RAVEN 距离世界第一的核心差距不在图算法（已足够好），而在搜索热路径的距离计算方式：当前用 f32 全精度算距离，而顶尖库用 LUT16-shuffle（pshufb + maddubs）加速的 PQ-ADC 实现了 3-5 倍的吞吐优势。补上这个差距（Phase 1，采用 4-bit PQ + 寄存器内查表，放弃 gather），再叠加 rerank 补偿 recall，RAVEN 有实力在 SIFT1M 上挑战 ann-benchmarks Pareto 前沿。但在此之前，必须先建立干净环境下的真实基线（排除 CPU 被后台进程抢占导致的污染数据），并修复 ann-benchmarks wrapper 每次重建索引的致命 bug（S1），否则一切性能数据都是假的。**

---

*文档结束*
