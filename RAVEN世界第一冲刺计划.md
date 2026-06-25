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

---

## 实验记录

### 基线数据（BEFORE）

> 所有优化的对比基准。在开始任何修改前先建立。

| 指标 | 数值 | 备注 |
|:--|:--|:--|
| cargo test | **155 passed, 0 failed** | 全部通过 |
| cargo clippy | **34 errors** | 已存在问题（needless_range_loop, module_inception 等） |
| recall@10 | **0.9517** | SIFT1M quick_recall_check (α=1.2, l_build=100, r_max=32, ef=100) |
| QPS | **1,287** | SIFT1M quick_recall_check |
| 建图时间 | **1,054.8s** | SIFT1M quick_recall_check |

**额外修复（FIX-0）**：`init_random_graph` 死循环 bug
- **问题**：`neighbor_count = config.r_max`（默认 64），当 n < r_max+1 时，while 循环永远无法凑够邻居数，导致死循环。`crc_corruption_detected` 和 `magic_mismatch_detected` 两个测试因此卡死。
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

| 工作点 | recall@10 | QPS | 路径 |
|:--|:--|:--|:--|
| α=1.2, r_max=64, ef=100 | 0.9961 | 2,434 | f32 全精度 |
| α=1.0, r_max=64, ef=50 | 0.9275 | 7,611 | f32 全精度 |
| α=1.2, ef=100, ADC+rerank | 0.9676 | 2,025 | AVQ 量化 |

### 1.2 世界第一 ScaNN 在 SIFT1M 上的大致性能（公开数据估算）

| 工作点 | recall@10 | QPS | 备注 |
|:--|:--|:--|:--|
| 高 recall 区 | 0.99 | ~5,000-8,000 | PQ-ADC + rerank |
| 中 recall 区 | 0.95 | ~10,000-15,000 | PQ-ADC |
| 低 recall 区 | 0.90 | ~15,000-20,000+ | PQ-ADC |

### 1.3 差距诊断

| 维度 | RAVEN | ScaNN | 差距倍数 |
|:--|:--|:--|:--|
| recall 0.95 处 QPS | ~4,000 | ~12,000 | **3x** |
| recall 0.99 处 QPS | ~2,400 | ~6,000 | **2.5x** |
| ADC 路径 vs f32 | ADC **更慢** (2025 vs 2434) | ADC 快 3-5x | **根本性缺陷** |
| 建图时间 | 4,754s (r_max=64) | ~200-500s | 10x |

**核心发现：RAVEN 的 ADC 路径比 f32 还慢，这是反直觉的。** ScaNN 之所以称霸，正是因为 PQ-ADC 路径比 f32 快 3-5 倍（1 字节 PQ code vs 4 字节 f32，内存带宽降低 4 倍）。RAVEN 有 AVQ 量化器但搜索时没有用 SIMD lookup table 加速 ADC，而是逐子空间标量计算——这等于浪费了量化的全部优势。

---

## 二、代码审计：已验证的缺陷清单

> 以下每一条均经源码逐行验证，**全部属实**。

### 2.1 严重问题（S 级）：未完整实现 / 不算生产级

#### S1. ann-benchmarks wrapper 每次查询重建索引 ✅ 已验证

**文件**：`ann_benchmarks/algorithms/raven/__init__.py:86` + `src/bin/raven_ann_bench.rs:99`

**证据**：
- `__init__.py` 的 `query()` 方法（第 86-126 行）调用二进制时传递了 `--train` 参数，但**未传递 `--load`**。
- `query_batch()` 方法（第 128-178 行）同样如此。
- `fit()` 方法（第 55-84 行）构建索引后**未传递 `--save`** 保存索引。
- `raven_ann_bench.rs` 第 100 行：`if !load_path.is_empty()` → 由于 `query()` 不传 `--load`，`load_path` 为空，必然走 `else` 分支（第 108-123 行）**重新构建索引**。
- 结论：**每次 `query()` / `query_batch()` 调用都会从训练集重新建图**，完全浪费了 `fit()` 阶段的构建工作。评测 QPS 被建图开销严重污染。

**影响**：benchmark 结果完全不可信，QPS 被建图时间淹没。

---

#### S2. 批量查询 GEMM 是标量回退 ✅ 已验证

**文件**：`src/memory/query_ctx.rs:75` + `src/memory/query_ctx.rs:133`

**证据**：
- 第 75 行注释明确写道：`当前实现为标量回退，GEMM 路径在 Week 3-4 接入`
- 第 133 行注释明确写道：`当前为标量回退实现，Week 3-4 接入真正 GEMM`
- `gemm_path()` 函数（第 137-151 行）实际实现是逐候选调用 `l2_simd`，与 `scalar_simd_path()` 完全相同——**没有任何 GEMM 矩阵乘法**。
- `pack_vectors_aligned()` 做了对齐打包，但打包后的数据只被 `gemm_path()` 逐行遍历，没有 BLAS/matrix-multiply 调用。

**影响**：批量吞吐不算完整实现，无法利用 SIMD batch 距离计算的吞吐优势。

---

#### S3. big-ann / SSD 路径只是预留接口 ✅ 已验证

**文件**：`Cargo.toml:18-19`

**证据**：
- `Cargo.toml` 第 18-19 行：`# big-ann SSD 扩展路径（Week 9+，当前仅预留接口）` + `big_ann = []`
- 全代码库中 `big_ann` feature 仅在 `Cargo.toml`、`lib.rs`、`config/mod.rs` 中被引用，均为 feature gate 声明或注释，**没有任何 SSD/磁盘索引的实现代码**。
- 设计文档中 big-ann 相关描述（约第 499 行后）属于愿景规划。

**影响**：设计文档暗示支持 big-ann，实际无任何实现。

---

#### S4. RP-Tuning B/C 方案未实现，静默回退 ✅ 已验证

**文件**：`src/graph/rp_tuning.rs:118-128`

**证据**：
- 第 116-128 行 `match config.scheme`：
  ```rust
  RPTuningStorageScheme::SchemeA => neighbors.clone(),
  RPTuningStorageScheme::SchemeB | RPTuningStorageScheme::SchemeC => {
      eprintln!("[rp_tuning] 警告：{:?} 未实现，回退到 SchemeA", config.scheme);
      neighbors.clone()  // 直接回退到 SchemeA 的逻辑
  }
  ```
- SchemeB/C 的行为与 SchemeA **完全相同**（都是 `neighbors.clone()`），仅多了一条 `eprintln!` 警告。
- 注释说明"方案 B/C 需要构建期保留候选集，当前未实现"。

**影响**：API 暴露了 SchemeB/C 选项但实际无效，可能误导使用者认为已支持。

---

#### S5. 维度分发是统一 SIMD 分发，非 const-generic 特化 ✅ 已验证

**文件**：`src/distance/dispatch.rs:59-67`

**证据**：
- `select_l2()` 函数（第 59-67 行）：
  ```rust
  pub fn select_l2(dim: usize) -> fn(&[f32], &[f32]) -> f32 {
      match dim {
          64 | 128 | 256 | 384 | 768 | 960 | 1536 => l2_simd,  // 已知维度
          _ => l2_simd,                                        // 未知维度
      }
  }
  ```
- **所有分支返回同一个 `l2_simd` 函数指针**，match 表达式完全是 no-op。
- 注释（第 62-63 行）说明"当前不使用 const generics 特化以避免 binary bloat，l2_simd 运行时自动选择最优 SIMD 核"。
- `dispatch_dim!` 宏虽然定义了 `::<64>()` 等 const-generic 调用形式，但返回的 `DispatchResult` 仅是一个标记枚举，不实际调用特化函数。

**影响**：文档示例暗示有 const-generic 多维特化，实际只有运行时 SIMD 分发（AVX-512 > AVX2 > scalar）。运行时分发本身已足够，但与文档描述有差距。

---

#### S6. 配置默认值自相矛盾 ✅ 已验证

**文件**：`src/config/config.rs:100-125` + `src/config/rules.rs:137-142`

**证据**：
- `Config::default()`（config.rs 第 100-125 行）：
  - `distance: "l2".to_string()` （第 103 行）
  - `avq: true` （第 115 行）
- 规则 `avq_l2_conflict`（rules.rs 第 137-142 行）：
  ```rust
  |cfg| !(cfg.avq && cfg.distance == "l2"),  // avq=true + distance=l2 → 返回 false → 违反
  ```
- 测试 `merge_config_defaults`（config.rs 第 322-326 行）**明确断言默认配置会校验失败**：
  ```rust
  // 默认配置 avq=true + distance=l2 违反 avq_l2_conflict（设计意图）
  let result = merge_config(None, None, false);
  assert!(result.is_err(), "default config should violate avq_l2_conflict");
  ```
- 结论：**`Config::default()` 产生的配置无法通过 `merge_config()` 校验**，调用方必须手动覆盖 `avq=false` 或 `distance="ip"` 才能通过。

**影响**：开箱即用的默认配置是坏的，新用户首次运行必然遇到校验错误。

---

### 2.2 代码质量问题（M/L 级）

#### M1. `eprintln!` 遍布热路径 ✅ 已验证

**证据**（grep 结果，共 40+ 处）：
- `src/quant/avq.rs`：3 处（finetune 日志）
- `src/quant/opq.rs`：3 处（train 日志 + test 日志）
- `src/graph/vamana.rs`：8 处（build 进度日志）
- `src/graph/rp_tuning.rs`：1 处（SchemeB/C 警告）
- `src/bin/raven_ann_bench.rs`：15 处（运行时日志）
- `src/bin/*.rs`：多处实验脚本日志

**影响**：生产构建中 `eprintln!` 会锁 stderr、格式化字符串、I/O 系统调用，在热路径中造成微秒级延迟。应替换为 `tracing`（已作为依赖引入但未使用）。

---

#### M2. `unwrap()` 散布于核心模块 ✅ 已验证

**证据**（grep count，src/ 目录）：
- `src/memory/graph.rs`：6 处
- `src/graph/vamana.rs`：3 处
- `src/quant/opq.rs`：3 处
- `src/build/pipeline.rs`：2 处
- `src/build/metadata.rs`：2 处
- `src/memory/serialize.rs`：1 处
- `src/config/config.rs`：1 处

**影响**：核心库中的 `unwrap()` 在异常输入时会 panic，不符合生产级健壮性要求。应替换为 `?` 错误传播或 `Result` 返回类型。

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

| 编号 | 任务 | 文件 | 原因 |
|:--|:--|:--|:--|
| 0.1 | **修复 ann-benchmarks wrapper**：`fit()` 传 `--save`，`query()`/`query_batch()` 传 `--load` | `__init__.py` + `raven_ann_bench.rs` | S1：每次查询重建索引 |
| 0.2 | `pipeline.rs` final_prune 改用 RobustPrune | `pipeline.rs` | 违反硬约束 |
| 0.3 | `pipeline.rs` max_iterations 改为 2 | `pipeline.rs` | 违反硬约束 |
| 0.4 | `pipeline.rs` quant_aware_prune 接通真实实现 | `pipeline.rs` | 当前是 no-op |
| 0.5 | `delayed_prune.rs` final_prune 改用 RobustPrune | `delayed_prune.rs` | 违反硬约束 |
| 0.6 | `rp_tuning.rs` SchemeB/C 实现或标注为 `unimplemented!()` | `rp_tuning.rs` | S4：静默回退 |
| 0.7 | **修配置默认冲突**：`default()` 中 `avq=false` 或 `distance="ip"` | `config.rs` | S6：默认配置校验失败 |
| 0.8 | 全代码库 `eprintln!` → `tracing` | `avq.rs`, `opq.rs`, `vamana.rs` 等 | M1：热路径 I/O |
| 0.9 | 魔法数字提取为具名常量 | `avq.rs` | 可维护性 |
| 0.10 | `try_into().unwrap()` → `?` 错误传播 | `graph.rs`, `vamana.rs` | M2：健壮性 |
| 0.11 | 清理过时 `#[allow(dead_code)]` | `kernel.rs` | 代码整洁 |
| 0.12 | 明确标注 big-ann / GEMM 为未实现 | `Cargo.toml`, README | S2/S3：文档诚实 |

---

### Phase 1：SIMD 加速 PQ-ADC 距离计算（3-5 天）🔴 最高优先级

**这是决定成败的关键。没有这个，不可能世界第一。**

#### 1.1 原理

当前 ADC 路径对每个候选节点：
1. 取 M 个 PQ codes（M 字节）
2. 对每个子空间，遍历 K=256 个聚类中心做标量 L2 → 极慢

ScaNN 的做法：
1. **查询时预计算** distance lookup table：对每个子空间 m，计算 query 到 256 个中心的 L2 距离 → M×256 个 float（SIFT1M: 8×256 = 2KB）
2. **候选计算**只需 M 次 table lookup + 求和 → 8 次内存访问 vs 128 次 f32 乘加

#### 1.2 实现方案

```
adc_distance(query, pq_codes[M]):
  // 预计算（每查询一次）
  for m in 0..M:
    for k in 0..256:
      lut[m][k] = l2_sq(query_sub[m], centroid[m][k])
  
  // 候选距离（每候选一次）
  dist = 0
  for m in 0..M:
    dist += lut[m][pq_codes[m]]   // 单次 lookup
  return dist
```

#### 1.3 SIMD 加速

**AVX-512 路径**（16 个 float 同时处理）：
- 将 M 个子空间的 lookup table 交错排列
- 用 `_mm512_i32gather_ps` 指令并行 gather 16 个候选的距离
- 或用 `_mm256_maddubs_epi16` 做 PQ code → table lookup 的 SIMD 加速

**AVX2 路径**（8 个 float）：
- `_mm256_i32gather_ps` 做 8-wide gather

**关键优化**：将 lookup table 转为 `u8` 定点（255 级量化），用 `_mm256_maddubs_epi16` 一次处理 32 个子空间的 lookup，性能再翻倍。

#### 1.4 预期收益

| 路径 | 当前 QPS | 预期 QPS | 加速比 |
|:--|:--|:--|:--|
| f32 全精度 (ef=100) | 2,434 | 2,434 | 1x (不变) |
| ADC 标量 (当前) | 2,025 | - | - |
| **ADC SIMD (目标)** | - | **8,000-12,000** | **4-5x** |
| ADC SIMD + rerank | - | **5,000-8,000** | **2.5-3x** |

---

### Phase 2：两阶段搜索管道优化（2-3 天）

**目标：ADC 快速粗筛 → f32 精确 rerank，在 recall 不变前提下最大化 QPS**

#### 2.1 搜索流程

```
greedy_search(query, ef_search):
  // Phase 1: 用 PQ-ADC 距离做图导航（快但粗略）
  candidates = graph_walk(entry_point, ef_search, adc_distance)
  
  // Phase 2: 对 top-N 候选用 f32 精确距离 rerank
  reranked = top_n_candidates.sort_by(|a, b| l2_f32(query, a).cmp(l2_f32(query, b)))
  
  return reranked[0..k]
```

#### 2.2 关键参数

- `ef_search`：图搜索宽度，ADC 路径下可以加大（因为 ADC 快 4-5x）
- `top_n`：rerank 候选数，设计文档已有 `top_n >= k` 规则校验
- `rerank_strategy`：全量 rerank vs 增量 rerank

#### 2.3 图导航用 ADC 还是 f32？

这是一个需要实验验证的关键决策点：
- **方案 A**：图导航用 ADC 距离（更快但可能走错路 → recall 下降）
- **方案 B**：图导航用 f32，仅最终 rerank 用 ADC（无意义，更慢）
- **方案 C**：图导航用 ADC，但 ef_search 加大到补偿 recall 损失

ScaNN 用的是方案 C——ADC 导航 + 大 ef + f32 rerank。

#### 2.4 预期 Pareto 前沿

| recall@10 | ef_search | top_n | 预期 QPS |
|:--|:--|:--|:--|
| 0.90 | 50 | 50 | 15,000+ |
| 0.95 | 100 | 100 | 8,000-12,000 |
| 0.99 | 200 | 200 | 3,000-5,000 |
| 0.995 | 400 | 400 | 1,500-2,500 |

---

### Phase 3：图质量与内存布局优化（3-5 天）

#### 3.1 图节点重排序（Cache Locality Optimization）

**原理**：当前节点 ID 是随机的，图遍历时访问 `vectors[node_id * dim]` 会在 493MB 的向量数据中随机跳跃，cache miss 率极高。

**方案**：按 BFS 遍历顺序重排节点 ID，使图遍历的内存访问模式变为顺序访问。

```
1. 从 entry_point 开始 BFS 遍历整个图
2. 按访问顺序分配新 node_id
3. 重排 vectors 和 graph storage
4. 效果：L2/L3 cache miss 率降低 50-70%
```

**预期收益**：QPS +30-50%（纯内存布局优化，recall 不变）

#### 3.2 PQ codes 连续存储

将所有节点的 PQ codes 存储为一个连续的 `Vec<u8>`（N × M 字节），而非 `Vec<Vec<u8>>`。

- SIFT1M, M=8: 8MB 连续数组，完全在 L2 cache 内
- 当前 `Vec<Vec<u8>>` 每次访问有指针跳转

#### 3.3 图质量提升

| 优化 | 当前 | 目标 | 对 QPS 的影响 |
|:--|:--|:--|:--|
| 初始图用 NN-guided | 随机图 | 用少量近邻引导 | 更少迭代收敛 |
| r_max 自适应 | 固定 64 | 按数据集自动选择 | 减少无效距离计算 |
| ef_build 加大 | 200 | 400 | 图质量↑ → ef_search↓ → QPS↑ |

---

### Phase 4：搜索热路径微优化（2-3 天）

#### 4.1 BinaryHeap 优化

当前 BinaryHeap 存储 `(OrderedF32, u32)` = 8 字节/元素。优化方向：
- 用 `BinaryHeap<u64>` 打包 `(distance_bits << 32 | node_id)`，减少比较开销
- 或用 `BinaryHeap<(u32, u32)` 将 distance 量化为 u32（乘以大常数后截断）

#### 4.2 VisitedTracker 优化

- 当前 `Vec<u8>` 已是最优
- 可考虑 `Vec<u64>` + bitmap，一次检查 8 个节点
- 但 Clear-List 重置开销需评估

#### 4.3 预取策略调优

OPT-2 已验证方案 B（预取堆顶邻居列表）最优，但需在 ADC 路径下重新调优：
- ADC 路径下向量访问模式不同（读 PQ codes 而非 f32 向量）
- 预取 PQ codes（8 字节 vs 512 字节）的 cache 影响完全不同

#### 4.4 分支消除

- `if !visited[node]` → 可用位运算无分支版本
- `if result.len() >= r_max` → 循环上界固定时可消除

---

### Phase 5：多数据集适配与参数自动调优（2-3 天）

#### 5.1 ann-benchmarks 全数据集支持

| 数据集 | dim | N | 距离 | 特殊要求 |
|:--|:--|:--|:--|:--|
| SIFT-128 | 128 | 1M | L2 | 当前主战场 |
| GIST-960 | 960 | 1M | L2 | 高维，PQ 子空间需调整 |
| GloVe-100 | 100 | 1.2M | L2 | 中维 |
| GloVe-200 | 200 | 1.2M | L2 | 中高维 |
| NYTimes-256 | 256 | 290K | L2 | 中维 |
| Fashion-MNIST | 784 | 60K | L2 | 小数据集 |

#### 5.2 自动参数选择

每个数据集需要不同的 `(r_max, α, ef_build, M, ef_search, top_n)` 组合：
- 实现自动参数扫描脚本
- 对每个数据集跑 Pareto 扫描
- 选择 Pareto 前沿最优参数

#### 5.3 ann-benchmarks wrapper 修复（与 Phase 0.1 关联）

- 修复 `query()` 单查询路径（当前每次重建索引）
- 确保 Docker 环境可复现
- 支持多线程查询（`use_threads()` 返回 True → QPS × 核数）

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

1. 用代表性查询跑 instrumented build
2. 收集 profile 数据
3. 重新编译，LLVM 根据 profile 优化分支预测和内联

**预期收益**：QPS +5-10%

#### 6.3 NUMA 亲和性

- 单线程查询绑定到特定核
- 避免 NUMA 跨节点内存访问
- `numactl --cpunodebind=0 --membind=0`

---

## 四、预期最终性能

### 4.1 SIFT1M 预期 Pareto 前沿

| recall@10 | 当前 QPS | 目标 QPS | vs ScaNN |
|:--|:--|:--|:--|
| 0.90 | 7,611 | 20,000+ | **超越** |
| 0.95 | ~4,000 | 12,000-15,000 | **持平/超越** |
| 0.99 | 2,434 | 5,000-8,000 | **持平** |
| 0.995 | ~1,200 | 2,500-4,000 | **持平** |

### 4.2 RAVEN 的差异化优势

如果上述优化完成，RAVEN 将在以下维度超越 ScaNN：

1. **RP-Tuning**：一次构建覆盖整条 Pareto 曲线（ScaNN 需要重建）
2. **AVQ 检索感知量化**：比标准 PQ 的 recall 更高（同 QPS 下）
3. **Rust 安全性**：无内存安全漏洞，适合生产部署
4. **确定性构建**：ChaCha8 + 固定种子，完全可复现

---

## 五、优先级排序与时间线

| 优先级 | Phase | 预期 QPS 提升 | 时间 | 风险 |
|:--|:--|:--|:--|:--|
| 🔴 P0 | Phase 0: 修复地基 | 0% (质量) | 1-2天 | 🟢 低 |
| 🔴 P0 | **Phase 1: SIMD PQ-ADC** | **+300-500%** | 3-5天 | 🟡 中 |
| 🟡 P1 | Phase 2: 两阶段搜索 | +50-100% | 2-3天 | 🟡 中 |
| 🟡 P1 | Phase 3: 内存布局 | +30-50% | 3-5天 | 🟢 低 |
| 🟢 P2 | Phase 4: 热路径微优化 | +10-20% | 2-3天 | 🟢 低 |
| 🟢 P2 | Phase 5: 多数据集 | 扩展覆盖 | 2-3天 | 🟡 中 |
| 🟢 P3 | Phase 6: PGO/NUMA | +5-10% | 2-3天 | 🟢 低 |

**总计**：约 15-24 天，预期 SIFT1M recall@10=0.95 处从 4,000 QPS 提升到 12,000+ QPS。

---

## 六、核心技术风险与对策

| 风险 | 概率 | 影响 | 对策 |
|:--|:--|:--|:--|
| ADC 距离精度不足导致 recall 暴跌 | 中 | 高 | 加大 ef_search 补偿；AVQ 优化 codebook 质量 |
| AVX-512 gather 指令实际不如预期 | 低 | 高 | 回退到 AVX2 + 手动 lookup table 交错 |
| 图重排序后 cache 提升不如预期 | 低 | 中 | 先用 `perf stat` 测 LLC miss 率再决定 |
| ann-benchmarks Docker 环境差异 | 中 | 中 | 在与 leaderboard 相同的 AWS c6i 实例上验证 |
| ScaNN 在某些数据集有不可逾越的优势 | 中 | 中 | 聚焦 SIFT1M 先拿一个数据集冠军 |

---

## 七、附录：审计发现优先修复清单

> 按"打榜影响 × 修复成本"排序

### 第一优先：直接影响 benchmark 结果

1. **S1 修复 wrapper** → `fit()` 用 `--save`，`query()` 用 `--load`（否则 QPS 数据全错）
2. **S6 修默认配置** → `default()` 中 `avq=false`（否则首次运行报错）
3. **Phase 1 SIMD PQ-ADC** → 核心性能突破点

### 第二优先：代码质量与可维护性

4. **S4 RP-Tuning B/C** → 标注 `unimplemented!()` 或实现
5. **M1 `eprintln!` → `tracing`** → 热路径零 I/O
6. **M2 `unwrap()` → `?`** → 核心库健壮性

### 第三优先：文档诚实度

7. **S2 GEMM 标为实验性** → 避免暗示已完整支持
8. **S3 big-ann 标为未实现** → 避免暗示已完整支持
9. **S5 维度分发说明** → 文档注明"运行时分发，非编译期特化"

---

## 八、一句话总结

**RAVEN 距离世界第一的核心差距不在图算法（已足够好），而在搜索热路径的距离计算方式：当前用 f32 全精度算距离，而 ScaNN 用 SIMD 加速的 PQ-ADC 实现了 4-5 倍的吞吐优势。补上这个差距（Phase 1），再叠加内存布局优化（Phase 3）和两阶段搜索（Phase 2），RAVEN 有实力在 SIFT1M 上挑战 ann-benchmarks 第一名。但在此之前，必须先修复 ann-benchmarks wrapper 每次重建索引的致命 bug（S1），否则一切性能数据都是假的。**

---

*文档结束*
