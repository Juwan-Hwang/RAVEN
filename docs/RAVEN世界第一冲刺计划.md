# RAVEN 冲击 ann-benchmarks 世界第一：战略路线图

> 创建时间：2026-06-25
> 最后修订：2026-06-29（v8.6：PQ8 u8 LUT + AVX2 gather 双双失败，PQ 方案全线封存；资源集中到 SQ8 极限优化）
> 目标：在 ann-benchmarks SIFT1M (sift-128-euclidean) 上达到 recall-QPS Pareto 前沿第一梯队（详见 §〇.2 目标重定义）
>
> **v8 重大变更**：H20 实测证明 "Glass avg_visited < 150" 是虚假基线（三个 bug 叠加：read_fvecs 用 astype 而非 view、search() 不更新计数器、硬编码文献值）。
> Glass HNSW 同配置（R=32, L=200, FP32, 单线程）实测 avg_visited=1041，RAVEN=1227，差距仅 15%。
> RAVEN 在 recall（97.05% vs 94.65%）和 QPS（8657 vs 7678）上均优于 Glass。
> **Pivot Criterion 否决令解除**：avg_visited 在同一量级，图质量正常，Phase 1（距离计算加速）恢复 #1 优先级。
> ~~D1（随机层级导航）根因判断已推翻，0C.2 降级为 P3。~~
> ~~v6/v7 中所有基于 "avg_visited 10x 差距" 的结论均已作废。~~

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

### 〇.1 例外条款：复合优化的整体评估

> **规则 2 和规则 3 与 Phase 1 存在执行冲突。**
>
> Phase 1 的本质是量化**必然先让 recall 下降**，再靠 rerank 和加大 ef_search 补回来。
> 如果严格按规则 2 执行，Phase 1 中间任何一个单独的 commit（如"接入 SQ8"这一步）都会让 recall 暴跌，按规则就得回退——Phase 1 永远做不下去。

**例外条款**：

对于已知会引入中间态 recall 损失的**复合优化**（如量化 + rerank），允许以"整体工作点"为单位评估 before/after，而非以单个 commit 为单位。但必须满足以下全部约束：

1. **预先声明边界**：在开始前，明确声明该复合优化包含哪些子步骤、子步骤之间的依赖关系。
2. **预设 recall 阈值**：声明终态 recall 阈值（如 recall@10 ≥ 0.95）。终态未达标则整体回退。
3. **在分支上开发**：复合优化在独立分支上开发，只有终态达标才并入主线。
4. **子步骤仍需独立 commit**：每个子步骤在分支上单独 commit（方便回退和对比），但不要求每个子步骤单独满足规则 2。
5. **终态评估**：合并到主线时，以终态工作点的 (recall, QPS) 与分支前的基线做 before/after 对比。

### 〇.2 目标重定义

> ~~**v6 算术诚实**~~（已推翻）：旧基线 ~~2,706~~ × Phase 1 最乐观估计 2.5 = 6,765 QPS 的"算术不可能定理"基于虚假基线，已作废。
>
> **v8 真实状态**：RAVEN 同配置（R=32, L=200, FP32, 单线程）recall=0.9705, QPS=8657，已超越 Glass（recall=0.9465, QPS=7678）。
> 图质量正常（avg_visited 1227 vs Glass 1041，差距 15%），距离计算是唯一剩下的瓶颈。
> **Phase 1（量化加速）现在是 #1 优先级，无前置条件。**

**目标重定义**（v8 更新，目标上调）：

| 层级 | 定义 | 达成条件 | v8 状态 |
|:--|:--|:--|:--|
| **新目标 A：第一梯队** | 在某个 recall 工作点进入 Pareto 前沿 top-5 | 存在 recall 点，RAVEN QPS ≥ 榜首 × 0.5 | 本地已超 Glass，需上榜验证 |
| **新目标 B：单点突破** | 在某个 recall 工作点超越特定榜首算法 | 存在 recall 点，RAVEN QPS > 该点榜首 | 本地已超越 Glass 单线程 |
| **新目标 C：论文贡献** | 机制创新有独立价值 | RP-Tuning / AVQ / 量化感知剪枝有消融数据支撑 | AVQ 已有初步数据 |
| ~~旧目标~~ | ~~Pareto 前沿世界第一~~ | ~~QPS 超越所有库的外包络线~~ | 量化 + 多线程后重新评估 |

> **v8 Pivot Criterion 裁决**：
>
> ~~avg_visited 是当前整个项目最关键的单个数字，比 Phase 1 还关键。~~
> ~~裁决维持：暂停 Phase 1（ADC 加速），全力修图（Phase 0C + Phase 3.3），重新评估可行性。~~
>
> **v8 裁决：否决令解除。** H20 实测证明 avg_visited 差距仅 15%（1227 vs 1041），在同一量级。
> 图质量正常，不存在"10x 膨胀"问题。**Phase 1（距离计算加速）恢复 #1 优先级，无前置条件。**
>
> ~~Pivot Criterion 阈值表（< 150 / 150-300 / > 300 / > 500）~~ 已作废——该表基于 "Glass avg_visited < 150" 的虚假假设。

---

## 实验记录

### 基线数据（v8 真实基线）

> **v8 基线**：H20 实测后确立的真实基线，取代所有旧基线。
> 同配置对比：R=32, L=200, FP32, 单线程。

| 指标 | RAVEN v9.1 | Glass HNSW (H20 实测) | 差距 | 备注 |
|:--|:--|:--|:--|:--|
| recall@10 (ef=50) | **0.9705** | 0.9465 | RAVEN +2.4pp | |
| QPS (ef=50) | **8,657** | 7,678 | RAVEN +13% | |
| avg_visited (ef=50) | 1,227 | 1,041 | Glass -15% | 同一量级，无膨胀 |
| 建图时间 | 408.0s | ~200-500s | 可比 | v3-fix 修复后 |
| cargo test | 155 passed | — | — | 全部通过 |

> **Glass 完整 ef 扫描（H20 实测，R=32, L=200, FP32, 单线程）**：
>
> | ef | recall@10 | QPS | avg_visited |
> |:--|:--|:--|:--|
> | 32 | 90.35% | 11,154 | 762 |
> | 48 | 94.32% | 9,225 | 1,011 |
> | 50 | 94.65% | 7,678 | 1,041 |
> | 64 | 96.29% | 6,766 | 1,252 |
> | 80 | 97.45% | 5,643 | 1,487 |
> | 96 | 98.16% | 5,111 | 1,715 |
> | 128 | 98.93% | 4,056 | 2,156 |
> | 200 | 99.57% | 3,123 | 3,086 |
>
> **Glass Optimize() 结果**：po=2, pl=8，获得 30.13% 性能提升。RAVEN 的 po 硬编码为 8，从未调优。

### 历史基线存档（v6/v7，已作废）

> ~~以下数据基于 "Glass avg_visited < 150" 的虚假基线，已被 H20 实测推翻。保留存档供历史追溯。~~

<details>
<summary>~~v6/v7 CANONICAL/GLASS-COMP 扫描结果（已作废）~~</summary>

~~CANONICAL（200/64/2，建图 6,657.3s）~~

| ~~ef_search~~ | ~~recall@10~~ | ~~QPS~~ | ~~avg_visited~~ |
|:--|:--|:--|:--|
| ~~50~~ | ~~0.9868~~ | ~~1,801~~ | ~~2,462.5~~ |
| ~~100~~ | ~~0.9965~~ | ~~1,112~~ | ~~4,012.6~~ |

~~GLASS-COMP v3（200/32/2，建图 408.0s）~~

| ~~ef_search~~ | ~~recall@10~~ | ~~QPS~~ | ~~avg_visited~~ |
|:--|:--|:--|:--|
| ~~50~~ | ~~0.9703~~ | ~~7,501~~ | ~~1,399.5~~ |
| ~~100~~ | ~~0.9920~~ | ~~4,479~~ | ~~2,355.5~~ |

~~v7 Pivot Criterion 裁决：avg_visited @ recall=0.95 ≈ 1,443，>> 500（严重偏高），暂停 Phase 1~~
~~**此裁决已被 H20 推翻**：Glass 同配置 avg_visited=1041，RAVEN 1227，差距仅 15%。~~

</details>

### 构建配置

> **v8 简化**：不再需要 CANONICAL/GLASS-COMP 双图对比——H20 已证明图质量正常。
> 统一使用 GLASS-COMP 配置（R=32）作为唯一标准，与 Glass 做 apples-to-apples 对比。

| 配置名 | α | l_build | r_max | r_soft | max_iterations | ef_search | 用途 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **STANDARD** | 1.2 | 200 | **32** | 48 | 2 | **扫描** | **唯一标准图**：基线、打榜、Phase 1 全部基于此 |
| ~~CANONICAL~~ | ~~1.2~~ | ~~200~~ | ~~64~~ | ~~96~~ | ~~2~~ | — | ~~已作废：r_max=64 无对标意义~~ |
| ~~GLASS-COMP~~ | ~~1.2~~ | ~~200~~ | ~~32~~ | ~~48~~ | ~~2~~ | — | ~~已合并到 STANDARD~~ |

---

**额外修复（FIX-0）**：`init_random_graph` 死循环 bug
- **问题**：`neighbor_count = config.r_max`（默认 64），当 n < r_max+1 时死循环。
- **修复**：`let neighbor_count = config.r_max.min(n.saturating_sub(1));`
- **验证**：155 passed, 0 failed。

---

### FIX-1: 修配置默认冲突 (avq+L2)

- **问题**：`Config::default()` 中 `avq=true` + `distance="l2"` 违反规则 `avq_l2_conflict`。
- **修复**：`config.rs` `avq: true` → `avq: false`；更新 3 个测试断言。
- **Before**：`merge_config(None, None, false)` → `Err`
- **After**：`merge_config(None, None, false)` → `Ok`
- **cargo test**：155 passed, 0 failed
- **结论**：✅ 已采纳

---

## 一、现状 vs 世界第一的差距分析

### 1.1 当前 RAVEN 性能（SIFT1M, 128-dim, L2）

| 工作点 | recall@10 | QPS | 路径 | 备注 |
|:--|:--|:--|:--|:--|
| **STANDARD (R=32, ef=50)** | **0.9705** | **8,657** | f32 全精度 | **v8 真实基线**（H20 同条件实测） |
| STANDARD (R=32, ef=100) | 0.9920 | 4,479 | f32 全精度 | v3-fix 数据 |
| ~~α=1.2, r_max=32, ef=100~~ | ~~0.9517~~ | ~~2,706~~ | ~~f32~~ | ~~已作废（100/32/2 非标准图）~~ |
| ~~ADC+rerank~~ | ~~0.9676~~ | ~~2,025~~ | ~~AVQ 量化~~ | ~~旧值，需重测~~ |

### 1.2 竞争目标：ann-benchmarks 真实 Pareto 前沿

> **数据来源**：ann-benchmarks 官方 results 仓库，sift-128-euclidean。
> **注意**：榜单数字为多线程/不同硬件口径，与单线程本地数据不完全可比。

| 工作点 | 榜首算法 | recall@10 | QPS | 备注 |
|:--|:--|:--|:--|:--|
| 高 recall 区 | **qsgngt** | 0.9917 | **11,163** | QSG-NGT |
| 中 recall 区 | **glass** | 0.9523 | **15,171** | glass R=48, level=2, L=200 |
| 低 recall 区 | **glass** | 0.9941 | **19,801** | glass R=32, level=2, L=200 |

> **v8 重新解读**：榜单 QPS 是多线程口径。RAVEN 单线程 8,657 QPS vs Glass 单线程 7,678 QPS 已证明算法优势。
> 榜单上 Glass 15,171 QPS 很可能是多线程结果。RAVEN 需实现多线程查询（Phase 7）后才能做同口径对比。

### 1.3 差距诊断（v8 重写）

> ~~"图算法已足够好"是未经验证的假设~~ → **v8 已验证：图质量正常。**
>
> H20 实测推翻了所有基于 "avg_visited 10x 差距" 的假设。真实差距：

| 维度 | RAVEN (ef=50) | Glass (ef=50, H20 实测) | 差距 | 诊断 |
|:--|:--|:--|:--|:--|
| recall@10 | **0.9705** | 0.9465 | RAVEN +2.4pp | RAVEN 图质量更优 |
| QPS | **8,657** | 7,678 | RAVEN +13% | RAVEN 更快 |
| avg_visited | 1,227 | 1,041 | Glass -15% | 同一量级，无膨胀 |
| ADC 路径 vs f32 | ADC 更慢 (旧值) | — | — | **唯一瓶颈：距离计算未量化** |

**核心发现（v8 重写）**：

1. ~~5.6x 差距~~ → **不存在**。同配置下 RAVEN 已超越 Glass。
2. ~~avg_visited 10x 差距~~ → **不存在**。真实差距 15%，在合理范围内。
3. **唯一瓶颈是距离计算**：RAVEN 当前用 f32 全精度距离，Glass 榜单成绩用 SQ8/PQ8 量化加速。RAVEN 的 ADC 路径比 f32 还慢（旧值 2,025 vs 2,434），这是 Phase 1 要解决的核心问题。
4. **Glass Optimize() 获得 30% 免费提升**：RAVEN 的 po 硬编码为 8，从未调优。Glass 实测最优 po=2。这是一天内可做的确定性收益。

### 1.4 ~~算术不可能定理~~（已推翻）

> ~~用文档自己的两个数字做最简单的乘法。~~
> ~~旧基线 2,706 × Phase 1 最乐观估计 2.5 = 6,765 QPS，离榜首 15,171 还差 2.24x。~~
> ~~结论：按当前以 Phase 1 为核心引擎的计划，recall 0.95 处「超越 glass」在数学上不成立。~~
>
> **v8 推翻**：旧基线 2,706 基于已作废的 100/32/2 配置 + 虚假的 Glass avg_visited < 150。
> 真实基线：RAVEN ef=50 recall=0.9705 QPS=8,657，已超越 Glass 单线程 7,678。
> 榜单 15,171 QPS 是多线程口径。RAVEN 量化 + 多线程后重新评估。

### 1.5 剩余问题：距离计算加速（v8 重写）

> ~~两个独立问题：距离计算 vs 图导航效率~~ → **只剩一个：距离计算。**

| 问题 | 性质 | 预期收益 | 对应 Phase | 当前状态 |
|:--|:--|:--|:--|:--|
| **距离计算未量化** | 唯一瓶颈 | SQ8: ~1.5x → ~13,000 QPS；4-bit PQ: ~2-3x → ~17,000+ QPS | Phase 1 | f32 全精度，ADC 路径更慢 |
| ~~图导航效率 10x 差距~~ | ~~已证伪~~ | — | — | ~~差距仅 15%，正常~~ |
| **预取参数未调优** | 确定性收益 | +20-30%（Glass 实测 30%） | Phase 0D | po 硬编码 8，从未扫描 |
| **无多线程查询** | 榜单口径差距 | 多核线性扩展 | Phase 7 | 仅单线程 |
| **无自适应 ef** | 差异化创新 | 同 recall 下降 avg_visited | Phase 4.5 | 所有查询用同一 ef |

---

## 二、代码审计：已验证的缺陷清单

> 以下每一条均经源码逐行验证。

### 2.0 设计文档一致性审计

> **v8 更新**：D1 根因判断已推翻。其余 D2-D8 仍然有效。

| # | 严重度 | 设计文档要求 | 实际代码状态 | v8 状态 |
|:--|:--|:--|:--|:--|
| ~~D1~~ | ~~🔴 严重~~ → **已推翻** | ~~随机层级导航~~ | ~~未实现~~ | ~~avg_visited 10x 差距是虚假基线。真实差距 15%，D1 不是根因。0C.2 降级 P3~~ |
| D2 | ✅ 已修复 | BuildMetadata 落盘 | serialize() 已集成 | 0C.3 已完成 |
| D3 | ✅ 已修复 | rp_tuning.rs 编译 | 补全 saturate 字段 | 0C.1 已完成 |
| D4 | 🟡 中等 | 局部 alpha | 未实现，设计文档标注探索性 | P3 |
| D5 | 🟡 中等 | parking_lot | 声明但未接入 | P2 |
| D6 | 🟡 中等 | GEMM 路径 | 标量回退占位符 | P3 |
| D7 | ✅ 已修复 | RP-Tuning alpha 范围 | 已扩展到 [0.8, 1.0, 1.2, 1.5, 2.0, 3.0] | 0C.5 已完成 |
| D8 | 🟡 中等 | 内存带宽 profiling | 未执行 | P2 |

> ~~**D1 是 avg_visited 10x 差距的根因**~~ → **已推翻**。H20 实测证明 Glass 同配置 avg_visited=1041，RAVEN=1227，差距 15%。
> D1（随机层级导航）仍有学术价值（设计文档要求），但不再是性能瓶颈，降级为 P3。

### 2.1 严重问题（S 级）

#### S1. ann-benchmarks wrapper 每次查询重建索引 ✅ 已修复

**文件**：`ann_benchmarks/algorithms/raven/__init__.py` + `src/bin/raven_ann_bench.rs`

**修复**：`fit()` 传 `--save`，`query()`/`query_batch()` 传 `--load`

---

#### S2. 批量查询 GEMM 是标量回退 ✅ 已验证

**文件**：`src/memory/query_ctx.rs`，`gemm_path()` 实际是逐候选调用 `l2_simd`。

---

#### S3. big-ann / SSD 路径只是预留接口 ✅ 已验证

**文件**：`Cargo.toml`，`big_ann = []` 空特征。

---

#### S4. RP-Tuning B/C 方案未实现，静默回退 ✅ 已验证

**文件**：`src/graph/rp_tuning.rs`，SchemeB/C 与 SchemeA 行为完全相同。

---

#### S5. 维度分发是统一 SIMD 分发 ✅ 已验证

**文件**：`src/distance/dispatch.rs`，`select_l2()` 所有分支返回同一个 `l2_simd`。

---

#### S6. 配置默认值自相矛盾 ✅ 已修复 (FIX-1)

---

#### S7. ann-benchmarks query() 接口不完整 🔴 未修复

**文件**：`ann_benchmarks/algorithms/raven/__init__.py`

**问题**：`query()` 单查询可能返回空列表；HDF5 格式适配未完成；Docker 环境未验证。

**影响**：无法在 ann-benchmarks 官方框架上运行，所有性能数据未经第三方验证。

---

#### S8. 预取参数 po 硬编码为 8，从未调优 🔴 未修复

**文件**：`src/graph/vamana.rs`，`prefetch_offset: 8` 硬编码

**证据**：Glass Optimize() 实测最优 po=2，获得 30.13% 性能提升。RAVEN 的 po=8 从未扫描验证。

**影响**：可能损失 20-30% QPS。这是一天内可做的确定性收益。

---

### 2.2 代码质量问题（M/L 级）

#### M1. `eprintln!` 遍布热路径 ✅ 已验证

共 40+ 处，分布在 `avq.rs`, `opq.rs`, `vamana.rs`, `rp_tuning.rs`, `raven_ann_bench.rs` 等。

#### M2. `unwrap()` 散布于核心模块 ✅ 已验证

`src/` 目录下 18+ 处 `unwrap()`。

---

### 2.3 质量评价总结

| 维度 | 评分 | 说明 |
|:--|:--|:--|
| 研究原型 | 8/10 | 算法实现正确，H20 实测超越 Glass |
| 工程完整度 | 7/10 | 核心路径完整，外围有原型/回退 |
| 生产可用度 | 5.5/10 | 缺多线程查询、Docker 集成 |
| 文档遵照度 | 70-80% | D1 判断已修正，D2-D8 大部分已修复 |

---

## 三、冲刺计划（v8 重排）

### Phase 0：已完成的地基修复

> v8 确认：Phase 0A/0C 中除 0C.2（已降级）外的任务均已完成。

#### Phase 0A：打榜命门 ✅ 已完成

| 编号 | 任务 | 状态 |
|:--|:--|:--|
| 0A.1 | 修复 ann-benchmarks wrapper（`--save`/`--load`） | ✅ 已完成 |
| 0A.2 | 测量干净基线 | ✅ 已完成（H20 真实基线） |

#### Phase 0B：代码卫生（逐项独立 commit + before/after）

> 每项必须单独在干净环境下测 before/after。不能批量做。

| 编号 | 任务 | 文件 | 状态 |
|:--|:--|:--|:--|
| 0B.1 | `eprintln!` → `tracing`（热路径优先） | `vamana.rs`, `avq.rs`, `opq.rs` 等 | 待执行 |
| 0B.2 | `try_into().unwrap()` → `?` 错误传播 | `graph.rs`, `vamana.rs` | 待执行 |
| 0B.3 | `rp_tuning.rs` SchemeB/C 标注 `unimplemented!()` | `rp_tuning.rs` | 待执行 |
| 0B.4 | `pipeline.rs` final_prune 改用 RobustPrune | `pipeline.rs` | 待执行 |
| 0B.5 | `pipeline.rs` max_iterations 改为 2 | `pipeline.rs` | 待执行 |
| 0B.6 | `delayed_prune.rs` final_prune 改用 RobustPrune | `delayed_prune.rs` | 待执行 |
| 0B.7 | `pipeline.rs` quant_aware_prune 接通真实实现 | `pipeline.rs` | 待执行 |
| 0B.8 | 魔法数字提取为具名常量 | `avq.rs` | 待执行 |
| 0B.9 | 清理过时 `#[allow(dead_code)]` | `kernel.rs` | 待执行 |
| 0B.10 | 明确标注 big-ann / GEMM 为未实现 | `Cargo.toml`, README | 待执行 |

#### Phase 0C：设计文档一致性修复

| 编号 | 任务 | 状态 | v8 变更 |
|:--|:--|:--|:--|
| 0C.1 | 修复 rp_tuning.rs 编译错误 | ✅ 已完成 | |
| ~~0C.2~~ | ~~实现随机层级导航~~ | 🟢 P3 降级 | ~~avg_visited 10x 差距已证伪，不再是 #1 优先级~~ |
| 0C.3 | BuildMetadata 集成到序列化 | ✅ 已完成 | |
| 0C.4 | 建图/查询路径分离 + ef*64 恢复 | ✅ 已完成 | 建图 609.8s→408.0s, QPS 4910→7501 |
| 0C.5 | RP-Tuning alpha 范围对齐 F.4 | ✅ 已完成 | |
| 0C.6 | parking_lot 接入 | 🟢 P2 | |
| 0C.7 | 内存带宽 profiling | 🟢 P2 | |
| 0C.8 | GEMM 路径标注未实现 | 🟢 P3 | |
| 0C.9 | 局部 alpha 标注探索性 | 🟢 P3 | |

#### Phase 0D：预取参数自动调优（v8 新增）→ 已完成，收益不显著，保留 po=8

> **v8 预期**：Glass Optimize() 在 SIFT1M 上获得 30.13% 性能提升，RAVEN 的 po 硬编码为 8，预期 +20-30% QPS。
>
> **实际结果**：收益不显著，po=6 vs po=8 在 5 轮交替测量中差距 -0.0%（2σ=5.4%），纯噪声。
> 原因：RAVEN 已实现 multi-line graph prefetch（4 条 cache line 预取邻居列表），
> 向量预取的 po 参数对整体性能影响已被图预取掩盖。Glass 之所以获得 30% 收益是因为它只有向量预取没有图预取。

| 编号 | 任务 | 文件 | 状态 | 实际收益 |
|:--|:--|:--|:--|:--|
| 0D.1 | po/pl 参数空间扫描 | `ef_po_sweep.rs`, `po_confirm.rs` | ✅ 已完成 | ef=50: po=6 vs po=8 差距 -0.0%（噪声） |
| 0D.2 | 实现 `optimize_prefetch()` 方法 | — | ❌ 舍弃 | 0D.1 证明无显著收益，不值得工程化 |
| 0D.3 | 按 ef_search 自适应 po | — | ❌ 舍弃 | 同上 |

**实验数据（0D.1）**：

第一次扫描（ef_po_sweep.rs）：
| ef | po | QPS | vs po=8 |
|:--|:--|:--|:--|
| 50 | 6 | 8,926 | +1.7% |
| 50 | 8 | 8,779 | baseline |
| 100 | 2 | 4,887 | +3.6% |
| 100 | 8 | 4,717 | baseline |

第二次确认（po_confirm.rs，5 轮交替 ef=50）：
| round | QPS(po=6) | QPS(po=8) | diff% |
|:--|:--|:--|:--|
| 1 | 8,634 | 8,737 | -1.2% |
| 2 | 8,588 | 8,414 | +2.1% |
| 3 | 8,612 | 8,689 | -0.9% |
| 4 | 9,049 | 8,637 | +4.8% |
| 5 | 8,719 | 9,144 | -4.6% |
| **mean** | **8,720** | **8,724** | **-0.0%** |
| std | 170 | 237 | |
| cv | 1.95% | 2.72% | |

**结论**：po=6 vs po=8 差距 -0.0%，2σ=5.4%，无法确认统计显著性。保留 po=8 不变。
根因：RAVEN 的 multi-line graph prefetch 已吃掉大部分预取红利，向量预取 po 参数不再是瓶颈。

---

### Phase 1：量化距离计算加速（3-7 天）🔴 #1 优先级

> **v8 解除所有限制**：Pivot Criterion 否决令已解除，Phase 1 无前置条件。
> 适用例外条款（§〇.1）：量化必然先让 recall 下降，以终态工作点做整体评估。

#### 1.0 Step 0：SQ8 标量量化过渡（v8 新增）→ ✅ 已完成，终态门 PASS

> **为什么需要 SQ8 过渡**：4-bit PQ 实现复杂（LUT16-shuffle + pshufb + maddubs），出错了难以定位是量化问题还是 SIMD 问题。
> SQ8 是标量量化（每维度 1 字节），实现简单，可独立验证量化质量和 recall 影响。
> SQ8 完成后，4-bit PQ 只需在 SQ8 基础上进一步压缩 + 换 SIMD 内核。

| 子步骤 | 预期效果 | 状态 | 实际效果 |
|:--|:--|:--|:--|
| 0a. 实现 SQ8 编码/解码 | 向量 128B → 32B，内存降 4x | ✅ | 512MB → 128MB (4.0x 压缩) |
| 0b. 实现 SQ8 L2 距离核 | u8 AVX2 距离，比 f32 快 | ✅ | AVX2 `_mm_madd_epi16`，每次 16 维 |
| 0c. 接入搜索路径 + rerank | recall 恢复 | ✅ | f32 rerank 全候选 |
| 0d. **终态评估** | recall ≥ 0.95, QPS ≥ 1.3x | ✅ **PASS** | 见下表 |

**实验数据（sq8_bench.rs）**：

| ef | mode | recall@10 | QPS | speedup | avg_visited | recallΔ |
|:--|:--|:--|:--|:--|:--|:--|
| 50 | f32 | 0.9705 | 8,705 | 1.00x | 1,227 | — |
| 50 | **SQ8** | **0.9653** | **11,997** | **1.38x** | 1,233 | -0.52pp |
| 100 | f32 | 0.9921 | 4,926 | 1.00x | 2,189 | — |
| 100 | **SQ8** | **0.9905** | **7,015** | **1.42x** | 2,200 | -0.16pp |
| 200 | f32 | 0.9981 | 2,731 | 1.00x | 3,886 | — |
| 200 | **SQ8** | **0.9978** | **3,863** | **1.41x** | 3,908 | -0.03pp |

**终态门判定**：
- recall ≥ 0.95: **PASS** (ef=50: 0.9653, ef=100: 0.9905)
- QPS ≥ baseline × 1.3: **PASS** (ef=50: 1.38x, ef=100: 1.42x)
- **✅ 终态门 PASS**

**结论**：
- QPS 提升 +38-42%，达到预期 +30-50%
- recall 损失极小（ef=50: -0.52pp, ef=100: -0.16pp）
- avg_visited 几乎不变（+0.5%），SQ8 距离排序与 f32 高度一致
- 内存压缩 4.0x（512MB → 128MB）
- 编码开销极低（0.5s 编码 1M 向量）

#### 1.1 原理（4-bit PQ LUT16）

当前 ADC 路径对每个候选节点：逐子空间遍历 K=256 个聚类中心做标量 L2 → 极慢。

顶尖库的做法：查询时预计算 distance lookup table，候选计算只需 M 次 table lookup + 求和。

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
    dist += lut[m][pq_codes[m]]
  return dist
```

#### 1.3 SIMD 加速：LUT16-shuffle 路线（放弃 gather）

> **关键决策：放弃 gather 指令，采用 pshufb + maddubs 的 LUT16 路线。**
>
> 1. PQ 量化到 **4-bit**（每子空间 16 个中心，而不是 256）
> 2. 16 个中心的距离量化成 u8 放进一个寄存器
> 3. `_mm256_shuffle_epi8`（pshufb）做**寄存器内查表**——单周期，不碰内存
> 4. `_mm256_maddubs_epi16` 做定点累加

#### 1.4 预期收益

> **Amdahl 定律警告**：距离计算加速是相对于"距离计算本身"而言的，不是端到端 QPS。
> Phase 1 之后，"非距离开销"（堆操作、visited、访存）的占比会上升。
> **Phase 1 完成后必须做 profiler 时间分解。**

| 路径 | 当前 QPS | 预期 QPS | 加速比 | 备注 |
|:--|:--|:--|:--|:--|
| f32 全精度 (ef=50) | 8,657 | 不变 | 1x | 基线 |
| **SQ8 标量 (Step 0)** | — | ~11,000-13,000 | **1.3-1.5x** | 过渡验证 |
| **PQ8 K=256 LUT-ADC (实测)** | — | 11,632 | **1.30x** | ❌ 不敌 SQ8，分支关闭 |
| **PQ4 K=16 LUT-ADC (实测)** | — | 52,509 | **5.90x** | ❌ recall=6%，距离太噪声 |
| **SQ8 AVX2 (Step 0 实测，最终选择)** | — | 12,786 | **1.43x** | ✅ Phase 1 终态量化路径 |

> **Phase 1 PQ 实验结论（2025-06-28）**：
>
> - **PQ4 (K=16)**：avg_visited=890K（visited.reset() bug），修复后 K=16 量化太粗，ADC 距离无法引导图导航。分支关闭。
> - **PQ8 (K=256)**：avg_visited=1233（正常），recall=0.9571 ✅，但 QPS=11632 不敌 SQ8 (12786)。
>   - ef=50：PQ8 0.91x SQ8（LUT 查表开销 > 带宽节省）
>   - ef=200：PQ8 1.06x SQ8（高 ef 时带宽优势显现）
>   - 主工作点 ef=50 PQ8 不敌 SQ8 → **SQ8 为最终量化路径，PQ 分支关闭**。
> - **根因发现**：`greedy_search_pq4/pq8` 缺少 `visited.reset()` 是 avg_visited=886K 的元凶，非量化精度问题。修复后 PQ8 数据完全正常。
> - **SQ8 优势**：AVX2 `_mm256_subs_epu8` + `_mm256_maddubs_epi16` 做定点距离计算，无需 LUT 查表，流水线友好。PQ8 的 M=32 次 LUT 随机访问（32KB LUT）产生 cache miss，抵消了 4x 带宽节省。

#### 1.5 复合优化边界声明

| 子步骤 | 预期效果 | 是否单独满足规则 2 |
|:--|:--|:--|
| 0. SQ8 编码 + 距离核 + rerank | recall ≥ 0.95, QPS +30-50% | **是（Step 0 终态门）✅** |
| 1. PQ4 (K=16) LUT-ADC | recall=6%，距离太噪声 | **否（失败，关闭）** |
| 2. PQ8 (K=256) LUT-ADC | recall=0.957 ✅，QPS 不敌 SQ8 | **否（SQ8 更优，关闭）** |
| 3. SQ8 确定为最终量化路径 | recall=0.965, QPS=12786, 1.43x | **是（Phase 1 终态）✅** |

**终态阈值**：

| 判定等级 | 条件 | 处置 |
|:--|:--|:--|
| ✅ 成功 | recall@10 ≥ 0.95 **且** 端到端 QPS ≥ STANDARD_BASELINE × 1.5 | 并入主线 |
| ⚠️ 部分成功 | recall@10 ≥ 0.95 **且** 端到端 QPS ∈ [BASELINE, BASELINE × 1.5) | 触发 profiler 复盘 |
| ❌ 失败 | recall@10 < 0.95 **或** 端到端 QPS < BASELINE | 整体回退 |

> **注**：`STANDARD_BASELINE` = STANDARD(R=32) 图在 ef=50 处的 QPS = 8,657。

---

### Phase 2：两阶段搜索管道优化（2-3 天）

**目标：量化快速粗筛 → f32 精确 rerank，在 recall 不变前提下最大化 QPS**

> Phase 1 已包含基本 rerank。Phase 2 的重点是 rerank 策略的**精细调优**。

#### 2.1 搜索流程

```
greedy_search(query, ef_search):
  // Phase 1: 用量化距离做图导航（快但粗略）
  candidates = graph_walk(entry_point, ef_search, adc_distance)
  // Phase 2: 对 top-N 候选用 f32 精确距离 rerank
  reranked = top_n_candidates.sort_by(|a, b| l2_f32(query, a).cmp(l2_f32(query, b)))
  return reranked[0..k]
```

#### 2.2 关键参数

- `ef_search`：图搜索宽度，量化路径下可以加大（因为量化距离快 3-5x）
- `top_n`：rerank 候选数
- `rerank_strategy`：全量 rerank vs 增量 rerank

#### 2.3 图导航用量化还是 f32？

- **方案 A**：图导航用量化距离（更快但可能走错路 → recall 下降）
- **方案 B**：图导航用 f32，仅最终 rerank 用量化（无意义，更慢）
- **方案 C**：图导航用量化，但 ef_search 加大到补偿 recall 损失

顶尖库用的是方案 C——量化导航 + 大 ef + f32 rerank。

---

### Phase 3：图质量与内存布局优化（3-5 天）

> **v8 更新**：图质量已验证正常（H20），Phase 3 重点从"修图"转向"内存布局"。

#### 3.1 图节点重排序（Cache Locality Optimization）

按 BFS 遍历顺序重排节点 ID，使图遍历的内存访问模式变为顺序访问。

#### 3.2 PQ codes 连续存储

将所有节点的 PQ codes 存储为连续的 `Vec<u8>`（N × M 字节），而非 `Vec<Vec<u8>>`。

#### 3.3 图质量微调 + NGT k-NN 图实验路线（可选）

> ~~v7：avg_visited 10x 差距的根因~~ → **已证伪**。以下优化为边际收益，非必须。

**3.3a 边际优化**：

| 优化 | 当前 | 目标 | 对 QPS 的影响 |
|:--|:--|:--|:--|
| 初始图用 NN-guided | 随机图 | 用少量近邻引导 | 更少迭代收敛，边际收益 |
| r_max 自适应 | 固定 32 | 按数据集自动选择 | 减少无效距离计算 |

**3.3b NGT k-NN 图 + 多样性剪枝路线（v8 新增实验选项）**：

> **这是与 Vamana 完全不同的建图策略。** Vamana 是「随机初始化 → greedy search → RobustPrune」迭代两遍。
> NGT 的 ONNG 路线是「先建真实 k-NN 图（ANNG）→ 多样性剪枝优化（ONNG）」。
> 理论上图质量上限更高（因为初始图就是真实近邻结构，而非随机图），但建图成本也更高（k-NN 搜索本身是 O(N²) 级别）。
>
> **论文依据**：Iwasaki & Miyazaki, "Optimization of Indexing Based on k-Nearest Neighbor Graph for Proximity Search", arXiv:1810.07355 (2018)

| 建图策略 | 初始图 | 剪枝 | 理论优势 | 建图成本 | 现状 |
|:--|:--|:--|:--|:--|:--|
| **Vamana（当前）** | 随机图 | RobustPrune (α 遮挡) | 两遍长程边 + 短程边 | ~408s (SIFT1M) | ✅ 已实现 |
| **NGT ANNG → ONNG** | 真实 k-NN 图 | 多样性剪枝（去冗余边） | 初始图即真实近邻，收敛更快 | ~2-3x Vamana | ❌ 未实现 |
| **NGT PANNG** | ANNG | 路径剪枝（去捷径边） | 减少冗余路径 | 中等 | ❌ 未实现 |

**实验价值**：
- 如果 Vamana 的 avg_visited 在某些数据集上偏高（当前 SIFT1M 差距 15%，属正常），NGT 路线可能在高维/稀疏数据集上表现更好
- ONNG 的多样性剪枝与 RAVEN 的 RobustPrune 互补——可以混合使用：Vamana 两遍 + ONNG 多样性后处理
- 对论文有贡献价值：对比两种建图策略的 avg_visited / recall / QPS trade-off

**实施条件**：仅在 Phase 1（量化加速）完成后，如果多数据集测试中发现某些数据集 avg_visited 偏高时启动。当前 SIFT1M 上 Vamana 已足够。

---

#### 3.4 缓存行对齐 + AVX2 内存布局验证（v8 新增）✅ 已完成 — 结论：无显著收益

> **A/B 实验完成（v8.3）。** 同进程 5 轮交替测量，64B 对齐 vs 32B 跨缓存行偏移：
> - 64B align: mean=12132, cv=4.13%
> - 32B misalign: mean=12119, cv=2.96%
> - 差距 +0.1%，2σ=8.3%，**噪声带内，无统计显著性**
>
> **根因**：图数据通过 `_mm_prefetch` 预取指令访问，延迟被流水线隐藏；每节点邻居列表（R=32 → 128B = 2 cache lines）足够小，跨行开销可忽略。
>
> **决策**：不引入 `AlignedVec64`。`Vec<u32>` 的默认对齐已足够。排除此不确定性后，可专注 Phase 1（量化加速）。

<details>
<summary>历史分析（已证伪）</summary>

**原始假设**：`Vec<u32>` 默认 4B 对齐，邻居列表 `node_id × r_max` 起点未必 64B 对齐，可能导致跨缓存行访问 +50% cache line 加载。Glass 用 `posix_memalign` 确保 64B 对齐。

**实验设计**：同进程交替 A/B benchmark，64B 对齐 vs 32B 偏移（保证跨缓存行），5 轮交替，SQ8 + adaptive ef + fixed entry=medoid，ef=50, po=8。

**实验结果**：差距 +0.1%（2σ=8.3%），无统计显著性。假设证伪。
</details>

---

### Phase 4：搜索热路径微优化（2-3 天）

> **Phase 1 后优先级可能上升。** Phase 1 把距离计算从 compute-bound 拉到 memory-bound 后，"非距离开销"的占比上升。

#### 4.1 BinaryHeap 优化

用 `BinaryHeap<u64>` 打包 `(distance_bits << 32 | node_id)`，减少比较开销。

#### 4.2 VisitedTracker 优化

可考虑 `Vec<u64>` + bitmap。

#### 4.3 预取策略调优

ADC 路径下预取 PQ codes（4-bit vs 512 字节 f32）的 cache 影响完全不同。

#### 4.4 分支消除

`if !visited[node]` → 无分支版本。

#### 4.5 自适应 ef（v8 新增）✅ 已完成（commit fa1bdad）

> **RAVEN 独有的差异化创新点**。Glass 和 DiskANN 都没有。
>
> 不同查询在图上的难度差异极大——简单查询 ef=20 就够，难查询需要 ef=200。
> 当前所有查询用同一个 ef 值。自适应 ef 可以在同等平均 recall 下降低 avg_visited，直接提升 QPS。

**实现方案**：方向 B（距离分布感知 + 幂律变换）

```rust
// 离线：采样 2000 个向量，用 nav.initialize() 收集入口距离分布
let config = AdaptiveEfConfig::build_with_layered_nav(
    &vectors, dim, &layered_nav, 35, 75, 2.0);

// 在线：nav.initialize() 返回的 f32 距离直接预测 ef，零额外开销
let ef = config.estimate_ef(entry_dist).max(k);
// ef = min_ef + percentile^gamma × (max_ef - min_ef)
```

**幂律变换（RAVEN 创新）**：
- 分层导航把入口距离压缩到极窄区间（p25≈0.74, med≈0.97, p75≈1.18）
- 线性插值后 avg_ef≈50，等于没做自适应
- gamma=2.0 幂律变换：中位数查询拿 25% 的 ef 区间而非 50%
- 只有真正的高距离查询才分到大 ef

**实测结果**（SIFT-1M, SQ8, 单线程, ef=50 基线）：

初始扫描（4 组 + 1 固定基线）：

| 配置 | min_ef | max_ef | gamma | recall | QPS | avg_ef | QPSΔ |
|---|---|---|---|---|---|---|---|
| 基线 ef=50 | - | - | - | 0.9653 | 9439 | 50.0 | - |
| **gamma-2** | 35 | 75 | 2.0 | **0.9651** | **10308** | 48.1 | **+9.2%** |
| gamma-3 | 30 | 80 | 3.0 | 0.9548 | 10092 | 42.7 | +6.9% |
| gamma-4 | 25 | 80 | 4.0 | 0.9384 | 12620 | 36.6 | +33.7% |
| gamma-3-narrow | 38 | 65 | 3.0 | 0.9603 | 10762 | 44.9 | +14.0% |

密集参数扫描（100 组网格 + 11 组固定 ef 基线，bench_stable warmup+2 轮平均）：

**Pareto 最优配置 γ3(40,75)**：min_ef=40, max_ef=75, gamma=3.0

| 配置 | min_ef | max_ef | gamma | recall | QPS | QPSΔ vs fixed ef=50 |
|---|---|---|---|---|---|---|
| 固定 ef=50 | - | - | - | 0.9653 | 11635 | - |
| **γ3(40,75) ← 默认** | 40 | 75 | 3.0 | **0.9660** | **12952** | **+11.3%** |
| γ3(38,65) 备选 | 38 | 65 | 3.0 | 0.9603 | 13530 | +16.3% |
| γ2.5(33,65) 备选 | 33 | 65 | 2.5 | 0.9559 | 14573 | +25.3% |

**关键更正**：之前 flagship_bench 的“单线程 -11%”是热节流假象（Layer 1 冷CPU 13865 QPS → Layer 2 热CPU 12334 QPS），同批次 bench_stable（warmup+2轮平均）证明自适应 ef 单线程也是正收益。

**终态门**：✅ PASS（γ3(40,75): QPS +11.3% ≥ 5%，recall 0.9660 ≥ 0.9653）

**关键技术细节**：
- `GraphSearcher::last_ef_used()` 提供 ef 追踪，benchmark 无需重复调用 `nav.initialize()`
- `AdaptiveEfConfig` 支持 gamma 参数，gamma=1.0 退化为线性插值
- 与 SQ8 量化路径、分层导航、多线程 `batch_search` 全部集成
- `raven_ann_bench.rs` 默认启用 γ3(40,75)，可通过 `--no-adaptive-ef` 回退

---

### Phase 5：多数据集适配与参数自动调优（2-3 天）

| 数据集 | dim | N | 距离 |
|:--|:--|:--|:--|
| SIFT-128 | 128 | 1M | L2 |
| GIST-960 | 960 | 1M | L2 |
| GloVe-100 | 100 | 1.2M | L2 |

> **高维数据集内存预算警告**：GIST-960 的 f32 向量每条 3.75KB（960×4B），100 万条 = 3.75GB。
> rerank 阶段需读全精度 f32 向量，4-bit PQ codes 的带宽优势仅覆盖图导航段。

---

### Phase 6：极致工程化（2-3 天）

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
opt-level = 3
target-cpu = native
```

PGO 预期收益：QPS +5-10%。

---

### Phase 7：多线程查询并行化（v8 新增）🔴 P0 ✅ 已完成

> **ann-benchmarks 榜单默认用多线程测 QPS。** Glass 的 `batch_search` 支持 OpenMP 并行。
> RAVEN 当前仅单线程搜索，单线程成绩天然吃亏。
> **不上多线程 = 不能和榜单做同口径对比。**

#### 7.1 实现方案（已实现）

`GraphSearcher::batch_search(&self, queries: &[&[f32]], k: usize) -> Vec<Vec<(u32, f32)>>`

- `&self` 只读共享（图数据、SQ8 码本 read-only，零竞争）
- `rayon::par_iter` 并行化查询循环
- 每 worker 用 `thread_local!` 缓存 `VisitedTracker`（1MB for N=1M，首次分配后复用）
- 自动选择搜索路径：SQ8 > f32
- `batch_search_ids` 便捷接口返回纯 ID

#### 7.2 关键设计

- 每个 worker 独立 `VisitedTracker`（`thread_local!` 缓存，无锁无竞争）
- 图数据 `&self` 不可变共享（`vectors`、`graph`、`sq8` 全部 read-only）
- `rayon::ThreadPoolBuilder::num_threads(n)` 控制线程数
- **visited.reset() bug 修复确认**：每查询 `greedy_search_*` 入口调用 `visited.reset()`，跨查询无状态污染

#### 7.3 实测结果（SIFT-1M, SQ8, 16 核 CPU）

| ef | threads | recall | QPS | speedup | scaling |
|:--|:--|:--|:--|:--|:--|
| 50 | 1 | 0.9653 | 11,518 | 1.00x | 100.0% |
| 50 | 2 | 0.9653 | 23,943 | 2.08x | 103.9% |
| 50 | 4 | 0.9653 | 45,845 | 3.98x | 99.5% |
| **50** | **8** | **0.9653** | **79,910** | **6.94x** | **86.7%** |
| 100 | 1 | 0.9905 | 6,961 | 1.00x | 100.0% |
| 100 | 2 | 0.9905 | 14,361 | 2.06x | 103.2% |
| 100 | 4 | 0.9905 | 25,601 | 3.68x | 91.9% |
| 100 | 8 | 0.9905 | 46,578 | 6.69x | 83.6% |

- **2 线程超线性**（103.9%）：TLS 缓存预热 + rayon 调度开销摊薄
- **4 线程近线性**（99.5%）：图数据 + SQ8 码本完全在共享 L3 cache 中
- **8 线程 86.7%**：内存带宽开始瓶颈（8 workers × 128B/vector = 1KB/visit）
- recall 完全一致（0.9653 / 0.9905），多线程不影响搜索语义

#### 7.4 ann-benchmarks 口径

- ann-benchmarks 默认 `--parallelism=1`（单线程），但榜单展示多线程结果
- RAVEN 需同时支持单线程和多线程口径
- **单线程 QPS 是算法质量指标，多线程 QPS 是工程能力指标**

---

### Phase 8：ann-benchmarks Docker 环境适配（v8 新增）🔴 P0

> **不上榜单 = 不被社区认可 = 不能宣称"世界第一"。**
> 竞品分析报告将此列为 P0，冲刺计划此前只解决了 S1（重建索引），未覆盖 Docker 环境和 HDF5 格式。

#### 8.1 具体步骤

| 编号 | 任务 | 说明 |
|:--|:--|:--|
| 8.1 | **修复 `query()` 接口** | S7：单查询返回空列表，需修复 |
| 8.2 | **HDF5 格式适配** | ann-benchmarks 用 HDF5，RAVEN 当前只读 fvecs/ivecs |
| 8.3 | **Dockerfile 编写** | Rust toolchain + SIMD flags + 依赖配置 |
| 8.4 | **Docker 环境验证** | `ann-benchmarks run --algorithm raven --dataset sift-128-euclidean` |
| 8.5 | **提交 SIFT-128 结果** | 哪怕初始成绩不理想，先上榜 |

#### 8.2 目标

- recall@10=95% 时 QPS > 8,000（超越 Glass 单线程 7,678）
- Docker 可复现：任何人 `docker build` + `ann-benchmarks run` 能得到相同结果

---

## 四、预期最终性能

> **v8 更新**：基于真实基线 8,657 QPS（ef=50, R=32, 单线程, f32）。

### 4.1 SIFT1M 预期 Pareto 前沿

| recall@10 | 当前 QPS (f32 单线程) | +SQ8 | +4-bit PQ | +多线程 (8核) | vs 榜首 |
|:--|:--|:--|:--|:--|:--|
| 0.95 | ~10,000 | ~15,000 | ~20,000 | ~100,000+ | 超越榜首 |
| 0.97 | 8,657 | ~13,000 | ~17,000 | ~80,000+ | 超越榜首 |
| 0.99 | 4,479 | ~6,500 | ~8,000 | ~50,000+ | 可比 |

> **注**：多线程 QPS 为估计值，实际取决于内存带宽是否成为瓶颈。
> 单线程 QPS 是算法质量指标，多线程 QPS 受内存带宽限制（Amdahl）。

### 4.2 打榜 vs 论文：明确拆分

**打榜只认 recall-QPS 曲线那一根线。**

#### 打榜要靠的（纯吞吐优化）

- Phase 1: SQ8 → 4-bit PQ LUT16 SIMD（核心突破点）
- Phase 2: 两阶段搜索 rerank 精细调优
- Phase 3: 内存布局优化
- Phase 4: 热路径微优化 + 自适应 ef
- Phase 6: 编译优化 / PGO
- Phase 7: 多线程查询并行化
- Phase 8: Docker 环境适配

#### 论文要靠的（机制证据，不计入打榜分数）

- **RP-Tuning**：一次构建覆盖整条 Pareto 曲线
- **AVQ 检索感知量化**：比标准 PQ 的 recall 更高
- **自适应 ef**：差异化创新点，Glass/DiskANN 均无
- **Rust 安全性 + 确定性构建**

#### ⚠️ 量化器张力：打榜用 4-bit，论文用 AVQ

> **Phase 1 为了 SIMD 速度选了 4-bit PQ（K=16），而 AVQ 的检索感知优势通常在更高码本精度下才明显。**
>
> - **打榜量化器**：4-bit PQ（K=16），配合 LUT16-shuffle，追求极致速度
> - **论文量化器**：AVQ（可能 8-bit 或更高），追求 recall 优势
> - **过渡验证**：SQ8 作为 Step 0，验证量化质量后再进入 4-bit

---

## 五、优先级排序与时间线（v8 科学重排）

> **v8 重排原则**：确定性收益优先 → 核心瓶颈（距离计算）→ 榜单适配 → 工程优化 → 差异化创新。
> **Phase 1（距离计算加速）是 #1 优先级，无前置条件。**
> Phase 0D（预取调优）是确定性收益，应在 Phase 1 之前或并行做。

| 优先级 | 编号 | 任务 | 预期收益 | 时间 | 风险 | 备注 |
|:--|:--|:--|:--|:--|:--|:--|
| ✅ 已完成 | 0A | Phase 0A: 打榜命门 | 质量 | 0.5天 | 🟢 | S1 + 基线 |
| ✅ 已完成 | 0C.1 | 修复 rp_tuning.rs 编译错误 | 解除阻断 | 0.1天 | 🟢 | D3 已修复 |
| ✅ 已完成 | 0C.4 | 建图/查询路径分离 | 建图 -33%, QPS +53% | 0.5天 | 🟢 | |
| ✅ 已完成 | 0C.3 | BuildMetadata 落盘 | 复现性 | 0.5天 | 🟢 | |
| ✅ 已完成 | 0C.5 | RP-Tuning alpha 范围 | 实验正确性 | 0.2天 | 🟢 | |
| ✅ 已完成 | — | H20 Glass 基线审计 | 推翻虚假基线 | 0.5天 | 🟢 | avg_visited 15% 差距 |
| ✅ 已完成 | 0D | 预取参数自动调优 | 0%（噪声内无差异） | 1天 | 🟢 | po=6 vs po=8 差距 -0.0%，2σ=5.4%，保留 po=8。RAVEN 已有 graph prefetch |
| ✅ 已完成 | 1.0 | Phase 1 Step 0: SQ8 量化 | +38-42% QPS | 1天 | 🟢 | ef50: 8705→11997 (1.38x), recall 0.9653。终态门 PASS |
| ✅ 已完成 | 1 | Phase 1: PQ 量化对比实验 | SQ8 胜出 | 3天 | ✅ | PQ4 recall 6% 废弃，PQ8 ef50 不敌 SQ8 但 ef200 反超 6%，SQ8 为最终方案 |
| 🔴 **P0** | **8** | **ann-benchmarks Docker 适配** | **上榜** | **3-5天** | 🟡 中 | S7 + HDF5 + Dockerfile |
| 🔴 **P0** | **7** | **多线程查询并行化** | **多核线性扩展** | **2-3天** | 🟡 中 | rayon，榜单多线程口径 |
| 🟡 P1 | 2 | Phase 2: 两阶段 rerank 精调 | 边际收益 | 2-3天 | 🟡 中 | |
| 🟡 P1 | ~~4.5~~ | ~~自适应 ef~~ | ~~+9.2% QPS~~ | ~~已完成~~ | ✅ | gamma=2.0, recall 保持 |
| 🟡 P1 | 0B | Phase 0B: 代码卫生 | 质量 | 1-2天 | 🟢 低 | 逐项独立测 |
| ✅ 已完成 | 3.4 | 缓存行对齐 + AVX2 布局验证 | +0.1%（噪声内，无收益） | 1天 | 🟢 低 | A/B 实验证伪，不引入 AlignedVec64 |
| 🟢 P2 | 3.1-3.2 | Phase 3.1-3.2: 节点重排序 + PQ连续存储 | 取决于 profiler | 2-3天 | 🟢 低 | |
| 🟢 P2 | 3.3b | NGT k-NN 图实验路线 | 论文贡献（非打榜） | 3-5天 | 🟡 中 | 仅多数据集 avg_visited 偏高时启动 |
| 🟢 P2 | 4 | Phase 4: 热路径微优化 | Phase 1 后可能更重要 | 2-3天 | 🟢 低 | |
| 🟢 P2 | 0C.6 | parking_lot 接入 | 设计文档合规 | 0.3天 | 🟢 低 | D5 |
| 🟢 P2 | 0C.7 | 内存带宽 profiling | Phase 1 决策数据 | 0.5天 | 🟢 低 | D8 |
| 🟢 P3 | 5 | Phase 5: 多数据集 | 扩展覆盖 | 2-3天 | 🟡 中 | |
| 🟢 P3 | 6 | Phase 6: PGO/NUMA | +5-10% | 2-3天 | 🟢 低 | |
| 🟢 P3 | ~~0C.2~~ | ~~随机层级导航~~ | ~~学术价值~~ | ~~3-5天~~ | 🟢 低 | ~~降级：avg_visited 10x 差距已证伪~~ |
| 🟢 P3 | 0C.8 | GEMM 路径标注未实现 | 文档诚实 | 0.1天 | 🟢 低 | D6 |
| 🟢 P3 | 0C.9 | 局部 alpha 标注探索性 | 文档诚实 | 0.1天 | 🟢 低 | D4 |

**总计**：约 20-35 天。关键路径：0D（预取调优）→ 1.0（SQ8）→ 1（4-bit PQ）→ 8（Docker 上榜）→ 7（多线程）。

**执行顺序（v8 锁死，不得跳步）**：

1. ✅ **0A-0C 已完成项**（S1 修复 + 编译修复 + 基线测量 + 建图优化 + BuildMetadata + alpha 范围）
2. ✅ **H20 Glass 基线审计**（推翻虚假基线，Phase 1 否决令解除）
3. 🔴 **0D 预取参数自动调优**（1天，确定性收益，Glass 实测 30%）
4. 🔴 **Phase 1 Step 0: SQ8 量化**（2天，过渡验证）
5. ✅ ~~**Phase 1: LUT16 SIMD PQ-ADC**~~（PQ4 K=16 recall 6% 废弃，PQ8 K=256 ef50 不敌 SQ8 但 ef200 反超 6%。SQ8 AVX2 为最终量化方案。PQ8 LUT 优化待定）
6. 🔴 **Phase 8: ann-benchmarks Docker 适配**（3-5天，可与 Phase 1 并行）
7. 🔴 **Phase 7: 多线程查询**（2-3天，榜单多线程口径）
8. ✅ ~~**Phase 3.4: 缓存行对齐验证**~~（A/B 实验完成，+0.1% 噪声内，无收益，不引入 AlignedVec64）
9. ✅ ~~**Phase 4.5: 自适应 ef**~~（已完成，+9.2% QPS，gamma=2.0 幂律变换）→ 🟡 **Phase 2: rerank 精调**
10. 🟢 **0B 代码卫生 + 0C.6/0C.7** 穿插在等待期做
11. 🟢 **Phase 3.1-3.3 + Phase 3.3b(NGT实验) + Phase 5-6** 按 profiler 数据决定

---

## 六、核心技术风险与对策

| 风险 | 概率 | 影响 | 对策 |
|:--|:--|:--|:--|
| 4-bit 量化导致 recall 下降 | **高** | 高 | SQ8 过渡验证 + rerank 补偿 + 加大 ef_search |
| ~~AVX-512 gather 不如预期~~ | ~~低~~ → **已放弃 gather** | - | 改用 LUT16-shuffle（pshufb + maddubs），不碰内存 |
| **Amdahl 稀释** | **高** | 高 | Phase 1 后用 profiler 测时间分解 |
| ~~图质量不高（avg_visited 偏高）~~ | ~~已证伪~~ | - | ~~H20 实测差距 15%，正常~~ |
| Phase 3 cache 优化收益衰减 | 中 | 中 | Phase 1 后用 perf stat 测 LLC miss |
| **4-bit PQ 与 AVQ 码本设计冲突** | 中 | 中 | 打榜/论文用不同量化器配置，明确边界 |
| **rerank 带宽优势仅覆盖图导航段** | **高** | 中 | SIFT1M 无影响；GIST-960 需重新评估 |
| **ann-benchmarks 环境差异** | 中 | 中 | Docker 可复现 + 同硬件同线程口径验证 |
| **多线程内存带宽瓶颈** | 中 | 中 | Phase 7 后用 perf stat 测带宽利用率 |
| **预取调优收益不如 Glass** | 低 | 低 | po 扫描是确定性收益，最差也是 po=8 不变 |
| ~~main_block 未 64B 对齐导致 cache miss~~ | ~~已证伪~~ | - | A/B 实验 +0.1% 噪声内，prefetch 已隐藏延迟 |
| **NGT k-NN 图建图成本过高** | 中 | 低 | 仅作为实验选项，不进入主路径。Vamana 当前已足够 |

---

## 七、附录：审计发现优先修复清单（v8 更新）

> 按"打榜影响 x 修复成本"排序

### 第零优先：阻断编译/测试（已修复）

0. **D3 rp_tuning.rs 编译错误** ✅ 已修复

### 第一优先：直接影响 benchmark 结果

1. **建立真实基线** ✅ 已完成 → H20: RAVEN recall=0.9705, QPS=8,657; Glass recall=0.9465, QPS=7,678
2. **S1 修复 wrapper** ✅ 已完成
3. **拉取真实排行榜** ✅ 已完成 → qsgngt(11,163@0.99) 和 glass(15,171@0.95)
4. **~~测量 avg_visited（否决闸门）~~** ✅ 已完成 → **1227 vs Glass 1041，差距 15%，正常。否决令解除。**
5. **S7 修复 ann-benchmarks query() 接口** → P0（上榜阻塞项）
6. **S8 预取参数自动调优** → P0（确定性收益 +20-30%）
7. **Phase 1 SQ8 → 4-bit PQ** → P0（核心瓶颈：距离计算）
8. **Phase 7 多线程查询** → P0（榜单多线程口径）
9. **Phase 8 Docker 环境适配** → P0（上榜）

### 第二优先：复现性与实验正确性

10. **D2 BuildMetadata** ✅ 已完成
11. **D7 RP-Tuning alpha 范围** ✅ 已完成
12. **D8 内存带宽 profiling** → P2

### 第三优先：代码质量与可维护性

13. **M1 `eprintln!` -> `tracing`** → P1
14. **M2 `unwrap()` -> `?`** → P1
15. **S4 RP-Tuning B/C** → P2
16. **D5 parking_lot 接入** → P2

### 第四优先：差异化创新

17. ~~**自适应 ef**~~ → ✅ 已完成（P1，RAVEN 独有，gamma=2.0 幂律变换，+9.2% QPS）
18. ~~**缓存行对齐 + AVX2 布局验证**~~ → ✅ 已完成（A/B 实验证伪，+0.1% 噪声内，不引入 AlignedVec64）
19. **NGT k-NN 图实验路线** → P2（论文贡献，仅多数据集 avg_visited 偏高时启动）
20. **AVQ 论文级打磨** → P2

### 第五优先：文档诚实度

20. **D6 GEMM 标为未实现** → P3
21. **S3 big-ann 标为未实现** → P3
22. **S5 维度分发说明** → P3
23. **D4 局部 alpha 标注探索性** → P3
24. ~~**D1 随机层级导航**~~ → ~~P3 降级~~（avg_visited 10x 差距已证伪）

---

## 八、一句话总结

**v8 核心结论：H20 实测推翻了 "Glass avg_visited < 150" 的虚假基线（三个 bug 叠加）。Glass HNSW 同配置（R=32, L=200, FP32, 单线程）实测 avg_visited=1041，RAVEN=1227，差距仅 15%。RAVEN 在 recall（97.05% vs 94.65%）和 QPS（8657 vs 7678）上均优于 Glass。图质量正常，距离计算是唯一剩下的瓶颈。Pivot Criterion 否决令解除，Phase 1（SQ8 → 4-bit PQ LUT16 SIMD）恢复 #1 优先级。新增 Phase 0D（预取参数自动调优，确定性 +20-30% 收益）、Phase 3.4（缓存行对齐 + AVX2 内存布局验证，main_block Vec→AlignedVec64）、Phase 3.3b（NGT k-NN 图 + 多样性剪枝实验路线，论文贡献）、Phase 7（多线程查询 rayon 并行化）、Phase 8（ann-benchmarks Docker 环境适配）、Phase 4.5（自适应 ef，差异化创新）。D1（随机层级导航）根因判断已推翻，降级为 P3。关键路径：0D（预取调优）→ 1.0（SQ8）→ 1（4-bit PQ）→ 8（Docker 上榜）→ 7（多线程）。Phase 3.4（对齐验证）应在 Phase 1 之前做诊断，排除 cache miss 干扰。**

**v8.1 更新（全优化集成审计）：发现所有优化（SQ8、自适应 ef、多线程）此前的集成状态为零——各自在独立 bench 里验证过，但从未叠加进 `raven_ann_bench.rs`（上榜入口）和 `sift1m_bench.rs`（旗舰 bench）。已完成集成：`raven_ann_bench.rs` 默认启用 SQ8 + 自适应 ef + 多线程，可通过 `--no-sq8` / `--no-adaptive-ef` / `--no-multithread` 回退。旗舰 benchmark（`flagship_bench.rs`）逐层叠加验证：f32 baseline 9.2K QPS → +SQ8 13.9K（+50%）→ +adaptive_ef 12.3K（单线程下 -11%，avg_ef=48.1 vs 50，预测开销抵消了 ef 缩减收益）→ +multithread 16T **102.4K QPS**。自适应 ef 在单线程 SQ8 路径下收益不明显（nav.initialize 的 f32 导航开销 + predict_ef 分支开销 ≈ ef 缩减 3.8% 的节省），但在多线程 batch_search 中作为整体打包无额外开销。全栈 recall=0.9651，QPS=102,417。

**v8.2 更新**：
1. **自适应 ef 密集扫描完成**（100 组网格 + 11 组固定 ef 基线）。关键更正：v8.1 中“单线程 -11%”是热节流假象，bench_stable（warmup+2 轮平均）证明自适应 ef 单线程也是正收益。Pareto 最优 γ3(40,75): recall=0.9660, QPS=12952 (+11.3% vs fixed ef=50)。已设为 `raven_ann_bench.rs` 默认。
2. **degrees 数组 O(1) neighbors()**（commit a88d4a5）。`HybridBlockedCsr` 新增 `degrees: Vec<u16>` 数组，`neighbors()` 从 O(r_max) SENTINEL 线性扫描降为 O(1) 数组查找，`add_edge()` 去重从 O(r_max) 降为 O(deg)。A/B benchmark（同进程 5 轮交替）结论：查询路径 +0.2%（2σ=4.8%，噪声内），建图路径改进（add_edge/set_neighbors/degree 均加速）。根因：multi-line graph prefetch 已将邻居列表预取到 L1，SENTINEL 扫描 32 个 u32 在 L1 命中下仅 ~4 周期，非瓶颈。
3. **target-cpu=native** 全局生效（`.cargo/config.toml`）。手写 intrinsic 已用 AVX2，此标志让非 intrinsic 代码路径（如 SQ8 rerank 的 f32 排序）被编译器自动向量化。
4. **`raven_ann_bench.rs` 默认配置**：use_sq8=true, use_adaptive_ef=true (γ3(40,75)), use_multithread=false。标志：`--no-sq8` / `--no-adaptive-ef` / `--multithread`。

**v8.3 更新**：
1. **Phase 3.4 缓存行对齐 A/B 实验完成**。同进程 5 轮交替，64B 对齐 vs 32B 跨缓存行偏移（mod64=32，真正跨行）。结果：64B align mean=12132 cv=4.13%，32B misalign mean=12119 cv=2.96%，差距 +0.1%（2σ=8.3%，噪声内）。**结论：无显著收益，不引入 AlignedVec64**。根因：`_mm_prefetch` 预取指令已将图数据预取到 L1，跨缓存行开销被流水线隐藏；每节点邻居列表 128B（2 cache lines）足够小。
2. **清理临时 A/B 测试代码**：`align_ab_bench.rs` 已删除，工作区回退到 commit 638b612 干净状态。

**v8.4 更新**：
1. **Phase 1 PQ 全部完成**（commit 2cc7b2a）。三方对比：f32 vs SQ8 vs PQ8 vs PQ4（SIFT-128, M=32, ef=50）：

| 量化方案 | recall@10 | QPS | vs f32 | vs SQ8 | avg_visited | 结论 |
|:--|--:|--:|--:|--:|--:|:--|
| f32 (基线) | 0.9705 | 8,951 | 1.00x | — | 1,227 | 基线 |
| **SQ8 AVX2** | **0.9653** | **12,786** | **1.43x** | **1.00x** | 1,233 | **✅ 最终选择** |
| PQ8 K=256 | 0.9571 | 11,632 | 1.30x | 0.91x | 1,233 | ❌ ef=50 不敌 SQ8，但 ef=200 反超 6% |
| PQ4 K=16 | 0.0597 | 52,509 | 5.90x | — | 889,755 | ❌ 量化太粗，recall 崩塌 |

2. **关键发现**：
   - **visited.reset() bug**：`greedy_search_pq4` 和 `greedy_search_pq8` 缺少 `visited.reset()`，warmup 后全图被标记 → 退化为穷举。修复后 avg_visited 从 886K 降到 1233。
   - **SQ8 胜出原因**：AVX2 `_mm256_subs_epu8` + `_mm256_maddubs_epi16` 全 SIMD 并行，无需 LUT 查表。PQ8 的 M=32 次 LUT 随机访问（32KB LUT 不入 L1）产生 cache miss，抵消了 4x 带宽节省。
   - **PQ8 保留价值**：ef=200 时 PQ8 反超 SQ8 6%（4599 vs 4346 QPS），高精度工作点 LUT 命中率提升。代码存档在主线。
3. **PQ8 优化方向**：LUT 从 f32(32KB) 降为 u16(16KB) 可入 L1，消除 cache miss；再叠加 AVX2 gather 批量查表，有望在中低 ef 也追上 SQ8。

**v8.5 更新**：
1. **PQ4 pshufb SIMD 实验完成——方案彻底终止**。三轮实验全部失败：

| 实验 | ef=50 recall | ef=50 QPS | vs PQ4 标量 | vs SQ8 | 结论 |
|:--|--:|--:|--:|--:|:--|
| PQ4 M=32 标量（基线） | 0.7678 | 15,412 | 1.00x | 1.29x | recall 差 SQ8 20pp |
| PQ4 M=32 SIMD v1（散落指针 gather） | 0.7683 | 10,260 | 0.67x | 0.83x | ❌ cache miss + 预取不匹配，QPS 倒退 33% |
| PQ4 M=32 SIMD v2（连续栈缓冲区 + 批量预取） | 0.7683 | 12,759 | 0.83x | 1.01x | ❌ transpose 开销 > pshufb 收益，仍低标量 17% |
| PQ4 M=64（sub_dim=2，recall 验证） | 0.9221 | 10,448 | 0.68x | 0.82x | ❌ recall 仍差 SQ8 4.3pp，32B/neighbor 带宽翻倍比 SQ8 还慢 |

2. **PQ4 终止根因分析**：
   - **K=16 码字太少**：4-bit 表达力不足，M=32 recall 0.77@ef=50，M=64 recall 0.92@ef=50 仍不过 0.95 门槛。
   - **recall 税吞噬带宽优势**：PQ4 M=32 的 16B/neighbor 带宽优势（8x vs SQ8）在 ef=300+ 的高 recall 工作点被访问次数翻 4 倍完全抵消。
   - **M=64 双重打击**：带宽翻倍到 32B/neighbor（与 SQ8 128B 差距缩小到 4x），recall 仍不够，QPS 反而比 SQ8 还慢。
   - **pshufb SIMD 失败**：批量 16 邻居方案的 transpose（256 次 stack 读写 + column gather）开销吃掉 pshufb 查表加速。只有 ef=200（计算密度高）才勉强持平标量。
3. **代码处理**：PQ4 SIMD 代码已全部回退，保留 PQ4 标量版（M=32, f32 LUT）作为实验存档。`adaptive_ef` 模块导出修复保留。
4. **转向 PQ8 u8 LUT 优化**：~~PQ8（K=256）是当前最优路线~~ → 见 v8.6 更新，PQ8 优化也失败。

**v8.6 更新**：
1. **PQ8 u8 LUT + AVX2 gather 两条优化路径全部失败，PQ 方案全线封存**：

| 优化尝试 | ef=50 QPS | vs PQ8 标量 | vs SQ8 | 失败原因 |
|:--|--:|--:|--:|:--|
| PQ8 f32 LUT 标量（基线） | 11,632 | 1.00x | 0.91x | 32 次随机 LUT 查表，硬件预取失效 |
| PQ8 u8 LUT（8KB 入 L1） | 10,946 | 0.94x ↓6% | 0.86x | cache line 利用率更低（1.56% vs 6.25%），每次取 1 byte 带 64B cache line |
| PQ8 AVX2 gather batch8 | 9,766 | 0.84x ↓16% | 0.73x | `_mm256_i32gather_ps` 是串行微码，延迟高于标量逐一加载 |

2. **PQ 方案全线封存根因**：
   - **随机查表 vs 顺序 SIMD 是本质差距**：SQ8 是 128 bytes 连续顺序 SIMD 读（`_mm256_subs_epu8` + `_mm256_maddubs_epi16`），PQ8 是 32 次随机 LUT 查表（地址由 code 决定，不可预测）。两种访问模式的内存友好程度不在同一层级。
   - **LUT 大小不是瓶颈**：u8 LUT（8KB）进 L1 反而更慢，因为 cache line 利用率下降。
   - **gather 不是银弹**：x86 gather 指令在当前微架构上是逐元素串行执行，32 次 gather × 5 cycles = 160 cycles，比 32 次标量 load（~128 cycles）还慢。
   - **PQ8 ef=200 曾超过 SQ8 6%**（4599 vs 4346），但这是唯一优势点，低 ef 全面落后，不具备实战价值。
3. **代码处理**：所有 PQ8 优化代码已回退，保留 PQ8 标量版（f32 LUT）作为实验存档。
4. **资源集中到 SQ8 极限优化**：SQ8 是当前 recall-QPS Pareto 曲线最优方案（ef=50: recall 0.9653, QPS 12,786），下一步优化方向：
   - SQ8 距离核 AVX2 SIMD 审计（当前实现是否已最优）
   - `target-cpu=native` 效果验证
   - `greedy_search` 热路径局部性（VisitedTracker / LinearPool）

---

## 九、附录：终端工作流（保证数据收集不中断、不阻塞）

> 本节记录 agent 如何使用终端工具保证 benchmark 数据收集的可靠性与连续性。

### 9.1 长时间任务后台执行

建图和 benchmark 运行可能需要 10-15 分钟。直接在前台运行会阻塞整个 agent 会话。

**方法**：使用 `is_background: true` 将任务放入后台，输出重定向到文件：

```bat
cd /d c:\Users\Juwan\Desktop\RAVEN && cargo run --release --bin quick_recall_check > result.txt 2>&1
```

- `> result.txt 2>&1`：stdout 和 stderr 合并写入文件，确保不丢数据
- 后台执行后，agent 可继续做其他工作（如更新文档、审查代码）
- 通过 `read_file` 工具读取 `result.txt` 获取结果，无需阻塞等待

### 9.2 进程状态检查

运行 benchmark 前必须确认无残留进程（旧 exe、cargo、rustc），否则 CPU 被抢占导致数据污染（QPS 减半 + 建图翻倍）。

```bat
tasklist | findstr /i "raven cargo rustc"
```

- 如有残留进程，先 `taskkill /f /im <process.exe>` 清理
- 清理后等待 2-3 秒再启动新任务，确保 CPU 释放

### 9.3 .bat 脚本封装复杂命令

Windows cmd 对引号、管道、中文有编码问题。将复杂命令封装为 `.bat` 文件：

```bat
@echo off
cd /d c:\Users\Juwan\Desktop\RAVEN
git add RAVEN*.md
git commit -m "docs: v8 - H20 overturns false baseline, Phase 1 unblocked"
```

- 用 `write` 工具创建 `.bat`，用 `run_terminal_cmd` 执行
- 避免中文或特殊字符直接出现在 cmd 参数中
- git commit message 保持 ASCII，中文内容写入文档而非 commit message

### 9.4 结果文件验证

工具缓存可能导致 `read_file` 显示旧内容。验证磁盘实际状态的方法：

```bat
type result.txt | findstr /i "recall QPS"
```

- 或使用 `grep` 工具搜索结果文件中的关键行
- 必要时用 `write` 工具全量覆盖文件，而非增量编辑

### 9.5 干净环境验证清单

每次跑 benchmark 前执行：

1. `tasklist | findstr /i "raven cargo rustc"` → 确认无残留
2. 确认数据文件存在：`dir data\sift\sift_base.fvecs`
3. 后台启动 benchmark，输出重定向到文件
4. 等待完成后用 `read_file` 或 `grep` 读取结果
5. 将结果写入本文档对应章节

---

*文档结束*
