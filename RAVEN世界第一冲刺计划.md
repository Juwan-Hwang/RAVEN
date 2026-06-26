# RAVEN 冲击 ann-benchmarks 世界第一：战略路线图

> 创建时间：2026-06-25
> 最后修订：2026-06-26（v6：算术不可能定理 + Pivot Criterion + 目标重定义 + 终端工作流）
> 目标：在 ann-benchmarks SIFT1M (sift-128-euclidean) 上达到 recall-QPS Pareto 前沿第一梯队（详见 §〇.2 目标重定义）
>
> **v6 重大变更**：用文档自己的两个数字（基线 2,706 × Phase 1 上限 2.5 = 6,765，目标 15,171）证明
> Phase 1 单独无法闭合 5.6x 差距。avg_visited 被提升为项目最关键单个数字，拥有 Phase 1 优先级否决权。

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
> Phase 1 的本质是 4-bit 量化**必然先让 recall 下降**，再靠 rerank 和加大 ef_search 补回来。
> 如果严格按规则 2 执行，Phase 1 中间任何一个单独的 commit（如"接入 4-bit PQ"这一步）都会让 recall 暴跌，按规则就得回退——Phase 1 永远做不下去。

**例外条款**：

对于已知会引入中间态 recall 损失的**复合优化**（如 4-bit 量化 + rerank），允许以"整体工作点"为单位评估 before/after，而非以单个 commit 为单位。但必须满足以下全部约束：

1. **预先声明边界**：在开始前，明确声明该复合优化包含哪些子步骤、子步骤之间的依赖关系。
2. **预设 recall 阈值**：声明终态 recall 阈值（如 recall@10 ≥ 0.95）。终态未达标则整体回退。
3. **在分支上开发**：复合优化在独立分支上开发，只有终态达标才并入主线。
4. **子步骤仍需独立 commit**：每个子步骤在分支上单独 commit（方便回退和对比），但不要求每个子步骤单独满足规则 2。
5. **终态评估**：合并到主线时，以终态工作点的 (recall, QPS) 与分支前的基线做 before/after 对比。

### 〇.2 目标重定义与 Pivot Criterion

> **算术诚实**：基线 2,706 × Phase 1 最乐观估计 2.5 = 6,765 QPS。目标线 glass = 15,171。
> 即使 Phase 1 拿到它能给的全部上限，离 recall 0.95 处的榜首还差 2.24x。
> Phase 3/4/6 在文档中已被诚实标注为「可能衰减」「+5-10%」，凑不出剩下的 2.24x。
> **结论：按当前以 Phase 1 为核心引擎的计划，「SIFT1M Pareto 前沿世界第一」在数学上不成立。**

**目标重定义**：

| 层级 | 定义 | 达成条件 |
|:--|:--|:--|
| ~~旧目标~~ | ~~Pareto 前沿世界第一~~ | ~~QPS 超越所有库的外包络线~~ |
| **新目标 A：第一梯队** | 在某个 recall 工作点进入 Pareto 前沿 top-5 | 存在 recall 点，RAVEN QPS ≥ 榜首 × 0.5 |
| **新目标 B：单点突破** | 在某个 recall 工作点超越特定榜首算法 | 存在 recall 点，RAVEN QPS > 该点榜首 |
| **新目标 C：论文贡献** | 机制创新有独立价值 | RP-Tuning / AVQ / 量化感知剪枝有消融数据支撑 |

> 如果 avg_visited 测量后发现图质量是主瓶颈并修复之，5.6x 差距可能大幅缩小，届时可重新评估是否回归「世界第一」目标。

**Pivot Criterion（avg_visited 否决权）**：

> avg_visited 是当前整个项目最关键的单个数字，比 Phase 1 还关键。
> 它必须在 Phase 1 之前测量，且结果有权否决 Phase 1 的优先级。

| avg_visited 测量结果 | 诊断 | 行动 |
|:--|:--|:--|
| **< 150**（glass 量级） | 图质量正常，瓶颈在距离计算 | **Phase 1 优先**（原计划不变） |
| **150-300**（中等） | 图质量有改进空间但不致命 | Phase 1 和 Phase 3.3 **并行推进** |
| **> 300**（偏高） | 图质量是主瓶颈，Phase 1 救不了 | **Phase 3.3 优先**，Phase 1 降级为 P2 |
| **> 500**（严重偏高） | 图导航效率根本性缺陷 | **暂停 Phase 1**，全力修图，重新评估可行性 |

> **为什么 avg_visited 比 QPS 更能暴露问题**：QPS 受 SIMD、内存布局、cache miss 等多重因素干扰。
> avg_visited 只衡量「图导航走了多少冤枉路」——如果 RAVEN 的 avg_visited 是 glass 的 3 倍，
> 那么距离计算再快 5 倍也没用，因为做了 3 倍于对手的距离计算次数。
> Amdahl 稀释之外还有这一层：在一条更长的路径上加速，对手压根不走那条路。

---

## 实验记录

### 基线数据（BEFORE）

> 所有优化的对比基准。必须在干净环境下测量。
> **旧数据可能被后台进程污染，不可直接引用。**

| 指标 | 旧值（可能污染） | 干净重测值 | 备注 |
|:--|:--|:--|:--|
| cargo test | 155 passed, 0 failed | 155 passed, 0 failed | 全部通过 |
| cargo clippy | 34 errors | 待测 | 已存在问题 |
| recall@10 (α=1.2, l_build=100, r_max=32, ef=100) | 0.9517 | **0.9517** | 一致（recall 不受 CPU 污染影响） |
| QPS (同上参数) | 1,287 | **2,706** | 旧值被污染，实际快 2.1x |
| 建图时间 | 1,054.8s | **912.1s** | 旧值被污染，实际快 13.5% |
| **avg_visited**（每查询平均距离计算次数） | 未测 | **待测** | 需加 instrumentation，见 §1.3 |

> **avg_visited 是比 QPS 更能暴露图质量的指标**，因为它不受 SIMD/内存布局干扰。
> 公开 HNSW/Vamana 实现在 SIFT1M recall@10=0.95 时的 avg_visited 通常在 100-300 次。
> 如果 RAVEN 的 avg_visited 显著偏高（如 >500），说明图本身的导航效率有问题，Phase 1 的 ADC 加速救不了。

---

### 唯一标准构建配置（Canonical Build Config）

> **v6.2 新增（2026-06-26）**
>
> **问题**：审计发现项目存在三张参数不同的图——打榜旗舰图（200/64/max_iter=1）、干净基线图（100/32/max_iter=2）、待测 avg_visited 图（100/32/max_iter=2）。5.6x 差距是拿 100/32/2 的小图基线去比 glass，却拿 200/64/1 的旗舰图去打榜，口径错位。旗舰图堆了大参数（l_build=200/r_max=64）却因 `max_iterations=1` 漏了 Vamana 第二遍长程边，是「用料足但没建完」的图。
>
> **裁决**：在此钉死唯一标准构建配置，此后所有基线、avg_visited 测量、Phase 1/3.3 评估**必须**在这张图上进行。

| 配置名 | α | l_build | r_max | r_soft | max_iterations | ef_search | 用途 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **CANONICAL** | 1.2 | 200 | 64 | 96 | **2** | 100 | **唯一标准图**：基线、avg_visited、打榜全部基于此 |
| **GLASS-COMP** | 1.2 | 200 | **32** | 48 | 2 | 100 | 同度数对照：与 glass R=32 做 apples-to-apples 对比 |

> - **CANONICAL** = 旗舰参数（l_build=200, r_max=64）+ max_iter=2（建完整两遍）。这是最终提交到 ann-benchmarks 的路径，基线必须长得和打榜图一样。
> - **GLASS-COMP** = 同 l_build、同 max_iter，仅 r_max=32。glass 榜首用 R=32 做到 recall=0.9941/QPS=19,801，RAVEN 需在同度数下对比 avg_visited 才能暴露图质量差距。大参数能靠堆出度硬撑 recall，但 glass 用小出度就又快又准——这才是图质量高低的真正标尺。
> - **旧的 2,706 QPS 基线（100/32/2）作废**。它既不是打榜图也不是新标准图，留着只会继续制造「和谁比」的混乱。

**实施顺序（锁死，不得跳步）**：

1. 修复 `VamanaBuildConfig::default()` 的 `max_iterations: 1` 地雷（纯 bug，独立 commit）
2. 给 `GraphSearcher` 添加 `avg_visited` 插桩（纯增量，不改行为）
3. 将 `quick_recall_check.rs` 改为 CANONICAL 配置，同时支持 `--r-max 32` 切换到 GLASS-COMP
4. 在两张图上分别测量 avg_visited + QPS + recall
5. 修复 `raven_ann_bench.rs` 硬编码 `max_iterations=1` + `__init__.py` wrapper S1
6. 将结果记入本文档，按 §〇.2 Pivot Criterion 裁决

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
| α=1.2, r_max=32, ef=100 | **0.9517** | **2,706** | f32 全精度 | ~~干净基线~~ **已作废**：非 CANONICAL 配置，见 §Canonical Build Config |
| α=1.2, r_max=64, ef=100 | 0.9961 | 2,434 | f32 全精度 | 旧值，需干净重测 |
| α=1.0, r_max=64, ef=50 | 0.9275 | 7,611 | f32 全精度 | 旧值，需干净重测 |
| α=1.2, ef=100, ADC+rerank | 0.9676 | 2,025 | AVQ 量化 | 旧值，需干净重测 |
| **CANONICAL (200/64/2)** | **待测** | **待测** | f32 全精度 | **新标准基线**：取代 2,706 |
| **GLASS-COMP (200/32/2)** | **待测** | **待测** | f32 全精度 | 同度数对照 glass R=32 |

### 1.2 竞争目标：ann-benchmarks 真实 Pareto 前沿（✅ 已填入）

> **数据来源**：ann-benchmarks 官方 results 仓库，sift-128-euclidean。
> **关键发现：榜首不是 ScaNN，是 qsgngt (NGT-QSG) 和 glass。** ScaNN 未出现在最优 Pareto 前沿中。
> 目标是超越当前 Pareto 前沿的**外包络线**（所有库取最优）。

| 工作点 | 榜首算法 | recall@10 | QPS | 备注 |
|:--|:--|:--|:--|:--|
| 高 recall 区 | **qsgngt** | 0.9917 | **11,163** | QSG-NGT(100,64,120,96,100,60,300,400,0,1.) |
| 中 recall 区 | **glass** | 0.9523 | **15,171** | glass_({'R':48,'level':2,'L':200}) |
| 低 recall 区 | **glass** | 0.9941 | **19,801** | glass_({'R':32,'level':2,'L':200}) |

> **注意**：低 recall 区的 glass 工作点 recall=0.9941 且 QPS=19,801，说明 glass 在低 R 值下同时保持高 recall 和高 QPS——这是图质量极高的标志。

### 1.3 差距诊断

> **"图算法已足够好"是未经验证的假设，不是结论。**
> 这个判断和 1.1 节那些可能被污染的旧数据是同源的。我们诚实地把性能数字改成"待测"了，但定性结论没有同样降级——这是一个逻辑漏洞。
>
> 如果干净重测后发现 f32 路径的 QPS 也明显低于榜首（不只是 ADC 慢），那可能说明图本身的导航效率（avg_visited）就有问题，Phase 1 的 ADC 加速救不了。
>
> **必须在干净基线中测量 avg_visited**，拿它和公开的 HNSW/Vamana 实现对比。

| 维度 | RAVEN（干净基线） | 榜首外包络 | 差距倍数 |
|:--|:--|:--|:--|
| recall ~0.95 处 QPS | **2,706** (recall=0.9517) | **15,171** (glass, recall=0.9523) | **5.6x** |
| recall ~0.99 处 QPS | ~2,434 (旧值, recall=0.9961) | **11,163** (qsgngt, recall=0.9917) | **4.6x** |
| recall ~0.93 处 QPS | 7,611 (旧值, recall=0.9275) | **19,801** (glass, recall=0.9941) | **2.6x** |
| **avg_visited @ recall 0.95** | **待测** | 100-300（公开数据） | 待对比 |
| ADC 路径 vs f32 | ADC **更慢** (旧值 2,025 vs 2,434) | 榜首用 PQ-ADC 快 3-5x | **根本性缺陷** |
| 建图时间 | **912.1s** (r_max=32) | ~200-500s | 2-4x |

**核心发现 1：RAVEN 的 ADC 路径比 f32 还慢。** 但这个发现能解释「为什么 RAVEN 的量化路径是坏的」，
却解释不了「为什么 glass 比 RAVEN 的 f32 还快 5.6x」。把 ADC 修好，顶多让 RAVEN 追上自己的 f32 再快个 2x 上下，
到 ~5,000-6,000 QPS，仍然到不了 15,000。

**核心发现 2：榜首不是 ScaNN，是 qsgngt 和 glass。**
- glass 在 recall=0.9523 时达到 15,171 QPS，而 RAVEN 在 recall=0.9517 时只有 2,706 QPS——**差距 5.6x**
- qsgngt 在 recall=0.9917 时达到 11,163 QPS，而 RAVEN 在 recall=0.9961 时只有 2,434 QPS——**差距 4.6x**
- glass 在低参数下同时保持 recall=0.9941 和 QPS=19,801——**图质量极高的标志**

**核心发现 3（v6 新增）：5.6x 差距的性质可能根本不是距离计算，而是图导航效率。**
glass 用 R=32 的小度数就能又准又快，说明它每次查询访问的节点数极少。
如果 RAVEN 的 avg_visited 是 glass 的好几倍，那么距离计算再快 5 倍也没用——
因为做了几倍于对手的距离计算次数。这正是 §1.3 表中那个一直「待测」的 avg_visited。

**待验证假设：图算法质量。** 如果 avg_visited 显著偏高，说明图导航效率有问题，
需要把图质量优化（Phase 3.3）的优先级提前到 Phase 1 之前或并行。详见 §〇.2 Pivot Criterion。

### 1.4 算术不可能定理

> **用文档自己的两个数字做最简单的乘法。**
>
> - §1.3：RAVEN 干净基线 recall 0.95 处 QPS = **2,706**
> - §1.4 预期收益表：Phase 1 端到端乐观估计 = **1.5-2.5x**（Amdahl 稀释后）
>
> 最乐观场景：2,706 × 2.5 = **6,765 QPS**
> 目标线（glass @ recall 0.9523）：**15,171 QPS**
> 即使 Phase 1 拿到全部上限，离榜首还差：15,171 / 6,765 = **2.24x**
>
> Phase 3/4/6 在文档中已被诚实标注为「可能衰减」「+5-10%」，凑不出剩下的 2.24x。
>
> **结论：按当前以 Phase 1 为核心引擎的计划，recall 0.95 处「超越 glass」在数学上不成立。**
> 真正的突破口更可能在 avg_visited 暴露的图质量问题（§1.5），而非距离计算加速。

### 1.5 两个独立问题：距离计算 vs 图导航效率

> **旧叙事**：「核心差距很可能在距离计算方式（f32 vs PQ-ADC）」。
> 这个叙事被 §1.1 的旧数据部分证伪：RAVEN 现在就有 ADC+rerank 路径（2,025 QPS），
> 它非但不快反而比 f32（2,434）还慢。把 ADC 修好，顶多追上 f32 再快 2x 上下，到 ~5,000-6,000，仍然到不了 15,000。

**必须把两个问题分开**：

| 问题 | 性质 | 预期收益 | 对应 Phase | 当前状态 |
|:--|:--|:--|:--|:--|
| **问题 A：RAVEN 的 ADC 路径是坏的** | 明确但收益有限 | ~2x → 到 ~5,000-6,000 QPS | Phase 1 | 旧 ADC 更慢（2,025 vs 2,434） |
| **问题 B：glass 比 RAVEN 的 f32 还快 5.6x** | 更大但更难 | 取决于 avg_visited 差距 | Phase 3.3 | 待诊断 |

> 距离计算这个叙事能解释「为什么 RAVEN 的量化路径是坏的」，
> 但它解释不了「为什么 glass 比 RAVEN 的 f32 还快 5.6x」。
> 后面这个 5.6x，答案更可能在图，不在距离。
> **别让「修 ADC」这个明确但收益有限的任务，掩盖了「图可能根本不够好」这个更大但更难的问题。**

---

## 二、代码审计：已验证的缺陷清单

> 以下每一条均经源码逐行验证，**全部属实**。

### 2.1 严重问题（S 级）

#### S1. ann-benchmarks wrapper 每次查询重建索引 ✅ 已验证

**文件**：`ann_benchmarks/algorithms/raven/__init__.py` + `src/bin/raven_ann_bench.rs`

**证据**：
- `fit()` 构建索引后未传 `--save` 保存。
- `query()` / `query_batch()` 调用二进制时传了 `--train` 但未传 `--load`，导致每次重新建图。

**影响**：benchmark 结果完全不可信，QPS 被建图时间淹没。

---

#### S2. 批量查询 GEMM 是标量回退 ✅ 已验证

**文件**：`src/memory/query_ctx.rs`

**证据**：`gemm_path()` 实际是逐候选调用 `l2_simd`，无 GEMM 矩阵乘法。

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

#### S6. 配置默认值自相矛盾 ✅ 已验证 → ✅ 已修复 (FIX-1)

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
| 研究原型 | 8/10 | 算法实现正确，实验有数据支撑 |
| 工程完整度 | 6.5/10 | 多处有原型/回退而非完整实现 |
| 生产可用度 | 5.5/10 | 缺错误处理、并发查询 |
| 文档遵照度 | 65-75% | 核心算法已实现，外围有差距 |

---

## 三、六阶段冲刺计划

### Phase 0：修复地基（1-3 天）

> **拆分为 0A（打榜命门）和 0B（代码卫生），优先级天差地别。**
> S1 是打榜命门——不修复它，所有 QPS 数据都是假的。
> 0B 的代码卫生改动（如 eprintln!→tracing）碰热路径，必须单独在干净环境下测 before/after，
> 不能和别的修改一起 commit（否则会产生"建图翻倍"的假象）。

#### Phase 0A：打榜命门（必须最先完成）

| 编号 | 任务 | 文件 | 原因 |
|:--|:--|:--|:--|
| 0A.1 | **修复 ann-benchmarks wrapper**：`fit()` 传 `--save`，`query()`/`query_batch()` 传 `--load` | `__init__.py` + `raven_ann_bench.rs` | S1：每次查询重建索引 |
| 0A.2 | **测量干净基线**：QPS + recall + avg_visited | `quick_recall_check.rs` | 所有后续优化的对比基准 |

#### Phase 0B：代码卫生（逐项独立 commit + before/after）

> 每项必须单独在干净环境下测 before/after。不能批量做。

| 编号 | 任务 | 文件 | 原因 |
|:--|:--|:--|:--|
| 0B.1 | `eprintln!` → `tracing`（热路径优先） | `vamana.rs`, `avq.rs`, `opq.rs` 等 | M1：热路径 I/O |
| 0B.2 | `try_into().unwrap()` → `?` 错误传播 | `graph.rs`, `vamana.rs` | M2：健壮性 |
| 0B.3 | `rp_tuning.rs` SchemeB/C 标注 `unimplemented!()` | `rp_tuning.rs` | S4 |
| 0B.4 | `pipeline.rs` final_prune 改用 RobustPrune | `pipeline.rs` | 违反硬约束 |
| 0B.5 | `pipeline.rs` max_iterations 改为 2 | `pipeline.rs` | 违反硬约束 |
| 0B.6 | `delayed_prune.rs` final_prune 改用 RobustPrune | `delayed_prune.rs` | 违反硬约束 |
| 0B.7 | `pipeline.rs` quant_aware_prune 接通真实实现 | `pipeline.rs` | 当前是 no-op |
| 0B.8 | 魔法数字提取为具名常量 | `avq.rs` | 可维护性 |
| 0B.9 | 清理过时 `#[allow(dead_code)]` | `kernel.rs` | 代码整洁 |
| 0B.10 | 明确标注 big-ann / GEMM 为未实现 | `Cargo.toml`, README | S2/S3 |

---

### Phase 1：LUT16 SIMD PQ-ADC 距离计算（3-5 天）🔴 条件性最高优先级

> **条件性优先级——仅当 avg_visited < 150 时成立（见 §〇.2 Pivot Criterion）。**
> 如果 avg_visited ≥ 150，Phase 3.3（图质量）必须先于或并行于 Phase 1。
> **此项未出 avg_visited 结果前，Phase 1 不得启动。**

> **适用例外条款（§〇.1）**：Phase 1 是复合优化，4-bit 量化必然先让 recall 下降，再靠 rerank 补回来。
> 在独立分支上开发，以终态工作点 (recall ≥ 0.95, QPS) 做整体 before/after 评估。

#### 1.1 原理

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
> gather 指令在绝大多数 x86 微架构上并不快：内部仍是串行访存，延迟很高。
> ScaNN 真正的快路径（LUT16）用的是完全不同的机制：
> 1. PQ 量化到 **4-bit**（每子空间 16 个中心，而不是 256）
> 2. 16 个中心的距离量化成 u8 放进一个寄存器
> 3. `_mm256_shuffle_epi8`（pshufb）做**寄存器内查表**——单周期，不碰内存
> 4. `_mm256_maddubs_epi16` 做定点累加

**连锁影响：4-bit 量化会降低 recall**

4-bit PQ（K=16）比 8-bit PQ（K=256）的量化误差更大，recall 会下降。
Phase 1 必须包含 4-bit PQ 训练 + LUT16 SIMD ADC + 基本 rerank，作为一个整体验证 recall。

#### 1.4 预期收益

> **Amdahl 定律警告：LUT16 的 3-5x 加速是相对于"距离计算本身"而言的，不是端到端 QPS。**
>
> 在图索引里，距离计算只是查询时间的一部分，还有堆操作、visited 检查、邻接表遍历、cache miss 等。
> 如果距离计算只占查询总时间的 50%，那 5x 的距离加速最多带来 2x 的端到端提升（Amdahl 定律）。
>
> **这意味着 Phase 1 之后，"非距离开销"（堆操作、visited、访存）的占比会上升，反而可能成为新瓶颈。**
> 所以"Phase 1 吃掉大部分收益、Phase 4 衰减"这个预判**可能恰好反了**——
> Phase 1 之后，Phase 4 的堆优化反而可能变得**更重要**。
>
> **Phase 1 完成后必须做 profiler 时间分解**（距离计算 vs 堆操作 vs 访存等待），用真实占比决定 Phase 4 的优先级。

| 路径 | 当前 QPS | 预期 QPS | 加速比 | 备注 |
|:--|:--|:--|:--|:--|
| f32 全精度 (ef=100) | 待测 | 不变 | 1x | 基线 |
| ADC 标量 (当前) | 待测 | - | - | 比全精度还慢 |
| **LUT16 SIMD ADC (距离层面)** | - | 待验证 | **预计 3-5x（距离层面）** | shuffle 不碰内存 |
| **LUT16 SIMD ADC (端到端 QPS)** | - | 待验证 | **预计 1.5-2.5x（Amdahl 稀释后）** | 取决于距离计算占比 |
| LUT16 SIMD + rerank | - | 待验证 | 待验证 | rerank 补偿 recall |

> **rerank 的带宽陷阱**：4-bit LUT16 的"带宽降 4 倍"优势**只在图导航阶段成立**。rerank 阶段仍然要读全精度 f32 向量（SIFT1M 493MB）。也就是说：
> - 既要存 4-bit PQ codes（为了快），又要存 f32 原始向量（为了 rerank），**内存占用不降反增**。
> - 带宽优势只覆盖查询的前半段（图导航），后半段（rerank）仍是全精度访存。
> - 对 SIFT1M（fit-in-RAM）无所谓，但会影响 Phase 5 的 GIST-960（960 维，f32 向量大得多）和未来 big-ann 的内存预算。
>
> **Phase 5 高维数据集需重新评估内存预算与 rerank 策略**（如降维 rerank、top_n 限制、或 rerank 子采样）。

#### 1.5 复合优化边界声明

| 子步骤 | 预期效果 | 是否单独满足规则 2 |
|:--|:--|:--|
| 1. 实现 4-bit PQ 训练 | 量化器就绪，recall 暂不评估 | 否（中间态） |
| 2. 实现 LUT16 SIMD ADC 距离 | 距离计算变快，但 recall 暴跌 | 否（中间态） |
| 3. 接入 f32 rerank | recall 恢复 | 否（需调参） |
| 4. 调参 ef_search / top_n | **终态：recall ≥ 0.95, QPS 提升** | **是（终态评估）** |

**终态阈值（分层判定）**：

| 判定等级 | 条件 | 处置 |
|:--|:--|:--|
| ✅ 成功 | recall@10 ≥ 0.95 **且** 端到端 QPS ≥ 2,706 × 1.5 = **4,059** | 并入主线 |
| ⚠️ 部分成功 | recall@10 ≥ 0.95 **且** 端到端 QPS ∈ [2,706, 4,059) | **不自动并入主线**；触发 profiler 时间分解复盘，找出 Amdahl 瓶颈，决定是否继续优化 |
| ❌ 失败 | recall@10 < 0.95 **或** 端到端 QPS < 2,706 | 整体回退，记录原因 |

> **为什么不是"非退步"？** Phase 1 是我们自己定义的"决定成败的关键"。如果做完 LUT16 只是"QPS 不低于 f32"，那这个 Phase 在打榜意义上等于失败，却会按规则被判定为"终态达标、并入主线"——这是"成功"的贬值。终态门必须写成量化的最低加速比。

---

### Phase 2：两阶段搜索管道优化（2-3 天）

**目标：LUT16 ADC 快速粗筛 → f32 精确 rerank，在 recall 不变前提下最大化 QPS**

> **注意**：Phase 1 已包含基本 rerank。Phase 2 的重点是 rerank 策略的**精细调优**。

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
- `top_n`：rerank 候选数
- `rerank_strategy`：全量 rerank vs 增量 rerank

#### 2.3 图导航用 ADC 还是 f32？

- **方案 A**：图导航用 ADC 距离（更快但可能走错路 → recall 下降）
- **方案 B**：图导航用 f32，仅最终 rerank 用 ADC（无意义，更慢）
- **方案 C**：图导航用 ADC，但 ef_search 加大到补偿 recall 损失

顶尖库用的是方案 C——ADC 导航 + 大 ef + f32 rerank。

---

### Phase 3：图质量与内存布局优化（3-5 天）

> **前提条件**：Phase 1 完成后，必须用 profiler 测时间分解 + `perf stat` 测 LLC miss。
> 如果 avg_visited 偏高（§1.3），图质量优化（3.3）的优先级应提前。

#### 3.1 图节点重排序（Cache Locality Optimization）

按 BFS 遍历顺序重排节点 ID，使图遍历的内存访问模式变为顺序访问。

#### 3.2 PQ codes 连续存储

将所有节点的 PQ codes 存储为连续的 `Vec<u8>`（N × M 字节），而非 `Vec<Vec<u8>>`。

#### 3.3 图质量提升

| 优化 | 当前 | 目标 | 对 QPS 的影响 |
|:--|:--|:--|:--|
| 初始图用 NN-guided | 随机图 | 用少量近邻引导 | 更少迭代收敛，降低 avg_visited |
| r_max 自适应 | 固定 64 | 按数据集自动选择 | 减少无效距离计算 |
| ef_build 加大 | 200 | 400 | 图质量↑ → ef_search↓ → QPS↑ |

---

### Phase 4：搜索热路径微优化（2-3 天）

> **Phase 1 后优先级可能上升，而非衰减。**
> Phase 1 把距离计算从 compute-bound 拉到 memory-bound 后，"非距离开销"（堆操作、visited、访存）的占比上升。
> 如果 profiler 显示堆操作占比 >20%，Phase 4 的优先级应从 P2 提升到 P1。
> **不要预设它衰减——用 profiler 数据决定。**

#### 4.1 BinaryHeap 优化

用 `BinaryHeap<u64>` 打包 `(distance_bits << 32 | node_id)`，减少比较开销。

#### 4.2 VisitedTracker 优化

可考虑 `Vec<u64>` + bitmap。

#### 4.3 预取策略调优

ADC 路径下预取 PQ codes（4-bit vs 512 字节 f32）的 cache 影响完全不同。

#### 4.4 分支消除

`if !visited[node]` → 无分支版本。

---

### Phase 5：多数据集适配与参数自动调优（2-3 天）

| 数据集 | dim | N | 距离 |
|:--|:--|:--|:--|
| SIFT-128 | 128 | 1M | L2 |
| GIST-960 | 960 | 1M | L2 |
| GloVe-100 | 100 | 1.2M | L2 |

> **高维数据集内存预算警告**：GIST-960 的 f32 向量每条 3.75KB（960×4B），100 万条 = 3.75GB。
> rerank 阶段需读全精度 f32 向量，4-bit PQ codes 的带宽优势仅覆盖图导航段。
> 高维数据集需重新评估内存预算与 rerank 策略（如降维 rerank、top_n 限制、或 rerank 子采样）。

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

## 四、预期最终性能

> **注意**：以下目标建立在"干净环境真实基线"之上。
> **"图算法已足够好"是待验证假设，不是结论。** 如果 avg_visited 偏高，图质量优化需提前。

### 4.1 SIFT1M 预期 Pareto 前沿

| recall@10 | 当前 QPS（待测） | 目标 QPS | vs 榜首外包络 |
|:--|:--|:--|:--|
| 0.90 | 待测 | 待定 | 目标：持平或超越 |
| 0.95 | 待测 | 待定 | 目标：持平或超越 |
| 0.99 | 待测 | 待定 | 目标：持平或超越 |

### 4.2 打榜 vs 论文：明确拆分（含量化器张力）

**打榜只认 recall-QPS 曲线那一根线。**

#### 打榜要靠的（纯吞吐优化）

- Phase 1: LUT16 SIMD PQ-ADC（核心突破点）
- Phase 2: 两阶段搜索 rerank 精细调优
- Phase 3: 内存布局优化（如果 profiler 证明还有 cache 空间）
- Phase 4: 热路径微优化（Phase 1 后优先级可能上升）
- Phase 6: 编译优化 / PGO

#### 论文要靠的（机制证据，不计入打榜分数）

- **RP-Tuning**：一次构建覆盖整条 Pareto 曲线。
- **AVQ 检索感知量化**：比标准 PQ 的 recall 更高。
- **量化感知剪枝**：β/α 协同调参。
- **Rust 安全性 + 确定性构建**。

#### ⚠️ 量化器张力：打榜用 4-bit，论文用 AVQ，可能不是同一套

> **Phase 1 为了 SIMD 速度选了 4-bit PQ（K=16），而 AVQ 的检索感知优势通常在更高码本精度下才明显。**
>
> 这意味着：
> - **打榜量化器**：4-bit PQ（K=16），配合 LUT16-shuffle，追求极致速度
> - **论文量化器**：AVQ（可能 8-bit 或更高），追求 recall 优势
>
> 这两套量化器在码本设计上是**打架的**。必须想清楚：
> 1. 打榜用的量化器和论文消融用的量化器是不是同一个？
> 2. 如果不是，"打榜 vs 论文"的拆分还要再深一层——连量化器本身都是两套。
> 3. 论文的 AVQ 消融实验需要单独的量化器配置，不能直接复用打榜的 4-bit。
>
> **TODO**：在 Phase 1 开始前，明确打榜量化器和论文量化器的配置边界。

---

## 五、优先级排序与时间线

> **收益不可简单连乘。** Phase 1（ADC SIMD）、Phase 3（内存布局）、Phase 4（热路径）针对的是不同瓶颈。
> Phase 1 后瓶颈会转移——距离计算变快后，非距离开销（堆/visited/访存）占比上升。
> **Phase 1 完成后必须用 profiler 测时间分解 + perf stat 测 LLC miss，再决定后续优先级。**
> **如果 avg_visited 偏高，Phase 3.3（图质量）应提前到 Phase 1 之前或并行。**

| 优先级 | Phase | 预期收益 | 时间 | 风险 | 备注 |
|:--|:--|:--|:--|:--|:--|
| 🔴 P0 | Phase 0A: 打榜命门 | 0% (质量) | 0.5-1天 | 🟢 低 | S1 + 基线 |
| 🟡 P1 | Phase 0B: 代码卫生 | 0% (质量) | 1-2天 | 🟢 低 | 逐项独立测 |
| 🔴 P0 | **图质量评估（否决闸门）** | 决定 Phase 1 命运 | 0.5天 | 🟢 低 | **此项未出结果前，Phase 1 不得启动**；avg_visited 决定 Phase 1 vs Phase 3.3 优先级 |
| 🔴 P0* | **Phase 1: LUT16 SIMD PQ-ADC** | **主要收益（待验证）** | 3-5天 | 🟡 中 | *条件性 P0：仅当 avg_visited < 150 时启动；否则降级 P2 |
| 🟡 P1 | Phase 2: 两阶段 rerank 精调 | 边际收益（待验证） | 2-3天 | 🟡 中 | |
| 🟡 P1→P0? | Phase 3: 内存布局 | **取决于 Phase 1 后的 profiler** | 3-5天 | 🟢 低 | 优先级可能上升 |
| 🟢 P2→P1? | Phase 4: 热路径微优化 | **Phase 1 后可能更重要** | 2-3天 | 🟢 低 | 优先级可能上升 |
| 🟢 P2 | Phase 5: 多数据集 | 扩展覆盖 | 2-3天 | 🟡 中 | |
| 🟢 P3 | Phase 6: PGO/NUMA | +5-10% | 2-3天 | 🟢 低 | |

**总计**：约 15-24 天。具体收益待 avg_visited 测量后重新估算。

**执行顺序（锁死，不得跳步）**：

1. **0A.1 修 S1**（无条件先做）→ 不修它，后面任何 QPS 都是假的
2. **0A.2 测 avg_visited**（命运分叉点）→ 这个数字一出来，§〇.2 Pivot Criterion 表自动裁决下一步
3. **照 §〇.2 表走**：avg_visited < 150 → Phase 1 先上；≥ 150 → Phase 3.3 先上
4. **0B 代码卫生**穿插在等待期做，逐项独立、干净环境测 before/after

---

## 六、核心技术风险与对策

| 风险 | 概率 | 影响 | 对策 |
|:--|:--|:--|:--|
| 4-bit 量化导致 recall 下降 | **高** | 高 | Phase 1 内含 rerank 补偿；加大 ef_search；AVQ 优化 codebook 质量 |
| ~~AVX-512 gather 不如预期~~ | ~~低~~ → **已放弃 gather** | - | 改用 LUT16-shuffle（pshufb + maddubs），不碰内存 |
| **Amdahl 稀释：端到端 QPS 提升远小于距离加速** | **高** | 高 | Phase 1 后用 profiler 测时间分解，用真实占比决定后续优化方向 |
| **图质量不高（avg_visited 偏高）** | **中** | 高 | 干净基线测 avg_visited，若偏高则提前 Phase 3.3 |
| Phase 3 cache 优化收益衰减 | 中 | 中 | Phase 1 后用 perf stat 测 LLC miss |
| **4-bit PQ 与 AVQ 码本设计冲突** | 中 | 中 | 打榜/论文用不同量化器配置，明确边界 |
| **rerank 带宽优势仅覆盖图导航段** | **高** | 中 | SIFT1M 无影响；GIST-960 需重新评估内存预算与 rerank 策略 |
| ann-benchmarks 环境差异 | 中 | 中 | 在与 leaderboard 相同的硬件和线程口径上验证 |
| 榜首算法不是 ScaNN（瞄错靶子） | 中 | 中 | 拉取真实排行榜，目标设为所有库的外包络线 |

---

## 七、附录：审计发现优先修复清单

> 按"打榜影响 × 修复成本"排序

### 第一优先：直接影响 benchmark 结果

1. **建立干净基线** ✅ 已完成 → QPS=2,706, recall=0.9517, 建图=912.1s（avg_visited 仍需测）
2. **S1 修复 wrapper** → `fit()` 用 `--save`，`query()` 用 `--load`（否则 QPS 数据全错）
3. **拉取真实排行榜** ✅ 已完成 → 榜首是 qsgngt(11,163 QPS@0.99) 和 glass(15,171 QPS@0.95)，不是 ScaNN
4. **测量 avg_visited（否决闸门）** → 决定 Phase 1 vs Phase 3.3 优先级（§〇.2 Pivot Criterion）
5. **Phase 1 LUT16 SIMD PQ-ADC** → 条件性最高优先级（仅当 avg_visited < 150 时启动）

### 第二优先：代码质量与可维护性（逐项独立测 before/after）

5. **M1 `eprintln!` → `tracing`** → 热路径零 I/O
6. **M2 `unwrap()` → `?`** → 核心库健壮性
7. **S4 RP-Tuning B/C** → 标注 `unimplemented!()` 或实现

### 第三优先：文档诚实度

8. **S2 GEMM 标为实验性**
9. **S3 big-ann 标为未实现**
10. **S5 维度分发说明**

---

## 八、一句话总结

**RAVEN 距离榜首的差距已用真实数据量化：干净基线 QPS=2,706 (recall=0.9517)，榜首 glass 在 recall=0.9523 时达到 15,171 QPS——差距 5.6x。但用文档自己的数字做乘法：基线 2,706 × Phase 1 最乐观 2.5 = 6,765，离目标还差 2.24x。Phase 1（修 ADC）是明确但收益有限的任务，它解释不了「为什么 glass 比 RAVEN 的 f32 还快 5.6x」。真正的答案更可能在 avg_visited 暴露的图导航效率。因此 avg_visited 是当前整个项目最关键的单个数字，拥有 Phase 1 优先级否决权（§〇.2 Pivot Criterion）。测完 avg_visited 再决定：如果图质量是主瓶颈，Phase 3.3 应先于 Phase 1；如果图质量正常，Phase 1 仍是最优路径。在此之前：修复 ann-benchmarks wrapper（S1）+ 测量 avg_visited。**

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
git commit -m "docs: v6 - arithmetic impossibility + pivot criterion"
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
