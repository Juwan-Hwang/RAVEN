# RAVEN 全探索实验记录（汇总版）

> **目标**：将 RAVEN 所有探索实验汇总到一处，避免重复踩坑。
>
> **数据集**：SIFT1M (n=1,000,000, dim=128, nq=10,000)，部分实验用 siftsmall (n=10,000, nq=1,000)
>
> **参照基线**：Glass HNSW — recall=0.9523, QPS=15,171, avg_visited=<150
>
> **文档来源**：`EXPLORATION_LOG.md`（avg_visited 调查）、`优化方案.md`（OPT-1~15）、`RAVEN世界第一冲刺计划.md`（S/D 审计 + 修复记录）、`设计文档.md`（内嵌验证实验）

---

## 一、全探索总览

### A. QPS 优化（查询热路径）

| # | 实验名称 | 探索方向 | 关键结果 | 有效？ |
|---|---------|---------|---------|-------|
| A1 | OPT-1: r_max/ef_search Pareto 扫描 | r_max∈{24~64} × ef∈{50~800} | r_max=48 是性价比拐点 (0.9934/2524) | ✅ 参数依据 |
| A2 | OPT-2: 预取策略优化 | 预取 heap_neighbors vs next_vec vs 无预取 | 方案 B QPS +28%，方案 A 是负优化 | ✅ 已采纳 |
| A3 | OPT-2.5: 距离复用避免重算 | greedy_search 返回距离 | QPS +20.2%，recall 不变 | ✅ 已采纳 |
| A4 | OPT-3: BinaryHeap 替换 | flat sorted Vec vs BinaryHeap | flat Vec 0.59x，BinaryHeap 已最优 | ❌ 已否决 |
| A5 | OPT-4: f16 距离计算加速 | F16C + AVX-512 | dim=128 上 0.50x，F16C 转换开销主导 | ❌ 已否决 |
| A6 | OPT-5: centroid overlay entry_point | medoid vs √N centroid | SIFT1M QPS 仅 +0.5% < 2% 阈值 | ❌ 已否决 |

### B. 建图速度优化

| # | 实验名称 | 探索方向 | 关键结果 | 有效？ |
|---|---------|---------|---------|-------|
| B1 | OPT-6: Fisher-Yates 采样 | 替代 HashSet 初始化随机图 | recall 0.95→0.33，循环间数组状态污染 | ❌ 已否决 |
| B2 | OPT-7: medoid 采样 | 采样 1K 替代全量扫描 | 加速 25x，隔离实验证明对 recall 无影响 | ✅ 已采纳 |
| B3 | OPT-8: rayon 并行粒度 | per-batch vs per-node | task 调度开销仅 0.2%，per_iter 已最优 | ❌ 已否决 |

### C. 量化层优化

| # | 实验名称 | 探索方向 | 关键结果 | 有效？ |
|---|---------|---------|---------|-------|
| C1 | OPT-9: PQ k-means++ 初始化 | k-means++ vs 取前 k 个 | loss 仅改善 1.92%，耗时 15x | ❌ 已否决 |
| C2 | OPT-10: AVQ 归一化方案消融 | Mean/StdDev/Mad/LogSumExp × β | 所有 β>0 均 ≤ β=0 基线 | ❌ 已否决 |
| C3 | OPT-11: β 假设 r_max=64 验证 | β∈{0~2.0} on r_max=64 | β≥0.1 recall 单调下降 | ❌ 已否决 |
| C4 | OPT-12: 对照组 1 (f32) | f32 全精度 baseline | recall=0.9961，量化降级 2.85% | ✅ 已记录 |

### D. 工程清理

| # | 实验名称 | 探索方向 | 关键结果 | 有效？ |
|---|---------|---------|---------|-------|
| D1 | OPT-13: cargo check 警告清理 | unused variable + missing_docs | 纯整洁，无功能影响 | ⏸️ 待执行 |
| D2 | OPT-14: 魔法数字常量化 | avq.rs 硬编码数字 | 纯重构，无功能影响 | ⏸️ 待执行 |
| D3 | OPT-15: 延迟分位数评测 | p50/p99/p999 | 已接入 sift1m_bench.rs | ✅ 已采纳 |

### E. 设计文档一致性审计

| # | 审计项 | 严重度 | 状态 | 关键结论 |
|---|-------|-------|------|---------|
| E1 | S1: ann-benchmarks wrapper 每次查询重建索引 | 🔴 严重 | ✅ 已修复 | fit() 传 --save，query() 传 --load |
| E2 | S2: 批量查询 GEMM 是标量回退 | 🔴 严重 | ✅ 已记录 | gemm_path() 逐候选调用 l2_simd |
| E3 | S3: big-ann/SSD 是空 feature | 🔴 严重 | ✅ 已记录 | big_ann = [] 空特征 |
| E4 | S4: RP-Tuning B/C 未实现 | 🔴 严重 | ✅ 已记录 | SchemeB/C 与 SchemeA 行为相同 |
| E5 | S5: 维度分发是统一 SIMD | 🟡 中等 | ✅ 已记录 | select_l2() 所有分支返回同一个 l2_simd |
| E6 | S6: 配置默认值自相矛盾 | 🔴 严重 | ✅ 已修复 | avq=true + L2 冲突，已修 avq=false |
| E7 | D1: 随机层级导航未实现 | 🔴 严重 | 🔴 #1 优先级 | avg_visited 10x 差距的架构根因 |
| E8 | D2: BuildMetadata 未落盘 | 🔴 严重 | ✅ 已修复 | serialize() 已写入 metadata |
| E9 | D3: rp_tuning.rs 编译错误 | 🔴 已修复 | ✅ 已修复 | 补全 saturate: true 字段 |
| E10 | D4: 局部 alpha 未实现 | 🟡 中等 | ✅ 已标注 | 设计文档说"探索性实验" |
| E11 | D5: parking_lot 未接入 | 🟡 中等 | ⏸️ P2 | Cargo.toml 声明但无 use |
| E12 | D6: GEMM 路径未实现 | 🟡 中等 | ✅ 已记录 | feature-gate 隔离 |
| E13 | D7: RP-Tuning alpha 范围不对齐 | 🟡 中等 | ✅ 已修复 | 补 0.8 和 3.0 |
| E14 | D8: 内存带宽 profiling 未做 | 🟡 中等 | ⏸️ P2 | 无 profiling 代码 |

### F. 关键 Bug 修复

| # | 修复名称 | 问题 | 修复 | 效果 |
|---|---------|------|------|------|
| F1 | FIX-0: init_random_graph 死循环 | n < r_max+1 时死循环 | `r_max.min(n.saturating_sub(1))` | 155 test passed |
| F2 | FIX-1: 配置默认冲突 | avq=true + L2 违反规则 | avq: true → false | merge_config 不再 Err |
| F3 | 0C.4: 建图/查询路径分离 | 建图误用 Two-Pass + ef*64→ef*3 回退 | 建图用简单循环，查询保留 Two-Pass | 建图 610s→408s (-33%)，QPS 4910→7501 (+53%) |

### G. 设计文档内嵌验证实验

| # | 实验名称 | 探索方向 | 关键结果 | 有效？ |
|---|---------|---------|---------|-------|
| G1 | centroid overlay (siftsmall 10K) | medoid vs centroid entry | QPS +3.5%，visited -5%，recall 不变 | ✅ 保留（可选层） |
| G2 | DelayedPruneController | 延迟剪枝 | 功能与 connect_bidirectional 完全重叠 | ❌ 保留作诊断工具 |
| G3 | BuildPipeline 验证 | β=0.0 vs β=0.3 | 两者 recall@10=1.0000（siftsmall） | ✅ 框架可用 |
| G4 | 死代码清理实验 | 6 项死代码扫描 | 核心库零死代码，bin/ 修复后零警告 | ✅ 已完成 |

### H. avg_visited 专项调查

| # | 实验名称 | 探索方向 | avg_visited | recall | 有效？ |
|---|---------|---------|-------------|--------|-------|
| H1 | v8 搜索层消融 | Two-pass/multi-line prefetch | 2444 (不变) | 0.9931 | QPS +40%，**avg_visited 无变化** |
| H2 | v8 saturation A/B | saturate=true vs false | 2444 vs 2442 | 0.9931 | **无效果** |
| H3 | v8 GLASS-COMP 配置 | r_max=32 vs r_max=64 | 1399.5 vs 2443.9 | 0.9703 vs 0.9926 | r_max=32 降 43%，但 recall 也降 |
| H4 | v9 分层导航 10K 验证 | layered nav 小规模 | 735.6 vs 792.3 | 0.9970 | 小规模有效 (-7%) |
| H5 | v9 分层导航 1M 验证 | layered nav 全量 | 1227.4 vs 1399.5 | 0.9705 | 1M 规模仅降 12%，**远不够** |
| H6 | v9.1 ef+po 联合扫描 | ef_search / prefetch_offset 调参 | 1227.4 (ef=50) | 0.9705 | **参数调优不影响 avg_visited** |
| H7 | v9.1 F16 混合精度 | F16 替代 F32 距离计算 | 1227.5 (不变) | 0.9705 | recall 不变，QPS 降 20%，**负收益** |
| H8 | v9.1 nav A/B (同进程) | flat vs layered_nav | 1402.7 vs 1227.9 | 0.9712 vs 0.9710 | nav 降 12%，**但绝对值仍太高** |
| H9 | saturation_probe | saturate + r_max 联合扫描 | 见下文 | 见下文 | **saturation 无效；r_max 降低线性降 visited 但牺牲 recall** |
| H10 | DirectionalPrune A/B | 方向性剪枝 vs RobustPrune | 1178.8 vs 1227.4 | 0.9657 vs 0.9705 | 度数降 14%，visited 仅降 4%，recall 降 0.5% |
| H11 | r_soft 1.5→1.3 | slack factor 对齐 DiskANN | 1227.9 vs 1227.4 | 0.9710 vs 0.9705 | **完全无效** |
| H12 | 搜索终止条件对比 | RAVEN vs Glass LinearPool | — | — | **算法逻辑完全相同，非差异来源** |

---

## 二、QPS 优化实验详细记录

### A1: OPT-1 — r_max / ef_search Pareto 扫描 ✅

**日期**：2026-06-24
**文件**：`pareto_scan.log`
**参数**：α=1.2, l_build=200, max_iter=2, OPQ+AVQ, SIFT1M

| r_max | ef=50 recall | ef=50 QPS | ef=100 recall | ef=100 QPS | avg_degree | build_time(s) |
|:--|:--|:--|:--|:--|:--|:--|
| 24 | 0.9047 | 7519 | 0.9554 | 4438 | 23.9 | 1192 |
| 32 | 0.9472 | 6103 | 0.9786 | 3635 | 31.8 | 1551 |
| 40 | 0.9679 | 5114 | 0.9891 | 2940 | 39.6 | 2124 |
| **48** | 0.9781 | 4227 | **0.9934** | **2524** | 47.2 | 2831 |
| 56 | 0.9838 | 3852 | 0.9953 | 2264 | 54.4 | 3696 |
| 64 | 0.9869 | 3234 | 0.9964 | 2000 | 61.2 | 4754 |

**关键发现**：
1. r_max=48 是性价比拐点：recall=0.9934，QPS=2524，建图比 r_max=64 快 40%
2. "recall≥0.99 且 QPS≥4000" 在当前架构下不可达
3. 建图时间与 r_max 近似线性：每增 8 增 600-1000s
4. QPS 与 avg_degree 反相关

**决策**：旗舰参数维持 r_max=64；r_max=48 作为"快速建图"备选。不修改代码，仅提供参数选择依据。

---

### A2: OPT-2 — 预取策略优化 ✅ (QPS +28%)

**日期**：2026-06-25
**文件**：`src/bin/opt2_bench.rs`
**参数**：SIFT1M base + 随机图 R=32，ef_search=100，10000 查询

| 策略 | QPS | avg_latency | 加速比 |
|:--|---:|---:|---:|
| A: 预取 next_vec（原始） | 17,451 | 57.30us | 1.00x |
| B: 预取 heap_neighbors | **22,305** | 44.83us | **1.28x** |
| C: 组合 A+B | 22,082 | 45.29us | 1.27x |
| D: 无预取 | 20,166 | 49.59us | 1.16x |
| E: 2-ahead 预取 | 21,412 | 46.70us | 1.23x |

**关键发现**：
1. **方案 A（原始预取）是负优化**：比无预取（D）还慢 16%！循环内预取 `neighbors[i+1]` 的向量数据，指令开销超过收益
2. 方案 B 最优：预取堆顶节点的邻居列表
3. recall 不变（0.9960）

**决策**：已采纳。删除循环内 `neighbors[i+1]` 预取，在 pop 后添加 `storage.prefetch_neighbors(top_node)`。

---

### A3: OPT-2.5 — 距离复用避免重算 ✅ (QPS +20.2%)

**日期**：2026-06-24
**参数**：SIFT1M, α=1.2, r_max=32, l_build=100, ef_search=100, k=10

| 指标 | 优化前 | 优化后 | 变化 |
|:--|:--|:--|:--|
| recall@10 | 0.9528 | 0.9528 | 不变 |
| QPS | 2454 | 2949 | +20.2% |
| avg_latency | 0.407ms | 0.339ms | -16.7% |

**决策**：已采纳，commit dc814c8。`greedy_search_vec_reuse` 返回 `Vec<(u32, f32)>`，`search()` 不再重算候选距离。

---

### A4: OPT-3 — BinaryHeap 替换 ❌ 已否决

**日期**：2026-06-25
**文件**：`src/bin/opt3_bench.rs`

| 方案 | QPS | avg_latency | 加速比 |
|:--|---:|---:|---:|
| A: BinaryHeap（当前） | 21,543 | 46.42us | 1.00x |
| B: flat sorted Vec | 12,638 | 79.13us | **0.59x** |

**否决原因**：flat sorted Vec 的 `insert` 是 O(n) memmove（移动 100-200 个 8 字节元素），远比 BinaryHeap 的 O(log n)（7-8 次比较+移动）慢。BinaryHeap 在 ef_search=100 场景下已是最优。

---

### A5: OPT-4 — f16 距离计算加速 ❌ 已否决

**日期**：2026-06-25
**文件**：`src/bin/opt4_bench.rs`, `src/distance/f16.rs`

| 方案 | QPS | avg_latency | 加速比 |
|:--|---:|---:|---:|
| A: f32 AVX-512 (l2_simd) | 11,773,413 | 0.08us | 1.00x |
| B: f16 SIMD (F16C + AVX-512) | 5,930,360 | 0.17us | **0.50x** |

**否决原因**：
1. F16C 转换开销主导：`_mm256_cvtph_ps` 指令数约为 f32 AVX-512 的 2 倍
2. dim=128 向量太小：8 个 AVX-512 chunk，F16C 固定开销占比大
3. 带宽收益无法抵消转换开销

**适用场景**：dim=768/1536 等高维数据集。SIFT1M dim=128 不适合。SIMD f16 距离核保留在 `f16.rs` 中供未来使用。

---

### A6: OPT-5 — centroid overlay entry_point ❌ 已否决

**日期**：2026-06-25
**文件**：`src/bin/opt5_centroid_ablation.rs`

**SIFT1M 数据** (r_max=32, ef=100, √N=1000 centroid)：

| 方案 | recall@10 | QPS | avg_latency |
|:--|:--|:--|:--|
| A. medoid entry | 0.9528 | 3098 | 0.323ms |
| B. centroid entry | 0.9528 | 3114 | 0.321ms |
| 差异 | 0.0000 | +0.5% | -0.6% |

**否决原因**：SIFT1M 上 QPS +0.5% < 2% 验收标准。找最近 centroid 的开销与收益抵消。medoid entry 已足够好（greedy_search 自收敛）。

**决策**：NavigationLayer 代码保留（设计文档要求实现），默认关闭（`enable_centroid_overlay: false`）。

---

## 三、建图速度优化实验详细记录

### B1: OPT-6 — Fisher-Yates 采样 ❌ 已否决（recall 回归）

**日期**：2026-06-24（2026-06-25 隔离实验确认）

**微基准** (n=100K, r_max=64)：Fisher-Yates 2.11x 加速

**SIFT1M 隔离实验**：

| 实验 | OPT-6 | OPT-7 | recall@10 | 建图时间 |
|:--|:--|:--|:--|:--|
| HEAD | ✅ | ✅ | 0.3307 | 121.6s |
| 只回退 OPT-7 | ✅ | ❌ | 0.3311 | 113.8s |
| 只回退 OPT-6 | ❌ | ✅ | **0.9517** | 797.3s |

**否决原因**：partial_shuffle 改变 indices 数组顺序，后续节点的采样分布被前序节点的 shuffle 结果影响 → 随机图质量严重下降 → recall 0.33。建图时间异常短（121s vs 852s）说明图结构极差。**微基准的 2.11x 加速在实际建图中是负优化（图质量 > 采样速度）。**

---

### B2: OPT-7 — medoid 采样 ✅ 已采纳

**日期**：2026-06-24（2026-06-25 隔离实验验证）

**微基准** (n=100K)：全量 3.02ms → 采样 0.12ms（25x 加速）

**隔离实验**：采样 medoid=594 和全量 medoid=123742 在 OPT-6 回退后 recall 均为 0.9517。**medoid 选择对图质量无影响**，greedy_search 自收敛。

**决策**：保留采样版本。全量 medoid 仅需 3ms（在 852s 建图中占 0.0003%），但采样版本也不影响质量。

---

### B3: OPT-8 — rayon 并行粒度 ❌ 已否决

**日期**：2026-06-25
**文件**：`src/bin/opt8_rayon_batch.rs`

| 指标 | 数值 |
|:--|:--|
| par_iter 建图时间 (100K) | 4.9s |
| task 调度总开销 | 10ms (100000 × 100ns) |
| 占比 | **0.2%** |

**外推 SIFT1M**：1M × 100ns = 100ms，占比 0.002%。建图瓶颈在 greedy_search + RobustPrune 计算，不在调度。

---

## 四、量化层优化实验详细记录

### C1: OPT-9 — PQ k-means++ 初始化 ❌ 已否决

**日期**：2026-06-24
**文件**：`src/bin/opt9_bench.rs`

| 方案 | 耗时 | loss | loss 改善 |
|:--|:--|:--|:--|
| A: 取前 k 个 | 1324.79ms | 2560.35 | 0% |
| B: k-means++ 全量 | 19701.52ms | 2511.21 | +1.92% |
| C: 采样 k-means++ | 2328.26ms | 2561.92 | -0.06% |

**否决原因**：SIFT 数据分布均匀，"取前 k 个"初始化已足够好。全量 k-means++ loss 仅改善 1.92%，耗时增 15 倍。

---

### C2: OPT-10 — AVQ 归一化方案消融 ❌ 已否决

**日期**：2026-06-25
**文件**：`src/bin/opt10_normalization_ablation.rs`
**参数**：siftsmall 10K, r_max=32, α=1.0

| 方案 | β=0.0 recall | β=0.3 recall | β=1.0 recall | β=2.0 recall |
|:--|:--|:--|:--|:--|
| **Baseline** | **0.9960** | — | — | — |
| Mean | — | 0.9920 | 0.9840 | 0.9870 |
| StdDev | — | 0.9850 | 0.9830 | 0.9870 |
| Mad | — | 0.9840 | 0.9860 | 0.9850 |
| LogSumExp | — | 0.9910 | 0.9830 | 0.9810 |

**否决原因**：所有 β>0 组合 recall 都低于 β=0 基线。4 种归一化方案都无法改变"量化误差反向影响剪枝"在均匀量化误差数据集上不成立的根本问题。

---

### C3: OPT-11 — β 假设在 r_max=64 上重新验证 ❌ 已否决

**日期**：2026-06-25
**文件**：`src/bin/opt11_beta_rmax64.rs`
**参数**：siftsmall 10K, r_max=64, α=1.0, ef=20（避免天花板）

| β | recall@10 | avg_deg | build_s |
|:--|:--|:--|:--|
| **0.00** | **0.9870** | 29.9 | 1.02 |
| 0.05 | 0.9880 | 29.7 | 1.43 |
| 0.10 | 0.9870 | 29.7 | 1.40 |
| 0.30 | 0.9820 | 29.5 | 1.45 |
| 0.50 | 0.9770 | 29.1 | 1.45 |
| 1.00 | 0.9670 | 28.5 | 1.68 |
| 2.00 | 0.9650 | 28.0 | 1.63 |

**否决原因**：β=0.05 的 0.1% 提升在统计噪声范围内。β≥0.1 后 recall 单调下降。**β 在 SIFT 数据上无正收益**（r_max=32 和 r_max=64 一致）。

**研究结论**：QuantAwareRobustPrune 核心假设"量化误差反向影响图剪枝决策"在 SIFT 数据集上不成立。SIFT 数据的量化误差分布均匀，避免高误差边会扭曲图结构而无收益。负面结果有研究价值，应在论文中报告。

---

### C4: OPT-12 — 对照组 1 (f32 全精度) ✅ 已记录

**日期**：2026-06-24
**参数**：α=1.2, r_max=64, l_build=200, max_iter=2, ef_search=100

| 路径 | recall@10 | QPS |
|:--|:--|:--|
| f32 全精度 | 0.9961 | 2434 |
| AVQ ADC (β=0) | 0.9676 | 2025 |
| ADC+rerank | 0.9676 | 2025 |

**结论**：量化引入 ~2.85% recall 降级，符合预期。对照组 1 补全完成。

---

## 五、设计文档内嵌验证实验详细记录

### G1: centroid overlay (设计文档第340-349行)

**日期**：2026-06-24
**参数**：siftsmall 10K, α=1.0, ef_search=100

| 方案 | recall@10 | QPS | avg_visited | latency_ms |
|:--|:--|:--|:--|:--|
| A. medoid entry | 1.0000 | 19973 | 974.5 | 0.050 |
| B. centroid entry | 1.0000 | 20670 | 925.4 | 0.048 |

**结论**：centroid overlay 有轻微正向收益（QPS +3.5%，visited -5%），recall 不变。centroid 从未恰好等于 medoid（0/100），二者是互补的起点选择策略。NavigationLayer 保留为可选层。（注：SIFT1M 上收益消失，见 A6）

---

### G2: DelayedPruneController (设计文档第374-389行)

**日期**：2026-06-24
**参数**：siftsmall 10K, α=1.0, r_max=32, r_soft=48

**结果**：build 后图状态 avg_degree=18.7, max_degree=32，超过 R_soft/R_max 的节点均为 0。`final_prune` 后图结构完全不变。

**结论**：DelayedPruneController 的 `should_prune` + `final_prune` 逻辑已由 `connect_bidirectional` + `VamanaGraph::final_prune` 内联实现，功能完全重叠。保留作诊断工具（prune_count 统计），不作为生产路径。

---

### G3: BuildPipeline 验证 (设计文档第422-431行)

**日期**：2026-06-24
**参数**：siftsmall 10K

| β | recall@10 |
|:--|:--|
| 0.0 (标准 RobustPrune) | 1.0000 |
| 0.3 (量化感知 RobustPrune) | 1.0000 |

**结论**：BuildPipeline 是消融实验框架的参考实现，能正常工作。主 benchmark 使用优化路径（learn 100K + iter=5），不经过 BuildPipeline。

---

### G4: 死代码清理实验 (设计文档附录 F.13)

**日期**：2026-06-24

1. `avq_recall_probe.rs`：三个死函数接入 main，验证 ADC 原则 + rerank 增益
2. `sift1m_bench.rs`：recall_at_k 工具函数消除代码重复
3. `beta_scan.rs`：adc_recall_rerank 函数删除（main 内联版本更高效）
4. `bandwidth_probe.rs`：删除冗余 elapsed_us 字段
5. `appendix_a_experiment.rs`：退化判定中打印 method + build_time_s
6. `kernel_select.rs`：删除 verbose 变量

**最终状态**：核心库零死代码，`cargo build --all-features` 零 "never used" / "never read" 警告。

---

## 六、avg_visited 专项调查详细记录

### H1: v8 搜索层消融 (opt_ablation)

**时间**: v8.0 时期
**文件**: `src/bin/opt_ablation.rs` → `opt_ablation_result.txt`
**图参数**: r_max=64, r_soft=96, alpha=1.2, l_build=200, max_iter=2, saturate=true

| 配置 | ef=50 QPS | ef=50 avg_visited | 相对 baseline |
|------|-----------|-------------------|---------------|
| baseline (无预取) | 2140 | 2444 | 1.00x |
| two_pass(po=4) | 2502 | 2444 | 1.17x |
| two_pass(po=8) | 2968 | 2444 | 1.39x |
| multi_pref (4 cache lines) | 2463 | 2444 | 1.15x |
| combined(po=8) | 2992 | 2444 | 1.40x |

**结论**: 搜索层预取优化可提升 QPS 40%，但 `avg_visited` 完全不变 — 预取只影响 cache miss 延迟，不影响搜索路径。**方向：搜索层微优化已到天花板。**

---

### H2: v8 saturation A/B

**时间**: v8.0 时期
**文件**: `src/bin/opt_ablation.rs` Part 2
**图参数**: r_max=64, alpha=1.2, l_build=200, max_iter=2

| 配置 | ef=50 recall | ef=50 QPS | ef=50 avg_visited |
|------|-------------|-----------|-------------------|
| saturate=true | 0.9931 | 3069 | 2444 |
| saturate=false | 0.9931 | 3362 | 2442 |

**结论**: saturation 对 avg_visited 和 recall 均无影响（差异 <0.1%）。alpha=1.2 本身会填满度数，saturation 只是锦上添花。**排除 saturation 作为根因。**

---

### H3: v8 GLASS-COMP 配置 (r_max 降级)

**时间**: v8.0 时期
**文件**: `src/bin/quick_recall_check.rs` → `v8_benchmark.txt`, `v8_final.txt`

| 配置 | r_max | ef=50 recall | ef=50 QPS | ef=50 avg_visited |
|------|-------|-------------|-----------|-------------------|
| CANONICAL | 64 | 0.9926 | 3607 | 2443.9 |
| GLASS-COMP | 32 | 0.9703 | 4910 | 1399.5 |

**结论**: r_max 从 64 降到 32，avg_visited 降 43%，但 recall 也降 2.3%。这是因为度数减半 → 邻居覆盖减少 → 需要更多跳数但每跳展开更少。**线性降 r_max 不是解法 — recall 代价太大。**

---

### H4: v9 分层导航 10K 验证

**时间**: v9.0 时期
**文件**: `src/bin/nav_verify.rs` → `nav_verify_result.txt`
**数据**: n=10,000, nq=200

| 配置 | ef=50 recall | ef=50 QPS | ef=50 avg_visited |
|------|-------------|-----------|-------------------|
| flat Vamana | 0.9970 | 16663 | 792.3 |
| v9 分层导航 | 0.9970 | 24009 | 735.6 |

**结论**: 10K 规模下分层导航有效：avg_visited 降 7%，QPS 提 44%。小规模上图密度高，导航层能快速定位到近邻区域。

---

### H5: v9 分层导航 1M 验证

**时间**: v9.0 时期
**文件**: `src/bin/nav_verify_1m.rs` → `nav_verify_1m_v91.txt`

| 配置 | ef=50 recall | ef=50 QPS | ef=50 avg_visited |
|------|-------------|-----------|-------------------|
| flat Vamana (v3 旧导航) | 0.9703 | 7501 | 1399.5 |
| v9.1 分层导航 | 0.9705 | 5019 | 1227.4 |
| Glass HNSW (参照) | 0.9523 | 15171 | <150 |

**结论**: 1M 规模下分层导航仅降 12% avg_visited（1399→1227），但 QPS 反而降了（导航层构建+搜索开销 > 收益）。**分层导航不是解决 avg_visited 的银弹。差距仍有 8 倍。**

---

### H6: v9.1 ef+po 联合参数扫描

**时间**: v9.1 时期
**文件**: `src/bin/ef_po_sweep.rs` → `ef_po_sweep.txt`

**Phase 1: ef 扫描 (po=8 固定)**

| ef | recall | QPS | avg_visited |
|----|--------|-----|------------|
| 20 | 0.8856 | 18029 | 593.1 |
| 30 | 0.9327 | 13788 | 810.4 |
| 40 | 0.9565 | 11105 | 1021.8 |
| **50** | **0.9705** | **9001** | **1227.4** |
| 60 | 0.9788 | 7588 | 1428.1 |
| 80 | 0.9877 | 5811 | 1816.7 |

**Phase 2: po 扫描 (ef=50 固定)**

| po | recall | QPS | avg_visited |
|----|--------|-----|------------|
| 0 | 0.9705 | 7499 | 1227.4 |
| **2** | **0.9705** | **9195** | **1227.4** |
| 4 | 0.9705 | 8704 | 1227.4 |
| 8 | 0.9705 | 8966 | 1227.4 |
| 16 | 0.9705 | 8219 | 1227.4 |

**结论**: avg_visited 与 ef 成正比（~24.5×ef），与 po 无关。ef=50/po=2 是最优工作点。**参数调优不改变 avg_visited 的本质 — 它由图结构决定，不由搜索参数决定。**

---

### H7: v9.1 F16 混合精度

**时间**: v9.1 时期
**文件**: `src/bin/f16_bench.rs` → `f16_bench.txt`

| 配置 | ef=50 recall | ef=50 QPS | avg_visited | 加速比 |
|------|-------------|-----------|------------|-------|
| F32 | 0.9705 | 8504 | 1227.4 | 1.00x |
| F16 | 0.9705 | 6724 | 1227.5 | 0.79x |

**结论**: F16 距离计算在 SIFT1M 上比 F32 慢 21%。128 维向量的 F16 转换开销 > 计算节省。recall 不变（精度无损）。**F16 在此场景下是负优化，已排除。**（与 A5/OPT-4 微基准结论一致）

---

### H8: v9.1 nav A/B (同进程对比)

**时间**: v9.1 时期
**文件**: `src/bin/nav_ab_test.rs` → `nav_ab_test.txt`

**ef=50, 三轮取最佳：**

| 配置 | recall | QPS | avg_visited |
|------|--------|-----|------------|
| A: flat Vamana (无导航) | 0.9712 | 8437 | 1402.7 |
| B: v9.1 分层导航 | 0.9710 | 9468 | 1227.9 |

**ef=100, 三轮取最佳：**

| 配置 | recall | QPS | avg_visited |
|------|--------|-----|------------|
| A: flat Vamana | 0.9920 | 4839 | 2359.4 |
| B: v9.1 分层导航 | 0.9921 | 5186 | 2189.9 |

**结论**: 分层导航稳定降低 ~12% avg_visited，同时 QPS 提升约 12%。但绝对值（1228 vs Glass <150）仍有 8 倍差距。**导航层的贪心下降只省了入口定位的开销，Layer 0 搜索才是大头。**

---

### H9: saturation_probe (saturation + r_max 联合扫描)

**时间**: v9.1 时期
**文件**: `src/bin/saturation_probe.rs` → `saturation_probe_result.txt`
**参数**: alpha=1.2, l_build=200, max_iter=2, layered_nav=true, ef=50, po=2

| 配置 | r_max | sat | degree | recall | QPS | avg_visited |
|------|-------|-----|--------|--------|-----|------------|
| baseline | 32 | true | 32.0 | 0.9705 | 9195 | 1227.4 |
| no_sat_r32 | 32 | false | 32.0 | 0.9704 | 9003 | 1227.3 |
| no_sat_r24 | 24 | false | 24.0 | 0.9511 | 10296 | 954.9 |
| no_sat_r20 | 20 | false | 20.0 | 0.9324 | 12461 | 811.4 |
| no_sat_r16 | 16 | false | 16.0 | 0.8995 | 14678 | 664.2 |

**结论**:
1. **saturation 无效果**: no_sat_r32 ≈ baseline，度数都是 32.0，avg_visited 差异 <0.01%。alpha=1.2 本身就会填满 r_max。
2. **r_max 线性降低可降 avg_visited**: r_max 每降 4，avg_visited 降 ~150。但 recall 也线性下降。
3. **降 r_max 到 16 时 avg_visited=664，仍远高于 Glass <150**。即使度数减半，差距仍有 4 倍。
4. **排除 saturation 作为根因；r_max 降低不是有效手段（recall 代价太大）。**

---

### H10: DirectionalPrune A/B

**时间**: v9.1 时期
**文件**: `src/bin/directional_prune_ab.rs` → `directional_prune_ab_result.txt`
**设计**: Pass 1 用 α=1.0 纯方向性扫描（保留朝向 query 方向的邻居）；Pass 2 仅在度数不足 r_min=r_max/4 时用 α=1.2 backfill。

| 配置 | degree | ef=50 recall | ef=50 QPS | ef=50 avg_visited |
|------|--------|-------------|-----------|-------------------|
| RobustPrune (α=1.2) | 32.0 | 0.9705 | 8272 | 1227.4 |
| DirectionalPrune (α=1.0+r_min) | 27.6 | 0.9657 | 3036 | 1178.8 |

**ef 扫描对比：**

| ef | RobustPrune recall | DirPrune recall | RobustPrune visited | DirPrune visited |
|----|-------------------|-----------------|---------------------|------------------|
| 40 | 0.9565 | 0.9496 | 1021.8 | 979.2 |
| 50 | 0.9705 | 0.9657 | 1227.4 | 1178.8 |
| 60 | 0.9788 | 0.9756 | 1428.1 | 1373.7 |
| 100 | 0.9921 | 0.9910 | 2189.2 | 2115.7 |

**结论**:
1. DirectionalPrune 将度数从 32.0 降到 27.6（-14%），但 avg_visited 仅降 4%（1227→1178）。
2. recall 在所有 ef 点都更低（-0.5%~-0.7%）。
3. **度数降 14% 只带来 4% avg_visited 下降，说明边质量更差** — 方向性剪枝去掉的边恰恰是帮助搜索快速收敛的"桥接边"。
4. Pass 2 在 128 维 SIFT 上从未触发（Pass 1 α=1.0 本身产出 ~27 条边，超过 r_min=8）。
5. **DirectionalPrune 框架已实现但效果不理想，已排除。**

---

### H11: r_soft (GRAPH_SLACK_FACTOR) 1.5 → 1.3

**时间**: v9.1 时期
**文件**: `src/bin/nav_ab_test.rs` (修改 r_soft) → `slack13_result.txt`
**背景**: DiskANN 源码中 `GRAPH_SLACK_FACTOR = 1.3`，RAVEN 原为 1.5。r_soft 控制反向边触发剪枝的阈值：r_soft = r_max × slack_factor。

| 配置 | r_soft | recall | QPS | avg_visited |
|------|--------|--------|-----|------------|
| A: flat, r_soft=1.5×32=48 | 48 | 0.9712 | 8274 | 1402.7 |
| B: v9.1 nav, r_soft=1.3×32=41.6 | 41.6 | 0.9710 | 9468 | 1227.9 |
| B (之前 r_soft=1.5) | 48 | 0.9705 | 9468 | 1227.4 |

**结论**: r_soft 从 1.5 降到 1.3，avg_visited 从 1227.4 变为 1227.9（**反而略升**）。recall 和 QPS 无变化。**slack factor 完全不是影响因素，已排除。**

---

### H12: 搜索终止条件对比 (RAVEN vs Glass)

**时间**: v9.1 时期
**文件**: `src/graph/linear_pool.rs` vs `glass-ref/pyglass-main/glass/neighbor.hpp`

**RAVEN LinearPool**:
```rust
pub fn has_next(&self) -> bool {
    self.cursor < self.size && self.cursor < self.ef
}

pub fn insert(&mut self, node: u32, dist: f32) -> bool {
    if self.size == self.capacity {
        let worst = unsafe { *self.data.get_unchecked(self.size - 1) };
        if dist >= worst.1 { return false; }
    }
    // ... 二分插入
}
```

**Glass LinearPool**:
```cpp
bool has_next() const { return cur_ < size_ && cur_ < ef_; }

bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
        return false;
    }
    // ... 二分插入
}
```

**结论**: **两者算法逻辑完全相同** — 终止条件 = 弹出 ef 个元素后停止；拒绝条件 = 池满且距离 >= worst。没有 early termination（候选距离远超结果集最差时提前停止）。**8 倍差距不在搜索算法层，在图结构层。**

**用户纠正**: avg_visited ≠ 跳数。avg_visited=1227 是被评估距离的唯一节点总数。每次 pop 展开一个节点的全部邻居（~32 个），其中未访问的才计入 visited。实际弹出次数 ≈ avg_visited / 每跳平均新增唯一邻居数。Glass 的 <150 visited 意味着搜索快速收敛到一个局部簇，后续邻居几乎全是已访问的重复节点。

---

## 七、关键数据对比汇总

### RAVEN 各版本演进

| 版本 | 配置 | recall | QPS | avg_visited | 建图时间 |
|------|------|--------|-----|------------|---------|
| v7 旧版 | r_max=64, 无 nav | 0.9517 | 2706 | — | 912s |
| v8 GLASS-COMP | r_max=32, 无 nav | 0.9703 | 4910 | 1399.5 | 610s |
| v8 CANONICAL | r_max=64, 无 nav | 0.9926 | 3607 | 2443.9 | 1045s |
| v9.1 baseline | r_max=32, nav, ef=50, po=2 | 0.9705 | 9195 | 1227.4 | 395s |
| v9.1 + DirPrune | r_max=32, nav, DirPrune | 0.9657 | 3036 | 1178.8 | 640s |
| v9.1 + r_soft=1.3 | r_max=32, nav, slack=1.3 | 0.9710 | 9468 | 1227.9 | 423s |
| **Glass HNSW** | R=32, L=200 | **0.9523** | **15171** | **<150** | **98s** |

### GLASS-COMP v3 版本对比（建图/查询路径分离修复前后）

| 版本 | 建图时间 | QPS@ef=50 | QPS@ef=100 | recall | avg_visited | 修复内容 |
|:--|:--|:--|:--|:--|:--|:--|
| v2 (0b99927) | 444.8s | 4,910 | 4,070 | 0.9703 | 1,399.5 | 基线 |
| v2-dirty (29a5f0e) | 609.8s | 4,910 | 4,070 | 0.9703 | 1,399.5 | visited.rs ef*64→ef*3 回退 |
| v3 (9e534e3) | 603.3s | 6,811 | 3,010 | 0.9703 | 1,399.5 | + map_init（但误用 Two-Pass 到建图） |
| **v3-fix (1ace197)** | **408.0s** | **7,501** | **4,479** | 0.9703 | 1,399.5 | 建图用简单循环 + 恢复 ef*64 + 查询保留 Two-Pass |

### RAVEN vs Glass 结构差异

| 指标 | RAVEN v9.1 | Glass HNSW |
|------|-----------|------------|
| 建图模式 | Vamana batch (全量 → 迭代优化) | HNSW incremental (逐点插入) |
| 度数 mean | 32.0 (全部饱和) | 20.7 (自然分布, min=1) |
| 建图时间 | 395s | 98s |
| avg_visited (ef=50) | 1227 | <150 |
| 搜索器逻辑 | LinearPool (与 Glass 相同) | LinearPool |
| 分层导航 | 独立上层图 (4 层) | HNSW 原生层级 |

### CANONICAL/GLASS-COMP 完整扫描数据

#### CANONICAL（200/64/2，建图 6,657.3s）

| ef_search | recall@10 | QPS | avg_visited | p50 | p99 |
|:--|:--|:--|:--|:--|:--|
| 50 | 0.9868 | 1,801 | **2,462.5** | 2,484 | 3,237 |
| 100 | 0.9965 | 1,112 | 4,012.6 | 4,100 | 5,280 |
| 200 | 0.9986 | 651 | 6,621.9 | 6,786 | 9,011 |

#### GLASS-COMP v3（200/32/2，建图 408.0s）

| ef_search | recall@10 | QPS | avg_visited | p50 | p99 |
|:--|:--|:--|:--|:--|:--|
| 50 | 0.9703 | **7,501** | **1,399.5** | 1,425 | 1,777 |
| 100 | 0.9920 | **4,479** | 2,355.5 | 2,440 | 2,939 |
| 200 | 0.9980 | **2,579** | 4,044.8 | 4,214 | 5,185 |

---

## 八、已排除的方向（不要重复尝试）

### 搜索层

1. **搜索层预取优化** — H1 证明只提 QPS 不降 avg_visited。（注：QPS 层面的预取优化 A2 仍然有效）
2. **ef/po 参数调优** — H6 证明 avg_visited 由图结构决定，不由搜索参数决定。
3. **搜索终止条件** — H12 证明 RAVEN 和 Glass 逻辑完全相同。
4. **BinaryHeap 替换** — A4 证明 flat sorted Vec 0.59x，BinaryHeap 已最优。

### 距离计算层

5. **F16 混合精度** — A5/H7 证明在 128 维上是负优化（0.50x~0.79x）。
6. **距离复用** — A3 已采纳（QPS +20.2%），但不影响 avg_visited。

### 剪枝策略层

7. **DirectionalPrune** — H10 证明度数降 14% 只换 4% visited 下降，边质量更差。
8. **r_soft (slack factor) 调整** — H11 证明 1.5→1.3 完全无效。
9. **saturation 开/关** — H2/H9 证明无效果。alpha=1.2 本身填满度数。

### 参数调优层

10. **r_max 线性降低** — H9 证明 recall 代价太大。r_max=16 时 avg_visited=664 仍远高于 150。
11. **r_max/ef_search Pareto 扫描** — A1 证明"recall≥0.99 且 QPS≥4000"在当前架构下不可达。

### 建图层

12. **Fisher-Yates 采样** — B1 证明导致 recall 0.95→0.33（图质量 > 采样速度）。
13. **rayon 并行粒度调整** — B3 证明 task 调度开销仅 0.2%。
14. **medoid 全量 vs 采样** — B2 证明无影响（greedy_search 自收敛）。

### 量化层

15. **PQ k-means++ 初始化** — C1 证明 loss 仅改善 1.92%，耗时 15x。
16. **AVQ 归一化方案（Mean/StdDev/Mad/LogSumExp）** — C2 证明所有 β>0 均 ≤ β=0 基线。
17. **β 假设在 r_max=64 上** — C3 证明 β≥0.1 recall 单调下降。**β 在 SIFT 数据上无正收益。**

### Entry Point 层

18. **centroid overlay** — A6 证明 SIFT1M 上 QPS 仅 +0.5%。siftsmall 上有轻微收益（G1），但 SIFT1M 上消失。

### 分层导航

19. **独立上层图分层导航** — H4/H5/H8 证明 1M 规模仅降 12% avg_visited，绝对值仍太高（1228 vs <150）。

---

## 九、核心结论

### 差距不在哪里

1. **不在搜索算法层**: LinearPool 的 `has_next()` / `insert()` 逻辑完全一致（H12）。
2. **不在剪枝策略层**: RobustPrune vs DirectionalPrune 差异微小（H10）。
3. **不在参数调优层**: saturation / r_soft / po 均无效（H2/H9/H11）。
4. **不在距离计算层**: F16 负优化，距离复用只提 QPS 不降 avg_visited（A3/A5/H7）。
5. **不在量化层**: β 在 SIFT 数据上无正收益（C2/C3）。
6. **不在并行调度层**: rayon task 开销 0.2%（B3）。
7. **不在随机图初始化**: Fisher-Yates 采样导致 recall 暴跌（B1）。
8. **不在 entry point**: centroid overlay SIFT1M 上收益可忽略（A6）。

### 差距在哪里（推测）

1. **在建图模式**: Vamana batch build 产出的图导航质量远不如 HNSW incremental insertion。同样的搜索器在 Glass 图上走几跳就收敛，在 RAVEN 图上要展开上千个节点。
2. **在图结构质量**: RAVEN 度数全部饱和到 32.0（均匀），Glass 度数自然分布（avg=20.7, min=1）。Glass 的稀疏节点可能充当"高速公路"。
3. **在邻居重叠率**: Glass 图上搜索快速收敛到一个局部簇，后续展开的邻居几乎全是已访问节点（高重叠）。RAVEN 图上邻居分散，每次展开都引入大量新节点（低重叠）。

### 下一步方向

1. **图结构指标对比**: 测量 RAVEN vs Glass 的聚类系数、平均路径长度、邻居重叠率，定位具体结构差异。
2. **搜索过程诊断**: 在搜索中统计 result_set 更新次数（improvements），判断是否在"无效展开"。
3. **建图模式对比**: 考虑实现 HNSW incremental insertion 作为替代建图策略。
4. **visited 标记时机**: 确认 RAVEN 在 `insert` 拒绝前就标记了 `visited`（当前代码确认如此），评估是否应改为 insert 成功后才标记。

---

## 十、待执行项

| 编号 | 任务 | 优先级 | 来源 | 状态 |
|:--|:--|:--|:--|:--|
| D1 | 实现随机层级导航 (HNSW 风格分层) | 🔴 P0 | 冲刺计划 0C.2 | 待执行 |
| D5 | parking_lot 接入 | 🟢 P2 | 冲刺计划 0C.6 | 待执行 |
| D8 | 内存带宽 profiling (LLC miss) | 🟢 P2 | 冲刺计划 0C.7 | 待执行 |
| OPT-13 | cargo check 警告清理 | 🟢 P3 | 优化方案 | 待执行 |
| OPT-14 | 魔法数字具名常量化 | 🟢 P3 | 优化方案 | 待执行 |
| — | 图结构指标对比 (聚类系数等) | 🟡 P1 | 本文核心结论 | 待执行 |
| — | 搜索过程 result_set 更新次数诊断 | 🟡 P1 | 本文核心结论 | 待执行 |
