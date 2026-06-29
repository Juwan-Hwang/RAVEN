//! Vamana/DiskANN 风格图构建
//!
//! 设计文档第三层：
//! 主线选 Vamana/DiskANN 风格。传统工作流中调整 α 常需重建或重跑构建流程，
//! RP-Tuning 提供了 post-hoc 调优路径，是系统级卖点。
//!
//! Vamana 构建流程：
//! 1. 初始化图（随机连接或健康度优先）
//! 2. 对每个节点执行贪心搜索得到候选集
//! 3. 对候选集执行 RobustPrune（带 α 参数）
//! 4. 双向连接 + 度数控制

use crate::distance::l2_simd;
use crate::memory::{HybridBlockedCsr, VisitedTracker};
use crate::quant::{SQ8Dataset, PQ4Dataset, PQ8Dataset};
use super::adaptive_ef::AdaptiveEfConfig;
use crate::build::ChaCha8Rng;
use crate::build::{BuildConfig, BuildMetadata};
use crate::graph::navigation::NavigationLayer;
use crate::graph::navigation::LayeredNavigation;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Vamana 图构建配置
#[derive(Debug, Clone)]
pub struct VamanaBuildConfig {
    /// 全局 α（构建时固定，数据集级默认值）
    /// 设计文档第三层 α 三段式①：全局 α
    pub alpha: f32,
    /// 构建期搜索宽度
    pub l_build: usize,
    /// 最终图最大出度 R_max（设计文档第四层：硬上限）
    pub r_max: usize,
/// 构建期软上限 R_soft = 1.3 × R_max（对齐 DiskANN GRAPH_SLACK_FACTOR）
/// 度数超过 R_soft 时触发延迟剪枝；R_soft 越小，反向边积累越少，低质量边更早被清除
pub r_soft: usize,
    /// 最大迭代轮数
    pub max_iterations: usize,
    /// 是否在 RobustPrune 后用剩余候选填充到 R_max（saturation）
    ///
    /// DiskANN: saturate_after_prune(true) + alpha > 1.0
    /// 但 saturation 会用远距离候选填充，拉低邻居重叠率 → avg_visited 膨胀
    /// 实验开关：false 时图自然稀疏，邻居全是 RobustPrune 精选的高质量边
    pub saturate: bool,
    /// 是否启用分层导航（设计文档：保留随机层级，可与 HNSW 直接对比）
    pub enable_layered_nav: bool,
    /// 分层导航参数 M（层间缩减比，默认 16）
    pub nav_m: usize,
    /// 剪枝策略：RobustPrune（Vamana 标准）或 DirectionalPrune（RAVEN 超越方案）
    /// DirectionalPrune 无 saturation，用 r_min 连通性补底替代 r_max 填充
    pub prune_strategy: super::robust_prune::PruneStrategy,
}

impl Default for VamanaBuildConfig {
    fn default() -> Self {
        let r_max = 64;
        Self {
            alpha: 1.2,
            l_build: 200,
            r_max,
            r_soft: (r_max as f32 * 1.3) as usize, // DiskANN GRAPH_SLACK_FACTOR=1.3
            // Vamana 论文要求 two passes：第一轮 α=1.0（连通性），第二轮 α=config（长程边）
            // max_iterations=1 只跑连通性轮，图质量显著下降（recall 0.33→0.95 的差距来源之一）
max_iterations: 2,
saturate: true,
enable_layered_nav: true,
nav_m: 16,
prune_strategy: super::robust_prune::PruneStrategy::RobustPrune,
        }
    }
}

/// Vamana 图
///
/// 设计文档第三层主线：Vamana/DiskANN 风格图索引
/// 内部使用 HybridBlockedCsr 存储
pub struct VamanaGraph {
/// 图存储
storage: HybridBlockedCsr,
/// 入口节点（随机层级导航起点）
entry_point: u32,
/// 向量维度
dim: usize,
/// 节点数
n: usize,
/// 构建元数据（设计文档 F.7：随索引文件落盘）
/// `None` 表示从旧格式反序列化或由 `from_storage` 构造
metadata: Option<BuildMetadata>,
/// HNSW 风格分层导航（设计文档：保留随机层级）
/// 搜索时从顶层贪心走到 Layer 0 入口，大幅减少 avg_visited
layered_nav: Option<LayeredNavigation>,
}

impl VamanaGraph {
    /// 构建图
    ///
    /// vectors: 扁平存储的向量，vectors[i*dim..(i+1)*dim] 是第 i 个向量
    pub fn build(
        vectors: &[f32],
        dim: usize,
        config: &VamanaBuildConfig,
        rng: &mut ChaCha8Rng,
    ) -> Self {
        let n = vectors.len() / dim;
        assert_eq!(vectors.len(), n * dim);
        let mut storage = HybridBlockedCsr::new(n, config.r_max);

        eprintln!("[build] rayon threads: {}", rayon::current_num_threads());

        // 1. 初始化图：随机连接（设计文档上层导航：保留随机层级）
        let _random_entry = Self::init_random_graph(&mut storage, n, config, rng);
        // Vamana/DiskANN 论文：entry_point 用 medoid（离质心最近的点），不是随机点
        let entry_point = Self::compute_medoid(vectors, dim, n);
        eprintln!("[build] init_random_graph done, entry=medoid[{}]", entry_point);

        // 2. 迭代优化：小批量并行 greedy_search + RobustPrune，批间顺序写入
        // Vamana 论文 two passes：第一轮 α=1.0（连通性），第二轮 α=config.alpha（长程边）
        //
        // v6.6 关键修复：原实现全量并行（1M 节点同时在同一张旧图上搜索），
        // 违反 Vamana 论文 Algorithm 1 的顺序处理要求。
        // 全量并行导致所有节点看到相同的（差的）图状态，图质量严重退化
        // （avg_visited=2325，正常应 100-300）。
        // 小批量处理：每批 BUILD_BATCH_SIZE 个节点并行搜索，
        // 批间顺序更新图，后续批次受益于前面批次的改进。
        const BUILD_BATCH_SIZE: usize = 10_000;
        for iter in 0..config.max_iterations {
            let order = Self::permutation(n, rng);
            let progress = AtomicUsize::new(0);
            let progress_interval = (n / 20).max(1);
            let alpha = if iter == 0 { 1.0 } else { config.alpha };
            eprintln!("[build] iter {}/{} alpha={}", iter + 1, config.max_iterations, alpha);

            for chunk in order.chunks(BUILD_BATCH_SIZE) {
                // 批内并行：在当前图状态上搜索 + 剪枝
                // map_init 复用 VisitedTracker：visited.reset() 只清 history（~1400 entries），
                // 避免 vec![0u8; 1M] 的 1MB memset（2M 次 × 1MB = 2TB memset 流量）
                // 用 greedy_search_vec_build（简单单循环），不用 Two-Pass Prefetch（建图场景纯开销）
                let new_neighbors: Vec<(u32, Vec<u32>)> = chunk
                    .par_iter()
                    .map_init(
                        || VisitedTracker::new(n, config.l_build),
                        |visited, &node_id| {
                            let idx = progress.fetch_add(1, Ordering::Relaxed);
                            if idx > 0 && idx % progress_interval == 0 {
                                eprintln!("[build] {}/{} ({}%)", idx, n, idx * 100 / n);
                            }
                            let query = &vectors[node_id as usize * dim..(node_id as usize + 1) * dim];
                            let _candidates = Self::greedy_search_vec_build(
                                vectors, dim, &storage, entry_point, query, config.l_build,
                                visited,
                            );
                            let pruned = prune_dispatch(
                                config.prune_strategy,
                                visited.visited_nodes(), node_id, vectors, dim, alpha, config.r_max,
                                config.saturate && alpha > 1.0,
                            );
                            (node_id, pruned)
                        }
                    )
                    .collect();

                // 批间顺序写入：更新图，下一批搜索受益
                for (node_id, pruned) in new_neighbors {
                    Self::connect_bidirectional(
                        &mut storage, node_id, &pruned, vectors, dim, alpha, config,
                    );
                }
            }
        }

        // 3. 全局 final prune 到 R_max（用 RobustPrune，不是 truncate）
        Self::final_prune(&mut storage, vectors, dim, config.alpha, config.r_max, config.saturate, config.prune_strategy);

        // 创建构建元数据（设计文档 F.7）
        let build_config = BuildConfig::default();
        let metadata = BuildMetadata::from_config(&build_config, n, dim);

        // 构建分层导航（设计文档：保留随机层级，可与 HNSW 直接对比）
        let layered_nav = if config.enable_layered_nav && n > 0 {
            eprintln!("[build] constructing layered navigation (M={})...", config.nav_m);
            let t0 = std::time::Instant::now();
            let nav = LayeredNavigation::build(
                vectors, dim, &storage, entry_point, config.nav_m, config.r_max / 2,
            );
            eprintln!("[build] layered nav done in {:.1}s (max_level={})",
                      t0.elapsed().as_secs_f64(), nav.max_level());
            Some(nav)
        } else {
            None
        };

        VamanaGraph { storage, entry_point, dim, n, metadata: Some(metadata), layered_nav }
    }

    /// 从已有存储构造（用于 RP-Tuning 生成变体）
    pub fn from_storage(storage: HybridBlockedCsr, entry_point: u32, dim: usize) -> Self {
        let n = storage.len();
        Self { storage, entry_point, dim, n, metadata: None, layered_nav: None }
    }

    /// 量化感知建图（Week 7：β/α 协同调参）
    ///
    /// 与标准 build 的区别：用 QuantAwareRobustPrune 替代 RobustPrune
    /// Score = dist / (μ_dist + ε) + β × error / (μ_error + ε)
    /// β > 0 时，剪枝会回避量化误差大的边
    ///
    /// error_fn: 量化误差函数 error(u, v) = mean(avq_error(u), avq_error(v))
    pub fn build_with_quant_aware_prune<F>(
        vectors: &[f32],
        dim: usize,
        config: &VamanaBuildConfig,
        qa_config: &QuantAwarePruneConfig,
        error_fn: F,
        rng: &mut ChaCha8Rng,
    ) -> Self
    where
        F: Fn(u32, u32) -> f32 + Sync,
    {
        let n = vectors.len() / dim;
        assert_eq!(vectors.len(), n * dim);
        let mut storage = HybridBlockedCsr::new(n, config.r_max);

        eprintln!("[build_qa] rayon threads: {}", rayon::current_num_threads());

        let _random_entry = Self::init_random_graph(&mut storage, n, config, rng);
        // Vamana/DiskANN 论文：entry_point 用 medoid
        let entry_point = Self::compute_medoid(vectors, dim, n);
        eprintln!("[build_qa] init_random_graph done, entry=medoid[{}]", entry_point);

        // Vamana 论文 two passes：第一轮 α=1.0（连通性），第二轮 α=config.alpha（长程边）
        // v6.6: 小批量顺序处理（同 build 方法）
        const BUILD_BATCH_SIZE: usize = 10_000;
        for iter in 0..config.max_iterations {
            let order = Self::permutation(n, rng);
            let alpha = if iter == 0 { 1.0 } else { config.alpha };
            let progress = AtomicUsize::new(0);
            let progress_interval = (n / 20).max(1);
            eprintln!("[build_qa] iter {}/{} alpha={} beta={}", iter + 1, config.max_iterations, alpha, qa_config.beta);

            for chunk in order.chunks(BUILD_BATCH_SIZE) {
                let new_neighbors: Vec<(u32, Vec<u32>)> = chunk
                    .par_iter()
                    .map_init(
                        || VisitedTracker::new(n, config.l_build),
                        |visited, &node_id| {
                            let idx = progress.fetch_add(1, Ordering::Relaxed);
                            if idx > 0 && idx % progress_interval == 0 {
                                eprintln!("[build_qa] {}/{} ({}%)", idx, n, idx * 100 / n);
                            }
                            let query = &vectors[node_id as usize * dim..(node_id as usize + 1) * dim];
                            let _candidates = Self::greedy_search_vec_build(
                                vectors, dim, &storage, entry_point, query, config.l_build,
                                visited,
                            );
                            let iter_qa_config = QuantAwarePruneConfig {
                                alpha,
                                ..*qa_config
                            };
                            let pruned = QuantAwareRobustPrune::prune(
                                visited.visited_nodes(),
                                node_id,
                                vectors,
                                dim,
                                &error_fn,
                                &iter_qa_config,
                            );
                            (node_id, pruned)
                        }
                    )
                    .collect();

                for (node_id, pruned) in new_neighbors {
                    Self::connect_bidirectional(
                        &mut storage, node_id, &pruned, vectors, dim, alpha, config,
                    );
                }
            }
        }

        Self::final_prune(&mut storage, vectors, dim, config.alpha, config.r_max, config.saturate, config.prune_strategy);

        let build_config = BuildConfig::default();
        let metadata = BuildMetadata::from_config(&build_config, n, dim);
        VamanaGraph { storage, entry_point, dim, n, metadata: Some(metadata), layered_nav: None }
    }

    /// 初始化随机图
    ///
    /// 设计文档上层导航：保留随机层级（可与 HNSW 直接对比）
    fn init_random_graph(
        storage: &mut HybridBlockedCsr,
        n: usize,
        config: &VamanaBuildConfig,
        rng: &mut ChaCha8Rng,
    ) -> u32 {
        if n == 0 {
            return 0;
        }
        let entry = rng.gen_range(0..n as u32);

        // 随机连接每个节点到若干邻居，加双向边（Vamana/DiskANN 要求无向初始图）
        // BUG FIX (v6.4): 原实现只加前向边不加反向边，导致初始图是有向的。
        // 第一轮 greedy_search 只能沿前向边导航，访问集质量极差，
        // RobustPrune 在噪声上剪枝 → 图导航效率崩塌（avg_visited=1400，glass 的 10x）。
        let neighbor_count = config.r_max.min(n.saturating_sub(1));
        for node in 0..n as u32 {
            let mut seen = std::collections::HashSet::with_capacity(neighbor_count);
            seen.insert(node);
            let mut neighbors = Vec::with_capacity(neighbor_count);
            while neighbors.len() < neighbor_count {
                let j = rng.gen_range(0..n as u32);
                if seen.insert(j) {
                    neighbors.push(j);
                }
            }
            for &j in &neighbors {
                storage.add_edge(node, j);
                storage.add_edge(j, node); // 反向边：无向图
            }
        }
        entry
    }

    /// 计算 medoid（离质心最近的点）
    ///
    /// Vamana/DiskANN 论文要求 entry_point 用 medoid，不是随机点。
    /// 质心法：先算所有向量的均值（质心），再找离质心最近的点。
    ///
    /// OPT-7：当 n > 10K 时用采样近似（1K 样本），O(1K*dim) vs O(n*dim)
    /// 采样用独立 rng（seed 42）保证确定性，不影响外部 rng 状态
    /// medoid 仅是搜索起点，采样近似对 recall 无影响（greedy_search 自收敛）
    fn compute_medoid(vectors: &[f32], dim: usize, n: usize) -> u32 {
        if n == 0 {
            return 0;
        }
        const SAMPLE_THRESHOLD: usize = 10_000;
        const SAMPLE_COUNT: usize = 1_000;
        if n <= SAMPLE_THRESHOLD {
            return Self::compute_medoid_full(vectors, dim, n);
        }
        // 采样近似：用独立 rng 保证确定性，不干扰外部 rng 状态
        let mut rng = ChaCha8Rng::seed_from(42);
        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.partial_shuffle(&mut rng, SAMPLE_COUNT);
        let sample: Vec<u32> = indices.iter().take(SAMPLE_COUNT).copied().collect();

        // 1. 用采样计算近似质心
        let mut centroid = vec![0.0f32; dim];
        for &idx in &sample {
            let v = &vectors[idx as usize * dim..(idx as usize + 1) * dim];
            for d in 0..dim {
                centroid[d] += v[d];
            }
        }
        for d in 0..dim {
            centroid[d] /= SAMPLE_COUNT as f32;
        }
        // 2. 在采样集中找离近似质心最近的点
        let mut best_id = sample[0];
        let mut best_dist = f32::MAX;
        for &idx in &sample {
            let dist = l2_simd(&centroid, &vectors[idx as usize * dim..(idx as usize + 1) * dim]);
            if dist < best_dist {
                best_dist = dist;
                best_id = idx;
            }
        }
        best_id
    }

    /// 全量计算 medoid（n <= 10K 时使用）
    fn compute_medoid_full(vectors: &[f32], dim: usize, n: usize) -> u32 {
        // 1. 计算质心（所有向量的均值）
        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            for d in 0..dim {
                centroid[d] += v[d];
            }
        }
        for d in 0..dim {
            centroid[d] /= n as f32;
        }
        // 2. 找离质心最近的点
        let mut best_id = 0u32;
        let mut best_dist = f32::MAX;
        for i in 0..n {
            let dist = l2_simd(&centroid, &vectors[i * dim..(i + 1) * dim]);
            if dist < best_dist {
                best_dist = dist;
                best_id = i as u32;
            }
        }
        best_id
    }

    /// 生成 0..n 的随机排列
    fn permutation(n: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.shuffle(rng);
        order
    }

    /// 贪心搜索（构建期，返回候选集 + 全部 visited）
    ///
    /// 设计文档第三层：从 entry_point 出发，贪心寻找距离 query 最近的节点
    /// 返回 (top-L 结果, 所有 visited 节点)
    /// Vamana/DiskANN 论文：BuildVamana 用 visited set 做 RobustPrune
    pub fn greedy_search(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query_node: u32,
        l: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let query = &vectors[query_node as usize * dim..(query_node as usize + 1) * dim];
        Self::greedy_search_vec(vectors, dim, storage, entry_point, query, l)
    }

    /// 贪心搜索（查询向量版本）
    ///
    /// 返回 (top-L 结果, 所有 visited 节点)
    /// 查询时用 top-L，建图时用 visited（Vamana 论文要求）
    ///
    /// v7.1 核心修复：建图搜索从无界 BinaryHeap 改为 LinearPool。
    ///
    /// 根因分析（avg_visited=2445 的真正根因）：
    /// 查询侧 LinearPool 已正确限制搜索宽度，但 avg_visited 仍高 →
    /// 问题在图质量本身。图质量由建图搜索决定：
    ///
    /// 原实现用无界 BinaryHeap，候选堆可膨胀到数千元素：
    /// - 建图搜索沿中距离候选发散，visited 集充斥远距离噪声节点
    /// - RobustPrune 在噪声候选集上剪枝 → 边选择质量差
    /// - 图邻居无局部重叠 → 查询时每次扩展引入大量新访问
    /// - ef=50, degree=64 → 50×49=2450 ≈ avg_visited=2445
    ///
    /// LinearPool 修复（对齐 DiskANN：build 和 query 用同一个固定容量搜索器）：
    /// - 候选池固定容量 = l_build，满时拒绝远候选
    /// - 搜索聚焦局部邻域，visited 集干净无噪声
    /// - RobustPrune 获得高质量候选集 → 更好的边 → 更好的图
    /// - VisitedTracker 仍记录所有距离计算过的节点（含被 pool 拒绝的），
    ///   保证 Vamana 论文要求的完整 visited set
    pub fn greedy_search_vec(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query: &[f32],
        l: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let n = vectors.len() / dim;
        let mut visited = VisitedTracker::new(n, l);
        let mut pool = LinearPool::new(l);

        // 插入入口节点
        let entry_dist = l2_simd(
            query,
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        // 主循环：弹出最近未扩展候选，扩展其邻居
        while let Some((node, _dist)) = pool.pop() {
            for &neighbor in storage.neighbors(node) {
                if visited.visit(neighbor) {
                    let d = l2_simd(
                        query,
                        &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                    );
                    // LinearPool 满时自动拒绝远候选，
                    // 但 visited 已记录该节点（距离已计算）
                    pool.insert(neighbor, d);
                }
            }
        }

        // top_results: pool 中所有元素（按距离升序）
        let top_results: Vec<u32> = pool
            .to_sorted_vec()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        // all_visited: VisitedTracker 记录的所有距离计算过的节点
        let all_visited: Vec<u32> = visited.visited_nodes().to_vec();
        (top_results, all_visited)
    }

    /// 建图专用贪心搜索（简单单循环，无 prefetch 开销）
    ///
    /// 建图路径调用 2M 次，每次 ~1400 迭代。Two-Pass Prefetch 的 3 循环 +
    /// _mm_prefetch 指令开销在建图场景下是纯浪费（建图访问模式与查询不同）。
    /// 此函数保留 map_init 的 memset 优化（visited.reset() 只清 history），
    /// 但用简单单循环代替 Two-Pass Prefetch。
    pub fn greedy_search_vec_build(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query: &[f32],
        l: usize,
        visited: &mut VisitedTracker,
    ) -> Vec<(u32, f32)> {
        visited.reset();

        let mut pool = LinearPool::new(l);

        let entry_dist = l2_simd(
            query,
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        while let Some((node, _dist)) = pool.pop() {
            if let Some((next_node, _)) = pool.peek_unchecked() {
                storage.prefetch_neighbors(next_node);
            }

            for &neighbor in storage.neighbors(node) {
                if visited.visit(neighbor) {
                    let d = l2_simd(
                        query,
                        &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                    );
                    pool.insert(neighbor, d);
                }
            }
        }

        pool.to_sorted_vec()
    }

    /// 查询专用贪心搜索（复用 VisitedTracker + LinearPool + Two-Pass Prefetch）
    ///
    /// v8.0 核心优化：Two-Pass Prefetch + Multi-line Graph Prefetch
    ///
    /// 消融实验验证（ef=50, recall=0.9931）：
    ///   baseline (v7):           QPS=911
    ///   + two_pass(po=8):        QPS=1258 (+38%)
    ///   + multi_pref:            QPS=1081 (+19%)
    ///   + combined(po=8):        QPS=1493 (+64%)  ← 采用此方案
    ///
    /// Two-Pass Prefetch（学习 Glass SearchImpl2）：
    ///   第一遍扫描邻居列表，收集未访问节点到 edge_buf
    ///   预取 edge_buf 前 po 个节点的向量数据到 L1 cache
    ///   第二遍计算距离，同时前瞻预取 i+po 处的向量
    ///   效果：距离计算时向量已在 cache，隐藏 ~100ns DRAM 延迟
    ///
    /// Multi-line Graph Prefetch：
    ///   R_max=64 → 邻居列表 256 bytes = 4 cache lines
    ///   原实现只预取 1 行，现在预取 4 行覆盖完整邻居列表
    pub fn greedy_search_vec_reuse(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query: &[f32],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
    ) -> Vec<(u32, f32)> {
        visited.reset();
        pool.reset_for(l);

        // 插入入口节点
        let entry_dist = l2_simd(
            query,
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        // Prefetch offset：前瞻距离（po=0 禁用向量预取）
        // 消融实验：po=8 在 ef=50 最佳 (+38%), po=4 在 ef=100 最佳 (+39%)
        // 默认 po=8（ef=50 是最常用的工作点），可通过 GraphSearcher::with_prefetch_offset 调整
        // 栈上 edge_buf，避免堆分配（R_max=64 → 64*4=256B fit 栈）
        let mut edge_buf: [u32; 128] = [0; 128];

        // 主循环：弹出最近未扩展候选，扩展其邻居
        while let Some((node, _dist)) = pool.pop() {
            // Multi-line graph prefetch：预取下一轮 pop 节点的完整邻居列表
            // R_max=64 → 256 bytes → 4 cache lines
            if let Some((next_node, _)) = pool.peek_unchecked() {
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<0>(ptr);
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居到 edge_buf
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                // SAFETY: v 来自图边，保证 < n
                if unsafe { visited.visit_unchecked(v) } {
                    edge_buf[edge_size] = v;
                    edge_size += 1;
                }
            }

            // 预取前 po 个邻居的向量数据
            let prefetch_count = po.min(edge_size);
            for i in 0..prefetch_count {
                let v = edge_buf[i] as usize;
                let ptr = &vectors[v * dim] as *const f32 as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }

            // 第二遍：计算距离，同时前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = &vectors[v * dim] as *const f32 as *const i8;
                    unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
                }
                let neighbor = edge_buf[i];
                let d = l2_simd(
                    query,
                    &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                );
                pool.insert(neighbor, d);
            }
        }

        pool.to_sorted_vec()
    }

    /// SQ8 量化贪心搜索（Phase 1 Step 0）
    ///
    /// 与 `greedy_search_vec_reuse` 相同的 Two-Pass Prefetch 架构，
    /// 但距离计算用 SQ8 u8 码代替 f32 全精度向量。
    ///
    /// 优势：
    /// - 内存带宽降 4x（128B/vector → 32B/vector for SIFT-128）
    /// - AVX2 u8 运算每次处理 16 维（vs f32 的 8 维）
    ///
    /// 搜索结束后对全部候选用 f32 重排序（rerank），恢复精确距离排序。
    pub fn greedy_search_sq8(
        sq8: &SQ8Dataset,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
    ) -> Vec<(u32, f32)> {
        visited.reset();
        pool.reset_for(l);

        // 入口节点距离（SQ8）
        let entry_dist = sq8.distance(query_code, entry_point as usize);
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        let mut edge_buf: [u32; 128] = [0; 128];

        while let Some((node, _dist)) = pool.pop() {
            // Multi-line graph prefetch（与 f32 路径相同，邻居列表大小不变）
            if let Some((next_node, _)) = pool.peek_unchecked() {
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<0>(ptr);
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                // SAFETY: v 来自图边，保证 < n
                if unsafe { visited.visit_unchecked(v) } {
                    edge_buf[edge_size] = v;
                    edge_size += 1;
                }
            }

            // 预取前 po 个邻居的 SQ8 码
            // SIFT-128: 128B = 2 cache lines，始终预取两行（分支消除）
            let prefetch_count = po.min(edge_size);
            for i in 0..prefetch_count {
                let v = edge_buf[i] as usize;
                let ptr = sq8.code(v).as_ptr() as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<0>(ptr);
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                }
            }

            // 第二遍：SQ8 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = sq8.code(v).as_ptr() as *const i8;
                    unsafe {
                        std::arch::x86_64::_mm_prefetch::<0>(ptr);
                        std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                    }
                }
                let neighbor = edge_buf[i];
                // 热路径：跳过 bounds check（neighbor 已通过 visited.visit 验证有效）
                let d = unsafe { sq8.distance_unchecked(query_code, neighbor as usize) };
                pool.insert(neighbor, d);
            }
        }

        pool.to_sorted_vec()
    }

    /// PQ4 图遍历（Phase 1 Step 1）
    ///
    /// 使用 4-bit PQ LUT-ADC 距离进行图遍历，码大小仅 M/2 bytes/vector（SIFT: 16B）。
    /// LUT (2KB) 完全在 L1 cache，距离计算为纯算术 + L1 查表。
    ///
    /// 返回 (节点ID, PQ4-ADC距离) 列表，需外部 f32 rerank。
    pub fn greedy_search_pq4(
        pq4: &PQ4Dataset,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        lut: &[f32],
        ef_search: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
    ) -> Vec<(u32, f32)> {
        if ef_search == 0 {
            return Vec::new();
        }

        visited.reset();
        pool.reset_for(ef_search);
        let m = pq4.codebook.m;

        // entry_point 距离
        let ep_dist = PQ4Dataset::adc_distance(lut, pq4.code(entry_point as usize), m);
        pool.insert(entry_point, ep_dist);
        visited.visit(entry_point);

        let mut edge_buf = [0u32; 128];

        while let Some((node, _dist)) = pool.pop() {
            // Multi-line graph prefetch
            if let Some((next_node, _)) = pool.peek_unchecked() {
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<0>(ptr);
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                if visited.visit(v) {
                    edge_buf[edge_size] = v;
                    edge_size += 1;
                }
            }

            // 预取前 po 个邻居的 PQ4 码（M/2 bytes，比 f32 小 16x）
            let prefetch_count = po.min(edge_size);
            for i in 0..prefetch_count {
                let v = edge_buf[i] as usize;
                let ptr = pq4.code(v).as_ptr() as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }

            // 第二遍：PQ4 ADC 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = pq4.code(v).as_ptr() as *const i8;
                    unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
                }
                let neighbor = edge_buf[i];
                let d = PQ4Dataset::adc_distance(lut, pq4.code(neighbor as usize), m);
                pool.insert(neighbor, d);
            }
        }

        pool.to_sorted_vec()
    }

    /// PQ8 图遍历（Phase 1 Step 1, K=256 标量 LUT-ADC）
    ///
    /// 使用 8-bit PQ LUT-ADC 距离进行图遍历，码大小 M bytes/vector（SIFT: 32B）。
    /// LUT (32KB) 在 L2 cache 内，距离计算为 M 次 table lookup。
    /// K=256 精度接近 SQ8，但带宽降低 4x（32B vs 128B per vector）。
    ///
    /// 返回 (节点ID, PQ8-ADC距离) 列表，需外部 f32 rerank。
    pub fn greedy_search_pq8(
        pq8: &PQ8Dataset,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        lut: &[f32],
        ef_search: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
    ) -> Vec<(u32, f32)> {
        if ef_search == 0 {
            return Vec::new();
        }

        visited.reset();
        pool.reset_for(ef_search);
        let m = pq8.codebook.m;
        let k = pq8.codebook.k;

        // entry_point 距离
        let ep_dist = PQ8Dataset::adc_distance(lut, pq8.code(entry_point as usize), m, k);
        pool.insert(entry_point, ep_dist);
        visited.visit(entry_point);

        let mut edge_buf = [0u32; 128];

        while let Some((node, _dist)) = pool.pop() {
            // Multi-line graph prefetch
            if let Some((next_node, _)) = pool.peek_unchecked() {
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<0>(ptr);
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                    std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                if visited.visit(v) {
                    edge_buf[edge_size] = v;
                    edge_size += 1;
                }
            }

            // 预取前 po 个邻居的 PQ8 码（M bytes，比 f32 小 16x）
            let prefetch_count = po.min(edge_size);
            for i in 0..prefetch_count {
                let v = edge_buf[i] as usize;
                let ptr = pq8.code(v).as_ptr() as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }

            // 第二遍：PQ8 ADC 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = pq8.code(v).as_ptr() as *const i8;
                    unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
                }
                let neighbor = edge_buf[i];
                let d = PQ8Dataset::adc_distance(lut, pq8.code(neighbor as usize), m, k);
                pool.insert(neighbor, d);
            }
        }

        pool.to_sorted_vec()
    }

    /// 贪心搜索（探索模式，建图期第一轮用）
    ///
    /// 与 greedy_search_vec 的区别：候选 > worst 时不 break，而是跳过插入继续扩展邻居
    /// 这样能 escape 局部最优，建连通图。限制 max_visited 防止遍历整个图。
    pub fn greedy_search_vec_explore(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query: &[f32],
        l: usize,
        max_visited: usize,
    ) -> Vec<u32> {
        let n = vectors.len() / dim;
        let mut visited = VisitedTracker::new(n, l);

        use std::cmp::Reverse;
        let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::new();

        let entry_dist = l2_simd(
            query,
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        candidates.push(Reverse((OrderedF32(entry_dist), entry_point)));
        visited.visit(entry_point);

        while let Some(Reverse((dist, node))) = candidates.pop() {
            // 跳过但扩展：候选 > worst 时不加入 results，但继续扩展邻居
            let should_add = if results.len() >= l {
                if let Some(&(worst, _)) = results.peek() {
                    dist.0 <= worst.0
                } else {
                    true
                }
            } else {
                true
            };

            if should_add {
                results.push((dist, node));
                if results.len() > l {
                    results.pop();
                }
            }

            for &neighbor in storage.neighbors(node) {
                if visited.visit(neighbor) {
                    let d = l2_simd(
                        query,
                        &vectors[neighbor as usize * dim..(neighbor as usize + 1) * dim],
                    );
                    candidates.push(Reverse((OrderedF32(d), neighbor)));
                }
            }

            if visited.visited_count() > max_visited {
                break;
            }
        }

        results.into_iter().map(|(_, id)| id).collect()
    }

    /// 贪心搜索（探索模式，query_node 版本）
    pub fn greedy_search_explore(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query_node: u32,
        l: usize,
        max_visited: usize,
    ) -> Vec<u32> {
        let query = &vectors[query_node as usize * dim..(query_node as usize + 1) * dim];
        Self::greedy_search_vec_explore(vectors, dim, storage, entry_point, query, l, max_visited)
    }

    /// 双向连接 + 度数控制
    ///
    /// Vamana 论文 Algorithm 1:
    ///   SetOutNeighbors(p, N)       ← 替换，不是追加
    ///   for n in N:
    ///     AddOutNeighbor(n, p)      ← 加反向边
    ///     if |OutN(n)| > R:
    ///       RobustPrune(n, α_t, R)  ← 用迭代特定 α_t
    ///
    /// BUG FIX (v6.4): 原实现用 add_edge 追加边，不清除旧边。
    /// 导致随机初始化边 + 各轮迭代边全部累积，final_prune 在噪声候选集上剪枝，
    /// 图导航结构被破坏，avg_visited 飙升到 1,443（glass 的 ~10 倍）。
    fn connect_bidirectional(
        storage: &mut HybridBlockedCsr,
        node: u32,
        neighbors: &[u32],
        vectors: &[f32],
        dim: usize,
        alpha: f32,
        config: &VamanaBuildConfig,
    ) {
        // 替换：清除旧边，设置剪枝后的新邻居（Vamana: SetOutNeighbors）
        storage.set_neighbors(node, neighbors);

        // 加反向边 + 延迟剪枝（Vamana: AddOutNeighbor + RobustPrune）
        for &nb in neighbors {
            storage.add_edge(nb, node);
            // 反向边：如果 nb 的度数超过 R_soft，用迭代特定 α 做 RobustPrune
            if storage.degree(nb) > config.r_soft {
                let (main, overflow) = storage.neighbors_full(nb);
                let mut all: Vec<u32> = main.to_vec();
                all.extend_from_slice(overflow);
                let pruned = prune_dispatch(
                    config.prune_strategy,
                    &all, nb, vectors, dim, alpha, config.r_max,
                    config.saturate && alpha > 1.0,
                );
                storage.set_neighbors(nb, &pruned);
            }
        }
    }

    /// 全局 final prune 到 R_max
    ///
    /// Vamana 论文：用 RobustPrune（不是 truncate），保留质量最好的边
    fn final_prune(
        storage: &mut HybridBlockedCsr,
        vectors: &[f32],
        dim: usize,
        alpha: f32,
        r_max: usize,
        saturate: bool,
        strategy: PruneStrategy,
    ) {
        for node in 0..storage.len() as u32 {
            let (main, overflow) = storage.neighbors_full(node);
            if main.len() + overflow.len() <= r_max {
                continue;
            }
            let mut all: Vec<u32> = main.to_vec();
            all.extend_from_slice(overflow);
            let pruned = prune_dispatch(strategy, &all, node, vectors, dim, alpha, r_max, saturate);
            storage.set_neighbors(node, &pruned);
        }
    }

    /// 节点数
    pub fn len(&self) -> usize {
        self.n
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// 向量维度
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 入口节点
    pub fn entry_point(&self) -> u32 {
        self.entry_point
    }

    /// 获取节点邻居
    pub fn neighbors(&self, node_id: u32) -> &[u32] {
        self.storage.neighbors(node_id)
    }

    /// 获取内部存储引用
    pub fn storage(&self) -> &HybridBlockedCsr {
        &self.storage
    }

    /// 获取内部存储可变引用
    pub fn storage_mut(&mut self) -> &mut HybridBlockedCsr {
        &mut self.storage
    }

    /// 获取构建元数据（设计文档 F.7）
    pub fn metadata(&self) -> Option<&BuildMetadata> {
        self.metadata.as_ref()
    }

    /// 获取分层导航（设计文档：保留随机层级）
    pub fn layered_nav(&self) -> Option<&LayeredNavigation> {
        self.layered_nav.as_ref()
    }

    /// 度数统计
    pub fn degree_stats(&self) -> crate::memory::graph::DegreeStats {
        self.storage.log_degree_distribution()
    }
}

/// VamanaGraph 序列化格式（含 16 字节文件头）
///
/// 设计文档 F.1：
///   [0..16)  IndexHeader: magic + version + flags + crc32
///   [16..24) n: u64
///   [24..32) dim: u64
///   [32..36) entry_point: u32
///   flags=0（旧格式）:
///     [36..)    HybridBlockedCsr body
///   flags=FLAG_HAS_METADATA（新格式）:
///     [36..40)  metadata_len: u32
///     [40..40+metadata_len) metadata TOML
///     [40+metadata_len..) HybridBlockedCsr body
///   flags=FLAG_HAS_METADATA|FLAG_HAS_LAYERED_NAV:
///     [36..40)  metadata_len: u32
///     [40..40+metadata_len) metadata TOML + padding
///     [after metadata] layered_nav_len: u32 + layered_nav bytes + padding
///     [after layered_nav] HybridBlockedCsr body
impl crate::memory::serialize::Serializable for VamanaGraph {
    fn serialize(&self) -> Vec<u8> {
        use crate::memory::serialize::{IndexHeader, crc32, FLAG_HAS_METADATA, FLAG_HAS_LAYERED_NAV};

        // 序列化 metadata TOML（如果有）
        let metadata_bytes: Vec<u8> = match &self.metadata {
            Some(m) => m.to_toml().unwrap_or_default().into_bytes(),
            None => Vec::new(),
        };

        // 序列化 layered nav（如果有）
        let nav_bytes: Vec<u8> = match &self.layered_nav {
            Some(nav) => nav.serialize_binary(),
            None => Vec::new(),
        };

        // 序列化文件体
        let mut body: Vec<u8> = Vec::with_capacity(
            20 + self.storage.main_block_bytes() + 4 + metadata_bytes.len() + 4 + nav_bytes.len(),
        );
        body.extend_from_slice(&(self.n as u64).to_le_bytes());
        body.extend_from_slice(&(self.dim as u64).to_le_bytes());
        body.extend_from_slice(&self.entry_point.to_le_bytes());

        // metadata trailer（设计文档 F.7）
        let has_metadata = !metadata_bytes.is_empty();
        if has_metadata {
            body.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
            body.extend_from_slice(&metadata_bytes);
            let padding = (4 - (metadata_bytes.len() % 4)) % 4;
            body.extend(std::iter::repeat(0u8).take(padding));
        }

        // layered nav trailer（设计文档：保留随机层级）
        let has_nav = !nav_bytes.is_empty();
        if has_nav {
            body.extend_from_slice(&(nav_bytes.len() as u32).to_le_bytes());
            body.extend_from_slice(&nav_bytes);
            let padding = (4 - (nav_bytes.len() % 4)) % 4;
            body.extend(std::iter::repeat(0u8).take(padding));
        }

        // HybridBlockedCsr body
        let storage_bytes = crate::memory::serialize::Serializable::serialize(&self.storage);
        body.extend_from_slice(&storage_bytes);

        // 计算校验和并构造文件头
        let crc = crc32(&body);
        let mut flags = 0u32;
        if has_metadata { flags |= FLAG_HAS_METADATA; }
        if has_nav { flags |= FLAG_HAS_LAYERED_NAV; }
        let header = IndexHeader {
            magic: crate::memory::serialize::INDEX_MAGIC,
            version: crate::memory::serialize::INDEX_VERSION,
            flags,
            crc32: crc,
        };
        let header_bytes = header.to_bytes();

        // 拼接：header + body
        let mut result = Vec::with_capacity(header_bytes.len() + body.len());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&body);
        result
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, crate::memory::serialize::SerializeError> {
        use crate::memory::serialize::{IndexHeader, HEADER_SIZE, FLAG_HAS_METADATA, FLAG_HAS_LAYERED_NAV};
        use std::convert::TryInto;

        if bytes.len() < HEADER_SIZE + 20 {
            return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too short for VamanaGraph header",
            )));
        }

        // 校验文件头
        let header = IndexHeader::from_bytes(&bytes[..HEADER_SIZE])?;
        header.validate()?;

        let body = &bytes[HEADER_SIZE..];

        // 校验 CRC32
        let expected_crc = crate::memory::serialize::crc32(body);
        if expected_crc != header.crc32 {
            return Err(crate::memory::serialize::SerializeError::CrcMismatch {
                expected: header.crc32,
                actual: expected_crc,
            });
        }

        // 解析文件体
        let n = u64::from_le_bytes(body[0..8].try_into().unwrap()) as usize;
        let dim = u64::from_le_bytes(body[8..16].try_into().unwrap()) as usize;
        let entry_point = u32::from_le_bytes(body[16..20].try_into().unwrap());

        let mut offset = 20usize;

        // 读取 metadata trailer（如果 flags 标记存在）
        let metadata = if header.flags & FLAG_HAS_METADATA != 0 {
            if body.len() < offset + 4 {
                return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "body too short for metadata trailer header",
                )));
            }
            let meta_len = u32::from_le_bytes(body[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            if body.len() < offset + meta_len {
                return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "body too short for metadata trailer",
                )));
            }
            let meta_bytes = &body[offset..offset + meta_len];
            let metadata = std::str::from_utf8(meta_bytes)
                .ok()
                .and_then(|s| BuildMetadata::from_toml(s).ok());
            // 跳过 4 字节对齐填充
            let padding = (4 - (meta_len % 4)) % 4;
            offset += meta_len + padding;
            metadata
        } else {
            None
        };

        // 读取 layered nav trailer（如果 flags 标记存在）
        let layered_nav = if header.flags & FLAG_HAS_LAYERED_NAV != 0 {
            if body.len() < offset + 4 {
                return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "body too short for layered nav trailer header",
                )));
            }
            let nav_len = u32::from_le_bytes(body[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            if body.len() < offset + nav_len {
                return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "body too short for layered nav trailer",
                )));
            }
            let nav_bytes = &body[offset..offset + nav_len];
            let nav = LayeredNavigation::deserialize_binary(nav_bytes);
            let padding = (4 - (nav_len % 4)) % 4;
            offset += nav_len + padding;
            nav
        } else {
            None
        };

        // 反序列化 HybridBlockedCsr
        let storage = HybridBlockedCsr::deserialize(&body[offset..])?;

        // 校验一致性
        if storage.len() != n {
            return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("n mismatch: header={}, storage={}", n, storage.len()),
            )));
        }

        Ok(Self { storage, entry_point, dim, n, metadata, layered_nav })
    }
}

/// 图搜索器（查询热路径）
pub struct GraphSearcher<'a> {
    vectors: &'a [f32],
    dim: usize,
    graph: &'a VamanaGraph,
    ef_search: usize,
    /// 预分配的 VisitedTracker，避免每次搜索分配 1MB visited 数组
    /// 设计文档 F.2：热路径零分配
    visited: VisitedTracker,
    /// 可选的 NavigationLayer（centroid overlay）
    /// 设计文档第三层：可选层，√N 个 centroid overlay 锚点节点
    /// 启用后搜索从最近 centroid 开始，而非默认 medoid
    navigation: Option<&'a NavigationLayer>,
    /// 上次搜索访问的唯一节点数（avg_visited 诊断接口）
    /// 此值在 search() 结束后被设置，不受 SIMD/内存布局干扰，纯粹衡量图导航效率
    last_visited_count: usize,
    /// 上次搜索实际使用的 ef 值（自适应 ef 诊断接口）
    /// 固定 ef 模式下等于 ef_search；自适应模式下等于 estimate_ef 返回值
    last_ef_used: usize,
    /// Prefetch offset：Two-Pass Prefetch 的前瞻距离
    /// po=8 适合 ef=50，po=4 适合 ef=100；po=0 禁用向量预取
    prefetch_offset: usize,
    /// 可选的 SQ8 量化数据集（Phase 1 Step 0）
    /// 启用后 search_sq8() 使用 SQ8 距离进行图遍历 + f32 rerank
    sq8: Option<&'a SQ8Dataset>,
    /// 可选的 4-bit PQ 量化数据集（Phase 1 Step 1）
    /// 启用后 search_pq4() 使用 LUT-ADC 距离进行图遍历 + f32 rerank
    pq4: Option<&'a PQ4Dataset>,
    /// 可选的 8-bit PQ 量化数据集（Phase 1 Step 1, K=256）
    /// 启用后 search_pq8() 使用 LUT-ADC 距离进行图遍历 + f32 rerank
    pq8: Option<&'a PQ8Dataset>,
    /// 可选的自适应 ef 配置（Phase 4.5）
    /// 启用后 search()/search_sq8()/batch_search() 根据查询难度动态分配 ef
    adaptive_ef: Option<AdaptiveEfConfig>,
    /// 预分配的 LinearPool，避免每次搜索堆分配
    pool: LinearPool,
}

impl<'a> GraphSearcher<'a> {
    /// 创建搜索器（默认 medoid entry_point）
    ///
    /// 预分配 VisitedTracker（O(N) 一次性分配），后续搜索复用
    pub fn new(vectors: &'a [f32], graph: &'a VamanaGraph, ef_search: usize) -> Self {
        let dim = graph.dim();
        let n = vectors.len() / dim;
        Self {
            vectors,
            dim,
            graph,
            ef_search,
            visited: VisitedTracker::new(n, ef_search),
            navigation: None,
            last_visited_count: 0,
            last_ef_used: ef_search,
            prefetch_offset: 8,
            sq8: None,
            pq4: None,
            pq8: None,
            adaptive_ef: None,
            pool: LinearPool::new(ef_search),
        }
    }

    /// 创建搜索器（启用 NavigationLayer centroid overlay）
    ///
    /// 设计文档第三层：可选层，√N 个 centroid overlay 锚点节点
    /// 启用后搜索从最近 centroid 开始
    pub fn new_with_navigation(
        vectors: &'a [f32],
        graph: &'a VamanaGraph,
        ef_search: usize,
        navigation: &'a NavigationLayer,
    ) -> Self {
        let dim = graph.dim();
        let n = vectors.len() / dim;
        Self {
            vectors,
            dim,
            graph,
            ef_search,
            visited: VisitedTracker::new(n, ef_search),
            navigation: Some(navigation),
            last_visited_count: 0,
            last_ef_used: ef_search,
            prefetch_offset: 8,
            sq8: None,
            pq4: None,
            pq8: None,
            adaptive_ef: None,
            pool: LinearPool::new(ef_search),
        }
    }

    /// 设置 prefetch offset（Two-Pass Prefetch 前瞻距离）
    ///
    /// po=8 适合 ef=50，po=4 适合 ef=100；po=0 禁用向量预取
    /// 返回 &mut Self 以支持链式调用
    pub fn with_prefetch_offset(&mut self, po: usize) -> &mut Self {
        self.prefetch_offset = po;
        self
    }

    /// 启用 SQ8 量化搜索（Phase 1 Step 0）
    ///
    /// 设置后可调用 search_sq8() 使用 SQ8 量化距离进行图遍历。
    /// 需要预先构建 SQ8Dataset（SQ8Dataset::build）。
    pub fn with_sq8(&mut self, sq8: &'a SQ8Dataset) -> &mut Self {
        self.sq8 = Some(sq8);
        self
    }

    /// 启用 4-bit PQ 量化搜索（Phase 1 Step 1）
    ///
    /// 设置后可调用 search_pq4() 使用 4-bit PQ LUT-ADC 距离进行图遍历。
    /// 需要预先构建 PQ4Dataset（PQ4Dataset::build）。
    pub fn with_pq4(&mut self, pq4: &'a PQ4Dataset) -> &mut Self {
        self.pq4 = Some(pq4);
        self
    }

    /// 启用 8-bit PQ 量化搜索（Phase 1 Step 1, K=256）
    ///
    /// 设置后可调用 search_pq8() 使用 8-bit PQ LUT-ADC 距离进行图遍历。
    /// 需要预先构建 PQ8Dataset（PQ8Dataset::build）。
    pub fn with_pq8(&mut self, pq8: &'a PQ8Dataset) -> &mut Self {
        self.pq8 = Some(pq8);
        self
    }

    /// 启用自适应 ef（Phase 4.5）
    ///
    /// 启用后 search()/search_sq8()/batch_search() 根据查询到入口点的距离
    /// 动态分配 ef：简单查询用小 ef（快），难查询用大 ef（准）。
    ///
    /// 需预先构建 AdaptiveEfConfig（AdaptiveEfConfig::build_with_layered_nav 等）。
    /// gamma 参数控制幂律曲率：gamma=1.0 线性，gamma>1 把多数查询压到小 ef。
    pub fn with_adaptive_ef(&mut self, config: AdaptiveEfConfig) -> &mut Self {
        // 若 max_ef > ef_search，扩容 VisitedTracker 的 history 容量
        // 避免难查询时 Vec 增长触发 reallocation
        if config.max_ef() > self.ef_search {
            self.visited = VisitedTracker::new(self.visited.len(), config.max_ef());
        }
        self.adaptive_ef = Some(config);
        self
    }

    /// 搜索最近邻
    ///
    /// 返回 (节点ID, 距离) 列表，按距离升序
    ///
    /// 使用标准 break 模式（Vamana 论文标准 greedy search）
    /// 复用预分配的 VisitedTracker，零堆分配热路径
    /// 若图有分层导航，从顶层贪心走到 Layer 0 入口（设计文档：保留随机层级）
    /// 若启用 NavigationLayer centroid overlay，从最近 centroid 开始
    /// 否则用 medoid entry_point
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        // 选择 entry_point 优先级：分层导航 > centroid overlay > medoid
        // 同时捕获 nav.initialize() 返回的 f32 距离用于自适应 ef 预测（Phase 4.5）
        let (entry_point, nav_entry_dist) = if let Some(nav) = self.graph.layered_nav() {
            let (ep, dist) = nav.initialize(self.vectors, self.dim, query);
            (ep, Some(dist))
        } else if let Some(nav) = self.navigation {
            (Self::nearest_centroid(nav.centroids(), self.vectors, self.dim, query), None)
        } else {
            (self.graph.entry_point(), None)
        };

        // 自适应 ef：用 nav.initialize 返回的 f32 距离预测（零额外开销）
        let ef = if let Some(ref adaptive) = self.adaptive_ef {
            let entry_dist = nav_entry_dist.unwrap_or_else(|| {
                l2_simd(
                    query,
                    &self.vectors[entry_point as usize * self.dim
                        ..(entry_point as usize + 1) * self.dim],
                )
            });
            adaptive.estimate_ef(entry_dist).max(k)
        } else {
            self.ef_search
        };
        self.last_ef_used = ef;

        let candidates = VamanaGraph::greedy_search_vec_reuse(
            self.vectors,
            self.dim,
            self.graph.storage(),
            entry_point,
            query,
            ef,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
        );

        // 记录本次搜索访问的唯一节点数（avg_visited 诊断）
        self.last_visited_count = self.visited.visited_count();

        // 距离已在 greedy_search_vec_reuse 中计算，只需排序取 top-k
        let mut results = candidates;
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// SQ8 量化搜索（Phase 1 Step 0）
    ///
    /// 图遍历使用 SQ8 u8 量化距离（4x 内存带宽降低），
    /// 终态对全部候选用 f32 精确距离重排序（rerank）。
    ///
    /// 返回 (节点ID, f32精确距离) 列表，按距离升序。
    /// 需先调用 with_sq8() 设置 SQ8Dataset。
    pub fn search_sq8(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let sq8 = self.sq8.expect("search_sq8 requires with_sq8() first");
        let dim = self.dim;

        // 1. 编码查询向量为 SQ8
        let query_code = sq8.params.encode(query);

        // 2. 选择 entry_point + 捕获 nav.initialize 的 f32 距离
        let (entry_point, nav_entry_dist) = if let Some(nav) = self.graph.layered_nav() {
            let (ep, dist) = nav.initialize(self.vectors, dim, query);
            (ep, Some(dist))
        } else if let Some(nav) = self.navigation {
            (Self::nearest_centroid(nav.centroids(), self.vectors, dim, query), None)
        } else {
            (self.graph.entry_point(), None)
        };

        // 3. 自适应 ef：用 nav.initialize 返回的 f32 距离预测（零额外开销）
        //    f32 距离比 SQ8 距离更精确，且分层导航已计算，不增加开销
        let ef = if let Some(ref adaptive) = self.adaptive_ef {
            let entry_dist = nav_entry_dist.unwrap_or_else(|| {
                l2_simd(
                    query,
                    &self.vectors[entry_point as usize * dim
                        ..(entry_point as usize + 1) * dim],
                )
            });
            adaptive.estimate_ef(entry_dist).max(k)
        } else {
            self.ef_search
        };
        self.last_ef_used = ef;

        // 4. SQ8 图遍历
        let candidates = VamanaGraph::greedy_search_sq8(
            sq8,
            self.graph.storage(),
            entry_point,
            &query_code,
            ef,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
        );

        self.last_visited_count = self.visited.visited_count();

        // 4. f32 rerank：用精确距离重排序全部候选
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, _sq8_dist)| {
                let f32_dist = l2_simd(
                    query,
                    &self.vectors[id as usize * dim..(id as usize + 1) * dim],
                );
                (id, f32_dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// 4-bit PQ 量化搜索（Phase 1 Step 1）
    ///
    /// 图遍历使用 4-bit PQ LUT-ADC 距离（16B/vector，比 SQ8 再小 8x），
    /// 终态对全部候选用 f32 精确距离重排序（rerank）。
    ///
    /// 返回 (节点ID, f32精确距离) 列表，按距离升序。
    /// 需先调用 with_pq4() 设置 PQ4Dataset。
    pub fn search_pq4(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let pq4 = self.pq4.expect("search_pq4 requires with_pq4() first");
        let dim = self.dim;

        // 1. 预计算 LUT（M×K f32 = 2KB，完全 L1 cache）
        let lut = pq4.codebook.compute_lut(query);

        // 2. 选择 entry_point（f32 路径，开销极小）
        let entry_point = if let Some(nav) = self.graph.layered_nav() {
            nav.initialize(self.vectors, dim, query).0
        } else if let Some(nav) = self.navigation {
            Self::nearest_centroid(nav.centroids(), self.vectors, dim, query)
        } else {
            self.graph.entry_point()
        };

        // 3. PQ4 图遍历
        let candidates = VamanaGraph::greedy_search_pq4(
            pq4,
            self.graph.storage(),
            entry_point,
            &lut,
            self.ef_search,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
        );

        self.last_visited_count = self.visited.visited_count();

        // 4. f32 rerank：用精确距离重排序全部候选
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, _pq4_dist)| {
                let f32_dist = l2_simd(
                    query,
                    &self.vectors[id as usize * dim..(id as usize + 1) * dim],
                );
                (id, f32_dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// 8-bit PQ 量化搜索（Phase 1 Step 1, K=256）
    ///
    /// 图遍历使用 8-bit PQ LUT-ADC 距离（32B/vector，比 SQ8 小 4x），
    /// 终态对全部候选用 f32 精确距离重排序（rerank）。
    ///
    /// 返回 (节点ID, f32精确距离) 列表，按距离升序。
    /// 需先调用 with_pq8() 设置 PQ8Dataset。
    pub fn search_pq8(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let pq8 = self.pq8.expect("search_pq8 requires with_pq8() first");
        let dim = self.dim;

        // 1. 预计算 LUT（M×K f32 = 32KB，L2 cache 内）
        let lut = pq8.codebook.compute_lut(query);

        // 2. 选择 entry_point（f32 路径，开销极小）
        let entry_point = if let Some(nav) = self.graph.layered_nav() {
            nav.initialize(self.vectors, dim, query).0
        } else if let Some(nav) = self.navigation {
            Self::nearest_centroid(nav.centroids(), self.vectors, dim, query)
        } else {
            self.graph.entry_point()
        };

        // 3. PQ8 图遍历
        let candidates = VamanaGraph::greedy_search_pq8(
            pq8,
            self.graph.storage(),
            entry_point,
            &lut,
            self.ef_search,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
        );

        self.last_visited_count = self.visited.visited_count();

        // 4. f32 rerank：用精确距离重排序全部候选
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, _pq8_dist)| {
                let f32_dist = l2_simd(
                    query,
                    &self.vectors[id as usize * dim..(id as usize + 1) * dim],
                );
                (id, f32_dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// 上次搜索访问的唯一节点数（avg_visited 诊断接口）
    ///
    /// 返回最近一次 search() 调用期间访问的唯一节点数。
    /// 此值不受 SIMD/内存布局干扰，纯粹衡量图导航效率。
    /// 用于 §〇.2 Pivot Criterion 裁决 Phase 1 vs Phase 3.3 优先级。
    pub fn last_visited_count(&self) -> usize {
        self.last_visited_count
    }

    /// 上次搜索实际使用的 ef 值（自适应 ef 诊断接口）
    ///
    /// 固定 ef 模式下等于 ef_search；自适应模式下等于 estimate_ef 返回值。
    /// 用于 benchmark 追踪 avg_ef 而无需重复调用 nav.initialize()。
    pub fn last_ef_used(&self) -> usize {
        self.last_ef_used
    }

    /// 找最近的 centroid 作为 entry_point
    /// O(√N * dim)，对 SIFT1M 约 1000*128 = 128K 次浮点运算
    #[inline]
    fn nearest_centroid(centroids: &[u32], vectors: &[f32], dim: usize, query: &[f32]) -> u32 {
        let mut best = centroids[0];
        let mut best_dist = f32::MAX;
        for &c in centroids {
            let cv = &vectors[c as usize * dim..(c as usize + 1) * dim];
            let d = l2_simd(query, cv);
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }
        best
    }

    // ──────────────────────────────────────────────────────────────────
    // Phase 7: 多线程批量搜索（rayon 并行）
    // ──────────────────────────────────────────────────────────────────

    /// 批量搜索（多线程并行）
    ///
    /// 图数据（vectors、graph、sq8）以 `&self` 只读共享，零竞争。
    /// 每个 rayon worker 独立创建 VisitedTracker（thread_local 缓存复用）。
    ///
    /// 使用 SQ8 量化路径（若已配置），否则回退 f32。
    ///
    /// 返回 Vec<Vec<(u32, f32)>>，每个查询的 top-k 结果按距离升序。
    pub fn batch_search(
        &self,
        queries: &[&[f32]],
        k: usize,
    ) -> Vec<Vec<(u32, f32)>> {
        use rayon::prelude::*;

        let n = self.vectors.len() / self.dim;
        let default_ef = self.ef_search;
        let po = self.prefetch_offset;
        let vectors = self.vectors;
        let dim = self.dim;
        let storage = self.graph.storage();
        let sq8 = self.sq8;
        let graph = self.graph;
        let navigation = self.navigation;
        let adaptive_ef = self.adaptive_ef.as_ref();

        queries
            .par_iter()
            .map(|query| {
                // 线程本地 VisitedTracker：每 worker 分配一次后复用
                // 设计文档 F.2：热路径零分配
                thread_local! {
                    static TLS_SEARCH: std::cell::RefCell<
                        (Option<VisitedTracker>, Option<LinearPool>)
                    > = std::cell::RefCell::new((None, None));
                }

                TLS_SEARCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    if borrow.0.is_none() {
                        let cap = adaptive_ef
                            .map(|c| c.max_ef())
                            .unwrap_or(default_ef);
                        borrow.0 = Some(VisitedTracker::new(n, cap));
                        borrow.1 = Some(LinearPool::new(cap));
                    }
                    let (visited_opt, pool_opt) = &mut *borrow;
                    let visited = visited_opt.as_mut().unwrap();
                    let pool = pool_opt.as_mut().unwrap();

                    // 选择搜索路径：SQ8 > f32
                    if let Some(sq8) = sq8 {
                        let query_code = sq8.params.encode(query);

                        let (entry_point, nav_entry_dist) = if let Some(nav) = graph.layered_nav() {
                            let (ep, dist) = nav.initialize(vectors, dim, query);
                            (ep, Some(dist))
                        } else if let Some(nav) = navigation {
                            (Self::nearest_centroid(nav.centroids(), vectors, dim, query), None)
                        } else {
                            (graph.entry_point(), None)
                        };

                        // 自适应 ef（Phase 4.5）：用 nav.initialize 的 f32 距离
                        let ef = if let Some(ref adaptive) = adaptive_ef {
                            let entry_dist = nav_entry_dist.unwrap_or_else(|| {
                                l2_simd(query,
                                    &vectors[entry_point as usize * dim
                                        ..(entry_point as usize + 1) * dim])
                            });
                            adaptive.estimate_ef(entry_dist).max(k)
                        } else {
                            default_ef
                        };

                        let cands = VamanaGraph::greedy_search_sq8(
                            sq8,
                            storage,
                            entry_point,
                            &query_code,
                            ef,
                            visited,
                            pool,
                            po,
                        );

                        // f32 rerank
                        let mut results: Vec<(u32, f32)> = cands
                            .into_iter()
                            .map(|(id, _)| {
                                let d = l2_simd(
                                    query,
                                    &vectors[id as usize * dim..(id as usize + 1) * dim],
                                );
                                (id, d)
                            })
                            .collect();
                        results.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        results.truncate(k);
                        results
                    } else {
                        let (entry_point, nav_entry_dist) = if let Some(nav) = graph.layered_nav() {
                            let (ep, dist) = nav.initialize(vectors, dim, query);
                            (ep, Some(dist))
                        } else if let Some(nav) = navigation {
                            (Self::nearest_centroid(nav.centroids(), vectors, dim, query), None)
                        } else {
                            (graph.entry_point(), None)
                        };

                        // 自适应 ef（Phase 4.5）：用 nav.initialize 的 f32 距离
                        let ef = if let Some(ref adaptive) = adaptive_ef {
                            let entry_dist = nav_entry_dist.unwrap_or_else(|| {
                                l2_simd(query,
                                    &vectors[entry_point as usize * dim
                                        ..(entry_point as usize + 1) * dim])
                            });
                            adaptive.estimate_ef(entry_dist).max(k)
                        } else {
                            default_ef
                        };

                        let mut cands = VamanaGraph::greedy_search_vec_reuse(
                            vectors,
                            dim,
                            storage,
                            entry_point,
                            query,
                            ef,
                            visited,
                            pool,
                            po,
                        );
                        cands.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        cands.truncate(k);
                        cands
                    }
                })
            })
            .collect()
    }

    /// 批量搜索（返回纯 ID，用于 benchmark/ann-benchmarks 接口）
    ///
    /// 内部调用 batch_search，提取 ID
    pub fn batch_search_ids(
        &self,
        queries: &[&[f32]],
        k: usize,
    ) -> Vec<Vec<u32>> {
        self.batch_search(queries, k)
            .into_iter()
            .map(|results| results.into_iter().map(|(id, _)| id).collect())
            .collect()
    }
}

/// 用于 BinaryHeap 排序的 f32 包装（BinaryHeap 要求 Ord）
#[derive(Debug, Clone, Copy)]
struct OrderedF32(f32);

impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// 引入 RobustPrune（避免循环引用，模块内使用）
use super::robust_prune::{prune_dispatch, PruneStrategy};
use super::quant_aware_prune::{QuantAwareRobustPrune, QuantAwarePruneConfig};
use super::linear_pool::LinearPool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::ChaCha8Rng;

    #[test]
    fn build_small_graph() {
        // 10 个 4 维向量
        let vectors: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let dim = 4;
        let mut rng = ChaCha8Rng::seed_from(42);
let config = VamanaBuildConfig {
alpha: 1.0,
l_build: 10,
r_max: 4,
r_soft: 6,
max_iterations: 1,
saturate: true,
enable_layered_nav: false,
nav_m: 16,
prune_strategy: Default::default(),
};
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        assert_eq!(graph.len(), 10);
        assert_eq!(graph.dim(), 4);
    }

    #[test]
    fn search_returns_results() {
        let vectors: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let dim = 10;
        let mut rng = ChaCha8Rng::seed_from(42);
let config = VamanaBuildConfig {
alpha: 1.0,
l_build: 20,
r_max: 8,
r_soft: 12,
max_iterations: 1,
saturate: true,
enable_layered_nav: false,
nav_m: 16,
prune_strategy: Default::default(),
};
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        let mut searcher = GraphSearcher::new(&vectors, &graph, 20);
        let query = vectors[0..dim].to_vec();
        let results = searcher.search(&query, 5);
        assert!(!results.is_empty());
        // 查询自身应排在最前
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn empty_graph() {
        let vectors: Vec<f32> = vec![];
        let dim = 4;
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig::default();
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        assert_eq!(graph.len(), 0);
    }

    /// 序列化往返测试：build → serialize → deserialize → 查询结果一致
    #[test]
    fn serialize_roundtrip() {
        use crate::memory::serialize::Serializable;

        let vectors: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let dim = 10;
        let mut rng = ChaCha8Rng::seed_from(42);
let config = VamanaBuildConfig {
alpha: 1.2,
l_build: 20,
r_max: 8,
r_soft: 12,
max_iterations: 1,
saturate: true,
enable_layered_nav: false,
nav_m: 16,
prune_strategy: Default::default(),
};
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);

        // 序列化到字节
        let bytes = graph.serialize();
        assert!(bytes.len() > 16, "serialized bytes should include header");

        // 反序列化
        let restored = VamanaGraph::deserialize(&bytes).expect("deserialize should succeed");
        assert_eq!(restored.len(), graph.len());
        assert_eq!(restored.dim(), graph.dim());
        assert_eq!(restored.entry_point(), graph.entry_point());

        // 验证 metadata roundtrip（设计文档 F.7）
        assert!(restored.metadata().is_some(), "metadata should be present after roundtrip");
        let meta = restored.metadata().unwrap();
        assert_eq!(meta.n, 20);
        assert_eq!(meta.dim, 10);
        assert_eq!(meta.rng_algorithm, "chacha8");
        assert_eq!(meta.rng_seed, 42);

        // 查询结果应一致
        let q = vectors[0..dim].to_vec();
        let r1 = GraphSearcher::new(&vectors, &graph, 20).search(&q, 5);
        let r2 = GraphSearcher::new(&vectors, &restored, 20).search(&q, 5);
        assert_eq!(r1, r2, "search results should match after roundtrip");
    }

    /// 序列化到文件往返测试
    #[test]
    fn serialize_file_roundtrip() {
        use crate::memory::serialize::Serializable;

        let vectors: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let dim = 5;
        let mut rng = ChaCha8Rng::seed_from(42);
let config = VamanaBuildConfig {
alpha: 1.0,
l_build: 10,
r_max: 4,
r_soft: 6,
max_iterations: 1,
saturate: true,
enable_layered_nav: false,
nav_m: 16,
prune_strategy: Default::default(),
};
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);

        let tmp = std::env::temp_dir().join("raven_serialize_test.bin");
        graph.save(&tmp).expect("save should succeed");
        let restored = VamanaGraph::load(&tmp).expect("load should succeed");
        assert_eq!(restored.len(), graph.len());
        assert_eq!(restored.dim(), graph.dim());
        let _ = std::fs::remove_file(&tmp);
    }

    /// CRC 校验：篡改文件体应导致加载失败
    #[test]
    fn crc_corruption_detected() {
        use crate::memory::serialize::Serializable;

        let vectors: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let dim = 5;
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig::default();
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);

        let mut bytes = graph.serialize();
        // 篡改文件体最后一个字节
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;

        let err = VamanaGraph::deserialize(&bytes);
        assert!(err.is_err(), "corrupted CRC should be detected");
    }

    /// magic 不匹配应报错
    #[test]
    fn magic_mismatch_detected() {
        use crate::memory::serialize::Serializable;

        let vectors: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let dim = 5;
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig::default();
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);

        let mut bytes = graph.serialize();
        // 篡改 magic
        bytes[0] = 0x00;

        let err = VamanaGraph::deserialize(&bytes);
        assert!(err.is_err(), "magic mismatch should be detected");
    }
}
