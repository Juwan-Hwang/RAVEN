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
use crate::distance::F16Dataset;
use crate::memory::{HybridBlockedCsr, VisitedTracker};
use crate::quant::{SQ8Dataset, SQ4Dataset, PQ4Dataset, PQ8Dataset};
use super::adaptive_ef::AdaptiveEfConfig;
use super::refiner::{rerank_f32, rerank_f16};
use crate::build::ChaCha8Rng;
use crate::build::{BuildConfig, BuildMetadata};
use crate::graph::navigation::NavigationLayer;
use crate::graph::navigation::LayeredNavigation;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::info;

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
    /// 分层导航参数 M（层间缩减比，默认 32）
/// 全参数扫描确认 nav_m=32 比 16 QPS +3.3%、recall +0.03pp、CV 更低（3层 vs 4层导航）
    pub nav_m: usize,
    /// 剪枝策略：RobustPrune（Vamana 标准）或 DirectionalPrune（RAVEN 超越方案）
    /// DirectionalPrune 无 saturation，用 r_min 连通性补底替代 r_max 填充
    pub prune_strategy: super::robust_prune::PruneStrategy,
    /// 跳跃边数量（仅 DirectionalPrune 有效，0 = 禁用）
    pub n_jumps: usize,
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
nav_m: 32,
prune_strategy: super::robust_prune::PruneStrategy::RobustPrune,
n_jumps: 0,
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

        info!("[build] rayon threads: {}", rayon::current_num_threads());

        // 1. 初始化图：随机连接（设计文档上层导航：保留随机层级）
        let _random_entry = Self::init_random_graph(&mut storage, n, config, rng);
        // Vamana/DiskANN 论文：entry_point 用 medoid（离质心最近的点），不是随机点
        let entry_point = Self::compute_medoid(vectors, dim, n);
        // 断点续建：环境变量 RAVEN_BUILD_CKPT=路径。每 ~5 分钟原子写入 (iter, 下一批, 迭代起点 rng 位置, 图)。
        // 续建时恢复图与 rng，重新生成同一置换并跳过已完成批次——结果与不中断逐位一致。
        let ckpt_path = std::env::var("RAVEN_BUILD_CKPT").ok();
        let (mut start_iter, mut start_chunk) = (0usize, 0usize);
        if let Some(cp) = &ckpt_path {
            if let Some((it, ch, wp, st)) = ckpt::load(cp, n, config.r_max) {
                storage = st;
                rng.set_word_pos(wp);
                start_iter = it;
                start_chunk = ch;
                info!("[build] resumed checkpoint {cp}: iter={it} chunk={ch}");
                eprintln!("[build] resumed checkpoint {cp}: iter={it} chunk={ch}");
            }
        }
        let mut last_ckpt = std::time::Instant::now();
        let ckpt_secs: u64 = std::env::var("RAVEN_BUILD_CKPT_SECS").ok().and_then(|x| x.parse().ok()).unwrap_or(300);
        info!("[build] init_random_graph done, entry=medoid[{}]", entry_point);

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
        for iter in start_iter..config.max_iterations {
            let iter_wp = rng.get_word_pos();
            let order = Self::permutation(n, rng);
            let skip = if iter == start_iter { start_chunk } else { 0 };
            let progress = AtomicUsize::new(0);
            let progress_interval = (n / 20).max(1);
            let alpha = if iter == 0 { 1.0 } else { config.alpha };
            info!("[build] iter {}/{} alpha={}", iter + 1, config.max_iterations, alpha);

            for (ci, chunk) in order.chunks(BUILD_BATCH_SIZE).enumerate() {
                if ci < skip { continue; }
                if let Some(cp) = &ckpt_path {
                    if last_ckpt.elapsed().as_secs() >= ckpt_secs {
                        ckpt::save(cp, iter, ci, iter_wp, &storage);
                        last_ckpt = std::time::Instant::now();
                    }
                }
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
                            if idx > 0 && idx.is_multiple_of(progress_interval) {
                                info!("[build] {}/{} ({}%)", idx, n, idx * 100 / n);
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

        // 3b. 确保所有节点邻居列表按距离升序排列
        // final_prune 只处理度数 > r_max 的节点；其余节点可能因反向边追加而未排序。
        // 搜索时度数上限（search_degree）依赖邻居列表已排序。
        Self::sort_all_neighbors(&mut storage, vectors, dim);

        // 创建构建元数据（设计文档 F.7）
        let build_config = BuildConfig::default();
        let metadata = BuildMetadata::from_config(&build_config, n, dim);

        // 构建分层导航（设计文档：保留随机层级，可与 HNSW 直接对比）
        let layered_nav = if config.enable_layered_nav && n > 0 {
            info!("[build] constructing layered navigation (M={})...", config.nav_m);
            let t0 = std::time::Instant::now();
            let nav = LayeredNavigation::build(
                vectors, dim, &storage, entry_point, config.nav_m, config.r_max / 2,
            );
            info!("[build] layered nav done in {:.1}s (max_level={})",
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

    /// 从预计算候选集建图（候选集质量实验）
    ///
    /// 与标准 `build` 的唯一区别：候选集来自外部（如暴力 kNN），
    /// 而非贪心搜索的 visited set。其余流程（prune_dispatch →
    /// connect_bidirectional → final_prune → sort_all_neighbors →
    /// layered_nav）完全一致，确保唯一变量是候选集质量。
    ///
    /// `candidates[i]` = 节点 i 的候选邻居列表（无需排序，DirectionalPrune 内部排序）
    pub fn build_from_candidates(
        vectors: &[f32],
        dim: usize,
        candidates: &[Vec<u32>],
        config: &VamanaBuildConfig,
    ) -> Self {
        let n = vectors.len() / dim;
        assert_eq!(vectors.len(), n * dim);
        assert_eq!(candidates.len(), n);
        let mut storage = HybridBlockedCsr::new(n, config.r_max);

        let entry_point = Self::compute_medoid(vectors, dim, n);
        info!("[build_from_candidates] entry=medoid[{}], n={}", entry_point, n);

        let alpha = config.alpha;
        for node_id in 0..n as u32 {
            let pruned = prune_dispatch(
                config.prune_strategy,
                &candidates[node_id as usize],
                node_id,
                vectors,
                dim,
                alpha,
                config.r_max,
                config.saturate && alpha > 1.0,
            );
            Self::connect_bidirectional(
                &mut storage, node_id, &pruned, vectors, dim, alpha, config,
            );
        }

        Self::final_prune(
            &mut storage, vectors, dim, config.alpha, config.r_max,
            config.saturate, config.prune_strategy,
        );
        Self::sort_all_neighbors(&mut storage, vectors, dim);

        let layered_nav = if config.enable_layered_nav && n > 0 {
            info!("[build_from_candidates] constructing layered navigation...");
            let nav = LayeredNavigation::build(
                vectors, dim, &storage, entry_point, config.nav_m, config.r_max / 2,
            );
            Some(nav)
        } else {
            None
        };

        VamanaGraph { storage, entry_point, dim, n, metadata: None, layered_nav }
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
        let build_t0 = std::time::Instant::now();

        eprintln!("[build_qa] n={} dim={} threads={} beta={}",
            n, dim, rayon::current_num_threads(), qa_config.beta);

        let _random_entry = Self::init_random_graph(&mut storage, n, config, rng);
        let entry_point = Self::compute_medoid(vectors, dim, n);
        eprintln!("[build_qa] init_random_graph done ({:.1}s), entry=medoid[{}]",
            build_t0.elapsed().as_secs_f64(), entry_point);

        const BUILD_BATCH_SIZE: usize = 10_000;
        for iter in 0..config.max_iterations {
            let order = Self::permutation(n, rng);
            let alpha = if iter == 0 { 1.0 } else { config.alpha };
            let progress = AtomicUsize::new(0);
            let progress_interval = (n / 20).max(1);
            let iter_t0 = std::time::Instant::now();
            eprintln!("[build_qa] iter {}/{} alpha={} beta={} ...",
                iter + 1, config.max_iterations, alpha, qa_config.beta);

            for chunk in order.chunks(BUILD_BATCH_SIZE) {
                let new_neighbors: Vec<(u32, Vec<u32>)> = chunk
                    .par_iter()
                    .map_init(
                        || VisitedTracker::new(n, config.l_build),
                        |visited, &node_id| {
                            let idx = progress.fetch_add(1, Ordering::Relaxed);
                            if idx > 0 && idx.is_multiple_of(progress_interval) {
                                eprintln!("[build_qa] iter {}  {}/{} ({}%)  {:.1}s",
                                    iter + 1, idx, n, idx * 100 / n,
                                    iter_t0.elapsed().as_secs_f64());
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

            eprintln!("[build_qa] iter {} done in {:.1}s",
                iter + 1, iter_t0.elapsed().as_secs_f64());
        }

        eprintln!("[build_qa] final_prune (QA-aware) ...");
        let prune_t0 = std::time::Instant::now();
        Self::final_prune_qa(&mut storage, vectors, dim, qa_config, &error_fn);
        eprintln!("[build_qa] final_prune done in {:.1}s", prune_t0.elapsed().as_secs_f64());

        let build_config = BuildConfig::default();
        let metadata = BuildMetadata::from_config(&build_config, n, dim);

        let layered_nav = if config.enable_layered_nav && n > 0 {
            eprintln!("[build_qa] constructing layered navigation (M={})...", config.nav_m);
            let t0 = std::time::Instant::now();
            let nav = LayeredNavigation::build(
                vectors, dim, &storage, entry_point, config.nav_m, config.r_max / 2,
            );
            eprintln!("[build_qa] layered nav done in {:.1}s (max_level={})",
                t0.elapsed().as_secs_f64(), nav.max_level());
            Some(nav)
        } else {
            None
        };

        eprintln!("[build_qa] total build: {:.1}s", build_t0.elapsed().as_secs_f64());

        VamanaGraph { storage, entry_point, dim, n, metadata: Some(metadata), layered_nav }
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
        for c in centroid.iter_mut().take(dim) {
            *c /= SAMPLE_COUNT as f32;
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
        for c in centroid.iter_mut().take(dim) {
            *c /= n as f32;
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
                    crate::arch::prefetch_l1(ptr);
                    crate::arch::prefetch_l1(ptr.add(64));
                    crate::arch::prefetch_l1(ptr.add(128));
                    crate::arch::prefetch_l1(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

// 第一遍：收集未访问邻居到 edge_buf（branchless — A8 优化）
// 与 SQ8/PQ4/PQ8 路径保持一致，用 cmov 代替不可预测分支
let mut edge_size = 0usize;
for &v in neighbors {
if edge_size >= 128 {
break;
}
// SAFETY: v 来自图边，保证 < n
let is_new = unsafe { visited.visit_unchecked_branchless(v) };
unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
edge_size += is_new as usize;
}

            // 预取前 po 个邻居的向量数据
            let prefetch_count = po.min(edge_size);
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = &vectors[v * dim] as *const f32 as *const i8;
                unsafe { crate::arch::prefetch_l1(ptr); }
            }

            // 第二遍：计算距离，同时前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = &vectors[v * dim] as *const f32 as *const i8;
                    unsafe { crate::arch::prefetch_l1(ptr); }
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
    pub fn greedy_search_sq8<const RAW: bool>(
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
        let entry_dist = if RAW {
            sq8.distance_raw(query_code, entry_point as usize)
        } else {
            sq8.distance(query_code, entry_point as usize)
        };
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        let mut edge_buf: [u32; 128] = [0; 128];

        while let Some((node, _dist)) = pool.pop() {
            // Multi-line graph prefetch：按 r_max 计算实际 cache line 数
            // R_max=32 → 32×4B=128B=2 lines；R_max=64 → 256B=4 lines
            if let Some((next_node, _)) = pool.peek_unchecked() {
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                let neighbor_bytes = storage.r_max() * 4; // u32 per neighbor
                let graph_lines = neighbor_bytes.div_ceil(64); // ceil to cache lines
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    for l in 1..graph_lines {
                        crate::arch::prefetch_l1(ptr.add(l * 64));
                    }
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居（branchless — A8 优化）
            // 图遍历中"邻居是否已访问"取决于图拓扑，本质不可预测。
            // branchless 消除分支误预测惩罚，用 cmov 代替 if-branch。
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                // SAFETY: v 来自图边，保证 < n
                let is_new = unsafe { visited.visit_unchecked_branchless(v) };
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += is_new as usize;
            }

            // 预取前 po 个邻居的 SQ8 码
            // SIFT-128: 128B = 2 cache lines，始终预取两行（分支消除）
            let prefetch_count = po.min(edge_size);
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = sq8.code(v).as_ptr() as *const i8;
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    crate::arch::prefetch_l1(ptr.add(64));
                }
            }

            // 第二遍：SQ8 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = sq8.code(v).as_ptr() as *const i8;
                    unsafe {
                        crate::arch::prefetch_l1(ptr);
                        crate::arch::prefetch_l1(ptr.add(64));
                    }
                }
                let neighbor = edge_buf[i];
                // 热路径：跳过 bounds check（neighbor 已通过 visited.visit 验证有效）
                let d = if RAW {
                    unsafe { sq8.distance_raw_unchecked(query_code, neighbor as usize) }
                } else {
                    unsafe { sq8.distance_unchecked(query_code, neighbor as usize) }
                };
                pool.insert(neighbor, d);
            }
        }

        pool.to_sorted_vec()
    }

    /// R53 ADC 版 SQ8 遍历（加权、查询不截断）
    pub fn greedy_search_sq8_adc(
        sq8: &SQ8Dataset,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        aq: &crate::quant::sq8::Sq8AdcQuery,
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
    ) -> Vec<(u32, f32)> {
        visited.reset();
        pool.reset_for(l);

        // 入口节点距离（SQ8）
        let entry_dist = unsafe { sq8.adc_dist_unchecked(aq, entry_point as usize) };
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        let mut edge_buf: [u32; 128] = [0; 128];

        while let Some((node, _dist)) = pool.pop() {
            // Multi-line graph prefetch：按 r_max 计算实际 cache line 数
            // R_max=32 → 32×4B=128B=2 lines；R_max=64 → 256B=4 lines
            if let Some((next_node, _)) = pool.peek_unchecked() {
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                let neighbor_bytes = storage.r_max() * 4; // u32 per neighbor
                let graph_lines = neighbor_bytes.div_ceil(64); // ceil to cache lines
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    for l in 1..graph_lines {
                        crate::arch::prefetch_l1(ptr.add(l * 64));
                    }
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居（branchless — A8 优化）
            // 图遍历中"邻居是否已访问"取决于图拓扑，本质不可预测。
            // branchless 消除分支误预测惩罚，用 cmov 代替 if-branch。
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                // SAFETY: v 来自图边，保证 < n
                let is_new = unsafe { visited.visit_unchecked_branchless(v) };
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += is_new as usize;
            }

            // 预取前 po 个邻居的 SQ8 码
            // SIFT-128: 128B = 2 cache lines，始终预取两行（分支消除）
            let prefetch_count = po.min(edge_size);
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = sq8.code(v).as_ptr() as *const i8;
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    crate::arch::prefetch_l1(ptr.add(64));
                }
            }

            // 第二遍：SQ8 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = sq8.code(v).as_ptr() as *const i8;
                    unsafe {
                        crate::arch::prefetch_l1(ptr);
                        crate::arch::prefetch_l1(ptr.add(64));
                    }
                }
                let neighbor = edge_buf[i];
                // 热路径：跳过 bounds check（neighbor 已通过 visited.visit 验证有效）
                let d = unsafe { sq8.adc_dist_unchecked(aq, neighbor as usize) };
                pool.insert(neighbor, d);
            }
        }

        pool.to_sorted_vec()
    }

    /// SQ4 标量量化图遍历
    ///
    /// 与 `greedy_search_sq8` 相同的 Two-Pass Prefetch 架构，
    /// 但距离计算用 SQ4 packed 4-bit 码代替 SQ8 u8 码。
    ///
    /// 优势：
    /// - 内存带宽降 2x（64B/vector vs SQ8 的 128B for SIFT-128）
    /// - AVX2 每次处理 64 维（32 bytes packed），SIFT-128 仅 2 次迭代（vs SQ8 的 4 次）
    /// - 1M 向量仅 64MB，部分进入 L3 cache（vs SQ8 的 128MB 完全不进 L3）
    ///
    /// 返回 (节点ID, SQ4距离) 列表，需外部 f32 rerank。
    pub fn greedy_search_sq4(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
    ) -> Vec<(u32, f32)> {
        visited.reset();
        pool.reset_for(l);

        // 入口节点距离（SQ4）
        let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        Self::sq4_expand::<false>(sq4, storage, query_code, visited, pool, po, search_degree);
        pool.to_sorted_vec()
    }

    /// H09：多种子 SQ4 图遍历。`seeds` 为 (节点, SQ4 raw 距离)，调用方保证已在 `visited` 中标记
    /// （导航层轨迹节点全部保持 visited，不重复计算）。不 reset `visited`。
    #[allow(clippy::too_many_arguments)]
    pub fn greedy_search_sq4_seeded(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        seeds: &[(u32, f32)],
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
    ) -> Vec<(u32, f32)> {
        Self::greedy_search_sq4_seeded_nv(sq4, storage, seeds, query_code, l, visited, pool, po, search_degree, false)
    }

    /// R27 S1：同 [`greedy_search_sq4_seeded`]，`novis = true` 时 L0 不用 visited（H153）。调用方必须保证：
    /// 所有已在 `visited` 中标记的节点（导航轨迹）都已作为种子提供（插入或被满池拒绝），否则不再逐位等价。
    #[allow(clippy::too_many_arguments)]
    pub fn greedy_search_sq4_seeded_nv(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        seeds: &[(u32, f32)],
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
        novis: bool,
    ) -> Vec<(u32, f32)> {
        pool.reset_for(l);
        for &(v, d) in seeds {
            pool.insert(v, orx_dist(v).unwrap_or(d));
        }
        if novis { Self::sq4_expand::<true>(sq4, storage, query_code, visited, pool, po, search_degree); }
        else { Self::sq4_expand::<false>(sq4, storage, query_code, visited, pool, po, search_degree); }
        pool.to_sorted_vec()
    }

    /// SQ4 图遍历主循环（greedy_search_sq4 / _seeded 共用，inline(always) 保持热路径代码不变）
    #[inline(always)]
    fn sq4_expand<const NV: bool>(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        query_code: &[u8],
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
    ) {
        let mut edge_buf: [u32; 128] = [0; 128];
        let hsel_rec = HSEL_REC_PTR.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let hsel_k = if hsel_rec.is_null() || HSEL_LUTONLY.load(std::sync::atomic::Ordering::Relaxed) { 0 } else { HSEL_K.load(std::sync::atomic::Ordering::Relaxed) };
        let hsel_epf = HSEL_EPF.load(std::sync::atomic::Ordering::Relaxed);
        let hsel_rpf = HSEL_RPF.load(std::sync::atomic::Ordering::Relaxed);
        let hsel_lut: *const f32 = HSEL_LUT.with(|l| l.get() as *const f32);
        // R23 HREL：每边残差 PCA 投影（行内顺序块，提前一跳预取）
        let hrel_base = HREL_PTR.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let hrel_k = if hrel_base.is_null() { 0 } else { HREL_K.load(std::sync::atomic::Ordering::Relaxed) };
        let hrel_bsz = HREL_BSZ.load(std::sync::atomic::Ordering::Relaxed);
        let hrel_epf = HREL_EPF.load(std::sync::atomic::Ordering::Relaxed);
        let hrel_pq: *const f32 = HREL_PQ.with(|l| l.get() as *const f32);
        let mut hrel_sc: [f32; 128] = [0.0; 128];
        let mut slot_buf: [u8; 128] = [0; 128];
        // H128：前瞻流水线移植到精简路径（第 11 轮：aarch64 默认 L0EA=0 后默认走本路径，原 L0 路径的 la 不再生效）。
        // ARM 4 轮：SIFT +3.2–5.7%（8/8），GloVe +2.0–6.5%（8/8 为正，均值 2.9% <3%）→ 默认关；RAVEN_LEAN_LA=8 可开，AB 旋钮 lla
        let la = lean_la_depth();
        let mut adj_ready: [u32; 2] = [u32::MAX; 2];
        let mindist_lambda = if cfg!(feature = "diag") { f32::from_bits(MINDIST_LAMBDA.load(std::sync::atomic::Ordering::Relaxed)) } else { 0.0 };
        let mindist_r = MINDIST_R.load(std::sync::atomic::Ordering::Relaxed) as *const f32;
        // A1（第 12 轮）：精简路径轨迹记录（仅 diag 构建；release 热路径不变）
        let a1 = cfg!(feature = "diag") && A1_TRACE_ON.load(std::sync::atomic::Ordering::Relaxed);
        if a1 { a1_begin(visited, pool); }
        let vh = vec_hop();
        let orx = cfg!(feature = "diag") && ORX_Q.load(std::sync::atomic::Ordering::Relaxed) != 0;
        let pfl = { let cl = sq4.code_bytes.div_ceil(64); match PF_LINES.load(std::sync::atomic::Ordering::Relaxed) { usize::MAX => if cl >= 4 { cl } else { 1 }, l => l.min(cl).max(1) } };
        #[cfg(target_arch = "aarch64")]
        let qdec = if vh & 4 != 0 && sq4.code_bytes == 64 { Some(crate::quant::sq4::QDec64::new(query_code)) } else { None };
        // R25 CM（代价模型裁决 oracle，仅 diag_cm 构建）：见 CmState
        let cm = cfg!(feature = "diag_cm") && CM_ON.load(std::sync::atomic::Ordering::Relaxed);
        let cmp: *mut CmState = if cm { CM.with(|c| c.as_ptr()) } else { std::ptr::null_mut() };
        let (cm_rec, cm_thr, cm_chain, cm_extra, cm_seed, cm_block) = if cm { let c = unsafe { &*cmp }; (c.rec, c.keep_thr, c.chain, c.extra, c.seed, c.block) } else { (false, u32::MAX, 0, 0, 0, 0) };
        let cm_zero: u32 = if cm { std::hint::black_box(0u32) } else { 0 };
        let cm_ah: u32 = if cm && !cm_rec { unsafe { (*cmp).ah } } else { 0 };
        let mut cm_hi: usize = 0;
        let (cm_tab, cm_mask) = if cm { (CM_TAB.load(std::sync::atomic::Ordering::Relaxed) as *const u32, CM_MASK.load(std::sync::atomic::Ordering::Relaxed)) } else { (std::ptr::null(), 0) };
        let mut cm_sink = 0f32;
        let cmp_u = cmp as usize; let cm_keep = move |v: u32| -> bool { (cm_mix(v ^ cm_seed) <= cm_thr) | unsafe { (*(cmp_u as *const CmState)).contains(v) } };
        // R27 ZA：热态逐项消融（计时轮；全部位都让池/轨迹/结果与基线逐位一致）。位：1 距离核→记录距离（不读码）· 2 距离核→记录距离 + 依赖读码首末字节
        // （理想计算、真实访存）· 4 去掉 L0 全部码预取指令 · 8 门槛比较理想化（不对逐码调 insert，跳末只插入记录的入池者）· 16 入池理想化（不插入，结束时装入记录的池；隐含 8|32）
        // · 32 弹出回放（按记录弹出序列，不调 pool.pop/peek）· 64 visited 回放（按记录的 is_new，不碰位图/history/visited 预取）· 128 邻接回放（直接装入记录的新邻居表；隐含 64）
        // · 256 整个 L0 循环置零（只装入记录的结果池）· 512 visited 不写 history（调用方在计时外按记录清位图）
        let cm_za: u32 = if cm && !cm_rec { unsafe { (*cmp).za } } else { 0 };
        if cm_za & 256 != 0 { pool.load_sorted(unsafe { &(*cmp).q_fin }); return; }
        let (z_kz, z_kl, z_pf, z_th, z_pi, z_pr, z_vr, z_ar, z_hz) = (cm_za & 1 != 0, cm_za & 2 != 0, cm_za & 4 != 0, cm_za & 24 != 0, cm_za & 16 != 0, cm_za & 48 != 0, cm_za & 192 != 0, cm_za & 128 != 0, cm_za & 512 != 0);
        let (mut z_di, mut z_vi) = (0usize, 0usize);
        let (z_qpop, z_npop, z_qisn, z_qdist): (*const u32, usize, *const u8, *const f32) = if cm_za != 0 { let c = unsafe { &*cmp }; (c.q_pop.as_ptr(), c.q_pop.len(), c.q_isn.as_ptr(), c.q_dist.as_ptr()) } else { (std::ptr::null(), 0, std::ptr::null(), std::ptr::null()) };
        let mut z_sink = 0u32;
        let cm_filt = cm && cm_thr != u32::MAX; // R27：f=1 时不做（全保留的）过滤，去掉每邻居一次哈希查表的测量开销

        // N4（第 13 轮）：确认阶段批量化（RAVEN_N4 = T | T'<<8 | H<<16 | flags<<24；0 = 关）
        let n4 = n4_cfg();
        let (n4_t, n4_tp, n4_h, n4_f) = ((n4 & 0xFF) as usize, ((n4 >> 8) & 0xFF) as usize, ((n4 >> 16) & 0xFF) as usize, n4 >> 24);
        let mut n4_hops = 0usize;
        // R27 S1：无 visited 仅在精确等价的纯路径上生效（诊断/选择类旋钮会撤销 visited 或改变候选集 → 回退到 visited）
        let nv = storage.r_max() <= 128 && NV && hsel_k == 0 && hrel_k == 0 && !cm && n4_t == 0 && !a1 && !orx && mindist_lambda == 0.0
            && G1_K.load(std::sync::atomic::Ordering::Relaxed) == 0;
        let nvrf = NvRf::begin(nv);
        let s2 = nv && s2_mode() != 0;
        loop {
            if n4_t > 0 && n4_hops >= n4_h && pool.cursor() >= n4_t && pool.has_next() {
                n4_batch(sq4, storage, query_code, visited, pool, search_degree, n4_tp, n4_f);
                continue;
            }
            let node = if z_pr { if cm_hi >= z_npop { break } unsafe { *z_qpop.add(cm_hi) } } else { let Some((node, _dist)) = pool.pop() else { break }; node };
            pop_rec(node);
            n4_hops += 1;
            crate::diag_count!(HOPS);
            // R25 CM：每跳串行插入 cm_chain 次相互依赖的随机访存，并让本跳邻接行地址依赖其结果（node + (x & 0)）
            let node = if cm && cm_chain > 0 && !cm_rec {
                let mut x = node as usize;
                for c in 0..cm_chain { x = unsafe { *cm_tab.add((cm_mix(x as u32 ^ c) as usize) & cm_mask) } as usize; }
                node + (x as u32 & cm_zero)
            } else { node };
            if cm { unsafe { (*cmp).hops += 1; } }
            if cm_rec { let c = unsafe { &mut *cmp }; c.r_pop.push(node); let l = c.r_new.len() as u32; c.r_off.push(l); let a = c.r_acc2.len() as u32; c.r_aoff.push(a); }
            // R26 H148 oracle：按记录轨迹提前 D 跳预取邻接行、提前 Dc 跳预取码（完美弹出序列预测的上界；预取本身计时）
            if cm_ah != 0 {
                let c = unsafe { &*cmp };
                let (d, dc, md) = ((cm_ah & 15) as usize, ((cm_ah >> 4) & 15) as usize, cm_ah >> 8);
                if md & 1 != 0 { if let Some(&u) = c.q_pop.get(cm_hi + d) {
                    let er = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                    let rp = storage.main_block().as_ptr().wrapping_add(u as usize * storage.r_max()) as *const i8;
                    for l in 0..(er * 4).div_ceil(64) { unsafe { if md & 8 != 0 { crate::arch::prefetch_l2(rp.add(l * 64)); } else { crate::arch::prefetch_l1(rp.add(l * 64)); } } } } }
                // R26 H150（md&8）：远期预取改发 L2 提示（避开 L1 填充缓冲上限）；md&16：同时预取第 i+Dc 跳新邻居的 visited 位图行
                if md & 22 != 0 { let j = cm_hi + dc; if j + 1 < c.q_off.len() {
                    for &v in &c.q_new[c.q_off[j] as usize..c.q_off[j + 1] as usize] {
                        if md & 16 != 0 { unsafe { visited.prefetch(v); } }
                        if md & 4 != 0 || (md & 2 != 0 && c.contains(v)) { let cp = sq4.code(v as usize).as_ptr() as *const i8;
                            unsafe { if md & 8 != 0 { crate::arch::prefetch_l2(cp); } else { crate::arch::prefetch_l1(cp); } } } } } }
            }
            cm_hi += 1;
            // CM 行内块（Descartes 式"邻居码内联"代价模型）：每跳一个由 node 决定位置的连续 cm_block 行块，跳开始时整块预取，
            // 在本跳距离计算之前逐行读取（块是本跳决策的输入 → 在链上）
            let cm_blk: usize = if cm && cm_block > 0 && !cm_rec {
                let b = (cm_mix(node ^ 0xB10C) as usize & cm_mask) & !15usize; let b = b.min(cm_mask + 1 - 16 * cm_block as usize);
                for l in 0..cm_block as usize { unsafe { crate::arch::prefetch_l1(cm_tab.add(b + l * 16) as *const i8); } }
                b } else { usize::MAX };
            // CM 额外取码：位 31 置位时在跳开始（弹出后）一次性预取全部额外码（与真实码的早预取同位置，不在链上等）
            if cm && cm_extra & 0x8000_0000 != 0 && !cm_rec {
                let n = sq4.n as u32; let base = cm_mix(node ^ cm_seed);
                for j in 0..(cm_extra & 0xFFFF) { let v = cm_mix(base.wrapping_add(j)) % n; unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); } }
            }
            if a1 { A1_TRACE.with(|t| { let mut t = t.borrow_mut(); t.push(A1_HOP); t.push(node); }); }
            let mut edge_size = 0usize;
            if z_ar {
                // R27 ZA 128：邻接回放——不读邻接行/度数、不做 visited，直接装入记录的本跳新邻居
                let c = unsafe { &*cmp }; let (a, b) = (c.q_off[cm_hi - 1] as usize, c.q_off[cm_hi] as usize);
                edge_buf[..b - a].copy_from_slice(&c.q_new[a..b]); edge_size = b - a;
            } else {
            if la > 0 {
                let mut nr: [u32; 2] = [u32::MAX; 2];
                if let Some((c, _)) = pool.peek_unchecked() {
                    nr[0] = c;
                    if c == adj_ready[0] || c == adj_ready[1] {
                        let nb = storage.neighbors(c);
                        let lim = la.min(nb.len()).min(if search_degree > 0 { search_degree } else { usize::MAX });
                        for &v in &nb[..lim] {
                            unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); }
                        }
                    }
                }
                if let Some(c2) = pool.peek_second_unchecked() {
                    nr[1] = c2;
                    let ptr = storage.main_block().as_ptr().wrapping_add(c2 as usize * storage.r_max()) as *const i8;
                    let effective_r = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                    unsafe { for l in 0..(effective_r * 4).div_ceil(64) { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
                    // R14（vh 位 64）：度数数组是邻接切片长度的来源，另一次随机访存 → 与邻接行一起预取
                    if vh & 64 != 0 { unsafe { crate::arch::prefetch_l1(storage.degrees_ptr().wrapping_add(c2 as usize) as *const i8); } }
                }
                adj_ready = nr;
            }
            // Multi-line graph prefetch（尊重 search_degree：只预取实际会读取的字节）
            let z_pk = if z_pr { if cm_hi < z_npop { Some((unsafe { *z_qpop.add(cm_hi) }, 0f32)) } else { None } } else { pool.peek_unchecked() };
            if let Some((next_node, _)) = z_pk {
                if vh & 64 != 0 { unsafe { crate::arch::prefetch_l1(storage.degrees_ptr().wrapping_add(next_node as usize) as *const i8); } }
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                // search_degree > 0 时只预取前 search_degree 个邻居的字节
                let effective_r = if search_degree > 0 {
                    search_degree.min(storage.r_max())
                } else {
                    storage.r_max()
                };
                let neighbor_bytes = effective_r * 4;
                let graph_lines = neighbor_bytes.div_ceil(64);
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    for l in 1..graph_lines {
                        crate::arch::prefetch_l1(ptr.add(l * 64));
                    }
                }
                if hrel_k > 0 { let b = hrel_base.wrapping_add(next_node as usize * hrel_bsz) as *const i8; unsafe { for l in 0..hrel_bsz / 64 { crate::arch::prefetch_l1(b.add(l * 64)); } } }
            }

            // R14b（vh 位 128）：不读 degrees（跳链上一次独立随机访存），直接取邻接行前 cap 槽，按 SENTINEL 计数得长度
            // （行按序填充、空槽为 SENTINEL；cap 槽已随邻接行预取进 L1）。与原路径逐位一致。
            let (neighbors, effective_len) = if vh & 128 != 0 {
                let cap = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                let row = unsafe { std::slice::from_raw_parts(storage.main_block().as_ptr().add(node as usize * storage.r_max()), cap) };
                let mut c = 0usize; for &x in row { c += (x != crate::SENTINEL) as usize; }
                (&row[..c], c)
            } else {
                let neighbors = storage.neighbors(node);
                // 搜索时度数上限：只看前 search_degree 个邻居（已按距离升序排列）
                // search_degree=0 表示不限制
                let effective_len = if search_degree > 0 {
                    search_degree.min(neighbors.len())
                } else {
                    neighbors.len()
                };
                (neighbors, effective_len)
            };

            // 第一遍：收集未访问邻居（branchless — A8）
            //
            // Visited 预取：visited 数组 = N bytes（1M = 1MB），超出 L2 cache。
            // 每个邻居的 visited 检查是随机访问，提前 po 步预取隐藏延迟。
            // H136（第 12 轮，A1 冷回放发现 visited 过滤位于"邻接→码取"关键路径上）：RAVEN_VH 位 8/16/32 = 过滤前直接按原始邻接
            // 预取前 8/16/全部 个邻居的码（已访问者的码多半已在缓存，浪费小），让首批码取不等 visited 过滤。结果逐位一致（只加预取）
            // R23 HREL：先对全部邻居打分（行内块已提前一跳预取），早码预取改为按分数取前 hrel_epf 个
            // R26 段归因：记录本跳邻接行（与上方预取同口径：前 effective_r 个槽的行）及度数项
            if cm_rec {
                let c = unsafe { &mut *cmp };
                let er = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                let rp = storage.main_block().as_ptr().wrapping_add(node as usize * storage.r_max()) as usize;
                for l in 0..(er * 4).div_ceil(64) { c.r_adj.push(rp + l * 64); }
                if vh & 128 == 0 { c.r_adj.push(storage.degrees_ptr().wrapping_add(node as usize) as usize); }
            }
            if hrel_k > 0 {
                unsafe { hrel_scores(hrel_base.add(node as usize * hrel_bsz), hrel_pq, effective_len, &mut hrel_sc); }
                if hrel_epf > 0 {
                    let mut tmp: [(f32, u8); 128] = [(0.0, 0); 128];
                    for i in 0..effective_len { tmp[i] = (hrel_sc[i], i as u8); }
                    let e = hrel_epf.min(effective_len);
                    if e < effective_len { tmp[..effective_len].select_nth_unstable_by(e - 1, |a, b| a.0.total_cmp(&b.0)); }
                    for t in &tmp[..e] { unsafe { crate::arch::prefetch_l1(sq4.code(neighbors[t.1 as usize] as usize).as_ptr() as *const i8); } }
                }
            } else if hsel_k > 0 {
                if hsel_rpf { for &v in neighbors.iter().take(effective_len) { unsafe { crate::arch::prefetch_l1(hsel_rec.add(v as usize * HSEL_REC) as *const i8); } } }
                let n_epf = hsel_epf.min(effective_len);
                for &v in neighbors.iter().take(n_epf) { unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); } }
            } else if vh & 56 != 0 && !z_pf {
                let n_epf = if vh & 32 != 0 { effective_len } else if vh & 16 != 0 { 16 } else { 8 }.min(effective_len);
                if cm_filt && !cm_rec {
                    // CM：完美选择下不会取被丢弃者的码 → 早码预取也只预取保留者
                    for &v in neighbors.iter().take(n_epf) { let k = cm_keep(v); let a = if k { sq4.code(v as usize).as_ptr() } else { query_code.as_ptr() }; unsafe { crate::arch::prefetch_l1(a as *const i8); } }
                } else {
                for &v in neighbors.iter().take(n_epf) { unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); } }
                }
            }
            let pf_vis = if z_vr || nv { 0 } else { po.min(effective_len) };
            for &v in neighbors.iter().take(pf_vis) {
                unsafe { visited.prefetch(v); }
            }

            // G1 oracle（第 13 轮）：只保留与 (q−u) 夹角最小的前 g1k 个邻居（作弊：用真 q 的 f32 方向）
            let g1k = G1_K.load(std::sync::atomic::Ordering::Relaxed);
            let mut g1_buf: [u32; 128] = [0; 128];
            let (neighbors, effective_len) = if g1k > 0 && effective_len > g1k { let n = g1_select(node, &neighbors[..effective_len], g1k, &mut g1_buf); (&g1_buf[..n], n) } else { (neighbors, effective_len) };
            if nv {
                // R27 S1：不过滤，原样取全部邻居（重复者在入池时由门槛/去重拒绝）
                let n = effective_len.min(128);
                edge_size = nvrf.fill(&neighbors[..n], &mut edge_buf);
            } else {
            for (i, &v) in neighbors.iter().take(effective_len).enumerate() {
                if edge_size >= 128 {
                    break;
                }
                if i + po < effective_len && !z_vr {
                    unsafe { visited.prefetch(neighbors[i + po]); }
                }
                if cm_rec { unsafe { (*cmp).r_vis.push(v); } }
                let is_new = if z_vr { let b = unsafe { *z_qisn.add(z_vi) } != 0; z_vi += 1; b }
                    else if z_hz { unsafe { visited.visit_nohist(v) } }
                    else { unsafe { visited.visit_unchecked_branchless(v) } };
                if cm_rec { unsafe { (*cmp).r_isn.push(is_new as u8); } }
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; *slot_buf.get_unchecked_mut(edge_size) = i as u8; }
                edge_size += is_new as usize;
            }
            } // R27 S1 nv else 结束
            if hrel_k > 0 && edge_size > hrel_k && pool.is_full() {
                let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                for j in 0..edge_size { sc[j] = (unsafe { *hrel_sc.get_unchecked(*slot_buf.get_unchecked(j) as usize) }, edge_buf[j]); }
                sc[..edge_size].select_nth_unstable_by(hrel_k - 1, |a, b| a.0.total_cmp(&b.0));
                for j in hrel_k..edge_size { visited.unvisit(sc[j].1); }
                for j in 0..hrel_k { edge_buf[j] = sc[j].1; }
                edge_size = hrel_k;
            }
            // R23 HSEL：池满且新邻居多于 k 时，用 1-bit RaBitQ 全局记录估计距离，只保留前 k 个取 SQ4（其余撤销 visited，可被再次发现）
            if hsel_k > 0 && edge_size > hsel_k && pool.is_full() {
                edge_size = unsafe { hsel_select(hsel_rec, hsel_lut, &mut edge_buf, edge_size, hsel_k, visited) };
            }

            // R25 CM：固定轨迹的选择 oracle——只保留"记录轮中插入池成功"者 + 被拒者中按哈希比例 f 保留的部分（被拒者不改变池 → 轨迹逐位一致）
            if cm {
                let c = unsafe { &mut *cmp };
                c.n_new += edge_size as u64;
                if cm_rec { c.r_new.extend_from_slice(&edge_buf[..edge_size]); }
                if !cm_rec && cm_filt {
                    let mut k = 0usize;
                    for j in 0..edge_size { let v = edge_buf[j]; edge_buf[k] = v; k += cm_keep(v) as usize; }
                    edge_size = k;
                }
                c.n_kept += edge_size as u64;
                if cm_blk != usize::MAX { let mut acc = 0u32; for l in 0..cm_block as usize { acc = acc.wrapping_add(unsafe { *cm_tab.add(cm_blk + l * 16) }); }
                    edge_size += (acc & cm_zero) as usize; }
            }
            } // R27 ZA 128 else 结束
            // 预取前 po 个邻居的 SQ4 码
            // SIFT-128: 64B = 1 cache line，只需 1 条预取指令
            let prefetch_count = if z_pf { 0 } else { po.min(edge_size) };
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = sq4.code(v).as_ptr() as *const i8;
                unsafe { pf_code_lines(ptr, pfl); }
            }

            // R28 S2（H152 批量归并池，RAVEN_S2 / AB s2，仅 S1 无 visited 生效时）：整跳先算完距离写 dbuf，再一次并入池
            // （跳初门槛压缩 → 稳定排序 → 去重 → 反向原地归并；与逐个 insert_dedup 逐位一致，见 LinearPool::merge_batch_dedup）
            if s2 {
                let mut dbm: [std::mem::MaybeUninit<f32>; 128] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                let dbuf = unsafe { &mut *(dbm.as_mut_ptr() as *mut [f32; 128]) };
                for i in 0..edge_size {
                    if i + po < edge_size && !z_pf {
                        let v = unsafe { *edge_buf.get_unchecked(i + po) } as usize;
                        unsafe { pf_code_lines(sq4.code(v).as_ptr() as *const i8, pfl); }
                    }
                    let nb = unsafe { *edge_buf.get_unchecked(i) } as usize;
                    #[cfg(target_arch = "aarch64")]
                    let d = if let Some(qd) = &qdec { unsafe { qd.dist(sq4.code_unchecked(nb).as_ptr()) } } else { unsafe { sq4.distance_raw_unchecked(query_code, nb) } };
                    #[cfg(not(target_arch = "aarch64"))]
                    let d = unsafe { sq4.distance_raw_unchecked(query_code, nb) };
                    unsafe { *dbuf.get_unchecked_mut(i) = d; }
                }
                pool.merge_batch_dedup(&edge_buf[..edge_size], &dbuf[..edge_size]);
                continue;
            }
            // H130（第 12 轮）：向量化跳执行（RAVEN_VH 位掩码，默认 0）：1 = 无分支插入位置，2 = 先整批算距离再按跳初阈值掩码过滤，
            // 4 = 64B 码查询预解码核。全部与原路径逐位一致（阈值只会单调变小，被掩码过滤者必被原 insert 拒绝）。
            if vh & 7 != 0 && !(cfg!(feature = "diag") && (mindist_lambda > 0.0 || a1 || orx)) {
                let mut dbuf: [f32; 128] = [0.0; 128];
                for i in 0..edge_size {
                    if i + po < edge_size {
                        let v = unsafe { *edge_buf.get_unchecked(i + po) } as usize;
                        unsafe { pf_code_lines(sq4.code(v).as_ptr() as *const i8, pfl); }
                    }
                    let nb = unsafe { *edge_buf.get_unchecked(i) } as usize;
                    #[cfg(target_arch = "aarch64")]
                    let d = if let Some(qd) = &qdec { unsafe { qd.dist(sq4.code_unchecked(nb).as_ptr()) } } else { unsafe { sq4.distance_raw_unchecked(query_code, nb) } };
                    #[cfg(not(target_arch = "aarch64"))]
                    let d = unsafe { sq4.distance_raw_unchecked(query_code, nb) };
                    unsafe { *dbuf.get_unchecked_mut(i) = d; }
                    if vh & 2 == 0 {
                        if nv { pool.insert_dedup(nb as u32, d); } else if vh & 1 != 0 { pool.insert_bl(nb as u32, d); } else { pool.insert(nb as u32, d); }
                    }
                }
                if vh & 2 != 0 {
                    let w = if pool.is_full() { pool.worst_distance() } else { f32::INFINITY };
                    let mut base = 0usize;
                    while base < edge_size {
                        let end = (base + 64).min(edge_size);
                        let mut m: u64 = 0;
                        for j in base..end { m |= ((unsafe { *dbuf.get_unchecked(j) } < w) as u64) << (j - base); }
                        while m != 0 {
                            let j = base + m.trailing_zeros() as usize; m &= m - 1;
                            let (nb, d) = unsafe { (*edge_buf.get_unchecked(j), *dbuf.get_unchecked(j)) };
                            if nv { pool.insert_dedup(nb, d); } else if vh & 1 != 0 { pool.insert_bl(nb, d); } else { pool.insert(nb, d); }
                        }
                        base = end;
                    }
                }
                continue;
            }
            if N4_STATS.load(std::sync::atomic::Ordering::Relaxed) { N4_CNT.with(|c| c.borrow_mut()[2] += edge_size as u64); }
            // 第二遍：SQ4 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size && !z_pf {
                    let v = edge_buf[i + po] as usize;
                    let ptr = sq4.code(v).as_ptr() as *const i8;
                    unsafe {
                        crate::arch::prefetch_l1(ptr);
                    }
                }
                let neighbor = edge_buf[i];
                let d = if z_kz || z_kl {
                    let rd = unsafe { *z_qdist.add(z_di) }; z_di += 1;
                    if z_kl { let cp = unsafe { sq4.code_unchecked(neighbor as usize) };
                        let b = unsafe { (*cp.as_ptr() ^ *cp.as_ptr().add(cp.len() - 1)) as u32 }; rd + f32::from_bits(b & cm_zero) } else { rd }
                } else { unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) } };
                if cm_rec { unsafe { (*cmp).r_dist.push(d); } }
                let d = if orx { orx_dist(neighbor).unwrap_or(d) } else { d };
                if cfg!(feature = "diag") && mindist_lambda > 0.0 {
                    // H129 oracle：池键 = 到节点邻域球（半径 λ·r_w）的 MINDIST²（R-tree 最佳优先 kNN 的节点排序迁移到图）
                    let r = unsafe { *mindist_r.add(neighbor as usize) } * mindist_lambda;
                    let sd_ = d.sqrt() - r;
                    pool.insert(neighbor, if sd_ > 0.0 { sd_ * sd_ } else { 0.0 });
                } else if z_th {
                    z_sink ^= d.to_bits(); // R27 ZA 8：不逐码调 insert（保留核输出的数据依赖）
                } else {
                    let ins = if nv { pool.insert_dedup(neighbor, d) } else { pool.insert(neighbor, d) };
                    if cm_rec && ins { unsafe { (*cmp).acc.push(neighbor); (*cmp).r_acc2.push((neighbor, d)); } }
                }
                if a1 { A1_TRACE.with(|t| { let mut t = t.borrow_mut(); t.push(neighbor); t.push(d.to_bits()); }); }
            }
            if z_th && !z_pi { // R27 ZA 8：跳末只插入记录的本跳入池者（无门槛分支、无被拒者调用）
                let c = unsafe { &*cmp }; let (a, b) = (c.q_aoff[cm_hi - 1] as usize, c.q_aoff[cm_hi] as usize);
                for &(v, d) in &c.q_acc2[a..b] { pool.insert(v, d); }
            }
            // R25 CM：每跳额外 cm_extra 个独立随机取码 + 距离（地址只依赖本跳 node，不在池依赖链上；结果丢弃）
            if cm && cm_extra > 0 && !cm_rec {
                let n = sq4.n as u32; let base = cm_mix(node ^ cm_seed);
                for j in 0..(cm_extra & 0xFFFF) {
                    if j + po as u32 > j { let v2 = cm_mix(base.wrapping_add(j + po as u32)) % n; unsafe { crate::arch::prefetch_l1(sq4.code(v2 as usize).as_ptr() as *const i8); } }
                    let v = cm_mix(base.wrapping_add(j)) % n;
                    cm_sink += unsafe { sq4.distance_raw_unchecked(query_code, v as usize) };
                }
            }
        }
        if cm { unsafe { (*cmp).sink += cm_sink + f32::from_bits(z_sink & 0x3F00_0000); } }
        if z_pi { pool.load_sorted(unsafe { &(*cmp).q_fin }); }
        if cm_rec { let c = unsafe { &mut *cmp }; c.r_fin = pool.to_sorted_vec(); let a = c.r_acc2.len() as u32; c.r_aoff.push(a); }
        if a1 { A1_TRACE.with(|t| t.borrow_mut().push(A1_END)); }
        if N4_STATS.load(std::sync::atomic::Ordering::Relaxed) { N4_CNT.with(|c| { let mut c = c.borrow_mut(); c[0] += n4_hops as u64; c[3] += 1; }); }
    }

    /// H07：带 Layer 0 剪枝选项的多种子 SQ4 图遍历（见 [`L0Opts`]）。语义同 `greedy_search_sq4_seeded`。
    #[allow(clippy::too_many_arguments)]
    pub fn greedy_search_sq4_seeded_l0(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        seeds: &[(u32, f32)],
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
        opts: &L0Opts,
    ) -> Vec<(u32, f32)> {
        Self::greedy_search_sq4_seeded_l0_nv(sq4, storage, seeds, query_code, l, visited, pool, po, search_degree, opts, false)
    }

    /// R27 S1：同 [`greedy_search_sq4_seeded_l0`]，`novis = true` 时 L0 不用 visited（前提同 `greedy_search_sq4_seeded_nv`）
    #[allow(clippy::too_many_arguments)]
    pub fn greedy_search_sq4_seeded_l0_nv(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        seeds: &[(u32, f32)],
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
        opts: &L0Opts,
        novis: bool,
    ) -> Vec<(u32, f32)> {
        pool.reset_for(l);
        for &(v, d) in seeds {
            pool.insert(v, d);
        }
        Self::sq4_expand_l0(sq4, storage, query_code, visited, pool, po, search_degree, opts, novis);
        if YT_MODE.load(std::sync::atomic::Ordering::Relaxed) { yt_finish(pool); }
        pool.to_sorted_vec()
    }

    /// H07 主循环：与 `sq4_expand` 相同的两遍预取结构，增加
    /// (a) 提前放弃：池满时以 `worst × scale[c]` 为第 c 检查点阈值的分段 SQ4 距离；
    /// (c) H12 早停：top-`es_k` 第 k 名距离连续 `es_patience` 跳未改善则停止扩展。
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn sq4_expand_l0(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        query_code: &[u8],
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
        opts: &L0Opts,
        novis: bool,
    ) {
        // H80：预取距离覆盖（0 = 用调用方 po）
        let po = { let o = PO_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed); if o > 0 { o } else { po } };
        let mut edge_buf: [u32; 128] = [0; 128];
        let mut thr = [f32::MAX; 8];
        let mut thr_w = f32::NAN;
        let ea = opts.early_abandon;
        // H35：查询预拆分（仅 aarch64 + 16B 分段 EA 路径）
        #[cfg(target_arch = "aarch64")]
        let qsplit = ea && !opts.gran16 && crate::quant::sq4::qsplit_enabled();
        #[cfg(target_arch = "aarch64")]
        let mut qs_buf: Vec<u8> = Vec::new();
        #[cfg(target_arch = "aarch64")]
        if qsplit { crate::quant::sq4::split_query(query_code, &mut qs_buf); }
        // H14：阈值数组长 8，只允许前 8 个检查点（高维时 check_mask 高位无意义）
        let check_mask = opts.check_mask & 0xFF;
        let es = opts.es_k > 0 && opts.es_patience > 0;
        let mut stale = 0usize;

        // H13a 前瞻流水线（环境变量 RAVEN_LA=N，N>0 开启）：对下一个候选（其邻接已在上一跳预取）
        // 提前一跳预取其前 N 个邻居的 SQ4 码；并预取第二候选的邻接，形成两级流水
        let la = crate::graph::vamana::la_depth();
        let h37 = H37_CACHE.load(std::sync::atomic::Ordering::Relaxed);
        let mut adj_ready: [u32; 2] = [u32::MAX; 2];
        let mut prof_peek: u32 = u32::MAX;
        let rad_sd = RAD_SD.load(std::sync::atomic::Ordering::Relaxed);
        let rad_r = RAD_R.load(std::sync::atomic::Ordering::Relaxed);
        // H121：跳序自适应度数——前 hsd_h 跳只看前 hsd_sd 个邻居（0 = 关）
        let hsd = HSD_CFG.load(std::sync::atomic::Ordering::Relaxed);
        let (hsd_h, hsd_sd) = ((hsd >> 16) as usize, (hsd & 0xFFFF) as usize);
        let mut hop_i: usize = 0;
        // H52：码预取距离与 visited 预取距离解耦（RAVEN_CPF；0 = 跟随 po）
        let pc = { let c = CODE_PF.load(std::sync::atomic::Ordering::Relaxed); if c == 0 { po } else { c } };
        let itp = ITP_MODE.load(std::sync::atomic::Ordering::Relaxed);
        let itp_lines = ((if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() }) * 4).div_ceil(64);
        let sg = opts.stop_gamma;
        let mut nhop = 0usize;
        let simple = SIMPLE_MODE.load(std::sync::atomic::Ordering::Relaxed);
        if (PAIR_MODE.load(std::sync::atomic::Ordering::Relaxed) || simple) && sg <= 0.0 && !es {
            Self::sq4_expand_l0_pair(sq4, storage, query_code, visited, pool, po, search_degree, opts, &mut thr, &mut thr_w, check_mask, h37, !simple, novis);
            return;
        }
        let defer = DEFER_MODE.load(std::sync::atomic::Ordering::Relaxed);
        let inl = INL_MODE.load(std::sync::atomic::Ordering::Relaxed);
        let inl_base = INL_BASE.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let inl_sd = INL_SD.load(std::sync::atomic::Ordering::Relaxed);
        let cbz = sq4.code_bytes;
        let inl_blk = inl_sd * cbz;
        let mut slot_buf: [u8; 128] = [0; 128];
        let la = if inl > 0 { 0 } else { la };
        let yt = YT_MODE.load(std::sync::atomic::Ordering::Relaxed);
        let mut ready: u32 = u32::MAX;
        // 第 10 轮：H63/H64 诊断分支改为编译开关（feature diag）；默认构建中 el_tau=0、dk_need=0 为常量，分支被消除
        let dir_k = if cfg!(feature = "diag") { DIR_K.load(std::sync::atomic::Ordering::Relaxed) } else { 0 };
        let hsel_rec = HSEL_REC_PTR.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let hsel_k = if hsel_rec.is_null() { 0 } else { HSEL_K.load(std::sync::atomic::Ordering::Relaxed) };
        let hsel_lut: *const f32 = HSEL_LUT.with(|l| l.get() as *const f32);
        let dir_mode = DIR_MODE.load(std::sync::atomic::Ordering::Relaxed);
        let el_tau = if cfg!(feature = "diag") { f32::from_bits(EL_TAU.load(std::sync::atomic::Ordering::Relaxed)) } else { 0.0 };
        let el_base = EL_BASE.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let el_lut = &*EL_LUT;
        let el_sd = EL_SD.load(std::sync::atomic::Ordering::Relaxed);
        let (dk_k0, dk_k1, dk_need, dk_r) = { let v = if cfg!(feature = "diag") { DK_CFG.load(std::sync::atomic::Ordering::Relaxed) } else { 0 }; ((v & 0xFF) as usize, ((v >> 8) & 0xFF) as usize, ((v >> 16) & 0xFF) as u8, (v >> 24) as usize) };
        // R27 S1：无 visited 仅在精确路径生效（选择/准入类诊断旋钮会撤销 visited → 回退）
        let nv = storage.r_max() <= 128 && novis && dir_k == 0 && hsel_k == 0 && el_tau == 0.0 && dk_need == 0 && !defer;
        let nvrf = NvRf::begin(nv);
        loop {
            // H50 延迟感知扩展顺序：优先扩展上一跳已预取邻接的节点（若仍在池内未扩展），新插队的更优者推迟一跳
            let pop_rank = pool.cursor();
            let popped = if defer && ready != u32::MAX { pool.pop_id(ready).or_else(|| pool.pop()) } else { pool.pop() };
            let Some((node, _dist)) = popped else { break };
            pop_rec(node);
            if sg > 0.0 {
                nhop += 1;
                if nhop > opts.stop_min && _dist > sg * pool.kth_distance(opts.stop_k)
                    && (opts.es_patience == 0 || stale >= opts.es_patience) { break; }
            }
            crate::diag_count!(HOPS);
            if yt { let w = pool.worst_distance(); YT.with(|t| { let mut t = t.borrow_mut(); t.popped.push(node); t.meta.push((node, _dist, w, pop_rank as u32)); }); }
            let yt_full = pool.is_full();
            let yt_start = if yt { YT.with(|t| t.borrow().ins.len()) } else { 0 };
            let _h0 = crate::counters::prof::tick();
            if cfg!(feature = "diag_prof") { crate::counters::prof::add(7, 1); if node != prof_peek { crate::counters::prof::add(6, 1); } }
            if la > 0 {
                let mut nr: [u32; 2] = [u32::MAX; 2];
                if let Some((c, _)) = pool.peek_unchecked() {
                    nr[0] = c;
                    if c == adj_ready[0] || c == adj_ready[1] {
                        let nb = storage.neighbors(c);
                        let lim = la.min(nb.len()).min(if search_degree > 0 { search_degree } else { usize::MAX });
                        for &v in &nb[..lim] {
                            unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); }
                        }
                    }
                }
                if let Some(c2) = pool.peek_second_unchecked() {
                    nr[1] = c2;
                    let ptr = storage.main_block().as_ptr().wrapping_add(c2 as usize * storage.r_max()) as *const i8;
                    let effective_r = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                    unsafe { for l in 0..(effective_r * 4).div_ceil(64) { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
                }
                adj_ready = nr;
            }
            if let Some((next_node, _)) = pool.peek_unchecked() {
                if cfg!(feature = "diag_prof") { prof_peek = next_node; }
                ready = next_node;
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                let effective_r = if search_degree > 0 {
                    search_degree.min(storage.r_max())
                } else {
                    storage.r_max()
                };
                if el_tau > 0.0 { unsafe { crate::arch::prefetch_l1(el_base.wrapping_add(next_node as usize * el_sd) as *const i8); } }
                let graph_lines = (effective_r * 4).div_ceil(64);
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    for l in 1..graph_lines {
                        crate::arch::prefetch_l1(ptr.add(l * 64));
                    }
                }
                if inl == 1 {
                    // H56：下一候选的内嵌码块整块预取（顺序块读）
                    let bp = inl_base.wrapping_add(next_node as usize * inl_blk) as *const i8;
                    let lines = (effective_r.min(inl_sd) * cbz).div_ceil(64);
                    unsafe { for l in 0..lines { crate::arch::prefetch_l1(bp.add(l * 64)); } }
                }
            }

            let _h1 = crate::counters::prof::tick();
            crate::counters::prof::add(3, _h1 - _h0);
            let neighbors = storage.neighbors(node);
            let mut effective_len = if search_degree > 0 {
                search_degree.min(neighbors.len())
            } else {
                neighbors.len()
            };
            // H54 名次自适应度数：弹出者在池中名次 ≥ rad_r 时（认证阶段的远候选）只看前 rad_sd 个邻居
            if rad_sd > 0 && pop_rank >= rad_r { effective_len = effective_len.min(rad_sd); }
            if hsd_sd > 0 { if hop_i < hsd_h { effective_len = effective_len.min(hsd_sd); } hop_i += 1; }
            if hsel_k > 0 { for &v in neighbors.iter().take(effective_len) { unsafe { crate::arch::prefetch_l1(hsel_rec.add(v as usize * HSEL_REC) as *const i8); } } }
            let pf_vis = if nv { 0 } else { po.min(effective_len) };
            for &v in neighbors.iter().take(pf_vis) {
                unsafe { visited.prefetch(v); }
            }
            let mut edge_size = 0usize;
            if nv {
                // R27 S1：不过滤（重复者在入池时由门槛/去重拒绝）
                let n = effective_len.min(128);
                if nvrf.on() {
                    for (j, &v) in neighbors[..n].iter().enumerate() { unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; *slot_buf.get_unchecked_mut(edge_size) = j as u8; edge_size += (!nvrf.seen(v)) as usize; } }
                } else {
                    edge_buf[..n].copy_from_slice(&neighbors[..n]);
                    for j in 0..n { slot_buf[j] = j as u8; }
                    edge_size = n;
                }
            } else {
            for (i, &v) in neighbors.iter().take(effective_len).enumerate() {
                if edge_size >= 128 {
                    break;
                }
                if i + po < effective_len {
                    unsafe { visited.prefetch(neighbors[i + po]); }
                }
                // H63 门卫准入：名次 ≥ k0 的未访问邻居需累计引用权重 ≥ need 才取码（未达标不标记 visited，留待再次被引用）
                // H64 动态：边长准入（池满且 |u−v|²_SQ4 > τ·worst 时不取码，不标记 visited）
                if el_tau > 0.0 && pool.is_full() && unsafe { *el_lut.get_unchecked(*el_base.add(node as usize * el_sd + i) as usize) } > el_tau * pool.worst_distance() { continue; }
                if dk_need > 0 && i >= dk_k0 && pop_rank >= dk_r && !visited.is_visited(v) {
                    let w = if i < dk_k1 { 2 } else { 1 };
                    if visited.dk_add(v, w) < dk_need { continue; }
                }
                let is_new = unsafe { visited.visit_unchecked_branchless(v) };
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; *slot_buf.get_unchecked_mut(edge_size) = i as u8; }
                edge_size += is_new as usize;
            }
            } // R27 S1 nv else 结束
            // R23 HSEL（L0 路径，x86 默认 l0ea=1 走这里）
            if hsel_k > 0 && edge_size > hsel_k && pool.is_full() {
                edge_size = unsafe { hsel_select(hsel_rec, hsel_lut, &mut edge_buf, edge_size, hsel_k, visited) };
            }
            // H69 oracle：方向准入——池满时只保留 cos(q−u, v−u) 最大的 dir_k 个新邻居（用 SQ4 码算三边，不计入 NTC）
            if dir_k > 0 && (dir_mode == 2 || dir_mode == 3) && edge_size > dir_k && pool.is_full() {
                // H69b（mode 2）：按签名一致位数排序；H69c（mode 3）：真实边长 duv（码算，不计 NTC）+ 签名夹角估计 dqv 排序
                DIR_SIG.with(|ds| {
                    let ds = ds.borrow();
                    let nb = ds.nb;
                    let pu = &ds.pu[node as usize * nb..node as usize * nb + nb];
                    let mut sq: u64 = 0;
                    for b in 0..nb { sq |= ((ds.pq[b] > pu[b]) as u64) << b; }
                    let mask: u64 = if nb == 64 { !0 } else { (1u64 << nb) - 1 };
                    let base = node as usize * ds.sd;
                    let (cu, dqu) = if dir_mode == 3 { let c = sq4.code(node as usize); (c, crate::quant::sq4::l2_sq4_raw(query_code, c).max(1e-6)) } else { (sq4.code(node as usize), 0.0) };
                    let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                    for j in 0..edge_size {
                        let sg = unsafe { *ds.sig.get_unchecked(base + slot_buf[j] as usize) };
                        let agree = ((!(sg ^ sq)) & mask).count_ones() as f32;
                        sc[j].1 = edge_buf[j];
                        if dir_mode == 2 { sc[j].0 = -(agree * 256.0 + 255.0 - slot_buf[j] as f32); }
                        else {
                            let duv = crate::quant::sq4::l2_sq4_raw(cu, sq4.code(edge_buf[j] as usize));
                            let cs = (std::f32::consts::PI * (1.0 - agree / nb as f32)).cos();
                            sc[j].0 = dqu + duv - 2.0 * (dqu * duv).sqrt() * cs;
                        }
                    }
                    if dir_mode == 3 { crate::diag_count_add!(QUANT_DIST, 0u64.wrapping_sub(1 + edge_size as u64)); }
                    sc[..edge_size].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    for j in dir_k..edge_size { visited.unvisit(sc[j].1); }
                    for j in 0..dir_k { edge_buf[j] = sc[j].1; }
                });
                edge_size = dir_k;
            }
            if dir_k > 0 && dir_mode == 5 && edge_size > dir_k && pool.is_full() {
                // H70：1-bit RaBitQ（随机 Hadamard 旋转后 sign）排序信号，查询 LUT 16×256；只对前 dir_k 取 SQ4
                RB.with(|rb| {
                    let rb = rb.borrow();
                    let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                    for j in 0..edge_size {
                        let v = edge_buf[j] as usize;
                        let code = unsafe { rb.bits.get_unchecked(v * 16..v * 16 + 16) };
                        let mut dot = 0f32;
                        for t in 0..16 { dot += unsafe { *rb.lut.get_unchecked(t * 256 + code[t] as usize) }; }
                        sc[j] = (rb.nrm2[v] - 2.0 * rb.ratio[v] * dot, edge_buf[j]);
                    }
                    RB_TOUCH.with(|c| c.set(c.get() + edge_size as u64));
                    sc[..edge_size].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    for j in dir_k..edge_size { visited.unvisit(sc[j].1); }
                    for j in 0..dir_k { edge_buf[j] = sc[j].1; }
                });
                edge_size = dir_k;
            }
            if dir_k > 0 && dir_mode == 6 && edge_size > dir_k && pool.is_full() {
                // H70b：PQ（m 子空间 × 256 码字，m 字节/点）ADC 排序信号
                PQ.with(|pq| {
                    let pq = pq.borrow();
                    let m = pq.m;
                    let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                    for j in 0..edge_size {
                        let v = edge_buf[j] as usize;
                        let code = unsafe { pq.codes.get_unchecked(v * m..v * m + m) };
                        let mut d = 0f32;
                        for t in 0..m { d += unsafe { *pq.lut.get_unchecked(t * 256 + code[t] as usize) }; }
                        sc[j] = (d, edge_buf[j]);
                    }
                    sc[..edge_size].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    for j in dir_k..edge_size { visited.unvisit(sc[j].1); }
                    for j in 0..dir_k { edge_buf[j] = sc[j].1; }
                });
                edge_size = dir_k;
            }
            if dir_k > 0 && dir_mode == 4 && edge_size > dir_k && pool.is_full() {
                // oracle：只按 dqv 排序（检验方向是否有独立价值）
                let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                for j in 0..edge_size { sc[j] = (crate::quant::sq4::l2_sq4_raw(query_code, sq4.code(edge_buf[j] as usize)), edge_buf[j]); }
                crate::diag_count_add!(QUANT_DIST, 0u64.wrapping_sub(edge_size as u64));
                sc[..edge_size].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for j in dir_k..edge_size { visited.unvisit(sc[j].1); }
                for j in 0..dir_k { edge_buf[j] = sc[j].1; }
                edge_size = dir_k;
            }
            if dir_k > 0 && dir_mode == 7 && edge_size > dir_k && pool.is_full() {
                // R23 跳内相对排序（hop-relative ranking）：每边存残差投影 P(v−u)（int8），按 ‖r_v‖² − 2⟨P(q−u), P r_v⟩ 取前 dir_k
                edge_size = hr_select(node, &mut edge_buf, &slot_buf, edge_size, dir_k, visited);
            }
            if dir_k > 0 && dir_mode == 1 && edge_size > dir_k && pool.is_full() {
                let cu = sq4.code(node as usize);
                let dqu = crate::quant::sq4::l2_sq4_raw(query_code, cu).max(1e-6);
                let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                for j in 0..edge_size {
                    let v = edge_buf[j];
                    let cv = sq4.code(v as usize);
                    let duv = crate::quant::sq4::l2_sq4_raw(cu, cv).max(1e-6);
                    let dqv = crate::quant::sq4::l2_sq4_raw(query_code, cv);
                    sc[j] = ((dqu + duv - dqv) / (2.0 * (dqu * duv).sqrt()), v);
                }
                // oracle 的 1+2·edge_size 次 SQ4 距离不计入 NTC
                crate::diag_count_add!(QUANT_DIST, 0u64.wrapping_sub(1 + 2 * edge_size as u64));
                sc[..edge_size].sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for j in dir_k..edge_size { visited.unvisit(sc[j].1); }
                for j in 0..dir_k { edge_buf[j] = sc[j].1; }
                DIR_REJ.with(|c| c.set(c.get() + (edge_size - dir_k) as u64));
                edge_size = dir_k;
            }
            if yt { YT.with(|t| t.borrow_mut().touched.extend_from_slice(&edge_buf[..edge_size])); }
            let blk = inl_base.wrapping_add(node as usize * inl_blk);
            let use_inl = inl > 0 && effective_len <= inl_sd;
            let prefetch_count = pc.min(edge_size);
            if use_inl {
                if inl == 2 { for j in 0..prefetch_count { unsafe { crate::arch::prefetch_l1(blk.add(slot_buf[j] as usize * cbz) as *const i8); } } }
            } else {
            for &node in edge_buf.iter().take(prefetch_count) {
                let ptr = sq4.code(node as usize).as_ptr() as *const i8;
                unsafe { crate::arch::prefetch_l1(ptr); }
            }
            }

            let _h2 = crate::counters::prof::tick();
            crate::counters::prof::add(4, _h2 - _h1);
            let kth_before = if es { pool.kth_distance(opts.es_k) } else { 0.0 };
            // H48 插入触发邻接预取：新插入者若成为"下一个弹出"，立刻预取其邻接（原 peek 预取会失效）
            let mut itp_best = if itp { pool.peek_unchecked().map(|x| x.1).unwrap_or(f32::MAX) } else { f32::MIN };
            for i in 0..edge_size {
                if i + pc < edge_size {
                    if use_inl {
                        if inl == 2 { unsafe { crate::arch::prefetch_l1(blk.add(slot_buf[i + pc] as usize * cbz) as *const i8); } }
                    } else {
                    let v = edge_buf[i + pc] as usize;
                    let ptr = sq4.code(v).as_ptr() as *const i8;
                    unsafe { crate::arch::prefetch_l1(ptr); }
                    }
                }
                let neighbor = edge_buf[i];
                let code: &[u8] = if use_inl { unsafe { std::slice::from_raw_parts(blk.add(slot_buf[i] as usize * cbz), cbz) } } else { unsafe { sq4.code_unchecked(neighbor as usize) } };
                if ea && pool.is_full() {
                    let w = pool.worst_distance();
                    // H37：仅当池最差距离变化时重算阈值（逐位等价）
                    if w != thr_w || !h37 {
                        thr_w = w;
                        for c in 0..8 {
                            thr[c] = w * opts.scale[c];
                        }
                    }
                    /* H56 code from above */
                    #[cfg(target_arch = "aarch64")]
                    let split_hit = qsplit;
                    #[cfg(not(target_arch = "aarch64"))]
                    let split_hit = false;
                    let (d, dims) = if split_hit {
                        #[cfg(target_arch = "aarch64")]
                        { unsafe { crate::quant::sq4::l2_sq4_bounded_split_neon(&qs_buf, code, &thr, check_mask) } }
                        #[cfg(not(target_arch = "aarch64"))]
                        { unreachable!() }
                    } else if opts.gran16 {
                        crate::quant::sq4::l2_sq4_raw_bounded16(query_code, code, &thr, check_mask)
                    } else {
                        crate::quant::sq4::l2_sq4_raw_bounded(query_code, code, &thr, check_mask)
                    };
                    if (dims as usize) < sq4.dim {
                        continue; // 部分和已超阈值：精确模式下必被 pool 拒绝
                    }
                    let ok = if nv { pool.insert_dedup(neighbor, d) } else { pool.insert(neighbor, d) };
                    if yt && ok { YT.with(|t| t.borrow_mut().ins.push(neighbor)); }
                    if ok && d < itp_best {
                        itp_best = d;
                        let ptr = storage.main_block().as_ptr().wrapping_add(neighbor as usize * storage.r_max()) as *const i8;
                        unsafe { for l in 0..itp_lines { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
                    }
                } else {
                    let d = crate::quant::sq4::l2_sq4_raw(query_code, code);
                    let ok = if nv { pool.insert_dedup(neighbor, d) } else { pool.insert(neighbor, d) };
                    if yt && ok { YT.with(|t| t.borrow_mut().ins.push(neighbor)); }
                    if ok && d < itp_best {
                        itp_best = d;
                        let ptr = storage.main_block().as_ptr().wrapping_add(neighbor as usize * storage.r_max()) as *const i8;
                        unsafe { for l in 0..itp_lines { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
                    }
                }
            }
            if yt { YT.with(|t| { let mut t = t.borrow_mut(); let l = t.ins.len() - yt_start; t.hops.push((edge_size as u32, yt_start as u32, l as u32, yt_full)); }); }
            if defer && la > 0 && ready != u32::MAX {
                // 下一跳必扩展 ready（除非被挤出）：其邻接已在本跳开始时预取，此处预取其前 la 个邻居码
                let nb = storage.neighbors(ready);
                let lim = la.min(nb.len()).min(if search_degree > 0 { search_degree } else { usize::MAX });
                for &v in &nb[..lim] { unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); } }
            }
            crate::counters::prof::add(5, crate::counters::prof::tick() - _h2);
            #[cfg(feature = "diag_counters")]
            {
                let _ = (kth_before,);
                crate::graph::stoptrace::on_hop(pool, edge_size as u32, 0);
            }
            if es {
                if pool.kth_distance(opts.es_k) < kth_before {
                    stale = 0;
                } else {
                    stale += 1;
                    if sg <= 0.0 && stale >= opts.es_patience {
                        break;
                    }
                }
            }
        }
    }

    /// H45 单查询双链交错：每轮弹出池内最好的两个未扩展候选（两条互不依赖的扩展链），
    /// 合并两者的未访问邻居后一次性流水取码，使两条链的随机访存在同一时间窗重叠；共享 visited 与结果池。
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn sq4_expand_l0_pair(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        query_code: &[u8],
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
        opts: &L0Opts,
        thr: &mut [f32; 8],
        thr_w: &mut f32,
        check_mask: u32,
        h37: bool,
        dual: bool,
        nv: bool,
    ) {
        let mut edge_buf: [u32; 256] = [0; 256];
        let ea = opts.early_abandon;
        let effective_r = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
        let graph_lines = (effective_r * 4).div_ceil(64);
        let pf_adj = |c: u32| {
            let ptr = storage.main_block().as_ptr().wrapping_add(c as usize * storage.r_max()) as *const i8;
            unsafe { for l in 0..graph_lines { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
        };
        // X3（x86 线）：H136 早码预取移植到 simple/pair 路径（RAVEN_VH 位 8/16/32；x86 默认 0）。只加预取，结果逐位一致
        let vh = vec_hop();
        let n_epf_cfg = if vh & 32 != 0 { usize::MAX } else if vh & 16 != 0 { 16 } else if vh & 8 != 0 { 8 } else { 0 };
        let nvrf = NvRf::begin(nv && storage.r_max() <= 128);
        let nv = nv && storage.r_max() <= 128;
        loop {
            let Some((n1, _)) = pool.pop() else { break };
            pop_rec(n1);
            let n2 = if dual { pool.pop().map(|x| x.0) } else { None };
            if let Some(m) = n2 { pop_rec(m); }
            crate::diag_count!(HOPS);
            if n2.is_some() { crate::diag_count!(HOPS); }
            // 预取下一对候选的邻接（在本轮距离计算期间到达）
            if let Some((c, _)) = pool.peek_unchecked() { pf_adj(c); }
            if dual { if let Some(c2) = pool.peek_second_unchecked() { pf_adj(c2); } }
            let mut edge_size = 0usize;
            for (ci, node) in [Some(n1), n2].into_iter().enumerate() {
                let Some(node) = node else { continue };
                let neighbors = storage.neighbors(node);
                let el = if search_degree > 0 { search_degree.min(neighbors.len()) } else { neighbors.len() };
                if n_epf_cfg > 0 { for &v in neighbors.iter().take(n_epf_cfg.min(el)) { unsafe { crate::arch::prefetch_l1(sq4.code_unchecked(v as usize).as_ptr() as *const i8); } } }
                if nv {
                    // R27 S1：不过滤、不碰 visited（重复者在入池时由门槛/去重拒绝）
                    let n = el.min(256 - edge_size);
                    edge_size += nvrf.fill(&neighbors[..n], &mut edge_buf[edge_size..]);
                    continue;
                }
                if ci == 0 { for &v in neighbors.iter().take(po.min(el)) { unsafe { visited.prefetch(v); } } }
                if ci == 0 { if let Some(m) = n2 { let nb2 = storage.neighbors(m); for &v in nb2.iter().take(po.min(nb2.len())) { unsafe { visited.prefetch(v); } } } }
                for (i, &v) in neighbors.iter().take(el).enumerate() {
                    if edge_size >= 256 { break; }
                    if i + po < el { unsafe { visited.prefetch(neighbors[i + po]); } }
                    let is_new = unsafe { visited.visit_unchecked_branchless(v) };
                    unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                    edge_size += is_new as usize;
                }
            }
            // H87：码跨行时（如 GloVe 50B）补发末字节所在行
            let pf2 = CODE_PF2.load(std::sync::atomic::Ordering::Relaxed);
            let cbm1 = sq4.code_bytes - 1;
            let pfc = |v: u32| unsafe {
                let p = sq4.code_unchecked(v as usize).as_ptr();
                crate::arch::prefetch_l1(p as *const i8);
                if pf2 && ((p as usize) & 63) + cbm1 >= 64 { crate::arch::prefetch_l1(p.add(cbm1) as *const i8); }
            };
            for &node in edge_buf.iter().take(po.min(edge_size)) { pfc(node); }
            for i in 0..edge_size {
                if i + po < edge_size { pfc(edge_buf[i + po]); }
                let neighbor = edge_buf[i];
                if ea && pool.is_full() {
                    let w = pool.worst_distance();
                    if w != *thr_w || !h37 {
                        *thr_w = w;
                        for c in 0..8 { thr[c] = w * opts.scale[c]; }
                    }
                    let code = unsafe { sq4.code_unchecked(neighbor as usize) };
                    let (d, dims) = if opts.gran16 {
                        crate::quant::sq4::l2_sq4_raw_bounded16(query_code, code, thr, check_mask)
                    } else {
                        crate::quant::sq4::l2_sq4_raw_bounded(query_code, code, thr, check_mask)
                    };
                    if (dims as usize) < sq4.dim { continue; }
                    if nv { pool.insert_dedup(neighbor, d); } else { pool.insert(neighbor, d); }
                } else {
                    let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
                    if nv { pool.insert_dedup(neighbor, d); } else { pool.insert(neighbor, d); }
                }
            }
        }
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
                    crate::arch::prefetch_l1(ptr);
                    crate::arch::prefetch_l1(ptr.add(64));
                    crate::arch::prefetch_l1(ptr.add(128));
                    crate::arch::prefetch_l1(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居（branchless — A8）
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                let is_new = unsafe { visited.visit_unchecked_branchless(v) };
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += is_new as usize;
            }

            // 预取前 po 个邻居的 PQ4 码（M/2 bytes，比 f32 小 16x）
            let prefetch_count = po.min(edge_size);
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = pq4.code(v).as_ptr() as *const i8;
                unsafe { crate::arch::prefetch_l1(ptr); }
            }

            // 第二遍：PQ4 ADC 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = pq4.code(v).as_ptr() as *const i8;
                    unsafe { crate::arch::prefetch_l1(ptr); }
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
                    crate::arch::prefetch_l1(ptr);
                    crate::arch::prefetch_l1(ptr.add(64));
                    crate::arch::prefetch_l1(ptr.add(128));
                    crate::arch::prefetch_l1(ptr.add(192));
                }
            }

            let neighbors = storage.neighbors(node);

            // 第一遍：收集未访问邻居（branchless — A8）
            let mut edge_size = 0usize;
            for &v in neighbors {
                if edge_size >= 128 {
                    break;
                }
                let is_new = unsafe { visited.visit_unchecked_branchless(v) };
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += is_new as usize;
            }

            // 预取前 po 个邻居的 PQ8 码（M bytes，比 f32 小 16x）
            let prefetch_count = po.min(edge_size);
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = pq8.code(v).as_ptr() as *const i8;
                unsafe { crate::arch::prefetch_l1(ptr); }
            }

            // 第二遍：PQ8 ADC 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size {
                    let v = edge_buf[i + po] as usize;
                    let ptr = pq8.code(v).as_ptr() as *const i8;
                    unsafe { crate::arch::prefetch_l1(ptr); }
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

    /// 对所有节点的邻居列表按距离排序（in-place）
    ///
    /// `final_prune` 只处理度数 > r_max 的节点。
    /// 度数 ≤ r_max 的节点保留了反向边追加的原始顺序（未排序）。
    ///
    /// 搜索时度数上限（search_degree）要求邻居列表按距离升序排列，
    /// 否则 `.take(D)` 取的是任意 D 个邻居而非最近的 D 个。
    ///
    /// 本函数对所有节点做 in-place 排序，不改变邻居集合，只改变顺序。
    fn sort_all_neighbors(storage: &mut HybridBlockedCsr, vectors: &[f32], dim: usize) {
        for node in 0..storage.len() as u32 {
            let (main, overflow) = storage.neighbors_full(node);
            if overflow.is_empty() && main.len() <= 1 {
                continue;
            }
            let mut all: Vec<u32> = main.to_vec();
            all.extend_from_slice(overflow);
            let node_vec = &vectors[node as usize * dim..(node as usize + 1) * dim];
            all.sort_by(|&a, &b| {
                let da = l2_simd(node_vec, &vectors[a as usize * dim..(a as usize + 1) * dim]);
                let db = l2_simd(node_vec, &vectors[b as usize * dim..(b as usize + 1) * dim]);
                da.total_cmp(&db)
            });
            storage.set_neighbors(node, &all);
        }
    }

    /// QA-aware final prune
    fn final_prune_qa<F>(
        storage: &mut HybridBlockedCsr,
        vectors: &[f32],
        dim: usize,
        qa_config: &QuantAwarePruneConfig,
        error_fn: &F,
    ) where
        F: Fn(u32, u32) -> f32 + Sync,
    {
        for node in 0..storage.len() as u32 {
            let (main, overflow) = storage.neighbors_full(node);
            if main.len() + overflow.len() <= qa_config.r_max {
                continue;
            }
            let mut all: Vec<u32> = main.to_vec();
            all.extend_from_slice(overflow);
            let pruned = QuantAwareRobustPrune::prune(
                &all, node, vectors, dim, error_fn, qa_config,
            );
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
    /// R26 H149 / R28：饱和填充——每节点保留原邻接，再用本图 f32 搜索（ef）得到的近似 kNN 中最近的未连接点补满到 `d` 槽，
    /// 行按到本节点的距离升序（供 search_degree 截断）。返回的新图不含上层导航（pipeline::prepare 会重建）。
    /// `base` 为建图所用向量（angular 数据已归一化，L2 序 = 角度序）。
    pub fn saturate_fill(&self, base: &[f32], dim: usize, d: usize, ef: usize) -> VamanaGraph {
        use rayon::prelude::*;
        let n = base.len() / dim;
        let l2 = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
        let rows: Vec<Vec<u32>> = (0..n).into_par_iter().map_init(|| GraphSearcher::new(base, self, ef), |s, u| {
            let x = &base[u * dim..(u + 1) * dim];
            let mut row: Vec<(f32, u32)> = self.storage().neighbors(u as u32).iter().filter(|&&v| v != crate::SENTINEL && v as usize != u)
                .map(|&v| (l2(x, &base[v as usize * dim..(v as usize + 1) * dim]), v)).collect();
            if row.len() < d {
                let have: std::collections::HashSet<u32> = row.iter().map(|e| e.1).collect();
                let r = s.search(x, ef);
                for &(v, _) in r.iter() { if row.len() >= d { break; } if v as usize != u && !have.contains(&v) { row.push((l2(x, &base[v as usize * dim..(v as usize + 1) * dim]), v)); } }
            }
            row.sort_by(|a, b| a.0.total_cmp(&b.0)); row.truncate(d);
            row.into_iter().map(|e| e.1).collect()
        }).collect();
        let rm = d.max(self.storage().r_max());
        let mut st = HybridBlockedCsr::new(n, rm);
        for (u, r) in rows.iter().enumerate() { for &v in r { st.add_edge(u as u32, v); } }
        VamanaGraph::from_storage(st, self.entry_point(), dim)
    }

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

    /// 设置分层导航（用于 BFS 重排后重新构建导航层）
    pub fn set_layered_nav(&mut self, nav: LayeredNavigation) {
        self.layered_nav = Some(nav);
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
        let metadata_bytes: Vec<u8> = self.metadata.as_ref()
            .map_or_else(Vec::new, |m| m.to_toml().unwrap_or_default().into_bytes());

        // 序列化 layered nav（如果有）
        let nav_bytes: Vec<u8> = self.layered_nav.as_ref()
            .map_or_else(Vec::new, super::navigation::LayeredNavigation::serialize_binary);

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
            body.extend(std::iter::repeat_n(0u8, padding));
        }

        // layered nav trailer（设计文档：保留随机层级）
        let has_nav = !nav_bytes.is_empty();
        if has_nav {
            body.extend_from_slice(&(nav_bytes.len() as u32).to_le_bytes());
            body.extend_from_slice(&nav_bytes);
            let padding = (4 - (nav_bytes.len() % 4)) % 4;
            body.extend(std::iter::repeat_n(0u8, padding));
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
        let n = u64::from_le_bytes(body[0..8].try_into()?) as usize;
        let dim = u64::from_le_bytes(body[8..16].try_into()?) as usize;
        let entry_point = u32::from_le_bytes(body[16..20].try_into()?);

        let mut offset = 20usize;

        // 读取 metadata trailer（如果 flags 标记存在）
        let metadata = if header.flags & FLAG_HAS_METADATA != 0 {
            if body.len() < offset + 4 {
                return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "body too short for metadata trailer header",
                )));
            }
            let meta_len = u32::from_le_bytes(body[offset..offset+4].try_into()?) as usize;
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
            let nav_len = u32::from_le_bytes(body[offset..offset+4].try_into()?) as usize;
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
    /// 可选的 SQ4 标量量化数据集（4-bit per dimension）
    /// 启用后 search_sq4() 使用 SQ4 距离进行图遍历 + f32 rerank
    sq4: Option<&'a SQ4Dataset>,
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
    /// 预分配的 SQ8 query code buffer，避免每次搜索堆分配
    query_code_buf: Vec<u8>,
    sq8_adc: crate::quant::sq8::Sq8AdcQuery,
    /// Rerank 倍数（放在 struct 末尾避免干扰热路径 cache 对齐）
    rerank_factor: usize,
    /// 可选的 f16 预量化数据集（用于 f16 rerank，高维场景带宽减半）
    f16: Option<&'a F16Dataset>,
    /// H53：u8 无损 rerank 向量（整数数据集）
    vec_u8: Option<&'a [u8]>,
    /// 搜索时度数上限（0 = 不限制）
    ///
    /// 建图用高 R 保证图质量，搜索时只看前 search_degree 个邻居（已按距离升序排列）。
    /// 这打破了"高 R 必然高 avg_visited"的困局：
    /// - 高 R 图 → 收敛所需跳数少
    /// - 低 search_degree → 每跳距离计算少
    /// - 乘积（跳数 × 每跳计算量）可能低于低 R 不 cap 的方案
    search_degree: usize,
    /// 上层导航搜索宽度（方案 B：Layer 1 beam search）
    ///
    /// 1 = 当前行为（ef=1 贪心下降），>1 = Layer 1 用 ef=upper_ef beam search。
    /// Layer 1 有 ~31K 节点（SIFT1M/M=32），ef=1 容易陷入局部最优。
    upper_ef: usize,
    /// 上层搜索专用 LinearPool（容量 = upper_ef，与 Layer 0 pool 独立）
    upper_pool: LinearPool,
    /// H09 导航层选项（默认 = 旧行为，search_sq4 走原路径）
    nav_opts: NavOpts,
    /// H09 导航轨迹缓冲（复用，零分配）
    nav_trace: Vec<(u32, f32)>,
    /// H07 Layer 0 剪枝选项（默认关闭）
    l0_opts: L0Opts,
    /// H07 SQ4 维度置换（与 `SQ4Dataset::build_permuted` 配套；None = 不置换）
    sq4_perm: Option<&'a [usize]>,
    /// H07 置换后查询缓冲
    query_perm_buf: Vec<f32>,
    /// R33 HR 模式（ef ≥ gate 时由 Prepared::searcher 挂上；None = 原路径，逐位一致）
    cc: Option<crate::graph::cellconfirm::CcActive<'a>>,
    cc_scratch: Option<Box<crate::graph::cellconfirm::CcScratch>>,
}

/// H09：上层导航搜索侧选项（仅 `search_sq4` 生效）。
///
/// 默认值（2026-10-29 H09a 接受后切换）：f32 贪心 + 去重，单入口 —— 返回结果与旧行为逐位一致，
/// 只是少算重复距离（SIFT-1M 导航层 f32 距离 117.8 → 79.4/查询）。
/// `NavOpts::legacy()` = 旧行为（不去重）。SQ4 导航 + 轨迹携带（H09b）为开关，默认关闭，待 x86 验证。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NavOpts {
    /// 去重：贪心下降中已算过距离的节点不再重算（结果与旧行为逐位一致）
    pub dedup: bool,
    /// 上层导航用 SQ4 距离代替 f32（隐含去重）
    pub sq4: bool,
    /// 把导航轨迹中 SQ4 距离最小的前 carry 个节点作为 Layer 0 种子（≤1 = 单入口）。
    /// sq4=true 时零额外距离；sq4=false 时需对种子补算 SQ4 距离。
    pub carry: usize,
    /// 下降停止层（1 = 完整下降；2 = 跳过 Layer 1，直接从 Layer 2 结果进入 Layer 0）
    pub stop_level: u8,
}

impl Default for NavOpts {
    fn default() -> Self {
        Self { dedup: true, sq4: false, carry: 1, stop_level: 1 }
    }
}

impl NavOpts {
    /// 旧行为（H09 之前）：f32 贪心，不去重，单入口
    pub fn legacy() -> Self {
        Self { dedup: false, sq4: false, carry: 1, stop_level: 1 }
    }

    /// 是否为旧行为
    pub fn is_legacy(&self) -> bool {
        *self == Self::legacy()
    }

    /// H09b 配置：SQ4 导航 + 全部轨迹作为 Layer 0 种子
    pub fn sq4_carry() -> Self {
        Self { dedup: true, sq4: true, carry: usize::MAX, stop_level: 1 }
    }
}

/// H07：Layer 0（SQ4 遍历）搜索侧剪枝选项。仅在 `search_sq4` 的 H09 导航路径
/// （`NavOpts::sq4_carry()`）中生效；默认全部关闭 = 行为不变。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L0Opts {
    /// (a) SQ4 分段提前放弃
    pub early_abandon: bool,
    /// 第 c 个检查点（32·(c+1) 维后）阈值 = 池最差距离 × scale[c]。
    /// 全 1.0 = 精确下界剪枝（结果与关闭时逐位一致）；<1.0 = ADSampling 式有损估计剪枝。
    pub scale: [f32; 8],
    /// 启用哪些检查点（bit c = 第 c 段后检查）
    pub check_mask: u32,
    /// 分段粒度：false = 32 维（16 字节），true = 16 维（8 字节）
    pub gran16: bool,
    /// (c) H12 早停：跟踪第 es_k 名距离（0 = 关闭）
    pub es_k: usize,
    /// (c) 连续多少跳未改善即停止
    pub es_patience: usize,
    /// H46 距离比终止：弹出候选距离 > stop_gamma × 池第 stop_k 名距离 且 hop ≥ stop_min 即停（0 = 关闭）
    pub stop_gamma: f32,
    pub stop_k: usize,
    pub stop_min: usize,
}

impl Default for L0Opts {
    fn default() -> Self {
        Self { early_abandon: false, scale: [1.0; 8], check_mask: 0, gran16: false, es_k: 0, es_patience: 0, stop_gamma: 0.0, stop_k: 10, stop_min: 0 }
    }
}

impl L0Opts {
    /// 是否有任何选项开启
    pub fn is_active(&self) -> bool {
        self.early_abandon || (self.es_k > 0 && self.es_patience > 0) || self.stop_gamma > 0.0
    }

    /// 精确提前放弃（下界剪枝，无损），检查点 = 全部
    pub fn exact() -> Self {
        Self { early_abandon: true, scale: [1.0; 8], check_mask: 0xFF, gran16: false, es_k: 0, es_patience: 0, stop_gamma: 0.0, stop_k: 10, stop_min: 0 }
    }
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
            prefetch_offset: 16,
            sq8: None,
            sq4: None,
            pq4: None,
            pq8: None,
            adaptive_ef: None,
            pool: LinearPool::new(ef_search),
            query_code_buf: vec![0u8; dim],
            sq8_adc: Default::default(),
            rerank_factor: 5,
            f16: None,
            vec_u8: None,
            search_degree: 0,
            upper_ef: 1,
            upper_pool: LinearPool::new(1),
            nav_opts: NavOpts::default(),
            nav_trace: Vec::new(),
            l0_opts: L0Opts::default(),
            sq4_perm: None,
            query_perm_buf: Vec::new(),
            cc: None,
            cc_scratch: None,
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
            prefetch_offset: 16,
            sq8: None,
            sq4: None,
            pq4: None,
            pq8: None,
            adaptive_ef: None,
            pool: LinearPool::new(ef_search),
            query_code_buf: vec![0u8; dim],
            sq8_adc: Default::default(),
            rerank_factor: 5,
            f16: None,
            vec_u8: None,
            search_degree: 0,
            upper_ef: 1,
            upper_pool: LinearPool::new(1),
            nav_opts: NavOpts::default(),
            nav_trace: Vec::new(),
            l0_opts: L0Opts::default(),
            sq4_perm: None,
            query_perm_buf: Vec::new(),
            cc: None,
            cc_scratch: None,
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

    /// 设置 rerank 倍数（SQ8 搜索后 f32 重排序的候选数 = k × factor）
    pub fn with_rerank_factor(&mut self, factor: usize) -> &mut Self {
        self.rerank_factor = factor.max(1);
        self
    }

    /// 设置搜索时度数上限（0 = 不限制，使用全部 R 个邻居）
    ///
    /// 邻居列表已按距离升序排列（DirectionalPrune 保证），
    /// 只看前 D 个 = 只看最近的 D 个邻居。
    ///
    /// 适用场景：建图用高 R（如 64）保证图质量，搜索用低 D（如 32）降低每跳计算量。
    pub fn with_search_degree(&mut self, degree: usize) -> &mut Self {
        self.search_degree = degree;
        self
    }

    /// H09：设置上层导航选项（见 [`NavOpts`]）
    pub fn with_nav_opts(&mut self, opts: NavOpts) -> &mut Self {
        self.nav_opts = opts;
        self
    }

    /// H09：当前上层导航选项
    pub fn nav_opts(&self) -> NavOpts {
        self.nav_opts
    }

    /// H07：设置 Layer 0 剪枝选项（仅 H09 SQ4 导航路径生效）
    pub fn with_l0_opts(&mut self, opts: L0Opts) -> &mut Self {
        self.l0_opts = opts;
        self
    }

    /// H07：设置 SQ4 维度置换（`sq4` 须由 `SQ4Dataset::build_permuted(.., perm)` 构建）
    pub fn with_sq4_perm(&mut self, perm: &'a [usize]) -> &mut Self {
        self.sq4_perm = Some(perm);
        self
    }


    /// 设置上层导航搜索宽度（方案 B：Layer 1 beam search）
    ///
    /// upper_ef=1（默认）= ef=1 贪心下降（当前行为）
    /// upper_ef>1 = Layer 1 用 ef beam search，获得更好的 Layer 0 入口点
    ///
    /// Layer 1（~31K 节点 for SIFT1M/M=32）远大于竞品上层，
    /// ef=1 贪心容易陷入局部最优。beam search 可降低 Layer 0 avg_visited。
    pub fn with_upper_ef(&mut self, upper_ef: usize) -> &mut Self {
        self.upper_ef = upper_ef;
        if upper_ef > 1 {
            self.upper_pool = LinearPool::new(upper_ef);
        }
        self
    }

    /// 启用 f16 预量化 rerank（高维场景带宽减半）
    ///
    /// 启用后 search_sq8 的 rerank 阶段使用 f16 半精度距离代替 f32。
    /// 内存带宽减半（2B/元素 vs 4B/元素），F16C 指令在寄存器中转回 f32 计算。
    ///
    /// 适用场景：维度 ≥ 512 的高维数据集
    /// 低维数据集（SIFT-128）：rerank 数据已在 L1 cache，收益极小
    pub fn with_vec_u8(&mut self, v: &'a [u8]) -> &mut Self {
        self.vec_u8 = Some(v);
        self
    }

    pub fn with_f16(&mut self, f16: &'a F16Dataset) -> &mut Self {
        self.f16 = Some(f16);
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

    /// 启用 SQ4 标量量化搜索（4-bit per dimension）
    ///
    /// 设置后可调用 search_sq4() 使用 SQ4 距离进行图遍历。
    /// SQ4 每维度 4-bit，码大小 dim/2 bytes（SIFT-128: 64B vs SQ8 的 128B）。
    /// 需要预先构建 SQ4Dataset（SQ4Dataset::build）。
    pub fn with_sq4(&mut self, sq4: &'a SQ4Dataset) -> &mut Self {
        self.sq4 = Some(sq4);
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
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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
    fn search_sq8_impl<const RAW: bool>(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let sq8 = self.sq8.expect("search_sq8 requires with_sq8() first");
        let dim = self.dim;

        // 1. 编码查询向量为 SQ8（零分配，复用预分配 buffer）
        sq8.params.encode_into(query, &mut self.query_code_buf);

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

        // 4. SQ8 图遍历（R53：RAVEN_SQ8_ADC=1 走 ADC）
        let candidates = if crate::quant::sq8::sq8_adc_enabled() {
            let mut aq = std::mem::take(&mut self.sq8_adc);
            sq8.adc_prepare(query, &mut aq);
            let c = VamanaGraph::greedy_search_sq8_adc(sq8, self.graph.storage(), entry_point, &aq, ef, &mut self.visited, &mut self.pool, self.prefetch_offset);
            self.sq8_adc = aq; c
        } else { VamanaGraph::greedy_search_sq8::<RAW>(
            sq8,
            self.graph.storage(),
            entry_point,
            &self.query_code_buf,
            ef,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
        ) };

        self.last_visited_count = self.visited.visited_count();

        // 5. 精量化 rerank（Refiner 框架）：candidates 已按 SQ8 距离升序，
        //    SQ8 排序与精确距离高度一致，只需对 top-N rerank 即可覆盖 top-k
        //    f16 模式：带宽减半（高维场景收益显著）
        //    f32 模式：全精度（低维默认）
        if let Some(f16_ds) = self.f16 {
            rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor)
        } else {
            rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor)
        }
    }

    /// SQ8 量化搜索（Phase 1 Step 0）
    ///
    /// 图遍历使用无加权 SQ8 距离（OPT-15：跳过 scale_sq），
    /// 终态对 top-N 候选用 f32 精确距离重排序（rerank）。
    ///
    /// 无加权距离在 SIFT-1M 上实测 recall +0.44pp（0.9680 vs 0.9636），
    /// QPS neutral-to-positive（scale_sq 加权扭曲了距离排序）。
    ///
    /// 返回 (节点ID, f32精确距离) 列表，按距离升序。
    /// 需先调用 with_sq8() 设置 SQ8Dataset。
    #[inline(always)]
    pub fn search_sq8(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_sq8_impl::<true>(query, k)
    }

    /// SQ8 加权量化搜索（OPT-15 之前的原始路径，保留用于对比）
    ///
    /// 图遍历使用加权 SQ8 距离（d = Σ (qa[i] - qb[i])² × scale_sq[i]）。
    #[inline(always)]
    pub fn search_sq8_weighted(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_sq8_impl::<false>(query, k)
    }

    /// H09：带导航层选项的 SQ4 搜索（query_code_buf 已编码）
    fn search_sq4_nav_opts(
        &mut self,
        nav: &LayeredNavigation,
        sq4: &SQ4Dataset,
        query: &[f32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        let dim = self.dim;
        let opts = self.nav_opts;
        let ef = self.ef_search;
        self.last_ef_used = ef;
        let want_trace = opts.carry > 1;
        self.nav_trace.clear();
        self.visited.reset();
        let vectors = self.vectors;
        let code = &self.query_code_buf;
        let use_vis = opts.dedup || opts.sq4;
        let _p0 = crate::counters::prof::tick();
        let navpf = NAV_PF.load(std::sync::atomic::Ordering::Relaxed);
        let cm_nrec = cfg!(feature = "diag_cm") && CM_ON.load(std::sync::atomic::Ordering::Relaxed) && CM.with(|c| c.borrow().rec);
        let (ep, ep_dist) = if opts.sq4 {
            nav.descend_with_pf(
                |v| { if cm_nrec { CM.with(|c| c.borrow_mut().r_nav.push(v)); } unsafe { sq4.distance_raw_unchecked(code, v as usize) } },
                |v| if navpf { unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); } },
                Some(&mut self.visited),
                opts.stop_level,
                if want_trace { Some(&mut self.nav_trace) } else { None },
            )
        } else {
            nav.descend_with(
                |v| { if cm_nrec { CM.with(|c| c.borrow_mut().r_nav.push(v)); } l2_simd(query, &vectors[v as usize * dim..(v as usize + 1) * dim]) },
                if use_vis { Some(&mut self.visited) } else { None },
                opts.stop_level,
                if want_trace { Some(&mut self.nav_trace) } else { None },
            )
        };
        let _p1 = crate::counters::prof::tick();
        crate::counters::prof::add(0, _p1 - _p0);
        if G1_K.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            G1_Q.store(query.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            G1_V.store(vectors.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            G1_DIM.store(dim, std::sync::atomic::Ordering::Relaxed);
        }
        if cfg!(feature = "diag") && ORX_ON.with(|o| *o) {
            ORX_Q.store(query.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            ORX_V.store(vectors.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            ORX_DIM.store(dim, std::sync::atomic::Ordering::Relaxed);
        }
        let candidates = if opts.sq4 && want_trace {
            // 轨迹全部保持 visited；按 SQ4 距离取前 carry 个作种子
            let t = &mut self.nav_trace;
            // R27 S1：无 visited 的前提——导航层标记过的节点（= 全部轨迹）都作为种子提供（未截断、无 oracle 注入）
            let nv = novis_on() && t.len() <= opts.carry && ORACLE_SEEDS.with(|o| o.borrow().is_empty());
            t.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            t.truncate(opts.carry);
            // H39 oracle 上界实验：注入查询专属种子（eval harness RAVEN_ORACLE），默认空
            ORACLE_SEEDS.with(|o| {
                for &v in o.borrow().iter() {
                    if self.visited.visit(v) { t.push((v, sq4.distance_raw(code, v as usize))); }
                }
            });
            if self.l0_opts.is_active() {
                VamanaGraph::greedy_search_sq4_seeded_l0_nv(
                    sq4, self.graph.storage(), t, code, ef,
                    &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
                    &self.l0_opts, nv,
                )
            } else {
                VamanaGraph::greedy_search_sq4_seeded_nv(
                    sq4, self.graph.storage(), t, code, ef,
                    &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree, nv,
                )
            }
        } else if want_trace {
            // f32 轨迹：取前 carry 个，补算 SQ4，重置 visited 后只标记种子
            let t = &mut self.nav_trace;
            t.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            t.truncate(opts.carry);
            self.visited.reset();
            for s in t.iter_mut() {
                self.visited.visit(s.0);
                s.1 = sq4.distance_raw(code, s.0 as usize);
            }
            VamanaGraph::greedy_search_sq4_seeded(
                sq4, self.graph.storage(), t, code, ef,
                &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
            )
        } else if opts.sq4 {
            // 单入口，入口 SQ4 距离已知；导航轨迹节点保持 visited
            let seeds = [(ep, ep_dist)];
            VamanaGraph::greedy_search_sq4_seeded(
                sq4, self.graph.storage(), &seeds, code, ef,
                &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
            )
        } else {
            VamanaGraph::greedy_search_sq4(
                sq4, self.graph.storage(), ep, code, ef,
                &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
            )
        };
        self.last_visited_count = self.visited.visited_count();
        let _p2 = crate::counters::prof::tick();
        crate::counters::prof::add(1, _p2 - _p1);
        if cm_nrec {
            let n = (k * self.rerank_factor).max(k).min(candidates.len());
            CM.with(|c| { let mut c = c.borrow_mut(); for x in &candidates[..n] { c.r_rr.push(x.0); } });
        }
        let rrlb = RRLB_E.load(std::sync::atomic::Ordering::Relaxed);
        let r = if rrlb != 0 {
            rerank_lb(self.vectors, dim, query, sq4, self.sq4_perm, unsafe { std::slice::from_raw_parts(rrlb as *const f32, sq4.n) }, candidates, k, self.rerank_factor)
        } else if let (Some(v8), true) = (self.vec_u8, crate::graph::refiner::U8_RR.load(std::sync::atomic::Ordering::Relaxed)) {
            crate::graph::refiner::rerank_u8(v8, dim, query, candidates, k, self.rerank_factor)
        } else if let Some(f16_ds) = self.f16 {
            rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor)
        } else {
            rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor)
        };
        crate::counters::prof::add(2, crate::counters::prof::tick() - _p2);
        r
    }

    /// SQ4 标量量化搜索（4-bit per dimension）
    ///
    /// 图遍历使用 SQ4 无加权距离（Σ (qa[i] - qb[i])²），
    /// 码大小 dim/2 bytes（SIFT-128: 64B vs SQ8 的 128B）。
    /// 终态对 top-N 候选用 f32 精确距离重排序（rerank）。
    ///
    /// 返回 (节点ID, f32精确距离) 列表，按距离升序。
    /// 需先调用 with_sq4() 设置 SQ4Dataset。
    /// R26 段归因：本 searcher 的 visited 结构中节点 v 所在字节的地址（diag_cm 预热用）
    pub fn visited_addr(&self, v: u32) -> usize { self.visited.addr_of(v) }
    /// R27 S1 校验：上次搜索结束时池的原始内容（含已扩展标记位）与游标
    pub fn pool_raw(&self) -> (Vec<(u32, f32)>, usize) { let (d, c, _ef) = self.pool.raw_view(); (d.to_vec(), c) }
    /// R27 ZA 512：按 id 列表清 visited 位（调用方在计时外用）
    pub fn visited_clear_ids(&mut self, ids: &[u32]) { self.visited.clear_ids(ids); }

    /// R33：挂上 HR 模式（图段 ef 应已设为 E；p = 单元数）
    pub fn with_cell_confirm(&mut self, cc: &'a crate::graph::cellconfirm::CellConfirm, p: usize) -> &mut Self {
        self.cc = Some(crate::graph::cellconfirm::CcActive { cc, p, margin: cc.params.margin, pat: cc.params.pat, pminr: cc.params.pminr });
        self.cc_scratch = Some(Box::new(cc.scratch()));
        self
    }

    /// R34：覆盖单元扫描耐心早停（A/B 用）
    pub fn set_cc_pat(&mut self, pat: usize, pminr: f32) { if let Some(a) = self.cc.as_mut() { a.pat = pat; a.pminr = pminr; } }
    /// R35：覆盖分段确认参数（sa=0 关）
    pub fn set_cc_stage(&mut self, sa: usize, su: usize, ss: usize) { if let Some(x) = self.cc_scratch.as_mut() { let a = &mut x.ext; a.sa = sa; a.su = su; a.ss = ss; a.sc = 0; a.saf = 0.0; } }
    /// R36：覆盖 RH 半径收割参数（rk = 0 关）
    pub fn set_cc_rh(&mut self, rk: usize, rmg: f32) { if let Some(x) = self.cc_scratch.as_mut() { x.ext.rk = rk; x.ext.rmg = rmg; } }
    /// R36 RH oracle：下一次查询的初始平方门槛（Some = 用它代替 LID 外推；诊断用，计时含此赋值）
    pub fn set_rh_oracle(&mut self, thr2: Option<f32>) { if let Some(x) = self.cc_scratch.as_mut() { x.ext.rh_oracle = thr2; } }
    /// R36：覆盖 S2C 覆盖判据 sc（0 = 不用）
    pub fn set_cc_sc(&mut self, sc: usize) { if let Some(x) = self.cc_scratch.as_mut() { x.ext.sc = sc; } }
    /// R35 诊断：图段 + 分段确认（总续扫）并返回第一段特征
    pub fn search_hr_diag(&mut self, query: &[f32], k: usize, sa: usize, feat: &mut crate::graph::cellconfirm::StageFeat) -> Vec<(u32, f32)> {
        let act = self.cc.expect("hr");
        let g = self.search_sq4(query, k);
        let mut sc = self.cc_scratch.take().expect("cc scratch");
        let dim = self.dim; let v = self.vectors;
        sc.top.clear();
        sc.top.extend(g.iter().map(|x| (crate::distance::l2_simd(query, &v[x.0 as usize * dim..(x.0 as usize + 1) * dim]), x.0)));
        sc.top.sort_by(|a, b| a.0.total_cmp(&b.0)); sc.top.truncate(k); sc.thr0 = f32::MAX;
        act.cc.confirm_staged(query, v, k, act.p, act.margin, sa, 0, 0, 0, 0, act.pminr, &mut sc, Some(feat));
        let r: Vec<(u32, f32)> = sc.top.iter().map(|x| (x.1, x.0)).collect();
        self.cc_scratch = Some(sc);
        r
    }

    /// R33 HR 模式是否挂上
    pub fn hr_active(&self) -> bool { self.cc.is_some() }

    /// R33 统一入口：挂了 HR（ef ≥ gate）走"图段 E + 单元确认"，否则就是 [`Self::search_sq4`]（原路径，代码未改）。
    /// 由 Prepared::searcher 得到的 searcher 一律用本入口（Python 绑定、eval harness 主循环）。
    #[inline]
    pub fn search_auto(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        match self.cc { Some(act) => self.search_sq4_hr(query, k, act), None => self.search_sq4(query, k) }
    }

    #[inline(never)]
    fn search_sq4_hr(&mut self, query: &[f32], k: usize, act: crate::graph::cellconfirm::CcActive<'a>) -> Vec<(u32, f32)> {
        let ext = self.cc_scratch.as_ref().expect("cc scratch").ext;
        let rh = ext.rk > 0 && k > ext.rk;
        let g = self.search_sq4(query, if rh { ext.rk } else { k });
        let mut sc = self.cc_scratch.take().expect("cc scratch");
        let dim = self.dim; let v = self.vectors;
        sc.top.clear();
        sc.top.extend(g.iter().map(|x| (crate::distance::l2_simd(query, &v[x.0 as usize * dim..(x.0 as usize + 1) * dim]), x.0)));
        sc.top.sort_by(|a, b| a.0.total_cmp(&b.0)); sc.top.truncate(k);
        if rh {
            // R36 RH：图段 top-rk 精确距离 → LID 外推第 k 名半径 → 初始门槛（oracle 覆盖时用给定值）
            sc.thr0 = match ext.rh_oracle { Some(t) => t, None => {
                let d2: Vec<f32> = sc.top.iter().map(|x| x.0).collect();
                let r2 = crate::graph::cellconfirm::lid_radius2(&d2, k);
                if r2 == f32::MAX { f32::MAX } else { r2 * ext.rmg * ext.rmg } } };
        } else { sc.thr0 = f32::MAX; }
        if ext.sa > 0 { let sa = ext.sa.max((ext.saf * act.p as f32).round() as usize);
            act.cc.confirm_staged(query, v, k, act.p, act.margin, sa, ext.su, ext.ss, ext.sc, act.pat, act.pminr, &mut sc, None); }
        else { act.cc.confirm_pat(query, v, k, act.p, act.margin, act.pat, act.pminr, &mut sc); }
        let r: Vec<(u32, f32)> = sc.top.iter().map(|x| (x.1, x.0)).collect();
        self.cc_scratch = Some(sc);
        r
    }

    pub fn search_sq4(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let sq4 = self.sq4.expect("search_sq4 requires with_sq4() first");
        let dim = self.dim;

        // 1. 编码查询向量为 SQ4 packed 4-bit（零分配，复用预分配 buffer）
        //    SQ4 码大小 = ceil(dim/2)，可能与 SQ8 的 dim 不同
        let code_bytes = sq4.code_bytes;
        if self.query_code_buf.len() != code_bytes {
            self.query_code_buf = vec![0u8; code_bytes];
        }
        if !sq4.params.proj.is_empty() {
            // R42c LP：投影矩阵在 SQ4 参数里（m = 码空间维数）
            let (pm, m) = (&sq4.params.proj, sq4.dim);
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear();
            let d = query.len();
            buf.extend((0..m).map(|j| crate::pipeline::dot_f32(&pm[j * d..(j + 1) * d], query)));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else if let Some(perm) = self.sq4_perm {
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear();
            buf.extend(perm.iter().map(|&d| query[d]));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else {
            sq4.params.encode_query_into(query, &mut self.query_code_buf);
        }
        hsel_query(query); // R23 HSEL（关闭时为一次原子读）
        hrel_query(query);

        // H09：非默认导航选项走独立路径（默认行为完全不变）
        if !self.nav_opts.is_legacy() && self.upper_ef <= 1 && self.adaptive_ef.is_none() {
            if let Some(nav) = self.graph.layered_nav() {
                return self.search_sq4_nav_opts(nav, sq4, query, k);
            }
        }

        // 2. 选择 entry_point + 捕获 nav.initialize 的 f32 距离
        let (entry_point, nav_entry_dist) = if let Some(nav) = self.graph.layered_nav() {
            if self.upper_ef > 1 {
                // 方案 B：Layer 1 用 ef>1 beam search
                let (ep, dist) = nav.initialize_with_upper_ef(
                    self.vectors, dim, query, self.upper_ef,
                    &mut self.visited, &mut self.upper_pool,
                );
                (ep, Some(dist))
            } else {
                let (ep, dist) = nav.initialize(self.vectors, dim, query);
                (ep, Some(dist))
            }
        } else if let Some(nav) = self.navigation {
            (Self::nearest_centroid(nav.centroids(), self.vectors, dim, query), None)
        } else {
            (self.graph.entry_point(), None)
        };

        // 3. 自适应 ef
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

        // 4. SQ4 图遍历
        let candidates = VamanaGraph::greedy_search_sq4(
            sq4,
            self.graph.storage(),
            entry_point,
            &self.query_code_buf,
            ef,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
            self.search_degree,
        );

        self.last_visited_count = self.visited.visited_count();

        // 5. f32 rerank（与 SQ8 路径完全相同）
        if let Some(f16_ds) = self.f16 {
            rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor)
        } else {
            rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor)
        }
    }

    /// R47 原型（查询内精度切换，第二段）：以调用方给的种子（例如 LP 码段的精排结果）作入口，
    /// 用本 searcher 的 SQ4 码（通常为全维码）做 seeded beam 搜索 + f32 精排。不改任何字段（布局不变）。
    pub fn search_sq4_from_seeds(&mut self, query: &[f32], k: usize, seeds: &[u32]) -> Vec<(u32, f32)> {
        let sq4 = self.sq4.expect("search_sq4_from_seeds requires with_sq4() first");
        let dim = self.dim;
        if self.query_code_buf.len() != sq4.code_bytes { self.query_code_buf = vec![0u8; sq4.code_bytes]; }
        if !sq4.params.proj.is_empty() {
            let (pm, m) = (&sq4.params.proj, sq4.dim);
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear(); let d = query.len();
            buf.extend((0..m).map(|j| crate::pipeline::dot_f32(&pm[j * d..(j + 1) * d], query)));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else if let Some(perm) = self.sq4_perm {
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear(); buf.extend(perm.iter().map(|&d| query[d]));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else {
            sq4.params.encode_query_into(query, &mut self.query_code_buf);
        }
        self.visited.reset();
        let code = &self.query_code_buf;
        let mut sd: Vec<(u32, f32)> = Vec::with_capacity(seeds.len());
        for &v in seeds { if self.visited.visit(v) { sd.push((v, sq4.distance_raw(code, v as usize))); } }
        sd.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let ef = self.ef_search; self.last_ef_used = ef;
        let candidates = VamanaGraph::greedy_search_sq4_seeded(
            sq4, self.graph.storage(), &sd, code, ef,
            &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
        );
        self.last_visited_count = self.visited.visited_count();
        if let Some(f16_ds) = self.f16 { rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor) }
        else { rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor) }
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
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        results
    }

    /// 上次搜索访问的唯一节点数（avg_visited 诊断接口）
    ///
    /// 返回最近一次 search() 调用期间访问的唯一节点数。
    /// 此值不受 SIMD/内存布局干扰，纯粹衡量图导航效率。
    /// 用于 §〇.2 Pivot Criterion 裁决 Phase 1 vs Phase 3.3 优先级。
    /// R43-2：确定性代价模型的形状 (图段 ef, rerank 倍数, 导航码字节, LP 投影矩阵元素数)
    pub fn cost_shape(&self) -> (usize, usize, usize, usize) { (self.ef_search, self.rerank_factor, self.sq4.map(|s| s.code_bytes).unwrap_or(0), self.sq4.map(|s| s.params.proj.len()).unwrap_or(0)) }

    /// R43-2：HR 确认段上次计数 (幸存者, 单元, 扫描槽, 每槽字节)；未挂 HR 时 None
    pub fn hr_last_counts(&self) -> Option<(usize, usize, usize, usize)> {
        match (&self.cc, &self.cc_scratch) { (Some(a), Some(sc)) => Some(a.cc.last_counts(sc)), _ => None }
    }

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
        let rerank_factor = self.rerank_factor;
        let search_degree = self.search_degree;
        let vectors = self.vectors;
        let dim = self.dim;
        let storage = self.graph.storage();
        let sq8 = self.sq8;
        let sq4 = self.sq4;
        let graph = self.graph;
        let navigation = self.navigation;
        let adaptive_ef = self.adaptive_ef.as_ref();
        let f16_codes = self.f16.map(super::super::distance::f16::F16Dataset::codes);

        // 公共逻辑：选择 entry_point + 计算 ef（SQ8/f32 路径共享，OPT-6 去重）
        let resolve_entry_and_ef = |query: &[f32]| -> (u32, usize) {
            let (entry_point, nav_entry_dist) = graph.layered_nav().map_or_else(|| {
                navigation.map_or_else(|| (graph.entry_point(), None), |nav| {
                    (Self::nearest_centroid(nav.centroids(), vectors, dim, query), None)
                })
            }, |nav| {
                let (ep, dist) = nav.initialize(vectors, dim, query);
                (ep, Some(dist))
            });

            let ef = adaptive_ef.map_or(default_ef, |adaptive| {
                let entry_dist = nav_entry_dist.unwrap_or_else(|| {
                    l2_simd(query,
                        &vectors[entry_point as usize * dim
                            ..(entry_point as usize + 1) * dim])
                });
                adaptive.estimate_ef(entry_dist).max(k)
            });
            (entry_point, ef)
        };

        // 公共逻辑：精量化 rerank（Refiner 框架，支持 f32/f16）
        let rerank_and_sort = move |cands: Vec<(u32, f32)>, query: &[f32]| -> Vec<(u32, f32)> {
            if let Some(f16_vecs) = f16_codes {
                rerank_f16(f16_vecs, dim, query, cands, k, rerank_factor)
            } else {
                rerank_f32(vectors, dim, query, cands, k, rerank_factor)
            }
        };

        queries
            .par_iter()
            .map(|query| {
                // 线程本地 VisitedTracker：每 worker 分配一次后复用
                // 设计文档 F.2：热路径零分配
                thread_local! {
                    static TLS_SEARCH: std::cell::RefCell<
                        (Option<VisitedTracker>, Option<LinearPool>, Option<Vec<u8>>)
                    > = const { std::cell::RefCell::new((None, None, None)) };
                }

                TLS_SEARCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    if borrow.0.is_none() {
                        let cap = adaptive_ef
                            .map_or(default_ef, super::adaptive_ef::AdaptiveEfConfig::max_ef);
                        borrow.0 = Some(VisitedTracker::new(n, cap));
                        borrow.1 = Some(LinearPool::new(cap));
                        // SQ8: 1 byte/dim, SQ4: ceil(dim/2) bytes
                        let code_len = if sq8.is_some() { dim } else { sq4.map_or(dim.div_ceil(2), |q| q.code_bytes) };
                        borrow.2 = Some(vec![0u8; code_len]);
                    }
                    let (visited_opt, pool_opt, code_buf_opt) = &mut *borrow;
                    let visited = visited_opt.as_mut().unwrap();
                    let pool = pool_opt.as_mut().unwrap();
                    let query_code_buf = code_buf_opt.as_mut().unwrap();

                    // 选择搜索路径：SQ8 > SQ4 > f32
                    if let (Some(sq8), true) = (sq8, crate::quant::sq8::sq8_adc_enabled()) {
                        thread_local! { static AQ: std::cell::RefCell<crate::quant::sq8::Sq8AdcQuery> = std::cell::RefCell::new(Default::default()); }
                        let (entry_point, ef) = resolve_entry_and_ef(query);
                        let cands = AQ.with(|a| { let mut aq = a.borrow_mut(); sq8.adc_prepare(query, &mut aq);
                            VamanaGraph::greedy_search_sq8_adc(sq8, storage, entry_point, &aq, ef, visited, pool, po) });
                        rerank_and_sort(cands, query)
                    } else if let Some(sq8) = sq8 {
                        sq8.params.encode_into(query, query_code_buf);
                        let (entry_point, ef) = resolve_entry_and_ef(query);

                        let cands = VamanaGraph::greedy_search_sq8::<true>(
                            sq8, storage, entry_point, query_code_buf,
                            ef, visited, pool, po,
                        );
                        rerank_and_sort(cands, query)
                    } else if let Some(sq4) = sq4 {
                        sq4.params.encode_query_into(query, query_code_buf);
                        // R24 审计：批量路径也须逐查询设置 HSEL/HREL 查询表（否则沿用上一个查询的 LUT）
                        hsel_query(query);
                        hrel_query(query);
                        let (entry_point, ef) = resolve_entry_and_ef(query);

                        let cands = VamanaGraph::greedy_search_sq4(
                            sq4, storage, entry_point, query_code_buf,
                            ef, visited, pool, po, search_degree,
                        );
                        rerank_and_sort(cands, query)
                    } else {
                        let (entry_point, ef) = resolve_entry_and_ef(query);

                        let mut cands = VamanaGraph::greedy_search_vec_reuse(
                            vectors, dim, storage, entry_point, query,
                            ef, visited, pool, po,
                        );
                        cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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
n_jumps: 0,
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
n_jumps: 0,
};
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        let mut searcher = GraphSearcher::new(&vectors, &graph, 20);
        let query = vectors[0..dim].to_vec();
        let results = searcher.search(&query, 5);
        assert!(!results.is_empty());
        // 查询自身应排在最前
        assert_eq!(results[0].0, 0);
    }

    /// R28 组合 1（aarch64 默认填充）：saturate_fill 保留全部自有边、每行按距离升序、度数 = min(D, n-1)、无自环/重复，搜索自身排第一
    #[test]
    fn r28_saturate_fill_keeps_own_edges_sorted() {
        use rand::RngCore;
        let mut rng = ChaCha8Rng::seed_from(5);
        let (n, dim) = (600usize, 8usize);
        let vectors: Vec<f32> = (0..n * dim).map(|_| (rng.next_u64() % 1000) as f32 / 10.0).collect();
        let config = VamanaBuildConfig { alpha: 1.2, l_build: 40, r_max: 12, r_soft: 18, max_iterations: 1, saturate: false,
            enable_layered_nav: false, nav_m: 8, prune_strategy: Default::default(), n_jumps: 0 };
        let g = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        let f = g.saturate_fill(&vectors, dim, 20, 64);
        let l2 = |a: usize, b: usize| -> f32 { (0..dim).map(|d| { let x = vectors[a * dim + d] - vectors[b * dim + d]; x * x }).sum() };
        for u in 0..n as u32 {
            let own: Vec<u32> = g.neighbors(u).iter().copied().filter(|&v| v != crate::SENTINEL && v != u).collect();
            let row: Vec<u32> = f.neighbors(u).iter().copied().filter(|&v| v != crate::SENTINEL).collect();
            assert!(row.len() >= own.len().min(20) && row.len() <= 20, "deg {u}");
            assert!(row.len() >= 20.min(own.len().max(1)));
            let set: std::collections::HashSet<u32> = row.iter().copied().collect();
            assert_eq!(set.len(), row.len(), "dup {u}"); assert!(!set.contains(&u));
            if own.len() <= 20 { for v in &own { assert!(set.contains(v), "own edge lost {u}->{v}"); } }
            for w in row.windows(2) { assert!(l2(u as usize, w[0] as usize) <= l2(u as usize, w[1] as usize)); }
        }
        let mean = (0..n as u32).map(|u| f.neighbors(u).iter().filter(|&&v| v != crate::SENTINEL).count()).sum::<usize>() as f64 / n as f64;
        assert!(mean > 19.0, "mean degree {mean}");
        let mut s = GraphSearcher::new(&vectors, &f, 30);
        for q in [0usize, 17, 333] { assert_eq!(s.search(&vectors[q * dim..(q + 1) * dim], 5)[0].0, q as u32); }
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
n_jumps: 0,
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
n_jumps: 0,
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

    /// H07：精确提前放弃（+ 维度置换）与关闭时结果逐位一致；有损模式只少不多
    #[test]
    fn h07_exact_early_abandon_matches_baseline() {
        use crate::quant::sq4::SQ4Dataset;
        use rand::Rng;
        let (n, dim) = (4000usize, 64usize);
        let mut rng = ChaCha8Rng::seed_from(11);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 40, r_max: 16, r_soft: 24, max_iterations: 1, saturate: false,
            enable_layered_nav: true, nav_m: 8, prune_strategy: Default::default(), n_jumps: 0,
        };
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        assert!(graph.layered_nav().is_some());
        let sq4 = SQ4Dataset::build(&vectors, dim);
        let perm: Vec<usize> = (0..dim).rev().collect();
        let sq4p = SQ4Dataset::build_permuted(&vectors, dim, &perm);
        let queries: Vec<Vec<f32>> = (0..40).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect();
        let run = |ds: &SQ4Dataset, p: Option<&[usize]>, l0: L0Opts| -> Vec<Vec<(u32, f32)>> {
            let mut s = GraphSearcher::new(&vectors, &graph, 30);
            s.with_sq4(ds);
            s.with_nav_opts(NavOpts::sq4_carry());
            s.with_l0_opts(l0);
            if let Some(p) = p { s.with_sq4_perm(p); }
            queries.iter().map(|q| s.search_sq4(q, 10)).collect()
        };
        let base = run(&sq4, None, L0Opts::default());
        assert_eq!(base, run(&sq4, None, L0Opts::exact()));
        assert_eq!(base, run(&sq4p, Some(&perm), L0Opts::default()));
        assert_eq!(base, run(&sq4p, Some(&perm), L0Opts::exact()));
        let mut g16 = L0Opts::exact();
        g16.gran16 = true;
        assert_eq!(base, run(&sq4p, Some(&perm), g16));
        // 有损（scale<1）：仍返回 k 个结果
        let mut lossy = L0Opts::exact();
        lossy.scale = [0.3, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0];
        for r in run(&sq4, None, lossy) {
            assert_eq!(r.len(), 10);
        }
    }


    /// H14：angular（归一化 + L2）d=100 端到端：SQ4 导航携带、精确提前放弃、维度置换、search_degree 均正确，
    /// 精确 EA / 置换结果与基线逐位一致，recall（按 cosine 暴力真值）达标。
    #[test]
    fn h14_angular_d100_end_to_end() {
        use crate::quant::sq4::SQ4Dataset;
        use rand::Rng;
        let (n, dim, nq) = (3000usize, 100usize, 50usize);
        let mut rng = ChaCha8Rng::seed_from(14);
        // 带簇结构的数据（纯均匀随机在 d=100 上近邻无意义）
        let centers: Vec<f32> = (0..30 * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let mut gen = |rng: &mut ChaCha8Rng| -> Vec<f32> {
            let c = rng.gen_range(0..30);
            (0..dim).map(|d| centers[c * dim + d] + (rng.gen::<f32>() - 0.5) * 0.6).collect()
        };
        let raw: Vec<f32> = (0..n).flat_map(|_| gen(&mut rng)).collect();
        let raw_q: Vec<Vec<f32>> = (0..nq).map(|_| gen(&mut rng)).collect();
        let mut vectors = raw.clone();
        assert_eq!(crate::distance::normalize_rows(&mut vectors, dim), 0);
        let queries: Vec<Vec<f32>> = raw_q.iter().map(|q| { let mut q = q.clone(); crate::distance::normalize_rows(&mut q, dim); q }).collect();
        // cosine 暴力真值（原始向量上）
        let cosd = |a: &[f32], b: &[f32]| -> f64 {
            let ab: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            1.0 - ab / (na * nb)
        };
        let gt: Vec<Vec<u32>> = raw_q.iter().map(|q| {
            let mut d: Vec<(u32, f64)> = (0..n).map(|i| (i as u32, cosd(q, &raw[i * dim..(i + 1) * dim]))).collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            d.iter().take(10).map(|x| x.0).collect()
        }).collect();
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 60, r_max: 24, r_soft: 36, max_iterations: 1, saturate: false,
            enable_layered_nav: true, nav_m: 8, prune_strategy: Default::default(), n_jumps: 0,
        };
        let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
        assert!(graph.layered_nav().is_some());
        let sq4 = SQ4Dataset::build(&vectors, dim);
        assert_eq!(sq4.code_bytes, 50);
        let perm: Vec<usize> = (0..dim).map(|j| (j * 37) % dim).collect(); // gcd(37,100)=1 → 置换
        let sq4p = SQ4Dataset::build_permuted(&vectors, dim, &perm);
        let run = |ds: &SQ4Dataset, p: Option<&[usize]>, nav: NavOpts, l0: L0Opts, sd: usize| -> Vec<Vec<(u32, f32)>> {
            let mut s = GraphSearcher::new(&vectors, &graph, 64);
            s.with_sq4(ds);
            s.with_nav_opts(nav);
            s.with_l0_opts(l0);
            s.with_search_degree(sd);
            if let Some(p) = p { s.with_sq4_perm(p); }
            queries.iter().map(|q| s.search_sq4(q, 10)).collect()
        };
        let recall = |res: &[Vec<(u32, f32)>]| -> f64 {
            res.iter().zip(&gt).map(|(r, g)| r.iter().filter(|x| g.contains(&x.0)).count()).sum::<usize>() as f64 / (nq * 10) as f64
        };
        let legacy = NavOpts { dedup: false, sq4: false, carry: 1, stop_level: 1 };
        let r_legacy = run(&sq4, None, legacy, L0Opts::default(), 0);
        let base = run(&sq4, None, NavOpts::sq4_carry(), L0Opts::default(), 0);
        assert!(recall(&r_legacy) >= 0.9, "legacy recall {}", recall(&r_legacy));
        assert!(recall(&base) >= 0.9, "sq4_carry recall {}", recall(&base));
        // 返回的 f32 距离 = 归一化向量 L2² = 2(1-cos)
        for (r, q) in base.iter().zip(&raw_q) {
            for &(id, d) in r {
                let c = cosd(q, &raw[id as usize * dim..(id as usize + 1) * dim]);
                assert!((d as f64 - 2.0 * c).abs() < 1e-4);
            }
        }
        // 精确提前放弃 / 置换 / 两者组合 / 16 维分段：逐位一致
        assert_eq!(base, run(&sq4, None, NavOpts::sq4_carry(), L0Opts::exact(), 0));
        assert_eq!(base, run(&sq4p, Some(&perm), NavOpts::sq4_carry(), L0Opts::default(), 0));
        assert_eq!(base, run(&sq4p, Some(&perm), NavOpts::sq4_carry(), L0Opts::exact(), 0));
        let mut g16 = L0Opts::exact();
        g16.gran16 = true;
        g16.check_mask = u32::MAX; // 越界掩码位必须被安全忽略
        assert_eq!(base, run(&sq4p, Some(&perm), NavOpts::sq4_carry(), g16, 0));
        // search_degree：EA 在截断下同样逐位一致，recall 仍合理
        let sd16 = run(&sq4, None, NavOpts::sq4_carry(), L0Opts::default(), 16);
        assert_eq!(sd16, run(&sq4p, Some(&perm), NavOpts::sq4_carry(), L0Opts::exact(), 16));
        assert!(recall(&sd16) >= 0.85, "sd16 recall {}", recall(&sd16));
    }

    /// R27 S1（H153 无 visited 搜索）：开关开/关在 SIFT 型（128 维整数 L2）与 GloVe 型（100 维归一化 angular）数据上
    /// 逐位一致——rerank 结果、终态池（含已扩展标记与游标）、L0 弹出序列；覆盖精简路径（aarch64 默认）、L0 精确提前放弃路径、
    /// simple 与 pair（dual）路径、search_degree 截断、小 ef（池早满、大量挤出）与大 ef（长期未满）。
    #[test]
    fn r27_s1_novis_bit_identical() {
        use crate::quant::sq4::SQ4Dataset;
        use rand::Rng;
        let rec = |f: &mut dyn FnMut() -> Vec<(u32, f32)>| -> (Vec<(u32, u32)>, Vec<u32>) {
            POPREC_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            POPREC.with(|p| p.borrow_mut().clear());
            let r = f();
            (r.iter().map(|x| (x.0, x.1.to_bits())).collect(), POPREC.with(|p| p.borrow().clone()))
        };
        for (mode, lg) in [(1u8, 11u32), (2, 6), (2, 11)] {
        NVRF_LOG2.store(lg, std::sync::atomic::Ordering::Relaxed);
        for (ds, dim, angular) in [("sift", 128usize, false), ("glove", 100usize, true)] {
            let (n, nq) = (4000usize, 40usize);
            let mut rng = ChaCha8Rng::seed_from(27);
            let centers: Vec<f32> = (0..40 * dim).map(|_| rng.gen::<f32>()).collect();
            let mut gen = |rng: &mut ChaCha8Rng| -> Vec<f32> {
                let c = rng.gen_range(0..40);
                (0..dim).map(|d| if angular { centers[c * dim + d] * 2.0 - 1.0 + (rng.gen::<f32>() - 0.5) * 0.8 }
                    else { ((centers[c * dim + d] * 120.0 + rng.gen::<f32>() * 60.0).floor()).min(255.0) }).collect()
            };
            let mut vectors: Vec<f32> = (0..n).flat_map(|_| gen(&mut rng)).collect();
            let mut queries: Vec<Vec<f32>> = (0..nq).map(|_| gen(&mut rng)).collect();
            if angular {
                crate::distance::normalize_rows(&mut vectors, dim);
                for q in queries.iter_mut() { crate::distance::normalize_rows(q, dim); }
            }
            let config = VamanaBuildConfig {
                alpha: 1.2, l_build: 60, r_max: 24, r_soft: 36, max_iterations: 1, saturate: false,
                enable_layered_nav: true, nav_m: 8, prune_strategy: Default::default(), n_jumps: 0,
            };
            let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);
            let sq4 = SQ4Dataset::build(&vectors, dim);
            let nchk = sq4.code_bytes / 16;
            let mut ea = L0Opts::exact();
            ea.check_mask = (((1u64 << (nchk - 1).min(32)) - 1) as u32) & 0xFF;
            let mut total_pops = 0usize;
            // (1) 完整搜索路径（导航 + L0 + rerank）：精简路径 / EA 路径
            for (l0, sd, ef) in [(L0Opts::default(), 0usize, 10usize), (L0Opts::default(), 0, 40), (L0Opts::default(), 16, 40), (L0Opts::default(), 0, 150),
                                 (ea, 0, 10), (ea, 0, 40), (ea, 16, 100)] {
                let mut s = GraphSearcher::new(&vectors, &graph, ef);
                s.with_sq4(&sq4); s.with_nav_opts(NavOpts::sq4_carry()); s.with_l0_opts(l0); s.with_search_degree(sd); s.with_rerank_factor(4);
                for q in &queries {
                    set_novis(0);
                    let (ra, pa) = rec(&mut || s.search_sq4(q, 10));
                    let pool_a = s.pool_raw();
                    set_novis(mode);
                    let (rb, pb) = rec(&mut || s.search_sq4(q, 10));
                    let pool_b = s.pool_raw();
                    // R28 S2：同一模式下再开批量归并池（仅精简路径生效；EA 路径自动忽略）
                    set_s2(1);
                    let (rc, pc) = rec(&mut || s.search_sq4(q, 10));
                    let pool_c = s.pool_raw();
                    set_s2(0);
                    set_novis(0);
                    assert_eq!(ra, rb, "{ds} ef{ef} sd{sd} ea{} results", l0.early_abandon);
                    assert_eq!(pa, pb, "{ds} ef{ef} sd{sd} ea{} popseq", l0.early_abandon);
                    assert_eq!(ra, rc, "{ds} ef{ef} sd{sd} ea{} s2 results", l0.early_abandon);
                    assert_eq!(pa, pc, "{ds} ef{ef} sd{sd} ea{} s2 popseq", l0.early_abandon);
                    let bits = |p: &(Vec<(u32, f32)>, usize)| (p.0.iter().map(|x| (x.0, x.1.to_bits())).collect::<Vec<_>>(), p.1);
                    assert_eq!(bits(&pool_a), bits(&pool_b), "{ds} ef{ef} pool");
                    assert_eq!(bits(&pool_a), bits(&pool_c), "{ds} ef{ef} s2 pool");
                    assert!(!pa.is_empty());
                    total_pops += pa.len();
                }
            }
            // (2) simple / pair（dual）路径直接调用（不改全局 SIMPLE/PAIR 开关，避免与并行测试互相干扰）
            let qc: Vec<Vec<u8>> = queries.iter().map(|q| sq4.params.encode(q)).collect();
            for dual in [false, true] {
                for (opts, ef, sd) in [(L0Opts::default(), 10usize, 0usize), (L0Opts::default(), 60, 16), (ea, 10, 0), (ea, 40, 0)] {
                    for code in &qc {
                        let ep = graph.entry_point();
                        let seeds = [(ep, sq4.distance_raw(code, ep as usize))];
                        let run = |nv: bool| {
                            let mut vis = VisitedTracker::new(n, ef);
                            let mut pool = LinearPool::new(ef);
                            vis.visit(ep);
                            pool.reset_for(ef);
                            pool.insert(seeds[0].0, seeds[0].1);
                            let (mut thr, mut thr_w) = ([f32::MAX; 8], f32::NAN);
                            let cm = opts.check_mask & 0xFF;
                            rec(&mut || { VamanaGraph::sq4_expand_l0_pair(&sq4, graph.storage(), code, &mut vis, &mut pool, 8, sd, &opts, &mut thr, &mut thr_w, cm, false, dual, nv); pool.to_sorted_vec() })
                        };
                        let a = run(false);
                        set_novis(mode);
                        let b = run(true);
                        set_novis(0);
                        assert_eq!(a, b, "{ds} pair dual={dual} ef{ef} ea{}", opts.early_abandon);
                        total_pops += a.1.len();
                    }
                }
            }
            assert!(total_pops > 1000, "{ds} pops {total_pops}");
        }
        }
        NVRF_LOG2.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
    }
}

/// 断点续建文件格式（小端）：magic "RVCK1\0\0\0", iter u64, chunk u64, word_pos u128, n u64, r_max u64,
/// main_block u32×n·r_max, 每节点 overflow 长度 u32 + 数据。
mod ckpt {
    use crate::memory::HybridBlockedCsr;
    use std::io::{Read, Write};

    pub fn save(path: &str, iter: usize, chunk: usize, wp: u128, st: &HybridBlockedCsr) {
        let tmp = format!("{path}.tmp");
        let t = std::time::Instant::now();
        let r = (|| -> std::io::Result<()> {
            let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
            f.write_all(b"RVCK1\0\0\0")?;
            f.write_all(&(iter as u64).to_le_bytes())?;
            f.write_all(&(chunk as u64).to_le_bytes())?;
            f.write_all(&wp.to_le_bytes())?;
            f.write_all(&(st.len() as u64).to_le_bytes())?;
            f.write_all(&(st.r_max() as u64).to_le_bytes())?;
            let mb = st.main_block();
            let bytes = unsafe { std::slice::from_raw_parts(mb.as_ptr() as *const u8, mb.len() * 4) };
            f.write_all(bytes)?;
            for o in st.overflow() {
                f.write_all(&(o.len() as u32).to_le_bytes())?;
                for &x in o { f.write_all(&x.to_le_bytes())?; }
            }
            f.flush()?;
            f.get_ref().sync_all()?;
            Ok(())
        })();
        match r {
            Ok(()) => { let _ = std::fs::rename(&tmp, path); eprintln!("[build] checkpoint iter={iter} chunk={chunk} saved in {:.1}s", t.elapsed().as_secs_f64()); }
            Err(e) => eprintln!("[build] checkpoint save failed: {e}"),
        }
    }

    pub fn load(path: &str, n: usize, r_max: usize) -> Option<(usize, usize, u128, HybridBlockedCsr)> {
        let mut f = std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path).ok()?);
        let mut m = [0u8; 8];
        f.read_exact(&mut m).ok()?;
        if &m != b"RVCK1\0\0\0" { return None; }
        let mut b8 = [0u8; 8];
        let mut rd = |f: &mut std::io::BufReader<std::fs::File>| -> Option<u64> { f.read_exact(&mut b8).ok()?; Some(u64::from_le_bytes(b8)) };
        let iter = rd(&mut f)? as usize;
        let chunk = rd(&mut f)? as usize;
        let mut b16 = [0u8; 16];
        f.read_exact(&mut b16).ok()?;
        let wp = u128::from_le_bytes(b16);
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).ok()?; let cn = u64::from_le_bytes(b8) as usize;
        f.read_exact(&mut b8).ok()?; let cr = u64::from_le_bytes(b8) as usize;
        if cn != n || cr != r_max { eprintln!("[build] checkpoint shape mismatch, ignoring"); return None; }
        let mut mb = vec![0u32; n * r_max];
        let bytes = unsafe { std::slice::from_raw_parts_mut(mb.as_mut_ptr() as *mut u8, mb.len() * 4) };
        f.read_exact(bytes).ok()?;
        let mut ov = Vec::with_capacity(n);
        let mut b4 = [0u8; 4];
        for _ in 0..n {
            f.read_exact(&mut b4).ok()?;
            let l = u32::from_le_bytes(b4) as usize;
            let mut v = Vec::with_capacity(l);
            for _ in 0..l { f.read_exact(&mut b4).ok()?; v.push(u32::from_le_bytes(b4)); }
            ov.push(v);
        }
        Some((iter, chunk, wp, HybridBlockedCsr::from_parts(n, r_max, mb, ov)))
    }
}

thread_local! {
    /// H39：oracle 种子（仅实验用，空 = 无影响）
    pub static ORACLE_SEEDS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// H39：设置下一次搜索的 oracle 种子
pub fn set_oracle_seeds(v: &[u32]) { ORACLE_SEEDS.with(|o| { let mut b = o.borrow_mut(); b.clear(); b.extend_from_slice(v); }); }

/// H13a：前瞻预取深度（RAVEN_LA 覆盖；默认 0 = 关闭（第 8 轮审计：GloVe 不达标），8 = 原设置）
/// （审计 2026-10-30：改为可运行时切换的原子量，供进程内 A/B）
static LA_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
pub fn la_depth() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = LA_DEPTH.load(Relaxed);
    if v != usize::MAX { return v; }
    let d = std::env::var("RAVEN_LA").ok().and_then(|x| x.parse().ok()).unwrap_or(if cfg!(target_arch = "aarch64") { 8 } else { 0 }); // H107：ARM pad32+rrpf1 之上 la8 两轮正反序 SIFT +3.3–4.0%、GloVe +2.6–3.6% → ARM 默认 8；x86 未重验 → 0。第 8 轮：GloVe +0.8% <3% → 默认关（SIFT +3.9%，RAVEN_LA=8 可开）
    LA_DEPTH.store(d, Relaxed);
    d
}
/// H129 oracle：节点邻域半径数组（f32，按内部 id）与 λ（仅 diag 构建生效）
pub static MINDIST_R: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static MINDIST_LAMBDA: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// R48b：L0 邻居码预取行数上限（RAVEN_PFL；默认 auto：码 ≥4 行（≥256B，fashion/mnist/gist 全维）预取整码，否则只取首行（nytimes 128B ARM ef40 −3.9%，故不动）；显式 N = 上限 N 行，1 = 原行为）。高维码（fashion 416B = 7 行）只预取首行时其余行为按需缺失。
pub static PF_LINES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
#[inline(always)]
unsafe fn pf_code_lines(p: *const i8, lines: usize) {
    crate::arch::prefetch_l1(p);
    let mut l = 1; while l < lines { crate::arch::prefetch_l1(p.wrapping_add(l * 64)); l += 1; }
}
/// R12-ORX oracle（仅 diag 构建）：L0 池键改用精确 f32 距离（查询指针 / 向量基址 / dim），量化噪声对 ef 需求的上界
thread_local! { static ORX_ON: bool = std::env::var("RAVEN_ORX").map(|x| x == "1").unwrap_or(false); }
pub static ORX_Q: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ORX_V: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ORX_DIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[inline(always)]
fn orx_dist(v: u32) -> Option<f32> {
    if !cfg!(feature = "diag") { return None; }
    use std::sync::atomic::Ordering::Relaxed;
    let q = ORX_Q.load(Relaxed); if q == 0 { return None; }
    let d = ORX_DIM.load(Relaxed);
    let qs = unsafe { std::slice::from_raw_parts(q as *const f32, d) };
    let vs = unsafe { std::slice::from_raw_parts((ORX_V.load(Relaxed) as *const f32).add(v as usize * d), d) };
    Some(crate::distance::l2_simd(qs, vs))
}
/// H128：精简路径前瞻深度（RAVEN_LEAN_LA，默认 0）
static LEAN_LA: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
pub fn lean_la_depth() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = LEAN_LA.load(Relaxed);
    if v != usize::MAX { return v; }
    let d = std::env::var("RAVEN_LEAN_LA").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
    LEAN_LA.store(d, Relaxed);
    d
}
/// A1（第 12 轮）精简路径轨迹：[A1_BEGIN, n_pool, (id,dist)*, cursor, n_vis, vis*, (A1_HOP, node, (id,dist)*)*, A1_END]
pub static A1_TRACE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub const A1_BEGIN: u32 = 0xFFFF_FFF0;
pub const A1_HOP: u32 = 0xFFFF_FFF1;
pub const A1_END: u32 = 0xFFFF_FFF2;
thread_local! { pub static A1_TRACE: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) }; }
fn a1_begin(visited: &VisitedTracker, pool: &LinearPool) {
    let (data, cursor, _ef) = pool.raw_view();
    A1_TRACE.with(|t| {
        let mut t = t.borrow_mut();
        t.push(A1_BEGIN); t.push(data.len() as u32);
        for &(id, d) in data { t.push(id); t.push(d.to_bits()); }
        t.push(cursor as u32);
        let v = visited.visited_nodes(); t.push(v.len() as u32); t.extend_from_slice(v);
    });
}
/// H130/H136（第 12 轮）跳执行位掩码（RAVEN_VH；aarch64 默认 16 = H136 早码预取，x86 默认 0；AB 旋钮 vh）
static VEC_HOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
#[inline(always)]
pub fn vec_hop() -> u32 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = VEC_HOP.load(Relaxed);
    if v != u32::MAX { return v; }
    // H136（第 12 轮）：aarch64 默认 16（过滤前按原始邻接预取前 16 个码；3 轮正反序 SIFT +7.6…+12.4%、GloVe +4.6…+8.1%，12/12，召回逐位一致）；
    // R14-C2：16 + 128（去 degrees 读）随组合接受（ARM、x86 各自验证）。x86：R14-X1 独立验证（Ryzen 7 8845H，taskset 单核，4 轮正反序 SIFT +7.1…+11.0%、GloVe +5.5…+7.9%，召回逐位一致）→ 16。RAVEN_VH 可覆盖（位 1/2/4 = H130 向量化跳执行，默认关）
    // R28 组合 1（aarch64）：+ 位 4（QDec64 预解码核，仅 64B 码生效）随组合接受 → aarch64 148；x86 144（位 4 无 x86 实现）
    let d = std::env::var("RAVEN_VH").ok().and_then(|x| x.parse().ok()).unwrap_or(if cfg!(target_arch = "aarch64") { 148 } else { 144 });
    VEC_HOP.store(d, Relaxed);
    d
}
/// R27 S1（H153 无 visited 搜索）：L0 不使用 visited（不测、不置位、不写 history、不预取 visited 行），
/// 重复候选由"固定容量池 + 单调门槛 + 池内同距离段去重"（`LinearPool::insert_dedup`）拒绝，轨迹/结果逐位一致。
/// 模式：0 = 关；1 = 纯无 visited（每个重复候选都重算距离，由池拒绝）；2 = 1 + 有损近期过滤（每查询一张 L1 常驻直接映射表
/// `2^RAVEN_NVRF_LOG2` 个 id，默认 2^11 = 8KB，查询开始整表置空；命中 = 本查询已提供过 ⇒ 由不变量必被池拒绝 ⇒ 直接跳过（不取码、不算距离）；
/// 未命中（含哈希冲突覆盖）走纯模式的精确路径。有损只影响"省多少重算"，不影响正确性——这正是不变量带来的：visited 从正确性结构降级为缓存）。
/// u8::MAX = 未初始化（读 RAVEN_NOVIS，默认见 `novis_default()`）。AB 旋钮 novis（值 0/1/2），cfg 第 8 字段
static NOVIS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);
/// 各平台默认（按 cfg；R27S1 合并门槛判定后设置）
/// R28：x86 = 2（s1l：与 l0ea=0 一起，SIFT +8.1…+12.2%、GloVe +11.6…+16.4%，3 轮 12/12）；aarch64 = 2（组合 1：填充 64 + novis2 + vh148，见 LOG R28）
const fn novis_default() -> u8 { 2 }
#[inline(always)]
pub fn novis_mode() -> u8 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = NOVIS.load(Relaxed);
    if v != u8::MAX { return v; }
    let d = std::env::var("RAVEN_NOVIS").ok().and_then(|x| x.parse().ok()).unwrap_or(novis_default()).min(2);
    NOVIS.store(d, Relaxed);
    d
}
#[inline(always)]
pub fn novis_on() -> bool { novis_mode() > 0 }
pub fn set_novis(mode: u8) { NOVIS.store(mode.min(2), std::sync::atomic::Ordering::Relaxed); }
/// R28 组合 1：建图后饱和填充的默认（度数, ef）。RAVEN_FILL=D,ef 覆盖，RAVEN_FILL=0 关。各平台默认按合并门槛判定后设置。
/// aarch64：R28 C（c1j48，SIFT R48→64 填充 + novis2 + vh148，ef 40→25 / 45→28 等召回，3 轮 6/6：+16.0/+22.1/+25.0%、+21.7/+19.8/+18.3%；
/// GloVe 组合 1 c1j +16.6/+10.7%）→ (64, 128)。x86：组合 1 SIFT 噪声大（c1j48 第 3 轮 −20%），待 R29 低噪声复测 → 关。
/// R29（2026-11-02）：单图进程 ABAB 公平复测（internal runs，4 轮正反序，等召回插值）：SIFT@0.95 +20.6/+12.1/+14.7/+12.4%（中位 +13.5%），
/// GloVe@0.8785 +3.3/−15.2/+4.3/+17.2%、@0.9083 +3.6/−3.0/+9.4/+5.9%（不同向）→ 按合并规则 aarch64 改回 None（两平台都关；RAVEN_FILL=64,128 可开）。
/// R28 的 +16～25% 来自双图同进程 A/B（先载入的 A 图被拖慢），见 CHARTER 测量规则。
/// R31（2026-11-02）x86：单图进程 ABAB 4 轮全档位合并测试（internal runs）：SIFT@0.95 +10.7/+0.6/+7.3/−9.8%（中位 +3.9%，3/4 正），
/// GloVe@0.8785 +14.2/+3.3/+8.9/−16.1%（中位 +6.1%，3/4 正）、@0.9083 +15.4/+4.2/+10.8/+11.5%（+11.1%，4/4）、@0.95 +5.6/+11.9/+6.4/+15.8%（+9.1%，4/4）；
/// 同会话另一组 4 轮（q_table.txt）SIFT +13.7%（4/4）、GloVe@0.95 +3.2%（4/4）→ x86 改为 (64, 128)；aarch64 保持 None（R29 不同向）。
#[cfg(target_arch = "x86_64")]
const FILL_DEFAULT: Option<(usize, usize)> = Some((64, 128));
#[cfg(not(target_arch = "x86_64"))]
const FILL_DEFAULT: Option<(usize, usize)> = None;
pub fn fill_default() -> Option<(usize, usize)> {
    match std::env::var("RAVEN_FILL") {
        Ok(v) if v == "0" => None,
        Ok(v) => { let f: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("RAVEN_FILL=D,ef")).collect(); Some((f[0], *f.get(1).unwrap_or(&128))) }
        Err(_) => FILL_DEFAULT,
    }
}
/// R28 S2（H152 批量归并池）：0 = 关（默认，两平台）；1 = 精简路径在无 visited 生效时整跳批量并入池。RAVEN_S2 / AB s2
static S2_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);
#[inline(always)]
pub fn s2_mode() -> u8 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = S2_MODE.load(Relaxed);
    if v != u8::MAX { return v; }
    let d = std::env::var("RAVEN_S2").ok().and_then(|x| x.parse().ok()).unwrap_or(0u8).min(1);
    S2_MODE.store(d, Relaxed);
    d
}
pub fn set_s2(mode: u8) { S2_MODE.store(mode.min(1), std::sync::atomic::Ordering::Relaxed); }
/// R27 S1 模式 2 的表大小（log2，默认 11）
pub static NVRF_LOG2: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
fn nvrf_log2() -> u32 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = NVRF_LOG2.load(Relaxed);
    if v != u32::MAX { return v; }
    let d = std::env::var("RAVEN_NVRF_LOG2").ok().and_then(|x| x.parse().ok()).unwrap_or(11u32).clamp(6, 20);
    NVRF_LOG2.store(d, Relaxed);
    d
}
thread_local! { static NVRF_BUF: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) }; }
/// R27 S1 模式 2：有损近期过滤（见 NOVIS 注释）。`p` 为空 = 关（纯模式）
struct NvRf { p: *mut u32, shift: u32 }
impl NvRf {
    #[inline(always)]
    fn begin(nv: bool) -> NvRf {
        if !nv || novis_mode() != 2 { return NvRf { p: std::ptr::null_mut(), shift: 0 }; }
        let lg = nvrf_log2();
        let p = NVRF_BUF.with(|b| { let mut b = b.borrow_mut(); let sz = 1usize << lg; if b.len() != sz { b.resize(sz, u32::MAX); } b.fill(u32::MAX); b.as_mut_ptr() });
        NvRf { p, shift: 32 - lg }
    }
    #[inline(always)]
    fn on(&self) -> bool { !self.p.is_null() }
    /// 返回 v 是否在本查询已被提供过（表命中），并记录 v
    #[inline(always)]
    unsafe fn seen(&self, v: u32) -> bool {
        let h = (v.wrapping_mul(0x9E37_79B1) >> self.shift) as usize;
        let q = self.p.add(h);
        let s = *q == v;
        *q = v;
        s
    }
    /// 把邻居 `nb` 写入 `out`（跳过表命中者），返回写入个数
    #[inline(always)]
    fn fill(&self, nb: &[u32], out: &mut [u32]) -> usize {
        if !self.on() { out[..nb.len()].copy_from_slice(nb); return nb.len(); }
        let mut k = 0usize;
        for &v in nb { unsafe { *out.get_unchecked_mut(k) = v; k += (!self.seen(v)) as usize; } }
        k
    }
}
// R27 S1 校验：L0 弹出序列记录（仅 test / diag 构建；默认 release 编译期消除）
#[cfg(any(test, feature = "diag"))]
pub static POPREC_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(any(test, feature = "diag"))]
thread_local! { pub static POPREC: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) }; }
#[inline(always)]
fn pop_rec(_n: u32) {
    #[cfg(any(test, feature = "diag"))]
    if POPREC_ON.load(std::sync::atomic::Ordering::Relaxed) { POPREC.with(|p| p.borrow_mut().push(_n)); }
}
pub fn set_vec_hop(v: u32) { VEC_HOP.store(v, std::sync::atomic::Ordering::Relaxed); }
pub fn set_lean_la(v: usize) { LEAN_LA.store(v, std::sync::atomic::Ordering::Relaxed); }
/// 进程内 A/B：设置前瞻深度
pub fn set_la_depth(v: usize) { LA_DEPTH.store(v, std::sync::atomic::Ordering::Relaxed); }

/// H37 开关（2026-10-30 审计：SIFT1M 9 点均值 +2.5% < 3% 门槛 → 默认关闭；true = 仅池最差变化时重算阈值，逐位等价）
/// H54 名次自适应度数（RAVEN_RAD=r,sd；AB knob rad="r,sd"；sd=0 关）
pub static RAD_SD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static RAD_R: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(10);
pub fn set_rad(spec: &str) { let v: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect(); RAD_R.store(v[0], std::sync::atomic::Ordering::Relaxed); RAD_SD.store(*v.get(1).unwrap_or(&0), std::sync::atomic::Ordering::Relaxed); }
/// H52 L0 码预取距离（RAVEN_CPF；0 = 同 po；AB knob cpf）
pub static CODE_PF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// H51 导航层邻居码批量预取（RAVEN_NAVPF=1；AB knob navpf）
/// R14-C2：ARM/x86 各自独立验证后默认开；RAVEN_NAVPF=0/1 覆盖
pub static NAV_PF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// H50 延迟感知扩展顺序（RAVEN_DEFER=1；AB knob defer）
pub static DEFER_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// H48 插入触发邻接预取开关（RAVEN_ITP=1；AB knob itp）
pub static ITP_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// H45 双链交错开关（RAVEN_PAIR=1，默认关）
pub static PAIR_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// H79a：单链精简扩展（复用 pair 的精简循环但每轮只弹 1 个）——检验"pair 的收益来自更简单的代码路径"而非双链 MLP；结果与主路径逐位一致
/// x86 默认开（H79 接受）；ARM 默认关，待 ARM 复核（ARM 主路径有 qsplit 等 NEON 专属优化）
pub static PO_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static SIMPLE_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(cfg!(target_arch = "x86_64"));
/// H56 邻居码内嵌：每节点前 INL_SD 个邻居的 SQ4 码连续存放（节点 u 的块 = INL_BASE + u*INL_SD*cb）。
/// INL_MODE: 0 = 关；1 = 内嵌取码 + 下一候选整块预取；2 = 内嵌取码，不做整块预取（只靠流水码预取）
/// H58 邻域产出 oracle：记录每跳（取码数, 插入的邻居），查询结束时判定"无用跳"（插入者全部在扩展前被挤出）
pub static YT_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// [hops, useless_hops, zero_ins_hops, codes, useless_codes, zero_ins_codes, full_hops]
pub static YT_STATS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
#[derive(Default)]
pub struct YieldTrace { pub hops: Vec<(u32, u32, u32, bool)>, pub ins: Vec<u32>, pub popped: Vec<u32>, pub meta: Vec<(u32, f32, f32, u32)>, pub dump: Option<Vec<u8>>, pub qid: u32, pub touched: Vec<u32>, pub keep_touch: bool, pub keep_all: bool }
thread_local! { pub static YT: std::cell::RefCell<YieldTrace> = std::cell::RefCell::new(YieldTrace::default()); }
pub fn yt_finish(pool: &LinearPool) {
    use std::sync::atomic::Ordering::Relaxed;
    YT.with(|t| {
        let mut t = t.borrow_mut();
        let mut useful: std::collections::HashSet<u32> = t.popped.iter().copied().collect();
        for (id, _) in pool.to_sorted_vec() { useful.insert(id); }
        let mut st = [0u64; 7];
        let qid = t.qid;
        let mut rows: Vec<u8> = Vec::new();
        for (hi, &(es, a, l, full)) in t.hops.iter().enumerate() {
            if t.dump.is_some() {
                let useless = !t.ins[a as usize..(a + l) as usize].iter().any(|v| useful.contains(v));
                let (nd, d, w, rk) = t.meta[hi];
                for x in [qid, nd, d.to_bits(), w.to_bits(), rk, es, l, useless as u32, hi as u32] { rows.extend_from_slice(&x.to_le_bytes()); }
            }
            st[0] += 1; st[3] += es as u64;
            if full { st[6] += 1; }
            if l == 0 { st[2] += 1; st[5] += es as u64; }
            if !t.ins[a as usize..(a + l) as usize].iter().any(|v| useful.contains(v)) { st[1] += 1; st[4] += es as u64; }
        }
        for i in 0..7 { YT_STATS[i].fetch_add(st[i], Relaxed); }
        if let Some(dv) = t.dump.as_mut() { dv.extend_from_slice(&rows); }
        if !t.keep_touch { t.touched.clear(); t.popped.clear(); }
        if !t.keep_all { t.hops.clear(); t.ins.clear(); t.meta.clear(); }
    });
}
pub static INL_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static INL_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static INL_SD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// H56 构建内嵌码数组（返回 64B 对齐缓冲；调用方负责保活并设置 INL_BASE/INL_SD）
pub fn build_inline_codes(sq4: &SQ4Dataset, storage: &HybridBlockedCsr, n: usize, isd: usize, huge: bool) -> crate::memory::aligned::AlignedBuf<u8> {
    let cb = sq4.code_bytes;
    let mut v = vec![0u8; n * isd * cb];
    for u in 0..n {
        let nb = storage.neighbors(u as u32);
        for (i, &w) in nb.iter().take(isd).enumerate() {
            let o = (u * isd + i) * cb;
            v[o..o + cb].copy_from_slice(sq4.code(w as usize));
        }
    }
    crate::memory::aligned::AlignedBuf::from_slice_opts(&v, 0, huge)
}
pub fn init_pair_from_env() { if let Ok(v) = std::env::var("RAVEN_RAD") { set_rad(&v); } if let Some(c) = std::env::var("RAVEN_CPF").ok().and_then(|v| v.parse().ok()) { CODE_PF.store(c, std::sync::atomic::Ordering::Relaxed); } if let Ok(v) = std::env::var("RAVEN_NAVPF") { NAV_PF.store(v == "1", std::sync::atomic::Ordering::Relaxed); } if std::env::var("RAVEN_DEFER").map(|v| v == "1").unwrap_or(false) { DEFER_MODE.store(true, std::sync::atomic::Ordering::Relaxed); } if std::env::var("RAVEN_ITP").map(|v| v == "1").unwrap_or(false) { ITP_MODE.store(true, std::sync::atomic::Ordering::Relaxed); } if std::env::var("RAVEN_PAIR").map(|v| v == "1").unwrap_or(false) { PAIR_MODE.store(true, std::sync::atomic::Ordering::Relaxed); } if let Ok(v) = std::env::var("RAVEN_SIMPLE") { SIMPLE_MODE.store(v == "1", std::sync::atomic::Ordering::Relaxed); } if let Some(c) = std::env::var("RAVEN_X2PF").ok().and_then(|v| v.parse().ok()) { X2_P1PF.store(c, std::sync::atomic::Ordering::Relaxed); } }
/// H63 门卫准入配置：k0 | k1<<8 | need<<16 | r<<24（need=0 关；仅当弹出者池内名次 ≥ r 时生效）。名次 < k0 直接准入；k0 ≤ 名次 < k1 每次引用加 2，≥ k1 加 1；累计 ≥ need 才取码
pub static DK_CFG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub fn set_dk(k0: u32, k1: u32, need: u32, r: u32) { DK_CFG.store(k0 | (k1 << 8) | (need << 16) | (r.min(255) << 24), std::sync::atomic::Ordering::Relaxed); }
/// H64 动态边长准入：EL_BASE 指向 n×EL_SD 的 u8 对数量化边长（SQ4 码空间平方距离，q=round(16·log2 L)，解码 EL_LUT），EL_TAU=0 关。
/// 池满且 EL_LUT[q] > τ·worst 时不取码（不标记 visited）。下一候选的边长行与其邻接同时预取。
pub static EL_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| { let mut t = [0f32; 256]; for (q, x) in t.iter_mut().enumerate() { *x = if q == 255 { f32::MAX } else { (q as f32 / 16.0).exp2() }; } t });
pub fn el_quant(l: f32) -> u8 { if !(l >= 1.0) { 0 } else { (16.0 * l.log2()).round().min(254.0) as u8 } }
/// 构建 n×sd 的 u8 边长表（缺邻居填 255 = 永远拒绝不会被用到）
pub fn build_edge_len_u8(sq4: &SQ4Dataset, storage: &HybridBlockedCsr, n: usize, sd: usize) -> crate::memory::aligned::AlignedBuf<u8> {
    let mut v = vec![255u8; n * sd];
    for u in 0..n { for (i, &w) in storage.neighbors(u as u32).iter().take(sd).enumerate() { v[u * sd + i] = el_quant(crate::quant::sq4::l2_sq4_raw(sq4.code(u), sq4.code(w as usize))); } }
    crate::memory::aligned::AlignedBuf::from_slice_opts(&v, 0, true)
}
pub static EL_TAU: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub static EL_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static EL_SD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static H37_CACHE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// H69 方向准入 oracle（feature diag）：池满时每跳只保留方向余弦最大的 DIR_K 个新邻居；0 = 关
pub static DIR_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
thread_local! { pub static DIR_REJ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
pub static DIR_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
pub struct DirSig { pub proj: Vec<f32>, pub pu: Vec<f32>, pub sig: Vec<u64>, pub sd: usize, pub dim: usize, pub nb: usize, pub pq: [f32; 64] }
thread_local! { pub static DIR_SIG: std::cell::RefCell<DirSig> = std::cell::RefCell::new(DirSig { proj: Vec::new(), pu: Vec::new(), sig: Vec::new(), sd: 0, dim: 0, nb: 0, pq: [0.0; 64] }); }
/// H69b/c：nb（≤64）个高斯随机投影、每节点投影 pu（n×nb f32）、每边签名 sign(Pv−Pu)（n×sd u64，低 nb 位）
pub fn build_dir_sig(vectors: &[f32], dim: usize, storage: &HybridBlockedCsr, n: usize, sd: usize, nb: usize, seed: u64) {
    let mut st = seed | 1;
    let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; ((st >> 11) as f64 / (1u64 << 53) as f64) as f32 };
    let mut proj = vec![0f32; nb * dim];
    for x in proj.iter_mut() { let (a, b) = (rnd().max(1e-12), rnd()); *x = (-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos(); }
    let mut pu = vec![0f32; n * nb];
    for u in 0..n { let v = &vectors[u * dim..(u + 1) * dim]; for b in 0..nb { pu[u * nb + b] = proj[b * dim..(b + 1) * dim].iter().zip(v).map(|(p, x)| p * x).sum(); } }
    let mut sig = vec![0u64; n * sd];
    for u in 0..n { for (i, &w) in storage.neighbors(u as u32).iter().take(sd).enumerate() {
        let mut s = 0u64; for b in 0..nb { s |= ((pu[w as usize * nb + b] > pu[u * nb + b]) as u64) << b; } sig[u * sd + i] = s; } }
    DIR_SIG.with(|d| { let mut d = d.borrow_mut(); d.proj = proj; d.pu = pu; d.sig = sig; d.sd = sd; d.dim = dim; d.nb = nb; });
}
/// 每个查询前设置 Pq
pub fn dir_set_query(q: &[f32]) { DIR_SIG.with(|d| { let mut d = d.borrow_mut(); if d.dim == 0 { return; } let (dim, nb) = (d.dim, d.nb); for b in 0..nb { let v: f32 = d.proj[b * dim..(b + 1) * dim].iter().zip(q).map(|(p, x)| p * x).sum(); d.pq[b] = v; } }); }

/// H70：L3 常驻 1-bit RaBitQ 排序码
pub struct RbCodes { pub signs: Vec<f32>, pub mean: Vec<f32>, pub bits: Vec<u8>, pub nrm2: Vec<f32>, pub ratio: Vec<f32>, pub lut: Vec<f32>, pub dim: usize }
thread_local! {
    pub static RB: std::cell::RefCell<RbCodes> = std::cell::RefCell::new(RbCodes { signs: Vec::new(), mean: Vec::new(), bits: Vec::new(), nrm2: Vec::new(), ratio: Vec::new(), lut: Vec::new(), dim: 0 });
    pub static RB_TOUCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
fn rb_rotate(x: &mut [f32], signs: &[f32]) {
    let d = x.len();
    for r in 0..2 {
        for i in 0..d { x[i] *= signs[r * d + i]; }
        let mut h = 1; while h < d { for i in (0..d).step_by(h * 2) { for j in i..i + h { let (a, b) = (x[j], x[j + h]); x[j] = a + b; x[j + h] = a - b; } } h *= 2; }
        let s = 1.0 / (d as f32).sqrt(); for v in x.iter_mut() { *v *= s; }
    }
}
/// dim 必须为 2 的幂（SIFT 128）
pub fn build_rb(vectors: &[f32], dim: usize, n: usize, seed: u64) {
    assert!(dim.is_power_of_two() && dim == 128, "H70 RB: dim must be 128 for now");
    let mut st = seed | 1;
    let mut signs = vec![0f32; 2 * dim];
    for x in signs.iter_mut() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; *x = if st & 1 == 0 { 1.0 } else { -1.0 }; }
    let mut mean = vec![0f32; dim];
    for u in 0..n { for i in 0..dim { mean[i] += vectors[u * dim + i]; } }
    for m in mean.iter_mut() { *m /= n as f32; }
    let mut bits = vec![0u8; n * 16]; let mut nrm2 = vec![0f32; n]; let mut ratio = vec![0f32; n];
    let mut o = vec![0f32; dim];
    for u in 0..n {
        for i in 0..dim { o[i] = vectors[u * dim + i] - mean[i]; }
        rb_rotate(&mut o, &signs);
        let n2: f32 = o.iter().map(|x| x * x).sum(); let l1: f32 = o.iter().map(|x| x.abs()).sum();
        nrm2[u] = n2; ratio[u] = if l1 > 0.0 { n2 / l1 } else { 0.0 };
        for i in 0..dim { if o[i] > 0.0 { bits[u * 16 + i / 8] |= 1 << (i % 8); } }
    }
    RB.with(|r| { let mut r = r.borrow_mut(); r.signs = signs; r.mean = mean; r.bits = bits; r.nrm2 = nrm2; r.ratio = ratio; r.lut = vec![0f32; 16 * 256]; r.dim = dim; });
}
pub fn rb_set_query(q: &[f32]) {
    RB.with(|r| { let mut r = r.borrow_mut(); if r.dim == 0 { return; } let dim = r.dim;
        let mut x: Vec<f32> = (0..dim).map(|i| q[i] - r.mean[i]).collect();
        let signs = r.signs.clone(); rb_rotate(&mut x, &signs);
        for t in 0..16 { for byte in 0..256usize { let mut s = 0f32; for b in 0..8 { let v = x[t * 8 + b]; s += if byte >> b & 1 == 1 { v } else { -v }; } r.lut[t * 256 + byte] = s; } }
    });
}
pub fn rb_est(v: usize) -> f32 { RB.with(|rb| { let rb = rb.borrow(); let code = &rb.bits[v * 16..v * 16 + 16]; let mut dot = 0f32; for t in 0..16 { dot += rb.lut[t * 256 + code[t] as usize]; } rb.nrm2[v] - 2.0 * rb.ratio[v] * dot }) }

/// H70b：PQ 排序码
pub struct PqCodes { pub m: usize, pub ds: usize, pub cent: Vec<f32>, pub codes: Vec<u8>, pub lut: Vec<f32> }
thread_local! { pub static PQ: std::cell::RefCell<PqCodes> = std::cell::RefCell::new(PqCodes { m: 0, ds: 0, cent: Vec::new(), codes: Vec::new(), lut: Vec::new() }); }
pub fn build_pq(vectors: &[f32], dim: usize, n: usize, m: usize, iters: usize, train: usize) {
    use rayon::prelude::*;
    let ds = dim / m;
    let step = (n / train).max(1);
    let sample: Vec<usize> = (0..n).step_by(step).collect();
    let mut cent = vec![0f32; m * 256 * ds];
    cent.par_chunks_mut(256 * ds).enumerate().for_each(|(t, c)| {
        for k in 0..256 { let id = sample[(k * 7919) % sample.len()]; c[k * ds..(k + 1) * ds].copy_from_slice(&vectors[id * dim + t * ds..id * dim + (t + 1) * ds]); }
        for _ in 0..iters {
            let mut sum = vec![0f64; 256 * ds]; let mut cnt = vec![0usize; 256];
            for &id in &sample { let x = &vectors[id * dim + t * ds..id * dim + (t + 1) * ds];
                let mut best = (f32::MAX, 0); for k in 0..256 { let d: f32 = (0..ds).map(|i| (x[i] - c[k * ds + i]).powi(2)).sum(); if d < best.0 { best = (d, k); } }
                cnt[best.1] += 1; for i in 0..ds { sum[best.1 * ds + i] += x[i] as f64; } }
            for k in 0..256 { if cnt[k] > 0 { for i in 0..ds { c[k * ds + i] = (sum[k * ds + i] / cnt[k] as f64) as f32; } } }
        }
    });
    let mut codes = vec![0u8; n * m];
    codes.par_chunks_mut(m).enumerate().for_each(|(u, cd)| { for t in 0..m { let x = &vectors[u * dim + t * ds..u * dim + (t + 1) * ds]; let c = &cent[t * 256 * ds..(t + 1) * 256 * ds];
        let mut best = (f32::MAX, 0u8); for k in 0..256 { let d: f32 = (0..ds).map(|i| (x[i] - c[k * ds + i]).powi(2)).sum(); if d < best.0 { best = (d, k as u8); } } cd[t] = best.1; } });
    PQ.with(|p| { let mut p = p.borrow_mut(); p.m = m; p.ds = ds; p.cent = cent; p.codes = codes; p.lut = vec![0f32; m * 256]; });
}
pub fn pq_set_query(q: &[f32]) { PQ.with(|p| { let mut p = p.borrow_mut(); if p.m == 0 { return; } let (m, ds) = (p.m, p.ds);
    for t in 0..m { for k in 0..256 { let mut d = 0f32; for i in 0..ds { let e = q[t * ds + i] - p.cent[(t * 256 + k) * ds + i]; d += e * e; } p.lut[t * 256 + k] = d; } } }); }
pub fn pq_est(v: usize) -> f32 { PQ.with(|p| { let p = p.borrow(); let mut d = 0f32; for t in 0..p.m { d += p.lut[t * 256 + p.codes[v * p.m + t] as usize]; } d }) }

// ===================== H86：跨查询双路交错（MLP across queries） =====================
// 两个独立查询按跳交替：P1(A) P1(B) P2(A) P2(B)。每个查询的搜索序列与单查询 simple 路径逐位相同
// （同样的弹出、visited、插入顺序），只是访存在两条独立依赖链之间重叠。NTC/hops 不变。
/// H87 码第二行预取（AB knob pf2）
pub static CODE_PF2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static X2_P1PF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct X2St<'s> {
    code: &'s [u8],
    visited: &'s mut VisitedTracker,
    pool: &'s mut LinearPool,
    thr: [f32; 8],
    thr_w: f32,
    edge: [u32; 256],
    n: usize,
    alive: bool,
}

impl VamanaGraph {
    #[inline(always)]
    fn x2_p1(st: &mut X2St, storage: &HybridBlockedCsr, sq4: &SQ4Dataset, po: usize, sd: usize, graph_lines: usize, p1pf: usize) {
        st.n = 0;
        let Some((node, _)) = st.pool.pop() else { st.alive = false; return };
        crate::diag_count!(HOPS);
        if let Some((c, _)) = st.pool.peek_unchecked() {
            let ptr = storage.main_block().as_ptr().wrapping_add(c as usize * storage.r_max()) as *const i8;
            unsafe { for l in 0..graph_lines { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
        }
        let neighbors = storage.neighbors(node);
        let el = if sd > 0 { sd.min(neighbors.len()) } else { neighbors.len() };
        for &v in neighbors.iter().take(po.min(el)) { unsafe { st.visited.prefetch(v); } }
        let mut n = 0usize;
        for (i, &v) in neighbors.iter().take(el).enumerate() {
            if n >= 256 { break; }
            if i + po < el { unsafe { st.visited.prefetch(neighbors[i + po]); } }
            let is_new = unsafe { st.visited.visit_unchecked_branchless(v) };
            unsafe { *st.edge.get_unchecked_mut(n) = v; }
            n += is_new as usize;
        }
        st.n = n;
        for &v in st.edge.iter().take(p1pf.min(n)) {
            unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); }
        }
    }

    #[inline(always)]
    fn x2_p2(st: &mut X2St, sq4: &SQ4Dataset, po: usize, opts: &L0Opts, check_mask: u32, h37: bool, p1pf: usize) {
        let n = st.n;
        let ea = opts.early_abandon;
        // p1pf < po 时补发 [p1pf, po) 段（否则这段永远不被预取）
        for i in p1pf.min(n)..po.min(n) { unsafe { crate::arch::prefetch_l1(sq4.code(st.edge[i] as usize).as_ptr() as *const i8); } }
        let la = po.max(p1pf);
        for i in 0..n {
            let j = i + la;
            if j < n { unsafe { crate::arch::prefetch_l1(sq4.code(st.edge[j] as usize).as_ptr() as *const i8); } }
            let neighbor = st.edge[i];
            if ea && st.pool.is_full() {
                let w = st.pool.worst_distance();
                if w != st.thr_w || !h37 {
                    st.thr_w = w;
                    for c in 0..8 { st.thr[c] = w * opts.scale[c]; }
                }
                let code = unsafe { sq4.code_unchecked(neighbor as usize) };
                let (d, dims) = if opts.gran16 {
                    crate::quant::sq4::l2_sq4_raw_bounded16(st.code, code, &st.thr, check_mask)
                } else {
                    crate::quant::sq4::l2_sq4_raw_bounded(st.code, code, &st.thr, check_mask)
                };
                if (dims as usize) < sq4.dim { continue; }
                st.pool.insert(neighbor, d);
            } else {
                let d = unsafe { sq4.distance_raw_unchecked(st.code, neighbor as usize) };
                st.pool.insert(neighbor, d);
            }
        }
        st.n = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn sq4_expand_l0_x2(sq4: &SQ4Dataset, storage: &HybridBlockedCsr, a: &mut X2St, b: &mut X2St, po: usize, sd: usize, opts: &L0Opts) {
        let po = { let o = PO_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed); if o > 0 { o } else { po } };
        let check_mask = opts.check_mask & 0xFF;
        let h37 = H37_CACHE.load(std::sync::atomic::Ordering::Relaxed);
        let effective_r = if sd > 0 { sd.min(storage.r_max()) } else { storage.r_max() };
        let graph_lines = (effective_r * 4).div_ceil(64);
        let p1pf = { let v = X2_P1PF.load(std::sync::atomic::Ordering::Relaxed); if v == 0 { po } else { v } };
        a.alive = true; b.alive = true;
        while a.alive || b.alive {
            if a.alive { Self::x2_p1(a, storage, sq4, po, sd, graph_lines, p1pf); }
            if b.alive { Self::x2_p1(b, storage, sq4, po, sd, graph_lines, p1pf); }
            if a.n > 0 { Self::x2_p2(a, sq4, po, opts, check_mask, h37, p1pf); }
            if b.n > 0 { Self::x2_p2(b, sq4, po, opts, check_mask, h37, p1pf); }
        }
    }
}

impl<'a> GraphSearcher<'a> {
    /// H86：编码查询 + 导航下降 + 种子入池。路径不受支持时返回 false（调用方回退单查询）。
    fn x2_prepare(&mut self, query: &[f32]) -> bool {
        let Some(sq4) = self.sq4 else { return false };
        let opts = self.nav_opts;
        let Some(nav) = self.graph.layered_nav() else { return false };
        if opts.is_legacy() || self.upper_ef > 1 || self.adaptive_ef.is_some() { return false; }
        if !(opts.sq4 && opts.carry > 1 && self.l0_opts.is_active()) { return false; }
        let l0 = &self.l0_opts;
        if !SIMPLE_MODE.load(std::sync::atomic::Ordering::Relaxed) || l0.stop_gamma > 0.0 || (l0.es_k > 0 && l0.es_patience > 0) { return false; }
        if YT_MODE.load(std::sync::atomic::Ordering::Relaxed) { return false; }
        // R24 审计：HSEL/HREL 的查询表是线程局部单份，两查询交错无法共用 → 退回逐查询路径
        if HSEL_K.load(std::sync::atomic::Ordering::Relaxed) > 0 || HREL_K.load(std::sync::atomic::Ordering::Relaxed) > 0 { return false; }
        let code_bytes = sq4.code_bytes;
        if self.query_code_buf.len() != code_bytes { self.query_code_buf = vec![0u8; code_bytes]; }
        if !sq4.params.proj.is_empty() {
            // R42c LP：投影矩阵在 SQ4 参数里（m = 码空间维数）
            let (pm, m) = (&sq4.params.proj, sq4.dim);
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear();
            let d = query.len();
            buf.extend((0..m).map(|j| crate::pipeline::dot_f32(&pm[j * d..(j + 1) * d], query)));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else if let Some(perm) = self.sq4_perm {
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear();
            buf.extend(perm.iter().map(|&d| query[d]));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else {
            sq4.params.encode_query_into(query, &mut self.query_code_buf);
        }
        self.last_ef_used = self.ef_search;
        self.nav_trace.clear();
        self.visited.reset();
        let code = &self.query_code_buf;
        let navpf = NAV_PF.load(std::sync::atomic::Ordering::Relaxed);
        let _ = nav.descend_with_pf(
            |v| unsafe { sq4.distance_raw_unchecked(code, v as usize) },
            |v| if navpf { unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); } },
            Some(&mut self.visited),
            opts.stop_level,
            Some(&mut self.nav_trace),
        );
        let t = &mut self.nav_trace;
        t.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        t.truncate(opts.carry);
        ORACLE_SEEDS.with(|o| {
            for &v in o.borrow().iter() {
                if self.visited.visit(v) { t.push((v, sq4.distance_raw(code, v as usize))); }
            }
        });
        self.pool.reset_for(self.ef_search);
        for &(v, d) in t.iter() { self.pool.insert(v, d); }
        true
    }

    fn x2_finish(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let candidates = self.pool.to_sorted_vec();
        self.last_visited_count = self.visited.visited_count();
        let dim = self.dim;
        if let (Some(v8), true) = (self.vec_u8, crate::graph::refiner::U8_RR.load(std::sync::atomic::Ordering::Relaxed)) {
            crate::graph::refiner::rerank_u8(v8, dim, query, candidates, k, self.rerank_factor)
        } else if let Some(f16_ds) = self.f16 {
            rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor)
        } else {
            rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor)
        }
    }

    /// H86：两个查询交错搜索；结果与分别调用 `search_sq4` 逐位相同。
    pub fn search_sq4_x2(a: &mut GraphSearcher<'a>, b: &mut GraphSearcher<'a>, qa: &[f32], qb: &[f32], k: usize) -> (Vec<(u32, f32)>, Vec<(u32, f32)>) {
        if !a.x2_prepare(qa) { return (a.search_sq4(qa, k), b.search_sq4(qb, k)); }
        if !b.x2_prepare(qb) { let ra = a.search_sq4(qa, k); return (ra, b.search_sq4(qb, k)); }
        let sq4 = a.sq4.unwrap();
        let storage = a.graph.storage();
        let (po, sd, opts) = (a.prefetch_offset, a.search_degree, a.l0_opts);
        {
            let mut sa = X2St { code: &a.query_code_buf, visited: &mut a.visited, pool: &mut a.pool, thr: [f32::MAX; 8], thr_w: f32::NAN, edge: [0; 256], n: 0, alive: true };
            let mut sb = X2St { code: &b.query_code_buf, visited: &mut b.visited, pool: &mut b.pool, thr: [f32::MAX; 8], thr_w: f32::NAN, edge: [0; 256], n: 0, alive: true };
            VamanaGraph::sq4_expand_l0_x2(sq4, storage, &mut sa, &mut sb, po, sd, &opts);
        }
        (a.x2_finish(qa, k), b.x2_finish(qb, k))
    }
}

/// H121：(hops << 16) | sd
pub static HSD_CFG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// N4（第 13 轮）：确认阶段批量化配置（RAVEN_N4="T,T',H,flags"；默认关）。
/// 触发：池前 T 名全部已扩展（游标 ≥ T）且已走 ≥H 跳 → 一次取出名次 T+1..ef 的全部未扩展节点，
/// 批量读邻接 + visited 过滤 + 批量取码（无依赖链），新邻居只与第 T' 名距离比较：更近者正常入池（回到逐跳模式），
/// 其余 flags&1 时以"已扩展"状态入池（保留给 rerank），否则丢弃。
static N4_CFG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
pub static N4_STATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! { pub static N4_CNT: std::cell::RefCell<[u64; 6]> = const { std::cell::RefCell::new([0; 6]) }; }
thread_local! { pub static N4_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
thread_local! { pub static N4_NS2: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
thread_local! { static N4_BUF: std::cell::RefCell<(Vec<u32>, Vec<u32>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) }; }
pub fn parse_n4(s: &str) -> u32 {
    let f: Vec<u32> = s.split(|c| c == ',' || c == '/').map(|x| x.parse().unwrap_or(0)).collect();
    let g = |i: usize| f.get(i).copied().unwrap_or(0);
    if g(0) == 0 { return 0; }
    let tp = if g(1) == 0 { g(0) } else { g(1) };
    (g(0) & 0xFF) | ((tp & 0xFF) << 8) | ((g(2) & 0xFF) << 16) | ((g(3) & 0xFF) << 24)
}
#[inline(always)]
pub fn n4_cfg() -> u32 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = N4_CFG.load(Relaxed);
    if v != u32::MAX { return v; }
    let d = std::env::var("RAVEN_N4").map(|x| parse_n4(&x)).unwrap_or(0);
    N4_CFG.store(d, Relaxed);
    d
}
pub fn set_n4(v: u32) { N4_CFG.store(v, std::sync::atomic::Ordering::Relaxed); }

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn n4_batch(sq4: &SQ4Dataset, storage: &HybridBlockedCsr, query_code: &[u8], visited: &mut VisitedTracker,
            pool: &mut LinearPool, search_degree: usize, tp: usize, flags: u32) {
    let t0 = if N4_STATS.load(std::sync::atomic::Ordering::Relaxed) { Some(std::time::Instant::now()) } else { None };
    N4_BUF.with(|b| {
        let mut b = b.borrow_mut();
        let (s, nb) = &mut *b;
        let ns = pool.take_unexpanded(s);
        let pd = ((flags >> 1) & 0x7F) as usize * 4; // 0 = 过滤时全部预取；>0 = 滚动预取距离
        let r_max = storage.r_max();
        let eff = if search_degree > 0 { search_degree.min(r_max) } else { r_max };
        let lines = (eff * 4).div_ceil(64);
        // 1) 全部邻接行一次性预取（id 已知、无依赖）
        for &u in s.iter() {
            let ptr = storage.main_block().as_ptr().wrapping_add(u as usize * r_max) as *const i8;
            for l in 0..lines { unsafe { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
        }
        // 2) visited 过滤，新邻居立即发码预取
        nb.clear();
        for &u in s.iter() {
            let nbr = storage.neighbors(u);
            let len = eff.min(nbr.len());
            for &v in &nbr[..len] {
                if unsafe { visited.visit_unchecked_branchless(v) } {
                    if pd == 0 || nb.len() < pd { unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); } }
                    nb.push(v);
                }
            }
        }
        if let Some(t) = t0 { N4_NS2.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64)); }
        // 3) 批量距离：只与第 T' 名比较
        let mut thr = pool.kth_distance(tp);
        let mut n_in = 0u64;
        let nn = nb.len();
        // H139（第 13 轮）：确认阶段 L2 常驻粗码拒绝（k-means 单元 id + 残差范数²；est = ‖q−c‖² + ‖x−c‖²），阈值 = 第 T' 名
        let h139 = H139.with(|h| h.borrow().as_ref().map(|t| (t.gamma, t.filter)));
        let cd: Vec<f32> = if h139.is_some() { H139.with(|h| { let h = h.borrow(); let t = h.as_ref().unwrap(); t.cdist(query_code) }) } else { Vec::new() };
        for i in 0..nn {
            let v = unsafe { *nb.get_unchecked(i) };
            if let Some((gamma, filt)) = h139 {
                let est = H139.with(|h| { let h = h.borrow(); let t = h.as_ref().unwrap(); cd[t.assign[v as usize] as usize] + t.r2[v as usize] });
                let d = unsafe { sq4.distance_raw_unchecked(query_code, v as usize) };
                H139_CNT.with(|c| { let mut c = c.borrow_mut(); c[0] += 1; let useful = d < thr; c[1] += useful as u64;
                    for (gi, g) in [0.8f32, 1.0, 1.2, 1.5, 2.0].iter().enumerate() { if est >= g * thr { c[2 + 2 * gi] += 1; c[3 + 2 * gi] += useful as u64; } } });
                // H139b：热段（置换后前 16 B = 32 维）部分距离作为确认阶段拒绝信号（β=1 时为精确下界）
                let dh = crate::quant::sq4::l2_sq4_raw(&query_code[..16], &sq4.code(v as usize)[..16]);
                H139_HOT.with(|r| { let mut r = r.borrow_mut(); let useful = d < thr; r[0] += 1; r[1] += useful as u64;
                    for (gi, b) in [0.3f32, 0.4, 0.5, 0.6, 0.8, 1.0].iter().enumerate() { if dh >= b * thr { r[2 + 2 * gi] += 1; r[3 + 2 * gi] += useful as u64; } } });
                H139_RATIO.with(|r| { let mut r = r.borrow_mut(); for (gi, g) in [1.1f32, 1.2, 1.35, 1.5, 2.0].iter().enumerate() { if d >= g * thr { r[gi] += 1; } } });
                if filt && est >= gamma * thr { continue; }
            }
            let d = unsafe { sq4.distance_raw_unchecked(query_code, v as usize) };
            if d < thr {
                pool.insert(v, d); n_in += 1;
                thr = pool.kth_distance(tp);
            } else if flags & 1 != 0 {
                pool.insert_checked(v, d);
            }
        }
        if N4_STATS.load(std::sync::atomic::Ordering::Relaxed) {
            N4_CNT.with(|c| { let mut c = c.borrow_mut(); c[1] += ns as u64; c[2] += nb.len() as u64; c[4] += 1; c[5] += n_in; });
        }
    });
    if let Some(t) = t0 { N4_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64)); }
}

/// H139（第 13 轮，oracle）：粗码表（SQ4 nibble 空间 k-means）
pub struct H139Table { pub k: usize, pub dim2: usize, pub cent: Vec<f32>, pub assign: Vec<u8>, pub r2: Vec<f32>, pub gamma: f32, pub filter: bool }
impl H139Table {
    pub fn nibbles(code: &[u8], out: &mut [f32]) { for (i, &b) in code.iter().enumerate() { out[2 * i] = (b & 15) as f32; out[2 * i + 1] = (b >> 4) as f32; } }
    pub fn cdist(&self, qc: &[u8]) -> Vec<f32> {
        let mut qn = vec![0f32; self.dim2]; Self::nibbles(qc, &mut qn);
        (0..self.k).map(|c| { let ce = &self.cent[c * self.dim2..(c + 1) * self.dim2]; qn.iter().zip(ce).map(|(a, b)| (a - b) * (a - b)).sum() }).collect()
    }
}
thread_local! { pub static H139: std::cell::RefCell<Option<H139Table>> = const { std::cell::RefCell::new(None) }; }
thread_local! { pub static H139_RATIO: std::cell::RefCell<[u64; 5]> = const { std::cell::RefCell::new([0; 5]) }; }
thread_local! { pub static H139_HOT: std::cell::RefCell<[u64; 14]> = const { std::cell::RefCell::new([0; 14]) }; }
thread_local! { pub static H139_CNT: std::cell::RefCell<[u64; 12]> = const { std::cell::RefCell::new([0; 12]) }; }
pub fn h139_build(sq4: &SQ4Dataset, k: usize, iters: usize, sample: usize) -> H139Table {
    let n = sq4.n; let d2 = sq4.code_bytes * 2;
    let mut x = vec![0f32; n * d2];
    for i in 0..n { H139Table::nibbles(sq4.code(i), &mut x[i * d2..(i + 1) * d2]); }
    let step = (n / sample.max(1)).max(1);
    let samp: Vec<usize> = (0..n).step_by(step).collect();
    let mut cent: Vec<f32> = (0..k).flat_map(|c| x[samp[c * samp.len() / k] * d2..][..d2].to_vec()).collect();
    let near = |v: &[f32], cent: &[f32]| -> (usize, f32) { let mut b = (0usize, f32::MAX); for c in 0..k { let dd = crate::distance::l2_simd(v, &cent[c * d2..(c + 1) * d2]); if dd < b.1 { b = (c, dd); } } b };
    for _ in 0..iters {
        let mut sum = vec![0f64; k * d2]; let mut cnt = vec![0usize; k];
        for &i in &samp { let v = &x[i * d2..(i + 1) * d2]; let (c, _) = near(v, &cent); cnt[c] += 1; for j in 0..d2 { sum[c * d2 + j] += v[j] as f64; } }
        for c in 0..k { if cnt[c] > 0 { for j in 0..d2 { cent[c * d2 + j] = (sum[c * d2 + j] / cnt[c] as f64) as f32; } } }
    }
    let mut assign = vec![0u8; n]; let mut r2 = vec![0f32; n];
    for i in 0..n { let (c, dd) = near(&x[i * d2..(i + 1) * d2], &cent); assign[i] = c as u8; r2[i] = dd; }
    H139Table { k, dim2: d2, cent, assign, r2, gamma: 1.0, filter: false }
}

/// N2（第 13 轮）：rerank 残差估计跳过。表 = 每节点 SQ4 量化残差范数 e_x = ‖x − x̂‖（置换空间）。
/// 对每个候选用已在缓存中的 SQ4 码算 est = ‖q − x̂‖² + e_x²（残差与 q−x̂ 近似正交时的无偏估计，交叉项相对误差 ~2/√d），
/// 按 est 升序精确计算，直到 est ≥ (1+δ)·当前第 k 名精确距离为止（δ = RRLB_DELTA）；其余候选不取 f32 向量。
pub static RRLB_E: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static RRLB_DELTA: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3DCCCCCD); // 0.1
pub static RRLB_STATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! { pub static RRLB_CNT: std::cell::RefCell<[u64; 3]> = const { std::cell::RefCell::new([0; 3]) }; }
pub fn rrlb_build(vectors: &[f32], dim: usize, sq4: &SQ4Dataset, perm: &[usize]) -> Vec<f32> {
    let n = sq4.n;
    let mut e = vec![0f32; n];
    for i in 0..n {
        let x = &vectors[i * dim..(i + 1) * dim];
        let xh = sq4.params.decode(&sq4.code(i)[..dim.div_ceil(2)]);
        let mut s = 0f32;
        for j in 0..dim { let xv = if perm.is_empty() { x[j] } else { x[perm[j]] }; let d = xv - xh[j]; s += d * d; }
        e[i] = s.sqrt();
    }
    e
}
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn rerank_lb(vectors: &[f32], dim: usize, query: &[f32], sq4: &SQ4Dataset, perm: Option<&[usize]>, etab: &[f32],
                 candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)> {
    let rerank_n = (k * rerank_factor).max(k).min(candidates.len());
    let delta = f32::from_bits(RRLB_DELTA.load(std::sync::atomic::Ordering::Relaxed));
    let pr = &sq4.params;
    // a_j = q_perm_j − min_j；est = Σ (a_j − n_j·s_j)² + e_x²
    let mut a = [0f32; 1024];
    let d = dim.min(1024);
    for j in 0..d { let qv = match perm { Some(p) => query[p[j]], None => query[j] }; a[j] = qv - pr.min[j]; }
    let sc = &pr.scale[..d];
    let mut est: Vec<(u32, f32)> = Vec::with_capacity(rerank_n);
    for c in candidates.iter().take(rerank_n) {
        let code = sq4.code(c.0 as usize);
        let mut acc = [0f32; 8];
        let mut j = 0;
        while j + 8 <= d {
            for t in 0..4 { let b = code[j / 2 + t];
                let x0 = (b & 15) as f32 * sc[j + 2 * t]; let x1 = (b >> 4) as f32 * sc[j + 2 * t + 1];
                let d0 = a[j + 2 * t] - x0; let d1 = a[j + 2 * t + 1] - x1;
                acc[2 * t] += d0 * d0; acc[2 * t + 1] += d1 * d1; }
            j += 8;
        }
        let mut sacc: f32 = acc.iter().sum();
        while j < d { let b = code[j / 2]; let nv = if j % 2 == 0 { b & 15 } else { b >> 4 }; let dd = a[j] - nv as f32 * sc[j]; sacc += dd * dd; j += 1; }
        let e = etab[c.0 as usize];
        est.push((c.0, sacc + e * e));
    }
    est.sort_unstable_by(|x, y| x.1.total_cmp(&y.1));
    let pf = crate::graph::refiner::rerank_pf_dist().max(1);
    let pf_vec = |id: u32| {
        let base = vectors.as_ptr().wrapping_add(id as usize * dim) as *const i8;
        let vb = dim * 4; let mut off = 0usize;
        while off < vb { unsafe { crate::arch::prefetch_l1(base.wrapping_add(off)); } off += 64; }
    };
    for c in est.iter().take(pf.min(est.len())) { pf_vec(c.0); }
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    let mut nexact = 0u64;
    for i in 0..est.len() {
        if top.len() >= k && est[i].1 >= (1.0 + delta) * top[k - 1].1 { break; }
        if i + pf < est.len() { pf_vec(est[i + pf].0); }
        let id = est[i].0;
        let dist = l2_simd(query, &vectors[id as usize * dim..(id as usize + 1) * dim]);
        nexact += 1;
        let pos = top.partition_point(|x| x.1 <= dist);
        if pos < k { top.insert(pos, (id, dist)); top.truncate(k); }
    }
    if RRLB_STATS.load(std::sync::atomic::Ordering::Relaxed) { RRLB_CNT.with(|c| { let mut c = c.borrow_mut(); c[0] += est.len() as u64; c[1] += nexact; c[2] += 1; }); }
    top
}

/// G1 oracle（第 13 轮）：方向选择邻居（作弊上界）。AB 旋钮 g1=k，0 = 关。
pub static G1_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static G1_Q: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static G1_V: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static G1_DIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn g1_select(u: u32, nb: &[u32], k: usize, out: &mut [u32; 128]) -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let d = G1_DIM.load(Relaxed);
    let q = unsafe { std::slice::from_raw_parts(G1_Q.load(Relaxed) as *const f32, d) };
    let vb = G1_V.load(Relaxed) as *const f32;
    let uv = unsafe { std::slice::from_raw_parts(vb.add(u as usize * d), d) };
    let qu: Vec<f32> = q.iter().zip(uv).map(|(a, b)| a - b).collect();
    let mut sc: Vec<(f32, u32)> = nb.iter().map(|&v| {
        let vv = unsafe { std::slice::from_raw_parts(vb.add(v as usize * d), d) };
        let (mut dot, mut nn) = (0f32, 0f32);
        for j in 0..d { let e = vv[j] - uv[j]; dot += e * qu[j]; nn += e * e; }
        (-(dot / nn.sqrt().max(1e-12)), v)
    }).collect();
    sc.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = k.min(sc.len()).min(128);
    for i in 0..n { out[i] = sc[i].1; }
    n
}


/// R23 跳内相对排序（hop-relative ranking）oracle/可行性（feature diag；DIR_MODE=7）
/// kind 0 = 全局随机正交投影；1 = 残差 PCA（学到的全局投影）；2 = 局部锚点基（前 B 个邻居的残差张成，每边存最小二乘系数）；3 = 精确（理智检查）
pub struct HopRank {
    pub kind: usize, pub m: usize, pub dim: usize, pub sd: usize, pub n: usize,
    pub vecs: Vec<f32>, pub q: Vec<f32>, pub proj: Vec<f32>, pub pu: Vec<f32>, pub pq: Vec<f32>,
    pub ecode: Vec<i8>, pub escale: Vec<f32>, pub enorm: Vec<f32>, pub nbr: Vec<u32>,
    pub st_hops: u64, pub st_ok: u64, pub st_tot: u64, pub st_ov: u64, pub st_k: u64, pub st_cand: u64,
}
thread_local! {
    pub static HR: std::cell::RefCell<HopRank> = std::cell::RefCell::new(HopRank { kind: 0, m: 0, dim: 0, sd: 0, n: 0, vecs: Vec::new(), q: Vec::new(), proj: Vec::new(), pu: Vec::new(), pq: Vec::new(),
        ecode: Vec::new(), escale: Vec::new(), enorm: Vec::new(), nbr: Vec::new(), st_hops: 0, st_ok: 0, st_tot: 0, st_ov: 0, st_k: 0, st_cand: 0 });
}
fn hr_l2(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() }
/// 对称矩阵循环 Jacobi 特征分解；返回按特征值降序的 (特征值, 特征向量行优先 m×d)
fn hr_jacobi(a: &mut [f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; d * d]; for i in 0..d { v[i * d + i] = 1.0; }
    for _sweep in 0..30 {
        let mut off = 0f64; for i in 0..d { for j in 0..d { if i != j { off += a[i * d + j] * a[i * d + j]; } } }
        if off < 1e-18 { break; }
        for p in 0..d { for q in p + 1..d {
            let apq = a[p * d + q]; if apq.abs() < 1e-300 { continue; }
            let th = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
            let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt()); let t = if th == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt(); let s = t * c;
            for k in 0..d { let (akp, akq) = (a[k * d + p], a[k * d + q]); a[k * d + p] = c * akp - s * akq; a[k * d + q] = s * akp + c * akq; }
            for k in 0..d { let (apk, aqk) = (a[p * d + k], a[q * d + k]); a[p * d + k] = c * apk - s * aqk; a[q * d + k] = s * apk + c * aqk; }
            for k in 0..d { let (vkp, vkq) = (v[k * d + p], v[k * d + q]); v[k * d + p] = c * vkp - s * vkq; v[k * d + q] = s * vkp + c * vkq; }
        } }
    }
    let mut idx: Vec<usize> = (0..d).collect(); idx.sort_by(|&x, &y| a[y * d + y].partial_cmp(&a[x * d + x]).unwrap());
    let ev: Vec<f64> = idx.iter().map(|&i| a[i * d + i]).collect();
    let mut rows = vec![0f64; d * d]; for (r, &i) in idx.iter().enumerate() { for k in 0..d { rows[r * d + k] = v[k * d + i]; } }
    (ev, rows)
}
/// 建 HopRank 表。返回说明字符串（PCA 时含前 m 维能量占比）
pub fn build_hop_rank(vectors: &[f32], dim: usize, storage: &HybridBlockedCsr, n: usize, sd: usize, kind: usize, m: usize, seed: u64) -> String {
    let mut info = String::new();
    let mut nbr = vec![u32::MAX; n * sd];
    for u in 0..n { for (i, &w) in storage.neighbors(u as u32).iter().take(sd).enumerate() { nbr[u * sd + i] = w; } }
    let mut proj = vec![0f32; m * dim];
    if kind == 0 {
        let mut st = seed | 1;
        let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; ((st >> 11) as f64 / (1u64 << 53) as f64) as f64 };
        let mut g = vec![0f64; m * dim];
        for x in g.iter_mut() { let (a, b) = (rnd().max(1e-12), rnd()); *x = (-2.0 * a.ln()).sqrt() * (6.283185307 * b).cos(); }
        for r in 0..m { for p in 0..r { let dp: f64 = (0..dim).map(|k| g[r * dim + k] * g[p * dim + k]).sum(); for k in 0..dim { g[r * dim + k] -= dp * g[p * dim + k]; } }
            let nn: f64 = (0..dim).map(|k| g[r * dim + k].powi(2)).sum::<f64>().sqrt(); for k in 0..dim { g[r * dim + k] /= nn; } }
        let sc = (dim as f64 / m as f64).sqrt();
        for i in 0..m * dim { proj[i] = (g[i] * sc) as f32; }
        info = format!("random orthonormal m={m} scaled sqrt(d/m)");
    } else if kind == 1 {
        let mut cov = vec![0f64; dim * dim]; let mut cnt = 0usize;
        let stride = ((n * sd) / 300_000).max(1);
        let mut r = vec![0f64; dim];
        for e in (0..n * sd).step_by(stride) { let (u, w) = (e / sd, nbr[e]); if w == u32::MAX { continue; }
            for k in 0..dim { r[k] = (vectors[w as usize * dim + k] - vectors[u * dim + k]) as f64; }
            for a in 0..dim { let ra = r[a]; if ra == 0.0 { continue; } for b in 0..dim { cov[a * dim + b] += ra * r[b]; } } cnt += 1; }
        for x in cov.iter_mut() { *x /= cnt as f64; }
        let tr: f64 = (0..dim).map(|i| cov[i * dim + i]).sum();
        let (ev, rows) = hr_jacobi(&mut cov, dim);
        let top: f64 = ev[..m].iter().sum();
        for i in 0..m * dim { proj[i] = rows[i] as f32; }
        info = format!("residual PCA m={m} edges={cnt} energy_top_m={:.3} (ev1={:.0} ev_m={:.0})", top / tr, ev[0], ev[m - 1]);
    }
    // RAVEN_HR_BITS：每投影坐标量化位数（默认 8；4 = 每边 m/2 字节）
    let bits: i32 = std::env::var("RAVEN_HR_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    info = format!("{info} bits={bits}");
    let mut pu = Vec::new();
    let mut ecode = vec![0i8; n * sd * m.max(1)]; let mut escale = vec![0f32; n]; let mut enorm = vec![0f32; n * sd];
    if kind == 0 || kind == 1 {
        pu = vec![0f32; n * m];
        for u in 0..n { let v = &vectors[u * dim..(u + 1) * dim]; for b in 0..m { pu[u * m + b] = proj[b * dim..(b + 1) * dim].iter().zip(v).map(|(p, x)| p * x).sum(); } }
        let mut tmp = vec![0f32; sd * m];
        for u in 0..n {
            let mut mx = 0f32;
            for i in 0..sd { let w = nbr[u * sd + i]; if w == u32::MAX { continue; }
                enorm[u * sd + i] = hr_l2(&vectors[w as usize * dim..(w as usize + 1) * dim], &vectors[u * dim..(u + 1) * dim]);
                for b in 0..m { let x = pu[w as usize * m + b] - pu[u * m + b]; tmp[i * m + b] = x; mx = mx.max(x.abs()); } }
            let s = if mx > 0.0 { mx / qmax } else { 1.0 }; escale[u] = s;
            for i in 0..sd * m { ecode[u * sd * m + i] = (tmp[i] / s).round().clamp(-qmax, qmax) as i8; }
        }
    } else if kind == 2 {
        let bsz = m; let mut tmp = vec![0f32; sd * bsz];
        let mut a = vec![0f64; bsz * dim]; let mut rv = vec![0f64; dim];
        let (mut res_e, mut tot_e) = (0f64, 0f64);
        for u in 0..n {
            let nb = storage.neighbors(u as u32).len().min(sd); let ba = bsz.min(nb);
            for b in 0..ba { let w = nbr[u * sd + b] as usize; for k in 0..dim { a[b * dim + k] = (vectors[w * dim + k] - vectors[u * dim + k]) as f64; } }
            // Gram + ridge, Cholesky
            let mut g = vec![0f64; ba * ba];
            for i in 0..ba { for j in 0..=i { let x: f64 = (0..dim).map(|k| a[i * dim + k] * a[j * dim + k]).sum(); g[i * ba + j] = x; g[j * ba + i] = x; } }
            let tr: f64 = (0..ba).map(|i| g[i * ba + i]).sum::<f64>() / ba.max(1) as f64; for i in 0..ba { g[i * ba + i] += 1e-6 * tr + 1e-12; }
            let mut l = vec![0f64; ba * ba];
            for i in 0..ba { for j in 0..=i { let mut s = g[i * ba + j]; for k in 0..j { s -= l[i * ba + k] * l[j * ba + k]; } if i == j { l[i * ba + i] = s.max(1e-300).sqrt(); } else { l[i * ba + j] = s / l[j * ba + j]; } } }
            let mut mx = 0f32;
            for i in 0..sd { let w = nbr[u * sd + i]; if w == u32::MAX { continue; }
                for k in 0..dim { rv[k] = (vectors[w as usize * dim + k] - vectors[u * dim + k]) as f64; }
                enorm[u * sd + i] = rv.iter().map(|x| x * x).sum::<f64>() as f32;
                let mut y: Vec<f64> = (0..ba).map(|b| (0..dim).map(|k| a[b * dim + k] * rv[k]).sum()).collect();
                for r in 0..ba { let mut s = y[r]; for k in 0..r { s -= l[r * ba + k] * y[k]; } y[r] = s / l[r * ba + r]; }
                for r in (0..ba).rev() { let mut s = y[r]; for k in r + 1..ba { s -= l[k * ba + r] * y[k]; } y[r] = s / l[r * ba + r]; }
                if i >= ba { let mut e = 0f64; for k in 0..dim { let mut p = 0f64; for b in 0..ba { p += y[b] * a[b * dim + k]; } e += (rv[k] - p).powi(2); }
                    res_e += e; tot_e += enorm[u * sd + i] as f64; }
                for b in 0..bsz { let x = if b < ba { y[b] as f32 } else { 0.0 }; tmp[i * bsz + b] = x; mx = mx.max(x.abs()); } }
            let s = if mx > 0.0 { mx / 127.0 } else { 1.0 }; escale[u] = s;
            for i in 0..sd * bsz { ecode[u * sd * bsz + i] = (tmp[i] / s).round().clamp(-127.0, 127.0) as i8; }
        }
        info = format!("local anchor basis B={bsz}: non-anchor residual energy captured {:.3}", 1.0 - res_e / tot_e.max(1e-30));
    } else { info = "exact (sanity)".into(); for u in 0..n { for i in 0..sd { let w = nbr[u * sd + i]; if w != u32::MAX { enorm[u * sd + i] = hr_l2(&vectors[w as usize * dim..(w as usize + 1) * dim], &vectors[u * dim..(u + 1) * dim]); } } } }
    // enorm 量化为每节点 u8（真实实现 1 字节/边）
    for u in 0..n { let mx = enorm[u * sd..(u + 1) * sd].iter().cloned().fold(0f32, f32::max); if mx > 0.0 { for i in 0..sd { enorm[u * sd + i] = (enorm[u * sd + i] / mx * 255.0).round() / 255.0 * mx; } } }
    HR.with(|h| { let mut h = h.borrow_mut(); *h = HopRank { kind, m, dim, sd, n, vecs: vectors.to_vec(), q: vec![0.0; dim], proj, pu, pq: vec![0.0; m],
        ecode, escale, enorm, nbr, st_hops: 0, st_ok: 0, st_tot: 0, st_ov: 0, st_k: 0, st_cand: 0 }; });
    info
}
pub fn hr_set_query(q: &[f32]) { HR.with(|h| { let mut h = h.borrow_mut(); if h.dim == 0 { return; } let (m, dim) = (h.m, h.dim); h.q.copy_from_slice(q);
    if h.kind <= 1 { for b in 0..m { let v: f32 = h.proj[b * dim..(b + 1) * dim].iter().zip(q).map(|(p, x)| p * x).sum(); h.pq[b] = v; } } }); }
/// 统计：(跳数, 成对一致率, top-k 重合率, 平均候选数)；读后清零
pub fn hr_stats_take() -> (u64, f64, f64, f64) { HR.with(|h| { let mut h = h.borrow_mut();
    let r = (h.st_hops, h.st_ok as f64 / h.st_tot.max(1) as f64, h.st_ov as f64 / h.st_k.max(1) as f64, h.st_cand as f64 / h.st_hops.max(1) as f64);
    h.st_hops = 0; h.st_ok = 0; h.st_tot = 0; h.st_ov = 0; h.st_k = 0; h.st_cand = 0; r }) }
#[inline(never)]
fn hr_select(node: u32, edge_buf: &mut [u32; 128], slot_buf: &[u8; 128], edge_size: usize, k: usize, visited: &mut crate::memory::visited::VisitedTracker) -> usize {
    HR.with(|h| {
        let mut h = h.borrow_mut();
        let (m, dim, sd, u) = (h.m, h.dim, h.sd, node as usize);
        let mut sc: [(f32, u32, f32); 128] = [(0.0, 0, 0.0); 128];
        let qu = hr_l2(&h.q, &h.vecs[u * dim..(u + 1) * dim]);
        let mut g = [0f32; 64];
        if h.kind == 2 { // 锚点内积 ⟨q−u, r_b⟩ = (d(q,u) + ‖r_b‖² − d(q,b))/2（锚点码按常规取码计入 NTC；已访问锚点视为距离已缓存）
            for b in 0..m.min(sd) { let w = h.nbr[u * sd + b]; if w == u32::MAX { continue; }
                g[b] = 0.5 * (qu + h.enorm[u * sd + b] - hr_l2(&h.q, &h.vecs[w as usize * dim..(w as usize + 1) * dim])); } }
        let mut pqu = [0f32; 64];
        if h.kind <= 1 { for b in 0..m { pqu[b] = h.pq[b] - h.pu[u * m + b]; } }
        let s = h.escale[u];
        for j in 0..edge_size {
            let v = edge_buf[j] as usize; let slot = slot_buf[j] as usize;
            let exact = hr_l2(&h.q, &h.vecs[v * dim..(v + 1) * dim]);
            let en = h.enorm[u * sd + slot];
            let score = match h.kind {
                0 | 1 => { let c = &h.ecode[(u * sd + slot) * m..(u * sd + slot + 1) * m]; let mut d = 0f32; for b in 0..m { d += pqu[b] * c[b] as f32; } en - 2.0 * s * d }
                2 => { if slot < m { f32::MIN + slot as f32 } else { let c = &h.ecode[(u * sd + slot) * m..(u * sd + slot + 1) * m]; let mut d = 0f32; for b in 0..m { d += g[b] * c[b] as f32; } en - 2.0 * s * d } }
                _ => exact - qu,
            };
            sc[j] = (score, edge_buf[j], exact);
        }
        // 统计（相对精确 f32 距离）
        h.st_hops += 1; h.st_cand += edge_size as u64;
        for a in 0..edge_size { for b in a + 1..edge_size { h.st_tot += 1; if (sc[a].0 < sc[b].0) == (sc[a].2 < sc[b].2) { h.st_ok += 1; } } }
        let mut ex: Vec<f32> = sc[..edge_size].iter().map(|x| x.2).collect(); ex.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); let thr = ex[k - 1];
        sc[..edge_size].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        h.st_k += k as u64; h.st_ov += sc[..k].iter().filter(|x| x.2 <= thr).count() as u64;
        for j in k..edge_size { visited.unvisit(sc[j].1); }
        for j in 0..k { edge_buf[j] = sc[j].1; }
        k
    })
}


// ── R23 HSEL：跳内排序选择（生产路径，RAVEN_HSEL=k 或 AB 旋钮 hsel 开启；默认关）──
// 每点 24B 记录 = 16B 符号位（随机 Hadamard×2 旋转后，维度补零到 128）+ f32 ‖o‖² + f32 2·‖o‖²/‖o‖₁（o = x − 均值）；
// 估计 d(q,x) − ‖q−μ‖² ≈ ‖o‖² − 2·ratio·⟨sign(o), Rq'⟩（1-bit RaBitQ 式），查询 LUT 16×256 f32（16KB，L1 常驻）
pub const HSEL_REC: usize = 24;
pub static HSEL_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static HSEL_EPF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// 是否预取全部邻居的排序记录（AB 值第三段，默认 1）
pub static HSEL_RPF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// 诊断：只做每查询 LUT（不选择），用于分离固定开销（AB 值第四段 = 1 时 hsel_k 视为 0 但仍建 LUT）
pub static HSEL_LUTONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static HSEL_REC_PTR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub struct HselTable { pub rec: crate::memory::aligned::AlignedBuf<u8>, pub signs: Vec<f32>, pub mean: Vec<f32>, pub dim: usize }
static HSEL_TAB: std::sync::OnceLock<HselTable> = std::sync::OnceLock::new();
thread_local! { pub static HSEL_LUT: std::cell::UnsafeCell<[f32; 4096]> = const { std::cell::UnsafeCell::new([0.0; 4096]) }; }
fn hsel_rotate(x: &mut [f32; 128], signs: &[f32]) {
    for r in 0..2 {
        for i in 0..128 { x[i] *= signs[r * 128 + i]; }
        let mut h = 1; while h < 128 { let mut i = 0; while i < 128 { for j in i..i + h { let (a, b) = (x[j], x[j + h]); x[j] = a + b; x[j + h] = a - b; } i += h * 2; } h *= 2; }
        let s = 1.0 / (128f32).sqrt(); for v in x.iter_mut() { *v *= s; }
    }
}
/// 构建 HSEL 记录（dim ≤ 128；内部 id 顺序 = vectors 顺序）。进程级只建一次。
pub fn hsel_build(vectors: &[f32], dim: usize, n: usize, seed: u64, thp: bool) -> usize {
    assert!(dim <= 128, "HSEL: dim must be <= 128");
    if let Some(t) = HSEL_TAB.get() { HSEL_REC_PTR.store(t.rec.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed); return t.rec.len(); }
    let mut st = seed | 1;
    let mut signs = vec![0f32; 256];
    for x in signs.iter_mut() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; *x = if st & 1 == 0 { 1.0 } else { -1.0 }; }
    let mut mean = vec![0f64; dim];
    for u in 0..n { for i in 0..dim { mean[i] += vectors[u * dim + i] as f64; } }
    let mean: Vec<f32> = mean.iter().map(|m| (*m / n as f64) as f32).collect();
    let mut rec = vec![0u8; n * HSEL_REC];
    for u in 0..n {
        let mut o = [0f32; 128];
        for i in 0..dim { o[i] = vectors[u * dim + i] - mean[i]; }
        hsel_rotate(&mut o, &signs);
        let n2: f32 = o.iter().map(|x| x * x).sum(); let l1: f32 = o.iter().map(|x| x.abs()).sum();
        let r = &mut rec[u * HSEL_REC..(u + 1) * HSEL_REC];
        for i in 0..128 { if o[i] > 0.0 { r[i / 8] |= 1 << (i % 8); } }
        r[16..20].copy_from_slice(&n2.to_le_bytes());
        r[20..24].copy_from_slice(&(if l1 > 0.0 { 2.0 * n2 / l1 } else { 0.0 }).to_le_bytes());
    }
    let t = HselTable { rec: crate::memory::aligned::AlignedBuf::from_slice_opts(&rec, 0, thp), signs, mean, dim };
    let _ = HSEL_TAB.set(t);
    let t = HSEL_TAB.get().unwrap();
    HSEL_REC_PTR.store(t.rec.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
    t.rec.len()
}
/// 每查询：旋转查询并建 LUT。每组 8 维 = 低 4 维表 lo[16] + 高 4 维表 hi[16]，lut[b] = lo[b&15] + hi[b>>4]
/// （R23 首版用增量 lut[b] = lut[b&(b−1)] + 2v 形成经内存的串行依赖链，每查询 ≈16µs，使 QPS 掉 1.6×；此版无依赖链）
#[inline]
pub fn hsel_query(q: &[f32]) {
    if HSEL_K.load(std::sync::atomic::Ordering::Relaxed) == 0 { return; }
    let Some(t) = HSEL_TAB.get() else { return };
    let mut x = [0f32; 128];
    for i in 0..t.dim { x[i] = q[i] - t.mean[i]; }
    hsel_rotate(&mut x, &t.signs);
    HSEL_LUT.with(|l| { let lut = unsafe { &mut *l.get() };
        for g in 0..16 { let v = &x[g * 8..g * 8 + 8]; let base = g * 256;
            let mut lo = [0f32; 16]; let mut hi = [0f32; 16];
            for t in 0..16usize { let (mut a, mut c) = (0f32, 0f32); for i in 0..4 { let sg = if t >> i & 1 == 1 { 1.0 } else { -1.0 }; a += sg * v[i]; c += sg * v[4 + i]; } lo[t] = a; hi[t] = c; }
            for h in 0..16usize { let hv = hi[h]; let o = &mut lut[base + h * 16..base + h * 16 + 16]; for l in 0..16 { o[l] = lo[l] + hv; } } } });
}
#[inline(always)]
unsafe fn hsel_select(rec: *const u8, lut: *const f32, edge_buf: &mut [u32; 128], edge_size: usize, k: usize, visited: &mut VisitedTracker) -> usize {
    let dbg = HSEL_DBG.load(std::sync::atomic::Ordering::Relaxed);
    if dbg & 2 != 0 { if dbg & 1 == 0 { for j in k..edge_size { visited.unvisit(*edge_buf.get_unchecked(j)); } } return k; }
    let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
    for j in 0..edge_size {
        let v = *edge_buf.get_unchecked(j);
        let r = rec.add(v as usize * HSEL_REC);
        let mut dot = 0f32;
        for g in 0..16 { dot += *lut.add(g * 256 + *r.add(g) as usize); }
        let n2 = (r.add(16) as *const f32).read_unaligned();
        let ra = (r.add(20) as *const f32).read_unaligned();
        *sc.get_unchecked_mut(j) = (n2 - ra * dot, v);
    }
    let s = &mut sc[..edge_size];
    s.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
    if dbg & 1 == 0 { for j in k..edge_size { visited.unvisit(s[j].1); } }
    for j in 0..k { *edge_buf.get_unchecked_mut(j) = s[j].1; }
    k
}
/// 诊断位：1 = 不撤销 visited；2 = 不打分（直接取前 k）
pub static HSEL_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


// ── R23 HREL：跳内相对排序（hop-relative ranking）生产原型（RAVEN_AB 旋钮 hrel；默认关）──
// 每节点一个顺序块（与邻接行同步提前一跳预取）：
//   行 0：Pu = P·u（f32×16）；行 1：escale f32、nscale f32、‖r_i‖² 量化 u8×sd；之后：P·(v_i − u) int8×16 × sd
// P = 边残差 v−u 的前 16 个主成分（学到的全局投影）。分数 s_i = ‖r_i‖² − 2⟨P(q−u), P r_i⟩ ≈ d(q,v_i) − d(q,u)（公共项 d(q,u) 消去）
pub const HREL_M: usize = 16;
pub static HREL_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static HREL_EPF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static HREL_PTR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static HREL_BSZ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub struct HrelTable { pub blk: crate::memory::aligned::AlignedBuf<u8>, pub proj: Vec<f32>, pub dim: usize }
static HREL_TAB: std::sync::OnceLock<HrelTable> = std::sync::OnceLock::new();
thread_local! { pub static HREL_PQ: std::cell::UnsafeCell<[f32; 16]> = const { std::cell::UnsafeCell::new([0.0; 16]) }; }
pub fn hrel_build(vectors: &[f32], dim: usize, storage: &HybridBlockedCsr, n: usize, sd: usize, thp: bool) -> String {
    use rayon::prelude::*;
    assert!(sd <= 56, "HREL: sd must be <= 56");
    if let Some(t) = HREL_TAB.get() { HREL_PTR.store(t.blk.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed); return "cached".into(); }
    let m = HREL_M;
    // PCA of edge residuals (sampled)
    let tot_e: usize = n * sd; let stride = (tot_e / 300_000).max(1);
    let samples: Vec<(usize, usize)> = (0..tot_e).step_by(stride).filter_map(|e| { let (u, i) = (e / sd, e % sd); storage.neighbors(u as u32).get(i).map(|&w| (u, w as usize)) }).collect();
    let cov = samples.par_chunks(4096).map(|ch| { let mut c = vec![0f64; dim * dim]; let mut r = vec![0f64; dim];
        for &(u, w) in ch { for k in 0..dim { r[k] = (vectors[w * dim + k] - vectors[u * dim + k]) as f64; }
            for a in 0..dim { let ra = r[a]; for b in 0..dim { c[a * dim + b] += ra * r[b]; } } } c })
        .reduce(|| vec![0f64; dim * dim], |mut a, b| { for i in 0..a.len() { a[i] += b[i]; } a });
    let mut cov: Vec<f64> = cov.iter().map(|x| x / samples.len() as f64).collect();
    let tr: f64 = (0..dim).map(|i| cov[i * dim + i]).sum();
    let (ev, rows) = hr_jacobi(&mut cov, dim);
    let energy = ev[..m].iter().sum::<f64>() / tr;
    let proj: Vec<f32> = rows[..m * dim].iter().map(|&x| x as f32).collect();
    let prj = &proj;
    let pu: Vec<f32> = (0..n).into_par_iter().flat_map_iter(|u| { let v = &vectors[u * dim..(u + 1) * dim]; (0..m).map(move |b| v.iter().zip(&prj[b * dim..(b + 1) * dim]).map(|(x, p)| x * p).sum::<f32>()).collect::<Vec<_>>() }).collect();
    let bsz = 128 + sd * m; let bsz = bsz.div_ceil(64) * 64;
    let mut blk = vec![0u8; n * bsz];
    blk.par_chunks_mut(bsz).enumerate().for_each(|(u, b)| {
        let nb = storage.neighbors(u as u32); let len = nb.len().min(sd);
        for t in 0..m { b[t * 4..t * 4 + 4].copy_from_slice(&pu[u * m + t].to_le_bytes()); }
        let mut pr = vec![0f32; len * m]; let mut en = vec![0f32; len]; let (mut mx, mut mn) = (0f32, 0f32);
        for i in 0..len { let w = nb[i] as usize;
            en[i] = (0..dim).map(|k| (vectors[w * dim + k] - vectors[u * dim + k]).powi(2)).sum(); mn = mn.max(en[i]);
            for t in 0..m { let x = pu[w * m + t] - pu[u * m + t]; pr[i * m + t] = x; mx = mx.max(x.abs()); } }
        let es = if mx > 0.0 { mx / 127.0 } else { 1.0 }; let ns = if mn > 0.0 { mn / 255.0 } else { 1.0 };
        b[64..68].copy_from_slice(&es.to_le_bytes()); b[68..72].copy_from_slice(&ns.to_le_bytes());
        for i in 0..len { b[72 + i] = (en[i] / ns).round().clamp(0.0, 255.0) as u8;
            for t in 0..m { b[128 + i * m + t] = ((pr[i * m + t] / es).round().clamp(-127.0, 127.0) as i8) as u8; } }
    });
    let t = HrelTable { blk: crate::memory::aligned::AlignedBuf::from_slice_opts(&blk, 0, thp), proj, dim };
    let _ = HREL_TAB.set(t);
    let t = HREL_TAB.get().unwrap();
    HREL_PTR.store(t.blk.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
    HREL_BSZ.store(bsz, std::sync::atomic::Ordering::Relaxed);
    format!("m={m} sd={sd} block={bsz}B total={:.1}MB residual-PCA energy(top16)={energy:.3} edges_sampled={}", (n * bsz) as f64 / 1e6, samples.len())
}
#[inline]
pub fn hrel_query(q: &[f32]) {
    if HREL_K.load(std::sync::atomic::Ordering::Relaxed) == 0 { return; }
    let Some(t) = HREL_TAB.get() else { return };
    let dim = t.dim;
    HREL_PQ.with(|l| { let pq = unsafe { &mut *l.get() };
        for b in 0..HREL_M { pq[b] = t.proj[b * dim..(b + 1) * dim].iter().zip(q).map(|(p, x)| p * x).sum(); } });
}
#[inline(always)]
unsafe fn hrel_scores(b: *const u8, pq: *const f32, len: usize, out: &mut [f32; 128]) {
    let mut pqu = [0f32; 16];
    for t in 0..16 { pqu[t] = *pq.add(t) - (b.add(t * 4) as *const f32).read(); }
    let es2 = 2.0 * (b.add(64) as *const f32).read();
    let ns = (b.add(68) as *const f32).read();
    #[cfg(target_arch = "aarch64")]
    {
        // NEON：P(q−u) 量化为 int8，每候选 1 次 16B 载入 + smull/smull2 + saddlp/sadalp + addv
        use std::arch::aarch64::*;
        let mut mx = 0f32; for t in 0..16 { mx = mx.max(pqu[t].abs()); }
        let sq = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        let mut q8 = [0i8; 16]; for t in 0..16 { q8[t] = (pqu[t] / sq).round() as i8; }
        let qv = vld1q_s8(q8.as_ptr());
        let k = es2 * sq;
        for i in 0..len {
            let cv = vld1q_s8(b.add(128 + i * 16) as *const i8);
            let lo = vmull_s8(vget_low_s8(qv), vget_low_s8(cv));
            let hi = vmull_high_s8(qv, cv);
            let acc = vpadalq_s16(vpaddlq_s16(lo), hi);
            let dot = vaddvq_s32(acc) as f32;
            *out.get_unchecked_mut(i) = (*b.add(72 + i)) as f32 * ns - k * dot;
        }
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..len {
        let c = b.add(128 + i * 16) as *const i8;
        let mut acc = [0f32; 4];
        for t in 0..16 { acc[t & 3] += pqu[t] * (*c.add(t)) as f32; }
        let dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        *out.get_unchecked_mut(i) = (*b.add(72 + i)) as f32 * ns - es2 * dot;
    }
}


/// R23-H144 门廊直通边（预算中性、与查询无关的拓扑改写）：每节点把前 sd 邻接的最后 r 个（最远）换成
/// "经保留邻居可达的 2 跳点"中、与已保留方向最大余弦最小（方向覆盖最空缺）的 r 个（贪心），候选按 d(u,x) 取前 c。
/// 动机（R22 DECOMP）：漏检者藏在已取码未扩展的门廊节点后一跳；把门廊后、且在 u 的方向覆盖空缺处的点直接接到 u。
/// 返回替换的边数。邻接行长度不变（只改前 sd 槽中的后 r 个）。
pub fn h144_rewire(storage: &mut HybridBlockedCsr, vecs: &[f32], dim: usize, sd: usize, r: usize, c: usize) -> usize {
    use rayon::prelude::*;
    if r == 0 { return 0; }
    let n = storage.len();
    let orig: Vec<Vec<u32>> = (0..n).map(|u| storage.neighbors(u as u32).to_vec()).collect();
    let l2 = |a: usize, b: usize| -> f32 { vecs[a * dim..(a + 1) * dim].iter().zip(&vecs[b * dim..(b + 1) * dim]).map(|(x, y)| (x - y) * (x - y)).sum() };
    let newl: Vec<Option<Vec<u32>>> = (0..n).into_par_iter().map(|u| {
        let o = &orig[u]; let lim = o.len().min(sd); if lim <= r { return None; }
        let keep: Vec<u32> = o[..lim - r].to_vec();
        let mut seen: std::collections::HashSet<u32> = keep.iter().cloned().collect(); seen.insert(u as u32);
        let mut cand: Vec<(f32, u32)> = Vec::new();
        for &w in &keep { for &x in orig[w as usize].iter().take(sd) { if seen.insert(x) { cand.push((l2(u, x as usize), x)); } } }
        cand.sort_by(|a, b| a.0.total_cmp(&b.0)); cand.truncate(c);
        let dirn = |x: usize| -> Vec<f32> { let mut d: Vec<f32> = (0..dim).map(|k| vecs[x * dim + k] - vecs[u * dim + k]).collect(); let nn = d.iter().map(|z| z * z).sum::<f32>().sqrt().max(1e-12); for z in d.iter_mut() { *z /= nn; } d };
        let mut dirs: Vec<Vec<f32>> = keep.iter().map(|&y| dirn(y as usize)).collect();
        let cd: Vec<Vec<f32>> = cand.iter().map(|c| dirn(c.1 as usize)).collect();
        let mut used = vec![false; cand.len()]; let mut nl = keep.clone();
        for _ in 0..r { let mut best = (f32::MAX, usize::MAX);
            for (ci, d) in cd.iter().enumerate() { if used[ci] { continue; }
                let mc = dirs.iter().map(|e| e.iter().zip(d).map(|(a, b)| a * b).sum::<f32>()).fold(f32::MIN, f32::max);
                if mc < best.0 { best = (mc, ci); } }
            if best.1 == usize::MAX { break; }
            used[best.1] = true; nl.push(cand[best.1].1); dirs.push(cd[best.1].clone()); }
        let mut extra = o[lim - r..lim].iter(); while nl.len() < lim { nl.push(*extra.next().unwrap()); }
        nl.extend_from_slice(&o[lim..]);
        Some(nl)
    }).collect();
    let mut changed = 0usize;
    for u in 0..n { if let Some(nl) = &newl[u] { changed += nl.iter().zip(&orig[u]).filter(|(a, b)| a != b).count(); storage.set_neighbors(u as u32, nl); } }
    changed
}


// ---------------------------------------------------------------------------------------------------------------
/// R25 代价模型裁决 oracle（diag_cm 构建）：在真实搜索里固定轨迹、分别改变"取码数"与"每跳串行链长"，直接测每码/每跳的边际成本。
/// rec=true：记录轮，记下本查询中 pool.insert 成功的节点（acc）；rec=false：只取 set 中的节点 + 被拒者中 cm_mix(v^seed) <= keep_thr 的比例，
/// 被跳过者全是"会被池拒绝"的点，故池/轨迹/结果与基线逐位一致（= 完美准入；keep_thr 线性扫取码数，跳数不变）。
/// chain = 每跳串行插入的依赖随机访存次数（表 CM_TAB，掩码 CM_MASK）；extra = 每跳额外独立取码数。
/// R26 段归因（warm 掩码，见 eval harness cm 第 5 字段）：记录轮另记每查询的 r_adj（邻接行绝对行地址）、r_vis（做 visited 检查的节点）、
/// r_new（visited 过滤后的新邻居 = 取码者）、r_nav（导航层算过 f32 距离的节点）、r_rr（rerank 候选）。
pub struct CmState { pub rec: bool, pub keep_thr: u32, pub chain: u32, pub extra: u32, pub block: u32, pub seed: u32,
    pub set: Vec<u32>, pub used: Vec<usize>, pub acc: Vec<u32>, pub n_new: u64, pub n_kept: u64, pub hops: u64, pub sink: f32,
    pub r_adj: Vec<usize>, pub r_vis: Vec<u32>, pub r_new: Vec<u32>, pub r_nav: Vec<u32>, pub r_rr: Vec<u32>,
    /// R26 H148 跳前预取 oracle：记录轮每跳弹出节点 r_pop 与 r_new 的跳偏移 r_off；计时轮载入 q_pop/q_off/q_new（同一查询的记录），
    /// ah = D | Dc<<4 | mode<<8：每跳 i 预取第 i+D 跳的邻接行（mode 1）与第 i+Dc 跳的入池码（mode 2）/ 全部新邻居码（mode 4）
    pub r_pop: Vec<u32>, pub r_off: Vec<u32>, pub q_pop: Vec<u32>, pub q_off: Vec<u32>, pub q_new: Vec<u32>, pub ah: u32,
    /// R27 热态逐项消融（za 掩码，见 sq4_expand 中 "R27 ZA"）：记录轮另记每个被算码的距离 r_dist、每个 visited 检查的结果 r_isn、
    /// 每跳入池成功者 (id, d) r_acc2 及其跳偏移 r_aoff、L0 结束时的池 r_fin；计时轮载入 q_* 后按 za 位把对应项换成"记录值"（理想成本），轨迹逐位一致。
    pub za: u32, pub r_dist: Vec<f32>, pub r_isn: Vec<u8>, pub r_acc2: Vec<(u32, f32)>, pub r_aoff: Vec<u32>, pub r_fin: Vec<(u32, f32)>,
    pub q_dist: Vec<f32>, pub q_isn: Vec<u8>, pub q_acc2: Vec<(u32, f32)>, pub q_aoff: Vec<u32>, pub q_fin: Vec<(u32, f32)> }
impl CmState {
    #[inline(always)]
    pub fn contains(&self, v: u32) -> bool {
        let m = self.set.len() - 1; let mut h = (v.wrapping_mul(0x9E37_79B1) >> 20) as usize & m;
        loop { let x = unsafe { *self.set.get_unchecked(h) }; if x == v { return true; } if x == u32::MAX { return false; } h = (h + 1) & m; }
    }
    pub fn load(&mut self, ids: &[u32]) {
        for &h in &self.used { self.set[h] = u32::MAX; } self.used.clear();
        let m = self.set.len() - 1;
        for &v in ids { let mut h = (v.wrapping_mul(0x9E37_79B1) >> 20) as usize & m;
            while self.set[h] != u32::MAX && self.set[h] != v { h = (h + 1) & m; }
            if self.set[h] == u32::MAX { self.set[h] = v; self.used.push(h); } }
    }
}
pub static CM_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static CM_TAB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static CM_MASK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
thread_local! { pub static CM: std::cell::RefCell<CmState> = std::cell::RefCell::new(CmState { rec: false, keep_thr: u32::MAX, chain: 0, extra: 0, block: 0, seed: 0x5bd1e995,
    set: vec![u32::MAX; 8192], used: Vec::new(), acc: Vec::new(), n_new: 0, n_kept: 0, hops: 0, sink: 0.0,
    r_adj: Vec::new(), r_vis: Vec::new(), r_new: Vec::new(), r_nav: Vec::new(), r_rr: Vec::new(),
    r_pop: Vec::new(), r_off: Vec::new(), q_pop: Vec::new(), q_off: Vec::new(), q_new: Vec::new(), ah: 0,
    za: 0, r_dist: Vec::new(), r_isn: Vec::new(), r_acc2: Vec::new(), r_aoff: Vec::new(), r_fin: Vec::new(),
    q_dist: Vec::new(), q_isn: Vec::new(), q_acc2: Vec::new(), q_aoff: Vec::new(), q_fin: Vec::new() }); }
#[inline(always)]
pub fn cm_mix(x: u32) -> u32 { let mut h = x.wrapping_mul(0x85EB_CA6B); h ^= h >> 13; h = h.wrapping_mul(0xC2B2_AE35); h ^ (h >> 16) }
