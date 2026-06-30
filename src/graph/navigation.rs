//! 上层导航：独立分层图 + 双向 RobustPrune + 多入口初始化
//!
//! v9 重写：上层图不再在 Layer 0 上搜索后筛选同层节点，
//! 而是在每层独立构建导航图：
//! - 小层（≤2000 节点）：暴力精建，候选 = 全部同层节点
//! - 大层（>2000 节点）：暴力种子 2000 + 图引导顺序插入剩余
//! 配合 Vamana RobustPrune（多轮 α 递增 + 遮挡因子）和双向加边，
//! 确保上层贪心下降能收敛到全局最近邻。
//!
//! 对比 Glass HNSW 的改进：
//! - 小层暴力精建 vs Glass 全层增量插入 → 上层图质量完美
//! - Vamana RobustPrune vs Glass heuristic2 → 邻居多样性更好
//! - 多入口初始化 vs Glass 单入口 → 消除冷启动问题

use crate::build::ChaCha8Rng;
use crate::distance::l2_simd;
use crate::graph::linear_pool::LinearPool;
use crate::graph::robust_prune::RobustPrune;
use crate::memory::{HybridBlockedCsr, VisitedTracker};
use rand::Rng;
use rand::seq::SliceRandom;
use std::convert::TryInto;

/// 导航层配置
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct NavigationConfig {
    /// 是否启用 centroid overlay 锚点节点
    /// 设计文档：可选层，可关闭
    pub enable_centroid_overlay: bool,
    /// centroid 数量，默认 √N
    /// 设计文档：√N 个 centroid overlay 锚点节点
    pub centroid_count: Option<usize>,
}


/// 导航层
///
/// 设计文档第三层上层导航：双机制
pub struct NavigationLayer {
    config: NavigationConfig,
    /// centroid 锚点节点列表
    centroids: Vec<u32>,
}

impl NavigationLayer {
    /// 创建导航层
    ///
    /// vectors: 扁平存储的向量数据
    /// dim: 维度
    /// config: 导航配置
    pub fn new(n: usize, vectors: &[f32], dim: usize, config: NavigationConfig) -> Self {
        let centroids = if config.enable_centroid_overlay && n > 0 && dim > 0 {
            let count = config.centroid_count.unwrap_or_else(|| (n as f64).sqrt() as usize);
            let count = count.min(n);
            // 评估报告 M2：用 k-means 聚类选择 centroid（原均匀采样）
            Self::kmeans_centroids(vectors, dim, n, count)
        } else {
            Vec::new()
        };
        Self { config, centroids }
    }

    /// k-means 聚类选择 centroid 锚点节点
    ///
    /// 算法：
    /// 1. 采样 min(n, 10000) 个样本（避免 O(n*k*dim) 全量扫描）
    /// 2. 在样本上跑 k-means++ 初始化 + 迭代分配
    /// 3. 从样本中选择离每个聚类中心最近的节点作为 centroid
    ///
    /// 采样加速：SIFT1M (n=1M, k=1000) 从 O(1.4T) 降到 O(15B)，几秒完成
    fn kmeans_centroids(vectors: &[f32], dim: usize, n: usize, k: usize) -> Vec<u32> {
        if k == 0 || n == 0 {
            return Vec::new();
        }

        // 采样 min(n, 10000) 个样本（ChaCha8Rng 保证确定性）
        let sample_size = n.min(10000);
        let mut rng = ChaCha8Rng::seed_from(42);
        let sample_indices: Vec<usize> = {
            use rand::seq::index::sample;
            sample(&mut rng, n, sample_size).into_vec()
        };

        // 1. k-means++ 初始化（在样本上）
        let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
        let first_idx = sample_indices.choose(&mut rng).copied().unwrap_or(0);
        centers.push(vectors[first_idx * dim..(first_idx + 1) * dim].to_vec());

        // 后续中心按 D(x)² 概率选择
        for _ in 1..k {
            let mut dists = vec![f32::MAX; sample_size];
            for (si, &vi) in sample_indices.iter().enumerate() {
                let v = &vectors[vi * dim..(vi + 1) * dim];
                for c in &centers {
                    let d = l2_simd(v, c);
                    if d < dists[si] {
                        dists[si] = d;
                    }
                }
            }
            let total: f32 = dists.iter().map(|d| d * d).sum();
            if total <= 0.0 {
                break;
            }
            let r: f32 = rng.gen();
            let mut cum = 0.0f32;
            let mut chosen = 0;
            for (si, &d) in dists.iter().enumerate().take(sample_size) {
                cum += d * d / total;
                if cum >= r {
                    chosen = si;
                    break;
                }
            }
            let vi = sample_indices[chosen];
            centers.push(vectors[vi * dim..(vi + 1) * dim].to_vec());
        }

        // 2. 迭代分配 + 更新中心（最多 10 次，在样本上）
        let mut assignments = vec![0usize; sample_size];
        for _ in 0..10 {
            let mut changed = false;
            for (si, &vi) in sample_indices.iter().enumerate() {
                let v = &vectors[vi * dim..(vi + 1) * dim];
                let mut best = 0;
                let mut best_dist = f32::MAX;
                for (j, c) in centers.iter().enumerate() {
                    let d = l2_simd(v, c);
                    if d < best_dist {
                        best_dist = d;
                        best = j;
                    }
                }
                if assignments[si] != best {
                    assignments[si] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            let mut new_centers = vec![vec![0.0f32; dim]; centers.len()];
            let mut counts = vec![0usize; centers.len()];
            for (si, &vi) in sample_indices.iter().enumerate() {
                let a = assignments[si];
                let v = &vectors[vi * dim..(vi + 1) * dim];
                for d in 0..dim {
                    new_centers[a][d] += v[d];
                }
                counts[a] += 1;
            }
            for j in 0..centers.len() {
                if counts[j] > 0 {
                    for c in new_centers[j].iter_mut().take(dim) {
                        *c /= counts[j] as f32;
                    }
                    centers[j] = std::mem::take(&mut new_centers[j]);
                }
            }
        }

        // 3. 从样本中选择离每个聚类中心最近的节点作为 centroid
        let mut result: Vec<u32> = Vec::with_capacity(centers.len());
        for c in &centers {
            let mut best = 0u32;
            let mut best_dist = f32::MAX;
            for &vi in &sample_indices {
                let v = &vectors[vi * dim..(vi + 1) * dim];
                let d = l2_simd(v, c);
                if d < best_dist {
                    best_dist = d;
                    best = vi as u32;
                }
            }
            if !result.contains(&best) {
                result.push(best);
            }
        }
        result
    }

    /// 获取 centroid 锚点节点
    pub fn centroids(&self) -> &[u32] {
        &self.centroids
    }

    /// 是否启用 centroid overlay
    pub fn is_overlay_enabled(&self) -> bool {
        self.config.enable_centroid_overlay
    }
}

// ============================================================================
// LayeredNavigation（v9 重写：独立上层图 + 双向 RobustPrune + 多入口）
// ============================================================================

/// 暴力精建的节点数阈值：≤ 此值的层暴力全对计算，> 此值的层用图引导
const BRUTE_FORCE_THRESHOLD: usize = 2000;

/// 多入口候选上限
const MAX_ENTRY_CANDIDATES: usize = 50;

/// HNSW 风格分层导航（v9 重写）
///
/// 在 Vamana flat 图之上构建多层独立导航图：
/// - Layer 0: 完整 Vamana 图（所有 N 个节点）
/// - Layer l > 0: 随机选取的稀疏子集，节点数 ~ N / M^l
///
/// 与旧实现的根本区别：
/// - 旧：在 Layer 0 上搜索 200 候选 → 筛选同层 → 上层图是 Layer 0 残渣
/// - 新：在每层独立图上搜索 → RobustPrune → 双向加边 → 上层图是独立导航图
///
/// 结构兼容 glass 的 `HNSWInitializer`：
/// - `node_levels[i]`: 节点 i 的层级
/// - `upper_lists[i]`: 节点 i 的上层边（flat array）
///   对 level L 的节点，有 L*K 条边，level l 的边在 `[(l-1)*K .. l*K)`
pub struct LayeredNavigation {
    /// 顶层入口节点（被强制为 max_level）
    entry_point: u32,
    /// 最高层级
    max_level: u8,
    /// 上层图度数 K = R/2
    k: usize,
    /// 每个节点的层级（0 = 仅 Layer 0）
    node_levels: Vec<u8>,
    /// 上层边：`upper_lists[node_id]` = flat array
    /// 对 level L 的节点，`[(l-1)*K .. l*K)` 是 level l 的 K 条边
    /// level 0 的节点为空 Vec
    upper_lists: Vec<Vec<u32>>,
    /// 多入口候选节点（通常 = max_level 层的全部节点，上限 50）
    /// 搜索时从中选距离 query 最近的作为贪心下降起点
    entry_candidates: Vec<u32>,
}

impl LayeredNavigation {
    /// 构建分层导航
    ///
    /// 在 flat Vamana 图构建完成后调用：
    /// 1. HNSW 层级分配（`level = floor(-ln(uniform) / ln(M))`，确定性 ChaCha8 种子 42）
    /// 2. 强制 medoid 为顶层
    /// 3. 自底向上构建每层独立导航图（暴力精建 / 图引导）
    /// 4. 选择多入口候选
    pub fn build(
        vectors: &[f32],
        dim: usize,
        _storage: &HybridBlockedCsr,
        entry_point: u32,
        m: usize,
        k: usize,
    ) -> Self {
        let n = vectors.len() / dim;
        if n == 0 || k == 0 {
            return Self::empty(0, k);
        }

        // 1. HNSW 层级分配
        let mut rng = ChaCha8Rng::seed_from(42);
        let ml = 1.0 / (m as f64).ln();
        let mut node_levels = vec![0u8; n];
        let mut max_level: u8 = 0;
        for level in node_levels.iter_mut().take(n) {
            // level = floor(-ln(U) * mL), U ~ Uniform(0,1)
            let u: f64 = rng.gen();
            *level = (-(u.max(1e-10).ln()) * ml).floor() as u8;
            if *level > max_level {
                max_level = *level;
            }
        }
        // 强制 medoid 为顶层（HNSW 的 enterpoint_node_ 始终在最高层）
        node_levels[entry_point as usize] = max_level;

        // 2. 构建上层图（独立于 Layer 0）
        let upper_lists = if max_level > 0 {
            Self::build_upper_layers(vectors, dim, &node_levels, max_level, k)
        } else {
            vec![Vec::new(); n]
        };

        // 3. 多入口候选选择
        let entry_candidates =
            Self::select_entry_candidates(vectors, dim, &node_levels, max_level, entry_point);

        Self {
            entry_point,
            max_level,
            k,
            node_levels,
            upper_lists,
            entry_candidates,
        }
    }

    fn empty(n: usize, k: usize) -> Self {
        Self {
            entry_point: 0,
            max_level: 0,
            k,
            node_levels: vec![0; n],
            upper_lists: vec![Vec::new(); n],
            entry_candidates: Vec::new(),
        }
    }

    /// 最高层级
    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    // ========================================================================
    // 上层图构建
    // ========================================================================

    /// 为每层构建独立导航图
    ///
    /// 自底向上（level 1 → max_level），每层独立建图：
    /// - 小层（≤2000）：暴力精建，候选 = 全部同层节点
    /// - 大层（>2000）：暴力种子 2000 + 图引导顺序插入剩余
    fn build_upper_layers(
        vectors: &[f32],
        dim: usize,
        node_levels: &[u8],
        max_level: u8,
        k: usize,
    ) -> Vec<Vec<u32>> {
        let n = node_levels.len();
        let mut upper_lists: Vec<Vec<u32>> = vec![Vec::new(); n];

        for level in 1..=max_level {
            let layer_nodes: Vec<u32> = (0..n as u32)
                .filter(|&i| node_levels[i as usize] >= level)
                .collect();

            if layer_nodes.len() <= 1 {
                continue;
            }

            eprintln!(
                "[nav] level {}: {} nodes → {}",
                level,
                layer_nodes.len(),
                if layer_nodes.len() <= BRUTE_FORCE_THRESHOLD {
                    "brute-force"
                } else {
                    "graph-guided"
                }
            );

            // 临时邻接表：仅存储本层边
            let mut level_adj: Vec<Vec<u32>> = vec![Vec::new(); n];

            if layer_nodes.len() <= BRUTE_FORCE_THRESHOLD {
                Self::build_level_brute_force(vectors, dim, &layer_nodes, &mut level_adj, k);
            } else {
                Self::build_level_graph_guided(vectors, dim, &layer_nodes, &mut level_adj, k);
            }

            // 合并到 upper_lists（level 1 的边在前，level 2 在后，...）
            // Pad to K 条边：保持 flat 布局一致性（每层恰好 K 条边）
            // 用自身填充，initialize 中跳过 v == u
            for &u in &layer_nodes {
                let mut edges = std::mem::take(&mut level_adj[u as usize]);
                while edges.len() < k {
                    edges.push(u);
                }
                upper_lists[u as usize].extend(edges);
            }
        }

        upper_lists
    }

    /// 暴力精建：候选 = 全部同层节点
    ///
    /// 对每个节点 u：
    /// 1. 候选集 = 本层所有其他节点
    /// 2. RobustPrune(α=1.0) 选 K 个邻居（多样性骨干）
    /// 3. 双向加边：对每个被选中的邻居 v，把 u 加到 v 的边列表
    ///    若 v 的度数超 K，用 v 的全部当前邻居（含 u）重跑 RobustPrune
    fn build_level_brute_force(
        vectors: &[f32],
        dim: usize,
        layer_nodes: &[u32],
        level_adj: &mut [Vec<u32>],
        k: usize,
    ) {
        for (i, &u) in layer_nodes.iter().enumerate() {
            // 候选 = 本层所有其他节点
            let candidates: Vec<u32> = layer_nodes
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .collect();

            // RobustPrune 选 K 个邻居（α=1.0：最严格多样性，不填充）
            let neighbors = RobustPrune::prune(&candidates, u, vectors, dim, 1.0, k, false);

            // 设置 u 的边
            level_adj[u as usize].clone_from(&neighbors);

            // 双向加边：u → v 已设，现在加 v → u
            for &v in &neighbors {
                Self::add_bidirectional_edge(vectors, dim, level_adj, v, u, k);
            }
        }
    }

    /// 图引导顺序插入：暴力种子 + 顺序扩展
    ///
    /// Phase 1：暴力精建前 min(2000, len) 个节点作为种子图
    /// Phase 2：对剩余每个节点，在当前 level-L 图上贪心搜索 ef_build 个候选，
    ///          RobustPrune 选 K 个邻居，双向加边
    ///
    /// 顺序插入保证：每个节点看到之前所有节点 → 搜索候选有意义 → 零一致性顾虑
    fn build_level_graph_guided(
        vectors: &[f32],
        dim: usize,
        layer_nodes: &[u32],
        level_adj: &mut [Vec<u32>],
        k: usize,
    ) {
        // Phase 1: 暴力精建种子图
        let seed_count = layer_nodes.len().min(BRUTE_FORCE_THRESHOLD);
        Self::build_level_brute_force(vectors, dim, &layer_nodes[..seed_count], level_adj, k);

        // Phase 2: 顺序插入剩余节点
        if layer_nodes.len() <= seed_count {
            return;
        }

        let n = vectors.len() / dim;
        let ef_build = k * 4; // ef >= K * 2 保证搜索宽度
        let mut visited = VisitedTracker::new(n, ef_build);

        // 搜索起点：种子图的第一个节点（已有完整边）
        let search_entry = layer_nodes[0];

        for &u in &layer_nodes[seed_count..] {
            let query = &vectors[u as usize * dim..(u as usize + 1) * dim];

            // 在当前 level-L 图上贪心搜索
            let candidates = Self::search_upper_layer(
                vectors,
                dim,
                level_adj,
                search_entry,
                query,
                ef_build,
                &mut visited,
            );

            // RobustPrune 选 K 个邻居
            let candidate_ids: Vec<u32> = candidates.iter().map(|(id, _)| *id).collect();
            let neighbors = RobustPrune::prune(&candidate_ids, u, vectors, dim, 1.0, k, false);

            // 设置 u 的边
            level_adj[u as usize].clone_from(&neighbors);

            // 双向加边
            for &v in &neighbors {
                Self::add_bidirectional_edge(vectors, dim, level_adj, v, u, k);
            }
        }
    }

    /// 双向加边：将 new_neighbor 添加到 node 的邻接表
    ///
    /// 若度数超过 K，用 node 的 **全部当前邻居**（含 new_neighbor）重跑 RobustPrune。
    /// 不能只传 new_neighbor——否则剪掉的边可能比保留的更好。
    fn add_bidirectional_edge(
        vectors: &[f32],
        dim: usize,
        adj: &mut [Vec<u32>],
        node: u32,
        new_neighbor: u32,
        k: usize,
    ) {
        let neighbors = &mut adj[node as usize];

        // 避免重复添加
        if neighbors.contains(&new_neighbor) {
            return;
        }

        neighbors.push(new_neighbor);

        // 度数超过 K → 用全部当前邻居（含 new_neighbor）重跑 RobustPrune
        if neighbors.len() > k {
            let all_candidates: Vec<u32> = neighbors.clone();
            let pruned = RobustPrune::prune(&all_candidates, node, vectors, dim, 1.0, k, false);
            *neighbors = pruned;
        }
    }

    /// 在上层图上贪心搜索
    ///
    /// 与 Layer 0 的 greedy_search_vec_build 逻辑相同，
    /// 但使用 `Vec<Vec<u32>>` 邻接表而非 HybridBlockedCsr。
    /// 复用 LinearPool（固定容量排序邻居池）和 VisitedTracker。
    fn search_upper_layer(
        vectors: &[f32],
        dim: usize,
        adj: &[Vec<u32>],
        entry: u32,
        query: &[f32],
        ef: usize,
        visited: &mut VisitedTracker,
    ) -> Vec<(u32, f32)> {
        visited.reset();

        let mut pool = LinearPool::new(ef);

        let entry_dist =
            l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
        visited.visit(entry);
        pool.insert(entry, entry_dist);

        while let Some((node, _dist)) = pool.pop() {
            for &neighbor in &adj[node as usize] {
                if neighbor == node {
                    continue; // 跳过自填充
                }
                if visited.visit(neighbor) {
                    let d = l2_simd(
                        query,
                        &vectors[neighbor as usize * dim..(neighbor as usize + 1) * dim],
                    );
                    pool.insert(neighbor, d);
                }
            }
        }

        pool.to_sorted_vec()
    }

    // ========================================================================
    // 多入口候选选择
    // ========================================================================

    /// 选择多入口候选节点
    ///
    /// 候选 = max_level 层的全部节点（通常 ~15 个 for SIFT1M/M=16）
    /// 若超过 MAX_ENTRY_CANDIDATES（50），用 farthest-point sampling 选 50 个
    fn select_entry_candidates(
        vectors: &[f32],
        dim: usize,
        node_levels: &[u8],
        max_level: u8,
        entry_point: u32,
    ) -> Vec<u32> {
        // 收集 max_level 层的所有节点
        let top_nodes: Vec<u32> = (0..node_levels.len() as u32)
            .filter(|&i| node_levels[i as usize] >= max_level)
            .collect();

        if top_nodes.len() <= MAX_ENTRY_CANDIDATES {
            return top_nodes;
        }

        // Farthest-point sampling：从 entry_point 开始，每次选离已选集最远的节点
        let mut selected: Vec<u32> = vec![entry_point];
        let ep_vec =
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim];
        let mut min_dists: Vec<f32> = top_nodes
            .iter()
            .map(|&v| {
                l2_simd(ep_vec, &vectors[v as usize * dim..(v as usize + 1) * dim])
            })
            .collect();

        while selected.len() < MAX_ENTRY_CANDIDATES {
            // 找到离已选集最远的节点
            let (idx, _) = min_dists
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            let new_node = top_nodes[idx];
            selected.push(new_node);

            // 更新每个节点到已选集的最小距离
            let new_vec =
                &vectors[new_node as usize * dim..(new_node as usize + 1) * dim];
            for (i, &v) in top_nodes.iter().enumerate() {
                let d = l2_simd(new_vec, &vectors[v as usize * dim..(v as usize + 1) * dim]);
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }
        }

        selected
    }

    // ========================================================================
    // 搜索初始化
    // ========================================================================

    /// 搜索初始化：从 entry_point 顶层贪心下降到 Layer 0
    ///
    /// 与 Glass `HNSWInitializer::initialize()` 对齐：
    /// 从单一 entry_point 出发，在每一层贪心走（ef=1），直到没有更近的邻居。
    ///
    /// v9.1 修正：多入口实验证明负收益（入口分散导致 Layer 0 起点更远），
    /// 回退到单一 entry_point。多入口候选字段保留（序列化兼容），但不用。
    #[inline]
    pub fn initialize(&self, vectors: &[f32], dim: usize, query: &[f32]) -> (u32, f32) {
        if self.max_level == 0 {
            return (
                self.entry_point,
                l2_simd(
                    query,
                    &vectors[self.entry_point as usize * dim
                        ..(self.entry_point as usize + 1) * dim],
                ),
            );
        }

        // 单一入口：从 entry_point 贪心下降
        let mut u = self.entry_point;
        let mut cur_dist =
            l2_simd(query, &vectors[u as usize * dim..(u as usize + 1) * dim]);

        // 贪心下降：从 max_level 逐层走到 Layer 0
        for level in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let edges = &self.upper_lists[u as usize];
                let start = (level as usize - 1) * self.k;
                if start >= edges.len() {
                    break; // 该节点在此层没有边，跳到下一层
                }
                let end = (start + self.k).min(edges.len());
                for &v in &edges[start..end] {
                    if v == u {
                        continue; // 跳过自填充
                    }
                    let d =
                        l2_simd(query, &vectors[v as usize * dim..(v as usize + 1) * dim]);
                    if d < cur_dist {
                        cur_dist = d;
                        u = v;
                        changed = true;
                    }
                }
            }
        }

        (u, cur_dist)
    }

    // ========================================================================
    // 序列化
    // ========================================================================

    /// 序列化为二进制（紧凑格式）
    ///
    /// 格式（v9 新增 entry_candidates 字段，向后兼容）：
    /// [0..4)   entry_point: u32
    /// [4]      max_level: u8
    /// [5..8)   padding
    /// [8..12)  k: u32
    /// [12..16) n: u32 (node_levels len)
    /// [16..16+n) node_levels
    /// [16+n..) upper_lists: count + entries
    /// [end]    entry_candidates: len + data (v9 新增，旧文件缺省为 [entry_point])
    pub fn serialize_binary(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            16
                + self.node_levels.len()
                + self.upper_lists.len() * 4
                + self.entry_candidates.len() * 4,
        );
        // entry_point: u32
        buf.extend_from_slice(&self.entry_point.to_le_bytes());
        // max_level: u8 + 3 bytes padding
        buf.push(self.max_level);
        buf.extend_from_slice(&[0u8; 3]);
        // k: u32
        buf.extend_from_slice(&(self.k as u32).to_le_bytes());
        // node_levels: u32 len + data
        buf.extend_from_slice(&(self.node_levels.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.node_levels);
        // upper_lists: u32 count + entries
        let non_empty: Vec<(usize, &Vec<u32>)> = self
            .upper_lists
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect();
        buf.extend_from_slice(&(non_empty.len() as u32).to_le_bytes());
        for (node_id, list) in non_empty {
            buf.extend_from_slice(&(node_id as u32).to_le_bytes());
            buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
            let bytes: &[u8] = bytemuck::cast_slice(list);
            buf.extend_from_slice(bytes);
        }
        // entry_candidates: u32 len + data（v9 新增）
        buf.extend_from_slice(&(self.entry_candidates.len() as u32).to_le_bytes());
        let bytes: &[u8] = bytemuck::cast_slice(&self.entry_candidates);
        buf.extend_from_slice(bytes);
        buf
    }

    /// 从二进制反序列化
    ///
    /// 向后兼容：旧格式文件不含 entry_candidates 字段，
    /// 此时默认为 `vec![entry_point]`（单入口，等价于旧行为）
    pub fn deserialize_binary(buf: &[u8]) -> Option<Self> {
        if buf.len() < 16 {
            return None;
        }
        let entry_point = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        let max_level = buf[4];
        let k = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
        let n = u32::from_le_bytes(buf[12..16].try_into().ok()?) as usize;
        if buf.len() < 16 + n {
            return None;
        }
        let node_levels = buf[16..16 + n].to_vec();

        let mut off = 16 + n;
        if buf.len() < off + 4 {
            return None;
        }
        let count = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;

        let mut upper_lists = vec![Vec::new(); n];
        for _ in 0..count {
            if buf.len() < off + 8 {
                return None;
            }
            let node_id = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
            let list_len = u32::from_le_bytes(buf[off + 4..off + 8].try_into().ok()?) as usize;
            off += 8;
            if buf.len() < off + list_len * 4 || node_id >= n {
                return None;
            }
            upper_lists[node_id] = bytemuck::cast_slice(&buf[off..off + list_len * 4]).to_vec();
            off += list_len * 4;
        }

        // entry_candidates（v9 新增，向后兼容：旧文件无此字段）
        let entry_candidates = if buf.len() >= off + 4 {
            let ec_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
            off += 4;
            if buf.len() >= off + ec_len * 4 {
                bytemuck::cast_slice(&buf[off..off + ec_len * 4]).to_vec()
            } else {
                vec![entry_point]
            }
        } else {
            // 旧格式文件：默认单入口
            vec![entry_point]
        };

        Some(Self {
            entry_point,
            max_level,
            k,
            node_levels,
            upper_lists,
            entry_candidates,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| i as f32).collect()
    }

    #[test]
    fn default_navigation_no_overlay() {
        let v = make_vectors(1000, 4);
        let nav = NavigationLayer::new(1000, &v, 4, NavigationConfig::default());
        assert!(!nav.is_overlay_enabled());
        assert!(nav.centroids().is_empty());
    }

    #[test]
    fn overlay_enabled_uses_sqrt_n() {
        let v = make_vectors(10000, 4);
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: None,
        };
        let nav = NavigationLayer::new(10000, &v, 4, config);
        assert!(nav.is_overlay_enabled());
        // √10000 = 100
        assert_eq!(nav.centroids().len(), 100);
    }

    #[test]
    fn overlay_custom_count() {
        let v = make_vectors(1000, 4);
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: Some(50),
        };
        let nav = NavigationLayer::new(1000, &v, 4, config);
        assert_eq!(nav.centroids().len(), 50);
    }

    #[test]
    fn kmeans_centroids_are_valid_nodes() {
        // 验证 k-means 选择的 centroid 都是有效的节点 ID
        let v = make_vectors(100, 4);
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: Some(10),
        };
        let nav = NavigationLayer::new(100, &v, 4, config);
        for &c in nav.centroids() {
            assert!(c < 100, "centroid {} should be < 100", c);
        }
    }

    // ========================================================================
    // LayeredNavigation 测试
    // ========================================================================

    #[test]
    fn layered_nav_serialization_roundtrip() {
        let nav = LayeredNavigation {
            entry_point: 42,
            max_level: 3,
            k: 16,
            node_levels: vec![0, 0, 1, 2, 3, 1, 0, 2],
            upper_lists: vec![
                Vec::new(),
                Vec::new(),
                vec![3, 4, 5],
                vec![4, 7, 2, 5, 1, 6, 0, 3],
                vec![3, 7, 2, 5],
                vec![2, 3, 4],
                Vec::new(),
                vec![4, 3, 2],
            ],
            entry_candidates: vec![42, 4],
        };

        let bytes = nav.serialize_binary();
        let restored = LayeredNavigation::deserialize_binary(&bytes).expect("deserialize");

        assert_eq!(restored.entry_point, 42);
        assert_eq!(restored.max_level, 3);
        assert_eq!(restored.k, 16);
        assert_eq!(restored.node_levels, nav.node_levels);
        assert_eq!(restored.upper_lists, nav.upper_lists);
        assert_eq!(restored.entry_candidates, nav.entry_candidates);
    }

    #[test]
    fn layered_nav_empty_roundtrip() {
        let nav = LayeredNavigation::empty(0, 16);
        let bytes = nav.serialize_binary();
        let restored = LayeredNavigation::deserialize_binary(&bytes).expect("deserialize");
        assert_eq!(restored.max_level, 0);
        assert_eq!(restored.k, 16);
        assert!(restored.node_levels.is_empty());
    }

    #[test]
    fn layered_nav_backward_compat_no_entry_candidates() {
        // 模拟旧格式：手动构造不含 entry_candidates 的 buffer
        let nav = LayeredNavigation {
            entry_point: 7,
            max_level: 2,
            k: 8,
            node_levels: vec![0, 0, 1, 2],
            upper_lists: vec![
                Vec::new(),
                Vec::new(),
                vec![3, 1],
                vec![2, 0],
            ],
            entry_candidates: Vec::new(), // 不写入
        };

        // 手动序列化不含 entry_candidates
        let mut buf = Vec::new();
        buf.extend_from_slice(&7u32.to_le_bytes()); // entry_point
        buf.push(2u8); // max_level
        buf.extend_from_slice(&[0u8; 3]); // padding
        buf.extend_from_slice(&8u32.to_le_bytes()); // k
        buf.extend_from_slice(&4u32.to_le_bytes()); // n
        buf.extend_from_slice(&[0, 0, 1, 2]); // node_levels
        // upper_lists: 2 non-empty
        buf.extend_from_slice(&2u32.to_le_bytes());
        // node 2: [3, 1]
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(bytemuck::cast_slice(&[3u32, 1u32]));
        // node 3: [2, 0]
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(bytemuck::cast_slice(&[2u32, 0u32]));
        // 无 entry_candidates 字段

        let restored = LayeredNavigation::deserialize_binary(&buf).expect("deserialize");
        assert_eq!(restored.entry_point, 7);
        // 旧格式 → 默认 entry_candidates = [entry_point]
        assert_eq!(restored.entry_candidates, vec![7]);
    }

    #[test]
    fn add_bidirectional_edge_prunes_correctly() {
        // 验证双向加边：当度数超 K 时，用全部邻居重跑 RobustPrune
        //
        // 构造向量使得节点 4 比节点 3 更近于节点 0：
        //   node 0 = [0, 0, 0, 0]
        //   node 1 = [1, 0, 0, 0]  dist² = 1
        //   node 2 = [0, 1, 0, 0]  dist² = 1
        //   node 3 = [9, 9, 9, 9]  dist² = 324（远）
        //   node 4 = [0, 0, 1, 0]  dist² = 1（近）
        // K=3，初始 [1,2,3]，加 4 → [1,2,3,4] → prune → 应保留 [1,2,4] 或 [1,2,3] 中较近的
        let v: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // node 0
            1.0, 0.0, 0.0, 0.0, // node 1
            0.0, 1.0, 0.0, 0.0, // node 2
            9.0, 9.0, 9.0, 9.0, // node 3 (远)
            0.0, 0.0, 1.0, 0.0, // node 4 (近)
        ];
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); 5];

        // 节点 0 已有 3 个邻居 [1, 2, 3]，K=3
        adj[0] = vec![1, 2, 3];

        // 添加新邻居 4 → 度数变 4 > K=3 → 触发 RobustPrune
        LayeredNavigation::add_bidirectional_edge(&v, 4, &mut adj, 0, 4, 3);

        // 验证：度数 ≤ K
        assert!(adj[0].len() <= 3, "degree should be <= K after prune");
        // 验证：远的节点 3 应被剪掉，近的节点 4 应保留
        assert!(
            adj[0].contains(&4),
            "closer neighbor 4 should survive prune, got {:?}",
            adj[0]
        );
        assert!(
            !adj[0].contains(&3),
            "far neighbor 3 should be pruned, got {:?}",
            adj[0]
        );
    }

    #[test]
    fn search_upper_layer_finds_neighbors() {
        // 验证上层图贪心搜索能找到近邻
        let v = make_vectors(100, 4);
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); 100];

        // 简单线性图：i → [i+1, i+2]
        for i in 0..98 {
            adj[i] = vec![(i + 1) as u32, (i + 2) as u32];
        }
        adj[98] = vec![99];
        adj[99] = vec![];

        let n = 100;
        let mut visited = VisitedTracker::new(n, 10);
        let query = &v[90 * 4..91 * 4]; // 查询 = 节点 90

        let results = LayeredNavigation::search_upper_layer(
            &v, 4, &adj, 0, query, 10, &mut visited,
        );

        // 应该找到节点 90 附近的节点
        assert!(!results.is_empty());
        // 节点 90 本身应该在结果中（距离=0）
        let found_90 = results.iter().any(|(id, dist)| *id == 90 && *dist == 0.0);
        assert!(found_90, "should find node 90 with dist=0");
    }
}
