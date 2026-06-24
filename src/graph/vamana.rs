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
use crate::build::ChaCha8Rng;
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
    /// 构建期软上限 R_soft = 1.5 × R_max（设计文档第四层：延迟剪枝）
    pub r_soft: usize,
    /// 最大迭代轮数
    pub max_iterations: usize,
}

impl Default for VamanaBuildConfig {
    fn default() -> Self {
        let r_max = 64;
        Self {
            alpha: 1.2,
            l_build: 200,
            r_max,
            r_soft: (r_max as f32 * 1.5) as usize, // 设计文档：1.5 × R_max
            max_iterations: 1,
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

        // 2. 迭代优化：并行 greedy_search + RobustPrune，顺序写入
        // Vamana 论文 two passes：第一轮 α=1.0（连通性），第二轮 α=config.alpha（长程边）
        for iter in 0..config.max_iterations {
            let order = Self::permutation(n, rng);
            let progress = AtomicUsize::new(0);
            let progress_interval = (n / 20).max(1);
            let alpha = if iter == 0 { 1.0 } else { config.alpha };
            eprintln!("[build] iter {}/{} alpha={}", iter + 1, config.max_iterations, alpha);

            // 并行计算每个节点的新邻居（storage 只读，安全并行）
            let new_neighbors: Vec<(u32, Vec<u32>)> = order
                .par_iter()
                .map(|&node_id| {
                    let idx = progress.fetch_add(1, Ordering::Relaxed);
                    if idx > 0 && idx % progress_interval == 0 {
                        eprintln!("[build] {}/{} ({}%)", idx, n, idx * 100 / n);
                    }
                    // 标准 break greedy search，返回 (top-L, all_visited)
                    // Vamana/DiskANN 论文：用 visited set 做 RobustPrune（不是 top-L）
                    let (_top, visited) = Self::greedy_search(
                        vectors, dim, &storage, entry_point, node_id, config.l_build,
                    );
                    let pruned = RobustPrune::prune(
                        &visited, node_id, vectors, dim, alpha, config.r_max,
                    );
                    (node_id, pruned)
                })
                .collect();

            // 顺序写入邻接表（避免锁竞争）+ 真正的延迟剪枝
            for (node_id, pruned) in new_neighbors {
                Self::connect_bidirectional(
                    &mut storage, node_id, &pruned, vectors, dim, config,
                );
            }
        }

        // 3. 全局 final prune 到 R_max（用 RobustPrune，不是 truncate）
        Self::final_prune(&mut storage, vectors, dim, config.alpha, config.r_max);

        VamanaGraph { storage, entry_point, dim, n }
    }

    /// 从已有存储构造（用于 RP-Tuning 生成变体）
    pub fn from_storage(storage: HybridBlockedCsr, entry_point: u32, dim: usize) -> Self {
        let n = storage.len();
        Self { storage, entry_point, dim, n }
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
        for iter in 0..config.max_iterations {
            let order = Self::permutation(n, rng);
            let alpha = if iter == 0 { 1.0 } else { config.alpha };
            let progress = AtomicUsize::new(0);
            let progress_interval = (n / 20).max(1);
            eprintln!("[build_qa] iter {}/{} alpha={} beta={}", iter + 1, config.max_iterations, alpha, qa_config.beta);

            // 并行计算每个节点的新邻居（storage 只读，安全并行）
            let new_neighbors: Vec<(u32, Vec<u32>)> = order
                .par_iter()
                .map(|&node_id| {
                    let idx = progress.fetch_add(1, Ordering::Relaxed);
                    if idx > 0 && idx % progress_interval == 0 {
                        eprintln!("[build_qa] {}/{} ({}%)", idx, n, idx * 100 / n);
                    }
                    let (_top, visited) = Self::greedy_search(
                        vectors, dim, &storage, entry_point, node_id, config.l_build,
                    );
                    // 量化感知剪枝（β > 0 时考虑量化误差），用 visited set
                    // two-pass: iter 0 用 α=1.0，iter 1 用 config.alpha
                    let iter_qa_config = QuantAwarePruneConfig {
                        alpha,
                        ..*qa_config
                    };
                    let pruned = QuantAwareRobustPrune::prune(
                        &visited,
                        node_id,
                        vectors,
                        dim,
                        &error_fn,
                        &iter_qa_config,
                    );
                    (node_id, pruned)
                })
                .collect();

            // 顺序写入邻接表（避免锁竞争）+ 真正的延迟剪枝
            for (node_id, pruned) in new_neighbors {
                Self::connect_bidirectional(
                    &mut storage, node_id, &pruned, vectors, dim, config,
                );
            }
        }

        Self::final_prune(&mut storage, vectors, dim, config.alpha, config.r_max);
        VamanaGraph { storage, entry_point, dim, n }
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

        // 随机连接每个节点到若干邻居
        // 修复：原实现对每个节点 shuffle 整个 indices（O(n²)），1M 节点需要数小时
        // 改为随机采样 neighbor_count 个不同节点，用 HashSet 去重（O(1) 查找）
        let neighbor_count = config.r_max;
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
            }
        }
        entry
    }

    /// 计算 medoid（离质心最近的点）
    ///
    /// Vamana/DiskANN 论文要求 entry_point 用 medoid，不是随机点。
    /// 质心法：先算所有向量的均值（质心），再找离质心最近的点。
    /// 复杂度 O(n*dim)，1M 向量也很快。
    fn compute_medoid(vectors: &[f32], dim: usize, n: usize) -> u32 {
        if n == 0 {
            return 0;
        }
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

        // 候选集：最小堆（距离小的优先）
        use std::cmp::Reverse;
        let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
        // 结果集：最大堆（距离大的在堆顶，便于淘汰）
        let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::new();

        let entry_dist = l2_simd(
            query,
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        candidates.push(Reverse((OrderedF32(entry_dist), entry_point)));
        visited.visit(entry_point);

        while let Some(Reverse((dist, node))) = candidates.pop() {
            // 标准终止条件：结果集已满 l，且候选最小距离 > 结果集最差距离
            if results.len() >= l {
                if let Some(&(worst, _)) = results.peek() {
                    if dist.0 > worst.0 {
                        break;
                    }
                }
            }

            results.push((dist, node));
            if results.len() > l {
                results.pop();
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
        }

        let top_results: Vec<u32> = results.into_iter().map(|(_, id)| id).collect();
        let all_visited: Vec<u32> = visited.visited_nodes().to_vec();
        (top_results, all_visited)
    }

    /// 贪心搜索（复用 VisitedTracker，零分配热路径）
    ///
    /// 设计文档 F.2：bumpalo arena 作用域——构建期临时候选集
    /// 此方法复用外部传入的 VisitedTracker，避免每次搜索分配 1MB visited 数组
    /// 适用于查询热路径（GraphSearcher::search）
    pub fn greedy_search_vec_reuse(
        vectors: &[f32],
        dim: usize,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query: &[f32],
        l: usize,
        visited: &mut VisitedTracker,
    ) -> Vec<u32> {
        // 复用 visited：reset 是 O(V) 而非 O(N)
        visited.reset();

        // 候选集：最小堆
        use std::cmp::Reverse;
        let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
        // 结果集：最大堆
        let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

        let entry_dist = l2_simd(
            query,
            &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        candidates.push(Reverse((OrderedF32(entry_dist), entry_point)));
        visited.visit(entry_point);

        while let Some(Reverse((dist, node))) = candidates.pop() {
            if results.len() >= l {
                if let Some(&(worst, _)) = results.peek() {
                    if dist.0 > worst.0 {
                        break;
                    }
                }
            }

            results.push((dist, node));
            if results.len() > l {
                results.pop();
            }

            for &neighbor in storage.neighbors(node) {
                if visited.visit(neighbor) {
                    let d = l2_simd(
                        query,
                        &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                    );
                    candidates.push(Reverse((OrderedF32(d), neighbor)));
                }
            }
        }

        results.into_iter().map(|(_, id)| id).collect()
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
    /// Vamana 论文：加反向边后，如果邻居度数超过 R_soft，立即对该邻居做 RobustPrune
    /// 之前是空实现（注释说"final prune 统一处理"），导致 overflow 膨胀 + final_prune
    /// 用 truncate 截断，破坏 RobustPrune 性质
    fn connect_bidirectional(
        storage: &mut HybridBlockedCsr,
        node: u32,
        neighbors: &[u32],
        vectors: &[f32],
        dim: usize,
        config: &VamanaBuildConfig,
    ) {
        for &nb in neighbors {
            storage.add_edge(node, nb);
            storage.add_edge(nb, node);
            // 反向边：如果 nb 的度数超过 R_soft，对 nb 做 RobustPrune
            if storage.degree(nb) > config.r_soft {
                let (main, overflow) = storage.neighbors_full(nb);
                let mut all: Vec<u32> = main.to_vec();
                all.extend_from_slice(overflow);
                let pruned = RobustPrune::prune(
                    &all, nb, vectors, dim, config.alpha, config.r_max,
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
    ) {
        for node in 0..storage.len() as u32 {
            let (main, overflow) = storage.neighbors_full(node);
            if main.len() + overflow.len() <= r_max {
                continue;
            }
            let mut all: Vec<u32> = main.to_vec();
            all.extend_from_slice(overflow);
            let pruned = RobustPrune::prune(&all, node, vectors, dim, alpha, r_max);
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
///   [36..)    HybridBlockedCsr body（见 src/memory/graph.rs）
impl crate::memory::serialize::Serializable for VamanaGraph {
    fn serialize(&self) -> Vec<u8> {
        use crate::memory::serialize::{IndexHeader, crc32};

        // 序列化文件体
        let mut body: Vec<u8> = Vec::with_capacity(20 + self.storage.main_block_bytes());
        body.extend_from_slice(&(self.n as u64).to_le_bytes());
        body.extend_from_slice(&(self.dim as u64).to_le_bytes());
        body.extend_from_slice(&self.entry_point.to_le_bytes());
        // HybridBlockedCsr body
        let storage_bytes = crate::memory::serialize::Serializable::serialize(&self.storage);
        body.extend_from_slice(&storage_bytes);

        // 计算校验和并构造文件头
        let crc = crc32(&body);
        let header = IndexHeader::new(crc);
        let header_bytes = header.to_bytes();

        // 拼接：header + body
        let mut result = Vec::with_capacity(header_bytes.len() + body.len());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&body);
        result
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, crate::memory::serialize::SerializeError> {
        use crate::memory::serialize::{IndexHeader, HEADER_SIZE};
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

        // 反序列化 HybridBlockedCsr
        let storage = HybridBlockedCsr::deserialize(&body[20..])?;

        // 校验一致性
        if storage.len() != n {
            return Err(crate::memory::serialize::SerializeError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("n mismatch: header={}, storage={}", n, storage.len()),
            )));
        }

        Ok(Self::from_storage(storage, entry_point, dim))
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
}

impl<'a> GraphSearcher<'a> {
    /// 创建搜索器
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
        }
    }

    /// 搜索最近邻
    ///
    /// 返回 (节点ID, 距离) 列表，按距离升序
    ///
    /// 使用标准 break 模式（Vamana 论文标准 greedy search）
    /// 复用预分配的 VisitedTracker，零堆分配热路径
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let candidates = VamanaGraph::greedy_search_vec_reuse(
            self.vectors,
            self.dim,
            self.graph.storage(),
            self.graph.entry_point(),
            query,
            self.ef_search,
            &mut self.visited,
        );

        // 按距离排序，取 top-k
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|id| {
                let v = &self.vectors[id as usize * self.dim..(id as usize + 1) * self.dim];
                (id, l2_simd(query, v))
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
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
use super::robust_prune::RobustPrune;
use super::quant_aware_prune::{QuantAwareRobustPrune, QuantAwarePruneConfig};

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
