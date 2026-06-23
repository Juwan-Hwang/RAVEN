//! RobustPrune（标准实现，β=0）
//!
//! 设计文档第三层：
//! RobustPrune 是 Vamana/DiskANN 的核心剪枝算法
//! α 参数控制剪枝激进度：α 越大保留的长程导航边越多
//!
//! 算法：
//! 1. 候选集按距离升序排序
//! 2. 从最近的开始，依次加入结果集
//! 3. 对每个候选 p'，若存在已加入的 p 使得 α × dist(p, p') ≤ dist(p', q)，
//!    则跳过 p'（被 p "遮挡"）
//! 4. 直到结果集达到 R_max

use crate::distance::l2_simd;

/// RobustPrune 配置
#[derive(Debug, Clone)]
pub struct RobustPruneConfig {
    /// α 参数：剪枝激进度
    /// 设计文档第三层 α 三段式①：全局 α（构建时固定）
    pub alpha: f32,
    /// 最大出度 R_max
    pub r_max: usize,
}

impl Default for RobustPruneConfig {
    fn default() -> Self {
        Self { alpha: 1.2, r_max: 64 }
    }
}

/// 标准 RobustPrune（β=0）
///
/// 设计文档第三层：第一阶段固定 β=0（标准 RobustPrune），扫 α baseline
pub struct RobustPrune;

impl RobustPrune {
    /// 执行 RobustPrune 剪枝
    ///
    /// candidates: 候选邻居集（节点 ID）
    /// query_node: 被剪枝的节点
    /// vectors: 扁平存储的向量
    /// dim: 维度
    /// alpha: α 参数
    /// r_max: 最大出度
    ///
    /// 返回剪枝后的邻居列表
    pub fn prune(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        alpha: f32,
        r_max: usize,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let query = &vectors[query_node as usize * dim..(query_node as usize + 1) * dim];

        // 计算每个候选到 query 的距离
        let mut scored: Vec<(f32, u32)> = candidates
            .iter()
            .filter(|&&c| c != query_node)
            .map(|&c| {
                let v = &vectors[c as usize * dim..(c as usize + 1) * dim];
                (l2_simd(query, v), c)
            })
            .collect();

        // 按距离升序排序
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<u32> = Vec::with_capacity(r_max);
        let mut pruned = vec![false; scored.len()];

        for i in 0..scored.len() {
            if pruned[i] {
                continue;
            }
            if result.len() >= r_max {
                break;
            }
            result.push(scored[i].1);

            // 对后续候选，若被当前节点遮挡则剪掉
            let p_vec = &vectors[scored[i].1 as usize * dim..(scored[i].1 as usize + 1) * dim];
            for j in (i + 1)..scored.len() {
                if pruned[j] {
                    continue;
                }
                let q_vec = &vectors[scored[j].1 as usize * dim..(scored[j].1 as usize + 1) * dim];
                let dist_p_q = l2_simd(p_vec, q_vec);
                // α × dist(p, p') ≤ dist(p', q) → p' 被 p 遮挡
                // α 越大，条件越难满足，保留更多长程边
                if alpha * dist_p_q <= scored[j].0 {
                    pruned[j] = true;
                }
            }
        }

        result
    }

    /// 带配置的剪枝
    pub fn prune_with_config(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        config: &RobustPruneConfig,
    ) -> Vec<u32> {
        Self::prune(candidates, query_node, vectors, dim, config.alpha, config.r_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| i as f32).collect()
    }

    #[test]
    fn prune_empty_candidates() {
        let v = make_vectors(10, 4);
        let result = RobustPrune::prune(&[], 0, &v, 4, 1.0, 8);
        assert!(result.is_empty());
    }

    #[test]
    fn prune_respects_r_max() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 5);
        assert!(result.len() <= 5);
    }

    #[test]
    fn prune_excludes_self() {
        let v = make_vectors(10, 4);
        let candidates = vec![0, 1, 2, 3];
        let result = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 8);
        assert!(!result.contains(&0));
    }

    #[test]
    fn prune_alpha_larger_keeps_more_long_edges() {
        // α 越大，剪枝越宽松，保留更多候选
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result_small = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 10);
        let result_large = RobustPrune::prune(&candidates, 0, &v, 4, 2.0, 10);
        // α=2.0 应保留 >= α=1.0 的数量
        assert!(result_large.len() >= result_small.len());
    }
}
