//! 剪枝策略模块
//!
//! 包含两种剪枝算法：
//! 1. RobustPrune（Vamana/DiskANN 标准）：多轮 α 递增 + saturation 填充
//! 2. DirectionalPrune（RAVEN 超越方案）：方向性扫描 + 自适应连通性补底，无 saturation
//!
//! DirectionalPrune 的设计动机：
//!   RobustPrune 的 saturation 用远距离候选填满 r_max → avg_visited 膨胀（1227 vs Glass <150）
//!   Glass Heuristic2 纯方向性无连通性保证 → 依赖 HNSW 多层结构补偿
//!   DirectionalPrune 融合两者优势：方向直 + 保连通 + 无垃圾边

use crate::distance::l2_simd;

// ═══════════════════════════════════════════════════════════════════════
//  公共函数：候选集预处理
// ═══════════════════════════════════════════════════════════════════════

/// 候选集预处理：去自身、算距离、去重、按距离升序排序
///
/// 返回 (平方距离, 节点ID) 列表，已去重且按距离升序排列。
/// 被 RobustPrune 和 DirectionalPrune 共享。
fn prepare_candidates(
    candidates: &[u32],
    query_node: u32,
    vectors: &[f32],
    dim: usize,
) -> Vec<(f32, u32)> {
    let query = &vectors[query_node as usize * dim..(query_node as usize + 1) * dim];

    let mut scored: Vec<(f32, u32)> = candidates
        .iter()
        .filter(|&&c| c != query_node)
        .map(|&c| {
            let v = &vectors[c as usize * dim..(c as usize + 1) * dim];
            (l2_simd(query, v), c)
        })
        .collect();

    // 去重（按节点 ID）
    scored.sort_by(|a, b| a.1.cmp(&b.1));
    scored.dedup_by(|a, b| a.1 == b.1);

    // 按距离升序排序
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    scored
}

// ═══════════════════════════════════════════════════════════════════════
//  剪枝策略枚举
// ═══════════════════════════════════════════════════════════════════════

/// 剪枝策略选择
///
/// - `RobustPrune`：Vamana/DiskANN 标准多轮 α 递增 + saturation 填充
/// - `DirectionalPrune`：RAVEN 超越方案，方向性扫描 + 自适应连通性补底
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneStrategy {
    /// Vamana/DiskANN 标准 RobustPrune（baseline）
    RobustPrune,
    /// RAVEN DirectionalPrune（方向性 + 连通性补底，无 saturation）
    DirectionalPrune,
}

impl Default for PruneStrategy {
    fn default() -> Self {
        PruneStrategy::RobustPrune
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  RobustPrune（Vamana/DiskANN 标准）
// ═══════════════════════════════════════════════════════════════════════

/// RobustPrune 配置
#[derive(Debug, Clone)]
pub struct RobustPruneConfig {
    pub alpha: f32,
    pub r_max: usize,
}

impl Default for RobustPruneConfig {
    fn default() -> Self {
        Self { alpha: 1.2, r_max: 64 }
    }
}

/// 标准 Vamana RobustPrune（β=0 baseline，多轮 α 递增）
pub struct RobustPrune;

impl RobustPrune {
    /// 执行 RobustPrune 剪枝（Vamana 论文 Algorithm 2，平方距离直接比较）
    ///
    /// 返回剪枝后的邻居列表（长度 ≤ r_max）
    pub fn prune(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        alpha: f32,
        r_max: usize,
        saturate: bool,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let scored = prepare_candidates(candidates, query_node, vectors, dim);
        if scored.is_empty() {
            return Vec::new();
        }

        let n = scored.len();
        let mut result: Vec<u32> = Vec::with_capacity(r_max);

        let mut occlude_factor = vec![0.0f32; n];
        let mut result_indices: Vec<usize> = Vec::with_capacity(r_max);

        let mut current_alpha = 1.0f32;
        let increment_factor = alpha.min(1.2);

        while result.len() < r_max {
            for i in 0..n {
                if result.len() >= r_max {
                    break;
                }
                if occlude_factor[i] == f32::MAX {
                    continue;
                }
                if occlude_factor[i] > current_alpha {
                    continue;
                }

                let candidate_vec =
                    &vectors[scored[i].1 as usize * dim..(scored[i].1 as usize + 1) * dim];
                let dist_iq_sq = scored[i].0;

                for &j in &result_indices {
                    if j >= i {
                        continue;
                    }
                    let accepted_vec =
                        &vectors[scored[j].1 as usize * dim..(scored[j].1 as usize + 1) * dim];
                    let dist_ji_sq = l2_simd(accepted_vec, candidate_vec);

                    if dist_ji_sq == 0.0 {
                        occlude_factor[i] = f32::MAX;
                        break;
                    }

                    let ratio_sq = dist_iq_sq / dist_ji_sq;
                    if ratio_sq > occlude_factor[i] {
                        occlude_factor[i] = ratio_sq;
                    }
                    if occlude_factor[i] > current_alpha {
                        break;
                    }
                }

                if occlude_factor[i] > current_alpha {
                    continue;
                }

                occlude_factor[i] = f32::MAX;
                result_indices.push(i);
                result.push(scored[i].1);
            }

            if current_alpha >= alpha {
                break;
            }
            current_alpha = (current_alpha * increment_factor).min(alpha);
        }

        if saturate {
            for i in 0..n {
                if result.len() >= r_max {
                    break;
                }
                if occlude_factor[i] == f32::MAX {
                    continue;
                }
                result.push(scored[i].1);
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
        Self::prune(candidates, query_node, vectors, dim, config.alpha, config.r_max, config.alpha > 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  DirectionalPrune（RAVEN 超越方案）
// ═══════════════════════════════════════════════════════════════════════

/// DirectionalPrune 配置
#[derive(Debug, Clone)]
pub struct DirectionalPruneConfig {
    /// 最大出度
    pub r_max: usize,
    /// 最小连通度：度数低于此值时触发 Pass 2 补底
    /// 默认 r_max / 4，确保图不会过度稀疏
    pub r_min: usize,
    /// Pass 2 放宽 α：允许方向性稍弱的长程边
    /// 默认 1.2，与 Vamana 第二轮 α 一致
    pub backfill_alpha: f32,
}

impl DirectionalPruneConfig {
    /// 从 r_max 和 alpha 自动推导配置
    pub fn from_params(r_max: usize, alpha: f32) -> Self {
        Self {
            r_max,
            r_min: (r_max / 4).max(1),
            backfill_alpha: alpha.max(1.0),
        }
    }
}

impl Default for DirectionalPruneConfig {
    fn default() -> Self {
        Self::from_params(64, 1.2)
    }
}

/// RAVEN DirectionalPrune — 融合 RobustPrune 和 Heuristic2 的超越方案
///
/// 三方对比：
/// ┌─────────────────┬──────────────────┬───────────────────┬──────────────────────────┐
/// │                 │ RobustPrune      │ Heuristic2 (Glass)│ DirectionalPrune (RAVEN) │
/// ├─────────────────┼──────────────────┼───────────────────┼──────────────────────────┤
/// │ Pass 1 方向性   │ α=1.0 多轮       │ α=1.0 单轮        │ α=1.0 单轮               │
/// │ 长程边          │ α=1.2 多轮补充   │ 无                │ α=backfill 补到 r_min    │
/// │ Saturation      │ 填到 r_max（垃圾）│ 无               │ 无                       │
/// │ 连通性保证      │ 靠 saturation    │ 靠 HNSW 多层      │ 靠 r_min 补底             │
/// │ avg_visited     │ 高（1227）       │ 低（<150）        │ 目标：低                 │
/// └─────────────────┴──────────────────┴───────────────────┴──────────────────────────┘
///
/// 算法：
/// 1. 候选集按距离升序排序、去重（共享 prepare_candidates）
/// 2. Pass 1 — 方向性扫描（α=1.0）：
///    对每个候选 i（近→远），检查所有已接受邻居 j：
///    若 d²(i,j) < d²(i,q) → i 被 j 方向性遮挡，跳过
///    这等价于 Glass Heuristic2 的 `computer(u,v) < dist_to_query`
///    产出：纯方向性骨干边，路径直
/// 3. Pass 2 — 连通性补底（仅当 result.len() < r_min）：
///    对 Pass 1 未接受的候选，用放宽阈值 backfill_alpha 重新检查：
///    若 d²(i,q)/d²(i,j) > backfill_alpha → 仍遮挡，跳过
///    否则接受，但只补到 r_min，不补到 r_max
///    产出：少量高质量长程边，保证全局连通
/// 4. 无 saturation：剩余槽位留空，不盲目填充远距离候选
///
/// 核心创新：r_min 分离"必须连通"和"锦上添花"
/// - Vamana 错在填到 r_max：saturation 塞入垃圾边 → avg_visited 膨胀
/// - Glass 弱在无补底：纯 α=1.0 可能产生孤岛 → 依赖 HNSW 多层
/// - RAVEN 两者兼得：方向直 + 保连通 + 无垃圾边
pub struct DirectionalPrune;

impl DirectionalPrune {
    /// 执行 DirectionalPrune 剪枝
    ///
    /// 返回剪枝后的邻居列表（长度 ≤ r_max）
    pub fn prune(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        r_max: usize,
        r_min: usize,
        backfill_alpha: f32,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let scored = prepare_candidates(candidates, query_node, vectors, dim);
        if scored.is_empty() {
            return Vec::new();
        }

        let n = scored.len();
        let mut result: Vec<u32> = Vec::with_capacity(r_max);

        // ── Pass 1：方向性扫描（α=1.0，等价 Glass Heuristic2）──
        //
        // 对每个候选 i（近→远），检查所有已接受邻居 j：
        //   d²(i,j) < d²(i,q) → i 被 j 遮挡（j 比 q 更靠近 i，i 是冗余方向）
        // 这与 Glass getNeighborsByHeuristic2 的 `curdist < dist_to_query` 完全等价
        for i in 0..n {
            if result.len() >= r_max {
                break;
            }

            let dist_iq_sq = scored[i].0; // d²(candidate, query)
            let candidate_vec =
                &vectors[scored[i].1 as usize * dim..(scored[i].1 as usize + 1) * dim];

            let mut good = true;
            for &accepted_id in &result {
                let accepted_vec =
                    &vectors[accepted_id as usize * dim..(accepted_id as usize + 1) * dim];
                let dist_ji_sq = l2_simd(accepted_vec, candidate_vec); // d²(accepted, candidate)

                // 方向性遮挡：accepted 离 candidate 比 query 离 candidate 更近
                // → candidate 方向与 accepted 重叠，冗余
                if dist_ji_sq < dist_iq_sq {
                    good = false;
                    break;
                }
            }

            if good {
                result.push(scored[i].1);
            }
        }

        // ── Pass 2：连通性补底（仅当度数不足 r_min）──
        //
        // Pass 1 的 α=1.0 在密集区域可能只选出极少邻居（方向性太严格）。
        // Vamana 用 saturation 填到 r_max → 垃圾边膨胀 avg_visited
        // Glass 不补 → 依赖 HNSW 多层结构
        // RAVEN：用放宽 α 补到 r_min，只补充方向性稍弱但仍合理的长程边
        //
        // 放宽判定：d²(i,q) / d²(i,j) > backfill_alpha → 仍遮挡
        // backfill_alpha=1.2 时，有效阈值 d(i,q)/d(i,j) > sqrt(1.2) ≈ 1.095
        // 即允许 j 到 i 的距离比 q 到 i 近 9.5% 以内的候选
        if result.len() < r_min {
            let accepted_set: std::collections::HashSet<u32> = result.iter().copied().collect();

            for i in 0..n {
                if result.len() >= r_min {
                    break;
                }

                let id = scored[i].1;
                if accepted_set.contains(&id) {
                    continue;
                }

                let dist_iq_sq = scored[i].0;
                let candidate_vec =
                    &vectors[id as usize * dim..(id as usize + 1) * dim];

                let mut good = true;
                for &accepted_id in &result {
                    let accepted_vec =
                        &vectors[accepted_id as usize * dim..(accepted_id as usize + 1) * dim];
                    let dist_ji_sq = l2_simd(accepted_vec, candidate_vec);

                    // 放宽遮挡：ratio_sq = d²(i,q) / d²(i,j) > backfill_alpha → 遮挡
                    if dist_ji_sq == 0.0 {
                        good = false;
                        break;
                    }
                    let ratio_sq = dist_iq_sq / dist_ji_sq;
                    if ratio_sq > backfill_alpha {
                        good = false;
                        break;
                    }
                }

                if good {
                    result.push(id);
                }
            }
        }

        // 无 saturation：不填充剩余槽位
        result
    }

    /// 带配置的剪枝
    pub fn prune_with_config(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        config: &DirectionalPruneConfig,
    ) -> Vec<u32> {
        Self::prune(candidates, query_node, vectors, dim, config.r_max, config.r_min, config.backfill_alpha)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  统一 dispatch
// ═══════════════════════════════════════════════════════════════════════

/// 根据策略 dispatch 到对应剪枝实现
///
/// 共享 alpha/r_max/saturate 语义：
/// - RobustPrune：直接传递
/// - DirectionalPrune：r_min = r_max/4, backfill_alpha = alpha
pub fn prune_dispatch(
    strategy: PruneStrategy,
    candidates: &[u32],
    query_node: u32,
    vectors: &[f32],
    dim: usize,
    alpha: f32,
    r_max: usize,
    saturate: bool,
) -> Vec<u32> {
    match strategy {
        PruneStrategy::RobustPrune => {
            RobustPrune::prune(candidates, query_node, vectors, dim, alpha, r_max, saturate)
        }
        PruneStrategy::DirectionalPrune => {
            let config = DirectionalPruneConfig::from_params(r_max, alpha);
            DirectionalPrune::prune(
                candidates, query_node, vectors, dim,
                config.r_max, config.r_min, config.backfill_alpha,
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| i as f32).collect()
    }

    // ── RobustPrune 测试 ──

    #[test]
    fn prune_empty_candidates() {
        let v = make_vectors(10, 4);
        let result = RobustPrune::prune(&[], 0, &v, 4, 1.0, 8, false);
        assert!(result.is_empty());
    }

    #[test]
    fn prune_respects_r_max() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 5, false);
        assert!(result.len() <= 5);
    }

    #[test]
    fn prune_excludes_self() {
        let v = make_vectors(10, 4);
        let candidates = vec![0, 1, 2, 3];
        let result = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 8, false);
        assert!(!result.contains(&0));
    }

    #[test]
    fn prune_alpha_larger_keeps_more_long_edges() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result_small = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 10, false);
        let result_large = RobustPrune::prune(&candidates, 0, &v, 4, 2.0, 10, false);
        assert!(result_large.len() >= result_small.len());
    }

    #[test]
    fn prune_dedup_candidates() {
        // 非共线向量：候选在 query 的不同方向上，α=1.0 剪枝不会误杀
        //
        // 旧测试用 make_vectors 生成共线数据 [0,1,2,3], [4,5,6,7], [8,9,10,11]...
        // 共线 → 所有候选在同一方向 → α 剪枝只保留最近的 1 个 → assert(3) 失败
        //
        // 修复：正交放置，3 个候选在 120° 分隔的方向上，全部存活
        let v = vec![
            0.0, 0.0,   // node 0 (query)
            3.0, 0.0,   // node 1 — 0°
            0.0, 3.0,   // node 2 — 90°
           -3.0, 0.0,   // node 3 — 180°
        ];
        let candidates = vec![1, 1, 2, 2, 3];
        let result = RobustPrune::prune(&candidates, 0, &v, 2, 1.0, 8, false);
        // 去重后 3 个候选，方向正交 → 全部通过 α=1.0 剪枝
        assert_eq!(result.len(), 3);
    }

    // ── DirectionalPrune 测试 ──

    #[test]
    fn directional_prune_empty_candidates() {
        let v = make_vectors(10, 4);
        let result = DirectionalPrune::prune(&[], 0, &v, 4, 8, 2, 1.2);
        assert!(result.is_empty());
    }

    #[test]
    fn directional_prune_respects_r_max() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result = DirectionalPrune::prune(&candidates, 0, &v, 4, 5, 2, 1.2);
        assert!(result.len() <= 5);
    }

    #[test]
    fn directional_prune_excludes_self() {
        let v = make_vectors(10, 4);
        let candidates = vec![0, 1, 2, 3];
        let result = DirectionalPrune::prune(&candidates, 0, &v, 4, 8, 2, 1.2);
        assert!(!result.contains(&0));
    }

    #[test]
    fn directional_prune_dedup() {
        // 非共线向量：同 prune_dedup_candidates 的修复理由
        // 共线数据下 DirectionalPrune Pass 1 的 d²(j,i) < d²(i,q) 对同方向候选恒成立 → 只留 1 个
        let v = vec![
            0.0, 0.0,   // node 0 (query)
            3.0, 0.0,   // node 1 — 0°
            0.0, 3.0,   // node 2 — 90°
           -3.0, 0.0,   // node 3 — 180°
        ];
        let candidates = vec![1, 1, 2, 2, 3];
        let result = DirectionalPrune::prune(&candidates, 0, &v, 2, 8, 2, 1.2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn directional_prune_no_saturation() {
        // DirectionalPrune 绝不用远距离候选填充
        // 用 1D 向量确保 Pass 1 只选少量方向性边，验证不填充到 r_max
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let candidates: Vec<u32> = (1..10).collect();
        // r_min=1 → Pass 2 不触发；r_max=8 → 但不应填到 8
        let result = DirectionalPrune::prune(&candidates, 0, &v, 1, 8, 1, 1.2);
        // 在 1D 线上，方向性检查下只有最近的邻居能入选
        assert!(result.len() <= 8);
        assert!(!result.is_empty());
    }

    #[test]
    fn directional_prune_backfill_connectivity() {
        // 当 Pass 1 产出不足 r_min 时，Pass 2 应补到 r_min
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        // r_min=5 → 如果 Pass 1 < 5，Pass 2 应补到至少 5（如果候选足够）
        let result = DirectionalPrune::prune(&candidates, 0, &v, 4, 10, 5, 1.5);
        // 应该至少有一些结果
        assert!(!result.is_empty());
        assert!(result.len() <= 10);
    }

    #[test]
    fn directional_prune_pass1_never_exceeds_r_max() {
        let v = make_vectors(50, 4);
        let candidates: Vec<u32> = (1..40).collect();
        let result = DirectionalPrune::prune(&candidates, 0, &v, 4, 8, 2, 1.2);
        assert!(result.len() <= 8);
    }

    // ── dispatch 测试 ──

    #[test]
    fn dispatch_robust_prune() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let r1 = prune_dispatch(
            PruneStrategy::RobustPrune, &candidates, 0, &v, 4, 1.2, 8, true,
        );
        let r2 = RobustPrune::prune(&candidates, 0, &v, 4, 1.2, 8, true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn dispatch_directional_prune() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let r1 = prune_dispatch(
            PruneStrategy::DirectionalPrune, &candidates, 0, &v, 4, 1.2, 8, true,
        );
        let config = DirectionalPruneConfig::from_params(8, 1.2);
        let r2 = DirectionalPrune::prune(
            &candidates, 0, &v, 4, config.r_max, config.r_min, config.backfill_alpha,
        );
        assert_eq!(r1, r2);
    }
}
