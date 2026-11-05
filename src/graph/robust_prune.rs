//! 剪枝策略模块
//!
//! 包含两种剪枝算法：
//! 1. RobustPrune（Vamana/DiskANN 标准）：多轮 α 递增 + saturation 填充
//! 2. DirectionalPrune（RAVEN 超越方案）：方向性扫描 + 自适应连通性补底 + 跳跃边，无 saturation
//!
//! DirectionalPrune 的设计动机：
//!   RobustPrune 的 saturation 用远距离候选填满 r_max → avg_visited 略高（1227 vs Glass 1041, H20 实测）
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
    scored.sort_by_key(|a| a.1);
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
/// - `DirectionalPrune`：RAVEN 超越方案，方向性扫描 + 自适应连通性补底 + 跳跃边
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum PruneStrategy {
    /// Vamana/DiskANN 标准 RobustPrune（baseline）
    #[default]
    RobustPrune,
    /// RAVEN DirectionalPrune（方向性 + 连通性补底 + 跳跃边，无 saturation）
    DirectionalPrune,
}


// ═══════════════════════════════════════════════════════════════════════
//  RobustPrune（Vamana/DiskANN 标准）
// ═══════════════════════════════════════════════════════════════════════

/// RobustPrune 配置
#[derive(Debug, Clone)]
pub struct RobustPruneConfig {
    /// 距离放大因子（α > 1 允许更多长程边，提升连通性）
    pub alpha: f32,
    /// 每个节点最大邻居数（度数上限）
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
    /// Pass 1 核心方向性判据的遮挡阈值
    ///
    /// 判据: d²(j,i) < alpha_core · d²(i,q) → j 遮挡 i（剪枝）
    ///
    /// - alpha_core = 1.0（默认）：标准方向性判据，等价 Glass Heuristic2
    /// - alpha_core < 1.0：阈值更严 → 更难遮挡 → 更多边存活 → 图更密
    /// - alpha_core > 1.0：阈值更松 → 更易遮挡 → 更少边存活 → 图更稀
    ///
    /// 这是真正决定图拓扑质量的核心自由度——
    /// 对比 `backfill_alpha` 只控制从不触发的 Pass 2 补底。
    pub alpha_core: f32,
    /// Pass 3 跳跃边数量（0 = 禁用）
    ///
    /// DirectionalPrune Pass 1 的遮挡判定 d(j,i) < d(i,q) 在几何上
    /// 等价于 cos(θ) > r_j/(2*r_i)。当已接受邻居 j 很近时，即使
    /// 候选 i 在 90° 方向上也会被遮挡——远距离但方向唯一的「跳跃边」
    /// 被系统性剪掉，导致搜索需要更多跳数才能到达目标。
    ///
    /// Pass 3 从被拒绝的远距离候选中，选择遮挡比 d²(j,i)/d²(i,q)
    /// 最接近 1.0 的（=方向最独特），保留 n_jumps 条。
    pub n_jumps: usize,
}

impl DirectionalPruneConfig {
    /// 从 r_max 和 alpha 自动推导配置
    ///
    /// r_min = r_max / divisor，divisor 由 `RAVEN_RMIN_DIVISOR` 环境变量控制（默认 4）。
    /// - divisor=4: r_min=R/4，Pass 2 仅补到 25% 底数（QPS 最优）
    /// - divisor=3: r_min=R/3，中间态
    /// - divisor=2: r_min=R/2，实测 QPS -6.8% 且 recall 无改善
    pub fn from_params(r_max: usize, alpha: f32) -> Self {
        let divisor = std::env::var("RAVEN_RMIN_DIVISOR")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&d| d > 0)
            .unwrap_or(4);
        Self {
            r_max,
            r_min: (r_max / divisor).max(1),
            backfill_alpha: alpha.max(1.0),
            alpha_core: 1.0,
            n_jumps: 0,
        }
    }

    /// 启用跳跃边
    pub fn with_jumps(mut self, n_jumps: usize) -> Self {
        self.n_jumps = n_jumps;
        self
    }
}

impl Default for DirectionalPruneConfig {
    fn default() -> Self {
        Self::from_params(64, 1.2)
    }
}

/// 跳跃边候选评分
///
/// occlusion_ratio = max over accepted j of d²(j,i) / d²(i,q)
/// - ratio ≈ 1.0: 勉强被遮挡，方向最独特 → 最佳跳跃边
/// - ratio ≈ 0.0: 被强遮挡，方向重叠 → 冗余
#[derive(Clone, Copy)]
struct JumpCandidate {
    occlusion_ratio: f32,
    id: u32,
}

/// RAVEN DirectionalPrune — 融合 RobustPrune 和 Heuristic2 的超越方案
///
/// 三方对比：
/// ┌─────────────────┬──────────────────┬───────────────────┬──────────────────────────┐
/// │                 │ RobustPrune      │ Heuristic2 (Glass)│ DirectionalPrune (RAVEN) │
/// ├─────────────────┼──────────────────┼───────────────────┼──────────────────────────┤
/// │ Pass 1 方向性   │ α=1.0 多轮       │ α=1.0 单轮        │ α=1.0 单轮               │
/// │ 长程边          │ α=1.2 多轮补充   │ 无                │ α=backfill 补到 r_min    │
/// │ 跳跃边          │ 无               │ 无                │ Pass 3: 方向最独特远边   │
/// │ Saturation      │ 填到 r_max（垃圾）│ 无               │ 无                       │
/// │ 连通性保证      │ 靠 saturation    │ 靠 HNSW 多层      │ 靠 r_min 补底             │
/// │ avg_visited     │ 1227             │ 1041 (H20 实测)   │ 目标：低                 │
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
/// 4. Pass 3 — 跳跃边（n_jumps > 0 时）：
///    对被 Pass 1 遮挡的候选，计算遮挡比 = max_j d²(j,i)/d²(i,q)。
///    比值越接近 1.0 = 勉强被遮挡 = 方向最独特 = 最佳跳跃边。
///    保留 n_jumps 条，占用 r_max 度数预算。
/// 5. 无 saturation：剩余槽位留空，不盲目填充远距离候选
pub struct DirectionalPrune;

impl DirectionalPrune {
    /// 执行 DirectionalPrune 剪枝（alpha_core=1.0, n_jumps=0，等价于无跳跃边）
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
        Self::prune_with_jumps(candidates, query_node, vectors, dim, r_max, r_min, backfill_alpha, 1.0, 0)
    }

    /// 执行 DirectionalPrune 剪枝（含跳跃边）
    ///
    /// `n_jumps > 0` 时在 Pass 2 之后执行 Pass 3：从被拒绝的远距离候选中
    /// 选择方向最独特的（遮挡比最接近 1.0），保留 n_jumps 条跳跃边。
    pub fn prune_with_jumps(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        r_max: usize,
        r_min: usize,
        backfill_alpha: f32,
        alpha_core: f32,
        n_jumps: usize,
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

        // ── Pass 1：方向性扫描（alpha_core 可调，默认 1.0 等价 Glass Heuristic2）──
        //
        // 判据: d²(j,i) < alpha_core · d²(i,q) → j 方向性遮挡 i
        //
        // alpha_core = 1.0：标准方向性判据
        // alpha_core < 1.0：更难遮挡 → 更多方向独特的边存活 → 图更密
        // alpha_core > 1.0：更易遮挡 → 边更少 → 图更稀，可能触发 Pass 2 补底
        for i in 0..n {
            if result.len() >= r_max {
                break;
            }

            let dist_iq_sq = scored[i].0;
            let candidate_vec =
                &vectors[scored[i].1 as usize * dim..(scored[i].1 as usize + 1) * dim];

            let mut good = true;
            for &accepted_id in &result {
                let accepted_vec =
                    &vectors[accepted_id as usize * dim..(accepted_id as usize + 1) * dim];
                let dist_ji_sq = l2_simd(accepted_vec, candidate_vec);

                if dist_ji_sq < alpha_core * dist_iq_sq {
                    good = false;
                    break;
                }
            }

            if good {
                result.push(scored[i].1);
            }
        }

        // ── Pass 2：连通性补底（仅当度数不足 r_min）──
        if result.len() < r_min {
            let accepted_set: std::collections::HashSet<u32> = result.iter().copied().collect();

            for &(dist_iq_sq, id) in scored.iter().take(n) {
                if result.len() >= r_min {
                    break;
                }

                if accepted_set.contains(&id) {
                    continue;
                }

                let candidate_vec =
                    &vectors[id as usize * dim..(id as usize + 1) * dim];

                let mut good = true;
                for &accepted_id in &result {
                    let accepted_vec =
                        &vectors[accepted_id as usize * dim..(accepted_id as usize + 1) * dim];
                    let dist_ji_sq = l2_simd(accepted_vec, candidate_vec);

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

        // ── Pass 3：跳跃边（n_jumps > 0 时执行）──
        //
        // Pass 1 的遮挡判定 d(j,i) < d(i,q) 在几何上等价于
        // cos(θ) > r_j / (2*r_i)。当 j 很近时，即使 θ≈90° 也会遮挡。
        // 这导致远距离但方向唯一的跳跃边被系统性剪掉。
        //
        // 修复：对被拒绝的候选，计算遮挡比 = max_j d²(j,i)/d²(i,q)。
        // 比值越接近 1.0 = 勉强被遮挡 = 方向最独特 = 最佳跳跃边。
        if n_jumps > 0 && result.len() < r_max {
            let slots = (r_max - result.len()).min(n_jumps);
            let accepted_set: std::collections::HashSet<u32> = result.iter().copied().collect();

            let mut jump_candidates: Vec<JumpCandidate> = Vec::new();

            for &(dist_iq_sq, id) in &scored {
                if accepted_set.contains(&id) {
                    continue;
                }

                let candidate_vec =
                    &vectors[id as usize * dim..(id as usize + 1) * dim];

                // 计算遮挡比 = max over accepted j of d²(j,i) / d²(i,q)
                let mut max_ratio = 0.0f32;
                for &accepted_id in &result {
                    let accepted_vec =
                        &vectors[accepted_id as usize * dim..(accepted_id as usize + 1) * dim];
                    let dist_ji_sq = l2_simd(accepted_vec, candidate_vec);
                    let ratio = dist_ji_sq / dist_iq_sq;
                    if ratio > max_ratio {
                        max_ratio = ratio;
                    }
                }

                jump_candidates.push(JumpCandidate {
                    occlusion_ratio: max_ratio,
                    id,
                });
            }

            // 按遮挡比降序（最接近 1.0 = 最独特 的优先）
            jump_candidates.sort_by(|a, b| {
                b.occlusion_ratio
                    .partial_cmp(&a.occlusion_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for jc in jump_candidates.into_iter().take(slots) {
                result.push(jc.id);
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
        Self::prune_with_jumps(
            candidates, query_node, vectors, dim,
            config.r_max, config.r_min, config.backfill_alpha,
            config.alpha_core, config.n_jumps,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  统一 dispatch
// ═══════════════════════════════════════════════════════════════════════

/// 根据策略 dispatch 到对应剪枝实现
///
/// 共享 alpha/r_max/saturate 语义：
/// - RobustPrune：直接传递
/// - DirectionalPrune：r_min = r_max/4, backfill_alpha = alpha, n_jumps 从 RAVEN_N_JUMPS 环境变量读取
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
            let mut config = DirectionalPruneConfig::from_params(r_max, alpha);
            config.n_jumps = std::env::var("RAVEN_N_JUMPS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            config.alpha_core = std::env::var("RAVEN_ALPHA_CORE")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|&v| v > 0.0)
                .unwrap_or(1.0);
            DirectionalPrune::prune_with_jumps(
                candidates, query_node, vectors, dim,
                config.r_max, config.r_min, config.backfill_alpha,
                config.alpha_core, config.n_jumps,
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
        let v = vec![
            0.0, 0.0,
            3.0, 0.0,
            0.0, 3.0,
           -3.0, 0.0,
        ];
        let candidates = vec![1, 1, 2, 2, 3];
        let result = RobustPrune::prune(&candidates, 0, &v, 2, 1.0, 8, false);
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
        let v = vec![
            0.0, 0.0,
            3.0, 0.0,
            0.0, 3.0,
           -3.0, 0.0,
        ];
        let candidates = vec![1, 1, 2, 2, 3];
        let result = DirectionalPrune::prune(&candidates, 0, &v, 2, 8, 2, 1.2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn directional_prune_no_saturation() {
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let candidates: Vec<u32> = (1..10).collect();
        let result = DirectionalPrune::prune(&candidates, 0, &v, 1, 8, 1, 1.2);
        assert!(result.len() <= 8);
        assert!(!result.is_empty());
    }

    #[test]
    fn directional_prune_backfill_connectivity() {
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result = DirectionalPrune::prune(&candidates, 0, &v, 4, 10, 5, 1.5);
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

    // ── 跳跃边测试 ──

    #[test]
    fn jump_edges_add_neighbors() {
        // 跳跃边应增加邻居数量
        let v = vec![
            0.0, 0.0,   // node 0 (query)
            1.0, 0.0,   // node 1: 近, 0°
            0.9, 0.1,   // node 2: 近, ~6°
            0.8, 0.2,   // node 3: 近, ~14°
            2.0, 2.0,   // node 4: 中, 45°
            1.5, 1.5,   // node 5: 中, 45° (被4遮挡)
            5.0, -1.0,  // node 6: 远
            5.0, 1.0,   // node 7: 远
            4.0, 3.0,   // node 8: 远
            3.0, 4.0,   // node 9: 远
            -3.0, 4.0,  // node 10: 远, ~127°
            -4.0, 3.0,  // node 11: 远, ~143°
            -5.0, 0.0,  // node 12: 远, 180°
            0.0, 5.0,   // node 13: 远, 90°
            0.0, -5.0,  // node 14: 远, -90°
        ];
        let candidates: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

        let base = DirectionalPrune::prune(&candidates, 0, &v, 2, 10, 3, 1.2);
        let with_jumps = DirectionalPrune::prune_with_jumps(&candidates, 0, &v, 2, 10, 3, 1.2, 1.0, 4);

        assert!(with_jumps.len() > base.len(), "跳跃边应增加邻居数: base={}, jumps={}", base.len(), with_jumps.len());
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
        // 确保 RAVEN_N_JUMPS 未设置
        std::env::remove_var("RAVEN_N_JUMPS");
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
