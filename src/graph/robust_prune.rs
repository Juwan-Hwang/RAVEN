//! RobustPrune（标准 Vamana 多轮 α 递增剪枝，β=0 baseline）
//!
//! 设计文档第三层：
//! 主线选 Vamana/DiskANN 风格。本模块是 β=0 的标准 RobustPrune（baseline）。
//! RAVEN 的创新在 QuantAwareRobustPrune（β>0，量化感知剪枝）、
//! RP-Tuning（post-hoc α 调优）和 AVQ（retrieval-aware 量化），
//! 这些模块构建在本 baseline 之上。
//!
//! 算法（Vamana 论文 Algorithm 2，平方距离直接比较实现）：
//! 1. 候选集按距离升序排序，去重
//! 2. 多轮 α 递增扫描：
//!    - 第一轮 current_alpha = 1.0（最严格，建立多样性骨干）
//!    - 每轮 current_alpha *= min(alpha, 1.2)，上限为 alpha
//!    - 累积遮挡因子 occlude_factor = max(d²(i,q) / d²(j,i))，
//!      超过 current_alpha² 的候选跳过，留待下一轮更宽松阈值重新评估
//!    - α=1.0 轮确保最多样性的近邻入选，后续轮补充长程导航边
//! 3. Saturation：若结果不足 r_max，用剩余候选按距离填充，避免图过度稀疏
//!
//! v6.5 修复（原实现 BUG）：
//! - 原单轮 α 扫描 → 多轮递增（Vamana 论文本身就是多轮的）
//! - 原无 saturation → 添加（图过度稀疏是 avg_visited 飙升的根因之一）
//! - 原无候选去重 → 添加（重复候选导致错误遮挡）
//!
//! v6.6 修复：
//! - Saturation 条件对齐 DiskANN build_disk_index：仅 alpha > 1.0 时填充
//!   DiskANN 源码：`saturate_after_prune(true)` + `if alpha > 1.0`
//!   第一轮 α=1.0 不填充（保多样性骨干），第二轮 α=1.2 填充到 r_max（保导航密度）
//! - 原实现无条件填充 → 改为条件填充，消除 avg_visited=2324 的根因

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

/// 标准 Vamana RobustPrune（β=0 baseline，多轮 α 递增）
///
/// 设计文档第三层：第一阶段固定 β=0（标准 RobustPrune），扫 α baseline
/// 这是 QuantAwareRobustPrune（β>0）的对照基线，不是 RAVEN 的创新点
pub struct RobustPrune;

impl RobustPrune {
    /// 执行 RobustPrune 剪枝（Vamana 论文 Algorithm 2，平方距离直接比较）
    ///
    /// candidates: 候选邻居集（节点 ID）
    /// query_node: 被剪枝的节点
    /// vectors: 扁平存储的向量
    /// dim: 维度
    /// alpha: α 参数
    /// r_max: 最大出度
    /// saturate: 是否在剪枝后填充到 r_max（DiskANN: saturate_after_prune && alpha > 1.0）
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

        let query = &vectors[query_node as usize * dim..(query_node as usize + 1) * dim];

        // 计算每个候选到 query 的平方距离，去重，按距离升序排序
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

        if scored.is_empty() {
            return Vec::new();
        }

        let n = scored.len();
        let mut result: Vec<u32> = Vec::with_capacity(r_max);

        // ── 多轮 α 遮挡扫描核心状态 ──
        //
        // occlude_factor[i] 的语义（平方距离空间，对齐 DiskANN）：
        //   0.0       → 尚未被任何已接受邻居检查
        //   (0, α]    → 已检查，累积遮挡因子（ratio_sq = d²(i,q) / d²(j,i)）
        //   f32::MAX  → 已接受为邻居
        //
        // 遮挡判定（对齐 DiskANN occlude_list）：
        //   DiskANN 的 Metric::L2 返回平方距离（SquaredL2），
        //   occlude_factor = distance_ik / distance_jk = d²(ik) / d²(jk)
        //   检查：occlude_factor > current_alpha（不是 alpha²！）
        //   有效阈值：d(ik)/d(jk) > sqrt(alpha)
        //   对于 alpha=1.2：有效阈值 ≈ 1.095（比直接用 alpha 更激进）
        let mut occlude_factor = vec![0.0f32; n];
        // result_indices[k] = 第 k 个已接受候选在 scored 中的下标
        let mut result_indices: Vec<usize> = Vec::with_capacity(r_max);

        // ── 多轮 α 递增扫描（Vamana 论文核心机制）──
        //
        // current_alpha 从 1.0 开始，每轮乘以 increment_factor = min(alpha, 1.2)
        // 对于 alpha=1.2：pass 1 用 α=1.0（多样性骨干），pass 2 用 α=1.2（长程边）
        // 对于 alpha=1.0：只有一轮，current_alpha == alpha 立即退出
        let mut current_alpha = 1.0f32;
        let increment_factor = alpha.min(1.2);

        while result.len() < r_max {
            // 对齐 DiskANN：平方距离直接比较 alpha（不是 alpha²）
            // DiskANN 源码：occlude_factor > current_alpha
            // 其中 occlude_factor = d²(ik) / d²(jk)（平方距离比）
            // 有效阈值：d(ik)/d(jk) > sqrt(current_alpha)

            for i in 0..n {
                if result.len() >= r_max {
                    break;
                }

                // 已接受的候选跳过
                if occlude_factor[i] == f32::MAX {
                    continue;
                }

                // 在当前 α 阈值下已被遮挡的候选跳过（留待下一轮更宽松的阈值）
                if occlude_factor[i] > current_alpha {
                    continue;
                }

                // 检查候选 i 对所有已接受邻居的遮挡因子
                let candidate_vec =
                    &vectors[scored[i].1 as usize * dim..(scored[i].1 as usize + 1) * dim];
                let dist_iq_sq = scored[i].0; // d²(candidate_i, query)

                for &j in &result_indices {
                    // 只检查距离更近的已接受邻居（j < i，sorted by distance）
                    if j >= i {
                        continue;
                    }

                    let accepted_vec =
                        &vectors[scored[j].1 as usize * dim..(scored[j].1 as usize + 1) * dim];
                    let dist_ji_sq = l2_simd(accepted_vec, candidate_vec); // d²(accepted_j, candidate_i)

                    // d(j, i) == 0 → 完全重叠，直接遮挡
                    if dist_ji_sq == 0.0 {
                        occlude_factor[i] = f32::MAX;
                        break;
                    }

                    // 累积遮挡因子：ratio_sq = d²(i, q) / d²(j, i)
                    // 对齐 DiskANN：直接用平方距离比，比较 alpha（非 alpha²）
                    let ratio_sq = dist_iq_sq / dist_ji_sq;
                    if ratio_sq > occlude_factor[i] {
                        occlude_factor[i] = ratio_sq;
                    }

                    // 超过当前 α 阈值 → 跳过（留待下一轮）
                    if occlude_factor[i] > current_alpha {
                        break;
                    }
                }

                // 仍在阈值内 → 接受为邻居
                if occlude_factor[i] > current_alpha {
                    continue;
                }

                occlude_factor[i] = f32::MAX; // 标记为已接受
                result_indices.push(i);
                result.push(scored[i].1);
            }

            // 已达到最终 α，退出多轮扫描
            if current_alpha >= alpha {
                break;
            }
            // 递增 α
            current_alpha = (current_alpha * increment_factor).min(alpha);
        }

        // Saturation：当 saturate=true 时，用剩余候选按距离填充到 r_max。
        //
        // DiskANN build_disk_index 明确设置 saturate_after_prune(true)，
        // 且仅在 alpha > 1.0 时执行填充（第二轮迭代 α=1.2）。
        // 第一轮 α=1.0 不填充：保多样性骨干。
        // 第二轮 α=1.2 填充：确保每个节点有 r_max 条边，提供充足导航路径。
        //
        // 无 saturation 时 α=1.2 在 SIFT 上典型度数仅 15-40，图过度稀疏，
        // 搜索导航效率崩塌（avg_visited=2324，正常应 100-300）。
        if saturate {
            for i in 0..n {
                if result.len() >= r_max {
                    break;
                }
                // 跳过已接受的候选（occlude_factor == MAX 表示已入选）
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| i as f32).collect()
    }

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
        // α 越大，剪枝越宽松，保留更多候选
        let v = make_vectors(20, 4);
        let candidates: Vec<u32> = (1..15).collect();
        let result_small = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 10, false);
        let result_large = RobustPrune::prune(&candidates, 0, &v, 4, 2.0, 10, false);
        // α=2.0 应保留 >= α=1.0 的数量
        assert!(result_large.len() >= result_small.len());
    }

    #[test]
    fn prune_dedup_candidates() {
        let v = make_vectors(10, 4);
        let candidates = vec![1, 1, 2, 2, 3];
        let result = RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 8, false);
        // 去重后只有 3 个候选
        assert_eq!(result.len(), 3);
    }
}
