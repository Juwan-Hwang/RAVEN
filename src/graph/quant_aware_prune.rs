//! 量化感知 RobustPrune（核心研究假设）
//!
//! 设计文档第三层：
//! 归一化打分函数（含数值稳定保护）：
//!   Score = dist / (μ_dist + ε) + β × error / (μ_error + ε)
//!
//!   dist    = 候选边的几何距离
//!   error   = 边两端点 AVQ 平行分量量化误差的均值
//!             error(u,v) = mean(avq_error(u), avq_error(v))
//!   μ_dist  = 当前节点所有候选邻居的平均几何距离
//!   μ_error = 当前节点所有候选邻居的平均量化误差
//!   ε       = 默认 1e-8，防除零导致 NaN/Inf
//!   β       = 融合权重，约 0.05–2.0 之间有意义
//!
//! 归一化消融变量（设计文档）：
//! | 均值归一化（主方案） | 密度均匀数据集 | 极端密度不均时均值不稳定 |
//! | 标准差归一化 | 方差差异大时更稳 | 计算量略高 |
//! | MAD 归一化 | 鲁棒性最强 | 实现复杂度稍高 |
//! | log-sum-exp / sigmoid | 备选非线性压缩 | 引入额外超参 |

use crate::distance::l2_simd;

/// ε 默认值：防除零导致 NaN/Inf（设计文档：默认 1e-8）
pub const EPSILON: f32 = 1e-8;

/// β 有效范围（设计文档：约 0.05–2.0 之间有意义）
pub const BETA_MIN: f32 = 0.05;
/// β 上限
pub const BETA_MAX: f32 = 2.0;

/// 归一化方案（设计文档：消融对比）
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationScheme {
    /// 均值归一化（主方案）：密度均匀数据集适用
    Mean,
    /// 标准差归一化：方差差异大时更稳
    StdDev,
    /// MAD 归一化（中位数绝对偏差）：鲁棒性最强
    Mad,
    /// log-sum-exp / sigmoid：备选非线性压缩
    LogSumExp,
}

impl Default for NormalizationScheme {
    fn default() -> Self {
        // 设计文档：均值归一化为主方案
        NormalizationScheme::Mean
    }
}

/// 量化感知 RobustPrune 配置
#[derive(Debug, Clone)]
pub struct QuantAwarePruneConfig {
    /// α 参数（与标准 RobustPrune 一致）
    pub alpha: f32,
    /// β 融合权重（设计文档：约 0.05–2.0）
    pub beta: f32,
    /// ε 防除零（设计文档：默认 1e-8）
    pub epsilon: f32,
    /// R_max 最大出度
    pub r_max: usize,
    /// 归一化方案（消融对比）
    pub normalization: NormalizationScheme,
}

impl Default for QuantAwarePruneConfig {
    fn default() -> Self {
        Self {
            alpha: 1.2,
            beta: 0.3,
            epsilon: EPSILON,
            r_max: 64,
            normalization: NormalizationScheme::Mean,
        }
    }
}

/// 量化误差函数类型
///
/// 设计文档 F.3：error(u, v) = mean(avq_error(u), avq_error(v))
/// 即边两端点 AVQ 平行分量量化误差的均值
/// 实现时须统一此口径，消融实验记录中须标注所用口径版本
pub type ErrorFn<'a> = &'a dyn Fn(u32, u32) -> f32;

/// 量化感知 RobustPrune
///
/// 设计文档第三层核心研究假设：
/// 量化误差反向影响图剪枝决策的图网络结构
pub struct QuantAwareRobustPrune;

impl QuantAwareRobustPrune {
    /// 执行量化感知剪枝
    ///
    /// 设计文档归一化打分函数：
    ///   Score = dist / (μ_dist + ε) + β × error / (μ_error + ε)
    ///
    /// candidates: 候选邻居集
    /// query_node: 被剪枝的节点
    /// vectors: 扁平存储的向量
    /// dim: 维度
    /// error_fn: 量化误差函数 error(u, v) = mean(avq_error(u), avq_error(v))
    /// config: 配置
    pub fn prune(
        candidates: &[u32],
        query_node: u32,
        vectors: &[f32],
        dim: usize,
        error_fn: ErrorFn,
        config: &QuantAwarePruneConfig,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let query = &vectors[query_node as usize * dim..(query_node as usize + 1) * dim];

        // 计算每个候选的距离和量化误差
        let mut scored: Vec<(f32, f32, u32)> = candidates
            .iter()
            .filter(|&&c| c != query_node)
            .map(|&c| {
                let v = &vectors[c as usize * dim..(c as usize + 1) * dim];
                let dist = l2_simd(query, v);
                let error = error_fn(query_node, c);
                (dist, error, c)
            })
            .collect();

        if scored.is_empty() {
            return Vec::new();
        }

        // 计算归一化基准
        let (mu_dist, mu_error) = Self::compute_normalization(&scored, config.normalization);

        // 计算归一化打分并排序（打分越低越优先保留）
        // 设计文档：Score = dist / (μ_dist + ε) + β × error / (μ_error + ε)
        let mut scored_normalized: Vec<(f32, u32)> = scored
            .iter()
            .map(|&(dist, error, c)| {
                let score = dist / (mu_dist + config.epsilon)
                    + config.beta * error / (mu_error + config.epsilon);
                (score, c)
            })
            .collect();

        // 按打分升序排序（低分优先保留）
        scored_normalized.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 原始距离排序用于 α 遮挡判定
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // RobustPrune 风格的 α 遮遮挡判定 + 量化感知打分
        let mut result: Vec<u32> = Vec::with_capacity(config.r_max);
        let mut pruned = vec![false; scored_normalized.len()];

        // 建立节点到 scored_normalized 索引的映射
        let mut idx_map = std::collections::HashMap::new();
        for (i, &(_, c)) in scored_normalized.iter().enumerate() {
            idx_map.insert(c, i);
        }

        // 建立节点到原始距离的映射（用于 α 遮挡判定）
        let mut dist_map = std::collections::HashMap::new();
        for &(dist, _, c) in scored.iter() {
            dist_map.insert(c, dist);
        }

        // 按量化感知打分顺序选择（β > 0 时影响选择顺序）
        for i in 0..scored_normalized.len() {
            let (_, node) = scored_normalized[i];
            if pruned[i] {
                continue;
            }
            if result.len() >= config.r_max {
                break;
            }
            result.push(node);

            // α 遮遮挡判定（用原始距离，与标准 RobustPrune 一致）
            let p_vec = &vectors[node as usize * dim..(node as usize + 1) * dim];
            for j in (i + 1)..scored_normalized.len() {
                let (_, other) = scored_normalized[j];
                if pruned[j] {
                    continue;
                }
                let q_vec = &vectors[other as usize * dim..(other as usize + 1) * dim];
                let dist_p_q = l2_simd(p_vec, q_vec);
                let q_dist = dist_map[&other];
                // α × dist(p, p') ≤ dist(p', q) → p' 被 p 遮挡（方向修正）
                if config.alpha * dist_p_q <= q_dist {
                    pruned[j] = true;
                }
            }
        }

        result
    }

    /// 计算归一化基准
    ///
    /// 设计文档：归一化消融变量
    fn compute_normalization(
        scored: &[(f32, f32, u32)],
        scheme: NormalizationScheme,
    ) -> (f32, f32) {
        if scored.is_empty() {
            return (EPSILON, EPSILON);
        }

        let dists: Vec<f32> = scored.iter().map(|(d, _, _)| *d).collect();
        let errors: Vec<f32> = scored.iter().map(|(_, e, _)| *e).collect();

        match scheme {
            NormalizationScheme::Mean => {
                // 设计文档主方案：均值归一化
                let mu_dist = dists.iter().sum::<f32>() / dists.len() as f32;
                let mu_error = errors.iter().sum::<f32>() / errors.len() as f32;
                (mu_dist, mu_error)
            }
            NormalizationScheme::StdDev => {
                // 标准差归一化：方差差异大时更稳
                let mu_dist = dists.iter().sum::<f32>() / dists.len() as f32;
                let mu_error = errors.iter().sum::<f32>() / errors.len() as f32;
                let var_dist = dists.iter().map(|d| (d - mu_dist).powi(2)).sum::<f32>()
                    / dists.len() as f32;
                let var_error = errors.iter().map(|e| (e - mu_error).powi(2)).sum::<f32>()
                    / errors.len() as f32;
                (var_dist.sqrt().max(EPSILON), var_error.sqrt().max(EPSILON))
            }
            NormalizationScheme::Mad => {
                // MAD 归一化：鲁棒性最强
                let mu_dist = Self::median(&dists);
                let mu_error = Self::median(&errors);
                let mad_dist = Self::median_abs_dev(&dists, mu_dist);
                let mad_error = Self::median_abs_dev(&errors, mu_error);
                (mad_dist.max(EPSILON), mad_error.max(EPSILON))
            }
            NormalizationScheme::LogSumExp => {
                // log-sum-exp / sigmoid：备选非线性压缩
                let max_dist = dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let max_error = errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse_dist = max_dist
                    + (dists.iter().map(|d| (d - max_dist).exp()).sum::<f32>()
                        / dists.len() as f32).ln();
                let lse_error = max_error
                    + (errors.iter().map(|e| (e - max_error).exp()).sum::<f32>()
                        / errors.len() as f32).ln();
                (lse_dist.max(EPSILON), lse_error.max(EPSILON))
            }
        }
    }

    /// 中位数
    fn median(sorted: &[f32]) -> f32 {
        if sorted.is_empty() {
            return 0.0;
        }
        let mut s = sorted.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = s.len() / 2;
        if s.len() % 2 == 0 {
            (s[mid - 1] + s[mid]) / 2.0
        } else {
            s[mid]
        }
    }

    /// 中位数绝对偏差
    fn median_abs_dev(values: &[f32], median: f32) -> f32 {
        let devs: Vec<f32> = values.iter().map(|v| (v - median).abs()).collect();
        Self::median(&devs)
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
        let error_fn = |_: u32, _: u32| 0.0f32;
        let config = QuantAwarePruneConfig::default();
        let result = QuantAwareRobustPrune::prune(&[], 0, &v, 4, &error_fn, &config);
        assert!(result.is_empty());
    }

    #[test]
    fn prune_respects_r_max() {
        let v = make_vectors(20, 4);
        let error_fn = |_: u32, _: u32| 0.1f32;
        let config = QuantAwarePruneConfig {
            r_max: 5,
            ..Default::default()
        };
        let candidates: Vec<u32> = (1..15).collect();
        let result = QuantAwareRobustPrune::prune(&candidates, 0, &v, 4, &error_fn, &config);
        assert!(result.len() <= 5);
    }

    #[test]
    fn beta_zero_matches_standard_robust_prune() {
        // β=0 时应退化为标准 RobustPrune
        let v = make_vectors(20, 4);
        let error_fn = |_: u32, _: u32| 0.5f32;
        let config = QuantAwarePruneConfig {
            alpha: 1.0,
            beta: 0.0,
            r_max: 10,
            ..Default::default()
        };
        let candidates: Vec<u32> = (1..15).collect();
        let result = QuantAwareRobustPrune::prune(&candidates, 0, &v, 4, &error_fn, &config);
        let standard = crate::graph::robust_prune::RobustPrune::prune(&candidates, 0, &v, 4, 1.0, 10, false);
        // β=0 时量化误差不影响排序，结果应一致
        assert_eq!(result, standard);
    }

    #[test]
    fn epsilon_prevents_division_by_zero() {
        // 设计文档：μ=0 时除零 → NaN/Inf，打分函数分母加 ε=1e-8
        let v = make_vectors(5, 4);
        // 所有误差为 0，μ_error = 0，应通过 ε 避免除零
        let error_fn = |_: u32, _: u32| 0.0f32;
        let config = QuantAwarePruneConfig {
            beta: 1.0,
            epsilon: EPSILON,
            r_max: 4,
            ..Default::default()
        };
        let candidates: Vec<u32> = vec![1, 2, 3];
        let result = QuantAwareRobustPrune::prune(&candidates, 0, &v, 4, &error_fn, &config);
        // 不应 panic 或产生 NaN
        assert!(!result.is_empty());
    }

    #[test]
    fn beta_range_meaningful() {
        // 设计文档：β 约 0.05–2.0 之间有意义
        assert!(BETA_MIN >= 0.05);
        assert!(BETA_MAX <= 2.0);
    }

    #[test]
    fn normalization_schemes_produce_valid_results() {
        let v = make_vectors(20, 4);
        let error_fn = |_: u32, _: u32| 0.3f32;
        let candidates: Vec<u32> = (1..10).collect();

        for scheme in [
            NormalizationScheme::Mean,
            NormalizationScheme::StdDev,
            NormalizationScheme::Mad,
            NormalizationScheme::LogSumExp,
        ] {
            let config = QuantAwarePruneConfig {
                normalization: scheme,
                ..Default::default()
            };
            let result = QuantAwareRobustPrune::prune(&candidates, 0, &v, 4, &error_fn, &config);
            assert!(!result.is_empty(), "scheme {:?} produced empty result", scheme);
        }
    }
}
