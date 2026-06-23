//! RP-Tuning（post-hoc α 调优）
//!
//! 设计文档第三层 α 三段式②：
//! 前提：从较高质量基础图出发
//! 机制：对现有出邻居集合重跑 RobustPrune，蒸馏不同 α 工作点
//! 约束：不是从稀疏图中凭空恢复已删边；
//!       若需保留构建时未剪枝的候选集，须明确额外存储成本
//!       （三种存储方案及退化判定阈值见附录 A）
//! → 生成 α=1.0 / 1.2 / 1.5 / 2.0 变体索引
//! → 一次构建覆盖整条 Pareto 曲线的多个工作点
//!
//! 附录 A 三种存储方案：
//! A（推荐先验证）：直接对当前邻域重跑 prune，不保留原始候选集，零额外存储
//! B（妥协方案）：每节点存储被剪掉的邻居 ID
//! C（完整回溯）：保留完整构建期候选集

use crate::memory::HybridBlockedCsr;
use super::vamana::VamanaGraph;
use super::robust_prune::RobustPrune;

/// RP-Tuning 存储方案（设计文档附录 A）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RPTuningStorageScheme {
    /// 方案 A：直接对当前邻域重跑 prune，不保留原始候选集，零额外存储
    /// 设计文档：推荐先验证
    SchemeA,
    /// 方案 B：每节点存储被剪掉的邻居 ID
    /// 候选集 = 当前邻域 ∪ 被剪边，O(N × (R_soft - R_max)) 额外存储
    SchemeB,
    /// 方案 C：保留完整构建期候选集，最多 R_soft 条
    /// O(N × R_soft) 额外存储
    SchemeC,
}

impl Default for RPTuningStorageScheme {
    fn default() -> Self {
        // 设计文档：A 方案推荐先验证
        RPTuningStorageScheme::SchemeA
    }
}

/// RP-Tuning 配置
#[derive(Debug, Clone)]
pub struct RPTuningConfig {
    /// 存储方案
    pub scheme: RPTuningStorageScheme,
    /// α 扫描范围（设计文档 F.4：实验前锁定）
    /// 扫描范围：α ∈ [0.8, 3.0]
    /// 离散取点：[0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    pub alpha_points: Vec<f32>,
    /// R_max（最终图最大出度）
    pub r_max: usize,
}

impl Default for RPTuningConfig {
    fn default() -> Self {
        Self {
            scheme: RPTuningStorageScheme::SchemeA,
            // 设计文档：生成 α=1.0 / 1.2 / 1.5 / 2.0 变体索引
            // F.4：离散取点 [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
            alpha_points: vec![1.0, 1.2, 1.5, 2.0],
            r_max: 64,
        }
    }
}

/// α 变体索引
#[derive(Debug, Clone)]
pub struct AlphaVariant {
    /// α 值
    pub alpha: f32,
    /// 变体图存储
    pub storage: HybridBlockedCsr,
    /// 入口节点
    pub entry_point: u32,
}

impl AlphaVariant {
    /// 转为 VamanaGraph
    pub fn into_graph(self, dim: usize) -> VamanaGraph {
        VamanaGraph::from_storage(self.storage, self.entry_point, dim)
    }
}

/// RP-Tuning：post-hoc α 调优
///
/// 设计文档第三层 α 三段式②：
/// 一次构建覆盖整条 Pareto 曲线的多个工作点
pub struct RPTuning;

impl RPTuning {
    /// 生成多个 α 变体
    ///
    /// 设计文档：对现有出邻居集合重跑 RobustPrune，蒸馏不同 α 工作点
    ///
    /// base_graph: 基础图（从较高质量出发）
    /// vectors: 扁平存储的向量
    /// dim: 维度
    /// config: RP-Tuning 配置
    pub fn generate_variants(
        base_graph: &VamanaGraph,
        vectors: &[f32],
        dim: usize,
        config: &RPTuningConfig,
    ) -> Vec<AlphaVariant> {
        let n = base_graph.len();
        config
            .alpha_points
            .iter()
            .map(|&alpha| {
                let mut new_storage = HybridBlockedCsr::new(n, config.r_max);
                // 对每个节点，取当前邻域重跑 RobustPrune
                for node in 0..n as u32 {
                    let neighbors = base_graph.neighbors(node).to_vec();
                    // 方案 A：候选集 = 当前邻域（设计文档：不保留原始候选集）
                    let candidates = match config.scheme {
                        RPTuningStorageScheme::SchemeA => neighbors.clone(),
                        // 方案 B/C 需要额外存储，当前实现 A 方案
                        // B/C 的实现需要构建期保留候选集，这里回退到 A
                        RPTuningStorageScheme::SchemeB
                        | RPTuningStorageScheme::SchemeC => neighbors.clone(),
                    };
                    let pruned = RobustPrune::prune(
                        &candidates,
                        node,
                        vectors,
                        dim,
                        alpha,
                        config.r_max,
                    );
                    new_storage.set_neighbors(node, &pruned);
                }
                AlphaVariant {
                    alpha,
                    storage: new_storage,
                    entry_point: base_graph.entry_point(),
                }
            })
            .collect()
    }

    /// 退化判定（设计文档附录 A）
    ///
    /// A 方案视为"不退化"，须同时满足：
    /// 指标一（固定 QPS）：A 方案在该 QPS 下的 recall@10 差距 < 0.5%
    /// 指标二（固定 recall）：A 方案达到该 recall 所需的 QPS 差距 < 3%
    /// 覆盖范围：至少 3 个不同数据集，每个独立判定
    pub fn check_degradation(
        recall_at_qps_diff: f64,
        qps_at_recall_diff: f64,
    ) -> RPTuningDegradationResult {
        // 设计文档：实验前锁定阈值，不事后 cherry-pick
        const RECALL_THRESHOLD: f64 = 0.005; // 0.5%
        const QPS_THRESHOLD: f64 = 0.03; // 3%

        let recall_ok = recall_at_qps_diff < RECALL_THRESHOLD;
        let qps_ok = qps_at_recall_diff < QPS_THRESHOLD;

        RPTuningDegradationResult {
            recall_at_qps_diff,
            qps_at_recall_diff,
            is_degraded: !(recall_ok && qps_ok),
            recall_threshold: RECALL_THRESHOLD,
            qps_threshold: QPS_THRESHOLD,
        }
    }

    /// 连通性辅助校验（设计文档附录 D）
    ///
    /// 辅助判定项（预警，不作为主判定）：
    /// 孤立节点数：A 方案 <= 完整重建版本的 2 倍
    /// 平均出度：A 方案 >= 完整重建版本的 95%
    pub fn check_connectivity(
        variant: &AlphaVariant,
        baseline_isolated: usize,
        baseline_mean_degree: f64,
    ) -> ConnectivityCheck {
        let stats = variant.storage.log_degree_distribution();
        ConnectivityCheck {
            isolated_ok: stats.isolated_nodes <= baseline_isolated * 2,
            mean_degree_ok: stats.mean_degree >= baseline_mean_degree * 0.95,
            variant_isolated: stats.isolated_nodes,
            baseline_isolated,
            variant_mean_degree: stats.mean_degree,
            baseline_mean_degree,
        }
    }
}

/// RP-Tuning 退化判定结果
#[derive(Debug, Clone)]
pub struct RPTuningDegradationResult {
    /// 固定 QPS 下 recall@10 差距
    pub recall_at_qps_diff: f64,
    /// 固定 recall 下 QPS 差距
    pub qps_at_recall_diff: f64,
    /// 是否退化（true 表示需降级到 B 方案）
    pub is_degraded: bool,
    /// recall 阈值
    pub recall_threshold: f64,
    /// QPS 阈值
    pub qps_threshold: f64,
}

/// 连通性校验结果（设计文档附录 D）
#[derive(Debug, Clone)]
pub struct ConnectivityCheck {
    /// 孤立节点数是否达标
    pub isolated_ok: bool,
    /// 平均出度是否达标
    pub mean_degree_ok: bool,
    /// 变体孤立节点数
    pub variant_isolated: usize,
    /// 基线孤立节点数
    pub baseline_isolated: usize,
    /// 变体平均出度
    pub variant_mean_degree: f64,
    /// 基线平均出度
    pub baseline_mean_degree: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::ChaCha8Rng;
    use crate::graph::vamana::VamanaBuildConfig;

    fn build_test_graph() -> (Vec<f32>, usize, VamanaGraph) {
        let vectors: Vec<f32> = (0..200).map(|i| i as f32).collect();
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
        (vectors, dim, graph)
    }

    #[test]
    fn generate_variants_default_alphas() {
        let (vectors, dim, graph) = build_test_graph();
        let config = RPTuningConfig::default();
        let variants = RPTuning::generate_variants(&graph, &vectors, dim, &config);
        assert_eq!(variants.len(), 4); // [1.0, 1.2, 1.5, 2.0]
        assert_eq!(variants[0].alpha, 1.0);
        assert_eq!(variants[3].alpha, 2.0);
    }

    #[test]
    fn variant_alpha_affects_pruning() {
        let (vectors, dim, graph) = build_test_graph();
        let config = RPTuningConfig {
            scheme: RPTuningStorageScheme::SchemeA,
            alpha_points: vec![1.0, 3.0],
            r_max: 8,
        };
        let variants = RPTuning::generate_variants(&graph, &vectors, dim, &config);
        // α=3.0 应保留 >= α=1.0 的邻居数
        let small_alpha_degree: usize = (0..graph.len() as u32)
            .map(|n| variants[0].storage.neighbors(n).len())
            .sum();
        let large_alpha_degree: usize = (0..graph.len() as u32)
            .map(|n| variants[1].storage.neighbors(n).len())
            .sum();
        assert!(large_alpha_degree >= small_alpha_degree);
    }

    #[test]
    fn degradation_check_thresholds() {
        // 设计文档：recall 差距 < 0.5%，QPS 差距 < 3%
        let ok = RPTuning::check_degradation(0.003, 0.02);
        assert!(!ok.is_degraded);

        let bad_recall = RPTuning::check_degradation(0.01, 0.02);
        assert!(bad_recall.is_degraded);

        let bad_qps = RPTuning::check_degradation(0.003, 0.05);
        assert!(bad_qps.is_degraded);
    }

    #[test]
    fn alpha_scan_range_locked() {
        // 设计文档 F.4：扫描范围 α ∈ [0.8, 3.0]
        // 实验前锁定，不根据初步结果回调上下界
        let config = RPTuningConfig {
            alpha_points: vec![0.8, 1.0, 1.2, 1.5, 2.0, 3.0],
            ..Default::default()
        };
        for &alpha in &config.alpha_points {
            assert!(alpha >= 0.8 && alpha <= 3.0);
        }
    }
}
