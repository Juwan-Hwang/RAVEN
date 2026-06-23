//! 消融实验设计（论文核心证据）
//!
//! 设计文档第三层：
//! 每个 β 值记录四层指标：
//! 1. 平均边长分布（边两端 L2 距离直方图）→ 证明 β 增大时，长程导航边比例提高
//! 2. 量化误差分布（保留边端点的 AVQ 平行分量误差均值）→ 证明 β 增大时，图系统性回避量化不稳定节点
//! 3. 图连通度指标（平均出度、最大出度、孤立节点数）→ 证明量化感知剪枝没有破坏导航连通性
//! 4. 跨随机种子 recall 方差（辅助稳定性指标）→ 验证量化感知剪枝是否放大构建随机性
//!
//! 闭合论证链：
//!   高 β → 量化误差低的边比例上升（拓扑证据）
//!        → recall 提高（性能证据）
//!        → 图导航路径避开了量化不稳定区域（机制解释）
//!
//! 补充对照组（设计文档附录 E）：
//!   对照组 1：标准 RobustPrune + 无量化（f32 全精度）
//!   对照组 2：标准 RobustPrune + AVQ 量化（先量化后建图，β=0）
//!   实验组：量化感知 RobustPrune + AVQ 量化（β > 0）

use crate::memory::HybridBlockedCsr;
use crate::distance::l2_simd;

/// 边长分布直方图
///
/// 设计文档消融指标 1：平均边长分布（边两端 L2 距离直方图）
#[derive(Debug, Clone)]
pub struct EdgeLengthHistogram {
    /// 直方图桶边界
    pub bins: Vec<f32>,
    /// 每个桶的边数
    pub counts: Vec<usize>,
    /// 总边数
    pub total_edges: usize,
    /// 平均边长
    pub mean: f32,
    /// 中位数边长
    pub median: f32,
    /// 95 分位数边长
    pub p95: f32,
    /// 99 分位数边长
    pub p99: f32,
}

impl EdgeLengthHistogram {
    /// 从图和向量计算边长分布
    ///
    /// bins_count: 直方图桶数
    pub fn compute(
        storage: &HybridBlockedCsr,
        vectors: &[f32],
        dim: usize,
        bins_count: usize,
    ) -> Self {
        let mut lengths: Vec<f32> = Vec::new();

        for node in 0..storage.len() as u32 {
            let v = &vectors[node as usize * dim..(node as usize + 1) * dim];
            for &nb in storage.neighbors(node) {
                let nv = &vectors[nb as usize * dim..(nb as usize + 1) * dim];
                lengths.push(l2_simd(v, nv));
            }
        }

        Self::from_values(&lengths, bins_count)
    }

    /// 从距离值列表构建直方图
    pub fn from_values(values: &[f32], bins_count: usize) -> Self {
        if values.is_empty() {
            return Self {
                bins: vec![],
                counts: vec![],
                total_edges: 0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let median = percentile(&sorted, 0.5);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let range = max - min;
        let bin_width = if range > 0.0 { range / bins_count as f32 } else { 1.0 };

        let bins: Vec<f32> = (0..=bins_count).map(|i| min + i as f32 * bin_width).collect();
        let mut counts = vec![0usize; bins_count];

        for &v in &sorted {
            let idx = if v >= max {
                bins_count - 1
            } else {
                ((v - min) / bin_width) as usize
            };
            counts[idx.min(bins_count - 1)] += 1;
        }

        Self {
            bins,
            counts,
            total_edges: values.len(),
            mean,
            median,
            p95,
            p99,
        }
    }
}

/// 量化误差分布
///
/// 设计文档消融指标 2：保留边端点的 AVQ 平行分量误差均值
#[derive(Debug, Clone)]
pub struct ErrorDistribution {
    /// 直方图桶边界
    pub bins: Vec<f32>,
    /// 每个桶的边数
    pub counts: Vec<usize>,
    /// 总边数
    pub total_edges: usize,
    /// 平均误差
    pub mean: f32,
    /// 中位数误差
    pub median: f32,
    /// 95 分位数误差
    pub p95: f32,
    /// 99 分位数误差
    pub p99: f32,
}

impl ErrorDistribution {
    /// 从误差值列表构建分布
    pub fn from_values(values: &[f32], bins_count: usize) -> Self {
        if values.is_empty() {
            return Self {
                bins: vec![],
                counts: vec![],
                total_edges: 0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let median = percentile(&sorted, 0.5);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let range = max - min;
        let bin_width = if range > 0.0 { range / bins_count as f32 } else { 1.0 };

        let bins: Vec<f32> = (0..=bins_count).map(|i| min + i as f32 * bin_width).collect();
        let mut counts = vec![0usize; bins_count];

        for &v in &sorted {
            let idx = if v >= max {
                bins_count - 1
            } else {
                ((v - min) / bin_width) as usize
            };
            counts[idx.min(bins_count - 1)] += 1;
        }

        Self {
            bins,
            counts,
            total_edges: values.len(),
            mean,
            median,
            p95,
            p99,
        }
    }
}

/// 图连通度指标
///
/// 设计文档消融指标 3：平均出度、最大出度、孤立节点数
#[derive(Debug, Clone)]
pub struct ConnectivityMetrics {
    /// 平均出度
    pub mean_degree: f64,
    /// 最大出度
    pub max_degree: usize,
    /// 孤立节点数
    pub isolated_nodes: usize,
    /// 总边数
    pub total_edges: usize,
}

impl ConnectivityMetrics {
    /// 从图存储计算连通度
    pub fn compute(storage: &HybridBlockedCsr) -> Self {
        let stats = storage.log_degree_distribution();
        Self {
            mean_degree: stats.mean_degree,
            max_degree: stats.max_degree,
            isolated_nodes: stats.isolated_nodes,
            total_edges: (0..storage.len() as u32).map(|n| storage.neighbors(n).len()).sum(),
        }
    }
}

/// 跨随机种子 recall 方差
///
/// 设计文档消融指标 4：辅助稳定性指标
/// 验证量化感知剪枝是否放大构建随机性
#[derive(Debug, Clone)]
pub struct RecallVariance {
    /// 不同种子下的 recall 值
    pub recalls: Vec<f32>,
    /// 平均 recall
    pub mean: f32,
    /// 标准差
    pub std_dev: f32,
    /// 方差
    pub variance: f32,
}

impl RecallVariance {
    /// 从 recall 列表计算方差
    pub fn from_recalls(recalls: &[f32]) -> Self {
        if recalls.is_empty() {
            return Self {
                recalls: vec![],
                mean: 0.0,
                std_dev: 0.0,
                variance: 0.0,
            };
        }
        let mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
        let variance = recalls.iter().map(|r| (r - mean).powi(2)).sum::<f32>()
            / recalls.len() as f32;
        let std_dev = variance.sqrt();
        Self {
            recalls: recalls.to_vec(),
            mean,
            std_dev,
            variance,
        }
    }
}

/// 完整消融指标
///
/// 设计文档：每个 β 值记录四层指标
#[derive(Debug, Clone)]
pub struct AblationMetrics {
    /// β 值
    pub beta: f32,
    /// 指标 1：边长分布
    pub edge_length: EdgeLengthHistogram,
    /// 指标 2：量化误差分布
    pub error_distribution: ErrorDistribution,
    /// 指标 3：连通度
    pub connectivity: ConnectivityMetrics,
    /// 指标 4：跨种子 recall 方差
    pub recall_variance: RecallVariance,
    /// recall@10
    pub recall_at_10: f32,
}

/// 消融实验框架
///
/// 设计文档第三层：消融实验设计（论文核心证据）
pub struct AblationFramework {
    /// β 扫描点（设计文档：[0, 0.1, 0.3, 1.0]）
    pub beta_points: Vec<f32>,
    /// 直方图桶数
    pub bins_count: usize,
}

impl Default for AblationFramework {
    fn default() -> Self {
        Self {
            // 设计文档第六层参数扫描空间：beta: [0, 0.1, 0.3, 1.0]
            beta_points: vec![0.0, 0.1, 0.3, 1.0],
            bins_count: 50,
        }
    }
}

impl AblationFramework {
    /// 计算单个 β 的消融指标
    pub fn compute_metrics(
        &self,
        beta: f32,
        storage: &HybridBlockedCsr,
        vectors: &[f32],
        dim: usize,
        error_fn: &dyn Fn(u32, u32) -> f32,
        recalls: &[f32],
        recall_at_10: f32,
    ) -> AblationMetrics {
        let edge_length = EdgeLengthHistogram::compute(storage, vectors, dim, self.bins_count);

        // 计算保留边的量化误差分布
        let mut errors: Vec<f32> = Vec::new();
        for node in 0..storage.len() as u32 {
            for &nb in storage.neighbors(node) {
                errors.push(error_fn(node, nb));
            }
        }
        let error_distribution = ErrorDistribution::from_values(&errors, self.bins_count);

        let connectivity = ConnectivityMetrics::compute(storage);
        let recall_variance = RecallVariance::from_recalls(recalls);

        AblationMetrics {
            beta,
            edge_length,
            error_distribution,
            connectivity,
            recall_variance,
            recall_at_10,
        }
    }

    /// 验证闭合论证链
    ///
    /// 设计文档：
    /// 高 β → 量化误差低的边比例上升（拓扑证据）
    ///      → recall 提高（性能证据）
    ///      → 图导航路径避开了量化不稳定区域（机制解释）
    pub fn verify_argument_chain(metrics: &[AblationMetrics]) -> ArgumentChainResult {
        if metrics.len() < 2 {
            return ArgumentChainResult {
                topology_evidence: false,
                performance_evidence: false,
                mechanism_explanation: false,
                chain_holds: false,
            };
        }

        // 按 β 升序排序
        let mut sorted = metrics.to_vec();
        sorted.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap_or(std::cmp::Ordering::Equal));

        // 拓扑证据：β 增大时，低误差边比例上升
        let low_error_ratio_increasing = sorted.windows(2).all(|w| {
            let low_before = w[0].error_distribution.counts.get(0).copied().unwrap_or(0) as f64
                / w[0].error_distribution.total_edges.max(1) as f64;
            let low_after = w[1].error_distribution.counts.get(0).copied().unwrap_or(0) as f64
                / w[1].error_distribution.total_edges.max(1) as f64;
            low_after >= low_before
        });

        // 性能证据：β 增大时 recall 提高
        let recall_increasing = sorted.windows(2).all(|w| w[1].recall_at_10 >= w[0].recall_at_10);

        // 机制解释：连通度未破坏
        let connectivity_preserved = sorted.iter().all(|m| {
            m.connectivity.isolated_nodes <= m.connectivity.total_edges
                && m.connectivity.mean_degree > 0.0
        });

        ArgumentChainResult {
            topology_evidence: low_error_ratio_increasing,
            performance_evidence: recall_increasing,
            mechanism_explanation: connectivity_preserved,
            chain_holds: low_error_ratio_increasing
                && recall_increasing
                && connectivity_preserved,
        }
    }
}

/// 闭合论证链验证结果
#[derive(Debug, Clone)]
pub struct ArgumentChainResult {
    /// 拓扑证据：β 增大时低误差边比例上升
    pub topology_evidence: bool,
    /// 性能证据：β 增大时 recall 提高
    pub performance_evidence: bool,
    /// 机制解释：连通度未破坏
    pub mechanism_explanation: bool,
    /// 论证链是否成立
    pub chain_holds: bool,
}

/// 计算已排序数组的分位数
fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_length_histogram_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let h = EdgeLengthHistogram::from_values(&values, 5);
        assert_eq!(h.total_edges, 5);
        assert!((h.mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn error_distribution_basic() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let d = ErrorDistribution::from_values(&values, 5);
        assert_eq!(d.total_edges, 5);
        assert!((d.mean - 0.3).abs() < 1e-6);
    }

    #[test]
    fn recall_variance_basic() {
        let recalls = vec![0.9, 0.91, 0.89, 0.92];
        let v = RecallVariance::from_recalls(&recalls);
        assert!((v.mean - 0.905).abs() < 1e-3);
        assert!(v.std_dev > 0.0);
    }

    #[test]
    fn empty_histogram() {
        let h = EdgeLengthHistogram::from_values(&[], 10);
        assert_eq!(h.total_edges, 0);
    }
}
