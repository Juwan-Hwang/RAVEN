//! 参数扫描空间
//!
//! 设计文档第六层：
//! 参数扫描空间：
//!   M:               [16, 32, 64]
//!   ef_construction: [100, 200, 400]
//!   alpha:           [1.0, 1.2, 1.5]
//!   kernel:          [auto]
//!   pq_mode:         [pq, opq, avq]
//!   prefetch_window: [2, 4, 8]
//!   beta:            [0, 0.1, 0.3, 1.0]
//!   r_soft_ratio:    [1.3, 1.5, 2.0]
//!   gemm_threshold:  [4, 8, 16]

use crate::config::Config;

/// 参数扫描空间
///
/// 设计文档第六层 ann-benchmarks 参数扫描空间
#[derive(Debug, Clone)]
pub struct ParamSpace {
    /// M 参数
    pub m: Vec<usize>,
    /// ef_construction
    pub ef_construction: Vec<usize>,
    /// α 参数
    pub alpha: Vec<f32>,
    /// 内核选择
    pub kernel: Vec<String>,
    /// PQ 模式
    pub pq_mode: Vec<String>,
    /// prefetch 窗口
    pub prefetch_window: Vec<usize>,
    /// β 参数
    pub beta: Vec<f32>,
    /// R_soft 比例
    pub r_soft_ratio: Vec<f32>,
    /// GEMM 阈值
    pub gemm_threshold: Vec<usize>,
}

impl Default for ParamSpace {
    fn default() -> Self {
        Self {
            // 设计文档第六层参数扫描空间
            m: vec![16, 32, 64],
            ef_construction: vec![100, 200, 400],
            alpha: vec![1.0, 1.2, 1.5],
            kernel: vec!["auto".to_string()],
            pq_mode: vec!["pq".to_string(), "opq".to_string(), "avq".to_string()],
            prefetch_window: vec![2, 4, 8],
            beta: vec![0.0, 0.1, 0.3, 1.0],
            r_soft_ratio: vec![1.3, 1.5, 2.0],
            gemm_threshold: vec![4, 8, 16],
        }
    }
}

impl ParamSpace {
    /// β 与 α 协同调参策略
    ///
    /// 设计文档修正摘要 #20：
    /// 协同调参策略：先固定 β 扫 α，再固定 α 扫 β，最后联合细扫
    ///
    /// optimal_alpha: 第一阶段实测得到的最优 α（评估报告 M5：原硬编码为中间值）
    pub fn coordinated_alpha_beta_scan(&self, optimal_alpha: f32) -> Vec<(f32, f32)> {
        let mut combinations = Vec::new();

        // 第一阶段：固定 β=0（标准 RobustPrune），扫 α baseline
        // 设计文档第三层：第一阶段固定 β=0
        for &alpha in &self.alpha {
            combinations.push((alpha, 0.0));
        }

        // 第二阶段：固定最优 α（第一阶段实测结果），扫 β
        // 设计文档：固定最优 α，扫 β
        for &beta in &self.beta {
            if beta > 0.0 {
                combinations.push((optimal_alpha, beta));
            }
        }

        // 第三阶段：在最优 (α, β) 附近做联合细扫
        // 设计文档：在最优 (α, β) 附近做联合细扫
        let fine_alphas = vec![optimal_alpha - 0.2, optimal_alpha - 0.1, optimal_alpha, optimal_alpha + 0.1, optimal_alpha + 0.2];
        let fine_betas = vec![0.2, 0.3, 0.4, 0.5];
        for &a in &fine_alphas {
            for &b in &fine_betas {
                combinations.push((a, b));
            }
        }

        combinations
    }

    /// 生成完整参数组合（笛卡尔积，用于完整扫描）
    pub fn full_grid(&self) -> Vec<Config> {
        let mut configs = Vec::new();
        for &m in &self.m {
            for &ef in &self.ef_construction {
                for &alpha in &self.alpha {
                    for pq_mode in &self.pq_mode {
                        for &beta in &self.beta {
                            for &r_soft in &self.r_soft_ratio {
                                configs.push(Config {
                                    m,
                                    ef_construction: ef,
                                    alpha,
                                    kernel: "auto".to_string(),
                                    pq_mode: pq_mode.clone(),
                                    prefetch_window: 4,
                                    beta,
                                    r_soft_ratio: r_soft,
                                    gemm_threshold: 8,
                                    avq: pq_mode == "avq",
                                    ..Default::default()
                                });
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    /// 总组合数
    pub fn total_combinations(&self) -> usize {
        self.m.len()
            * self.ef_construction.len()
            * self.alpha.len()
            * self.kernel.len()
            * self.pq_mode.len()
            * self.prefetch_window.len()
            * self.beta.len()
            * self.r_soft_ratio.len()
            * self.gemm_threshold.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_param_space() {
        let ps = ParamSpace::default();
        assert_eq!(ps.m, vec![16, 32, 64]);
        assert_eq!(ps.beta, vec![0.0, 0.1, 0.3, 1.0]);
    }

    #[test]
    fn coordinated_scan_has_phases() {
        let ps = ParamSpace::default();
        // 评估报告 M5：optimal_alpha 应从第一阶段实测结果取（这里用 1.2 模拟）
        let combos = ps.coordinated_alpha_beta_scan(1.2);
        // 第一阶段：β=0 扫 α
        assert!(combos.iter().any(|&(a, b)| b == 0.0 && a == 1.0));
        // 第二阶段：固定 α 扫 β>0
        assert!(combos.iter().any(|&(_, b)| b > 0.0));
        // 第三阶段：联合细扫包含 optimal_alpha 附近
        assert!(combos.iter().any(|&(a, b)| (a - 1.2).abs() < 0.01 && b == 0.3));
    }

    #[test]
    fn full_grid_generates_configs() {
        let ps = ParamSpace::default();
        let configs = ps.full_grid();
        assert!(!configs.is_empty());
    }
}
