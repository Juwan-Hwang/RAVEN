//! 配置合并（全入口覆盖）
//!
//! 设计文档第六层 F.12：
//! 规则引擎校验必须覆盖所有参数来源，不允许校验后静默覆盖：
//!   参数来源优先级（从低到高）：
//!     1. 默认值（代码内置）
//!     2. TOML 配置文件
//!     3. CLI 参数
//!     4. 环境变量
//!     5. feature flag（编译期，优先级最高）
//!
//! 合并顺序：按优先级从低到高依次覆盖，得到唯一最终配置
//! 校验时机：合并完成后统一校验一次
//! 校验后：Config 不可变，任何运行时覆盖须走完整 merge + validate 流程

use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::config::rules::ConflictError;

/// 配置来源
///
/// 设计文档 F.12：参数来源优先级（从低到高）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    /// 1. 默认值（代码内置）
    Default,
    /// 2. TOML 配置文件
    Toml,
    /// 3. CLI 参数
    Cli,
    /// 4. 环境变量
    Env,
    /// 5. feature flag（编译期，优先级最高）
    Feature,
}

/// 运行时配置
///
/// 设计文档第六层：所有配置入口必须先合并成唯一最终配置，再统一校验
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// 距离度量
    pub distance: String,
    /// M 参数（HNSW 风格）
    pub m: usize,
    /// ef_construction
    pub ef_construction: usize,
    /// α 参数（Vamana 剪枝激进度）
    pub alpha: f32,
    /// 内核选择
    pub kernel: String,
    /// PQ 模式：pq / opq / avq
    pub pq_mode: String,
    /// prefetch 窗口
    pub prefetch_window: usize,
    /// β 参数（量化感知权重）
    pub beta: f32,
    /// R_soft 比例
    pub r_soft_ratio: f32,
    /// GEMM 阈值
    pub gemm_threshold: usize,
    /// 是否启用 AVX-512
    pub avx512: bool,
    /// 是否启用批量模式
    pub batch_mode: bool,
    /// 是否启用 AVQ
    pub avq: bool,
    /// 是否走 GEMM 路径
    pub gemm_path: bool,
    /// 候选数（用于规则校验）
    pub candidate_count: usize,
    /// 向量维度（用于 sub_dim 整除校验）
    #[serde(default = "default_dim")]
    pub dim: usize,
    /// 最终返回数 k（top_n ≥ k 校验）
    #[serde(default = "default_k")]
    pub k: usize,
    /// rerank 候选数 top_n（top_n ≥ k 校验）
    #[serde(default = "default_top_n")]
    pub top_n: usize,
    /// AVQ codebook 大小 K（必须是 2 的幂次）
    #[serde(default = "default_codebook_k")]
    pub codebook_k: usize,
    /// AVQ 子维度（必须整除 dim）
    #[serde(default = "default_sub_dim")]
    pub sub_dim: usize,
    /// AVQ 混合权重 α（reconstruction + retrieval-aware）
    /// 设计文档：α * recon_loss + (1-α) * ret_loss，α ∈ [0.0, 1.0]
    #[serde(default = "default_avq_alpha")]
    pub avq_alpha: f32,
}

fn default_dim() -> usize { 128 }
fn default_k() -> usize { 10 }
fn default_top_n() -> usize { 100 }
fn default_codebook_k() -> usize { 256 }
fn default_sub_dim() -> usize { 8 }
fn default_avq_alpha() -> f32 { 0.30 }

impl Default for Config {
    fn default() -> Self {
        Self {
            distance: "l2".to_string(),
            m: 32,
            ef_construction: 200,
            alpha: 1.2,
            kernel: "auto".to_string(),
            pq_mode: "avq".to_string(),
            prefetch_window: 4,
            beta: 0.3,
            r_soft_ratio: 1.5,
            gemm_threshold: 8,
            avx512: false,
            batch_mode: false,
            avq: true,
            gemm_path: false,
            candidate_count: 0,
            dim: default_dim(),
            k: default_k(),
            top_n: default_top_n(),
            codebook_k: default_codebook_k(),
            sub_dim: default_sub_dim(),
            avq_alpha: default_avq_alpha(),
        }
    }
}

impl Config {
    /// 从 TOML 文件加载
    pub fn from_toml(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let s = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&s)?)
    }

    /// 从环境变量覆盖
    ///
    /// 设计文档 F.12：环境变量优先级高于 CLI
    pub fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("RAVEN_DISTANCE") {
            self.distance = v;
        }
        if let Ok(v) = std::env::var("RAVEN_M") {
            if let Ok(m) = v.parse() {
                self.m = m;
            }
        }
        if let Ok(v) = std::env::var("RAVEN_ALPHA") {
            if let Ok(a) = v.parse() {
                self.alpha = a;
            }
        }
        if let Ok(v) = std::env::var("RAVEN_BETA") {
            if let Ok(b) = v.parse() {
                self.beta = b;
            }
        }
        if let Ok(v) = std::env::var("RAVEN_KERNEL") {
            self.kernel = v;
        }
        if let Ok(v) = std::env::var("RAVEN_PQ_MODE") {
            self.pq_mode = v;
        }
        if let Ok(v) = std::env::var("RAVEN_AVX512") {
            self.avx512 = v == "1" || v == "true";
        }
    }

    /// 应用 feature flag（编译期，优先级最高）
    ///
    /// 设计文档 F.12：feature flag（编译期，优先级最高）
    pub fn apply_feature_flags(&mut self) {
        #[cfg(feature = "batch_mode")]
        {
            self.batch_mode = true;
        }
        #[cfg(feature = "avx512")]
        {
            self.avx512 = true;
        }
    }
}

/// 合并所有配置来源
///
/// 设计文档第六层 build_config：
/// 先合并所有来源，再统一校验，校验后 cfg 不可变
pub fn merge_config(
    toml_path: Option<&Path>,
    cli: Option<&Config>,
    env: bool,
) -> Result<Config, crate::config::rules::ConflictError> {
    // 1. 默认值（代码内置）
    let mut cfg = Config::default();

    // 2. TOML 配置文件
    if let Some(path) = toml_path {
        if path.exists() {
            cfg = Config::from_toml(path)
                .map_err(|e| ConflictError::ConfigLoad(e.to_string()))?;
        }
    }

    // 3. CLI 参数
    if let Some(cli_cfg) = cli {
        cfg.distance = cli_cfg.distance.clone();
        cfg.m = cli_cfg.m;
        cfg.ef_construction = cli_cfg.ef_construction;
        cfg.alpha = cli_cfg.alpha;
        cfg.kernel = cli_cfg.kernel.clone();
        cfg.pq_mode = cli_cfg.pq_mode.clone();
        cfg.prefetch_window = cli_cfg.prefetch_window;
        cfg.beta = cli_cfg.beta;
        cfg.r_soft_ratio = cli_cfg.r_soft_ratio;
        cfg.gemm_threshold = cli_cfg.gemm_threshold;
        cfg.avq = cli_cfg.avq;
        cfg.gemm_path = cli_cfg.gemm_path;
        cfg.candidate_count = cli_cfg.candidate_count;
        // Week 7-8：新参数也走 CLI 覆盖
        cfg.dim = cli_cfg.dim;
        cfg.k = cli_cfg.k;
        cfg.top_n = cli_cfg.top_n;
        cfg.codebook_k = cli_cfg.codebook_k;
        cfg.sub_dim = cli_cfg.sub_dim;
        cfg.avq_alpha = cli_cfg.avq_alpha;
        // 注：avx512 和 batch_mode 不走 CLI，由 feature flag 控制（优先级最高）
    }

    // 4. 环境变量
    if env {
        cfg.apply_env();
    }

    // 5. feature flag（编译期，优先级最高）
    cfg.apply_feature_flags();

    // 校验：合并完成后统一校验一次（设计文档 F.12）
    let engine = crate::config::rules::RuleEngine::default();
    engine.validate_and_warn(&cfg)?;

    // 校验后 cfg 不可变
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = Config::default();
        assert_eq!(cfg.distance, "l2");
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.alpha, 1.2);
    }

    #[test]
    fn merge_config_defaults() {
        // 默认配置 avq=true + distance=l2 违反 avq_l2_conflict（设计意图）
        let result = merge_config(None, None, false);
        assert!(result.is_err(), "default config should violate avq_l2_conflict");
    }

    #[test]
    fn merge_config_cli_overrides() {
        let mut cli = Config::default();
        cli.m = 64;
        cli.avq = false; // 避免违反 avq_l2_conflict
        let cfg = merge_config(None, Some(&cli), false).unwrap();
        assert_eq!(cfg.m, 64);
    }
}
