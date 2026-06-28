//! 第三层：图索引核心
//!
//! 设计文档要点：
//! - 主线选 Vamana/DiskANN 风格（RP-Tuning 提供了 post-hoc 调优路径，是系统级卖点）
//! - α 三段式机制：全局 α（构建时）→ RP-Tuning（post-hoc）→ 局部 α（可选）
//! - 量化感知 RobustPrune（核心研究假设）：归一化打分函数 dist/(μ_dist+ε) + β×error/(μ_error+ε)
//! - 消融实验设计：四层指标（边长分布 + 量化误差分布 + 连通度 + 随机种子方差）
//! - 上层导航：双机制（随机层级 + √N centroid overlay 锚点节点）

pub mod vamana;
pub mod robust_prune;
pub mod rp_tuning;
pub mod quant_aware_prune;
pub mod ablation;
pub mod navigation;
pub mod linear_pool;
pub mod adaptive_ef;

pub use vamana::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
pub use adaptive_ef::AdaptiveEfConfig;
pub use robust_prune::{RobustPrune, RobustPruneConfig, PruneStrategy, DirectionalPrune, DirectionalPruneConfig, prune_dispatch};
pub use rp_tuning::{RPTuning, RPTuningConfig, AlphaVariant};
pub use quant_aware_prune::{QuantAwareRobustPrune, QuantAwarePruneConfig, NormalizationScheme};
pub use ablation::{AblationFramework, AblationMetrics, EdgeLengthHistogram};
pub use navigation::{NavigationLayer, NavigationConfig};
pub use linear_pool::LinearPool;
