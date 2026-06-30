//! 第四层：并发构建
//!
//! 设计文档要点：
//! - 锁实现策略：std::sync（RAVEN 构建期为单线程或 rayon 并行，无细粒度锁竞争）
//! - 延迟剪枝策略：硬上限 R_max + 软上限 R_soft = 1.5 × R_max
//! - 构建可复现性约定：ChaCha8 RNG + 固定种子 + 确定性分片，元数据落盘
//! - 纯函数式 Pipeline（离线层，绝不进入查询热路径）

pub mod rng;
pub mod metadata;
pub mod pipeline;
pub mod delayed_prune;

pub use rng::ChaCha8Rng;
pub use metadata::{BuildMetadata, BuildConfig};
pub use pipeline::{BuildPipeline, PipelineStage};
pub use delayed_prune::DelayedPruneController;
