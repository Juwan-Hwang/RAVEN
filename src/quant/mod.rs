//! 第五层：量化层
//!
//! 设计文档：
//! PQ（基线）：优化 reconstruction loss，作为对照组
//! OPQ：构建前旋转向量使各子空间方差均等，再做 PQ；开发成本低，是进入 AVQ 的工程桥梁
//! AVQ（参考 ScaNN score-aware quantization 思路）：
//!   优化 retrieval-aware quantization loss
//!   核心洞见：量化误差的平行分量对 inner product 高分候选的召回率影响远大于正交分量
//!   → codebook 训练时用检索目标加权
//!   → 与量化感知 RobustPrune 的归一化误差项直接联动
//!
//! ⚠️ retrieval-aware 信号来源须在实现阶段明确指定：
//!   AVQ codebook 训练在图构建之前，不依赖图结构；
//!   检索感知信号来自 sampled high-score pairs 或预采样近邻对。
//!   Week 5 同时实现最小双分支，用消融数据决定主线（见附录 F.11）。

pub mod pq;
pub mod opq;
pub mod avq;
pub mod sq8;
pub mod pq4;

pub use pq::{PQCodebook, PQ, PQ8Dataset};
pub use opq::OPQRotation;
pub use avq::{AVQCodebook, AVQ, QuantizationMode, TrainingSignal};
pub use sq8::{SQ8Params, SQ8Dataset, l2_sq8, l2_sq8_avx2, l2_sq8_raw, l2_sq8_raw_avx2, is_sq8_avx2_supported};
pub use pq4::{PQ4Codebook, PQ4Dataset};
