//! RAVEN (Retrieval-Aware Vector Engine with Navigation)
//!
//! 高性能 Rust ANN 检索库，合并三条技术路线：
//! - RP-Tuning 后验 α 调优（Vamana/DiskANN 风格）
//! - AVQ 检索感知量化（优化 retrieval loss 而非 reconstruction loss）
//! - 量化误差反向影响图剪枝决策的图网络结构
//!
//! 主战场：ann-benchmarks 单查询 recall-QPS Pareto frontier（fit-in-RAM，单 CPU）
//! 扩展场景：big-ann-benchmarks 100M/1B 级 SSD（⚠️ 未实现，仅预留接口）
//!
//! 设计文档冻结版（修订 2），实现依照文档分层组织：
//! - 第一层：距离计算核
//! - 第二层：内存布局
//! - 第三层：图索引核心
//! - 第四层：并发构建
//! - 第五层：量化层
//! - 第六层：评测自动化

#![warn(missing_docs)]

pub mod distance;
pub mod memory;
pub mod graph;
pub mod build;
pub mod quant;
pub mod config;
pub mod bench;

#[cfg(feature = "python")]
pub mod python;

/// 库版本号，写入索引元数据随索引文件落盘（设计文档 F.7）
pub const BUILD_VERSION: &str = "0.1.0";

/// u32 节点 ID sentinel 值，标记无效邻居槽位、overflow 链末端
/// 系统支持最大节点数为 2^32-2（约 42.9 亿），u32::MAX 保留（设计文档第二层）
pub const SENTINEL: u32 = u32::MAX;

/// 重新导出常用类型，方便外部使用
pub use distance::{DistanceMetric, l2_dynamic, l2_simd};
pub use memory::{VisitedTracker, HybridBlockedCsr, QueryContext};
pub use graph::{VamanaGraph, RobustPrune, RPTuning};
pub use build::{BuildConfig, BuildMetadata, ChaCha8Rng};
