//! 评测自动化模块入口
//!
//! 设计文档第六层：评测自动化
//! - 微基准工具：divan（日常迭代）+ criterion（正式回归）
//! - 分阶段接入策略：先用纯 Rust 扁平二进制格式把本地基准和核心内核跑通

pub use crate::config::{LocalBenchmark, BenchmarkResult, ParamSpace};
