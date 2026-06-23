//! 第六层：评测自动化
//!
//! 设计文档要点：
//! - 微基准工具：divan（日常迭代）+ criterion（正式回归）
//! - 分阶段接入策略：先用纯 Rust 扁平二进制格式把本地基准和核心内核跑通
//! - 两种模式：ann-benchmarks（主战场）+ big-ann-benchmarks（扩展目标）
//! - 规则驱动 Auto-tuner（含 DAG 冲突校验）

pub mod config;
pub mod rules;
pub mod local_bench;
pub mod params;

pub use config::{Config, ConfigSource, merge_config};
pub use rules::{RuleEngine, Rule, RuleSeverity, ConflictError};
pub use local_bench::{LocalBenchmark, BenchmarkResult};
pub use params::ParamSpace;
