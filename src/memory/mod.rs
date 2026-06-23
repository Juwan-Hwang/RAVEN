//! 第二层：内存布局
//!
//! 设计文档要点：
//! - 图结构索引化原则：用 u32 节点 ID 做邻接表、候选集、visited history
//! - 节点 ID 边界：最大节点数 2^32-2，u32::MAX 保留为 sentinel
//! - 索引序列化格式：16 字节文件头（magic + version + flags + crc32）
//! - Hybrid Blocked-CSR 图存储：主块固定 R_max 槽 + overflow 区
//! - Visited 标记：Clear-List 模式 Vec<u8>，O(V) 重置
//! - 批量查询架构：feature-gate 隔离，独立 QueryContext

pub mod visited;
pub mod graph;
pub mod serialize;
pub mod query_ctx;

pub use visited::VisitedTracker;
pub use graph::{HybridBlockedCsr, GraphStorage};
pub use serialize::{IndexHeader, INDEX_MAGIC, INDEX_VERSION, Serializable};
pub use query_ctx::QueryContext;
