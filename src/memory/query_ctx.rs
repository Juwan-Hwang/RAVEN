//! 批量查询架构（feature-gate 隔离，不污染主路径）
//!
//! 设计文档第二层：
//! 不同查询强行共享 visited 会改变搜索语义，直接损害召回率。
//! 批量模式通过 feature-gate 与单查询主路径隔离，使用显式 QueryContext 传参，
//! 预留 NUMA 亲和性扩展位。
//!
//! GEMM 阈值（默认 8）进入自动调参空间，以基准结果为准。
//! 该模式作为辅助吞吐评测，不参与主成绩。

use crate::distance::AlignedVec;
use super::VisitedTracker;

/// 查询上下文
///
/// 设计文档原文：
/// - visited: VisitedTracker（独立 visited，避免共享损害召回）
/// - candidate_buf: 候选集缓冲
/// - vector_pack_buf: GEMM 路径的向量打包缓冲（对齐）
/// - numa_node: 预留 NUMA 亲和性扩展位，当前不强制绑定
///
/// 设计文档 F.2：QueryContext 由调用方持有，不走 arena
/// 设计文档：thread_local! 平台行为不确定，热路径不押注 TLS，用显式 QueryContext
pub struct QueryContext {
    /// 独立 visited 标记，避免不同查询共享损害召回
    pub visited: VisitedTracker,
    /// 候选集缓冲
    pub candidate_buf: Vec<u32>,
    /// GEMM 路径的向量打包缓冲（32 字节对齐）
    pub vector_pack_buf: AlignedVec<f32>,
    /// NUMA 亲和性扩展位，当前不强制绑定
    /// 设计文档第二层：预留 NUMA 亲和性扩展位
    pub numa_node: Option<usize>,
}

impl QueryContext {
    /// 创建查询上下文
    ///
    /// n: 节点总数
    /// ef_search: 搜索宽度
    /// dim: 向量维度（用于 vector_pack_buf 预分配）
    pub fn new(n: usize, ef_search: usize, dim: usize) -> Self {
        Self {
            visited: VisitedTracker::new(n, ef_search),
            candidate_buf: Vec::with_capacity(ef_search * 3),
            vector_pack_buf: crate::distance::aligned::aligned_vec_f32(ef_search * dim),
            numa_node: None,
        }
    }

    /// 重置查询上下文（用于复用）
    pub fn reset(&mut self) {
        self.visited.reset();
        self.candidate_buf.clear();
        // vector_pack_buf 不需要清空，写入时覆盖
    }

    /// 设置 NUMA 亲和性（预留接口，当前不强制绑定）
    ///
    /// ⚠️ 未实现：仅存储偏好，不实际绑定 CPU 亲和性
    #[allow(dead_code)]
    pub fn set_numa_node(&mut self, node: usize) {
        self.numa_node = Some(node);
    }
}

/// GEMM 阈值：候选数超过此值才走 GEMM 路径
/// 设计文档：默认 8，进入自动调参空间，以基准结果为准
pub const GEMM_THRESHOLD: usize = 8;

/// 批量距离计算（feature-gate 隔离）
///
/// 设计文档第二层 batch_distance：
/// - 候选数 <= 8 走标量 SIMD 路径
/// - 候选数 > 8 走 GEMM 路径（pack_vectors_aligned + gemm_path）
///
/// ⚠️ GEMM 路径未实现：当前 gemm_path 为标量回退（评估报告 M1）
#[cfg(feature = "batch_mode")]
pub fn batch_distance(
    ctx: &mut QueryContext,
    queries: &[&[f32]],
    candidates: &[u32],
    vectors: &[f32],
    dim: usize,
) -> Vec<Vec<f32>> {
    if candidates.len() <= GEMM_THRESHOLD {
        scalar_simd_path(queries, candidates, vectors, dim)
    } else {
        pack_vectors_aligned(candidates, &mut ctx.vector_pack_buf, vectors, dim);
        gemm_path(queries, &ctx.vector_pack_buf, dim)
    }
}

/// 标量 SIMD 路径（候选数少时）
#[cfg(feature = "batch_mode")]
fn scalar_simd_path(
    queries: &[&[f32]],
    candidates: &[u32],
    vectors: &[f32],
    dim: usize,
) -> Vec<Vec<f32>> {
    use crate::distance::l2_simd;
    queries
        .iter()
        .map(|q| {
            candidates
                .iter()
                .map(|&c| {
                    let v = &vectors[c as usize * dim..(c as usize + 1) * dim];
                    l2_simd(q, v)
                })
                .collect()
        })
        .collect()
}

/// 对齐打包候选向量
#[cfg(feature = "batch_mode")]
fn pack_vectors_aligned(
    candidates: &[u32],
    buf: &mut AlignedVec<f32>,
    vectors: &[f32],
    dim: usize,
) {
    let needed = candidates.len() * dim;
    if buf.len() < needed {
        *buf = crate::distance::aligned::aligned_vec_f32(needed);
    }
    for (i, &c) in candidates.iter().enumerate() {
        let v = &vectors[c as usize * dim..(c as usize + 1) * dim];
        buf[i * dim..(i + 1) * dim].copy_from_slice(v);
    }
}

/// GEMM 路径（候选数多时）
///
/// ⚠️ 未实现：当前为标量回退，尚未接入真正矩阵乘法（评估报告 M1）
/// 真正 GEMM 需要 BLIS/mkl-blis 或 hand-written AVX-512 GEMM kernel
#[cfg(feature = "batch_mode")]
fn gemm_path(queries: &[&[f32]], packed: &[f32], dim: usize) -> Vec<Vec<f32>> {
    use crate::distance::l2_simd;
    let n_candidates = packed.len() / dim;
    queries
        .iter()
        .map(|q| {
            (0..n_candidates)
                .map(|i| {
                    let v = &packed[i * dim..(i + 1) * dim];
                    l2_simd(q, v)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_context_new() {
        let ctx = QueryContext::new(1000, 200, 768);
        assert_eq!(ctx.visited.len(), 1000);
        assert!(ctx.numa_node.is_none());
    }

    #[test]
    fn query_context_reset() {
        let mut ctx = QueryContext::new(1000, 200, 768);
        ctx.visited.visit(5);
        ctx.candidate_buf.push(10);
        ctx.reset();
        assert_eq!(ctx.visited.visited_count(), 0);
        assert!(ctx.candidate_buf.is_empty());
    }

    #[test]
    fn query_context_numa_placeholder() {
        let mut ctx = QueryContext::new(100, 64, 128);
        assert!(ctx.numa_node.is_none());
        ctx.set_numa_node(1);
        assert_eq!(ctx.numa_node, Some(1));
    }

    #[test]
    fn gemm_threshold_is_8() {
        assert_eq!(GEMM_THRESHOLD, 8);
    }
}
