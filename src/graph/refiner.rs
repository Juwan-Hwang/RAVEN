//! Refiner 框架：粗量化搜索 + 精量化 rerank 通用管道
//!
//! 设计文档第四层 Refiner 框架：
//! 粗量化搜索（SQ8/PQ4/PQ8）→ 候选集 → 精量化 rerank → 最终结果
//!
//! Glass 有通用的 Refiner 框架，RAVEN 此前是手动拼接。
//! 本模块提取通用 rerank 逻辑，支持任意精量化器组合：
//! - F32Refiner: f32 全精度（当前主线，SIFT-128 默认）
//! - F16Refiner: f16 半精度（带宽减半，高维数据集 768+ 收益显著）

use crate::distance::{l2_simd, f16::{F16, l2_f16_mixed_simd}};

/// 精量化 Refiner trait：对粗搜索候选用精确距离重排序
///
/// 设计文档：粗量化距离排序与精确距离高度一致，
/// 只需对 top-N 候选 rerank 即可覆盖 top-k。
pub trait Refiner {
    /// 对候选集用精确距离重排序，返回 top-k
    ///
    /// `candidates` 已按粗量化距离升序排列
    /// `rerank_factor` 控制重排序的候选数 = k × factor
    fn rerank(&self, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)>;
}

/// f32 全精度 Refiner（默认主线）
///
/// 使用 f32 全精度向量计算精确 L2 距离
/// 适用场景：所有数据集（SIFT-128 等低维数据集的默认选择）
pub struct F32Refiner<'a> {
    /// f32 全精度向量
    vectors: &'a [f32],
    /// 维度
    dim: usize,
}

impl<'a> F32Refiner<'a> {
    /// 创建 f32 全精度 Refiner
    pub fn new(vectors: &'a [f32], dim: usize) -> Self {
        Self { vectors, dim }
    }
}

impl<'a> Refiner for F32Refiner<'a> {
    #[inline]
    fn rerank(&self, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)> {
        let rerank_n = (k * rerank_factor).max(k).min(candidates.len());
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(rerank_n)
            .map(|(id, _)| {
                let dist = l2_simd(
                    query,
                    &self.vectors[id as usize * self.dim..(id as usize + 1) * self.dim],
                );
                (id, dist)
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        results
    }
}

/// f16 半精度 Refiner（带宽优化）
///
/// 数据库预量化为 f16（2B/元素 vs f32 4B/元素），
/// rerank 阶段内存带宽减半，F16C 指令在寄存器中转回 f32 计算
///
/// 适用场景：维度 ≥ 512 的高维数据集（Cohere-768、GIST-960 等）
/// 低维数据集（SIFT-128）：rerank 数据已在 L1 cache，带宽不是瓶颈，收益极小
pub struct F16Refiner<'a> {
    /// f16 预量化向量
    vectors: &'a [F16],
    /// 维度
    dim: usize,
}

impl<'a> F16Refiner<'a> {
    /// 创建 f16 半精度 Refiner
    pub fn new(vectors: &'a [F16], dim: usize) -> Self {
        Self { vectors, dim }
    }
}

impl<'a> Refiner for F16Refiner<'a> {
    #[inline]
    fn rerank(&self, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)> {
        let rerank_n = (k * rerank_factor).max(k).min(candidates.len());
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(rerank_n)
            .map(|(id, _)| {
                let dist = l2_f16_mixed_simd(
                    query,
                    &self.vectors[id as usize * self.dim..(id as usize + 1) * self.dim],
                );
                (id, dist)
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        results
    }
}

/// 通用 rerank 函数（零分配热路径之外的辅助函数）
///
/// 供 batch_search 等无法直接使用 trait 对象的场景调用
#[inline]
pub fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rerank_factor: usize,
) -> Vec<(u32, f32)> {
    F32Refiner::new(vectors, dim).rerank(query, candidates, k, rerank_factor)
}

/// f16 rerank 函数（同上）
#[inline]
pub fn rerank_f16(
    f16_vectors: &[F16],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rerank_factor: usize,
) -> Vec<(u32, f32)> {
    F16Refiner::new(f16_vectors, dim).rerank(query, candidates, k, rerank_factor)
}
