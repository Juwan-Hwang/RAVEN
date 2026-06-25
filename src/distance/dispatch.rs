//! 运行时维度分发（避免 binary bloat）
//!
//! 设计文档第一层：
//! ann-benchmarks 数据集维度在运行时从 HDF5 文件读取，涵盖 25、50、100、128、
//! 256、784、960、1536 等多种规格。对所有维度做编译期特化会导致 binary bloat
//! 和编译时间失控。
//!
//! 推荐实现：宏预先特化高频维度，其余走动态兜底。

use super::l2_simd;

/// 维度分发宏：预先特化高频维度，其余走动态兜底
///
/// 设计文档原文实现。使用方式：
/// ```ignore
/// dispatch_dim!(dim, l2_kernel);
/// ```
/// 其中 `$kernel` 是一个泛型函数 `fn f<const D: usize>()`。
///
/// 注意：const generics 在函数指针传递上有局限，实际使用时通常直接 match
/// 并调用特化函数。这里保留宏形式以匹配设计文档，并提供基于维度的分发辅助。
#[macro_export]
macro_rules! dispatch_dim {
    ($dim:expr, $kernel:expr) => {
        match $dim {
            64   => $kernel::<64>(),
            128  => $kernel::<128>(),
            256  => $kernel::<256>(),
            384  => $kernel::<384>(),
            768  => $kernel::<768>(),
            960  => $kernel::<960>(),
            1536 => $kernel::<1536>(),
            _    => $crate::distance::dispatch::dispatch_dynamic_slice($dim),
        }
    };
}

/// 动态兜底入口（宏的 fallback 分支调用）
///
/// 返回一个标识，表示走动态路径。实际距离计算由调用方在动态路径上完成。
pub fn dispatch_dynamic_slice(dim: usize) -> DispatchResult {
    DispatchResult::Dynamic { dim }
}

/// 维度分发结果
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub enum DispatchResult {
    /// 命中编译期特化维度
    Specialized { dim: usize },
    /// 走动态兜底路径
    Dynamic { dim: usize },
}

/// 按维度选择距离计算函数（返回函数指针）
///
/// 返回 l2_simd 统一分发函数（AVX-512 > AVX2 > dynamic），
/// 运行时自动选择最优 SIMD 核。
pub fn select_l2(dim: usize) -> fn(&[f32], &[f32]) -> f32 {
    match dim {
        // 高频维度走 SIMD 实现（AVX-512 > AVX2 > dynamic）
        // 注：当前不使用 const generics 特化以避免 binary bloat，
        // l2_simd 运行时自动选择最优 SIMD 核
        64 | 128 | 256 | 384 | 768 | 960 | 1536 => l2_simd,
        _ => l2_simd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_known_dim() {
        let r = dispatch_dynamic_slice(128);
        assert!(matches!(r, DispatchResult::Dynamic { dim: 128 }));
    }

    #[test]
    fn select_l2_returns_function() {
        let f = select_l2(768);
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let d = f(&a, &b);
        assert!((d - 27.0).abs() < 1e-6);
    }
}
