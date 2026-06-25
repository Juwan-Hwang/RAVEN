//! 第一层：距离计算核
//!
//! 设计文档要点：
//! - 运行时维度分发（避免 binary bloat）：宏预先特化高频维度，其余走动态兜底
//! - 动态兜底路径：chunks_exact(8) 有助于 LLVM 识别规则循环并做自动向量化
//! - 内核选择策略：三阶段筛选（延迟粗筛 + 瞬时 QPS + 持续稳定性验证）
//! - 内存对齐策略：GEMM 缓冲区和向量主存储保证 32/64 字节对齐
//! - 精度三层：f32（主线）/ f16（Week 7+ 按需）/ PQ-OPQ-AVQ（Week 5-6 主线）

pub mod scalar;
pub mod dispatch;
pub mod dynamic;
pub mod kernel;
pub mod aligned;
pub mod avx2;
pub mod avx512;
pub mod f16;

/// 距离度量类型
///
/// 设计文档 F.8：当前文档核心围绕 L2 距离展开
/// 规则层明确标注互斥保护：skip if avq == true && distance == L2
/// IP 场景下打分归一化口径与 β 范围需单独校准，当前不实现
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// L2 欧氏距离（主线）
    L2,
    /// 内积距离（扩展，当前不实现，设计文档 F.8）
    #[allow(dead_code)]
    InnerProduct,
}

/// 距离计算结果类型
pub type Distance = f32;

/// 距离计算核统一接口
pub trait DistanceKernel: Send + Sync {
    /// 计算两个向量之间的距离
    fn distance(&self, a: &[f32], b: &[f32]) -> Distance;
    /// 度量类型
    fn metric(&self) -> DistanceMetric;
    /// 内核名称（用于日志与配置缓存）
    fn name(&self) -> &'static str;
}

// 重新导出核心函数
pub use dynamic::l2_dynamic;
pub use scalar::l2_scalar;
pub use kernel::{KernelVariant, select_kernel, validate_kernel_stability};
pub use aligned::AlignedVec;
pub use avx2::{l2_avx2, l2_dispatch, Avx2Kernel, is_avx2_supported};
pub use avx512::{l2_avx512, Avx512Kernel, is_avx512_supported};
pub use f16::{
    F16, l2_f16, l2_f16_packed, l2_f16_mixed, l2_f16_mixed_simd,
    l2_f16_mixed_avx512, l2_f16_mixed_avx2,
    f32_to_f16_slice, f16_to_f32_slice,
    is_f16c_supported, is_avx512_and_f16c_supported, is_avx2_and_f16c_supported,
};

/// 统一 SIMD 分发：AVX-512 > AVX2 > 动态兜底
///
/// 运行时检测 CPU 特性，优先使用最宽的 SIMD 核：
/// - AVX-512（16-wide）：理论峰值 16x 标量
/// - AVX2（8-wide）：理论峰值 8x 标量
/// - 动态兜底：chunks_exact(8)，依赖编译器自动向量化
///
/// 设计文档 F.10：AVX-512 在部分 Intel 平台可能因降频不如 AVX2，
/// 三阶段筛选（kernel.rs）负责在运行时决定是否降级。
/// 此函数为热路径默认入口，若需精细控制可用 RAVEN_KERNEL 环境变量。
#[inline(always)]
pub fn l2_simd(a: &[f32], b: &[f32]) -> Distance {
    if is_avx512_supported() {
        unsafe { avx512::l2_avx512(a, b) }
    } else if is_avx2_supported() {
        unsafe { avx2::l2_avx2(a, b) }
    } else {
        dynamic::l2_dynamic(a, b)
    }
}
