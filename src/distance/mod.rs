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
#[cfg(target_arch = "aarch64")]
pub mod neon;

/// 距离度量类型
///
/// 设计文档 F.8：当前文档核心围绕 L2 距离展开
/// 规则层明确标注互斥保护：skip if avq == true && distance == L2
/// IP 场景下打分归一化口径与 β 范围需单独校准，当前不实现
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// L2 欧氏距离（主线）
    L2,
    /// 内积距离（⚠️ 未实现，设计文档 F.8 预留适配边界）
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
    F16Dataset,
};

/// 统一 SIMD 分发：AVX-512 > AVX2 > 动态兜底
///
/// 当 `target-cpu=native` 编译时，`#[cfg(target_feature)]` 在编译期确定 CPU 特性，
/// 消除每次调用的 `is_x86_feature_detected!` 运行时开销（atomic load + branch）。
/// 与 SQ8 距离核 (`l2_sq8`) 使用相同的编译期分发手法。
///
/// - AVX-512（16-wide）：理论峰值 16x 标量
/// - AVX2（8-wide）：理论峰值 8x 标量
/// - 动态兜底：chunks_exact(8)，依赖编译器自动向量化
///
/// 设计文档 F.10：AVX-512 在部分 Intel 平台可能因降频不如 AVX2，
/// 三阶段筛选（kernel.rs）负责在运行时决定是否降级（仅 DistanceKernel trait 路径）。
/// 此函数为热路径默认入口，编译期已选定最优内核。
#[inline(always)]
pub fn l2_simd(a: &[f32], b: &[f32]) -> Distance {
    crate::diag_count!(FULL_DIST);
    // 编译期已知 AVX-512F：零运行时开销
    #[cfg(target_feature = "avx512f")]
    {
        unsafe { avx512::l2_avx512(a, b) }
    }
    // 编译期已知 AVX2+FMA（非 AVX-512 CPU）
    #[cfg(all(target_feature = "avx2", target_feature = "fma", not(target_feature = "avx512f")))]
    {
        unsafe { avx2::l2_avx2(a, b) }
    }
    // 运行时检测（x86 非 native 编译或无 SIMD 的 CPU）
    #[cfg(all(
        target_arch = "x86_64",
        not(any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma")
        ))
    ))]
    {
        if is_avx512_supported() {
            unsafe { avx512::l2_avx512(a, b) }
        } else if is_avx2_supported() {
            unsafe { avx2::l2_avx2(a, b) }
        } else {
            dynamic::l2_dynamic(a, b)
        }
    }
    // aarch64：NEON 为基线特性，编译期直选
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::l2_neon(a, b) }
    }
    // 其他架构：动态兜底
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dynamic::l2_dynamic(a, b)
    }
}

/// H14：angular（cosine）数据集预处理——逐行 L2 归一化（就地）。
///
/// 单位向量上 `||x - q||² = 2 - 2·cos(x, q)`，因此在归一化后的数据上用 L2 建图/搜索，
/// 排序与 cosine 距离（ann-benchmarks `angular` = 1 - cos）完全等价。零向量保持为零（cos 未定义），返回零向量个数。
pub fn normalize_rows(data: &mut [f32], dim: usize) -> usize {
    assert!(dim > 0 && data.len() % dim == 0);
    let mut zeros = 0;
    for row in data.chunks_exact_mut(dim) {
        let s: f64 = row.iter().map(|&x| x as f64 * x as f64).sum();
        if s == 0.0 {
            zeros += 1;
            continue;
        }
        let inv = (1.0 / s.sqrt()) as f32;
        row.iter_mut().for_each(|x| *x *= inv);
    }
    zeros
}

#[cfg(test)]
mod h14_angular_tests {
    use super::*;

    #[test]
    fn h14_normalized_l2_order_equals_cosine_order() {
        let dim = 100;
        let mut s = 7u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0 };
        let mut data: Vec<f32> = (0..500 * dim).map(|_| rnd() * 3.0).collect();
        let mut q: Vec<f32> = (0..dim).map(|_| rnd()).collect();
        let raw = data.clone();
        let rq = q.clone();
        data[..dim].iter_mut().for_each(|x| *x = 0.0); // 一个零向量
        assert_eq!(normalize_rows(&mut data, dim), 1);
        assert_eq!(normalize_rows(&mut q, dim), 0);
        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let ab: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
            ab / (na * nb)
        };
        let mut by_l2: Vec<(usize, f32)> = (1..500).map(|i| (i, l2_simd(&q, &data[i * dim..(i + 1) * dim]))).collect();
        let mut by_cos: Vec<(usize, f64)> = (1..500).map(|i| (i, 1.0 - cos(&rq, &raw[i * dim..(i + 1) * dim]))).collect();
        by_l2.sort_by(|a, b| a.1.total_cmp(&b.1));
        by_cos.sort_by(|a, b| a.1.total_cmp(&b.1));
        // 前 50 名完全一致；且 ||x-q||² ≈ 2(1-cos)
        assert_eq!(by_l2.iter().take(50).map(|x| x.0).collect::<Vec<_>>(), by_cos.iter().take(50).map(|x| x.0).collect::<Vec<_>>());
        for (i, d) in by_cos.iter().take(50) {
            let l2 = l2_simd(&q, &data[i * dim..(i + 1) * dim]) as f64;
            assert!((l2 - 2.0 * d).abs() < 1e-4, "l2={l2} 2(1-cos)={}", 2.0 * d);
        }
        // 归一化后单位长度
        for i in 1..500 {
            let n: f32 = data[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum();
            assert!((n - 1.0).abs() < 1e-5);
        }
    }
}
