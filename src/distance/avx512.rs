//! AVX-512 距离核（Week 7-8）
//!
//! 设计文档第一层：
//! AVX-512 内核，16-wide f32，理论峰值是标量的 16 倍、AVX2 的 2 倍。
//!
//! 实现要点：
//! - 使用 _mm512_loadu_ps 加载 16 个 f32
//! - 使用 _mm512_sub_ps 做减法
//! - 使用 _mm512_fmadd_ps（FMA）做 d*d + sum
//! - 尾部用掩码加载 _mm512_maskz_loadu_ps，避免越界
//! - _mm512_reduce_add_ps 做水平求和
//!
//! 注意（设计文档 F.10）：
//! AVX-512 在部分 Intel 平台上可能因降频导致 wall-clock 时间不如 AVX2，
//! 三阶段筛选（kernel.rs）负责在运行时决定是否启用。

use super::{Distance, DistanceKernel, DistanceMetric};
use std::arch::x86_64::*;

/// AVX-512 L2 距离（平方欧氏距离）
///
/// 使用 16-wide SIMD + FMA：
/// - _mm512_loadu_ps: 加载 16 个 f32（unaligned）
/// - _mm512_sub_ps: 向量减法
/// - _mm512_fmadd_ps: d*d + sum（融合乘加）
/// - 尾部用 _mm512_maskz_loadu_ps 掩码加载，避免越界
///
/// 每 cycle 处理 16 个 f32，理论峰值是 AVX2 的 2 倍
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;

    // 主循环：每次处理 16 个 f32
    let mut sum = _mm512_setzero_ps();
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
        let d = _mm512_sub_ps(va, vb);
        // FMA: sum = d*d + sum
        sum = _mm512_fmadd_ps(d, d, sum);
    }

    // 尾部用掩码加载（remainder < 16）
    let mut result = 0.0f32;
    if remainder > 0 {
        let mask = (1u16 << remainder) - 1;
        let offset = chunks * 16;
        let va = _mm512_maskz_loadu_ps(mask, a.as_ptr().add(offset));
        let vb = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(offset));
        let d = _mm512_sub_ps(va, vb);
        let tail_sum = _mm512_fmadd_ps(d, d, _mm512_setzero_ps());
        result = _mm512_reduce_add_ps(tail_sum);
    }

    // 水平求和主循环累加结果
    result + _mm512_reduce_add_ps(sum)
}

/// AVX-512 距离核实现
pub struct Avx512Kernel {
    /// 度量类型
    pub metric: DistanceMetric,
}

impl Avx512Kernel {
    /// 创建 L2 AVX-512 核
    pub fn l2() -> Self {
        Self { metric: DistanceMetric::L2 }
    }
}

impl DistanceKernel for Avx512Kernel {
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> Distance {
        unsafe { l2_avx512(a, b) }
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn name(&self) -> &'static str {
        match self.metric {
            DistanceMetric::L2 => "avx512_l2",
            DistanceMetric::InnerProduct => "avx512_ip",
        }
    }
}

/// 检测当前 CPU 是否支持 AVX-512F（基础指令集）
///
/// 只需要 avx512f：L2 距离只用基础的加减乘加
pub fn is_avx512_supported() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512_matches_scalar() {
        if !is_avx512_supported() {
            eprintln!("AVX-512 not supported, skipping test");
            return;
        }
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32) * 2.0).collect();
        let d_scalar: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d_avx512 = unsafe { l2_avx512(&a, &b) };
        assert!((d_scalar - d_avx512).abs() < 1e-2, "scalar={} avx512={}", d_scalar, d_avx512);
    }

    #[test]
    fn avx512_small_vectors() {
        if !is_avx512_supported() {
            return;
        }
        // 维度 < 16，测试掩码尾部处理
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 5.0, 6.0, 7.0];
        // (1-4)^2+(2-5)^2+(3-6)^2+(4-7)^2 = 9+9+9+9 = 36
        let result = unsafe { l2_avx512(&a, &b) };
        assert!((result - 36.0).abs() < 1e-5, "got {} expected {}", result, 36.0);
    }

    #[test]
    fn avx512_aligned_16() {
        if !is_avx512_supported() {
            return;
        }
        // 维度恰好是 16 的倍数
        let a = [1.0f32; 32];
        let b = [2.0f32; 32];
        // 每个分量差 1，平方 1，共 32 个 = 32
        let result = unsafe { l2_avx512(&a, &b) };
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn avx512_dim_128() {
        if !is_avx512_supported() {
            return;
        }
        // siftsmall 维度 128 = 8 × 16，测试主循环
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0).sin()).collect();
        let d_scalar: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d_avx512 = unsafe { l2_avx512(&a, &b) };
        // 大数值累加，用相对误差避免浮点累加顺序差异
        let rel_err = (d_scalar - d_avx512).abs() / d_scalar.max(1.0);
        assert!(rel_err < 1e-4, "scalar={} avx512={} rel_err={}", d_scalar, d_avx512, rel_err);
    }

    #[test]
    fn avx512_zero_distance() {
        if !is_avx512_supported() {
            return;
        }
        let a = [1.0f32; 100];
        let result = unsafe { l2_avx512(&a, &a) };
        assert!(result.abs() < 1e-4);
    }

    #[test]
    fn avx512_kernel_trait() {
        if !is_avx512_supported() {
            return;
        }
        let kernel = Avx512Kernel::l2();
        let a = [1.0f32; 100];
        let b = [2.0f32; 100];
        let d = kernel.distance(&a, &b);
        assert!((d - 100.0).abs() < 1e-4);
        assert_eq!(kernel.name(), "avx512_l2");
    }
}
