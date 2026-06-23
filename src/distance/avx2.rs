//! AVX2 距离核（Week 3-4）
//!
//! 设计文档第一层：
//! AVX2 距离核 + cargo asm 验证向量化
//!
//! 实现要点：
//! - 使用 _mm256_loadu_ps 加载 8 个 f32
//! - 使用 _mm256_sub_ps 做减法
//! - 使用 _mm256_fmadd_ps（FMA）做 d*d + sum
//! - 尾部用标量处理
//! - aligned vs unaligned load 以基准结果为准（当前用 loadu，兼容未对齐数据）

use super::{Distance, DistanceKernel, DistanceMetric};
use std::arch::x86_64::*;

/// AVX2 L2 距离（平方欧氏距离）
///
/// 使用 8-wide SIMD + FMA：
/// - _mm256_loadu_ps: 加载 8 个 f32（unaligned）
/// - _mm256_sub_ps: 向量减法
/// - _mm256_fmadd_ps: d*d + sum（融合乘加，单指令完成）
///
/// 每 cycle 处理 8 个 f32，理论峰值是标量的 8 倍
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    // 主循环：每次处理 8 个 f32
    let mut sum = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let d = _mm256_sub_ps(va, vb);
        // FMA: sum = d*d + sum
        sum = _mm256_fmadd_ps(d, d, sum);
    }

    // 水平求和：将 8 个 lane 相加
    let mut result = horizontal_sum_ps(sum);

    // 尾部标量处理
    for i in 0..remainder {
        let idx = chunks * 8 + i;
        let d = a[idx] - b[idx];
        result += d * d;
    }

    result
}

/// AVX2 内积距离（1 - inner product，MIPS 场景预留）
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn ip_avx2(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        // FMA: sum = a*b + sum
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = horizontal_sum_ps(sum);

    for i in 0..remainder {
        let idx = chunks * 8 + i;
        result += a[idx] * b[idx];
    }

    -result
}

/// __m256 水平求和
#[inline(always)]
unsafe fn horizontal_sum_ps(v: __m256) -> f32 {
    // 提取高低 128 bit
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    // 128 bit 内水平加
    let sum128 = _mm_add_ps(hi, lo);
    // 再水平加
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum128, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

/// AVX2 距离核实现
pub struct Avx2Kernel {
    /// 度量类型
    pub metric: DistanceMetric,
}

impl Avx2Kernel {
    /// 创建 L2 AVX2 核
    pub fn l2() -> Self {
        Self { metric: DistanceMetric::L2 }
    }

    /// 创建内积 AVX2 核
    #[allow(dead_code)]
    pub fn ip() -> Self {
        Self { metric: DistanceMetric::InnerProduct }
    }
}

impl DistanceKernel for Avx2Kernel {
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> Distance {
        unsafe {
            match self.metric {
                DistanceMetric::L2 => l2_avx2(a, b),
                DistanceMetric::InnerProduct => ip_avx2(a, b),
            }
        }
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn name(&self) -> &'static str {
        match self.metric {
            DistanceMetric::L2 => "avx2_l2",
            DistanceMetric::InnerProduct => "avx2_ip",
        }
    }
}

/// 检测当前 CPU 是否支持 AVX2 + FMA
pub fn is_avx2_supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// 运行时分发的 L2 距离
///
/// 若 CPU 支持 AVX2+FMA 则走 AVX2 路径，否则回退到动态兜底
#[inline(always)]
pub fn l2_dispatch(a: &[f32], b: &[f32]) -> Distance {
    if is_avx2_supported() {
        unsafe { l2_avx2(a, b) }
    } else {
        super::dynamic::l2_dynamic(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_matches_scalar() {
        if !is_avx2_supported() {
            eprintln!("AVX2 not supported, skipping test");
            return;
        }
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32) * 2.0).collect();
        let d_scalar: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d_avx2 = unsafe { l2_avx2(&a, &b) };
        assert!((d_scalar - d_avx2).abs() < 1e-2, "scalar={} avx2={}", d_scalar, d_avx2);
    }

    #[test]
    fn avx2_small_vectors() {
        if !is_avx2_supported() {
            return;
        }
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 5.0, 6.0, 7.0];
        let expected = 27.0f32; // 9+9+9+9... wait: (1-4)^2+(2-5)^2+(3-6)^2+(4-7)^2 = 9+9+9+9 = 36
        let expected = 36.0f32;
        let result = unsafe { l2_avx2(&a, &b) };
        assert!((result - expected).abs() < 1e-5, "got {} expected {}", result, expected);
    }

    #[test]
    fn avx2_aligned_8() {
        if !is_avx2_supported() {
            return;
        }
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        // 每个分量差 1，平方 1，共 16 个 = 16
        let result = unsafe { l2_avx2(&a, &b) };
        assert!((result - 16.0).abs() < 1e-5);
    }

    #[test]
    fn avx2_zero_distance() {
        if !is_avx2_supported() {
            return;
        }
        let a = [1.0f32; 100];
        let result = unsafe { l2_avx2(&a, &a) };
        assert!(result.abs() < 1e-4);
    }

    #[test]
    fn dispatch_uses_avx2_when_available() {
        let a = [1.0f32; 100];
        let b = [2.0f32; 100];
        let result = l2_dispatch(&a, &b);
        assert!((result - 100.0).abs() < 1e-4);
    }

    #[test]
    fn avx2_kernel_trait() {
        if !is_avx2_supported() {
            return;
        }
        let kernel = Avx2Kernel::l2();
        let a = [1.0f32; 100];
        let b = [2.0f32; 100];
        let d = kernel.distance(&a, &b);
        assert!((d - 100.0).abs() < 1e-4);
        assert_eq!(kernel.name(), "avx2_l2");
    }
}
