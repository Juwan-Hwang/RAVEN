//! NEON 距离核（H01：aarch64 移植）
//!
//! f32 L2：4 路独立 float32x4 累加器（16 f32 / 迭代），隐藏 FMA 延迟。
//! SIFT-128 = 8 次迭代。尾部标量。
//!
//! 仅在 `target_arch = "aarch64"` 编译；NEON 是 AArch64 基线特性，无需运行时检测。

use super::{Distance, DistanceKernel, DistanceMetric};
use std::arch::aarch64::*;

/// NEON L2 距离（平方欧氏距离）
///
/// # Safety
/// `a.len() == b.len()`（debug 断言）。NEON 为 AArch64 基线，始终可用。
#[inline(always)]
pub unsafe fn l2_neon(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);
    let chunks = n / 16;
    for c in 0..chunks {
        let o = c * 16;
        let d0 = vsubq_f32(vld1q_f32(pa.add(o)), vld1q_f32(pb.add(o)));
        let d1 = vsubq_f32(vld1q_f32(pa.add(o + 4)), vld1q_f32(pb.add(o + 4)));
        let d2 = vsubq_f32(vld1q_f32(pa.add(o + 8)), vld1q_f32(pb.add(o + 8)));
        let d3 = vsubq_f32(vld1q_f32(pa.add(o + 12)), vld1q_f32(pb.add(o + 12)));
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }
    let mut o = chunks * 16;
    while o + 4 <= n {
        let d = vsubq_f32(vld1q_f32(pa.add(o)), vld1q_f32(pb.add(o)));
        s0 = vfmaq_f32(s0, d, d);
        o += 4;
    }
    let mut result = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    while o < n {
        let d = *pa.add(o) - *pb.add(o);
        result = d.mul_add(d, result);
        o += 1;
    }
    result
}

/// NEON 内积距离（负内积，MIPS 预留）
///
/// # Safety
/// 同 [`l2_neon`]。
#[inline(always)]
pub unsafe fn ip_neon(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s0 = vdupq_n_f32(0.0);
    let mut o = 0;
    while o + 4 <= n {
        s0 = vfmaq_f32(s0, vld1q_f32(a.as_ptr().add(o)), vld1q_f32(b.as_ptr().add(o)));
        o += 4;
    }
    let mut result = vaddvq_f32(s0);
    while o < n {
        result = a[o].mul_add(b[o], result);
        o += 1;
    }
    -result
}

/// NEON 距离核实现
pub struct NeonKernel {
    /// 度量类型
    pub metric: DistanceMetric,
}

impl DistanceKernel for NeonKernel {
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> Distance {
        unsafe {
            match self.metric {
                DistanceMetric::L2 => l2_neon(a, b),
                DistanceMetric::InnerProduct => ip_neon(a, b),
            }
        }
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn name(&self) -> &'static str {
        match self.metric {
            DistanceMetric::L2 => "neon_l2",
            DistanceMetric::InnerProduct => "neon_ip",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::scalar::l2_scalar;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 200.0 - 100.0
            })
            .collect()
    }

    #[test]
    fn neon_l2_matches_scalar_various_dims() {
        for dim in [1usize, 3, 4, 7, 15, 16, 17, 31, 64, 100, 128, 129, 256, 960] {
            let a = rnd(dim, dim as u64);
            let b = rnd(dim, dim as u64 + 99);
            let s = l2_scalar(&a, &b);
            let v = unsafe { l2_neon(&a, &b) };
            assert!((s - v).abs() <= 1e-4 * s.abs().max(1.0), "dim={dim} scalar={s} neon={v}");
        }
    }

    #[test]
    fn neon_l2_self_is_zero() {
        let a = rnd(128, 7);
        assert_eq!(unsafe { l2_neon(&a, &a) }, 0.0);
    }

    #[test]
    fn neon_ip_matches_scalar() {
        let a = rnd(130, 1);
        let b = rnd(130, 2);
        let s: f32 = -a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>();
        let v = unsafe { ip_neon(&a, &b) };
        assert!((s - v).abs() <= 1e-3 * s.abs().max(1.0));
    }
}
