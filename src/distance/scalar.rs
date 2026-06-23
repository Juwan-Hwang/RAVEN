//! f32 标量距离核（Week 1-2 主线）
//!
//! 设计文档第一层：
//! - f32 用于构建真值 + 图决策，全程不降精度
//! - 标量实现作为基线，后续 AVX2/AVX-512/NEON 在此基础上扩展

use super::{Distance, DistanceKernel, DistanceMetric};

/// f32 标量 L2 距离（平方欧氏距离，不开根号以避免 sqrt 开销）
///
/// 图检索中比较距离大小时，平方距离与真实距离排序一致，
/// 省去 sqrt 可显著降低热路径开销。
#[inline(always)]
pub fn l2_scalar(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// f32 标量内积距离（1 - inner product，用于 MIPS 场景，当前预留）
#[allow(dead_code)]
#[inline(always)]
pub fn ip_scalar(a: &[f32], b: &[f32]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    -sum // 返回负内积，使"更小=更相似"语义统一
}

/// 标量距离核实现
pub struct ScalarKernel {
    /// 度量类型
    pub metric: DistanceMetric,
}

impl ScalarKernel {
    /// 创建 L2 标量核
    pub fn l2() -> Self {
        Self { metric: DistanceMetric::L2 }
    }

    /// 创建内积标量核
    #[allow(dead_code)]
    pub fn ip() -> Self {
        Self { metric: DistanceMetric::InnerProduct }
    }
}

impl DistanceKernel for ScalarKernel {
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> Distance {
        match self.metric {
            DistanceMetric::L2 => l2_scalar(a, b),
            DistanceMetric::InnerProduct => ip_scalar(a, b),
        }
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn name(&self) -> &'static str {
        match self.metric {
            DistanceMetric::L2 => "scalar_l2",
            DistanceMetric::InnerProduct => "scalar_ip",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_scalar_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert!((l2_scalar(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn l2_scalar_zero() {
        let a = [1.0f32, 2.0, 3.0];
        assert!(l2_scalar(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn l2_scalar_ordering() {
        let q = [0.0f32, 0.0];
        let near = [1.0f32, 0.0];
        let far = [3.0f32, 4.0];
        assert!(l2_scalar(&q, &near) < l2_scalar(&q, &far));
    }
}
