//! 动态兜底路径
//!
//! 设计文档第一层：
//! chunks_exact、固定步长循环、尾部单独处理，有助于 LLVM 识别规则循环
//! 并做自动向量化；是否真正生成 FMA 或消除边界检查，以 cargo asm / objdump
//! 和基准结果为准。

/// 动态维度 L2 距离（平方欧氏距离）
///
/// 设计文档原文实现：
/// - 使用 chunks_exact(8) 让 LLVM 识别规则循环并做自动向量化
/// - 尾部 remainder 单独处理
/// - 不开根号以避免热路径 sqrt 开销
#[inline(always)]
pub fn l2_dynamic(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let (rem_a, rem_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..8 {
            let d = ca[i] - cb[i];
            sum += d * d;
        }
    }
    for (x, y) in rem_a.iter().zip(rem_b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// 动态维度内积距离（1 - inner product，MIPS 场景预留）
#[allow(dead_code)]
#[inline(always)]
pub fn ip_dynamic(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let (rem_a, rem_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..8 {
            sum += ca[i] * cb[i];
        }
    }
    for (x, y) in rem_a.iter().zip(rem_b) {
        sum += x * y;
    }
    -sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_dynamic_matches_scalar() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32) * 2.0).collect();
        let d_scalar: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d_dynamic = l2_dynamic(&a, &b);
        assert!((d_scalar - d_dynamic).abs() < 1e-3);
    }

    #[test]
    fn l2_dynamic_small() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((l2_dynamic(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn l2_dynamic_aligned_8() {
        // 维度恰好是 8 的倍数
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        // 每个分量差 1，平方 1，共 16 个 = 16
        assert!((l2_dynamic(&a, &b) - 16.0).abs() < 1e-6);
    }

    #[test]
    fn l2_dynamic_zero() {
        let a = [1.0f32; 100];
        assert!(l2_dynamic(&a, &a).abs() < 1e-6);
    }
}
