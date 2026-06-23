//! 内存对齐策略
//!
//! 设计文档第一层/第二层：
//! GEMM 缓冲区和向量主存储保证 32/64 字节对齐有助于 SIMD 访问效率；
//! aligned vs unaligned load 哪个更快，以具体 CPU 和基准结果为准。

pub use aligned_vec::{AVec, ConstAlign};

/// 32 字节对齐的 f32 向量（AVX2 友好）
///
/// 设计文档：保证 32/64 字节对齐有助于 SIMD 访问效率
pub type AlignedVec<T> = AVec<T, ConstAlign<32>>;

/// 64 字节对齐的 f32 向量（AVX-512 友好，Week 7-8）
#[allow(dead_code)]
pub type AlignedVec64<T> = AVec<T, ConstAlign<64>>;

/// 创建 32 字节对齐的 f32 向量
pub fn aligned_vec_f32(len: usize) -> AlignedVec<f32> {
    // aligned-vec 0.5 API: AVec::new(align) 返回空 Vec，再 push
    // 使用 from_slice 从零初始化切片创建
    AVec::from_slice(32, &vec![0.0f32; len])
}

/// 从切片创建 32 字节对齐的 f32 向量
pub fn aligned_from_slice(slice: &[f32]) -> AlignedVec<f32> {
    AVec::from_slice(32, slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_vec_32_alignment() {
        let v = aligned_vec_f32(64);
        let ptr = v.as_ptr();
        // 32 字节对齐：指针低 5 位应为 0
        assert_eq!(ptr as usize & 0x1F, 0);
    }

    #[test]
    fn aligned_from_slice_copies() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let v = aligned_from_slice(&src);
        assert_eq!(&v[..], &src);
    }
}
