//! H42：64 字节对齐的扁平数组（Deref 为 `[T]`）。
//!
//! 大块 `Vec` 由 mmap 分配，起点通常为 16 mod 64，使 64B SQ4 码、512B 向量、192B 邻接行
//! 都多跨一条 cache line（而软件预取按行数发，尾行变成按需缺失）。H42 测得 SIFT1M 对齐后 QPS +15~27%。

use std::alloc::{alloc_zeroed, dealloc, Layout};

/// H47：≥ 4 MB 的块按 2 MB 对齐分配并 madvise(MADV_HUGEPAGE)（THP=madvise 的系统上默认全是 4 KB 页，
/// 1M 级随机访问几乎每次都是 TLB 缺失）。开关：默认开（2026-10-30 第 8 轮合入：SIFT1M +16.2% 9/9、GloVe +12.2% 9/9），RAVEN_THP=0 关闭，或 `from_slice_opts(.., huge)`。
const HUGE_MIN: usize = 4 << 20;
const HUGE_ALIGN: usize = 2 << 20;

pub fn thp_default() -> bool {
    std::env::var("RAVEN_THP").map(|v| v != "0").unwrap_or(true)
}

#[cfg(target_os = "linux")]
fn advise_huge(p: *mut u8, len: usize) {
    extern "C" { fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32; }
    const MADV_HUGEPAGE: i32 = 14;
    unsafe { madvise(p as *mut core::ffi::c_void, len, MADV_HUGEPAGE); }
}
#[cfg(not(target_os = "linux"))]
fn advise_huge(_p: *mut u8, _len: usize) {}

/// 64B 对齐的 `[T]` 容器（`T: Copy`，按字节拷贝）。`off` 仅供 A/B 实验人为错位。
pub struct AlignedBuf<T: Copy> {
    ptr: *mut u8,
    layout: Layout,
    off: usize,
    len: usize,
    huge: bool,
    _t: std::marker::PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for AlignedBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedBuf<T> {}

impl<T: Copy> AlignedBuf<T> {
    /// 拷贝 `src`，起点相对 64B 边界偏移 `off_bytes`（0 = 对齐）；`huge` = 大块走 2MB 对齐 + THP 建议。
    pub fn from_slice_opts(src: &[T], off_bytes: usize, huge: bool) -> Self {
        let sz = std::mem::size_of::<T>();
        let al = std::mem::align_of::<T>();
        let off = (off_bytes % 64) / al * al;
        let bytes = src.len() * sz;
        let total = (bytes + off).div_ceil(64).max(1) * 64;
        let huge = huge && total >= HUGE_MIN;
        let (align, size) = if huge { (HUGE_ALIGN, total.div_ceil(HUGE_ALIGN) * HUGE_ALIGN) } else { (64, total) };
        let layout = Layout::from_size_align(size, align).expect("layout");
        let ptr = if huge {
            // 先 madvise 再触页，使缺页直接分配 2MB 页
            let p = unsafe { std::alloc::alloc(layout) };
            assert!(!p.is_null(), "alloc failed");
            advise_huge(p, size);
            unsafe { std::ptr::write_bytes(p, 0, size); }
            p
        } else {
            let p = unsafe { alloc_zeroed(layout) };
            assert!(!p.is_null(), "alloc failed");
            p
        };
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, ptr.add(off), bytes); }
        AlignedBuf { ptr, layout, off, len: src.len(), huge, _t: std::marker::PhantomData }
    }
    pub fn from_slice_with_offset(src: &[T], off_bytes: usize) -> Self { Self::from_slice_opts(src, off_bytes, thp_default()) }
    pub fn from_slice(src: &[T]) -> Self { Self::from_slice_with_offset(src, 0) }
    pub fn is_huge(&self) -> bool { self.huge }
    /// 零初始化，`huge_min` 字节以上才走大页
    pub fn zeroed_opts(len: usize, huge: bool, huge_min: usize) -> Self {
        let sz = std::mem::size_of::<T>().max(1);
        let total = (len * sz).div_ceil(64).max(1) * 64;
        let huge = huge && total >= huge_min;
        let (align, size) = if huge { (HUGE_ALIGN, total.div_ceil(HUGE_ALIGN) * HUGE_ALIGN) } else { (64, total) };
        let layout = Layout::from_size_align(size, align).expect("layout");
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "alloc failed");
        if huge { advise_huge(ptr, size); }
        unsafe { std::ptr::write_bytes(ptr, 0, size); }
        AlignedBuf { ptr, layout, off: 0, len, huge, _t: std::marker::PhantomData }
    }
    /// 逻辑长度截断（不重新分配）
    pub fn truncated(mut self, len: usize) -> Self { self.len = self.len.min(len); self }
}

impl<T: Copy> Drop for AlignedBuf<T> {
    fn drop(&mut self) { unsafe { dealloc(self.ptr, self.layout) } }
}

impl<T: Copy> From<Vec<T>> for AlignedBuf<T> {
    fn from(v: Vec<T>) -> Self { Self::from_slice(&v) }
}

impl<T: Copy> std::ops::Deref for AlignedBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.add(self.off) as *const T, self.len) }
    }
}

impl<T: Copy> std::ops::DerefMut for AlignedBuf<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(self.off) as *mut T, self.len) }
    }
}

impl<T: Copy> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self { Self::from_slice_opts(self, self.off, self.huge) }
}

impl<T: Copy> std::fmt::Debug for AlignedBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuf(len={}, off={})", self.len, self.off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aligned_roundtrip() {
        let v: Vec<u32> = (0..1000).collect();
        let a = AlignedBuf::from_slice(&v);
        assert_eq!(a.as_ptr() as usize % 64, 0);
        assert_eq!(&a[..], &v[..]);
        let b = AlignedBuf::from_slice_with_offset(&v, 16);
        assert_eq!(b.as_ptr() as usize % 64, 16);
        assert_eq!(&b[..], &v[..]);
        let c = b.clone();
        assert_eq!(&c[..], &v[..]);
        let big: Vec<u8> = (0..(5usize << 20)).map(|i| i as u8).collect();
        let h = AlignedBuf::from_slice_opts(&big, 0, true);
        assert!(h.is_huge());
        assert_eq!(h.as_ptr() as usize % (2 << 20), 0);
        assert_eq!(&h[..], &big[..]);
        let h2 = AlignedBuf::from_slice_opts(&big, 16, false);
        assert_eq!(&h2[..], &big[..]);
    }
}
