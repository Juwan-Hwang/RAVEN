//! 平台抽象层：跨架构的 prefetch shim（H01：aarch64 移植）
//!
//! - x86_64：与移植前完全一致，直接发 `_mm_prefetch`
//!   （`prefetch_l1` 保留原代码的 hint=0，即 `_MM_HINT_NTA`；`prefetch_t0` 为 hint=3 `_MM_HINT_T0`）
//! - aarch64：`prfm pldl1keep`
//! - 其他架构：空操作

/// 读预取到 L1（热路径专用）。x86 上等价于原来的 `_mm_prefetch::<0>`。
///
/// # Safety
/// prefetch 只是 hint，不会解引用；仍标为 unsafe 以与原 intrinsic 调用点保持一致。
#[inline(always)]
pub unsafe fn prefetch_l1(ptr: *const i8) {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_mm_prefetch::<0>(ptr);
    }
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// R12：读预取到 L2（aarch64 `prfm pldl2keep`，x86 `_MM_HINT_T1`）。
///
/// # Safety
/// 同 [`prefetch_l1`]。
#[inline(always)]
pub unsafe fn prefetch_l2(ptr: *const i8) {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_mm_prefetch::<2>(ptr);
    }
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("prfm pldl2keep, [{0}]", in(reg) ptr, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// 读预取（x86 hint=3 `_MM_HINT_T0`）。aarch64 上同 `prefetch_l1`。
///
/// # Safety
/// 同 [`prefetch_l1`]。
#[inline(always)]
pub unsafe fn prefetch_t0(ptr: *const i8) {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_mm_prefetch::<3>(ptr);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        prefetch_l1(ptr);
    }
}
