//! 诊断计数器（feature `diag_counters`）
//!
//! 未启用 feature 时 `diag_count!` 展开为空，QPS 构建零开销。
//! 启用时使用 Relaxed 原子计数（单线程评测下等价于普通加法）。
//! 计数点：
//! - `FULL_DIST`：f32 全精度距离（`l2_simd`：导航层 + rerank + 入口）
//! - `QUANT_DIST`：量化距离（`l2_sq4_raw` / `l2_sq8` / `l2_sq8_raw`）
//! - `HOPS`：底层图搜索扩展（pop）的节点数（`greedy_search_sq4`）

use std::sync::atomic::{AtomicU64, Ordering};

/// f32 全精度距离计算次数
pub static FULL_DIST: AtomicU64 = AtomicU64::new(0);
/// 量化距离计算次数
pub static QUANT_DIST: AtomicU64 = AtomicU64::new(0);
/// 上层导航（LayeredNavigation::descend_with）距离计算次数（f32 或 SQ4，同时计入 FULL/QUANT）
pub static NAV_DIST: AtomicU64 = AtomicU64::new(0);
/// H07：提前放弃（early abandon）的 SQ4 距离次数（不计入 QUANT_DIST）
pub static QUANT_PARTIAL: AtomicU64 = AtomicU64::new(0);
/// H07：提前放弃的 SQ4 距离实际处理的维度总数（NDC 折算 = 维度数 / dim）
pub static QUANT_PARTIAL_DIMS: AtomicU64 = AtomicU64::new(0);
/// 底层扩展节点数
pub static HOPS: AtomicU64 = AtomicU64::new(0);

/// 计数器快照 (full, quant, hops)
pub fn snapshot() -> (u64, u64, u64) {
    (
        FULL_DIST.load(Ordering::Relaxed),
        QUANT_DIST.load(Ordering::Relaxed),
        HOPS.load(Ordering::Relaxed),
    )
}

/// 清零全部计数器
pub fn reset() {
    FULL_DIST.store(0, Ordering::Relaxed);
    QUANT_DIST.store(0, Ordering::Relaxed);
    HOPS.store(0, Ordering::Relaxed);
    NAV_DIST.store(0, Ordering::Relaxed);
    QUANT_PARTIAL.store(0, Ordering::Relaxed);
    QUANT_PARTIAL_DIMS.store(0, Ordering::Relaxed);
}

/// H07：提前放弃快照 (次数, 已处理维度总数)
pub fn partial_snapshot() -> (u64, u64) {
    (QUANT_PARTIAL.load(Ordering::Relaxed), QUANT_PARTIAL_DIMS.load(Ordering::Relaxed))
}

/// 按数量累加（H07 部分距离维度）
#[macro_export]
macro_rules! diag_count_add {
    ($c:ident, $n:expr) => {
        #[cfg(feature = "diag_counters")]
        {
            $crate::counters::$c.fetch_add(($n) as u64, ::std::sync::atomic::Ordering::Relaxed);
        }
    };
}

/// 导航层距离计数快照
pub fn nav_snapshot() -> u64 {
    NAV_DIST.load(Ordering::Relaxed)
}

/// 计数器是否编译启用
pub const ENABLED: bool = cfg!(feature = "diag_counters");

/// 计数宏：未启用 `diag_counters` 时为空操作
#[macro_export]
macro_rules! diag_count {
    ($c:ident) => {
        #[cfg(feature = "diag_counters")]
        {
            $crate::counters::$c.fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
        }
    };
}

/// 分段计时（feature diag_prof）：0 nav，1 L0 总，2 rerank，3 hop-pop+预取，4 visited 过滤，5 距离+插入
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static ACC: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    #[inline(always)]
    pub fn tick() -> u64 {
        #[cfg(all(feature = "diag_prof", target_arch = "aarch64"))]
        { let v: u64; unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) v); } v }
        #[cfg(all(feature = "diag_prof", target_arch = "x86_64"))]
        { unsafe { core::arch::x86_64::_rdtsc() } }
        #[cfg(not(all(feature = "diag_prof", any(target_arch = "aarch64", target_arch = "x86_64"))))]
        { 0 }
    }
    #[inline(always)]
    pub fn add(i: usize, d: u64) { if cfg!(feature = "diag_prof") { ACC[i].fetch_add(d, Relaxed); } }
    pub fn snapshot() -> [u64; 8] { let mut o = [0; 8]; for i in 0..8 { o[i] = ACC[i].load(Relaxed); } o }
    pub fn reset() { for a in ACC.iter() { a.store(0, Relaxed); } }
}
