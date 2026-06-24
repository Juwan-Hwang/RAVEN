//! 内核选择策略（三阶段筛选）
//!
//! 设计文档第一层：
//! 三阶段筛选，避免 thermal throttling 误判（AVX-512 在部分 Intel 平台上混合使用时
//! 可能因降频导致 wall-clock 时间不如 AVX2）：
//! 1. 第一阶段：延迟粗筛
//! 2. 第二阶段：瞬时 QPS 筛选
//! 3. 第三阶段：持续稳定性验证（结果缓存，首次约 5 分钟）
//!
//! 支持配置入口：缓存最优配置（首次选定后落盘）、强制指定内核（环境变量或配置文件覆盖）

use std::time::{Duration, Instant};
use super::{DistanceKernel, DistanceMetric};

/// 延迟粗筛阈值（纳秒），超过此阈值的内核被淘汰
/// 设计文档第一阶段：延迟粗筛
pub const LATENCY_THRESHOLD_NS: u64 = 10_000;

/// 稳定性验证持续时间（设计文档 F.10：连续运行 5 分钟）
/// 注：实际验证时可通过配置缩短用于测试
const STABILITY_TEST_DURATION: Duration = Duration::from_secs(300);

/// 稳定性通过标准：steady-state QPS >= 瞬时 QPS × 0.95（设计文档 F.10）
const STABILITY_THRESHOLD: f64 = 0.95;

/// 内核变体
///
/// 设计文档第一层 KernelVariant。当前 Week 1-2 仅标量核，
/// Week 3-4 接入 AVX2，Week 7-8 接入 AVX-512/NEON。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// f32 标量核（Week 1-2 主线）
    Scalar,
    /// AVX2 核（Week 3-4）
    Avx2,
    /// AVX-512 核（Week 7-8）
    Avx512,
    /// NEON 核（Week 7-8，ARM 平台）
    Neon,
}

impl KernelVariant {
    /// 内核名称
    pub fn name(self) -> &'static str {
        match self {
            KernelVariant::Scalar => "scalar",
            KernelVariant::Avx2 => "avx2",
            KernelVariant::Avx512 => "avx512",
            KernelVariant::Neon => "neon",
        }
    }

    /// 构造对应的距离核
    pub fn build_kernel(self, metric: DistanceMetric) -> Box<dyn DistanceKernel> {
        match self {
            KernelVariant::Scalar => Box::new(super::scalar::ScalarKernel { metric }),
            KernelVariant::Avx2 => {
                // 运行时检测：若 CPU 不支持 AVX2，回退到标量
                if super::avx2::is_avx2_supported() {
                    Box::new(super::avx2::Avx2Kernel { metric })
                } else {
                    tracing::warn!("AVX2 requested but not supported, falling back to scalar");
                    Box::new(super::scalar::ScalarKernel { metric })
                }
            }
            KernelVariant::Avx512 => {
                // Week 7-8：AVX-512 接入
                if super::avx512::is_avx512_supported() {
                    Box::new(super::avx512::Avx512Kernel { metric })
                } else {
                    tracing::warn!("AVX-512 requested but not supported, falling back to scalar");
                    Box::new(super::scalar::ScalarKernel { metric })
                }
            }
            KernelVariant::Neon => {
                // ARM 平台，当前 x86_64 不支持
                tracing::warn!("NEON requested but not supported on x86_64, falling back to scalar");
                Box::new(super::scalar::ScalarKernel { metric })
            }
        }
    }
}

/// 获取当前平台可用的内核列表
///
/// 设计文档第一阶段：available_kernels()
/// Week 3-4：AVX2 在支持时加入候选
/// Week 7-8：AVX-512 在支持时加入候选
pub fn available_kernels() -> Vec<KernelVariant> {
    let mut kernels = vec![KernelVariant::Scalar];
    // Week 3-4：AVX2 在 CPU 支持时加入候选
    if super::avx2::is_avx2_supported() {
        kernels.push(KernelVariant::Avx2);
    }
    // Week 7-8：AVX-512 在 CPU 支持时加入候选
    if super::avx512::is_avx512_supported() {
        kernels.push(KernelVariant::Avx512);
    }
    kernels
}

/// 测量内核延迟（纳秒）
///
/// 设计文档第一阶段：measure_latency_ns
/// 对单个距离计算调用测量纳秒级延迟
pub fn measure_latency_ns(variant: KernelVariant, dim: usize) -> u64 {
    let kernel = variant.build_kernel(DistanceMetric::L2);
    let a = vec![1.0f32; dim];
    let b = vec![2.0f32; dim];

    // 预热
    for _ in 0..16 {
        let _ = kernel.distance(&a, &b);
    }

    // 测量
    let iterations = 1024;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = kernel.distance(&a, &b);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as u64 / iterations
}

/// 测量内核瞬时 QPS
///
/// 设计文档第二阶段：measure_qps
pub fn measure_qps(variant: KernelVariant, dim: usize) -> u64 {
    let kernel = variant.build_kernel(DistanceMetric::L2);
    let a = vec![1.0f32; dim];
    let b = vec![2.0f32; dim];

    // 预热
    for _ in 0..256 {
        let _ = kernel.distance(&a, &b);
    }

    let duration = Duration::from_millis(100);
    let start = Instant::now();
    let mut count = 0u64;
    while start.elapsed() < duration {
        for _ in 0..1024 {
            let _ = kernel.distance(&a, &b);
        }
        count += 1024;
    }
    let elapsed_secs = start.elapsed().as_secs_f64();
    (count as f64 / elapsed_secs) as u64
}

/// 持续稳定性验证
///
/// 设计文档 F.10：
/// - 测试方式：连续运行 5 分钟，每 30 秒采样一次 QPS
/// - 通过标准：steady-state QPS >= 瞬时 QPS × 0.95
/// - 失败处理：降级到下一候选，记录降级原因和平台信息
///
/// 注：完整 5 分钟测试仅在首次内核选择时执行，结果缓存后不重复。
/// 此处提供可配置时长以便测试。
pub fn validate_kernel_stability(
    variant: KernelVariant,
    dim: usize,
    test_duration: Duration,
) -> bool {
    let baseline = measure_qps(variant, dim);
    let sustained = measure_qps_sustained(variant, dim, test_duration);
    sustained as f64 >= baseline as f64 * STABILITY_THRESHOLD
}

/// 测量持续 QPS（设计文档 F.10：measure_qps_sustained）
pub fn measure_qps_sustained(variant: KernelVariant, dim: usize, duration: Duration) -> u64 {
    let kernel = variant.build_kernel(DistanceMetric::L2);
    let a = vec![1.0f32; dim];
    let b = vec![2.0f32; dim];

    // 预热
    for _ in 0..256 {
        let _ = kernel.distance(&a, &b);
    }

    let start = Instant::now();
    let mut count = 0u64;
    while start.elapsed() < duration {
        for _ in 0..1024 {
            let _ = kernel.distance(&a, &b);
        }
        count += 1024;
    }
    let elapsed_secs = start.elapsed().as_secs_f64();
    (count as f64 / elapsed_secs) as u64
}

/// 三阶段内核选择
///
/// 设计文档第一层 select_kernel：
/// 1. 第一阶段：延迟粗筛 - 淘汰延迟超过阈值的内核
/// 2. 第二阶段：瞬时 QPS 筛选 - 选 QPS 最高的
/// 3. 第三阶段：持续稳定性验证 - 5 分钟持续测试
///
/// 优先级（从高到低）：
/// - 环境变量 RAVEN_KERNEL 强制覆盖（设计文档：强制指定内核）
/// - KernelCache 缓存命中（设计文档：缓存最优配置）
/// - 三阶段筛选（首次执行，结果落盘）
///
/// 失败时降级到 fallback_kernel 并记录原因
pub fn select_kernel(dim: usize) -> KernelVariant {
    select_kernel_with_duration(dim, STABILITY_TEST_DURATION)
}

/// 带可配置稳定性测试时长的内核选择（用于测试）
pub fn select_kernel_with_duration(dim: usize, stability_duration: Duration) -> KernelVariant {
    // 优先级 0：环境变量强制覆盖（设计文档：强制指定内核）
    if let Some(forced) = env_forced_kernel() {
        tracing::info!(
            kernel = forced.name(),
            dim,
            "kernel forced by RAVEN_KERNEL env var"
        );
        return forced;
    }

    // 优先级 1：缓存命中（设计文档：缓存最优配置，跳过微基准）
    if let Some(cached) = load_cached_kernel(dim) {
        tracing::info!(
            kernel = cached.name(),
            dim,
            "kernel selected from cache"
        );
        return cached;
    }

    // 优先级 2：三阶段筛选
    let selected = run_three_stage_selection(dim, stability_duration);

    // 落盘缓存
    save_cached_kernel(dim, selected);

    selected
}

/// 环境变量强制内核（RAVEN_KERNEL=scalar|avx2|avx512|neon）
///
/// 设计文档：强制指定内核（环境变量或配置文件覆盖）
fn env_forced_kernel() -> Option<KernelVariant> {
    let val = std::env::var("RAVEN_KERNEL").ok()?;
    match val.trim().to_ascii_lowercase().as_str() {
        "scalar" => Some(KernelVariant::Scalar),
        "avx2" => Some(KernelVariant::Avx2),
        "avx512" => Some(KernelVariant::Avx512),
        "neon" => Some(KernelVariant::Neon),
        _ => {
            tracing::warn!(value = %val, "invalid RAVEN_KERNEL value, ignored");
            None
        }
    }
}

/// 默认缓存路径（~/.cache/raven/kernel_cache.toml）
fn default_cache_path() -> std::path::PathBuf {
    let mut p = std::env::var("RAVEN_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_fallback_cache_dir()
        });
    p.push("kernel_cache.toml");
    p
}

/// 不依赖外部 crate 的 cache 目录兜底
fn dirs_fallback_cache_dir() -> std::path::PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache");
        p.push("raven");
        std::fs::create_dir_all(&p).ok();
        p
    } else {
        std::path::PathBuf::from(".")
    }
}

/// 从缓存加载内核选择
fn load_cached_kernel(dim: usize) -> Option<KernelVariant> {
    let path = default_cache_path();
    let cache = KernelCache::load(&path).ok()?;
    cache.get(dim).filter(|k| {
        // 缓存命中后仍需校验 CPU 支持
        let avail = available_kernels();
        avail.contains(k)
    })
}

/// 保存内核选择到缓存
fn save_cached_kernel(dim: usize, variant: KernelVariant) {
    let path = default_cache_path();
    let mut cache = KernelCache::load(&path).unwrap_or_default();
    cache.set(dim, variant);
    if let Err(e) = cache.save(&path) {
        tracing::warn!(path = %path.display(), error = %e, "failed to save kernel cache");
    }
}

/// 三阶段筛选核心逻辑（不含缓存与环境变量覆盖）
fn run_three_stage_selection(dim: usize, stability_duration: Duration) -> KernelVariant {
    // 第一阶段：延迟粗筛
    let candidates: Vec<_> = available_kernels()
        .into_iter()
        .filter(|k| measure_latency_ns(*k, dim) < LATENCY_THRESHOLD_NS)
        .collect();

    if candidates.is_empty() {
        return fallback_kernel(dim);
    }

    // 第二阶段：瞬时 QPS 筛选
    let finalist = candidates
        .into_iter()
        .max_by_key(|k| measure_qps(*k, dim))
        .unwrap_or(fallback_kernel(dim));

    // 第三阶段：持续稳定性验证
    if !validate_kernel_stability(finalist, dim, stability_duration) {
        tracing::warn!(
            kernel = finalist.name(),
            dim,
            "kernel stability validation failed, falling back"
        );
        return fallback_kernel(dim);
    }

    tracing::info!(
        kernel = finalist.name(),
        dim,
        "selected kernel after three-stage validation"
    );
    finalist
}

/// 降级内核（设计文档：fallback_kernel）
pub fn fallback_kernel(_dim: usize) -> KernelVariant {
    KernelVariant::Scalar
}

/// 内核选择缓存配置
///
/// 设计文档：缓存最优配置（首次选定后落盘，跳过微基准）
/// 支持强制指定内核（环境变量或配置文件覆盖）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelCache {
    /// 维度到内核的映射
    pub entries: Vec<KernelCacheEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// 单条内核缓存条目
pub struct KernelCacheEntry {
    /// 维度
    pub dim: usize,
    /// 内核名称
    pub kernel: String,
    /// 选择时的时间戳
    pub selected_at: u64,
}

impl KernelCache {
    /// 从文件加载内核选择缓存
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cache: Self = toml::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(cache)
    }

    /// 保存内核选择缓存到文件
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let data = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, data)
    }

    /// 查询某维度已缓存的内核
    pub fn get(&self, dim: usize) -> Option<KernelVariant> {
        self.entries.iter().find(|e| e.dim == dim).and_then(|e| match e.kernel.as_str() {
            "scalar" => Some(KernelVariant::Scalar),
            "avx2" => Some(KernelVariant::Avx2),
            "avx512" => Some(KernelVariant::Avx512),
            "neon" => Some(KernelVariant::Neon),
            _ => None,
        })
    }

    /// 记录某维度的内核选择
    pub fn set(&mut self, dim: usize, variant: KernelVariant) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(entry) = self.entries.iter_mut().find(|e| e.dim == dim) {
            entry.kernel = variant.name().to_string();
            entry.selected_at = now;
        } else {
            self.entries.push(KernelCacheEntry {
                dim,
                kernel: variant.name().to_string(),
                selected_at: now,
            });
        }
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_kernels_has_scalar() {
        let kernels = available_kernels();
        assert!(kernels.contains(&KernelVariant::Scalar));
    }

    #[test]
    fn available_kernels_has_avx2_when_supported() {
        let kernels = available_kernels();
        if super::super::avx2::is_avx2_supported() {
            assert!(kernels.contains(&KernelVariant::Avx2), "AVX2 supported but not in available_kernels");
        }
    }

    #[test]
    fn measure_latency_returns_positive() {
        let lat = measure_latency_ns(KernelVariant::Scalar, 128);
        assert!(lat > 0);
    }

    #[test]
    fn measure_qps_returns_positive() {
        let qps = measure_qps(KernelVariant::Scalar, 128);
        assert!(qps > 0);
    }

    #[test]
    fn select_kernel_picks_available_candidate() {
        // Week 3-4：AVX2 已加入候选，选择结果必须来自 available_kernels
        // 用极短稳定性测试避免 5 分钟等待
        let k = select_kernel_with_duration(128, Duration::from_millis(50));
        let avail = available_kernels();
        assert!(avail.contains(&k), "selected kernel {:?} not in available {:?}", k, avail);
    }

    #[test]
    fn env_var_forces_scalar() {
        // 临时设置环境变量强制 scalar
        std::env::set_var("RAVEN_KERNEL", "scalar");
        let k = select_kernel_with_duration(128, Duration::from_millis(50));
        std::env::remove_var("RAVEN_KERNEL");
        assert_eq!(k, KernelVariant::Scalar, "RAVEN_KERNEL=scalar should force scalar");
    }

    #[test]
    fn env_var_forces_avx2_when_supported() {
        if !super::super::avx2::is_avx2_supported() {
            return;
        }
        std::env::set_var("RAVEN_KERNEL", "avx2");
        let k = select_kernel_with_duration(128, Duration::from_millis(50));
        std::env::remove_var("RAVEN_KERNEL");
        assert_eq!(k, KernelVariant::Avx2, "RAVEN_KERNEL=avx2 should force avx2");
    }

    #[test]
    fn env_var_invalid_value_ignored() {
        std::env::set_var("RAVEN_KERNEL", "nonsense");
        // 无效值应被忽略，走正常选择路径
        let k = select_kernel_with_duration(128, Duration::from_millis(50));
        std::env::remove_var("RAVEN_KERNEL");
        let avail = available_kernels();
        assert!(avail.contains(&k));
    }

    #[test]
    fn kernel_cache_set_get() {
        let mut cache = KernelCache::default();
        cache.set(768, KernelVariant::Scalar);
        assert_eq!(cache.get(768), Some(KernelVariant::Scalar));
        assert_eq!(cache.get(128), None);
    }

    #[test]
    fn kernel_cache_roundtrip() {
        let mut cache = KernelCache::default();
        cache.set(768, KernelVariant::Avx2);
        cache.set(1536, KernelVariant::Scalar);

        let tmp = std::env::temp_dir().join("raven_kernel_cache_test.toml");
        cache.save(&tmp).expect("save");
        let loaded = KernelCache::load(&tmp).expect("load");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.get(768), Some(KernelVariant::Avx2));
        assert_eq!(loaded.get(1536), Some(KernelVariant::Scalar));
        assert_eq!(loaded.get(256), None);
    }

    #[test]
    fn fallback_kernel_is_scalar() {
        assert_eq!(fallback_kernel(768), KernelVariant::Scalar);
    }
}
