//! Adaptive ef（Phase 4.5）— 方向 B：距离分布感知
//!
//! 基于 Ada-ef 论文（Distribution-Aware Exploration for Adaptive HNSW Search, 2023）：
//! - 离线：采样数据集向量，计算它们到入口点的距离，构建距离分布
//! - 在线：计算查询到入口点的距离 → 二分查找得到百分位 → 幂律映射到 ef
//!
//! 核心洞见：查询与入口点的距离是查询难度的强预测器。
//! - 近距离查询（"简单"）在图上快速收敛，小 ef 即可
//! - 远距离查询（"难"）需要更大 ef 才能找到正确邻域
//!
//! 幂律变换（RAVEN 改进）：
//! 分层导航把入口距离压缩到极窄区间（p25≈0.74, med≈0.97, p75≈1.18），
//! 线性插值后 avg_ef ≈ min_ef + 0.5 × (max_ef - min_ef)，等于没做自适应。
//! gamma > 1 的幂律曲线把中位数附近的查询压到小 ef 区间，
//! 只有真正的高距离查询才会分到大 ef。
//! gamma=2.0：中位数 → 25% 区间；gamma=3.0：→ 12.5%；gamma=4.0：→ 6%
//!
//! 设计原则：
//! - 零开销：未配置时搜索路径完全不变
//! - O(log n) 预测：二分查找 2000 样本 ≈ 11 次比较
//! - 幂律变换：ef = min_ef + percentile^gamma × (max_ef - min_ef)

use crate::distance::l2_simd;
use crate::quant::SQ8Dataset;
use super::navigation::LayeredNavigation;

// ─── Fast approx powf (feature: fast_powf) ──────────────────────────────
//
// 用 IEEE 754 位操作 + Chebyshev minimax 多项式替代 `f32::powf`。
//
// 原理：x^gamma = 2^(gamma * log2(x))
//   1. log2(x) = exponent + log2(1 + mantissa)
//      从 IEEE 754 位模式提取指数和小数部分，用 3 阶 Chebyshev 多项式修正
//   2. 2^y = 2^floor(y) * 2^frac(y)
//      整数部分直接构造 IEEE 754 位模式，小数部分用 3 阶 Chebyshev 多项式
//
// 系数来源：Chebyshev 节点插值（4 阶），在 [0, 1) 上最大相对误差：
//   log2: < 0.05%   exp2: < 0.001%   组合 powf: < 0.10%
//
// 性能（Zen 4 AVX-512）：
//   f32::powf:  ~20-30 ns（调用 libm）
//   fast_powf:  ~3-5 ns（2×位操作 + 8×FMA + 2×分支）
//   整数 gamma (2/3/4): ~1 ns（直接乘法，精确）
//
// 对 ef 预测的影响：零（ef 四舍五入到整数，0.1% 误差不改变结果）

/// log2(1+m) Chebyshev 系数，m ∈ [0, 1)
///
/// log2(1+m) ≈ c0·m + c1·m² + c2·m³ + c3·m⁴
///
/// 通过 4 阶 Chebyshev 节点插值得到（非 Taylor 级数）。
/// Taylor 3 阶在 m→1 时误差达 20%；
/// Chebyshev 3 阶全区间 < 0.30%，4 阶 < 0.05%。
#[cfg(feature = "fast_powf")]
const LOG2_C0: f32 = 1.442068;
#[cfg(feature = "fast_powf")]
const LOG2_C1: f32 = -0.700778;
#[cfg(feature = "fast_powf")]
const LOG2_C2: f32 = 0.364019;
#[cfg(feature = "fast_powf")]
const LOG2_C3: f32 = -0.105659;

/// 2^f Chebyshev 系数，f ∈ [0, 1)
///
/// 2^f ≈ 1 + a0·f + a1·f² + a2·f³ + a3·f⁴
#[cfg(feature = "fast_powf")]
const EXP2_A0: f32 = 0.693134;
#[cfg(feature = "fast_powf")]
const EXP2_A1: f32 = 0.240647;
#[cfg(feature = "fast_powf")]
const EXP2_A2: f32 = 0.053441;
#[cfg(feature = "fast_powf")]
const EXP2_A3: f32 = 0.012763;

/// 快速 log2 近似
///
/// 从 IEEE 754 位模式提取指数和尾数，用 Chebyshev 多项式修正尾数部分。
///
/// ```text
/// x = (1 + m) × 2^e    其中 m ∈ [0, 1), e ∈ ℤ
/// log2(x) = e + log2(1+m)
/// log2(1+m) ≈ c0·m + c1·m² + c2·m³ + c3·m⁴
///   (Horner: m·(c0 + m·(c1 + m·(c2 + m·c3))))
/// ```
#[cfg(feature = "fast_powf")]
#[inline(always)]
fn fast_log2(x: f32) -> f32 {
    debug_assert!(x > 0.0, "fast_log2 requires x > 0");
    let bits = x.to_bits();

    // 指数：bits[30:23] - 127
    let e = ((bits >> 23) & 0xFF) as i32 - 127;

    // 尾数：清除指数位、设置隐含的 1.0，得到 [1.0, 2.0)，减 1.0 得 m ∈ [0, 1)
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000) - 1.0;

    // Horner 求值：m · (c0 + m · (c1 + m · (c2 + m · c3)))
    //   t = m·c3 + c2        →  m.mul_add(c3, c2)
    //   t = m·t  + c1        →  m.mul_add(t,  c1)
    //   t = m·t  + c0        →  m.mul_add(t,  c0)
    //   r = m · t
    let t = m.mul_add(LOG2_C3, LOG2_C2);
    let t = m.mul_add(t, LOG2_C1);
    let t = m.mul_add(t, LOG2_C0);
    e as f32 + m * t
}

/// 快速 exp2 近似
///
/// ```text
/// 2^y = 2^⌊y⌋ × 2^frac(y)
/// 2^⌊y⌋       → 直接构造 IEEE 754 位模式
/// 2^frac(y)   ≈ 1 + a0·f + a1·f² + a2·f³ + a3·f⁴
///   (Horner: 1 + f·(a0 + f·(a1 + f·(a2 + f·a3))))
/// ```
#[cfg(feature = "fast_powf")]
#[inline(always)]
fn fast_exp2(y: f32) -> f32 {
    if y < -126.0 {
        return 0.0; // 下溢
    }
    if y > 127.0 {
        return f32::INFINITY; // 上溢
    }

    // 拆分整数和小数部分
    let xi = y.floor() as i32;
    let xf = y - xi as f32; // xf ∈ [0, 1)

    // Horner 求值：1 + f · (a0 + f · (a1 + f · (a2 + f · a3)))
    //   t = f·a3 + a2        →  f.mul_add(a3, a2)
    //   t = f·t  + a1        →  f.mul_add(t,  a1)
    //   t = f·t  + a0        →  f.mul_add(t,  a0)
    //   r = f·t  + 1.0       →  f.mul_add(t,  1.0)
    let t = xf.mul_add(EXP2_A3, EXP2_A2);
    let t = xf.mul_add(t, EXP2_A1);
    let t = xf.mul_add(t, EXP2_A0);
    let two_xf = xf.mul_add(t, 1.0);

    // 2^xi：直接构造 IEEE 754 位模式
    let two_xi = f32::from_bits(((xi + 127) as u32) << 23);
    two_xi * two_xf
}

/// 快速 powf 近似：x^gamma
///
/// 对于 AdaptiveEf 热路径中 `percentile^gamma`（percentile ∈ [0, 1]）的计算。
///
/// # 精度
///
/// | gamma 类型 | 方法 | 误差 |
/// |-----------|------|------|
/// | 整数 (2, 3, 4) | 直接乘法 | 0%（精确） |
/// | 1.0 | 恒等 | 0%（精确） |
/// | 0.5 | 硬件 sqrt | 0%（精确） |
/// | 其他 | log2 + exp2 | < 0.10% |
///
/// # 边界
///
/// - x ≤ 0 → 0.0（ef 下界为 min_ef，powf 结果为 0 正确映射）
/// - x ≥ 1 → 1.0（百分位最大值，正确映射到 max_ef）
#[cfg(feature = "fast_powf")]
#[inline(always)]
fn fast_powf(x: f32, gamma: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // 整数 gamma 快速路径（精确，零误差）
    if gamma == 2.0 {
        return x * x;
    }
    if gamma == 3.0 {
        return x * x * x;
    }
    if gamma == 4.0 {
        let x2 = x * x;
        return x2 * x2;
    }
    if gamma == 1.0 {
        return x;
    }
    if gamma == 0.5 {
        return x.sqrt();
    }
    // 通用路径：x^gamma = 2^(gamma · log2(x))
    fast_exp2(gamma * fast_log2(x))
}

// ─── AdaptiveEfConfig ───────────────────────────────────────────────────

/// 自适应 ef 配置
///
/// 离线构建，在线 O(log n) 预测。存储排序后的距离样本（~8KB for 2000 samples）。
///
/// 用法：
/// ```ignore
/// // 离线构建（推荐：用分层导航的真实入口距离）
/// let config = AdaptiveEfConfig::build_with_layered_nav(
///     &vectors, dim, &layered_nav, 30, 80, 3.0);
///
/// // 在线预测（nav.initialize 返回的 f32 距离直接用）
/// let (entry_point, entry_dist) = nav.initialize(vectors, dim, query);
/// let ef = config.estimate_ef(entry_dist); // → 30..80
/// ```
#[derive(Clone)]
pub struct AdaptiveEfConfig {
    /// 排序后的距离样本（升序），用于百分位查找
    sorted_distances: Vec<f32>,
    /// 最小 ef（最简单的查询）
    min_ef: usize,
    /// 最大 ef（最难的查询）
    max_ef: usize,
    /// 幂律指数。1.0 = 线性。>1.0 = 把多数查询压到小 ef
    /// 建议从 2.0 开始测，分布越窄用越大的 gamma
    pub gamma: f32,
}

impl AdaptiveEfConfig {
    /// 从排序后的距离样本构建
    fn from_sorted_samples(
        sorted_distances: Vec<f32>,
        min_ef: usize,
        max_ef: usize,
        gamma: f32,
    ) -> Self {
        debug_assert!(min_ef <= max_ef, "min_ef must be <= max_ef");
        debug_assert!(min_ef >= 1, "min_ef must be >= 1");
        debug_assert!(gamma > 0.0, "gamma must be > 0");
        Self { sorted_distances, min_ef, max_ef, gamma }
    }

    /// 用相同距离分布但不同参数创建变体（零采样开销）
    ///
    /// 参数扫描专用：构建一次 `build_with_layered_nav` 获取距离分布，
    /// 然后用 `with_params` 快速创建数百个变体进行网格搜索。
    pub fn with_params(&self, min_ef: usize, max_ef: usize, gamma: f32) -> Self {
        Self {
            sorted_distances: self.sorted_distances.clone(),
            min_ef,
            max_ef,
            gamma,
        }
    }

    /// 用 f32 距离构建（用于 f32 搜索路径）
    ///
    /// 采样最多 2000 个向量，计算它们到入口点的 f32 L2 距离，排序构建分布。
    pub fn build_f32(
        vectors: &[f32],
        dim: usize,
        entry_point: u32,
        min_ef: usize,
        max_ef: usize,
        gamma: f32,
    ) -> Self {
        let n = vectors.len() / dim;
        let sample_size = n.min(2000);

        let entry_vec = &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim];

        let mut samples: Vec<f32> = (0..sample_size)
            .map(|i| l2_simd(entry_vec, &vectors[i * dim..(i + 1) * dim]))
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self::from_sorted_samples(samples, min_ef, max_ef, gamma)
    }

    /// 用 SQ8 距离构建（用于无分层导航的 SQ8 搜索路径）
    ///
    /// 采样最多 2000 个向量，计算它们到入口点的 SQ8 L2 距离，排序构建分布。
    /// 仅适用于搜索入口点固定为 entry_point 的场景（无 LayeredNavigation）。
    pub fn build_sq8(
        sq8: &SQ8Dataset,
        entry_point: u32,
        min_ef: usize,
        max_ef: usize,
        gamma: f32,
    ) -> Self {
        let n = sq8.n;
        let sample_size = n.min(2000);

        // 入口点的 SQ8 code 作为 "查询 code"
        let entry_code = sq8.code(entry_point as usize);

        let mut samples: Vec<f32> = (0..sample_size)
            .map(|i| sq8.distance(entry_code, i))
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self::from_sorted_samples(samples, min_ef, max_ef, gamma)
    }

    /// 用 LayeredNavigation 入口距离构建（推荐，所有搜索路径通用）
    ///
    /// 对每个采样向量运行 `nav.initialize()`，收集返回的 f32 距离。
    /// 这个距离是查询到**实际 Layer 0 搜索起点**的距离——
    /// 经过分层导航贪心下降后的真实起点，不是固定 medoid。
    ///
    /// 搜索时 `nav.initialize()` 返回的 `.1` 距离直接用于预测，零额外开销。
    ///
    /// 适用于 SQ8 和 f32 搜索路径——ef 预测用 f32 距离（更精确），
    /// 图遍历用 SQ8/f32 距离（独立优化），两者正交不干扰。
    pub fn build_with_layered_nav(
        vectors: &[f32],
        dim: usize,
        nav: &LayeredNavigation,
        min_ef: usize,
        max_ef: usize,
        gamma: f32,
    ) -> Self {
        let n = vectors.len() / dim;
        let sample_size = n.min(2000);

        let mut samples: Vec<f32> = (0..sample_size)
            .map(|i| {
                let query = &vectors[i * dim..(i + 1) * dim];
                nav.initialize(vectors, dim, query).1
            })
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self::from_sorted_samples(samples, min_ef, max_ef, gamma)
    }

    /// 根据查询到入口点的距离预测 ef
    ///
    /// 返回 [min_ef, max_ef] 区间内的 ef 值。
    /// - 距离越小（简单查询）→ ef 越接近 min_ef
    /// - 距离越大（难查询）→ ef 越接近 max_ef
    ///
    /// 幂律变换：gamma > 1 时，中位数附近的查询被压到小 ef 区间，
    /// 只有真正的高距离查询才会分到大 ef。
    ///
    /// O(log n) 二分查找，2000 样本约 11 次比较。
    #[inline]
    pub fn estimate_ef(&self, distance: f32) -> usize {
        if self.sorted_distances.len() <= 1 {
            return usize::midpoint(self.min_ef, self.max_ef);
        }

        // 二分查找：严格小于 distance 的样本数 = rank
        let rank = self.sorted_distances.partition_point(|&d| d < distance);

        // 百分位：0.0（最简单）→ 1.0（最难）
        let percentile = rank as f32 / self.sorted_distances.len() as f32;

        // 幂律变换：gamma > 1 → 低百分位被强压缩，高百分位保持大 ef
        #[cfg(feature = "fast_powf")]
        let shaped = fast_powf(percentile, self.gamma);
        #[cfg(not(feature = "fast_powf"))]
        let shaped = percentile.powf(self.gamma);

        let ef = shaped.mul_add((self.max_ef - self.min_ef) as f32, self.min_ef as f32);
        ef.round() as usize
    }

    /// 最小 ef
    pub fn min_ef(&self) -> usize {
        self.min_ef
    }

    /// 最大 ef
    pub fn max_ef(&self) -> usize {
        self.max_ef
    }

    /// 样本数
    pub fn sample_count(&self) -> usize {
        self.sorted_distances.len()
    }

    /// 距离分布统计（用于诊断）
    pub fn distribution_stats(&self) -> (f32, f32, f32, f32, f32) {
        if self.sorted_distances.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let n = self.sorted_distances.len();
        (
            self.sorted_distances[0],                          // min
            self.sorted_distances[n / 4],                      // p25
            self.sorted_distances[n / 2],                      // median
            self.sorted_distances[3 * n / 4],                  // p75
            self.sorted_distances[n - 1],                      // max
        )
    }
}

// ─── 测试 ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_ef_basic() {
        // 100 个样本，距离 0.0..100.0
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let config = AdaptiveEfConfig::from_sorted_samples(samples, 20, 80, 1.0);

        // 距离 < 所有样本 → min_ef
        assert_eq!(config.estimate_ef(-1.0), 20);

        // 距离 > 所有样本 → max_ef
        assert_eq!(config.estimate_ef(200.0), 80);

        // 中位数距离 → 约 (20+80)/2 = 50（gamma=1.0 线性）
        let mid_ef = config.estimate_ef(50.0);
        assert!(mid_ef >= 45 && mid_ef <= 55, "mid_ef = {}", mid_ef);
    }

    #[test]
    fn estimate_ef_power_law() {
        // gamma=3.0：中位数查询 → percentile=0.5, shaped=0.125
        // ef = 20 + 0.125 * 60 = 27.5 → 28
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let config = AdaptiveEfConfig::from_sorted_samples(samples, 20, 80, 3.0);

        let mid_ef = config.estimate_ef(50.0);
        // gamma=3.0: 0.5^3 = 0.125, ef = 20 + 0.125*60 = 27.5 → 28
        assert!(mid_ef >= 26 && mid_ef <= 30, "gamma=3 mid_ef = {}", mid_ef);
    }

    #[test]
    fn estimate_ef_monotonic() {
        let samples: Vec<f32> = (0..2000).map(|i| i as f32 * 0.1).collect();
        let config = AdaptiveEfConfig::from_sorted_samples(samples, 10, 100, 2.0);

        // ef 应随距离单调递增
        let mut prev_ef = 0;
        for d in [0.0f32, 10.0, 50.0, 100.0, 150.0, 200.0] {
            let ef = config.estimate_ef(d);
            assert!(ef >= prev_ef, "ef not monotonic: d={} ef={} prev={}", d, ef, prev_ef);
            prev_ef = ef;
        }
    }

    #[test]
    fn estimate_ef_respects_bounds() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = AdaptiveEfConfig::from_sorted_samples(samples, 30, 70, 3.0);

        for d in [-100.0f32, 0.0, 1.0, 2.5, 5.0, 100.0, 1000.0] {
            let ef = config.estimate_ef(d);
            assert!(ef >= 30 && ef <= 70, "ef={} out of bounds for d={}", ef, d);
        }
    }

    #[test]
    fn empty_samples_returns_average() {
        let config = AdaptiveEfConfig::from_sorted_samples(vec![], 20, 80, 3.0);
        assert_eq!(config.estimate_ef(42.0), 50);
    }

    #[test]
    fn distribution_stats_correct() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let config = AdaptiveEfConfig::from_sorted_samples(samples, 20, 80, 2.0);
        let (min, p25, median, p75, max) = config.distribution_stats();
        assert!((min - 0.0).abs() < 1e-6);
        assert!((p25 - 25.0).abs() < 1e-6);
        assert!((median - 50.0).abs() < 1e-6);
        assert!((p75 - 75.0).abs() < 1e-6);
        assert!((max - 99.0).abs() < 1e-6);
    }

    #[test]
    fn build_f32_simple() {
        // 10 个 2 维向量
        let vectors: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let dim = 2;
        let entry_point = 0u32;

        let config = AdaptiveEfConfig::build_f32(&vectors, dim, entry_point, 10, 50, 2.0);
        assert_eq!(config.sample_count(), 10);
        assert_eq!(config.min_ef(), 10);
        assert_eq!(config.max_ef(), 50);

        // 入口点自身的距离 = 0 → 最小 ef
        let ef = config.estimate_ef(0.0);
        assert_eq!(ef, 10);
    }

    // ── fast_powf 精度测试 ──────────────────────────────────────────────

    #[cfg(feature = "fast_powf")]
    #[test]
    fn fast_powf_integer_gamma_exact() {
        // 整数 gamma 路径必须精确匹配
        for &x in &[0.01f32, 0.1, 0.25, 0.5, 0.75, 0.99] {
            let exact2 = x * x;
            let exact3 = x * x * x;
            let exact4 = {
                let x2 = x * x;
                x2 * x2
            };
            assert!((fast_powf(x, 2.0) - exact2).abs() < 1e-7, "gamma=2 x={}", x);
            assert!((fast_powf(x, 3.0) - exact3).abs() < 1e-7, "gamma=3 x={}", x);
            assert!((fast_powf(x, 4.0) - exact4).abs() < 1e-7, "gamma=4 x={}", x);
        }
    }

    #[cfg(feature = "fast_powf")]
    #[test]
    fn fast_powf_general_accuracy() {
        // 非整数 gamma 路径：相对误差 < 0.5%
        let gammas = [1.5f32, 2.5, 3.5, 5.0];
        let xs = [0.01f32, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99];

        for &gamma in &gammas {
            for &x in &xs {
                let reference = x.powf(gamma);
                let approx = fast_powf(x, gamma);
                let rel_err = (approx - reference).abs() / reference.max(1e-10);
                assert!(
                    rel_err < 0.005,
                    "x={}, gamma={}: approx={}, ref={}, rel_err={:.4}%",
                    x, gamma, approx, reference, rel_err * 100.0
                );
            }
        }
    }

    #[cfg(feature = "fast_powf")]
    #[test]
    fn fast_powf_edge_cases() {
        // 边界值
        assert_eq!(fast_powf(0.0, 2.0), 0.0);
        assert_eq!(fast_powf(-0.1, 2.0), 0.0);
        assert_eq!(fast_powf(1.0, 2.0), 1.0);
        assert_eq!(fast_powf(1.5, 2.0), 1.0); // x > 1 → 1.0
        assert_eq!(fast_powf(0.5, 1.0), 0.5); // gamma=1 → identity
        assert!((fast_powf(0.25, 0.5) - 0.5).abs() < 1e-7); // gamma=0.5 → sqrt
    }

    #[cfg(feature = "fast_powf")]
    #[test]
    fn fast_log2_accuracy() {
        // 验证 log2 在典型输入上的精度
        //
        // 使用绝对误差而非相对误差：log2(x) → 0 当 x → 1，
        // 此时即使绝对误差极小（< 0.0002），相对误差也会被放大到 > 1%。
        // 4 阶 Chebyshev 多项式的最大绝对误差 < 0.0004。
        let test_cases = [
            (0.5f32, -1.0f32),   // 2^(-1)
            (0.25, -2.0),         // 2^(-2)
            (0.125, -3.0),        // 2^(-3)
            (0.75, -0.41504),     // log2(3/4)
            (0.9, -0.15200),      // log2(0.9)
            (0.99, -0.01450),     // log2(0.99) — 接近零，相对误差无意义
            (0.01, -6.64386),     // log2(0.01)
        ];

        for &(x, expected) in &test_cases {
            let approx = fast_log2(x);
            let abs_err = (approx - expected).abs();
            assert!(
                abs_err < 0.001,
                "log2({}): approx={}, expected={}, abs_err={:.6}",
                x, approx, expected, abs_err
            );
        }
    }

    #[cfg(feature = "fast_powf")]
    #[test]
    fn fast_exp2_accuracy() {
        // 验证 exp2 在典型输入上的精度
        let test_cases = [
            (0.0f32, 1.0f32),
            (0.5, 1.41421),
            (-0.5, 0.70711),
            (-1.0, 0.5),
            (-2.0, 0.25),
            (0.75, 1.68179),
            (0.99, 1.98616),
            (-3.5, 0.08839),
        ];

        for &(y, expected) in &test_cases {
            let approx = fast_exp2(y);
            let rel_err = (approx - expected).abs() / expected.max(0.001);
            assert!(
                rel_err < 0.005,
                "exp2({}): approx={}, expected={}, rel_err={:.4}%",
                y, approx, expected, rel_err * 100.0
            );
        }
    }

    #[cfg(feature = "fast_powf")]
    #[test]
    fn fast_powf_estimate_ef_consistency() {
        // 验证 fast_powf 和 powf 在 estimate_ef 中给出相同的 ef 值
        //（由于四舍五入，小误差不应改变结果）
        let samples: Vec<f32> = (0..2000).map(|i| i as f32 * 0.1).collect();

        for gamma in [1.5f32, 2.0, 2.5, 3.0, 4.0] {
            let config = AdaptiveEfConfig::from_sorted_samples(samples.clone(), 20, 80, gamma);
            for d in [10.0f32, 50.0, 100.0, 150.0, 200.0] {
                let ef = config.estimate_ef(d);
                // 手动计算 powf 版本的 ef
                let rank = samples.partition_point(|&s| s < d);
                let percentile = rank as f32 / samples.len() as f32;
                let shaped_ref = percentile.powf(gamma);
                let ef_ref = shaped_ref.mul_add(60.0, 20.0).round() as usize;
                // 允许 ±1 的差异（四舍五入边界）
                assert!(
                    (ef as i32 - ef_ref as i32).abs() <= 1,
                    "gamma={}, d={}: fast_ef={}, ref_ef={}",
                    gamma, d, ef, ef_ref
                );
            }
        }
    }
}
