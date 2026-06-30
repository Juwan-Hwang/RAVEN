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
}
