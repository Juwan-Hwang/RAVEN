//! 上层导航：双机制
//!
//! 设计文档第三层：
//! 默认层：保留随机层级（可与 HNSW 直接对比）
//! 可选层：√N 个 centroid overlay 锚点节点（可关闭）

use crate::build::ChaCha8Rng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

/// 导航层配置
#[derive(Debug, Clone)]
pub struct NavigationConfig {
    /// 是否启用 centroid overlay 锚点节点
    /// 设计文档：可选层，可关闭
    pub enable_centroid_overlay: bool,
    /// centroid 数量，默认 √N
    /// 设计文档：√N 个 centroid overlay 锚点节点
    pub centroid_count: Option<usize>,
}

impl Default for NavigationConfig {
    fn default() -> Self {
        Self {
            enable_centroid_overlay: false, // 默认关闭，保留随机层级
            centroid_count: None,           // None 表示自动 √N
        }
    }
}

/// 导航层
///
/// 设计文档第三层上层导航：双机制
pub struct NavigationLayer {
    config: NavigationConfig,
    /// centroid 锚点节点列表
    centroids: Vec<u32>,
}

impl NavigationLayer {
    /// 创建导航层
    ///
    /// vectors: 扁平存储的向量数据
    /// dim: 维度
    /// config: 导航配置
    pub fn new(n: usize, vectors: &[f32], dim: usize, config: NavigationConfig) -> Self {
        let centroids = if config.enable_centroid_overlay && n > 0 && dim > 0 {
            let count = config.centroid_count.unwrap_or_else(|| (n as f64).sqrt() as usize);
            let count = count.min(n);
            // 评估报告 M2：用 k-means 聚类选择 centroid（原均匀采样）
            Self::kmeans_centroids(vectors, dim, n, count)
        } else {
            Vec::new()
        };
        Self { config, centroids }
    }

    /// k-means 聚类选择 centroid 锚点节点
    ///
    /// 算法：
    /// 1. k-means++ 初始化选择 k 个聚类中心
    /// 2. 迭代分配 + 更新中心
    /// 3. 选择离每个聚类中心最近的节点作为 centroid
    fn kmeans_centroids(vectors: &[f32], dim: usize, n: usize, k: usize) -> Vec<u32> {
        use crate::distance::l2_simd;

        if k == 0 || n == 0 {
            return Vec::new();
        }

        // 1. k-means++ 初始化（ChaCha8Rng 保证确定性）
        let mut rng = ChaCha8Rng::seed_from(42);
        let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);

        // 第一个中心随机选择
        let first_idx = (0..n).collect::<Vec<_>>().choose(&mut rng).copied().unwrap_or(0);
        centers.push(vectors[first_idx * dim..(first_idx + 1) * dim].to_vec());

        // 后续中心按 D(x)² 概率选择
        for _ in 1..k {
            let mut dists = vec![f32::MAX; n];
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                for c in &centers {
                    let d = l2_simd(v, c);
                    if d < dists[i] {
                        dists[i] = d;
                    }
                }
            }
            // 按距离平方加权选择
            let total: f32 = dists.iter().map(|d| d * d).sum();
            if total <= 0.0 {
                break;
            }
            let r: f32 = rng.gen();
            let mut cum = 0.0f32;
            let mut chosen = 0;
            for i in 0..n {
                cum += dists[i] * dists[i] / total;
                if cum >= r {
                    chosen = i;
                    break;
                }
            }
            centers.push(vectors[chosen * dim..(chosen + 1) * dim].to_vec());
        }

        // 2. 迭代分配 + 更新中心（最多 10 次）
        let mut assignments = vec![0usize; n];
        for _ in 0..10 {
            let mut changed = false;
            // 分配
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                let mut best = 0;
                let mut best_dist = f32::MAX;
                for (j, c) in centers.iter().enumerate() {
                    let d = l2_simd(v, c);
                    if d < best_dist {
                        best_dist = d;
                        best = j;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            // 更新中心
            let mut new_centers = vec![vec![0.0f32; dim]; centers.len()];
            let mut counts = vec![0usize; centers.len()];
            for i in 0..n {
                let a = assignments[i];
                let v = &vectors[i * dim..(i + 1) * dim];
                for d in 0..dim {
                    new_centers[a][d] += v[d];
                }
                counts[a] += 1;
            }
            for j in 0..centers.len() {
                if counts[j] > 0 {
                    for d in 0..dim {
                        new_centers[j][d] /= counts[j] as f32;
                    }
                    centers[j] = std::mem::replace(&mut new_centers[j], Vec::new());
                }
            }
        }

        // 3. 选择离每个聚类中心最近的节点作为 centroid
        let mut result: Vec<u32> = Vec::with_capacity(centers.len());
        for c in &centers {
            let mut best = 0u32;
            let mut best_dist = f32::MAX;
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                let d = l2_simd(v, c);
                if d < best_dist {
                    best_dist = d;
                    best = i as u32;
                }
            }
            if !result.contains(&best) {
                result.push(best);
            }
        }
        result
    }

    /// 获取 centroid 锚点节点
    pub fn centroids(&self) -> &[u32] {
        &self.centroids
    }

    /// 是否启用 centroid overlay
    pub fn is_overlay_enabled(&self) -> bool {
        self.config.enable_centroid_overlay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| i as f32).collect()
    }

    #[test]
    fn default_navigation_no_overlay() {
        let v = make_vectors(1000, 4);
        let nav = NavigationLayer::new(1000, &v, 4, NavigationConfig::default());
        assert!(!nav.is_overlay_enabled());
        assert!(nav.centroids().is_empty());
    }

    #[test]
    fn overlay_enabled_uses_sqrt_n() {
        let v = make_vectors(10000, 4);
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: None,
        };
        let nav = NavigationLayer::new(10000, &v, 4, config);
        assert!(nav.is_overlay_enabled());
        // √10000 = 100
        assert_eq!(nav.centroids().len(), 100);
    }

    #[test]
    fn overlay_custom_count() {
        let v = make_vectors(1000, 4);
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: Some(50),
        };
        let nav = NavigationLayer::new(1000, &v, 4, config);
        assert_eq!(nav.centroids().len(), 50);
    }

    #[test]
    fn kmeans_centroids_are_valid_nodes() {
        // 验证 k-means 选择的 centroid 都是有效的节点 ID
        let v = make_vectors(100, 4);
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: Some(10),
        };
        let nav = NavigationLayer::new(100, &v, 4, config);
        for &c in nav.centroids() {
            assert!(c < 100, "centroid {} should be < 100", c);
        }
    }
}
