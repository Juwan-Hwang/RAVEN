//! 上层导航：双机制
//!
//! 设计文档第三层：
//! 默认层：保留随机层级（可与 HNSW 直接对比）
//! 可选层：√N 个 centroid overlay 锚点节点（可关闭）

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
    pub fn new(n: usize, config: NavigationConfig) -> Self {
        let centroids = if config.enable_centroid_overlay {
            let count = config.centroid_count.unwrap_or_else(|| (n as f64).sqrt() as usize);
            // 简化：均匀采样作为 centroid
            // 实际实现应使用 k-means 聚类
            (0..count as u32)
                .map(|i| (i as usize * n / count) as u32)
                .filter(|&c| c < n as u32)
                .collect()
        } else {
            Vec::new()
        };
        Self { config, centroids }
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

    #[test]
    fn default_navigation_no_overlay() {
        let nav = NavigationLayer::new(1000, NavigationConfig::default());
        assert!(!nav.is_overlay_enabled());
        assert!(nav.centroids().is_empty());
    }

    #[test]
    fn overlay_enabled_uses_sqrt_n() {
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: None,
        };
        let nav = NavigationLayer::new(10000, config);
        assert!(nav.is_overlay_enabled());
        // √10000 = 100
        assert_eq!(nav.centroids().len(), 100);
    }

    #[test]
    fn overlay_custom_count() {
        let config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: Some(50),
        };
        let nav = NavigationLayer::new(1000, config);
        assert_eq!(nav.centroids().len(), 50);
    }
}
