//! 延迟剪枝策略
//!
//! 设计文档第四层：
//! 硬上限 R_max（如 64）：最终图的出度上界
//! 软上限 R_soft = 1.5 × R_max（如 96）：构建期允许临时膨胀
//!
//! 触发时机：
//!   1. 节点出度超过 R_soft 才触发单节点 RobustPrune
//!   2. 整个数据集插入完成后，freeze 前做一次全局 final prune
//!
//! 预期效果（待基准验证）：
//!   RobustPrune 调用次数显著降低，建图速度预期大幅提升

use crate::memory::HybridBlockedCsr;

/// 延迟剪枝控制器
///
/// 设计文档第四层：延迟剪枝策略
pub struct DelayedPruneController {
    /// 硬上限 R_max
    pub r_max: usize,
    /// 软上限 R_soft = 1.5 × R_max
    pub r_soft: usize,
    /// 单节点 prune 触发次数（统计用）
    pub single_prune_count: usize,
    /// 全局 final prune 触发次数
    pub final_prune_count: usize,
}

impl DelayedPruneController {
    /// 创建控制器
    pub fn new(r_max: usize) -> Self {
        Self {
            r_max,
            r_soft: (r_max as f32 * 1.5) as usize,
            single_prune_count: 0,
            final_prune_count: 0,
        }
    }

    /// 检查节点是否需要触发单节点 prune
    ///
    /// 设计文档：节点出度超过 R_soft 才触发单节点 RobustPrune
    #[inline]
    pub fn should_prune(&self, storage: &HybridBlockedCsr, node: u32) -> bool {
        storage.degree(node) > self.r_soft
    }

    /// 记录单节点 prune 触发
    pub fn record_single_prune(&mut self) {
        self.single_prune_count += 1;
    }

    /// 执行全局 final prune
    ///
    /// 设计文档：整个数据集插入完成后，freeze 前做一次全局 final prune
    pub fn final_prune(&mut self, storage: &mut HybridBlockedCsr) {
        self.final_prune_count += 1;
        for node in 0..storage.len() as u32 {
            let (main, overflow) = storage.neighbors_full(node);
            let total = main.len() + overflow.len();
            if total <= self.r_max {
                continue;
            }
            // 合并主块和 overflow，截断到 r_max
            let mut all: Vec<u32> = main.to_vec();
            all.extend_from_slice(overflow);
            all.truncate(self.r_max);
            storage.set_neighbors(node, &all);
        }
    }

    /// 统计当前超过 R_soft 的节点数
    pub fn count_over_soft(&self, storage: &HybridBlockedCsr) -> usize {
        (0..storage.len() as u32)
            .filter(|&n| storage.degree(n) > self.r_soft)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_defaults() {
        let c = DelayedPruneController::new(64);
        assert_eq!(c.r_max, 64);
        assert_eq!(c.r_soft, 96); // 1.5 × 64
    }

    #[test]
    fn should_prune_only_over_soft() {
        let mut storage = HybridBlockedCsr::new(10, 100);
        let c = DelayedPruneController::new(4);
        // r_soft = 6
        storage.add_edge(0, 1);
        assert!(!c.should_prune(&storage, 0)); // degree 1
        for i in 2..7 {
            storage.add_edge(0, i);
        }
        assert!(!c.should_prune(&storage, 0)); // degree 6, not > 6
        storage.add_edge(0, 7);
        assert!(c.should_prune(&storage, 0)); // degree 7 > 6
    }

    #[test]
    fn final_prune_truncates_to_r_max() {
        let mut storage = HybridBlockedCsr::new(10, 100);
        let mut c = DelayedPruneController::new(4);
        // 添加超过 r_max 的邻居
        for i in 1..10 {
            storage.add_edge(0, i);
        }
        assert!(storage.degree(0) > c.r_max);
        c.final_prune(&mut storage);
        assert!(storage.degree(0) <= c.r_max);
        assert_eq!(c.final_prune_count, 1);
    }
}
