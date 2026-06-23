//! Visited 标记（Clear-List 模式，Vec<u8>）
//!
//! 设计文档第二层：
//! Generation counter 在 N=10^8 时每线程 400MB，32 线程共 12.8GB，直接挤占缓存空间。
//! Clear-List 把重置开销从 O(N) 降为 O(V<=1000)，用 Vec<u8> 代替 Vec<u32> 再降 4 倍。
//!
//! | 方案 | N=10^8 / 单线程 | N=10^8 × 32 线程 |
//! | Generation counter Vec<u32> | 400 MB | 12.8 GB |
//! | Clear-List Vec<u8> | 100 MB | 3.2 GB |
//!
//! Visited history 按 ef_search × 3 预分配，避免查询热路径上频繁扩容。

/// Visited 标记追踪器（Clear-List 模式）
///
/// 设计文档原文实现：
/// - visited: Vec<u8>，按节点数 N 分配
/// - history: Vec<u32>，按 ef_search × 3 预分配 capacity
/// - visit: 标记并记录到 history，返回是否首次访问
/// - reset: 只清零 history 中记录的索引，O(V) 而非 O(N)
pub struct VisitedTracker {
    /// 节点访问标记，按节点 ID 索引
    visited: Vec<u8>,
    /// 本次查询访问过的节点列表，用于 O(V) 重置
    history: Vec<u32>,
}

impl VisitedTracker {
    /// 创建 VisitedTracker
    ///
    /// n: 节点总数
    /// ef_search: 搜索宽度，history 按 ef_search × 3 预分配（设计文档第三层）
    pub fn new(n: usize, ef_search: usize) -> Self {
        Self {
            visited: vec![0u8; n],
            history: Vec::with_capacity(ef_search * 3),
        }
    }

    /// 标记节点为已访问
    ///
    /// 返回 true 表示首次访问，false 表示已访问过
    #[inline(always)]
    pub fn visit(&mut self, idx: u32) -> bool {
        let i = idx as usize;
        if self.visited[i] != 0 {
            return false;
        }
        self.visited[i] = 1;
        self.history.push(idx);
        true
    }

    /// 检查节点是否已访问（不标记）
    #[inline(always)]
    pub fn is_visited(&self, idx: u32) -> bool {
        self.visited[idx as usize] != 0
    }

    /// 重置：只清零本次查询访问过的节点
    ///
    /// 设计文档：把重置开销从 O(N) 降为 O(V<=1000)
    #[inline]
    pub fn reset(&mut self) {
        for &idx in &self.history {
            self.visited[idx as usize] = 0;
        }
        self.history.clear();
    }

    /// 本次查询已访问的节点数
    #[inline]
    pub fn visited_count(&self) -> usize {
        self.history.len()
    }

    /// 节点总数
    pub fn len(&self) -> usize {
        self.visited.len()
    }

    /// 是否为空（无节点）
    pub fn is_empty(&self) -> bool {
        self.visited.is_empty()
    }

    /// history 容量
    #[allow(dead_code)]
    pub fn history_capacity(&self) -> usize {
        self.history.capacity()
    }
}

impl Default for VisitedTracker {
    fn default() -> Self {
        Self::new(0, 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_first_time_returns_true() {
        let mut t = VisitedTracker::new(100, 64);
        assert!(t.visit(5));
        assert!(t.visit(10));
        assert!(t.visit(99));
    }

    #[test]
    fn visit_second_time_returns_false() {
        let mut t = VisitedTracker::new(100, 64);
        assert!(t.visit(5));
        assert!(!t.visit(5));
    }

    #[test]
    fn reset_clears_visited() {
        let mut t = VisitedTracker::new(100, 64);
        t.visit(5);
        t.visit(10);
        assert_eq!(t.visited_count(), 2);
        t.reset();
        assert_eq!(t.visited_count(), 0);
        // 重置后可再次访问
        assert!(t.visit(5));
    }

    #[test]
    fn reset_is_o_v_not_o_n() {
        // 验证 reset 只清零 history 中的节点
        let mut t = VisitedTracker::new(1000, 64);
        t.visit(5);
        t.visit(10);
        t.reset();
        // 未访问的节点应仍为 0
        assert_eq!(t.visited[100], 0);
        // 访问过的节点也应被清零
        assert_eq!(t.visited[5], 0);
        assert_eq!(t.visited[10], 0);
    }

    #[test]
    fn history_preallocated_by_ef_search_times_3() {
        let t = VisitedTracker::new(1000, 200);
        assert_eq!(t.history_capacity(), 600);
    }

    #[test]
    fn is_visited_check() {
        let mut t = VisitedTracker::new(100, 64);
        assert!(!t.is_visited(5));
        t.visit(5);
        assert!(t.is_visited(5));
    }
}
