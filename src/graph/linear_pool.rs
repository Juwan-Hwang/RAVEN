//! LinearPool —— 固定容量排序邻居池
//!
//! 融合 Glass 的 LinearPool 与 DiskANN 的 NeighborPriorityQueue 精华，
//! 替换 RAVEN 原有的无界 BinaryHeap，根治 avg_visited 爆炸问题。
//!
//! ## 核心设计
//!
//! 候选集与结果集合并为单一数据结构：
//! - **固定容量** = ef_search：满时直接拒绝比最差更远的候选
//! - **排序数组**（升序）：二分查找插入位置，O(1) 访问最优/最差
//! - **游标弹出**：跟踪已扩展元素，无需单独的候选堆
//!
//! ## 与原实现的对比
//!
//! | 特性 | 原 BinaryHeap | LinearPool |
//! |------|--------------|------------|
//! | 候选池容量 | 无界 | 固定 = ef |
//! | 结果集 | 独立最大堆 | 同一数组 |
//! | 最差访问 | O(1) peek | O(1) data[size-1] |
//! | 拒绝远候选 | 仅 results 满时 | 始终（满时） |
//! | 内存分配 | 持续 push 增长 | 零分配（预分配） |
//!
//! ## 性能原理
//!
//! 无界堆允许大量中距离候选涌入，搜索沿远候选发散，visited 爆炸。
//! 固定容量池在满时立即拒绝远候选，阈值随每次插入动态收紧，
//! 自然将搜索限制在 ~ef × avg_degree 的 visited 范围内。

/// 固定容量排序邻居池
///
/// 融合 Glass LinearPool + DiskANN NeighborPriorityQueue
pub struct LinearPool {
    /// 排序数组 (node_id, distance)，按 distance 升序
    /// 使用 checked 标记位（id 最高位）跟踪已扩展状态
    data: Vec<(u32, f32)>,
    /// 当前元素数量
    size: usize,
    /// 游标：下一个未扩展元素的索引
    cursor: usize,
    /// 最大容量（= ef_search）
    capacity: usize,
    /// ef 参数：搜索在游标达到 ef 时终止
    ef: usize,
}

/// checked 标记位掩码（Glass 技巧：用 id 最高位标记已扩展）
const CHECKED_MASK: u32 = 1u32 << 31;
/// 获取真实 node_id（去除 checked 标记）
#[inline(always)]
fn raw_id(id: u32) -> u32 {
    id & !CHECKED_MASK
}
/// 检查是否已扩展
#[inline(always)]
fn is_checked(id: u32) -> bool {
    (id >> 31) & 1 == 1
}

impl LinearPool {
    /// 创建 LinearPool
    ///
    /// capacity = ef_search：候选池容量等于搜索宽度
    pub fn new(ef: usize) -> Self {
        Self {
            data: vec![(0, 0.0); ef + 1],
            size: 0,
            cursor: 0,
            capacity: ef,
            ef,
        }
    }

    /// 重置池（复用分配的内存）
    #[inline]
    pub fn clear(&mut self) {
        self.size = 0;
        self.cursor = 0;
    }

    /// 为新的 ef 重置池（复用分配的内存，按需扩容）
    ///
    /// 适用于自适应 ef：每次查询的 ef 可能不同。
    /// 若新 ef > 当前容量，自动扩容 data 数组。
    #[inline]
    pub fn reset_for(&mut self, ef: usize) {
        if ef + 1 > self.data.len() {
            self.data.resize(ef + 1, (0, 0.0));
        }
        self.capacity = ef;
        self.ef = ef;
        self.size = 0;
        self.cursor = 0;
    }

    /// 尝试插入候选
    ///
    /// 返回 true 表示成功插入，false 表示被拒绝（池满且距离 >= 最差）
    /// 插入后数组保持升序排列
    #[inline]
    pub fn insert(&mut self, node: u32, dist: f32) -> bool {
        // 池满时：比最差还远（或相等）→ 拒绝
        if self.size == self.capacity {
            // SAFETY: size == capacity > 0 (capacity = ef >= 1)
            let worst = unsafe { *self.data.get_unchecked(self.size - 1) };
            if dist >= worst.1 {
                return false;
            }
        }

        // 二分查找插入位置（升序排列）
        let lo = self.find_insert_pos(dist);

        // 移动元素腾出空间
        let move_count = self.size - lo;
        if move_count > 0 {
            // 保留多分配的 1 个槽位，memmove 不会越界
            unsafe {
                std::ptr::copy(
                    self.data.as_ptr().add(lo),
                    self.data.as_mut_ptr().add(lo + 1),
                    move_count,
                );
            }
        }

        // 写入新元素
        // SAFETY: lo < data.len() (因为 lo <= size <= capacity < data.len() = capacity + 1)
        unsafe {
            *self.data.as_mut_ptr().add(lo) = (node, dist);
        }

        if self.size < self.capacity {
            self.size += 1;
        }
        // else: 最后一个元素被 memmove 挤出，size 不变

        // 如果新元素插入在游标之前，回退游标使其被优先弹出
        if lo < self.cursor {
            self.cursor = lo;
        }

        true
    }

    /// 弹出最近的未扩展元素
    ///
    /// 标记当前元素为已扩展，推进游标到下一个未扩展元素
    /// 返回 None 表示没有更多可扩展元素
    #[inline]
    pub fn pop(&mut self) -> Option<(u32, f32)> {
        if !self.has_next() {
            return None;
        }

        let idx = self.cursor;
        // 标记为已扩展
        // SAFETY: cursor < size < data.len()
        let entry = &mut self.data[idx];
        entry.0 |= CHECKED_MASK;

        // 推进游标到下一个未扩展元素
        self.cursor += 1;
        while self.cursor < self.size && is_checked(self.data[self.cursor].0) {
            self.cursor += 1;
        }

        // 返回真实 ID 和距离
        // SAFETY: idx < size < data.len()
        let (id, dist) = unsafe { *self.data.get_unchecked(idx) };
        Some((raw_id(id), dist))
    }

    /// 是否还有可扩展的元素
    ///
    /// 搜索终止条件：游标 >= ef 或 游标 >= size
    #[inline]
    pub fn has_next(&self) -> bool {
        self.cursor < self.size && self.cursor < self.ef
    }

    /// 查看下一个要弹出的元素（不移除、不标记）
    ///
    /// 用于预取：在 pop 之前获取下一个节点的 ID 以预取其邻居列表
    /// 返回 (node_id, distance) 或 None
    #[inline]
    pub fn peek_unchecked(&self) -> Option<(u32, f32)> {
        if !self.has_next() {
            return None;
        }
        let (id, dist) = self.data[self.cursor];
        Some((raw_id(id), dist))
    }

    /// 当前最差距离（最后一个元素的 distance）
    ///
    /// 池满时用于快速判断是否拒绝新候选
    #[inline]
    pub fn worst_distance(&self) -> f32 {
        if self.size == 0 {
            f32::MAX
        } else {
            // SAFETY: size > 0 guarantees size-1 is valid
            unsafe { self.data.get_unchecked(self.size - 1).1 }
        }
    }

    /// 当前元素数量
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// 提取所有结果（按距离升序），返回 (node_id, distance) 列表
    ///
    /// 清除 checked 标记，返回真实 ID
    pub fn into_sorted_vec(self) -> Vec<(u32, f32)> {
        let mut result = Vec::with_capacity(self.size);
        for i in 0..self.size {
            let (id, dist) = self.data[i];
            result.push((raw_id(id), dist));
        }
        result
    }

    /// 提取结果（不消费 self，用于复用场景）
    pub fn to_sorted_vec(&self) -> Vec<(u32, f32)> {
        let mut result = Vec::with_capacity(self.size);
        for i in 0..self.size {
            let (id, dist) = self.data[i];
            result.push((raw_id(id), dist));
        }
        result
    }

    /// 二分查找插入位置
    ///
    /// 返回第一个 distance > dist 的位置（保持升序）
    #[inline]
    fn find_insert_pos(&self, dist: f32) -> usize {
        let mut lo = 0usize;
        let mut hi = self.size;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            // SAFETY: mid < size
            let mid_dist = unsafe { self.data.get_unchecked(mid).1 };
            if mid_dist > dist {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_pop_basic() {
        let mut pool = LinearPool::new(5);
        assert!(pool.is_empty());

        pool.insert(1, 3.0);
        pool.insert(2, 1.0);
        pool.insert(3, 2.0);

        assert_eq!(pool.len(), 3);

        // Pop should return closest first
        let (id, dist) = pool.pop().unwrap();
        assert_eq!(id, 2);
        assert!((dist - 1.0).abs() < 1e-6);

        let (id, _) = pool.pop().unwrap();
        assert_eq!(id, 3);

        let (id, _) = pool.pop().unwrap();
        assert_eq!(id, 1);

        assert!(pool.pop().is_none());
    }

    #[test]
    fn capacity_rejects_worst() {
        let mut pool = LinearPool::new(3);
        pool.insert(1, 1.0);
        pool.insert(2, 2.0);
        pool.insert(3, 3.0);
        assert_eq!(pool.len(), 3);

        // This should be rejected (dist >= worst = 3.0)
        assert!(!pool.insert(4, 4.0));
        assert_eq!(pool.len(), 3);

        // This should be accepted (dist < worst = 3.0)
        assert!(pool.insert(5, 2.5));
        assert_eq!(pool.len(), 3); // size stays at capacity

        // Verify the worst (3.0) was replaced
        assert!((pool.worst_distance() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn has_next_terminates_at_ef() {
        let mut pool = LinearPool::new(3);
        pool.insert(1, 1.0);
        pool.insert(2, 2.0);
        pool.insert(3, 3.0);

        // Pop all 3 (ef = capacity = 3)
        assert!(pool.has_next());
        pool.pop();
        assert!(pool.has_next());
        pool.pop();
        assert!(pool.has_next());
        pool.pop();
        assert!(!pool.has_next()); // cursor >= ef
    }

    #[test]
    fn insert_before_cursor_adjusts_cursor() {
        let mut pool = LinearPool::new(10);
        pool.insert(1, 5.0);
        pool.insert(2, 10.0);

        // Pop first element (id=1, dist=5.0), cursor moves to 1
        pool.pop();
        assert_eq!(pool.cursor, 1);

        // Insert a closer element - should move cursor back
        pool.insert(3, 3.0);
        assert_eq!(pool.cursor, 0); // cursor adjusted to insertion point

        // Next pop should return the new closer element
        let (id, _) = pool.pop().unwrap();
        assert_eq!(id, 3);
    }

    #[test]
    fn sorted_order_maintained() {
        let mut pool = LinearPool::new(20);
        let distances = [5.0, 1.0, 3.0, 2.0, 4.0, 0.5, 2.5];
        for (i, &d) in distances.iter().enumerate() {
            pool.insert(i as u32, d);
        }

        let result = pool.to_sorted_vec();
        for i in 0..result.len() - 1 {
            assert!(result[i].1 <= result[i + 1].1, "Not sorted at index {}", i);
        }
    }

    #[test]
    fn empty_pool_operations() {
        let mut pool = LinearPool::new(5);
        assert!(pool.is_empty());
        assert!(pool.pop().is_none());
        assert!(!pool.has_next());
        assert_eq!(pool.worst_distance(), f32::MAX);
    }

    #[test]
    fn clear_resets_state() {
        let mut pool = LinearPool::new(5);
        pool.insert(1, 1.0);
        pool.insert(2, 2.0);
        pool.pop();

        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert!(!pool.has_next());
    }

    #[test]
    fn large_capacity_stress() {
        let mut pool = LinearPool::new(100);
        for i in 0..200 {
            pool.insert(i, (i as f32) * 0.5);
        }
        assert_eq!(pool.len(), 100); // capped at capacity

        // All elements should be sorted
        let result = pool.to_sorted_vec();
        for i in 0..result.len() - 1 {
            assert!(result[i].1 <= result[i + 1].1);
        }

        // Pop all and verify sorted order
        let mut prev = f32::MIN;
        while let Some((_, dist)) = pool.pop() {
            assert!(dist >= prev);
            prev = dist;
        }
    }
}
