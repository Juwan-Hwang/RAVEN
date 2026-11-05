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
/// R28 S2：幸存者 ≤ 此数时逐个 insert_dedup（批量排序/归并的固定开销不划算）
pub const S2_SEQ_MAX: usize = 2;
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
            data: vec![(0, 0.0); ef + 4], // +4：H130 SIMD 计数按 4 对读取的尾部余量
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
        if ef + 4 > self.data.len() {
            self.data.resize(ef + 4, (0, 0.0));
        }
        self.capacity = ef;
        self.ef = ef;
        self.size = 0;
        self.cursor = 0;
    }

    /// 中途扩容：增大 capacity 和 ef，保留现有数据与游标
    ///
    /// 用于运行时自适应 ef：搜索进行中判定 query 难度后，
    /// 不重置池、不丢失已收集的候选，直接扩大池容量继续搜索。
    ///
    /// 扩容后：
    /// - `capacity` 和 `ef` 更新为 `new_ef`
    /// - `size`、`cursor`、`data` 内容全部不变
    /// - 后续 `insert` 用新容量判断是否满（更宽的接受阈值）
    /// - 后续 `pop` 用新 ef 判断是否终止（搜索更多元素）
    ///
    /// 注意：扩容前被旧 capacity 拒绝的候选不可恢复。
    /// 这部分丢失的候选对 recall 的影响需通过实验验证。
    #[inline]
    pub fn expand_ef(&mut self, new_ef: usize) {
        debug_assert!(new_ef >= self.capacity,
            "expand_ef: new_ef ({}) must >= current capacity ({})", new_ef, self.capacity);
        if new_ef + 4 > self.data.len() {
            self.data.resize(new_ef + 4, (0, 0.0));
        }
        self.capacity = new_ef;
        self.ef = new_ef;
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

    /// R27 S1（H153 无 visited 搜索，RAVEN_NOVIS）：带池内去重的插入。语义 = 若 `node` 已在池中（含已扩展项）则拒绝，
    /// 否则同 [`insert`]。同一节点每次被提供时距离相同（SQ4 整数累加 → f32，逐位确定），故若它在池中，必位于
    /// "插入位置之前紧邻的同距离段"内：只需从插入位置向前扫描 `d == dist` 的元素比较 id（通常 0–1 次比较）。
    /// 池满且 `dist >= worst` 的拒绝在去重之前（与 insert 一致，不改变池）。
    #[inline]
    pub fn insert_dedup(&mut self, node: u32, dist: f32) -> bool {
        if self.size == self.capacity {
            let worst = unsafe { self.data.get_unchecked(self.size - 1).1 };
            if dist >= worst { return false; }
        }
        let lo = self.find_insert_pos(dist);
        let mut j = lo;
        while j > 0 {
            let e = unsafe { *self.data.get_unchecked(j - 1) };
            if e.1 != dist { break; }
            if raw_id(e.0) == node { return false; }
            j -= 1;
        }
        let move_count = self.size - lo;
        if move_count > 0 {
            unsafe { std::ptr::copy(self.data.as_ptr().add(lo), self.data.as_mut_ptr().add(lo + 1), move_count); }
        }
        unsafe { *self.data.as_mut_ptr().add(lo) = (node, dist); }
        if self.size < self.capacity { self.size += 1; }
        if lo < self.cursor { self.cursor = lo; }
        true
    }

    /// R28 S2（H152 批量归并池，建在 S1 无 visited 之上，RAVEN_S2）：一跳的全部候选 (ids[i], ds[i]) 一次并入池，
    /// 结果（池内容、并列次序、游标）与按原序逐个调用 [`insert_dedup`] 逐位一致：
    /// (1) 跳初门槛 w（池满时 = worst，否则 +inf）：d >= w 者逐个插入时也必被拒（worst 单调不增）→ 先压缩幸存者；
    /// (2) 幸存者按 (d, 原序) 稳定排序；逐个插入的并列规则是"新元素排在所有 d 相等的已有元素之后"，
    ///     故最终序 = (池 ∪ 批) 按 (d, 池在前, 批内原序) 的稳定归并的前 cap 个；
    /// (3) 去重：若某幸存者 id 已在池中，其池内副本 d 相同 → 在归并前用一次反向协同扫描（池指针单调）检出并剔除；
    ///     批内同 id（同距离）只保留第一个；被挤出后再出现者在逐个插入时由 `d >= worst` 拒绝，在归并中落在前 cap 之外，二者一致；
    /// (4) 游标 = min(原游标, 新元素在最终数组中的最小下标)（逐个插入时每次 lo < cursor 即回退，最小插入位置在最终数组中仍由新元素占据）。
    /// 返回实际并入的新元素个数（含其后被同批挤出者不计）。`ids.len() <= 128`。
    #[inline]
    pub fn merge_batch_dedup(&mut self, ids: &[u32], ds: &[f32]) -> usize {
        let n = ids.len();
        debug_assert!(n <= 128 && ds.len() == n);
        let w = if self.size == self.capacity { unsafe { self.data.get_unchecked(self.size - 1).1 } } else { f32::INFINITY };
        // 幸存者缓冲不初始化（每跳省 1KB 置零）；只读写前 k 个
        let mut svm: [std::mem::MaybeUninit<(f32, u32)>; 128] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let sv = svm.as_mut_ptr() as *mut (f32, u32);
        let mut k = 0usize;
        for j in 0..n {
            let d = unsafe { *ds.get_unchecked(j) };
            unsafe { *sv.add(k) = (d, *ids.get_unchecked(j)); }
            k += (d < w) as usize;
        }
        if k == 0 { return 0; }
        // 少量幸存者：逐个 insert_dedup（与批量语义相同，省去排序/归并的固定开销）
        if k <= S2_SEQ_MAX {
            let mut a = 0usize;
            for j in 0..k { let (d, id) = unsafe { *sv.add(j) }; a += self.insert_dedup(id, d) as usize; }
            return a;
        }
        let sv = unsafe { std::slice::from_raw_parts_mut(sv, k) };
        // 稳定插入排序
        for i in 1..k {
            let x = sv[i];
            let mut j = i;
            while j > 0 && sv[j - 1].0 > x.0 { sv[j] = sv[j - 1]; j -= 1; }
            sv[j] = x;
        }
        // 去重：反向协同扫描（sv 升序、池升序；池指针 i 单调下降）；重复者 id 置为 DUP 后压缩
        const DUP: u32 = u32::MAX;
        let mut i = self.size as isize - 1;
        let mut ndup = 0usize;
        for j in (0..k).rev() {
            let (d, id) = sv[j];
            while i >= 0 && unsafe { self.data.get_unchecked(i as usize).1 } > d { i -= 1; }
            let mut t = i;
            let mut dup = false;
            while t >= 0 { let e = unsafe { *self.data.get_unchecked(t as usize) }; if e.1 != d { break; } if raw_id(e.0) == id { dup = true; break; } t -= 1; }
            if !dup { let mut u = j; while u > 0 && sv[u - 1].0 == d { u -= 1; if sv[u].1 == id { dup = true; break; } } }
            if dup { sv[j].1 = DUP; ndup += 1; }
        }
        if ndup > 0 { let mut m = 0usize; for j in 0..k { if sv[j].1 != DUP { sv[m] = sv[j]; m += 1; } } k = m; if k == 0 { return 0; } }
        // 反向原地归并：先丢弃超出容量的最大者（并列时批元素在后 → 先丢批元素），再写回
        let total = self.size + k;
        let newsize = total.min(self.capacity);
        let mut pi = self.size as isize - 1;
        let mut bj = k as isize - 1;
        let mut out = total as isize - 1;
        let mut minpos = usize::MAX;
        let mut added = 0usize;
        while bj >= 0 {
            let b = sv[bj as usize];
            let take_b = pi < 0 || b.0 >= unsafe { self.data.get_unchecked(pi as usize).1 };
            if (out as usize) < newsize {
                if take_b { unsafe { *self.data.get_unchecked_mut(out as usize) = (b.1, b.0); } minpos = out as usize; added += 1; }
                else { unsafe { *self.data.get_unchecked_mut(out as usize) = *self.data.get_unchecked(pi as usize); } }
            }
            if take_b { bj -= 1; } else { pi -= 1; }
            out -= 1;
        }
        self.size = newsize;
        if minpos < self.cursor { self.cursor = minpos; }
        added
    }

    /// H130（第 12 轮）：无分支插入位置——位置 = 池中 d <= dist 的元素个数（数组升序，等价于 find_insert_pos 的
    /// 从尾向前线性扫描），NEON 每次 vld2 解交错 4 对、比较计数，无数据相关分支。语义与 `insert` 逐位一致。
    #[inline(always)]
    pub fn insert_bl(&mut self, node: u32, dist: f32) -> bool {
        if self.size == self.capacity {
            let worst = unsafe { self.data.get_unchecked(self.size - 1).1 };
            if dist >= worst { return false; }
        }
        let lo = self.count_le(dist);
        let move_count = self.size - lo;
        unsafe {
            let p = self.data.as_mut_ptr();
            // 小块右移：从尾向前逐个拷贝（≤ ef 个 8B 元素），避免 memmove 调用
            let mut j = lo + move_count;
            while j > lo { *p.add(j) = *p.add(j - 1); j -= 1; }
            *p.add(lo) = (node, dist);
        }
        if self.size < self.capacity { self.size += 1; }
        if lo < self.cursor { self.cursor = lo; }
        true
    }

    #[inline(always)]
    fn count_le(&self, dist: f32) -> usize {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let base = self.data.as_ptr() as *const f32;
            let dv = vdupq_n_f32(dist);
            let sz = vdupq_n_u32(self.size as u32);
            let iota: [u32; 4] = [0, 1, 2, 3];
            let mut idx = vld1q_u32(iota.as_ptr());
            let four = vdupq_n_u32(4);
            let mut cnt = vdupq_n_u32(0);
            let mut c = 0usize;
            while c < self.size {
                let v = vld2q_f32(base.add(2 * c)).1;
                let m = vandq_u32(vcleq_f32(v, dv), vcltq_u32(idx, sz));
                cnt = vsubq_u32(cnt, m);
                idx = vaddq_u32(idx, four);
                c += 4;
            }
            vaddvq_u32(cnt) as usize
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut n = 0usize;
            for i in 0..self.size { n += (unsafe { self.data.get_unchecked(i).1 } <= dist) as usize; }
            n
        }
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

    /// R27 ZA oracle：直接装入一个已排序的池内容（size = len，游标置末 = 全部视为已扩展）
    #[inline]
    pub fn load_sorted(&mut self, v: &[(u32, f32)]) {
        if v.len() + 4 > self.data.len() { self.data.resize(v.len() + 4, (0, 0.0)); }
        self.data[..v.len()].copy_from_slice(v);
        self.size = v.len();
        self.cursor = v.len();
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

    /// H13a：第二个待扩展元素（cursor 之后第一个未检查项）
    #[inline]
    pub fn peek_second_unchecked(&self) -> Option<u32> {
        if !self.has_next() {
            return None;
        }
        let end = self.size.min(self.ef);
        let mut i = self.cursor + 1;
        while i < end {
            let id = self.data[i].0;
            if !is_checked(id) {
                return Some(raw_id(id));
            }
            i += 1;
        }
        None
    }

    /// 池是否已满（size == capacity）。H07：满时 `worst_distance()` 即插入阈值。
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
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

    /// 第 k 近的距离（1-based：k=1 是最近，k=size 是最远）
    ///
    /// 用于早停启发式：追踪 top-k 距离是否改善。
    /// 若 k > size，返回 f32::MAX。
    #[inline]
    pub fn kth_distance(&self, k: usize) -> f32 {
        if k == 0 || k > self.size {
            f32::MAX
        } else {
            // SAFETY: 1 <= k <= size guarantees k-1 is valid
            unsafe { self.data.get_unchecked(k - 1).1 }
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

    /// 线性查找插入位置（OPT-10：小数组优于二分查找）
    ///
    /// 从尾部向前扫描，返回第一个 distance > dist 的位置（保持升序）。
    ///
    /// 为什么线性扫描优于二分查找（ef ≤ 128）：
    /// - 数组 ≤ 128 元素 = ≤ 1024B = ≤ 16 cache lines，全部在 L1
    /// - 顺序访问 → CPU prefetcher 完美预测，每个元素 ~1 cycle
    /// - 二分查找的 mid 跳转 → 不可预测分支，每次 ~5 cycles（mispredict penalty）
    /// - 二分查找 6 次迭代 × 5 cycles = 30 cycles vs 线性平均 25 × 1 cycle = 25 cycles
    /// - 从尾部扫描的优势：满池时大多数插入距离接近 worst（尾部），1-2 次比较即可定位
    #[inline]
    /// H38：弹出并返回下标（供 `rekey_checked` 使用）
    #[inline]
    pub fn pop_idx(&mut self) -> Option<(u32, f32, usize)> {
        if !self.has_next() { return None; }
        let idx = self.cursor;
        self.data[idx].0 |= CHECKED_MASK;
        self.cursor += 1;
        while self.cursor < self.size && is_checked(self.data[self.cursor].0) { self.cursor += 1; }
        let (id, dist) = self.data[idx];
        Some((raw_id(id), dist, idx))
    }

    /// H38：把已扩展条目（下标 idx，可能已因插入后移，调用方保证仍正确）改为新键并保持升序与已扩展标记。
    #[inline]
    pub fn rekey_checked(&mut self, idx: usize, new_dist: f32) {
        let id = self.data[idx].0;
        // 删除 idx
        unsafe { std::ptr::copy(self.data.as_ptr().add(idx + 1), self.data.as_mut_ptr().add(idx), self.size - idx - 1); }
        self.size -= 1;
        if idx < self.cursor { self.cursor -= 1; }
        let pos = self.find_insert_pos(new_dist);
        unsafe {
            std::ptr::copy(self.data.as_ptr().add(pos), self.data.as_mut_ptr().add(pos + 1), self.size - pos);
            *self.data.as_mut_ptr().add(pos) = (id | CHECKED_MASK, new_dist);
        }
        self.size += 1;
        if pos <= self.cursor { self.cursor += 1; }
        while self.cursor < self.size && is_checked(self.data[self.cursor].0) { self.cursor += 1; }
    }

    fn find_insert_pos(&self, dist: f32) -> usize {
        let mut i = self.size as isize - 1;
        while i >= 0 {
            // SAFETY: i < size
            let d = unsafe { self.data.get_unchecked(i as usize).1 };
            if d <= dist {
                return (i + 1) as usize;
            }
            i -= 1;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r28_s2_merge_batch_equals_sequential_insert_dedup() {
        // R28 S2：批量归并 = 逐个 insert_dedup（随机并列距离、池内重复、批内重复、交错弹出、各种 ef）
        for ef in [1usize, 2, 3, 5, 10, 40, 41, 100, 150] {
            let (mut a, mut b) = (LinearPool::new(ef), LinearPool::new(ef));
            let mut x: u64 = 0x9E37_79B9 ^ (ef as u64 * 7919);
            let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
            for step in 0..20000u32 {
                let r = rnd();
                if r % 4 == 0 { assert_eq!(a.pop(), b.pop(), "pop ef={ef} step={step}"); continue; }
                let n = (rnd() % 40) as usize;
                let mut ids = Vec::new(); let mut ds = Vec::new();
                for _ in 0..n {
                    let id = (rnd() % 300) as u32;
                    // 同一 id 距离固定（与真实搜索一致）：d = f(id)，量化成少量取值制造并列
                    let d = ((id.wrapping_mul(2654435761) >> 20) % 50) as f32;
                    ids.push(id); ds.push(d);
                }
                // 批内同 id 至多出现在同一行一次（真实邻接无重复），但保留少量重复以覆盖批内去重
                for (id, d) in ids.iter().zip(&ds) { a.insert_dedup(*id, *d); }
                b.merge_batch_dedup(&ids, &ds);
                assert_eq!(a.size, b.size, "size ef={ef} step={step}");
                assert_eq!(a.cursor, b.cursor, "cursor ef={ef} step={step}");
                assert_eq!(&a.data[..a.size], &b.data[..b.size], "data ef={ef} step={step}");
            }
        }
    }

    #[test]
    fn insert_bl_matches_insert() {
        // H130：无分支插入与原插入逐位一致（含相等距离的并列、池满拒绝、弹出交错）
        for ef in [1usize, 3, 4, 5, 10, 40, 41, 100] {
            let (mut a, mut b) = (LinearPool::new(ef), LinearPool::new(ef));
            let mut x: u64 = 0x1234_5678 + ef as u64;
            for step in 0..5000u32 {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let d = (x % 64) as f32; // 大量并列
                assert_eq!(a.insert(step, d), b.insert_bl(step, d));
                if x % 7 == 0 { assert_eq!(a.pop(), b.pop()); }
                assert_eq!(a.raw_view(), b.raw_view());
            }
        }
    }

    #[test]
    fn insert_dedup_equals_visited_filter() {
        // R27 S1：去重插入（无 visited，同一节点可被重复提供）与"visited 过滤 + insert"逐位一致：
        // 固定容量 + 单调门槛 ⇒ 重复者要么在池内（同距离段去重），要么距离 ≥ 现门槛（直接拒绝）。含大量并列距离、弹出交错。
        for ef in [1usize, 2, 3, 5, 10, 40, 100] {
            let (mut a, mut b) = (LinearPool::new(ef), LinearPool::new(ef));
            let mut vis = std::collections::HashSet::new();
            let mut x: u64 = 0x9E37_79B9 + ef as u64;
            let dist_of = |v: u32| ((v.wrapping_mul(2654435761) >> 7) % 23) as f32; // 每节点固定距离，大量并列
            for _step in 0..20000u32 {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let v = (x % 300) as u32; // 小 id 空间 → 大量重复提供
                let d = dist_of(v);
                let ra = if vis.insert(v) { a.insert(v, d) } else { false };
                let rb = b.insert_dedup(v, d);
                assert_eq!(ra, rb);
                if x % 5 == 0 { assert_eq!(a.pop(), b.pop()); }
                assert_eq!(a.raw_view(), b.raw_view());
            }
        }
    }

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

impl LinearPool {
    /// H40：诊断用只读视图（id 含 checked 最高位）、游标、ef
    pub fn raw_view(&self) -> (&[(u32, f32)], usize, usize) {
        (&self.data[..self.size], self.cursor, self.ef)
    }
}

impl LinearPool {
    /// H50：弹出指定 id（若其仍在 [cursor, ef) 内且未扩展），否则 None。语义同 pop（标记已扩展、推进游标）。
    #[inline]
    pub fn pop_id(&mut self, id: u32) -> Option<(u32, f32)> {
        let end = self.size.min(self.ef);
        let mut i = self.cursor;
        while i < end {
            let e = self.data[i].0;
            if e == id { // 未扩展项的 id 不带最高位
                self.data[i].0 |= CHECKED_MASK;
                if i == self.cursor {
                    self.cursor += 1;
                    while self.cursor < self.size && is_checked(self.data[self.cursor].0) { self.cursor += 1; }
                }
                return Some((id, self.data[i].1));
            }
            i += 1;
        }
        None
    }
}

impl LinearPool {
    /// H54：当前游标（下一个将弹出元素的名次）
    #[inline(always)]
    pub fn cursor(&self) -> usize { self.cursor }
}

impl LinearPool {
    /// N4（第 13 轮）：把 [cursor, min(size, ef)) 内全部未扩展项标记为已扩展并写入 `out`（返回个数），游标推到末尾。
    #[inline]
    pub fn take_unexpanded(&mut self, out: &mut Vec<u32>) -> usize {
        out.clear();
        let end = self.size.min(self.ef);
        let mut i = self.cursor;
        while i < end {
            let e = self.data[i].0;
            if !is_checked(e) { out.push(e); self.data[i].0 = e | CHECKED_MASK; }
            i += 1;
        }
        self.cursor = end;
        out.len()
    }

    /// N4：以"已扩展"状态插入（参与排名与 rerank，但永不被弹出）。语义同 `insert` 的拒绝规则。
    #[inline]
    pub fn insert_checked(&mut self, node: u32, dist: f32) -> bool {
        if self.size == self.capacity {
            let worst = unsafe { self.data.get_unchecked(self.size - 1).1 };
            if dist >= worst { return false; }
        }
        let lo = self.find_insert_pos(dist);
        let move_count = self.size - lo;
        if move_count > 0 {
            unsafe { std::ptr::copy(self.data.as_ptr().add(lo), self.data.as_mut_ptr().add(lo + 1), move_count); }
        }
        unsafe { *self.data.as_mut_ptr().add(lo) = (node | CHECKED_MASK, dist); }
        if self.size < self.capacity { self.size += 1; }
        if lo <= self.cursor { self.cursor += 1; }
        if self.cursor > self.size { self.cursor = self.size; }
        true
    }
}

