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
    visited: crate::memory::aligned::AlignedBuf<u8>,
    /// 本次查询访问过的节点列表，用于 O(V) 重置
    history: Vec<u32>,
    /// H49：位图模式（1 bit/节点，1M = 128KB 常驻 L2）
    bits: bool,
    n: usize,
    /// H63 门卫准入（TinyLFU doorkeeper → ANN）：每节点"被已扩展节点引用次数"计数（惰性分配，默认空）
    dk: Vec<u8>,
    dk_hist: Vec<u32>,
    /// H135（第 12 轮）：L1 常驻开放寻址哈希集合（槽存 id+1，0 = 空；线性探测；按 history 清零）
    hash: bool,
    htab: Vec<u32>,
    hmask: u32,
    hshift: u32,
}

impl VisitedTracker {
    /// 创建 VisitedTracker
    ///
    /// n: 节点总数
    /// ef_search: 搜索宽度，history 按 ef_search × 256 预分配
    ///
    /// 容量公式：ef_search × max_degree_estimate。max_degree_estimate = 256
    /// 覆盖 R ≤ 256（远超实际使用范围），确保 visit_unchecked_branchless
    /// 的无检查写入不会越界。内存开销：ef=100 → 100KB，远小于 visited 数组。
    pub fn new(n: usize, ef_search: usize) -> Self {
        Self {
            visited: { let b = VIS_BITS.load(std::sync::atomic::Ordering::Relaxed); let len = if b { n.div_ceil(8) } else { n }; crate::memory::aligned::AlignedBuf::zeroed_opts(len.max(1), VIS_HUGE.load(std::sync::atomic::Ordering::Relaxed), 256 << 10).truncated(len) },
            bits: VIS_BITS.load(std::sync::atomic::Ordering::Relaxed),
            n,
            history: Vec::with_capacity(ef_search * 256),
            dk: Vec::new(),
            dk_hist: Vec::new(),
            hash: false,
            htab: Vec::new(),
            hmask: 0,
            hshift: 0,
        }
        .with_hash(VIS_HASH.load(std::sync::atomic::Ordering::Relaxed), ef_search)
    }

    fn with_hash(mut self, on: bool, ef: usize) -> Self {
        if on {
            let cap = (ef.max(8) * 96).next_power_of_two().max(1024);
            self.set_hash_cap(cap);
            self.hash = true;
        }
        self
    }

    fn set_hash_cap(&mut self, cap: usize) {
        self.htab = vec![0u32; cap];
        self.hmask = (cap - 1) as u32;
        self.hshift = 32 - cap.trailing_zeros();
    }

    #[inline(always)]
    fn hslot(&self, idx: u32) -> usize { (idx.wrapping_mul(0x9E37_79B1) >> self.hshift) as usize }

    /// H135：查找或插入；返回是否首次
    #[inline(always)]
    fn hvisit(&mut self, idx: u32) -> bool {
        let key = idx + 1;
        let mut h = self.hslot(idx);
        loop {
            let s = unsafe { *self.htab.get_unchecked(h) };
            if s == key { return false; }
            if s == 0 {
                unsafe { *self.htab.get_unchecked_mut(h) = key; }
                self.history.push(idx);
                if self.history.len() * 8 > self.htab.len() * 5 { self.hgrow(); }
                return true;
            }
            h = (h + 1) & self.hmask as usize;
        }
    }

    #[cold]
    fn hgrow(&mut self) {
        let cap = self.htab.len() * 2;
        self.set_hash_cap(cap);
        for i in 0..self.history.len() {
            let idx = self.history[i];
            let mut h = self.hslot(idx);
            while self.htab[h] != 0 { h = (h + 1) & self.hmask as usize; }
            self.htab[h] = idx + 1;
        }
    }

    #[inline(always)]
    fn hcontains(&self, idx: u32) -> bool {
        let key = idx + 1;
        let mut h = self.hslot(idx);
        loop {
            let s = unsafe { *self.htab.get_unchecked(h) };
            if s == key { return true; }
            if s == 0 { return false; }
            h = (h + 1) & self.hmask as usize;
        }
    }

    /// 标记节点为已访问
    ///
    /// 返回 true 表示首次访问，false 表示已访问过
    #[inline(always)]
    pub fn visit(&mut self, idx: u32) -> bool {
        if self.hash { return self.hvisit(idx); }
        let i = idx as usize;
        if self.bits {
            let (w, m) = (i >> 3, 1u8 << (i & 7));
            if self.visited[w] & m != 0 { return false; }
            self.visited[w] |= m;
            self.history.push(idx);
            return true;
        }
        if self.visited[i] != 0 {
            return false;
        }
        self.visited[i] = 1;
        self.history.push(idx);
        true
    }

    /// 标记节点为已访问（无边界检查，热路径专用）
    ///
    /// 返回 true 表示首次访问，false 表示已访问过
    ///
    /// SAFETY: idx 必须 < visited.len()
    #[inline(always)]
    pub unsafe fn visit_unchecked(&mut self, idx: u32) -> bool {
        if self.hash { return self.hvisit(idx); }
        let i = idx as usize;
        if self.bits { return self.visit(idx); }
        if *self.visited.get_unchecked(i) != 0 {
            return false;
        }
        *self.visited.get_unchecked_mut(i) = 1;
        self.history.push(idx);
        true
    }

    /// 标记节点为已访问（branchless，无分支预测惩罚）
    ///
    /// A8 优化：消除 visited 检查中的数据依赖分支。
    ///
    /// 图遍历中"邻居是否已访问"取决于图拓扑，本质不可预测。
    /// branchless 版本用 cmov 代替 branch：
    /// - 无条件写 visited[v] = 1
    /// - 无条件写 history[len] = v，仅当首次访问时推进 len
    /// - 消除 history.push() 内部的 capacity 检查分支
    ///
    /// SAFETY: idx 必须 < visited.len()
    ///
    /// history 容量通过 new(ef_search × 256) 保证充足，
    /// 极端情况下若仍溢出，自动 fallback 到 push（带扩容）。
    #[inline(always)]
    /// H69 oracle：撤销标记（被方向准入拒绝的邻居可再次被发现）
    #[inline(always)]
    pub fn unvisit(&mut self, idx: u32) {
        assert!(!self.hash, "unvisit unsupported in H135 hash mode");
        let i = idx as usize;
        if self.bits { self.visited[i >> 3] &= !(1u8 << (i & 7)); } else { self.visited[i] = 0; }
    }

    pub unsafe fn visit_unchecked_branchless(&mut self, idx: u32) -> bool {
        if self.hash { return self.hvisit(idx); }
        let i = idx as usize;
        let is_new = if self.bits {
            let (w, m) = (i >> 3, 1u8 << (i & 7));
            let was = *self.visited.get_unchecked(w);
            *self.visited.get_unchecked_mut(w) = was | m;
            (was & m == 0) as usize
        } else {
            let was = *self.visited.get_unchecked(i);
            *self.visited.get_unchecked_mut(i) = 1;
            (was == 0) as usize
        };
        // Branchless history push: always write, only advance if new
        let len = self.history.len();
        if len < self.history.capacity() {
            *self.history.as_mut_ptr().add(len) = idx;
            self.history.set_len(len + is_new);
        } else if is_new == 1 {
            // 极端溢出保护：fallback 到 push（自动扩容）
            self.history.push(idx);
        }
        is_new == 1
    }

    /// R27 ZA 512 oracle：位图/字节 visited 测试置位，但不写 history（调用方负责按 id 清位，见 clear_ids）
    #[inline(always)]
    pub unsafe fn visit_nohist(&mut self, idx: u32) -> bool {
        let i = idx as usize;
        if self.bits {
            let (w, m) = (i >> 3, 1u8 << (i & 7));
            let was = *self.visited.get_unchecked(w);
            *self.visited.get_unchecked_mut(w) = was | m;
            was & m == 0
        } else {
            let was = *self.visited.get_unchecked(i);
            *self.visited.get_unchecked_mut(i) = 1;
            was == 0
        }
    }

    /// R27 ZA 512：按 id 清位（不经 history）
    pub fn clear_ids(&mut self, ids: &[u32]) {
        if self.hash { return; }
        for &idx in ids { if self.bits { self.visited[(idx >> 3) as usize] = 0; } else { self.visited[idx as usize] = 0; } }
    }

    /// H63：引用计数 +w（饱和），返回加之后的计数
    #[inline(always)]
    pub fn dk_add(&mut self, idx: u32, w: u8) -> u8 {
        if self.dk.is_empty() { self.dk = vec![0u8; self.n.max(1)]; }
        let c = unsafe { self.dk.get_unchecked_mut(idx as usize) };
        if *c == 0 { self.dk_hist.push(idx); }
        *c = c.saturating_add(w);
        *c
    }

    /// 检查节点是否已访问（不标记）
    #[inline(always)]
    pub fn is_visited(&self, idx: u32) -> bool {
        if self.hash { return self.hcontains(idx); }
        if self.bits { let i = idx as usize; return self.visited[i >> 3] & (1u8 << (i & 7)) != 0; }
        self.visited[idx as usize] != 0
    }

    /// 预取节点对应的 visited 字节到 L1 cache
    ///
    /// visited 数组大小 = N bytes（1M 节点 = 1MB），超出 L2 cache。
    /// 搜索时每个邻居的 visited 检查是随机访问，可能 cache miss。
    /// 在第一遍收集未访问邻居前，提前 po 步预取，隐藏内存延迟。
    #[inline(always)]
    /// R26：节点 idx 的 visited 标记所在地址（哈希模式返回表首地址）
    pub fn addr_of(&self, idx: u32) -> usize {
        if self.hash { return self.htab.as_ptr() as usize; }
        let off = if self.bits { (idx >> 3) as usize } else { idx as usize };
        self.visited.as_ptr() as usize + off
    }

    pub unsafe fn prefetch(&self, idx: u32) {
        if self.hash { return; } // H135：哈希表常驻 L1，无需预取
        let off = if self.bits { (idx >> 3) as usize } else { idx as usize };
        let ptr = self.visited.as_ptr().add(off) as *const i8;
        crate::arch::prefetch_l1(ptr);
    }

    /// 重置：只清零本次查询访问过的节点
    ///
    /// 设计文档：把重置开销从 O(N) 降为 O(V<=1000)
    #[inline]
    pub fn reset(&mut self) {
        if self.hash {
            // 按 history 重新探测清槽（只清本查询写过的槽；探测链上的键都在 history 中，逐个清零后链仍完整，因为清零按插入顺序重探测）
            if self.history.len() * 16 > self.htab.len() { self.htab.iter_mut().for_each(|x| *x = 0); }
            else { for i in 0..self.history.len() { let idx = self.history[i]; let mut h = self.hslot(idx); while self.htab[h] != idx + 1 { h = (h + 1) & self.hmask as usize; } self.htab[h] = 0; } }
            self.history.clear();
            return;
        }
        if self.bits {
            for &idx in &self.history { self.visited[(idx >> 3) as usize] = 0; }
        } else {
            for &idx in &self.history {
                self.visited[idx as usize] = 0;
            }
        }
        self.history.clear();
        if !self.dk_hist.is_empty() { for &idx in &self.dk_hist { self.dk[idx as usize] = 0; } self.dk_hist.clear(); }
    }

    /// 本次查询已访问的节点数
    #[inline]
    pub fn visited_count(&self) -> usize {
        self.history.len()
    }

    /// 获取所有已访问节点的 ID 列表
    ///
    /// Vamana/DiskANN 论文要求：BuildVamana 用 GreedySearch 返回的整个 visited set
    /// 做 RobustPrune，而不是只用 top-L。此方法暴露 history 供建图使用。
    pub fn visited_nodes(&self) -> &[u32] {
        &self.history
    }

    /// 节点总数
    pub fn len(&self) -> usize {
        self.n
    }

    /// 是否为空（无节点）
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// history 容量（诊断接口，用于验证预分配是否充足）
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
    fn history_preallocated_by_ef_search_times_256() {
        let t = VisitedTracker::new(1000, 200);
        assert_eq!(t.history_capacity(), 200 * 256); // 410acea 已将容量改为 ef×256，测试未同步
    }

    #[test]
    fn hash_mode_matches_bitmap() {
        VIS_HASH.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut h = VisitedTracker::new(1 << 20, 8);
        VIS_HASH.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut b = VisitedTracker::new(1 << 20, 8);
        assert!(h.hash && !b.hash);
        let mut x: u64 = 77;
        for round in 0..5 {
            for _ in 0..(3000 * (round + 1)) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let id = (x % (1 << 20)) as u32;
                assert_eq!(h.visit(id), b.visit(id));
                assert_eq!(h.is_visited(id), b.is_visited(id));
            }
            assert_eq!(h.visited_nodes(), b.visited_nodes());
            h.reset(); b.reset();
            assert!(h.htab.iter().all(|&s| s == 0));
        }
    }

    #[test]
    fn is_visited_check() {
        let mut t = VisitedTracker::new(100, 64);
        assert!(!t.is_visited(5));
        t.visit(5);
        assert!(t.is_visited(5));
    }
}

/// H47b：visited 数组走 2MB 大页（RAVEN_THP=1 或此开关；AB knob vthp）
pub static VIS_HUGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// H49：visited 位图模式（RAVEN_VBITS=1；AB knob vbits），在 VisitedTracker::new 时读取
pub static VIS_BITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// H135（第 12 轮）：visited 走 L1 常驻开放寻址哈希集合（RAVEN_VHASH=1；AB knob vhash），在 VisitedTracker::new 时读取
pub static VIS_HASH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
