//! Hybrid Blocked-CSR 图存储
//!
//! 设计文档第二层：
//! 主块：每个节点固定 R_max 个槽（64 字节 cache-line 对齐）
//!       邻居起点 = node_id × block_size，无需 offset 数组
//!       空槽填充 sentinel = u32::MAX，加载时可快速跳过
//!
//! overflow 区：度数超 R_max 的极少数节点指向溢出链
//!              不浪费主存，不影响常规节点性能
//!
//! 建图阶段增加度数分位数统计钩子（设计文档第二层 log_degree_distribution）

use crate::SENTINEL;

/// Hybrid Blocked-CSR 图存储
///
/// 设计文档原文：
/// - 主块：每个节点固定 R_max 个槽，邻居起点 = node_id × block_size
/// - 空槽填充 sentinel = u32::MAX
/// - overflow 区：度数超 R_max 的极少数节点指向溢出链
#[derive(Debug, Clone)]
pub struct HybridBlockedCsr {
    /// 节点数
    n: usize,
    /// 每个节点的最大出度（主块槽位数）
    r_max: usize,
    /// 主块：n × r_max 的连续数组，空槽填 SENTINEL
    /// 邻居起点 = node_id × r_max，无需 offset 数组
    main_block: Vec<u32>,
    /// overflow 区：度数超 R_max 的节点的额外邻居
    /// key = node_id, value = 溢出的邻居列表
    overflow: Vec<Vec<u32>>,
    /// 每个节点在主块中的实际邻居数（不含 overflow）
    ///
    /// O(1) neighbors() 切片：消除 SENTINEL 线性扫描。
    /// 每次 neighbors() 调用从 O(r_max) 线性扫描降为 O(1) 数组查找。
    /// r_max=32 时，每查询 ~1200 次 neighbors() 调用节省 ~38K 次比较。
    /// 同时 add_edge() 去重扫描从 O(r_max) 降为 O(degree)。
    degrees: Vec<u16>,
}

impl HybridBlockedCsr {
    /// 创建图存储
    ///
    /// n: 节点数
    /// r_max: 每个节点主块的最大出度
    pub fn new(n: usize, r_max: usize) -> Self {
        debug_assert!(r_max <= u16::MAX as usize, "r_max exceeds u16::MAX");
        Self {
            n,
            r_max,
            main_block: vec![SENTINEL; n * r_max],
            overflow: vec![Vec::new(); n],
            degrees: vec![0u16; n],
        }
    }

    /// 节点数
    pub fn len(&self) -> usize {
        self.n
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// 每个节点的最大出度
    pub fn r_max(&self) -> usize {
        self.r_max
    }

    /// 获取节点 node_id 的邻居切片（主块部分）
    ///
    /// 邻居起点 = node_id × r_max，无需 offset 数组。
    /// 使用 degrees 数组直接切片，O(1) 复杂度（无需 SENTINEL 线性扫描）。
    #[inline(always)]
    pub fn neighbors(&self, node_id: u32) -> &[u32] {
        let start = node_id as usize * self.r_max;
        let len = self.degrees[node_id as usize] as usize;
        &self.main_block[start..start + len]
    }

    /// 预取节点 node_id 的邻居列表（OPT-2: 方案 B 预取策略）
    ///
    /// 只发出 prefetch hint，不访问数据，不触发 cache miss。
    /// 用于图搜索热路径：在 pop 候选节点前，预取堆顶节点的邻居列表，
    /// 让下一步访问 neighbors() 时数据已在 cache 中。
    ///
    /// 实测比"循环内预取下一个邻居的向量"（方案 A）快 28%，
    /// 因为方案 A 的循环内预取指令开销超过收益（SIFT1M + 随机图微基准）。
    #[inline(always)]
    pub fn prefetch_neighbors(&self, node_id: u32) {
        let start = node_id as usize * self.r_max;
        let ptr = self.main_block.as_ptr().wrapping_add(start) as *const i8;
        unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
    }

    /// 获取节点 node_id 的所有邻居（主块 + overflow）
    ///
    /// 返回主块邻居切片和 overflow 邻居切片
    #[inline]
    pub fn neighbors_full(&self, node_id: u32) -> (&[u32], &[u32]) {
        (self.neighbors(node_id), &self.overflow[node_id as usize])
    }

    /// 添加一条有向边 from -> to（自动去重）
    ///
    /// 如果主块未满，写入主块；否则写入 overflow 区。
    /// 去重：若 from -> to 已存在则跳过（DiskANN AdjacencyList 语义）。
    ///
    /// BUG FIX (v6.5): 原实现不去重，connect_bidirectional 反复添加反向边，
    /// 导致度数虚高、触发不必要的重剪枝、图结构退化。
    pub fn add_edge(&mut self, from: u32, to: u32) {
        debug_assert!(to != SENTINEL, "cannot add edge to sentinel");
        let start = from as usize * self.r_max;
        let deg = self.degrees[from as usize] as usize;

        // 去重：只扫描有效条目（O(deg) 而非 O(r_max)）
        for i in 0..deg {
            if self.main_block[start + i] == to {
                return; // 已存在，跳过
            }
        }

        // 主块有空槽，直接写入（槽位 [deg] 一定是 SENTINEL，由不变式保证）
        if deg < self.r_max {
            self.main_block[start + deg] = to;
            self.degrees[from as usize] += 1;
            return;
        }

        // 主块已满，检查 overflow
        if self.overflow[from as usize].contains(&to) {
            return;
        }
        self.overflow[from as usize].push(to);
    }

    /// 设置节点的邻居列表（替换主块，清空 overflow）
    ///
    /// 如果邻居数超过 r_max，主块填满后剩余写入 overflow
    pub fn set_neighbors(&mut self, node_id: u32, neighbors: &[u32]) {
        let start = node_id as usize * self.r_max;
        let old_deg = self.degrees[node_id as usize] as usize;

        // 清空旧主块条目（只需清 old_deg 个，剩余槽位已由不变式保证为 SENTINEL）
        for i in 0..old_deg {
            self.main_block[start + i] = SENTINEL;
        }
        // 清空 overflow
        self.overflow[node_id as usize].clear();

        // 填入新邻居
        let main_count = neighbors.len().min(self.r_max);
        self.main_block[start..start + main_count].copy_from_slice(&neighbors[..main_count]);
        // 超出部分写入 overflow
        if neighbors.len() > self.r_max {
            self.overflow[node_id as usize].extend_from_slice(&neighbors[self.r_max..]);
        }

        // 更新度数
        self.degrees[node_id as usize] = main_count as u16;
    }

    /// 获取节点的出度（主块 + overflow）
    #[inline]
    pub fn degree(&self, node_id: u32) -> usize {
        self.degrees[node_id as usize] as usize + self.overflow[node_id as usize].len()
    }

    /// 度数分位数统计钩子
    ///
    /// 设计文档第二层 log_degree_distribution：
    /// - overflow_ratio: 度数超 R_max 的节点比例
    /// - p95_degree: 95 分位数出度
    /// - p99_degree: 99 分位数出度
    pub fn log_degree_distribution(&self) -> DegreeStats {
        let mut degrees: Vec<usize> = (0..self.n).map(|i| self.degree(i as u32)).collect();
        let overflow_count = degrees.iter().filter(|&&d| d > self.r_max).count();

        degrees.sort_unstable();
        let p95 = percentile(&degrees, 0.95);
        let p99 = percentile(&degrees, 0.99);
        let mean = degrees.iter().sum::<usize>() as f64 / self.n.max(1) as f64;
        let max_degree = degrees.last().copied().unwrap_or(0);
        let isolated = degrees.iter().filter(|&&d| d == 0).count();

        tracing::info!(
            overflow_ratio = overflow_count as f64 / self.n.max(1) as f64,
            p95_degree = p95,
            p99_degree = p99,
            mean_degree = mean,
            max_degree = max_degree,
            isolated_nodes = isolated,
            "graph degree distribution"
        );

        DegreeStats {
            overflow_count,
            overflow_ratio: overflow_count as f64 / self.n.max(1) as f64,
            p95_degree: p95,
            p99_degree: p99,
            mean_degree: mean,
            max_degree,
            isolated_nodes: isolated,
        }
    }

    /// 主块内存占用（字节）
    pub fn main_block_bytes(&self) -> usize {
        self.main_block.len() * std::mem::size_of::<u32>()
    }

    /// overflow 区内存占用（字节）
    pub fn overflow_bytes(&self) -> usize {
        self.overflow.iter().map(|v| v.len() * std::mem::size_of::<u32>()).sum()
    }

    /// 从已有数据构造（用于反序列化）
    ///
    /// 参数须满足：main_block.len() == n * r_max, overflow.len() == n
    pub fn from_parts(
        n: usize,
        r_max: usize,
        main_block: Vec<u32>,
        overflow: Vec<Vec<u32>>,
    ) -> Self {
        debug_assert_eq!(main_block.len(), n * r_max, "main_block size mismatch");
        debug_assert_eq!(overflow.len(), n, "overflow size mismatch");
        debug_assert!(r_max <= u16::MAX as usize, "r_max exceeds u16::MAX");

        // 从 main_block 一次性计算 degrees（O(n*r_max)，仅加载时执行一次）
        let degrees: Vec<u16> = (0..n)
            .map(|i| {
                let start = i * r_max;
                let slice = &main_block[start..start + r_max];
                slice.iter().position(|&x| x == SENTINEL).unwrap_or(r_max) as u16
            })
            .collect();

        Self { n, r_max, main_block, overflow, degrees }
    }

    /// 导出主块引用（用于序列化）
    pub fn main_block(&self) -> &[u32] {
        &self.main_block
    }

    /// 导出 overflow 引用（用于序列化）
    pub fn overflow(&self) -> &[Vec<u32>] {
        &self.overflow
    }
}

/// 度数统计结果
#[derive(Debug, Clone)]
pub struct DegreeStats {
    /// 度数超 R_max 的节点数
    pub overflow_count: usize,
    /// 度数超 R_max 的节点比例
    pub overflow_ratio: f64,
    /// 95 分位数出度
    pub p95_degree: usize,
    /// 99 分位数出度
    pub p99_degree: usize,
    /// 平均出度
    pub mean_degree: f64,
    /// 最大出度
    pub max_degree: usize,
    /// 孤立节点数（出度为 0）
    pub isolated_nodes: usize,
}

/// 计算已排序数组的分位数
fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// 图存储统一接口（用于序列化与跨实现切换）
pub trait GraphStorage: Send + Sync {
    /// 节点数
    fn len(&self) -> usize;
    /// 是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// 获取节点邻居
    fn neighbors(&self, node_id: u32) -> &[u32];
    /// 获取节点出度
    fn degree(&self, node_id: u32) -> usize {
        self.neighbors(node_id).len()
    }
}

impl GraphStorage for HybridBlockedCsr {
    fn len(&self) -> usize {
        self.n
    }
    fn neighbors(&self, node_id: u32) -> &[u32] {
        HybridBlockedCsr::neighbors(self, node_id)
    }
}

/// HybridBlockedCsr 序列化格式（不含文件头，文件头由 VamanaGraph 统一管理）
///
/// 布局（little-endian）：
///   [0..8)   n: u64
///   [8..16)  r_max: u64
///   [16..24) main_block_len: u64 (= n * r_max)
///   [24..24+main_block_len*4)  main_block: [u32]
///   [..+8)   overflow_nonempty_count: u64
///   对每个非空 overflow 条目：
///     [4 字节] node_id: u32
///     [8 字节] len: u64
///     [len*4 字节] data: [u32]
impl super::serialize::Serializable for HybridBlockedCsr {
    fn serialize(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(
            24 + self.main_block.len() * 4 + 16 + self.overflow_bytes() + 16,
        );
        // n, r_max, main_block_len
        buf.extend_from_slice(&(self.n as u64).to_le_bytes());
        buf.extend_from_slice(&(self.r_max as u64).to_le_bytes());
        buf.extend_from_slice(&(self.main_block.len() as u64).to_le_bytes());
        // main_block（u32 数组，直接按字节拷贝）
        let mb_bytes: &[u8] = bytemuck::cast_slice(&self.main_block);
        buf.extend_from_slice(mb_bytes);
        // overflow：只存非空条目以节省空间
        let nonempty: Vec<(usize, &Vec<u32>)> = self.overflow.iter().enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect();
        buf.extend_from_slice(&(nonempty.len() as u64).to_le_bytes());
        for (node_id, vec) in nonempty {
            buf.extend_from_slice(&(node_id as u32).to_le_bytes());
            buf.extend_from_slice(&(vec.len() as u64).to_le_bytes());
            let v_bytes: &[u8] = bytemuck::cast_slice(vec);
            buf.extend_from_slice(v_bytes);
        }
        buf
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, super::serialize::SerializeError> {
        use std::convert::TryInto;
        if bytes.len() < 24 {
            return Err(super::serialize::SerializeError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HybridBlockedCsr body too short",
            )));
        }
        let n = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
        let r_max = u64::from_le_bytes(bytes[8..16].try_into()?) as usize;
        let mb_len = u64::from_le_bytes(bytes[16..24].try_into()?) as usize;
        let mb_bytes_needed = mb_len * 4;
        if bytes.len() < 24 + mb_bytes_needed + 8 {
            return Err(super::serialize::SerializeError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "main_block truncated",
            )));
        }
        let main_block: Vec<u32> = bytemuck::cast_slice(
            &bytes[24..24 + mb_bytes_needed],
        ).to_vec();

        let mut off = 24 + mb_bytes_needed;
        let ov_count = u64::from_le_bytes(bytes[off..off+8].try_into()?) as usize;
        off += 8;

        let mut overflow: Vec<Vec<u32>> = vec![Vec::new(); n];
        for _ in 0..ov_count {
            if off + 4 + 8 > bytes.len() {
                return Err(super::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "overflow entry header truncated",
                )));
            }
            let node_id = u32::from_le_bytes(bytes[off..off+4].try_into()?) as usize;
            off += 4;
            let v_len = u64::from_le_bytes(bytes[off..off+8].try_into()?) as usize;
            off += 8;
            if off + v_len * 4 > bytes.len() {
                return Err(super::serialize::SerializeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "overflow entry data truncated",
                )));
            }
            let data: Vec<u32> = bytemuck::cast_slice(&bytes[off..off + v_len * 4]).to_vec();
            off += v_len * 4;
            if node_id < n {
                overflow[node_id] = data;
            }
        }

        Ok(Self::from_parts(n, r_max, main_block, overflow))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_graph_all_sentinel() {
        let g = HybridBlockedCsr::new(10, 4);
        assert_eq!(g.len(), 10);
        assert_eq!(g.r_max(), 4);
        // 所有槽位应为 SENTINEL
        for i in 0..10 {
            assert!(g.neighbors(i).is_empty());
        }
    }

    #[test]
    fn add_edge_to_main_block() {
        let mut g = HybridBlockedCsr::new(10, 4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        assert_eq!(g.neighbors(0), &[1, 2]);
        assert_eq!(g.degree(0), 2);
    }

    #[test]
    fn add_edge_overflow() {
        let mut g = HybridBlockedCsr::new(10, 2);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3); // overflow
        assert_eq!(g.degree(0), 3);
        let (main, overflow) = g.neighbors_full(0);
        assert_eq!(main, &[1, 2]);
        assert_eq!(overflow, &[3]);
    }

    #[test]
    fn set_neighbors_replaces() {
        let mut g = HybridBlockedCsr::new(10, 3);
        g.add_edge(0, 1);
        g.set_neighbors(0, &[5, 6, 7]);
        assert_eq!(g.neighbors(0), &[5, 6, 7]);
    }

    #[test]
    fn set_neighbors_overflow() {
        let mut g = HybridBlockedCsr::new(10, 2);
        g.set_neighbors(0, &[1, 2, 3, 4, 5]);
        assert_eq!(g.degree(0), 5);
        let (main, overflow) = g.neighbors_full(0);
        assert_eq!(main, &[1, 2]);
        assert_eq!(overflow, &[3, 4, 5]);
    }

    #[test]
    fn degree_stats_basic() {
        let mut g = HybridBlockedCsr::new(10, 3);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 2);
        // 节点 2-9 度数为 0
        let stats = g.log_degree_distribution();
        assert_eq!(stats.overflow_count, 0);
        assert_eq!(stats.isolated_nodes, 8);
    }

    #[test]
    fn sentinel_skipped_in_neighbors() {
        let mut g = HybridBlockedCsr::new(10, 4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        // 中间不应有 SENTINEL
        let n = g.neighbors(0);
        assert_eq!(n, &[1, 2]);
    }
}
