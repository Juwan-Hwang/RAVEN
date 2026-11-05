//! Benchmark 工具函数 — 供 `src/bin/` 下所有 benchmark 二进制共享
//!
//! 提取自 50+ 个 benchmark 中重复的 read_fvecs / read_ivecs / compute_recall / median3，
//! 消除 copy-paste 代码债务。

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;

use crate::graph::navigation::LayeredNavigation;
use crate::graph::VamanaGraph;
use crate::memory::graph::HybridBlockedCsr;
use crate::quant::SQ4Dataset;
use crate::SENTINEL;

/// SIFT-1M 数据集（train/test/gt），已归一化（÷255）
pub struct SiftDataset {
    /// 训练集向量（扁平存储，已归一化）
    pub train: Vec<f32>,
    /// 查询集向量（扁平存储，已归一化）
    pub test: Vec<f32>,
    /// groundtruth ID（扁平存储）
    pub gt: Vec<i32>,
    /// 向量维度
    pub dim: usize,
    /// 训练集大小
    pub n: usize,
    /// 查询集大小
    pub nq: usize,
    /// groundtruth k（每条查询的真值数）
    pub gt_k: usize,
}

impl SiftDataset {
    /// 从 `data/sift/` 加载并归一化（SIFT uint8 → f32 ÷ 255）
    pub fn load() -> Self {
        let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
        let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
        let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

        for v in &mut train {
            *v /= 255.0;
        }
        for v in &mut test {
            *v /= 255.0;
        }

        Self { train, test, gt, dim, n, nq, gt_k }
    }
}

/// 读取 fvecs 文件 → (扁平向量, dim, n)
pub fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
    }
    (vectors, dim, n)
}

/// 读取 ivecs 文件 → (扁平数组, dim, n)
pub fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("open ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read ivecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            gt.push(i32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

/// 计算 recall@k
pub fn compute_recall(results: &[Vec<u32>], gt: &[i32], gt_k: usize, k: usize, nq: usize) -> f64 {
    let mut hits = 0;
    for q in 0..nq {
        let found = &results[q];
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

/// 三值取中位数（1 warmup + 3 rounds 去抖）
pub fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
}

// ─── BFS 图重排 ───────────────────────────────────────────────────
//
// 从 entry point 做 BFS 遍历，将逻辑上邻近的节点重排到物理上连续的内存位置，
// 提升 cache 局部性。重排是纯后处理：不改变图拓扑，不改变搜索结果，
// 只改变节点在内存中的存储顺序。
//
// 验证结论（reorder_verify）：
//   - 20 个随机 query 的 top-10 结果 ID 集合完全一致
//   - 200 对距离全部一致（<0.01 浮动）
//   - recall 不变

/// BFS 重排后的完整图 + 数据包
pub struct ReorderedGraph {
    /// 重排后的图（新 ID 空间）
    pub graph: VamanaGraph,
    /// 重排后的 SQ4 编码（新 ID 空间）
    pub sq4: SQ4Dataset,
    /// 重排后的训练向量（新 ID 空间）
    pub train: Vec<f32>,
    /// 新 ID → 原始 ID 逆映射（搜索结果翻译回原始 ID 空间）
    pub new_to_old: Vec<u32>,
}

/// 对已建好的图执行 BFS 重排
///
/// 返回重排后的图 + SQ4 + 向量 + 逆映射。
/// 搜索时用重排后的 graph/train/sq4，返回的结果 ID 用 `new_to_old` 翻译回原始空间。
pub fn bfs_reorder(
    graph: &VamanaGraph,
    sq4: &SQ4Dataset,
    train: &[f32],
    dim: usize,
    nav_m: usize,
    nav_k: usize,
) -> ReorderedGraph {
    let entry = graph.entry_point();
    let n = graph.len();

    // 1. BFS 遍历生成 old_to_new 映射
    let old_to_new = bfs_order(graph, entry, n);

    // 2. 重排图存储
    let new_storage = remap_graph(graph, &old_to_new);
    let new_sq4 = remap_sq4(sq4, &old_to_new);
    let new_train = remap_vectors(train, dim, &old_to_new);
    let new_entry = old_to_new[entry as usize];

    // 3. 逆映射
    let mut new_to_old = vec![0u32; n];
    for old_id in 0..n {
        new_to_old[old_to_new[old_id] as usize] = old_id as u32;
    }

    // 4. 重建分层导航
    let new_nav =
        LayeredNavigation::build(&new_train, dim, &new_storage, new_entry, nav_m, nav_k);
    let mut new_graph = VamanaGraph::from_storage(new_storage, new_entry, dim);
    new_graph.set_layered_nav(new_nav);

    ReorderedGraph {
        graph: new_graph,
        sq4: new_sq4,
        train: new_train,
        new_to_old,
    }
}


/// H02：按给定 old_to_new 重排（导航层用 remapped 保持等价，不重建）。
pub fn reorder_with(graph: &VamanaGraph, sq4: &SQ4Dataset, train: &[f32], dim: usize, old_to_new: &[u32]) -> ReorderedGraph {
    let n = graph.len();
    let new_storage = remap_graph(graph, old_to_new);
    let new_sq4 = remap_sq4(sq4, old_to_new);
    let new_train = remap_vectors(train, dim, old_to_new);
    let new_entry = old_to_new[graph.entry_point() as usize];
    let mut new_to_old = vec![0u32; n];
    for o in 0..n { new_to_old[old_to_new[o] as usize] = o as u32; }
    let mut g = VamanaGraph::from_storage(new_storage, new_entry, dim);
    if let Some(nav) = graph.layered_nav() { g.set_layered_nav(nav.remapped(old_to_new)); }
    ReorderedGraph { graph: g, sq4: new_sq4, train: new_train, new_to_old }
}

/// H02：BFS 序（公开版）
pub fn bfs_order_pub(graph: &VamanaGraph) -> Vec<u32> { bfs_order(graph, graph.entry_point(), graph.len()) }

/// H02：按导航层级降序 + BFS（上层节点集中在头部，导航访存局部）
pub fn level_bfs_order(graph: &VamanaGraph) -> Vec<u32> {
    let n = graph.len();
    let bfs = bfs_order(graph, graph.entry_point(), n);
    let lv: Vec<u8> = match graph.layered_nav() { Some(nav) => (0..n as u32).map(|i| nav.level_of(i)).collect(), None => vec![0; n] };
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|&a, &b| lv[b as usize].cmp(&lv[a as usize]).then(bfs[a as usize].cmp(&bfs[b as usize])));
    let mut o2n = vec![0u32; n];
    for (nw, &o) in idx.iter().enumerate() { o2n[o as usize] = nw as u32; }
    o2n
}

/// 将搜索结果中的新 ID 翻译回原始 ID（原地修改）
pub fn translate_results(results: &mut [Vec<u32>], new_to_old: &[u32]) {
    for r in results.iter_mut() {
        for id in r.iter_mut() {
            *id = new_to_old[*id as usize];
        }
    }
}

// ─── 内部辅助函数 ─────────────────────────────────────────────────

fn bfs_order(graph: &VamanaGraph, entry: u32, n: usize) -> Vec<u32> {
    let mut old_to_new = vec![u32::MAX; n];
    let mut queue = VecDeque::new();
    let mut next_id = 0u32;

    old_to_new[entry as usize] = next_id;
    next_id += 1;
    queue.push_back(entry);

    while let Some(node) = queue.pop_front() {
        for &neighbor in graph.neighbors(node) {
            if neighbor == SENTINEL {
                continue;
            }
            let idx = neighbor as usize;
            if old_to_new[idx] == u32::MAX {
                old_to_new[idx] = next_id;
                next_id += 1;
                queue.push_back(neighbor);
            }
        }
    }

    // 孤立节点（BFS 不可达）追加到末尾
    for i in 0..n {
        if old_to_new[i] == u32::MAX {
            old_to_new[i] = next_id;
            next_id += 1;
        }
    }
    old_to_new
}

fn remap_graph(graph: &VamanaGraph, old_to_new: &[u32]) -> HybridBlockedCsr {
    let n = graph.len();
    let r_max = graph.storage().r_max();
    let mut new_storage = HybridBlockedCsr::new(n, r_max);
    for old_id in 0..n as u32 {
        let new_id = old_to_new[old_id as usize];
        let neighbors = graph.neighbors(old_id);
        let remapped: Vec<u32> = neighbors
            .iter()
            .map(|&old_n| old_to_new[old_n as usize])
            .collect();
        new_storage.set_neighbors(new_id, &remapped);
    }
    new_storage
}

pub fn remap_sq4(sq4: &SQ4Dataset, old_to_new: &[u32]) -> SQ4Dataset {
    let n = sq4.n;
    let bytes_per_vec = sq4.code_bytes;
    let mut new_codes = vec![0u8; sq4.codes.len()];
    for old_id in 0..n {
        let new_id = old_to_new[old_id] as usize;
        let old_start = old_id * bytes_per_vec;
        let new_start = new_id * bytes_per_vec;
        new_codes[new_start..new_start + bytes_per_vec]
            .copy_from_slice(&sq4.codes[old_start..old_start + bytes_per_vec]);
    }
    SQ4Dataset {
        codes: new_codes.into(),
        params: sq4.params.clone(),
        dim: sq4.dim,
        n,
        code_bytes: bytes_per_vec,
    }
}

fn remap_vectors(train: &[f32], dim: usize, old_to_new: &[u32]) -> Vec<f32> {
    let n = old_to_new.len();
    let mut new_train = vec![0.0f32; n * dim];
    for old_id in 0..n {
        let new_id = old_to_new[old_id] as usize;
        new_train[new_id * dim..(new_id + 1) * dim]
            .copy_from_slice(&train[old_id * dim..(old_id + 1) * dim]);
    }
    new_train
}
