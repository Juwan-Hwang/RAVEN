//! OPT-2 微基准：预取策略优化对比
//!
//! 目标：验证不同预取策略对图搜索 QPS 的影响
//!
//! 核心问题：当前预取 neighbors[i+1] 的向量数据，但下一步真正需要的是
//! candidates 堆顶节点的邻居。哪种预取策略最优？
//!
//! 实验方案：
//! - 方案 A（当前）：预取 neighbors[i+1] 的向量数据
//! - 方案 B（新）：预取 candidates 堆顶节点的 neighbors 指针
//! - 方案 C（组合）：A + B
//! - 方案 D（无预取）：删除所有 _mm_prefetch
//! - 方案 E（2-ahead）：预取 neighbors[i+1] 和 neighbors[i+2] 的向量数据
//!
//! 数据：SIFT1M base（1M × dim=128 = 493MB，超出 L3 cache）
//! 图：随机图（R=32，不建 Vamana 图，省 782s）
//! 注意：随机图 recall 很低，但预取策略不影响 recall，只测 QPS

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::l2_simd;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use rand::SeedableRng;

/// 读取 fvecs 文件
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 文件长度不对齐");

    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            vectors.push(v);
        }
    }
    (vectors, dim, n)
}

/// OrderedF32 包装（用于 BinaryHeap）
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// 生成随机邻居列表（每个节点 R 个随机邻居）
fn gen_random_graph(n: usize, r: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    use rand::seq::SliceRandom;
    let mut graph: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut indices: Vec<u32> = (0..n as u32).collect();
    for node in 0..n as u32 {
        indices.partial_shuffle(&mut rng, r + 1);
        let neighbors: Vec<u32> = indices.iter()
            .take(r + 1)
            .filter(|&&x| x != node)
            .take(r)
            .copied()
            .collect();
        graph.push(neighbors);
    }
    graph
}

/// 简单的 visited tracker
struct VisitedTracker {
    visited: Vec<u8>,
    gen: u8,
}

impl VisitedTracker {
    fn new(n: usize) -> Self {
        Self { visited: vec![0; n], gen: 1 }
    }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.visited.fill(0);
            self.gen = 1;
        }
    }
    fn visit(&mut self, node: u32) -> bool {
        let idx = node as usize;
        if self.visited[idx] == self.gen {
            false
        } else {
            self.visited[idx] = self.gen;
            true
        }
    }
}

/// 方案 A：预取 neighbors[i+1] 的向量数据（当前实现）
fn search_prefetch_next_vec(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        let neighbors = &graph[node as usize];
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // 方案 A：预取下一个邻居的向量
            if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr); }
            }
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 方案 B：预取 candidates 堆顶节点的 neighbors 指针
fn search_prefetch_heap_neighbors(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        // 方案 B：预取堆顶节点的邻居列表
        if let Some(&Reverse((_, top_node))) = candidates.peek() {
            let top_neighbors = &graph[top_node as usize];
            // 预取堆顶节点的邻居列表数据（Vec<u32> 的堆内存）
            let ptr = top_neighbors.as_ptr() as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        let neighbors = &graph[node as usize];
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 方案 C：组合 A + B
fn search_prefetch_combo(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        // 方案 C 的 B 部分：预取堆顶节点的邻居列表
        if let Some(&Reverse((_, top_node))) = candidates.peek() {
            let top_neighbors = &graph[top_node as usize];
            let ptr = top_neighbors.as_ptr() as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        let neighbors = &graph[node as usize];
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // 方案 C 的 A 部分：预取下一个邻居的向量
            if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr); }
            }
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 方案 D：无预取
fn search_no_prefetch(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        let neighbors = &graph[node as usize];
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 方案 E：2-ahead 预取（neighbors[i+1] 和 neighbors[i+2]）
fn search_prefetch_2ahead(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        let neighbors = &graph[node as usize];
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // 方案 E：预取 2-ahead
            if i + 2 < neighbors.len() {
                let next1 = neighbors[i + 1];
                let next2 = neighbors[i + 2];
                let ptr1 = vectors.as_ptr().wrapping_add(next1 as usize * dim) as *const i8;
                let ptr2 = vectors.as_ptr().wrapping_add(next2 as usize * dim) as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<3>(ptr1);
                    std::arch::x86_64::_mm_prefetch::<3>(ptr2);
                }
            } else if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr); }
            }
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

fn main() {
    println!("=== OPT-2 微基准：预取策略优化对比（SIFT1M + 随机图）===");
    println!();

    // 加载 SIFT1M base
    println!("加载 SIFT1M base 数据...");
    let t0 = Instant::now();
    let (vectors, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    println!("加载完成: {} vecs, dim={}, {:.1}s, {:.1}MB",
        n, dim, t0.elapsed().as_secs_f64(), vectors.len() * 4 / 1024 / 1024);

    // 生成随机图（R=32，模拟 Vamana 图的度数）
    let r = 32;
    println!("生成随机图 (R={})...", r);
    let t0 = Instant::now();
    let graph = gen_random_graph(n, r, 42);
    println!("图生成: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 加载查询集
    let (queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    println!("查询集: {} queries", nq);
    let ef_search = 100;
    let k = 10;
    println!("参数: ef_search={}, k={}, R={}", ef_search, k, r);
    println!();

    // 预热
    let mut visited = VisitedTracker::new(n);
    let warmup_query = &queries[0..dim];
    search_prefetch_next_vec(&vectors, dim, &graph, 0, warmup_query, ef_search, &mut visited);
    println!("预热完成");
    println!();

    // 运行 5 种策略
    let strategies: &[(&str, fn(&[f32], usize, &[Vec<u32>], u32, &[f32], usize, &mut VisitedTracker) -> Vec<(u32, f32)>)] = &[
        ("A: 预取 next_vec (当前)", search_prefetch_next_vec),
        ("B: 预取 heap_neighbors", search_prefetch_heap_neighbors),
        ("C: 组合 A+B", search_prefetch_combo),
        ("D: 无预取", search_no_prefetch),
        ("E: 2-ahead 预取", search_prefetch_2ahead),
    ];

    let mut results_summary: Vec<(String, f64, usize)> = Vec::new();

    for (name, search_fn) in strategies {
        let t0 = Instant::now();
        let mut total_results = 0usize;
        for q in 0..nq {
            let query = &queries[q * dim..(q + 1) * dim];
            let result = search_fn(&vectors, dim, &graph, 0, query, ef_search, &mut visited);
            total_results += result.len();
        }
        let time = t0.elapsed().as_secs_f64();
        let qps = nq as f64 / time;
        println!("{}: 时间={:.4}s, QPS={:.0}, avg_latency={:.2}us, results={}",
            name, time, qps, time * 1e6 / nq as f64, total_results);
        results_summary.push((name.to_string(), qps, total_results));
    }

    // 汇总
    println!();
    println!("=== 汇总 ===");
    println!("{:<30} {:>10} {:>12}", "策略", "QPS", "加速比");
    println!("{:-<55}", "");
    let baseline_qps = results_summary[0].1;
    for (name, qps, _) in &results_summary {
        let speedup = qps / baseline_qps;
        println!("{:<30} {:>10.0} {:>10.2}x", name, qps, speedup);
    }
    println!();

    // 结论
    let best = results_summary.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let speedup_vs_baseline = best.1 / baseline_qps;
    if speedup_vs_baseline >= 1.03 {
        println!("结论: 最优策略 '{}' 加速 {:.2}x (≥1.03x)，达到 OPT-2 验收标准", best.0, speedup_vs_baseline);
    } else {
        println!("结论: 最优策略 '{}' 加速 {:.2}x (<1.03x)，未达验收标准", best.0, speedup_vs_baseline);
    }
}
