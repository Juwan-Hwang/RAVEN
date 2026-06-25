//! OPT-3 微基准：BinaryHeap vs flat sorted Vec 堆性能对比
//!
//! 目标：验证 flat sorted Vec 是否比 BinaryHeap 更快
//!
//! 核心假设：BinaryHeap 的堆操作涉及随机内存访问（树结构），
//! 而 flat sorted Vec 是连续内存，cache 友好。
//! 在 ef_search=100（堆大小约 100-200）时，flat vec 的 O(n) 插入
//! 可能比 BinaryHeap 的 O(log n) 更快（因为 cache 命中率高）。
//!
//! 实验方案：
//! - 方案 A（当前）：BinaryHeap<Reverse<(OrderedF32, u32)>>
//! - 方案 B：flat sorted Vec（降序，pop 从尾部 O(1)，push 用 binary_search + insert）
//!
//! 数据：SIFT1M base + 随机图（复用 opt2_bench 框架）

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::l2_simd;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use rand::SeedableRng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap()));
        }
    }
    (vectors, dim, n)
}

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

struct VisitedTracker {
    visited: Vec<u8>,
    gen: u8,
}
impl VisitedTracker {
    fn new(n: usize) -> Self { Self { visited: vec![0; n], gen: 1 } }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 { self.visited.fill(0); self.gen = 1; }
    }
    fn visit(&mut self, node: u32) -> bool {
        let idx = node as usize;
        if self.visited[idx] == self.gen { false }
        else { self.visited[idx] = self.gen; true }
    }
}

/// 方案 A：BinaryHeap（当前实现，含 OPT-2 预取策略 B）
fn search_binary_heap(
    vectors: &[f32], dim: usize, graph: &[Vec<u32>],
    entry: u32, query: &[f32], l: usize, visited: &mut VisitedTracker,
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

        // OPT-2 预取策略 B
        if let Some(&Reverse((_, top_node))) = candidates.peek() {
            let top_neighbors = &graph[top_node as usize];
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

/// 方案 B：flat sorted Vec（降序，pop 从尾部取最小值 O(1)）
fn search_flat_sorted_vec(
    vectors: &[f32], dim: usize, graph: &[Vec<u32>],
    entry: u32, query: &[f32], l: usize, visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    // candidates: 降序 vec，最小值在尾部，pop() O(1)
    // (dist, node)，降序排列
    let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(l * 2);
    // results: 升序 vec，最大值在尾部，pop() O(1)
    let mut results: Vec<(f32, u32)> = Vec::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    // push 到降序 candidates：binary_search 找位置 + insert
    let pos = candidates.partition_point(|(d, _)| d > &entry_dist);
    candidates.insert(pos, (entry_dist, entry));
    visited.visit(entry);

    while let Some((dist, node)) = candidates.pop() { // pop 从尾部取最小值
        if results.len() >= l {
            if let Some(&(worst, _)) = results.last() {
                if dist > worst { break; }
            }
        }
        // push 到升序 results：binary_search 找位置 + insert
        let pos = results.partition_point(|(d, _)| d < &dist);
        results.insert(pos, (dist, node));
        if results.len() > l { results.pop(); } // pop 从尾部取最大值

        // OPT-2 预取策略 B
        if let Some(&(_, top_node)) = candidates.last() {
            let top_neighbors = &graph[top_node as usize];
            let ptr = top_neighbors.as_ptr() as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        let neighbors = &graph[node as usize];
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                // push 到降序 candidates
                let pos = candidates.partition_point(|(cd, _)| cd > &d);
                candidates.insert(pos, (d, neighbor));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist)).collect()
}

fn main() {
    println!("=== OPT-3 微基准：BinaryHeap vs flat sorted Vec ===");
    println!();

    let t0 = Instant::now();
    let (vectors, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    println!("SIFT1M: {} vecs, dim={}, {:.1}s, {:.1}MB",
        n, dim, t0.elapsed().as_secs_f64(), vectors.len() * 4 / 1024 / 1024);

    let r = 32;
    let t0 = Instant::now();
    let graph = gen_random_graph(n, r, 42);
    println!("随机图 R={}: {:.1}s", r, t0.elapsed().as_secs_f64());

    let (queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let ef_search = 100;
    println!("查询: {} queries, ef_search={}", nq, ef_search);
    println!();

    let mut visited = VisitedTracker::new(n);
    // 预热
    search_binary_heap(&vectors, dim, &graph, 0, &queries[0..dim], ef_search, &mut visited);
    search_flat_sorted_vec(&vectors, dim, &graph, 0, &queries[0..dim], ef_search, &mut visited);
    println!("预热完成");
    println!();

    // 方案 A：BinaryHeap
    let t0 = Instant::now();
    let mut sum_a = 0u64;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let result = search_binary_heap(&vectors, dim, &graph, 0, query, ef_search, &mut visited);
        sum_a += result.len() as u64;
    }
    let time_a = t0.elapsed().as_secs_f64();
    let qps_a = nq as f64 / time_a;
    println!("方案 A: BinaryHeap");
    println!("  时间={:.4}s, QPS={:.0}, avg_latency={:.2}us, results={}",
        time_a, qps_a, time_a * 1e6 / nq as f64, sum_a);

    // 方案 B：flat sorted Vec
    let t0 = Instant::now();
    let mut sum_b = 0u64;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let result = search_flat_sorted_vec(&vectors, dim, &graph, 0, query, ef_search, &mut visited);
        sum_b += result.len() as u64;
    }
    let time_b = t0.elapsed().as_secs_f64();
    let qps_b = nq as f64 / time_b;
    println!("方案 B: flat sorted Vec");
    println!("  时间={:.4}s, QPS={:.0}, avg_latency={:.2}us, results={}",
        time_b, qps_b, time_b * 1e6 / nq as f64, sum_b);
    println!();

    // 汇总
    println!("=== 汇总 ===");
    println!("{:<25} {:>10} {:>12} {:>10}", "方案", "QPS", "avg_latency", "加速比");
    println!("{:-<60}", "");
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "A: BinaryHeap", qps_a, time_a * 1e6 / nq as f64, 1.0);
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "B: flat sorted Vec", qps_b, time_b * 1e6 / nq as f64, qps_b / qps_a);
    println!();

    let speedup = qps_b / qps_a;
    if speedup >= 1.05 {
        println!("结论: flat sorted Vec 加速 {:.2}x (≥1.05x)，达到 OPT-3 验收标准", speedup);
    } else if speedup >= 1.02 {
        println!("结论: flat sorted Vec 加速 {:.2}x (1.02-1.05x)，收益有限", speedup);
    } else {
        println!("结论: flat sorted Vec 加速 {:.2}x (<1.02x)，无收益，OPT-3 否决", speedup);
    }
}
