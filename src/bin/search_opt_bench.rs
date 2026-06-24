//! 搜索优化对比基准
//!
//! 建图一次，分别跑基线和优化版搜索，对比 QPS/recall。
//!
//! 优化点：
//! 1. 复用距离：greedy_search 内部已算距离，search() 不重算
//! 2. Software prefetch：内层循环预取下一个 neighbor 的向量数据
//!
//! 用法：cargo run --release --bin search_opt_bench

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::memory::{VisitedTracker, HybridBlockedCsr};
use raven::l2_simd;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// f32 包装为 Ord（BinaryHeap 需要）
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
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

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            gt.push(v);
        }
    }
    (gt, dim, n)
}

fn recall_at_k(found: &[u32], gt_slice: &[i32], k: usize) -> f64 {
    let mut hits = 0usize;
    for &g in gt_slice.iter().take(k) {
        if found.contains(&(g as u32)) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

/// 优化版 greedy_search：返回 (id, dist) 对，避免 search() 重算距离
///
/// 优化点：
/// 1. 返回 (id, dist) 而非仅 id，调用方无需重算距离
/// 2. 内层循环 software prefetch 下一个 neighbor 的向量
fn greedy_search_optimized(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();

    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(
        query,
        &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
    );
    candidates.push(Reverse((OrderedF32(entry_dist), entry_point)));
    visited.visit(entry_point);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 {
                    break;
                }
            }
        }

        results.push((dist, node));
        if results.len() > l {
            results.pop();
        }

        let neighbors = storage.neighbors(node);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // Software prefetch: 预取下一个 neighbor 的向量数据
            // 即使下一个 neighbor 已 visited，prefetch 只是 hint，无副作用
            if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe {
                    // _MM_HINT_T0 = 3: 预取到所有 cache 层级
                    std::arch::x86_64::_mm_prefetch::<3>(ptr);
                }
            }

            if visited.visit(neighbor) {
                let d = l2_simd(
                    query,
                    &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                );
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }

    // 返回 (id, dist) 对，避免调用方重算距离
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

fn main() {
    println!("=== 搜索优化对比基准 (SIFT1M) ===");
    println!();

    // 1. 加载数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("数据加载: {:.2}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 归一化到 [0,1]
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }

    // 2. 建图（只建一次）
    println!("=== f32 建图（Vamana α=1.2, r_max=32, l_build=100, max_iter=2）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图时间: {:.2}s ({:.0} vec/s)", build_time, n as f64 / build_time);
    println!();

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;

    // 3. 基线搜索（当前 GraphSearcher::search）
    println!("=== 基线搜索（当前 GraphSearcher::search）===");
    let mut searcher = GraphSearcher::new(&train, &graph, ef_search);
    // 预热
    for q in 0..100.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search(query, k);
    }
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let baseline_time = t0.elapsed().as_secs_f64();
    let baseline_recall = recall_sum / nq as f64;
    let baseline_qps = nq as f64 / baseline_time;
    println!("基线 recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        baseline_recall, baseline_qps, baseline_time * 1000.0 / nq as f64);
    println!();

    // 4. 优化版搜索（距离复用 + prefetch）
    println!("=== 优化版搜索（距离复用 + software prefetch）===");
    let n_nodes = train.len() / dim;
    let mut visited = VisitedTracker::new(n_nodes, ef_search);
    let entry_point = graph.entry_point();
    let storage = graph.storage();

    // 预热
    for q in 0..100.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = greedy_search_optimized(
            &train, dim, storage, entry_point, query, ef_search, &mut visited,
        );
        let mut reranked = candidates;
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let _ = reranked.truncate(k);
    }

    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = greedy_search_optimized(
            &train, dim, storage, entry_point, query, ef_search, &mut visited,
        );
        // 距离已包含在结果中，只需排序取 top-k
        let mut reranked = candidates;
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let found: Vec<u32> = reranked.iter().take(k).map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let opt_time = t0.elapsed().as_secs_f64();
    let opt_recall = recall_sum / nq as f64;
    let opt_qps = nq as f64 / opt_time;
    println!("优化 recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        opt_recall, opt_qps, opt_time * 1000.0 / nq as f64);
    println!();

    // 5. 对比汇总
    println!("=== 对比汇总 ===");
    println!("基线:  recall={:.4}, QPS={:.0}, latency={:.3}ms", baseline_recall, baseline_qps, baseline_time * 1000.0 / nq as f64);
    println!("优化:  recall={:.4}, QPS={:.0}, latency={:.3}ms", opt_recall, opt_qps, opt_time * 1000.0 / nq as f64);
    let qps_delta = (opt_qps - baseline_qps) / baseline_qps * 100.0;
    let recall_delta = (opt_recall - baseline_recall) * 100.0;
    println!("QPS 变化: {:+.1}%", qps_delta);
    println!("recall 变化: {:+.4}pp", recall_delta);
    if qps_delta > 5.0 && recall_delta.abs() < 0.001 {
        println!("结论: 优化有效（QPS 提升 >5%, recall 不变）");
    } else if qps_delta > 0.0 && recall_delta.abs() < 0.001 {
        println!("结论: 优化有轻微效果（QPS 提升 <5%, recall 不变）");
    } else if recall_delta.abs() >= 0.001 {
        println!("结论: 优化影响 recall，需检查正确性");
    } else {
        println!("结论: 优化无效或反作用");
    }
}
