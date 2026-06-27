//! 搜索过程诊断：insert 拒绝率分析
//!
//! 验证假设：RAVEN 图的 insert 拒绝率远低于 Glass，
//! 因为 RobustPrune 的 alpha 角度约束剔除了局部近邻，
//! 导致 worst 距离收紧慢，大量候选通过 insert 检验被继续展开。
//!
//! 用法：cargo run --release --bin reject_rate

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;
use raven::memory::VisitedTracker;
use raven::graph::linear_pool::LinearPool;
use raven::distance::l2_simd;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
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
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (vectors, dim, n)
}

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
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
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

/// 搜索诊断统计
#[derive(Default, Debug)]
struct SearchStats {
    n_popped: usize,
    n_inserted: usize,
    n_rejected: usize,
    n_visited: usize,
    /// worst 距离变化轨迹（每 pop 一次记录当前 worst）
    worst_trace: Vec<f32>,
}

/// 带插桩的搜索（复制 greedy_search_vec_reuse 逻辑，加计数器）
///
/// 返回 (top_k 结果, 统计)
fn search_instrumented(
    vectors: &[f32],
    dim: usize,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
) -> (Vec<(u32, f32)>, SearchStats) {
    visited.reset();
    let mut pool = LinearPool::new(ef);
    let mut stats = SearchStats::default();

    // 插入入口节点
    let entry_dist = l2_simd(
        query,
        &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
    );
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);
    stats.n_visited = 1;

    while let Some((node, _dist)) = pool.pop() {
        stats.n_popped += 1;
        stats.worst_trace.push(pool.worst_distance());

        let neighbors = storage.neighbors(node);
        for &neighbor in neighbors {
            if neighbor == u32::MAX {
                continue;
            }
            if visited.visit(neighbor) {
                stats.n_visited += 1;
                let d = l2_simd(
                    query,
                    &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                );
                if pool.insert(neighbor, d) {
                    stats.n_inserted += 1;
                } else {
                    stats.n_rejected += 1;
                }
            }
        }
    }

    (pool.to_sorted_vec(), stats)
}

fn run_diagnostic(
    label: &str,
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
) {
    let storage = graph.storage();
    let n = train.len() / dim;
    let mut visited = VisitedTracker::new(n, ef);

    let mut total_popped = 0u64;
    let mut total_inserted = 0u64;
    let mut total_rejected = 0u64;
    let mut total_visited = 0u64;
    let mut hits = 0usize;
    let mut total = 0usize;

    // 每查询的拒绝率分布（用于直方图）
    let mut per_query_reject_rates: Vec<f64> = Vec::with_capacity(nq);

    // worst 距离轨迹（取前 100 个查询的平均）
    let mut avg_worst_trace: Vec<f64> = Vec::new();
    let trace_queries = 100usize.min(nq);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];

        // 入口点选择（与 GraphSearcher::search 一致）
        let entry_point = if let Some(nav) = graph.layered_nav() {
            nav.initialize(train, dim, query).0
        } else {
            graph.entry_point()
        };

        let (result, stats) = search_instrumented(
            train, dim, storage, entry_point, query, ef, &mut visited,
        );

        total_popped += stats.n_popped as u64;
        total_inserted += stats.n_inserted as u64;
        total_rejected += stats.n_rejected as u64;
        total_visited += stats.n_visited as u64;

        let total_attempts = stats.n_inserted + stats.n_rejected;
        let reject_rate = if total_attempts > 0 {
            stats.n_rejected as f64 / total_attempts as f64
        } else {
            0.0
        };
        per_query_reject_rates.push(reject_rate);

        if q < trace_queries {
            let max_len = avg_worst_trace.len().max(stats.worst_trace.len());
            avg_worst_trace.resize(max_len, 0.0);
            for (i, &w) in stats.worst_trace.iter().enumerate() {
                avg_worst_trace[i] += w as f64 / trace_queries as f64;
            }
        }

        // recall 计算
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
        total += k;
    }
    let dt = t0.elapsed();

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / dt.as_secs_f64();
    let avg_visited = total_visited as f64 / nq as f64;
    let avg_popped = total_popped as f64 / nq as f64;
    let avg_inserted = total_inserted as f64 / nq as f64;
    let avg_rejected = total_rejected as f64 / nq as f64;
    let overall_reject_rate = total_rejected as f64 / (total_inserted + total_rejected) as f64;

    // 拒绝率分位数
    per_query_reject_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = per_query_reject_rates[nq / 2];
    let p95 = per_query_reject_rates[(nq as f64 * 0.95) as usize];
    let p99 = per_query_reject_rates[(nq as f64 * 0.99) as usize];
    let min_rr = per_query_reject_rates[0];
    let max_rr = per_query_reject_rates[nq - 1];

    println!("\n=== {} (ef={}) ===", label, ef);
    println!("  recall={:.4}  QPS={:.0}  avg_visited={:.1}", recall, qps, avg_visited);
    println!("  popped={:.1}  inserted={:.1}  rejected={:.1}", avg_popped, avg_inserted, avg_rejected);
    println!("  insert 拒绝率: {:.1}%  (inserted+rejected={:.0})", overall_reject_rate * 100.0, avg_inserted + avg_rejected);
    println!("  拒绝率分位数: min={:.1}%  p50={:.1}%  p95={:.1}%  p99={:.1}%  max={:.1}%",
             min_rr * 100.0, p50 * 100.0, p95 * 100.0, p99 * 100.0, max_rr * 100.0);

    // worst 距离收紧曲线（前 20 跳 + 后 5 跳）
    let trace_len = avg_worst_trace.len();
    if trace_len > 0 {
        print!("  worst 收紧曲线: [0]={:.1}", avg_worst_trace[0]);
        let head = 20.min(trace_len);
        for i in 1..head {
            print!(" [{i}]={:.1}", avg_worst_trace[i]);
        }
        if trace_len > 25 {
            print!(" ... [{:.0}]={:.1}", trace_len as f64 - 5.0, avg_worst_trace[trace_len - 5]);
            print!(" [{:.0}]={:.1}", trace_len as f64 - 1.0, avg_worst_trace[trace_len - 1]);
        }
        println!();
        // 收紧比：第 5 跳 vs 第 0 跳
        if trace_len > 5 {
            let ratio = avg_worst_trace[5] / avg_worst_trace[0];
            println!("  worst[5]/worst[0] = {:.3}  (越小=收紧越快)", ratio);
        }
    }
}

fn main() {
    println!("=== 搜索 insert 拒绝率诊断 ===");
    println!("假设: RAVEN 图 worst 收紧慢 → insert 拒绝率低 → 无效展开多");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    // === 建图 A: flat Vamana ===
    println!("\n--- 建图 A: flat Vamana ---");
    let t0 = Instant::now();
    let mut rng_a = ChaCha8Rng::seed_from(42);
    let config_a = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: false,
        nav_m: 16,
        ..Default::default()
    };
    let graph_a = VamanaGraph::build(&train, dim, &config_a, &mut rng_a);
    println!("建图 A: {:.1}s", t0.elapsed().as_secs_f64());

    // === 建图 B: v9.1 分层导航 ===
    println!("\n--- 建图 B: v9.1 分层导航 ---");
    let t0 = Instant::now();
    let mut rng_b = ChaCha8Rng::seed_from(42);
    let config_b = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: true,
        nav_m: 16,
        ..Default::default()
    };
    let graph_b = VamanaGraph::build(&train, dim, &config_b, &mut rng_b);
    println!("建图 B: {:.1}s", t0.elapsed().as_secs_f64());

    // 度数对比
    let stats_a = graph_a.degree_stats();
    let stats_b = graph_b.degree_stats();
    println!("\n度数对比: A={:.1} B={:.1}", stats_a.mean_degree, stats_b.mean_degree);

    // === 诊断 ===
    // 只用前 1000 个查询（足够统计，快）
    let diag_nq = 1000usize.min(nq);

    for &ef in &[50, 100] {
        run_diagnostic("A: flat Vamana", &train, &graph_a, &test, dim, diag_nq, &gt, gt_k, k, ef);
        run_diagnostic("B: v9.1 nav", &train, &graph_b, &test, dim, diag_nq, &gt, gt_k, k, ef);
    }

    // === Glass 参照 ===
    println!("\n=== Glass HNSW 参照（理论值）===");
    println!("  avg_visited < 150, insert 拒绝率 ~91% (文献值)");
    println!("  worst 收紧：前 3 跳即收敛到真实 k-NN 距离");
}
