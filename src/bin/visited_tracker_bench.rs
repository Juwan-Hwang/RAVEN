//! VisitedTracker 复用隔离对比基准（科学验证）
//!
//! 目标：隔离测试 VisitedTracker 复用对搜索性能的影响
//!
//! 方法：
//! 1. 用 learn 集（100K）快速建图（~5 分钟）
//! 2. 对同一张图，分别用：
//!    - baseline：greedy_search_vec（每次新建 VisitedTracker，分配 100KB）
//!    - reuse：greedy_search_vec_reuse（复用 VisitedTracker，零分配）
//! 3. 测量 QPS、recall@10、p50/p99 延迟
//!
//! 判据：
//! - QPS 提升 ≥5%，recall 完全不变 → 优化有效
//! - QPS 无变化或下降 → 优化无效，回退
//! - recall 下降 → 存在 bug，立即修复

use std::fs::File;
use std::io::Read;
use std::time::{Instant, Duration};
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::memory::VisitedTracker;
use raven::distance::l2_simd;
use raven::build::ChaCha8Rng;

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

/// 读取 ivecs 文件
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

/// baseline 搜索（每次新建 VisitedTracker）
///
/// 模拟优化前的行为：每次搜索都分配 visited 数组
fn search_baseline(
    vectors: &[f32],
    graph: &VamanaGraph,
    query: &[f32],
    ef_search: usize,
    k: usize,
    dim: usize,
) -> Vec<(u32, f32)> {
    let (candidates, _visited) = VamanaGraph::greedy_search_vec(
        vectors,
        dim,
        graph.storage(),
        graph.entry_point(),
        query,
        ef_search,
    );
    let mut results: Vec<(u32, f32)> = candidates
        .into_iter()
        .map(|id| {
            let v = &vectors[id as usize * dim..(id as usize + 1) * dim];
            (id, l2_simd(query, v))
        })
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// reuse 搜索（复用 VisitedTracker）
///
/// 优化后的行为：复用预分配的 VisitedTracker
fn search_reuse(
    vectors: &[f32],
    graph: &VamanaGraph,
    query: &[f32],
    ef_search: usize,
    k: usize,
    dim: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    let candidates = VamanaGraph::greedy_search_vec_reuse(
        vectors,
        dim,
        graph.storage(),
        graph.entry_point(),
        query,
        ef_search,
        visited,
    );
    // 距离已在 greedy_search_vec_reuse 中计算，只需排序取 top-k
    let mut results = candidates;
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// 运行搜索基准，返回 (recall, qps, latencies)
fn run_bench<F>(
    name: &str,
    vectors: &[f32],
    _graph: &VamanaGraph,
    queries: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    _ef_search: usize,
    k: usize,
    mut search_fn: F,
) -> (f64, f64, Vec<Duration>)
where
    F: FnMut(&[f32], &[f32]) -> Vec<(u32, f32)>,
{
    let gt_stride = 100;
    let mut hits = 0usize;
    let mut latencies = Vec::with_capacity(nq);

    // 预热：跑前 100 个 query 不计时
    for q in 0..100.min(nq) {
        let query = &queries[q * dim..(q + 1) * dim];
        let _ = search_fn(query, vectors);
    }

    // 正式计时
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let tq = Instant::now();
        let result = search_fn(query, vectors);
        latencies.push(tq.elapsed());

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    let qps = nq as f64 / elapsed;

    // 计算延迟分位数
    let mut sorted_lat = latencies.clone();
    sorted_lat.sort();
    let p50 = sorted_lat[sorted_lat.len() / 2];
    let p99 = sorted_lat[(sorted_lat.len() as f64 * 0.99) as usize];

    println!("{}: recall={:.4}, QPS={:.0}, p50={:.2}µs, p99={:.2}µs",
        name, recall, qps,
        p50.as_secs_f64() * 1e6,
        p99.as_secs_f64() * 1e6);

    (recall, qps, latencies)
}

fn main() {
    println!("=== VisitedTracker 复用隔离对比基准 ===");
    println!("目标：科学验证 VisitedTracker 复用对搜索性能的影响");
    println!("方法：同一张图，对比 baseline（每次新建）vs reuse（复用）");
    println!();

    // 1. 加载 base 集前 100K 子集（确保 groundtruth 有效）
    // 用前 100K 而非 learn 集，因为 groundtruth 是基于完整 base 集的
    let t0 = Instant::now();
    let (full_base, dim, n_full) = read_fvecs("data/sift/sift_base.fvecs");
    let n_db = 100_000.min(n_full); // 取前 100K
    let mut db = full_base[..n_db * dim].to_vec();
    drop(full_base); // 释放完整 base
    let (mut queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("数据加载: {:.1}s", t0.elapsed().as_secs_f64());
    println!("db(base 前 100K): {} vecs, queries: {} vecs, dim={}, gt_k={}", n_db, nq, dim, gt_k);
    println!("注意：groundtruth 基于完整 base，recall 会低于全量");
    println!();

    // 归一化到 [0,1]
    for v in db.iter_mut() { *v /= 255.0; }
    for v in queries.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. 建图（保守参数，快速构建）
    println!("=== 建图（learn 100K, r_max=32, l_build=100, α=1.2）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&db, dim, &config, &mut rng);
    println!("建图完成: {:.1}s, avg_degree={:.1}", t0.elapsed().as_secs_f64(), graph.degree_stats().mean_degree);
    println!();

    // 3. 预分配 VisitedTracker（reuse 模式用）
    let mut visited = VisitedTracker::new(n_db, ef_search);

    // 4. baseline 搜索（每次新建 VisitedTracker）
    println!("=== 搜索基准（10K 查询, ef=100）===");
    println!();

    let (baseline_recall, baseline_qps, baseline_lat) = run_bench(
        "baseline (每次新建 VisitedTracker)",
        &db,
        &graph,
        &queries,
        &gt,
        dim,
        nq,
        ef_search,
        k,
        |query, vectors| search_baseline(vectors, &graph, query, ef_search, k, dim),
    );

    // 5. reuse 搜索（复用 VisitedTracker）
    println!();

    let (reuse_recall, reuse_qps, reuse_lat) = run_bench(
        "reuse (复用 VisitedTracker)",
        &db,
        &graph,
        &queries,
        &gt,
        dim,
        nq,
        ef_search,
        k,
        |query, vectors| search_reuse(vectors, &graph, query, ef_search, k, dim, &mut visited),
    );

    // 6. 对比分析
    println!();
    println!("=== 对比分析 ===");
    println!("{:<30} {:>12} {:>12} {:>12}", "", "baseline", "reuse", "diff");
    println!("{:-<70}", "");
    println!("{:<30} {:>12.4} {:>12.4} {:>12.4}", "recall@10", baseline_recall, reuse_recall, reuse_recall - baseline_recall);
    println!("{:<30} {:>12.0} {:>12.0} {:>11.1}%", "QPS", baseline_qps, reuse_qps, (reuse_qps / baseline_qps - 1.0) * 100.0);

    // 延迟分位数
    let mut bl_sorted = baseline_lat.clone();
    bl_sorted.sort();
    let mut ru_sorted = reuse_lat.clone();
    ru_sorted.sort();
    let bl_p50 = bl_sorted[bl_sorted.len() / 2];
    let ru_p50 = ru_sorted[ru_sorted.len() / 2];
    let bl_p99 = bl_sorted[(bl_sorted.len() as f64 * 0.99) as usize];
    let ru_p99 = ru_sorted[(ru_sorted.len() as f64 * 0.99) as usize];

    println!("{:<30} {:>10.2}µs {:>10.2}µs {:>11.1}%", "p50 延迟",
        bl_p50.as_secs_f64() * 1e6, ru_p50.as_secs_f64() * 1e6,
        (ru_p50.as_secs_f64() / bl_p50.as_secs_f64() - 1.0) * 100.0);
    println!("{:<30} {:>10.2}µs {:>10.2}µs {:>11.1}%", "p99 延迟",
        bl_p99.as_secs_f64() * 1e6, ru_p99.as_secs_f64() * 1e6,
        (ru_p99.as_secs_f64() / bl_p99.as_secs_f64() - 1.0) * 100.0);

    println!();
    println!("=== 判定 ===");
    let qps_improvement = (reuse_qps / baseline_qps - 1.0) * 100.0;
    let recall_diff = reuse_recall - baseline_recall;

    if recall_diff.abs() > 1e-6 {
        println!("FAIL: recall 变化 {:.6}，存在 bug，需要修复", recall_diff);
    } else if qps_improvement >= 5.0 {
        println!("PASS: QPS 提升 {:.1}%，recall 不变，优化有效", qps_improvement);
    } else if qps_improvement > 0.0 {
        println!("MARGINAL: QPS 提升 {:.1}%（< 5%），优化效果不显著", qps_improvement);
    } else {
        println!("FAIL: QPS 下降 {:.1}%，优化无效，需要回退", qps_improvement);
    }
}
