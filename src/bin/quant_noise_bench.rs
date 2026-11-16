//! SQ4 vs f32 同图对比实验
//!
//! 在同一张图、同一批 query 上分别用 SQ4 量化距离和 f32 精确距离搜索，
//! 对比 avg_visited 和 recall，隔离"量化噪声"与"图拓扑质量"两个变量。
//!
//! 如果 f32 的 avg_visited 显著低于 SQ4 -> 瓶颈部分来自量化噪声
//! 如果两者接近 -> 瓶颈确认在图拓扑本身
//!
//! 同时测试 SQ8 作为中间精度参照。
//!
//! 用法：cargo run --release --bin quant_noise_bench

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::linear_pool::LinearPool;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::VisitedTracker;
use raven::quant::{SQ4Dataset, SQ8Dataset};

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
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
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
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

fn compute_recall(results: &[Vec<u32>], gt: &[i32], gt_k: usize, k: usize, nq: usize) -> f64 {
    let mut hits = 0u64;
    for q in 0..nq {
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &id in &results[q] {
            if gt_slice.contains(&(id as i32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rerank_factor: usize,
) -> Vec<(u32, f32)> {
    let n_rerank = (k * rerank_factor).min(candidates.len());
    let mut results: Vec<(u32, f32)> = candidates[..n_rerank]
        .iter()
        .map(|(id, _)| {
            let dist = l2_simd(
                query,
                &vectors[*id as usize * dim..(*id as usize + 1) * dim],
            );
            (*id, dist)
        })
        .collect();
    results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(k);
    results
}

#[derive(Clone, Copy, PartialEq)]
enum DistanceMetric {
    F32,
    SQ8,
    SQ4,
}

impl DistanceMetric {
    fn label(&self) -> &'static str {
        match self {
            DistanceMetric::F32 => "f32",
            DistanceMetric::SQ8 => "SQ8",
            DistanceMetric::SQ4 => "SQ4",
        }
    }
}

fn search(
    metric: DistanceMetric,
    vectors: &[f32],
    dim: usize,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    query_code_sq4: &[u8],
    query_code_sq8: &[u8],
    ef: usize,
    visited: &mut VisitedTracker,
    pool: &mut LinearPool,
    po: usize,
    sq4: &SQ4Dataset,
    sq8: &SQ8Dataset,
) -> (Vec<(u32, f32)>, usize) {
    match metric {
        DistanceMetric::F32 => {
            let r = VamanaGraph::greedy_search_vec_reuse(
                vectors, dim, storage, entry_point, query, ef,
                visited, pool, po,
            );
            let vc = visited.visited_count();
            (r, vc)
        }
        DistanceMetric::SQ8 => {
            let r = VamanaGraph::greedy_search_sq8::<true>(
                sq8, storage, entry_point, query_code_sq8, ef,
                visited, pool, po,
            );
            let vc = visited.visited_count();
            (r, vc)
        }
        DistanceMetric::SQ4 => {
            let r = VamanaGraph::greedy_search_sq4(
                sq4, storage, entry_point, query_code_sq4, ef,
                visited, pool, po, 0,
            );
            let vc = visited.visited_count();
            (r, vc)
        }
    }
}

struct BenchResult {
    recall: f64,
    avg_visited: f64,
    qps: f64,
}

fn run_bench(
    metric: DistanceMetric,
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    gt_k: usize,
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
    ef: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
    sq8: &SQ8Dataset,
    graph: &VamanaGraph,
) -> BenchResult {
    let nav = graph.layered_nav().expect("need layered nav");
    let mut visited = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);

    let code_bytes_sq4 = dim.div_ceil(2);
    let code_bytes_sq8 = dim;
    let mut query_code_sq4 = vec![0u8; code_bytes_sq4];
    let mut query_code_sq8 = vec![0u8; code_bytes_sq8];

    let mut results = Vec::with_capacity(nq);
    let mut total_visited = 0usize;
    let t0 = Instant::now();

    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];

        sq4.params.encode_into(query, &mut query_code_sq4);
        sq8.params.encode_into(query, &mut query_code_sq8);

        let (entry_point, _) = nav.initialize(train, dim, query);

        let (candidates, vc) = search(
            metric, train, dim, graph.storage(), entry_point, query,
            &query_code_sq4, &query_code_sq8, ef, &mut visited, &mut pool, po,
            sq4, sq8,
        );

        total_visited += vc;
        let r = rerank_f32(train, dim, query, candidates, k, rr);
        results.push(r.iter().map(|(id, _)| *id).collect());
    }

    let dt = t0.elapsed();
    BenchResult {
        recall: compute_recall(&results, gt, gt_k, k, nq),
        avg_visited: total_visited as f64 / nq as f64,
        qps: nq as f64 / dt.as_secs_f64(),
    }
}

fn main() {
    println!("=== SQ4 vs f32 同图对比 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = 1000.min(nq_full);
    let k = 10;
    let po = 16;
    let rr = 5;

    println!("--- 建图 ---");
    let config = VamanaBuildConfig {
        alpha: 1.2, l_build: 384, r_max: 40, r_soft: 60,
        max_iterations: 2, saturate: false,
        enable_layered_nav: true, nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.1}s\n", t0.elapsed().as_secs_f64());

    println!("--- 构建量化数据集 ---");
    let t0 = Instant::now();
    let sq4 = SQ4Dataset::build(&train, dim);
    let sq8 = SQ8Dataset::build(&train, dim);
    println!("SQ4+SQ8 构建时间: {:.1}s\n", t0.elapsed().as_secs_f64());

    let ef_values: Vec<usize> = vec![35, 50, 65, 80, 100];
    let metrics = [DistanceMetric::F32, DistanceMetric::SQ8, DistanceMetric::SQ4];

    println!("--- 搜索 (k={}, nq={}, po={}, rr={}) ---\n", k, nq, po, rr);

    for &ef in &ef_values {
        println!("=== ef={} ===", ef);
        println!(
            "{:<8} {:<10} {:<14} {:<10}",
            "metric", "recall", "avg_visited", "qps"
        );
        println!("{}", "-".repeat(50));

        let mut f32_visited = 0.0;

        for &metric in &metrics {
            let r = run_bench(
                metric, &train, &test, &gt, gt_k, dim, n, nq, k, ef, po, rr,
                &sq4, &sq8, &graph,
            );

            let delta_str = if metric == DistanceMetric::F32 {
                f32_visited = r.avg_visited;
                "(baseline)".to_string()
            } else {
                let delta = (r.avg_visited - f32_visited) / f32_visited * 100.0;
                format!("({:+.1}% vs f32)", delta)
            };

            println!(
                "{:<8} {:<10.4} {:<10.1}    {:<10.0}  {}",
                metric.label(), r.recall, r.avg_visited, r.qps, delta_str,
            );
        }
        println!();
    }

    // 汇总表
    println!("=== 汇总：量化噪声对 avg_visited 的影响 ===\n");
    println!("（正值 = SQ4/SQ8 比 f32 多访问的百分比 = 量化噪声代价）\n");
    println!("{:<6} {:<10} {:<10} {:<10} {:<15} {:<15}", "ef", "f32", "SQ8", "SQ4", "SQ8 Δ%", "SQ4 Δ%");
    println!("{}", "-".repeat(70));

    for &ef in &ef_values {
        let mut f32_v = 0.0;
        let mut sq8_v = 0.0;
        let mut sq4_v = 0.0;

        for &metric in &metrics {
            let r = run_bench(
                metric, &train, &test, &gt, gt_k, dim, n, nq, k, ef, po, rr,
                &sq4, &sq8, &graph,
            );
            match metric {
                DistanceMetric::F32 => f32_v = r.avg_visited,
                DistanceMetric::SQ8 => sq8_v = r.avg_visited,
                DistanceMetric::SQ4 => sq4_v = r.avg_visited,
            }
        }

        let sq8_delta = (sq8_v - f32_v) / f32_v * 100.0;
        let sq4_delta = (sq4_v - f32_v) / f32_v * 100.0;

        println!(
            "{:<6} {:<10.1} {:<10.1} {:<10.1} {:+<15.1} {:+<15.1}",
            ef, f32_v, sq8_v, sq4_v, sq8_delta, sq4_delta,
        );
    }

    println!("\n=== 实验完成 ===");
}
