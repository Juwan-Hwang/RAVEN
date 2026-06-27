//! DirectionalPrune r_max 扫描
//!
//! A/B 结果确认：Pass 1 α=1.0 在 128 维 SIFT 上产出 ~27 条边，
//! Pass 2 (r_min=8) 从未触发。backfill_alpha 无关紧要。
//! 要降 avg_visited 必须降 r_max。
//!
//! 本实验：DirectionalPrune 在 r_max=16/20/24/32 下的表现，
//! 对照 saturation_probe 的 RobustPrune no_sat 数据。
//! 关键问题：DirectionalPrune 的纯方向性边在低 r_max 下是否比 RobustPrune α=1.2 边质量更高？
//!
//! 用法：cargo run --release --bin directional_prune_ab

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, PruneStrategy};
use raven::build::ChaCha8Rng;

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

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_vis: f64,
}

fn run_search(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    po: usize,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);

    let mut hits = 0usize;
    let mut total = 0usize;
    let mut total_visited = 0usize;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        total_visited += searcher.last_visited_count();

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) { hits += 1; }
        }
        total += k;
    }
    let dt = t0.elapsed();

    BenchResult {
        recall: hits as f64 / total as f64,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_vis: total_visited as f64 / nq as f64,
    }
}

fn build_and_bench(
    r_max: usize,
    train: &[f32],
    test: &[f32],
    dim: usize,
    n: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
) {
    let label = format!("DirPrune r_max={}", r_max);
    println!("\n--- {} ---", label);
    let r_soft = (r_max as f32 * 1.5) as usize;
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max,
        r_soft,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: true,
        nav_m: 16,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };

    let t0 = Instant::now();
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("  build: {:.1}s", build_time);

    let stats = graph.degree_stats();
    println!("  degree: mean={:.1} max={} p95={} p99={} isolated={}",
             stats.mean_degree, stats.max_degree, stats.p95_degree, stats.p99_degree,
             stats.isolated_nodes);

    if let Some(nav) = graph.layered_nav() {
        println!("  nav: max_level={}", nav.max_level());
    }

    // ef=50, po=2, 3 rounds
    for round in 1..=3 {
        let r = run_search(train, &graph, test, dim, nq, gt, gt_k, k, 50, 2);
        let prefix = if round == 3 { ">>" } else { "  " };
        println!("  {} ef=50  recall={:.4}  QPS={:>6.0}  avg_visited={:.1}",
                 prefix, r.recall, r.qps, r.avg_vis);
    }

    // ef=100, po=4, 1 round
    let r = run_search(train, &graph, test, dim, nq, gt, gt_k, k, 100, 4);
    println!("  {} ef=100 recall={:.4}  QPS={:>6.0}  avg_visited={:.1}",
             "  ", r.recall, r.qps, r.avg_vis);
}

fn main() {
    println!("=== DirectionalPrune r_max scan ===");
    println!("DirectionalPrune: Pass1 a=1.0 directional, Pass2 backfill to r_min=r_max/4");
    println!("Fixed: alpha=1.2, l_build=200, max_iter=2, layered_nav=true");
    println!();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    println!("data: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    for &r_max in &[32, 24, 20, 16] {
        build_and_bench(r_max, &train, &test, dim, n, nq, &gt, gt_k, k);
    }

    println!("\n=== RobustPrune no_sat baseline (from saturation_probe) ===");
    println!("  r_max=32: recall=0.9704  QPS=8351   avg_visited=1227  degree=32.0");
    println!("  r_max=24: recall=0.9511  QPS=10296  avg_visited=955    degree=24.0");
    println!("  r_max=20: recall=0.9324  QPS=11690  avg_visited=811    degree=20.0");
    println!("  r_max=16: recall=0.8995  QPS=14678  avg_visited=664    degree=16.0");
    println!();
    println!("  Glass HNSW: recall=0.9523  QPS=15171  avg_visited=<150");
    println!();
    println!("Key question: does DirectionalPrune beat RobustPrune at same r_max?");
    println!("If DirPrune r_max=16 recall > 0.8995, directionality edges are higher quality.");
}
