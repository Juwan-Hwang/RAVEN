//! SIFT1M 参数扫描：R=32/L=200 vs R=70/L=75
//!
//! 核心假设：RAVEN avg_visited 膨胀的根因是 R=32 度数太低（DiskANN 用 R=70）。
//! 低度数 → 图直径大 → 搜索需要更多跳 → avg_visited 膨胀。
//!
//! 用法: cargo run --release --bin param_sweep_1m

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, PruneStrategy};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
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

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            gt.push(i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap()));
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

struct BuildResult {
    name: &'static str,
    build_time: f64,
    #[allow(dead_code)]
    graph: VamanaGraph,
}

fn build_and_eval(
    name: &'static str,
    train: &[f32],
    dim: usize,
    n: usize,
    test: &[f32],
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    config: &VamanaBuildConfig,
) -> BuildResult {
    let k = 10;
    println!("\n--- {} (R={}, L={}, r_soft={}, alpha={}) ---",
        name, config.r_max, config.l_build, config.r_soft, config.alpha);

    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(train, dim, config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("  build: {:.1}s ({:.0} vec/s)", build_time, n as f64 / build_time);

    // ef sweep
    for &ef in &[50, 100, 200] {
        let mut searcher = GraphSearcher::new(train, &graph, ef);
        let mut recall_sum = 0.0f64;
        let mut visited_sum = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = searcher.search(query, k);
            let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
            let gt_slice = &gt[q * gt_k..q * gt_k + k];
            recall_sum += recall_at_k(&found, gt_slice, k);
            visited_sum += searcher.last_visited_count();
        }
        let search_time = t0.elapsed().as_secs_f64();
        let recall = recall_sum / nq as f64;
        let qps = nq as f64 / search_time;
        let avg_visited = visited_sum as f64 / nq as f64;
        println!("  ef={:>3}: recall@10={:.4}, QPS={:>6.0}, avg_visited={:.0}",
            ef, recall, qps, avg_visited);
    }

    BuildResult { name, build_time, graph }
}

fn main() {
    println!("=== SIFT1M Parameter Sweep: R=32/L=200 vs R=70/L=75 ===\n");

    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }
    println!("Data loaded: {}s (n={}, dim={}, nq={})", t0.elapsed().as_secs_f64(), n, dim, nq);

    // Config A: RAVEN 当前参数（基线）
    let config_a = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
    };

    // Config B: DiskANN Vamana 参数
    let config_b = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 75,
        r_soft: 105,  // 1.5 × 70
        r_max: 70,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
    };

    let result_a = build_and_eval("RAVEN R=32 L=200", &train, dim, n, &test, nq, &gt, gt_k, &config_a);
    let result_b = build_and_eval("DiskANN R=70 L=75", &train, dim, n, &test, nq, &gt, gt_k, &config_b);

    println!("\n=== Summary ===");
    println!("{:<24} {:>10}", "Config", "build_s");
    println!("{:-<36}", "");
    println!("{:<24} {:>10.1}", result_a.name, result_a.build_time);
    println!("{:<24} {:>10.1}", result_b.name, result_b.build_time);
    println!("\nDiskANN ref: 129s (R=70, L=75, 1-pass)");
    println!("RAVEN ref:   260s (R=32, L=200, 2-pass, 16 threads)");
}
