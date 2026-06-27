//! 同进程 A/B 对比：flat Vamana vs v9.1 分层导航
//!
//! 同一次运行连续建两张图（相同种子=相同 Layer 0），
//! 一张不带导航（flat），一张带 v9.1 导航。
//! 消除跨进程环境噪声。
//!
//! 用法：cargo run --release --bin nav_ab_test

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
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

fn run_search(
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
    let mut searcher = GraphSearcher::new(train, graph, ef);
    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;

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

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / dt.as_secs_f64();
    let avg_vis = total_visited as f64 / nq as f64;

    println!(
        "  {:>20}  recall={:.4}  QPS={:>6.0}  avg_visited={:.1}",
        label, recall, qps, avg_vis
    );
}

fn main() {
    println!("=== v9.1 同进程 A/B 对比 ===");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    // === A: flat Vamana（无分层导航）===
    println!("\n--- 建图 A: flat Vamana（enable_layered_nav=false）---");
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

    // === B: v9.1 分层导航 ===
    println!("\n--- 建图 B: v9.1 分层导航（enable_layered_nav=true）---");
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

    if let Some(nav) = graph_b.layered_nav() {
        println!("[nav] max_level={}", nav.max_level());
    }

    // === 对比搜索 ===
    println!("\n=== 搜索对比（ef=50, 三轮消除 cache 冷启动）===");
    for round in 1..=3 {
        println!("Round {}:", round);
        run_search("A: flat", &train, &graph_a, &test, dim, nq, &gt, gt_k, k, 50);
        run_search("B: v9.1 nav", &train, &graph_b, &test, dim, nq, &gt, gt_k, k, 50);
    }

    println!("\n=== 搜索对比（ef=100, 三轮）===");
    for round in 1..=3 {
        println!("Round {}:", round);
        run_search("A: flat", &train, &graph_a, &test, dim, nq, &gt, gt_k, k, 100);
        run_search("B: v9.1 nav", &train, &graph_b, &test, dim, nq, &gt, gt_k, k, 100);
    }

    println!("\n=== 历史基线参照 ===");
    println!("  v3 旧导航:    recall=0.9703  QPS=7,501  avg_visited=1,399.5");
    println!("  Glass HNSW (H20 实测):   recall=0.9465  QPS=7,678 avg_visited=1,041");
}
