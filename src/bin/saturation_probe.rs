//! Saturation + r_max 参数扫描实验
//!
//! 验证假设：禁用 saturation + 降低 r_max 是否能降低 avg_visited 同时维持 recall？
//!
//! 之前数据：
//!   saturate=true,  r_max=32: avg_visited=1227, recall=0.9705, QPS=9195
//!   saturate=false, r_max=32: avg_visited=2324（图太稀疏，反而更差）
//!
//! 本实验：saturate=false + 逐步降低 r_max，找到"自然稀疏但不至于孤岛"的甜点。
//! 如果存在某个 r_max 使得 avg_visited 下降 + recall 持平，
//! 则 DirectionalPrune 的 r_min 设计方向正确。
//!
//! 用法：cargo run --release --bin saturation_probe

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

struct Config {
    label: &'static str,
    r_max: usize,
    saturate: bool,
}

fn run_config(
    cfg: &Config,
    train: &[f32],
    test: &[f32],
    dim: usize,
    n: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    po: usize,
) {
    println!("\n--- {} (r_max={}, saturate={}) ---", cfg.label, cfg.r_max, cfg.saturate);

    let r_soft = (cfg.r_max as f32 * 1.3) as usize;
    let mut rng = ChaCha8Rng::seed_from(42);
    let build_config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: cfg.r_max,
        r_soft,
        max_iterations: 2,
        saturate: cfg.saturate,
        enable_layered_nav: true,
        nav_m: 16,
        ..Default::default()
    };

    let t0 = Instant::now();
    let graph = VamanaGraph::build(train, dim, &build_config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("  建图: {:.1}s", build_time);

    // 度数统计
    let stats = graph.degree_stats();
    println!("  度数: mean={:.1} max={} p95={} p99={} isolated={} overflow={:.1}%",
             stats.mean_degree, stats.max_degree, stats.p95_degree, stats.p99_degree,
             stats.isolated_nodes, stats.overflow_ratio * 100.0);

    if let Some(nav) = graph.layered_nav() {
        println!("  导航层: max_level={}", nav.max_level());
    }

    // 搜索：3 轮取最佳（消除 cache 冷启动）
    let mut best_qps = 0.0f64;
    let mut best_recall = 0.0f64;
    let mut best_avg_vis = 0.0f64;

    for round in 1..=3 {
        let mut searcher = GraphSearcher::new(train, &graph, ef);
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

        let recall = hits as f64 / total as f64;
        let qps = nq as f64 / dt.as_secs_f64();
        let avg_vis = total_visited as f64 / nq as f64;

        if round == 3 || qps > best_qps {
            best_qps = qps;
            best_recall = recall;
            best_avg_vis = avg_vis;
        }
    }

    println!("  结果: recall={:.4}  QPS={:>6.0}  avg_visited={:.1}",
             best_recall, best_qps, best_avg_vis);
}

fn main() {
    println!("=== Saturation + r_max 参数扫描 ===");
    println!("假设：禁用 saturation + 降低 r_max → avg_visited 下降 + recall 持平");
    println!("固定：alpha=1.2, l_build=200, max_iter=2, layered_nav=true, ef=50, po=2");
    println!();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;
    let ef = 50usize;
    let po = 2usize;

    let configs = [
        Config { label: "baseline",         r_max: 32, saturate: true  },
        Config { label: "no_sat_r32",       r_max: 32, saturate: false },
        Config { label: "no_sat_r24",       r_max: 24, saturate: false },
        Config { label: "no_sat_r20",       r_max: 20, saturate: false },
        Config { label: "no_sat_r16",       r_max: 16, saturate: false },
    ];

    for cfg in &configs {
        run_config(cfg, &train, &test, dim, n, nq, &gt, gt_k, k, ef, po);
    }

    println!("\n=== 历史基线参照 ===");
    println!("  v9.1 baseline:   recall=0.9705  QPS=9195  avg_visited=1227");
    println!("  Glass HNSW:      recall=0.9523  QPS=15171 avg_visited=<150");
    println!("\n如果 no_sat_r24 或 no_sat_r20 的 avg_visited < 1227 且 recall >= 0.95，");
    println!("则 DirectionalPrune 的 r_min 补底方向正确，值得实现。");
}
