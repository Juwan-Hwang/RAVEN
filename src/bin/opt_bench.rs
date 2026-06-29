//! OPT 系列实验快速 benchmark
//!
//! 只测 SQ8 ef=50 (warmup + 3 rounds, 取中位数)
//! 用法：cargo run --release --bin opt_bench
//!
//! 输出格式：
//!   round 1: QPS=xxxxx recall=0.xxxx
//!   round 2: QPS=xxxxx recall=0.xxxx
//!   round 3: QPS=xxxxx recall=0.xxxx
//!   median: QPS=xxxxx recall=0.xxxx

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, VamanaBuildConfig, VamanaGraph};
use raven::memory::serialize::Serializable;
use raven::quant::SQ8Dataset;

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

struct RoundResult {
    qps: f64,
    recall: f64,
}

fn run_round(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    sq8: &SQ8Dataset,
) -> RoundResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_sq8(sq8);

    let mut hits = 0usize;
    let mut total = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, k);
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

    RoundResult {
        qps: nq as f64 / dt.as_secs_f64(),
        recall: hits as f64 / total as f64,
    }
}

fn main() {
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    let k = 10usize;
    let ef = 50usize;

    eprintln!("=== OPT Benchmark (SQ8, ef={}, k={}) ===", ef, k);
    eprintln!("data: n={}, dim={}, nq={}", n, dim, nq);

    // 建图（带缓存：首次建图后存盘，后续直接加载，消除 ~400s 建图热源）
    let graph_path = std::path::Path::new("data/sift/graph_cache.bin");
    let graph = if graph_path.exists() {
        eprintln!("loading cached graph...");
        let t0 = Instant::now();
        let g = VamanaGraph::load(graph_path).expect("load graph cache");
        eprintln!("load: {:.1}s", t0.elapsed().as_secs_f64());
        g
    } else {
        eprintln!("building graph (first run)...");
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 200,
            r_max: 32,
            r_soft: 48,
            max_iterations: 2,
            saturate: true,
            enable_layered_nav: true,
            nav_m: 16,
            ..Default::default()
        };
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        eprintln!("build: {:.1}s", t0.elapsed().as_secs_f64());
        let _ = g.save(graph_path);
        eprintln!("graph cached to data/sift/graph_cache.bin");
        g
    };

    // SQ8 编码
    let sq8 = SQ8Dataset::build(&train, dim);

    // warmup
    eprintln!("warmup...");
    let _ = run_round(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8);

    // 3 rounds
    let mut rounds = Vec::with_capacity(3);
    for i in 0..3 {
        let r = run_round(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8);
        eprintln!("round {}: QPS={:.0} recall={:.4}", i + 1, r.qps, r.recall);
        rounds.push(r);
    }

    // median
    let mut qps_sorted: Vec<f64> = rounds.iter().map(|r| r.qps).collect();
    qps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_qps = qps_sorted[1];
    let median_recall = rounds[1].recall;

    println!("median: QPS={:.0} recall={:.4}", median_qps, median_recall);
}
