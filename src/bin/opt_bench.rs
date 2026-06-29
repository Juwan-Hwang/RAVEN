//! OPT 系列实验 benchmark（v2 — 长测量窗口）
//!
//! 设计目标：每轮 ~7s（100K queries），5 轮取中位数，方差 < ±3%
//! recall 分离计时：首轮算 recall，后续轮只测 QPS（用 sink 防优化消除）
//!
//! 用法：cargo run --release --bin opt_bench

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

/// 单轮 benchmark 结果
struct RoundResult {
    qps: f64,
    recall: f64,
    elapsed_secs: f64,
}

/// 跑一轮：先算 recall（不计入时间），再纯测 QPS
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
    repeats: usize,
    weighted: bool,
) -> RoundResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_sq8(sq8);

    // ── Pass 0: recall（不计入 timing） ──
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = if weighted {
            searcher.search_sq8_weighted(query, k)
        } else {
            searcher.search_sq8(query, k)
        };
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
        total += k;
    }
    let recall = hits as f64 / total as f64;

    // ── Pass 1..N: 纯 QPS（sink 防优化消除） ──
    let mut sink: u64 = 0;
    let t0 = Instant::now();
    for _ in 0..repeats {
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = if weighted {
                searcher.search_sq8_weighted(query, k)
            } else {
                searcher.search_sq8(query, k)
            };
            sink = sink.wrapping_add(result[0].0 as u64);
        }
    }
    let dt = t0.elapsed();

    // 防编译器消除
    if sink == u64::MAX {
        eprintln!("impossible");
    }

    let total_queries = nq * repeats;
    RoundResult {
        qps: total_queries as f64 / dt.as_secs_f64(),
        recall,
        elapsed_secs: dt.as_secs_f64(),
    }
}

fn main() {
    let weighted = std::env::args().any(|a| a == "--weighted");
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
    const REPEATS: usize = 10; // 10K × 10 = 100K queries/轮 ≈ 7s
    const ROUNDS: usize = 5;

    eprintln!("=== OPT Benchmark v2 (SQ8 {}, ef={}, k={}) ===", if weighted { "WEIGHTED" } else { "RAW" }, ef, k);
    eprintln!("data: n={}, dim={}, nq={}, repeats={}, rounds={}", n, dim, nq, REPEATS, ROUNDS);

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
    let _ = run_round(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, REPEATS, weighted);

    // 5 rounds
    let mut rounds = Vec::with_capacity(ROUNDS);
    for i in 0..ROUNDS {
        let r = run_round(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, REPEATS, weighted);
        eprintln!(
            "round {}: QPS={:.0} recall={:.4} ({:.1}s)",
            i + 1, r.qps, r.recall, r.elapsed_secs
        );
        rounds.push(r);
    }

    // 统计：median, mean, min, max, CV
    let mut qps_vals: Vec<f64> = rounds.iter().map(|r| r.qps).collect();
    qps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = qps_vals[ROUNDS / 2];
    let mean = qps_vals.iter().sum::<f64>() / ROUNDS as f64;
    let min = qps_vals[0];
    let max = qps_vals[ROUNDS - 1];
    let variance = qps_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ROUNDS as f64;
    let cv = variance.sqrt() / mean * 100.0; // 变异系数 %
    let recall = rounds[0].recall;

    eprintln!(
        "stats: median={:.0} mean={:.0} min={:.0} max={:.0} CV={:.1}% recall={:.4}",
        median, mean, min, max, cv, recall
    );

    println!(
        "median: QPS={:.0} recall={:.4} CV={:.1}%",
        median, recall, cv
    );
}
