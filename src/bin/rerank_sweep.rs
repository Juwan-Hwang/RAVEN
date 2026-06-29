//! rerank_n 参数扫描器
//!
//! 建图一次，对每个 rerank_n 值跑全部 10K 查询，输出 QPS + recall。
//! 用法：cargo run --release --bin rerank_sweep

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{LinearPool, VamanaBuildConfig, VamanaGraph};
use raven::memory::VisitedTracker;
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

struct SweepResult {
    rerank_n: usize,
    qps: f64,
    recall: f64,
}

fn run_sweep(
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
    rerank_n: usize,
) -> SweepResult {
    let n = train.len() / dim;
    let mut visited = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);
    let storage = graph.storage();

    let mut hits = 0usize;
    let mut total = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let query_code = sq8.params.encode(query);

        // entry point
        let (entry_point, _nav_dist) = if let Some(nav) = graph.layered_nav() {
            let (ep, dist) = nav.initialize(train, dim, query);
            (ep, Some(dist))
        } else {
            (graph.entry_point(), None)
        };

        let candidates = VamanaGraph::greedy_search_sq8(
            sq8,
            storage,
            entry_point,
            &query_code,
            ef,
            &mut visited,
            &mut pool,
            8, // po=8
        );

        // 部分 rerank
        let actual_n = rerank_n.min(candidates.len());
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(actual_n)
            .map(|(id, _)| {
                let d = l2_simd(query, &train[id as usize * dim..(id as usize + 1) * dim]);
                (id, d)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if results.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
        total += k;
    }
    let dt = t0.elapsed();

    SweepResult {
        rerank_n,
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

    eprintln!("=== rerank_n Sweep (SQ8, ef={}, k={}) ===", ef, k);
    eprintln!("data: n={}, dim={}, nq={}", n, dim, nq);

    // 建图
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
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    eprintln!("build: {:.1}s", t0.elapsed().as_secs_f64());

    // SQ8 编码
    let sq8 = SQ8Dataset::build(&train, dim);

    // warmup
    eprintln!("warmup...");
    let _ = run_sweep(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, 30);

    // 扫描 rerank_n = [15, 20, 25, 28, 30, 32, 35, 38, 40, 45, 50]
    let ns: Vec<usize> = vec![15, 20, 25, 28, 30, 32, 35, 38, 40, 45, 50];
    let mut results = Vec::with_capacity(ns.len());

    for &rn in &ns {
        let r = run_sweep(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, rn);
        eprintln!(
            "rerank_n={:3}: QPS={:.0} recall={:.4}",
            r.rerank_n, r.qps, r.recall
        );
        println!(
            "rerank_n={:3}: QPS={:.0} recall={:.4}",
            r.rerank_n, r.qps, r.recall
        );
        results.push(r);
    }

    // 汇总
    eprintln!("\n=== Summary ===");
    eprintln!("{:<15} {:>10} {:>10} {:>10}", "rerank_n", "QPS", "recall", "Δqps%");
    let baseline_qps = results.last().unwrap().qps; // N=50 as baseline
    for r in &results {
        let delta = (r.qps / baseline_qps - 1.0) * 100.0;
        eprintln!(
            "{:<15} {:>10.0} {:>10.4} {:>+9.1}%",
            r.rerank_n, r.qps, r.recall, delta
        );
    }
}
