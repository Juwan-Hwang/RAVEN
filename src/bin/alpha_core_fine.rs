//! Alpha_core 精细扫描 + R 复核
//!
//! 阶段 1：alpha_core 精细扫描 [0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.95]
//!         只测 ef=50，1 warmup + 3 rounds 取中位数
//! 阶段 2：确认最优 alpha_core 后，R=36/40/44 快速复核
//!
//! 用法：cargo run --release --bin alpha_core_fine

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{
    GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
};
use raven::quant::SQ4Dataset;

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
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
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
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

fn compute_recall(results: &[Vec<u32>], gt: &[i32], gt_k: usize, k: usize, nq: usize) -> f64 {
    let mut hits = 0;
    for q in 0..nq {
        let found = &results[q];
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
}

fn bench_ef(
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
    rr: usize,
    sq4: &SQ4Dataset,
) -> (f64, f64, f64) {
    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rr);
        searcher.with_sq4(sq4);

        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = searcher.search_sq4(query, k);
            total_visited += searcher.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        let recall = compute_recall(&results, gt, gt_k, k, nq);
        (recall, nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };

    run();
    let r1 = run();
    let r2 = run();
    let r3 = run();
    (
        median3(r1.0, r2.0, r3.0),
        median3(r1.1, r2.1, r3.1),
        median3(r1.2, r2.2, r3.2),
    )
}

fn build_and_bench(
    train: &[f32],
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
    alpha_core: f32,
    r_max: usize,
    ef: usize,
) -> (f64, f64, f64, f64) {
    std::env::set_var("RAVEN_ALPHA_CORE", alpha_core.to_string());

    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 384,
        r_max,
        r_soft: (r_max as f32 * 1.5) as usize,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let t0 = Instant::now();
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();

    let (recall, qps, avg_visited) = bench_ef(
        train, &graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sq4,
    );

    (recall, qps, avg_visited, build_time)
}

fn main() {
    let mut out = String::new();

    macro_rules! println_both {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{}", line);
            out.push_str(&line);
            out.push('\n');
        }};
    }

    println_both!("=== RAVEN Alpha_core Fine Sweep + R Verification ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10usize;
    let po = 16usize;
    let rr = 5usize;
    let ef = 50usize;
    let sq4 = SQ4Dataset::build(&train, dim);

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!("固定: L=384, po={}, rr={}, ef={}, SQ4, DirectionalPrune, seed=42", po, rr, ef);
    println_both!("");

    // ════════════════════════════════════════════════════════════════
    //  阶段 1：精细 alpha_core 扫描
    // ════════════════════════════════════════════════════════════════
    const FINE_VALUES: [f32; 7] = [0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.95];

    println_both!("--- 阶段 1：精细 alpha_core 扫描 (R=40, ef=50) ---");
    println_both!("  {:>12}  {:>10}  {:>12}  {:>12}  {:>8}",
        "alpha_core", "recall", "QPS", "avg_visited", "build_s");
    println_both!("  {}", "-".repeat(60));

    let mut fine_results: Vec<(f32, f64, f64, f64)> = Vec::new();

    for &ac in &FINE_VALUES {
        let (recall, qps, visited, build_t) = build_and_bench(
            &train, &test, dim, nq, &gt, gt_k, k, po, rr, &sq4, ac, 40, ef,
        );
        println_both!("  {:>12.2}  {:>10.4}  {:>12.0}  {:>12.1}  {:>8.0}",
            ac, recall, qps, visited, build_t);
        fine_results.push((ac, recall, qps, visited));
    }

    // 找最优 alpha_core
    let best_ac = fine_results.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(ac, _, _, _)| *ac)
        .unwrap_or(0.9);
    let best_recall = fine_results.iter()
        .find(|(ac, _, _, _)| *ac == best_ac)
        .map(|(_, r, _, _)| *r)
        .unwrap_or(0.0);

    println_both!("\n  >>> 最优 alpha_core = {:.2} (recall={:.4})", best_ac, best_recall);

    // ════════════════════════════════════════════════════════════════
    //  阶段 2：R 复核（使用最优 alpha_core）
    // ════════════════════════════════════════════════════════════════
    const R_VALUES: [usize; 3] = [36, 40, 44];

    println_both!("\n--- 阶段 2：R 复核 (alpha_core={:.2}, ef=50) ---", best_ac);
    println_both!("  {:>6}  {:>10}  {:>12}  {:>12}  {:>8}",
        "R", "recall", "QPS", "avg_visited", "build_s");
    println_both!("  {}", "-".repeat(54));

    let mut r_results: Vec<(usize, f64, f64, f64)> = Vec::new();

    for &r in &R_VALUES {
        let (recall, qps, visited, build_t) = build_and_bench(
            &train, &test, dim, nq, &gt, gt_k, k, po, rr, &sq4, best_ac, r, ef,
        );
        println_both!("  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}  {:>8.0}",
            r, recall, qps, visited, build_t);
        r_results.push((r, recall, qps, visited));
    }

    let best_r = r_results.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(r, _, _, _)| *r)
        .unwrap_or(40);

    println_both!("\n  >>> 最优 R = {} (alpha_core={:.2})", best_r, best_ac);

    // ════════════════════════════════════════════════════════════════
    //  汇总
    // ════════════════════════════════════════════════════════════════
    println_both!("\n=======================================================");
    println_both!("  最终结论");
    println_both!("=======================================================");
    println_both!("  最优 alpha_core = {:.2}", best_ac);
    println_both!("  最优 R = {}", best_r);
    println_both!("  recall = {:.4} (vs 旧基线 0.9686, delta = {:+.4})",
        best_recall, best_recall - 0.9686);

    std::env::remove_var("RAVEN_ALPHA_CORE");

    let mut f = File::create("experiments/alpha_core_fine_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/alpha_core_fine_result.txt");
}
