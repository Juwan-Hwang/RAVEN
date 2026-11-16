//! R=36/44 精细扫描 — 确认 R=32-48 之间的精确峰值
//!
//! 用法：cargo run --release --bin r36_44_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

macro_rules! println_both {
    ($out:expr, $($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{}", line);
        $out.push_str(&line);
        $out.push('\n');
    }};
}

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

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

fn run_once(
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
) -> BenchResult {
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
    BenchResult {
        recall,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
    }
}

fn bench_sq4(
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
) -> BenchResult {
    run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sq4); // warmup
    let runs: Vec<BenchResult> = (0..3)
        .map(|_| run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sq4))
        .collect();
    let mut qps_sorted: Vec<f64> = runs.iter().map(|r| r.qps).collect();
    qps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_qps = qps_sorted[1];
    let avg_recall = runs.iter().map(|r| r.recall).sum::<f64>() / 3.0;
    let avg_visited = runs.iter().map(|r| r.avg_visited).sum::<f64>() / 3.0;
    BenchResult { recall: avg_recall, qps: median_qps, avg_visited }
}

fn build_graph(train: &[f32], dim: usize, r_max: usize, l_build: usize) -> VamanaGraph {
    let mut rng = ChaCha8Rng::seed_from(42);
    let r_soft = (r_max as f32 * 1.5) as usize;
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build,
        r_max,
        r_soft,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    VamanaGraph::build(train, dim, &config, &mut rng)
}

fn main() {
    let mut out = String::new();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10usize;
    let po = 16usize;
    let rr = 5usize;
    let l_build = 384usize;
    let r_values: [usize; 2] = [36, 44];
    let ef_values: [usize; 7] = [30, 35, 40, 50, 60, 80, 100];

    println_both!(&mut out, "RAVEN R=36/44 精细扫描（2 点 × 3 次重复）");
    println_both!(&mut out, "数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(&mut out, "参数: L={}, po={}, rr={}", l_build, po, rr);
    println_both!(&mut out, "测量：1 次 warmup + 3 次正式，取中位数 QPS");

    let mut all_results: std::collections::BTreeMap<usize, Vec<(usize, BenchResult)>> =
        std::collections::BTreeMap::new();

    for (idx, &r) in r_values.iter().enumerate() {
        println_both!(&mut out, "\n=== 建图 {}/{}: R={}, L={} ===", idx + 1, r_values.len(), r, l_build);
        let t0 = Instant::now();
        let graph = build_graph(&train, dim, r, l_build);
        let build_time = t0.elapsed().as_secs_f64();
        println_both!(&mut out, "建图: {:.1}s", build_time);

        let sq4 = SQ4Dataset::build(&train, dim);

        println_both!(&mut out, "\n--- R={} L={} (po={}, rr={}) ---", r, l_build, po, rr);
        println_both!(&mut out, "  {:>6}  {:>10}  {:>12}  {:>12}", "ef", "recall", "QPS(med)", "avg_visited");
        println_both!(out, "  {}", "-".repeat(48));

        let mut r_results: Vec<(usize, BenchResult)> = Vec::new();
        for &ef in &ef_values {
            let result = bench_sq4(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4);
            println_both!(&mut out, "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}", ef, result.recall, result.qps, result.avg_visited);
            r_results.push((ef, result));
        }
        all_results.insert(r, r_results);
        drop(graph);
    }

    // === 对比汇总：加入 R=32/40/48 的已有数据 ===
    println_both!(&mut out, "\n=== 完整 R 值 Pareto 对比（含已有 R=32/40/48 数据）===");
    println_both!(&mut out, "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "recall", "R=32", "R=36", "R=40", "R=44", "R=48");
    println_both!(out, "  {}", "-".repeat(62));

    // R=32/40/48 的已有数据（从 r_sweep_32_64.txt 手动录入）
    let r32_data: [(f64, f64); 7] = [
        (0.9104, 31085.0), (0.9290, 27888.0), (0.9422, 25221.0),
        (0.9609, 21223.0), (0.9709, 18661.0), (0.9830, 15043.0), (0.9886, 12895.0),
    ];
    let r40_data: [(f64, f64); 7] = [
        (0.9226, 31068.0), (0.9398, 27650.0), (0.9521, 25105.0),
        (0.9686, 21141.0), (0.9777, 18386.0), (0.9872, 15027.0), (0.9916, 12685.0),
    ];
    let r48_data: [(f64, f64); 7] = [
        (0.9313, 27931.0), (0.9472, 24065.0), (0.9587, 23292.0),
        (0.9734, 19007.0), (0.9817, 16778.0), (0.9898, 13857.0), (0.9934, 11552.0),
    ];

    let r36_data: &Vec<(usize, BenchResult)> = &all_results[&36];
    let r44_data: &Vec<(usize, BenchResult)> = &all_results[&44];

    let recall_targets: [f64; 6] = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98];
    for &target in &recall_targets {
        let q32 = interpolate(&r32_data, target);
        let q36 = interpolate_bench(r36_data, target);
        let q40 = interpolate(&r40_data, target);
        let q44 = interpolate_bench(r44_data, target);
        let q48 = interpolate(&r48_data, target);
        println_both!(&mut out, "  {:>8.3}  {:>10.0}  {:>10.0}  {:>10.0}  {:>10.0}  {:>10.0}",
            target, q32, q36, q40, q44, q48);
    }

    let result_path = "experiments/r_sweep_36_44.txt";
    let mut f = File::create(result_path).expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 {}", result_path);
}

/// 线性插值：在 (recall, qps) 数据点中找 target recall 对应的 QPS
fn interpolate(data: &[(f64, f64)], target: f64) -> f64 {
    // data 按 recall 升序
    for i in 0..data.len() - 1 {
        let (r1, q1) = data[i];
        let (r2, q2) = data[i + 1];
        if target >= r1 && target <= r2 {
            if r2 - r1 < 1e-9 { return q1; }
            let t = (target - r1) / (r2 - r1);
            return q1 + t * (q2 - q1);
        }
    }
    // 外推
    if target < data[0].0 { return data[0].1; }
    data[data.len() - 1].1
}

fn interpolate_bench(data: &[(usize, BenchResult)], target: f64) -> f64 {
    let pairs: Vec<(f64, f64)> = data.iter().map(|(_, r)| (r.recall, r.qps)).collect();
    interpolate(&pairs, target)
}
