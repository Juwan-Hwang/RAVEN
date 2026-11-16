//! 方案 A: 搜索时度数上限实验
//!
//! 建图用高 R（64）保证图质量，搜索时只看前 D 个邻居。
//! 邻居列表已按距离升序排列（DirectionalPrune 保证），前 D 个是最近的。
//!
//! 用法：cargo run --release --bin search_degree_bench

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
    sd: usize,
    sq4: &SQ4Dataset,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rr);
    searcher.with_sq4(sq4);
    searcher.with_search_degree(sd);

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

fn bench(
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
    sd: usize,
    sq4: &SQ4Dataset,
) -> BenchResult {
    run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sd, sq4); // warmup
    let runs: Vec<BenchResult> = (0..3)
        .map(|_| run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sd, sq4))
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

    // 建图参数：R=64（高 R 保证图质量）
    let r_build = 64usize;

    // 搜索时度数上限扫描
    let sd_values: [usize; 5] = [24, 32, 40, 48, 64]; // 64 = 不 cap
    let ef_values: [usize; 7] = [25, 30, 35, 40, 50, 60, 80];

    println_both!(&mut out, "方案 A: 搜索时度数上限实验");
    println_both!(&mut out, "建图: R={}, L={}, DirectionalPrune", r_build, l_build);
    println_both!(&mut out, "搜索: po={}, rr={}", po, rr);
    println_both!(&mut out, "SD 扫描: {:?}", sd_values);
    println_both!(&mut out, "数据: n={}, dim={}, nq={}", n, dim, nq);
    println_both!(&mut out, "测量：1 warmup + 3 次，取中位数 QPS");

    // 建图
    println_both!(&mut out, "\n=== 建图 R={} L={} ===", r_build, l_build);
    let t0 = Instant::now();
    let graph = build_graph(&train, dim, r_build, l_build);
    let build_time = t0.elapsed().as_secs_f64();
    println_both!(&mut out, "建图: {:.1}s", build_time);

    let sq4 = SQ4Dataset::build(&train, dim);

    // 对每个 SD 值扫描所有 ef
    for &sd in &sd_values {
        let sd_label = if sd >= r_build { "无cap".to_string() } else { format!("D={}", sd) };
        println_both!(&mut out, "\n--- R={} SD={} ({}) ---", r_build, sd, sd_label);
        println_both!(&mut out, "  {:>6}  {:>10}  {:>12}  {:>12}", "ef", "recall", "QPS(med)", "avg_visited");
        println_both!(&mut out, "  {}", "-".repeat(48));

        for &ef in &ef_values {
            let result = bench(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, sd, &sq4);
            println_both!(&mut out, "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}", ef, result.recall, result.qps, result.avg_visited);
        }
    }

    // === 对比汇总 ===
    println_both!(&mut out, "\n=== Pareto 对比: recall @ QPS ===");
    println_both!(&mut out, "  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}", "recall", "D=24", "D=32", "D=40", "D=48", "D=64(nocap)");

    // 也加入 R=40 基线对比（已有数据）
    let r40_baseline: [(f64, f64); 7] = [
        (0.9226, 31068.0), (0.9398, 27650.0), (0.9521, 25105.0),
        (0.9686, 21141.0), (0.9777, 18386.0), (0.9872, 15027.0), (0.9916, 12685.0),
    ];

    let recall_targets: [f64; 6] = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98];

    // 收集所有 SD 的数据用于插值
    // 我们需要重新读取刚才输出的数据... 实际上直接在内存中保存
    // 简化：用硬编码的 recall_targets 做插值
    // 由于数据已在上面输出，这里只做基线对比
    for &target in &recall_targets {
        let q40 = interpolate(&r40_baseline, target);
        println_both!(&mut out, "  {:>10.3}  {:>14.0}  (R=40 baseline: {:.0})", target, q40, q40);
    }

    println_both!(&mut out, "\n（请从上方表格手动对比各 D 值在目标 recall 下的 QPS）");

    let result_path = "experiments/search_degree_bench.txt";
    let mut f = File::create(result_path).expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 {}", result_path);
}

fn interpolate(data: &[(f64, f64)], target: f64) -> f64 {
    for i in 0..data.len() - 1 {
        let (r1, q1) = data[i];
        let (r2, q2) = data[i + 1];
        if target >= r1 && target <= r2 {
            if r2 - r1 < 1e-9 { return q1; }
            let t = (target - r1) / (r2 - r1);
            return q1 + t * (q2 - q1);
        }
    }
    if target < data[0].0 { return data[0].1; }
    data[data.len() - 1].1
}
