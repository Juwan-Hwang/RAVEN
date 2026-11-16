//! R 值 Pareto 扫描实验（5 点 + 3 次重复）
//!
//! 修正历史分析错误：之前在相同 ef 下对比 R=32/48/64，
//! 但正确对比应在相同 recall 下。R=48 用更低 ef 达到相同 recall。
//!
//! 本实验：R=32/40/48/56/64，全部用 L=384, po=16, rr=5（当前最优参数）
//! 每个数据点重复 3 次测量，取中位数 QPS
//! 用法：cargo run --release --bin r48_bench

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

/// 单次搜索运行
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

/// 3 次重复测量，取中位数 QPS（recall 和 avg_visited 取均值）
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

    let mut runs: Vec<BenchResult> = (0..3)
        .map(|_| run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sq4))
        .collect();

    // 中位数 QPS
    runs.sort_by(|a, b| a.qps.partial_cmp(&b.qps).unwrap());
    let median_qps = runs[1].qps;

    // recall 和 avg_visited 取均值
    let avg_recall = runs.iter().map(|r| r.recall).sum::<f64>() / 3.0;
    let avg_visited = runs.iter().map(|r| r.avg_visited).sum::<f64>() / 3.0;

    let _ = runs.iter().map(|r| r.qps).sum::<f64>() / 3.0; // mean_qps (diagnostic only)

    BenchResult {
        recall: avg_recall,
        qps: median_qps,
        avg_visited,
    }
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

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    let k = 10usize;
    let po = 16usize;
    let rr = 5usize;
    let l_build = 384usize;
    // R=48 可能是甜点，补 R=40/56 确认峰值位置
    let r_values: [usize; 5] = [32, 40, 48, 56, 64];
    let ef_values: [usize; 7] = [30, 35, 40, 50, 60, 80, 100];

    println_both!(&mut out, "RAVEN R 值 Pareto 扫描（5 点 × 3 次重复）");
    println_both!(&mut out, "数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(&mut out, "参数: L={}, po={}, rr={}", l_build, po, rr);
    println_both!(&mut out, "对比方式：相同 recall 下比较 QPS（非相同 ef）");
    println_both!(&mut out, "测量：1 次 warmup + 3 次正式，取中位数 QPS");

    let mut all_results: std::collections::BTreeMap<usize, Vec<(usize, BenchResult)>> =
        std::collections::BTreeMap::new();

    for (idx, &r) in r_values.iter().enumerate() {
        println_both!(
            &mut out,
            "\n=== 建图 {}/{}: R={}, L={} ===",
            idx + 1,
            r_values.len(),
            r,
            l_build
        );
        let t0 = Instant::now();
        let graph = build_graph(&train, dim, r, l_build);
        let build_time = t0.elapsed().as_secs_f64();
        println_both!(&mut out, "建图: {:.1}s", build_time);

        let sq4 = SQ4Dataset::build(&train, dim);

        println_both!(
            &mut out,
            "\n--- R={} L={} (po={}, rr={}) ---",
            r,
            l_build,
            po,
            rr
        );
        println_both!(
            &mut out,
            "  {:>6}  {:>10}  {:>12}  {:>12}",
            "ef",
            "recall",
            "QPS(med)",
            "avg_visited"
        );
        println_both!(out, "  {}", "-".repeat(48));

        let mut r_results: Vec<(usize, BenchResult)> = Vec::new();
        for &ef in &ef_values {
            let result =
                bench_sq4(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4);
            println_both!(
                &mut out,
                "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}",
                ef,
                result.recall,
                result.qps,
                result.avg_visited
            );
            r_results.push((ef, result));
        }

        all_results.insert(r, r_results);
        drop(graph);
    }

    // === Pareto 对比汇总 ===
    println_both!(
        &mut out,
        "\n=== Pareto 对比汇总（相同 recall 下 QPS 对比）==="
    );

    // 收集所有 (recall, qps, R, ef) 数据点
    let mut all_points: Vec<(f64, f64, usize, usize)> = Vec::new();
    for (&r, results) in &all_results {
        for (ef, res) in results {
            all_points.push((res.recall, res.qps, r, *ef));
        }
    }
    // 按 recall 排序
    all_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println_both!(
        &mut out,
        "  {:>8}  {:>6}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "recall",
        "ef",
        "R",
        "QPS",
        "",
        "",
        "",
        "",
        ""
    );

    // 简化输出：每个 R 值的 Pareto 前沿
    println_both!(&mut out, "\n各 R 值 Pareto 前沿点（按 recall 排序）：");
    println_both!(
        &mut out,
        "  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "recall",
        "R=32",
        "R=40",
        "R=48",
        "R=56",
        "R=64"
    );
    println_both!(out, "  {}", "-".repeat(62));

    // 对每个 recall 档位，找各 R 最接近的数据点
    let recall_targets: [f64; 6] = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98];
    for &target in &recall_targets {
        let mut row = format!("  {:>8.3}", target);
        for &r in &r_values {
            let results = &all_results[&r];
            // 找 recall 最接近 target 的点
            let best = results
                .iter()
                .min_by_key(|(_, res)| (res.recall - target).abs() as u64);
            if let Some((_ef, res)) = best {
                row.push_str(&format!("  {:>10.0}", res.qps));
            } else {
                row.push_str("         -");
            }
        }
        println_both!(out, "{}", row);
    }

    // 建图时间汇总
    println_both!(&mut out, "\n建图时间汇总：");
    for &r in &r_values {
        // 重新建图计时太浪费，从日志中提取
        // 这里只输出提示
        println_both!(&mut out, "  R={}: 见上方日志", r);
    }

    let result_path = "experiments/r_sweep_32_64.txt";
    let mut f = File::create(result_path).expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 {}", result_path);
}
