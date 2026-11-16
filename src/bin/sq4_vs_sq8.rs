//! SQ4 vs SQ8 头对头对比 — 当前最优参数下
//!
//! 核心假设：SQ4 的 4-bit 量化噪声导致搜索访问大量"假近邻"，
//! SQ8 的 16x 精细量化可能减少 avg_visited，即使单次访问略贵，
//! 总 QPS 也可能更高。
//!
//! 参数：R=40, L=384, po=16, rr=5, alpha_core=0.93, DirectionalPrune
//! 对比：SQ4 (64B/vec) vs SQ8 (128B/vec)，相同 ef 扫描
//!
//! 用法：cargo run --release --bin sq4_vs_sq8

use std::io::Write;
use std::time::Instant;

use raven::bench::util::{SiftDataset, compute_recall, median3};
use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::{SQ4Dataset, SQ8Dataset};

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;
const EF_LIST: [usize; 7] = [25, 30, 35, 40, 50, 65, 80];

struct BenchResult {
    recall: f64,
    qps: f64,
    visited: f64,
}

fn bench_sq4(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    gt: &[i32], gt_k: usize, ef: usize, sq4: &SQ4Dataset,
) -> BenchResult {
    let run = || {
        let mut s = GraphSearcher::new(train, graph, ef);
        s.with_prefetch_offset(PO);
        s.with_rerank_factor(RR);
        s.with_sq4(sq4);
        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let r = s.search_sq4(&test[q * dim..(q + 1) * dim], K);
            total_visited += s.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        (compute_recall(&results, gt, gt_k, K, nq), nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };
    run(); // warmup
    let r1 = run();
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: median3(r1.0, r2.0, r3.0),
        qps: median3(r1.1, r2.1, r3.1),
        visited: median3(r1.2, r2.2, r3.2),
    }
}

fn bench_sq8(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    gt: &[i32], gt_k: usize, ef: usize, sq8: &SQ8Dataset,
) -> BenchResult {
    let run = || {
        let mut s = GraphSearcher::new(train, graph, ef);
        s.with_prefetch_offset(PO);
        s.with_rerank_factor(RR);
        s.with_sq8(sq8);
        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let r = s.search_sq8(&test[q * dim..(q + 1) * dim], K);
            total_visited += s.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        (compute_recall(&results, gt, gt_k, K, nq), nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };
    run(); // warmup
    let r1 = run();
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: median3(r1.0, r2.0, r3.0),
        qps: median3(r1.1, r2.1, r3.1),
        visited: median3(r1.2, r2.2, r3.2),
    }
}

/// 在 recall vs visited 平面上做 Pareto 插值对比
/// 返回 (sq4_qps_at_recall, sq8_qps_at_recall, delta_pct)
fn interpolate_qps_at_recall(
    target: f64,
    sq4_data: &[(usize, BenchResult)],
    sq8_data: &[(usize, BenchResult)],
) -> Option<(f64, f64, f64)> {
    // 按 recall 排序后线性插值 QPS
    let interp = |data: &[(usize, BenchResult)]| -> Option<f64> {
        let mut sorted: Vec<&BenchResult> = data.iter().map(|(_, r)| r).collect();
        sorted.sort_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap());
        if target <= sorted.first()?.recall {
            return Some(sorted.first()?.qps);
        }
        if target >= sorted.last()?.recall {
            return Some(sorted.last()?.qps);
        }
        for i in 0..sorted.len() - 1 {
            if sorted[i].recall <= target && target <= sorted[i + 1].recall {
                let t = (target - sorted[i].recall) / (sorted[i + 1].recall - sorted[i].recall);
                return Some(sorted[i].qps * (1.0 - t) + sorted[i + 1].qps * t);
            }
        }
        None
    };
    let sq4_qps = interp(sq4_data)?;
    let sq8_qps = interp(sq8_data)?;
    let delta = (sq8_qps - sq4_qps) / sq4_qps * 100.0;
    Some((sq4_qps, sq8_qps, delta))
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

    let data = SiftDataset::load();
    let sq4 = SQ4Dataset::build(&data.train, data.dim);
    let sq8 = SQ8Dataset::build(&data.train, data.dim);

    println_both!("=== SQ4 vs SQ8 头对头对比 (alpha_core=0.93) ===\n");
    println_both!("数据: n={}, dim={}, nq={}, k={}", data.n, data.dim, data.nq, K);
    println_both!("参数: R=40, L=384, po={}, rr={}, DirectionalPrune, alpha_core=0.93", PO, RR);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

    println_both!("索引大小: SQ4={:.1} MB (64B/vec) vs SQ8={:.1} MB (128B/vec)",
        data.n as f64 * 64.0 / 1e6, data.n as f64 * 128.0 / 1e6);

    // 建图
    std::env::set_var("RAVEN_ALPHA_CORE", "0.93");
    println_both!("\n建图中... ");
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 384,
        r_max: 40,
        r_soft: 60,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&data.train, data.dim, &config, &mut rng);
    println_both!("建图: {:.0}s\n", t0.elapsed().as_secs_f64());

    // 扫描
    let mut sq4_results: Vec<(usize, BenchResult)> = Vec::new();
    let mut sq8_results: Vec<(usize, BenchResult)> = Vec::new();

    println_both!("--- 原始数据 ---\n");
    println_both!("  {:>6}  {:>6}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>8}",
        "ef", "mode", "recall", "QPS", "visited", "ns/vis", "recall", "QPS", "visited", "ns/vis");
    println_both!("  {}", "-".repeat(95));

    for &ef in &EF_LIST {
        let r4 = bench_sq4(&data.train, &graph, &data.test, data.dim, data.nq, &data.gt, data.gt_k, ef, &sq4);
        let r8 = bench_sq8(&data.train, &graph, &data.test, data.dim, data.nq, &data.gt, data.gt_k, ef, &sq8);

        let ns4 = 1e9 / r4.qps / r4.visited;
        let ns8 = 1e9 / r8.qps / r8.visited;

        println_both!("  {:>6}  {:>6}  {:>8.4}  {:>10.0}  {:>10.1}  {:>6.1}ns  {:>8.4}  {:>10.0}  {:>10.1}  {:>6.1}ns",
            ef, "SQ4", r4.recall, r4.qps, r4.visited, ns4,
            r8.recall, r8.qps, r8.visited, ns8);
        println_both!("  {:>6}  {:>6}  {:>+8.4}  {:>+10.0}  {:>+10.1}  {:>+6.1}ns  {:>8}",
            "", "Δ", r8.recall - r4.recall, r8.qps - r4.qps, r8.visited - r4.visited, ns8 - ns4,
            if r8.qps > r4.qps { "SQ8↑" } else { "SQ4↑" });
        println_both!("  {}", "-".repeat(95));

        sq4_results.push((ef, r4));
        sq8_results.push((ef, r8));
    }

    // Pareto 插值对比
    println_both!("\n--- Pareto 插值：相同 recall 下 QPS 对比 ---\n");
    println_both!("  {:>8}  {:>12}  {:>12}  {:>8}  {:>8}", "recall", "SQ4 QPS", "SQ8 QPS", "Δ%", "胜者");
    println_both!("  {}", "-".repeat(55));

    for &target in &[0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99] {
        if let Some((q4, q8, delta)) = interpolate_qps_at_recall(target, &sq4_results, &sq8_results) {
            let winner = if q8 > q4 { "SQ8" } else { "SQ4" };
            println_both!("  {:>8.2}  {:>12.0}  {:>12.0}  {:>+7.1}%  {:>8}", target, q4, q8, delta, winner);
        }
    }

    // 关键指标总结
    println_both!("\n--- 关键指标总结 ---\n");

    // 找 recall≈0.95 的点
    for (ef, r4) in &sq4_results {
        if (r4.recall - 0.95).abs() < 0.01 {
            let r8 = sq8_results.iter().find(|(e, _)| e == ef).map(|(_, r)| r);
            if let Some(r8) = r8 {
                println_both!("  recall≈0.95 (ef={}):", ef);
                println_both!("    SQ4: recall={:.4} QPS={:.0} visited={:.0} ns/visit={:.1}",
                    r4.recall, r4.qps, r4.visited, 1e9 / r4.qps / r4.visited);
                println_both!("    SQ8: recall={:.4} QPS={:.0} visited={:.0} ns/visit={:.1}",
                    r8.recall, r8.qps, r8.visited, 1e9 / r8.qps / r8.visited);
                println_both!("    visited 比: {:.2}x (SQ8/SQ4)", r8.visited / r4.visited);
                println_both!("    QPS 比: {:.2}x (SQ8/SQ4)", r8.qps / r4.qps);
            }
        }
    }

    // 竞品对比
    println_both!("\n--- 竞品对比 (recall=0.95, 单线程) ---\n");
    println_both!("  {:>12}  {:>12}  {:>12}  {:>12}", "算法", "QPS", "索引大小", "vs RAVEN");
    println_both!("  {}", "-".repeat(55));

    // 用 SQ4/SQ8 在 recall=0.95 的最佳 QPS
    let best_sq4 = interpolate_qps_at_recall(0.95, &sq4_results, &sq8_results);
    if let Some((q4, q8, _)) = best_sq4 {
        let best = q4.max(q8);
        let best_label = if q8 > q4 { "SQ8" } else { "SQ4" };
        let best_size = if q8 > q4 { 128.0 } else { 64.0 };
        println_both!("  {:>12}  {:>12.0}  {:>10.1} MB  {:>12}", format!("RAVEN-{}", best_label), best, best_size, "—");

        let competitors = [
            ("descartes", 37723.0, 7.07),
            ("qsgngt", 34841.0, 5.44),
            ("kgn", 32185.0, 3.26),
        ];
        for (name, qps, size) in &competitors {
            let ratio = best / qps;
            println_both!("  {:>12}  {:>12.0}  {:>10.2} MB  {:>11.1}%", name, qps, size, ratio * 100.0);
        }
    }

    // 写文件
    let mut f = std::fs::File::create("experiments/sq4_vs_sq8_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/sq4_vs_sq8_result.txt");
}
