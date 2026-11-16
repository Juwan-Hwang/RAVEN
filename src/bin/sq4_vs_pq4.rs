//! SQ4 vs PQ4 头对头对比 — 当前最优参数下
//!
//! SQ4: 64B/vec, AVX2 SIMD 距离计算
//! PQ4: 16B/vec (4x 更小), LUT-ADC 查表距离计算
//!
//! 核心假设：PQ4 的 4x 更小码 → 更好缓存利用 → 更低 ns/visit → 更高 QPS
//! 即使 PQ4 recall 可能略低（量化误差更大），ns/visit 的降低可能补偿
//!
//! 参数：R=40, L=384, po=16, rr=5, alpha_core=0.93, DirectionalPrune
//!
//! 用法：cargo run --release --bin sq4_vs_pq4

use std::io::Write;
use std::time::Instant;

use raven::bench::util::{SiftDataset, compute_recall, median3};
use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::{PQ4Dataset, SQ4Dataset};

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
    run();
    let r1 = run();
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: median3(r1.0, r2.0, r3.0),
        qps: median3(r1.1, r2.1, r3.1),
        visited: median3(r1.2, r2.2, r3.2),
    }
}

fn bench_pq4(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    gt: &[i32], gt_k: usize, ef: usize, pq4: &PQ4Dataset,
) -> BenchResult {
    let run = || {
        let mut s = GraphSearcher::new(train, graph, ef);
        s.with_prefetch_offset(PO);
        s.with_rerank_factor(RR);
        s.with_pq4(pq4);
        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let r = s.search_pq4(&test[q * dim..(q + 1) * dim], K);
            total_visited += s.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        (compute_recall(&results, gt, gt_k, K, nq), nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };
    run();
    let r1 = run();
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: median3(r1.0, r2.0, r3.0),
        qps: median3(r1.1, r2.1, r3.1),
        visited: median3(r1.2, r2.2, r3.2),
    }
}

fn interpolate_qps_at_recall(
    target: f64,
    a_data: &[(usize, BenchResult)],
    b_data: &[(usize, BenchResult)],
) -> Option<(f64, f64, f64)> {
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
    let a_qps = interp(a_data)?;
    let b_qps = interp(b_data)?;
    let delta = (b_qps - a_qps) / a_qps * 100.0;
    Some((a_qps, b_qps, delta))
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

    println_both!("=== SQ4 vs PQ4 头对头对比 (alpha_core=0.93) ===\n");
    println_both!("数据: n={}, dim={}, nq={}, k={}", data.n, data.dim, data.nq, K);
    println_both!("参数: R=40, L=384, po={}, rr={}, DirectionalPrune, alpha_core=0.93", PO, RR);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

    println_both!("索引大小: SQ4={:.1} MB (64B/vec) vs PQ4={:.1} MB (16B/vec, 4x 更小)",
        data.n as f64 * 64.0 / 1e6, data.n as f64 * 16.0 / 1e6);

    // PQ4 训练 + 编码
    print!("PQ4 训练中... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let pq4 = PQ4Dataset::build(&data.train, data.dim, 32); // M=32, sub_dim=4
    println!("{:.1}s", t0.elapsed().as_secs_f64());

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

    let mut sq4_results: Vec<(usize, BenchResult)> = Vec::new();
    let mut pq4_results: Vec<(usize, BenchResult)> = Vec::new();

    println_both!("--- 原始数据 ---\n");
    println_both!("  {:>6}  {:>6}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>8}",
        "ef", "mode", "recall", "QPS", "visited", "ns/vis", "recall", "QPS", "visited", "ns/vis");
    println_both!("  {}", "-".repeat(95));

    for &ef in &EF_LIST {
        let r4 = bench_sq4(&data.train, &graph, &data.test, data.dim, data.nq, &data.gt, data.gt_k, ef, &sq4);
        let rp = bench_pq4(&data.train, &graph, &data.test, data.dim, data.nq, &data.gt, data.gt_k, ef, &pq4);

        let ns4 = 1e9 / r4.qps / r4.visited;
        let nsp = 1e9 / rp.qps / rp.visited;

        println_both!("  {:>6}  {:>6}  {:>8.4}  {:>10.0}  {:>10.1}  {:>6.1}ns  {:>8.4}  {:>10.0}  {:>10.1}  {:>6.1}ns",
            ef, "SQ4", r4.recall, r4.qps, r4.visited, ns4,
            rp.recall, rp.qps, rp.visited, nsp);
        println_both!("  {:>6}  {:>6}  {:>+8.4}  {:>+10.0}  {:>+10.1}  {:>+6.1}ns  {:>8}",
            "", "Δ", rp.recall - r4.recall, rp.qps - r4.qps, rp.visited - r4.visited, nsp - ns4,
            if rp.qps > r4.qps { "PQ4↑" } else { "SQ4↑" });
        println_both!("  {}", "-".repeat(95));

        sq4_results.push((ef, r4));
        pq4_results.push((ef, rp));
    }

    // Pareto 插值对比
    println_both!("\n--- Pareto 插值：相同 recall 下 QPS 对比 ---\n");
    println_both!("  {:>8}  {:>12}  {:>12}  {:>8}  {:>8}", "recall", "SQ4 QPS", "PQ4 QPS", "Δ%", "胜者");
    println_both!("  {}", "-".repeat(55));

    for &target in &[0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99] {
        if let Some((q4, qp, delta)) = interpolate_qps_at_recall(target, &sq4_results, &pq4_results) {
            let winner = if qp > q4 { "PQ4" } else { "SQ4" };
            println_both!("  {:>8.2}  {:>12.0}  {:>12.0}  {:>+7.1}%  {:>8}", target, q4, qp, delta, winner);
        }
    }

    // 竞品对比
    println_both!("\n--- 竞品对比 (recall=0.95, 单线程) ---\n");
    println_both!("  {:>12}  {:>12}  {:>12}  {:>12}", "算法", "QPS", "索引大小", "vs RAVEN");
    println_both!("  {}", "-".repeat(55));

    let best = interpolate_qps_at_recall(0.95, &sq4_results, &pq4_results);
    if let Some((q4, qp, _)) = best {
        let best_qps = q4.max(qp);
        let (best_label, best_size) = if qp > q4 {
            ("PQ4", 16.0)
        } else {
            ("SQ4", 64.0)
        };
        println_both!("  {:>12}  {:>12.0}  {:>10.1} MB  {:>12}", format!("RAVEN-{}", best_label), best_qps, best_size, "—");

        let competitors = [
            ("descartes", 37723.0, 7.07),
            ("qsgngt", 34841.0, 5.44),
            ("kgn", 32185.0, 3.26),
        ];
        for (name, qps, size) in &competitors {
            let ratio = best_qps / qps;
            println_both!("  {:>12}  {:>12.0}  {:>10.2} MB  {:>11.1}%", name, qps, size, ratio * 100.0);
        }
    }

    // 写文件
    let mut f = std::fs::File::create("experiments/sq4_vs_pq4_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/sq4_vs_pq4_result.txt");
}
