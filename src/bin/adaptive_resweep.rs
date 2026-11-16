//! AdaptiveEf 重新扫描 — 基于新 alpha_core=0.93 曲线
//!
//! 旧 8 组 γ 配置基于 alpha_core=1.0 调优，新曲线上移后需重新选择填隙点。
//!
//! 三大目标区域：
//! 1. 低 recall 间隙（0.954-0.964）：min_ef=25/30, max_ef=45/50
//! 2. 中 recall 现有区域（0.967-0.977）：复测旧配置 + 微调
//! 3. 高 recall 间隙（0.977-0.985）：min_ef=35, max_ef=90/95/100
//!
//! 输出：合并曲线1固定 ef 点后的完整 Pareto 前沿，自动选出最优 8 组填隙。
//!
//! 用法：cargo run --release --bin adaptive_resweep

use std::io::Write;
use std::time::Instant;

use raven::bench::util::{SiftDataset, compute_recall, median3};
use raven::build::ChaCha8Rng;
use raven::graph::{AdaptiveEfConfig, GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;
const FIXED_EF: [usize; 9] = [35, 40, 45, 50, 55, 65, 70, 80, 100];

/// (gamma, min_ef, max_ef)
const SWEEP_CONFIGS: &[(f32, usize, usize)] = &[
    // ── 区域 1：低 recall 间隙 (0.954-0.964) ──
    (3.5, 25, 45),
    (2.5, 25, 45),
    (3.5, 25, 50),
    (2.5, 25, 50),
    (3.5, 30, 50),
    (2.5, 30, 50),
    (2.0, 30, 50),
    (3.5, 30, 55),
    (2.5, 30, 55),
    // ── 区域 2：旧配置复测 ──
    (3.5, 35, 65),
    (2.4, 35, 65),
    (2.0, 35, 65),
    (2.3, 35, 75),
    (2.0, 35, 75),
    (2.3, 35, 85),
    (2.0, 35, 85),
    // ── 区域 2b：微调 ──
    (3.0, 35, 60),
    (2.5, 35, 60),
    (2.0, 35, 60),
    // ── 区域 3：高 recall 间隙 (0.977-0.985) ──
    (3.0, 35, 90),
    (2.5, 35, 90),
    (2.0, 35, 90),
    (3.0, 35, 95),
    (2.5, 35, 95),
    (2.0, 35, 95),
    (3.0, 35, 100),
    (2.0, 35, 100),
    (2.5, 40, 90),
    (2.0, 40, 90),
    (2.5, 40, 100),
    (2.0, 40, 100),
];

struct Point {
    label: String,
    recall: f64,
    qps: f64,
    visited: f64,
    is_fixed: bool,
}

fn bench_fixed(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    gt: &[i32], gt_k: usize, ef: usize, sq4: &SQ4Dataset,
) -> (f64, f64, f64) {
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
    (median3(r1.0, r2.0, r3.0), median3(r1.1, r2.1, r3.1), median3(r1.2, r2.2, r3.2))
}

fn bench_adaptive(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    gt: &[i32], gt_k: usize, max_ef: usize, sq4: &SQ4Dataset,
    config: &AdaptiveEfConfig,
) -> (f64, f64, f64) {
    let run = || {
        let mut s = GraphSearcher::new(train, graph, max_ef);
        s.with_prefetch_offset(PO);
        s.with_rerank_factor(RR);
        s.with_sq4(sq4);
        s.with_adaptive_ef(config.clone());
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
    (median3(r1.0, r2.0, r3.0), median3(r1.1, r2.1, r3.1), median3(r1.2, r2.2, r3.2))
}

/// 从所有点中选出 Pareto 前沿（最大化 recall 和 QPS）
///
/// 正确算法：按 recall 降序排列，保留 QPS 严格递增的点。
/// 即从高 recall 往低 recall 扫，只有 QPS 比之前见过的都高才入选。
fn pareto_frontier(points: &[Point]) -> Vec<&Point> {
    let mut sorted: Vec<&Point> = points.iter().collect();
    // recall 降序，同 recall 时 QPS 降序（取 QPS 最高的）
    sorted.sort_by(|a, b| {
        b.recall.partial_cmp(&a.recall).unwrap()
            .then(b.qps.partial_cmp(&a.qps).unwrap())
    });

    let mut frontier = Vec::new();
    let mut max_qps = f64::NEG_INFINITY;
    for p in &sorted {
        if p.qps > max_qps {
            frontier.push(*p);
            max_qps = p.qps;
        }
    }
    // 反转为 recall 升序，便于阅读
    frontier.reverse();
    frontier
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

    println_both!("=== RAVEN AdaptiveEf Re-sweep (alpha_core=0.93) ===\n");
    println_both!("数据: n={}, dim={}, nq={}, k={}", data.n, data.dim, data.nq, K);
    println_both!("固定: L=384, po={}, rr={}, SQ4, DirectionalPrune, alpha_core=0.93", PO, RR);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

    // 建图
    std::env::set_var("RAVEN_ALPHA_CORE", "0.93");
    println_both!("建图中... ");
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

    // AdaptiveEf 基础配置
    let nav = graph.layered_nav().expect("layered nav required");
    let adaptive_base = AdaptiveEfConfig::build_with_layered_nav(&data.train, data.dim, nav, 30, 100, 2.0);

    let mut all_points: Vec<Point> = Vec::new();

    // 曲线 1：固定 ef
    println_both!("--- 曲线 1: 固定 ef ---");
    println_both!("  {:>6}  {:>10}  {:>10}  {:>10}", "ef", "recall", "QPS", "visited");
    println_both!("  {}", "-".repeat(42));
    for &ef in &FIXED_EF {
        let (recall, qps, visited) = bench_fixed(
            &data.train, &graph, &data.test, data.dim, data.nq,
            &data.gt, data.gt_k, ef, &sq4,
        );
        println_both!("  {:>6}  {:>10.4}  {:>10.0}  {:>10.1}", ef, recall, qps, visited);
        all_points.push(Point {
            label: format!("ef={}", ef),
            recall, qps, visited,
            is_fixed: true,
        });
    }

    // 扫描 adaptive 配置
    println_both!("\n--- Adaptive 扫描 ({} 组配置) ---", SWEEP_CONFIGS.len());
    println_both!("  {:>18}  {:>10}  {:>10}  {:>10}", "config", "recall", "QPS", "visited");
    println_both!("  {}", "-".repeat(54));
    for &(gamma, min_ef, max_ef) in SWEEP_CONFIGS {
        let cfg = adaptive_base.with_params(min_ef, max_ef, gamma);
        let (recall, qps, visited) = bench_adaptive(
            &data.train, &graph, &data.test, data.dim, data.nq,
            &data.gt, data.gt_k, max_ef, &sq4, &cfg,
        );
        let label = format!("γ{:.1}({},{})", gamma, min_ef, max_ef);
        println_both!("  {:>18}  {:>10.4}  {:>10.0}  {:>10.1}", label, recall, qps, visited);
        all_points.push(Point {
            label,
            recall, qps, visited,
            is_fixed: false,
        });
    }

    // ════════════════════════════════════════════════════════════════
    //  Pareto 分析
    // ════════════════════════════════════════════════════════════════
    let frontier = pareto_frontier(&all_points);

    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  Pareto 前沿（全部点合并后）");
    println_both!("═══════════════════════════════════════════════════════════════\n");
    println_both!("  {:>6}  {:>18}  {:>10}  {:>10}  {:>10}", "#", "config", "recall", "QPS", "visited");
    println_both!("  {}", "-".repeat(60));

    let mut adaptive_on_frontier: Vec<&Point> = Vec::new();
    for (i, p) in frontier.iter().enumerate() {
        let tag = if p.is_fixed { "FIX" } else { "ADP" };
        println_both!("  {:>3}{}  {:>18}  {:>10.4}  {:>10.0}  {:>10.1}",
            i + 1, tag, p.label, p.recall, p.qps, p.visited);
        if !p.is_fixed {
            adaptive_on_frontier.push(*p);
        }
    }

    // 选出最优 8 组 adaptive（在 Pareto 前沿上的 adaptive 点）
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  最优 Adaptive 填隙点（Pareto 前沿上的 adaptive 点）");
    println_both!("═══════════════════════════════════════════════════════════════\n");

    if adaptive_on_frontier.is_empty() {
        println_both!("  无 adaptive 点在 Pareto 前沿上！");
    } else {
        println_both!("  {:>4}  {:>18}  {:>10}  {:>10}  {:>10}", "#", "config", "recall", "QPS", "visited");
        println_both!("  {}", "-".repeat(56));
        for (i, p) in adaptive_on_frontier.iter().enumerate() {
            println_both!("  {:>4}  {:>18}  {:>10.4}  {:>10.0}  {:>10.1}",
                i + 1, p.label, p.recall, p.qps, p.visited);
        }
    }

    // 额外信息：被支配的固定 ef 点
    println_both!("\n--- 被支配的固定 ef 点（adaptive 点提供了更好的 recall/QPS 组合）---");
    let frontier_labels: Vec<&str> = frontier.iter().map(|p| p.label.as_str()).collect();
    for p in &all_points {
        if p.is_fixed && !frontier_labels.contains(&p.label.as_str()) {
            // 找支配它的 adaptive 点
            let dominator = all_points.iter()
                .find(|q| !q.is_fixed && q.recall >= p.recall && q.qps >= p.qps);
            if let Some(dom) = dominator {
                println_both!("  {} ({:.4}, {:.0}) ← 被 {} ({:.4}, {:.0}) 支配",
                    p.label, p.recall, p.qps, dom.label, dom.recall, dom.qps);
            }
        }
    }

    // 写文件
    let mut f = std::fs::File::create("experiments/adaptive_resweep_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/adaptive_resweep_result.txt");
}
