//! Adaptive ef 密集参数扫描
//!
//! 填补 gamma-2 (+9.2%, recall≈0.965) 与 gamma-3-narrow (+14%, recall≈0.960) 之间的 Pareto 空隙。
//!
//! 扫描网格：
//!   gamma:  [2.0, 2.25, 2.5, 2.75, 3.0]
//!   min_ef: [30, 33, 35, 38, 40]
//!   max_ef: [65, 70, 75, 80]
//!   → 100 组自适应配置
//!
//! 外加 11 组固定 ef 基线 (30..80) 用于绘制 fixed-ef Pareto 曲线。
//!
//! 每组配置：warmup + 2 轮取平均（bench_stable）
//! 预计运行时间：~5 分钟
//!
//! 用法：cargo run --release --bin adaptive_ef_sweep

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{AdaptiveEfConfig, GraphSearcher, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ8Dataset;

// ── 数据读取 ──

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = 4 + dim * 4;
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
    let record_bytes = 4 + dim * 4;
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

// ── benchmark 核心 ──

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
    avg_ef: f64,
}

/// 单轮搜索（固定 ef，SQ8 路径）
fn run_once_fixed(
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
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_sq8(sq8);

    let mut hits = 0u64;
    let mut total_visited = 0u64;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, k);
        total_visited += searcher.last_visited_count() as u64;

        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
    }
    let dt = t0.elapsed();

    BenchResult {
        recall: hits as f64 / (nq * k) as f64,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
        avg_ef: ef as f64,
    }
}

/// 单轮搜索（自适应 ef，SQ8 路径）
fn run_once_adaptive(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    nominal_ef: usize,
    sq8: &SQ8Dataset,
    config: &AdaptiveEfConfig,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, nominal_ef);
    searcher.with_sq8(sq8);
    searcher.with_adaptive_ef(config.clone());

    let mut hits = 0u64;
    let mut total_visited = 0u64;
    let mut total_ef = 0u64;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, k);
        total_visited += searcher.last_visited_count() as u64;
        total_ef += searcher.last_ef_used() as u64;

        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
    }
    let dt = t0.elapsed();

    BenchResult {
        recall: hits as f64 / (nq * k) as f64,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
        avg_ef: total_ef as f64 / nq as f64,
    }
}

/// warmup + 2 轮取平均
fn bench_stable<F>(bench_fn: F) -> BenchResult
where
    F: Fn() -> BenchResult,
{
    bench_fn(); // warmup
    let r2 = bench_fn();
    let r3 = bench_fn();
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
        avg_ef: (r2.avg_ef + r3.avg_ef) / 2.0,
    }
}

// ── Pareto 前沿计算 ──

#[derive(Clone)]
struct SweepEntry {
    label: String,
    recall: f64,
    qps: f64,
    avg_ef: f64,
    avg_visited: f64,
    is_pareto: bool,
}

/// 标记 Pareto 前沿：recall 和 qps 都不会被同时超越
fn mark_pareto(entries: &mut [SweepEntry]) {
    for i in 0..entries.len() {
        entries[i].is_pareto = true;
        for j in 0..entries.len() {
            if i == j {
                continue;
            }
            // j 同时优于 i → i 不在 Pareto 前沿
            if entries[j].recall >= entries[i].recall
                && entries[j].qps >= entries[i].qps
                && (entries[j].recall > entries[i].recall || entries[j].qps > entries[i].qps)
            {
                entries[i].is_pareto = false;
                break;
            }
        }
    }
}

// ── main ──

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

    println_both!("=== Adaptive ef 密集参数扫描 ===\n");

    // 读取数据
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
    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);

    // 建图
    print!("建图... ");
    std::io::stdout().flush().ok();
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
    println_both!("{:.1}s", t0.elapsed().as_secs_f64());

    // SQ8 编码
    print!("SQ8 编码... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let sq8 = SQ8Dataset::build(&train, dim);
    println_both!("{:.1}s", t0.elapsed().as_secs_f64());

    // 构建基础自适应配置（获取距离分布）
    let layered_nav = graph.layered_nav().expect("layered nav required");
    let base_config =
        AdaptiveEfConfig::build_with_layered_nav(&train, dim, layered_nav, 35, 75, 2.0);
    let (pmin, p25, median, p75, pmax) = base_config.distribution_stats();
    println_both!(
        "距离分布: min={:.4} p25={:.4} med={:.4} p75={:.4} max={:.4}",
        pmin,
        p25,
        median,
        p75,
        pmax
    );

    // ════════════════════════════════════════
    // Phase 1: 固定 ef 基线曲线
    // ════════════════════════════════════════
    println_both!("\n--- 固定 ef 基线 ---\n");
    println_both!(
        "  {:>6}  {:>8}  {:>10}  {:>10}  {:>8}",
        "ef",
        "recall",
        "QPS",
        "avg_visit",
        "avg_ef"
    );
    println_both!("  {}", "-".repeat(52));

    let ef_list: Vec<usize> = (30..=80).step_by(5).collect();
    let mut fixed_entries: Vec<SweepEntry> = Vec::new();

    for &ef in &ef_list {
        let r =
            bench_stable(|| run_once_fixed(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8));
        println_both!(
            "  {:>6}  {:>8.4}  {:>10.0}  {:>10.1}  {:>8.1}",
            ef,
            r.recall,
            r.qps,
            r.avg_visited,
            r.avg_ef
        );
        fixed_entries.push(SweepEntry {
            label: format!("ef={}", ef),
            recall: r.recall,
            qps: r.qps,
            avg_ef: r.avg_ef,
            avg_visited: r.avg_visited,
            is_pareto: false,
        });
    }

    // 基线 ef=50 用于 QPS delta 对比
    let baseline_qps = fixed_entries
        .iter()
        .find(|e| e.label == "ef=50")
        .map(|e| e.qps)
        .unwrap_or(0.0);
    let baseline_recall = fixed_entries
        .iter()
        .find(|e| e.label == "ef=50")
        .map(|e| e.recall)
        .unwrap_or(0.0);
    println_both!(
        "\n  基线: ef=50, recall={:.4}, QPS={:.0}",
        baseline_recall,
        baseline_qps
    );

    // ════════════════════════════════════════
    // Phase 2: 自适应 ef 网格扫描
    // ════════════════════════════════════════
    let gammas: [f32; 5] = [2.0, 2.25, 2.5, 2.75, 3.0];
    let min_efs: [usize; 5] = [30, 33, 35, 38, 40];
    let max_efs: [usize; 4] = [65, 70, 75, 80];

    let total = gammas.len() * min_efs.len() * max_efs.len();
    println_both!("\n--- 自适应 ef 网格扫描 ({} 组) ---\n", total);

    let mut adaptive_entries: Vec<SweepEntry> = Vec::with_capacity(total);
    let mut idx = 0usize;

    for &gamma in &gammas {
        for &min_ef in &min_efs {
            for &max_ef in &max_efs {
                idx += 1;
                if min_ef >= max_ef {
                    continue;
                }

                let config = base_config.with_params(min_ef, max_ef, gamma);
                let label = format!("γ{}({},{})", gamma, min_ef, max_ef);

                let r = bench_stable(|| {
                    run_once_adaptive(
                        &train, &graph, &test, dim, nq, &gt, gt_k, k,
                        max_ef, // nominal_ef = max_ef 确保 VisitedTracker 不resize
                        &sq8, &config,
                    )
                });

                let qps_delta = (r.qps - baseline_qps) / baseline_qps * 100.0;
                print!(
                    "\r  [{:>3}/{}] {:>20}  recall={:.4}  QPS={:>6.0}  Δ={:+.1}%  ef={:.1}  ",
                    idx, total, label, r.recall, r.qps, qps_delta, r.avg_ef
                );
                std::io::stdout().flush().ok();

                adaptive_entries.push(SweepEntry {
                    label,
                    recall: r.recall,
                    qps: r.qps,
                    avg_ef: r.avg_ef,
                    avg_visited: r.avg_visited,
                    is_pareto: false,
                });
            }
        }
    }
    println!();
    out.push('\n');

    // ════════════════════════════════════════
    // Phase 3: Pareto 前沿分析
    // ════════════════════════════════════════

    let mut all_entries: Vec<SweepEntry> = Vec::new();
    for e in &fixed_entries {
        all_entries.push(SweepEntry {
            label: format!("[fixed] {}", e.label),
            recall: e.recall,
            qps: e.qps,
            avg_ef: e.avg_ef,
            avg_visited: e.avg_visited,
            is_pareto: false,
        });
    }
    for e in &adaptive_entries {
        all_entries.push(SweepEntry {
            label: format!("[adapt] {}", e.label),
            recall: e.recall,
            qps: e.qps,
            avg_ef: e.avg_ef,
            avg_visited: e.avg_visited,
            is_pareto: false,
        });
    }

    mark_pareto(&mut all_entries);

    // 按 QPS 降序输出全部结果
    all_entries.sort_by(|a, b| {
        b.qps
            .partial_cmp(&a.qps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println_both!("\n--- 全部结果（按 QPS 降序）---\n");
    println_both!(
        "  {:>4}  {:>28}  {:>8}  {:>10}  {:>+8}  {:>8}  {:>10}",
        "#",
        "config",
        "recall",
        "QPS",
        "ΔQPS%",
        "avg_ef",
        "avg_visit"
    );
    println_both!("  {}", "-".repeat(90));

    for (i, e) in all_entries.iter().enumerate() {
        let qps_delta = (e.qps - baseline_qps) / baseline_qps * 100.0;
        let marker = if e.is_pareto { "★" } else { " " };
        println_both!(
            "  {:>3}{} {:>28}  {:>8.4}  {:>10.0}  {:>+7.1}%  {:>8.1}  {:>10.1}",
            i + 1,
            marker,
            e.label,
            e.recall,
            e.qps,
            qps_delta,
            e.avg_ef,
            e.avg_visited
        );
    }

    // Pareto 前沿单独输出
    let pareto: Vec<&SweepEntry> = all_entries.iter().filter(|e| e.is_pareto).collect();
    println_both!("\n--- Pareto 前沿（{} 个点）---\n", pareto.len());
    println_both!(
        "  {:>28}  {:>8}  {:>10}  {:>+8}  {:>8}  {:>10}",
        "config",
        "recall",
        "QPS",
        "ΔQPS%",
        "avg_ef",
        "avg_visit"
    );
    println_both!("  {}", "-".repeat(82));

    for e in &pareto {
        let qps_delta = (e.qps - baseline_qps) / baseline_qps * 100.0;
        println_both!(
            "  {:>28}  {:>8.4}  {:>10.0}  {:>+7.1}%  {:>8.1}  {:>10.1}",
            e.label,
            e.recall,
            e.qps,
            qps_delta,
            e.avg_ef,
            e.avg_visited
        );
    }

    // 分 recall 层级找最佳
    println_both!("\n--- 分 recall 层级最佳自适应配置 ---\n");
    let tiers: &[(f64, &str)] = &[
        (0.9650, "recall≥0.9650 (近无损)"),
        (0.9600, "recall≥0.9600 (可接受)"),
        (0.9500, "recall≥0.9500 (低门槛)"),
        (0.9400, "recall≥0.9400 (激进)"),
    ];

    for &(threshold, tier_name) in tiers {
        let best = adaptive_entries
            .iter()
            .filter(|e| e.recall >= threshold)
            .max_by(|a, b| {
                a.qps
                    .partial_cmp(&b.qps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(b) = best {
            let qps_delta = (b.qps - baseline_qps) / baseline_qps * 100.0;
            println_both!(
                "  {:>30}  {:>20}  recall={:.4}  QPS={:.0}  Δ={:+.1}%  avg_ef={:.1}",
                tier_name,
                b.label,
                b.recall,
                b.qps,
                qps_delta,
                b.avg_ef
            );
        } else {
            println_both!("  {:>30}  (无满足条件的配置)", tier_name);
        }
    }

    // 写文件
    let _ = std::fs::remove_file("adaptive_ef_sweep_result.txt");
    let mut f = File::create("adaptive_ef_sweep_result.txt").expect("create result file");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 adaptive_ef_sweep_result.txt");
}
