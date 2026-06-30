//! AdaptiveEf A/B Benchmark — powf vs fast_powf + 配置网格扫描
//!
//! 修复之前版本的核心问题：AdaptiveEf ON 时遍历不同 ef 值毫无意义
//! （ef 被 estimate_ef 完全替代），所有行输出完全相同的 avg_ef。
//!
//! 正确做法：
//!   Phase A: 固定 ef 基线曲线（ef=40..100）
//!   Phase B/C: AdaptiveEf 配置网格（gamma × min_ef × max_ef）
//!     Phase B 用原始 powf（不带 fast_powf feature）
//!     Phase C 用 fast_powf（带 fast_powf feature）
//!
//! 搜索路径：SQ8 RAW + f32 rerank（当前最优配置）
//! 建图：nav_m=32, DirectionalPrune, alpha=1.2, L=200, R=32
//!
//! 用法：
//!   cargo run --release --bin adaptive_ef_bench              # Phase B (powf)
//!   cargo run --release --features fast_powf --bin adaptive_ef_bench  # Phase C (fast_powf)

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{
    AdaptiveEfConfig, GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
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

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
    avg_ef: f64,
}

/// 固定 ef 搜索（SQ4 路径 — 当前主力配置）
fn bench_fixed(
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
    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rr);
        searcher.with_sq4(sq4);

        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0u64;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = searcher.search_sq4(query, k);
            total_visited += searcher.last_visited_count() as u64;
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        BenchResult {
            recall: compute_recall(&results, gt, gt_k, k, nq),
            qps: nq as f64 / dt.as_secs_f64(),
            avg_visited: total_visited as f64 / nq as f64,
            avg_ef: ef as f64,
        }
    };

    run(); // warmup
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
        avg_ef: (r2.avg_ef + r3.avg_ef) / 2.0,
    }
}

/// 自适应 ef 搜索（SQ4 路径 — 当前主力配置）
///
/// nominal_ef 设为 max_ef，确保 VisitedTracker 容量足够大，
/// 实际 ef 由 estimate_ef 决定（与 nominal_ef 无关）。
fn bench_adaptive(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    max_ef: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
    config: &AdaptiveEfConfig,
) -> BenchResult {
    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, max_ef);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rr);
        searcher.with_sq4(sq4);
        searcher.with_adaptive_ef(config.clone());

        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0u64;
        let mut total_ef = 0u64;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = searcher.search_sq4(query, k);
            total_visited += searcher.last_visited_count() as u64;
            total_ef += searcher.last_ef_used() as u64;
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        BenchResult {
            recall: compute_recall(&results, gt, gt_k, k, nq),
            qps: nq as f64 / dt.as_secs_f64(),
            avg_visited: total_visited as f64 / nq as f64,
            avg_ef: total_ef as f64 / nq as f64,
        }
    };

    run(); // warmup
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
        avg_ef: (r2.avg_ef + r3.avg_ef) / 2.0,
    }
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

    let mode = if cfg!(feature = "fast_powf") {
        "fast_powf"
    } else {
        "powf"
    };
    println_both!("=== RAVEN AdaptiveEf Benchmark (mode={}) ===\n", mode);

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
    let po = 8usize;
    let rr = 8usize;

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(
        "配置: nav_m=32, DirectionalPrune, alpha=1.2, L=200, R=32, po={}, rr={}",
        po,
        rr
    );
    println_both!("搜索: SQ4 + f32 rerank (当前主力配置)");

    // ── 建图 ──
    println_both!("\n--- 建图 ---");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 48,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println_both!("建图: {:.1}s", t0.elapsed().as_secs_f64());

    // ── SQ4 编码 ──
    let sq4 = SQ4Dataset::build(&train, dim);
    println_both!("SQ4: {:.1} MB", sq4.codes.len() as f64 / 1e6);

    // ── 构建 AdaptiveEfConfig（距离分布采样）──
    println_both!("\n--- 构建 AdaptiveEfConfig ---");
    let nav = graph.layered_nav().expect("layered nav required");
    let t0 = Instant::now();
    let base_config = AdaptiveEfConfig::build_with_layered_nav(&train, dim, nav, 30, 100, 2.0);
    println_both!("构建: {:.1}s", t0.elapsed().as_secs_f64());

    let (min_d, p25, med, p75, max_d) = base_config.distribution_stats();
    println_both!(
        "距离分布: min={:.4} p25={:.4} med={:.4} p75={:.4} max={:.4}",
        min_d,
        p25,
        med,
        p75,
        max_d
    );

    // ══════════════════════════════════════════════════════════════
    // Phase A: 固定 ef 基线
    // ══════════════════════════════════════════════════════════════
    println_both!("\n--- Phase A: 固定 ef 基线 (SQ4) ---");
    println_both!(
        "  {:>6}  {:>10}  {:>12}  {:>12}  {:>8}",
        "ef",
        "recall",
        "QPS",
        "avg_visited",
        "avg_ef"
    );
    println_both!("  {}", "-".repeat(56));

    let ef_list: [usize; 7] = [40, 45, 50, 55, 65, 80, 100];
    let mut baseline_50_qps = 0.0f64;
    let mut baseline_50_recall = 0.0f64;

    for &ef in &ef_list {
        let r = bench_fixed(
            &train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4,
        );
        if ef == 50 {
            baseline_50_qps = r.qps;
            baseline_50_recall = r.recall;
        }
        println_both!(
            "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}  {:>8.1}",
            ef,
            r.recall,
            r.qps,
            r.avg_visited,
            r.avg_ef
        );
    }
    println_both!(
        "\n  基线 ef=50: recall={:.4}  QPS={:.0}",
        baseline_50_recall,
        baseline_50_qps
    );

    // ══════════════════════════════════════════════════════════════
    // Phase B/C: AdaptiveEf 配置网格
    // ══════════════════════════════════════════════════════════════
    let phase_label = if cfg!(feature = "fast_powf") {
        "Phase C: AdaptiveEf ON (fast_powf)"
    } else {
        "Phase B: AdaptiveEf ON (powf)"
    };
    println_both!("\n--- {} ---", phase_label);
    println_both!(
        "  {:>20}  {:>10}  {:>12}  {:>12}  {:>8}  {:>8}",
        "config",
        "recall",
        "QPS",
        "avg_visited",
        "avg_ef",
        "ΔQPS%"
    );
    println_both!("  {}", "-".repeat(80));

    // 配置网格：聚焦 ef=50 工作点附近的参数空间
    // 细粒度 gamma（2.1-2.4）填补 0.95-0.96 recall 间隙——ann-benchmarks 评分最密集区域
    let gammas: [f32; 8] = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0, 3.5];
    let min_efs: [usize; 3] = [30, 35, 40];
    let max_efs: [usize; 4] = [65, 75, 85, 100];

    let mut best_config = String::new();
    let mut best_qps = 0.0f64;
    let mut best_recall = 0.0f64;
    let mut best_avg_ef = 0.0f64;

    for &gamma in &gammas {
        for &min_ef in &min_efs {
            for &max_ef in &max_efs {
                if min_ef >= max_ef {
                    continue;
                }

                let config = base_config.with_params(min_ef, max_ef, gamma);
                let label = format!("γ{:.1}({},{})", gamma, min_ef, max_ef);

                let r = bench_adaptive(
                    &train, &graph, &test, dim, nq, &gt, gt_k, k, max_ef, po, rr, &sq4, &config,
                );

                let delta = (r.qps - baseline_50_qps) / baseline_50_qps * 100.0;
                println_both!(
                    "  {:>20}  {:>10.4}  {:>12.0}  {:>12.1}  {:>8.1}  {:>+7.1}%",
                    label,
                    r.recall,
                    r.qps,
                    r.avg_visited,
                    r.avg_ef,
                    delta
                );

                // 记录 recall >= 基线 0.95 倍时的最佳 QPS
                if r.recall >= baseline_50_recall * 0.95 && r.qps > best_qps {
                    best_qps = r.qps;
                    best_config = label;
                    best_recall = r.recall;
                    best_avg_ef = r.avg_ef;
                }
            }
        }
    }

    // ── 汇总 ──
    println_both!("\n--- 汇总 ---");
    println_both!(
        "  基线 ef=50: recall={:.4}  QPS={:.0}",
        baseline_50_recall,
        baseline_50_qps
    );
    if !best_config.is_empty() {
        let delta = (best_qps - baseline_50_qps) / baseline_50_qps * 100.0;
        println_both!(
            "  最佳自适应: {}  recall={:.4}  QPS={:.0}  Δ={:+.1}%  avg_ef={:.1}",
            best_config,
            best_recall,
            best_qps,
            delta,
            best_avg_ef
        );
    } else {
        println_both!(
            "  (无自适应配置达到 recall >= {:.4})",
            baseline_50_recall * 0.95
        );
    }

    // 写文件
    let suffix = if cfg!(feature = "fast_powf") {
        "_fast_powf"
    } else {
        "_powf"
    };
    let out_path = format!("experiments/adaptive_ef_bench{}.txt", suffix);
    let mut f = File::create(&out_path).expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 {}", out_path);
}
