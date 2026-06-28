//! Adaptive ef benchmark (Phase 4.5)
//!
//! 对比固定 ef vs 自适应 ef（方向 B：距离分布感知 + 幂律变换）。
//! 基线：SQ8 量化搜索 ef=50。
//! 实验：多组 (min_ef, max_ef, gamma) 配置，寻找最优 Pareto 点。
//!
//! 结果自动写入 adaptive_ef_bench_result.txt
//!
//! 用法：cargo run --release --bin adaptive_ef_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{AdaptiveEfConfig, GraphSearcher, VamanaBuildConfig, VamanaGraph};
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

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
    avg_ef: f64,
}

/// 固定 ef SQ8 benchmark
fn bench_fixed_ef(
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

    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, k);
        total_visited += searcher.last_visited_count();

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
        total += k;
    }
    let dt = t0.elapsed();

    BenchResult {
        recall: hits as f64 / total as f64,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
        avg_ef: ef as f64,
    }
}

/// 自适应 ef SQ8 benchmark
///
/// 关键修复：不再单独调用 nav.initialize() 追踪 ef，
/// 改用 searcher.last_ef_used()——零额外开销，QPS 测量不被污染。
fn bench_adaptive_ef(
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
    adaptive_config: &AdaptiveEfConfig,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, nominal_ef);
    searcher.with_sq8(sq8);
    searcher.with_adaptive_ef(adaptive_config.clone());

    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;
    let mut total_ef = 0usize;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];

        // search_sq8 内部一次性调用 nav.initialize()，
        // 同时完成入口点选择 + ef 预测 + 图遍历。
        // last_ef_used() 返回本次搜索实际使用的 ef，零额外开销。
        let result = searcher.search_sq8(query, k);
        total_visited += searcher.last_visited_count();
        total_ef += searcher.last_ef_used();

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
        total += k;
    }
    let dt = t0.elapsed();

    BenchResult {
        recall: hits as f64 / total as f64,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
        avg_ef: total_ef as f64 / nq as f64,
    }
}

/// warm-up + 2 轮取平均
fn bench_stable<F>(bench_fn: F) -> BenchResult
where
    F: Fn() -> BenchResult,
{
    bench_fn(); // warm-up
    let r2 = bench_fn();
    let r3 = bench_fn();
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

    println_both!("=== Adaptive ef Benchmark (Phase 4.5, Direction B + Power-law) ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    println_both!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    // 建图
    println_both!("\n--- 建图 ---");
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
    println_both!("建图: {:.1}s", t0.elapsed().as_secs_f64());

    // SQ8 编码
    println_both!("\n--- SQ8 编码 ---");
    let t0 = Instant::now();
    let sq8 = SQ8Dataset::build(&train, dim);
    println_both!("SQ8 编码: {:.1}s", t0.elapsed().as_secs_f64());

    // 构建自适应 ef 配置（用 LayeredNavigation 真实入口距离）
    let layered_nav = graph.layered_nav()
        .expect("layered nav required for adaptive ef");
    println_both!("\n--- 自适应 ef 配置 ---");
    println_both!("分层导航: max_level={}", layered_nav.max_level());

    // gamma 扫描配置
    // 基线 ef=50, SQ8: recall≈0.965, QPS≈11000
    // gamma>1 把多数查询压到小 ef，目标 avg_ef < 45 且 recall ≥ 0.9603
    let configs: &[(usize, usize, f32, &str)] = &[
        (35, 75, 2.0, "gamma-2"),
        (30, 80, 3.0, "gamma-3"),
        (25, 80, 4.0, "gamma-4"),
        (38, 65, 3.0, "gamma-3-narrow"),
    ];

    // 为每组配置构建 AdaptiveEfConfig 并打印分布统计
    let adaptive_configs: Vec<(usize, usize, f32, &str, AdaptiveEfConfig)> = configs
        .iter()
        .map(|&(min_ef, max_ef, gamma, name)| {
            let config = AdaptiveEfConfig::build_with_layered_nav(
                &train, dim, layered_nav, min_ef, max_ef, gamma);
            let (pmin, p25, median, p75, pmax) = config.distribution_stats();
            println_both!(
                "  [{:>14}] min_ef={:>3} max_ef={:>3} gamma={:.1} | dist: min={:.4} p25={:.4} med={:.4} p75={:.4} max={:.4}",
                name, min_ef, max_ef, gamma, pmin, p25, median, p75, pmax
            );
            (min_ef, max_ef, gamma, name, config)
        })
        .collect();

    // === 基线：固定 ef ===
    println_both!("\n--- 基线：固定 ef (SQ8) ---\n");
    println_both!(
        "  {:>8}  {:>8}  {:>12}  {:>12}  {:>8}",
        "ef", "recall", "QPS", "avg_visited", "avg_ef"
    );
    println_both!("  {}", "-".repeat(60));

    let ef_list = [50, 100];
    let mut baselines = Vec::new();
    for &ef in &ef_list {
        let r = bench_stable(|| {
            bench_fixed_ef(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8)
        });
        println_both!(
            "  {:>8}  {:>8.4}  {:>12.0}  {:>12.1}  {:>8.1}",
            ef, r.recall, r.qps, r.avg_visited, r.avg_ef
        );
        baselines.push((ef, r));
    }

    // === 实验：自适应 ef ===
    println_both!("\n--- 实验：自适应 ef (SQ8 + Power-law) ---\n");
    println_both!(
        "  {:>14}  {:>6}  {:>6}  {:>5}  {:>8}  {:>12}  {:>8}  {:>10}  {:>10}",
        "config", "min_ef", "max_ef", "gamma", "recall", "QPS", "avg_ef", "QPSΔ", "visitedΔ"
    );
    println_both!("  {}", "-".repeat(100));

    // 基线 ef=50 用于对比
    let baseline = &baselines[0].1; // ef=50

    let mut best_qps = 0.0;
    let mut best_config_name = "";

    for &(min_ef, max_ef, gamma, name, ref adaptive_config) in &adaptive_configs {
        let r = bench_stable(|| {
            bench_adaptive_ef(
                &train,
                &graph,
                &test,
                dim,
                nq,
                &gt,
                gt_k,
                k,
                50, // nominal ef (用于 VisitedTracker 容量)
                &sq8,
                adaptive_config,
            )
        });

        let qps_delta = (r.qps - baseline.qps) / baseline.qps * 100.0;
        let visited_delta = r.avg_visited - baseline.avg_visited;

        println_both!(
            "  {:>14}  {:>6}  {:>6}  {:>5.1}  {:>8.4}  {:>12.0}  {:>8.1}  {:>+9.1}%  {:>+10.1}",
            name, min_ef, max_ef, gamma, r.recall, r.qps, r.avg_ef, qps_delta, visited_delta
        );

        // 记录 recall ≥ 基线 recall - 0.005 的最佳 QPS 配置
        if r.recall >= baseline.recall - 0.005 && r.qps > best_qps {
            best_qps = r.qps;
            best_config_name = name;
        }
    }

    // === 终态门判定 ===
    println_both!("\n--- 终态门判定 (§〇.1) ---\n");
    println_both!("  基线: ef=50, recall={:.4}, QPS={:.0}", baseline.recall, baseline.qps);
    println_both!("  最佳配置: {} (QPS={:.0}, recall≥{:.4})", best_config_name, best_qps, baseline.recall - 0.005);

    let qps_improvement = (best_qps - baseline.qps) / baseline.qps * 100.0;
    let passes_qps = qps_improvement >= 5.0;
    let passes_recall = best_qps > 0.0; // recall 已在筛选条件中保证

    println_both!("\n  QPS 提升 ≥ 5%: {}", if passes_qps { "PASS" } else { "FAIL" });
    println_both!("  Recall ≥ 基线 - 0.5pp: {}", if passes_recall { "PASS" } else { "FAIL" });
    println_both!(
        "  终态门: {} (QPS 提升: {:+.1}%)",
        if passes_qps && passes_recall { "✅ PASS" } else { "❌ FAIL" },
        qps_improvement
    );

    // 历史基线
    println_both!("\n  历史基线:");
    println_both!("    SQ8 ef=50 (固定):  recall=0.9653  QPS=11,997  avg_visited=1,233");

    // 写文件（先删旧文件避免锁冲突）
    let _ = std::fs::remove_file("adaptive_ef_bench_result.txt");
    let mut f = File::create("adaptive_ef_bench_result.txt").expect("create result file");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 adaptive_ef_bench_result.txt");
}
