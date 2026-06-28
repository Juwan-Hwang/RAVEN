//! ef + po 联合参数扫描
//!
//! 只建一次 v9.1 图，然后扫描 ef 和 prefetch offset，
//! 找到 recall-QPS 帕累托最优点。
//!
//! 用法：cargo run --release --bin ef_po_sweep

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

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

/// 单次搜索测量结果
struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

/// 对给定 ef + po 跑 nq 条查询，返回 recall / QPS / avg_visited
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
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);

    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
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
    }
}

/// 跑 3 轮（第 1 轮 warm-up 丢弃），返回后 2 轮的平均
fn bench_stable(
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
) -> BenchResult {
    // warm-up
    bench(train, graph, test, dim, nq, gt, gt_k, k, ef, po);

    let r2 = bench(train, graph, test, dim, nq, gt, gt_k, k, ef, po);
    let r3 = bench(train, graph, test, dim, nq, gt, gt_k, k, ef, po);

    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
    }
}

fn main() {
    println!("=== ef + po 联合参数扫描 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    // === 建图：v9.1 分层导航（只建一次）===
    println!("\n--- 建图: v9.1 分层导航 ---");
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
    println!("建图: {:.1}s", t0.elapsed().as_secs_f64());
    if let Some(nav) = graph.layered_nav() {
        println!("[nav] max_level={}", nav.max_level());
    }

    // ================================================================
    // Phase 1: ef 扫描（固定 po=8）
    // ================================================================
    println!("\n{}", "=".repeat(70));
    println!("Phase 1: ef 扫描 (po=8 固定)");
    println!("{}", "=".repeat(70));

    let ef_list = [20, 30, 40, 50, 60, 80];
    let mut ef_results: Vec<(usize, BenchResult)> = Vec::new();

    // 表头
    println!(
        "\n  {:>6}  {:>8}  {:>8}  {:>12}",
        "ef", "recall", "QPS", "avg_visited"
    );
    println!("  {}", "-".repeat(42));

    for &ef in &ef_list {
        let r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, 8);
        println!(
            "  {:>6}  {:>8.4}  {:>8.0}  {:>12.1}",
            ef, r.recall, r.qps, r.avg_visited
        );
        ef_results.push((ef, r));
    }

    // 找最优 ef：recall >= 0.97 中 QPS 最高的
    let best_ef = ef_results
        .iter()
        .filter(|(_, r)| r.recall >= 0.97)
        .max_by(|a, b| a.1.qps.partial_cmp(&b.1.qps).unwrap())
        .map(|(ef, _)| *ef)
        .unwrap_or(50);

    println!(
        "\n  → 最优 ef={} (recall≥0.97 中 QPS 最高)",
        best_ef
    );

    // ================================================================
    // Phase 2: po 扫描（在 ef=50 和 ef=100 两个关键工作点分别扫描）
    // ================================================================
    let po_list = [0, 1, 2, 3, 4, 6, 8, 12, 16, 32];

    // all_po_results: (ef, po, BenchResult)
    let mut all_po_results: Vec<(usize, usize, BenchResult)> = Vec::new();

    for &sweep_ef in &[50, 100] {
        println!("\n{}", "=".repeat(70));
        println!("Phase 2: po 扫描 (ef={} 固定)", sweep_ef);
        println!("{}", "=".repeat(70));

        println!(
            "\n  {:>6}  {:>8}  {:>8}  {:>12}",
            "po", "recall", "QPS", "avg_visited"
        );
        println!("  {}", "-".repeat(42));

        for &po in &po_list {
            let r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, sweep_ef, po);
            println!(
                "  {:>6}  {:>8.4}  {:>8.0}  {:>12.1}",
                po, r.recall, r.qps, r.avg_visited
            );
            all_po_results.push((sweep_ef, po, BenchResult {
                recall: r.recall,
                qps: r.qps,
                avg_visited: r.avg_visited,
            }));
        }

        let baseline = all_po_results.iter()
            .find(|(ef, po, _)| *ef == sweep_ef && *po == 8)
            .map(|(_, _, r)| r)
            .unwrap();
        let best = all_po_results.iter()
            .filter(|(ef, _, _)| *ef == sweep_ef)
            .max_by(|a, b| a.2.qps.partial_cmp(&b.2.qps).unwrap())
            .unwrap();
        let improvement = (best.2.qps / baseline.qps - 1.0) * 100.0;
        println!(
            "\n  -> 最优 po={} (QPS={:.0}, vs po=8 QPS={:.0}, {:+.1}%)",
            best.1, best.2.qps, baseline.qps, improvement
        );
    }

    // ================================================================
    // 汇总：各 ef 的最优 po
    // ================================================================
    println!("\n{}", "=".repeat(70));
    println!("汇总：各 ef 工作点的最优 po");
    println!("{}", "=".repeat(70));

    println!(
        "\n  {:>4}  {:>4}  {:>8}  {:>10}  {:>10}  {:>8}",
        "ef", "po", "recall", "QPS", "vs po=8", "visited"
    );
    println!("  {}", "-".repeat(52));

    for &sweep_ef in &[50, 100] {
        let baseline = all_po_results.iter()
            .find(|(ef, po, _)| *ef == sweep_ef && *po == 8)
            .map(|(_, _, r)| r)
            .unwrap();
        let best = all_po_results.iter()
            .filter(|(ef, _, _)| *ef == sweep_ef)
            .max_by(|a, b| a.2.qps.partial_cmp(&b.2.qps).unwrap())
            .unwrap();
        let improvement = (best.2.qps / baseline.qps - 1.0) * 100.0;
        println!(
            "  {:>4}  {:>4}  {:>8.4}  {:>10.0}  {:>+9.1}%  {:>8.1}",
            sweep_ef, best.1, best.2.recall, best.2.qps, improvement, best.2.avg_visited
        );
    }

    println!("\n  历史基线 (po=8 硬编码):");
    println!("    H20 ef=50 po=8:  recall=0.9705  QPS=9,195  avg_visited=1,227");
    println!("    Glass HNSW (H20):  recall=0.9465  QPS=7,678  avg_visited=1,041");
    println!("    Glass Optimize(): po=2 -> +30.13% QPS");
}
