//! Alpha_core 参数扫描 — Pass 1 核心方向性判据的遮挡阈值
//!
//! 这是真正决定图拓扑的参数：
//!   判据: d²(j,i) < alpha_core · d²(i,q) → j 方向性遮挡 i（剪枝）
//!
//! - alpha_core = 1.0（默认）：标准方向性判据，等价 Glass Heuristic2
//! - alpha_core < 1.0：阈值更严 → 更难遮挡 → 更多边存活 → 图更密
//! - alpha_core > 1.0：阈值更松 → 更易遮挡 → 更少边存活 → 图更稀
//!
//! 对比第十二轮的 alpha 扫描：那个测的是 Pass 2 的 backfill_alpha（死参数），
//! 这次测的是 Pass 1 的 alpha_core（核心判据）。
//!
//! 方法论：多组 alpha_core 值建图，每组在 6 个 ef 点上测量
//! 1 warmup + 3 rounds 取中位数
//! 基线：R=40, L=384, po=16, rr=5, DirectionalPrune, SQ4, 单线程
//!
//! 用法：cargo run --release --bin alpha_core_sweep

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{
    GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
};
use raven::quant::SQ4Dataset;

// ═══════════════════════════════════════════════════════════════════════
//  I/O 与 recall（复用 submission_bench 逻辑）
// ═══════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════
//  Benchmark：1 warmup + 3 rounds 取中位数
// ═══════════════════════════════════════════════════════════════════════

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
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
    let run = || {
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
    };

    run(); // warmup
    let r1 = run();
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: median3(r1.recall, r2.recall, r3.recall),
        qps: median3(r1.qps, r2.qps, r3.qps),
        avg_visited: median3(r1.avg_visited, r2.avg_visited, r3.avg_visited),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════

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

    println_both!("=== RAVEN Alpha_core Sweep (Pass 1 core criterion) ===\n");

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

    // alpha_core 扫描范围
    // 1.0 = 当前默认（标准方向性判据）
    // <1.0 = 更严（更多边存活，图更密）
    // >1.0 = 更松（更少边存活，图更稀，可能触发 Pass 2）
    const ALPHA_CORE_VALUES: [f32; 7] = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
    const EF_VALUES: [usize; 6] = [35, 40, 45, 50, 55, 65];

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(
        "配置: DirectionalPrune, L=384, R=40, po={}, rr={}, nav_m=32, saturate=false, alpha(backfill)=1.2",
        po, rr
    );
    println_both!("Alpha_core 值: {:?}", ALPHA_CORE_VALUES);
    println_both!("EF 值: {:?}", EF_VALUES);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

    // 编码 SQ4（所有 alpha_core 共用同一个编码）
    let sq4 = SQ4Dataset::build(&train, dim);

    // 存储所有结果
    let mut all_results: Vec<(f32, Vec<(usize, BenchResult)>)> = Vec::new();

    for &alpha_core in &ALPHA_CORE_VALUES {
        println_both!("--- alpha_core={:.1} ---", alpha_core);

        // 设置环境变量，prune_dispatch 会读取
        std::env::set_var("RAVEN_ALPHA_CORE", alpha_core.to_string());

        // 建图
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2, // backfill_alpha（不影响 Pass 1）
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
        let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
        let build_time = t0.elapsed().as_secs_f64();
        println_both!("  建图: {:.1}s", build_time);

        // 测量每个 ef 点
        println_both!(
            "  {:>6}  {:>10}  {:>12}  {:>12}",
            "ef", "recall", "QPS", "avg_visited"
        );
        println_both!("  {}", "-".repeat(48));

        let mut ef_results: Vec<(usize, BenchResult)> = Vec::new();

        for &ef in &EF_VALUES {
            let r = bench_sq4(
                &train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4,
            );
            println_both!(
                "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}",
                ef, r.recall, r.qps, r.avg_visited
            );
            ef_results.push((ef, r));
        }

        all_results.push((alpha_core, ef_results));
        println_both!("");
    }

    // 清除环境变量
    std::env::remove_var("RAVEN_ALPHA_CORE");

    // ════════════════════════════════════════════════════════════════
    //  汇总
    // ════════════════════════════════════════════════════════════════
    println_both!("=======================================================");
    println_both!("  汇总: 各 alpha_core 在 ef=50 的对比（主工作点）");
    println_both!("=======================================================");
    println_both!(
        "  {:>12}  {:>10}  {:>12}  {:>12}",
        "alpha_core", "recall", "QPS", "avg_visited"
    );
    println_both!("  {}", "-".repeat(52));

    for (alpha_core, ef_results) in &all_results {
        if let Some((_, r)) = ef_results.iter().find(|(ef, _)| *ef == 50) {
            println_both!(
                "  {:>12.1}  {:>10.4}  {:>12.0}  {:>12.1}",
                alpha_core, r.recall, r.qps, r.avg_visited
            );
        }
    }

    // 各 ef 点的最佳 alpha_core
    println_both!("\n--- 各 ef 点最佳 alpha_core（按 QPS 排序）---");
    println_both!(
        "  {:>6}  {:>12}  {:>10}  {:>12}  {:>12}",
        "ef", "best_a_c", "recall", "QPS", "avg_visited"
    );
    println_both!("  {}", "-".repeat(58));

    for &ef in &EF_VALUES {
        let mut best: Option<(f32, &BenchResult)> = None;
        for (alpha_core, ef_results) in &all_results {
            if let Some((_, r)) = ef_results.iter().find(|(e, _)| *e == ef) {
                if best.is_none() || r.qps > best.unwrap().1.qps {
                    best = Some((*alpha_core, r));
                }
            }
        }
        if let Some((alpha_core, r)) = best {
            println_both!(
                "  {:>6}  {:>12.1}  {:>10.4}  {:>12.0}  {:>12.1}",
                ef, alpha_core, r.recall, r.qps, r.avg_visited
            );
        }
    }

    // 全表对比（所有 alpha_core × 所有 ef）
    println_both!("\n--- 全表: recall @ 各 alpha_core x ef ---");
    print!("  {:>6}", "ef");
    for &ac in &ALPHA_CORE_VALUES {
        print!("  {:>10.1}", ac);
    }
    println!();
    for &ef in &EF_VALUES {
        print!("  {:>6}", ef);
        for (ac, ef_results) in &all_results {
            if let Some((_, r)) = ef_results.iter().find(|(e, _)| *e == ef) {
                print!("  {:>10.4}", r.recall);
            }
        }
        println!();
    }

    println_both!("\n--- 全表: avg_visited @ 各 alpha_core x ef ---");
    print!("  {:>6}", "ef");
    for &ac in &ALPHA_CORE_VALUES {
        print!("  {:>10.1}", ac);
    }
    println!();
    for &ef in &EF_VALUES {
        print!("  {:>6}", ef);
        for (ac, ef_results) in &all_results {
            if let Some((_, r)) = ef_results.iter().find(|(e, _)| *e == ef) {
                print!("  {:>10.1}", r.avg_visited);
            }
        }
        println!();
    }

    // Pareto 前沿
    println_both!("\n--- Pareto 前沿（recall vs QPS）---");
    println_both!(
        "  {:>12}  {:>6}  {:>10}  {:>12}",
        "alpha_core", "ef", "recall", "QPS"
    );
    println_both!("  {}", "-".repeat(46));

    let mut all_points: Vec<(f32, usize, f64, f64)> = Vec::new();
    for (alpha_core, ef_results) in &all_results {
        for (ef, r) in ef_results {
            all_points.push((*alpha_core, *ef, r.recall, r.qps));
        }
    }

    let mut pareto: Vec<&(f32, usize, f64, f64)> = Vec::new();
    for p in &all_points {
        let dominated = all_points.iter().any(|q| {
            (q.2 >= p.2 && q.3 >= p.3) && (q.2 > p.2 || q.3 > p.3)
        });
        if !dominated {
            pareto.push(p);
        }
    }
    pareto.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for &(alpha_core, ef, recall, qps) in &pareto {
        println_both!(
            "  {:>12.1}  {:>6}  {:>10.4}  {:>12.0}",
            alpha_core, ef, recall, qps
        );
    }

    // 写文件
    let mut f = File::create("experiments/alpha_core_sweep_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/alpha_core_sweep_result.txt");
}
