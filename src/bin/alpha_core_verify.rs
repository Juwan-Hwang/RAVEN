//! Alpha_core 验证 — 多 seed 重复建图，排除单次随机性
//!
//! 固定 alpha_core=0.9 和 1.0，各用 3 个不同 seed 独立建图
//! 只测 ef=50 主工作点
//!
//! 目标：
//! 1. 同 seed 同 alpha_core 多次运行 → 验证建图确定性
//! 2. 不同 seed 同 alpha_core → 验证 recall 自然波动幅度
//! 3. 不同 seed 下 0.9 vs 1.0 → 验证 +0.67pp 是否系统性效应
//!
//! 用法：cargo run --release --bin alpha_core_verify

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{
    GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
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

fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
}

fn bench_ef50(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
) -> (f64, f64, f64) {
    let ef = 50;
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
        (recall, nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };

    run(); // warmup
    let r1 = run();
    let r2 = run();
    let r3 = run();
    (
        median3(r1.0, r2.0, r3.0),
        median3(r1.1, r2.1, r3.1),
        median3(r1.2, r2.2, r3.2),
    )
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

    println_both!("=== RAVEN Alpha_core Verification (multi-seed) ===\n");

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
    let sq4 = SQ4Dataset::build(&train, dim);

    const ALPHA_CORE_VALUES: [f32; 2] = [0.9, 1.0];
    const SEEDS: [u64; 3] = [42, 123, 7];

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!("配置: R=40, L=384, po={}, rr={}, ef=50, SQ4, DirectionalPrune", po, rr);
    println_both!("Alpha_core: {:?}", ALPHA_CORE_VALUES);
    println_both!("Seeds: {:?}", SEEDS);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

    // 结构: (alpha_core, seed, recall, qps, avg_visited)
    let mut all_results: Vec<(f32, u64, f64, f64, f64)> = Vec::new();

    for &alpha_core in &ALPHA_CORE_VALUES {
        std::env::set_var("RAVEN_ALPHA_CORE", alpha_core.to_string());

        for &seed in &SEEDS {
            print!("--- alpha_core={:.1}, seed={} ---  建图中... ", alpha_core, seed);

            let t0 = Instant::now();
            let mut rng = ChaCha8Rng::seed_from(seed);
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
            let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
            let build_time = t0.elapsed().as_secs_f64();

            let (recall, qps, avg_visited) = bench_ef50(
                &train, &graph, &test, dim, nq, &gt, gt_k, k, po, rr, &sq4,
            );

            println_both!("建图: {:.0}s | recall={:.4}  QPS={:.0}  visited={:.1}",
                build_time, recall, qps, avg_visited);

            all_results.push((alpha_core, seed, recall, qps, avg_visited));
        }
    }

    std::env::remove_var("RAVEN_ALPHA_CORE");

    // ════════════════════════════════════════════════════════════════
    //  分析
    // ════════════════════════════════════════════════════════════════
    println_both!("\n=======================================================");
    println_both!("  分析");
    println_both!("=======================================================");

    // 1. 确定性检查：同 alpha_core 同 seed 是否产出同 recall
    //    （由于代码用 ChaCha8Rng 且批间顺序更新，理论上应确定性）
    //    注意：我们没有重复跑同 seed，但可以检查 recall 是否在不同 seed 间波动

    // 2. 各 alpha_core 的 recall 统计
    println_both!("\n--- 各 alpha_core 的 recall 统计（跨 seed）---");
    println_both!("  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "alpha_core", "recall_min", "recall_max", "range", "recall_mean");
    println_both!("  {}", "-".repeat(58));

    for &alpha_core in &ALPHA_CORE_VALUES {
        let recalls: Vec<f64> = all_results.iter()
            .filter(|(ac, _, _, _, _)| *ac == alpha_core)
            .map(|(_, _, r, _, _)| *r)
            .collect();
        let rmin = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
        let rmax = recalls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rmean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        let range = rmax - rmin;
        println_both!("  {:>12.1}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            alpha_core, rmin, rmax, range, rmean);
    }

    // 3. 0.9 vs 1.0 的逐 seed 对比
    println_both!("\n--- 逐 seed 对比：alpha_core=0.9 vs 1.0 ---");
    println_both!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "seed", "recall.9", "recall.10", "delta", "sig?");
    println_both!("  {}", "-".repeat(50));

    for &seed in &SEEDS {
        let r09 = all_results.iter()
            .find(|(ac, s, _, _, _)| *ac == 0.9 && *s == seed)
            .map(|(_, _, r, _, _)| *r)
            .unwrap();
        let r10 = all_results.iter()
            .find(|(ac, s, _, _, _)| *ac == 1.0 && *s == seed)
            .map(|(_, _, r, _, _)| *r)
            .unwrap();
        let delta = r09 - r10;
        // 0.67pp = 0.0067
        let sig = if delta > 0.003 { "YES" } else if delta > 0.001 { "maybe" } else { "NO" };
        println_both!("  {:>6}  {:>10.4}  {:>10.4}  {:>>+10.4}  {:>10}",
            seed, r09, r10, delta, sig);
    }

    // 4. 全表
    println_both!("\n--- 全部结果 ---");
    println_both!("  {:>12}  {:>6}  {:>10}  {:>12}  {:>12}",
        "alpha_core", "seed", "recall", "QPS", "avg_visited");
    println_both!("  {}", "-".repeat(58));
    for (ac, seed, recall, qps, visited) in &all_results {
        println_both!("  {:>12.1}  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}",
            ac, seed, recall, qps, visited);
    }

    // 写文件
    let mut f = File::create("experiments/alpha_core_verify_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/alpha_core_verify_result.txt");
}
