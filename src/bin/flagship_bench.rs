//! 旗舰 Benchmark — 叠加全部已验证优化，展示累积 QPS 提升
//!
//! 逐层叠加优化，每层对比前一层基线：
//!   Layer 0: f32 baseline (layered nav + po=8 prefetch)
//!   Layer 1: + SQ8 量化
//!   Layer 2: + 自适应 ef (gamma=2.0)
//!   Layer 3: + 多线程 (rayon, 全核心)
//!
//! 结果自动写入 flagship_bench_result.txt
//!
//! 用法：cargo run --release --bin flagship_bench

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
}

fn compute_recall(results: &[Vec<u32>], gt: &[i32], gt_k: usize, k: usize, nq: usize) -> f64 {
    let mut hits = 0;
    let mut total = 0;
    for q in 0..nq {
        let found = &results[q];
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f64 / total as f64
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

    println_both!("=== RAVEN Flagship Benchmark (全优化叠加) ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10usize;
    let ef = 50usize;
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!("ef_search={}, threads={}", ef, num_threads);

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
    println_both!("SQ8 编码: {:.1}s ({} MB)", t0.elapsed().as_secs_f64(), sq8.codes.len() / 1_000_000);

    // 自适应 ef 配置
    println_both!("\n--- 自适应 ef 配置 ---");
    let layered_nav = graph.layered_nav().expect("layered nav required");
    let t0 = Instant::now();
    let adaptive_config = AdaptiveEfConfig::build_with_layered_nav(
        &train, dim, layered_nav, 35, 75, 2.0);
    println_both!("自适应 ef: {:.1}s (min_ef=35, max_ef=75, gamma=2.0)", t0.elapsed().as_secs_f64());

    // 准备查询引用
    let queries: Vec<&[f32]> = (0..nq).map(|q| &test[q * dim..(q + 1) * dim]).collect();

    // ── Layer 0: f32 baseline ──
    println_both!("\n--- Layer 0: f32 baseline (layered nav + po=8) ---");
    {
        let mut searcher = GraphSearcher::new(&train, &graph, ef);
        // warmup
        for q in 0..nq.min(100) { let _ = searcher.search(&queries[q], k); }
        let t0 = Instant::now();
        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0;
        for q in 0..nq {
            let r = searcher.search(&queries[q], k);
            total_visited += searcher.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        let recall = compute_recall(&results, &gt, gt_k, k, nq);
        let qps = nq as f64 / dt.as_secs_f64();
        println_both!("  recall={:.4}  QPS={:.0}  avg_visited={:.1}", recall, qps, total_visited as f64 / nq as f64);
        println_both!("  → 基线 QPS = {:.0}", qps);
    }

    // ── Layer 1: + SQ8 ──
    println_both!("\n--- Layer 1: + SQ8 量化 ---");
    {
        let mut searcher = GraphSearcher::new(&train, &graph, ef);
        searcher.with_sq8(&sq8);
        // warmup
        for q in 0..nq.min(100) { let _ = searcher.search_sq8(&queries[q], k); }
        let t0 = Instant::now();
        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0;
        for q in 0..nq {
            let r = searcher.search_sq8(&queries[q], k);
            total_visited += searcher.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        let recall = compute_recall(&results, &gt, gt_k, k, nq);
        let qps = nq as f64 / dt.as_secs_f64();
        println_both!("  recall={:.4}  QPS={:.0}  avg_visited={:.1}", recall, qps, total_visited as f64 / nq as f64);
        println_both!("  → SQ8 QPS = {:.0}", qps);
    }

    // ── Layer 2: + 自适应 ef ──
    println_both!("\n--- Layer 2: + SQ8 + 自适应 ef (gamma=2.0) ---");
    {
        let mut searcher = GraphSearcher::new(&train, &graph, ef);
        searcher.with_sq8(&sq8);
        searcher.with_adaptive_ef(adaptive_config.clone());
        // warmup
        for q in 0..nq.min(100) { let _ = searcher.search_sq8(&queries[q], k); }
        let t0 = Instant::now();
        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0;
        let mut total_ef = 0usize;
        for q in 0..nq {
            let r = searcher.search_sq8(&queries[q], k);
            total_visited += searcher.last_visited_count();
            total_ef += searcher.last_ef_used();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        let recall = compute_recall(&results, &gt, gt_k, k, nq);
        let qps = nq as f64 / dt.as_secs_f64();
        println_both!("  recall={:.4}  QPS={:.0}  avg_visited={:.1}  avg_ef={:.1}",
            recall, qps, total_visited as f64 / nq as f64, total_ef as f64 / nq as f64);
        println_both!("  → SQ8+adaptive_ef QPS = {:.0}", qps);
    }

    // ── Layer 3: + 多线程 ──
    println_both!("\n--- Layer 3: + SQ8 + 自适应 ef + 多线程 ({} threads) ---", num_threads);
    {
        let mut searcher = GraphSearcher::new(&train, &graph, ef);
        searcher.with_sq8(&sq8);
        searcher.with_adaptive_ef(adaptive_config.clone());
        // warmup
        let warmup_n = nq.min(100);
        let _ = searcher.batch_search(&queries[..warmup_n], k);

        let t0 = Instant::now();
        let batch_results = searcher.batch_search(&queries, k);
        let dt = t0.elapsed();

        let results: Vec<Vec<u32>> = batch_results.into_iter()
            .map(|r| r.into_iter().map(|(id, _)| id).collect())
            .collect();
        let recall = compute_recall(&results, &gt, gt_k, k, nq);
        let qps = nq as f64 / dt.as_secs_f64();
        println_both!("  recall={:.4}  QPS={:.0}", recall, qps);
        println_both!("  → 全栈 QPS = {:.0} ({} threads)", qps, num_threads);
    }

    // ── 汇总 ──
    println_both!("\n--- 累积提升汇总 ---");
    println_both!("  (见上方各层 QPS，最终全栈 QPS 即为上榜 QPS)");

    // 写文件
    let _ = std::fs::remove_file("flagship_bench_result.txt");
    let mut f = File::create("flagship_bench_result.txt").expect("create result file");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 flagship_bench_result.txt");
}
