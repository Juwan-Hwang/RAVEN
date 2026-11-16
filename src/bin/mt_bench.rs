//! 多线程查询 benchmark (Phase 7)
//!
//! 对比单线程 vs 多线程（rayon）批量搜索 QPS。
//! 使用 SQ8 量化路径（Phase 1 最终选择）。
//!
//! 扫描不同线程数：1, 2, 4, 8
//! 结果自动写入 multithread_bench_result.txt
//!
//! 用法：cargo run --release --bin mt_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, VamanaBuildConfig, VamanaGraph};
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
}

fn bench_single_thread(
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

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, k);

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
    }
}

fn bench_multi_thread(
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
    num_threads: usize,
) -> BenchResult {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("build thread pool");

    let queries: Vec<&[f32]> = (0..nq).map(|q| &test[q * dim..(q + 1) * dim]).collect();

    pool.install(|| {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_sq8(sq8);

        // warmup
        searcher.batch_search(&queries[..nq.min(100)], k);

        let mut hits = 0;
        let mut total = 0;

        let t0 = Instant::now();
        let results = searcher.batch_search(&queries, k);
        let dt = t0.elapsed();

        for (q, result) in results.iter().enumerate() {
            let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
            let gt_slice = &gt[q * gt_k..q * gt_k + k];
            for &g in gt_slice {
                if found.contains(&(g as u32)) {
                    hits += 1;
                }
            }
            total += k;
        }

        BenchResult {
            recall: hits as f64 / total as f64,
            qps: nq as f64 / dt.as_secs_f64(),
        }
    })
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

    println_both!("=== Multi-thread Benchmark (Phase 7) ===\n");

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
    println_both!("CPU 核心数: {}", num_cpus());
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
    let sq8 = SQ8Dataset::build(&train, dim);
    println_both!("SQ8 编码完成: {:.1} MB", sq8.codes.len() as f64 / 1e6);

    // 单线程基线
    let ef_list = [50, 100];

    println_both!(
        "\n  {:>6}  {:>8}  {:>10}  {:>12}  {:>10}  {:>10}",
        "ef", "threads", "recall", "QPS", "speedup", "scaling"
    );
    println_both!("  {}", "-".repeat(66));

    for &ef in &ef_list {
        // warm-up
        bench_single_thread(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8);

        let st = bench_single_thread(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8);

        println_both!(
            "  {:>6}  {:>8}  {:>10.4}  {:>12.0}  {:>10.2}x  {:>10.1}%",
            ef, 1, st.recall, st.qps, 1.0, 100.0
        );

        for &nt in &[2, 4, 8] {
            let mt = bench_multi_thread(
                &train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, nt,
            );

            let speedup = mt.qps / st.qps;
            let scaling = speedup / nt as f64 * 100.0;

            println_both!(
                "  {:>6}  {:>8}  {:>10.4}  {:>12.0}  {:>10.2}x  {:>10.1}%",
                ef, nt, mt.recall, mt.qps, speedup, scaling
            );
        }
        println_both!("  {}", "-".repeat(66));
    }

    // 写文件
    let mut f = File::create("multithread_bench_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 multithread_bench_result.txt");
}

/// 获取 CPU 核心数
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
