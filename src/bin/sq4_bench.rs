//! SQ4 vs SQ8 A/B Benchmark
//!
//! SQ4: 4-bit per dimension scalar quantization (64B/vector for SIFT-128)
//! SQ8: 8-bit per dimension scalar quantization (128B/vector for SIFT-128)
//!
//! 同图同配置，仅量化路径不同。建图一次，多 ef 扫描。
//!
//! 用法：cargo run --release --bin sq4_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, VamanaBuildConfig, VamanaGraph, PruneStrategy};
use raven::quant::{SQ8Dataset, SQ4Dataset};

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

enum SearchMode<'a> {
    SQ8(&'a SQ8Dataset),
    SQ4(&'a SQ4Dataset),
}

fn bench_mode(
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
    mode: &SearchMode,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rr);
    match mode {
        SearchMode::SQ8(sq8) => {
            searcher.with_sq8(sq8);
        }
        SearchMode::SQ4(sq4) => {
            searcher.with_sq4(sq4);
        }
    }

    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = match mode {
            SearchMode::SQ8(_) => searcher.search_sq8(query, k),
            SearchMode::SQ4(_) => searcher.search_sq4(query, k),
        };
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

/// warm-up + 2 轮取平均
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
    rr: usize,
    mode: &SearchMode,
) -> BenchResult {
    bench_mode(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, mode); // warm-up
    let r2 = bench_mode(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, mode);
    let r3 = bench_mode(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, mode);
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
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

    println_both!("=== SQ4 vs SQ8 A/B Benchmark ===\n");

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
    let po = 8usize;
    let rr = 3usize;

    // 建图（当前最优配置：nav_m=32, DirectionalPrune）
    println_both!("\n--- 建图 (nav_m=32, DirectionalPrune) ---");
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

    // SQ8 编码
    print!("SQ8 编码... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let sq8 = SQ8Dataset::build(&train, dim);
    println!("{:.1}s ({} bytes, {:.1} MB)", t0.elapsed().as_secs_f64(), sq8.codes.len(), sq8.codes.len() as f64 / 1e6);

    // SQ4 编码
    print!("SQ4 编码... ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let sq4 = SQ4Dataset::build(&train, dim);
    println!("{:.1}s ({} bytes, {:.1} MB)", t0.elapsed().as_secs_f64(), sq4.codes.len(), sq4.codes.len() as f64 / 1e6);

    println_both!(
        "\n内存对比: SQ8={:.1} MB ({:.1}x) | SQ4={:.1} MB ({:.1}x) | SQ4/SQ8={:.1}x",
        sq8.codes.len() as f64 / 1e6,
        (n * dim * 4) as f64 / sq8.codes.len() as f64,
        sq4.codes.len() as f64 / 1e6,
        (n * dim * 4) as f64 / sq4.codes.len() as f64,
        sq8.codes.len() as f64 / sq4.codes.len() as f64,
    );

    // 扫描多个 ef
    let ef_list = [50, 55, 65, 80, 100];

    println_both!(
        "\n  {:>6}  {:>6}  {:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "ef", "mode", "recall", "QPS", "vs SQ8 QPS", "avg_visited", "recallΔ"
    );
    println_both!("  {}", "-".repeat(82));

    for &ef in &ef_list {
        // SQ8
        let sq8_r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &SearchMode::SQ8(&sq8));

        println_both!(
            "  {:>6}  {:>6}  {:>10.4}  {:>12.0}  {:>12}  {:>12.1}  {:>10}",
            ef, "SQ8", sq8_r.recall, sq8_r.qps, "1.00x", sq8_r.avg_visited, ""
        );

        // SQ4
        let sq4_r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &SearchMode::SQ4(&sq4));

        let speedup = sq4_r.qps / sq8_r.qps;
        let recall_delta = sq4_r.recall - sq8_r.recall;

        println_both!(
            "  {:>6}  {:>6}  {:>10.4}  {:>12.0}  {:>11.2}x  {:>12.1}  {:>+10.4}",
            ef, "SQ4", sq4_r.recall, sq4_r.qps, speedup, sq4_r.avg_visited, recall_delta
        );
        println_both!("  {}", "-".repeat(82));
    }

    // 结论
    println_both!("\n--- 结论 ---");
    let sq8_50 = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, 50, po, rr, &SearchMode::SQ8(&sq8));
    let sq4_50 = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, 50, po, rr, &SearchMode::SQ4(&sq4));
    let speedup_50 = sq4_50.qps / sq8_50.qps;
    let recall_delta_50 = sq4_50.recall - sq8_50.recall;

    if speedup_50 > 1.05 && recall_delta_50 > -0.02 {
        println_both!("✅ SQ4 在 ef=50 有显著 QPS 优势 (+{:.1}%)，recall 变化 {:+.4}pp", (speedup_50 - 1.0) * 100.0, recall_delta_50);
    } else if speedup_50 > 1.05 && recall_delta_50 <= -0.02 {
        println_both!("⚠️ SQ4 QPS +{:.1}% 但 recall 掉 {:.4}pp，需提高 rerank", (speedup_50 - 1.0) * 100.0, -recall_delta_50);
    } else {
        println_both!("❌ SQ4 无显著优势 (QPS {:+.1}%, recall {:+.4}pp)", (speedup_50 - 1.0) * 100.0, recall_delta_50);
    }

    let mut f = File::create("experiments/sq4_bench_result.txt").expect("create result file");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/sq4_bench_result.txt");
}
