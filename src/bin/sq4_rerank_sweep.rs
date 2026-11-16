//! SQ4 rerank sweep — 找到弥补 recall gap 的最优 rerank_factor
//!
//! SQ4 在 ef=50 有 +14% QPS 但 -1pp recall。
//! 提高 rerank_factor 可能弥补 recall（f32 rerank 开销极小）。
//!
//! 用法：cargo run --release --bin sq4_rerank_sweep

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
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rr);
    searcher.with_sq4(sq4);

    // warm-up
    for q in 0..100.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search_sq4(query, k);
    }

    let mut hits = 0;
    let mut total = 0;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq4(query, k);
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

fn bench_sq8(
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
    sq8: &SQ8Dataset,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rr);
    searcher.with_sq8(sq8);

    // warm-up
    for q in 0..100.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search_sq8(query, k);
    }

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

    println_both!("=== SQ4 Rerank Sweep ===\n");

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
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("{:.1}s", t0.elapsed().as_secs_f64());

    let sq8 = SQ8Dataset::build(&train, dim);
    let sq4 = SQ4Dataset::build(&train, dim);

    // SQ8 baseline (rr=3, ef=50)
    let sq8_base = bench_sq8(&train, &graph, &test, dim, nq, &gt, gt_k, k, 50, po, 3, &sq8);
    println_both!(
        "\nSQ8 baseline (ef=50, rr=3): recall={:.4}, QPS={:.0}",
        sq8_base.recall,
        sq8_base.qps
    );

    // SQ4 rerank sweep at ef=50
    println_both!("\n--- SQ4 rerank sweep (ef=50) ---");
    println_both!(
        "  {:>6}  {:>10}  {:>12}  {:>12}  {:>10}",
        "rr", "recall", "QPS", "vs SQ8 QPS", "recallΔ"
    );
    println_both!("  {}", "-".repeat(62));

    for &rr in &[3, 5, 8, 12, 16, 24] {
        let r = bench_sq4(&train, &graph, &test, dim, nq, &gt, gt_k, k, 50, po, rr, &sq4);
        let speedup = r.qps / sq8_base.qps;
        let recall_delta = r.recall - sq8_base.recall;
        println_both!(
            "  {:>6}  {:>10.4}  {:>12.0}  {:>11.2}x  {:>+10.4}",
            rr,
            r.recall,
            r.qps,
            speedup,
            recall_delta
        );
    }

    // Also sweep ef=55 with higher rerank
    println_both!("\n--- SQ4 rerank sweep (ef=55) ---");
    let sq8_55 = bench_sq8(&train, &graph, &test, dim, nq, &gt, gt_k, k, 55, po, 3, &sq8);
    println_both!(
        "SQ8 baseline (ef=55, rr=3): recall={:.4}, QPS={:.0}\n",
        sq8_55.recall,
        sq8_55.qps
    );
    println_both!(
        "  {:>6}  {:>10}  {:>12}  {:>12}  {:>10}",
        "rr", "recall", "QPS", "vs SQ8 QPS", "recallΔ"
    );
    println_both!("  {}", "-".repeat(62));

    for &rr in &[3, 5, 8, 12, 16] {
        let r = bench_sq4(&train, &graph, &test, dim, nq, &gt, gt_k, k, 55, po, rr, &sq4);
        let speedup = r.qps / sq8_55.qps;
        let recall_delta = r.recall - sq8_55.recall;
        println_both!(
            "  {:>6}  {:>10.4}  {:>12.0}  {:>11.2}x  {:>+10.4}",
            rr,
            r.recall,
            r.qps,
            speedup,
            recall_delta
        );
    }

    let mut f = File::create("experiments/sq4_rerank_sweep_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/sq4_rerank_sweep_result.txt");
}
