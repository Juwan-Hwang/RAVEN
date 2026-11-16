//! P5b: po 精细扫描 13-20，步长 1
//!
//! 用法：cargo run --release --bin optimization_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

macro_rules! println_both {
    ($out:expr, $($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{}", line);
        $out.push_str(&line);
        $out.push('\n');
    }};
}

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
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
    }
}

fn build_graph(train: &[f32], dim: usize, r_max: usize, l_build: usize) -> VamanaGraph {
    let mut rng = ChaCha8Rng::seed_from(42);
    let r_soft = (r_max as f32 * 1.5) as usize;
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build,
        r_max,
        r_soft,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    VamanaGraph::build(train, dim, &config, &mut rng)
}

fn main() {
    let mut out = String::new();

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
    println_both!(&mut out, "RAVEN P5b: po 精细扫描 (13-20, step=1)");
    println_both!(&mut out, "数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);

    println_both!(&mut out, "\n=== 建图: R=32, L=200 ===");
    let t0 = Instant::now();
    let graph = build_graph(&train, dim, 32, 200);
    println_both!(&mut out, "建图: {:.1}s", t0.elapsed().as_secs_f64());

    let sq4 = SQ4Dataset::build(&train, dim);

    // po 精细扫描: 13, 14, 15, 16, 17, 18, 19, 20
    let po_values: [usize; 8] = [13, 14, 15, 16, 17, 18, 19, 20];
    let ef_values: [usize; 4] = [40, 50, 60, 80];
    let rr = 5usize;

    println_both!(
        &mut out,
        "\n=== po 精细扫描 (rr={}, R=32, L=200) ===",
        rr
    );
    println_both!(
        &mut out,
        "  {:>4}  {:>6}  {:>10}  {:>12}  {:>12}",
        "po", "ef", "recall", "QPS", "avg_visited"
    );
    println_both!(out, "  {}", "-".repeat(52));

    for &ef in &ef_values {
        for &po in &po_values {
            let r = bench_sq4(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4);
            println_both!(
                &mut out,
                "  {:>4}  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}",
                po, ef, r.recall, r.qps, r.avg_visited
            );
        }
        println_both!(out, "");
    }

    let result_path = "experiments/optimization_bench_p5b.txt";
    let mut f = File::create(result_path).expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 {}", result_path);
}
