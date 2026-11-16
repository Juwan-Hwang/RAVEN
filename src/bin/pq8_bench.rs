//! PQ8 (K=256) LUT-ADC benchmark (Phase 1 Step 1)
//!
//! 三方对比：f32 全精度 vs SQ8 vs PQ8（均含 f32 rerank）。
//! 建图一次，扫描多个 ef 工作点。
//!
//! 结果自动写入 pq8_bench_result.txt
//!
//! 用法：cargo run --release --bin pq8_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, VamanaBuildConfig, VamanaGraph};
use raven::quant::{PQ8Dataset, SQ8Dataset};

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
    F32,
    SQ8(&'a SQ8Dataset),
    PQ8(&'a PQ8Dataset),
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
    mode: &SearchMode,
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    match mode {
        SearchMode::F32 => {}
        SearchMode::SQ8(sq8) => {
            searcher.with_sq8(sq8);
        }
        SearchMode::PQ8(pq8) => {
            searcher.with_pq8(pq8);
        }
    }

    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = match mode {
            SearchMode::F32 => searcher.search(query, k),
            SearchMode::SQ8(_) => searcher.search_sq8(query, k),
            SearchMode::PQ8(_) => searcher.search_pq8(query, k),
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
    mode: &SearchMode,
) -> BenchResult {
    bench_mode(train, graph, test, dim, nq, gt, gt_k, k, ef, mode); // warm-up
    let r2 = bench_mode(train, graph, test, dim, nq, gt, gt_k, k, ef, mode);
    let r3 = bench_mode(train, graph, test, dim, nq, gt, gt_k, k, ef, mode);
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

    println_both!("=== PQ8 (K=256) vs SQ8 vs f32 Benchmark (Phase 1 Step 1) ===\n");

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
    let m = 32usize; // SIFT-128: M=32, sub_dim=4

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
    println_both!(
        "SQ8 编码: {:.1}s ({} bytes, {:.1} MB)",
        t0.elapsed().as_secs_f64(),
        sq8.codes.len(),
        sq8.codes.len() as f64 / 1e6
    );

    // PQ8 编码
    println_both!("\n--- PQ8 编码 (M={}, K=256) ---", m);
    let t0 = Instant::now();
    let pq8 = PQ8Dataset::build(&train, dim, m);
    println_both!(
        "PQ8 编码: {:.1}s ({} bytes, {:.1} MB)",
        t0.elapsed().as_secs_f64(),
        pq8.codes.len(),
        pq8.codes.len() as f64 / 1e6
    );

    println_both!(
        "\n内存对比: f32={:.1} MB | SQ8={:.1} MB ({:.1}x) | PQ8={:.1} MB ({:.1}x)",
        (n * dim * 4) as f64 / 1e6,
        sq8.codes.len() as f64 / 1e6,
        (n * dim * 4) as f64 / sq8.codes.len() as f64,
        pq8.codes.len() as f64 / 1e6,
        (n * dim * 4) as f64 / pq8.codes.len() as f64
    );

    // 扫描
    let ef_list = [50, 100, 200];

    println_both!(
        "\n  {:>6}  {:>8}  {:>8}  {:>12}  {:>8}  {:>12}  {:>10}  {:>10}",
        "ef", "mode", "recall", "QPS", "speedup", "avg_visited", "recallΔ", "visitedΔ"
    );
    println_both!("  {}", "-".repeat(92));

    // 先跑 f32 基线
    let mut f32_results = std::collections::HashMap::new();
    for &ef in &ef_list {
        let r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &SearchMode::F32);
        f32_results.insert(ef, r);
    }

    for &ef in &ef_list {
        let f32_r = &f32_results[&ef];

        println_both!(
            "  {:>6}  {:>8}  {:>8.4}  {:>12.0}  {:>8.2}x  {:>12.1}  {:>+10.4}  {:>+10.1}",
            ef, "f32", f32_r.recall, f32_r.qps, 1.0, f32_r.avg_visited, 0.0, 0.0
        );

        let sq8_r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &SearchMode::SQ8(&sq8));
        let sq8_speedup = sq8_r.qps / f32_r.qps;
        println_both!(
            "  {:>6}  {:>8}  {:>8.4}  {:>12.0}  {:>8.2}x  {:>12.1}  {:>+10.4}  {:>+10.1}",
            ef, "SQ8", sq8_r.recall, sq8_r.qps, sq8_speedup, sq8_r.avg_visited,
            sq8_r.recall - f32_r.recall, sq8_r.avg_visited - f32_r.avg_visited
        );

        let pq8_r = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, &SearchMode::PQ8(&pq8));
        let pq8_speedup = pq8_r.qps / f32_r.qps;
        println_both!(
            "  {:>6}  {:>8}  {:>8.4}  {:>12.0}  {:>8.2}x  {:>12.1}  {:>+10.4}  {:>+10.1}",
            ef, "PQ8", pq8_r.recall, pq8_r.qps, pq8_speedup, pq8_r.avg_visited,
            pq8_r.recall - f32_r.recall, pq8_r.avg_visited - f32_r.avg_visited
        );

        // SQ8→PQ8 delta
        println_both!(
            "  {:>6}  {:>8}  {:>8.4}  {:>12.0}  {:>8.2}x  {:>12.1}  {:>+10.4}  {:>+10.1}",
            "", "PQ8/SQ8", pq8_r.recall, pq8_r.qps, pq8_r.qps / sq8_r.qps, pq8_r.avg_visited,
            pq8_r.recall - sq8_r.recall, pq8_r.avg_visited - sq8_r.avg_visited
        );
        println_both!("  {}", "-".repeat(92));
    }

    // 终态门判定（§〇.1 例外条款）
    println_both!("\n--- 终态门判定 (§〇.1) ---\n");
    let ef50_f32 = &f32_results[&50];
    let ef50_pq8 = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, 50, &SearchMode::PQ8(&pq8));
    let ef100_pq8 = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, 100, &SearchMode::PQ8(&pq8));
    let ef200_pq8 = bench_stable(&train, &graph, &test, dim, nq, &gt, gt_k, k, 200, &SearchMode::PQ8(&pq8));

    let speedup_50 = ef50_pq8.qps / ef50_f32.qps;
    let passes_recall = ef50_pq8.recall >= 0.95 || ef100_pq8.recall >= 0.95 || ef200_pq8.recall >= 0.95;
    let passes_qps = speedup_50 >= 1.5 || ef100_pq8.qps / ef50_f32.qps >= 1.5;

    println_both!("  ef=50:  f32 recall={:.4} QPS={:.0} | PQ8 recall={:.4} QPS={:.0} | speedup={:.2}x",
        ef50_f32.recall, ef50_f32.qps, ef50_pq8.recall, ef50_pq8.qps, speedup_50);
    println_both!("  ef=100: PQ8 recall={:.4} QPS={:.0}", ef100_pq8.recall, ef100_pq8.qps);
    println_both!("  ef=200: PQ8 recall={:.4} QPS={:.0}", ef200_pq8.recall, ef200_pq8.qps);
    println_both!("\n  recall ≥ 0.95: {}", if passes_recall { "PASS" } else { "FAIL" });
    println_both!("  QPS ≥ baseline × 1.5: {}", if passes_qps { "PASS" } else { "FAIL" });
    println_both!("  终态门: {}", if passes_recall && passes_qps { "✅ PASS" } else { "❌ FAIL" });

    println_both!("\n  历史基线:");
    println_both!("    f32 ef=50:     recall=0.9705  QPS=8,657  avg_visited=1,227");
    println_both!("    SQ8 ef=50:     recall=0.9653  QPS=13,111  avg_visited=1,233  (Phase 1.0 实测)");
    println_both!("    PQ4 ef=50:     recall=0.0597  QPS=52,509  avg_visited=889,755  (K=16 失败)");

    // 写文件
    let mut f = File::create("pq8_bench_result.txt").expect("create result file");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 pq8_bench_result.txt");
}
