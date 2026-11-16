//! 预取效果验证 — Visited 数组预取对 ns/visit 的影响
//!
//! 对比：有 visited 预取 vs 无 visited 预取（通过环境变量控制）
//! 在 ef=50/65 两个点各跑 5 轮，测量 ns/visit 波动
//!
//! 用法：cargo run --release --bin prefetch_bench

use std::io::Write;
use std::time::Instant;

use raven::bench::util::SiftDataset;
use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;

fn single_run(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    ef: usize,
    sq4: &SQ4Dataset,
) -> (f64, f64) {
    let mut s = GraphSearcher::new(train, graph, ef);
    s.with_prefetch_offset(PO);
    s.with_rerank_factor(RR);
    s.with_sq4(sq4);
    let mut total_visited = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        s.search_sq4(&test[q * dim..(q + 1) * dim], K);
        total_visited += s.last_visited_count();
    }
    let dt = t0.elapsed();
    let qps = nq as f64 / dt.as_secs_f64();
    let visited = total_visited as f64 / nq as f64;
    (qps, visited)
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

    let data = SiftDataset::load();
    let dim = data.dim;
    let nq = data.nq;

    println_both!("=== Visited 预取效果验证 ===\n");
    println_both!("数据: n={}, dim={}, nq={}", data.n, dim, nq);
    println_both!("配置: po={}, rr={}, K={}", PO, RR, K);

    std::env::set_var("RAVEN_ALPHA_CORE", "0.93");
    println_both!("\n建图中... ");
    let mut rng = ChaCha8Rng::seed_from(42);
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
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&data.train, dim, &config, &mut rng);
    println_both!("建图: {:.0}s", t0.elapsed().as_secs_f64());

    let sq4 = SQ4Dataset::build(&data.train, dim);
    println_both!("SQ4 编码完成\n");

    // ════════════════════════════════════════════════════════════════
    //  对比测量：ef=50/65 各 1 warmup + 5 rounds
    // ════════════════════════════════════════════════════════════════
    println_both!("═══════════════════════════════════════════════════════════════");
    println_both!("  Visited 预取效果（ef=50/65 各 1 warmup + 5 rounds）");
    println_both!("═══════════════════════════════════════════════════════════════\n");

    for &ef in &[50, 65] {
        println_both!("  --- ef={} ---\n", ef);
        println_both!("  {:>6}  {:>10}  {:>10}  {:>10}",
            "round", "QPS", "avg_vis", "ns/visit");
        println_both!("  {}", "-".repeat(50));

        // warmup
        single_run(&data.train, &graph, &data.test, dim, nq, ef, &sq4);

        let mut ns_list = Vec::new();
        let mut qps_list = Vec::new();

        for round in 1..=5 {
            let (qps, visited) = single_run(&data.train, &graph, &data.test, dim, nq, ef, &sq4);
            let ns = 1e9 / qps / visited;
            println_both!("  {:>6}  {:>10.0}  {:>10.1}  {:>8.1}ns",
                round, qps, visited, ns);
            ns_list.push(ns);
            qps_list.push(qps);
        }

        let avg_ns: f64 = ns_list.iter().sum::<f64>() / 5.0;
        let avg_qps: f64 = qps_list.iter().sum::<f64>() / 5.0;
        let min_ns = ns_list.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ns = ns_list.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread = max_ns - min_ns;

        println_both!("  {}", "-".repeat(50));
        println_both!("  avg: ns/visit={:.1}  QPS={:.0}  spread={:.1}ns\n", avg_ns, avg_qps, spread);
    }

    // 对比基线（submission_bench 的数字）
    println_both!("═══════════════════════════════════════════════════════════════");
    println_both!("  对比基线（submission_bench with BFS reorder）");
    println_both!("  ef=50: QPS=15352, avg_visited=1461.3, ns/visit={:.1}", 1e9/15352.0/1461.3);
    println_both!("  ef=65: QPS=14135, avg_visited=1812.9, ns/visit={:.1}", 1e9/14135.0/1812.9);
    println_both!("═══════════════════════════════════════════════════════════════\n");

    let mut f = std::fs::File::create("experiments/prefetch_bench_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/prefetch_bench_result.txt");
}
