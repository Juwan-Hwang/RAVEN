//! 候选集质量子集验证 — 暴力 kNN vs Vamana 增量插入
//!
//! 目标：判断 Vamana 增量插入产生的候选集质量是否是 avg_visited 偏高的根因。
//!
//! 方法：
//! 1. 取 SIFT-1M 前 10 万向量作为子集
//! 2. 标准 Vamana 建图（贪心搜索 → DirectionalPrune）→ baseline 图
//! 3. 暴力 kNN（k=200）→ DirectionalPrune → perfect-candidate 图
//! 4. 两个图用相同 ef 搜索，对比 avg_visited 和 recall
//!
//! 唯一变量：候选集来源（贪心搜索 visited set vs 暴力 kNN）
//! 其余完全一致：DirectionalPrune + alpha_core=0.93 + connect_bidirectional + final_prune
//!
//! 用法：cargo run --release --bin candidate_quality_bench

use std::io::Write;
use std::time::Instant;

use raven::bench::util::{compute_recall, SiftDataset};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph, GraphSearcher};
use raven::quant::SQ4Dataset;
use rayon::prelude::*;

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;
const SUBSET_N: usize = 100_000;
const BRUTE_K: usize = 200; // 暴力 kNN 的 k 值（>= r_soft=60）

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

    // 取前 SUBSET_N 个向量作为子集
    let subset_n = SUBSET_N.min(data.n);
    let subset_train: Vec<f32> = data.train[..subset_n * dim].to_vec();

    println_both!("=== 候选集质量子集验证 ===\n");
    println_both!("数据: subset_n={}, dim={}, nq={}", subset_n, dim, nq);
    println_both!("暴力 kNN: k={}", BRUTE_K);

    std::env::set_var("RAVEN_ALPHA_CORE", "0.93");

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

    // ════════════════════════════════════════════════════════════════
    //  1. 标准 Vamana 建图（baseline）
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[1/4] 标准 Vamana 建图（贪心搜索候选集）...");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let vamana_graph = VamanaGraph::build(&subset_train, dim, &config, &mut rng);
    let vamana_build_time = t0.elapsed().as_secs_f64();
    println_both!("  建图: {:.1}s", vamana_build_time);

    let vamana_sq4 = SQ4Dataset::build(&subset_train, dim);
    println_both!("  SQ4 编码完成");

    // 度数统计
    let stats = vamana_graph.degree_stats();
    println_both!("  avg_degree={:.1}, max_degree={}, p95={}, p99={}",
        stats.mean_degree, stats.max_degree, stats.p95_degree, stats.p99_degree);

    // ════════════════════════════════════════════════════════════════
    //  2. 暴力 kNN
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[2/4] 暴力 kNN 计算（{} × {}，k={}）...", subset_n, subset_n, BRUTE_K);
    let t0 = Instant::now();
    let brute_candidates: Vec<Vec<u32>> = (0..subset_n)
        .into_par_iter()
        .map(|i| {
            let query = &subset_train[i * dim..(i + 1) * dim];
            // 计算到所有其他节点的距离
            let mut dists: Vec<(f32, u32)> = (0..subset_n)
                .filter(|&j| j != i)
                .map(|j| {
                    let v = &subset_train[j * dim..(j + 1) * dim];
                    (l2_simd(query, v), j as u32)
                })
                .collect();
            // 部分排序取前 k 个（f32 无 Ord，用 select_nth_unstable_by）
            dists.select_nth_unstable_by(BRUTE_K - 1, |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(BRUTE_K);
            dists.iter().map(|(_, id)| *id).collect()
        })
        .collect();
    let brute_time = t0.elapsed().as_secs_f64();
    println_both!("  暴力 kNN: {:.1}s ({:.0} distances/s/thread)",
        brute_time, (subset_n as f64 * subset_n as f64) / brute_time / rayon::current_num_threads() as f64);

    // ════════════════════════════════════════════════════════════════
    //  3. 从暴力 kNN 候选集建图
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[3/4] 从暴力 kNN 候选集建图（perfect-candidate）...");
    let t0 = Instant::now();
    let perfect_graph = VamanaGraph::build_from_candidates(
        &subset_train, dim, &brute_candidates, &config,
    );
    let perfect_build_time = t0.elapsed().as_secs_f64();
    println_both!("  建图: {:.1}s", perfect_build_time);

    let perfect_sq4 = SQ4Dataset::build(&subset_train, dim);
    println_both!("  SQ4 编码完成");

    let stats2 = perfect_graph.degree_stats();
    println_both!("  avg_degree={:.1}, max_degree={}, p95={}, p99={}",
        stats2.mean_degree, stats2.max_degree, stats2.p95_degree, stats2.p99_degree);

    // ════════════════════════════════════════════════════════════════
    //  4. 搜索对比
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[4/4] 搜索对比（ef=30/40/50/65）\n");
    println_both!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "ef", "V_recall", "P_recall", "V_avgvis", "P_avgvis", "Δvis%", "V_QPS");
    println_both!("  {}", "-".repeat(80));

    for &ef in &[30, 40, 50, 65] {
        // Vamana baseline
        let mut s_v = GraphSearcher::new(&subset_train, &vamana_graph, ef);
        s_v.with_prefetch_offset(PO);
        s_v.with_rerank_factor(RR);
        s_v.with_sq4(&vamana_sq4);

        let mut v_results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut v_total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let r = s_v.search_sq4(&data.test[q * dim..(q + 1) * dim], K);
            v_results.push(r.iter().map(|(id, _)| *id).collect());
            v_total_visited += s_v.last_visited_count();
        }
        let v_dt = t0.elapsed();
        let v_recall = compute_recall(&v_results, &data.gt, data.gt_k, K, nq);
        let v_avg_vis = v_total_visited as f64 / nq as f64;
        let v_qps = nq as f64 / v_dt.as_secs_f64();

        // Perfect candidate
        let mut s_p = GraphSearcher::new(&subset_train, &perfect_graph, ef);
        s_p.with_prefetch_offset(PO);
        s_p.with_rerank_factor(RR);
        s_p.with_sq4(&perfect_sq4);

        let mut p_results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut p_total_visited = 0usize;
        for q in 0..nq {
            let r = s_p.search_sq4(&data.test[q * dim..(q + 1) * dim], K);
            p_results.push(r.iter().map(|(id, _)| *id).collect());
            p_total_visited += s_p.last_visited_count();
        }
        let p_recall = compute_recall(&p_results, &data.gt, data.gt_k, K, nq);
        let p_avg_vis = p_total_visited as f64 / nq as f64;

        let delta_vis = (p_avg_vis - v_avg_vis) / v_avg_vis * 100.0;

        println_both!("  {:>6}  {:>10.4}  {:>10.4}  {:>10.1}  {:>10.1}  {:>+9.1}%  {:>10.0}",
            ef, v_recall, p_recall, v_avg_vis, p_avg_vis, delta_vis, v_qps);
    }

    // ════════════════════════════════════════════════════════════════
    //  结论
    // ════════════════════════════════════════════════════════════════
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  结论分析");
    println_both!("═══════════════════════════════════════════════════════════════\n");
    println_both!("  V = Vamana (贪心搜索候选集), P = Perfect (暴力 kNN 候选集)");
    println_both!("  Δvis% < 0: P 的 avg_visited 更低 → 候选集质量是根因");
    println_both!("  Δvis% ≈ 0: 候选集质量无影响 → 排除此方向");
    println_both!("  Δvis% > 0: P 反而更差 → DirectionalPrune 已足够好");

    let mut f = std::fs::File::create("experiments/candidate_quality_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/candidate_quality_result.txt");
}
