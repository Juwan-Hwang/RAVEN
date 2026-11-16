//! Search-on-Graph 候选集升级实验（1M 全量）
//!
//! 发现：暴力 kNN 候选集在 100k 子集上将 avg_visited 降低 26.3%。
//! 但暴力 kNN 在 1M 上需要 ~3 小时，不可行。
//!
//! 替代方案：search-on-graph（NSG 核心思路）
//! 1. 标准建图（2 轮迭代，L_build=384）→ 候选集来自部分图上的贪心搜索
//! 2. 用完整图重新搜索每个节点（L_search=1000）→ 候选集来自完整图上的高 ef 搜索
//! 3. 用新候选集重建图（build_from_candidates）
//! 4. 对比 avg_visited
//!
//! 原理：标准 Vamana 建图时，每个节点看到的是部分图（前面 batch 的边），
//! 候选集质量受限。完整图上用高 ef 搜索能找到更好的候选邻居。
//!
//! 优化：SoG 候选收集用 SQ4 距离（而非 f32），速度 ~3x，不影响候选质量
//! （DirectionalPrune 内部用 f32 精确距离剪枝）。
//!
//! 用法：cargo run --release --bin search_on_graph_bench

use std::io::Write;
use std::time::Instant;

use raven::bench::util::{compute_recall, SiftDataset};
use raven::build::ChaCha8Rng;
use raven::graph::linear_pool::LinearPool;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph, GraphSearcher};
use raven::memory::VisitedTracker;
use raven::quant::SQ4Dataset;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;
const SEARCH_L: usize = 1000; // search-on-graph 的搜索宽度

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
    let n = data.n;

    println_both!("=== Search-on-Graph 候选集升级实验（1M 全量）===\n");
    println_both!("数据: n={}, dim={}, nq={}", n, dim, nq);
    println_both!("Search-on-graph: L_search={} (SQ4 加速)", SEARCH_L);

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
    //  1. 标准 Vamana 建图
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[1/4] 标准 Vamana 建图...");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let vamana_graph = VamanaGraph::build(&data.train, dim, &config, &mut rng);
    let vamana_build_time = t0.elapsed().as_secs_f64();
    println_both!("  建图: {:.0}s", vamana_build_time);

    let vamana_sq4 = SQ4Dataset::build(&data.train, dim);
    println_both!("  SQ4 编码完成");

    // ════════════════════════════════════════════════════════════════
    //  2. Search-on-Graph：用 SQ4 搜索获取高质量候选集
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[2/4] Search-on-Graph 候选集收集（L_search={}，SQ4）...", SEARCH_L);
    let t0 = Instant::now();
    let storage = vamana_graph.storage();
    let entry_point = vamana_graph.entry_point();
    let progress = AtomicUsize::new(0);
    let progress_interval = (n / 20).max(1);

    // 并行 SQ4 搜索：每个节点在完整图上搜索自身，收集 visited set 作为候选
    let sog_candidates: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let visited = VisitedTracker::new(n, SEARCH_L);
                let pool = LinearPool::new(SEARCH_L);
                (visited, pool)
            },
            |(visited, pool), node_id| {
                let idx = progress.fetch_add(1, Ordering::Relaxed);
                if idx > 0 && idx.is_multiple_of(progress_interval) {
                    eprintln!("[sog] {}/{} ({}%)", idx, n, idx * 100 / n);
                }

                // 直接用 SQ4 码作为查询（节点自身的 SQ4 编码）
                let query_code = vamana_sq4.code(node_id);

                // SQ4 贪心搜索（比 f32 快 ~3x）
                VamanaGraph::greedy_search_sq4(
                    &vamana_sq4, storage, entry_point, query_code,
                    SEARCH_L, visited, pool, PO, 0,
                );

                // visited set 就是候选集（排除自身）
                visited.visited_nodes()
                    .iter()
                    .filter(|&&v| v != node_id as u32)
                    .copied()
                    .collect()
            },
        )
        .collect();
    let sog_time = t0.elapsed().as_secs_f64();
    println_both!("  Search-on-Graph: {:.0}s", sog_time);

    // 候选集大小统计
    let avg_cand: f64 = sog_candidates.iter().map(|c| c.len()).sum::<usize>() as f64 / n as f64;
    let min_cand = sog_candidates.iter().map(|c| c.len()).min().unwrap();
    let max_cand = sog_candidates.iter().map(|c| c.len()).max().unwrap();
    println_both!("  候选集大小: avg={:.0}, min={}, max={}", avg_cand, min_cand, max_cand);

    // ════════════════════════════════════════════════════════════════
    //  3. 从 search-on-graph 候选集重建图
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[3/4] 从 search-on-graph 候选集重建图...");
    let t0 = Instant::now();
    let sog_graph = VamanaGraph::build_from_candidates(
        &data.train, dim, &sog_candidates, &config,
    );
    let sog_build_time = t0.elapsed().as_secs_f64();
    println_both!("  重建: {:.0}s", sog_build_time);

    let sog_sq4 = SQ4Dataset::build(&data.train, dim);
    println_both!("  SQ4 编码完成");

    let stats_v = vamana_graph.degree_stats();
    let stats_s = sog_graph.degree_stats();
    println_both!("  Vamana:  avg_degree={:.1}, max={}, p95={}, p99={}",
        stats_v.mean_degree, stats_v.max_degree, stats_v.p95_degree, stats_v.p99_degree);
    println_both!("  SoG:     avg_degree={:.1}, max={}, p95={}, p99={}",
        stats_s.mean_degree, stats_s.max_degree, stats_s.p95_degree, stats_s.p99_degree);

    // ════════════════════════════════════════════════════════════════
    //  4. 搜索对比
    // ════════════════════════════════════════════════════════════════
    println_both!("\n[4/4] 搜索对比（ef=35/40/50/65/100）\n");
    println_both!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "ef", "V_recall", "S_recall", "V_avgvis", "S_avgvis", "Δvis%", "V_QPS", "S_QPS");
    println_both!("  {}", "-".repeat(90));

    for &ef in &[35, 40, 50, 65, 100] {
        // Vamana baseline
        let mut s_v = GraphSearcher::new(&data.train, &vamana_graph, ef);
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

        // Search-on-Graph
        let mut s_s = GraphSearcher::new(&data.train, &sog_graph, ef);
        s_s.with_prefetch_offset(PO);
        s_s.with_rerank_factor(RR);
        s_s.with_sq4(&sog_sq4);

        let mut s_results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut s_total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let r = s_s.search_sq4(&data.test[q * dim..(q + 1) * dim], K);
            s_results.push(r.iter().map(|(id, _)| *id).collect());
            s_total_visited += s_s.last_visited_count();
        }
        let s_dt = t0.elapsed();
        let s_recall = compute_recall(&s_results, &data.gt, data.gt_k, K, nq);
        let s_avg_vis = s_total_visited as f64 / nq as f64;
        let s_qps = nq as f64 / s_dt.as_secs_f64();

        let delta_vis = (s_avg_vis - v_avg_vis) / v_avg_vis * 100.0;

        println_both!("  {:>6}  {:>10.4}  {:>10.4}  {:>10.1}  {:>10.1}  {:>+9.1}%  {:>10.0}  {:>10.0}",
            ef, v_recall, s_recall, v_avg_vis, s_avg_vis, delta_vis, v_qps, s_qps);
    }

    // ════════════════════════════════════════════════════════════════
    //  结论
    // ════════════════════════════════════════════════════════════════
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  结论分析");
    println_both!("═══════════════════════════════════════════════════════════════\n");
    println_both!("  V = Vamana（标准建图，候选集来自部分图贪心搜索）");
    println_both!("  S = Search-on-Graph（完整图高 ef 搜索候选集，NSG 思路）");
    println_both!("  100k 子集实验已证明：暴力 kNN 候选集降低 avg_visited 26.3%");
    println_both!("  本实验验证：search-on-graph 能否在 1M 全量复现类似改善");
    println_both!("\n  Δvis% < 0: SoG 有效 → 候选集质量是根因，值得投入 NSG 完整实现");
    println_both!("  Δvis% ≈ 0: SoG 无效 → Vamana 两轮迭代已足够，需其他方向");

    let mut f = std::fs::File::create("experiments/search_on_graph_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/search_on_graph_result.txt");
}
