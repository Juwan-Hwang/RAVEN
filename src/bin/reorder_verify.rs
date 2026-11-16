//! BFS 重排正确性抽查 + 稳定性验证
//!
//! 正确性抽查：
//!   随机选 20 个 query，对每个 query：
//!   1. 原始图搜索 top-10
//!   2. 重排图搜索 top-10，映射回原始 ID
//!   3. 计算这些 ID 对应向量到 query 的真实 f32 L2 距离
//!   4. 与 ground truth 的距离对比，确认确实是最近邻
//!   5. 检查两种图返回的 ID 集合是否一致（拓扑相同时应完全相同）
//!
//! 稳定性验证：
//!   对 ef=50 和 ef=65，各重复跑 5 次（1 warmup + 5 rounds），看 ns/visit 波动
//!
//! 用法：cargo run --release --bin reorder_verify

use std::collections::VecDeque;
use std::io::Write;
use std::time::Instant;

use raven::bench::util::SiftDataset;
use raven::build::ChaCha8Rng;
use raven::graph::navigation::LayeredNavigation;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::graph::HybridBlockedCsr;
use raven::quant::SQ4Dataset;
use raven::SENTINEL;
use rand::Rng;

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;

// ─── BFS 重排（复用 reorder_bfs_bench 的逻辑） ─────────────────────

fn bfs_order(graph: &VamanaGraph, entry: u32, n: usize) -> Vec<u32> {
    let mut old_to_new = vec![u32::MAX; n];
    let mut queue = VecDeque::new();
    let mut next_id = 0u32;

    old_to_new[entry as usize] = next_id;
    next_id += 1;
    queue.push_back(entry);

    while let Some(node) = queue.pop_front() {
        for &neighbor in graph.neighbors(node) {
            if neighbor == SENTINEL {
                continue;
            }
            let idx = neighbor as usize;
            if old_to_new[idx] == u32::MAX {
                old_to_new[idx] = next_id;
                next_id += 1;
                queue.push_back(neighbor);
            }
        }
    }

    for i in 0..n {
        if old_to_new[i] == u32::MAX {
            old_to_new[i] = next_id;
            next_id += 1;
        }
    }
    old_to_new
}

fn remap_graph(graph: &VamanaGraph, old_to_new: &[u32]) -> HybridBlockedCsr {
    let n = graph.len();
    let r_max = graph.storage().r_max();
    let mut new_storage = HybridBlockedCsr::new(n, r_max);
    for old_id in 0..n as u32 {
        let new_id = old_to_new[old_id as usize];
        let neighbors = graph.neighbors(old_id);
        let remapped: Vec<u32> = neighbors
            .iter()
            .map(|&old_n| old_to_new[old_n as usize])
            .collect();
        new_storage.set_neighbors(new_id, &remapped);
    }
    new_storage
}

fn remap_sq4(sq4: &SQ4Dataset, old_to_new: &[u32]) -> SQ4Dataset {
    let n = sq4.n;
    let bytes_per_vec = sq4.code_bytes;
    let mut new_codes = vec![0u8; sq4.codes.len()];
    for old_id in 0..n {
        let new_id = old_to_new[old_id] as usize;
        let old_start = old_id * bytes_per_vec;
        let new_start = new_id * bytes_per_vec;
        new_codes[new_start..new_start + bytes_per_vec]
            .copy_from_slice(&sq4.codes[old_start..old_start + bytes_per_vec]);
    }
    SQ4Dataset {
        codes: new_codes.into(),
        params: sq4.params.clone(),
        dim: sq4.dim,
        n,
        code_bytes: bytes_per_vec,
    }
}

fn remap_vectors(train: &[f32], dim: usize, old_to_new: &[u32]) -> Vec<f32> {
    let n = old_to_new.len();
    let mut new_train = vec![0.0f32; n * dim];
    for old_id in 0..n {
        let new_id = old_to_new[old_id] as usize;
        new_train[new_id * dim..(new_id + 1) * dim]
            .copy_from_slice(&train[old_id * dim..(old_id + 1) * dim]);
    }
    new_train
}

// ─── f32 L2 距离 ───────────────────────────────────────────────────

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─── 单次搜索测量 ──────────────────────────────────────────────────

struct SingleRun {
    qps: f64,
    visited: f64,
}

fn single_run(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    ef: usize, sq4: &SQ4Dataset,
) -> SingleRun {
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
    SingleRun {
        qps: nq as f64 / dt.as_secs_f64(),
        visited: total_visited as f64 / nq as f64,
    }
}

fn main() {
    let mut out = String::new();
    macro_rules! println_both {
        () => {{ println!(); out.push('\n'); }};
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

    println_both!("=== BFS 重排正确性抽查 + 稳定性验证 ===\n");
    println_both!("数据: n={}, dim={}, nq={}", data.n, dim, nq);

    // 建图
    std::env::set_var("RAVEN_ALPHA_CORE", "0.93");
    println_both!("建图中... ");
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
    let entry = graph.entry_point();
    let n = graph.len();

    // BFS 重排
    println_both!("BFS 重排中... ");
    let old_to_new = bfs_order(&graph, entry, n);
    let new_storage = remap_graph(&graph, &old_to_new);
    let new_sq4 = remap_sq4(&sq4, &old_to_new);
    let new_train = remap_vectors(&data.train, dim, &old_to_new);
    let new_entry = old_to_new[entry as usize];

    // new_to_old 逆映射
    let mut new_to_old = vec![0u32; n];
    for old_id in 0..n {
        new_to_old[old_to_new[old_id] as usize] = old_id as u32;
    }

    let new_nav = LayeredNavigation::build(&new_train, dim, &new_storage, new_entry, 32, 20);
    let mut reordered_graph = VamanaGraph::from_storage(new_storage, new_entry, dim);
    reordered_graph.set_layered_nav(new_nav);
    println_both!("重排完成\n");

    // ════════════════════════════════════════════════════════════════
    //  Part 1: 正确性抽查
    // ════════════════════════════════════════════════════════════════
    println_both!("═══════════════════════════════════════════════════════════════");
    println_both!("  Part 1: 正确性抽查（20 个随机 query）");
    println_both!("═══════════════════════════════════════════════════════════════\n");

    // 用确定性种子选 20 个 query（均匀分布）
    let mut qrng = ChaCha8Rng::seed_from(999);
    let mut sample_queries = Vec::with_capacity(20);
    for _ in 0..20 {
        sample_queries.push((qrng.inner().gen::<u32>() as usize) % nq);
    }

    let ef_check = 50;

    // 原始图搜索
    let mut orig_searcher = GraphSearcher::new(&data.train, &graph, ef_check);
    orig_searcher.with_prefetch_offset(PO);
    orig_searcher.with_rerank_factor(RR);
    orig_searcher.with_sq4(&sq4);

    // 重排图搜索
    let mut reord_searcher = GraphSearcher::new(&new_train, &reordered_graph, ef_check);
    reord_searcher.with_prefetch_offset(PO);
    reord_searcher.with_rerank_factor(RR);
    reord_searcher.with_sq4(&new_sq4);

    let mut all_pass = true;

    println_both!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>8}",
        "q#", "orig_top1", "reord_top1", "gt_top1", "orig_dist", "reord_d", "gt_dist", "ID匹配");
    println_both!("  {}", "-".repeat(80));

    for (i, &qi) in sample_queries.iter().enumerate() {
        let query = &data.test[qi * dim..(qi + 1) * dim];

        // 原始图搜索
        let orig_result = orig_searcher.search_sq4(query, K);
        let orig_ids: Vec<u32> = orig_result.iter().map(|(id, _)| *id).collect();

        // 重排图搜索 → 映射回原始 ID
        let reord_result = reord_searcher.search_sq4(query, K);
        let reord_ids_mapped: Vec<u32> = reord_result
            .iter()
            .map(|(new_id, _)| new_to_old[*new_id as usize])
            .collect();

        // Ground truth
        let gt_ids: Vec<u32> = data.gt[qi * data.gt_k..qi * data.gt_k + K]
            .iter()
            .map(|&g| g as u32)
            .collect();

        // 计算距离
        let orig_top1 = orig_ids[0];
        let reord_top1 = reord_ids_mapped[0];
        let gt_top1 = gt_ids[0];

        let orig_dist = l2_distance(query, &data.train[orig_top1 as usize * dim..(orig_top1 as usize + 1) * dim]);
        let reord_dist = l2_distance(query, &data.train[reord_top1 as usize * dim..(reord_top1 as usize + 1) * dim]);
        let gt_dist = l2_distance(query, &data.train[gt_top1 as usize * dim..(gt_top1 as usize + 1) * dim]);

        // 检查 ID 集合是否一致（拓扑相同时应完全相同）
        let mut orig_sorted = orig_ids.clone();
        orig_sorted.sort();
        let mut reord_sorted = reord_ids_mapped.clone();
        reord_sorted.sort();
        let id_match = orig_sorted == reord_sorted;

        // 检查距离是否合理（重排后 top1 距离应 ≤ gt_top1 距离，或非常接近）
        let dist_ok = reord_dist <= gt_dist * 1.001; // 允许 0.1% 浮动（SQ4 rerank 可能微调顺序）

        if !id_match || !dist_ok {
            all_pass = false;
        }

        let match_str = if id_match { "OK" } else { "MISMATCH" };
        let dist_str = if dist_ok { "OK" } else { "FAIL" };

        println_both!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>10.1}  {:>8.1}  {:>8.1}  {:>4} {:>4}",
            i + 1, orig_top1, reord_top1, gt_top1,
            orig_dist, reord_dist, gt_dist,
            match_str, dist_str);
    }

    println_both!("\n  正确性结论: {}", if all_pass { "ALL PASS" } else { "FAIL" });

    // 额外检查：重排后 top-10 的距离是否与原始 top-10 完全一致
    println_both!("\n  --- 距离一致性详细检查 ---\n");
    let mut dist_mismatches = 0;
    for &qi in &sample_queries {
        let query = &data.test[qi * dim..(qi + 1) * dim];

        let orig_result = {
            let mut s = GraphSearcher::new(&data.train, &graph, ef_check);
            s.with_prefetch_offset(PO);
            s.with_rerank_factor(RR);
            s.with_sq4(&sq4);
            s.search_sq4(query, K)
        };

        let reord_result = {
            let mut s = GraphSearcher::new(&new_train, &reordered_graph, ef_check);
            s.with_prefetch_offset(PO);
            s.with_rerank_factor(RR);
            s.with_sq4(&new_sq4);
            s.search_sq4(query, K)
        };

        // 计算原始图 top-10 的真实距离
        let orig_dists: Vec<f32> = orig_result.iter()
            .map(|(id, _)| {
                let v = &data.train[*id as usize * dim..(*id as usize + 1) * dim];
                l2_distance(query, v)
            })
            .collect();

        // 重排图 top-10 映射回原始 ID 后的真实距离
        let reord_dists: Vec<f32> = reord_result.iter()
            .map(|(new_id, _)| {
                let old_id = new_to_old[*new_id as usize] as usize;
                let v = &data.train[old_id * dim..(old_id + 1) * dim];
                l2_distance(query, v)
            })
            .collect();

        // 排序后对比
        let mut orig_sorted = orig_dists.clone();
        orig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut reord_sorted = reord_dists.clone();
        reord_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for j in 0..K {
            let diff = (orig_sorted[j] - reord_sorted[j]).abs();
            if diff > 0.01 {
                dist_mismatches += 1;
                if dist_mismatches <= 5 {
                    println_both!("  q{} rank{}: orig_dist={:.2} reord_dist={:.2} diff={:.4}",
                        qi, j + 1, orig_sorted[j], reord_sorted[j], diff);
                }
            }
        }
    }

    if dist_mismatches == 0 {
        println_both!("  距离一致性: ALL 20 queries × 10 results = 200 pairs 全部一致 (<0.01)");
    } else {
        println_both!("  距离一致性: {}/200 不一致", dist_mismatches);
    }

    // ════════════════════════════════════════════════════════════════
    //  Part 2: 稳定性验证 — ef=50 和 ef=65 各跑 5 次
    // ════════════════════════════════════════════════════════════════
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  Part 2: 稳定性验证（ef=50/65 各 1 warmup + 5 rounds）");
    println_both!("═══════════════════════════════════════════════════════════════\n");

    for &ef in &[50, 65] {
        println_both!("  --- ef={} ---\n", ef);
        println_both!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "round", "orig_QPS", "orig_vis", "orig_ns", "reord_QPS", "reord_vis", "reord_ns", "dns", "dQPS%");
        println_both!("  {}", "-".repeat(95));

        // warmup
        single_run(&data.train, &graph, &data.test, dim, nq, ef, &sq4);
        single_run(&new_train, &reordered_graph, &data.test, dim, nq, ef, &new_sq4);

        let mut orig_ns_list = Vec::new();
        let mut reord_ns_list = Vec::new();
        let mut orig_qps_list = Vec::new();
        let mut reord_qps_list = Vec::new();

        for round in 1..=5 {
            let ro = single_run(&data.train, &graph, &data.test, dim, nq, ef, &sq4);
            let rr = single_run(&new_train, &reordered_graph, &data.test, dim, nq, ef, &new_sq4);

            let ns_o = 1e9 / ro.qps / ro.visited;
            let ns_r = 1e9 / rr.qps / rr.visited;
            let dns = ns_r - ns_o;
            let dqps = (rr.qps - ro.qps) / ro.qps * 100.0;

            println_both!("  {:>6}  {:>10.0}  {:>10.1}  {:>8.1}ns  {:>10.0}  {:>10.1}  {:>8.1}ns  {:>+6.1}ns  {:>+6.1}%",
                round, ro.qps, ro.visited, ns_o, rr.qps, rr.visited, ns_r, dns, dqps);

            orig_ns_list.push(ns_o);
            reord_ns_list.push(ns_r);
            orig_qps_list.push(ro.qps);
            reord_qps_list.push(rr.qps);
        }

        let avg_orig_ns: f64 = orig_ns_list.iter().sum::<f64>() / 5.0;
        let avg_reord_ns: f64 = reord_ns_list.iter().sum::<f64>() / 5.0;
        let avg_orig_qps: f64 = orig_qps_list.iter().sum::<f64>() / 5.0;
        let avg_reord_qps: f64 = reord_qps_list.iter().sum::<f64>() / 5.0;
        let min_orig_ns = orig_ns_list.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_orig_ns = orig_ns_list.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_reord_ns = reord_ns_list.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_reord_ns = reord_ns_list.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let ns_range_orig = max_orig_ns - min_orig_ns;
        let ns_range_reord = max_reord_ns - min_reord_ns;

        println_both!("  {}", "-".repeat(95));
        println_both!("  {:>6}  orig  ns/visit avg={:.1}  range=[{:.1}-{:.1}]  spread={:.1}ns",
            "avg", avg_orig_ns, min_orig_ns, max_orig_ns, ns_range_orig);
        println_both!("  {:>6}  reord ns/visit avg={:.1}  range=[{:.1}-{:.1}]  spread={:.1}ns",
            "avg", avg_reord_ns, min_reord_ns, max_reord_ns, ns_range_reord);
        println_both!("  {:>6}  dns/visit={:+.1}ns ({:+.1}%)  dQPS={:+.1}%",
            "Δ", avg_reord_ns - avg_orig_ns,
            (avg_reord_ns - avg_orig_ns) / avg_orig_ns * 100.0,
            (avg_reord_qps - avg_orig_qps) / avg_orig_qps * 100.0);

        // 判断：波动是否大于改善幅度
        if ns_range_orig > (avg_orig_ns - avg_reord_ns).abs() {
            println_both!("  ⚠ 原始图 ns/visit 波动范围 ({:.1}ns) > 改善幅度 ({:.1}ns)，改善可能被噪声掩盖", 
                ns_range_orig, (avg_orig_ns - avg_reord_ns).abs());
        } else {
            println_both!("  ✓ 改善幅度 ({:.1}ns) > 原始图波动范围 ({:.1}ns)，改善信号清晰",
                (avg_orig_ns - avg_reord_ns).abs(), ns_range_orig);
        }
        println_both!();
    }

    // 写文件
    let mut f = std::fs::File::create("experiments/reorder_verify_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/reorder_verify_result.txt");
}
