//! BFS 图重排最小验证 — 实测 ns/visit 变化
//!
//! 核心假设：当前节点 ID 是建图顺序的任意排列，
//! BFS 重排后相连节点内存相邻 → cache miss 降低 → ns/visit 降低。
//!
//! 方法：
//! 1. 正常建图
//! 2. 从 entry_point BFS，得到 old_to_new 映射
//! 3. 用映射重建图存储 + SQ4 codes + 训练向量
//! 4. 重建分层导航
//! 5. 对比原始 vs 重排后的 ns/visit
//!
//! 注意：图拓扑完全相同（相同边），只改节点 ID → 内存布局。
//! 分层导航会略有不同（确定性 RNG 但节点顺序变了），不影响验证。
//!
//! 用法：cargo run --release --bin reorder_bfs_bench

use std::collections::VecDeque;
use std::io::Write;
use std::time::Instant;

use raven::bench::util::{SiftDataset, compute_recall, median3};
use raven::build::ChaCha8Rng;
use raven::graph::{
    GraphSearcher, PruneStrategy, VamanaBuildConfig,
    VamanaGraph,
};
use raven::graph::navigation::LayeredNavigation;
use raven::memory::graph::HybridBlockedCsr;
use raven::quant::SQ4Dataset;
use raven::SENTINEL;

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;
const EF_LIST: [usize; 5] = [35, 40, 50, 65, 80];

struct BenchResult {
    recall: f64,
    qps: f64,
    visited: f64,
}

fn bench(
    train: &[f32], graph: &VamanaGraph, test: &[f32], dim: usize, nq: usize,
    gt: &[i32], gt_k: usize, ef: usize, sq4: &SQ4Dataset,
    id_remap: Option<&[u32]>,
) -> BenchResult {
    let run = || {
        let mut s = GraphSearcher::new(train, graph, ef);
        s.with_prefetch_offset(PO);
        s.with_rerank_factor(RR);
        s.with_sq4(sq4);
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let r = s.search_sq4(&test[q * dim..(q + 1) * dim], K);
            total_visited += s.last_visited_count();
            // 如果提供了 id_remap（new_id -> old_id），将搜索结果映射回原始 ID
            if let Some(remap) = id_remap {
                results.push(r.iter().map(|(id, _)| remap[*id as usize]).collect());
            } else {
                results.push(r.iter().map(|(id, _)| *id).collect());
            }
        }
        let dt = t0.elapsed();
        (compute_recall(&results, gt, gt_k, K, nq), nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };
    run(); // warmup
    let r1 = run();
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: median3(r1.0, r2.0, r3.0),
        qps: median3(r1.1, r2.1, r3.1),
        visited: median3(r1.2, r2.2, r3.2),
    }
}

/// 从 entry_point 开始 BFS，返回 old_to_new 映射
/// old_to_new[old_id] = new_id
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

    // 处理孤立节点（BFS 未到达的）
    for i in 0..n {
        if old_to_new[i] == u32::MAX {
            old_to_new[i] = next_id;
            next_id += 1;
        }
    }

    old_to_new
}

/// 用 old_to_new 映射重建图存储
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

/// 用 old_to_new 映射重排 SQ4 codes
fn remap_sq4(sq4: &SQ4Dataset, old_to_new: &[u32]) -> SQ4Dataset {
    let n = sq4.n;
    let bytes_per_vec = sq4.codes.len() / n;
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
        code_bytes: sq4.code_bytes,
    }
}

/// 用 old_to_new 映射重排训练向量
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

    println_both!("=== BFS 图重排最小验证 (alpha_core=0.93) ===\n");
    println_both!("数据: n={}, dim={}, nq={}, k={}", data.n, data.dim, data.nq, K);
    println_both!("参数: R=40, L=384, po={}, rr={}, DirectionalPrune, alpha_core=0.93", PO, RR);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

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
    let graph = VamanaGraph::build(&data.train, data.dim, &config, &mut rng);
    println_both!("建图: {:.0}s\n", t0.elapsed().as_secs_f64());

    let sq4 = SQ4Dataset::build(&data.train, data.dim);
    let entry = graph.entry_point();
    let n = graph.len();

    // BFS 重排
    println_both!("BFS 重排中... ");
    let t0 = Instant::now();
    let old_to_new = bfs_order(&graph, entry, n);

    // 验证映射完整性
    let mut seen = vec![false; n];
    for &new_id in &old_to_new {
        seen[new_id as usize] = true;
    }
    assert!(seen.iter().all(|&s| s), "BFS mapping incomplete!");

    let new_storage = remap_graph(&graph, &old_to_new);
    let new_sq4 = remap_sq4(&sq4, &old_to_new);
    let new_train = remap_vectors(&data.train, data.dim, &old_to_new);
    let new_entry = old_to_new[entry as usize];

    // 重建分层导航
    let new_nav = LayeredNavigation::build(
        &new_train, data.dim, &new_storage, new_entry, 32, 20,
    );

    // 构建重排后的 VamanaGraph
    let mut reordered_graph = VamanaGraph::from_storage(new_storage, new_entry, data.dim);
    reordered_graph.set_layered_nav(new_nav);

    println_both!("重排完成: {:.1}s\n", t0.elapsed().as_secs_f64());

    // 构建 new_to_old 逆映射（用于搜索结果映射回原始 ID 计算 recall）
    let mut new_to_old = vec![0u32; n];
    for old_id in 0..n {
        new_to_old[old_to_new[old_id] as usize] = old_id as u32;
    }

    // 对比 benchmark
    println_both!("--- 原始 vs BFS重排 ---\n");
    println_both!(
        "  {:>6}  {:>6}  {:>8}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}  {:>8}  {:>8}",
        "ef", "mode", "recall", "QPS", "visited", "ns/vis",
        "recall", "QPS", "visited", "ns/vis", "dns/vis", "dQPS%"
    );
    println_both!("  {}", "-".repeat(110));

    let mut orig_results: Vec<(usize, BenchResult)> = Vec::new();
    let mut reordered_results: Vec<(usize, BenchResult)> = Vec::new();

    for &ef in &EF_LIST {
        let r_orig = bench(&data.train, &graph, &data.test, data.dim, data.nq, &data.gt, data.gt_k, ef, &sq4, None);
        let r_reord = bench(&new_train, &reordered_graph, &data.test, data.dim, data.nq, &data.gt, data.gt_k, ef, &new_sq4, Some(&new_to_old));

        let ns_orig = 1e9 / r_orig.qps / r_orig.visited;
        let ns_reord = 1e9 / r_reord.qps / r_reord.visited;
        let delta_ns = ns_reord - ns_orig;
        let delta_qps = (r_reord.qps - r_orig.qps) / r_orig.qps * 100.0;

        println_both!(
            "  {:>6}  {:>6}  {:>8.4}  {:>10.0}  {:>10.1}  {:>6.1}ns  {:>8.4}  {:>10.0}  {:>10.1}  {:>6.1}ns  {:>+6.1}ns  {:>+6.1}%",
            ef, "orig", r_orig.recall, r_orig.qps, r_orig.visited, ns_orig,
            r_reord.recall, r_reord.qps, r_reord.visited, ns_reord,
            delta_ns, delta_qps
        );

        orig_results.push((ef, r_orig));
        reordered_results.push((ef, r_reord));
    }

    // 汇总
    println_both!("\n--- 汇总 ---\n");

    let avg_ns_orig: f64 = orig_results.iter()
        .map(|(_, r)| 1e9 / r.qps / r.visited)
        .sum::<f64>() / orig_results.len() as f64;
    let avg_ns_reord: f64 = reordered_results.iter()
        .map(|(_, r)| 1e9 / r.qps / r.visited)
        .sum::<f64>() / reordered_results.len() as f64;
    let avg_qps_orig: f64 = orig_results.iter().map(|(_, r)| r.qps).sum::<f64>() / orig_results.len() as f64;
    let avg_qps_reord: f64 = reordered_results.iter().map(|(_, r)| r.qps).sum::<f64>() / reordered_results.len() as f64;

    println_both!("  平均 ns/visit: 原始={:.1}ns -> 重排={:.1}ns ({:+.1}%)",
        avg_ns_orig, avg_ns_reord, (avg_ns_reord - avg_ns_orig) / avg_ns_orig * 100.0);
    println_both!("  平均 QPS:       原始={:.0} -> 重排={:.0} ({:+.1}%)",
        avg_qps_orig, avg_qps_reord, (avg_qps_reord - avg_qps_orig) / avg_qps_orig * 100.0);

    // recall 一致性检查
    let max_recall_diff: f64 = orig_results.iter()
        .zip(reordered_results.iter())
        .map(|((_, o), (_, r))| (r.recall - o.recall).abs())
        .fold(0.0f64, f64::max);
    println_both!("  最大 recall 差异: {:.4} (应接近0，差异来自分层导航重建)", max_recall_diff);

    // 写文件
    let mut f = std::fs::File::create("experiments/reorder_bfs_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/reorder_bfs_result.txt");
}
