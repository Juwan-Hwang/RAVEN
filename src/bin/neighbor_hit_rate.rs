//! 邻居命中率分析
//!
//! 在真实 SQ4 搜索路径上统计：
//! 1. 候选池插入率（主指标）：每次 pop 的第 i 个邻居 insert 成功的比例
//! 2. 按 neighbor rank 切片：第 1 个 vs 第 40 个邻居的插入率变化
//! 3. 池存活率：插入后存活到搜索结束的比例（仍在 final pool 中）
//!
//! 定义：
//! - insert 成功 = pool.insert(v, d) 返回 true（v 比 pool 最差的还近）
//! - insert 失败 = v 比 pool 最差的还远（被拒绝）
//! - 分母 = 所有未访问邻居（即 insert 尝试次数）
//! - 存活 = 插入成功后，节点出现在搜索结束时的 pool 中
//!
//! 用法：cargo run --release --bin neighbor_hit_rate

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::linear_pool::LinearPool;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::VisitedTracker;
use raven::quant::SQ4Dataset;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
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
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
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

fn compute_recall(results: &[Vec<u32>], gt: &[i32], gt_k: usize, k: usize, nq: usize) -> f64 {
    let mut hits = 0u64;
    for q in 0..nq {
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &id in &results[q] {
            if gt_slice.contains(&(id as i32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rerank_factor: usize,
) -> Vec<(u32, f32)> {
    let n_rerank = (k * rerank_factor).min(candidates.len());
    let mut results: Vec<(u32, f32)> = candidates[..n_rerank]
        .iter()
        .map(|(id, _)| {
            let dist = l2_simd(
                query,
                &vectors[*id as usize * dim..(*id as usize + 1) * dim],
            );
            (*id, dist)
        })
        .collect();
    results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(k);
    results
}

/// 搜索追踪结果
struct TrackResult {
    /// rerank 后的 top-k 结果
    results: Vec<(u32, f32)>,
    /// per-rank: (insert_attempts, insert_successes)
    rank_stats: Vec<(u64, u64)>,
    /// per-rank: 插入成功的 [(node_id, insert_pop_order)]
    rank_inserted: Vec<Vec<(u32, usize)>>,
    /// 搜索结束时 pool 中的节点集合
    final_pool: HashSet<u32>,
    /// 总访问数
    visited_count: usize,
    /// 总 pop 次数
    total_pops: usize,
}

/// 带 neighbor rank 追踪的 SQ4 搜索
fn search_with_tracking(
    sq4: &SQ4Dataset,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query_code: &[u8],
    ef: usize,
    visited: &mut VisitedTracker,
    pool: &mut LinearPool,
    po: usize,
    k: usize,
    rr: usize,
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    max_r: usize,
) -> TrackResult {
    visited.reset();
    pool.reset_for(ef);

    let mut rank_stats: Vec<(u64, u64)> = vec![(0, 0); max_r];
    let mut rank_inserted: Vec<Vec<(u32, usize)>> = vec![Vec::new(); max_r];

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];
    let mut pop_order: usize = 0;

    while let Some((node, _dist)) = pool.pop() {
        pop_order += 1;

        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            unsafe {
                raven::arch::prefetch_l1(ptr);
                raven::arch::prefetch_l1(ptr.add(64));
                raven::arch::prefetch_l1(ptr.add(128));
                raven::arch::prefetch_l1(ptr.add(192));
            }
        }

        let neighbors = storage.neighbors(node);

        // 第一遍：收集未访问邻居到 edge_buf，同时记录 rank
        // 使用并行数组记录每个 edge_buf 元素对应的 rank
        let mut edge_ranks: [usize; 128] = [0; 128];
        let mut edge_size = 0usize;

        for (rank, &v) in neighbors.iter().enumerate() {
            if rank >= max_r || edge_size >= 128 {
                break;
            }
            let is_new = unsafe { visited.visit_unchecked_branchless(v) };
            if is_new {
                unsafe {
                    *edge_buf.get_unchecked_mut(edge_size) = v;
                    *edge_ranks.get_unchecked_mut(edge_size) = rank;
                }
                edge_size += 1;
            }
        }

        // 预取
        let prefetch_count = po.min(edge_size);
        for i in 0..prefetch_count {
            let v = edge_buf[i] as usize;
            let ptr = sq4.code(v).as_ptr() as *const i8;
            unsafe { raven::arch::prefetch_l1(ptr); }
        }

        // 第二遍：距离计算 + insert + 按 rank 统计
        for i in 0..edge_size {
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = sq4.code(v).as_ptr() as *const i8;
                unsafe { raven::arch::prefetch_l1(ptr); }
            }
            let neighbor = edge_buf[i];
            let rank = edge_ranks[i];
            let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
            let inserted = pool.insert(neighbor, d);

            rank_stats[rank].0 += 1;
            if inserted {
                rank_stats[rank].1 += 1;
                rank_inserted[rank].push((neighbor, pop_order));
            }
        }
    }

    let vc = visited.visited_count();
    let total_pops = pop_order;

    // 提取 final pool
    let candidates = pool.to_sorted_vec();
    let final_pool: HashSet<u32> = candidates.iter().map(|(id, _)| *id).collect();

    let results = rerank_f32(vectors, dim, query, candidates, k, rr);

    TrackResult {
        results,
        rank_stats,
        rank_inserted,
        final_pool,
        visited_count: vc,
        total_pops,
    }
}

fn main() {
    println!("=== 邻居命中率分析 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = 1000.min(nq_full);
    let k = 10;
    let ef = 50;
    let po = 16;
    let rr = 5;
    let max_r = 40;

    println!("--- 建图 ---");
    let config = VamanaBuildConfig {
        alpha: 1.2, l_build: 384, r_max: 40, r_soft: 60,
        max_iterations: 2, saturate: false,
        enable_layered_nav: true, nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.1}s\n", t0.elapsed().as_secs_f64());

    println!("--- 构建 SQ4 ---");
    let t0 = Instant::now();
    let sq4 = SQ4Dataset::build(&train, dim);
    println!("SQ4 构建时间: {:.1}s\n", t0.elapsed().as_secs_f64());

    let nav = graph.layered_nav().expect("need layered nav");
    let mut visited = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);
    let code_bytes = dim.div_ceil(2);
    let mut query_code_buf = vec![0u8; code_bytes];

    // 累积统计
    let mut global_attempts: Vec<u64> = vec![0; max_r];
    let mut global_successes: Vec<u64> = vec![0; max_r];
    // 存活：插入成功后仍在 final pool 中
    let mut global_survived: Vec<u64> = vec![0; max_r];

    let mut total_visited = 0usize;
    let mut total_pops = 0usize;
    let mut all_results = Vec::with_capacity(nq);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, _) = nav.initialize(&train, dim, query);

        let tr = search_with_tracking(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            ef, &mut visited, &mut pool, po, k, rr,
            &train, dim, query, max_r,
        );

        total_visited += tr.visited_count;
        total_pops += tr.total_pops;
        all_results.push(tr.results.iter().map(|(id, _)| *id).collect());

        // 累积 per-rank 统计
        for r in 0..max_r {
            global_attempts[r] += tr.rank_stats[r].0;
            global_successes[r] += tr.rank_stats[r].1;

            // 存活：插入成功的节点是否仍在 final pool
            for &(node, _pop) in &tr.rank_inserted[r] {
                if tr.final_pool.contains(&node) {
                    global_survived[r] += 1;
                }
            }
        }
    }

    let dt = t0.elapsed();
    let recall = compute_recall(&all_results, &gt, gt_k, k, nq);
    let avg_visited = total_visited as f64 / nq as f64;
    let avg_pops = total_pops as f64 / nq as f64;

    println!("--- 搜索结果 ---");
    println!("recall@{}: {:.4}", k, recall);
    println!("avg_visited: {:.1}", avg_visited);
    println!("avg_pops: {:.1}", avg_pops);
    println!("搜索时间: {:.1}s\n", dt.as_secs_f64());

    // ── per-rank 报告 ──
    println!("=== Per-Rank 候选池插入率 + 存活率 ===\n");
    println!(
        "{:<6} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "rank", "attempts", "inserted", "insert%", "survived", "survive%"
    );
    println!("{}", "-".repeat(60));

    for r in 0..max_r {
        let att = global_attempts[r];
        let suc = global_successes[r];
        let sur = global_survived[r];

        let ins_rate = if att > 0 { suc as f64 / att as f64 * 100.0 } else { 0.0 };
        let sur_rate = if suc > 0 { sur as f64 / suc as f64 * 100.0 } else { 0.0 };

        println!(
            "{:<6} {:<10} {:<10} {:<10.1} {:<10} {:<10.1}",
            r + 1, att, suc, ins_rate, sur, sur_rate,
        );
    }

    // 汇总
    println!("\n=== 汇总 ===");
    let total_att: u64 = global_attempts.iter().sum();
    let total_suc: u64 = global_successes.iter().sum();
    let total_sur: u64 = global_survived.iter().sum();
    println!("总邻居尝试: {}", total_att);
    println!("总插入成功: {} ({:.1}%)", total_suc, total_suc as f64 / total_att as f64 * 100.0);
    println!("总存活到结束: {} ({:.1}% of inserted)", total_sur, total_sur as f64 / total_suc as f64 * 100.0);

    // ── 按 rank 区间统计 ──
    let bins: [(usize, usize); 6] = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40)];
    println!("\n=== 按 rank 区间 ===\n");
    println!(
        "{:<12} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "rank_range", "attempts", "inserted", "insert%", "survived", "survive%"
    );
    println!("{}", "-".repeat(65));

    for (lo, hi) in &bins {
        let mut att = 0u64;
        let mut suc = 0u64;
        let mut sur = 0u64;
        for r in *lo..*hi {
            att += global_attempts[r];
            suc += global_successes[r];
            sur += global_survived[r];
        }
        let ins_rate = if att > 0 { suc as f64 / att as f64 * 100.0 } else { 0.0 };
        let sur_rate = if suc > 0 { sur as f64 / suc as f64 * 100.0 } else { 0.0 };
        println!(
            "{:<12} {:<10} {:<10} {:<10.1} {:<10} {:<10.1}",
            format!("{}-{}", lo + 1, hi), att, suc, ins_rate, sur, sur_rate,
        );
    }

    // ── 边际贡献分析 ──
    println!("\n=== 边际贡献：每个 rank 贡献的插入数占总插入数的比例 ===\n");
    let total_inserts = total_suc as f64;
    for r in 0..max_r {
        let pct = global_successes[r] as f64 / total_inserts * 100.0;
        let bar = "#".repeat((pct * 2.0) as usize);
        println!(
            "rank {:<3} {:<8} ({:<5.1}%) {}",
            r + 1, global_successes[r], pct, bar,
        );
    }

    // ── 存活贡献分析 ──
    println!("\n=== 存活贡献：每个 rank 贡献的存活数占总存活数的比例 ===\n");
    let total_survived = total_sur as f64;
    for r in 0..max_r {
        let pct = global_survived[r] as f64 / total_survived * 100.0;
        let bar = "#".repeat((pct * 2.0) as usize);
        println!(
            "rank {:<3} {:<8} ({:<5.1}%) {}",
            r + 1, global_survived[r], pct, bar,
        );
    }

    // ── 关键指标：前 N 个邻居的贡献占比 ──
    println!("\n=== 累积贡献占比 ===\n");
    println!("{:<12} {:<12} {:<12} {:<12}", "front_N", "insert%", "survive%", "attempts%");
    println!("{}", "-".repeat(50));

    for &cutoff in &[5, 10, 15, 20, 25, 30, 40] {
        let cutoff = cutoff.min(max_r);
        let cum_ins: u64 = global_successes[..cutoff].iter().sum();
        let cum_sur: u64 = global_survived[..cutoff].iter().sum();
        let cum_att: u64 = global_attempts[..cutoff].iter().sum();

        println!(
            "{:<12} {:<12.1} {:<12.1} {:<12.1}",
            format!("1-{}", cutoff),
            cum_ins as f64 / total_suc as f64 * 100.0,
            cum_sur as f64 / total_sur as f64 * 100.0,
            cum_att as f64 / total_att as f64 * 100.0,
        );
    }

    println!("\n=== 实验完成 ===");
}
