//! 邻居命中率 — 按 query 难度切片
//!
//! 在搜索追踪的基础上，将 query 按 visited_count 分为 easy/hard 组，
//! 分别统计 per-rank 存活贡献占比。
//!
//! 如果 hard 组的尾部边存活贡献显著高于 easy 组 → cap 会伤害难 query
//! 如果两组差异不大 → cap 安全，值得投入完整扫描
//!
//! 用法：cargo run --release --bin hit_rate_by_difficulty

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

struct QueryStats {
    /// per-rank: (attempts, successes)
    rank_stats: Vec<(u64, u64)>,
    /// per-rank: 插入成功的节点列表 + pop 序号
    rank_inserted: Vec<Vec<(u32, usize)>>,
    /// 搜索结束时 pool 中的节点集合
    final_pool: HashSet<u32>,
    /// 总访问数
    visited_count: usize,
    /// rerank 后的 top-k 结果
    result_ids: Vec<u32>,
}

fn search_tracked(
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
) -> QueryStats {
    visited.reset();
    pool.reset_for(ef);

    let mut rank_stats: Vec<(u64, u64)> = vec![(0, 0); max_r];
    let mut rank_inserted: Vec<Vec<(u32, usize)>> = vec![Vec::new(); max_r];

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];
    let mut edge_ranks: [usize; 128] = [0; 128];
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

        let prefetch_count = po.min(edge_size);
        for i in 0..prefetch_count {
            let v = edge_buf[i] as usize;
            let ptr = sq4.code(v).as_ptr() as *const i8;
            unsafe { raven::arch::prefetch_l1(ptr); }
        }

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
    let candidates = pool.to_sorted_vec();
    let final_pool: HashSet<u32> = candidates.iter().map(|(id, _)| *id).collect();
    let results = rerank_f32(vectors, dim, query, candidates, k, rr);

    QueryStats {
        rank_stats,
        rank_inserted,
        final_pool,
        visited_count: vc,
        result_ids: results.iter().map(|(id, _)| *id).collect(),
    }
}

/// 聚合一组 query 的 per-rank 统计
struct GroupStats {
    attempts: Vec<u64>,
    successes: Vec<u64>,
    survived: Vec<u64>,
    total_visited: usize,
    count: usize,
    recall: f64,
}

fn aggregate_group(
    queries: &[usize],
    all_stats: &[QueryStats],
    all_results: &[Vec<u32>],
    gt: &[i32],
    gt_k: usize,
    k: usize,
    max_r: usize,
) -> GroupStats {
    let mut attempts = vec![0u64; max_r];
    let mut successes = vec![0u64; max_r];
    let mut survived = vec![0u64; max_r];
    let mut total_visited = 0usize;
    let mut hits = 0u64;

    for &q in queries {
        let qs = &all_stats[q];
        total_visited += qs.visited_count;

        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &id in &all_results[q] {
            if gt_slice.contains(&(id as i32)) {
                hits += 1;
            }
        }

        for r in 0..max_r {
            attempts[r] += qs.rank_stats[r].0;
            successes[r] += qs.rank_stats[r].1;
            for &(node, _pop) in &qs.rank_inserted[r] {
                if qs.final_pool.contains(&node) {
                    survived[r] += 1;
                }
            }
        }
    }

    let nq = queries.len();
    GroupStats {
        attempts,
        successes,
        survived,
        total_visited,
        count: nq,
        recall: hits as f64 / (nq * k) as f64,
    }
}

fn print_group_report(
    label: &str,
    gs: &GroupStats,
    max_r: usize,
) {
    let total_sur: u64 = gs.survived.iter().sum();
    let total_ins: u64 = gs.successes.iter().sum();
    let total_att: u64 = gs.attempts.iter().sum();

    println!("\n--- {} (n={}, recall={:.4}, avg_visited={:.0}) ---", label, gs.count, gs.recall, gs.total_visited as f64 / gs.count as f64);
    println!("总尝试={} 总插入={} ({:.1}%) 总存活={} ({:.1}% of inserted)", total_att, total_ins, total_ins as f64 / total_att as f64 * 100.0, total_sur, total_sur as f64 / total_ins.max(1) as f64 * 100.0);

    // 按 rank 区间
    let bins: [(usize, usize); 6] = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40)];
    println!("\n{:<12} {:<10} {:<10} {:<10} {:<12} {:<12}", "rank_range", "insert%", "survive%", "insert_share", "survive_share", "att_share");
    println!("{}", "-".repeat(75));

    for (lo, hi) in &bins {
        let mut att = 0u64;
        let mut suc = 0u64;
        let mut sur = 0u64;
        for r in *lo..*hi.min(&max_r) {
            att += gs.attempts[r];
            suc += gs.successes[r];
            sur += gs.survived[r];
        }
        let ins_rate = if att > 0 { suc as f64 / att as f64 * 100.0 } else { 0.0 };
        let sur_rate = if suc > 0 { sur as f64 / suc as f64 * 100.0 } else { 0.0 };
        let ins_share = if total_ins > 0 { suc as f64 / total_ins as f64 * 100.0 } else { 0.0 };
        let sur_share = if total_sur > 0 { sur as f64 / total_sur as f64 * 100.0 } else { 0.0 };
        let att_share = if total_att > 0 { att as f64 / total_att as f64 * 100.0 } else { 0.0 };

        println!(
            "{:<12} {:<10.1} {:<10.1} {:<12.1} {:<12.1} {:<12.1}",
            format!("{}-{}", lo + 1, hi), ins_rate, sur_rate, ins_share, sur_share, att_share,
        );
    }

    // 累积
    println!("\n{:<12} {:<12} {:<12}", "front_N", "survive_share", "att_share");
    println!("{}", "-".repeat(40));
    for &cutoff in &[5, 10, 15, 20, 25, 30, 40] {
        let cutoff = cutoff.min(max_r);
        let cum_sur: u64 = gs.survived[..cutoff].iter().sum();
        let cum_att: u64 = gs.attempts[..cutoff].iter().sum();
        println!(
            "{:<12} {:<12.1} {:<12.1}",
            format!("1-{}", cutoff),
            cum_sur as f64 / total_sur.max(1) as f64 * 100.0,
            cum_att as f64 / total_att.max(1) as f64 * 100.0,
        );
    }
}

fn main() {
    println!("=== 邻居命中率 — 按 query 难度切片 ===\n");

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

    // 搜索全部 query，记录 per-query 统计
    let mut all_stats: Vec<QueryStats> = Vec::with_capacity(nq);
    let mut all_results: Vec<Vec<u32>> = Vec::with_capacity(nq);

    println!("--- 搜索 (ef={}, k={}, nq={}) ---\n", ef, k, nq);
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, _) = nav.initialize(&train, dim, query);

        let qs = search_tracked(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            ef, &mut visited, &mut pool, po, k, rr,
            &train, dim, query, max_r,
        );

        all_results.push(qs.result_ids.clone());
        all_stats.push(qs);
    }
    let dt = t0.elapsed();

    let total_recall = compute_recall(&all_results, &gt, gt_k, k, nq);
    let total_visited: usize = all_stats.iter().map(|s| s.visited_count).sum();
    println!("总 recall@{}: {:.4}", k, total_recall);
    println!("avg_visited: {:.1}", total_visited as f64 / nq as f64);
    println!("搜索时间: {:.1}s\n", dt.as_secs_f64());

    // 按 visited_count 排序 query
    let mut query_indices: Vec<usize> = (0..nq).collect();
    query_indices.sort_by_key(|&q| all_stats[q].visited_count);

    // 分组：easy (bottom 25%), medium (middle 50%), hard (top 25%)
    let q25 = nq / 4;
    let q75 = nq * 3 / 4;

    let easy_queries: Vec<usize> = query_indices[..q25].to_vec();
    let medium_queries: Vec<usize> = query_indices[q25..q75].to_vec();
    let hard_queries: Vec<usize> = query_indices[q75..].to_vec();

    // 打印分组信息
    let easy_vc: Vec<usize> = easy_queries.iter().map(|&q| all_stats[q].visited_count).collect();
    let hard_vc: Vec<usize> = hard_queries.iter().map(|&q| all_stats[q].visited_count).collect();

    println!("=== 难度分组 ===");
    println!("Easy   (n={}): visited [{}, {}], avg={:.0}", easy_queries.len(), easy_vc.first().unwrap(), easy_vc.last().unwrap(), easy_vc.iter().sum::<usize>() as f64 / easy_vc.len() as f64);
    println!("Medium (n={}): visited [{}, {}]", medium_queries.len(), all_stats[medium_queries[0]].visited_count, all_stats[medium_queries[medium_queries.len()-1]].visited_count);
    println!("Hard   (n={}): visited [{}, {}], avg={:.0}", hard_queries.len(), hard_vc.first().unwrap(), hard_vc.last().unwrap(), hard_vc.iter().sum::<usize>() as f64 / hard_vc.len() as f64);

    // 聚合各组统计
    let easy_stats = aggregate_group(&easy_queries, &all_stats, &all_results, &gt, gt_k, k, max_r);
    let medium_stats = aggregate_group(&medium_queries, &all_stats, &all_results, &gt, gt_k, k, max_r);
    let hard_stats = aggregate_group(&hard_queries, &all_stats, &all_results, &gt, gt_k, k, max_r);

    // 打印各组报告
    print_group_report("All Queries", &aggregate_group(&(0..nq).collect::<Vec<_>>(), &all_stats, &all_results, &gt, gt_k, k, max_r), max_r);
    print_group_report("Easy (bottom 25% by visited)", &easy_stats, max_r);
    print_group_report("Medium (middle 50%)", &medium_stats, max_r);
    print_group_report("Hard (top 25% by visited)", &hard_stats, max_r);

    // ── 核心对比表：rank 31-40 的存活贡献占比 ──
    println!("\n=== 核心对比：rank 31-40 的存活贡献 ===\n");

    let groups = [
        ("Easy", &easy_stats),
        ("Medium", &medium_stats),
        ("Hard", &hard_stats),
    ];

    println!("{:<10} {:<12} {:<12} {:<12} {:<12}", "group", "surv_31-40", "surv_total", "surv_share", "att_share");
    println!("{}", "-".repeat(60));

    for (label, gs) in &groups {
        let sur_tail: u64 = gs.survived[30..40].iter().sum();
        let sur_total: u64 = gs.survived.iter().sum();
        let att_tail: u64 = gs.attempts[30..40].iter().sum();
        let att_total: u64 = gs.attempts.iter().sum();

        println!(
            "{:<10} {:<12} {:<12} {:<12.1} {:<12.1}",
            label, sur_tail, sur_total,
            sur_tail as f64 / sur_total.max(1) as f64 * 100.0,
            att_tail as f64 / att_total.max(1) as f64 * 100.0,
        );
    }

    // 更细致：rank 21-30 和 31-40 分别对比
    println!("\n=== 细分对比：rank 21-30 vs 31-40 存活贡献 ===\n");
    println!("{:<10} {:<15} {:<15} {:<15}", "group", "surv_21-30 %", "surv_31-40 %", "ratio H/E");
    println!("{}", "-".repeat(55));

    let easy_sur_31_40 = easy_stats.survived[30..40].iter().sum::<u64>() as f64 / easy_stats.survived.iter().sum::<u64>().max(1) as f64 * 100.0;
    let med_sur_31_40 = medium_stats.survived[30..40].iter().sum::<u64>() as f64 / medium_stats.survived.iter().sum::<u64>().max(1) as f64 * 100.0;
    let hard_sur_31_40 = hard_stats.survived[30..40].iter().sum::<u64>() as f64 / hard_stats.survived.iter().sum::<u64>().max(1) as f64 * 100.0;

    let easy_sur_21_30 = easy_stats.survived[20..30].iter().sum::<u64>() as f64 / easy_stats.survived.iter().sum::<u64>().max(1) as f64 * 100.0;
    let med_sur_21_30 = medium_stats.survived[20..30].iter().sum::<u64>() as f64 / medium_stats.survived.iter().sum::<u64>().max(1) as f64 * 100.0;
    let hard_sur_21_30 = hard_stats.survived[20..30].iter().sum::<u64>() as f64 / hard_stats.survived.iter().sum::<u64>().max(1) as f64 * 100.0;

    println!("{:<10} {:<15.1} {:<15.1} {:<15.2}", "Easy", easy_sur_21_30, easy_sur_31_40, 1.0);
    println!("{:<10} {:<15.1} {:<15.1} {:<15.2}", "Medium", med_sur_21_30, med_sur_31_40, med_sur_31_40 / easy_sur_31_40.max(0.01));
    println!("{:<10} {:<15.1} {:<15.1} {:<15.2}", "Hard", hard_sur_21_30, hard_sur_31_40, hard_sur_31_40 / easy_sur_31_40.max(0.01));

    // ── 结论判断 ──
    println!("\n=== 结论 ===\n");

    let ratio = hard_sur_31_40 / easy_sur_31_40.max(0.01);
    if ratio > 2.0 {
        println!("⚠ Hard 组 rank 31-40 存活贡献是 Easy 组的 {:.1}x", ratio);
        println!("  → cap 会精准伤害难 query，大概率重蹈 P10 覆辙");
        println!("  → 不建议投入 Search-Degree Cap 实验");
    } else if ratio > 1.5 {
        println!("⚠ Hard 组 rank 31-40 存活贡献是 Easy 组的 {:.1}x", ratio);
        println!("  → 存在差异但不是数量级差异，cap 可能有边际价值");
        println!("  → 建议用保守 cap (D=30) + 提高 ef 补偿做小范围验证");
    } else {
        println!("✓ Hard 组 rank 31-40 存活贡献是 Easy 组的 {:.1}x", ratio);
        println!("  → 尾部边在难 query 里同样没什么用");
        println!("  → Search-Degree Cap 安全，值得投入完整 D×ef 扫描");
    }

    println!("\n=== 实验完成 ===");
}
