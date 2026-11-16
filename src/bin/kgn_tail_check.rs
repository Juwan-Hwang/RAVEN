//! KGN 式高 R 尾部命中率快检
//!
//! 目标：判断 KGN 的"R=128 建图 + 搜索时 prune"策略是否值得跟进
//!
//! 方法：在现有 R=40 图上，模拟搜索时 cap 到不同 D 值（10/15/20/25/30/40），
//! 按 query 难度（Easy/Medium/Hard）切片统计 recall 和 avg_visited 变化。
//!
//! 逻辑：
//! - 如果 cap=D 在 Hard query 上 recall 大幅下降（与 P10 一致），说明尾部边不可替代
//! - 此时 R=128 的前 D 个邻居必须显著优于 R=40 的前 D 个邻居才能补偿
//! - P10 已证明 R=64 的前 D 个邻居不比 R=40 更好
//! - 如果"Hard 组尾部贡献 8x"规律在 cap 模拟中也成立，直接排除方向 C
//!
//! 用法：cargo run --release --bin kgn_tail_check

use std::io::Write;
use std::time::Instant;

use raven::bench::util::SiftDataset;
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::linear_pool::LinearPool;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::VisitedTracker;
use raven::quant::SQ4Dataset;

const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;

/// 带度数上限的 SQ4 搜索（复用 greedy_search_sq4 逻辑但加 cap）
fn search_with_cap(
    sq4: &SQ4Dataset,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query_code: &[u8],
    ef: usize,
    visited: &mut VisitedTracker,
    pool: &mut LinearPool,
    po: usize,
    search_degree: usize, // 0 = 不限制
) -> Vec<(u32, f32)> {
    visited.reset();
    pool.reset_for(ef);

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];

    while let Some((node, _dist)) = pool.pop() {
        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            let effective_r = if search_degree > 0 {
                search_degree.min(storage.r_max())
            } else {
                storage.r_max()
            };
            let neighbor_bytes = effective_r * 4;
            let graph_lines = neighbor_bytes.div_ceil(64);
            unsafe {
                raven::arch::prefetch_l1(ptr);
                for l in 1..graph_lines {
                    raven::arch::prefetch_l1(ptr.add(l * 64));
                }
            }
        }

        let neighbors = storage.neighbors(node);
        let effective_len = if search_degree > 0 {
            search_degree.min(neighbors.len())
        } else {
            neighbors.len()
        };

        // Visited 预取
        let pf_vis = po.min(effective_len);
        for &v in neighbors.iter().take(pf_vis) {
            unsafe { visited.prefetch(v); }
        }

        let mut edge_size = 0usize;
        for (i, &v) in neighbors.iter().take(effective_len).enumerate() {
            if edge_size >= 128 { break; }
            if i + po < effective_len {
                unsafe { visited.prefetch(neighbors[i + po]); }
            }
            let is_new = unsafe { visited.visit_unchecked_branchless(v) };
            unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
            edge_size += is_new as usize;
        }

        let prefetch_count = po.min(edge_size);
        for &node in edge_buf.iter().take(prefetch_count) {
            let ptr = sq4.code(node as usize).as_ptr() as *const i8;
            unsafe { raven::arch::prefetch_l1(ptr); }
        }

        for i in 0..edge_size {
            if i + po < edge_size {
                let ptr = sq4.code(edge_buf[i + po] as usize).as_ptr() as *const i8;
                unsafe { raven::arch::prefetch_l1(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
            pool.insert(neighbor, d);
        }
    }

    pool.to_sorted_vec()
}

fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rr: usize,
) -> Vec<u32> {
    let n_rerank = (k * rr).min(candidates.len());
    let mut results: Vec<(u32, f32)> = candidates[..n_rerank]
        .iter()
        .map(|(id, _)| {
            let dist = l2_simd(query, &vectors[*id as usize * dim..(*id as usize + 1) * dim]);
            (*id, dist)
        })
        .collect();
    results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(k);
    results.iter().map(|(id, _)| *id).collect()
}

fn compute_recall_one(found: &[u32], gt_slice: &[i32]) -> f64 {
    let hits = gt_slice.iter().filter(|&&g| found.contains(&(g as u32))).count();
    hits as f64 / gt_slice.len() as f64
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

    println_both!("=== KGN 式高 R 尾部命中率快检 ===\n");
    println_both!("数据: n={}, dim={}, nq={}", data.n, dim, nq);

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
    let nav = graph.layered_nav().expect("need layered nav");
    let storage = graph.storage();

    let ef = 50;
    let mut visited = VisitedTracker::new(data.n, ef);
    let mut pool = LinearPool::new(ef);
    let code_bytes = dim.div_ceil(2);
    let mut query_code_buf = vec![0u8; code_bytes];

    // 先跑一次无 cap（D=0=不限制）获取每个 query 的 visited_count，用于难度分组
    println_both!("\n第一步：无 cap 搜索，收集每个 query 的 visited_count...");
    let mut query_visited: Vec<usize> = Vec::with_capacity(nq);
    let mut query_results: Vec<Vec<u32>> = Vec::with_capacity(nq);

    for q in 0..nq {
        let query = &data.test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, _) = nav.initialize(&data.train, dim, query);

        let candidates = search_with_cap(
            &sq4, storage, entry_point, &query_code_buf,
            ef, &mut visited, &mut pool, PO, 0, // D=0 = no cap
        );
        let vc = visited.visited_count();
        query_visited.push(vc);

        let result = rerank_f32(&data.train, dim, query, candidates, K, RR);
        query_results.push(result);
    }

    // 按 visited_count 排序，分 Easy (bottom 25%) / Medium (middle 50%) / Hard (top 25%)
    let mut sorted_indices: Vec<usize> = (0..nq).collect();
    sorted_indices.sort_by_key(|&i| query_visited[i]);

    let n_easy = nq / 4;
    let n_hard = nq / 4;
    let easy_idx = &sorted_indices[..n_easy];
    let medium_idx = &sorted_indices[n_easy..nq - n_hard];
    let hard_idx = &sorted_indices[nq - n_hard..];

    println_both!("  Easy: {} queries (visited ≤ {})", n_easy, query_visited[easy_idx[n_easy - 1]]);
    println_both!("  Medium: {} queries", medium_idx.len());
    println_both!("  Hard: {} queries (visited ≥ {})", n_hard, query_visited[hard_idx[0]]);

    // 计算无 cap 基线的 recall
    let calc_recall = |indices: &[usize]| -> f64 {
        let mut hits = 0u64;
        for &qi in indices {
            let gt_slice = &data.gt[qi * data.gt_k..qi * data.gt_k + K];
            for &g in gt_slice {
                if query_results[qi].contains(&(g as u32)) { hits += 1; }
            }
        }
        hits as f64 / (indices.len() * K) as f64
    };

    let calc_avg_visited = |indices: &[usize]| -> f64 {
        indices.iter().map(|&i| query_visited[i] as f64).sum::<f64>() / indices.len() as f64
    };

    let r_all = calc_recall(&sorted_indices[..]);
    let r_easy = calc_recall(easy_idx);
    let r_med = calc_recall(medium_idx);
    let r_hard = calc_recall(hard_idx);
    let v_all = calc_avg_visited(&sorted_indices[..]);
    let v_easy = calc_avg_visited(easy_idx);
    let v_med = calc_avg_visited(medium_idx);
    let v_hard = calc_avg_visited(hard_idx);

    println_both!("\n  无 cap 基线 (D=40):");
    println_both!("    All:    recall={:.4}  avg_visited={:.0}", r_all, v_all);
    println_both!("    Easy:   recall={:.4}  avg_visited={:.0}", r_easy, v_easy);
    println_both!("    Medium: recall={:.4}  avg_visited={:.0}", r_med, v_med);
    println_both!("    Hard:   recall={:.4}  avg_visited={:.0}", r_hard, v_hard);

    // ════════════════════════════════════════════════════════════════
    //  对不同 cap 值 D 重复搜索，统计 recall 和 visited 变化
    // ════════════════════════════════════════════════════════════════
    println_both!("\n第二步：不同 cap 值 D 的效果对比\n");
    println_both!("  {:>6}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "D", "group", "recall", "Δrecall", "avg_vis", "Δvis%", "Easy_r", "Med_r", "Hard_r");
    println_both!("  {}", "-".repeat(100));

    let caps: [usize; 7] = [5, 10, 15, 20, 25, 30, 40];
    let mut results_by_cap: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();

    for &d in &caps {
        // 对每个 cap 值重新搜索
        let mut cap_results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut cap_visited: Vec<usize> = Vec::with_capacity(nq);

        for q in 0..nq {
            let query = &data.test[q * dim..(q + 1) * dim];
            sq4.params.encode_into(query, &mut query_code_buf);
            let (entry_point, _) = nav.initialize(&data.train, dim, query);

            let candidates = search_with_cap(
                &sq4, storage, entry_point, &query_code_buf,
                ef, &mut visited, &mut pool, PO, d,
            );
            cap_visited.push(visited.visited_count());
            cap_results.push(rerank_f32(&data.train, dim, query, candidates, K, RR));
        }

        let cr_all = calc_recall_from(&cap_results, &sorted_indices[..], &data.gt, data.gt_k, K);
        let cr_easy = calc_recall_from(&cap_results, easy_idx, &data.gt, data.gt_k, K);
        let cr_med = calc_recall_from(&cap_results, medium_idx, &data.gt, data.gt_k, K);
        let cr_hard = calc_recall_from(&cap_results, hard_idx, &data.gt, data.gt_k, K);
        let cv_all: f64 = sorted_indices[..].iter()
            .map(|&i| cap_visited[i] as f64).sum::<f64>() / nq as f64;

        let dr = cr_all - r_all;
        let dv_pct = (cv_all - v_all) / v_all * 100.0;

        println_both!("  {:>6}  {:>8}  {:>10.4}  {:>+10.4}  {:>10.0}  {:>+9.1}%  {:>10.4}  {:>10.4}  {:>10.4}",
            d, "All", cr_all, dr, cv_all, dv_pct, cr_easy, cr_med, cr_hard);

        results_by_cap.push((d, cr_all, cr_easy, cr_med, cr_hard, cv_all));
    }

    // ════════════════════════════════════════════════════════════════
    //  结论分析
    // ════════════════════════════════════════════════════════════════
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  结论分析");
    println_both!("═══════════════════════════════════════════════════════════════\n");

    // 检查 Hard query 在 cap=D 时的 recall 损失
    let baseline_hard = r_hard;
    for &(d, _, _, _, cr_hard, _) in &results_by_cap {
        if d < 40 {
            let loss = baseline_hard - cr_hard;
            if loss > 0.01 {
                println_both!("  D={:<3} Hard recall 损失: {:.4} ({:.2}pp) — 尾部边对 Hard query 不可替代", d, loss, loss * 100.0);
            }
        }
    }

    // 检查"Hard 组尾部贡献 8x"规律是否成立
    // 通过 D=30 vs D=40 的 recall 差异来验证（rank 31-40 的贡献）
    let d30 = results_by_cap.iter().find(|&&(d, _, _, _, _, _)| d == 30);
    let d40 = results_by_cap.iter().find(|&&(d, _, _, _, _, _)| d == 40);
    if let (Some(&(30, _, _, _, cr_hard_30, _)), Some(&(40, _, _, _, cr_hard_40, _))) = (d30, d40) {
        let hard_tail_contrib = cr_hard_40 - cr_hard_30;
        let d30_easy = results_by_cap.iter().find(|&&(d, _, _, _, _, _)| d == 30).map(|&(_, _, cr_easy, _, _, _)| cr_easy).unwrap();
        let d40_easy = results_by_cap.iter().find(|&&(d, _, _, _, _, _)| d == 40).map(|&(_, _, cr_easy, _, _, _)| cr_easy).unwrap();
        let easy_tail_contrib = d40_easy - d30_easy;

        println_both!("\n  rank 31-40 贡献（D=30→D=40 的 recall 增量）:");
        println_both!("    Easy:   +{:.4}pp", easy_tail_contrib * 100.0);
        println_both!("    Hard:   +{:.4}pp", hard_tail_contrib * 100.0);
        if easy_tail_contrib > 0.0001 {
            println_both!("    比值:   {:.1}x", hard_tail_contrib / easy_tail_contrib);
        }
    }

    println_both!("\n  如果 Hard 组的 recall 损失随 cap D 递减而加速增长，");
    println_both!("  且 rank 31-40 的 Hard/Easy 贡献比 ≥ 4x，");
    println_both!("  则 KGN 的 R=128+prune 策略需要其前 D 个邻居质量显著优于 R=40，");
    println_both!("  否则 cap 会伤害 Hard query。P10 已证明 R=64 的前 D 个邻居不更优，");
    println_both!("  因此 R=128 大概率也不更优（DirectionalPrune 邻居排序与 R 无关）。");

    let mut f = std::fs::File::create("experiments/kgn_tail_check_result.txt").expect("create");
    f.write_all(out.as_bytes()).expect("write");
    println!("\n结果已写入 experiments/kgn_tail_check_result.txt");
}

fn calc_recall_from(
    results: &[Vec<u32>],
    indices: &[usize],
    gt: &[i32],
    gt_k: usize,
    k: usize,
) -> f64 {
    let mut hits = 0u64;
    for &qi in indices {
        let gt_slice = &gt[qi * gt_k..qi * gt_k + k];
        for &g in gt_slice {
            if results[qi].contains(&(g as u32)) { hits += 1; }
        }
    }
    hits as f64 / (indices.len() * k) as f64
}
