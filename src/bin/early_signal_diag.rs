//! 早期信号诊断：哪些运行时信号最能预测 query 难度
//!
//! 在 ef=50 搜索过程中，在 pop 5/10/15/20/30 检查点记录：
//! - kth_dist: pool 中第 k 近的距离 (k=10)
//! - worst_dist: pool 中最差的距离
//! - pool_size: 当前 pool 大小
//! - insert_rate: 累计插入成功率
//! - kth_delta: kth_dist 自上一个检查点的变化量
//!
//! 搜索结束后记录 visited_count（难度标签），
//! 计算每个信号与 visited_count 的 Pearson 相关系数。
//!
//! 同时记录 nav_entry_dist（当前 adaptive ef 用的信号）作为对比基线。
//!
//! 用法：cargo run --release --bin early_signal_diag

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

fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rr: usize,
) -> Vec<(u32, f32)> {
    let n = (k * rr).min(candidates.len());
    let mut r: Vec<(u32, f32)> = candidates[..n]
        .iter()
        .map(|(id, _)| {
            (*id, l2_simd(query, &vectors[*id as usize * dim..(*id as usize + 1) * dim]))
        })
        .collect();
    r.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    r.truncate(k);
    r
}

/// 检查点
const CHECKPOINTS: [usize; 5] = [5, 10, 15, 20, 30];

/// 每个 query 的检查点信号
#[derive(Default, Clone)]
struct CheckpointSignals {
    /// k-th 距离 (k=10)
    kth_dist: Vec<f32>,
    /// worst (pool 最差) 距离
    worst_dist: Vec<f32>,
    /// pool 当前大小
    pool_size: Vec<usize>,
    /// 累计插入成功率
    insert_rate: Vec<f32>,
    /// k-th 距离自上一个检查点的变化量（负 = 改善）
    kth_delta: Vec<f32>,
}

struct QueryRecord {
    /// nav_entry_dist（当前 adaptive ef 信号）
    nav_entry_dist: f32,
    /// 检查点信号
    signals: CheckpointSignals,
    /// 最终 visited_count（难度标签）
    visited_count: usize,
}

/// 带检查点记录的 SQ4 搜索
fn search_with_checkpoints(
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
) -> (Vec<(u32, f32)>, CheckpointSignals, usize) {
    visited.reset();
    pool.reset_for(ef);

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];

    let mut signals = CheckpointSignals {
        kth_dist: vec![f32::MAX; CHECKPOINTS.len()],
        worst_dist: vec![f32::MAX; CHECKPOINTS.len()],
        pool_size: vec![0; CHECKPOINTS.len()],
        insert_rate: vec![0.0; CHECKPOINTS.len()],
        kth_delta: vec![0.0; CHECKPOINTS.len()],
    };

    let mut pop_order: usize = 0;
    let mut total_attempts: usize = 0;
    let mut total_inserts: usize = 0;
    let mut prev_kth: f32 = f32::MAX;

    let mut cp_idx: usize = 0;

    while let Some((node, _dist)) = pool.pop() {
        pop_order += 1;

        // 检查点记录
        while cp_idx < CHECKPOINTS.len() && pop_order == CHECKPOINTS[cp_idx] {
            let cur_kth = pool.kth_distance(k);
            let cur_worst = pool.worst_distance();
            let cur_size = pool.len();
            let ins_rate = if total_attempts > 0 {
                total_inserts as f32 / total_attempts as f32
            } else {
                0.0
            };

            signals.kth_dist[cp_idx] = cur_kth;
            signals.worst_dist[cp_idx] = cur_worst;
            signals.pool_size[cp_idx] = cur_size;
            signals.insert_rate[cp_idx] = ins_rate;
            signals.kth_delta[cp_idx] = cur_kth - prev_kth;

            prev_kth = cur_kth;
            cp_idx += 1;
        }

        // Multi-line prefetch
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

        // 收集未访问邻居
        let mut edge_size = 0usize;
        for &v in neighbors.iter() {
            if edge_size >= 128 {
                break;
            }
            let is_new = unsafe { visited.visit_unchecked_branchless(v) };
            if is_new {
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += 1;
            }
        }

        // 预取
        let pc = po.min(edge_size);
        for i in 0..pc {
            let v = edge_buf[i] as usize;
            let ptr = sq4.code(v).as_ptr() as *const i8;
            unsafe { raven::arch::prefetch_l1(ptr); }
        }

        // 距离计算 + insert
        for i in 0..edge_size {
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = sq4.code(v).as_ptr() as *const i8;
                unsafe { raven::arch::prefetch_l1(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
            total_attempts += 1;
            if pool.insert(neighbor, d) {
                total_inserts += 1;
            }
        }
    }

    let vc = visited.visited_count();
    let candidates = pool.to_sorted_vec();
    let results = rerank_f32(vectors, dim, query, candidates, k, rr);

    (results, signals, vc)
}

/// Pearson 相关系数
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-20 || var_y < 1e-20 {
        return 0.0;
    }
    cov / (var_x * var_y).sqrt()
}

fn main() {
    println!("=== 早期信号诊断 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = 1000.min(nq_full);
    let k = 10;
    let ef = 50;
    let po = 16;
    let rr = 5;

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
    let sq4 = SQ4Dataset::build(&train, dim);

    let nav = graph.layered_nav().expect("need layered nav");
    let mut visited = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);
    let code_bytes = dim.div_ceil(2);
    let mut query_code_buf = vec![0u8; code_bytes];

    // 搜索 + 记录
    let mut records: Vec<QueryRecord> = Vec::with_capacity(nq);

    println!("--- 搜索 (ef={}, k={}, nq={}) ---\n", ef, k, nq);
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, nav_dist) = nav.initialize(&train, dim, query);

        let (_results, signals, vc) = search_with_checkpoints(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            ef, &mut visited, &mut pool, po, k, rr,
            &train, dim, query,
        );

        records.push(QueryRecord {
            nav_entry_dist: nav_dist,
            signals,
            visited_count: vc,
        });
    }
    println!("搜索时间: {:.1}s\n", t0.elapsed().as_secs_f64());

    // 提取 visited_count 数组
    let vc_arr: Vec<f64> = records.iter().map(|r| r.visited_count as f64).collect();

    // ── 基线：nav_entry_dist ──
    let nav_dist_arr: Vec<f64> = records.iter().map(|r| r.nav_entry_dist as f64).collect();
    let nav_corr = pearson(&nav_dist_arr, &vc_arr);

    println!("=== Pearson 相关系数（|r| 越大预测力越强）===\n");
    println!("基线信号（当前 adaptive ef 用）:");
    println!("  nav_entry_dist:     {:+.4}", nav_corr);

    // ── 各检查点信号 ──
    println!("\n运行时信号（按检查点）:\n");
    println!(
        "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "signal@pop", "5", "10", "15", "20", "30"
    );
    println!("{}", "-".repeat(70));

    let signal_names = ["kth_dist", "worst_dist", "pool_size", "insert_rate", "kth_delta"];

    for &sig_name in &signal_names {
        print!("{:<20}", sig_name);
        for &cp in &CHECKPOINTS {
            let cp_idx = CHECKPOINTS.iter().position(|&c| c == cp).unwrap();
            let arr: Vec<f64> = records
                .iter()
                .map(|r| {
                    let v = match sig_name {
                        "kth_dist" => r.signals.kth_dist[cp_idx],
                        "worst_dist" => r.signals.worst_dist[cp_idx],
                        "pool_size" => r.signals.pool_size[cp_idx] as f32,
                        "insert_rate" => r.signals.insert_rate[cp_idx],
                        "kth_delta" => r.signals.kth_delta[cp_idx],
                        _ => 0.0,
                    };
                    // 过滤 MAX 值
                    if v.is_finite() {
                        v as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect();

            // 过滤 NaN
            let pairs: Vec<(f64, f64)> = arr
                .iter()
                .zip(vc_arr.iter())
                .filter(|(x, _)| x.is_finite())
                .map(|(x, y)| (*x, *y))
                .collect();

            if pairs.len() < 10 {
                print!(" {:<10}", "N/A");
                continue;
            }

            let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
            let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
            let corr = pearson(&xs, &ys);
            print!(" {:+.4}    ", corr);
        }
        println!();
    }

    // ── 最佳信号排名 ──
    println!("\n=== 信号排名（按 |r| 降序）===\n");

    let mut all_signals: Vec<(String, f64)> = Vec::new();
    all_signals.push(("nav_entry_dist".to_string(), nav_corr));

    for &sig_name in &signal_names {
        for (cp_idx, &cp) in CHECKPOINTS.iter().enumerate() {
            let arr: Vec<f64> = records
                .iter()
                .map(|r| {
                    let v = match sig_name {
                        "kth_dist" => r.signals.kth_dist[cp_idx],
                        "worst_dist" => r.signals.worst_dist[cp_idx],
                        "pool_size" => r.signals.pool_size[cp_idx] as f32,
                        "insert_rate" => r.signals.insert_rate[cp_idx],
                        "kth_delta" => r.signals.kth_delta[cp_idx],
                        _ => 0.0,
                    };
                    if v.is_finite() { v as f64 } else { f64::NAN }
                })
                .collect();

            let pairs: Vec<(f64, f64)> = arr.iter().zip(vc_arr.iter())
                .filter(|(x, _)| x.is_finite())
                .map(|(x, y)| (*x, *y))
                .collect();

            if pairs.len() >= 10 {
                let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
                let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
                let corr = pearson(&xs, &ys);
                all_signals.push((format!("{}@{}", sig_name, cp), corr));
            }
        }
    }

    all_signals.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    for (i, (name, corr)) in all_signals.iter().enumerate() {
        let marker = if corr.abs() > nav_corr.abs() { " ★" } else { "" };
        println!("  {:<3} {:<25} {:+.4}{}", i + 1, name, corr, marker);
    }

    // ── 难度分组信号对比 ──
    println!("\n=== 难度分组信号均值 ===\n");

    let mut sorted_indices: Vec<usize> = (0..nq).collect();
    sorted_indices.sort_by_key(|&i| records[i].visited_count);

    let q25 = nq / 4;
    let q75 = nq * 3 / 4;
    let easy = &sorted_indices[..q25];
    let hard = &sorted_indices[q75..];

    println!(
        "{:<25} {:<15} {:<15} {:<10}",
        "signal", "Easy(avg)", "Hard(avg)", "ratio H/E"
    );
    println!("{}", "-".repeat(65));

    // nav_entry_dist
    let easy_nav: f64 = easy.iter().map(|&i| records[i].nav_entry_dist as f64).sum::<f64>() / easy.len() as f64;
    let hard_nav: f64 = hard.iter().map(|&i| records[i].nav_entry_dist as f64).sum::<f64>() / hard.len() as f64;
    println!("{:<25} {:<15.1} {:<15.1} {:<10.2}", "nav_entry_dist", easy_nav, hard_nav, hard_nav / easy_nav.max(0.01));

    for &sig_name in &signal_names {
        for (cp_idx, &cp) in CHECKPOINTS.iter().enumerate() {
            let easy_vals: Vec<f64> = easy.iter().filter_map(|&i| {
                let v = match sig_name {
                    "kth_dist" => records[i].signals.kth_dist[cp_idx],
                    "worst_dist" => records[i].signals.worst_dist[cp_idx],
                    "pool_size" => records[i].signals.pool_size[cp_idx] as f32,
                    "insert_rate" => records[i].signals.insert_rate[cp_idx],
                    "kth_delta" => records[i].signals.kth_delta[cp_idx],
                    _ => 0.0,
                };
                if v.is_finite() { Some(v as f64) } else { None }
            }).collect();
            let hard_vals: Vec<f64> = hard.iter().filter_map(|&i| {
                let v = match sig_name {
                    "kth_dist" => records[i].signals.kth_dist[cp_idx],
                    "worst_dist" => records[i].signals.worst_dist[cp_idx],
                    "pool_size" => records[i].signals.pool_size[cp_idx] as f32,
                    "insert_rate" => records[i].signals.insert_rate[cp_idx],
                    "kth_delta" => records[i].signals.kth_delta[cp_idx],
                    _ => 0.0,
                };
                if v.is_finite() { Some(v as f64) } else { None }
            }).collect();

            if easy_vals.is_empty() || hard_vals.is_empty() {
                continue;
            }

            let em = easy_vals.iter().sum::<f64>() / easy_vals.len() as f64;
            let hm = hard_vals.iter().sum::<f64>() / hard_vals.len() as f64;
            let ratio = if em.abs() > 0.001 { hm / em } else { f64::NAN };

            println!(
                "{:<25} {:<15.4} {:<15.4} {:<10.2}",
                format!("{}@{}", sig_name, cp), em, hm, ratio,
            );
        }
    }

    println!("\n=== 诊断完成 ===");
}
