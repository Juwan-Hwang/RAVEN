//! 组合信号分类器 —— 特征收集
//!
//! 收集每个 query 的多维运行时信号（pop=5/10/15 检查点），
//! 保存到 CSV 供 Python 逻辑回归/随机森林分析。
//!
//! 特征（10 维）：
//!   nav_entry_dist, insert_rate@5, kth_dist@5, worst_dist@5,
//!   insert_rate@10, kth_dist@10, worst_dist@10,
//!   insert_rate@15, kth_dist@15, worst_dist@15
//!
//! 标签：recall_35, recall_65, recall_100, visited_35, visited_65
//!
//! 用法：cargo run --release --bin combo_signal_bench

use std::fs::File;
use std::io::{Read, Write, BufWriter};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::linear_pool::LinearPool;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::VisitedTracker;
use raven::memory::serialize::Serializable;
use raven::quant::SQ4Dataset;

// ── IO helpers ──────────────────────────────────────────────────────────

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

fn recall_at_k(results: &[(u32, f32)], gt: &[i32], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<i32> = gt[..k].iter().copied().collect();
    let hits = results.iter().filter(|(id, _)| gt_set.contains(&(*id as i32))).count();
    hits as f32 / k as f32
}

// ── Checkpoint signals ──────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CpSignals {
    insert_rate: f32,
    kth_dist: f32,
    worst_dist: f32,
}

impl Default for CpSignals {
    fn default() -> Self {
        Self { insert_rate: 0.0, kth_dist: f32::MAX, worst_dist: f32::MAX }
    }
}

const CHECKPOINTS: [usize; 3] = [5, 10, 15];

/// 带 3 个检查点信号追踪的 SQ4 搜索
fn search_with_signals(
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
) -> (Vec<(u32, f32)>, usize, [CpSignals; 3]) {
    visited.reset();
    pool.reset_for(ef);

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];
    let mut pop_order: usize = 0;
    let mut total_attempts: usize = 0;
    let mut total_inserts: usize = 0;
    let mut cp_idx: usize = 0;
    let mut signals = [CpSignals::default(); 3];

    while let Some((node, _dist)) = pool.pop() {
        pop_order += 1;

        // 检查点记录
        while cp_idx < CHECKPOINTS.len() && pop_order == CHECKPOINTS[cp_idx] {
            let ins_rate = if total_attempts > 0 {
                total_inserts as f32 / total_attempts as f32
            } else {
                0.0
            };
            signals[cp_idx] = CpSignals {
                insert_rate: ins_rate,
                kth_dist: pool.kth_distance(k),
                worst_dist: pool.worst_distance(),
            };
            cp_idx += 1;
        }

        // Prefetch
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
        for &v in neighbors.iter() {
            if edge_size >= 128 { break; }
            let is_new = unsafe { visited.visit_unchecked_branchless(v) };
            if is_new {
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += 1;
            }
        }

        let pc = po.min(edge_size);
        for i in 0..pc {
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
            let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
            total_attempts += 1;
            if pool.insert(neighbor, d) {
                total_inserts += 1;
            }
        }
    }

    // 填充未到达的检查点（用最终值）
    while cp_idx < CHECKPOINTS.len() {
        let ins_rate = if total_attempts > 0 {
            total_inserts as f32 / total_attempts as f32
        } else {
            0.0
        };
        signals[cp_idx] = CpSignals {
            insert_rate: ins_rate,
            kth_dist: pool.kth_distance(k),
            worst_dist: pool.worst_distance(),
        };
        cp_idx += 1;
    }

    let vc = visited.visited_count();
    let candidates = pool.to_sorted_vec();
    let results = rerank_f32(vectors, dim, query, candidates, k, rr);

    (results, vc, signals)
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    println!("=== 组合信号特征收集 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = 1000.min(nq_full);
    let k = 10;
    let po = 16;
    let rr = 5;

    // 加载缓存图
    let graph_path = std::path::Path::new("data/sift/graph_cache_dir_rmin4.bin");
    let graph = if graph_path.exists() {
        println!("loading cached DirectionalPrune graph...");
        let t0 = Instant::now();
        let g = VamanaGraph::load(graph_path).expect("load graph cache");
        println!("load: {:.1}s", t0.elapsed().as_secs_f64());
        g
    } else {
        println!("building graph (first run)...");
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 384, r_max: 40, r_soft: 60,
            max_iterations: 2, saturate: false,
            enable_layered_nav: true, nav_m: 32,
            prune_strategy: PruneStrategy::DirectionalPrune,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from(42);
        let t0 = Instant::now();
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        println!("build: {:.1}s", t0.elapsed().as_secs_f64());
        let _ = g.save(graph_path);
        g
    };

    println!("building SQ4...");
    let sq4 = SQ4Dataset::build(&train, dim);

    let nav = graph.layered_nav().expect("need layered nav");
    let mut visited = VisitedTracker::new(n, 100);
    let mut pool = LinearPool::new(100);
    let code_bytes = dim.div_ceil(2);
    let mut query_code_buf = vec![0u8; code_bytes];

    // 输出 CSV
    let csv_path = "experiments/combo_features.csv";
    let mut csv = BufWriter::new(File::create(csv_path).expect("create csv"));

    // 表头
    writeln!(csv, "nav_entry_dist,insert_rate_5,kth_dist_5,worst_dist_5,insert_rate_10,kth_dist_10,worst_dist_10,insert_rate_15,kth_dist_15,worst_dist_15,recall_35,recall_65,recall_100,visited_35,visited_65").unwrap();

    println!("searching (ef=35/65/100, nq={})...", nq);
    let t0 = Instant::now();

    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, nav_dist) = nav.initialize(&train, dim, query);
        let gt_slice = &gt[q * gt_k..(q + 1) * gt_k];

        // ef=35 with signals
        let (r35, vis35, sig) = search_with_signals(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            35, &mut visited, &mut pool, po, k, rr, &train, dim, query,
        );
        let recall_35 = recall_at_k(&r35, gt_slice, k);

        // ef=65
        let (r65, vis65, _) = search_with_signals(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            65, &mut visited, &mut pool, po, k, rr, &train, dim, query,
        );
        let recall_65 = recall_at_k(&r65, gt_slice, k);

        // ef=100
        let (r100, _, _) = search_with_signals(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            100, &mut visited, &mut pool, po, k, rr, &train, dim, query,
        );
        let recall_100 = recall_at_k(&r100, gt_slice, k);

        // 写 CSV 行
        // 替换 f32::MAX 为 NaN（Python 端好处理）
        let k5 = if sig[0].kth_dist.is_finite() { sig[0].kth_dist } else { f32::NAN };
        let w5 = if sig[0].worst_dist.is_finite() { sig[0].worst_dist } else { f32::NAN };
        let k10 = if sig[1].kth_dist.is_finite() { sig[1].kth_dist } else { f32::NAN };
        let w10 = if sig[1].worst_dist.is_finite() { sig[1].worst_dist } else { f32::NAN };
        let k15 = if sig[2].kth_dist.is_finite() { sig[2].kth_dist } else { f32::NAN };
        let w15 = if sig[2].worst_dist.is_finite() { sig[2].worst_dist } else { f32::NAN };

        writeln!(csv, "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},",
            nav_dist,
            sig[0].insert_rate, k5, w5,
            sig[1].insert_rate, k10, w10,
            sig[2].insert_rate, k15, w15,
            recall_35, recall_65, recall_100,
            vis35, vis65,
        ).unwrap();
    }

    csv.flush().unwrap();
    println!("done: {:.1}s, saved to {}", t0.elapsed().as_secs_f64(), csv_path);
}
