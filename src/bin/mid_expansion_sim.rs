//! 中途扩容模拟 —— 验证"从 pop=15 扩容继续搜索"vs"从零用 ef=65 完整搜索"的 recall 差距
//!
//! 对每个 query 执行 7 种搜索模式：
//!  1. ef=35 (full)           → recall_35, visited_35
//!  2. ef=50 (full)           → recall_50, visited_50  (基线)
//!  3. ef=65 (full)           → recall_65, visited_65  (两阶段目标)
//!  4. ef=80 (full)           → recall_80, visited_80  (两阶段目标)
//!  5. expand_65: ef=35 → pop=15 → expand_ef(65) → continue → recall_exp65, visited_exp65
//!  6. expand_80: ef=35 → pop=15 → expand_ef(80) → continue → recall_exp80, visited_exp80
//!  7. 同时在 ef=35 搜索的 pop=5/10/15 收集 10 维特征
//!
//! 输出 CSV 供 Python 分析：
//!  - 特征（10 维）+ recall/visited（6 组）
//!  - 用全部 10000 queries 验证 AUC 稳定性
//!
//! 用法：cargo run --release --bin mid_expansion_sim

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

#[derive(Clone, Copy, Default)]
struct CpSignals {
    insert_rate: f32,
    kth_dist: f32,
    worst_dist: f32,
}

const CHECKPOINTS: [usize; 3] = [5, 10, 15];
const EXPAND_AT: usize = 15;

/// 搜索模式
enum SearchMode {
    /// 固定 ef 完整搜索
    Fixed(usize),
    /// 中途扩容：初始 ef → pop=EXPAND_AT → expand to target_ef → continue
    MidExpand { initial_ef: usize, target_ef: usize },
}

/// 通用搜索函数，支持固定 ef 和中途扩容两种模式
fn search_generic(
    sq4: &SQ4Dataset,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query_code: &[u8],
    mode: SearchMode,
    visited: &mut VisitedTracker,
    pool: &mut LinearPool,
    po: usize,
    k: usize,
    rr: usize,
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    mut signals: Option<&mut [CpSignals; 3]>,
) -> (Vec<(u32, f32)>, usize) {
    let (initial_ef, target_ef) = match mode {
        SearchMode::Fixed(ef) => (ef, ef),
        SearchMode::MidExpand { initial_ef, target_ef } => (initial_ef, target_ef),
    };

    visited.reset();
    pool.reset_for(initial_ef);

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];
    let mut pop_order: usize = 0;
    let mut total_attempts: usize = 0;
    let mut total_inserts: usize = 0;
    let mut cp_idx: usize = 0;
    let mut expanded = initial_ef == target_ef; // Fixed mode = already "expanded"

    while let Some((node, _dist)) = pool.pop() {
        pop_order += 1;

        // 检查点信号记录
        if let Some(sigs) = signals.as_deref_mut() {
            while cp_idx < CHECKPOINTS.len() && pop_order == CHECKPOINTS[cp_idx] {
                let ins_rate = if total_attempts > 0 {
                    total_inserts as f32 / total_attempts as f32
                } else {
                    0.0
                };
                sigs[cp_idx] = CpSignals {
                    insert_rate: ins_rate,
                    kth_dist: pool.kth_distance(k),
                    worst_dist: pool.worst_distance(),
                };
                cp_idx += 1;
            }
        }

        // 中途扩容
        if !expanded && pop_order == EXPAND_AT {
            pool.expand_ef(target_ef);
            expanded = true;
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

    // 填充未到达的检查点
    if let Some(sigs) = signals.as_deref_mut() {
        while cp_idx < CHECKPOINTS.len() {
            let ins_rate = if total_attempts > 0 {
                total_inserts as f32 / total_attempts as f32
            } else {
                0.0
            };
            sigs[cp_idx] = CpSignals {
                insert_rate: ins_rate,
                kth_dist: pool.kth_distance(k),
                worst_dist: pool.worst_distance(),
            };
            cp_idx += 1;
        }
    }

    let vc = visited.visited_count();
    let candidates = pool.to_sorted_vec();
    let results = rerank_f32(vectors, dim, query, candidates, k, rr);

    (results, vc)
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    println!("=== 中途扩容模拟 (10K queries) ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = nq_full; // 使用全部 queries
    let k = 10;
    let po = 16;
    let rr = 5;

    println!("data: n={}, dim={}, nq={}, gt_k={}\n", n, dim, nq, gt_k);

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
    let csv_path = "experiments/mid_expansion_data.csv";
    let mut csv = BufWriter::new(File::create(csv_path).expect("create csv"));

    // 表头
    writeln!(csv, "nav_dist,ir5,kth5,worst5,ir10,kth10,worst10,ir15,kth15,worst15,r35,v35,r50,v50,r65,v65,r80,v80,r_exp65,v_exp65,r_exp80,v_exp80").unwrap();

    println!("searching (7 modes x {} queries)...", nq);
    let t0 = Instant::now();

    for q in 0..nq {
        if q % 2000 == 0 && q > 0 {
            println!("  {}/{} ({:.1}s)", q, nq, t0.elapsed().as_secs_f64());
        }

        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, nav_dist) = nav.initialize(&train, dim, query);
        let gt_slice = &gt[q * gt_k..(q + 1) * gt_k];

        // 模式 1: ef=35 with signal collection
        let mut sig = [CpSignals::default(); 3];
        let (r35, v35) = search_generic(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            SearchMode::Fixed(35), &mut visited, &mut pool,
            po, k, rr, &train, dim, query,
            Some(&mut sig),
        );
        let recall_35 = recall_at_k(&r35, gt_slice, k);

        // 模式 2: ef=50
        let (r50, v50) = search_generic(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            SearchMode::Fixed(50), &mut visited, &mut pool,
            po, k, rr, &train, dim, query, None,
        );
        let recall_50 = recall_at_k(&r50, gt_slice, k);

        // 模式 3: ef=65
        let (r65, v65) = search_generic(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            SearchMode::Fixed(65), &mut visited, &mut pool,
            po, k, rr, &train, dim, query, None,
        );
        let recall_65 = recall_at_k(&r65, gt_slice, k);

        // 模式 4: ef=80
        let (r80, v80) = search_generic(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            SearchMode::Fixed(80), &mut visited, &mut pool,
            po, k, rr, &train, dim, query, None,
        );
        let recall_80 = recall_at_k(&r80, gt_slice, k);

        // 模式 5: mid-expand ef=35 → pop=15 → expand to 65
        let (r_exp65, v_exp65) = search_generic(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            SearchMode::MidExpand { initial_ef: 35, target_ef: 65 },
            &mut visited, &mut pool,
            po, k, rr, &train, dim, query, None,
        );
        let recall_exp65 = recall_at_k(&r_exp65, gt_slice, k);

        // 模式 6: mid-expand ef=35 → pop=15 → expand to 80
        let (r_exp80, v_exp80) = search_generic(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            SearchMode::MidExpand { initial_ef: 35, target_ef: 80 },
            &mut visited, &mut pool,
            po, k, rr, &train, dim, query, None,
        );
        let recall_exp80 = recall_at_k(&r_exp80, gt_slice, k);

        // 写 CSV
        let k5 = if sig[0].kth_dist.is_finite() { sig[0].kth_dist } else { f32::NAN };
        let w5 = if sig[0].worst_dist.is_finite() { sig[0].worst_dist } else { f32::NAN };
        let k10 = if sig[1].kth_dist.is_finite() { sig[1].kth_dist } else { f32::NAN };
        let w10 = if sig[1].worst_dist.is_finite() { sig[1].worst_dist } else { f32::NAN };
        let k15 = if sig[2].kth_dist.is_finite() { sig[2].kth_dist } else { f32::NAN };
        let w15 = if sig[2].worst_dist.is_finite() { sig[2].worst_dist } else { f32::NAN };

        writeln!(csv, "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{},{:.4},{},{:.4},{},{:.4},{},{:.4},{}",
            nav_dist,
            sig[0].insert_rate, k5, w5,
            sig[1].insert_rate, k10, w10,
            sig[2].insert_rate, k15, w15,
            recall_35, v35,
            recall_50, v50,
            recall_65, v65,
            recall_80, v80,
            recall_exp65, v_exp65,
            recall_exp80, v_exp80,
        ).unwrap();
    }

    csv.flush().unwrap();
    println!("\ndone: {:.1}s, saved to {}", t0.elapsed().as_secs_f64(), csv_path);
}
