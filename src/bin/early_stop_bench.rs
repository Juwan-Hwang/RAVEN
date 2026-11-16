//! 早停启发式实验 v2
//!
//! 三种早停判据对比：
//! 1. any_inserted: 连续 N 次 pop 没有新节点进入 pool（v1，已知太弱）
//! 2. topk_stable:  连续 N 次 pop 没有 top-k 距离改善
//! 3. dist_ratio:   popped 距离 > α × k-th 距离（hnswlib 式）
//!
//! 不需要重建图——同一张图、同一套 SQ4 码，只改搜索终止条件。
//!
//! 用法：cargo run --release --bin early_stop_bench

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

/// 早停策略
#[derive(Clone, Copy)]
enum EarlyStop {
    /// 禁用（baseline）
    None,
    /// 连续 N 次 pop 没有新节点进入 pool
    AnyInserted { n: usize },
    /// 连续 N 次 pop 没有 top-k 距离改善
    TopkStable { n: usize },
    /// popped 距离 > α × k-th 距离（连续触发 N 次）
    DistRatio { alpha: f32, n: usize },
}

impl EarlyStop {
    fn label(&self) -> String {
        match self {
            EarlyStop::None => "baseline".to_string(),
            EarlyStop::AnyInserted { n } => format!("any_ins({})", n),
            EarlyStop::TopkStable { n } => format!("topk({})", n),
            EarlyStop::DistRatio { alpha, n } => format!("ratio({},{})", alpha, n),
        }
    }
}

/// 带早停的 SQ4 贪心搜索
fn greedy_search_sq4_es(
    sq4: &SQ4Dataset,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query_code: &[u8],
    ef: usize,
    visited: &mut VisitedTracker,
    pool: &mut LinearPool,
    po: usize,
    search_degree: usize,
    es: EarlyStop,
    k: usize,
) -> (Vec<(u32, f32)>, usize) {
    visited.reset();
    pool.reset_for(ef);

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];

    // 早停状态
    let mut stale_pops: usize = 0;

    while let Some((node, pop_dist)) = pool.pop() {
        // Multi-line graph prefetch
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

        // 第一遍：收集未访问邻居
        let mut edge_size = 0usize;
        for &v in neighbors.iter().take(effective_len) {
            if edge_size >= 128 {
                break;
            }
            let is_new = unsafe { visited.visit_unchecked_branchless(v) };
            unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
            edge_size += is_new as usize;
        }

        // 预取
        let prefetch_count = po.min(edge_size);
        for &node in edge_buf.iter().take(prefetch_count) {
            let v = node as usize;
            let ptr = sq4.code(v).as_ptr() as *const i8;
            unsafe { raven::arch::prefetch_l1(ptr); }
        }

        // 记录展开前的 top-k 距离（用于 TopkStable 判据）
        let pre_kth = match es {
            EarlyStop::TopkStable { .. } => pool.kth_distance(k),
            EarlyStop::DistRatio { .. } => pool.kth_distance(k),
            _ => 0.0,
        };

        // 第二遍：距离计算 + insert
        let mut any_inserted = false;
        for i in 0..edge_size {
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = sq4.code(v).as_ptr() as *const i8;
                unsafe { raven::arch::prefetch_l1(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
            if pool.insert(neighbor, d) {
                any_inserted = true;
            }
        }

        // 早停判定
        match es {
            EarlyStop::None => {}
            EarlyStop::AnyInserted { n } => {
                if any_inserted {
                    stale_pops = 0;
                } else {
                    stale_pops += 1;
                    if stale_pops >= n {
                        break;
                    }
                }
            }
            EarlyStop::TopkStable { n } => {
                let post_kth = pool.kth_distance(k);
                // top-k 距离改善 = post_kth < pre_kth
                // 注意：pool 可能不满 k 个，此时 kth_distance 返回 MAX
                if post_kth < pre_kth {
                    stale_pops = 0;
                } else {
                    stale_pops += 1;
                    if stale_pops >= n {
                        break;
                    }
                }
            }
            EarlyStop::DistRatio { alpha, n } => {
                // pop_dist 是当前弹出的节点距离，pre_kth 是展开前的 k-th 距离
                // 如果 pop_dist > alpha * kth_dist，说明弹出的节点已经远离 top-k
                // 但 pre_kth 可能是 MAX（pool 不足 k 个），此时不触发
                if pre_kth > 0.0 && pre_kth < f32::MAX && pop_dist > alpha * pre_kth {
                    stale_pops += 1;
                    if stale_pops >= n {
                        break;
                    }
                } else {
                    stale_pops = 0;
                }
            }
        }
    }

    let vc = visited.visited_count();
    (pool.to_sorted_vec(), vc)
}

/// f32 rerank（与生产代码一致）
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

struct BenchResult {
    recall: f64,
    avg_visited: f64,
    qps: f64,
}

fn run_bench(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    gt_k: usize,
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
    ef: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
    graph: &VamanaGraph,
    es: EarlyStop,
) -> BenchResult {
    let nav = graph.layered_nav().expect("need layered nav");
    let mut visited = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);
    let code_bytes = dim.div_ceil(2);
    let mut query_code_buf = vec![0u8; code_bytes];

    let mut results = Vec::with_capacity(nq);
    let mut total_visited = 0usize;
    let t0 = Instant::now();

    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, _) = nav.initialize(train, dim, query);

        let (candidates, vc) = greedy_search_sq4_es(
            sq4, graph.storage(), entry_point, &query_code_buf,
            ef, &mut visited, &mut pool, po, 0, es, k,
        );

        total_visited += vc;
        let r = rerank_f32(train, dim, query, candidates, k, rr);
        results.push(r.iter().map(|(id, _)| *id).collect());
    }

    let dt = t0.elapsed();
    BenchResult {
        recall: compute_recall(&results, gt, gt_k, k, nq),
        avg_visited: total_visited as f64 / nq as f64,
        qps: nq as f64 / dt.as_secs_f64(),
    }
}

fn main() {
    println!("=== 早停启发式实验 v2 ===\n");

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

    // 建图
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

    // SQ4
    println!("--- 构建 SQ4 ---");
    let t0 = Instant::now();
    let sq4 = SQ4Dataset::build(&train, dim);
    println!("SQ4 构建时间: {:.1}s\n", t0.elapsed().as_secs_f64());

    // 所有早停策略
    let strategies: Vec<EarlyStop> = vec![
        EarlyStop::None,
        // v1: any_inserted（已知结果，快速复现）
        EarlyStop::AnyInserted { n: 5 },
        EarlyStop::AnyInserted { n: 10 },
        // v2: topk_stable
        EarlyStop::TopkStable { n: 3 },
        EarlyStop::TopkStable { n: 5 },
        EarlyStop::TopkStable { n: 8 },
        EarlyStop::TopkStable { n: 10 },
        EarlyStop::TopkStable { n: 15 },
        EarlyStop::TopkStable { n: 20 },
        EarlyStop::TopkStable { n: 30 },
        // v3: dist_ratio (hnswlib 式)
        EarlyStop::DistRatio { alpha: 1.5, n: 5 },
        EarlyStop::DistRatio { alpha: 2.0, n: 5 },
        EarlyStop::DistRatio { alpha: 2.0, n: 10 },
        EarlyStop::DistRatio { alpha: 3.0, n: 10 },
        EarlyStop::DistRatio { alpha: 3.0, n: 20 },
    ];

    println!("--- 搜索 (ef={}, k={}, nq={}, po={}, rr={}) ---\n", ef, k, nq, po, rr);
    println!("{:<20} {:<10} {:<12} {:<10} {:<15}", "strategy", "recall", "avg_visited", "qps", "Δvisited/Δrecall");
    println!("{}", "-".repeat(75));

    let mut baseline_visited = 0.0;
    let mut baseline_recall = 0.0;

    for es in &strategies {
        let r = run_bench(
            &train, &test, &gt, gt_k, dim, n, nq, k, ef, po, rr,
            &sq4, &graph, *es,
        );

        let label = es.label();
        if matches!(es, EarlyStop::None) {
            baseline_visited = r.avg_visited;
            baseline_recall = r.recall;
            println!(
                "{:<20} {:<10.4} {:<12.1} {:<10.0}",
                label, r.recall, r.avg_visited, r.qps,
            );
        } else {
            let dv = (r.avg_visited - baseline_visited) / baseline_visited * 100.0;
            let dr = (r.recall - baseline_recall) * 100.0; // in pp
            println!(
                "{:<20} {:<10.4} {:<12.1} {:<10.0} Δv:{:+.1}% Δr:{:+.2}pp",
                label, r.recall, r.avg_visited, r.qps, dv, dr,
            );
        }
    }

    println!("\n=== 实验完成 ===");
}
