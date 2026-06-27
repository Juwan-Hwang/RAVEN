//! 邻居重叠率诊断 (1M)
//!
//! 核心假设：RAVEN avg_visited 高的根因是邻居重叠率低。
//!
//! Glass HNSW: 增量插入 → 每个节点的邻居是当时图里已有的节点 →
//!   密集局部簇，邻居之间互相连接 → 搜索时展开 u，u 的邻居大部分已在 visited
//!   avg_visited=1041 (H20 实测), ef=50 → 每次 pop 平均引入 ~21 个新节点 (1041/50=21)
//!   邻居重叠率 ≈ (32-21)/32 ≈ 34%
//!
//! RAVEN Vamana: 批量建图 → 打破局部性 →
//!   每次展开都引入大量新节点 → avg_visited=1247, ef=50 → 24 新节点/pop
//!   邻居重叠率 ≈ (32-24)/32 ≈ 25%
//!
//! 本实验直接测量：每次 pop 节点 u 时：
//!   - u 的邻居总数 (degree)
//!   - 已在 visited 的数量 (overlap)
//!   - 新节点数量 (new = degree - overlap)
//!   - 重叠率 = overlap / degree
//!
//! 用法：cargo run --release --bin overlap_probe

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, PruneStrategy};
use raven::build::ChaCha8Rng;
use raven::memory::VisitedTracker;
use raven::graph::linear_pool::LinearPool;
use raven::distance::l2_simd;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
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
    let mut file = File::open(path).expect("open ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read ivecs");
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

/// 按 pop 顺序记录每跳的邻居重叠统计
#[derive(Default)]
struct OverlapStats {
    /// 每跳的邻居总数
    total_neighbors: Vec<usize>,
    /// 每跳的已访问邻居数
    overlap_count: Vec<usize>,
    /// 每跳的新节点数
    new_count: Vec<usize>,
    /// 每跳的 pop 后 visited 总数
    visited_after_pop: Vec<usize>,
}

fn search_with_overlap(
    vectors: &[f32],
    dim: usize,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
) -> (Vec<(u32, f32)>, OverlapStats) {
    visited.reset();
    let mut pool = LinearPool::new(ef);
    let mut stats = OverlapStats::default();

    let entry_dist = l2_simd(
        query,
        &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
    );
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    while let Some((node, _dist)) = pool.pop() {
        let neighbors = storage.neighbors(node);

        // 统计这一跳的邻居重叠
        let mut total_nb = 0usize;
        let mut overlap_nb = 0usize;
        let mut new_nb = 0usize;

        for &neighbor in neighbors {
            if neighbor == u32::MAX { continue; }
            total_nb += 1;
            if visited.is_visited(neighbor) {
                overlap_nb += 1;
            } else {
                new_nb += 1;
            }
        }

        stats.total_neighbors.push(total_nb);
        stats.overlap_count.push(overlap_nb);
        stats.new_count.push(new_nb);
        stats.visited_after_pop.push(visited.visited_count());

        // 实际展开（标记 + 距离计算 + 插入 pool）
        for &neighbor in neighbors {
            if neighbor == u32::MAX { continue; }
            if visited.visit(neighbor) {
                let d = l2_simd(
                    query,
                    &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                );
                pool.insert(neighbor, d);
            }
        }
    }

    (pool.to_sorted_vec(), stats)
}

fn run_overlap_diagnostic(
    label: &str,
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
) {
    let storage = graph.storage();
    let n = train.len() / dim;
    let mut visited = VisitedTracker::new(n, ef);

    // 全局统计
    let mut total_popped = 0u64;
    let mut total_neighbors_sum = 0u64;
    let mut total_overlap_sum = 0u64;
    let mut total_new_sum = 0u64;
    let mut hits = 0usize;
    let mut total = 0usize;

    // 按跳数聚合（前 60 跳）
    let max_hops = 60;
    let mut hop_neighbors: Vec<u64> = vec![0; max_hops];
    let mut hop_overlap: Vec<u64> = vec![0; max_hops];
    let mut hop_new: Vec<u64> = vec![0; max_hops];
    let mut hop_count: Vec<u64> = vec![0; max_hops];

    // 每查询的总重叠率
    let mut per_query_overlap: Vec<f64> = Vec::with_capacity(nq);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];

        let entry_point = if let Some(nav) = graph.layered_nav() {
            nav.initialize(train, dim, query).0
        } else {
            graph.entry_point()
        };

        let (result, stats) = search_with_overlap(
            train, dim, storage, entry_point, query, ef, &mut visited,
        );

        let n_pops = stats.total_neighbors.len();
        total_popped += n_pops as u64;
        let q_total_nb: u64 = stats.total_neighbors.iter().map(|&x| x as u64).sum();
        let q_overlap: u64 = stats.overlap_count.iter().map(|&x| x as u64).sum();
        let q_new: u64 = stats.new_count.iter().map(|&x| x as u64).sum();
        total_neighbors_sum += q_total_nb;
        total_overlap_sum += q_overlap;
        total_new_sum += q_new;

        let q_overlap_rate = if q_total_nb > 0 { q_overlap as f64 / q_total_nb as f64 } else { 0.0 };
        per_query_overlap.push(q_overlap_rate);

        // 按跳数聚合
        for hop in 0..n_pops.min(max_hops) {
            hop_neighbors[hop] += stats.total_neighbors[hop] as u64;
            hop_overlap[hop] += stats.overlap_count[hop] as u64;
            hop_new[hop] += stats.new_count[hop] as u64;
            hop_count[hop] += 1;
        }

        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) { hits += 1; }
        }
        total += k;
    }
    let dt = t0.elapsed();

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / dt.as_secs_f64();
    let avg_visited = (total_overlap_sum + total_new_sum + nq as u64) as f64 / nq as f64; // approx
    // 更精确：用 visited_after_pop 的最后一个值
    let avg_popped = total_popped as f64 / nq as f64;
    let avg_new_per_pop = total_new_sum as f64 / total_popped as f64;
    let avg_overlap_per_pop = total_overlap_sum as f64 / total_popped as f64;
    let avg_degree_per_pop = total_neighbors_sum as f64 / total_popped as f64;
    let overall_overlap_rate = total_overlap_sum as f64 / total_neighbors_sum as f64;

    per_query_overlap.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = per_query_overlap[nq / 2];
    let p05 = per_query_overlap[(nq as f64 * 0.05) as usize];

    println!("\n=== {} (ef={}) ===", label, ef);
    println!("  recall={:.4}  QPS={:.0}  avg_popped={:.1}", recall, qps, avg_popped);
    println!("  avg_visited ≈ {:.0}  (overlap+new+entry per query)", avg_visited);
    println!();
    println!("  邻居重叠率: {:.1}%  (avg {}/{:.1} neighbors already visited)",
             overall_overlap_rate * 100.0,
             avg_overlap_per_pop as i32,
             avg_degree_per_pop);
    println!("  每跳新增节点: {:.1}  (Glass 理论值: 3.0)", avg_new_per_pop);
    println!("  每跳邻居总数: {:.1}  (degree)", avg_degree_per_pop);
    println!();
    println!("  每查询重叠率: p5={:.1}%  p50={:.1}%", p05 * 100.0, p50 * 100.0);

    // 按跳数的重叠率曲线
    println!("\n  按跳数重叠率 (前 {} 跳):", max_hops);
    print!("    hop:  ");
    for h in 0..max_hops.min(30) { print!("{:>5}", h); }
    println!();
    print!("    overlap%: ");
    for h in 0..max_hops.min(30) {
        if hop_count[h] > 0 {
            let rate = hop_overlap[h] as f64 / hop_neighbors[h] as f64 * 100.0;
            print!("{:>4.0}%", rate);
        } else {
            print!("    --");
        }
    }
    println!();
    print!("    new/pop:  ");
    for h in 0..max_hops.min(30) {
        if hop_count[h] > 0 {
            let new_avg = hop_new[h] as f64 / hop_count[h] as f64;
            print!("{:>5.1}", new_avg);
        } else {
            print!("    --");
        }
    }
    println!();
}

fn main() {
    println!("=== 邻居重叠率诊断 (1M) ===");
    println!("测量: 展开 u 时，u 的邻居有多少已在 visited");
    println!("Glass 实测 (H20): 重叠率 ~34%, 新节点 21/pop, avg_visited=1041");
    println!("RAVEN 预期: 重叠率 ~25%, 新节点 24/pop, avg_visited=1247");
    println!();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    // 建图: RobustPrune α=1.2 (baseline)
    println!("\n--- 建图: RobustPrune α=1.2 ---");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        alpha_layer0: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let stats = graph.degree_stats();
    println!("建图: {:.1}s, mean_degree={:.1}", t0.elapsed().as_secs_f64(), stats.mean_degree);

    // 诊断
    let diag_nq = 1000usize.min(nq);
    for &ef in &[50, 100] {
        run_overlap_diagnostic("RobustPrune α=1.2", &train, &graph, &test, dim, diag_nq, &gt, gt_k, k, ef);
    }

    println!("\n=== 结论 ===");
    println!("如果重叠率 << 91% → 验证了根因是局部簇密度不足");
    println!("修复方向: 后处理增加邻居重叠（如三角剖分闭合）或增量插入模拟 HNSW");
}
