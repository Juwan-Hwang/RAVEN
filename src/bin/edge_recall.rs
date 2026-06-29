//! edge_recall 诊断：PureDistance vs RobustPrune 在 10K 上的边召回率
//!
//! 核心发现：
//!   - candidate recall@32 = 0.9993（候选池有真实近邻）
//!   - RobustPrune α=1.0 两轮 edge_recall = 0.2495（最差！角度遮挡+不填满）
//!   - RobustPrune α=1.2 两轮 edge_recall = 0.3866
//!   - RobustPrune iter1=1.0+iter2=1.2 edge_recall = 0.4173（旧代码，最好）
//!
//! α=1.0 并非"纯距离排序"——RobustPrune 的 α=1.0 仍有角度遮挡（occlude_factor > 1.0），
//! 且只跑一轮不填满 r_max，导致 edge_recall 反而最低。
//!
//! PureDistance 跳过所有角度检查，直接取最近的 r_max 个候选。
//! 预期 edge_recall 接近 candidate recall = 0.9993。
//!
//! 用法：cargo run --release --bin edge_recall

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, PruneStrategy};
use raven::build::ChaCha8Rng;
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

/// 暴力搜索 top-k 近邻（返回节点 ID 列表，已按距离升序）
fn brute_force_knn(
    vectors: &[f32],
    dim: usize,
    n: usize,
    query_idx: usize,
    k: usize,
) -> Vec<u32> {
    let query = &vectors[query_idx * dim..(query_idx + 1) * dim];
    let mut dists: Vec<(f32, u32)> = (0..n)
        .filter(|&i| i != query_idx)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            (l2_simd(query, v), i as u32)
        })
        .collect();
    dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    dists.into_iter().take(k).map(|(_, id)| id).collect()
}

/// 测量 edge_recall：图中邻居列表覆盖真实 top-k 的比例
fn measure_edge_recall(
    label: &str,
    graph: &VamanaGraph,
    vectors: &[f32],
    dim: usize,
    n: usize,
    sample_indices: &[usize],
    true_knn: &[Vec<u32>],
    k_true: usize,
) {
    let storage = graph.storage();
    let mut total_recall = 0.0f64;
    let mut total_top1 = 0.0f64;
    let mut total_top5 = 0.0f64;
    let mut total_neighbor_rank = 0.0f64;
    let mut rank_count = 0usize;
    let mut recall_histogram = vec![0u32; 11]; // 0.0, 0.1, ..., 1.0

    for (i, &node_idx) in sample_indices.iter().enumerate() {
        let neighbors = storage.neighbors(node_idx as u32);
        let neighbor_set: std::collections::HashSet<u32> = neighbors.iter().copied().collect();

        let true_nn = &true_knn[i];
        let hits = true_nn.iter().filter(|&&id| neighbor_set.contains(&id)).count();
        let recall = hits as f64 / k_true as f64;
        total_recall += recall;

        if !true_nn.is_empty() && neighbor_set.contains(&true_nn[0]) {
            total_top1 += 1.0;
        }
        let top5 = true_nn.iter().take(5).filter(|&&id| neighbor_set.contains(&id)).count();
        total_top5 += top5 as f64 / 5.0;

        // 邻居在真实 top-k 中的排名（0=最近, k_true=最远/不在 top-k）
        for &nb in neighbors {
            if let Some(rank) = true_nn.iter().position(|&id| id == nb) {
                total_neighbor_rank += rank as f64;
                rank_count += 1;
            } else {
                total_neighbor_rank += k_true as f64;
                rank_count += 1;
            }
        }

        let bucket = (recall * 10.0) as usize;
        recall_histogram[bucket.min(10)] += 1;
    }

    let m = sample_indices.len() as f64;
    println!("\n结果 ({})", label);
    println!("  edge_recall@{}  = {:.4}  (图中邻居覆盖真实 top-{} 的比例)", k_true, total_recall / m, k_true);
    println!("  top-1 hit rate  = {:.4}  (最近真实近邻是否在邻居列表中)", total_top1 / m);
    println!("  top-5 recall    = {:.4}  (真实 top-5 中有多少在邻居列表中)", total_top5 / m);
    println!("  avg neighbor rank = {:.2}  (邻居在真实排名中的位置, 0=最近, {}=不在 top-{})",
             total_neighbor_rank / rank_count as f64, k_true, k_true);

    println!("\n  edge_recall 分布:");
    for (bucket, &count) in recall_histogram.iter().enumerate() {
        if count > 0 {
            let lo = bucket as f64 * 0.1;
            let hi = lo + 0.1;
            let bar = "#".repeat((count as f64 / m * 200.0) as usize);
            println!("    [{:.1}-{:.1}): {:>4} ({:>5.1}%) {}", lo, hi, count, count as f64 / m * 100.0, bar);
        }
    }
}

fn main() {
    println!("=== edge_recall 诊断: PureDistance vs RobustPrune (10K) ===");
    println!("edge_recall = |graph_neighbors ∩ true_knn| / r_max");
    println!();

    // 读取 SIFT 数据，只取前 10K（与旧实验一致）
    let (mut train, dim, n_full) = read_fvecs("data/sift/sift_base.fvecs");
    let n = 10_000usize.min(n_full);
    train.truncate(n * dim);
    for v in train.iter_mut() { *v /= 255.0; }
    println!("数据: n={}, dim={}", n, dim);

    let k_true = 32usize;
    let sample_size = 1000usize;

    // 1. 采样 1000 个节点，暴力搜索真实 top-32
    println!("\n--- 暴力搜索 {} 个采样节点的真实 top-{} ---", sample_size, k_true);
    let t0 = Instant::now();
    let step = n / sample_size;
    let sample_indices: Vec<usize> = (0..n).step_by(step).take(sample_size).collect();
    let true_knn: Vec<Vec<u32>> = sample_indices
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| {
            if i % 200 == 0 && i > 0 {
                eprintln!("[brute] {}/{}", i, sample_size);
            }
            brute_force_knn(&train, dim, n, node_idx, k_true)
        })
        .collect();
    println!("暴力搜索完成: {:.1}s", t0.elapsed().as_secs_f64());

    // 2. 建图 A: RobustPrune α=1.0 两轮（之前的 baseline）
    println!("\n--- 建图 A: RobustPrune α=1.0 (两轮) ---");
    let t0 = Instant::now();
    let mut rng_a = ChaCha8Rng::seed_from(42);
    let config_a = VamanaBuildConfig {
        alpha: 1.2,
        alpha_layer0: 1.0,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
        ..Default::default()
    };
    let graph_a = VamanaGraph::build(&train, dim, &config_a, &mut rng_a);
    let stats_a = graph_a.degree_stats();
    println!("建图 A: {:.1}s, mean_degree={:.1}", t0.elapsed().as_secs_f64(), stats_a.mean_degree);

    // 3. 建图 B: RobustPrune α=1.2 两轮
    println!("\n--- 建图 B: RobustPrune α=1.2 (两轮) ---");
    let t0 = Instant::now();
    let mut rng_b = ChaCha8Rng::seed_from(42);
    let config_b = VamanaBuildConfig {
        alpha: 1.2,
        alpha_layer0: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
        ..Default::default()
    };
    let graph_b = VamanaGraph::build(&train, dim, &config_b, &mut rng_b);
    let stats_b = graph_b.degree_stats();
    println!("建图 B: {:.1}s, mean_degree={:.1}", t0.elapsed().as_secs_f64(), stats_b.mean_degree);

    // 4. 建图 C: PureDistance（无角度遮挡，直接取最近 r_max 个）
    println!("\n--- 建图 C: PureDistance (无角度遮挡) ---");
    let t0 = Instant::now();
    let mut rng_c = ChaCha8Rng::seed_from(42);
    let config_c = VamanaBuildConfig {
        alpha: 1.2,
        alpha_layer0: 1.0,       // PureDistance 忽略 alpha
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::PureDistance,
        ..Default::default()
    };
    let graph_c = VamanaGraph::build(&train, dim, &config_c, &mut rng_c);
    let stats_c = graph_c.degree_stats();
    println!("建图 C: {:.1}s, mean_degree={:.1}", t0.elapsed().as_secs_f64(), stats_c.mean_degree);

    // 5. 测量 edge_recall
    measure_edge_recall("A: RobustPrune α=1.0 (两轮)", &graph_a, &train, dim, n, &sample_indices, &true_knn, k_true);
    measure_edge_recall("B: RobustPrune α=1.2 (两轮)", &graph_b, &train, dim, n, &sample_indices, &true_knn, k_true);
    measure_edge_recall("C: PureDistance", &graph_c, &train, dim, n, &sample_indices, &true_knn, k_true);

    // 6. 诊断结论
    println!("\n=== 诊断结论 ===");
    println!("已知数据:");
    println!("  candidate recall@32 = 0.9993 (候选池质量)");
    println!("  RobustPrune iter1=1.0+iter2=1.2 edge_recall = 0.4173 (旧代码)");
    println!("  RobustPrune α=1.0 两轮 edge_recall = 0.2495 (角度遮挡+不填满)");
    println!("  RobustPrune α=1.2 两轮 edge_recall = 0.3866");
    println!();
    println!("PureDistance 预期 edge_recall ≈ 0.90+ (接近 candidate recall)");
    println!("如果成立 → 上 1M 测 avg_visited");
}
