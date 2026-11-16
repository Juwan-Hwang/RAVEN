//! 候选池质量诊断（1M 规模）：greedy_search 找到的候选中，真实近邻占多少？
//!
//! 10K 测试无效（pool 覆盖 18% 数据集，recall=1.0 是假象）。
//! 本版本在 1M 上测试：暴力搜索仅 1000 个采样节点的真实 top-32，
//! 然后对建好的图跑 greedy_search，统计候选池召回率。
//!
//! 用法：cargo run --release --bin candidate_recall

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;
use raven::memory::VisitedTracker;
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

fn main() {
    println!("=== 候选池质量诊断 (1M) ===");
    println!("假设: 1M 规模下 greedy_search 候选池不含真实近邻");
    println!();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    for v in train.iter_mut() { *v /= 255.0; }
    println!("数据: n={}, dim={}", n, dim);

    let l_build = 200usize;
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
            if i % 100 == 0 && i > 0 {
                eprintln!("[brute] {}/{}", i, sample_size);
            }
            brute_force_knn(&train, dim, n, node_idx, k_true)
        })
        .collect();
    println!("暴力搜索完成: {:.1}s ({} nodes x {} distance computations)",
             t0.elapsed().as_secs_f64(), sample_size, n);

    // 2. 构建 1M 图 (iter=2, 与生产配置一致)
    println!("\n--- 构建 Vamana 1M (max_iterations=2) ---");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: false,
        nav_m: 16,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图完成: {:.1}s", build_time);

    let stats = graph.degree_stats();
    println!("度数: mean={:.1} max={} p95={} isolated={}",
             stats.mean_degree, stats.max_degree, stats.p95_degree, stats.isolated_nodes);

    // 3. 测量候选池召回率
    let storage = graph.storage();
    let entry_point = graph.entry_point();
    let mut visited = VisitedTracker::new(n, l_build);

    let mut total_recall = 0.0f64;
    let mut total_top1 = 0.0f64;
    let mut total_pool = 0.0f64;
    let mut recall_histogram = vec![0u32; 11]; // 0.0, 0.1, ..., 1.0

    println!("\n--- 候选池召回率诊断 (sample={}) ---", sample_size);
    for (i, &node_idx) in sample_indices.iter().enumerate() {
        let query = &train[node_idx * dim..(node_idx + 1) * dim];

        let _top = VamanaGraph::greedy_search_vec_build(
            &train, dim, storage, entry_point, query, l_build, &mut visited,
        );

        let candidates = visited.visited_nodes();
        let pool_size = candidates.len();
        total_pool += pool_size as f64;

        let candidate_set: std::collections::HashSet<u32> = candidates.iter().copied().collect();
        let true_nn = &true_knn[i];
        let hits = true_nn.iter().filter(|&&id| candidate_set.contains(&id)).count();
        let recall = hits as f64 / k_true as f64;
        total_recall += recall;

        if !true_nn.is_empty() && candidate_set.contains(&true_nn[0]) {
            total_top1 += 1.0;
        }

        let bucket = (recall * 10.0) as usize;
        recall_histogram[bucket.min(10)] += 1;
    }

    let m = sample_size as f64;
    println!("\n结果:");
    println!("  candidate recall@32 = {:.4}", total_recall / m);
    println!("  top-1 hit rate      = {:.4}", total_top1 / m);
    println!("  avg pool size       = {:.1}  ({:.3}% of dataset)",
             total_pool / m, (total_pool / m) / n as f64 * 100.0);

    println!("\n召回率分布:");
    for (bucket, &count) in recall_histogram.iter().enumerate() {
        if count > 0 {
            let lo = bucket as f64 * 0.1;
            let hi = lo + 0.1;
            println!("  [{:.1}-{:.1}): {} ({:.1}%)", lo, hi, count, count as f64 / m * 100.0);
        }
    }

    println!("\n=== 诊断结论 ===");
    let avg_recall = total_recall / m;
    if avg_recall < 0.5 {
        println!("candidate recall@32 = {:.4} < 0.5 → 候选池是根因", avg_recall);
        println!("greedy_search 在 1M 图上找不到真实近邻 → 剪枝策略无论怎么改都无效");
        println!("修复方向: 增加 l_build / 增加迭代轮数 / 改进搜索入口点");
    } else if avg_recall < 0.9 {
        println!("candidate recall@32 = {:.4} (0.5-0.9) → 候选池部分缺失", avg_recall);
        println!("部分真实近邻不在候选池中，剪枝策略和候选池质量都有改进空间");
    } else {
        println!("candidate recall@32 = {:.4} > 0.9 → 候选池质量良好", avg_recall);
        println!("问题不在候选池，而在剪枝策略或搜索过程的展开效率");
    }
}
