//! PureDistance 搜索质量验证 (10K)
//!
//! PureDistance edge_recall=0.9488 已验证，但代价是零桥接边。
//! 本实验在 10K 上测搜索质量（avg_visited + recall），确认 recall 不会暴跌。
//!
//! 三组配置：
//!   A: RobustPrune α=1.2（baseline，有桥接边）
//!   B: PureDistance（纯局部，零桥接边）
//!   C: PureDistance Layer0 + RobustPrune nav（分离：局部+导航）
//!
//! 注意：sift_groundtruth.ivecs 是针对完整 1M 的，截断到 10K 后无效。
//! 必须自己算 10K 子集的暴力 ground truth。
//!
//! 用法：cargo run --release --bin pure_distance_10k

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, PruneStrategy};
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
///
/// 10K 子集的 ground truth 必须自己算——sift_groundtruth.ivecs 是针对完整 1M 的。
fn brute_force_knn(
    vectors: &[f32],
    dim: usize,
    n: usize,
    query: &[f32],
    k: usize,
) -> Vec<u32> {
    let mut dists: Vec<(f32, u32)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            (l2_simd(query, v), i as u32)
        })
        .collect();
    dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    dists.into_iter().take(k).map(|(_, id)| id).collect()
}

struct SearchStats {
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

fn run_search(
    label: &str,
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
) -> SearchStats {
    let storage = graph.storage();
    let n = train.len() / dim;
    let mut visited = VisitedTracker::new(n, ef);

    let mut hits = 0usize;
    let mut total = 0usize;
    let mut total_visited = 0u64;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];

        let entry_point = if let Some(nav) = graph.layered_nav() {
            nav.initialize(train, dim, query).0
        } else {
            graph.entry_point()
        };

        visited.reset();
        use raven::graph::linear_pool::LinearPool;
        let mut pool = LinearPool::new(ef);

        let entry_dist = l2_simd(
            query,
            &train[entry_point as usize * dim..(entry_point as usize + 1) * dim],
        );
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        while let Some((node, _dist)) = pool.pop() {
            for &neighbor in storage.neighbors(node) {
                if neighbor == u32::MAX {
                    continue;
                }
                if visited.visit(neighbor) {
                    let d = l2_simd(
                        query,
                        &train[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                    );
                    pool.insert(neighbor, d);
                }
            }
        }

        total_visited += visited.visited_count() as u64;

        let result = pool.to_sorted_vec();
        let gt_slice = &gt[q];
        for &g in gt_slice.iter().take(k) {
            if result.iter().any(|(id, _)| *id == g) {
                hits += 1;
            }
        }
        total += k;
    }
    let dt = t0.elapsed();

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / dt.as_secs_f64();
    let avg_visited = total_visited as f64 / nq as f64;

    println!("  {:<40} recall={:.4}  QPS={:.0}  avg_visited={:.1}", label, recall, qps, avg_visited);

    SearchStats { recall, qps, avg_visited }
}

fn build_and_benchmark(
    label: &str,
    train: &[f32],
    dim: usize,
    config: &VamanaBuildConfig,
) -> VamanaGraph {
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let graph = VamanaGraph::build(train, dim, config, &mut rng);
    let stats = graph.degree_stats();
    println!("  {:<40} build={:.1}s  mean_degree={:.1}", label, t0.elapsed().as_secs_f64(), stats.mean_degree);
    graph
}

fn main() {
    println!("=== PureDistance 搜索质量验证 (10K) ===");
    println!("验证 PureDistance 的 recall 是否守住（vs RobustPrune baseline）");
    println!();

    let (mut train, dim, n_full) = read_fvecs("data/sift/sift_base.fvecs");
    let n = 10_000usize.min(n_full);
    train.truncate(n * dim);
    for v in train.iter_mut() { *v /= 255.0; }

    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    for v in test.iter_mut() { *v /= 255.0; }
    let k = 10usize;
    let gt_k = 100usize;

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    println!("计算 10K 子集暴力 ground truth ({} queries x {} nodes)...", nq, n);
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = (0..nq)
        .map(|q| {
            if q % 1000 == 0 && q > 0 { eprintln!("[brute] {}/{}", q, nq); }
            brute_force_knn(&train, dim, n, &test[q * dim..(q + 1) * dim], gt_k)
        })
        .collect();
    println!("暴力 ground truth 完成: {:.1}s", t0.elapsed().as_secs_f64());

    // === 建图 ===
    println!("\n--- 建图 ---");
    let graph_a = build_and_benchmark("A: RobustPrune α=1.2", &train, dim, &VamanaBuildConfig {
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
    });

    let graph_b = build_and_benchmark("B: PureDistance (纯局部)", &train, dim, &VamanaBuildConfig {
        alpha: 1.2,
        alpha_layer0: 1.0,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::PureDistance,
        ..Default::default()
    });

    let graph_c = build_and_benchmark("C: PureDistance + nav α=1.2", &train, dim, &VamanaBuildConfig {
        alpha: 1.2,
        alpha_layer0: 1.0,
        l_build: 200,
        r_max: 32,
        r_soft: 41,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 16,
        prune_strategy: PruneStrategy::PureDistance,
        ..Default::default()
    });

    // === 搜索质量 ===
    println!("\n--- 搜索质量 (recall / QPS / avg_visited) ---");
    for &ef in &[50, 100] {
        println!("\n  ef={}", ef);
        run_search("A: RobustPrune α=1.2", &train, &graph_a, &test, dim, nq, &gt, k, ef);
        run_search("B: PureDistance (纯局部)", &train, &graph_b, &test, dim, nq, &gt, k, ef);
        run_search("C: PureDistance + nav α=1.2", &train, &graph_c, &test, dim, nq, &gt, k, ef);
    }

    println!("\n=== 结论 ===");
    println!("如果 B recall 暴跌 → PureDistance 不能单独用，需要 nav 层桥接");
    println!("如果 C recall 守住 → 分离架构可行，上 1M 测 avg_visited");
    println!("Glass 参照 (H20 实测): avg_visited=1041, recall~0.95 (ef=50)");
}
