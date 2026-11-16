//! 入度分布诊断
//!
//! 核心问题：R=40 图中入度分布是否严重不均？
//!   - 有多少节点入度为 0 或个位数？（"孤岛"节点）
//!   - 低入度节点是否集中在高 avg_visited 查询路径上？
//!
//! 用法：cargo run --release --bin indegree_check [subset_n]

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{
    GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
};
use raven::quant::SQ4Dataset;

fn read_fvecs(path: &str, n_limit: Option<usize>) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n_full = bytes.len() / record_bytes;
    let n = n_limit.unwrap_or(n_full).min(n_full);

    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
    }
    (vectors, dim, n)
}

fn brute_force_knn(
    queries: &[f32],
    base: &[f32],
    dim: usize,
    nq: usize,
    nb: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    let mut results = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let mut dists: Vec<(f32, u32)> = (0..nb)
            .map(|i| {
                let v = &base[i * dim..(i + 1) * dim];
                (l2_simd(query, v), i as u32)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.push(dists[..k].iter().map(|(_, id)| *id).collect());
    }
    results
}

fn compute_recall(results: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    let mut hits = 0;
    for q in 0..results.len().min(gt.len()) {
        for &g in &gt[q][..k.min(gt[q].len())] {
            if results[q].contains(&g) {
                hits += 1;
            }
        }
    }
    hits as f64 / (results.len() * k) as f64
}

fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (p * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subset_n: usize = if args.len() > 1 {
        args[1].parse().expect("subset_n must be a number")
    } else {
        100000
    };

    println!("=== 入度分布诊断 ===\n");
    println!("数据: {} 万节点子集", subset_n / 10000);

    // 读取数据
    let (mut train, dim, _n_full) = read_fvecs("data/sift/sift_base.fvecs", Some(subset_n));
    let (mut test_full, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs", None);

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test_full.iter_mut() {
        *v /= 255.0;
    }

    let n = train.len() / dim;
    let nq = 100.min(nq_full);
    let test = &test_full[..nq * dim];

    // GT
    println!("重新计算 ground truth...");
    let t0 = Instant::now();
    let gt = brute_force_knn(test, &train, dim, nq, n, 100);
    println!("GT 计算完成: {:.1}s", t0.elapsed().as_secs_f64());

    // 建图
    println!("\n--- 建图 ---");
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

    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.1}s", t0.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    //  Part 1: 入度分布统计
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Part 1: 入度分布统计 ---");
    let t0 = Instant::now();

    let mut in_degrees = vec![0usize; n];
    let mut out_degrees = vec![0usize; n];

    for node in 0..n as u32 {
        let neighbors = graph.neighbors(node);
        out_degrees[node as usize] = neighbors.len();
        for &nbr in neighbors {
            in_degrees[nbr as usize] += 1;
        }
    }

    // 统计
    let mut in_sorted = in_degrees.clone();
    in_sorted.sort_unstable();
    let mut out_sorted = out_degrees.clone();
    out_sorted.sort_unstable();

    let total_in: usize = in_degrees.iter().sum();
    let total_out: usize = out_degrees.iter().sum();

    println!("分析时间: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n出度统计（R=40 上限）:");
    println!("  总边数: {}", total_out);
    println!("  平均出度: {:.2}", total_out as f64 / n as f64);
    println!("  p50: {}", percentile(&out_sorted, 0.50));
    println!("  p95: {}", percentile(&out_sorted, 0.95));
    println!("  p99: {}", percentile(&out_sorted, 0.99));
    println!("  max: {}", out_sorted.last().unwrap());

    println!("\n入度统计:");
    println!("  总边数: {} (应等于出度总和)", total_in);
    println!("  平均入度: {:.2}", total_in as f64 / n as f64);
    println!("  p50: {}", percentile(&in_sorted, 0.50));
    println!("  p95: {}", percentile(&in_sorted, 0.95));
    println!("  p99: {}", percentile(&in_sorted, 0.99));
    println!("  max: {}", in_sorted.last().unwrap());

    // 入度分布直方图
    println!("\n入度分布直方图:");
    println!("  入度范围    | 节点数  | 占比");
    println!("  ------------|---------|--------");

    let buckets: &[(usize, usize, &str)] = &[
        (0, 0, "入度 = 0 (孤岛)"),
        (1, 1, "入度 = 1"),
        (2, 3, "入度 2-3"),
        (4, 5, "入度 4-5"),
        (6, 10, "入度 6-10"),
        (11, 20, "入度 11-20"),
        (21, 30, "入度 21-30"),
        (31, 40, "入度 31-40"),
        (41, 60, "入度 41-60"),
        (61, 80, "入度 61-80"),
        (81, 120, "入度 81-120"),
        (121, 200, "入度 121-200"),
        (201, usize::MAX, "入度 > 200"),
    ];

    for &(lo, hi, label) in buckets {
        let count = in_degrees
            .iter()
            .filter(|&&d| d >= lo && (hi == usize::MAX || d <= hi))
            .count();
        let pct = count as f64 / n as f64 * 100.0;
        if count > 0 {
            println!("  {:>12} | {:>7} | {:>5.1}%", label, count, pct);
        }
    }

    // 入度 = 0 的节点数（关键指标）
    let zero_in = in_degrees.iter().filter(|&&d| d == 0).count();
    let low_in = in_degrees.iter().filter(|&&d| d <= 3).count();
    let high_in = in_degrees.iter().filter(|&&d| d > 80).count();
    println!("\n关键指标:");
    println!("  入度 = 0 (完全孤岛): {} 节点 ({:.1}%)", zero_in, zero_in as f64 / n as f64 * 100.0);
    println!("  入度 ≤ 3 (低入度): {} 节点 ({:.1}%)", low_in, low_in as f64 / n as f64 * 100.0);
    println!("  入度 > 80 (热点): {} 节点 ({:.1}%)", high_in, high_in as f64 / n as f64 * 100.0);

    // Gini 系数（衡量分布不均匀程度）
    let mut gini_sum = 0i64;
    for i in 0..n {
        gini_sum += in_sorted[i] as i64 * (2 * (i + 1) as i64 - n as i64 - 1);
    }
    let gini = gini_sum as f64 / (n as f64 * total_in as f64);
    println!("  Gini 系数: {:.4} (0=完全均匀, 1=完全不均匀)", gini);

    // ═══════════════════════════════════════════════════════════════
    //  Part 2: 低入度节点与查询路径的关联
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- Part 2: 低入度节点与查询路径关联 ---");

    let sq4 = SQ4Dataset::build(&train, dim);

    // 用 ef=50 跑 100 个查询，记录每个查询的 visited 集合
    let ef = 50;
    let k = 10;

    let mut searcher = GraphSearcher::new(&train, &graph, ef);
    searcher.with_prefetch_offset(16);
    searcher.with_rerank_factor(5);
    searcher.with_sq4(&sq4);

    let mut query_results = Vec::with_capacity(nq);
    let mut query_visited: Vec<Vec<u32>> = Vec::with_capacity(nq);
    let mut total_visited = 0usize;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let r = searcher.search_sq4(query, k);
        total_visited += searcher.last_visited_count();

        // 记录 visited 节点（通过 last_visited_count 不够，需要路径追踪）
        // 我们用一个变通方法：记录结果 ID 和 visited count
        query_results.push(r.iter().map(|(id, _)| *id).collect::<Vec<u32>>());
        query_visited.push(Vec::new()); // 占位
    }

    let recall = compute_recall(&query_results, &gt, k);
    let avg_visited = total_visited as f64 / nq as f64;
    let qps = nq as f64 / t0.elapsed().as_secs_f64();

    println!("ef={}: recall={:.4}, QPS={:.0}, avg_visited={:.1}", ef, recall, qps, avg_visited);

    // Part 2b: 对结果节点（top-k）统计它们的入度
    println!("\n结果节点 (top-{}) 入度分布:", k);
    let mut result_in_degrees: Vec<usize> = Vec::new();
    for result in &query_results {
        for &id in result {
            result_in_degrees.push(in_degrees[id as usize]);
        }
    }
    result_in_degrees.sort_unstable();
    println!("  平均入度: {:.1}", result_in_degrees.iter().sum::<usize>() as f64 / result_in_degrees.len() as f64);
    println!("  p50: {}", percentile(&result_in_degrees, 0.50));
    println!("  p95: {}", percentile(&result_in_degrees, 0.95));
    println!("  min: {}", result_in_degrees.first().unwrap());
    println!("  max: {}", result_in_degrees.last().unwrap());

    // Part 2c: 对比——高入度 vs 低入度节点的"被搜索命中率"
    // 统计每个节点在 100 个查询中被访问的次数（近似：通过 visited count 估算）
    // 由于我们无法直接拿到 visited 列表，改用另一个角度：
    // 看低入度节点在 top-k 结果中的比例 vs 高入度节点在 top-k 结果中的比例
    let low_in_ids: std::collections::HashSet<u32> = (0..n as u32)
        .filter(|&i| in_degrees[i as usize] <= 3)
        .collect();
    let high_in_ids: std::collections::HashSet<u32> = (0..n as u32)
        .filter(|&i| in_degrees[i as usize] > 80)
        .collect();

    let low_in_in_results: usize = query_results
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&id| low_in_ids.contains(&id))
        .count();
    let high_in_in_results: usize = query_results
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&id| high_in_ids.contains(&id))
        .count();

    println!("\n低入度 (≤3) 节点在 top-k 结果中的比例: {}/{} ({:.1}%)",
        low_in_in_results, nq * k, low_in_in_results as f64 / (nq * k) as f64 * 100.0);
    println!("高入度 (>80) 节点在 top-k 结果中的比例: {}/{} ({:.1}%)",
        high_in_in_results, nq * k, high_in_in_results as f64 / (nq * k) as f64 * 100.0);

    // Part 2d: 实际追踪搜索路径
    // 重新跑一遍，这次用 search (非 SQ4) 来拿到 visited 标记
    // 用 VisitedTracker 的 history 来获取实际访问的节点
    println!("\n--- Part 2b: 搜索路径追踪 ---");

    // 重新搜索并追踪 visited 节点
    let mut visited_counts: HashMap<u32, usize> = HashMap::new();
    let mut searcher2 = GraphSearcher::new(&train, &graph, ef);
    searcher2.with_prefetch_offset(16);
    searcher2.with_rerank_factor(5);

    let mut total_visited2 = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher2.search(query, k);
        let visited = searcher2.last_visited_count();
        total_visited2 += visited;
    }

    // 我们无法直接拿到 visited 节点列表（API 限制），
    // 但可以用另一个方法：检查 top-k 结果节点的入度分布
    // 如果低入度节点很少出现在 top-k 中，说明搜索确实难以到达它们

    // 另一个角度：暴力搜索的 top-100 GT 中有多少是低入度节点？
    let mut gt_low_in = 0;
    let mut gt_high_in = 0;
    for q in 0..nq {
        for &gt_id in &gt[q][..10] {
            if in_degrees[gt_id as usize] <= 3 {
                gt_low_in += 1;
            }
            if in_degrees[gt_id as usize] > 80 {
                gt_high_in += 1;
            }
        }
    }
    println!("暴力 GT top-10 中低入度(≤3)节点: {}/{} ({:.1}%)",
        gt_low_in, nq * 10, gt_low_in as f64 / (nq * 10) as f64 * 100.0);
    println!("暴力 GT top-10 中高入度(>80)节点: {}/{} ({:.1}%)",
        gt_high_in, nq * 10, gt_high_in as f64 / (nq * 10) as f64 * 100.0);

    // 如果低入度节点在 GT top-10 中占一定比例，但在搜索结果 top-10 中占比更低
    // → 说明搜索确实"错过"了低入度节点
    if gt_low_in > 0 {
        let recall_gap = 1.0 - low_in_in_results as f64 / gt_low_in as f64;
        println!("\n低入度节点召回率: {}/{} = {:.1}% (差距: {:.1}%)",
            low_in_in_results, gt_low_in,
            low_in_in_results as f64 / gt_low_in as f64 * 100.0,
            recall_gap * 100.0);
    } else {
        println!("\n暴力 GT top-10 中没有低入度(≤3)节点 — 低入度节点不是正确答案");
    }

    println!("\n=== 诊断完成 ===");
}