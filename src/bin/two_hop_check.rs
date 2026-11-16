//! 2-hop 路径冗余验证程序
//!
//! 核心问题：DirectionalPrune 保留下来的边中，有多少被 2-hop 路径覆盖？
//!
//! 2-hop 冗余定义（NGT adjustPathsEffectively）：
//!   对于有向边 u→k，如果存在 u 的邻居 j 满足：
//!     dist(u, j) < dist(u, k)  且  dist(j, k) < dist(u, k)
//!   则 u→k 是 2-hop 冗余的——路径 u→j→k 比直连 u→k 更高效。
//!
//! 用法：
//!   cargo run --release --bin two_hop_check [subset_n]
//!
//!   subset_n: 子集节点数（默认 100000，快速验证）

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{
    GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
};
use raven::memory::HybridBlockedCsr;
use raven::quant::SQ4Dataset;

// ═══════════════════════════════════════════════════════════════════════
//  I/O 工具函数
// ═══════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════
//  2-hop 冗余分析
// ═══════════════════════════════════════════════════════════════════════

struct RedundancyStats {
    total_edges: usize,
    redundant_edges: usize,
    /// 按度数统计的冗余率：[(degree, total_edges, redundant_edges), ...]
    by_degree: Vec<(usize, usize, usize)>,
    /// 按邻居位置（距离排序后）统计的冗余率
    by_position: Vec<(usize, usize, usize)>,
}

/// 分析 2-hop 路径冗余
fn analyze_2hop_redundancy(
    graph: &VamanaGraph,
    vectors: &[f32],
    dim: usize,
) -> RedundancyStats {
    let n_nodes = vectors.len() / dim;

    let mut total_edges = 0usize;
    let mut redundant_edges = 0usize;

    // 按度数统计：degree -> (total, redundant)
    let mut by_degree: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();

    // 按位置统计：position (0-based) -> (total, redundant)
    let mut by_position: Vec<(usize, usize)> = Vec::new();
    let max_degree = 64; // R_max = 64
    by_position.resize(max_degree, (0, 0));

    for node in 0..n_nodes as u32 {
        let neighbors = graph.neighbors(node);
        let deg = neighbors.len();

        if deg == 0 {
            continue;
        }

        // 按距离排序邻居
        let node_vec = &vectors[node as usize * dim..(node as usize + 1) * dim];
        let mut sorted: Vec<(f32, u32)> = neighbors
            .iter()
            .map(|&n| {
                let v = &vectors[n as usize * dim..(n as usize + 1) * dim];
                (l2_simd(node_vec, v), n)
            })
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 对每个邻居 k，检查是否存在 u→j→k 的 2-hop 路径
        for (k_pos, (dist_uk, k)) in sorted.iter().enumerate() {
            let dist_uk = *dist_uk;
            let k = *k;
            total_edges += 1;

            // 记录度数统计
            let entry = by_degree.entry(deg).or_insert((0, 0));
            entry.0 += 1;

            let mut is_redundant = false;

            // 检查是否有 j 满足：j 是 u 的邻居，k 是 j 的邻居，且 dist(u,j) < dist(u,k), dist(j,k) < dist(u,k)
            for (dist_uj, j) in sorted.iter() {
                let dist_uj = *dist_uj;
                let j = *j;

                if j == k {
                    continue;
                }

                // j 必须比 k 更近
                if dist_uj >= dist_uk {
                    continue;
                }

                // 检查 k 是否是 j 的邻居
                let neighbors_j = graph.neighbors(j);
                let j_vec = &vectors[j as usize * dim..(j as usize + 1) * dim];
                let k_vec = &vectors[k as usize * dim..(k as usize + 1) * dim];

                for &nj in neighbors_j {
                    if nj == k {
                        let dist_jk = l2_simd(j_vec, k_vec);

                        // NGT 的判定：两段都比直连短
                        if dist_jk < dist_uk {
                            is_redundant = true;
                            break;
                        }
                    }
                }

                if is_redundant {
                    break;
                }
            }

            if is_redundant {
                redundant_edges += 1;

                // 记录度数统计
                let entry = by_degree.entry(deg).or_insert((0, 0));
                entry.1 += 1;

                // 记录位置统计
                if k_pos < max_degree {
                    by_position[k_pos].0 += 1;
                    by_position[k_pos].1 += 1;
                }
            } else {
                // 记录位置统计（非冗余）
                if k_pos < max_degree {
                    by_position[k_pos].0 += 1;
                }
            }
        }
    }

    // 转换为有序 Vec
    let mut by_degree_sorted: Vec<(usize, usize, usize)> = by_degree
        .into_iter()
        .map(|(d, (tot, red))| (d, tot, red))
        .collect();
    by_degree_sorted.sort_by_key(|&(d, _, _)| d);

    RedundancyStats {
        total_edges,
        redundant_edges,
        by_degree: by_degree_sorted,
        by_position: by_position
            .iter()
            .enumerate()
            .map(|(pos, (tot, red))| (pos + 1, *tot, *red))
            .collect(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  暴力 k-NN（为子集重新计算 ground truth）
// ═══════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════
//  图后处理：移除 2-hop 冗余边
// ═══════════════════════════════════════════════════════════════════════

fn remove_2hop_redundant_edges(
    graph: &VamanaGraph,
    vectors: &[f32],
    dim: usize,
    min_degree: usize,
) -> VamanaGraph {
    let n_nodes = vectors.len() / dim;

    // 创建新的图存储
    let mut new_storage = HybridBlockedCsr::new(n_nodes, 40); // R=40

    for node in 0..n_nodes as u32 {
        let neighbors = graph.neighbors(node);
        if neighbors.is_empty() {
            continue;
        }

        // 按距离排序邻居
        let node_vec = &vectors[node as usize * dim..(node as usize + 1) * dim];
        let mut sorted: Vec<(f32, u32)> = neighbors
            .iter()
            .map(|&n| {
                let v = &vectors[n as usize * dim..(n as usize + 1) * dim];
                (l2_simd(node_vec, v), n)
            })
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 找出非冗余的边
        let mut keep: Vec<u32> = Vec::new();
        let mut is_redundant: Vec<bool> = vec![false; sorted.len()];

        for (k_pos, (dist_uk, k)) in sorted.iter().enumerate() {
            let dist_uk = *dist_uk;
            let k = *k;

            for (dist_uj, j) in sorted.iter() {
                let dist_uj = *dist_uj;
                let j = *j;

                if j == k {
                    continue;
                }

                // j 必须比 k 更近
                if dist_uj >= dist_uk {
                    continue;
                }

                // 检查 k 是否是 j 的邻居
                let neighbors_j = graph.neighbors(j);
                let j_vec = &vectors[j as usize * dim..(j as usize + 1) * dim];
                let k_vec = &vectors[k as usize * dim..(k as usize + 1) * dim];

                for &nj in neighbors_j {
                    if nj == k {
                        let dist_jk = l2_simd(j_vec, k_vec);

                        // NGT 的判定：两段都比直连短
                        if dist_jk < dist_uk {
                            is_redundant[k_pos] = true;
                            break;
                        }
                    }
                }

                if is_redundant[k_pos] {
                    break;
                }
            }
        }

        // 保留非冗余边（从近到远，确保保留最近的）
        let mut non_redundant: Vec<(f32, u32)> = sorted
            .iter()
            .zip(is_redundant.iter())
            .filter(|(_, &red)| !red)
            .map(|(d, _)| *d)
            .collect();
        non_redundant.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 按距离从近到远保留，但确保至少 min_degree 条
        let to_keep = non_redundant.iter().take(min_degree.max(non_redundant.len()));
        for (_, id) in to_keep {
            keep.push(*id);
        }

        // 设置新的邻居列表
        new_storage.set_neighbors(node, &keep);
    }

    // 从新的存储构建 VamanaGraph（entry_point 使用原图的）
    VamanaGraph::from_storage(new_storage, graph.entry_point(), dim)
}

// ═══════════════════════════════════════════════════════════════════════
//  Benchmark
// ═══════════════════════════════════════════════════════════════════════

fn benchmark_graph(
    graph: &VamanaGraph,
    vectors: &[f32],
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
    sq4: &SQ4Dataset,
) -> (f64, f64, f64) {
    let mut searcher = GraphSearcher::new(vectors, graph, ef);
    searcher.with_prefetch_offset(16);
    searcher.with_rerank_factor(5);
    searcher.with_sq4(sq4);

    let mut results = Vec::with_capacity(nq);
    let mut total_visited = 0;
    let t0 = Instant::now();

    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let r = searcher.search_sq4(query, k);
        total_visited += searcher.last_visited_count();
        results.push(r.iter().map(|(id, _)| *id).collect());
    }

    let dt = t0.elapsed();
    let recall = compute_recall(&results, gt, k);
    let qps = nq as f64 / dt.as_secs_f64();
    let avg_visited = total_visited as f64 / nq as f64;

    (recall, qps, avg_visited)
}

// ═══════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subset_n: usize = if args.len() > 1 {
        args[1].parse().expect("subset_n must be a number")
    } else {
        100000 // 默认 10 万
    };

    println!("=== 2-hop 路径冗余验证 ===\n");
    println!("数据: {} 万节点子集", subset_n / 10000);

    // 读取数据
    let (mut train, dim, _n_full) = read_fvecs("data/sift/sift_base.fvecs", Some(subset_n));
    let (mut test_full, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs", None);

    // 归一化
    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test_full.iter_mut() {
        *v /= 255.0;
    }

    let n = train.len() / dim;

    // 读取前 100 个查询（减少暴力 GT 计算时间）
    let nq = 100.min(nq_full);
    let test = &test_full[..nq * dim];

    // 重新计算 ground truth（暴力 k-NN）
    println!("重新计算 ground truth（{} 查询 × {} 基础）...", nq, n);
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
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图时间: {:.1}s", build_time);

    // 编码 SQ4
    let sq4 = SQ4Dataset::build(&train, dim);

    // 2-hop 冗余分析
    println!("\n--- 2-hop 冗余分析 ---");
    let t0 = Instant::now();
    let stats = analyze_2hop_redundancy(&graph, &train, dim);
    let analysis_time = t0.elapsed().as_secs_f64();

    let redundancy_rate = stats.redundant_edges as f64 / stats.total_edges as f64 * 100.0;

    println!("分析时间: {:.1}s", analysis_time);
    println!("总边数: {}", stats.total_edges);
    println!("2-hop 冗余边: {}", stats.redundant_edges);
    println!("冗余率: {:.1}%", redundancy_rate);
    println!("\n按度数统计:");
    println!("  度数 | 总边数 | 冗余边 | 冗余率");
    println!("  -----|--------|--------|--------");
    for (deg, tot, red) in &stats.by_degree {
        let rate = *red as f64 / *tot as f64 * 100.0;
        println!("  {:>4} | {:>6} | {:>6} | {:>5.1}%", deg, tot, red, rate);
    }

    println!("\n按邻居位置统计（距离排序后第 N 个邻居）:");
    println!("  位置 | 总边数 | 冗余边 | 冗余率");
    println!("  ----|--------|--------|--------");
    for (pos, tot, red) in &stats.by_position {
        if *tot > 0 {
            let rate = *red as f64 / *tot as f64 * 100.0;
            println!("  {:>3} | {:>6} | {:>6} | {:>5.1}%", pos, tot, red, rate);
        }
    }

    // 如果冗余率 > 15%，执行 benchmark
    if redundancy_rate > 15.0 {
        println!("\n冗余率 {:.1}% > 15%，执行消融实验...", redundancy_rate);

        // 原图 benchmark
        println!("\n--- 原图 benchmark ---");
        let ef = 50;
        let (recall_orig, qps_orig, avg_visited_orig) =
            benchmark_graph(&graph, &train, test, dim, nq, &gt, 10, ef, &sq4);
        println!("ef={}: recall={:.4}, QPS={:.0}, avg_visited={:.1}", ef, recall_orig, qps_orig, avg_visited_orig);

        // 清理图（移除 2-hop 冗余边）
        println!("\n--- 清理图（移除冗余边）---");
        let min_degree = 10; // 保持最小连通性
        let t0 = Instant::now();
        let cleaned_graph = remove_2hop_redundant_edges(&graph, &train, dim, min_degree);
        let clean_time = t0.elapsed().as_secs_f64();
        println!("清理时间: {:.1}s", clean_time);

        // 计算清理后的平均度数
        let mut total_deg = 0usize;
        for node in 0..n as u32 {
            total_deg += cleaned_graph.neighbors(node).len();
        }
        let avg_deg_cleaned = total_deg as f64 / n as f64;
        println!("清理后平均度数: {:.1}", avg_deg_cleaned);

        // 清理图 benchmark
        println!("\n--- 清理图 benchmark ---");
        let (recall_cleaned, qps_cleaned, avg_visited_cleaned) =
            benchmark_graph(&cleaned_graph, &train, test, dim, nq, &gt, 10, ef, &sq4);
        println!("ef={}: recall={:.4}, QPS={:.0}, avg_visited={:.1}", ef, recall_cleaned, qps_cleaned, avg_visited_cleaned);

        // 对比
        println!("\n--- 对比 ---");
        let delta_qps = (qps_cleaned - qps_orig) / qps_orig * 100.0;
        let delta_visited = (avg_visited_cleaned - avg_visited_orig) / avg_visited_orig * 100.0;
        let delta_recall = recall_cleaned - recall_orig;

        println!("QPS 变化: {:+.1}% ({:.0} → {:.0})", delta_qps, qps_orig, qps_cleaned);
        println!("avg_visited 变化: {:+.1}% ({:.1} → {:.1})", delta_visited, avg_visited_orig, avg_visited_cleaned);
        println!("recall 变化: {:+.4}pp ({:.4} → {:.4})", delta_recall, recall_orig, recall_cleaned);
    } else {
        println!("\n冗余率 {:.1}% ≤ 15%，2-hop 冗余消除无收益，跳过消融实验。", redundancy_rate);
    }

    println!("\n=== 验证完成 ===");
}