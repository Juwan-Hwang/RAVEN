//! 附录 A 退化判定实验
//!
//! 设计文档附录 A：RP-Tuning 额外存储方案三选一
//! 退化判定阈值（实验前锁定）：
//!   指标一（固定 QPS）：A 方案 recall@10 差距 < 0.5%
//!   指标二（固定 recall）：A 方案 QPS 差距 < 3%
//!   覆盖范围：至少 3 个不同数据集，每个独立判定
//!
//! 实验设计：
//!   1. 完整重建版本（baseline）：分别用 α=1.0/1.5/2.0 从零构建
//!   2. A 方案：建一次 α=1.2 基础图，RP-Tuning 后验生成 α=1.0/1.5/2.0 变体
//!   3. 对比同等 QPS 下的 recall@10 差距

use std::time::Instant;
use rand::Rng;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::rp_tuning::{RPTuning, RPTuningConfig, RPTuningStorageScheme};
use raven::build::ChaCha8Rng;

/// 实验结果
#[derive(Debug, Clone)]
struct ExperimentResult {
    alpha: f32,
    method: String,  // "rebuild" or "rp_tuning_A"
    build_time_s: f64,
    qps: f64,
    recall: f64,
}

/// 生成带聚类结构的合成数据集（模拟真实分布）
fn generate_clustered_data(
    n: usize,
    dim: usize,
    nq: usize,
    k: usize,
    n_clusters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from(seed);
    let mut train = vec![0.0f32; n * dim];
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);

    // 生成聚类中心
    for _ in 0..n_clusters {
        let c: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0).collect();
        centroids.push(c);
    }

    // 每个向量 = 聚类中心 + 高斯噪声
    for i in 0..n {
        let cluster = i % n_clusters;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            train[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    // 生成查询向量
    let mut test = vec![0.0f32; nq * dim];
    for i in 0..nq {
        let cluster = (i % n_clusters) as usize;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            test[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    // 暴力计算 ground truth
    let mut gt = vec![0i32; nq * k];
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &train[i * dim..(i + 1) * dim];
                let d: f32 = v.iter().zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for j in 0..k {
            gt[q * k + j] = dists[j].0 as i32;
        }
    }

    (train, test, gt)
}

/// 运行查询并计算 recall
fn run_queries(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    k: usize,
    ef_search: usize,
) -> (f64, f64) {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let start = Instant::now();
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * k..(q + 1) * k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    let elapsed = start.elapsed();
    let qps = nq as f64 / elapsed.as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    (qps, recall)
}

fn main() {
    println!("=== 附录 A 退化判定实验 ===");
    println!("设计文档附录 A：RP-Tuning 额外存储方案三选一");
    println!("退化判定阈值：recall@10 差距 < 0.5%, QPS 差距 < 3%");
    println!();

    // 3 个数据集（不同规模/维度，满足"至少 3 个不同数据集"要求）
    let datasets = [
        ("dataset_1", 1000usize, 128usize, 100usize, 10usize, 20),
        ("dataset_2", 2000, 256, 100, 10, 30),
        ("dataset_3", 3000, 128, 100, 10, 40),
    ];

    let mut all_pass = true;

    for (name, n, dim, nq, k, n_clusters) in &datasets {
        println!("=== {} (n={}, dim={}, nq={}, k={}, clusters={}) ===", name, n, dim, nq, k, n_clusters);
        let (train, test, gt) = generate_clustered_data(*n, *dim, *nq, *k, *n_clusters, 42);

        let alpha_points = vec![1.0f32, 1.5, 2.0];
        let ef_search = 100;

        // === Baseline: 完整重建 ===
        println!("[Baseline] 完整重建（分别用不同 α 从零构建）");
        let mut rebuild_results: Vec<ExperimentResult> = Vec::new();
        for &alpha in &alpha_points {
            let mut rng = ChaCha8Rng::seed_from(42);
            let config = VamanaBuildConfig {
                alpha,
                l_build: 200,
                r_max: 64,
                r_soft: 96,
                max_iterations: 1,
            };
            let start = Instant::now();
            let graph = VamanaGraph::build(&train, *dim, &config, &mut rng);
            let build_time = start.elapsed().as_secs_f64();

            let (qps, recall) = run_queries(&train, &graph, &test, &gt, *dim, *nq, *k, ef_search);
            println!("  α={:.1}: build={:.3}s, QPS={:.0}, recall@{}={:.4}",
                alpha, build_time, qps, k, recall);
            rebuild_results.push(ExperimentResult {
                alpha,
                method: "rebuild".to_string(),
                build_time_s: build_time,
                qps,
                recall,
            });
        }

        // === A 方案：RP-Tuning ===
        println!("[Scheme A] RP-Tuning（建一次 α=1.2 基础图，后验生成变体）");
        let mut rng = ChaCha8Rng::seed_from(42);
        let base_config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 200,
            r_max: 64,
            r_soft: 96,
            max_iterations: 1,
        };
        let start = Instant::now();
        let base_graph = VamanaGraph::build(&train, *dim, &base_config, &mut rng);
        let base_build_time = start.elapsed().as_secs_f64();
        println!("  base graph (α=1.2): build={:.3}s", base_build_time);

        let rp_config = RPTuningConfig {
            scheme: RPTuningStorageScheme::SchemeA,
            alpha_points: alpha_points.clone(),
            r_max: 64,
        };
        let rp_start = Instant::now();
        let variants = RPTuning::generate_variants(&base_graph, &train, *dim, &rp_config);
        let rp_time = rp_start.elapsed().as_secs_f64();
        println!("  RP-Tuning generate_variants: {:.3}s ({} variants)", rp_time, variants.len());

        let mut rp_results: Vec<ExperimentResult> = Vec::new();
        for variant in &variants {
            let graph = variant.clone().into_graph(*dim);
            let stats = graph.degree_stats();
            println!("  α={:.1}: degree stats: mean={:.1}, p95={}, p99={}, max={}, isolated={}, overflow_ratio={:.4}",
                variant.alpha, stats.mean_degree, stats.p95_degree, stats.p99_degree,
                stats.max_degree, stats.isolated_nodes, stats.overflow_ratio);
            let (qps, recall) = run_queries(&train, &graph, &test, &gt, *dim, *nq, *k, ef_search);
            println!("  α={:.1}: QPS={:.0}, recall@{}={:.4}",
                variant.alpha, qps, k, recall);
            rp_results.push(ExperimentResult {
                alpha: variant.alpha,
                method: "rp_tuning_A".to_string(),
                build_time_s: base_build_time + rp_time,
                qps,
                recall,
            });
        }

        // === 退化判定 ===
        println!("[退化判定]");
        let mut dataset_pass = true;
        for i in 0..alpha_points.len() {
            let rebuild = &rebuild_results[i];
            let rp = &rp_results[i];

            // recall 退化：只在 Scheme A recall 低于 baseline 时算退化
            // Scheme A recall 更高是优势，不算退化
            let recall_drop = if rp.recall < rebuild.recall {
                rebuild.recall - rp.recall
            } else {
                0.0  // Scheme A 更好，不算退化
            };

            // QPS 退化：只在 Scheme A QPS 低于 baseline 时算退化
            let qps_drop = if rp.qps < rebuild.qps {
                (rebuild.qps - rp.qps) / rebuild.qps
            } else {
                0.0  // Scheme A 更快，不算退化
            };

            let recall_ok = recall_drop < 0.005;  // recall 下降 < 0.5%
            let qps_ok = qps_drop < 0.03;  // QPS 下降 < 3%

            println!("  α={:.1}: recall_drop={:.4} ({}), qps_drop={:.2}%, {}",
                rebuild.alpha,
                recall_drop,
                if recall_ok { "PASS" } else { "FAIL" },
                qps_drop * 100.0,
                if qps_ok { "PASS" } else { "FAIL" }
            );

            if !recall_ok || !qps_ok {
                dataset_pass = false;
            }
        }

        if dataset_pass {
            println!("  → {} PASS（A 方案不退化）", name);
        } else {
            println!("  → {} FAIL（A 方案退化，需降级 B）", name);
            all_pass = false;
        }
        println!();
    }

    // === 最终结论 ===
    println!("=== 最终结论 ===");
    if all_pass {
        println!("所有数据集均通过退化判定");
        println!("→ 选定方案 A：zero-cost RP-Tuning");
        println!("→ 论文亮点：RP-Tuning 无额外存储代价，不退化");
    } else {
        println!("存在数据集未通过退化判定");
        println!("→ 降级方案 B：每节点存储被剪掉的邻居 ID");
    }
}
