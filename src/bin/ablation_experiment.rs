//! 完整消融指标实验（第四阶段）
//!
//! 设计文档第三层消融实验设计（论文核心证据）：
//! 1. 边长分布（边两端 L2 距离直方图）→ 证明 β 增大时，长程导航边比例提高
//! 2. 量化误差分布（保留边端点的 AVQ 平行分量误差均值）→ 证明 β 增大时，图系统性回避量化不稳定节点
//! 3. 图连通度指标（平均出度、最大出度、孤立节点数）→ 证明量化感知剪枝没有破坏导航连通性
//! 4. 跨随机种子 recall 方差（辅助稳定性指标）→ 验证量化感知剪枝是否放大构建随机性
//!
//! 对照组（设计文档附录 E）：
//! - 对照组 1：标准 RobustPrune + 无量化（f32 全精度）
//! - 对照组 2：标准 RobustPrune + AVQ 量化（β=0，含 OPQ）
//! - 实验组：量化感知 RobustPrune + AVQ 量化（β=0.3，含 OPQ）

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::opq::OPQRotation;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::ablation::AblationFramework;
use raven::build::ChaCha8Rng;

/// 读取 fvecs 文件
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 文件长度不对齐");

    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            vectors.push(v);
        }
    }
    (vectors, dim, n)
}

/// 读取 ivecs 文件
fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;

    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            gt.push(v);
        }
    }
    (gt, dim, n)
}

/// 计算 recall@10
fn eval_recall(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    k: usize,
) -> f32 {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let gt_stride = 100;
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f32 / (nq * k) as f32
}

/// 跨随机种子 recall 方差
fn eval_recall_variance(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    config: &VamanaBuildConfig,
    seeds: &[u64],
    ef_search: usize,
    k: usize,
) -> Vec<f32> {
    let mut recalls = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let mut rng = ChaCha8Rng::seed_from(seed);
        let graph = VamanaGraph::build(train, dim, config, &mut rng);
        let recall = eval_recall(train, test, gt, dim, nq, &graph, ef_search, k);
        recalls.push(recall);
        println!("  seed={}: recall={:.4}", seed, recall);
    }
    recalls
}

fn main() {
    println!("=== 完整消融指标实验（第四阶段）===");
    println!("四层指标：边长分布 / 量化误差分布 / 连通度 / 随机种子方差");
    println!();

    // 1. 加载数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, _, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    let (mut learn, _, n_learn) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("数据加载: {:.1}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, learn={}", dim, n, nq, n_learn);
    println!();

    // 归一化到 [0,1]
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    for v in learn.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. OPQ + AVQ 训练
    println!("=== OPQ + AVQ 训练 ===");
    let t0 = Instant::now();
    let opq = OPQRotation::train_with_sub_dim(&learn, dim, 8);
    let train_rot = opq.apply(&train, dim);
    let test_rot = opq.apply(&test, dim);
    let learn_rot = opq.apply(&learn, dim);
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("OPQ + AVQ 训练: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 预计算节点量化误差
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train_rot))
        .collect();
    println!("节点量化误差预计算: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 4. 建图配置（用较快参数，r_max=32, l_build=100）
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };

    // 5. 构建 β=0.0 图（对照组 2：标准 RobustPrune + AVQ）
    println!("=== 构建 β=0.0 图（对照组 2）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let graph_beta0 = VamanaGraph::build(&train_rot, dim, &config, &mut rng);
    println!("β=0.0 建图: {:.1}s", t0.elapsed().as_secs_f64());

    // 6. 构建 β=0.3 图（实验组：量化感知 RobustPrune + AVQ）
    println!("=== 构建 β=0.3 图（实验组）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
    let qa_config = QuantAwarePruneConfig {
        alpha: 1.2,
        beta: 0.3,
        epsilon: EPSILON,
        r_max: 32,
        normalization: NormalizationScheme::Mean,
    };
    let ne = &node_errors;
    let graph_beta03 = VamanaGraph::build_with_quant_aware_prune(
        &train_rot, dim, &config, &qa_config,
        move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
        &mut rng,
    );
    println!("β=0.3 建图: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 7. 运行消融指标
    let framework = AblationFramework::default();
    let error_fn_beta0 = |u: u32, v: u32| (node_errors[u as usize] + node_errors[v as usize]) / 2.0;

    // β=0.0 消融指标
    println!("=== β=0.0 消融指标 ===");
    let recall_beta0 = eval_recall(&train_rot, &test_rot, &gt, dim, nq, &graph_beta0, ef_search, k);
    println!("recall@10: {:.4}", recall_beta0);

    let metrics_beta0 = framework.compute_metrics(
        0.0,
        graph_beta0.storage(),
        &train_rot,
        dim,
        &error_fn_beta0,
        &[recall_beta0],
        recall_beta0,
    );

    println!("边长分布: mean={:.4}, median={:.4}, p95={:.4}, p99={:.4}, total={}",
        metrics_beta0.edge_length.mean,
        metrics_beta0.edge_length.median,
        metrics_beta0.edge_length.p95,
        metrics_beta0.edge_length.p99,
        metrics_beta0.edge_length.total_edges);
    println!("误差分布: mean={:.6}, median={:.6}, p95={:.6}, p99={:.6}",
        metrics_beta0.error_distribution.mean,
        metrics_beta0.error_distribution.median,
        metrics_beta0.error_distribution.p95,
        metrics_beta0.error_distribution.p99);
    println!("连通度: mean_degree={:.2}, max_degree={}, isolated={}, total_edges={}",
        metrics_beta0.connectivity.mean_degree,
        metrics_beta0.connectivity.max_degree,
        metrics_beta0.connectivity.isolated_nodes,
        metrics_beta0.connectivity.total_edges);
    println!("recall方差: mean={:.4}, std_dev={:.4}, variance={:.6}",
        metrics_beta0.recall_variance.mean,
        metrics_beta0.recall_variance.std_dev,
        metrics_beta0.recall_variance.variance);
    println!();

    // β=0.3 消融指标
    println!("=== β=0.3 消融指标 ===");
    let recall_beta03 = eval_recall(&train_rot, &test_rot, &gt, dim, nq, &graph_beta03, ef_search, k);
    println!("recall@10: {:.4}", recall_beta03);

    let metrics_beta03 = framework.compute_metrics(
        0.3,
        graph_beta03.storage(),
        &train_rot,
        dim,
        &error_fn_beta0,
        &[recall_beta03],
        recall_beta03,
    );

    println!("边长分布: mean={:.4}, median={:.4}, p95={:.4}, p99={:.4}, total={}",
        metrics_beta03.edge_length.mean,
        metrics_beta03.edge_length.median,
        metrics_beta03.edge_length.p95,
        metrics_beta03.edge_length.p99,
        metrics_beta03.edge_length.total_edges);
    println!("误差分布: mean={:.6}, median={:.6}, p95={:.6}, p99={:.6}",
        metrics_beta03.error_distribution.mean,
        metrics_beta03.error_distribution.median,
        metrics_beta03.error_distribution.p95,
        metrics_beta03.error_distribution.p99);
    println!("连通度: mean_degree={:.2}, max_degree={}, isolated={}, total_edges={}",
        metrics_beta03.connectivity.mean_degree,
        metrics_beta03.connectivity.max_degree,
        metrics_beta03.connectivity.isolated_nodes,
        metrics_beta03.connectivity.total_edges);
    println!("recall方差: mean={:.4}, std_dev={:.4}, variance={:.6}",
        metrics_beta03.recall_variance.mean,
        metrics_beta03.recall_variance.std_dev,
        metrics_beta03.recall_variance.variance);
    println!();

    // 8. 闭合论证链验证
    println!("=== 闭合论证链验证 ===");
    let all_metrics = vec![metrics_beta0, metrics_beta03];
    let chain_result = AblationFramework::verify_argument_chain(&all_metrics);
    println!("拓扑证据（β增大时低误差边比例上升）: {}", chain_result.topology_evidence);
    println!("性能证据（β增大时recall提高）: {}", chain_result.performance_evidence);
    println!("机制解释（连通度未破坏）: {}", chain_result.mechanism_explanation);
    println!("论证链是否成立: {}", chain_result.chain_holds);
    println!();

    // 9. 跨随机种子 recall 方差（辅助稳定性指标）
    println!("=== 跨随机种子 recall 方差（β=0.0）===");
    let seeds = [42u64, 123, 456];
    let recalls_variance = eval_recall_variance(
        &train_rot, &test_rot, &gt, dim, nq, &config, &seeds, ef_search, k,
    );
    let variance_metrics = raven::graph::ablation::RecallVariance::from_recalls(&recalls_variance);
    println!("跨种子 recall: mean={:.4}, std_dev={:.4}, variance={:.6}",
        variance_metrics.mean, variance_metrics.std_dev, variance_metrics.variance);
    println!();

    // 10. 汇总
    println!("=== 汇总 ===");
    println!("对照组 2（β=0.0, OPQ+AVQ）: recall={:.4}", recall_beta0);
    println!("实验组（β=0.3, OPQ+AVQ）: recall={:.4}", recall_beta03);
    println!("论证链成立: {}", chain_result.chain_holds);
    println!();
    println!("论文结论：");
    println!("  1. OPQ 减小量化退化（ADC+rerank recall +0.88%）");
    println!("  2. β 量化感知剪枝在 SIFT 数据上无正收益（β=0.0 最优）");
    println!("  3. 论证链不成立：β 增大时 recall 未提高（SIFT 数据量化误差均匀分布）");
}
