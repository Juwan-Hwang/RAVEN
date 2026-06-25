//! OPT-10: 归一化方案消融实验（siftsmall）
//!
//! 目标：验证 QuantAwareRobustPrune 的 4 种归一化方案对 recall/QPS 的影响
//!
//! 归一化方案：
//!   Mean      - 均值归一化（主方案）
//!   StdDev    - 标准差归一化
//!   Mad       - 中位数绝对偏差
//!   LogSumExp - log-sum-exp 非线性压缩
//!
//! 打分函数：Score = dist / (μ_dist + ε) + β × error / (μ_error + ε)
//!
//! 实验矩阵：4 方案 × β ∈ {0.0, 0.3, 1.0, 2.0}
//!   β=0 时归一化方案不影响（error 项被消除），只测一次作为基线
//!
//! 之前 SIFT1M β 消融显示 β 无收益，本实验验证 siftsmall 上是否同样无收益，
//! 以及归一化方案能否改变这一结论。

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap()));
        }
    }
    (vectors, dim, n)
}

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            gt.push(i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap()));
        }
    }
    (gt, dim, n)
}

/// f32 搜索，返回 (recall@10, qps, avg_degree)
fn eval_f32(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    gt_stride: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    k: usize,
) -> (f64, f64, f64) {
    let avg_deg = graph.degree_stats().mean_degree;
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let mut hits = 0usize;
    let t0 = Instant::now();
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
    let elapsed = t0.elapsed().as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    let qps = nq as f64 / elapsed;
    (recall, qps, avg_deg)
}

fn main() {
    println!("=== OPT-10: 归一化方案消融实验（siftsmall）===");
    println!("目标: 验证 4 种归一化方案 × β 对 recall/QPS 的影响");
    println!();

    // 1. 加载数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("数据加载: {:.2}s", t0.elapsed().as_secs_f64());
    println!("siftsmall: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 归一化到 [0,1]（SIFT 数据 0-255，防梯度爆炸）
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;

    // 2. AVQ 训练（用 base 作为 learn 数据，siftsmall 10K 足够）
    println!("=== AVQ 训练（base 10K, K=256, sub_dim=8, α=0.30, iter=5）===");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 训练: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 预计算节点量化误差
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train))
        .collect();
    println!("节点量化误差预计算: {:.2}s", t0.elapsed().as_secs_f64());
    println!("误差统计: min={:.6}, max={:.6}, mean={:.6}",
        node_errors.iter().cloned().fold(f32::INFINITY, f32::min),
        node_errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        node_errors.iter().sum::<f32>() / node_errors.len() as f32,
    );
    println!();

    // 4. 建图配置（固定）
    let build_config = VamanaBuildConfig {
        alpha: 1.0,  // siftsmall 最优 α=1.0（memory 记录）
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };

    // 5. 实验矩阵
    let schemes = [
        (NormalizationScheme::Mean, "Mean"),
        (NormalizationScheme::StdDev, "StdDev"),
        (NormalizationScheme::Mad, "Mad"),
        (NormalizationScheme::LogSumExp, "LogSumExp"),
    ];
    let betas = [0.0f32, 0.3, 1.0, 2.0];

    println!("=== 消融结果 ===");
    println!("{:>12} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "scheme", "beta", "recall@10", "QPS", "avg_deg", "build_s");
    println!("{:-<62}", "");

    // β=0 基线（标准 RobustPrune，与归一化方案无关，只测一次）
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph_base = VamanaGraph::build(&train, dim, &build_config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    let (recall_base, qps_base, deg_base) = eval_f32(
        &train, &test, &gt, dim, nq, gt_stride, &graph_base, ef_search, k,
    );
    println!("{:>12} {:>6.1} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
        "Baseline", 0.0, recall_base, qps_base, deg_base, build_time);

    // β>0 扫描
    let mut best_recall = recall_base;
    let mut best_combo = "Baseline β=0".to_string();

    for (scheme, name) in &schemes {
        for &beta in &betas[1..] {  // 跳过 β=0（已测基线）
            let mut rng = ChaCha8Rng::seed_from(42);
            let qa_config = QuantAwarePruneConfig {
                alpha: 1.0,
                beta,
                epsilon: EPSILON,
                r_max: 32,
                normalization: *scheme,
            };
            let ne = &node_errors;
            let t0 = Instant::now();
            let graph = VamanaGraph::build_with_quant_aware_prune(
                &train, dim, &build_config, &qa_config,
                move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
                &mut rng,
            );
            let build_time = t0.elapsed().as_secs_f64();

            let (recall, qps, avg_deg) = eval_f32(
                &train, &test, &gt, dim, nq, gt_stride, &graph, ef_search, k,
            );

            println!("{:>12} {:>6.1} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
                name, beta, recall, qps, avg_deg, build_time);

            if recall > best_recall {
                best_recall = recall;
                best_combo = format!("{:?} β={:.1}", scheme, beta);
            }
        }
    }

    println!();
    println!("=== 结论 ===");
    println!("基线 (β=0): recall={:.4}, QPS={:.0}", recall_base, qps_base);
    println!("最佳组合: {} recall={:.4}", best_combo, best_recall);
    let delta = best_recall - recall_base;
    if delta > 0.005 {
        println!("归一化方案有效: recall 提升 {:.4} (>0.5%)", delta);
    } else if delta > 0.0 {
        println!("归一化方案收益可忽略: recall 提升 {:.4} (<0.5%)", delta);
    } else {
        println!("归一化方案无效: recall 未提升（β>0 全部 ≤ β=0 基线）");
    }
}
