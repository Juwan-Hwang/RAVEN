//! 附录 B：AVQ 训练信号双分支消融实验
//!
//! 设计文档附录 B：AVQ 训练信号的具体实现（Week 5-6 决策）
//! 两个选项都做最小版本，不提前拍板：
//!   选项一：批次内高分对采样（100行以内）
//!   选项二：预采样近邻对（复用已有粗量化构建）
//!
//! 对比指标：
//!   1. 同等训练时间下的 retrieval-aware loss
//!   2. 下游 recall@10 差异
//!
//! Week 5 末看数据，哪个 recall 更高选哪个，同时在论文里报告两个结果。

use std::time::Instant;
use rand::Rng;
use raven::quant::avq::{AVQCodebook, QuantizationMode, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

/// 生成带聚类结构的合成数据集
fn generate_data(n: usize, dim: usize, nq: usize, k: usize, n_clusters: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from(seed);
    let mut train = vec![0.0f32; n * dim];
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);

    for _ in 0..n_clusters {
        let c: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0).collect();
        centroids.push(c);
    }

    for i in 0..n {
        let cluster = i % n_clusters;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            train[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    let mut test = vec![0.0f32; nq * dim];
    for i in 0..nq {
        let cluster = (i % n_clusters) as usize;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            test[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

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

/// 计算量化后的 recall@10（用量化向量重建图）
fn quantized_recall(
    codebook: &AVQCodebook,
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
) -> (f64, f64) {
    // 量化所有训练向量
    let quantized: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 用量化向量构建图
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_max: 32,
        r_soft: 48,
        max_iterations: 1,
    };
    let graph = VamanaGraph::build(&quantized, dim, &config, &mut rng);

    // 用量化向量查询
    let searcher = GraphSearcher::new(&quantized, &graph, 100);
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
    (recall, qps)
}

fn main() {
    println!("=== 附录 B：AVQ 训练信号双分支消融实验 ===");
    println!("设计文档附录 B：Week 5 同时实现最小双分支，用消融数据决定主线");
    println!();

    // 3 个数据集
    let datasets = [
        ("dataset_1", 500usize, 64usize, 50usize, 10usize, 10),
        ("dataset_2", 1000, 128, 100, 10, 20),
        ("dataset_3", 2000, 128, 100, 10, 30),
    ];

    for (name, n, dim, nq, k, n_clusters) in &datasets {
        println!("=== {} (n={}, dim={}, nq={}, k={}, clusters={}) ===", name, n, dim, nq, k, n_clusters);
        let (train, test, gt) = generate_data(*n, *dim, *nq, *k, *n_clusters, 42);

        // 选项一：批次内高分对采样
        let t1_start = Instant::now();
        let cb1 = AVQCodebook::train_with_signal(&train, *dim, 16, QuantizationMode::Avq, TrainingSignal::BatchHighScorePairs);
        let t1_train = t1_start.elapsed();
        let loss1 = cb1.retrieval_aware_loss(&train);
        let (recall1, qps1) = quantized_recall(&cb1, &train, &test, &gt, *dim, *n, *nq, *k);

        // 选项二：预采样近邻对
        let t2_start = Instant::now();
        let cb2 = AVQCodebook::train_with_signal(&train, *dim, 16, QuantizationMode::Avq, TrainingSignal::PreSampledNeighborPairs);
        let t2_train = t2_start.elapsed();
        let loss2 = cb2.retrieval_aware_loss(&train);
        let (recall2, qps2) = quantized_recall(&cb2, &train, &test, &gt, *dim, *n, *nq, *k);

        println!("[选项一] BatchHighScorePairs:");
        println!("  train_time={:.3}s, retrieval_loss={:.6}, recall@{}={:.4}, QPS={:.0}",
            t1_train.as_secs_f64(), loss1, k, recall1, qps1);
        println!("[选项二] PreSampledNeighborPairs:");
        println!("  train_time={:.3}s, retrieval_loss={:.6}, recall@{}={:.4}, QPS={:.0}",
            t2_train.as_secs_f64(), loss2, k, recall2, qps2);

        // 对比
        let loss_ratio = if loss2 > 0.0 { loss1 / loss2 } else { 1.0 };
        let recall_diff = recall1 - recall2;
        println!("[对比]");
        println!("  loss ratio (option1/option2): {:.3}x ({})", loss_ratio,
            if loss_ratio < 1.0 { "option1 更优" } else { "option2 更优" });
        println!("  recall diff (option1 - option2): {:+.4} ({})", recall_diff,
            if recall_diff > 0.0 { "option1 更优" } else if recall_diff < 0.0 { "option2 更优" } else { "持平" });
        println!();
    }

    println!("=== 最终决策建议 ===");
    println!("看 recall 更高的选项作为主线，同时在论文里报告两个结果");
}
