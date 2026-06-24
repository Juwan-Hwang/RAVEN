//! Week 7：β/α 协同调参 — 阶段二
//!
//! 固定 Vamana α=1.0（Week 5 先验最优），AVQ α=0.30（Week 6 最优）
//! 扫描 β=0/0.1/0.3/0.5/1.0（量化感知 RobustPrune 权重）
//!
//! β=0：标准 RobustPrune（对照组）
//! β>0：量化感知剪枝，回避量化误差大的边

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
use raven::build::ChaCha8Rng;
use raven::l2_simd;

/// 读取 fvecs 文件（siftsmall 格式）
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
    let record_size = 4 + 128 * 4;
    let n = bytes.len() / record_size;
    let mut vectors = vec![0.0f32; n * 128];
    for i in 0..n {
        let offset = i * record_size;
        let dim = i32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        assert_eq!(dim, 128, "维度不是 128");
        for d in 0..128 {
            let off = offset + 4 + d * 4;
            vectors[i * 128 + d] = f32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]
            ]);
        }
    }
    (vectors, 128, n)
}

/// 读取 ivecs 文件（groundtruth 格式）
fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");
    let record_size = 4 + 100 * 4;
    let n = bytes.len() / record_size;
    let mut gt = vec![0i32; n * 100];
    for i in 0..n {
        let offset = i * record_size;
        let dim = i32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        assert_eq!(dim, 100, "groundtruth 维度不是 100");
        for d in 0..100 {
            let off = offset + 4 + d * 4;
            gt[i * 100 + d] = i32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]
            ]);
        }
    }
    (gt, 100, n)
}

/// ADC recall（无 rerank）
fn adc_recall(
    codebook: &AVQCodebook,
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
    graph: &VamanaGraph,
) -> f64 {
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    let mut searcher = GraphSearcher::new(&quantized_db, graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100;
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
    hits as f64 / (nq * k) as f64
}

fn main() {
    println!("=== Week 7：β/α 协同调参 — QPS + recall ===");
    println!();

    // 加载 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, _, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}", dim, n, nq);

    // 归一化到 [0,1]
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    println!("数据归一化: /255 → [0,1]");
    println!();

    // 训练 AVQ codebook（AVQ α=0.30, K=256, sub_dim=8）
    let sub_dim = 8;
    let k = 256;
    let avq_alpha = 0.30;
    println!("训练 AVQ codebook (AVQ α={}, K={}, sub_dim={})...", avq_alpha, k, sub_dim);
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let codebook = AVQCodebook::train_full(
        &train, dim, k, TrainingSignal::BatchHighScorePairs, 25, sub_dim, avq_alpha, avq_rng.inner(),
    );
    println!("codebook 训练完成");
    println!();

    // 量化数据库（ADC 用）
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 扫描 Vamana α × β
    for vamana_alpha in [1.0, 1.2] {
        let build_config = VamanaBuildConfig {
            alpha: vamana_alpha,
            l_build: 100,
            r_max: 32,
            r_soft: 48,
            max_iterations: 1,
        };

        println!("=== Vamana α={}, AVQ α={} ===", vamana_alpha, avq_alpha);
        println!("{:>8} {:>12} {:>12} {:>12} {:>10}",
            "beta", "adc_recall", "rerank_recall", "qps", "avg_degree");

        for beta in [0.0, 0.1, 0.3, 0.5, 1.0] {
            let mut rng = ChaCha8Rng::seed_from(42);
            let graph = if beta == 0.0 {
                VamanaGraph::build(&train, dim, &build_config, &mut rng)
            } else {
                let qa_config = QuantAwarePruneConfig {
                    alpha: vamana_alpha,
                    beta,
                    epsilon: EPSILON,
                    r_max: 32,
                    normalization: NormalizationScheme::Mean,
                };
                let cb_ref = &codebook;
                let train_ref = &train;
                VamanaGraph::build_with_quant_aware_prune(
                    &train, dim, &build_config, &qa_config,
                    |u, v| cb_ref.edge_error(u, v, train_ref),
                    &mut rng,
                )
            };

            // 计算 avg degree
            let avg_deg = graph.degree_stats().mean_degree;

            // ADC recall
            let recall_adc = adc_recall(&codebook, &train, &test, &gt, dim, n, nq, 10, &graph);

            // QPS + rerank recall（计时）
            let mut searcher = GraphSearcher::new(&quantized_db, &graph, 100);
            let gt_stride = 100;
            let top_n = 100;
            let k_ret = 10;
            let mut hits = 0usize;
            let start = Instant::now();
            for _warmup in 0..3 { // 3 轮预热
                for q in 0..nq {
                    let query = &test[q * dim..(q + 1) * dim];
                    let _ = searcher.search(query, top_n);
                }
            }
            let start = Instant::now();
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let candidates = searcher.search(query, top_n);
                // rerank
                let mut reranked: Vec<(u32, f32)> = candidates
                    .iter()
                    .map(|(id, _)| {
                        let v = &train[*id as usize * dim..(*id as usize + 1) * dim];
                        (*id, l2_simd(query, v))
                    })
                    .collect();
                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let found: Vec<u32> = reranked.iter().take(k_ret).map(|(id, _)| *id).collect();
                let gt_slice = &gt[q * gt_stride..q * gt_stride + k_ret];
                for &g in gt_slice {
                    if found.contains(&(g as u32)) {
                        hits += 1;
                    }
                }
            }
            let elapsed = start.elapsed();
            let qps = nq as f64 / elapsed.as_secs_f64();
            let recall_rerank = hits as f64 / (nq * k_ret) as f64;

            println!("{:>8.2} {:>12.4} {:>12.4} {:>10.0} {:>10.1}",
                beta, recall_adc, recall_rerank, qps, avg_deg);
        }
        println!();
    }

    println!("=== 结论 ===");
    println!("目标：recall@10 > 0.95 + QPS 对比");
    println!("β 影响判断：QPS 差距 < 5% → β 作用不显著，推荐 β=0.1");
}

