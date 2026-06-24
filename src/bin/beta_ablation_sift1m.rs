//! SIFT1M β 消融实验
//!
//! 固定 Vamana α=1.2（RP-Tuning 确认最优），AVQ α=0.30
//! 扫描 β=0.0/0.1/0.3/1.0（量化感知 RobustPrune 权重）
//!
//! β=0.0：标准 RobustPrune（对照组，已有基线）
//! β>0：量化感知剪枝，回避量化误差大的边
//!
//! 目标：验证 QuantAwareRobustPrune 是否能减小 AVQ 量化退化
//! 当前 β=0 退化：f32 0.9528 → AVQ ADC+rerank 0.9228（退化 3%）

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
use raven::build::ChaCha8Rng;
use raven::l2_simd;

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

/// ADC + rerank 搜索，返回 (recall@10, qps, avg_degree)
fn eval_adc_rerank(
    _codebook: &AVQCodebook,
    train: &[f32],
    quantized_db: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    _n: usize,
    nq: usize,
    gt_stride: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    top_n: usize,
    k: usize,
) -> (f64, f64, f64) {
    let avg_deg = graph.degree_stats().mean_degree;

    // ADC 搜索 + rerank
    let searcher = GraphSearcher::new(quantized_db, graph, ef_search);
    let mut hits = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = searcher.search(query, top_n);
        // f32 rerank
        let mut reranked: Vec<(u32, f32)> = candidates
            .iter()
            .map(|(id, _)| {
                let v = &train[*id as usize * dim..(*id as usize + 1) * dim];
                (*id, l2_simd(query, v))
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let found: Vec<u32> = reranked.iter().take(k).map(|(id, _)| *id).collect();
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

/// f32 搜索（无量化），返回 (recall@10, qps)
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
) -> (f64, f64) {
    let searcher = GraphSearcher::new(train, graph, ef_search);
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
    (recall, qps)
}

fn main() {
    println!("=== SIFT1M β 消融实验 ===");
    println!("固定 Vamana α=1.2, AVQ α=0.30, K=256, sub_dim=8");
    println!("扫描 β=0.0/0.1/0.3/1.0");
    println!();

    // 1. 加载数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    let (mut learn, _, n_learn) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("数据加载: {:.1}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}, learn={}", dim, n, nq, gt_nq, gt_k, n_learn);
    println!();

    // 归一化到 [0,1]
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    for v in learn.iter_mut() { *v /= 255.0; }

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;
    let top_n = 100;

    // 2. AVQ 训练（只训练一次，所有 β 共用）
    println!("=== AVQ 训练（sift_learn 100K, K=256, sub_dim=8, α=0.30, iter=5）===");
    let t0 = Instant::now();
    let cb = AVQCodebook::train_full(
        &learn, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30,
    );
    println!("AVQ 训练: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 量化数据库（所有 β 共用同一个 codebook）
    let t0 = Instant::now();
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            cb.decode(&cb.encode(v))
        })
        .collect();
    println!("量化数据库构造: {:.1}s", t0.elapsed().as_secs_f64());

    // 3.5 预计算所有节点的量化误差（避免建图时重复 encode+decode）
    // edge_error(u,v) = mean(node_error(u), node_error(v))
    // 不预计算的话，1M 节点 × ~100 候选 = 1 亿次 encode+decode，建图要数小时
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train))
        .collect();
    println!("节点量化误差预计算: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 4. Vamana 建图配置（固定）
    let build_config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };

    // 5. 扫描 β
    let betas = [0.0f32, 0.1, 0.3, 1.0];

    println!("=== β 消融结果 ===");
    println!("{:>6} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "beta", "f32_recall", "f32_qps", "adc_rerank", "adc_qps", "degrad", "avg_deg");
    println!("{:-<82}", "");

    // f32 基线 recall（β=0 的图，用于对比量化退化）
    let mut f32_baseline_recall = 0.0f64;

    for &beta in &betas {
        let mut rng = ChaCha8Rng::seed_from(42);

        let t0 = Instant::now();
        let graph = if beta == 0.0 {
            println!("[β={:.1}] 建图（标准 RobustPrune）...", beta);
            VamanaGraph::build(&train, dim, &build_config, &mut rng)
        } else {
            println!("[β={:.1}] 建图（量化感知 RobustPrune）...", beta);
            let qa_config = QuantAwarePruneConfig {
                alpha: 1.2,
                beta,
                epsilon: EPSILON,
                r_max: 32,
                normalization: NormalizationScheme::Mean,
            };
            // 用预计算的 node_errors，O(1) 查表替代 O(dim) encode+decode
            let ne = &node_errors;
            VamanaGraph::build_with_quant_aware_prune(
                &train, dim, &build_config, &qa_config,
                move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
                &mut rng,
            )
        };
        let build_time = t0.elapsed().as_secs_f64();
        println!("[β={:.1}] 建图完成: {:.1}s", beta, build_time);

        // f32 搜索（无量化，测量图本身质量）
        let (f32_recall, f32_qps) = eval_f32(
            &train, &test, &gt, dim, nq, gt_stride, &graph, ef_search, k,
        );

        if beta == 0.0 {
            f32_baseline_recall = f32_recall;
        }

        // ADC + rerank 搜索
        let (adc_recall, adc_qps, avg_deg) = eval_adc_rerank(
            &cb, &train, &quantized_db, &test, &gt, dim, n, nq, gt_stride,
            &graph, ef_search, top_n, k,
        );

        let degrad = f32_baseline_recall - adc_recall;

        println!("{:>6.1} {:>10.4} {:>10.0} {:>12.4} {:>12.0} {:>10.4} {:>10.1}",
            beta, f32_recall, f32_qps, adc_recall, adc_qps, degrad, avg_deg);
        println!();
    }

    println!("=== 结论 ===");
    println!("β=0 基线: f32 recall → AVQ ADC+rerank recall（量化退化）");
    println!("β>0: 量化感知剪枝是否减小退化？");
    println!("判断标准: recall 提升 > 0.5% 且 QPS 下降 < 5% → β 有效");
}
