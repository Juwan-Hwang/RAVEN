//! OPT-11: β 假设在 r_max=64 上重新验证（siftsmall 快速版）
//!
//! OPT-10 已在 siftsmall r_max=32 上证明 β 无收益。
//! OPT-11 假设：β 可能在 r_max=64 的高质量图上才生效。
//!
//! 本实验用 siftsmall + r_max=64 快速验证：
//! - 如果 β>0 仍然 ≤ β=0 基线，说明 β 假设在 SIFT 数据上不成立
//! - 如果 β>0 > β=0 基线，需要在 SIFT1M 上完整验证
//!
//! 注意：siftsmall 10K 节点 + r_max=64 可能出现 recall 天花板效应（recall=1.0），
//! 此时降低 ef_search 让 recall 有提升空间。

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

fn eval_f32(
    train: &[f32], test: &[f32], gt: &[i32],
    dim: usize, nq: usize, gt_stride: usize,
    graph: &VamanaGraph, ef_search: usize, k: usize,
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
            if found.contains(&(g as u32)) { hits += 1; }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (hits as f64 / (nq * k) as f64, nq as f64 / elapsed, avg_deg)
}

fn main() {
    println!("=== OPT-11: β 假设 r_max=64 验证（siftsmall）===");
    println!();

    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("数据加载: {:.2}s", t0.elapsed().as_secs_f64());
    println!("siftsmall: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);
    println!();

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let gt_stride = gt_k;
    let k = 10;

    // AVQ 训练
    println!("=== AVQ 训练 ===");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 训练: {:.2}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train))
        .collect();
    println!("节点量化误差: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // r_max=64 建图配置
    let build_config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 64,  // OPT-11 核心变量
        max_iterations: 2,
    };

    // 先测 β=0 基线，用多个 ef_search 找到非天花板 recall
    println!("=== β=0 基线（r_max=64）===");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph_base = VamanaGraph::build(&train, dim, &build_config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图: {:.2}s", build_time);

    let ef_values = [20, 50, 100];
    println!("{:>6} {:>10} {:>10} {:>10}", "ef", "recall@10", "QPS", "avg_deg");
    println!("{:-<40}", "");
    for &ef in &ef_values {
        let (recall, qps, deg) = eval_f32(&train, &test, &gt, dim, nq, gt_stride, &graph_base, ef, k);
        println!("{:>6} {:>10.4} {:>10.0} {:>10.1}", ef, recall, qps, deg);
    }
    println!();

    // 选一个 recall 非天花板的 ef_search 来测 β
    // ef=20 recall 应该 < 1.0，有提升空间
    let ef_test = 20;
    let (recall_base, qps_base, deg_base) = eval_f32(&train, &test, &gt, dim, nq, gt_stride, &graph_base, ef_test, k);
    println!("选定 ef={} 测 β：基线 recall={:.4}", ef_test, recall_base);
    println!();

    // 扫描 β
    let betas = [0.05f32, 0.1, 0.3, 0.5, 1.0, 2.0];
    println!("=== β 扫描（r_max=64, ef={}）===", ef_test);
    println!("{:>6} {:>10} {:>10} {:>10} {:>10}", "beta", "recall@10", "QPS", "avg_deg", "build_s");
    println!("{:-<50}", "");
    println!("{:>6.2} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
        0.0, recall_base, qps_base, deg_base, build_time);

    let mut best_recall = recall_base;
    let mut best_beta = 0.0f32;

    for &beta in &betas {
        let mut rng = ChaCha8Rng::seed_from(42);
        let qa_config = QuantAwarePruneConfig {
            alpha: 1.0,
            beta,
            epsilon: EPSILON,
            r_max: 64,
            normalization: NormalizationScheme::Mean,
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
            &train, &test, &gt, dim, nq, gt_stride, &graph, ef_test, k,
        );

        println!("{:>6.2} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
            beta, recall, qps, avg_deg, build_time);

        if recall > best_recall {
            best_recall = recall;
            best_beta = beta;
        }
    }

    println!();
    println!("=== 结论 ===");
    println!("r_max=64, ef={}: β=0 基线 recall={:.4}", ef_test, recall_base);
    println!("最佳: β={:.2} recall={:.4}", best_beta, best_recall);
    let delta = best_recall - recall_base;
    if delta > 0.005 {
        println!("β 假设可能成立: recall 提升 {:.4} (>0.5%)，需 SIFT1M 完整验证", delta);
    } else if delta > 0.0 {
        println!("β 收益可忽略: recall 提升 {:.4} (<0.5%)", delta);
    } else {
        println!("β 假设不成立: recall 未提升（β>0 全部 ≤ β=0 基线）");
    }
}
