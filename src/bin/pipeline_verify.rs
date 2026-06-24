//! Pipeline 验证实验
//!
//! 验证 S1-S5 修复后 pipeline 能正常工作
//! 跑 beta=0.0 和 beta=0.3 两种配置，对比 recall@10
//! - beta=0.0：quant_aware_prune 跳过（标准 RobustPrune）
//! - beta=0.3：quant_aware_prune 真正执行（S1 修复验证）

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::build::{BuildConfig, BuildPipeline};
use raven::graph::{VamanaGraph, GraphSearcher};

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
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

fn main() {
    println!("=== Pipeline 验证实验（S1-S5 修复后）===");
    println!("验证 pipeline 能正常工作，beta=0.0 和 beta=0.3 两种配置");
    println!();

    // 1. 加载 siftsmall 数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, _, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("数据加载: {:.1}s", t0.elapsed().as_secs_f64());
    println!("siftsmall: dim={}, base={}, query={}", dim, n, nq);
    println!();

    // 归一化到 [0,1]（设计文档：SIFT 数据 0-255 范围会导致梯度爆炸）
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. 跑 pipeline（beta=0.0：标准 RobustPrune，quant_aware_prune 跳过）
    println!("=== Pipeline beta=0.0（标准 RobustPrune）===");
    let config0 = BuildConfig {
        beta: 0.0,
        r_max: 32,
        r_soft: 48,
        l_build: 100,
        ..Default::default()
    };
    let pipeline0 = BuildPipeline::new(config0);
    let t0 = Instant::now();
    let result0 = pipeline0.run(train.clone(), dim);
    println!("Pipeline beta=0.0 构建: {:.1}s", t0.elapsed().as_secs_f64());

    // 用返回的 opq 旋转 train 和 test
    let opq0 = result0.opq.as_ref().expect("opq should be trained");
    let train_rot0 = opq0.apply(&train, dim);
    let test_rot0 = opq0.apply(&test, dim);

    let recall0 = eval_recall(&train_rot0, &test_rot0, &gt, dim, nq, &result0.graph, ef_search, k);
    println!("recall@10: {:.4}", recall0);
    println!("alpha_variants: {}", result0.alpha_variants.len());
    println!("final_stage: {:?}", result0.final_stage);
    println!();

    // 3. 跑 pipeline（beta=0.3：quant_aware_prune 真正执行，S1 修复验证）
    println!("=== Pipeline beta=0.3（量化感知 RobustPrune，S1 修复验证）===");
    let config3 = BuildConfig {
        beta: 0.3,
        r_max: 32,
        r_soft: 48,
        l_build: 100,
        ..Default::default()
    };
    let pipeline3 = BuildPipeline::new(config3);
    let t0 = Instant::now();
    let result3 = pipeline3.run(train.clone(), dim);
    println!("Pipeline beta=0.3 构建: {:.1}s", t0.elapsed().as_secs_f64());

    let opq3 = result3.opq.as_ref().expect("opq should be trained");
    let train_rot3 = opq3.apply(&train, dim);
    let test_rot3 = opq3.apply(&test, dim);

    let recall3 = eval_recall(&train_rot3, &test_rot3, &gt, dim, nq, &result3.graph, ef_search, k);
    println!("recall@10: {:.4}", recall3);
    println!("alpha_variants: {}", result3.alpha_variants.len());
    println!("final_stage: {:?}", result3.final_stage);
    println!();

    // 4. 汇总
    println!("=== 汇总 ===");
    println!("beta=0.0（标准 RobustPrune）: recall={:.4}", recall0);
    println!("beta=0.3（量化感知 RobustPrune）: recall={:.4}", recall3);
    println!();

    if recall0 > 0.9 && recall3 > 0.9 {
        println!("PASS: S1-S5 修复后 pipeline 正常工作，两个配置 recall 都合理");
    } else {
        println!("FAIL: recall 异常，需要检查 S1-S5 修复");
    }
}
