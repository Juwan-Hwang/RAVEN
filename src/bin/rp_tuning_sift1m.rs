//! RP-Tuning SIFT1M Pareto 曲线实验
//!
//! 1. 建图（f32, α=1.2, max_iter=2）
//! 2. RP-Tuning 生成 α 变体（α=1.0, 1.2, 1.5, 2.0）
//! 3. 对每个变体，用不同 ef_search（50, 100, 200, 400）跑搜索
//! 4. 输出 Pareto 曲线数据（recall-QPS）

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::rp_tuning::{RPTuning, RPTuningConfig, RPTuningStorageScheme};
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

/// 对单个变体 + ef_search 跑搜索，返回 (recall, qps)
fn eval_variant(
    vectors: &[f32],
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    dim: usize,
    test: &[f32],
    gt: &[i32],
    nq: usize,
    k: usize,
    ef_search: usize,
) -> (f64, f64) {
    let graph = VamanaGraph::from_storage(storage.clone(), entry_point, dim);
    let mut searcher = GraphSearcher::new(vectors, &graph, ef_search);
    let gt_stride = 100;

    let t0 = Instant::now();
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
    let search_time = t0.elapsed().as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    let qps = nq as f64 / search_time;
    (recall, qps)
}

fn main() {
    println!("=== RP-Tuning SIFT1M Pareto 曲线实验 ===");
    println!();

    // 1. 加载数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("数据加载: {}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 归一化到 [0,1]
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }

    // 2. 建图（f32, α=1.2, max_iter=2）
    println!("=== 建图（Vamana α=1.2, r_max=32, l_build=100, max_iter=2）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let base_graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. RP-Tuning 生成 α 变体（秒级）
    println!("=== RP-Tuning 生成 α 变体（Scheme A）===");
    let t0 = Instant::now();
    let rp_config = RPTuningConfig {
        scheme: RPTuningStorageScheme::SchemeA,
        alpha_points: vec![1.0, 1.2, 1.5, 2.0],
        r_max: 32,
    };
    let variants = RPTuning::generate_variants(&base_graph, &train, dim, &rp_config);
    println!("RP-Tuning 生成 {} 个变体: {:.2}s", variants.len(), t0.elapsed().as_secs_f64());
    println!();

    // 4. 对每个变体 + 每个 ef_search 跑搜索
    let ef_points = vec![50, 100, 200, 400];
    let k = 10;

    println!("=== Pareto 曲线数据 ===");
    println!("{:<8} {:<10} {:>10} {:>10}", "alpha", "ef_search", "recall@10", "QPS");
    println!("{:-<42}", "");

    for variant in &variants {
        for &ef in &ef_points {
            let (recall, qps) = eval_variant(
                &train, &variant.storage, variant.entry_point,
                dim, &test, &gt, nq, k, ef,
            );
            println!("{:<8.1} {:<10} {:>10.4} {:>10.0}", variant.alpha, ef, recall, qps);
        }
        println!();
    }

    // 5. 汇总
    println!("=== 汇总 ===");
    println!("基础图: α=1.2, r_max=32, max_iter=2");
    println!("RP-Tuning 变体: α=[1.0, 1.2, 1.5, 2.0], Scheme A");
    println!("ef_search: [50, 100, 200, 400]");
    println!();
    println!("Pareto 前沿分析：α 越大保留更多长程边，recall 越高但 QPS 越低");
}
