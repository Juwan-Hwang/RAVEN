//! SIFT1M 端到端基准测试
//!
//! 目标：测量 1M 向量下的真实瓶颈
//! - f32 建图时间 + 搜索 QPS + recall
//! - AVQ 训练时间 + ADC 搜索 QPS
//! - ADC + rerank QPS + recall
//!
//! 参数（Week 6 最优）：
//! - Vamana: α=1.0, r_max=32, l_build=100, r_soft=48
//! - AVQ: K=256, sub_dim=8, α=0.30, iterations=25
//! - rerank: top-100 → top-10, ef_search=100

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
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

/// recall@k 计算（gt_stride = groundtruth 每查询邻居数）
fn recall_at_k(found: &[u32], gt_slice: &[i32], k: usize) -> f64 {
    let mut hits = 0usize;
    for &g in gt_slice.iter().take(k) {
        if found.contains(&(g as u32)) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

fn main() {
    println!("=== SIFT1M 端到端基准测试 ===");
    println!();

    // 1. 加载数据
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    // 加载 sift_learn（100K）用于 AVQ 训练，比用 1M base 快 10 倍
    let (mut learn, _, n_learn) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("数据加载: {}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}, learn={}", dim, n, nq, gt_nq, gt_k, n_learn);
    println!();

    // 归一化到 [0,1]（SIFT 原始 0-255，AVQ 训练需要）
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }
    for v in learn.iter_mut() { *v /= max_val; }

    // 2. f32 建图（Vamana two passes: α=1.0→1.2, rayon 并行）
    println!("=== f32 建图（Vamana α=1.2, r_max=32, l_build=100, max_iter=2, rayon 并行）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图时间: {:.2}s ({:.0} vec/s)", build_time, n as f64 / build_time);
    println!();

    // 3. f32 搜索 QPS + recall
    println!("=== f32 搜索（ef_search=100, k=10）===");
    let mut searcher = GraphSearcher::new(&train, &graph, 100);
    let t0 = Instant::now();
    let gt_stride = gt_k;
    let k = 10;
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let search_time = t0.elapsed().as_secs_f64();
    let recall_f32 = recall_sum / nq as f64;
    let qps_f32 = nq as f64 / search_time;
    println!("f32 recall@10={:.4}, QPS={:.0}, avg_latency={:.2}ms",
        recall_f32, qps_f32, search_time * 1000.0 / nq as f64);
    println!();

    // 4. AVQ 训练（用 sift_learn 100K + iter=5 加速，工业标准）
    println!("=== AVQ 训练（sift_learn 100K, K=256, sub_dim=8, α=0.30, iter=5）===");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    let avq_train_time = t0.elapsed().as_secs_f64();
    println!("AVQ 训练时间: {:.2}s", avq_train_time);
    println!();

    // 5. 构造量化数据库（ADC 粗筛用）
    let t0 = Instant::now();
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            cb.decode(&cb.encode(v))
        })
        .collect();
    println!("量化数据库构造: {:.2}s", t0.elapsed().as_secs_f64());

    // 6. ADC 搜索 QPS（无 rerank）
    println!();
    println!("=== ADC 搜索（量化距离, ef_search=100, k=10）===");
    let mut searcher_q = GraphSearcher::new(&quantized_db, &graph, 100);
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher_q.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let adc_time = t0.elapsed().as_secs_f64();
    let recall_adc = recall_sum / nq as f64;
    let qps_adc = nq as f64 / adc_time;
    println!("ADC recall@10={:.4}, QPS={:.0}, avg_latency={:.2}ms",
        recall_adc, qps_adc, adc_time * 1000.0 / nq as f64);
    println!();

    // 7. ADC + rerank QPS + recall（top-100 → f32 rerank → top-10）
    println!("=== ADC + rerank（top-100 粗筛 → f32 精排 → top-10）===");
    let top_n = 100;
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = searcher_q.search(query, top_n);
        // f32 rerank（SIMD 加速）
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
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let rerank_time = t0.elapsed().as_secs_f64();
    let recall_rerank = recall_sum / nq as f64;
    let qps_rerank = nq as f64 / rerank_time;
    println!("ADC+rerank recall@10={:.4}, QPS={:.0}, avg_latency={:.2}ms",
        recall_rerank, qps_rerank, rerank_time * 1000.0 / nq as f64);
    println!();

    // 8. 汇总
    println!("=== 汇总 ===");
    println!("{:<20} {:>10} {:>10} {:>12}", "方法", "recall@10", "QPS", "latency_ms");
    println!("{:-<52}", "");
    println!("{:<20} {:>10.4} {:>10.0} {:>12.2}", "f32 baseline", recall_f32, qps_f32, search_time * 1000.0 / nq as f64);
    println!("{:<20} {:>10.4} {:>10.0} {:>12.2}", "AVQ ADC", recall_adc, qps_adc, adc_time * 1000.0 / nq as f64);
    println!("{:<20} {:>10.4} {:>10.0} {:>12.2}", "AVQ ADC+rerank", recall_rerank, qps_rerank, rerank_time * 1000.0 / nq as f64);
    println!();
    println!("建图时间: {:.2}s | AVQ 训练: {:.2}s", build_time, avq_train_time);
}
