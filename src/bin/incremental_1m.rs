//! SIFT1M 增量插入 vs 批量建图对比实验
//!
//! 核心假设：增量插入产出的图导航质量更好（avg_visited 更低），
//! 因为每个节点的邻居选择基于真实图导航结果，而非随机初始化。
//!
//! 增量插入（HNSW 风格）：
//!   - 空图起步，逐点串行插入
//!   - 单 pass，alpha=1.2 固定
//!   - 每次插入看到当前已优化图状态
//!
//! 批量建图（Vamana 标准）：
//!   - 随机初始化 → 2-pass 并行优化 → final prune
//!   - 16 线程 rayon 并行
//!
//! 用法: cargo run --release --bin incremental_1m

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, PruneStrategy};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
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
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");
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

fn recall_at_k(found: &[u32], gt_slice: &[i32], k: usize) -> f64 {
    let mut hits = 0usize;
    for &g in gt_slice.iter().take(k) {
        if found.contains(&(g as u32)) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

/// 评测搜索质量：recall, QPS, avg_visited
fn eval_search(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
) {
    let k = 10;
    for &ef in &[50, 100, 200] {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        let mut recall_sum = 0.0f64;
        let mut visited_sum = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = searcher.search(query, k);
            let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
            let gt_slice = &gt[q * gt_k..q * gt_k + k];
            recall_sum += recall_at_k(&found, gt_slice, k);
            visited_sum += searcher.last_visited_count();
        }
        let search_time = t0.elapsed().as_secs_f64();
        let recall = recall_sum / nq as f64;
        let qps = nq as f64 / search_time;
        let avg_visited = visited_sum as f64 / nq as f64;
        println!("  ef={:>3}: recall@10={:.4}, QPS={:>6.0}, avg_visited={:.0}",
            ef, recall, qps, avg_visited);
    }
}

fn main() {
    println!("=== SIFT1M Incremental vs Batch Build ===\n");

    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }
    println!("Data loaded: {:.3}s (n={}, dim={}, nq={})\n", t0.elapsed().as_secs_f64(), n, dim, nq);

    // ── 增量插入建图 ──
    println!("--- Incremental Build (serial, 1-pass, alpha=1.2) ---");
    let config_inc = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_soft: 48,
        r_max: 32,
        max_iterations: 1,      // 增量插入固定单 pass
        saturate: true,
        enable_layered_nav: true,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
    };
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph_inc = VamanaGraph::build_incremental(&train, dim, &config_inc, &mut rng);
    let inc_build_time = t0.elapsed().as_secs_f64();
    println!("  build: {:.1}s ({:.0} vec/s)\n", inc_build_time, n as f64 / inc_build_time);

    eval_search(&train, &graph_inc, &test, dim, nq, &gt, gt_k);

    // 释放增量图内存，避免与 batch 同时占用
    drop(graph_inc);

    // ── 批量建图（基线）──
    println!("\n--- Batch Build (16 threads, 2-pass, alpha=1.0→1.2) ---");
    let config_batch = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: true,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
    };
    let mut rng2 = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph_batch = VamanaGraph::build(&train, dim, &config_batch, &mut rng2);
    let batch_build_time = t0.elapsed().as_secs_f64();
    println!("  build: {:.1}s ({:.0} vec/s)\n", batch_build_time, n as f64 / batch_build_time);

    eval_search(&train, &graph_batch, &test, dim, nq, &gt, gt_k);

    // ── 汇总 ──
    println!("\n=== Summary ===");
    println!("{:<28} {:>10} {:>12} {:>10}", "Method", "build_s", "avg_visited", "recall@10");
    println!("{:-<64}", "");
    println!("(ef=50)");
    println!("{:<28} {:>10.1} {:>12} {:>10}", "incremental (serial, 1-pass)", inc_build_time, "see above", "see above");
    println!("{:<28} {:>10.1} {:>12} {:>10}", "batch (16 thr, 2-pass)", batch_build_time, "see above", "see above");
    println!("\nGlass HNSW ref (H20 实测): 98s build, 1041 avg_visited, 0.9465 recall@10 (ef=50)");
}
