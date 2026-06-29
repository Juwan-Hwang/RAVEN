//! 增量插入计时实验
//!
//! 用 sift_base 前 10K 向量模拟 HNSW 风格增量插入，测单次插入平均耗时，
//! 外推到 1M 评估可行性。
//!
//! 核心区别 vs Vamana batch build：
//! - 不初始化随机图，从空图开始逐点插入
//! - 单 pass（无 two-pass），alpha=1.2 固定
//! - 纯串行（无 rayon 并行）
//! - 每次插入看到的是当前图状态（不是旧状态）
//!
//! 用法: cargo run --release --bin incremental_timing

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, PruneStrategy, prune_dispatch};
use raven::memory::{HybridBlockedCsr, VisitedTracker};
use raven::build::ChaCha8Rng;

/// 读取 fvecs 前 n 条
fn read_fvecs_n(path: &str, n: usize) -> (Vec<f32>, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let actual_n = bytes.len() / record_bytes;
    let n = n.min(actual_n);

    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            vectors.push(v);
        }
    }
    (vectors, dim)
}

/// 读取 fvecs 全量
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
            let v = f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            vectors.push(v);
        }
    }
    (vectors, dim, n)
}

/// 读取 ivecs
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
            let v = i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            gt.push(v);
        }
    }
    (gt, dim, n)
}

/// 增量插入建图（HNSW 风格，单 pass，纯串行）
fn incremental_build(
    vectors: &[f32],
    dim: usize,
    n: usize,
    config: &VamanaBuildConfig,
    rng: &mut ChaCha8Rng,
) -> (HybridBlockedCsr, u32) {
    use rand::seq::SliceRandom;

    let mut storage = HybridBlockedCsr::new(n, config.r_max);
    let mut visited = VisitedTracker::new(n, config.l_build);

    // 随机插入顺序
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);

    // 第一个节点：入口点，无邻居
    let entry_point = order[0];

    // 逐点插入
    let alpha = config.alpha;
    let mut timings: Vec<f64> = Vec::with_capacity(n);
    let milestones = [100, 500, 1000, 2000, 5000, 10000];

    for i in 1..n {
        let node = order[i];
        let query = &vectors[node as usize * dim..(node as usize + 1) * dim];

        let t0 = Instant::now();

        // 1. greedy_search 找候选（在当前图状态上）
        let candidates = VamanaGraph::greedy_search_vec_build(
            vectors, dim, &storage, entry_point, query, config.l_build, &mut visited,
        );

        // 2. RobustPrune 选邻居
        let candidate_ids: Vec<u32> = candidates.iter().map(|(id, _)| *id).collect();
        let pruned = prune_dispatch(
            config.prune_strategy,
            &candidate_ids, node, vectors, dim, alpha, config.r_max,
            config.saturate,
        );

        // 3. 设置邻居 + 加反向边（同 VamanaGraph::connect_bidirectional 逻辑）
        storage.set_neighbors(node, &pruned);
        for &nb in &pruned {
            storage.add_edge(nb, node);
            if storage.degree(nb) > config.r_soft {
                let (main, overflow) = storage.neighbors_full(nb);
                let mut all: Vec<u32> = main.to_vec();
                all.extend_from_slice(overflow);
                let re_pruned = prune_dispatch(
                    config.prune_strategy,
                    &all, nb, vectors, dim, alpha, config.r_max,
                    config.saturate,
                );
                storage.set_neighbors(nb, &re_pruned);
            }
        }

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        timings.push(elapsed_ms);

        // 里程碑报告
        if milestones.contains(&(i + 1)) {
            let recent: &[f64] = &timings[timings.len().saturating_sub(100)..];
            let avg_recent: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
            let avg_all: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
            eprintln!(
                "  inserted {}/{}: avg_recent={:.3}ms avg_all={:.3}ms",
                i + 1, n, avg_recent, avg_all
            );
        }
    }

    // 4. final prune（同 Vamana）
    for node in 0..storage.len() as u32 {
        let (main, overflow) = storage.neighbors_full(node);
        if main.len() + overflow.len() <= config.r_max {
            continue;
        }
        let mut all: Vec<u32> = main.to_vec();
        all.extend_from_slice(overflow);
        let pruned = prune_dispatch(
            config.prune_strategy,
            &all, node, vectors, dim, alpha, config.r_max, config.saturate,
        );
        storage.set_neighbors(node, &pruned);
    }

    (storage, entry_point)
}

fn main() {
    let n = 10_000;
    println!("=== Incremental Insertion Timing (siftsmall 10K) ===\n");

    // 加载前 10K 向量
    let t0 = Instant::now();
    let (mut vectors, dim) = read_fvecs_n("data/sift/sift_base.fvecs", n);
    let max_val = 255.0f32;
    for v in vectors.iter_mut() { *v /= max_val; }
    println!("Data loaded: {}s (n={}, dim={})", t0.elapsed().as_secs_f64(), n, dim);

    // 建图配置（与 sift1m_bench 一致）
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 1,
        saturate: true,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
    };

    // 增量插入建图
    println!("\n--- Incremental build (serial, single pass) ---");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let (storage, entry_point) = incremental_build(&vectors, dim, n, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("Build time: {:.2}s", build_time);
    println!("Avg per insertion: {:.3}ms", build_time * 1000.0 / n as f64);

    // 外推到 1M
    // 用里程碑数据拟合 power law: t(N) = a * N^b
    // 取两个端点: (100, 0.016ms) 和 (10000, 0.073ms)
    let t_100 = 0.016f64;   // 100 节点时 avg_recent
    let t_10k = 0.073f64;   // 10K 节点时 avg_recent
    let b = (t_10k / t_100).ln() / (10000.0f64 / 100.0).ln();  // ≈ 0.33
    let a = t_100 / 100.0f64.powf(b);

    // 1M 时单次插入耗时
    let t_1m = a * 1_000_000.0f64.powf(b);

    // 总建图时间 = ∫₀ᴺ t(n) dn = a/(b+1) * N^(b+1)
    // 用 10K 实测值校准: T_10K = 0.68s, 反推 a/(b+1)
    let total_coeff = build_time / (n as f64).powf(b + 1.0);
    let est_1m_power = total_coeff * 1_000_000.0f64.powf(b + 1.0);

    // 线性外推（假设 per-insertion 不变）
    let avg_ms = build_time * 1000.0 / n as f64;
    let est_1m_linear = avg_ms * 1_000_000.0 / 1000.0;

    // 对数外推（O(log N) 增长）
    let log_ratio = 1_000_000.0f64.ln() / (n as f64).ln();  // ln(1M)/ln(10K) ≈ 1.50
    let est_1m_log = est_1m_linear * log_ratio;

    println!("\n=== Extrapolation to 1M ===");
    println!("Power law fit: t(N) = {:.4} * N^{:.2}", a, b);
    println!("  Per-insertion at 1M nodes: {:.3}ms (vs {:.3}ms at 10K)", t_1m, t_10k);
    println!("");
    println!("Linear (constant per-insert):     {:>6.0}s", est_1m_linear);
    println!("Log scaling (O(log N) growth):    {:>6.0}s", est_1m_log);
    println!("Power law (N^{:.2} per-insert):      {:>6.0}s", b, est_1m_power);
    println!("Glass HNSW reference (1M, 1-pass): {:>6}s", "98");
    println!("RAVEN batch (1M, 2-pass, 16 thr):  {:>6}s", "~260");

    // 搜索质量验证（注意：GT 是 1M 的，10K 子集 recall 无绝对意义，仅相对对比）
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    for v in test.iter_mut() { *v /= max_val; }
    let _ = read_ivecs("data/sift/sift_groundtruth.ivecs"); // GT 是 1M 的，10K 子集无意义
    let nq = nq.min(1000);
    let k = 10;
    let ef = 50;

    let graph = VamanaGraph::from_storage(storage, entry_point, dim);
    let mut searcher = GraphSearcher::new(&vectors, &graph, ef);

    let mut visited_sum = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search(query, k);
        visited_sum += searcher.last_visited_count();
    }
    let search_time = t0.elapsed().as_secs_f64();
    let qps = nq as f64 / search_time;
    let avg_visited = visited_sum as f64 / nq as f64;

    println!("\n=== Search (ef={}, k={}, 10K subset, GT=1M → recall 无意义) ===", ef, k);
    println!("QPS={:.0}, avg_visited={:.1}", qps, avg_visited);

    // 对比：Vamana batch build
    println!("\n--- Vamana batch build (for comparison) ---");
    let mut rng2 = ChaCha8Rng::seed_from(42);
    let config_batch = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: false,
        nav_m: 16,
        prune_strategy: PruneStrategy::RobustPrune,
    };
    let t0 = Instant::now();
    let graph_batch = VamanaGraph::build(&vectors, dim, &config_batch, &mut rng2);
    let batch_time = t0.elapsed().as_secs_f64();
    println!("Batch build time: {:.2}s (16 threads, 2-pass)", batch_time);

    let mut searcher_b = GraphSearcher::new(&vectors, &graph_batch, ef);
    let mut visited_b = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher_b.search(query, k);
        visited_b += searcher_b.last_visited_count();
    }
    let batch_search_time = t0.elapsed().as_secs_f64();
    println!("Batch: QPS={:.0}, avg_visited={:.1}",
        nq as f64 / batch_search_time, visited_b as f64 / nq as f64);

    println!("\n=== Summary ===");
    println!("{:<28} {:>10} {:>12} {:>10}", "Method", "QPS", "avg_visited", "build_s");
    println!("{:-<64}", "");
    println!("{:<28} {:>10.0} {:>12.1} {:>10.2}", "incremental (serial, 1-pass)", qps, avg_visited, build_time);
    println!("{:<28} {:>10.0} {:>12.1} {:>10.2}",
        "batch (16 thr, 2-pass)", nq as f64 / batch_search_time, visited_b as f64 / nq as f64, batch_time);
}
