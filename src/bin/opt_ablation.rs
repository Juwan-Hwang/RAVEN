//! v8.0 消融实验：逐项独立验证每项优化
//!
//! 实验设计（科学方法论）：
//! 1. 建图一次（saturate=true），保存到内存
//! 2. 在同一张图上跑多个搜索变体，隔离搜索层优化效果
//! 3. 另建 saturate=false 的图，对比图结构差异
//!
//! 搜索变体（同一张图，仅搜索代码不同）：
//!   A: baseline     — 当前搜索（bitset 已生效）
//!   B: two_pass     — Glass SearchImpl2 模式（批量收集 + 预取向量）
//!   C: multi_pref   — 多行图预取（4 cache lines vs 1）
//!   D: all_combined — B+C 合并
//!
//! 图结构变体（不同图，相同搜索代码）：
//!   E: saturate_off — saturate=false 建图，用 baseline 搜索
//!
//! 用法：
//!   cargo run --release --bin opt_ablation

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::memory::{HybridBlockedCsr, VisitedTracker};
use raven::graph::linear_pool::LinearPool;

// ── 数据读取 ──

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (vectors, dim, n)
}

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("open ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read ivecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            gt.push(i32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

// ── 搜索变体 A: baseline（当前代码，bitset 已生效） ──

fn search_baseline(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    while let Some((node, _dist)) = pool.pop() {
        let neighbors = storage.neighbors(node);
        if let Some((next_node, _)) = pool.peek_unchecked() {
            storage.prefetch_neighbors(next_node);
        }
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                pool.insert(neighbor, d);
            }
        }
    }

    pool.to_sorted_vec()
}

// ── 搜索变体 B: two_pass（Glass SearchImpl2 模式） ──
//
// Glass 的核心技巧：
// 1. Pop 节点后，第一遍扫描邻居列表，收集未访问的到 edge_buf
// 2. 预取 edge_buf 前 po 个邻居的向量数据
// 3. 第二遍扫描 edge_buf，每次预取 i+po 前瞻的向量
// 4. 计算距离时向量已在 L1/L2 cache
//
// 与 baseline 的区别：baseline 逐个邻居计算距离，
// 每次距离计算前向量数据可能在 DRAM，cache miss 延迟 ~100ns。
// two_pass 批量预取，距离计算时数据已在 cache，延迟 ~4ns。

fn search_two_pass(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
    po: usize,  // prefetch offset（前瞻距离）
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    // 预分配 edge_buf（栈上，避免堆分配）
    // R_max=64 → 64 * 4 = 256 bytes，fit 栈
    let mut edge_buf: [u32; 128] = [0; 128];

    while let Some((node, _dist)) = pool.pop() {
        let neighbors = storage.neighbors(node);

        // 图预取：预取下一轮要 pop 的节点的邻居列表
        if let Some((next_node, _)) = pool.peek_unchecked() {
            storage.prefetch_neighbors(next_node);
        }

        // 第一遍：收集未访问邻居到 edge_buf
        let mut edge_size = 0usize;
        for &v in neighbors {
            if edge_size >= 128 {
                break;
            }
            if visited.visit(v) {
                edge_buf[edge_size] = v;
                edge_size += 1;
            }
        }

        // 预取前 po 个邻居的向量数据
        let prefetch_count = po.min(edge_size);
        for i in 0..prefetch_count {
            let v = edge_buf[i] as usize;
            let ptr = &vectors[v * dim] as *const f32 as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        // 第二遍：计算距离，同时前瞻预取
        for i in 0..edge_size {
            // 前瞻预取：i + po 处的向量
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = &vectors[v * dim] as *const f32 as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
            pool.insert(neighbor, d);
        }
    }

    pool.to_sorted_vec()
}

// ── 搜索变体 C: multi_pref（多行图预取） ──
//
// 当前 prefetch_neighbors 只预取 1 个 cache line（64 bytes）。
// R_max=64 → 邻居列表 64*4=256 bytes = 4 cache lines。
// 多行预取确保整个邻居列表在 L1 cache。

fn search_multi_pref(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    while let Some((node, _dist)) = pool.pop() {
        // 多行图预取：预取完整邻居列表（4 cache lines for R=64）
        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            // R_max=64 → 256 bytes → 4 cache lines
            // 预取 4 行覆盖整个邻居列表
            unsafe {
                std::arch::x86_64::_mm_prefetch::<0>(ptr);
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
            }
        }

        let neighbors = storage.neighbors(node);
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                pool.insert(neighbor, d);
            }
        }
    }

    pool.to_sorted_vec()
}

// ── 搜索变体 D: all_combined（two_pass + multi_pref） ──

fn search_combined(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
    po: usize,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];

    while let Some((node, _dist)) = pool.pop() {
        // 多行图预取
        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            unsafe {
                std::arch::x86_64::_mm_prefetch::<0>(ptr);
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
            }
        }

        let neighbors = storage.neighbors(node);
        let mut edge_size = 0usize;
        for &v in neighbors {
            if edge_size >= 128 {
                break;
            }
            if visited.visit(v) {
                edge_buf[edge_size] = v;
                edge_size += 1;
            }
        }

        let prefetch_count = po.min(edge_size);
        for i in 0..prefetch_count {
            let v = edge_buf[i] as usize;
            let ptr = &vectors[v * dim] as *const f32 as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        for i in 0..edge_size {
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = &vectors[v * dim] as *const f32 as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
            pool.insert(neighbor, d);
        }
    }

    pool.to_sorted_vec()
}

// ── 基准测试框架 ──

struct BenchResult {
    name: String,
    ef: usize,
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

fn bench_search_variant(
    name: &str,
    vectors: &[f32],
    dim: usize,
    graph: &VamanaGraph,
    test: &[f32],
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    search_fn: impl Fn(&[f32], usize, &HybridBlockedCsr, u32, &[f32], usize, &mut VisitedTracker) -> Vec<(u32, f32)>,
) -> BenchResult {
    let storage = graph.storage();
    let entry = graph.entry_point();
    let mut visited = VisitedTracker::new(vectors.len() / dim, ef);

    // warmup
    for q in 0..nq.min(100) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = search_fn(vectors, dim, storage, entry, query, ef, &mut visited);
    }

    let mut hits = 0usize;
    let mut total = 0usize;
    let mut total_visited = 0usize;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = search_fn(vectors, dim, storage, entry, query, ef, &mut visited);
        total_visited += visited.visited_count();

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
        total += k;
    }
    let elapsed = t0.elapsed();

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / elapsed.as_secs_f64();
    let avg_visited = total_visited as f64 / nq as f64;

    println!(
        "  {:>16} ef={:>4}  recall={:.4}  QPS={:>8.0}  avg_visited={:.0}",
        name, ef, recall, qps, avg_visited
    );

    BenchResult {
        name: name.to_string(),
        ef,
        recall,
        qps,
        avg_visited,
    }
}

/// 版本横幅
fn print_banner() {
    let pkg_ver = env!("CARGO_PKG_VERSION");
    let git_hash = env!("RAVEN_GIT_HASH");
    let build_ts = env!("RAVEN_BUILD_TS");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  RAVEN v{}  git:{}  build:{}  ║", pkg_ver, git_hash, build_ts);
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn main() {
    print_banner();
    println!("=== v8.0 消融实验：逐项独立验证 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    println!("数据: n={}, dim={}, nq={}\n", n, dim, nq);

    // ════════════════════════════════════════════════════
    //  Part 1: 搜索层优化（同一张图，不同搜索代码）
    // ════════════════════════════════════════════════════
    println!("══ Part 1: 搜索层优化（同一张 saturate=true 图）══\n");

    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
        saturate: true,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图完成: {:.1}s\n", build_time);

    // 度数统计
    let stats = graph.degree_stats();
    println!(
        "[degree] mean={:.1} p95={} p99={} max={} isolated={}",
        stats.mean_degree, stats.p95_degree, stats.p99_degree,
        stats.max_degree, stats.isolated_nodes
    );

    // VisitedTracker 内存对比
    let vt = VisitedTracker::new(n, 200);
    println!("[visited] bitset 内存 = {} bytes ({:.1} KB)", vt.bits_bytes(), vt.bits_bytes() as f64 / 1024.0);
    println!("[visited] 对比 Vec<u8> = {} bytes ({:.1} KB)\n", n, n as f64 / 1024.0);

    let ef_list: &[usize] = &[50, 100, 200];

    for &ef in ef_list {
        println!("--- ef_search={} ---", ef);

        // A: baseline
        let _ra = bench_search_variant(
            "baseline", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_baseline(v, d, s, e, q, ef, vis),
        );

        // B: two_pass (po=4)
        let _rb = bench_search_variant(
            "two_pass(po=4)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_two_pass(v, d, s, e, q, ef, vis, 4),
        );

        // B2: two_pass (po=8)
        let _rb2 = bench_search_variant(
            "two_pass(po=8)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_two_pass(v, d, s, e, q, ef, vis, 8),
        );

        // C: multi_pref
        let _rc = bench_search_variant(
            "multi_pref", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_multi_pref(v, d, s, e, q, ef, vis),
        );

        // D: combined (two_pass po=4 + multi_pref)
        let _rd = bench_search_variant(
            "combined(po=4)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_combined(v, d, s, e, q, ef, vis, 4),
        );

        // D2: combined (two_pass po=8 + multi_pref)
        let _rd2 = bench_search_variant(
            "combined(po=8)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_combined(v, d, s, e, q, ef, vis, 8),
        );

        println!();
    }

    // ════════════════════════════════════════════════════
    //  Part 2: 图结构优化（saturate=true vs false）
    // ════════════════════════════════════════════════════
    println!("══ Part 2: 图结构优化（saturate=true vs false）══\n");

    let t0 = Instant::now();
    let mut rng2 = ChaCha8Rng::seed_from(42);
    let config_off = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
        saturate: false,
    };
    let graph_off = VamanaGraph::build(&train, dim, &config_off, &mut rng2);
    let build_time_off = t0.elapsed().as_secs_f64();
    println!("建图(saturate=false): {:.1}s", build_time_off);

    let stats_off = graph_off.degree_stats();
    println!(
        "[degree] mean={:.1} p95={} p99={} max={} isolated={}",
        stats_off.mean_degree, stats_off.p95_degree, stats_off.p99_degree,
        stats_off.max_degree, stats_off.isolated_nodes
    );
    println!();

    for &ef in ef_list {
        println!("--- ef_search={} ---", ef);

        // E: saturate=true (baseline search on saturate=true graph)
        let _re = bench_search_variant(
            "sat=true", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_baseline(v, d, s, e, q, ef, vis),
        );

        // F: saturate=false (baseline search on saturate=false graph)
        let _rf = bench_search_variant(
            "sat=false", &train, dim, &graph_off, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_baseline(v, d, s, e, q, ef, vis),
        );

        println!();
    }

    println!("\n=== 实验完成 ===");
}
