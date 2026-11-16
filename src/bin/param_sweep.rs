//! 统一参数扫描 benchmark
//!
//! Phase 1: 搜索参数（用缓存图，快速）
//!   - ef_search: 40, 45, 50, 55, 60, 65, 70
//!   - prefetch_offset: 0, 4, 8, 12, 16
//!   - rerank_factor: 1, 2, 3, 4, 5
//!
//! Phase 2: 图构建参数（需重建图，慢）
//!   - r_max: 24, 32, 48, 64
//!   - r_min_divisor: 3, 4, 5, 6
//!   - alpha: 1.0, 1.1, 1.2, 1.3, 1.5
//!   - l_build: 100, 150, 200, 300
//!   - nav_m: 8, 12, 16, 24
//!   - prune_strategy: DirectionalPrune, RobustPrune
//!   - saturate: true, false
//!
//! 用法�?
//!   cargo run --release --bin param_sweep -- --phase1     # 搜索参数扫描
//!   cargo run --release --bin param_sweep -- --phase2     # 图构建参数扫�?
//!   cargo run --release --bin param_sweep -- --phase2 --rmax-sweep  # 只扫 r_max
//!
//! 结果写入 tuning/ 目录

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, VamanaBuildConfig, VamanaGraph, PruneStrategy};
use raven::memory::serialize::Serializable;
use raven::quant::SQ8Dataset;

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

const REPEATS: usize = 10;
const ROUNDS: usize = 5;
const K: usize = 10;

/// 单次 benchmark 运行
struct BenchResult {
    qps: f64,
    recall: f64,
    cv: f64,
}

fn run_bench(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    sq8: &SQ8Dataset,
    ef: usize,
    po: usize,
    rerank: usize,
) -> BenchResult {
    // recall pass
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_sq8(sq8);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rerank);

    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, K);
        let gt_slice = &gt[q * gt_k..q * gt_k + K];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
        total += K;
    }
    let recall = hits as f64 / total as f64;

    // warmup
    let mut sink: u64 = 0;
    for _ in 0..REPEATS {
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = searcher.search_sq8(query, K);
            sink = sink.wrapping_add(result[0].0 as u64);
        }
    }

    // QPS rounds
    let mut qps_vals = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_sq8(sq8);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rerank);

        let t0 = Instant::now();
        for _ in 0..REPEATS {
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let result = searcher.search_sq8(query, K);
                sink = sink.wrapping_add(result[0].0 as u64);
            }
        }
        let dt = t0.elapsed();
        let total_queries = nq * REPEATS;
        qps_vals.push(total_queries as f64 / dt.as_secs_f64());
    }

    if sink == u64::MAX { eprintln!("impossible"); }

    qps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = qps_vals[ROUNDS / 2];
    let mean = qps_vals.iter().sum::<f64>() / ROUNDS as f64;
    let variance = qps_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ROUNDS as f64;
    let cv = variance.sqrt() / mean * 100.0;

    BenchResult { qps: median, recall, cv }
}

fn load_or_build_graph(
    train: &[f32],
    dim: usize,
    config: &VamanaBuildConfig,
    cache_path: &str,
) -> VamanaGraph {
    let path = std::path::Path::new(cache_path);
    if path.exists() {
        VamanaGraph::load(path).expect("load graph cache")
    } else {
        eprintln!("  building graph (cache={})...", cache_path);
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let g = VamanaGraph::build(train, dim, config, &mut rng);
        eprintln!("  build: {:.1}s", t0.elapsed().as_secs_f64());
        let _ = g.save(path);
        g
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let phase1 = args.iter().any(|a| a == "--phase1");
    let phase2 = args.iter().any(|a| a == "--phase2");
    let rmax_sweep = args.iter().any(|a| a == "--rmax-sweep");
    let rmin_sweep = args.iter().any(|a| a == "--rmin-sweep");
    let alpha_sweep = args.iter().any(|a| a == "--alpha-sweep");
    let lbuild_sweep = args.iter().any(|a| a == "--lbuild-sweep");
    let navm_sweep = args.iter().any(|a| a == "--navm-sweep");
    let prune_sweep = args.iter().any(|a| a == "--prune-sweep");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let sq8 = SQ8Dataset::build(&train, dim);

    // ══════════════════════════════════════════════════════════════�?
    // Phase 1: 搜索参数扫描（用缓存 DirectionalPrune 图）
    // ══════════════════════════════════════════════════════════════�?
    if phase1 {
        eprintln!("=== Phase 1: 搜索参数扫描 ===");
        eprintln!("data: n={}, dim={}, nq={}", n, dim, nq);

        let graph_path = "data/sift/graph_cache_dir_rmin4.bin";
        let graph = VamanaGraph::load(std::path::Path::new(graph_path)).expect("load graph");

        let mut out = String::new();
        out.push_str("phase,param,value,qps,recall,cv_pct\n");

        // 1a. ef_search 扫描（po=8, rr=3 固定�?
        eprintln!("\n--- 1a: ef_search sweep (po=8, rr=3) ---");
        for &ef in &[40, 45, 50, 55, 60, 65, 70] {
            let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, ef, 8, 3);
            eprintln!("  ef={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%", ef, r.qps, r.recall, r.cv);
            out.push_str(&format!("1a,ef_search,{},,{:.0},{:.4},{:.1}\n", ef, r.qps, r.recall, r.cv));
        }

        // 1b. prefetch_offset 扫描（ef=50, rr=3 固定�?
        eprintln!("\n--- 1b: prefetch_offset sweep (ef=50, rr=3) ---");
        for &po in &[0, 2, 4, 6, 8, 10, 12, 16] {
            let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, po, 3);
            eprintln!("  po={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%", po, r.qps, r.recall, r.cv);
            out.push_str(&format!("1b,prefetch_offset,{},,{:.0},{:.4},{:.1}\n", po, r.qps, r.recall, r.cv));
        }

        // 1c. rerank_factor 扫描（ef=50, po=8 固定�?
        eprintln!("\n--- 1c: rerank_factor sweep (ef=50, po=8) ---");
        for &rr in &[1, 2, 3, 4, 5, 8] {
            let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, rr);
            eprintln!("  rr={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%", rr, r.qps, r.recall, r.cv);
            out.push_str(&format!("1c,rerank_factor,{},,{:.0},{:.4},{:.1}\n", rr, r.qps, r.recall, r.cv));
        }

        let path = "tuning/phase1_search_params.csv";
        std::fs::write(path, out).expect("write");
        eprintln!("\nPhase 1 结果写入 {}", path);
    }

    // ══════════════════════════════════════════════════════════════�?
    // Phase 2: 图构建参数扫描（需重建图）
    // ══════════════════════════════════════════════════════════════�?
    if phase2 {
        let mut out = String::new();
        out.push_str("phase,param,value,qps,recall,cv_pct,build_time_s\n");

        // 2a. r_max 扫描
        if rmax_sweep || (!rmin_sweep && !alpha_sweep && !lbuild_sweep && !navm_sweep && !prune_sweep) {
            eprintln!("\n=== 2a: r_max sweep ===");
            for &r_max in &[24, 32, 48, 64] {
                let cache = format!("data/sift/sweep_rmax{}.bin", r_max);
                let cfg = VamanaBuildConfig {
                    alpha: 1.2, l_build: 200, r_max, r_soft: (r_max as f32 * 1.5) as usize,
                    max_iterations: 2, saturate: false,
                    enable_layered_nav: true, nav_m: 32,
                    prune_strategy: PruneStrategy::DirectionalPrune,
                    ..Default::default()
                };
                std::env::set_var("RAVEN_RMIN_DIVISOR", "4");
                let t0 = Instant::now();
                let graph = load_or_build_graph(&train, dim, &cfg, &cache);
                let build_time = t0.elapsed().as_secs_f64();
                let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
                eprintln!("  r_max={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                    r_max, r.qps, r.recall, r.cv, build_time);
                out.push_str(&format!("2a,r_max,{},{:.0},{:.4},{:.1},{:.0}\n", r_max, r.qps, r.recall, r.cv, build_time));
            }
        }

        // 2b. r_min_divisor 扫描
        if rmin_sweep || (!rmax_sweep && !alpha_sweep && !lbuild_sweep && !navm_sweep && !prune_sweep) {
            eprintln!("\n=== 2b: r_min_divisor sweep (r_max=32) ===");
            for &div in &[2, 3, 4, 5, 6, 8] {
                let cache = format!("data/sift/sweep_rmindiv{}.bin", div);
                let cfg = VamanaBuildConfig {
                    alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
                    max_iterations: 2, saturate: false,
                    enable_layered_nav: true, nav_m: 32,
                    prune_strategy: PruneStrategy::DirectionalPrune,
                    ..Default::default()
                };
                std::env::set_var("RAVEN_RMIN_DIVISOR", div.to_string());
                let t0 = Instant::now();
                let graph = load_or_build_graph(&train, dim, &cfg, &cache);
                let build_time = t0.elapsed().as_secs_f64();
                let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
                eprintln!("  rmin_div={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                    div, r.qps, r.recall, r.cv, build_time);
                out.push_str(&format!("2b,r_min_divisor,{},{:.0},{:.4},{:.1},{:.0}\n", div, r.qps, r.recall, r.cv, build_time));
            }
        }

        // 2c. alpha 扫描
        if alpha_sweep || (!rmax_sweep && !rmin_sweep && !lbuild_sweep && !navm_sweep && !prune_sweep) {
            eprintln!("\n=== 2c: alpha sweep (r_max=32) ===");
            for &alpha in &[0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0] {
                let cache = format!("data/sift/sweep_alpha{}.bin", (alpha * 10.0) as u32);
                let cfg = VamanaBuildConfig {
                    alpha, l_build: 200, r_max: 32, r_soft: 48,
                    max_iterations: 2, saturate: false,
                    enable_layered_nav: true, nav_m: 32,
                    prune_strategy: PruneStrategy::DirectionalPrune,
                    ..Default::default()
                };
                std::env::set_var("RAVEN_RMIN_DIVISOR", "4");
                let t0 = Instant::now();
                let graph = load_or_build_graph(&train, dim, &cfg, &cache);
                let build_time = t0.elapsed().as_secs_f64();
                let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
                eprintln!("  alpha={:.1}  QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                    alpha, r.qps, r.recall, r.cv, build_time);
                out.push_str(&format!("2c,alpha,{},{:.0},{:.4},{:.1},{:.0}\n", alpha, r.qps, r.recall, r.cv, build_time));
            }
        }

        // 2d. l_build 扫描
        if lbuild_sweep || (!rmax_sweep && !rmin_sweep && !alpha_sweep && !navm_sweep && !prune_sweep) {
            eprintln!("\n=== 2d: l_build sweep (r_max=32, alpha=1.2) ===");
            for &l_build in &[50, 100, 150, 200, 300, 400] {
                let cache = format!("data/sift/sweep_lbuild{}.bin", l_build);
                let cfg = VamanaBuildConfig {
                    alpha: 1.2, l_build, r_max: 32, r_soft: 48,
                    max_iterations: 2, saturate: false,
                    enable_layered_nav: true, nav_m: 32,
                    prune_strategy: PruneStrategy::DirectionalPrune,
                    ..Default::default()
                };
                std::env::set_var("RAVEN_RMIN_DIVISOR", "4");
                let t0 = Instant::now();
                let graph = load_or_build_graph(&train, dim, &cfg, &cache);
                let build_time = t0.elapsed().as_secs_f64();
                let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
                eprintln!("  l_build={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                    l_build, r.qps, r.recall, r.cv, build_time);
                out.push_str(&format!("2d,l_build,{},{:.0},{:.4},{:.1},{:.0}\n", l_build, r.qps, r.recall, r.cv, build_time));
            }
        }

        // 2e. nav_m 扫描
        if navm_sweep || (!rmax_sweep && !rmin_sweep && !alpha_sweep && !lbuild_sweep && !prune_sweep) {
            eprintln!("\n=== 2e: nav_m sweep (r_max=32, alpha=1.2) ===");
            for &nav_m in &[4, 8, 12, 16, 24, 32] {
                let cache = format!("data/sift/sweep_navm{}.bin", nav_m);
                let cfg = VamanaBuildConfig {
                    alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
                    max_iterations: 2, saturate: false,
                    enable_layered_nav: true, nav_m,
                    prune_strategy: PruneStrategy::DirectionalPrune,
                    ..Default::default()
                };
                std::env::set_var("RAVEN_RMIN_DIVISOR", "4");
                let t0 = Instant::now();
                let graph = load_or_build_graph(&train, dim, &cfg, &cache);
                let build_time = t0.elapsed().as_secs_f64();
                let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
                eprintln!("  nav_m={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                    nav_m, r.qps, r.recall, r.cv, build_time);
                out.push_str(&format!("2e,nav_m,{},{:.0},{:.4},{:.1},{:.0}\n", nav_m, r.qps, r.recall, r.cv, build_time));
            }
        }

        // 2f. PruneStrategy A/B
        if prune_sweep || (!rmax_sweep && !rmin_sweep && !alpha_sweep && !lbuild_sweep && !navm_sweep) {
            eprintln!("\n=== 2f: PruneStrategy A/B (r_max=32, alpha=1.2) ===");

            // DirectionalPrune (saturate=false)
            let cache_dir = "data/sift/graph_cache_dir_rmin4.bin";
            let cfg_dir = VamanaBuildConfig {
                alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
                max_iterations: 2, saturate: false,
                enable_layered_nav: true, nav_m: 32,
                prune_strategy: PruneStrategy::DirectionalPrune,
                ..Default::default()
            };
            std::env::set_var("RAVEN_RMIN_DIVISOR", "4");
            let t0 = Instant::now();
            let graph = load_or_build_graph(&train, dim, &cfg_dir, cache_dir);
            let build_time = t0.elapsed().as_secs_f64();
            let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
            eprintln!("  DirectionalPrune  QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                r.qps, r.recall, r.cv, build_time);
            out.push_str(&format!("2f,prune_dir,1,{:.0},{:.4},{:.1},{:.0}\n", r.qps, r.recall, r.cv, build_time));

            // RobustPrune (saturate=true)
            let cache_rob = "data/sift/graph_cache.bin";
            let cfg_rob = VamanaBuildConfig {
                alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
                max_iterations: 2, saturate: true,
                enable_layered_nav: true, nav_m: 32,
                prune_strategy: PruneStrategy::RobustPrune,
                ..Default::default()
            };
            let t0 = Instant::now();
            let graph = load_or_build_graph(&train, dim, &cfg_rob, cache_rob);
            let build_time = t0.elapsed().as_secs_f64();
            let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
            eprintln!("  RobustPrune+sat   QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                r.qps, r.recall, r.cv, build_time);
            out.push_str(&format!("2f,prune_rob_sat,1,{:.0},{:.4},{:.1},{:.0}\n", r.qps, r.recall, r.cv, build_time));

            // RobustPrune (saturate=false) �?对比 saturation 本身的影�?
            let cache_rob_nosat = "data/sift/sweep_robust_nosat.bin";
            let cfg_rob_nosat = VamanaBuildConfig {
                alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
                max_iterations: 2, saturate: false,
                enable_layered_nav: true, nav_m: 32,
                prune_strategy: PruneStrategy::RobustPrune,
                ..Default::default()
            };
            let t0 = Instant::now();
            let graph = load_or_build_graph(&train, dim, &cfg_rob_nosat, cache_rob_nosat);
            let build_time = t0.elapsed().as_secs_f64();
            let r = run_bench(&train, &graph, &test, dim, nq, &gt, gt_k, &sq8, 50, 8, 3);
            eprintln!("  RobustPrune       QPS={:>7.0}  recall={:.4}  CV={:.1}%  build={:.0}s",
                r.qps, r.recall, r.cv, build_time);
            out.push_str(&format!("2f,prune_rob_nosat,1,{:.0},{:.4},{:.1},{:.0}\n", r.qps, r.recall, r.cv, build_time));
        }

        let path = "tuning/phase2_graph_params.csv";
        std::fs::write(path, out).expect("write");
        eprintln!("\nPhase 2 结果写入 {}", path);
    }

    if !phase1 && !phase2 {
        eprintln!("用法�?);
        eprintln!("  cargo run --release --bin param_sweep -- --phase1           # 搜索参数扫描");
        eprintln!("  cargo run --release --bin param_sweep -- --phase2           # 图构建参数扫描（全部�?);
        eprintln!("  cargo run --release --bin param_sweep -- --phase2 --rmax-sweep    # 只扫 r_max");
        eprintln!("  cargo run --release --bin param_sweep -- --phase2 --rmin-sweep    # 只扫 r_min_divisor");
        eprintln!("  cargo run --release --bin param_sweep -- --phase2 --alpha-sweep   # 只扫 alpha");
        eprintln!("  cargo run --release --bin param_sweep -- --phase2 --lbuild-sweep  # 只扫 l_build");
        eprintln!("  cargo run --release --bin param_sweep -- --phase2 --navm-sweep    # 只扫 nav_m");
        eprintln!("  cargo run --release --bin param_sweep -- --phase2 --prune-sweep   # 只扫 prune_strategy");
    }
}
