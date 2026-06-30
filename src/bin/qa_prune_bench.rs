//! QuantAwarePrune benchmark — 测量化感知剪枝在 SIFT-1M 上的实际效果
//!
//! 对比 DirectionalPrune(生产基线) vs QuantAwarePrune（β=0.3）的 QPS/recall
//! 配置与生产一致：saturate=false, QA-aware final prune
//!
//! 用法：cargo run --release --bin qa_prune_bench

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{
    AdaptiveEfConfig, GraphSearcher, PruneStrategy, QuantAwarePruneConfig, VamanaBuildConfig,
    VamanaGraph,
};
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

fn run_bench(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    sq8: &SQ8Dataset,
    adaptive_config: Option<&AdaptiveEfConfig>,
) -> (f64, f64, f64) {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_sq8(sq8);
    searcher.with_prefetch_offset(8);
    searcher.with_rerank_factor(3);
    if let Some(ac) = adaptive_config {
        searcher.with_adaptive_ef(ac.clone());
    }

    // recall
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search_sq8(query, k);
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
        total += k;
    }
    let recall = hits as f64 / total as f64;

    // QPS
    let mut sink: u64 = 0;
    let mut qps_vals = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        for _ in 0..REPEATS {
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let result = searcher.search_sq8(query, k);
                sink = sink.wrapping_add(result[0].0 as u64);
            }
        }
        let dt = t0.elapsed();
        qps_vals.push(nq as f64 * REPEATS as f64 / dt.as_secs_f64());
    }
    if sink == u64::MAX { eprintln!("impossible"); }

    qps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = qps_vals[ROUNDS / 2];
    let mean = qps_vals.iter().sum::<f64>() / ROUNDS as f64;
    let cv = (qps_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ROUNDS as f64).sqrt() / mean * 100.0;
    (median, recall, cv)
}

fn main() {
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10usize;
    let ef = 50usize;

    eprintln!("=== QuantAwarePrune Benchmark (SIFT-1M, ef={}, k={}) ===", ef, k);
    eprintln!("data: n={}, dim={}, nq={}", n, dim, nq);
    eprintln!("config: saturate=false, DirectionalPrune baseline, QA-aware final prune");

    // SQ8 编码（先构建，用于 error_fn 和搜索）
    let sq8 = SQ8Dataset::build(&train, dim);

    // 预计算每个向量的 SQ8 量化误差
    eprintln!("computing per-vector SQ8 quantization error...");
    let t0 = Instant::now();
    let per_vec_error: Vec<f32> = (0..n)
        .map(|i| {
            let orig = &train[i * dim..(i + 1) * dim];
            let decoded = sq8.params.decode(sq8.code(i));
            let mut err = 0.0f32;
            for d in 0..dim {
                let diff = orig[d] - decoded[d];
                err += diff * diff;
            }
            err
        })
        .collect();
    eprintln!("error computation: {:.1}s", t0.elapsed().as_secs_f64());

    let error_fn = move |u: u32, v: u32| -> f32 {
        (per_vec_error[u as usize] + per_vec_error[v as usize]) * 0.5
    };

    // ── 加载 DirectionalPrune 基线图（生产配置，已缓存） ──
    let robust_path = std::path::Path::new("data/sift/graph_cache_dir_rmin4.bin");
    let robust_graph = if robust_path.exists() {
        eprintln!("loading cached DirectionalPrune graph (r_min=R/4, saturate=false)...");
        VamanaGraph::load(robust_path).expect("load")
    } else {
        eprintln!("building DirectionalPrune graph (saturate=false)...");
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
            max_iterations: 2, saturate: false,
            enable_layered_nav: true, nav_m: 16,
            prune_strategy: PruneStrategy::DirectionalPrune,
            ..Default::default()
        };
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        eprintln!("build: {:.1}s", t0.elapsed().as_secs_f64());
        let _ = g.save(robust_path);
        g
    };

    // ── 构建 QuantAwarePrune 图（β=0.3, saturate=false, QA final prune） ──
    let qa_path = std::path::Path::new("data/sift/graph_cache_qa_v2_beta03.bin");
    let qa_graph = if qa_path.exists() {
        eprintln!("loading cached QuantAwarePrune graph (v2, β=0.3)...");
        VamanaGraph::load(qa_path).expect("load")
    } else {
        eprintln!("building QuantAwarePrune graph (v2, β=0.3, QA final prune)...");
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
            max_iterations: 2, saturate: false,
            enable_layered_nav: true, nav_m: 16,
            prune_strategy: PruneStrategy::DirectionalPrune,
            ..Default::default()
        };
        let qa_config = QuantAwarePruneConfig {
            alpha: 1.2, beta: 0.3, r_max: 32, ..Default::default()
        };
        let g = VamanaGraph::build_with_quant_aware_prune(
            &train, dim, &config, &qa_config, error_fn, &mut rng,
        );
        eprintln!("build: {:.1}s", t0.elapsed().as_secs_f64());
        let _ = g.save(qa_path);
        g
    };

    // 自适应 ef 配置
    let adaptive_config = if let Some(nav) = robust_graph.layered_nav() {
        Some(AdaptiveEfConfig::build_with_layered_nav(&train, dim, nav, 40, 75, 3.0))
    } else { None };

    // ── Benchmark DirectionalPrune（生产基线） ──
    eprintln!("\n--- DirectionalPrune (saturate=false) + SQ8 + AdaptiveEf ---");
    // warmup
    let _ = run_bench(&train, &robust_graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, adaptive_config.as_ref());
    let (robust_qps, robust_recall, robust_cv) = run_bench(
        &train, &robust_graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, adaptive_config.as_ref(),
    );
    eprintln!("  QPS={:.0}  recall={:.4}  CV={:.1}%", robust_qps, robust_recall, robust_cv);

    // ── Benchmark QuantAwarePrune ──
    let qa_adaptive_config = if let Some(nav) = qa_graph.layered_nav() {
        Some(AdaptiveEfConfig::build_with_layered_nav(&train, dim, nav, 40, 75, 3.0))
    } else { None };

    eprintln!("\n--- QuantAwarePrune(v2, β=0.3, QA final prune) + SQ8 + AdaptiveEf ---");
    // warmup
    let _ = run_bench(&train, &qa_graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, qa_adaptive_config.as_ref());
    let (qa_qps, qa_recall, qa_cv) = run_bench(
        &train, &qa_graph, &test, dim, nq, &gt, gt_k, k, ef, &sq8, qa_adaptive_config.as_ref(),
    );
    eprintln!("  QPS={:.0}  recall={:.4}  CV={:.1}%", qa_qps, qa_recall, qa_cv);

    // ── Summary ──
    eprintln!("\n=== Summary ===");
    eprintln!("{:<25} {:>8} {:>8} {:>6}", "", "QPS", "recall", "CV%");
    eprintln!("{:<25} {:>8.0} {:>8.4} {:>5.1}%", "DirectionalPrune", robust_qps, robust_recall, robust_cv);
    eprintln!("{:<25} {:>8.0} {:>8.4} {:>5.1}%", "QuantAware(v2,β=0.3)", qa_qps, qa_recall, qa_cv);
    let qps_delta = (qa_qps - robust_qps) / robust_qps * 100.0;
    let recall_delta = (qa_recall - robust_recall) * 100.0;
    eprintln!("QPS delta: {:+.1}%", qps_delta);
    eprintln!("recall delta: {:+.4}pp", recall_delta);
}
