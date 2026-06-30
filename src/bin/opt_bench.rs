//! OPT 系列实验 benchmark（v3 — 全优化参数支持）
//!
//! 设计目标：每轮 ~7s（100K queries），5 轮取中位数，方差 < ±3%
//! recall 分离计时：首轮算 recall，后续轮只测 QPS（用 sink 防优化消除）
//!
//! 用法：
//!   cargo run --release --bin opt_bench                              # 基线 RobustPrune ef=50
//!   cargo run --release --bin opt_bench --directional               # DirectionalPrune r_min=R/4 ef=50
//!   cargo run --release --bin opt_bench --directional --rmin-divisor=3  # r_min=R/3
//!   cargo run --release --bin opt_bench --directional --ef-sweep=50,52,55,58,60  # ef 扫描
//!   cargo run --release --bin opt_bench --directional --adaptive-ef             # + 自适应 ef (消融实验用，DirectionalPrune 图上无收益)
//!   cargo run --release --bin opt_bench --directional --rerank=5                 # rerank_factor=5
//!   cargo run --release --bin opt_bench --directional --po=4                     # prefetch offset=4

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{AdaptiveEfConfig, GraphSearcher, VamanaBuildConfig, VamanaGraph, PruneStrategy};
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

/// 单轮 benchmark 结果
struct RoundResult {
    qps: f64,
    recall: f64,
    elapsed_secs: f64,
}

/// 搜索配置（所有优化参数集中管理）
struct SearchConfig<'a> {
    train: &'a [f32],
    graph: &'a VamanaGraph,
    test: &'a [f32],
    dim: usize,
    nq: usize,
    gt: &'a [i32],
    gt_k: usize,
    k: usize,
    sq8: &'a SQ8Dataset,
    repeats: usize,
    weighted: bool,
    po: usize,
    rerank_factor: usize,
    adaptive_ef: Option<&'a AdaptiveEfConfig>,
}

/// 跑一轮：先算 recall（不计入时间），再纯测 QPS
fn run_round(ef: usize, cfg: &SearchConfig) -> RoundResult {
    let mut searcher = GraphSearcher::new(cfg.train, cfg.graph, ef);
    searcher.with_sq8(cfg.sq8);
    searcher.with_prefetch_offset(cfg.po);
    searcher.with_rerank_factor(cfg.rerank_factor);
    if let Some(ac) = cfg.adaptive_ef {
        searcher.with_adaptive_ef(ac.clone());
    }

    // ── Pass 0: recall（不计入 timing） ──
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..cfg.nq {
        let query = &cfg.test[q * cfg.dim..(q + 1) * cfg.dim];
        let result = if cfg.weighted {
            searcher.search_sq8_weighted(query, cfg.k)
        } else {
            searcher.search_sq8(query, cfg.k)
        };
        let gt_slice = &cfg.gt[q * cfg.gt_k..q * cfg.gt_k + cfg.k];
        for &g in gt_slice {
            if result.iter().any(|(id, _)| *id == g as u32) {
                hits += 1;
            }
        }
        total += cfg.k;
    }
    let recall = hits as f64 / total as f64;

    // ── Pass 1..N: 纯 QPS（sink 防优化消除） ──
    let mut sink: u64 = 0;
    let t0 = Instant::now();
    for _ in 0..cfg.repeats {
        for q in 0..cfg.nq {
            let query = &cfg.test[q * cfg.dim..(q + 1) * cfg.dim];
            let result = if cfg.weighted {
                searcher.search_sq8_weighted(query, cfg.k)
            } else {
                searcher.search_sq8(query, cfg.k)
            };
            sink = sink.wrapping_add(result[0].0 as u64);
        }
    }
    let dt = t0.elapsed();

    // 防编译器消除
    if sink == u64::MAX {
        eprintln!("impossible");
    }

    let total_queries = cfg.nq * cfg.repeats;
    RoundResult {
        qps: total_queries as f64 / dt.as_secs_f64(),
        recall,
        elapsed_secs: dt.as_secs_f64(),
    }
}

/// 解析 ef 列表：支持 --ef-sweep=50,52,55,58,60 和 --ef=55
fn parse_ef_list() -> Vec<usize> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(sweep) = args.iter().find_map(|a| a.strip_prefix("--ef-sweep=")) {
        sweep
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    } else {
        let ef = args
            .iter()
            .find_map(|a| a.strip_prefix("--ef=").and_then(|v| v.parse::<usize>().ok()))
            .unwrap_or(50);
        vec![ef]
    }
}

fn main() {
    let weighted = std::env::args().any(|a| a == "--weighted");
    let directional = std::env::args().any(|a| a == "--directional");
    // AdaptiveEf 在 DirectionalPrune 图上无 measurable 收益（avg_ef=48.9 ≈ 固定 50）
    // 原因：分层导航 + DirectionalPrune 把入口距离压缩到极窄范围，幂律变换后几乎不改变 ef 分配
    // 仍可通过 --adaptive-ef 手动启用进行消融实验
    let use_adaptive_ef = std::env::args().any(|a| a == "--adaptive-ef");
    let rmin_divisor = std::env::args()
        .find_map(|a| a.strip_prefix("--rmin-divisor=").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(4);
    let rerank_factor = std::env::args()
        .find_map(|a| a.strip_prefix("--rerank=").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(3);
    let po = std::env::args()
        .find_map(|a| a.strip_prefix("--po=").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(8);
    let ef_list = parse_ef_list();

    // 设置环境变量，让 DirectionalPruneConfig::from_params 读取
    std::env::set_var("RAVEN_RMIN_DIVISOR", rmin_divisor.to_string());

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    let k = 10usize;
    const REPEATS: usize = 10; // 10K × 10 = 100K queries/轮 ≈ 7s
    const ROUNDS: usize = 5;

    let prune_name = if directional {
        format!("Directional(r_min=R/{})", rmin_divisor)
    } else {
        "Robust".to_string()
    };

    let opt_tags = {
        let mut tags: Vec<String> = vec![];
        if use_adaptive_ef { tags.push("adef".to_string()); }
        if rerank_factor != 3 { tags.push(format!("rr={}", rerank_factor)); }
        if po != 8 { tags.push(format!("po={}", po)); }
        if tags.is_empty() { "none".to_string() } else { tags.join(",") }
    };

    eprintln!(
        "=== OPT Benchmark v3 (SQ8 {}, ef_sweep={:?}, k={}, prune={}, opts=[{}]) ===",
        if weighted { "WEIGHTED" } else { "RAW" },
        ef_list, k, prune_name, opt_tags
    );
    eprintln!(
        "data: n={}, dim={}, nq={}, repeats={}, rounds={}",
        n, dim, nq, REPEATS, ROUNDS
    );

    // 建图（带缓存：首次建图后存盘，后续直接加载）
    // 缓存路径包含 rmin_divisor，避免不同配置混用
    let graph_path = if directional {
        format!("data/sift/graph_cache_dir_rmin{}.bin", rmin_divisor)
    } else {
        "data/sift/graph_cache.bin".to_string()
    };
    let graph_path = std::path::Path::new(&graph_path);
    let graph = if graph_path.exists() {
        eprintln!("loading cached graph ({})...", prune_name);
        let t0 = Instant::now();
        let g = VamanaGraph::load(graph_path).expect("load graph cache");
        eprintln!("load: {:.1}s", t0.elapsed().as_secs_f64());
        g
    } else {
        eprintln!("building graph (first run, prune={})...", prune_name);
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 200,
            r_max: 32,
            r_soft: 48,
            max_iterations: 2,
            saturate: !directional,
            enable_layered_nav: true,
            nav_m: 32,
            prune_strategy: if directional {
                PruneStrategy::DirectionalPrune
            } else {
                PruneStrategy::RobustPrune
            },
            ..Default::default()
        };
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        eprintln!("build: {:.1}s", t0.elapsed().as_secs_f64());
        let _ = g.save(graph_path);
        eprintln!("graph cached to {}", graph_path.display());
        g
    };

    // SQ8 编码
    let sq8 = SQ8Dataset::build(&train, dim);

    // 自适应 ef 配置（若启用）
    let adaptive_config = if use_adaptive_ef {
        if let Some(nav) = graph.layered_nav() {
            eprintln!("building adaptive ef config...");
            let t0 = Instant::now();
            // 与 raven_ann_bench.rs 一致的参数
            let ac = AdaptiveEfConfig::build_with_layered_nav(
                &train, dim, nav, 40, 75, 3.0,
            );
            eprintln!("adaptive ef build: {:.3}s", t0.elapsed().as_secs_f64());
            Some(ac)
        } else {
            eprintln!("warning: no layered nav, skipping adaptive ef");
            None
        }
    } else {
        None
    };

    let search_cfg = SearchConfig {
        train: &train,
        graph: &graph,
        test: &test,
        dim,
        nq,
        gt: &gt,
        gt_k,
        k,
        sq8: &sq8,
        repeats: REPEATS,
        weighted,
        po,
        rerank_factor,
        adaptive_ef: adaptive_config.as_ref(),
    };

    // ── ef sweep：对每个 ef 值跑 warmup + 5 rounds ──
    eprintln!("\n--- Sweep results (prune={}, opts=[{}]) ---", prune_name, opt_tags);
    eprintln!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>6}  {:>8}",
        "ef", "median", "mean", "min", "CV%", "recall"
    );

    for &ef in &ef_list {
        // warmup（每个 ef 都需要 warmup，因为 searcher 的 ef 容量不同）
        let _ = run_round(ef, &search_cfg);

        let mut rounds = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let r = run_round(ef, &search_cfg);
            rounds.push(r);
        }

        let mut qps_vals: Vec<f64> = rounds.iter().map(|r| r.qps).collect();
        qps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = qps_vals[ROUNDS / 2];
        let mean = qps_vals.iter().sum::<f64>() / ROUNDS as f64;
        let min = qps_vals[0];
        let variance = qps_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ROUNDS as f64;
        let cv = variance.sqrt() / mean * 100.0;
        let recall = rounds[0].recall;

        eprintln!(
            "{:>6}  {:>8.0}  {:>8.0}  {:>8.0}  {:>5.1}%  {:>8.4}",
            ef, median, mean, min, cv, recall
        );
    }
}
