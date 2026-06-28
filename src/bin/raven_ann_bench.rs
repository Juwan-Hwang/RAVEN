//! ann-benchmarks 接入二进制
//!
//! ann-benchmarks 评测口径（源码审计确认）：
//!   - runner.py:38  batch = False（硬编码，即使传 --batch 也无效）
//!   - runner.py:136 逐条顺序查询 [single_query(x) for x in X_test]
//!   - metrics.py:71  QPS = 1.0 / best_search_time（avg 单条耗时）
//!   - hnswlib 显式 set_num_threads(1)，Glass 用 prepare_query 逐条
//!   - Docker --parallelism 仅控制算法间并行，非查询内并行
//!
//! 默认配置：单线程 + SQ8 + 自适应 ef（ann-benchmarks 逐条顺序查询口径）
//!
//! 默认启用：
//!   1. 分层导航 (LayeredNavigation) — 建图时构建
//!   2. Two-Pass Prefetch (po=8) — GraphSearcher 默认
//!   3. SQ8 标量量化 — 1.47x QPS，recall 几乎无损
//!   4. 自适应 ef (gamma=3.0, min=40, max=75) — Pareto 最优 +11.3%
//!   5. degrees 数组 O(1) neighbors() — 建图路径加速，查询路径 +0.2%（噪声）
//!   6. target-cpu=native — .cargo/config.toml 全局生效
//!
//! 可选标志：
//!   --no-sq8           禁用 SQ8，回退 f32 全精度
//!   --no-adaptive-ef   禁用自适应 ef，用固定 ef
//!   --multithread      启用多线程 batch_search（ann-benchmarks 不使用）
//!   --threads N        指定线程数（默认全部核心）
//!
//! 数据格式（由 Python wrapper 准备）：
//!   train.bin: [n × dim] f32 连续存储
//!   test.bin:  [nq × dim] f32 连续存储
//!   neighbors.bin: [nq × k] i32 连续存储（ground truth）
//!
//! 用法：
//!   raven_ann_bench --train train.bin --test test.bin --neighbors neighbors.bin \
//!     --dim 128 --n 10000 --nq 100 --k 10 \
//!     --alpha 1.2 --l-build 200 --r-max 32 --ef-search 50
//!
//! 可选：
//!   --save index.bin    构建后保存索引到文件
//!   --load index.bin    从文件加载索引（跳过构建）
//!
//! 输出（JSON 到 stdout）：
//!   {"build_time_s": ..., "query_time_s": ..., "qps": ..., "recall@10": ...}

use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, AdaptiveEfConfig};
use raven::quant::SQ8Dataset;
use raven::build::ChaCha8Rng;
use raven::memory::serialize::Serializable;

/// 自适应 ef 最佳参数（密集参数扫描 Pareto 最优，gamma=3.0）
/// recall=0.9660, QPS=12952 (+11.3% vs fixed ef=50)
const ADAPTIVE_EF_MIN: usize = 40;
const ADAPTIVE_EF_MAX: usize = 75;
const ADAPTIVE_EF_GAMMA: f32 = 3.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut train_path = String::new();
    let mut test_path = String::new();
    let mut neighbors_path = String::new();
    let mut dim: usize = 0;
    let mut n: usize = 0;
    let mut nq: usize = 0;
    let mut k: usize = 10;
    let mut alpha: f32 = 1.2;
    let mut l_build: usize = 200;
    let mut r_max: usize = 32;
    let mut max_iterations: usize = 2;
    let mut ef_search: usize = 50;
    let mut output_path = String::new();
    let mut save_path = String::new();
    let mut load_path = String::new();

    // 优化控制标志（默认符合 ann-benchmarks 单线程口径）
    let mut use_sq8 = true;
    let mut use_adaptive_ef = true;   // Pareto 最优 γ3(40,75): +11.3% QPS, recall 无损
    let mut use_multithread = false;   // ann-benchmarks 逐条顺序查询
    let mut num_threads: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => { i += 1; train_path = args[i].clone(); }
            "--test" => { i += 1; test_path = args[i].clone(); }
            "--neighbors" => { i += 1; neighbors_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--save" => { i += 1; save_path = args[i].clone(); }
            "--load" => { i += 1; load_path = args[i].clone(); }
            "--dim" => { i += 1; dim = args[i].parse().expect("invalid dim"); }
            "--n" => { i += 1; n = args[i].parse().expect("invalid n"); }
            "--nq" => { i += 1; nq = args[i].parse().expect("invalid nq"); }
            "--k" => { i += 1; k = args[i].parse().expect("invalid k"); }
            "--alpha" => { i += 1; alpha = args[i].parse().expect("invalid alpha"); }
            "--l-build" => { i += 1; l_build = args[i].parse().expect("invalid l_build"); }
            "--r-max" => { i += 1; r_max = args[i].parse().expect("invalid r_max"); }
            "--max-iterations" => { i += 1; max_iterations = args[i].parse().expect("invalid max_iterations"); }
            "--ef-search" => { i += 1; ef_search = args[i].parse().expect("invalid ef_search"); }
            "--no-sq8" => { use_sq8 = false; }
            "--no-adaptive-ef" => { use_adaptive_ef = false; }
            "--multithread" => { use_multithread = true; }
            "--threads" => { i += 1; num_threads = Some(args[i].parse().expect("invalid threads")); }
            "--help" | "-h" => { print_help(); return; }
            _ => { eprintln!("unknown argument: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    // 读取训练数据（load 模式下仍需向量用于查询）
    let train: Vec<f32> = if !train_path.is_empty() {
        let train_bytes = std::fs::read(&train_path).expect("failed to read train file");
        assert_eq!(train_bytes.len(), n * dim * 4, "train file size mismatch");
        bytemuck_cast(&train_bytes)
    } else {
        Vec::new()
    };

    // 读取测试数据
    let test: Vec<f32> = if test_path.is_empty() || nq == 0 {
        Vec::new()
    } else {
        let test_bytes = std::fs::read(&test_path).expect("failed to read test file");
        assert_eq!(test_bytes.len(), nq * dim * 4, "test file size mismatch");
        bytemuck_cast(&test_bytes)
    };

    // 读取 ground truth
    let ground_truth: Vec<i32> = if neighbors_path.is_empty() {
        Vec::new()
    } else {
        let nb_bytes = std::fs::read(&neighbors_path).expect("failed to read neighbors file");
        assert_eq!(nb_bytes.len(), nq * k * 4, "neighbors file size mismatch");
        bytemuck_cast(&nb_bytes)
    };

    eprintln!("RAVEN ann-benchmarks runner (全优化叠加版)");
    eprintln!("  dim={}, n={}, nq={}, k={}", dim, n, nq, k);
    eprintln!("  alpha={}, l_build={}, r_max={}, max_iterations={}, ef_search={}", alpha, l_build, r_max, max_iterations, ef_search);
    eprintln!("  optimizations: sq8={}, adaptive_ef={}, multithread={}", use_sq8, use_adaptive_ef, use_multithread);

    // 构建或加载索引
    let (graph, build_time) = if !load_path.is_empty() {
        eprintln!("loading index from {}...", load_path);
        let load_start = Instant::now();
        let path = std::path::Path::new(&load_path);
        let g = VamanaGraph::load(path).expect("failed to load index");
        let t = load_start.elapsed();
        eprintln!("  load time: {:.3}s", t.as_secs_f64());
        (g, t)
    } else {
        eprintln!("building index...");
        let mut rng = ChaCha8Rng::new();
        let config = VamanaBuildConfig {
            alpha,
            l_build,
            r_max,
            r_soft: (r_max as f32 * 1.5) as usize,
            max_iterations,
            saturate: true,
            enable_layered_nav: true,
            nav_m: 16,
            ..Default::default()
        };
        let build_start = Instant::now();
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        let t = build_start.elapsed();
        eprintln!("  build time: {:.3}s", t.as_secs_f64());
        (g, t)
    };

    // 保存索引（可选）
    if !save_path.is_empty() {
        eprintln!("saving index to {}...", save_path);
        let path = std::path::Path::new(&save_path);
        graph.save(path).expect("failed to save index");
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        eprintln!("  saved {} bytes", file_size);
    }

    // 查询
    if test.is_empty() || nq == 0 {
        // 仅构建/加载，不查询
        let result = serde_json::json!({
            "build_time_s": build_time.as_secs_f64(),
            "n": n,
            "dim": dim,
            "alpha": alpha,
            "l_build": l_build,
            "r_max": r_max,
            "max_iterations": max_iterations,
        });
        println!("{}", result);
        return;
    }

    // ── 优化层 1: SQ8 量化 ──
    let sq8 = if use_sq8 {
        eprintln!("encoding SQ8...");
        let sq8_start = Instant::now();
        let s = SQ8Dataset::build(&train, dim);
        eprintln!("  SQ8 encode: {:.3}s ({} MB)", sq8_start.elapsed().as_secs_f64(), s.codes.len() / 1_000_000);
        Some(s)
    } else {
        None
    };

    // ── 优化层 2: 自适应 ef ──
    let adaptive_config = if use_adaptive_ef {
        if let Some(nav) = graph.layered_nav() {
            eprintln!("building adaptive ef config (gamma={})...", ADAPTIVE_EF_GAMMA);
            let ac_start = Instant::now();
            let ac = AdaptiveEfConfig::build_with_layered_nav(
                &train, dim, nav, ADAPTIVE_EF_MIN, ADAPTIVE_EF_MAX, ADAPTIVE_EF_GAMMA);
            eprintln!("  adaptive ef build: {:.3}s", ac_start.elapsed().as_secs_f64());
            Some(ac)
        } else {
            eprintln!("warning: no layered nav, skipping adaptive ef");
            None
        }
    } else {
        None
    };

    eprintln!("running {} queries...", nq);

    // ── 查询执行 ──
    let (query_time, results): (std::time::Duration, Vec<Vec<u32>>) = if use_multithread {
        // 多线程批量搜索
        let queries: Vec<&[f32]> = (0..nq).map(|q| &test[q * dim..(q + 1) * dim]).collect();

        // 配置线程池
        if let Some(nt) = num_threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .expect("build thread pool");
            pool.install(|| {
                run_batch(&train, &graph, &sq8, &adaptive_config, &queries, k, ef_search)
            })
        } else {
            run_batch(&train, &graph, &sq8, &adaptive_config, &queries, k, ef_search)
        }
    } else {
        // 单线程搜索
        let mut searcher = GraphSearcher::new(&train, &graph, ef_search);
        if let Some(ref sq8) = sq8 {
            searcher.with_sq8(sq8);
        }
        if let Some(ref ac) = adaptive_config {
            searcher.with_adaptive_ef(ac.clone());
        }

        let query_start = Instant::now();
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = if sq8.is_some() {
                searcher.search_sq8(query, k)
            } else {
                searcher.search(query, k)
            };
            results.push(result.iter().map(|(id, _)| *id).collect());
        }
        (query_start.elapsed(), results)
    };

    let qps = nq as f64 / query_time.as_secs_f64();
    eprintln!("  query time: {:.3}s ({:.0} QPS)", query_time.as_secs_f64(), qps);

    // 输出邻居 ID 到文件（raw binary, i32）
    if !output_path.is_empty() {
        let flat: Vec<i32> = results.iter()
            .flat_map(|r| r.iter().map(|&id| id as i32))
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&flat);
        std::fs::write(&output_path, bytes)
            .expect("failed to write output file");
        eprintln!("  neighbors written to {}", output_path);
    }

    // 计算 recall@k
    let recall = if !ground_truth.is_empty() {
        let mut hits = 0usize;
        for q in 0..nq {
            let gt = &ground_truth[q * k..(q + 1) * k];
            let found = &results[q];
            for &g in gt {
                if found.contains(&(g as u32)) {
                    hits += 1;
                }
            }
        }
        hits as f64 / (nq * k) as f64
    } else {
        -1.0
    };

    if recall >= 0.0 {
        eprintln!("  recall@{}: {:.4}", k, recall);
    }

    let result = serde_json::json!({
        "build_time_s": build_time.as_secs_f64(),
        "query_time_s": query_time.as_secs_f64(),
        "qps": qps,
        "recall@k": recall,
        "k": k,
        "n": n,
        "nq": nq,
        "dim": dim,
        "alpha": alpha,
        "l_build": l_build,
        "r_max": r_max,
        "max_iterations": max_iterations,
        "ef_search": ef_search,
        "sq8": use_sq8,
        "adaptive_ef": use_adaptive_ef,
        "multithread": use_multithread,
    });
    println!("{}", result);
}

/// 批量搜索（多线程，自动选择 SQ8/f32 + 自适应 ef/固定 ef）
fn run_batch(
    train: &[f32],
    graph: &VamanaGraph,
    sq8: &Option<SQ8Dataset>,
    adaptive_config: &Option<AdaptiveEfConfig>,
    queries: &[&[f32]],
    k: usize,
    ef_search: usize,
) -> (std::time::Duration, Vec<Vec<u32>>) {
    let searcher = GraphSearcher::new(train, graph, ef_search);
    let mut searcher = searcher;
    if let Some(ref sq8) = sq8 {
        searcher.with_sq8(sq8);
    }
    if let Some(ref ac) = adaptive_config {
        searcher.with_adaptive_ef(ac.clone());
    }

    // warmup
    let warmup_n = queries.len().min(100);
    searcher.batch_search(&queries[..warmup_n], k);

    let query_start = Instant::now();
    let batch_results = searcher.batch_search(queries, k);
    let query_time = query_start.elapsed();

    let results: Vec<Vec<u32>> = batch_results.into_iter()
        .map(|r| r.into_iter().map(|(id, _)| id).collect())
        .collect();

    (query_time, results)
}

/// 零拷贝 byte→f32/i32 转换
fn bytemuck_cast<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    bytemuck::cast_slice(bytes).to_vec()
}

fn print_help() {
    println!("RAVEN ann-benchmarks runner (全优化叠加版)");
    println!();
    println!("用法:");
    println!("  raven_ann_bench --train <path> --test <path> --neighbors <path> \\");
    println!("    --dim <N> --n <N> --nq <N> --k <N> \\");
    println!("    --alpha <F> --l-build <N> --r-max <N> --ef-search <N>");
    println!();
    println!("优化控制（默认: SQ8 开 / 自适应 ef 开 / 多线程关）:");
    println!("  --no-sq8             禁用 SQ8 量化");
    println!("  --no-adaptive-ef     禁用自适应 ef");
    println!("  --multithread        启用多线程（ann-benchmarks 不使用）");
    println!("  --threads N          指定线程数");
    println!();
    println!("可选:");
    println!("  --save <path>        构建后保存索引");
    println!("  --load <path>        从文件加载索引");
    println!("  --output <path>      输出邻居 ID 到文件");
    println!("  --max-iterations <N> Vamana 构建迭代轮数（默认 2）");
}
