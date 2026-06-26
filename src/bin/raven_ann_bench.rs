//! ann-benchmarks 接入二进制
//!
//! 设计文档 Week 3-4：接入 ann-benchmarks Docker，跑出第一条真实 Pareto 曲线
//! 设计文档第六层模式一：ann_benchmarks/algorithms/yourlib/
//!
//! 数据格式（由 Python wrapper 准备）：
//!   train.bin: [n × dim] f32 连续存储
//!   test.bin:  [nq × dim] f32 连续存储
//!   neighbors.bin: [nq × k] i32 连续存储（ground truth，用于 recall 计算）
//!
//! 用法：
//!   raven_ann_bench --train train.bin --test test.bin --neighbors neighbors.bin \
//!     --dim 128 --n 10000 --nq 100 --k 10 \
//!     --alpha 1.2 --l-build 200 --r-max 64 --ef-search 200
//!
//! 可选：
//!   --save index.bin    构建后保存索引到文件
//!   --load index.bin    从文件加载索引（跳过构建）
//!
//! 输出（JSON 到 stdout）：
//!   {"build_time_s": ..., "query_time_s": ..., "qps": ..., "recall@10": ...}

use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::memory::serialize::Serializable;

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
    let mut r_max: usize = 64;
    let mut max_iterations: usize = 2; // Canonical Config: max_iterations=2 (Vamana two-pass)
    let mut ef_search: usize = 200;
    let mut output_path = String::new();
    let mut save_path = String::new();
    let mut load_path = String::new();

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

    eprintln!("RAVEN ann-benchmarks runner");
    eprintln!("  dim={}, n={}, nq={}, k={}", dim, n, nq, k);
    eprintln!("  alpha={}, l_build={}, r_max={}, max_iterations={}, ef_search={}", alpha, l_build, r_max, max_iterations, ef_search);

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

    eprintln!("running {} queries...", nq);
    let mut searcher = GraphSearcher::new(&train, &graph, ef_search);
    let query_start = Instant::now();
    let mut results: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        results.push(result.iter().map(|(id, _)| *id).collect());
    }
    let query_time = query_start.elapsed();
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
    });
    println!("{}", result);
}

/// 零拷贝 byte→f32/i32 转换
fn bytemuck_cast<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    bytemuck::cast_slice(bytes).to_vec()
}

fn print_help() {
    println!("RAVEN ann-benchmarks runner");
    println!();
    println!("用法:");
    println!("  raven_ann_bench --train <path> --test <path> --neighbors <path> \\");
    println!("    --dim <N> --n <N> --nq <N> --k <N> \\");
    println!("    --alpha <F> --l-build <N> --r-max <N> --ef-search <N>");
    println!();
    println!("可选:");
    println!("  --save <path>         构建后保存索引");
    println!("  --load <path>         从文件加载索引（跳过构建）");
    println!("  --output <path>       输出邻居 ID 到文件");
    println!("  --max-iterations <N>  Vamana 构建迭代轮数（默认 2）");
}
