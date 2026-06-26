//! recall + avg_visited 扫描工具（SIFT1M）
//!
//! 对 CANONICAL(200/64/2) 和 GLASS-COMP(200/32/2) 两张图，
//! 各跑 ef_search ∈ {50, 75, 100, 150, 200, 300} 扫描，
//! 输出 (recall, QPS, avg_visited) 三元组曲线。
//!
//! 用法：
//!   cargo run --release --bin quick_recall_check              # 双图全扫描
//!   cargo run --release --bin quick_recall_check -- canonical # 仅 CANONICAL
//!   cargo run --release --bin quick_recall_check -- glass     # 仅 GLASS-COMP
//!
//! 退出码 0 = recall OK, 1 = recall BAD

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

/// ef_search 扫描列表（文档 §Canonical Build Config 定义）
const EF_SEARCH_LIST: &[usize] = &[50, 75, 100, 150, 200, 300];

/// 单次扫描结果
struct ScanResult {
    ef_search: usize,
    recall: f64,
    qps: f64,
    avg_visited: f64,
    p50_visited: usize,
    p95_visited: usize,
    p99_visited: usize,
    max_visited: usize,
}

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

/// 对一张图跑 ef_search 扫描
fn run_scan(
    name: &str,
    train: &[f32],
    dim: usize,
    graph: &VamanaGraph,
    test: &[f32],
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
) -> Vec<ScanResult> {
    println!("\n--- {} ef_search 扫描 ---", name);
    println!(
        "{:>10} {:>10} {:>12} {:>14} {:>8} {:>8} {:>8} {:>8}",
        "ef_search", "recall@10", "QPS", "avg_visited", "p50", "p95", "p99", "max"
    );

    let mut results = Vec::with_capacity(EF_SEARCH_LIST.len());

    for &ef_search in EF_SEARCH_LIST {
        let mut searcher = GraphSearcher::new(train, graph, ef_search);

        let mut hits = 0usize;
        let mut total = 0usize;
        let mut total_visited = 0usize;
        let mut visited_counts: Vec<usize> = Vec::with_capacity(nq);

        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = searcher.search(query, k);
            let vc = searcher.last_visited_count();
            total_visited += vc;
            visited_counts.push(vc);

            let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
            let gt_slice = &gt[q * gt_k..q * gt_k + k];
            for &g in gt_slice {
                if found.contains(&(g as u32)) {
                    hits += 1;
                }
            }
            total += k;
        }
        let query_time = t0.elapsed();

        let recall = hits as f64 / total as f64;
        let qps = nq as f64 / query_time.as_secs_f64();
        let avg_visited = total_visited as f64 / nq as f64;

        visited_counts.sort_unstable();
        let p50 = visited_counts[nq / 2];
        let p95 = visited_counts[(nq as f64 * 0.95) as usize];
        let p99 = visited_counts[(nq as f64 * 0.99) as usize];
        let max_vc = visited_counts[nq - 1];

        println!(
            "{:>10} {:>10.4} {:>12.0} {:>14.1} {:>8} {:>8} {:>8} {:>8}",
            ef_search, recall, qps, avg_visited, p50, p95, p99, max_vc
        );

        results.push(ScanResult {
            ef_search,
            recall,
            qps,
            avg_visited,
            p50_visited: p50,
            p95_visited: p95,
            p99_visited: p99,
            max_visited: max_vc,
        });
    }

    results
}

fn build_and_scan(
    name: &str,
    alpha: f32,
    l_build: usize,
    r_max: usize,
    r_soft: usize,
    max_iterations: usize,
    train: &[f32],
    dim: usize,
    test: &[f32],
    nq: usize,
    gt: &[i32],
    gt_k: usize,
) -> Vec<ScanResult> {
    println!("\n=== {} ===", name);
    println!(
        "构建参数: alpha={}, l_build={}, r_max={}, r_soft={}, max_iterations={}",
        alpha, l_build, r_max, r_soft, max_iterations
    );

    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha,
        l_build,
        r_max,
        r_soft,
        max_iterations,
        saturate: true,
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图: {:.1}s", build_time);

    // 度数分布诊断
    let stats = graph.degree_stats();
    println!(
        "[degree] mean={:.1} p95={} p99={} max={} isolated={} overflow={:.4}% overflow_count={}",
        stats.mean_degree, stats.p95_degree, stats.p99_degree,
        stats.max_degree, stats.isolated_nodes, stats.overflow_ratio, stats.overflow_count
    );
    // 采样前 10 个节点的度数
    let mut sample_degrees = Vec::new();
    for i in 0..10.min(graph.len()) {
        sample_degrees.push(graph.neighbors(i as u32).len());
    }
    println!("[degree] sample[0..10]: {:?}", sample_degrees);

    run_scan(name, train, dim, &graph, test, nq, gt, gt_k, 10)
}

/// 版本横幅：编译时注入，运行时打印
fn print_banner() {
    let pkg_ver = env!("CARGO_PKG_VERSION");
    let git_hash = env!("RAVEN_GIT_HASH");
    let build_ts = env!("RAVEN_BUILD_TS");
    // 多行横幅，终端中一眼可见
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  RAVEN v{}  git:{}  build:{}  ║",
             pkg_ver, git_hash, build_ts);
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run_canonical = args.is_empty()
        || args.iter().any(|a| a == "canonical" || a == "all");
    let run_glass = args.is_empty()
        || args.iter().any(|a| a == "glass" || a == "all");

    print_banner();
    println!("=== RAVEN SIFT1M recall + avg_visited 扫描 ===");
    println!("ef_search 列表: {:?}", EF_SEARCH_LIST);

    let (mut train, dim, _n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    // SIFT 数据归一化（与历史一致）
    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    println!("数据: n={}, dim={}, nq={}", _n, dim, nq);

    let mut all_results: Vec<(&str, Vec<ScanResult>)> = Vec::new();

    if run_canonical {
        let results = build_and_scan(
            "CANONICAL",
            1.2,
            200,
            64,
            96,
            2,
            &train,
            dim,
            &test,
            nq,
            &gt,
            gt_k,
        );
        all_results.push(("CANONICAL", results));
    }

    if run_glass {
        let results = build_and_scan(
            "GLASS-COMP",
            1.2,
            200,
            32,
            48,
            2,
            &train,
            dim,
            &test,
            nq,
            &gt,
            gt_k,
        );
        all_results.push(("GLASS-COMP", results));
    }

    // 汇总：找 recall≈0.95 的最近点
    println!("\n=== 汇总：recall≈0.95 锚点对比 ===");
    println!(
        "{:>12} {:>10} {:>10} {:>12} {:>14} {:>8} {:>8} {:>8} {:>8}",
        "config", "ef_search", "recall", "QPS", "avg_visited", "p50", "p95", "p99", "max"
    );

    for (name, results) in &all_results {
        // 找最接近 recall=0.95 的扫描点
        let best = results
            .iter()
            .min_by(|a, b| {
                let da = (a.recall - 0.95).abs();
                let db = (b.recall - 0.95).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        println!(
            "{:>12} {:>10} {:>10.4} {:>12.0} {:>14.1} {:>8} {:>8} {:>8} {:>8}",
            name, best.ef_search, best.recall, best.qps, best.avg_visited,
            best.p50_visited, best.p95_visited, best.p99_visited, best.max_visited
        );
    }

    // 检查最低 recall 是否达标
    let min_recall: f64 = all_results
        .iter()
        .flat_map(|(_, rs)| rs.iter())
        .map(|r| r.recall)
        .fold(f64::MAX, f64::min);

    if min_recall < 0.9 {
        println!("\nFAIL: 最低 recall {:.4} < 0.9", min_recall);
        std::process::exit(1);
    } else {
        println!("\nPASS: 所有扫描点 recall >= 0.9");
    }
}
