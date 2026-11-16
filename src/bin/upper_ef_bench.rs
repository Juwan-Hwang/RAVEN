//! 方案 B 验证：上层 ef>1 beam search vs ef=1 贪心下降
//!
//! 实验设计：
//!   - 同一张 R=40 图（与 submission_bench 完全相同的建图参数）
//!   - 4 组 upper_ef：[1, 5, 10, 20] × 3 个 ef_search：[35, 40, 50]
//!   - 交替测量（round-robin），1 warmup + 3 轮取中位数 QPS
//!   - 单次实验内完成，排除机器状态干扰
//!
//! 用法：cargo run --release --bin upper_ef_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

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
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
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
                bytes[offset + d * 4..offset + d * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

fn compute_recall(results: &[Vec<u32>], gt: &[i32], gt_k: usize, k: usize, nq: usize) -> f64 {
    let mut hits = 0;
    for q in 0..nq {
        let found = &results[q];
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

#[derive(Clone)]
struct RunResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

/// 单次搜索测量
fn run_once(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
    upper_ef: usize,
) -> RunResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rr);
    searcher.with_sq4(sq4);
    if upper_ef > 1 {
        searcher.with_upper_ef(upper_ef);
    }

    let mut results = Vec::with_capacity(nq);
    let mut total_visited = 0;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let r = searcher.search_sq4(query, k);
        total_visited += searcher.last_visited_count();
        results.push(r.iter().map(|(id, _)| *id).collect());
    }
    let dt = t0.elapsed();

    RunResult {
        recall: compute_recall(&results, gt, gt_k, k, nq),
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn main() {
    let mut out = String::new();

    macro_rules! println_both {
        () => {{
            println!();
            out.push('\n');
        }};
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{}", line);
            out.push_str(&line);
            out.push('\n');
        }};
    }

    println_both!("=== RAVEN 方案 B 验证：上层 ef>1 beam search ===\n");

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
    let po = 16usize;
    let rr = 5usize;

    // 实验参数
    let ef_list: [usize; 3] = [35, 40, 50];
    let upper_ef_list: [usize; 4] = [1, 5, 10, 20];

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(
        "配置: R=40, L=384, po={}, rr={}, DirectionalPrune, alpha=1.2",
        po, rr
    );
    println_both!(
        "实验: ef={:?}, upper_ef={:?}",
        ef_list, upper_ef_list
    );
    println_both!("测量: 1 warmup + 3 轮交替，取中位数 QPS\n");

    // 建图
    println_both!("--- 建图 ---");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 384,
        r_max: 40,
        r_soft: 60,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println_both!("建图: {:.1}s", t0.elapsed().as_secs_f64());

    // 编码
    let sq4 = SQ4Dataset::build(&train, dim);
    println_both!("编码: SQ4={:.1} MB\n", sq4.codes.len() as f64 / 1e6);

    // ════════════════════════════════════════════════════════════════
    // 交替测量：对每个 ef，round-robin 遍历所有 upper_ef
    // ════════════════════════════════════════════════════════════════

    // 收集结果：results[ef_idx][upper_ef_idx] = Vec<RunResult>
    let mut all_results: Vec<Vec<Vec<RunResult>>> = ef_list.iter().map(|_| {
        upper_ef_list.iter().map(|_| Vec::new()).collect()
    }).collect();

    // Warmup：每个 (ef, upper_ef) 组合跑 1 次
    println_both!("--- Warmup ---");
    for &ef in &ef_list {
        for &uef in &upper_ef_list {
            run_once(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4, uef);
        }
    }
    println_both!("Warmup 完成\n");

    // 3 轮交替
    for round in 0..3 {
        println_both!("--- Round {} ---", round + 1);
        for (ei, &ef) in ef_list.iter().enumerate() {
            for (ui, &uef) in upper_ef_list.iter().enumerate() {
                let r = run_once(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4, uef);
                println_both!(
                    "  ef={:>3} upper_ef={:>3}  recall={:.4}  QPS={:>8.0}  avg_visited={:.1}",
                    ef, uef, r.recall, r.qps, r.avg_visited
                );
                all_results[ei][ui].push(r);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 汇总：中位数 QPS + recall + avg_visited
    // ════════════════════════════════════════════════════════════════
    println_both!("\n=== 汇总（3 轮中位数）===");
    println_both!(
        "  {:>6}  {:>10}  {:>10}  {:>12}  {:>12}",
        "ef", "upper_ef", "recall", "QPS(med)", "avg_visited"
    );
    println_both!("  {}", "-".repeat(60));

    for (ei, &ef) in ef_list.iter().enumerate() {
        for (ui, &uef) in upper_ef_list.iter().enumerate() {
            let runs = &all_results[ei][ui];
            let med_qps = median(&runs.iter().map(|r| r.qps).collect::<Vec<_>>());
            let med_recall = median(&runs.iter().map(|r| r.recall).collect::<Vec<_>>());
            let med_visited = median(&runs.iter().map(|r| r.avg_visited).collect::<Vec<_>>());
            println_both!(
                "  {:>6}  {:>10}  {:>10.4}  {:>12.0}  {:>12.1}",
                ef, uef, med_recall, med_qps, med_visited
            );
        }
        println_both!();
    }

    // ════════════════════════════════════════════════════════════════
    // 对比分析：upper_ef>1 vs upper_ef=1
    // ════════════════════════════════════════════════════════════════
    println_both!("=== 对比分析（vs upper_ef=1 基线）===");
    println_both!(
        "  {:>6}  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "ef", "upper_ef", "QPS", "ΔQPS%", "Δvisited%", "Δrecall"
    );
    println_both!("  {}", "-".repeat(72));

    for (ei, &ef) in ef_list.iter().enumerate() {
        let baseline_qps = median(&all_results[ei][0].iter().map(|r| r.qps).collect::<Vec<_>>());
        let baseline_visited =
            median(&all_results[ei][0].iter().map(|r| r.avg_visited).collect::<Vec<_>>());
        let baseline_recall =
            median(&all_results[ei][0].iter().map(|r| r.recall).collect::<Vec<_>>());

        for (ui, &uef) in upper_ef_list.iter().enumerate() {
            let runs = &all_results[ei][ui];
            let med_qps = median(&runs.iter().map(|r| r.qps).collect::<Vec<_>>());
            let med_visited = median(&runs.iter().map(|r| r.avg_visited).collect::<Vec<_>>());
            let med_recall = median(&runs.iter().map(|r| r.recall).collect::<Vec<_>>());

            let delta_qps = (med_qps / baseline_qps - 1.0) * 100.0;
            let delta_visited = (med_visited / baseline_visited - 1.0) * 100.0;
            let delta_recall = med_recall - baseline_recall;

            println_both!(
                "  {:>6}  {:>10}  {:>12.0}  {:>11.1}%  {:>11.1}%  {:>+12.4}",
                ef, uef, med_qps, delta_qps, delta_visited, delta_recall
            );
        }
        println_both!();
    }

    // 写文件
    let mut f = File::create("experiments/upper_ef_bench_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/upper_ef_bench_result.txt");
}
