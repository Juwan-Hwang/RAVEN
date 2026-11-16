//! 提交配置真旗舰 Benchmark — 与 ann-benchmarks config.yml 完全一致
//!
//! 三条曲线：
//!   1. raven-sq4:          ef=[35,40,45,50,55,65,70,80,100], rr=5, SQ4, 单线程
//!   2. raven-sq4-adaptive:  AdaptiveEf 精选配置, rr=5, SQ4, 单线程 (Pareto 填隙)
//!   3. raven-sq4-mt:       ef=50, rr=5, SQ4, 全核多线程
//!
//! 用法：cargo run --release --bin submission_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::bench::util::{bfs_reorder, translate_results};
use raven::build::ChaCha8Rng;
use raven::graph::{
    AdaptiveEfConfig, GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
};
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

struct BenchResult {
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

/// 单线程 SQ4 benchmark：warmup + 2 轮取平均
///
/// `new_to_old` 非空时，搜索结果在 compute_recall 前翻译回原始 ID 空间
fn bench_sq4(
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
    new_to_old: &[u32],
) -> BenchResult {
    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rr);
        searcher.with_sq4(sq4);

        let mut results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut total_visited = 0;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = searcher.search_sq4(query, k);
            total_visited += searcher.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        if !new_to_old.is_empty() {
            translate_results(&mut results, new_to_old);
        }
        let recall = compute_recall(&results, gt, gt_k, k, nq);
        BenchResult {
            recall,
            qps: nq as f64 / dt.as_secs_f64(),
            avg_visited: total_visited as f64 / nq as f64,
        }
    };

    run(); // warmup
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
    }
}

/// 自适应 ef 单线程 SQ4 benchmark：warmup + 2 轮取平均
///
/// `max_ef` 作为 VisitedTracker 容量上限，实际 ef 由 `estimate_ef` 动态决定。
/// `new_to_old` 非空时，搜索结果在 compute_recall 前翻译回原始 ID 空间
fn bench_adaptive_sq4(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    max_ef: usize,
    po: usize,
    rr: usize,
    sq4: &SQ4Dataset,
    config: &AdaptiveEfConfig,
    new_to_old: &[u32],
) -> BenchResult {
    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, max_ef);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rr);
        searcher.with_sq4(sq4);
        searcher.with_adaptive_ef(config.clone());

        let mut results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        let mut total_visited = 0;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = searcher.search_sq4(query, k);
            total_visited += searcher.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        if !new_to_old.is_empty() {
            translate_results(&mut results, new_to_old);
        }
        let recall = compute_recall(&results, gt, gt_k, k, nq);
        BenchResult {
            recall,
            qps: nq as f64 / dt.as_secs_f64(),
            avg_visited: total_visited as f64 / nq as f64,
        }
    };

    run(); // warmup
    let r2 = run();
    let r3 = run();
    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (r2.qps + r3.qps) / 2.0,
        avg_visited: (r2.avg_visited + r3.avg_visited) / 2.0,
    }
}

/// 多线程 batch_search benchmark：warmup + 2 轮取平均
///
/// `new_to_old` 非空时，搜索结果在 compute_recall 前翻译回原始 ID 空间
fn bench_multi_thread(
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
    num_threads: usize,
    new_to_old: &[u32],
) -> BenchResult {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("build thread pool");

    let queries: Vec<&[f32]> = (0..nq).map(|q| &test[q * dim..(q + 1) * dim]).collect();

    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_prefetch_offset(po);
        searcher.with_rerank_factor(rr);
        searcher.with_sq4(sq4);

        let batch = pool.install(|| searcher.batch_search(&queries, k));

        let mut results: Vec<Vec<u32>> = Vec::with_capacity(nq);
        for result in &batch {
            results.push(result.iter().map(|(id, _)| *id).collect());
        }
        if !new_to_old.is_empty() {
            translate_results(&mut results, new_to_old);
        }
        let recall = compute_recall(&results, gt, gt_k, k, nq);
        BenchResult {
            recall,
            qps: 0.0,
            avg_visited: 0.0,
        }
    };

    run(); // warmup
    let t0 = Instant::now();
    let r2 = run();
    let r3 = run();
    let dt = t0.elapsed();

    BenchResult {
        recall: (r2.recall + r3.recall) / 2.0,
        qps: (2.0 * nq as f64) / dt.as_secs_f64(),
        avg_visited: 0.0,
    }
}

fn main() {
    let mut out = String::new();

    macro_rules! println_both {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{}", line);
            out.push_str(&line);
            out.push('\n');
        }};
    }

    println_both!("=== RAVEN Submission Benchmark (config.yml final) ===\n");

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
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let ef_list: [usize; 9] = [35, 40, 45, 50, 55, 65, 70, 80, 100];

    println_both!("数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(
        "配置: nav_m=32, DirectionalPrune, alpha=1.2, alpha_core=0.93, L=384, R=40, po={}, rr={}",
        po, rr
    );
    println_both!("线程: {}", num_threads);

    // 建图
    println_both!("\n--- 建图 ---");
    // alpha_core=0.93: Pass 1 核心方向性判据的最优遮挡阈值
    // 第十三轮诊断：精细扫描发现 0.93 > 0.91 > 0.95，3-seed 验证确认每 seed 下 0.93 均最优
    // recall: 0.9763 (vs 旧基线 0.9686, +0.77pp), avg_visited 几乎不变
    // R=44 Pareto 对比：0/15 胜出，R=44 优势完全来自多访问节点，保留 R=40
    std::env::set_var("RAVEN_ALPHA_CORE", "0.93");
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
    println_both!(
        "编码: SQ4={:.1} MB ({}x 压缩)",
        sq4.codes.len() as f64 / 1e6,
        (n * dim * 4) as f64 / sq4.codes.len() as f64,
    );

    // BFS 重排 — 纯后处理，不改变图拓扑，只提升 cache 局部性
    // 验证结论（reorder_verify）：20 query × top-10 ID 集合完全一致，200 对距离全部一致
    let t0 = Instant::now();
    let reord = bfs_reorder(&graph, &sq4, &train, dim, 32, 20);
    println_both!("BFS 重排: {:.1}s", t0.elapsed().as_secs_f64());
    let train = reord.train;
    let graph = reord.graph;
    let sq4 = reord.sq4;
    let new_to_old = reord.new_to_old;

    // ════════════════════════════════════════════════════════════════
    // 曲线 1: raven-sq4 (单线程, rr=8)
    // ════════════════════════════════════════════════════════════════
    println_both!("\n--- 曲线 1: raven-sq4 (单线程, rr={}) ---", rr);
    println_both!(
        "  {:>6}  {:>10}  {:>12}  {:>12}",
        "ef",
        "recall",
        "QPS",
        "avg_visited"
    );
    println_both!("  {}", "-".repeat(48));

    let mut sq4_results: Vec<(usize, BenchResult)> = Vec::new();

    for &ef in &ef_list {
        let r = bench_sq4(
            &train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4, &new_to_old,
        );
        println_both!(
            "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}",
            ef,
            r.recall,
            r.qps,
            r.avg_visited
        );
        sq4_results.push((ef, r));
    }

    // ════════════════════════════════════════════════════════════════
    // 曲线 2: raven-sq4-adaptive (单线程, rr=5, AdaptiveEf)
    //
    // AdaptiveEf 根据 query→entry-point 距离分布动态预测 ef，
    // 在固定 ef 曲线的 recall 间隙处生成 Pareto 最优点。
    //
    // 8 组配置基于 alpha_core=0.93 新曲线的 Pareto 重新扫描选定
    // (adaptive_resweep)，覆盖 0.945-0.981 全 recall 区间：
    //   - 低 recall 区（min_ef=25/30）：填补 ef=35 以下及 ef=35→40 间隙
    //   - 中 recall 区（min_ef=35）：填补 ef=40→50 间隙
    //   - 高 recall 区（max_ef=95/100, min_ef=35/40）：填补 ef=50→65 间隙
    // ════════════════════════════════════════════════════════════════
    println_both!("\n--- 曲线 2: raven-sq4-adaptive (单线程, rr={}, AdaptiveEf) ---", rr);
    println_both!(
        "  {:>20}  {:>10}  {:>12}  {:>12}",
        "config",
        "recall",
        "QPS",
        "avg_visited"
    );
    println_both!("  {}", "-".repeat(60));

    let nav = graph.layered_nav().expect("layered nav required");
    let adaptive_base = AdaptiveEfConfig::build_with_layered_nav(&train, dim, nav, 30, 100, 2.0);

    // 精选 Pareto 填隙配置 — adaptive_resweep 验证 (alpha_core=0.93)
    // 12 个 adaptive 点在 Pareto 前沿上，精选 8 个覆盖最大间隙
    let adaptive_configs: [(f32, usize, usize); 8] = [
        (2.5, 25, 50),  // recall≈0.945 — 延伸 ef=35 以下
        (2.5, 30, 50),  // recall≈0.956 — 填 ef=35(0.954)→ef=40(0.964) 下半
        (2.5, 30, 55),  // recall≈0.959 — 填 ef=35→ef=40 上半
        (3.5, 35, 65),  // recall≈0.967 — 填 ef=40(0.964)→ef=45(0.971)
        (2.3, 35, 75),  // recall≈0.973 — 填 ef=45(0.971)→ef=50(0.976) 下半
        (3.0, 35, 95),  // recall≈0.974 — 填 ef=45→ef=50 上半
        (2.0, 35, 100), // recall≈0.980 — 填 ef=50(0.976)→ef=55(0.980)
        (2.5, 40, 100), // recall≈0.981 — 填 ef=55(0.980)→ef=65(0.985)
    ];

    let mut adaptive_results: Vec<(String, BenchResult)> = Vec::new();

    for &(gamma, min_ef, max_ef) in &adaptive_configs {
        let config = adaptive_base.with_params(min_ef, max_ef, gamma);
        let r = bench_adaptive_sq4(
            &train, &graph, &test, dim, nq, &gt, gt_k, k, max_ef, po, rr, &sq4, &config, &new_to_old,
        );
        let label = format!("γ{:.1}({},{})", gamma, min_ef, max_ef);
        println_both!(
            "  {:>20}  {:>10.4}  {:>12.0}  {:>12.1}",
            label,
            r.recall,
            r.qps,
            r.avg_visited
        );
        adaptive_results.push((label, r));
    }

    // ════════════════════════════════════════════════════════════════
    // 曲线 3: raven-sq4-mt (多线程, rr=8, ef=50)
    // ════════════════════════════════════════════════════════════════
    println_both!(
        "\n--- 曲线 3: raven-sq4-mt ({} threads, rr={}, ef=50) ---",
        num_threads, rr
    );

    let mt_r = bench_multi_thread(
        &train,
        &graph,
        &test,
        dim,
        nq,
        &gt,
        gt_k,
        k,
        50,
        po,
        rr,
        &sq4,
        num_threads,
        &new_to_old,
    );
    let sq4_50_qps = sq4_results
        .iter()
        .find(|(ef, _)| *ef == 50)
        .map(|(_, r)| r.qps)
        .unwrap_or(1.0);
    println_both!(
        "  recall={:.4}  QPS={:.0}  ({:.1}x vs single-thread)",
        mt_r.recall,
        mt_r.qps,
        mt_r.qps / sq4_50_qps
    );

    // ════════════════════════════════════════════════════════════════
    // 汇总
    // ════════════════════════════════════════════════════════════════
    println_both!("\n--- 提交汇总 ---");
    let sq4_50 = sq4_results
        .iter()
        .find(|(ef, _)| *ef == 50)
        .map(|(_, r)| r)
        .unwrap();
    println_both!(
        "  SQ4  ef=50:  recall={:.4}  QPS={:.0}  (主力工作点)",
        sq4_50.recall,
        sq4_50.qps
    );
    println_both!(
        "  SQ4-mt ef=50: recall={:.4}  QPS={:.0}  (多线程峰值)",
        mt_r.recall,
        mt_r.qps
    );

    // AdaptiveEf Pareto 填隙点汇总
    if let Some(best) = adaptive_results
        .iter()
        .max_by(|a, b| a.1.qps.partial_cmp(&b.1.qps).unwrap())
    {
        println_both!(
            "  SQ4-adaptive 最高QPS: {}  recall={:.4}  QPS={:.0}",
            best.0,
            best.1.recall,
            best.1.qps
        );
    }
    if let Some(highest_recall) = adaptive_results
        .iter()
        .max_by(|a, b| a.1.recall.partial_cmp(&b.1.recall).unwrap())
    {
        println_both!(
            "  SQ4-adaptive 最高recall: {}  recall={:.4}  QPS={:.0}",
            highest_recall.0,
            highest_recall.1.recall,
            highest_recall.1.qps
        );
    }

    // 写文件
    let mut f = File::create("experiments/submission_bench_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/submission_bench_result.txt");
}
