//! P12: 跳跃边实验 — DirectionalPrune + 长程跳跃边
//!
//! DirectionalPrune Pass 1 的遮挡判定 d(j,i) < d(i,q) 在几何上
//! 等价于 cos(θ) > r_j/(2*r_i)。当已接受邻居 j 很近时，即使候选 i
//! 在 90° 方向上也会被遮挡——远距离但方向唯一的「跳跃边」被系统性剪掉。
//!
//! Pass 3 修复：从被拒绝的候选中，选择遮挡比 d²(j,i)/d²(i,q) 最接近
//! 1.0 的（=方向最独特），保留 n_jumps 条。
//!
//! 本实验：n_jumps=0(baseline)/2/4/6/8，R=40, L=384, po=16, rr=5
//! 每个数据点 1 warmup + 3 次测量取中位数 QPS
//!
//! 用法：cargo run --release --bin jump_edge_bench

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

macro_rules! println_both {
    ($out:expr, $($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{}", line);
        $out.push_str(&line);
        $out.push('\n');
    }};
    ($out:expr) => {{
        println!();
        $out.push('\n');
    }};
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
) -> BenchResult {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);
    searcher.with_rerank_factor(rr);
    searcher.with_sq4(sq4);

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
    let recall = compute_recall(&results, gt, gt_k, k, nq);
    BenchResult {
        recall,
        qps: nq as f64 / dt.as_secs_f64(),
        avg_visited: total_visited as f64 / nq as f64,
    }
}

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
) -> BenchResult {
    run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sq4); // warmup

    let mut runs: Vec<BenchResult> = (0..3)
        .map(|_| run_once(train, graph, test, dim, nq, gt, gt_k, k, ef, po, rr, sq4))
        .collect();

    runs.sort_by(|a, b| a.qps.partial_cmp(&b.qps).unwrap());
    let median_qps = runs[1].qps;
    let avg_recall = runs.iter().map(|r| r.recall).sum::<f64>() / 3.0;
    let avg_visited = runs.iter().map(|r| r.avg_visited).sum::<f64>() / 3.0;

    BenchResult {
        recall: avg_recall,
        qps: median_qps,
        avg_visited,
    }
}

fn build_graph(train: &[f32], dim: usize, r_max: usize, l_build: usize, n_jumps: usize) -> VamanaGraph {
    // 通过环境变量传递 n_jumps 给 prune_dispatch
    std::env::set_var("RAVEN_N_JUMPS", n_jumps.to_string());

    let mut rng = ChaCha8Rng::seed_from(42);
    let r_soft = (r_max as f32 * 1.5) as usize;
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build,
        r_max,
        r_soft,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        n_jumps,
        ..Default::default()
    };
    VamanaGraph::build(train, dim, &config, &mut rng)
}

fn fmt_label(nj: usize) -> String {
    if nj == 0 { "base".to_string() } else { format!("j{}", nj) }
}

fn main() {
    let mut out = String::new();

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
    let l_build = 384usize;
    let r_max = 40usize;

    // n_jumps 扫描：0(baseline) / 4 / 8（修复后重新实验）
    let n_jumps_values: [usize; 3] = [0, 4, 8];
    let ef_values: [usize; 6] = [35, 40, 50, 60, 80, 100];

    println_both!(&mut out, "P12: 跳跃边实验 — DirectionalPrune + 长程跳跃边");
    println_both!(&mut out, "数据: n={}, dim={}, nq={}, k={}", n, dim, nq, k);
    println_both!(&mut out, "参数: R={}, L={}, po={}, rr={}", r_max, l_build, po, rr);
    println_both!(&mut out, "n_jumps 扫描: {:?}", n_jumps_values);
    println_both!(&mut out, "ef 扫描: {:?}", ef_values);
    println_both!(&mut out, "测量：1 次 warmup + 3 次正式，取中位数 QPS");
    println_both!(&mut out, "核心指标：avg_visited（跳跃边应降低跳数）");

    let mut all_results: std::collections::BTreeMap<usize, Vec<(usize, BenchResult)>> =
        std::collections::BTreeMap::new();
    let mut build_times: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();

    for (idx, &nj) in n_jumps_values.iter().enumerate() {
        let label = if nj == 0 { " (baseline)".to_string() } else { String::new() };
        println_both!(
            &mut out,
            "\n=== 建图 {}/{}: n_jumps={}{} ===",
            idx + 1,
            n_jumps_values.len(),
            nj,
            label
        );
        let t0 = Instant::now();
        let graph = build_graph(&train, dim, r_max, l_build, nj);
        let build_time = t0.elapsed().as_secs_f64();
        build_times.insert(nj, build_time);
        println_both!(&mut out, "建图: {:.1}s", build_time);

        // 统计平均度数
        let mut total_degree = 0usize;
        let mut max_degree = 0usize;
        for node in 0..n {
            let deg = graph.neighbors(node as u32).len();
            total_degree += deg;
            if deg > max_degree {
                max_degree = deg;
            }
        }
        let avg_degree = total_degree as f64 / n as f64;
        println_both!(
            &mut out,
            "度数统计: avg={:.1}, max={}",
            avg_degree,
            max_degree
        );

        let sq4 = SQ4Dataset::build(&train, dim);

        println_both!(
            &mut out,
            "\n--- n_jumps={} (po={}, rr={}) ---",
            nj,
            po,
            rr
        );
        let header = format!(
            "  {:>6}  {:>10}  {:>12}  {:>12}",
            "ef", "recall", "QPS(med)", "avg_visited"
        );
        println_both!(&mut out, "{}", header);
        println_both!(&mut out, "  {}", "-".repeat(48));

        let mut nj_results: Vec<(usize, BenchResult)> = Vec::new();
        for &ef in &ef_values {
            let result = bench_sq4(&train, &graph, &test, dim, nq, &gt, gt_k, k, ef, po, rr, &sq4);
            println_both!(
                &mut out,
                "  {:>6}  {:>10.4}  {:>12.0}  {:>12.1}",
                ef,
                result.recall,
                result.qps,
                result.avg_visited
            );
            nj_results.push((ef, result));
        }

        all_results.insert(nj, nj_results);
        drop(graph);
    }

    // === 汇总表 ===
    let ncols = n_jumps_values.len();

    // avg_visited 对比
    println_both!(&mut out, "\n=== avg_visited 对比（越低越好）===");
    let mut hdr = String::from("  ef     ");
    for &nj in &n_jumps_values {
        hdr.push_str(&format!("  {:>10}", fmt_label(nj)));
    }
    println_both!(&mut out, "{}", hdr);
    println_both!(&mut out, "  {}", "-".repeat(8 + ncols * 12));
    for &ef in &ef_values {
        let mut row = format!("  {:<6}", ef);
        for &nj in &n_jumps_values {
            if let Some(results) = all_results.get(&nj) {
                if let Some((_, res)) = results.iter().find(|(e, _)| *e == ef) {
                    row.push_str(&format!("  {:>10.1}", res.avg_visited));
                    continue;
                }
            }
            row.push_str("  {:>10}");
            row.push_str("  -");
        }
        println_both!(&mut out, "{}", row);
    }

    // QPS 对比
    println_both!(&mut out, "\n=== QPS 对比（越高越好）===");
    let mut hdr = String::from("  ef     ");
    for &nj in &n_jumps_values {
        hdr.push_str(&format!("  {:>10}", fmt_label(nj)));
    }
    println_both!(&mut out, "{}", hdr);
    println_both!(&mut out, "  {}", "-".repeat(8 + ncols * 12));
    for &ef in &ef_values {
        let mut row = format!("  {:<6}", ef);
        for &nj in &n_jumps_values {
            if let Some(results) = all_results.get(&nj) {
                if let Some((_, res)) = results.iter().find(|(e, _)| *e == ef) {
                    row.push_str(&format!("  {:>10.0}", res.qps));
                    continue;
                }
            }
            row.push_str("  -");
        }
        println_both!(&mut out, "{}", row);
    }

    // Recall 对比
    println_both!(&mut out, "\n=== Recall 对比 ===");
    let mut hdr = String::from("  ef     ");
    for &nj in &n_jumps_values {
        hdr.push_str(&format!("  {:>10}", fmt_label(nj)));
    }
    println_both!(&mut out, "{}", hdr);
    println_both!(&mut out, "  {}", "-".repeat(8 + ncols * 12));
    for &ef in &ef_values {
        let mut row = format!("  {:<6}", ef);
        for &nj in &n_jumps_values {
            if let Some(results) = all_results.get(&nj) {
                if let Some((_, res)) = results.iter().find(|(e, _)| *e == ef) {
                    row.push_str(&format!("  {:>10.4}", res.recall));
                    continue;
                }
            }
            row.push_str("  -");
        }
        println_both!(&mut out, "{}", row);
    }

    // 建图时间对比
    println_both!(&mut out, "\n=== 建图时间对比 ===");
    for &nj in &n_jumps_values {
        println_both!(
            &mut out,
            "  {:>10}: {:.1}s",
            fmt_label(nj),
            build_times[&nj]
        );
    }

    // 相对变化汇总
    println_both!(&mut out, "\n=== 相对变化（vs baseline, ef=50）===");
    let baseline_ef50 = all_results.get(&0).and_then(|v| v.iter().find(|(e, _)| *e == 50));
    if let Some((_, base)) = baseline_ef50 {
        for &nj in &n_jumps_values[1..] {
            if let Some(results) = all_results.get(&nj) {
                if let Some((_, res)) = results.iter().find(|(e, _)| *e == 50) {
                    let d_visited = (res.avg_visited - base.avg_visited) / base.avg_visited * 100.0;
                    let d_qps = (res.qps - base.qps) / base.qps * 100.0;
                    let d_recall = (res.recall - base.recall) * 100.0; // in pp
                    println_both!(
                        &mut out,
                        "  j{}: avg_visited {:+.1}%, QPS {:+.1}%, recall {:+.2}pp",
                        nj,
                        d_visited,
                        d_qps,
                        d_recall
                    );
                }
            }
        }
    }

    let result_path = "experiments/jump_edge_bench_result.txt";
    let mut f = File::create(result_path).expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 {}", result_path);
}
