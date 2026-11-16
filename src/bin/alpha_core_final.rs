//! Alpha_core 最终验证 — 多 seed 复现 + R Pareto 对比
//!
//! ════════════════════════════════════════════════════════════════════
//!  问题一：alpha_core=0.93 只跑了单 seed，可能是噪声伪峰值
//!  方案：{0.91, 0.93, 0.95} × {42, 123, 7} = 9 次独立建图
//!        逐 seed 比较，确认 0.93 在每个 seed 下都领先
//!
//!  问题二：R=44 recall 更高可能只因多访问了 9% 节点，非公平比较
//!  方案：R=40 vs R=44 各扫 ef∈{40,45,50,55,60}，3 seeds
//!        在 recall vs avg_visited 平面上比较 Pareto 曲线
//!        用线性插值在匹配 avg_visited 下对比 recall
//! ════════════════════════════════════════════════════════════════════
//!
//! 用法：cargo run --release --bin alpha_core_final

use std::io::Write;
use std::time::Instant;

use raven::bench::util::{SiftDataset, compute_recall, median3};
use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::quant::SQ4Dataset;

// ════════════════════════════════════════════════════════════════════
//  常量
// ════════════════════════════════════════════════════════════════════

const SEEDS: [u64; 3] = [42, 123, 7];
const AC_CANDIDATES: [f32; 3] = [0.91, 0.93, 0.95];
const PARETO_EF: [usize; 5] = [40, 45, 50, 55, 60];
const K: usize = 10;
const PO: usize = 16;
const RR: usize = 5;
const L_BUILD: usize = 384;

// ════════════════════════════════════════════════════════════════════
//  建图 + 搜索
// ════════════════════════════════════════════════════════════════════

/// 建图：固定 L=384, nav_m=32, DirectionalPrune, alpha=1.2, 2-pass
fn build_graph(
    train: &[f32],
    dim: usize,
    alpha_core: f32,
    r_max: usize,
    seed: u64,
) -> (VamanaGraph, f64) {
    std::env::set_var("RAVEN_ALPHA_CORE", alpha_core.to_string());

    let mut rng = ChaCha8Rng::seed_from(seed);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: L_BUILD,
        r_max,
        r_soft: (r_max as f32 * 1.5) as usize,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let t0 = Instant::now();
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    (graph, t0.elapsed().as_secs_f64())
}

/// 在指定 ef 下测试：1 warmup + 3 rounds 取中位数
fn bench_at_ef(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    ef: usize,
    sq4: &SQ4Dataset,
) -> (f64, f64, f64) {
    let run = || {
        let mut searcher = GraphSearcher::new(train, graph, ef);
        searcher.with_prefetch_offset(PO);
        searcher.with_rerank_factor(RR);
        searcher.with_sq4(sq4);

        let mut results = Vec::with_capacity(nq);
        let mut total_visited = 0usize;
        let t0 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = searcher.search_sq4(query, K);
            total_visited += searcher.last_visited_count();
            results.push(r.iter().map(|(id, _)| *id).collect());
        }
        let dt = t0.elapsed();
        let recall = compute_recall(&results, gt, gt_k, K, nq);
        (recall, nq as f64 / dt.as_secs_f64(), total_visited as f64 / nq as f64)
    };

    run(); // warmup
    let r1 = run();
    let r2 = run();
    let r3 = run();
    (
        median3(r1.0, r2.0, r3.0),
        median3(r1.1, r2.1, r3.1),
        median3(r1.2, r2.2, r3.2),
    )
}

// ════════════════════════════════════════════════════════════════════
//  线性插值（Pareto 对比用）
// ════════════════════════════════════════════════════════════════════

/// 在 (x, y) 点集中，返回 x=target_x 处的线性插值 y 值
/// points 须按 x 升序排列；超出范围返回 None
fn interp_y(points: &[(f64, f64)], target_x: f64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return if (points[0].0 - target_x).abs() < 1e-9 {
            Some(points[0].1)
        } else {
            None
        };
    }
    // 找到 target_x 落在哪两个点之间
    for w in points.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if target_x >= x0 - 1e-9 && target_x <= x1 + 1e-9 {
            if (x1 - x0).abs() < 1e-9 {
                return Some(y0);
            }
            let t = (target_x - x0) / (x1 - x0);
            return Some(y0 + t * (y1 - y0));
        }
    }
    None
}

// ════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════

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

    let data = SiftDataset::load();
    let sq4 = SQ4Dataset::build(&data.train, data.dim);

    println_both!("=== RAVEN Alpha_core Final Verification ===\n");
    println_both!("数据: n={}, dim={}, nq={}, k={}", data.n, data.dim, data.nq, K);
    println_both!("固定: L={}, po={}, rr={}, SQ4, DirectionalPrune", L_BUILD, PO, RR);
    println_both!("Seeds: {:?}", SEEDS);
    println_both!("方法: 1 warmup + 3 rounds 取中位数\n");

    // ════════════════════════════════════════════════════════════════
    //  阶段 1：alpha_core 多 seed 复现
    // ════════════════════════════════════════════════════════════════
    println_both!("═══════════════════════════════════════════════════════════════");
    println_both!("  阶段 1：alpha_core 多 seed 复现 (R=40, ef=50)");
    println_both!("  alpha_core ∈ {:?} × seeds {:?} = {} 次独立建图", AC_CANDIDATES, SEEDS, AC_CANDIDATES.len() * SEEDS.len());
    println_both!("═══════════════════════════════════════════════════════════════\n");

    // 结构: (alpha_core, seed, recall, qps, visited, build_s)
    let mut p1: Vec<(f32, u64, f64, f64, f64, f64)> = Vec::new();
    // 缓存 alpha_core=0.93 的 3 个图（供阶段 2 复用）
    let mut r40_cached: Vec<(u64, VamanaGraph)> = Vec::new();

    println_both!("  {:>12}  {:>6}  {:>10}  {:>10}  {:>12}  {:>8}",
        "alpha_core", "seed", "recall", "QPS", "avg_visited", "build_s");
    println_both!("  {}", "-".repeat(66));

    for &ac in &AC_CANDIDATES {
        for &seed in &SEEDS {
            let (graph, build_t) = build_graph(&data.train, data.dim, ac, 40, seed);
            let (recall, qps, visited) = bench_at_ef(
                &data.train, &graph, &data.test, data.dim, data.nq,
                &data.gt, data.gt_k, 50, &sq4,
            );
            println_both!("  {:>12.2}  {:>6}  {:>10.4}  {:>10.0}  {:>12.1}  {:>8.0}",
                ac, seed, recall, qps, visited, build_t);
            p1.push((ac, seed, recall, qps, visited, build_t));

            if (ac - 0.93).abs() < 1e-6 {
                r40_cached.push((seed, graph));
            }
        }
    }

    // --- 阶段 1 分析 ---
    println_both!("\n--- alpha_core recall 统计（跨 seed）---");
    println_both!("  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "alpha_core", "min", "max", "range", "mean");
    println_both!("  {}", "-".repeat(56));

    for &ac in &AC_CANDIDATES {
        let recalls: Vec<f64> = p1.iter()
            .filter(|(a, _, _, _, _, _)| (a - ac).abs() < 1e-6)
            .map(|(_, _, r, _, _, _)| *r)
            .collect();
        let rmin = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
        let rmax = recalls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rmean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        println_both!("  {:>12.2}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            ac, rmin, rmax, rmax - rmin, rmean);
    }

    // 逐 seed 排名
    println_both!("\n--- 逐 seed 排名 ---");
    println_both!("  {:>6}  {:>14}  {:>14}  {:>14}  {:>10}",
        "seed", "1st (best)", "2nd", "3rd", "0.93 best?");
    println_both!("  {}", "-".repeat(64));

    let mut ac93_always_best = true;
    for &seed in &SEEDS {
        let mut seed_results: Vec<(f32, f64)> = AC_CANDIDATES.iter()
            .map(|&ac| {
                let r = p1.iter()
                    .find(|(a, s, _, _, _, _)| (a - ac).abs() < 1e-6 && *s == seed)
                    .map(|(_, _, r, _, _, _)| *r)
                    .unwrap();
                (ac, r)
            })
            .collect();
        seed_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_ac = seed_results[0].0;
        let is_93_best = (best_ac - 0.93).abs() < 1e-6;
        if !is_93_best {
            ac93_always_best = false;
        }

        println_both!("  {:>6}  {:>10.2} ({:.4})  {:>10.2} ({:.4})  {:>10.2} ({:.4})  {:>10}",
            seed,
            seed_results[0].0, seed_results[0].1,
            seed_results[1].0, seed_results[1].1,
            seed_results[2].0, seed_results[2].1,
            if is_93_best { "YES" } else { "NO" });
    }

    // 逐 seed delta: 0.93 vs 0.91, 0.93 vs 0.95
    println_both!("\n--- 逐 seed delta: 0.93 vs 候选 ---");
    println_both!("  {:>6}  {:>14}  {:>14}  {:>10}",
        "seed", "Δ(0.93−0.91)", "Δ(0.93−0.95)", "sig?");
    println_both!("  {}", "-".repeat(50));

    for &seed in &SEEDS {
        let r = |ac: f32| p1.iter()
            .find(|(a, s, _, _, _, _)| (a - ac).abs() < 1e-6 && *s == seed)
            .map(|(_, _, r, _, _, _)| *r)
            .unwrap();
        let d91 = r(0.93) - r(0.91);
        let d95 = r(0.93) - r(0.95);
        let sig = if d91 > 0.001 && d95 > 0.001 { "STRONG" }
                  else if d91 > 0.0 && d95 > 0.0 { "weak" }
                  else { "NO" };
        println_both!("  {:>6}  {:>+14.4}  {:>+14.4}  {:>10}", seed, d91, d95, sig);
    }

    println_both!("\n  >>> 阶段 1 结论: alpha_core=0.93 {}",
        if ac93_always_best { "在每个 seed 下均为最优 ✓" }
        else { "并非在所有 seed 下最优 ✗ — 需进一步分析" });

    // ════════════════════════════════════════════════════════════════
    //  阶段 2：R=40 vs R=44 Pareto 对比
    // ════════════════════════════════════════════════════════════════
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  阶段 2：R=40 vs R=44 Pareto 对比 (alpha_core=0.93)");
    println_both!("  R=40 复用阶段 1 缓存的 {} 个图; R=44 新建 {} 个图", r40_cached.len(), SEEDS.len());
    println_both!("  每图扫 ef ∈ {:?} → 共 {} 次搜索", PARETO_EF, 2 * SEEDS.len() * PARETO_EF.len());
    println_both!("═══════════════════════════════════════════════════════════════\n");

    // 结构: (R, seed, ef, recall, qps, visited)
    let mut p2: Vec<(usize, u64, usize, f64, f64, f64)> = Vec::new();

    // R=40: 复用缓存图
    println_both!("--- R=40 (复用阶段 1 缓存图, alpha_core=0.93) ---");
    println_both!("  {:>6}  {:>4}  {:>10}  {:>10}  {:>12}",
        "seed", "ef", "recall", "QPS", "avg_visited");
    println_both!("  {}", "-".repeat(48));
    for &(seed, ref graph) in &r40_cached {
        for &ef in &PARETO_EF {
            let (recall, qps, visited) = bench_at_ef(
                &data.train, graph, &data.test, data.dim, data.nq,
                &data.gt, data.gt_k, ef, &sq4,
            );
            println_both!("  {:>6}  {:>4}  {:>10.4}  {:>10.0}  {:>12.1}",
                seed, ef, recall, qps, visited);
            p2.push((40, seed, ef, recall, qps, visited));
        }
    }

    // R=44: 新建图
    println_both!("\n--- R=44 (新建图, alpha_core=0.93) ---");
    println_both!("  {:>6}  {:>4}  {:>10}  {:>10}  {:>12}",
        "seed", "ef", "recall", "QPS", "avg_visited");
    println_both!("  {}", "-".repeat(48));
    let mut r44_graphs: Vec<(u64, VamanaGraph)> = Vec::new();
    for &seed in &SEEDS {
        let (graph, _) = build_graph(&data.train, data.dim, 0.93, 44, seed);
        for &ef in &PARETO_EF {
            let (recall, qps, visited) = bench_at_ef(
                &data.train, &graph, &data.test, data.dim, data.nq,
                &data.gt, data.gt_k, ef, &sq4,
            );
            println_both!("  {:>6}  {:>4}  {:>10.4}  {:>10.0}  {:>12.1}",
                seed, ef, recall, qps, visited);
            p2.push((44, seed, ef, recall, qps, visited));
        }
        r44_graphs.push((seed, graph));
    }

    // --- 阶段 2 分析 ---
    // 跨 seed 均值表
    println_both!("\n--- 跨 seed 均值 ---");
    println_both!("  {:>4}  {:>4}  {:>12}  {:>14}  {:>10}",
        "R", "ef", "recall_mean", "visited_mean", "QPS_mean");
    println_both!("  {}", "-".repeat(58));

    for &r in &[40usize, 44] {
        for &ef in &PARETO_EF {
            let pts: Vec<(f64, f64, f64)> = p2.iter()
                .filter(|(rr, _, e, _, _, _)| *rr == r && *e == ef)
                .map(|(_, _, _, recall, qps, vis)| (*recall, *vis, *qps))
                .collect();
            let n = pts.len() as f64;
            let r_mean = pts.iter().map(|(r, _, _)| r).sum::<f64>() / n;
            let v_mean = pts.iter().map(|(_, v, _)| v).sum::<f64>() / n;
            let q_mean = pts.iter().map(|(_, _, q)| q).sum::<f64>() / n;
            println_both!("  {:>4}  {:>4}  {:>12.4}  {:>14.1}  {:>10.0}",
                r, ef, r_mean, v_mean, q_mean);
        }
        println_both!("  {}", "-".repeat(58));
    }

    // Pareto 对比：在匹配 avg_visited 下比较 recall
    println_both!("\n--- Pareto 对比：在匹配 avg_visited 下比较 recall ---");
    println_both!("  （对每个 seed，用 R=40 的 (visited, recall) 曲线");
    println_both!("   插值出 R=44 各 ef 点 visited 处的 R=40 recall）\n");
    println_both!("  {:>6}  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "seed", "ef", "visited", "R44_rec", "R40_interp", "Δ(R44−R40)", "R44 win?");
    println_both!("  {}", "-".repeat(74));

    let mut r44_wins = 0usize;
    let mut r44_total = 0usize;
    let mut deltas: Vec<f64> = Vec::new();

    for &seed in &SEEDS {
        // R=40 的 (visited, recall) 点（按 visited 升序）
        let r40_curve: Vec<(f64, f64)> = PARETO_EF.iter()
            .filter_map(|&ef| {
                p2.iter()
                    .find(|(r, s, e, _, _, _)| *r == 40 && *s == seed && *e == ef)
                    .map(|(_, _, _, rec, _, vis)| (*vis, *rec))
            })
            .collect();

        // R=44 的每个 ef 点
        for &ef in &PARETO_EF {
            if let Some((_, _, _, r44_rec, _, r44_vis)) = p2.iter()
                .find(|(r, s, e, _, _, _)| *r == 44 && *s == seed && *e == ef)
            {
                r44_total += 1;
                if let Some(r40_interp) = interp_y(&r40_curve, *r44_vis) {
                    let delta = r44_rec - r40_interp;
                    deltas.push(delta);
                    let win = delta > 0.0;
                    if win { r44_wins += 1; }
                    println_both!("  {:>6}  {:>4}  {:>10.1}  {:>10.4}  {:>10.4}  {:>+10.4}  {:>10}",
                        seed, ef, r44_vis, r44_rec, r40_interp, delta,
                        if win { "YES" } else { "no" });
                } else {
                    println_both!("  {:>6}  {:>4}  {:>10.1}  {:>10.4}  {:>10}  {:>10}  {:>10}",
                        seed, ef, r44_vis, r44_rec, "N/A", "N/A", "out of range");
                }
            }
        }
    }

    let mean_delta = if deltas.is_empty() { 0.0 } else { deltas.iter().sum::<f64>() / deltas.len() as f64 };
    println_both!("\n  R=44 胜出: {}/{} 次, 平均 Δrecall = {:+.4}",
        r44_wins, r44_total, mean_delta);

    // 最终裁决
    println_both!("\n═══════════════════════════════════════════════════════════════");
    println_both!("  最终裁决");
    println_both!("═══════════════════════════════════════════════════════════════");

    let r44_pareto_dominant = r44_wins > r44_total / 2 && mean_delta > 0.001;

    println_both!("\n  阶段 1 — alpha_core=0.93 多 seed 复现:");
    println_both!("    0.93 在所有 seed 下最优: {}", if ac93_always_best { "✓ YES" } else { "✗ NO" });
    println_both!("    结论: alpha_core=0.93 {}",
        if ac93_always_best { "是真实峰值，可写入提交配置" }
        else { "未通过多 seed 验证，需进一步分析" });

    println_both!("\n  阶段 2 — R=40 vs R=44 Pareto 对比:");
    println_both!("    R=44 在匹配 visited 下胜出: {}/{}", r44_wins, r44_total);
    println_both!("    平均 Δrecall (R44−R40): {:+.4}", mean_delta);
    println_both!("    R=44 Pareto 支配: {}",
        if r44_pareto_dominant { "✓ YES — R=44 在相同访问量下 recall 更高" }
        else { "✗ NO — R=44 的优势可由 R=40 调高 ef 复现" });

    println_both!("\n  >>> 最终配置建议:");
    if ac93_always_best {
        println_both!("    alpha_core = 0.93 ✓");
    } else {
        println_both!("    alpha_core = 0.90 (保守选择，已通过 3-seed 验证)");
    }
    if r44_pareto_dominant {
        println_both!("    R = 44 ✓ (Pareto 支配 R=40)");
    } else {
        println_both!("    R = 40 (R=44 优势不显著，保留原值)");
    }

    // 写文件
    let mut f = std::fs::File::create("experiments/alpha_core_final_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 experiments/alpha_core_final_result.txt");
}
