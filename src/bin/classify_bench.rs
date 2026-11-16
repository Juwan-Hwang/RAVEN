//! insert_rate@10 分类准确率验证
//!
//! 核心问题：r=-0.59 的相关系数能否撑起可靠的二分类判据？
//!
//! 对每个 query 搜索三次（ef=35 / ef=65 / ef=100），收集：
//! - insert_rate@10（ef=35 搜索的前 10 次 pop 中邻居插入成功率）
//! - visited_count（ef=35）
//! - recall@10（对比 ground truth，三个 ef 各算一次）
//! - nav_entry_dist（当前 adaptive ef 用的静态信号，作基线）
//!
//! 分析：
//! 1. insert_rate@10 在 Hard/Easy 组的完整分布（p10/p25/p50/p75/p90 + 直方图）
//! 2. 多阈值二分类混淆矩阵（TPR/FPR/Accuracy/Precision/F1）
//! 3. 与 nav_entry_dist 基线对比
//! 4. 两阶段方案的成本/收益估算
//!
//! 用法：cargo run --release --bin classify_bench

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::linear_pool::LinearPool;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::VisitedTracker;
use raven::memory::serialize::Serializable;
use raven::quant::SQ4Dataset;

// ── IO helpers ──────────────────────────────────────────────────────────

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
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
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
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

// ── Search with tracking ────────────────────────────────────────────────

/// 搜索结果 + 追踪数据
struct TrackResult {
    /// rerank 后的 top-k 结果 (id, dist)
    results: Vec<(u32, f32)>,
    /// 总访问量
    visited_count: usize,
    /// 前 10 次 pop 时的累计 insert_rate
    insert_rate_at_10: f32,
    /// nav 入口距离（静态信号基线）
    nav_entry_dist: f32,
}

/// 带追踪的 SQ4 搜索
fn search_tracked(
    sq4: &SQ4Dataset,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query_code: &[u8],
    ef: usize,
    visited: &mut VisitedTracker,
    pool: &mut LinearPool,
    po: usize,
    k: usize,
    rr: usize,
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    nav_dist: f32,
) -> TrackResult {
    visited.reset();
    pool.reset_for(ef);

    let entry_dist = sq4.distance_raw(query_code, entry_point as usize);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];

    let mut pop_order: usize = 0;
    let mut total_attempts: usize = 0;
    let mut total_inserts: usize = 0;
    let mut insert_rate_at_10: f32 = 0.0;

    while let Some((node, _dist)) = pool.pop() {
        pop_order += 1;

        // 在第 10 次 pop 时记录 insert_rate
        if pop_order == 10 {
            insert_rate_at_10 = if total_attempts > 0 {
                total_inserts as f32 / total_attempts as f32
            } else {
                0.0
            };
        }

        // Multi-line prefetch
        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            unsafe {
                raven::arch::prefetch_l1(ptr);
                raven::arch::prefetch_l1(ptr.add(64));
                raven::arch::prefetch_l1(ptr.add(128));
                raven::arch::prefetch_l1(ptr.add(192));
            }
        }

        let neighbors = storage.neighbors(node);

        let mut edge_size = 0usize;
        for &v in neighbors.iter() {
            if edge_size >= 128 {
                break;
            }
            let is_new = unsafe { visited.visit_unchecked_branchless(v) };
            if is_new {
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; }
                edge_size += 1;
            }
        }

        let pc = po.min(edge_size);
        for i in 0..pc {
            let v = edge_buf[i] as usize;
            let ptr = sq4.code(v).as_ptr() as *const i8;
            unsafe { raven::arch::prefetch_l1(ptr); }
        }

        for i in 0..edge_size {
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = sq4.code(v).as_ptr() as *const i8;
                unsafe { raven::arch::prefetch_l1(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = unsafe { sq4.distance_raw_unchecked(query_code, neighbor as usize) };
            total_attempts += 1;
            if pool.insert(neighbor, d) {
                total_inserts += 1;
            }
        }
    }

    // 如果 pop 不到 10 次（极短搜索），用最终 insert_rate
    if pop_order < 10 {
        insert_rate_at_10 = if total_attempts > 0 {
            total_inserts as f32 / total_attempts as f32
        } else {
            0.0
        };
    }

    let vc = visited.visited_count();
    let candidates = pool.to_sorted_vec();
    let results = rerank_f32(vectors, dim, query, candidates, k, rr);

    TrackResult {
        results,
        visited_count: vc,
        insert_rate_at_10,
        nav_entry_dist: nav_dist,
    }
}

fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rr: usize,
) -> Vec<(u32, f32)> {
    let n = (k * rr).min(candidates.len());
    let mut r: Vec<(u32, f32)> = candidates[..n]
        .iter()
        .map(|(id, _)| {
            (*id, l2_simd(query, &vectors[*id as usize * dim..(*id as usize + 1) * dim]))
        })
        .collect();
    r.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    r.truncate(k);
    r
}

/// 计算 recall@k
fn recall_at_k(results: &[(u32, f32)], gt: &[i32], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<i32> = gt[..k].iter().copied().collect();
    let hits = results.iter().filter(|(id, _)| gt_set.contains(&(*id as i32))).count();
    hits as f32 / k as f32
}

// ── Statistics helpers ──────────────────────────────────────────────────

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() as f32 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f32>() / xs.len() as f32 }
}

fn stddev(xs: &[f32]) -> f32 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (xs.len() - 1) as f32;
    var.sqrt()
}

// ── Binary classification analysis ──────────────────────────────────────

struct ConfusionMatrix {
    tp: usize,  // Hard correctly identified as Hard
    fp: usize,  // Easy incorrectly identified as Hard
    tn: usize,  // Easy correctly identified as Easy
    fn_: usize, // Hard incorrectly identified as Easy
}

impl ConfusionMatrix {
    fn tpr(&self) -> f32 { // sensitivity / recall (Hard)
        if self.tp + self.fn_ == 0 { 0.0 } else { self.tp as f32 / (self.tp + self.fn_) as f32 }
    }
    fn fpr(&self) -> f32 { // false positive rate
        if self.fp + self.tn == 0 { 0.0 } else { self.fp as f32 / (self.fp + self.tn) as f32 }
    }
    fn precision(&self) -> f32 {
        if self.tp + self.fp == 0 { 0.0 } else { self.tp as f32 / (self.tp + self.fp) as f32 }
    }
    fn accuracy(&self) -> f32 {
        let total = self.tp + self.fp + self.tn + self.fn_;
        if total == 0 { 0.0 } else { (self.tp + self.tn) as f32 / total as f32 }
    }
    fn f1(&self) -> f32 {
        let p = self.precision();
        let r = self.tpr();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

/// 在给定阈值下计算混淆矩阵
/// signal < threshold → 预测为 Hard（insert_rate 低 = 难）
fn classify(signal_values: &[f32], is_hard: &[bool], threshold: f32) -> ConfusionMatrix {
    let mut cm = ConfusionMatrix { tp: 0, fp: 0, tn: 0, fn_: 0 };
    for i in 0..signal_values.len() {
        let predicted_hard = signal_values[i] < threshold;
        match (predicted_hard, is_hard[i]) {
            (true, true) => cm.tp += 1,
            (true, false) => cm.fp += 1,
            (false, false) => cm.tn += 1,
            (false, true) => cm.fn_ += 1,
        }
    }
    cm
}

/// 计算 ROC AUC（梯形法）
fn roc_auc(signal_values: &[f32], is_hard: &[bool]) -> f32 {
    // 对所有可能的阈值计算 TPR/FPR
    let mut thresholds: Vec<f32> = signal_values.to_vec();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup();

    let mut points: Vec<(f32, f32)> = Vec::new(); // (fpr, tpr)
    points.push((0.0, 0.0));

    for &t in &thresholds {
        let cm = classify(signal_values, is_hard, t);
        points.push((cm.fpr(), cm.tpr()));
    }

    // 按 FPR 排序
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // 梯形积分
    let mut auc = 0.0;
    for i in 1..points.len() {
        let dx = points[i].0 - points[i - 1].0;
        let dy = points[i].1 + points[i - 1].1;
        auc += dx * dy / 2.0;
    }
    auc
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    println!("=== insert_rate@10 分类准确率验证 ===\n");

    // ── 加载数据 ──
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = 1000.min(nq_full);
    let k = 10;
    let po = 16;
    let rr = 5;

    println!("data: n={}, dim={}, nq={}, gt_k={}\n", n, dim, nq, gt_k);

    // ── 加载/构建图 ──
    let graph_path = std::path::Path::new("data/sift/graph_cache_dir_rmin4.bin");
    let graph = if graph_path.exists() {
        println!("loading cached DirectionalPrune graph...");
        let t0 = Instant::now();
        let g = VamanaGraph::load(graph_path).expect("load graph cache");
        println!("load: {:.1}s\n", t0.elapsed().as_secs_f64());
        g
    } else {
        println!("building graph (first run)...");
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 384, r_max: 40, r_soft: 60,
            max_iterations: 2, saturate: false,
            enable_layered_nav: true, nav_m: 32,
            prune_strategy: PruneStrategy::DirectionalPrune,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from(42);
        let t0 = Instant::now();
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        println!("build: {:.1}s\n", t0.elapsed().as_secs_f64());
        let _ = g.save(graph_path);
        g
    };

    // ── 构建 SQ4 ──
    println!("building SQ4...");
    let sq4 = SQ4Dataset::build(&train, dim);

    let nav = graph.layered_nav().expect("need layered nav");
    let mut visited = VisitedTracker::new(n, 100);
    let mut pool = LinearPool::new(100);
    let code_bytes = dim.div_ceil(2);
    let mut query_code_buf = vec![0u8; code_bytes];

    // ── 搜索：每个 query 在 ef=35/65/100 各搜一次 ──
    struct QueryData {
        insert_rate_10: f32,
        nav_entry_dist: f32,
        visited_35: usize,
        visited_65: usize,
        recall_35: f32,
        recall_65: f32,
        recall_100: f32,
    }

    let mut data: Vec<QueryData> = Vec::with_capacity(nq);

    println!("searching (ef=35/65/100, k={}, nq={})...\n", k, nq);
    let t0 = Instant::now();

    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, nav_dist) = nav.initialize(&train, dim, query);

        let gt_slice = &gt[q * gt_k..(q + 1) * gt_k];

        // ef=35
        let r35 = search_tracked(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            35, &mut visited, &mut pool, po, k, rr,
            &train, dim, query, nav_dist,
        );
        let recall_35 = recall_at_k(&r35.results, gt_slice, k);

        // ef=65
        let r65 = search_tracked(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            65, &mut visited, &mut pool, po, k, rr,
            &train, dim, query, nav_dist,
        );
        let recall_65 = recall_at_k(&r65.results, gt_slice, k);

        // ef=100
        let r100 = search_tracked(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            100, &mut visited, &mut pool, po, k, rr,
            &train, dim, query, nav_dist,
        );
        let recall_100 = recall_at_k(&r100.results, gt_slice, k);

        data.push(QueryData {
            insert_rate_10: r35.insert_rate_at_10,
            nav_entry_dist: r35.nav_entry_dist,
            visited_35: r35.visited_count,
            visited_65: r65.visited_count,
            recall_35,
            recall_65,
            recall_100,
        });
    }

    println!("search time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // ════════════════════════════════════════════════════════════════════
    // 1. 总体 recall 统计
    // ════════════════════════════════════════════════════════════════════
    println!("═══ 1. Recall 总览 ═══\n");
    let avg_r35 = mean(&data.iter().map(|d| d.recall_35).collect::<Vec<_>>());
    let avg_r65 = mean(&data.iter().map(|d| d.recall_65).collect::<Vec<_>>());
    let avg_r100 = mean(&data.iter().map(|d| d.recall_100).collect::<Vec<_>>());
    let avg_vis35 = mean(&data.iter().map(|d| d.visited_35 as f32).collect::<Vec<_>>());

    println!("  ef=35:  avg_recall={:.4}, avg_visited={:.0}", avg_r35, avg_vis35);
    println!("  ef=65:  avg_recall={:.4}", avg_r65);
    println!("  ef=100: avg_recall={:.4}", avg_r100);
    println!("  recall gain (35→65):  {:+.4}", avg_r65 - avg_r35);
    println!("  recall gain (65→100): {:+.4}", avg_r100 - avg_r65);

    // ════════════════════════════════════════════════════════════════════
    // 2. 定义难度标签
    // ════════════════════════════════════════════════════════════════════
    println!("\n═══ 2. 难度定义 ═══\n");

    // Hard = recall@10 at ef=35 < 1.0 (有任何 NN 没找到)
    // 也测试 recall < 0.9 作为更严格的 Hard 定义
    for &recall_threshold in &[1.0, 0.9, 0.8] {
        let n_hard = data.iter().filter(|d| d.recall_35 < recall_threshold).count();
        let n_easy = nq - n_hard;
        println!("  recall@35 < {:.1}: Hard={} ({:.1}%), Easy={} ({:.1}%)",
            recall_threshold, n_hard, 100.0 * n_hard as f32 / nq as f32,
            n_easy, 100.0 * n_easy as f32 / nq as f32);
    }

    // 使用 recall@35 < 1.0 作为主要 Hard 定义
    let recall_threshold = 1.0;
    let is_hard: Vec<bool> = data.iter().map(|d| d.recall_35 < recall_threshold).collect();
    let n_hard = is_hard.iter().filter(|&&h| h).count();
    let n_easy = nq - n_hard;

    println!("\n  → 使用 recall@35 < 1.0 作为 Hard 定义: Hard={}, Easy={}\n", n_hard, n_easy);

    // ════════════════════════════════════════════════════════════════════
    // 3. insert_rate@10 分布分析
    // ════════════════════════════════════════════════════════════════════
    println!("═══ 3. insert_rate@10 分布分析 ═══\n");

    let mut easy_rates: Vec<f32> = data.iter().zip(&is_hard)
        .filter(|(_, &h)| !h).map(|(d, _)| d.insert_rate_10).collect();
    let mut hard_rates: Vec<f32> = data.iter().zip(&is_hard)
        .filter(|(_, &h)| h).map(|(d, _)| d.insert_rate_10).collect();
    easy_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hard_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("  {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "group", "n", "p10", "p25", "p50", "p75", "p90");
    println!("  {}", "-".repeat(84));

    for (name, rates) in &[("Easy", &easy_rates), ("Hard", &hard_rates)] {
        if rates.is_empty() {
            println!("  {:<12} {:<12} (empty)", name, 0);
            continue;
        }
        println!("  {:<12} {:<12} {:<12.4} {:<12.4} {:<12.4} {:<12.4} {:<12.4}",
            name, rates.len(),
            percentile(rates, 0.10),
            percentile(rates, 0.25),
            percentile(rates, 0.50),
            percentile(rates, 0.75),
            percentile(rates, 0.90),
        );
    }

    println!("\n  {:<12} {:<12} {:<12}", "group", "mean", "stddev");
    println!("  {}", "-".repeat(36));
    println!("  {:<12} {:<12.4} {:<12.4}", "Easy", mean(&easy_rates), stddev(&easy_rates));
    if !hard_rates.is_empty() {
        println!("  {:<12} {:<12.4} {:<12.4}", "Hard", mean(&hard_rates), stddev(&hard_rates));
    }

    // 直方图
    println!("\n  insert_rate@10 直方图 (bin=0.05):\n");
    println!("  {:<10} {:<30} {:<30} {:<10}", "range", "Easy", "Hard", "ratio");
    println!("  {}", "-".repeat(80));

    let hist_min = 0.0f32;
    let hist_max = 1.0f32;
    let bin_width = 0.05f32;
    let n_bins = ((hist_max - hist_min) / bin_width) as usize;

    for b in 0..n_bins {
        let lo = hist_min + b as f32 * bin_width;
        let hi = lo + bin_width;
        let e_count = easy_rates.iter().filter(|&&r| r >= lo && r < hi).count();
        let h_count = hard_rates.iter().filter(|&&r| r >= lo && r < hi).count();
        let e_bar = "#".repeat(e_count * 100 / easy_rates.len().max(1));
        let h_bar = "#".repeat(h_count * 100 / hard_rates.len().max(1));
        let ratio = if e_count > 0 { h_count as f32 / e_count as f32 } else { f32::INFINITY };
        println!("  [{:.2},{:.2}) {:<30} {:<30} {:.2}", lo, hi, e_bar, h_bar, ratio);
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. 二分类混淆矩阵（多阈值扫描）
    // ════════════════════════════════════════════════════════════════════
    println!("\n═══ 4. 二分类混淆矩阵（insert_rate@10 < threshold → Hard）═══\n");

    let signal_values: Vec<f32> = data.iter().map(|d| d.insert_rate_10).collect();

    println!("  {:<10} {:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8}",
        "thresh", "TP", "FP", "TN", "FN", "TPR", "FPR", "Acc", "Prec", "F1");
    println!("  {}", "-".repeat(80));

    // 扫描阈值 0.30 ~ 0.70
    let mut best_f1 = 0.0f32;
    let mut best_thresh = 0.0f32;
    for i in 0..=40 {
        let threshold = 0.30 + i as f32 * 0.01;
        let cm = classify(&signal_values, &is_hard, threshold);
        if cm.tp + cm.fp + cm.tn + cm.fn_ == 0 { continue; }
        let f1 = cm.f1();
        if f1 > best_f1 {
            best_f1 = f1;
            best_thresh = threshold;
        }
        println!("  {:<10.2} {:<6} {:<6} {:<6} {:<6} {:<8.3} {:<8.3} {:<8.3} {:<8.3} {:<8.3}",
            threshold, cm.tp, cm.fp, cm.tn, cm.fn_,
            cm.tpr(), cm.fpr(), cm.accuracy(), cm.precision(), f1);
    }

    println!("\n  → Best F1={:.3} at threshold={:.2}", best_f1, best_thresh);

    // ROC AUC
    let auc = roc_auc(&signal_values, &is_hard);
    println!("  → ROC AUC(insert_rate@10) = {:.4}", auc);

    // ════════════════════════════════════════════════════════════════════
    // 5. 与 nav_entry_dist 基线对比
    // ════════════════════════════════════════════════════════════════════
    println!("\n═══ 5. 与 nav_entry_dist 基线对比 ═══\n");

    let nav_values: Vec<f32> = data.iter().map(|d| d.nav_entry_dist).collect();

    // nav_entry_dist > threshold → Hard (距离大 = 难)
    println!("  nav_entry_dist > threshold → Hard:\n");
    println!("  {:<10} {:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8}",
        "thresh", "TP", "FP", "TN", "FN", "TPR", "FPR", "Acc", "Prec", "F1");
    println!("  {}", "-".repeat(80));

    // 对 nav_entry_dist 用 > threshold 判 Hard（方向相反）
    let mut best_f1_nav = 0.0f32;
    let mut best_thresh_nav = 0.0f32;
    let nav_sorted: Vec<f32> = {
        let mut v = nav_values.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };

    for p in &[0.25, 0.50, 0.75, 0.90] {
        let threshold = percentile(&nav_sorted, *p);
        // nav_entry_dist > threshold → Hard
        let mut cm = ConfusionMatrix { tp: 0, fp: 0, tn: 0, fn_: 0 };
        for i in 0..nav_values.len() {
            let predicted_hard = nav_values[i] > threshold;
            match (predicted_hard, is_hard[i]) {
                (true, true) => cm.tp += 1,
                (true, false) => cm.fp += 1,
                (false, false) => cm.tn += 1,
                (false, true) => cm.fn_ += 1,
            }
        }
        let f1 = cm.f1();
        if f1 > best_f1_nav {
            best_f1_nav = f1;
            best_thresh_nav = threshold;
        }
        println!("  {:<10.4} {:<6} {:<6} {:<6} {:<6} {:<8.3} {:<8.3} {:<8.3} {:<8.3} {:<8.3}",
            threshold, cm.tp, cm.fp, cm.tn, cm.fn_,
            cm.tpr(), cm.fpr(), cm.accuracy(), cm.precision(), f1);
    }

    // ROC AUC for nav_entry_dist (invert direction: higher = harder)
    let nav_auc = {
        // Use -nav_values so that "higher signal = harder" like insert_rate
        let neg_nav: Vec<f32> = nav_values.iter().map(|&v| -v).collect();
        roc_auc(&neg_nav, &is_hard)
    };
    println!("\n  → Best F1={:.3} at threshold={:.4}", best_f1_nav, best_thresh_nav);
    println!("  → ROC AUC(nav_entry_dist) = {:.4}", nav_auc);
    println!("  → ROC AUC(insert_rate@10) = {:.4}", auc);
    println!("  → AUC 提升量: {:+.4}", auc - nav_auc);

    // ════════════════════════════════════════════════════════════════════
    // 6. 两阶段方案成本/收益估算
    // ════════════════════════════════════════════════════════════════════
    println!("\n═══ 6. 两阶段方案成本/收益估算 ═══\n");

    // 方案：ef=35 搜索 → insert_rate@10 < threshold → 重跑 ef=65
    // 基线：固定 ef=50
    // 需要计算 ef=50 的 visited 和 recall
    let mut visited_50: Vec<usize> = Vec::with_capacity(nq);
    let mut recall_50: Vec<f32> = Vec::with_capacity(nq);

    println!("  补测 ef=50 基线...");
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        sq4.params.encode_into(query, &mut query_code_buf);
        let (entry_point, nav_dist) = nav.initialize(&train, dim, query);
        let gt_slice = &gt[q * gt_k..(q + 1) * gt_k];

        let r50 = search_tracked(
            &sq4, graph.storage(), entry_point, &query_code_buf,
            50, &mut visited, &mut pool, po, k, rr,
            &train, dim, query, nav_dist,
        );
        visited_50.push(r50.visited_count);
        recall_50.push(recall_at_k(&r50.results, gt_slice, k));
    }

    let avg_vis50 = mean(&visited_50.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let avg_r50 = mean(&recall_50);

    println!("  ef=50 基线: avg_recall={:.4}, avg_visited={:.0}\n", avg_r50, avg_vis50);

    // 估算两阶段方案在不同阈值下的效果
    println!("  {:<10} {:<10} {:<12} {:<12} {:<12} {:<12} {:<10}",
        "thresh", "%rerun", "avg_visited", "vs_ef50", "avg_recall", "vs_ef50", "net_win");
    println!("  {}", "-".repeat(78));

    for i in 0..=20 {
        let threshold = 0.30 + i as f32 * 0.02;
        let mut total_visited: usize = 0;
        let mut total_recall: f32 = 0.0;

        for q in 0..nq {
            // Phase 1: ef=35 (always)
            let vis35 = data[q].visited_35;
            let recall35 = data[q].recall_35;

            if data[q].insert_rate_10 < threshold {
                // Phase 2: rerun at ef=65 (real visited count, measured above)
                total_visited += vis35 + data[q].visited_65;
                total_recall += data[q].recall_65;
            } else {
                total_visited += vis35;
                total_recall += recall35;
            }
        }

        let avg_vis = total_visited as f32 / nq as f32;
        let avg_recall = total_recall / nq as f32;
        let pct_rerun = data.iter().filter(|d| d.insert_rate_10 < threshold).count() as f32 / nq as f32 * 100.0;
        let vis_delta = avg_vis - avg_vis50;
        let recall_delta = avg_recall - avg_r50;
        // net_win: positive = recall gain outweighs visit cost
        // Simple metric: recall_delta * 1000 - vis_delta (arbitrary scaling)
        let net_win = recall_delta * 1000.0 - vis_delta;

        println!("  {:<10.2} {:<10.1} {:<12.0} {:<12.0} {:<12.4} {:<12.4} {:<10.1}",
            threshold, pct_rerun, avg_vis, vis_delta, avg_recall, recall_delta, net_win);
    }

    // ════════════════════════════════════════════════════════════════════
    // 7. 关键结论
    // ════════════════════════════════════════════════════════════════════
    println!("\n═══ 7. 关键结论 ═══\n");

    // 分布重叠度
    let easy_p10 = percentile(&easy_rates, 0.10);
    let hard_p90 = if !hard_rates.is_empty() { percentile(&hard_rates, 0.90) } else { f32::NAN };
    let overlap = easy_p10 < hard_p90;

    println!("  insert_rate@10:");
    println!("    Easy p10={:.4}, Hard p90={:.4}", easy_p10, hard_p90);
    println!("    分布重叠: {}", if overlap { "是（存在误判空间）" } else { "否" });
    println!("    ROC AUC={:.4} (vs nav_entry_dist={:.4})", auc, nav_auc);
    println!("    Best F1={:.3} @ threshold={:.2} (vs nav best F1={:.3})", best_f1, best_thresh, best_f1_nav);

    if auc < 0.65 {
        println!("\n  ⚠ AUC < 0.65：信号区分力不足，二分类误判率高，不建议直接用于两阶段方案");
    } else if auc < 0.75 {
        println!("\n  ⚠ AUC 0.65-0.75：中等区分力，可考虑但需配合连续映射而非硬阈值");
    } else {
        println!("\n  ✓ AUC > 0.75：区分力足够，可考虑用于两阶段方案");
    }

    println!("\n=== 诊断完成 ===");
}
