//! Angle-based pruning 分歧率验证
//!
//! 核心问题：DirectionalPrune 的距离比判据 d²(j,i) < d²(i,q) 与纯角度判据
//! 在多少候选上给出不同结论？
//!
//! 几何关系：
//!   DirectionalPrune 拒绝 i 当 d(j,i) < d(i,q)
//!   等价于 cos(θ) > r_j / (2 * r_i)   （r_j=|q→j|, r_i=|q→i|, θ=∠jqi）
//!
//!   纯角度判据拒绝 i 当 cos(θ) > cos(α)   （α=固定角度阈值）
//!
//! "被角度救回"的候选 = DirectionalPrune 拒绝但角度不拒绝：
//!   r_j/(2*r_i) < cos(θ) ≤ cos(α)
//!
//! 如果这个比例高 → 角度剪枝会保留更多方向独特边 → 值得全量实验
//! 如果这个比例低 → 两种判据几乎等价 → 角度剪枝无收益
//!
//! 用法：cargo run --release --bin angle_prune_check [subset_n]

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{
    PruneStrategy, VamanaBuildConfig, VamanaGraph,
};

fn read_fvecs(path: &str, n_limit: Option<usize>) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect(&format!("open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect(&format!("read {}", path));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n_full = bytes.len() / record_bytes;
    let n = n_limit.unwrap_or(n_full).min(n_full);

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

/// 从三个平方距离计算 cos(θ)，θ 是在 q 处 j 与 i 的夹角
///
/// d_qj_sq = d²(q, j), d_qi_sq = d²(q, i), d_ji_sq = d²(j, i)
/// 余弦定理: d_ji² = d_qj² + d_qi² - 2·d_qj·d_qi·cos(θ)
#[inline(always)]
fn cos_angle(d_qj_sq: f32, d_qi_sq: f32, d_ji_sq: f32) -> f32 {
    let d_qj = d_qj_sq.sqrt();
    let d_qi = d_qi_sq.sqrt();
    if d_qj < 1e-12 || d_qi < 1e-12 {
        return 1.0; // 退化情况
    }
    (d_qj_sq + d_qi_sq - d_ji_sq) / (2.0 * d_qj * d_qi)
}

struct DivergenceStats {
    /// 总候选数（所有节点的所有候选之和）
    total_candidates: usize,
    /// DirectionalPrune 拒绝的候选数
    dir_rejected: usize,
    /// 各角度阈值下"被角度救回"的候选数
    rescued_by_angle: Vec<(f32, usize)>, // (angle_degrees, count)
    /// 按距离比分桶统计分歧率
    /// ratio_bin = r_j / r_i 的区间
    divergence_by_ratio: Vec<(String, usize, usize)>, // (bin_label, total_rejected, rescued)
}

/// 对单个节点执行分歧分析
///
/// candidates: 已按距离升序排序的 (dist_sq, id) 列表
/// vectors, dim: 向量数据
///
/// 返回:
///   total_candidates: 候选总数
///   dir_rejected: 被 DirectionalPrune Pass 1 拒绝的数量
///   rescued: 各角度阈值下被"救回"的数量
fn analyze_node(
    sorted: &[(f32, u32)],
    vectors: &[f32],
    dim: usize,
    angle_thresholds_cos: &[f32], // cos(α) for each threshold
) -> (usize, usize, Vec<usize>, Vec<(String, usize, usize)>) {
    let n = sorted.len();
    let mut accepted: Vec<(f32, u32)> = Vec::with_capacity(40); // (dist_sq, id)

    let mut dir_rejected = 0usize;

    // 按距离比 r_j/r_i 分桶
    // bins: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0)
    let bin_labels = ["r_j/r_i<0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"];
    let mut bin_rejected = [0usize; 4];
    let mut bin_rescued: Vec<Vec<usize>> = (0..4).map(|_| vec![0; angle_thresholds_cos.len()]).collect();

    for i in 0..n {
        let dist_qi_sq = sorted[i].0;
        let id_i = sorted[i].1;
        let vec_i = &vectors[id_i as usize * dim..(id_i as usize + 1) * dim];

        // DirectionalPrune Pass 1 判定
        let mut dir_good = true;
        let mut occluder_j: Option<usize> = None; // 记录哪个 j 导致了遮挡

        for (j_idx, &(dist_qj_sq, id_j)) in accepted.iter().enumerate() {
            let vec_j = &vectors[id_j as usize * dim..(id_j as usize + 1) * dim];
            let dist_ji_sq = l2_simd(vec_j, vec_i);

            // DirectionalPrune: d(j,i) < d(i,q) → 遮挡
            if dist_ji_sq < dist_qi_sq {
                dir_good = false;
                occluder_j = Some(j_idx);
                break;
            }
        }

        if dir_good {
            accepted.push((dist_qi_sq, id_i));
        } else {
            dir_rejected += 1;

            // 计算角度并检查各阈值
            let j_idx = occluder_j.unwrap();
            let (dist_qj_sq, id_j) = accepted[j_idx];
            let vec_j = &vectors[id_j as usize * dim..(id_j as usize + 1) * dim];
            let dist_ji_sq = l2_simd(vec_j, vec_i);

            let cos_theta = cos_angle(dist_qj_sq, dist_qi_sq, dist_ji_sq);

            // DirectionalPrune 的有效阈值: cos(θ) > r_j/(2*r_i)
            let r_j = dist_qj_sq.sqrt();
            let r_i = dist_qi_sq.sqrt();
            let dir_threshold = r_j / (2.0 * r_i);

            // 分桶
            let ratio = r_j / r_i;
            let bin_idx = if ratio < 0.25 { 0 }
                else if ratio < 0.50 { 1 }
                else if ratio < 0.75 { 2 }
                else { 3 };
            bin_rejected[bin_idx] += 1;

            // 检查各角度阈值
            // 被角度救回 = DirectionalPrune 拒绝 (cos_θ > dir_threshold)
            //              但角度不拒绝 (cos_θ ≤ cos_α)
            // 即: dir_threshold < cos_θ ≤ cos_α
            for (t_idx, &cos_alpha) in angle_thresholds_cos.iter().enumerate() {
                if cos_theta <= cos_alpha {
                    bin_rescued[bin_idx][t_idx] += 1;
                }
            }
        }
    }

    // 汇总各 bin
    let mut by_ratio: Vec<(String, usize, usize)> = Vec::new();
    for b in 0..4 {
        // 只报告第一个角度阈值（60°）的 rescue 数
        let rescued = if !bin_rescued[b].is_empty() { bin_rescued[b][0] } else { 0 };
        by_ratio.push((bin_labels[b].to_string(), bin_rejected[b], rescued));
    }

    // 汇总各角度阈值的总 rescue 数
    let mut total_rescued: Vec<usize> = vec![0; angle_thresholds_cos.len()];
    for b in 0..4 {
        for (t, &count) in bin_rescued[b].iter().enumerate() {
            total_rescued[t] += count;
        }
    }

    (n, dir_rejected, total_rescued, by_ratio)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subset_n: usize = if args.len() > 1 {
        args[1].parse().expect("subset_n must be a number")
    } else {
        100000
    };

    println!("=== Angle-based pruning 分歧率验证 ===\n");
    println!("数据: {} 万节点子集", subset_n / 10000);

    // 读取数据
    let (mut train, dim, _n_full) = read_fvecs("data/sift/sift_base.fvecs", Some(subset_n));
    for v in train.iter_mut() {
        *v /= 255.0;
    }
    let n = train.len() / dim;

    // 建图
    println!("\n--- 建图 ---");
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

    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.1}s", t0.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    //  构建候选集：N(q) ∪ N2(q)
    // ═══════════════════════════════════════════════════════════════
    println!("\n--- 构建候选集并分析 ---");

    // 角度阈值: 30°, 45°, 60°, 75°
    let angle_degrees: Vec<f32> = vec![30.0, 45.0, 60.0, 75.0];
    let angle_thresholds_cos: Vec<f32> = angle_degrees.iter().map(|&d| (d * std::f32::consts::PI / 180.0).cos()).collect();

    // 抽样 10000 个节点（10%）
    let sample_size = 10000.min(n);
    let sample_step = n / sample_size;

    let mut total_candidates = 0usize;
    let mut total_dir_rejected = 0usize;
    let mut total_rescued: Vec<usize> = vec![0; angle_thresholds_cos.len()];
    let mut total_by_ratio: Vec<(String, usize, usize)> = vec![
        ("r_j/r_i<0.25".to_string(), 0, 0),
        ("0.25-0.50".to_string(), 0, 0),
        ("0.50-0.75".to_string(), 0, 0),
        ("0.75-1.00".to_string(), 0, 0),
    ];

    let t0 = Instant::now();
    let mut nodes_analyzed = 0usize;

    for node_idx in 0..n {
        if node_idx % sample_step != 0 {
            continue;
        }

        let node = node_idx as u32;
        let neighbors = graph.neighbors(node);
        if neighbors.is_empty() {
            continue;
        }

        // 构建候选集 = N(q) ∪ N2(q)
        let mut candidate_set: HashSet<u32> = HashSet::new();
        for &nbr in neighbors {
            candidate_set.insert(nbr);
            for &nbr2 in graph.neighbors(nbr) {
                if nbr2 != node {
                    candidate_set.insert(nbr2);
                }
            }
        }

        // 计算距离并排序
        let node_vec = &train[node as usize * dim..(node as usize + 1) * dim];
        let mut sorted: Vec<(f32, u32)> = candidate_set
            .iter()
            .map(|&id| {
                let v = &train[id as usize * dim..(id as usize + 1) * dim];
                (l2_simd(node_vec, v), id)
            })
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 分析
        let (n_cand, n_rejected, n_rescued, by_ratio) =
            analyze_node(&sorted, &train, dim, &angle_thresholds_cos);

        total_candidates += n_cand;
        total_dir_rejected += n_rejected;
        for (t, &count) in n_rescued.iter().enumerate() {
            total_rescued[t] += count;
        }
        for (b, (_, tot, res)) in by_ratio.iter().enumerate() {
            total_by_ratio[b].1 += tot;
            total_by_ratio[b].2 += res;
        }

        nodes_analyzed += 1;
    }

    let analysis_time = t0.elapsed().as_secs_f64();

    // ═══════════════════════════════════════════════════════════════
    //  报告
    // ═══════════════════════════════════════════════════════════════
    println!("分析节点数: {} / {}", nodes_analyzed, n);
    println!("分析时间: {:.1}s", analysis_time);
    println!("\n候选集统计:");
    println!("  总候选数: {}", total_candidates);
    println!("  平均每节点候选: {:.1}", total_candidates as f64 / nodes_analyzed as f64);
    println!("  DirectionalPrune 拒绝: {} ({:.1}%)", total_dir_rejected,
        total_dir_rejected as f64 / total_candidates as f64 * 100.0);

    println!("\n方向分歧分析 (被角度救回的候选):");
    println!("  角度阈值  | 救回数  | 占拒绝比  | 占总候选  | 含义");
    println!("  ----------|---------|-----------|-----------|-----");

    let threshold_meanings = [
        "极保守(几乎只救 90°+)",
        "保守(救 >45° 的)",
        "中等(救 >30° 的)",
        "激进(几乎不剪)",
    ];

    for (i, &angle) in angle_degrees.iter().enumerate() {
        let rescued = total_rescued[i];
        let pct_rejected = if total_dir_rejected > 0 {
            rescued as f64 / total_dir_rejected as f64 * 100.0
        } else {
            0.0
        };
        let pct_total = rescued as f64 / total_candidates as f64 * 100.0;
        println!("  α={:>5.0}°  | {:>7} | {:>7.1}%  | {:>7.1}%  | {}",
            angle, rescued, pct_rejected, pct_total, threshold_meanings[i]);
    }

    // 按距离比分桶（使用 60° 阈值）
    println!("\n按 r_j/r_i 比值分桶（α=60° 阈值）:");
    println!("  比值区间      | 拒绝数  | 救回数  | 救回率  | 解读");
    println!("  --------------|---------|---------|---------|----");

    let interpretations = [
        "j 极近→DirPrune 阈值≈90°, 角度更保守",
        "j 较近→DirPrune 阈值≈75°, 有分歧空间",
        "j 中等→DirPrune 阈值≈48°, 分歧收窄",
        "j 与 i 等距→DirPrune 阈值≈30°, 几乎无分歧",
    ];

    for (b, (label, tot, res)) in total_by_ratio.iter().enumerate() {
        let rate = if *tot > 0 { *res as f64 / *tot as f64 * 100.0 } else { 0.0 };
        println!("  {:>14} | {:>7} | {:>7} | {:>6.1}% | {}", label, tot, res, rate, interpretations[b]);
    }

    // 核心结论
    let rescue_rate_60 = if total_dir_rejected > 0 {
        total_rescued[2] as f64 / total_dir_rejected as f64 * 100.0  // 60° is index 2
    } else {
        0.0
    };
    let rescue_rate_45 = if total_dir_rejected > 0 {
        total_rescued[1] as f64 / total_dir_rejected as f64 * 100.0
    } else {
        0.0
    };

    println!("\n=== 结论 ===");
    if rescue_rate_60 > 20.0 {
        println!("60° 阈值救回率 {:.1}% > 20%，角度剪枝有显著差异 → 值得全量实验", rescue_rate_60);
    } else if rescue_rate_45 > 15.0 {
        println!("45° 阈值救回率 {:.1}% > 15%，角度剪枝有中等差异 → 可考虑实验", rescue_rate_45);
    } else {
        println!("60° 阈值救回率 {:.1}%, 45° 阈值救回率 {:.1}% — 差异太小", rescue_rate_60, rescue_rate_45);
        println!("DirectionalPrune 的距离判据与纯角度判据几乎等价，角度剪枝无收益。");
    }

    println!("\n=== 验证完成 ===");
}