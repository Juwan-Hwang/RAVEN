//! 搜索深度诊断（100万全量版）
//!
//! 用真实搜索路径（LayeredNavigation + greedy_search_vec_reuse）追踪：
//! 真正最近邻（暴力 GT #1）在访问序列中排第几。
//!
//! 用法：cargo run --release --bin search_depth_full
//!       cargo run --release --bin search_depth_full 100000  (10万子集对比)

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};
use raven::memory::{HybridBlockedCsr, VisitedTracker};
use raven::graph::linear_pool::LinearPool;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subset_n: Option<usize> = if args.len() > 1 {
        Some(args[1].parse().expect("subset_n"))
    } else {
        None
    };

    let label = if subset_n.is_some() { "子集" } else { "全量" };
    println!("=== 搜索深度诊断 ({}) ===\n", label);

    // 读取数据
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs", subset_n);
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs", None);

    // 全量时读取已有 GT；子集时暴力计算
    let true_nns: Vec<u32>;

    if subset_n.is_none() {
        // 全量：使用已有的 groundtruth 文件
        let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
        true_nns = (0..nq_full).map(|q| gt[q * gt_k] as u32).collect();
        println!("使用已有 GT 文件 (n={})", n);
    } else {
        // 子集：暴力计算 top-1
        println!("暴力计算 GT (n={})...", n);
        let t0 = Instant::now();
        true_nns = (0..nq_full).map(|q| {
            let query = &test[q * dim..(q + 1) * dim];
            (0..n).map(|i| {
                let v = &train[i * dim..(i + 1) * dim];
                (l2_simd(query, v), i as u32)
            }).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
        }).collect();
        println!("GT 完成: {:.1}s", t0.elapsed().as_secs_f64());
    }

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let nq = 200.min(nq_full);

    // 建图
    println!("\n--- 建图 ---");
    let config = VamanaBuildConfig {
        alpha: 1.2, l_build: 384, r_max: 40, r_soft: 60,
        max_iterations: 2, saturate: false,
        enable_layered_nav: true, nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.1}s", t0.elapsed().as_secs_f64());

    // 获取导航层
    let nav = graph.layered_nav().expect("需要分层导航");

    // 准备搜索资源
    let ef = 50;
    let k = 10;
    let po = 16;

    let mut visited = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);

    // 搜索追踪
    println!("\n--- 搜索深度追踪 (ef={}, {} 查询) ---", ef, nq);

    let mut total_visited = 0usize;
    let mut total_nn_pop = 0usize;
    let mut nn_found_count = 0usize;
    let mut ratio_bins = [0usize; 5];
    let mut nn_pop_values: Vec<f64> = Vec::with_capacity(nq);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];

        // 1. 用导航层选 entry_point（与 GraphSearcher::search 完全一致）
        let (entry_point, _nav_dist) = nav.initialize(&train, dim, query);

        // 2. 执行 greedy search（真实 f32 路径）
        let _results = VamanaGraph::greedy_search_vec_reuse(
            &train, dim, graph.storage(), entry_point, query, ef,
            &mut visited, &mut pool, po,
        );

        // 3. 从 visited.history 获取访问顺序
        let visited_nodes = visited.visited_nodes();
        let total_v = visited_nodes.len();
        total_visited += total_v;

        // 4. 找 NN 在访问序列中的位置
        let true_nn = true_nns[q];
        if let Some(pos) = visited_nodes.iter().position(|&n| n == true_nn) {
            let pop_order = pos + 1; // 1-based
            total_nn_pop += pop_order;
            nn_found_count += 1;
            nn_pop_values.push(pop_order as f64);

            let ratio = pop_order as f64 / total_v as f64;
            let bin = if ratio < 0.05 { 0 } else if ratio < 0.10 { 1 }
                else if ratio < 0.20 { 2 } else if ratio < 0.50 { 3 } else { 4 };
            ratio_bins[bin] += 1;
        }
    }
    println!("搜索时间: {:.1}s", t0.elapsed().as_secs_f64());

    // 报告
    println!("\n=== 核心结果 ===");
    println!("平均总访问: {:.1}", total_visited as f64 / nq as f64);
    println!("NN 找到率: {}/{} ({:.1}%)", nn_found_count, nq, nn_found_count as f64 / nq as f64 * 100.0);

    if nn_found_count > 0 {
        let avg_nn_pop = total_nn_pop as f64 / nn_found_count as f64;
        let avg_visited = total_visited as f64 / nq as f64;
        let avg_ratio = avg_nn_pop / avg_visited;

        println!("\n平均 NN 发现序号: {:.1} (前 {:.1}% 的访问就找到了真正最近邻)", avg_nn_pop, avg_ratio * 100.0);
        println!("平均总访问: {:.1}", avg_visited);
        println!("\"找路\"开销: {:.0} 个节点 ({:.1}%)", avg_nn_pop, avg_ratio * 100.0);
        println!("\"验证\"开销: {:.0} 个节点 ({:.1}%)", avg_visited - avg_nn_pop, (1.0 - avg_ratio) * 100.0);

        nn_pop_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = nn_pop_values[nn_pop_values.len() / 2];
        let p90 = nn_pop_values[nn_pop_values.len() * 9 / 10];
        let p99 = nn_pop_values[nn_pop_values.len() * 99 / 100];
        let mx = *nn_pop_values.last().unwrap();

        println!("\nNN 发现序号分布:");
        println!("  p50: {:.0}", p50);
        println!("  p90: {:.0}", p90);
        println!("  p99: {:.0}", p99);
        println!("  max: {:.0}", mx);

        println!("\nNN 发现序号 / 总访问 比值分布:");
        println!("  < 5%  (前 5% 就找到): {} ({:.1}%)", ratio_bins[0], ratio_bins[0] as f64 / nq as f64 * 100.0);
        println!("  5-10%:                {} ({:.1}%)", ratio_bins[1], ratio_bins[1] as f64 / nq as f64 * 100.0);
        println!("  10-20%:               {} ({:.1}%)", ratio_bins[2], ratio_bins[2] as f64 / nq as f64 * 100.0);
        println!("  20-50%:               {} ({:.1}%)", ratio_bins[3], ratio_bins[3] as f64 / nq as f64 * 100.0);
        println!("  50-100% (最后才找到): {} ({:.1}%)", ratio_bins[4], ratio_bins[4] as f64 / nq as f64 * 100.0);

        println!("\n=== 结论 ===");
        if avg_ratio < 0.10 {
            println!("NN 在前 {:.1}% 被发现 → 瓶颈是\"验证\"而非\"找路\"", avg_ratio * 100.0);
            println!("分层无法帮助：搜索已快速定位目标区域，剩余访问是 ef-bounded 验证开销。");
        } else if avg_ratio < 0.30 {
            println!("NN 在前 {:.1}% 被发现 → 中等\"找路\"开销", avg_ratio * 100.0);
            println!("分层可能有边际收益，但大头仍在验证阶段。");
        } else {
            println!("NN 在前 {:.1}% 才被发现 → 瓶颈是\"找路\"（导航路径太长）", avg_ratio * 100.0);
            println!("分层有显著收益空间：缩短导航路径可直接降低 avg_visited。");
        }
    }

    println!("\n=== 诊断完成 ===");
}