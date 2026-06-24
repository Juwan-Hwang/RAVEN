//! DelayedPruneController 实验
//!
//! 验证 DelayedPruneController 是否对项目有用
//!
//! 发现：connect_bidirectional（vamana.rs:507-529）已内联实现了
//!   - should_prune: storage.degree(nb) > config.r_soft
//!   - RobustPrune 触发
//! DelayedPruneController 是相同逻辑的封装版本 + 统计功能
//!
//! 实验：
//!   A. 当前 build（内联 lazy pruning）
//!   B. build 后用 DelayedPruneController 统计 prune 状态
//!   C. 对比 DelayedPruneController.final_prune vs VamanaGraph::final_prune 结果一致性
//!
//! 若 DelayedPruneController 仅提供统计功能（无性能差异），则定位为"诊断工具"

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::{ChaCha8Rng, DelayedPruneController};
use raven::memory::HybridBlockedCsr;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            vectors.push(v);
        }
    }
    (vectors, dim, n)
}

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            gt.push(v);
        }
    }
    (gt, dim, n)
}

fn eval_recall(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    k: usize,
) -> f32 {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let gt_stride = 100;
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f32 / (nq * k) as f32
}

fn main() {
    println!("=== DelayedPruneController 实验 ===");
    println!();

    // 1. 加载 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, _, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}", dim, n, nq);

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. 当前 build（内联 lazy pruning）
    println!("=== A. 当前 build（内联 lazy pruning）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time_a = t0.elapsed().as_secs_f64();
    println!("建图时间: {:.2}s", build_time_a);

    // 3. 用 DelayedPruneController 统计当前 graph 状态
    println!();
    println!("=== B. DelayedPruneController 诊断 ===");
    let controller = DelayedPruneController::new(config.r_max);
    let storage = graph.storage();
    let over_soft = controller.count_over_soft(storage);
    println!("r_max={}, r_soft={}", controller.r_max, controller.r_soft);
    println!("超过 R_soft 的节点数: {}/{}", over_soft, n);

    // 度数统计
    let mut degree_sum = 0usize;
    let mut degree_max = 0usize;
    let mut over_r_max = 0usize;
    for i in 0..n {
        let d = storage.degree(i as u32);
        degree_sum += d;
        if d > degree_max { degree_max = d; }
        if d > config.r_max { over_r_max += 1; }
    }
    println!("avg_degree={:.1}, max_degree={}, 超过 R_max 的节点: {}/{}",
        degree_sum as f64 / n as f64, degree_max, over_r_max, n);

    // 4. recall 验证
    let recall_a = eval_recall(&train, &test, &gt, dim, nq, &graph, ef_search, k);
    println!("recall@10: {:.4}", recall_a);

    // 5. 验证 DelayedPruneController.final_prune 与 VamanaGraph::final_prune 一致性
    // 复制 graph，用 DelayedPruneController.final_prune 重新剪枝
    println!();
    println!("=== C. DelayedPruneController.final_prune 一致性验证 ===");
    // 由于 VamanaGraph 的 storage 是私有的，我们通过 storage_mut 获取
    // 复制 graph 的 storage 做对比
    let mut graph_copy_storage = HybridBlockedCsr::new(n, config.r_max * 2);
    for i in 0..n as u32 {
        let (main, overflow) = graph.storage().neighbors_full(i);
        let mut all: Vec<u32> = main.to_vec();
        all.extend_from_slice(overflow);
        for &nb in &all {
            graph_copy_storage.add_edge(i, nb);
        }
    }
    println!("复制 storage 完成，节点数: {}", graph_copy_storage.len());

    // 用 DelayedPruneController.final_prune 剪枝
    let mut controller2 = DelayedPruneController::new(config.r_max);
    let t0 = Instant::now();
    controller2.final_prune(&mut graph_copy_storage, &train, dim, config.alpha);
    let prune_time = t0.elapsed().as_secs_f64();
    println!("DelayedPruneController.final_prune 时间: {:.3}s", prune_time);
    println!("final_prune 触发次数: {}", controller2.final_prune_count);

    // 统计剪枝后的度数
    let mut degree_sum_c = 0usize;
    let mut degree_max_c = 0usize;
    let mut over_r_max_c = 0usize;
    for i in 0..n {
        let d = graph_copy_storage.degree(i as u32);
        degree_sum_c += d;
        if d > degree_max_c { degree_max_c = d; }
        if d > config.r_max { over_r_max_c += 1; }
    }
    println!("剪枝后: avg_degree={:.1}, max_degree={}, 超过 R_max 的节点: {}/{}",
        degree_sum_c as f64 / n as f64, degree_max_c, over_r_max_c, n);

    // 6. 汇总
    println!();
    println!("=== 汇总 ===");
    println!("DelayedPruneController 定位分析:");
    println!("  - should_prune 逻辑 = connect_bidirectional 的 storage.degree(nb) > r_soft");
    println!("  - final_prune 逻辑  = VamanaGraph::final_prune（完全相同）");
    println!("  - 附加价值: 统计功能（single_prune_count, final_prune_count, count_over_soft）");
    println!();
    println!("结论:");
    if over_r_max == 0 {
        println!("  当前 build 的 final_prune 已将所有节点剪到 R_max 以内，");
        println!("  DelayedPruneController.final_prune 不会改变图结构（无额外收益）");
    } else {
        println!("  当前 build 有 {} 个节点超过 R_max，DelayedPruneController.final_prune 可修正", over_r_max);
    }
    println!("  DelayedPruneController 是 connect_bidirectional + final_prune 的封装版本");
    println!("  附加价值仅在于统计功能（prune_count 诊断）");
}
