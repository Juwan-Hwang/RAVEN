//! 搜索深度诊断：avg_visited 的 1180 个节点花在哪里？
//!
//! 追踪真正最近邻（暴力 GT #1）在搜索过程中第几个被访问到。
//! - 如果前 5% 就找到了 → 瓶颈是"验证"→ 分层无用
//! - 如果 50% 才找到 → 瓶颈是"找路"→ 分层有用
//!
//! 用法：cargo run --release --bin search_depth_check [subset_n]

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};

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

/// f32 距离 + u32 节点 ID，用 total_cmp 排序
#[derive(Clone, Copy)]
struct DistNode(f32, u32);

impl PartialEq for DistNode { fn eq(&self, o: &Self) -> bool { self.0.total_cmp(&o.0) == std::cmp::Ordering::Equal } }
impl Eq for DistNode {}
impl PartialOrd for DistNode { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for DistNode { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.0.total_cmp(&o.0).then(self.1.cmp(&o.1)) } }

struct SearchResult {
    total_visited: usize,
    nn_pop_order: usize,
    nn_found: bool,
}

fn search_with_depth(
    graph: &VamanaGraph,
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    entry_point: u32,
    ef: usize,
    k: usize,
    true_nn: u32,
) -> SearchResult {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let mut visited: HashSet<u32> = HashSet::new();
    // min-heap: 最近候选先出
    let mut cand: BinaryHeap<Reverse<DistNode>> = BinaryHeap::new();
    // max-heap: 最远结果先出（用于 ef 截断）
    let mut results: BinaryHeap<DistNode> = BinaryHeap::new();

    let ep_vec = &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim];
    let ep_dist = l2_simd(query, ep_vec);
    cand.push(Reverse(DistNode(ep_dist, entry_point)));

    let mut pop_order = 0usize;
    let mut nn_pop_order = 0usize;
    let mut nn_found = false;

    while let Some(Reverse(DistNode(dist, node))) = cand.pop() {
        if visited.contains(&node) { continue; }
        visited.insert(node);
        pop_order += 1;

        if node == true_nn {
            nn_pop_order = pop_order;
            nn_found = true;
        }

        // Vamana break: 如果候选比结果堆中最远的还远，停止
        if results.len() >= k {
            if let Some(DistNode(worst, _)) = results.peek() {
                if dist > *worst { break; }
            }
        }

        results.push(DistNode(dist, node));
        if results.len() > ef { results.pop(); }

        // 扩展邻居
        for &nbr in graph.neighbors(node) {
            if !visited.contains(&nbr) {
                let nv = &vectors[nbr as usize * dim..(nbr as usize + 1) * dim];
                let nd = l2_simd(query, nv);
                let should_add = if results.len() < ef { true }
                    else if let Some(DistNode(worst, _)) = results.peek() { nd < *worst }
                    else { true };
                if should_add {
                    cand.push(Reverse(DistNode(nd, nbr)));
                }
            }
        }
    }

    SearchResult { total_visited: visited.len(), nn_pop_order, nn_found }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subset_n: usize = if args.len() > 1 { args[1].parse().expect("subset_n") } else { 100000 };

    println!("=== 搜索深度诊断 ===\n");
    println!("数据: {} 万节点子集", subset_n / 10000);

    let (mut train, dim, _) = read_fvecs("data/sift/sift_base.fvecs", Some(subset_n));
    let (mut test, _, nq_full) = read_fvecs("data/sift/sift_query.fvecs", None);
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    let n = train.len() / dim;
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

    // 暴力 GT（只算 top-1）
    println!("\n计算暴力 GT (top-1)...");
    let t0 = Instant::now();
    let true_nns: Vec<u32> = (0..nq).map(|q| {
        let query = &test[q * dim..(q + 1) * dim];
        (0..n).map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            (l2_simd(query, v), i as u32)
        }).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
    }).collect();
    println!("GT 完成: {:.1}s", t0.elapsed().as_secs_f64());

    // 搜索追踪
    println!("\n--- 搜索深度追踪 (ef=50, {} 查询) ---", nq);
    let ef = 50;
    let k = 10;
    let entry_point = graph.entry_point();

    let mut total_visited = 0usize;
    let mut total_nn_pop = 0usize;
    let mut nn_found_count = 0usize;
    let mut ratio_bins = [0usize; 5];
    let mut nn_pop_values: Vec<f64> = Vec::with_capacity(nq);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let r = search_with_depth(&graph, &train, dim, query, entry_point, ef, k, true_nns[q]);

        total_visited += r.total_visited;
        if r.nn_found {
            total_nn_pop += r.nn_pop_order;
            nn_found_count += 1;
            nn_pop_values.push(r.nn_pop_order as f64);
            let ratio = r.nn_pop_order as f64 / r.total_visited as f64;
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