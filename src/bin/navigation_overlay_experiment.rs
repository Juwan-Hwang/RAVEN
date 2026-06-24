//! NavigationLayer centroid overlay 实验
//!
//! 验证 NavigationLayer 是否对搜索有用
//! 对比：
//!   A. 默认 medoid entry_point（当前生产路径）
//!   B. 最近 centroid entry_point（NavigationLayer 提供）
//!
//! 指标：recall@10, QPS, avg_visited
//! 若 B 的 recall/QPS 不劣于 A，则 NavigationLayer 有用，可集成

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, NavigationLayer, NavigationConfig};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::memory::VisitedTracker;

/// f32 包装（BinaryHeap 要求 Ord）
#[derive(Debug, Clone, Copy)]
struct OrdF32(f32);
impl PartialEq for OrdF32 { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { self.0.partial_cmp(&o.0) } }
impl Ord for OrdF32 { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal) } }

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

/// 从指定 entry_point 搜索，返回 (top-L, visited_count)
fn search_from_entry(
    vectors: &[f32],
    dim: usize,
    graph: &VamanaGraph,
    entry: u32,
    query: &[f32],
    ef_search: usize,
) -> (Vec<u32>, usize) {
    // 手动实现 greedy search，统计 visited 数量
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = vectors.len() / dim;
    let mut visited = VisitedTracker::new(n, ef_search);
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(ef_search * 2);
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(ef_search + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrdF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= ef_search {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 {
                    break;
                }
            }
        }
        results.push((dist, node));
        if results.len() > ef_search {
            results.pop();
        }
        for &neighbor in graph.storage().neighbors(node) {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor as usize + 1) * dim]);
                candidates.push(Reverse((OrdF32(d), neighbor)));
            }
        }
    }

    let top: Vec<u32> = results.into_iter().map(|(_, id)| id).collect();
    let visited_count = visited.visited_nodes().len();
    (top, visited_count)
}

fn main() {
    println!("=== NavigationLayer centroid overlay 实验 ===");
    println!();

    // 1. 加载 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);

    // 归一化
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    // 2. 构建 VamanaGraph（默认 medoid entry）
    println!();
    println!("=== 构建 VamanaGraph（α=1.0, r_max=32, l_build=100）===");
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
    println!("建图: {:.1}s", t0.elapsed().as_secs_f64());
    println!("entry_point (medoid): {}", graph.entry_point());

    // 3. 构建 NavigationLayer（centroid overlay，√N 个 centroid）
    println!();
    println!("=== 构建 NavigationLayer（centroid overlay, √N={}）===", (n as f64).sqrt() as usize);
    let t0 = Instant::now();
    let nav_config = NavigationConfig {
        enable_centroid_overlay: true,
        centroid_count: None, // √N
    };
    let nav = NavigationLayer::new(n, &train, dim, nav_config);
    println!("NavigationLayer 构建: {:.1}s", t0.elapsed().as_secs_f64());
    println!("centroid 数量: {}", nav.centroids().len());

    // 4. 对比搜索
    let k = 10;
    let ef_search = 100;
    let gt_stride = gt_k;

    // A. 默认 medoid entry
    println!();
    println!("=== A. 默认 medoid entry_point ===");
    let t0 = Instant::now();
    let mut hits_a = 0usize;
    let mut visited_sum_a = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let (top, visited) = search_from_entry(&train, dim, &graph, graph.entry_point(), query, ef_search);
        visited_sum_a += visited;
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if top.contains(&(g as u32)) {
                hits_a += 1;
            }
        }
    }
    let time_a = t0.elapsed().as_secs_f64();
    let recall_a = hits_a as f64 / (nq * k) as f64;
    let qps_a = nq as f64 / time_a;
    let avg_visited_a = visited_sum_a as f64 / nq as f64;
    println!("recall@10={:.4}, QPS={:.0}, avg_visited={:.1}, avg_latency={:.3}ms",
        recall_a, qps_a, avg_visited_a, time_a * 1000.0 / nq as f64);

    // B. 最近 centroid entry
    println!();
    println!("=== B. 最近 centroid entry_point（NavigationLayer）===");
    let t0 = Instant::now();
    let mut hits_b = 0usize;
    let mut visited_sum_b = 0usize;
    let mut entry_match_count = 0usize; // centroid 恰好是 medoid 的次数
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        // 找最近的 centroid
        let mut best_centroid = nav.centroids()[0];
        let mut best_dist = f32::MAX;
        for &c in nav.centroids() {
            let cv = &train[c as usize * dim..(c as usize + 1) * dim];
            let d = l2_simd(query, cv);
            if d < best_dist {
                best_dist = d;
                best_centroid = c;
            }
        }
        if best_centroid == graph.entry_point() {
            entry_match_count += 1;
        }
        let (top, visited) = search_from_entry(&train, dim, &graph, best_centroid, query, ef_search);
        visited_sum_b += visited;
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if top.contains(&(g as u32)) {
                hits_b += 1;
            }
        }
    }
    let time_b = t0.elapsed().as_secs_f64();
    let recall_b = hits_b as f64 / (nq * k) as f64;
    let qps_b = nq as f64 / time_b;
    let avg_visited_b = visited_sum_b as f64 / nq as f64;
    println!("recall@10={:.4}, QPS={:.0}, avg_visited={:.1}, avg_latency={:.3}ms",
        recall_b, qps_b, avg_visited_b, time_b * 1000.0 / nq as f64);
    println!("(centroid 恰好是 medoid 的次数: {}/{})", entry_match_count, nq);

    // 5. 汇总
    println!();
    println!("=== 汇总 ===");
    println!("{:<25} {:>10} {:>10} {:>12} {:>12}", "方案", "recall@10", "QPS", "avg_visited", "latency_ms");
    println!("{:-<69}", "");
    println!("{:<25} {:>10.4} {:>10.0} {:>12.1} {:>12.3}", "A. medoid entry", recall_a, qps_a, avg_visited_a, time_a * 1000.0 / nq as f64);
    println!("{:<25} {:>10.4} {:>10.0} {:>12.1} {:>12.3}", "B. centroid entry", recall_b, qps_b, avg_visited_b, time_b * 1000.0 / nq as f64);
    println!();

    // 判定
    let recall_diff = recall_b - recall_a;
    let qps_diff_pct = (qps_b - qps_a) / qps_a * 100.0;
    let visited_diff_pct = (avg_visited_b - avg_visited_a) / avg_visited_a * 100.0;
    println!("差异: recall {:+.4}, QPS {:+.1}%, visited {:+.1}%", recall_diff, qps_diff_pct, visited_diff_pct);
    println!();

    if recall_b >= recall_a - 0.001 && qps_b >= qps_a * 0.95 {
        println!("结论: NavigationLayer centroid overlay 不劣于 medoid，可集成");
    } else if recall_b > recall_a + 0.001 || (recall_b >= recall_a - 0.001 && qps_b > qps_a * 1.05) {
        println!("结论: NavigationLayer centroid overlay 有正向收益，建议集成");
    } else {
        println!("结论: NavigationLayer centroid overlay 无明显收益，建议删除");
    }
}
