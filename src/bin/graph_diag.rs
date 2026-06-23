//! 图质量诊断
//!
//! 验证：
//! 1. 边长分布（是否缺少长程导航边）
//! 2. 建图节点自身能否找到自己的近邻（图 navigability）
//! 3. 标准 break 搜索访问的节点数（验证是否过早终止）

use std::fs::File;
use std::io::Read;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取失败");
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

fn main() {
    println!("=== 图质量诊断 ===");
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    println!("siftsmall: dim={}, n={}", dim, n);

    // 建图（标准 break, Vamana 论文参数）
    println!("\n[1] 建图（l_build=200, r_max=64, alpha=1.2, max_iter=2, 标准 break）...");
    let t0 = std::time::Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图时间: {:.2}s", t0.elapsed().as_secs_f64());

    // [2] 边长分布统计
    println!("\n[2] 边长分布统计...");
    let mut edge_lens: Vec<f32> = Vec::new();
    for u in 0..n as u32 {
        let u_vec = &train[u as usize * dim..(u as usize + 1) * dim];
        for &v in graph.storage().neighbors(u) {
            let v_vec = &train[v as usize * dim..(v as usize + 1) * dim];
            edge_lens.push(l2_simd(u_vec, v_vec));
        }
    }
    edge_lens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total = edge_lens.len();
    let avg = edge_lens.iter().sum::<f32>() / total as f32;
    println!("  总边数: {} (平均度数 {:.1})", total, total as f64 / n as f64);
    println!("  平均边长: {:.6}", avg);
    println!("  p50: {:.6}", edge_lens[total / 2]);
    println!("  p90: {:.6}", edge_lens[total * 90 / 100]);
    println!("  p99: {:.6}", edge_lens[total * 99 / 100]);
    println!("  max: {:.6}", edge_lens[total - 1]);
    println!("  min: {:.6}", edge_lens[0]);

    // [3] 建图节点自身搜索：用节点自己作 query，看能否找回自己的邻居
    println!("\n[3] 节点自身搜索验证（用节点自己作 query, ef=100）...");
    let mut self_recall_sum = 0.0f64;
    let mut visited_count_sum = 0usize;
    let mut pop_count_sum = 0usize;
    let sample_count = 100usize;
    for i in 0..sample_count {
        let node_id = i as u32;
        let query = &train[node_id as usize * dim..(node_id as usize + 1) * dim];
        // 标准 break 搜索，返回 (候选, 访问数, pop数)
        let (candidates, visited_cnt, pop_cnt) = greedy_search_with_stats(
            &train, dim, graph.storage(), graph.entry_point(), query, 100,
        );
        // 图中 node_id 的邻居
        let neighbors: std::collections::HashSet<u32> =
            graph.storage().neighbors(node_id).iter().copied().collect();
        // top-10 候选里有多少是图的邻居
        let hits = candidates.iter().take(10)
            .filter(|&&c| neighbors.contains(&c)).count();
        self_recall_sum += hits as f64 / 10.0;
        visited_count_sum += visited_cnt;
        pop_count_sum += pop_cnt;
    }
    println!("  节点自身 top-10 命中图邻居比例: {:.4}", self_recall_sum / sample_count as f64);
    println!("  平均访问节点数: {:.1}", visited_count_sum as f64 / sample_count as f64);
    println!("  平均 pop 次数: {:.1}", pop_count_sum as f64 / sample_count as f64);

    // [4] 标准 break 访问节点数 vs ef_search
    println!("\n[4] 标准 break 访问节点数 vs ef_search...");
    let query = &train[0..dim];
    for &ef in &[50usize, 100, 200, 500, 1000] {
        let (_, visited_cnt, pop_cnt) = greedy_search_with_stats(
            &train, dim, graph.storage(), graph.entry_point(), query, ef,
        );
        println!("  ef={}: 访问 {} 节点, pop {} 次", ef, visited_cnt, pop_cnt);
    }
}

/// 标准 break 搜索，返回 (候选, 访问节点数, pop 次数)
fn greedy_search_with_stats(
    vectors: &[f32],
    dim: usize,
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    l: usize,
) -> (Vec<u32>, usize, usize) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use raven::memory::VisitedTracker;

    /// f32 的 Ord wrapper（BinaryHeap 需要 Ord）
    #[derive(PartialEq, PartialOrd, Clone, Copy)]
    struct OrdF32(f32);
    impl Eq for OrdF32 {}
    impl Ord for OrdF32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let n = vectors.len() / dim;
    let mut visited = VisitedTracker::new(n, l);
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    candidates.push(Reverse((OrdF32(entry_dist), entry_point)));
    visited.visit(entry_point);

    let mut pop_count = 0usize;

    while let Some(Reverse((dist, node))) = candidates.pop() {
        pop_count += 1;
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 {
                    break;
                }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }
        for &neighbor in storage.neighbors(node) {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor as usize + 1) * dim]);
                candidates.push(Reverse((OrdF32(d), neighbor)));
            }
        }
    }

    let visited_cnt = visited.visited_count();
    let result: Vec<u32> = results.into_iter().map(|(_, id)| id).collect();
    (result, visited_cnt, pop_count)
}
