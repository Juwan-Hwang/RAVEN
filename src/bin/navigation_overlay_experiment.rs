//! NavigationLayer centroid overlay 瀹為獙
//!
//! 楠岃瘉 NavigationLayer 鏄惁瀵规悳绱㈡湁鐢?
//! 瀵规瘮锛?
//!   A. 榛樿 medoid entry_point锛堝綋鍓嶇敓浜ц矾寰勶級
//!   B. 鏈€杩?centroid entry_point锛圢avigationLayer 鎻愪緵锛?
//!
//! 鎸囨爣锛歳ecall@10, QPS, avg_visited
//! 鑻?B 鐨?recall/QPS 涓嶅姡浜?A锛屽垯 NavigationLayer 鏈夌敤锛屽彲闆嗘垚

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, NavigationLayer, NavigationConfig};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::memory::VisitedTracker;

/// f32 鍖呰锛圔inaryHeap 瑕佹眰 Ord锛?
#[derive(Debug, Clone, Copy)]
struct OrdF32(f32);
impl PartialEq for OrdF32 { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { self.0.partial_cmp(&o.0) } }
impl Ord for OrdF32 { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal) } }

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");
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
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 ivecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 ivecs 澶辫触");
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

/// 浠庢寚瀹?entry_point 鎼滅储锛岃繑鍥?(top-L, visited_count)
fn search_from_entry(
    vectors: &[f32],
    dim: usize,
    graph: &VamanaGraph,
    entry: u32,
    query: &[f32],
    ef_search: usize,
) -> (Vec<u32>, usize) {
    // 鎵嬪姩瀹炵幇 greedy search锛岀粺璁?visited 鏁伴噺
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
    println!("=== NavigationLayer centroid overlay 瀹為獙 ===");
    println!();

    // 1. 鍔犺浇 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);

    // 褰掍竴鍖?
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    // 2. 鏋勫缓 VamanaGraph锛堥粯璁?medoid entry锛?
    println!();
    println!("=== 鏋勫缓 VamanaGraph锛埼?1.0, r_max=32, l_build=100锛?==");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("寤哄浘: {:.1}s", t0.elapsed().as_secs_f64());
    println!("entry_point (medoid): {}", graph.entry_point());

    // 3. 鏋勫缓 NavigationLayer锛坈entroid overlay锛屸垰N 涓?centroid锛?
    println!();
    println!("=== 鏋勫缓 NavigationLayer锛坈entroid overlay, 鈭歂={}锛?==", (n as f64).sqrt() as usize);
    let t0 = Instant::now();
    let nav_config = NavigationConfig {
        enable_centroid_overlay: true,
        centroid_count: None, // 鈭歂
    };
    let nav = NavigationLayer::new(n, &train, dim, nav_config);
    println!("NavigationLayer 鏋勫缓: {:.1}s", t0.elapsed().as_secs_f64());
    println!("centroid 鏁伴噺: {}", nav.centroids().len());

    // 4. 瀵规瘮鎼滅储
    let k = 10;
    let ef_search = 100;
    let gt_stride = gt_k;

    // A. 榛樿 medoid entry
    println!();
    println!("=== A. 榛樿 medoid entry_point ===");
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

    // B. 鏈€杩?centroid entry
    println!();
    println!("=== B. 鏈€杩?centroid entry_point锛圢avigationLayer锛?==");
    let t0 = Instant::now();
    let mut hits_b = 0usize;
    let mut visited_sum_b = 0usize;
    let mut entry_match_count = 0usize; // centroid 鎭板ソ鏄?medoid 鐨勬鏁?
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        // 鎵炬渶杩戠殑 centroid
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
    println!("(centroid 鎭板ソ鏄?medoid 鐨勬鏁? {}/{})", entry_match_count, nq);

    // 5. 姹囨€?
    println!();
    println!("=== 姹囨€?===");
    println!("{:<25} {:>10} {:>10} {:>12} {:>12}", "鏂规", "recall@10", "QPS", "avg_visited", "latency_ms");
    println!("{:-<69}", "");
    println!("{:<25} {:>10.4} {:>10.0} {:>12.1} {:>12.3}", "A. medoid entry", recall_a, qps_a, avg_visited_a, time_a * 1000.0 / nq as f64);
    println!("{:<25} {:>10.4} {:>10.0} {:>12.1} {:>12.3}", "B. centroid entry", recall_b, qps_b, avg_visited_b, time_b * 1000.0 / nq as f64);
    println!();

    // 鍒ゅ畾
    let recall_diff = recall_b - recall_a;
    let qps_diff_pct = (qps_b - qps_a) / qps_a * 100.0;
    let visited_diff_pct = (avg_visited_b - avg_visited_a) / avg_visited_a * 100.0;
    println!("宸紓: recall {:+.4}, QPS {:+.1}%, visited {:+.1}%", recall_diff, qps_diff_pct, visited_diff_pct);
    println!();

    if recall_b >= recall_a - 0.001 && qps_b >= qps_a * 0.95 {
        println!("缁撹: NavigationLayer centroid overlay 涓嶅姡浜?medoid锛屽彲闆嗘垚");
    } else if recall_b > recall_a + 0.001 || (recall_b >= recall_a - 0.001 && qps_b > qps_a * 1.05) {
        println!("缁撹: NavigationLayer centroid overlay 鏈夋鍚戞敹鐩婏紝寤鸿闆嗘垚");
    } else {
        println!("缁撹: NavigationLayer centroid overlay 鏃犳槑鏄炬敹鐩婏紝寤鸿鍒犻櫎");
    }
}
