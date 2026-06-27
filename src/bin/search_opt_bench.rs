//! 鎼滅储浼樺寲瀵规瘮鍩哄噯
//!
//! 寤哄浘涓€娆★紝鍒嗗埆璺戝熀绾垮拰浼樺寲鐗堟悳绱紝瀵规瘮 QPS/recall銆?
//!
//! 浼樺寲鐐癸細
//! 1. 澶嶇敤璺濈锛歡reedy_search 鍐呴儴宸茬畻璺濈锛宻earch() 涓嶉噸绠?
//! 2. Software prefetch锛氬唴灞傚惊鐜鍙栦笅涓€涓?neighbor 鐨勫悜閲忔暟鎹?
//!
//! 鐢ㄦ硶锛歝argo run --release --bin search_opt_bench

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::memory::{VisitedTracker, HybridBlockedCsr};
use raven::l2_simd;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// f32 鍖呰涓?Ord锛圔inaryHeap 闇€瑕侊級
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

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

fn recall_at_k(found: &[u32], gt_slice: &[i32], k: usize) -> f64 {
    let mut hits = 0usize;
    for &g in gt_slice.iter().take(k) {
        if found.contains(&(g as u32)) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

/// 浼樺寲鐗?greedy_search锛氳繑鍥?(id, dist) 瀵癸紝閬垮厤 search() 閲嶇畻璺濈
///
/// 浼樺寲鐐癸細
/// 1. 杩斿洖 (id, dist) 鑰岄潪浠?id锛岃皟鐢ㄦ柟鏃犻渶閲嶇畻璺濈
/// 2. 鍐呭眰寰幆 software prefetch 涓嬩竴涓?neighbor 鐨勫悜閲?
fn greedy_search_optimized(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();

    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(
        query,
        &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim],
    );
    candidates.push(Reverse((OrderedF32(entry_dist), entry_point)));
    visited.visit(entry_point);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 {
                    break;
                }
            }
        }

        results.push((dist, node));
        if results.len() > l {
            results.pop();
        }

        let neighbors = storage.neighbors(node);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // Software prefetch: 棰勫彇涓嬩竴涓?neighbor 鐨勫悜閲忔暟鎹?
            // 鍗充娇涓嬩竴涓?neighbor 宸?visited锛宲refetch 鍙槸 hint锛屾棤鍓綔鐢?
            if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe {
                    // _MM_HINT_T0 = 3: 棰勫彇鍒版墍鏈?cache 灞傜骇
                    std::arch::x86_64::_mm_prefetch::<3>(ptr);
                }
            }

            if visited.visit(neighbor) {
                let d = l2_simd(
                    query,
                    &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim],
                );
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }

    // 杩斿洖 (id, dist) 瀵癸紝閬垮厤璋冪敤鏂归噸绠楄窛绂?
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

fn main() {
    println!("=== 鎼滅储浼樺寲瀵规瘮鍩哄噯 (SIFT1M) ===");
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.2}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }

    // 2. 寤哄浘锛堝彧寤轰竴娆★級
    println!("=== f32 寤哄浘锛圴amana 伪=1.2, r_max=32, l_build=100, max_iter=2锛?==");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("寤哄浘鏃堕棿: {:.2}s ({:.0} vec/s)", build_time, n as f64 / build_time);
    println!();

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;

    // 3. 鍩虹嚎鎼滅储锛堝綋鍓?GraphSearcher::search锛?
    println!("=== 鍩虹嚎鎼滅储锛堝綋鍓?GraphSearcher::search锛?==");
    let mut searcher = GraphSearcher::new(&train, &graph, ef_search);
    // 棰勭儹
    for q in 0..100.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search(query, k);
    }
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let baseline_time = t0.elapsed().as_secs_f64();
    let baseline_recall = recall_sum / nq as f64;
    let baseline_qps = nq as f64 / baseline_time;
    println!("鍩虹嚎 recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        baseline_recall, baseline_qps, baseline_time * 1000.0 / nq as f64);
    println!();

    // 4. 浼樺寲鐗堟悳绱紙璺濈澶嶇敤 + prefetch锛?
    println!("=== 浼樺寲鐗堟悳绱紙璺濈澶嶇敤 + software prefetch锛?==");
    let n_nodes = train.len() / dim;
    let mut visited = VisitedTracker::new(n_nodes, ef_search);
    let entry_point = graph.entry_point();
    let storage = graph.storage();

    // 棰勭儹
    for q in 0..100.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = greedy_search_optimized(
            &train, dim, storage, entry_point, query, ef_search, &mut visited,
        );
        let mut reranked = candidates;
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let _ = reranked.truncate(k);
    }

    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = greedy_search_optimized(
            &train, dim, storage, entry_point, query, ef_search, &mut visited,
        );
        // 璺濈宸插寘鍚湪缁撴灉涓紝鍙渶鎺掑簭鍙?top-k
        let mut reranked = candidates;
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let found: Vec<u32> = reranked.iter().take(k).map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let opt_time = t0.elapsed().as_secs_f64();
    let opt_recall = recall_sum / nq as f64;
    let opt_qps = nq as f64 / opt_time;
    println!("浼樺寲 recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        opt_recall, opt_qps, opt_time * 1000.0 / nq as f64);
    println!();

    // 5. 瀵规瘮姹囨€?
    println!("=== 瀵规瘮姹囨€?===");
    println!("鍩虹嚎:  recall={:.4}, QPS={:.0}, latency={:.3}ms", baseline_recall, baseline_qps, baseline_time * 1000.0 / nq as f64);
    println!("浼樺寲:  recall={:.4}, QPS={:.0}, latency={:.3}ms", opt_recall, opt_qps, opt_time * 1000.0 / nq as f64);
    let qps_delta = (opt_qps - baseline_qps) / baseline_qps * 100.0;
    let recall_delta = (opt_recall - baseline_recall) * 100.0;
    println!("QPS 鍙樺寲: {:+.1}%", qps_delta);
    println!("recall 鍙樺寲: {:+.4}pp", recall_delta);
    if qps_delta > 5.0 && recall_delta.abs() < 0.001 {
        println!("缁撹: 浼樺寲鏈夋晥锛圦PS 鎻愬崌 >5%, recall 涓嶅彉锛?);
    } else if qps_delta > 0.0 && recall_delta.abs() < 0.001 {
        println!("缁撹: 浼樺寲鏈夎交寰晥鏋滐紙QPS 鎻愬崌 <5%, recall 涓嶅彉锛?);
    } else if recall_delta.abs() >= 0.001 {
        println!("缁撹: 浼樺寲褰卞搷 recall锛岄渶妫€鏌ユ纭€?);
    } else {
        println!("缁撹: 浼樺寲鏃犳晥鎴栧弽浣滅敤");
    }
}
