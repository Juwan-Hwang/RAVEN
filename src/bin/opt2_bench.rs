//! OPT-2 寰熀鍑嗭細棰勫彇绛栫暐浼樺寲瀵规瘮
//!
//! 鐩爣锛氶獙璇佷笉鍚岄鍙栫瓥鐣ュ鍥炬悳绱?QPS 鐨勫奖鍝?//!
//! 鏍稿績闂锛氬綋鍓嶉鍙?neighbors[i+1] 鐨勫悜閲忔暟鎹紝浣嗕笅涓€姝ョ湡姝ｉ渶瑕佺殑鏄?//! candidates 鍫嗛《鑺傜偣鐨勯偦灞呫€傚摢绉嶉鍙栫瓥鐣ユ渶浼橈紵
//!
//! 瀹為獙鏂规锛?//! - 鏂规 A锛堝綋鍓嶏級锛氶鍙?neighbors[i+1] 鐨勫悜閲忔暟鎹?//! - 鏂规 B锛堟柊锛夛細棰勫彇 candidates 鍫嗛《鑺傜偣鐨?neighbors 鎸囬拡
//! - 鏂规 C锛堢粍鍚堬級锛欰 + B
//! - 鏂规 D锛堟棤棰勫彇锛夛細鍒犻櫎鎵€鏈?_mm_prefetch
//! - 鏂规 E锛?-ahead锛夛細棰勫彇 neighbors[i+1] 鍜?neighbors[i+2] 鐨勫悜閲忔暟鎹?//!
//! 鏁版嵁锛歋IFT1M base锛?M 脳 dim=128 = 493MB锛岃秴鍑?L3 cache锛?//! 鍥撅細闅忔満鍥撅紙R=32锛屼笉寤?Vamana 鍥撅紝鐪?782s锛?//! 娉ㄦ剰锛氶殢鏈哄浘 recall 寰堜綆锛屼絾棰勫彇绛栫暐涓嶅奖鍝?recall锛屽彧娴?QPS

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::l2_simd;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use rand::SeedableRng;

/// 璇诲彇 fvecs 鏂囦欢
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 鏂囦欢闀垮害涓嶅榻?);

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

/// OrderedF32 鍖呰锛堢敤浜?BinaryHeap锛?#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// 鐢熸垚闅忔満閭诲眳鍒楄〃锛堟瘡涓妭鐐?R 涓殢鏈洪偦灞咃級
fn gen_random_graph(n: usize, r: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    use rand::seq::SliceRandom;
    let mut graph: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut indices: Vec<u32> = (0..n as u32).collect();
    for node in 0..n as u32 {
        indices.partial_shuffle(&mut rng, r + 1);
        let neighbors: Vec<u32> = indices.iter()
            .take(r + 1)
            .filter(|&&x| x != node)
            .take(r)
            .copied()
            .collect();
        graph.push(neighbors);
    }
    graph
}

/// 绠€鍗曠殑 visited tracker
struct VisitedTracker {
    visited: Vec<u8>,
    gen: u8,
}

impl VisitedTracker {
    fn new(n: usize) -> Self {
        Self { visited: vec![0; n], gen: 1 }
    }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.visited.fill(0);
            self.gen = 1;
        }
    }
    fn visit(&mut self, node: u32) -> bool {
        let idx = node as usize;
        if self.visited[idx] == self.gen {
            false
        } else {
            self.visited[idx] = self.gen;
            true
        }
    }
}

/// 鏂规 A锛氶鍙?neighbors[i+1] 鐨勫悜閲忔暟鎹紙褰撳墠瀹炵幇锛?fn search_prefetch_next_vec(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        let neighbors = &graph[node as usize];
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // 鏂规 A锛氶鍙栦笅涓€涓偦灞呯殑鍚戦噺
            if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr); }
            }
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 鏂规 B锛氶鍙?candidates 鍫嗛《鑺傜偣鐨?neighbors 鎸囬拡
fn search_prefetch_heap_neighbors(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        // 鏂规 B锛氶鍙栧爢椤惰妭鐐圭殑閭诲眳鍒楄〃
        if let Some(&Reverse((_, top_node))) = candidates.peek() {
            let top_neighbors = &graph[top_node as usize];
            // 棰勫彇鍫嗛《鑺傜偣鐨勯偦灞呭垪琛ㄦ暟鎹紙Vec<u32> 鐨勫爢鍐呭瓨锛?            let ptr = top_neighbors.as_ptr() as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        let neighbors = &graph[node as usize];
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 鏂规 C锛氱粍鍚?A + B
fn search_prefetch_combo(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        // 鏂规 C 鐨?B 閮ㄥ垎锛氶鍙栧爢椤惰妭鐐圭殑閭诲眳鍒楄〃
        if let Some(&Reverse((_, top_node))) = candidates.peek() {
            let top_neighbors = &graph[top_node as usize];
            let ptr = top_neighbors.as_ptr() as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        let neighbors = &graph[node as usize];
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // 鏂规 C 鐨?A 閮ㄥ垎锛氶鍙栦笅涓€涓偦灞呯殑鍚戦噺
            if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr); }
            }
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 鏂规 D锛氭棤棰勫彇
fn search_no_prefetch(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        let neighbors = &graph[node as usize];
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

/// 鏂规 E锛?-ahead 棰勫彇锛坣eighbors[i+1] 鍜?neighbors[i+2]锛?fn search_prefetch_2ahead(
    vectors: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    entry: u32,
    query: &[f32],
    l: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l * 2);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    candidates.push(Reverse((OrderedF32(entry_dist), entry)));
    visited.visit(entry);

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if results.len() >= l {
            if let Some(&(worst, _)) = results.peek() {
                if dist.0 > worst.0 { break; }
            }
        }
        results.push((dist, node));
        if results.len() > l { results.pop(); }

        let neighbors = &graph[node as usize];
        for (i, &neighbor) in neighbors.iter().enumerate() {
            // 鏂规 E锛氶鍙?2-ahead
            if i + 2 < neighbors.len() {
                let next1 = neighbors[i + 1];
                let next2 = neighbors[i + 2];
                let ptr1 = vectors.as_ptr().wrapping_add(next1 as usize * dim) as *const i8;
                let ptr2 = vectors.as_ptr().wrapping_add(next2 as usize * dim) as *const i8;
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<3>(ptr1);
                    std::arch::x86_64::_mm_prefetch::<3>(ptr2);
                }
            } else if i + 1 < neighbors.len() {
                let next = neighbors[i + 1];
                let ptr = vectors.as_ptr().wrapping_add(next as usize * dim) as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr); }
            }
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                candidates.push(Reverse((OrderedF32(d), neighbor)));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist.0)).collect()
}

fn main() {
    println!("=== OPT-2 寰熀鍑嗭細棰勫彇绛栫暐浼樺寲瀵规瘮锛圫IFT1M + 闅忔満鍥撅級===");
    println!();

    // 鍔犺浇 SIFT1M base
    println!("鍔犺浇 SIFT1M base 鏁版嵁...");
    let t0 = Instant::now();
    let (vectors, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    println!("鍔犺浇瀹屾垚: {} vecs, dim={}, {:.1}s, {:.1}MB",
        n, dim, t0.elapsed().as_secs_f64(), vectors.len() * 4 / 1024 / 1024);

    // 鐢熸垚闅忔満鍥撅紙R=32锛屾ā鎷?Vamana 鍥剧殑搴︽暟锛?    let r = 32;
    println!("鐢熸垚闅忔満鍥?(R={})...", r);
    let t0 = Instant::now();
    let graph = gen_random_graph(n, r, 42);
    println!("鍥剧敓鎴? {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 鍔犺浇鏌ヨ闆?    let (queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    println!("鏌ヨ闆? {} queries", nq);
    let ef_search = 100;
    let k = 10;
    println!("鍙傛暟: ef_search={}, k={}, R={}", ef_search, k, r);
    println!();

    // 棰勭儹
    let mut visited = VisitedTracker::new(n);
    let warmup_query = &queries[0..dim];
    search_prefetch_next_vec(&vectors, dim, &graph, 0, warmup_query, ef_search, &mut visited);
    println!("棰勭儹瀹屾垚");
    println!();

    // 杩愯 5 绉嶇瓥鐣?    let strategies: &[(&str, fn(&[f32], usize, &[Vec<u32>], u32, &[f32], usize, &mut VisitedTracker) -> Vec<(u32, f32)>)] = &[
        ("A: 棰勫彇 next_vec (褰撳墠)", search_prefetch_next_vec),
        ("B: 棰勫彇 heap_neighbors", search_prefetch_heap_neighbors),
        ("C: 缁勫悎 A+B", search_prefetch_combo),
        ("D: 鏃犻鍙?, search_no_prefetch),
        ("E: 2-ahead 棰勫彇", search_prefetch_2ahead),
    ];

    let mut results_summary: Vec<(String, f64, usize)> = Vec::new();

    for (name, search_fn) in strategies {
        let t0 = Instant::now();
        let mut total_results = 0usize;
        for q in 0..nq {
            let query = &queries[q * dim..(q + 1) * dim];
            let result = search_fn(&vectors, dim, &graph, 0, query, ef_search, &mut visited);
            total_results += result.len();
        }
        let time = t0.elapsed().as_secs_f64();
        let qps = nq as f64 / time;
        println!("{}: 鏃堕棿={:.4}s, QPS={:.0}, avg_latency={:.2}us, results={}",
            name, time, qps, time * 1e6 / nq as f64, total_results);
        results_summary.push((name.to_string(), qps, total_results));
    }

    // 姹囨€?    println!();
    println!("=== 姹囨€?===");
    println!("{:<30} {:>10} {:>12}", "绛栫暐", "QPS", "鍔犻€熸瘮");
    println!("{:-<55}", "");
    let baseline_qps = results_summary[0].1;
    for (name, qps, _) in &results_summary {
        let speedup = qps / baseline_qps;
        println!("{:<30} {:>10.0} {:>10.2}x", name, qps, speedup);
    }
    println!();

    // 缁撹
    let best = results_summary.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let speedup_vs_baseline = best.1 / baseline_qps;
    if speedup_vs_baseline >= 1.03 {
        println!("缁撹: 鏈€浼樼瓥鐣?'{}' 鍔犻€?{:.2}x (鈮?.03x)锛岃揪鍒?OPT-2 楠屾敹鏍囧噯", best.0, speedup_vs_baseline);
    } else {
        println!("缁撹: 鏈€浼樼瓥鐣?'{}' 鍔犻€?{:.2}x (<1.03x)锛屾湭杈鹃獙鏀舵爣鍑?, best.0, speedup_vs_baseline);
    }
}
