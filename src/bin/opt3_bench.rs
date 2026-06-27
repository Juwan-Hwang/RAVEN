//! OPT-3 寰熀鍑嗭細BinaryHeap vs flat sorted Vec 鍫嗘€ц兘瀵规瘮
//!
//! 鐩爣锛氶獙璇?flat sorted Vec 鏄惁姣?BinaryHeap 鏇村揩
//!
//! 鏍稿績鍋囪锛欱inaryHeap 鐨勫爢鎿嶄綔娑夊強闅忔満鍐呭瓨璁块棶锛堟爲缁撴瀯锛夛紝
//! 鑰?flat sorted Vec 鏄繛缁唴瀛橈紝cache 鍙嬪ソ銆?//! 鍦?ef_search=100锛堝爢澶у皬绾?100-200锛夋椂锛宖lat vec 鐨?O(n) 鎻掑叆
//! 鍙兘姣?BinaryHeap 鐨?O(log n) 鏇村揩锛堝洜涓?cache 鍛戒腑鐜囬珮锛夈€?//!
//! 瀹為獙鏂规锛?//! - 鏂规 A锛堝綋鍓嶏級锛欱inaryHeap<Reverse<(OrderedF32, u32)>>
//! - 鏂规 B锛歠lat sorted Vec锛堥檷搴忥紝pop 浠庡熬閮?O(1)锛宲ush 鐢?binary_search + insert锛?//!
//! 鏁版嵁锛歋IFT1M base + 闅忔満鍥撅紙澶嶇敤 opt2_bench 妗嗘灦锛?
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::l2_simd;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use rand::SeedableRng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇澶辫触");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap()));
        }
    }
    (vectors, dim, n)
}

#[derive(Clone, Copy, PartialEq)]
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

struct VisitedTracker {
    visited: Vec<u8>,
    gen: u8,
}
impl VisitedTracker {
    fn new(n: usize) -> Self { Self { visited: vec![0; n], gen: 1 } }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 { self.visited.fill(0); self.gen = 1; }
    }
    fn visit(&mut self, node: u32) -> bool {
        let idx = node as usize;
        if self.visited[idx] == self.gen { false }
        else { self.visited[idx] = self.gen; true }
    }
}

/// 鏂规 A锛欱inaryHeap锛堝綋鍓嶅疄鐜帮紝鍚?OPT-2 棰勫彇绛栫暐 B锛?fn search_binary_heap(
    vectors: &[f32], dim: usize, graph: &[Vec<u32>],
    entry: u32, query: &[f32], l: usize, visited: &mut VisitedTracker,
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

        // OPT-2 棰勫彇绛栫暐 B
        if let Some(&Reverse((_, top_node))) = candidates.peek() {
            let top_neighbors = &graph[top_node as usize];
            let ptr = top_neighbors.as_ptr() as *const i8;
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

/// 鏂规 B锛歠lat sorted Vec锛堥檷搴忥紝pop 浠庡熬閮ㄥ彇鏈€灏忓€?O(1)锛?fn search_flat_sorted_vec(
    vectors: &[f32], dim: usize, graph: &[Vec<u32>],
    entry: u32, query: &[f32], l: usize, visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    // candidates: 闄嶅簭 vec锛屾渶灏忓€煎湪灏鹃儴锛宲op() O(1)
    // (dist, node)锛岄檷搴忔帓鍒?    let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(l * 2);
    // results: 鍗囧簭 vec锛屾渶澶у€煎湪灏鹃儴锛宲op() O(1)
    let mut results: Vec<(f32, u32)> = Vec::with_capacity(l + 1);

    let entry_dist = l2_simd(query, &vectors[entry as usize * dim..(entry as usize + 1) * dim]);
    // push 鍒伴檷搴?candidates锛歜inary_search 鎵句綅缃?+ insert
    let pos = candidates.partition_point(|(d, _)| d > &entry_dist);
    candidates.insert(pos, (entry_dist, entry));
    visited.visit(entry);

    while let Some((dist, node)) = candidates.pop() { // pop 浠庡熬閮ㄥ彇鏈€灏忓€?        if results.len() >= l {
            if let Some(&(worst, _)) = results.last() {
                if dist > worst { break; }
            }
        }
        // push 鍒板崌搴?results锛歜inary_search 鎵句綅缃?+ insert
        let pos = results.partition_point(|(d, _)| d < &dist);
        results.insert(pos, (dist, node));
        if results.len() > l { results.pop(); } // pop 浠庡熬閮ㄥ彇鏈€澶у€?
        // OPT-2 棰勫彇绛栫暐 B
        if let Some(&(_, top_node)) = candidates.last() {
            let top_neighbors = &graph[top_node as usize];
            let ptr = top_neighbors.as_ptr() as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        let neighbors = &graph[node as usize];
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                // push 鍒伴檷搴?candidates
                let pos = candidates.partition_point(|(cd, _)| cd > &d);
                candidates.insert(pos, (d, neighbor));
            }
        }
    }
    results.into_iter().map(|(dist, id)| (id, dist)).collect()
}

fn main() {
    println!("=== OPT-3 寰熀鍑嗭細BinaryHeap vs flat sorted Vec ===");
    println!();

    let t0 = Instant::now();
    let (vectors, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    println!("SIFT1M: {} vecs, dim={}, {:.1}s, {:.1}MB",
        n, dim, t0.elapsed().as_secs_f64(), vectors.len() * 4 / 1024 / 1024);

    let r = 32;
    let t0 = Instant::now();
    let graph = gen_random_graph(n, r, 42);
    println!("闅忔満鍥?R={}: {:.1}s", r, t0.elapsed().as_secs_f64());

    let (queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let ef_search = 100;
    println!("鏌ヨ: {} queries, ef_search={}", nq, ef_search);
    println!();

    let mut visited = VisitedTracker::new(n);
    // 棰勭儹
    search_binary_heap(&vectors, dim, &graph, 0, &queries[0..dim], ef_search, &mut visited);
    search_flat_sorted_vec(&vectors, dim, &graph, 0, &queries[0..dim], ef_search, &mut visited);
    println!("棰勭儹瀹屾垚");
    println!();

    // 鏂规 A锛欱inaryHeap
    let t0 = Instant::now();
    let mut sum_a = 0u64;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let result = search_binary_heap(&vectors, dim, &graph, 0, query, ef_search, &mut visited);
        sum_a += result.len() as u64;
    }
    let time_a = t0.elapsed().as_secs_f64();
    let qps_a = nq as f64 / time_a;
    println!("鏂规 A: BinaryHeap");
    println!("  鏃堕棿={:.4}s, QPS={:.0}, avg_latency={:.2}us, results={}",
        time_a, qps_a, time_a * 1e6 / nq as f64, sum_a);

    // 鏂规 B锛歠lat sorted Vec
    let t0 = Instant::now();
    let mut sum_b = 0u64;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let result = search_flat_sorted_vec(&vectors, dim, &graph, 0, query, ef_search, &mut visited);
        sum_b += result.len() as u64;
    }
    let time_b = t0.elapsed().as_secs_f64();
    let qps_b = nq as f64 / time_b;
    println!("鏂规 B: flat sorted Vec");
    println!("  鏃堕棿={:.4}s, QPS={:.0}, avg_latency={:.2}us, results={}",
        time_b, qps_b, time_b * 1e6 / nq as f64, sum_b);
    println!();

    // 姹囨€?    println!("=== 姹囨€?===");
    println!("{:<25} {:>10} {:>12} {:>10}", "鏂规", "QPS", "avg_latency", "鍔犻€熸瘮");
    println!("{:-<60}", "");
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "A: BinaryHeap", qps_a, time_a * 1e6 / nq as f64, 1.0);
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "B: flat sorted Vec", qps_b, time_b * 1e6 / nq as f64, qps_b / qps_a);
    println!();

    let speedup = qps_b / qps_a;
    if speedup >= 1.05 {
        println!("缁撹: flat sorted Vec 鍔犻€?{:.2}x (鈮?.05x)锛岃揪鍒?OPT-3 楠屾敹鏍囧噯", speedup);
    } else if speedup >= 1.02 {
        println!("缁撹: flat sorted Vec 鍔犻€?{:.2}x (1.02-1.05x)锛屾敹鐩婃湁闄?, speedup);
    } else {
        println!("缁撹: flat sorted Vec 鍔犻€?{:.2}x (<1.02x)锛屾棤鏀剁泭锛孫PT-3 鍚﹀喅", speedup);
    }
}
