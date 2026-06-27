//! v8.0 娑堣瀺瀹為獙锛氶€愰」鐙珛楠岃瘉姣忛」浼樺寲
//!
//! 瀹為獙璁捐锛堢瀛︽柟娉曡锛夛細
//! 1. 寤哄浘涓€娆★紙saturate=true锛夛紝淇濆瓨鍒板唴瀛?
//! 2. 鍦ㄥ悓涓€寮犲浘涓婅窇澶氫釜鎼滅储鍙樹綋锛岄殧绂绘悳绱㈠眰浼樺寲鏁堟灉
//! 3. 鍙﹀缓 saturate=false 鐨勫浘锛屽姣斿浘缁撴瀯宸紓
//!
//! 鎼滅储鍙樹綋锛堝悓涓€寮犲浘锛屼粎鎼滅储浠ｇ爜涓嶅悓锛夛細
//!   A: baseline     鈥?褰撳墠鎼滅储锛坆itset 宸茬敓鏁堬級
//!   B: two_pass     鈥?Glass SearchImpl2 妯″紡锛堟壒閲忔敹闆?+ 棰勫彇鍚戦噺锛?
//!   C: multi_pref   鈥?澶氳鍥鹃鍙栵紙4 cache lines vs 1锛?
//!   D: all_combined 鈥?B+C 鍚堝苟
//!
//! 鍥剧粨鏋勫彉浣擄紙涓嶅悓鍥撅紝鐩稿悓鎼滅储浠ｇ爜锛夛細
//!   E: saturate_off 鈥?saturate=false 寤哄浘锛岀敤 baseline 鎼滅储
//!
//! 鐢ㄦ硶锛?
//!   cargo run --release --bin opt_ablation

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;
use raven::memory::{HybridBlockedCsr, VisitedTracker};
use raven::graph::linear_pool::LinearPool;

// 鈹€鈹€ 鏁版嵁璇诲彇 鈹€鈹€

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
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
    let mut file = File::open(path).expect("open ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read ivecs");
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

// 鈹€鈹€ 鎼滅储鍙樹綋 A: baseline锛堝綋鍓嶄唬鐮侊紝bitset 宸茬敓鏁堬級 鈹€鈹€

fn search_baseline(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    while let Some((node, _dist)) = pool.pop() {
        let neighbors = storage.neighbors(node);
        if let Some((next_node, _)) = pool.peek_unchecked() {
            storage.prefetch_neighbors(next_node);
        }
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                pool.insert(neighbor, d);
            }
        }
    }

    pool.to_sorted_vec()
}

// 鈹€鈹€ 鎼滅储鍙樹綋 B: two_pass锛圙lass SearchImpl2 妯″紡锛?鈹€鈹€
//
// Glass 鐨勬牳蹇冩妧宸э細
// 1. Pop 鑺傜偣鍚庯紝绗竴閬嶆壂鎻忛偦灞呭垪琛紝鏀堕泦鏈闂殑鍒?edge_buf
// 2. 棰勫彇 edge_buf 鍓?po 涓偦灞呯殑鍚戦噺鏁版嵁
// 3. 绗簩閬嶆壂鎻?edge_buf锛屾瘡娆￠鍙?i+po 鍓嶇灮鐨勫悜閲?
// 4. 璁＄畻璺濈鏃跺悜閲忓凡鍦?L1/L2 cache
//
// 涓?baseline 鐨勫尯鍒細baseline 閫愪釜閭诲眳璁＄畻璺濈锛?
// 姣忔璺濈璁＄畻鍓嶅悜閲忔暟鎹彲鑳藉湪 DRAM锛宑ache miss 寤惰繜 ~100ns銆?
// two_pass 鎵归噺棰勫彇锛岃窛绂昏绠楁椂鏁版嵁宸插湪 cache锛屽欢杩?~4ns銆?

fn search_two_pass(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
    po: usize,  // prefetch offset锛堝墠鐬昏窛绂伙級
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    // 棰勫垎閰?edge_buf锛堟爤涓婏紝閬垮厤鍫嗗垎閰嶏級
    // R_max=64 鈫?64 * 4 = 256 bytes锛宖it 鏍?
    let mut edge_buf: [u32; 128] = [0; 128];

    while let Some((node, _dist)) = pool.pop() {
        let neighbors = storage.neighbors(node);

        // 鍥鹃鍙栵細棰勫彇涓嬩竴杞 pop 鐨勮妭鐐圭殑閭诲眳鍒楄〃
        if let Some((next_node, _)) = pool.peek_unchecked() {
            storage.prefetch_neighbors(next_node);
        }

        // 绗竴閬嶏細鏀堕泦鏈闂偦灞呭埌 edge_buf
        let mut edge_size = 0usize;
        for &v in neighbors {
            if edge_size >= 128 {
                break;
            }
            if visited.visit(v) {
                edge_buf[edge_size] = v;
                edge_size += 1;
            }
        }

        // 棰勫彇鍓?po 涓偦灞呯殑鍚戦噺鏁版嵁
        let prefetch_count = po.min(edge_size);
        for i in 0..prefetch_count {
            let v = edge_buf[i] as usize;
            let ptr = &vectors[v * dim] as *const f32 as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        // 绗簩閬嶏細璁＄畻璺濈锛屽悓鏃跺墠鐬婚鍙?
        for i in 0..edge_size {
            // 鍓嶇灮棰勫彇锛歩 + po 澶勭殑鍚戦噺
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = &vectors[v * dim] as *const f32 as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
            pool.insert(neighbor, d);
        }
    }

    pool.to_sorted_vec()
}

// 鈹€鈹€ 鎼滅储鍙樹綋 C: multi_pref锛堝琛屽浘棰勫彇锛?鈹€鈹€
//
// 褰撳墠 prefetch_neighbors 鍙鍙?1 涓?cache line锛?4 bytes锛夈€?
// R_max=64 鈫?閭诲眳鍒楄〃 64*4=256 bytes = 4 cache lines銆?
// 澶氳棰勫彇纭繚鏁翠釜閭诲眳鍒楄〃鍦?L1 cache銆?

fn search_multi_pref(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    while let Some((node, _dist)) = pool.pop() {
        // 澶氳鍥鹃鍙栵細棰勫彇瀹屾暣閭诲眳鍒楄〃锛? cache lines for R=64锛?
        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            // R_max=64 鈫?256 bytes 鈫?4 cache lines
            // 棰勫彇 4 琛岃鐩栨暣涓偦灞呭垪琛?
            unsafe {
                std::arch::x86_64::_mm_prefetch::<0>(ptr);
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
            }
        }

        let neighbors = storage.neighbors(node);
        for &neighbor in neighbors {
            if visited.visit(neighbor) {
                let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
                pool.insert(neighbor, d);
            }
        }
    }

    pool.to_sorted_vec()
}

// 鈹€鈹€ 鎼滅储鍙樹綋 D: all_combined锛坱wo_pass + multi_pref锛?鈹€鈹€

fn search_combined(
    vectors: &[f32],
    dim: usize,
    storage: &HybridBlockedCsr,
    entry_point: u32,
    query: &[f32],
    ef: usize,
    visited: &mut VisitedTracker,
    po: usize,
) -> Vec<(u32, f32)> {
    visited.reset();
    let mut pool = LinearPool::new(ef);

    let entry_dist = l2_simd(query, &vectors[entry_point as usize * dim..(entry_point as usize + 1) * dim]);
    visited.visit(entry_point);
    pool.insert(entry_point, entry_dist);

    let mut edge_buf: [u32; 128] = [0; 128];

    while let Some((node, _dist)) = pool.pop() {
        // 澶氳鍥鹃鍙?
        if let Some((next_node, _)) = pool.peek_unchecked() {
            let start = next_node as usize * storage.r_max();
            let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
            unsafe {
                std::arch::x86_64::_mm_prefetch::<0>(ptr);
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(64));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(128));
                std::arch::x86_64::_mm_prefetch::<0>(ptr.add(192));
            }
        }

        let neighbors = storage.neighbors(node);
        let mut edge_size = 0usize;
        for &v in neighbors {
            if edge_size >= 128 {
                break;
            }
            if visited.visit(v) {
                edge_buf[edge_size] = v;
                edge_size += 1;
            }
        }

        let prefetch_count = po.min(edge_size);
        for i in 0..prefetch_count {
            let v = edge_buf[i] as usize;
            let ptr = &vectors[v * dim] as *const f32 as *const i8;
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
        }

        for i in 0..edge_size {
            if i + po < edge_size {
                let v = edge_buf[i + po] as usize;
                let ptr = &vectors[v * dim] as *const f32 as *const i8;
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr); }
            }
            let neighbor = edge_buf[i];
            let d = l2_simd(query, &vectors[neighbor as usize * dim..(neighbor + 1) as usize * dim]);
            pool.insert(neighbor, d);
        }
    }

    pool.to_sorted_vec()
}

// 鈹€鈹€ 鍩哄噯娴嬭瘯妗嗘灦 鈹€鈹€

struct BenchResult {
    name: String,
    ef: usize,
    recall: f64,
    qps: f64,
    avg_visited: f64,
}

fn bench_search_variant(
    name: &str,
    vectors: &[f32],
    dim: usize,
    graph: &VamanaGraph,
    test: &[f32],
    nq: usize,
    gt: &[i32],
    gt_k: usize,
    k: usize,
    ef: usize,
    search_fn: impl Fn(&[f32], usize, &HybridBlockedCsr, u32, &[f32], usize, &mut VisitedTracker) -> Vec<(u32, f32)>,
) -> BenchResult {
    let storage = graph.storage();
    let entry = graph.entry_point();
    let mut visited = VisitedTracker::new(vectors.len() / dim, ef);

    // warmup
    for q in 0..nq.min(100) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = search_fn(vectors, dim, storage, entry, query, ef, &mut visited);
    }

    let mut hits = 0usize;
    let mut total = 0usize;
    let mut total_visited = 0usize;

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = search_fn(vectors, dim, storage, entry, query, ef, &mut visited);
        total_visited += visited.visited_count();

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
        total += k;
    }
    let elapsed = t0.elapsed();

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / elapsed.as_secs_f64();
    let avg_visited = total_visited as f64 / nq as f64;

    println!(
        "  {:>16} ef={:>4}  recall={:.4}  QPS={:>8.0}  avg_visited={:.0}",
        name, ef, recall, qps, avg_visited
    );

    BenchResult {
        name: name.to_string(),
        ef,
        recall,
        qps,
        avg_visited,
    }
}

/// 鐗堟湰妯箙
fn print_banner() {
    let pkg_ver = env!("CARGO_PKG_VERSION");
    let git_hash = env!("RAVEN_GIT_HASH");
    let build_ts = env!("RAVEN_BUILD_TS");
    println!("鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽");
    println!("鈺? RAVEN v{}  git:{}  build:{}  鈺?, pkg_ver, git_hash, build_ts);
    println!("鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆");
}

fn main() {
    print_banner();
    println!("=== v8.0 娑堣瀺瀹為獙锛氶€愰」鐙珛楠岃瘉 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    println!("鏁版嵁: n={}, dim={}, nq={}\n", n, dim, nq);

    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    //  Part 1: 鎼滅储灞備紭鍖栵紙鍚屼竴寮犲浘锛屼笉鍚屾悳绱唬鐮侊級
    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    println!("鈺愨晲 Part 1: 鎼滅储灞備紭鍖栵紙鍚屼竴寮?saturate=true 鍥撅級鈺愨晲\n");

    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
        saturate: true,
enable_layered_nav: false,
nav_m: 16,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("寤哄浘瀹屾垚: {:.1}s\n", build_time);

    // 搴︽暟缁熻
    let stats = graph.degree_stats();
    println!(
        "[degree] mean={:.1} p95={} p99={} max={} isolated={}",
        stats.mean_degree, stats.p95_degree, stats.p99_degree,
        stats.max_degree, stats.isolated_nodes
    );

    // VisitedTracker 鍐呭瓨瀵规瘮
    let vt = VisitedTracker::new(n, 200);
    println!("[visited] bitset 鍐呭瓨 = {} bytes ({:.1} KB)", vt.bits_bytes(), vt.bits_bytes() as f64 / 1024.0);
    println!("[visited] 瀵规瘮 Vec<u8> = {} bytes ({:.1} KB)\n", n, n as f64 / 1024.0);

    let ef_list: &[usize] = &[50, 100, 200];

    for &ef in ef_list {
        println!("--- ef_search={} ---", ef);

        // A: baseline
        let _ra = bench_search_variant(
            "baseline", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_baseline(v, d, s, e, q, ef, vis),
        );

        // B: two_pass (po=4)
        let _rb = bench_search_variant(
            "two_pass(po=4)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_two_pass(v, d, s, e, q, ef, vis, 4),
        );

        // B2: two_pass (po=8)
        let _rb2 = bench_search_variant(
            "two_pass(po=8)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_two_pass(v, d, s, e, q, ef, vis, 8),
        );

        // C: multi_pref
        let _rc = bench_search_variant(
            "multi_pref", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_multi_pref(v, d, s, e, q, ef, vis),
        );

        // D: combined (two_pass po=4 + multi_pref)
        let _rd = bench_search_variant(
            "combined(po=4)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_combined(v, d, s, e, q, ef, vis, 4),
        );

        // D2: combined (two_pass po=8 + multi_pref)
        let _rd2 = bench_search_variant(
            "combined(po=8)", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_combined(v, d, s, e, q, ef, vis, 8),
        );

        println!();
    }

    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    //  Part 2: 鍥剧粨鏋勪紭鍖栵紙saturate=true vs false锛?
    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    println!("鈺愨晲 Part 2: 鍥剧粨鏋勪紭鍖栵紙saturate=true vs false锛夆晲鈺怽n");

    let t0 = Instant::now();
    let mut rng2 = ChaCha8Rng::seed_from(42);
    let config_off = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
        saturate: false,
enable_layered_nav: false,
nav_m: 16,
    };
    let graph_off = VamanaGraph::build(&train, dim, &config_off, &mut rng2);
    let build_time_off = t0.elapsed().as_secs_f64();
    println!("寤哄浘(saturate=false): {:.1}s", build_time_off);

    let stats_off = graph_off.degree_stats();
    println!(
        "[degree] mean={:.1} p95={} p99={} max={} isolated={}",
        stats_off.mean_degree, stats_off.p95_degree, stats_off.p99_degree,
        stats_off.max_degree, stats_off.isolated_nodes
    );
    println!();

    for &ef in ef_list {
        println!("--- ef_search={} ---", ef);

        // E: saturate=true (baseline search on saturate=true graph)
        let _re = bench_search_variant(
            "sat=true", &train, dim, &graph, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_baseline(v, d, s, e, q, ef, vis),
        );

        // F: saturate=false (baseline search on saturate=false graph)
        let _rf = bench_search_variant(
            "sat=false", &train, dim, &graph_off, &test, nq, &gt, gt_k, 10, ef,
            |v, d, s, e, q, ef, vis| search_baseline(v, d, s, e, q, ef, vis),
        );

        println!();
    }

    println!("\n=== 瀹為獙瀹屾垚 ===");
}
