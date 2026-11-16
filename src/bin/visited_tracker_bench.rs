//! VisitedTracker 澶嶇敤闅旂瀵规瘮鍩哄噯锛堢瀛﹂獙璇侊級
//!
//! 鐩爣锛氶殧绂绘祴璇?VisitedTracker 澶嶇敤瀵规悳绱㈡€ц兘鐨勫奖鍝?
//!
//! 鏂规硶锛?
//! 1. 鐢?learn 闆嗭紙100K锛夊揩閫熷缓鍥撅紙~5 鍒嗛挓锛?
//! 2. 瀵瑰悓涓€寮犲浘锛屽垎鍒敤锛?
//!    - baseline锛歡reedy_search_vec锛堟瘡娆℃柊寤?VisitedTracker锛屽垎閰?100KB锛?
//!    - reuse锛歡reedy_search_vec_reuse锛堝鐢?VisitedTracker锛岄浂鍒嗛厤锛?
//! 3. 娴嬮噺 QPS銆乺ecall@10銆乸50/p99 寤惰繜
//!
//! 鍒ゆ嵁锛?
//! - QPS 鎻愬崌 鈮?%锛宺ecall 瀹屽叏涓嶅彉 鈫?浼樺寲鏈夋晥
//! - QPS 鏃犲彉鍖栨垨涓嬮檷 鈫?浼樺寲鏃犳晥锛屽洖閫€
//! - recall 涓嬮檷 鈫?瀛樺湪 bug锛岀珛鍗充慨澶?

use std::fs::File;
use std::io::Read;
use std::time::{Instant, Duration};
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::memory::VisitedTracker;
use raven::distance::l2_simd;
use raven::build::ChaCha8Rng;

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

/// 璇诲彇 ivecs 鏂囦欢
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

/// baseline 鎼滅储锛堟瘡娆℃柊寤?VisitedTracker锛?
///
/// 妯℃嫙浼樺寲鍓嶇殑琛屼负锛氭瘡娆℃悳绱㈤兘鍒嗛厤 visited 鏁扮粍
fn search_baseline(
    vectors: &[f32],
    graph: &VamanaGraph,
    query: &[f32],
    ef_search: usize,
    k: usize,
    dim: usize,
) -> Vec<(u32, f32)> {
    let (candidates, _visited) = VamanaGraph::greedy_search_vec(
        vectors,
        dim,
        graph.storage(),
        graph.entry_point(),
        query,
        ef_search,
    );
    let mut results: Vec<(u32, f32)> = candidates
        .into_iter()
        .map(|id| {
            let v = &vectors[id as usize * dim..(id as usize + 1) * dim];
            (id, l2_simd(query, v))
        })
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// reuse 鎼滅储锛堝鐢?VisitedTracker锛?
///
/// 浼樺寲鍚庣殑琛屼负锛氬鐢ㄩ鍒嗛厤鐨?VisitedTracker
fn search_reuse(
    vectors: &[f32],
    graph: &VamanaGraph,
    query: &[f32],
    ef_search: usize,
    k: usize,
    dim: usize,
    visited: &mut VisitedTracker,
) -> Vec<(u32, f32)> {
    let candidates = VamanaGraph::greedy_search_vec_reuse(
        vectors,
        dim,
        graph.storage(),
        graph.entry_point(),
        query,
        ef_search,
        visited,
        8, // po: prefetch offset
    );
    // 璺濈宸插湪 greedy_search_vec_reuse 涓绠楋紝鍙渶鎺掑簭鍙?top-k
    let mut results = candidates;
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// 杩愯鎼滅储鍩哄噯锛岃繑鍥?(recall, qps, latencies)
fn run_bench<F>(
    name: &str,
    vectors: &[f32],
    _graph: &VamanaGraph,
    queries: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    _ef_search: usize,
    k: usize,
    mut search_fn: F,
) -> (f64, f64, Vec<Duration>)
where
    F: FnMut(&[f32], &[f32]) -> Vec<(u32, f32)>,
{
    let gt_stride = 100;
    let mut hits = 0usize;
    let mut latencies = Vec::with_capacity(nq);

    // 棰勭儹锛氳窇鍓?100 涓?query 涓嶈鏃?
    for q in 0..100.min(nq) {
        let query = &queries[q * dim..(q + 1) * dim];
        let _ = search_fn(query, vectors);
    }

    // 姝ｅ紡璁℃椂
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let tq = Instant::now();
        let result = search_fn(query, vectors);
        latencies.push(tq.elapsed());

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    let qps = nq as f64 / elapsed;

    // 璁＄畻寤惰繜鍒嗕綅鏁?
    let mut sorted_lat = latencies.clone();
    sorted_lat.sort();
    let p50 = sorted_lat[sorted_lat.len() / 2];
    let p99 = sorted_lat[(sorted_lat.len() as f64 * 0.99) as usize];

    println!("{}: recall={:.4}, QPS={:.0}, p50={:.2}碌s, p99={:.2}碌s",
        name, recall, qps,
        p50.as_secs_f64() * 1e6,
        p99.as_secs_f64() * 1e6);

    (recall, qps, latencies)
}

fn main() {
    println!("=== VisitedTracker 澶嶇敤闅旂瀵规瘮鍩哄噯 ===");
    println!("鐩爣锛氱瀛﹂獙璇?VisitedTracker 澶嶇敤瀵规悳绱㈡€ц兘鐨勫奖鍝?);
    println!("鏂规硶锛氬悓涓€寮犲浘锛屽姣?baseline锛堟瘡娆℃柊寤猴級vs reuse锛堝鐢級");
    println!();

    // 1. 鍔犺浇 base 闆嗗墠 100K 瀛愰泦锛堢‘淇?groundtruth 鏈夋晥锛?
    // 鐢ㄥ墠 100K 鑰岄潪 learn 闆嗭紝鍥犱负 groundtruth 鏄熀浜庡畬鏁?base 闆嗙殑
    let t0 = Instant::now();
    let (full_base, dim, n_full) = read_fvecs("data/sift/sift_base.fvecs");
    let n_db = 100_000.min(n_full); // 鍙栧墠 100K
    let mut db = full_base[..n_db * dim].to_vec();
    drop(full_base); // 閲婃斁瀹屾暣 base
    let (mut queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.1}s", t0.elapsed().as_secs_f64());
    println!("db(base 鍓?100K): {} vecs, queries: {} vecs, dim={}, gt_k={}", n_db, nq, dim, gt_k);
    println!("娉ㄦ剰锛歡roundtruth 鍩轰簬瀹屾暣 base锛宺ecall 浼氫綆浜庡叏閲?);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    for v in db.iter_mut() { *v /= 255.0; }
    for v in queries.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. 寤哄浘锛堜繚瀹堝弬鏁帮紝蹇€熸瀯寤猴級
    println!("=== 寤哄浘锛坙earn 100K, r_max=32, l_build=100, 伪=1.2锛?==");
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
    let graph = VamanaGraph::build(&db, dim, &config, &mut rng);
    println!("寤哄浘瀹屾垚: {:.1}s, avg_degree={:.1}", t0.elapsed().as_secs_f64(), graph.degree_stats().mean_degree);
    println!();

    // 3. 棰勫垎閰?VisitedTracker锛坮euse 妯″紡鐢級
    let mut visited = VisitedTracker::new(n_db, ef_search);

    // 4. baseline 鎼滅储锛堟瘡娆℃柊寤?VisitedTracker锛?
    println!("=== 鎼滅储鍩哄噯锛?0K 鏌ヨ, ef=100锛?==");
    println!();

    let (baseline_recall, baseline_qps, baseline_lat) = run_bench(
        "baseline (姣忔鏂板缓 VisitedTracker)",
        &db,
        &graph,
        &queries,
        &gt,
        dim,
        nq,
        ef_search,
        k,
        |query, vectors| search_baseline(vectors, &graph, query, ef_search, k, dim),
    );

    // 5. reuse 鎼滅储锛堝鐢?VisitedTracker锛?
    println!();

    let (reuse_recall, reuse_qps, reuse_lat) = run_bench(
        "reuse (澶嶇敤 VisitedTracker)",
        &db,
        &graph,
        &queries,
        &gt,
        dim,
        nq,
        ef_search,
        k,
        |query, vectors| search_reuse(vectors, &graph, query, ef_search, k, dim, &mut visited),
    );

    // 6. 瀵规瘮鍒嗘瀽
    println!();
    println!("=== 瀵规瘮鍒嗘瀽 ===");
    println!("{:<30} {:>12} {:>12} {:>12}", "", "baseline", "reuse", "diff");
    println!("{:-<70}", "");
    println!("{:<30} {:>12.4} {:>12.4} {:>12.4}", "recall@10", baseline_recall, reuse_recall, reuse_recall - baseline_recall);
    println!("{:<30} {:>12.0} {:>12.0} {:>11.1}%", "QPS", baseline_qps, reuse_qps, (reuse_qps / baseline_qps - 1.0) * 100.0);

    // 寤惰繜鍒嗕綅鏁?
    let mut bl_sorted = baseline_lat.clone();
    bl_sorted.sort();
    let mut ru_sorted = reuse_lat.clone();
    ru_sorted.sort();
    let bl_p50 = bl_sorted[bl_sorted.len() / 2];
    let ru_p50 = ru_sorted[ru_sorted.len() / 2];
    let bl_p99 = bl_sorted[(bl_sorted.len() as f64 * 0.99) as usize];
    let ru_p99 = ru_sorted[(ru_sorted.len() as f64 * 0.99) as usize];

    println!("{:<30} {:>10.2}碌s {:>10.2}碌s {:>11.1}%", "p50 寤惰繜",
        bl_p50.as_secs_f64() * 1e6, ru_p50.as_secs_f64() * 1e6,
        (ru_p50.as_secs_f64() / bl_p50.as_secs_f64() - 1.0) * 100.0);
    println!("{:<30} {:>10.2}碌s {:>10.2}碌s {:>11.1}%", "p99 寤惰繜",
        bl_p99.as_secs_f64() * 1e6, ru_p99.as_secs_f64() * 1e6,
        (ru_p99.as_secs_f64() / bl_p99.as_secs_f64() - 1.0) * 100.0);

    println!();
    println!("=== 鍒ゅ畾 ===");
    let qps_improvement = (reuse_qps / baseline_qps - 1.0) * 100.0;
    let recall_diff = reuse_recall - baseline_recall;

    if recall_diff.abs() > 1e-6 {
        println!("FAIL: recall 鍙樺寲 {:.6}锛屽瓨鍦?bug锛岄渶瑕佷慨澶?, recall_diff);
    } else if qps_improvement >= 5.0 {
        println!("PASS: QPS 鎻愬崌 {:.1}%锛宺ecall 涓嶅彉锛屼紭鍖栨湁鏁?, qps_improvement);
    } else if qps_improvement > 0.0 {
        println!("MARGINAL: QPS 鎻愬崌 {:.1}%锛? 5%锛夛紝浼樺寲鏁堟灉涓嶆樉钁?, qps_improvement);
    } else {
        println!("FAIL: QPS 涓嬮檷 {:.1}%锛屼紭鍖栨棤鏁堬紝闇€瑕佸洖閫€", qps_improvement);
    }
}
