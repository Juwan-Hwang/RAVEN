//! SIFT1M 绔埌绔熀鍑嗘祴璇?
//!
//! 鐩爣锛氭祴閲?1M 鍚戦噺涓嬬殑鐪熷疄鐡堕
//! - f32 寤哄浘鏃堕棿 + 鎼滅储 QPS + recall
//! - AVQ 璁粌鏃堕棿 + ADC 鎼滅储 QPS
//! - ADC + rerank QPS + recall
//!
//! 鍙傛暟锛圵eek 6 鏈€浼橈級锛?
//! - Vamana: 伪=1.0, r_max=32, l_build=100, r_soft=48
//! - AVQ: K=256, sub_dim=8, 伪=0.30, iterations=25
//! - rerank: top-100 鈫?top-10, ef_search=100

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::l2_simd;

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

/// recall@k 璁＄畻锛坓t_stride = groundtruth 姣忔煡璇㈤偦灞呮暟锛?
fn recall_at_k(found: &[u32], gt_slice: &[i32], k: usize) -> f64 {
    let mut hits = 0usize;
    for &g in gt_slice.iter().take(k) {
        if found.contains(&(g as u32)) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

/// 璁＄畻寤惰繜鍒嗕綅鏁帮紙OPT-15锛?
///
/// 杈撳叆锛氬欢杩熸暟缁勶紙绾崇锛夛紝杩斿洖 (p50, p99, p999) 姣
fn latency_percentiles(latencies_ns: &[u64]) -> (f64, f64, f64) {
    if latencies_ns.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = latencies_ns.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p50 = sorted[n / 2] as f64 / 1_000_000.0;
    let p99 = sorted[(n as f64 * 0.99) as usize] as f64 / 1_000_000.0;
    let p999 = sorted[(n as f64 * 0.999) as usize] as f64 / 1_000_000.0;
    (p50, p99, p999)
}

fn main() {
    println!("=== SIFT1M 绔埌绔熀鍑嗘祴璇?===");
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    // 鍔犺浇 sift_learn锛?00K锛夌敤浜?AVQ 璁粌锛屾瘮鐢?1M base 蹇?10 鍊?
    let (mut learn, _, n_learn) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("鏁版嵁鍔犺浇: {}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}, learn={}", dim, n, nq, gt_nq, gt_k, n_learn);
    println!();

    // 褰掍竴鍖栧埌 [0,1]锛圫IFT 鍘熷 0-255锛孉VQ 璁粌闇€瑕侊級
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }
    for v in learn.iter_mut() { *v /= max_val; }

    // 2. f32 寤哄浘锛圴amana two passes: 伪=1.0鈫?.2, rayon 骞惰锛?
    println!("=== f32 寤哄浘锛圴amana 伪=1.2, r_max=32, l_build=100, max_iter=2, rayon 骞惰锛?==");
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

    // 3. f32 鎼滅储 QPS + recall + 寤惰繜鍒嗕綅鏁帮紙OPT-15锛?
    println!("=== f32 鎼滅储锛坋f_search=100, k=10锛?==");
    let mut searcher = GraphSearcher::new(&train, &graph, 100);
    let t0 = Instant::now();
    let gt_stride = gt_k;
    let k = 10;
    let mut recall_sum = 0.0f64;
    let mut latencies_f32: Vec<u64> = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let tq = Instant::now();
        let result = searcher.search(query, k);
        latencies_f32.push(tq.elapsed().as_nanos() as u64);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let search_time = t0.elapsed().as_secs_f64();
    let recall_f32 = recall_sum / nq as f64;
    let qps_f32 = nq as f64 / search_time;
    let (p50, p99, p999) = latency_percentiles(&latencies_f32);
    println!("f32 recall@10={:.4}, QPS={:.0}, avg_latency={:.2}ms, p50={:.2}ms, p99={:.2}ms, p999={:.2}ms",
        recall_f32, qps_f32, search_time * 1000.0 / nq as f64, p50, p99, p999);
    println!();

    // 4. AVQ 璁粌锛堢敤 sift_learn 100K + iter=5 鍔犻€燂紝宸ヤ笟鏍囧噯锛?
    println!("=== AVQ 璁粌锛坰ift_learn 100K, K=256, sub_dim=8, 伪=0.30, iter=5锛?==");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    let avq_train_time = t0.elapsed().as_secs_f64();
    println!("AVQ 璁粌鏃堕棿: {:.2}s", avq_train_time);
    println!();

    // 5. 鏋勯€犻噺鍖栨暟鎹簱锛圓DC 绮楃瓫鐢級
    let t0 = Instant::now();
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            cb.decode(&cb.encode(v))
        })
        .collect();
    println!("閲忓寲鏁版嵁搴撴瀯閫? {:.2}s", t0.elapsed().as_secs_f64());

    // 6. ADC 鎼滅储 QPS锛堟棤 rerank锛? 寤惰繜鍒嗕綅鏁帮紙OPT-15锛?
    println!();
    println!("=== ADC 鎼滅储锛堥噺鍖栬窛绂? ef_search=100, k=10锛?==");
    let mut searcher_q = GraphSearcher::new(&quantized_db, &graph, 100);
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    let mut latencies_adc: Vec<u64> = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let tq = Instant::now();
        let result = searcher_q.search(query, k);
        latencies_adc.push(tq.elapsed().as_nanos() as u64);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let adc_time = t0.elapsed().as_secs_f64();
    let recall_adc = recall_sum / nq as f64;
    let qps_adc = nq as f64 / adc_time;
    let (p50_adc, p99_adc, p999_adc) = latency_percentiles(&latencies_adc);
    println!("ADC recall@10={:.4}, QPS={:.0}, avg_latency={:.2}ms, p50={:.2}ms, p99={:.2}ms, p999={:.2}ms",
        recall_adc, qps_adc, adc_time * 1000.0 / nq as f64, p50_adc, p99_adc, p999_adc);
    println!();

    // 7. ADC + rerank QPS + recall + 寤惰繜鍒嗕綅鏁帮紙top-100 鈫?f32 rerank 鈫?top-10锛?
    println!("=== ADC + rerank锛坱op-100 绮楃瓫 鈫?f32 绮炬帓 鈫?top-10锛?==");
    let top_n = 100;
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    let mut latencies_rerank: Vec<u64> = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let tq = Instant::now();
        let candidates = searcher_q.search(query, top_n);
        // f32 rerank锛圫IMD 鍔犻€燂級
        let mut reranked: Vec<(u32, f32)> = candidates
            .iter()
            .map(|(id, _)| {
                let v = &train[*id as usize * dim..(*id as usize + 1) * dim];
                (*id, l2_simd(query, v))
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        latencies_rerank.push(tq.elapsed().as_nanos() as u64);
        let found: Vec<u32> = reranked.iter().take(k).map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        recall_sum += recall_at_k(&found, gt_slice, k);
    }
    let rerank_time = t0.elapsed().as_secs_f64();
    let recall_rerank = recall_sum / nq as f64;
    let qps_rerank = nq as f64 / rerank_time;
    let (p50_rr, p99_rr, p999_rr) = latency_percentiles(&latencies_rerank);
    println!("ADC+rerank recall@10={:.4}, QPS={:.0}, avg_latency={:.2}ms, p50={:.2}ms, p99={:.2}ms, p999={:.2}ms",
        recall_rerank, qps_rerank, rerank_time * 1000.0 / nq as f64, p50_rr, p99_rr, p999_rr);
    println!();

    // 8. 姹囨€?
    println!("=== 姹囨€?===");
    println!("{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}", "鏂规硶", "recall@10", "QPS", "avg_ms", "p50_ms", "p99_ms", "p999_ms");
    println!("{:-<82}", "");
    println!("{:<20} {:>10.4} {:>10.0} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "f32 baseline", recall_f32, qps_f32, search_time * 1000.0 / nq as f64, p50, p99, p999);
    println!("{:<20} {:>10.4} {:>10.0} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "AVQ ADC", recall_adc, qps_adc, adc_time * 1000.0 / nq as f64, p50_adc, p99_adc, p999_adc);
    println!("{:<20} {:>10.4} {:>10.0} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "AVQ ADC+rerank", recall_rerank, qps_rerank, rerank_time * 1000.0 / nq as f64, p50_rr, p99_rr, p999_rr);
    println!();
    println!("寤哄浘鏃堕棿: {:.2}s | AVQ 璁粌: {:.2}s", build_time, avq_train_time);
}
