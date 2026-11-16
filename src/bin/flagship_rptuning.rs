//! 鏃楄埌绱㈠紩 + RP-Tuning Pareto 鏇茬嚎瀹為獙锛堢浜岄樁娈碉級
//!
//! 鏈€缁堝弬鏁帮細伪=1.2, 尾=0.0, l_build=200, r_max=64, r_soft=96
//! 娴佺▼锛?
//! 1. OPQ 鏃嬭浆锛坙earn 闆?100K锛?
//! 2. AVQ 璁粌锛堟棆杞悗 learn, K=256, sub_dim=8, 伪=0.30, iter=5锛?
//! 3. 鏃楄埌鍥炬瀯寤猴紙鏃嬭浆鍚?train, 伪=1.2, l_build=200, r_max=64, r_soft=96, max_iter=2锛?
//! 4. RP-Tuning 鐢熸垚 伪=1.0/1.2/1.5/2.0 鍙樹綋锛堢绾э級
//! 5. 瀵规瘡涓彉浣撹窇 f32 鍜?ADC+rerank 鎼滅储锛岀粯鍒?Pareto 鏇茬嚎
//!
//! 鐩爣锛氳瘉鏄?RP-Tuning 鍦ㄧ瀵嗗浘锛坮_max=64锛変笂浠嶆棤閫€鍖栵紝涓旇兘瑕嗙洊鏁存潯鍓嶆部

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::opq::OPQRotation;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::rp_tuning::{RPTuning, RPTuningConfig, RPTuningStorageScheme};
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

/// f32 鎼滅储锛岃繑鍥?(recall@10, qps)
fn eval_f32(
    train: &[f32],
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    dim: usize,
    test: &[f32],
    gt: &[i32],
    nq: usize,
    k: usize,
    ef_search: usize,
) -> (f64, f64) {
    let graph = VamanaGraph::from_storage(storage.clone(), entry_point, dim);
    let mut searcher = GraphSearcher::new(train, &graph, ef_search);
    let gt_stride = 100;

    let t0 = Instant::now();
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
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
    (recall, qps)
}

/// ADC + rerank 鎼滅储锛岃繑鍥?(recall@10, qps)
fn eval_adc_rerank(
    train: &[f32],
    quantized_db: &[f32],
    storage: &raven::memory::HybridBlockedCsr,
    entry_point: u32,
    dim: usize,
    test: &[f32],
    gt: &[i32],
    nq: usize,
    k: usize,
    ef_search: usize,
    top_n: usize,
) -> (f64, f64) {
    let graph = VamanaGraph::from_storage(storage.clone(), entry_point, dim);
    let mut searcher = GraphSearcher::new(quantized_db, &graph, ef_search);
    let gt_stride = 100;

    let t0 = Instant::now();
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = searcher.search(query, top_n);
        // f32 rerank
        let mut reranked: Vec<(u32, f32)> = candidates
            .iter()
            .map(|(id, _)| {
                let v = &train[*id as usize * dim..(*id as usize + 1) * dim];
                (*id, l2_simd(query, v))
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let found: Vec<u32> = reranked.iter().take(k).map(|(id, _)| *id).collect();
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
    (recall, qps)
}

fn main() {
    println!("=== 鏃楄埌绱㈠紩 + RP-Tuning Pareto 鏇茬嚎瀹為獙锛堢浜岄樁娈碉級===");
    println!("鏈€缁堝弬鏁帮細伪=1.2, 尾=0.0, l_build=200, r_max=64, r_soft=96, max_iter=2");
    println!("OPQ 鏃嬭浆 + AVQ 閲忓寲 + RP-Tuning 伪 鍙樹綋");
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    let (mut learn, _, n_learn) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("鏁版嵁鍔犺浇: {:.1}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}, learn={}", dim, n, nq, gt_nq, gt_k, n_learn);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    for v in learn.iter_mut() { *v /= 255.0; }

    let k = 10;
    let top_n = 100;
    let ef_points = vec![50, 100, 200, 400];

    // 2. OPQ 璁粌 + 鏃嬭浆
    println!("=== OPQ 璁粌 + 鏃嬭浆 ===");
    let t0 = Instant::now();
    let opq = OPQRotation::train_with_sub_dim(&learn, dim, 8);
    let train_rot = opq.apply(&train, dim);
    let test_rot = opq.apply(&test, dim);
    let learn_rot = opq.apply(&learn, dim);
    println!("OPQ 璁粌 + 鏃嬭浆: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. AVQ 璁粌 + 閲忓寲鏁版嵁搴?
    println!("=== AVQ 璁粌锛堟棆杞悗 learn, K=256, sub_dim=8, 伪=0.30, iter=5锛?==");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 璁粌: {:.1}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train_rot[i * dim..(i + 1) * dim];
            cb.decode(&cb.encode(v))
        })
        .collect();
    println!("閲忓寲鏁版嵁搴撴瀯閫? {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 4. 鏃楄埌鍥炬瀯寤猴紙鏈€缁堝弬鏁帮級
    println!("=== 鏃楄埌鍥炬瀯寤猴紙伪=1.2, l_build=200, r_max=64, r_soft=96, max_iter=2锛?==");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_soft: 96,
        r_max: 64,
        max_iterations: 2,
..Default::default()
    };
    let base_graph = VamanaGraph::build(&train_rot, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("鏃楄埌鍥炬瀯寤哄畬鎴? {:.1}s", build_time);
    println!("骞冲潎鍑哄害: {:.1}", base_graph.degree_stats().mean_degree);
    println!();

    // 5. RP-Tuning 鐢熸垚 伪 鍙樹綋锛堢绾э級
    println!("=== RP-Tuning 鐢熸垚 伪 鍙樹綋锛圫cheme A锛?==");
    let t0 = Instant::now();
    let rp_config = RPTuningConfig {
        scheme: RPTuningStorageScheme::SchemeA,
        alpha_points: vec![1.0, 1.2, 1.5, 2.0],
        r_max: 64,
    };
    let variants = RPTuning::generate_variants(&base_graph, &train_rot, dim, &rp_config);
    println!("RP-Tuning 鐢熸垚 {} 涓彉浣? {:.2}s", variants.len(), t0.elapsed().as_secs_f64());
    println!();

    // 6. f32 Pareto 鏇茬嚎
    println!("=== f32 Pareto 鏇茬嚎 ===");
    println!("{:<8} {:<10} {:>12} {:>10}", "alpha", "ef_search", "f32_recall", "f32_qps");
    println!("{:-<44}", "");

    for variant in &variants {
        for &ef in &ef_points {
            let (recall, qps) = eval_f32(
                &train_rot, &variant.storage, variant.entry_point,
                dim, &test_rot, &gt, nq, k, ef,
            );
            println!("{:<8.1} {:<10} {:>12.4} {:>10.0}", variant.alpha, ef, recall, qps);
        }
        println!();
    }

    // 7. ADC + rerank Pareto 鏇茬嚎
    println!("=== ADC + rerank Pareto 鏇茬嚎 ===");
    println!("{:<8} {:<10} {:>14} {:>12}", "alpha", "ef_search", "adc_rerank", "adc_qps");
    println!("{:-<48}", "");

    for variant in &variants {
        for &ef in &ef_points {
            let (recall, qps) = eval_adc_rerank(
                &train_rot, &quantized_db, &variant.storage, variant.entry_point,
                dim, &test_rot, &gt, nq, k, ef, top_n,
            );
            println!("{:<8.1} {:<10} {:>14.4} {:>12.0}", variant.alpha, ef, recall, qps);
        }
        println!();
    }

    // 8. 姹囨€?
    println!("=== 姹囨€?===");
    println!("鏃楄埌鍥惧弬鏁? 伪=1.2, l_build=200, r_max=64, r_soft=96, max_iter=2");
    println!("OPQ: sub_dim=8, AVQ: K=256, sub_dim=8, 伪=0.30, iter=5");
    println!("RP-Tuning 鍙樹綋: 伪=[1.0, 1.2, 1.5, 2.0], Scheme A, r_max=64");
    println!("ef_search: [50, 100, 200, 400]");
    println!();
    println!("Pareto 鍓嶆部鍒嗘瀽锛?);
    println!("  f32: 楠岃瘉 RP-Tuning 鍦ㄧ瀵嗗浘锛坮_max=64锛変笂鏃犻€€鍖?);
    println!("  ADC+rerank: 楠岃瘉 OPQ+AVQ 閲忓寲鍚庣殑 Pareto 鍓嶆部瑕嗙洊");
}
