//! RP-Tuning SIFT1M Pareto 鏇茬嚎瀹為獙
//!
//! 1. 寤哄浘锛坒32, 伪=1.2, max_iter=2锛?
//! 2. RP-Tuning 鐢熸垚 伪 鍙樹綋锛埼?1.0, 1.2, 1.5, 2.0锛?
//! 3. 瀵规瘡涓彉浣擄紝鐢ㄤ笉鍚?ef_search锛?0, 100, 200, 400锛夎窇鎼滅储
//! 4. 杈撳嚭 Pareto 鏇茬嚎鏁版嵁锛坮ecall-QPS锛?

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::rp_tuning::{RPTuning, RPTuningConfig, RPTuningStorageScheme};
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

/// 瀵瑰崟涓彉浣?+ ef_search 璺戞悳绱紝杩斿洖 (recall, qps)
fn eval_variant(
    vectors: &[f32],
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
    let mut searcher = GraphSearcher::new(vectors, &graph, ef_search);
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
    let search_time = t0.elapsed().as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    let qps = nq as f64 / search_time;
    (recall, qps)
}

fn main() {
    println!("=== RP-Tuning SIFT1M Pareto 鏇茬嚎瀹為獙 ===");
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }

    // 2. 寤哄浘锛坒32, 伪=1.2, max_iter=2锛?
    println!("=== 寤哄浘锛圴amana 伪=1.2, r_max=32, l_build=100, max_iter=2锛?==");
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
    let base_graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("寤哄浘鏃堕棿: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. RP-Tuning 鐢熸垚 伪 鍙樹綋锛堢绾э級
    println!("=== RP-Tuning 鐢熸垚 伪 鍙樹綋锛圫cheme A锛?==");
    let t0 = Instant::now();
    let rp_config = RPTuningConfig {
        scheme: RPTuningStorageScheme::SchemeA,
        alpha_points: vec![1.0, 1.2, 1.5, 2.0],
        r_max: 32,
    };
    let variants = RPTuning::generate_variants(&base_graph, &train, dim, &rp_config);
    println!("RP-Tuning 鐢熸垚 {} 涓彉浣? {:.2}s", variants.len(), t0.elapsed().as_secs_f64());
    println!();

    // 4. 瀵规瘡涓彉浣?+ 姣忎釜 ef_search 璺戞悳绱?
    let ef_points = vec![50, 100, 200, 400];
    let k = 10;

    println!("=== Pareto 鏇茬嚎鏁版嵁 ===");
    println!("{:<8} {:<10} {:>10} {:>10}", "alpha", "ef_search", "recall@10", "QPS");
    println!("{:-<42}", "");

    for variant in &variants {
        for &ef in &ef_points {
            let (recall, qps) = eval_variant(
                &train, &variant.storage, variant.entry_point,
                dim, &test, &gt, nq, k, ef,
            );
            println!("{:<8.1} {:<10} {:>10.4} {:>10.0}", variant.alpha, ef, recall, qps);
        }
        println!();
    }

    // 5. 姹囨€?
    println!("=== 姹囨€?===");
    println!("鍩虹鍥? 伪=1.2, r_max=32, max_iter=2");
    println!("RP-Tuning 鍙樹綋: 伪=[1.0, 1.2, 1.5, 2.0], Scheme A");
    println!("ef_search: [50, 100, 200, 400]");
    println!();
    println!("Pareto 鍓嶆部鍒嗘瀽锛毼?瓒婂ぇ淇濈暀鏇村闀跨▼杈癸紝recall 瓒婇珮浣?QPS 瓒婁綆");
}
