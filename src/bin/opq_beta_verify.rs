//! OPQ + AVQ + 尾 楠岃瘉瀹為獙锛堢涓€闃舵锛?
//!
//! 鐩爣锛氬垽鏂?OPQ 绌洪棿鏃嬭浆鏄惁璁?尾 澶嶆椿
//!
//! 娴佺▼锛?
//! 1. 璁粌 OPQ 鏃嬭浆鐭╅樀锛坙earn 闆?100K锛?
//! 2. 鐢?OPQ 鏃嬭浆鍚戦噺锛坙earn + train + test锛?
//! 3. 鐢ㄦ棆杞悗鐨勫悜閲忚缁?AVQ codebook锛圞=256, sub_dim=8, 伪=0.30, iter=5锛?
//! 4. 瀵?尾=0.0 鍜?0.3 寤哄浘瀵规瘮
//! 5. 璇勪及 f32 recall 鍜?ADC+rerank recall
//!
//! 鍒ゆ柇鏍囧噯锛?
//! - 鑻?尾=0.3 鐨?ADC+rerank recall 鏄捐憲浼樹簬 尾=0.0锛?0.5%锛夛紝鍒?OPQ 璁?尾 澶嶆椿
//! - 鍚﹀垯 尾 浠嶄繚鎸?0.0
//!
//! 鍏抽敭鎬ц川锛歄PQ 鏃嬭浆鏄浜ゅ彉鎹紝淇濇寔 L2 璺濈锛屾墍浠?groundtruth 浠嶇劧鏈夋晥

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::opq::OPQRotation;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
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

/// ADC + rerank 鎼滅储锛岃繑鍥?(recall@10, qps, avg_degree)
fn eval_adc_rerank(
    train: &[f32],
    quantized_db: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    gt_stride: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    top_n: usize,
    k: usize,
) -> (f64, f64, f64) {
    let avg_deg = graph.degree_stats().mean_degree;

    let mut searcher = GraphSearcher::new(quantized_db, graph, ef_search);
    let mut hits = 0usize;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let candidates = searcher.search(query, top_n);
        // f32 rerank锛堝湪鏃嬭浆鍚庣殑绌洪棿锛孡2 璺濈淇濇寔锛?
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
    (recall, qps, avg_deg)
}

/// f32 鎼滅储锛堟棤閲忓寲锛夛紝杩斿洖 (recall@10, qps)
fn eval_f32(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    gt_stride: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    k: usize,
) -> (f64, f64) {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let mut hits = 0usize;
    let t0 = Instant::now();
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

fn main() {
    println!("=== OPQ + AVQ + 尾 楠岃瘉瀹為獙锛堢涓€闃舵锛?==");
    println!("鐩爣锛氬垽鏂?OPQ 绌洪棿鏃嬭浆鏄惁璁?尾 澶嶆椿");
    println!("娴佺▼锛歄PQ 璁粌 鈫?鏃嬭浆鍚戦噺 鈫?AVQ 璁粌 鈫?尾=0.0/0.3 寤哄浘瀵规瘮");
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

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;
    let top_n = 100;

    // 2. 璁粌 OPQ 鏃嬭浆鐭╅樀锛堢敤 learn 闆?100K锛?
    println!("=== OPQ 璁粌锛坙earn 闆?100K, sub_dim=8锛?==");
    let t0 = Instant::now();
    let opq = OPQRotation::train_with_sub_dim(&learn, dim, 8);
    println!("OPQ 璁粌: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 鐢?OPQ 鏃嬭浆鍚戦噺锛坙earn + train + test锛?
    // OPQ 鏄浜ゅ彉鎹紝淇濇寔 L2 璺濈锛屾墍浠?groundtruth 浠嶇劧鏈夋晥
    println!("=== 搴旂敤 OPQ 鏃嬭浆 ===");
    let t0 = Instant::now();
    let train_rot = opq.apply(&train, dim);
    let test_rot = opq.apply(&test, dim);
    let learn_rot = opq.apply(&learn, dim);
    println!("鍚戦噺鏃嬭浆: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 4. 鐢ㄦ棆杞悗鐨勫悜閲忚缁?AVQ codebook
    println!("=== AVQ 璁粌锛堟棆杞悗 learn 100K, K=256, sub_dim=8, 伪=0.30, iter=5锛?==");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 璁粌: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 5. 閲忓寲鏁版嵁搴擄紙鐢ㄦ棆杞悗鐨?train锛?
    let t0 = Instant::now();
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train_rot[i * dim..(i + 1) * dim];
            cb.decode(&cb.encode(v))
        })
        .collect();
    println!("閲忓寲鏁版嵁搴撴瀯閫? {:.1}s", t0.elapsed().as_secs_f64());

    // 6. 棰勮绠楁墍鏈夎妭鐐圭殑閲忓寲璇樊
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train_rot))
        .collect();
    println!("鑺傜偣閲忓寲璇樊棰勮绠? {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 7. Vamana 寤哄浘閰嶇疆锛堝浐瀹氾級
    let build_config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };

    // 8. 鎵弿 尾=0.0 鍜?0.3
    let betas = [0.0f32, 0.3];

    println!("=== OPQ + AVQ + 尾 楠岃瘉缁撴灉 ===");
    println!("{:>6} {:>12} {:>10} {:>14} {:>12} {:>10} {:>10}",
        "beta", "f32_recall", "f32_qps", "adc_rerank", "adc_qps", "degrad", "avg_deg");
    println!("{:-<82}", "");

    let mut f32_baseline_recall = 0.0f64;

    for &beta in &betas {
        let mut rng = ChaCha8Rng::seed_from(42);

        let t0 = Instant::now();
        let graph = if beta == 0.0 {
            println!("[尾={:.1}] 寤哄浘锛堟爣鍑?RobustPrune锛孫PQ 鏃嬭浆绌洪棿锛?..", beta);
            VamanaGraph::build(&train_rot, dim, &build_config, &mut rng)
        } else {
            println!("[尾={:.1}] 寤哄浘锛堥噺鍖栨劅鐭?RobustPrune锛孫PQ 鏃嬭浆绌洪棿锛?..", beta);
            let qa_config = QuantAwarePruneConfig {
                alpha: 1.2,
                beta,
                epsilon: EPSILON,
                r_max: 32,
                normalization: NormalizationScheme::Mean,
            };
            let ne = &node_errors;
            VamanaGraph::build_with_quant_aware_prune(
                &train_rot, dim, &build_config, &qa_config,
                move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
                &mut rng,
            )
        };
        let build_time = t0.elapsed().as_secs_f64();
        println!("[尾={:.1}] 寤哄浘瀹屾垚: {:.1}s", beta, build_time);

        // f32 鎼滅储锛堝湪鏃嬭浆鍚庣殑绌洪棿锛孡2 璺濈淇濇寔锛?
        let (f32_recall, f32_qps) = eval_f32(
            &train_rot, &test_rot, &gt, dim, nq, gt_stride, &graph, ef_search, k,
        );

        if beta == 0.0 {
            f32_baseline_recall = f32_recall;
        }

        // ADC + rerank 鎼滅储
        let (adc_recall, adc_qps, avg_deg) = eval_adc_rerank(
            &train_rot, &quantized_db, &test_rot, &gt, dim, nq, gt_stride,
            &graph, ef_search, top_n, k,
        );

        let degrad = f32_baseline_recall - adc_recall;

        println!("{:>6.1} {:>12.4} {:>10.0} {:>14.4} {:>12.0} {:>10.4} {:>10.1}",
            beta, f32_recall, f32_qps, adc_recall, adc_qps, degrad, avg_deg);
        println!();
    }

    println!("=== 缁撹鍒ゆ柇 ===");
    println!("瀵规瘮 尾=0.0 鍜?尾=0.3 鐨?ADC+rerank recall锛?);
    println!("  鑻?尾=0.3 recall 鏄捐憲浼樹簬 尾=0.0锛?0.5%锛夆啋 OPQ 璁?尾 澶嶆椿锛岄攣瀹?尾=0.3");
    println!("  鍚﹀垯 鈫?尾 浠嶄繚鎸?0.0锛圤PQ 鏈兘鏀瑰彉 SIFT 鏁版嵁閲忓寲璇樊鍧囧寑鍒嗗竷鐨勭壒鎬э級");
    println!();
    println!("鍙傝€冿細鏈姞 OPQ 鐨?尾 娑堣瀺缁撴灉锛堝凡瀹為獙锛?);
    println!("  尾=0.0: adc_rerank=0.9213, 尾=0.3: adc_rerank=0.9177锛埼?鏃犳鏀剁泭锛?);
}
