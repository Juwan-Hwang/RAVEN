//! SIFT1M 尾 娑堣瀺瀹為獙
//!
//! 鍥哄畾 Vamana 伪=1.2锛圧P-Tuning 纭鏈€浼橈級锛孉VQ 伪=0.30
//! 鎵弿 尾=0.0/0.1/0.3/1.0锛堥噺鍖栨劅鐭?RobustPrune 鏉冮噸锛?
//!
//! 尾=0.0锛氭爣鍑?RobustPrune锛堝鐓х粍锛屽凡鏈夊熀绾匡級
//! 尾>0锛氶噺鍖栨劅鐭ュ壀鏋濓紝鍥為伩閲忓寲璇樊澶х殑杈?
//!
//! 鐩爣锛氶獙璇?QuantAwareRobustPrune 鏄惁鑳藉噺灏?AVQ 閲忓寲閫€鍖?
//! 褰撳墠 尾=0 閫€鍖栵細f32 0.9528 鈫?AVQ ADC+rerank 0.9228锛堥€€鍖?3%锛?

use std::fs::File;
use std::io::Read;
use std::time::Instant;
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
    _codebook: &AVQCodebook,
    train: &[f32],
    quantized_db: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    _n: usize,
    nq: usize,
    gt_stride: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    top_n: usize,
    k: usize,
) -> (f64, f64, f64) {
    let avg_deg = graph.degree_stats().mean_degree;

    // ADC 鎼滅储 + rerank
    let mut searcher = GraphSearcher::new(quantized_db, graph, ef_search);
    let mut hits = 0usize;
    let t0 = Instant::now();
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
    println!("=== SIFT1M 尾 娑堣瀺瀹為獙 ===");
    println!("鍥哄畾 Vamana 伪=1.2, AVQ 伪=0.30, K=256, sub_dim=8");
    println!("鎵弿 尾=0.0/0.1/0.3/1.0");
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

    // 2. AVQ 璁粌锛堝彧璁粌涓€娆★紝鎵€鏈?尾 鍏辩敤锛?
    println!("=== AVQ 璁粌锛坰ift_learn 100K, K=256, sub_dim=8, 伪=0.30, iter=5锛?==");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 璁粌: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 閲忓寲鏁版嵁搴擄紙鎵€鏈?尾 鍏辩敤鍚屼竴涓?codebook锛?
    let t0 = Instant::now();
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            cb.decode(&cb.encode(v))
        })
        .collect();
    println!("閲忓寲鏁版嵁搴撴瀯閫? {:.1}s", t0.elapsed().as_secs_f64());

    // 3.5 棰勮绠楁墍鏈夎妭鐐圭殑閲忓寲璇樊锛堥伩鍏嶅缓鍥炬椂閲嶅 encode+decode锛?
    // edge_error(u,v) = mean(node_error(u), node_error(v))
    // 涓嶉璁＄畻鐨勮瘽锛?M 鑺傜偣 脳 ~100 鍊欓€?= 1 浜挎 encode+decode锛屽缓鍥捐鏁板皬鏃?
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train))
        .collect();
    println!("鑺傜偣閲忓寲璇樊棰勮绠? {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 4. Vamana 寤哄浘閰嶇疆锛堝浐瀹氾級
    let build_config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };

    // 5. 鎵弿 尾
    let betas = [0.0f32, 0.1, 0.3, 1.0];

    println!("=== 尾 娑堣瀺缁撴灉 ===");
    println!("{:>6} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "beta", "f32_recall", "f32_qps", "adc_rerank", "adc_qps", "degrad", "avg_deg");
    println!("{:-<82}", "");

    // f32 鍩虹嚎 recall锛埼?0 鐨勫浘锛岀敤浜庡姣旈噺鍖栭€€鍖栵級
    let mut f32_baseline_recall = 0.0f64;

    for &beta in &betas {
        let mut rng = ChaCha8Rng::seed_from(42);

        let t0 = Instant::now();
        let graph = if beta == 0.0 {
            println!("[尾={:.1}] 寤哄浘锛堟爣鍑?RobustPrune锛?..", beta);
            VamanaGraph::build(&train, dim, &build_config, &mut rng)
        } else {
            println!("[尾={:.1}] 寤哄浘锛堥噺鍖栨劅鐭?RobustPrune锛?..", beta);
            let qa_config = QuantAwarePruneConfig {
                alpha: 1.2,
                beta,
                epsilon: EPSILON,
                r_max: 32,
                normalization: NormalizationScheme::Mean,
            };
            // 鐢ㄩ璁＄畻鐨?node_errors锛孫(1) 鏌ヨ〃鏇夸唬 O(dim) encode+decode
            let ne = &node_errors;
            VamanaGraph::build_with_quant_aware_prune(
                &train, dim, &build_config, &qa_config,
                move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
                &mut rng,
            )
        };
        let build_time = t0.elapsed().as_secs_f64();
        println!("[尾={:.1}] 寤哄浘瀹屾垚: {:.1}s", beta, build_time);

        // f32 鎼滅储锛堟棤閲忓寲锛屾祴閲忓浘鏈韩璐ㄩ噺锛?
        let (f32_recall, f32_qps) = eval_f32(
            &train, &test, &gt, dim, nq, gt_stride, &graph, ef_search, k,
        );

        if beta == 0.0 {
            f32_baseline_recall = f32_recall;
        }

        // ADC + rerank 鎼滅储
        let (adc_recall, adc_qps, avg_deg) = eval_adc_rerank(
            &cb, &train, &quantized_db, &test, &gt, dim, n, nq, gt_stride,
            &graph, ef_search, top_n, k,
        );

        let degrad = f32_baseline_recall - adc_recall;

        println!("{:>6.1} {:>10.4} {:>10.0} {:>12.4} {:>12.0} {:>10.4} {:>10.1}",
            beta, f32_recall, f32_qps, adc_recall, adc_qps, degrad, avg_deg);
        println!();
    }

    println!("=== 缁撹 ===");
    println!("尾=0 鍩虹嚎: f32 recall 鈫?AVQ ADC+rerank recall锛堥噺鍖栭€€鍖栵級");
    println!("尾>0: 閲忓寲鎰熺煡鍓灊鏄惁鍑忓皬閫€鍖栵紵");
    println!("鍒ゆ柇鏍囧噯: recall 鎻愬崌 > 0.5% 涓?QPS 涓嬮檷 < 5% 鈫?尾 鏈夋晥");
}
