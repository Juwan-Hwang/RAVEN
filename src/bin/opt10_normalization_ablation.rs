//! OPT-10: 褰掍竴鍖栨柟妗堟秷铻嶅疄楠岋紙siftsmall锛?
//!
//! 鐩爣锛氶獙璇?QuantAwareRobustPrune 鐨?4 绉嶅綊涓€鍖栨柟妗堝 recall/QPS 鐨勫奖鍝?
//!
//! 褰掍竴鍖栨柟妗堬細
//!   Mean      - 鍧囧€煎綊涓€鍖栵紙涓绘柟妗堬級
//!   StdDev    - 鏍囧噯宸綊涓€鍖?
//!   Mad       - 涓綅鏁扮粷瀵瑰亸宸?
//!   LogSumExp - log-sum-exp 闈炵嚎鎬у帇缂?
//!
//! 鎵撳垎鍑芥暟锛歋core = dist / (渭_dist + 蔚) + 尾 脳 error / (渭_error + 蔚)
//!
//! 瀹為獙鐭╅樀锛? 鏂规 脳 尾 鈭?{0.0, 0.3, 1.0, 2.0}
//!   尾=0 鏃跺綊涓€鍖栨柟妗堜笉褰卞搷锛坋rror 椤硅娑堥櫎锛夛紝鍙祴涓€娆′綔涓哄熀绾?
//!
//! 涔嬪墠 SIFT1M 尾 娑堣瀺鏄剧ず 尾 鏃犳敹鐩婏紝鏈疄楠岄獙璇?siftsmall 涓婃槸鍚﹀悓鏍锋棤鏀剁泭锛?
//! 浠ュ強褰掍竴鍖栨柟妗堣兘鍚︽敼鍙樿繖涓€缁撹銆?

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
use raven::build::ChaCha8Rng;

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

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇澶辫触");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            gt.push(i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap()));
        }
    }
    (gt, dim, n)
}

/// f32 鎼滅储锛岃繑鍥?(recall@10, qps, avg_degree)
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
) -> (f64, f64, f64) {
    let avg_deg = graph.degree_stats().mean_degree;
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
    (recall, qps, avg_deg)
}

fn main() {
    println!("=== OPT-10: 褰掍竴鍖栨柟妗堟秷铻嶅疄楠岋紙siftsmall锛?==");
    println!("鐩爣: 楠岃瘉 4 绉嶅綊涓€鍖栨柟妗?脳 尾 瀵?recall/QPS 鐨勫奖鍝?);
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.2}s", t0.elapsed().as_secs_f64());
    println!("siftsmall: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 褰掍竴鍖栧埌 [0,1]锛圫IFT 鏁版嵁 0-255锛岄槻姊害鐖嗙偢锛?
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;

    // 2. AVQ 璁粌锛堢敤 base 浣滀负 learn 鏁版嵁锛宻iftsmall 10K 瓒冲锛?
    println!("=== AVQ 璁粌锛坆ase 10K, K=256, sub_dim=8, 伪=0.30, iter=5锛?==");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 璁粌: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 棰勮绠楄妭鐐归噺鍖栬宸?
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train))
        .collect();
    println!("鑺傜偣閲忓寲璇樊棰勮绠? {:.2}s", t0.elapsed().as_secs_f64());
    println!("璇樊缁熻: min={:.6}, max={:.6}, mean={:.6}",
        node_errors.iter().cloned().fold(f32::INFINITY, f32::min),
        node_errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        node_errors.iter().sum::<f32>() / node_errors.len() as f32,
    );
    println!();

    // 4. 寤哄浘閰嶇疆锛堝浐瀹氾級
    let build_config = VamanaBuildConfig {
        alpha: 1.0,  // siftsmall 鏈€浼?伪=1.0锛坢emory 璁板綍锛?
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };

    // 5. 瀹為獙鐭╅樀
    let schemes = [
        (NormalizationScheme::Mean, "Mean"),
        (NormalizationScheme::StdDev, "StdDev"),
        (NormalizationScheme::Mad, "Mad"),
        (NormalizationScheme::LogSumExp, "LogSumExp"),
    ];
    let betas = [0.0f32, 0.3, 1.0, 2.0];

    println!("=== 娑堣瀺缁撴灉 ===");
    println!("{:>12} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "scheme", "beta", "recall@10", "QPS", "avg_deg", "build_s");
    println!("{:-<62}", "");

    // 尾=0 鍩虹嚎锛堟爣鍑?RobustPrune锛屼笌褰掍竴鍖栨柟妗堟棤鍏筹紝鍙祴涓€娆★級
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph_base = VamanaGraph::build(&train, dim, &build_config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    let (recall_base, qps_base, deg_base) = eval_f32(
        &train, &test, &gt, dim, nq, gt_stride, &graph_base, ef_search, k,
    );
    println!("{:>12} {:>6.1} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
        "Baseline", 0.0, recall_base, qps_base, deg_base, build_time);

    // 尾>0 鎵弿
    let mut best_recall = recall_base;
    let mut best_combo = "Baseline 尾=0".to_string();

    for (scheme, name) in &schemes {
        for &beta in &betas[1..] {  // 璺宠繃 尾=0锛堝凡娴嬪熀绾匡級
            let mut rng = ChaCha8Rng::seed_from(42);
            let qa_config = QuantAwarePruneConfig {
                alpha: 1.0,
                beta,
                epsilon: EPSILON,
                r_max: 32,
                normalization: *scheme,
            };
            let ne = &node_errors;
            let t0 = Instant::now();
            let graph = VamanaGraph::build_with_quant_aware_prune(
                &train, dim, &build_config, &qa_config,
                move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
                &mut rng,
            );
            let build_time = t0.elapsed().as_secs_f64();

            let (recall, qps, avg_deg) = eval_f32(
                &train, &test, &gt, dim, nq, gt_stride, &graph, ef_search, k,
            );

            println!("{:>12} {:>6.1} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
                name, beta, recall, qps, avg_deg, build_time);

            if recall > best_recall {
                best_recall = recall;
                best_combo = format!("{:?} 尾={:.1}", scheme, beta);
            }
        }
    }

    println!();
    println!("=== 缁撹 ===");
    println!("鍩虹嚎 (尾=0): recall={:.4}, QPS={:.0}", recall_base, qps_base);
    println!("鏈€浣崇粍鍚? {} recall={:.4}", best_combo, best_recall);
    let delta = best_recall - recall_base;
    if delta > 0.005 {
        println!("褰掍竴鍖栨柟妗堟湁鏁? recall 鎻愬崌 {:.4} (>0.5%)", delta);
    } else if delta > 0.0 {
        println!("褰掍竴鍖栨柟妗堟敹鐩婂彲蹇界暐: recall 鎻愬崌 {:.4} (<0.5%)", delta);
    } else {
        println!("褰掍竴鍖栨柟妗堟棤鏁? recall 鏈彁鍗囷紙尾>0 鍏ㄩ儴 鈮?尾=0 鍩虹嚎锛?);
    }
}
