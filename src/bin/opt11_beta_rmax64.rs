//! OPT-11: 尾 鍋囪鍦?r_max=64 涓婇噸鏂伴獙璇侊紙siftsmall 蹇€熺増锛?
//!
//! OPT-10 宸插湪 siftsmall r_max=32 涓婅瘉鏄?尾 鏃犳敹鐩娿€?
//! OPT-11 鍋囪锛毼?鍙兘鍦?r_max=64 鐨勯珮璐ㄩ噺鍥句笂鎵嶇敓鏁堛€?
//!
//! 鏈疄楠岀敤 siftsmall + r_max=64 蹇€熼獙璇侊細
//! - 濡傛灉 尾>0 浠嶇劧 鈮?尾=0 鍩虹嚎锛岃鏄?尾 鍋囪鍦?SIFT 鏁版嵁涓婁笉鎴愮珛
//! - 濡傛灉 尾>0 > 尾=0 鍩虹嚎锛岄渶瑕佸湪 SIFT1M 涓婂畬鏁撮獙璇?
//!
//! 娉ㄦ剰锛歴iftsmall 10K 鑺傜偣 + r_max=64 鍙兘鍑虹幇 recall 澶╄姳鏉挎晥搴旓紙recall=1.0锛夛紝
//! 姝ゆ椂闄嶄綆 ef_search 璁?recall 鏈夋彁鍗囩┖闂淬€?

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

fn eval_f32(
    train: &[f32], test: &[f32], gt: &[i32],
    dim: usize, nq: usize, gt_stride: usize,
    graph: &VamanaGraph, ef_search: usize, k: usize,
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
            if found.contains(&(g as u32)) { hits += 1; }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (hits as f64 / (nq * k) as f64, nq as f64 / elapsed, avg_deg)
}

fn main() {
    println!("=== OPT-11: 尾 鍋囪 r_max=64 楠岃瘉锛坰iftsmall锛?==");
    println!();

    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.2}s", t0.elapsed().as_secs_f64());
    println!("siftsmall: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);
    println!();

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let gt_stride = gt_k;
    let k = 10;

    // AVQ 璁粌
    println!("=== AVQ 璁粌 ===");
    let t0 = Instant::now();
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ 璁粌: {:.2}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train))
        .collect();
    println!("鑺傜偣閲忓寲璇樊: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // r_max=64 寤哄浘閰嶇疆
    let build_config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 64,  // OPT-11 鏍稿績鍙橀噺
        max_iterations: 2,
..Default::default()
    };

    // 鍏堟祴 尾=0 鍩虹嚎锛岀敤澶氫釜 ef_search 鎵惧埌闈炲ぉ鑺辨澘 recall
    println!("=== 尾=0 鍩虹嚎锛坮_max=64锛?==");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph_base = VamanaGraph::build(&train, dim, &build_config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("寤哄浘: {:.2}s", build_time);

    let ef_values = [20, 50, 100];
    println!("{:>6} {:>10} {:>10} {:>10}", "ef", "recall@10", "QPS", "avg_deg");
    println!("{:-<40}", "");
    for &ef in &ef_values {
        let (recall, qps, deg) = eval_f32(&train, &test, &gt, dim, nq, gt_stride, &graph_base, ef, k);
        println!("{:>6} {:>10.4} {:>10.0} {:>10.1}", ef, recall, qps, deg);
    }
    println!();

    // 閫変竴涓?recall 闈炲ぉ鑺辨澘鐨?ef_search 鏉ユ祴 尾
    // ef=20 recall 搴旇 < 1.0锛屾湁鎻愬崌绌洪棿
    let ef_test = 20;
    let (recall_base, qps_base, deg_base) = eval_f32(&train, &test, &gt, dim, nq, gt_stride, &graph_base, ef_test, k);
    println!("閫夊畾 ef={} 娴?尾锛氬熀绾?recall={:.4}", ef_test, recall_base);
    println!();

    // 鎵弿 尾
    let betas = [0.05f32, 0.1, 0.3, 0.5, 1.0, 2.0];
    println!("=== 尾 鎵弿锛坮_max=64, ef={}锛?==", ef_test);
    println!("{:>6} {:>10} {:>10} {:>10} {:>10}", "beta", "recall@10", "QPS", "avg_deg", "build_s");
    println!("{:-<50}", "");
    println!("{:>6.2} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
        0.0, recall_base, qps_base, deg_base, build_time);

    let mut best_recall = recall_base;
    let mut best_beta = 0.0f32;

    for &beta in &betas {
        let mut rng = ChaCha8Rng::seed_from(42);
        let qa_config = QuantAwarePruneConfig {
            alpha: 1.0,
            beta,
            epsilon: EPSILON,
            r_max: 64,
            normalization: NormalizationScheme::Mean,
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
            &train, &test, &gt, dim, nq, gt_stride, &graph, ef_test, k,
        );

        println!("{:>6.2} {:>10.4} {:>10.0} {:>10.1} {:>10.2}",
            beta, recall, qps, avg_deg, build_time);

        if recall > best_recall {
            best_recall = recall;
            best_beta = beta;
        }
    }

    println!();
    println!("=== 缁撹 ===");
    println!("r_max=64, ef={}: 尾=0 鍩虹嚎 recall={:.4}", ef_test, recall_base);
    println!("鏈€浣? 尾={:.2} recall={:.4}", best_beta, best_recall);
    let delta = best_recall - recall_base;
    if delta > 0.005 {
        println!("尾 鍋囪鍙兘鎴愮珛: recall 鎻愬崌 {:.4} (>0.5%)锛岄渶 SIFT1M 瀹屾暣楠岃瘉", delta);
    } else if delta > 0.0 {
        println!("尾 鏀剁泭鍙拷鐣? recall 鎻愬崌 {:.4} (<0.5%)", delta);
    } else {
        println!("尾 鍋囪涓嶆垚绔? recall 鏈彁鍗囷紙尾>0 鍏ㄩ儴 鈮?尾=0 鍩虹嚎锛?);
    }
}
