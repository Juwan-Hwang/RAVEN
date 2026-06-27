//! 闄勫綍 B锛欰VQ 璁粌淇″彿鍙屽垎鏀秷铻嶅疄楠?
//!
//! 璁捐鏂囨。闄勫綍 B锛欰VQ 璁粌淇″彿鐨勫叿浣撳疄鐜帮紙Week 5-6 鍐崇瓥锛?
//! 涓や釜閫夐」閮藉仛鏈€灏忕増鏈紝涓嶆彁鍓嶆媿鏉匡細
//!   閫夐」涓€锛氭壒娆″唴楂樺垎瀵归噰鏍凤紙100琛屼互鍐咃級
//!   閫夐」浜岋細棰勯噰鏍疯繎閭诲锛堝鐢ㄥ凡鏈夌矖閲忓寲鏋勫缓锛?
//!
//! 瀵规瘮鎸囨爣锛?
//!   1. 鍚岀瓑璁粌鏃堕棿涓嬬殑 retrieval-aware loss
//!   2. 涓嬫父 recall@10 宸紓
//!
//! Week 5 鏈湅鏁版嵁锛屽摢涓?recall 鏇撮珮閫夊摢涓紝鍚屾椂鍦ㄨ鏂囬噷鎶ュ憡涓や釜缁撴灉銆?

use std::time::Instant;
use rand::Rng;
use raven::quant::avq::{AVQCodebook, QuantizationMode, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

/// 鐢熸垚甯﹁仛绫荤粨鏋勭殑鍚堟垚鏁版嵁闆?
fn generate_data(n: usize, dim: usize, nq: usize, k: usize, n_clusters: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from(seed);
    let mut train = vec![0.0f32; n * dim];
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);

    for _ in 0..n_clusters {
        let c: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0).collect();
        centroids.push(c);
    }

    for i in 0..n {
        let cluster = i % n_clusters;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            train[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    let mut test = vec![0.0f32; nq * dim];
    for i in 0..nq {
        let cluster = (i % n_clusters) as usize;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            test[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    let mut gt = vec![0i32; nq * k];
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &train[i * dim..(i + 1) * dim];
                let d: f32 = v.iter().zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for j in 0..k {
            gt[q * k + j] = dists[j].0 as i32;
        }
    }

    (train, test, gt)
}

/// 璁＄畻閲忓寲鍚庣殑 recall@10锛堢敤閲忓寲鍚戦噺閲嶅缓鍥撅級
fn quantized_recall(
    codebook: &AVQCodebook,
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
) -> (f64, f64) {
    // 閲忓寲鎵€鏈夎缁冨悜閲?
    let quantized: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 鐢ㄩ噺鍖栧悜閲忔瀯寤哄浘
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_max: 32,
        r_soft: 48,
        max_iterations: 1,
..Default::default()
    };
    let graph = VamanaGraph::build(&quantized, dim, &config, &mut rng);

    // 鐢ㄩ噺鍖栧悜閲忔煡璇?
    let mut searcher = GraphSearcher::new(&quantized, &graph, 100);
    let start = Instant::now();
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * k..(q + 1) * k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    let elapsed = start.elapsed();
    let qps = nq as f64 / elapsed.as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    (recall, qps)
}

fn main() {
    println!("=== 闄勫綍 B锛欰VQ 璁粌淇″彿鍙屽垎鏀秷铻嶅疄楠?===");
    println!("璁捐鏂囨。闄勫綍 B锛歐eek 5 鍚屾椂瀹炵幇鏈€灏忓弻鍒嗘敮锛岀敤娑堣瀺鏁版嵁鍐冲畾涓荤嚎");
    println!();

    // 3 涓暟鎹泦
    let datasets = [
        ("dataset_1", 500usize, 64usize, 50usize, 10usize, 10),
        ("dataset_2", 1000, 128, 100, 10, 20),
        ("dataset_3", 2000, 128, 100, 10, 30),
    ];

    for (name, n, dim, nq, k, n_clusters) in &datasets {
        println!("=== {} (n={}, dim={}, nq={}, k={}, clusters={}) ===", name, n, dim, nq, k, n_clusters);
        let (train, test, gt) = generate_data(*n, *dim, *nq, *k, *n_clusters, 42);

        // 閫夐」涓€锛氭壒娆″唴楂樺垎瀵归噰鏍?
        let t1_start = Instant::now();
        let cb1 = AVQCodebook::train_with_signal(&train, *dim, 16, QuantizationMode::Avq, TrainingSignal::BatchHighScorePairs);
        let t1_train = t1_start.elapsed();
        let loss1 = cb1.retrieval_aware_loss(&train);
        let (recall1, qps1) = quantized_recall(&cb1, &train, &test, &gt, *dim, *n, *nq, *k);

        // 閫夐」浜岋細棰勯噰鏍疯繎閭诲
        let t2_start = Instant::now();
        let cb2 = AVQCodebook::train_with_signal(&train, *dim, 16, QuantizationMode::Avq, TrainingSignal::PreSampledNeighborPairs);
        let t2_train = t2_start.elapsed();
        let loss2 = cb2.retrieval_aware_loss(&train);
        let (recall2, qps2) = quantized_recall(&cb2, &train, &test, &gt, *dim, *n, *nq, *k);

        println!("[閫夐」涓€] BatchHighScorePairs:");
        println!("  train_time={:.3}s, retrieval_loss={:.6}, recall@{}={:.4}, QPS={:.0}",
            t1_train.as_secs_f64(), loss1, k, recall1, qps1);
        println!("[閫夐」浜宂 PreSampledNeighborPairs:");
        println!("  train_time={:.3}s, retrieval_loss={:.6}, recall@{}={:.4}, QPS={:.0}",
            t2_train.as_secs_f64(), loss2, k, recall2, qps2);

        // 瀵规瘮
        let loss_ratio = if loss2 > 0.0 { loss1 / loss2 } else { 1.0 };
        let recall_diff = recall1 - recall2;
        println!("[瀵规瘮]");
        println!("  loss ratio (option1/option2): {:.3}x ({})", loss_ratio,
            if loss_ratio < 1.0 { "option1 鏇翠紭" } else { "option2 鏇翠紭" });
        println!("  recall diff (option1 - option2): {:+.4} ({})", recall_diff,
            if recall_diff > 0.0 { "option1 鏇翠紭" } else if recall_diff < 0.0 { "option2 鏇翠紭" } else { "鎸佸钩" });
        println!();
    }

    println!("=== 鏈€缁堝喅绛栧缓璁?===");
    println!("鐪?recall 鏇撮珮鐨勯€夐」浣滀负涓荤嚎锛屽悓鏃跺湪璁烘枃閲屾姤鍛婁袱涓粨鏋?);
}
