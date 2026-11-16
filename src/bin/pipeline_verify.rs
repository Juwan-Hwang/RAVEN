//! Pipeline 楠岃瘉瀹為獙
//!
//! 楠岃瘉 S1-S5 淇鍚?pipeline 鑳芥甯稿伐浣?//! 璺?beta=0.0 鍜?beta=0.3 涓ょ閰嶇疆锛屽姣?recall@10
//! - beta=0.0锛歲uant_aware_prune 璺宠繃锛堟爣鍑?RobustPrune锛?//! - beta=0.3锛歲uant_aware_prune 鐪熸鎵ц锛圫1 淇楠岃瘉锛?
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::build::{BuildConfig, BuildPipeline};
use raven::graph::{VamanaGraph, GraphSearcher};

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
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

fn eval_recall(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    k: usize,
) -> f32 {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let gt_stride = 100;
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
    hits as f32 / (nq * k) as f32
}

fn main() {
    println!("=== Pipeline 楠岃瘉瀹為獙锛圫1-S5 淇鍚庯級===");
    println!("楠岃瘉 pipeline 鑳芥甯稿伐浣滐紝beta=0.0 鍜?beta=0.3 涓ょ閰嶇疆");
    println!();

    // 1. 鍔犺浇 siftsmall 鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, _, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.1}s", t0.elapsed().as_secs_f64());
    println!("siftsmall: dim={}, base={}, query={}", dim, n, nq);
    println!();

    // 褰掍竴鍖栧埌 [0,1]锛堣璁℃枃妗ｏ細SIFT 鏁版嵁 0-255 鑼冨洿浼氬鑷存搴︾垎鐐革級
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. 璺?pipeline锛坆eta=0.0锛氭爣鍑?RobustPrune锛宷uant_aware_prune 璺宠繃锛?    println!("=== Pipeline beta=0.0锛堟爣鍑?RobustPrune锛?==");
    let config0 = BuildConfig {
        beta: 0.0,
        r_max: 32,
        r_soft: 48,
        l_build: 100,
        ..Default::default()
    };
    let pipeline0 = BuildPipeline::new(config0);
    let t0 = Instant::now();
    let result0 = pipeline0.run(train.clone(), dim);
    println!("Pipeline beta=0.0 鏋勫缓: {:.1}s", t0.elapsed().as_secs_f64());

    // 鐢ㄨ繑鍥炵殑 opq 鏃嬭浆 train 鍜?test
    let opq0 = result0.opq.as_ref().expect("opq should be trained");
    let train_rot0 = opq0.apply(&train, dim);
    let test_rot0 = opq0.apply(&test, dim);

    let recall0 = eval_recall(&train_rot0, &test_rot0, &gt, dim, nq, &result0.graph, ef_search, k);
    println!("recall@10: {:.4}", recall0);
    println!("alpha_variants: {}", result0.alpha_variants.len());
    println!("final_stage: {:?}", result0.final_stage);
    println!();

    // 3. 璺?pipeline锛坆eta=0.3锛歲uant_aware_prune 鐪熸鎵ц锛孲1 淇楠岃瘉锛?    println!("=== Pipeline beta=0.3锛堥噺鍖栨劅鐭?RobustPrune锛孲1 淇楠岃瘉锛?==");
    let config3 = BuildConfig {
        beta: 0.3,
        r_max: 32,
        r_soft: 48,
        l_build: 100,
        ..Default::default()
    };
    let pipeline3 = BuildPipeline::new(config3);
    let t0 = Instant::now();
    let result3 = pipeline3.run(train.clone(), dim);
    println!("Pipeline beta=0.3 鏋勫缓: {:.1}s", t0.elapsed().as_secs_f64());

    let opq3 = result3.opq.as_ref().expect("opq should be trained");
    let train_rot3 = opq3.apply(&train, dim);
    let test_rot3 = opq3.apply(&test, dim);

    let recall3 = eval_recall(&train_rot3, &test_rot3, &gt, dim, nq, &result3.graph, ef_search, k);
    println!("recall@10: {:.4}", recall3);
    println!("alpha_variants: {}", result3.alpha_variants.len());
    println!("final_stage: {:?}", result3.final_stage);
    println!();

    // 4. 姹囨€?    println!("=== 姹囨€?===");
    println!("beta=0.0锛堟爣鍑?RobustPrune锛? recall={:.4}", recall0);
    println!("beta=0.3锛堥噺鍖栨劅鐭?RobustPrune锛? recall={:.4}", recall3);
    println!();

    if recall0 > 0.9 && recall3 > 0.9 {
        println!("PASS: S1-S5 淇鍚?pipeline 姝ｅ父宸ヤ綔锛屼袱涓厤缃?recall 閮藉悎鐞?);
    } else {
        println!("FAIL: recall 寮傚父锛岄渶瑕佹鏌?S1-S5 淇");
    }
}
