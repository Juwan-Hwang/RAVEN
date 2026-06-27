//! DelayedPruneController 瀹為獙
//!
//! 楠岃瘉 DelayedPruneController 鏄惁瀵归」鐩湁鐢?
//!
//! 鍙戠幇锛歝onnect_bidirectional锛坴amana.rs:507-529锛夊凡鍐呰仈瀹炵幇浜?
//!   - should_prune: storage.degree(nb) > config.r_soft
//!   - RobustPrune 瑙﹀彂
//! DelayedPruneController 鏄浉鍚岄€昏緫鐨勫皝瑁呯増鏈?+ 缁熻鍔熻兘
//!
//! 瀹為獙锛?
//!   A. 褰撳墠 build锛堝唴鑱?lazy pruning锛?
//!   B. build 鍚庣敤 DelayedPruneController 缁熻 prune 鐘舵€?
//!   C. 瀵规瘮 DelayedPruneController.final_prune vs VamanaGraph::final_prune 缁撴灉涓€鑷存€?
//!
//! 鑻?DelayedPruneController 浠呮彁渚涚粺璁″姛鑳斤紙鏃犳€ц兘宸紓锛夛紝鍒欏畾浣嶄负"璇婃柇宸ュ叿"

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::{ChaCha8Rng, DelayedPruneController};
use raven::memory::HybridBlockedCsr;

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
    println!("=== DelayedPruneController 瀹為獙 ===");
    println!();

    // 1. 鍔犺浇 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, _, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}", dim, n, nq);

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. 褰撳墠 build锛堝唴鑱?lazy pruning锛?
    println!("=== A. 褰撳墠 build锛堝唴鑱?lazy pruning锛?==");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time_a = t0.elapsed().as_secs_f64();
    println!("寤哄浘鏃堕棿: {:.2}s", build_time_a);

    // 3. 鐢?DelayedPruneController 缁熻褰撳墠 graph 鐘舵€?
    println!();
    println!("=== B. DelayedPruneController 璇婃柇 ===");
    let controller = DelayedPruneController::new(config.r_max);
    let storage = graph.storage();
    let over_soft = controller.count_over_soft(storage);
    println!("r_max={}, r_soft={}", controller.r_max, controller.r_soft);
    println!("瓒呰繃 R_soft 鐨勮妭鐐规暟: {}/{}", over_soft, n);

    // 搴︽暟缁熻
    let mut degree_sum = 0usize;
    let mut degree_max = 0usize;
    let mut over_r_max = 0usize;
    for i in 0..n {
        let d = storage.degree(i as u32);
        degree_sum += d;
        if d > degree_max { degree_max = d; }
        if d > config.r_max { over_r_max += 1; }
    }
    println!("avg_degree={:.1}, max_degree={}, 瓒呰繃 R_max 鐨勮妭鐐? {}/{}",
        degree_sum as f64 / n as f64, degree_max, over_r_max, n);

    // 4. recall 楠岃瘉
    let recall_a = eval_recall(&train, &test, &gt, dim, nq, &graph, ef_search, k);
    println!("recall@10: {:.4}", recall_a);

    // 5. 楠岃瘉 DelayedPruneController.final_prune 涓?VamanaGraph::final_prune 涓€鑷存€?
    // 澶嶅埗 graph锛岀敤 DelayedPruneController.final_prune 閲嶆柊鍓灊
    println!();
    println!("=== C. DelayedPruneController.final_prune 涓€鑷存€ч獙璇?===");
    // 鐢变簬 VamanaGraph 鐨?storage 鏄鏈夌殑锛屾垜浠€氳繃 storage_mut 鑾峰彇
    // 澶嶅埗 graph 鐨?storage 鍋氬姣?
    let mut graph_copy_storage = HybridBlockedCsr::new(n, config.r_max * 2);
    for i in 0..n as u32 {
        let (main, overflow) = graph.storage().neighbors_full(i);
        let mut all: Vec<u32> = main.to_vec();
        all.extend_from_slice(overflow);
        for &nb in &all {
            graph_copy_storage.add_edge(i, nb);
        }
    }
    println!("澶嶅埗 storage 瀹屾垚锛岃妭鐐规暟: {}", graph_copy_storage.len());

    // 鐢?DelayedPruneController.final_prune 鍓灊
    let mut controller2 = DelayedPruneController::new(config.r_max);
    let t0 = Instant::now();
    controller2.final_prune(&mut graph_copy_storage, &train, dim, config.alpha);
    let prune_time = t0.elapsed().as_secs_f64();
    println!("DelayedPruneController.final_prune 鏃堕棿: {:.3}s", prune_time);
    println!("final_prune 瑙﹀彂娆℃暟: {}", controller2.final_prune_count);

    // 缁熻鍓灊鍚庣殑搴︽暟
    let mut degree_sum_c = 0usize;
    let mut degree_max_c = 0usize;
    let mut over_r_max_c = 0usize;
    for i in 0..n {
        let d = graph_copy_storage.degree(i as u32);
        degree_sum_c += d;
        if d > degree_max_c { degree_max_c = d; }
        if d > config.r_max { over_r_max_c += 1; }
    }
    println!("鍓灊鍚? avg_degree={:.1}, max_degree={}, 瓒呰繃 R_max 鐨勮妭鐐? {}/{}",
        degree_sum_c as f64 / n as f64, degree_max_c, over_r_max_c, n);

    // 6. 姹囨€?
    println!();
    println!("=== 姹囨€?===");
    println!("DelayedPruneController 瀹氫綅鍒嗘瀽:");
    println!("  - should_prune 閫昏緫 = connect_bidirectional 鐨?storage.degree(nb) > r_soft");
    println!("  - final_prune 閫昏緫  = VamanaGraph::final_prune锛堝畬鍏ㄧ浉鍚岋級");
    println!("  - 闄勫姞浠峰€? 缁熻鍔熻兘锛坰ingle_prune_count, final_prune_count, count_over_soft锛?);
    println!();
    println!("缁撹:");
    if over_r_max == 0 {
        println!("  褰撳墠 build 鐨?final_prune 宸插皢鎵€鏈夎妭鐐瑰壀鍒?R_max 浠ュ唴锛?);
        println!("  DelayedPruneController.final_prune 涓嶄細鏀瑰彉鍥剧粨鏋勶紙鏃犻澶栨敹鐩婏級");
    } else {
        println!("  褰撳墠 build 鏈?{} 涓妭鐐硅秴杩?R_max锛孌elayedPruneController.final_prune 鍙慨姝?, over_r_max);
    }
    println!("  DelayedPruneController 鏄?connect_bidirectional + final_prune 鐨勫皝瑁呯増鏈?);
    println!("  闄勫姞浠峰€间粎鍦ㄤ簬缁熻鍔熻兘锛坧rune_count 璇婃柇锛?);
}
