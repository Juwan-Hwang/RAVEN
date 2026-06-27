//! OPT-1: r_max / ef_search Pareto 鎵弿
//!
//! 瀵规瘡涓?r_max 寤哄浘涓€娆★紝鐒跺悗鎵弿澶氫釜 ef_search 鍊硷紝缁樺埗 Pareto 鍓嶆部銆?
//!
//! 鐢ㄦ硶锛歝argo run --release --bin pareto_scan
//!
//! 杈撳嚭锛欳SV 鏍煎紡锛屽彲鐩存帴瀵煎叆缁樺浘宸ュ叿

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

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

fn recall_at_k(found: &[u32], gt_slice: &[i32], k: usize) -> f64 {
    let mut hits = 0usize;
    for &g in gt_slice.iter().take(k) {
        if found.contains(&(g as u32)) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

fn main() {
    println!("=== OPT-1: r_max / ef_search Pareto 鎵弿 ===");
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.2}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }

    // 鍙傛暟
    let r_max_values: Vec<usize> = vec![24, 32, 40, 48, 56, 64];
    let ef_search_values: Vec<usize> = vec![50, 100, 200, 400, 800];
    let k = 10;
    let gt_stride = gt_k;

    // CSV header
    println!("r_max,ef_search,recall@10,QPS,avg_latency_ms,avg_degree,build_time_s");

    for &r_max in &r_max_values {
        // 2. 寤哄浘
        let r_soft = (r_max as f32 * 1.5) as usize;
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 200,
            r_soft,
            r_max,
            max_iterations: 2,
..Default::default()
        };
        let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
        let build_time = t0.elapsed().as_secs_f64();

        // 璁＄畻骞冲潎搴︽暟
        let n_nodes = train.len() / dim;
        let mut total_degree = 0u64;
        let mut degrees = Vec::with_capacity(n_nodes);
        for node in 0..n_nodes as u32 {
            let deg = graph.storage().neighbors(node).len();
            total_degree += deg as u64;
            degrees.push(deg);
        }
        degrees.sort();
        let avg_degree = total_degree as f64 / n_nodes as f64;
        let p99_degree = degrees[(n_nodes as f64 * 0.99) as usize];

        println!("--- r_max={} 寤哄浘瀹屾垚: {:.1}s, avg_degree={:.1}, p99_degree={} ---",
            r_max, build_time, avg_degree, p99_degree);

        // 3. 鎵弿 ef_search
        for &ef_search in &ef_search_values {
            let mut searcher = GraphSearcher::new(&train, &graph, ef_search);

            // 棰勭儹
            for q in 0..100.min(nq) {
                let query = &test[q * dim..(q + 1) * dim];
                let _ = searcher.search(query, k);
            }

            // 姝ｅ紡娴嬮噺
            let t0 = Instant::now();
            let mut recall_sum = 0.0f64;
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let result = searcher.search(query, k);
                let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
                let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
                recall_sum += recall_at_k(&found, gt_slice, k);
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let recall = recall_sum / nq as f64;
            let qps = nq as f64 / elapsed;
            let avg_latency = elapsed * 1000.0 / nq as f64;

            println!("{},{},{:.4},{:.0},{:.3},{:.1},{:.1}",
                r_max, ef_search, recall, qps, avg_latency, avg_degree, build_time);
        }

        // 閲婃斁 graph 鍐呭瓨锛堟樉寮?drop锛?
        drop(graph);
        println!();
    }

    println!("=== Pareto 鎵弿瀹屾垚 ===");
}
