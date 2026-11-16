//! OPT-2 楠岃瘉锛歴iftsmall 寤哄浘 + 鎼滅储锛岄獙璇侀鍙栫瓥鐣ユ敼鍙樺悗 recall 涓嶅彉
//!
//! 棰勫彇绛栫暐鍙奖鍝嶆暟鎹姞杞芥椂鏈猴紝涓嶆敼鍙樻悳绱㈣矾寰勶紝recall 搴斾笉鍙?

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
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

fn main() {
    println!("=== OPT-2 楠岃瘉锛歴iftsmall recall 涓嶅彉鎬?===");
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    // 寤哄浘锛坰iftsmall 10K锛屽嚑绉掑畬鎴愶級
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.0, l_build: 100, r_soft: 48, r_max: 32, max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("寤哄浘: {:.2}s", t0.elapsed().as_secs_f64());

    // 鎼滅储锛圤PT-2 鏂规 B 棰勫彇绛栫暐宸叉帴鍏?greedy_search_vec_reuse锛?
    let mut searcher = GraphSearcher::new(&train, &graph, 100);
    let k = 10;
    let gt_stride = gt_k;
    let mut recall_sum = 0.0f64;
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        let mut hits = 0;
        for &g in gt_slice.iter().take(k) {
            if found.contains(&(g as u32)) { hits += 1; }
        }
        recall_sum += hits as f64 / k as f64;
    }
    let time = t0.elapsed().as_secs_f64();
    let recall = recall_sum / nq as f64;
    let qps = nq as f64 / time;
    println!("OPT-2 鏂规 B: recall@10={:.4}, QPS={:.0}", recall, qps);

    if recall > 0.95 {
        println!("楠岃瘉閫氳繃: recall > 0.95锛岄鍙栫瓥鐣ユ敼鍙樻湭褰卞搷鎼滅储璐ㄩ噺");
    } else {
        println!("楠岃瘉澶辫触: recall < 0.95锛岄渶瑕佹鏌?);
    }
}
