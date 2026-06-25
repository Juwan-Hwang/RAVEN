//! 快速 recall 验证（建图 + 搜索，约 100s）
//! 用法：cargo run --release --bin quick_recall_check
//! 退出码 0 = recall OK, 1 = recall BAD

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
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
    let mut file = File::open(path).expect("open ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read ivecs");
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
    println!("=== Quick Recall Check (SIFT1M) ===");

    let (mut train, dim, _n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2, l_build: 100, r_soft: 48, r_max: 32, max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图: {:.1}s", t0.elapsed().as_secs_f64());

    let k = 10;
    let ef_search = 100;
    let mut searcher = GraphSearcher::new(&train, &graph, ef_search);

    let t0 = Instant::now();
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) { hits += 1; }
        }
        total += k;
    }
    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / t0.elapsed().as_secs_f64();
    println!("recall@10={:.4}, QPS={:.0}", recall, qps);

    if recall < 0.9 {
        println!("FAIL: recall < 0.9");
        std::process::exit(1);
    } else {
        println!("PASS: recall >= 0.9");
    }
}
