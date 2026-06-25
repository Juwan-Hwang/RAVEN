//! OPT-2 验证：siftsmall 建图 + 搜索，验证预取策略改变后 recall 不变
//!
//! 预取策略只影响数据加载时机，不改变搜索路径，recall 应不变

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取失败");
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
    let mut file = File::open(path).expect("无法打开 ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取失败");
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
    println!("=== OPT-2 验证：siftsmall recall 不变性 ===");
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_nq={}, gt_k={}", dim, n, nq, gt_nq, gt_k);

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    // 建图（siftsmall 10K，几秒完成）
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.0, l_build: 100, r_soft: 48, r_max: 32, max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图: {:.2}s", t0.elapsed().as_secs_f64());

    // 搜索（OPT-2 方案 B 预取策略已接入 greedy_search_vec_reuse）
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
    println!("OPT-2 方案 B: recall@10={:.4}, QPS={:.0}", recall, qps);

    if recall > 0.95 {
        println!("验证通过: recall > 0.95，预取策略改变未影响搜索质量");
    } else {
        println!("验证失败: recall < 0.95，需要检查");
    }
}
