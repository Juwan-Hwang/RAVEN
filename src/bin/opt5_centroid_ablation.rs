//! OPT-5: centroid overlay 数量优化验证（siftsmall）
//!
//! SIFT1M 上 centroid overlay（√N=1000）仅提升 QPS 0.5%（已有日志）。
//! OPT-5 假设：减少 centroid 数量可能有更大收益（降低找最近 centroid 的开销）。
//!
//! 本实验用 siftsmall 快速验证不同 centroid_count 的效果。
//! 反效果预警：如果所有组合 QPS 提升 < 2%，标记为"已验证无显著收益"。

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, NavigationLayer, NavigationConfig};
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

fn eval(
    train: &[f32], test: &[f32], gt: &[i32],
    dim: usize, nq: usize, gt_stride: usize,
    graph: &VamanaGraph, ef_search: usize, k: usize,
    navigation: Option<&NavigationLayer>,
) -> (f64, f64) {
    let mut searcher = if let Some(nav) = navigation {
        GraphSearcher::new_with_navigation(train, graph, ef_search, nav)
    } else {
        GraphSearcher::new(train, graph, ef_search)
    };
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
    (hits as f64 / (nq * k) as f64, nq as f64 / elapsed)
}

fn main() {
    println!("=== OPT-5: centroid overlay 数量优化（siftsmall）===");
    println!();

    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let gt_stride = gt_k;
    let k = 10;
    let ef_search = 100;

    // 建图（只建一次）
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.0, l_build: 100, r_soft: 48, r_max: 32, max_iterations: 2,
    };
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println!("建图: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // 基线：medoid entry（无 navigation）
    let (recall_base, qps_base) = eval(
        &train, &test, &gt, dim, nq, gt_stride, &graph, ef_search, k, None,
    );
    println!("基线 (medoid entry): recall={:.4}, QPS={:.0}", recall_base, qps_base);
    println!();

    // 扫描 centroid_count
    let centroid_counts = [10usize, 50, 100];  // siftsmall √N=100
    println!("{:>12} {:>10} {:>10} {:>10}", "centroid_n", "recall@10", "QPS", "qps_delta");
    println!("{:-<46}", "");

    for &count in &centroid_counts {
        let t0 = Instant::now();
        let nav_config = NavigationConfig {
            enable_centroid_overlay: true,
            centroid_count: Some(count),
        };
        let nav = NavigationLayer::new(n, &train, dim, nav_config);
        let nav_time = t0.elapsed().as_secs_f64();

        let (recall, qps) = eval(
            &train, &test, &gt, dim, nq, gt_stride, &graph, ef_search, k, Some(&nav),
        );
        let delta = (qps - qps_base) / qps_base * 100.0;
        println!("{:>12} {:>10.4} {:>10.0} {:>8.1}%  (nav_build={:.2}s, centroids={})",
            count, recall, qps, delta, nav_time, nav.centroids().len());
    }

    println!();
    println!("=== 结论 ===");
    println!("SIFT1M 已有数据: centroid overlay (√N=1000) QPS +0.5%");
    println!("siftsmall 验证: 不同 centroid_count 的 QPS 变化");
    println!("反效果预警: 如果 QPS 提升 < 2%，标记为'已验证无显著收益'");
}
