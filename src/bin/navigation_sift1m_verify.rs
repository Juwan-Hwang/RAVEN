//! NavigationLayer SIFT1M 集成验证实验
//!
//! 验证 NavigationLayer 集成到 GraphSearcher 后在 SIFT1M 上的性能
//! 对比：
//!   A. GraphSearcher::new（默认 medoid entry_point）
//!   B. GraphSearcher::new_with_navigation（centroid entry_point）
//!
//! 指标：recall@10, QPS, avg_latency
//! SIFT1M: √N=1000 个 centroid，找最近 centroid 开销 O(1000*128)=128K flops/query

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, NavigationLayer, NavigationConfig};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");
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
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");
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

fn main() {
    println!("=== NavigationLayer SIFT1M 集成验证 ===");
    println!();

    // 1. 加载 SIFT1M
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("数据加载: {:.1}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);

    // 归一化
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;
    let gt_stride = gt_k;

    // 2. 构建 VamanaGraph
    println!();
    println!("=== 构建 VamanaGraph（α=1.2, r_max=32, l_build=100, max_iter=2）===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图时间: {:.1}s ({:.0} vec/s)", build_time, n as f64 / build_time);
    println!("entry_point (medoid): {}", graph.entry_point());

    // 3. 构建 NavigationLayer（centroid overlay, √N=1000）
    println!();
    println!("=== 构建 NavigationLayer（centroid overlay, √N={}）===", (n as f64).sqrt() as usize);
    let t0 = Instant::now();
    let nav_config = NavigationConfig {
        enable_centroid_overlay: true,
        centroid_count: None, // √N
    };
    let nav = NavigationLayer::new(n, &train, dim, nav_config);
    let nav_build_time = t0.elapsed().as_secs_f64();
    println!("NavigationLayer 构建: {:.1}s", nav_build_time);
    println!("centroid 数量: {}", nav.centroids().len());

    // 4. A. 默认 medoid entry
    println!();
    println!("=== A. GraphSearcher::new（默认 medoid entry）===");
    let t0 = Instant::now();
    let mut searcher_a = GraphSearcher::new(&train, &graph, ef_search);
    let mut hits_a = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher_a.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits_a += 1;
            }
        }
    }
    let time_a = t0.elapsed().as_secs_f64();
    let recall_a = hits_a as f64 / (nq * k) as f64;
    let qps_a = nq as f64 / time_a;
    println!("recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        recall_a, qps_a, time_a * 1000.0 / nq as f64);

    // 5. B. NavigationLayer centroid entry
    println!();
    println!("=== B. GraphSearcher::new_with_navigation（centroid entry）===");
    let t0 = Instant::now();
    let mut searcher_b = GraphSearcher::new_with_navigation(&train, &graph, ef_search, &nav);
    let mut hits_b = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher_b.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits_b += 1;
            }
        }
    }
    let time_b = t0.elapsed().as_secs_f64();
    let recall_b = hits_b as f64 / (nq * k) as f64;
    let qps_b = nq as f64 / time_b;
    println!("recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        recall_b, qps_b, time_b * 1000.0 / nq as f64);

    // 6. 汇总
    println!();
    println!("=== 汇总 ===");
    println!("{:<25} {:>10} {:>10} {:>12}", "方案", "recall@10", "QPS", "latency_ms");
    println!("{:-<57}", "");
    println!("{:<25} {:>10.4} {:>10.0} {:>12.3}", "A. medoid entry", recall_a, qps_a, time_a * 1000.0 / nq as f64);
    println!("{:<25} {:>10.4} {:>10.0} {:>12.3}", "B. centroid entry", recall_b, qps_b, time_b * 1000.0 / nq as f64);
    println!();

    let recall_diff = recall_b - recall_a;
    let qps_diff_pct = (qps_b - qps_a) / qps_a * 100.0;
    println!("差异: recall {:+.4}, QPS {:+.1}%", recall_diff, qps_diff_pct);
    println!();

    // 判定
    if recall_b >= recall_a - 0.001 && qps_b >= qps_a * 1.02 {
        println!("结论: NavigationLayer 在 SIFT1M 上有正向收益，集成成功");
    } else if recall_b >= recall_a - 0.001 && qps_b >= qps_a * 0.98 {
        println!("结论: NavigationLayer 在 SIFT1M 上无明显收益（QPS 差异 <2%），但 recall 不劣");
    } else if recall_b < recall_a - 0.001 {
        println!("结论: NavigationLayer 在 SIFT1M 上 recall 下降，不应集成");
    } else {
        println!("结论: NavigationLayer 在 SIFT1M 上 QPS 下降（centroid 查找开销 > visited 减少），不应集成");
    }
}
