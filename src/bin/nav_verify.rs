//! v9 导航层验证：取 SIFT1M 前 10K 节点建小图，暴力算 ground truth，对比 avg_visited
//!
//! 用法：cargo run --release --bin nav_verify

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::distance::l2_simd;

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
            vectors.push(f32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (vectors, dim, n)
}

fn main() {
    println!("=== v9 导航层 avg_visited 验证（10K 子集）===");

    let (mut train_all, dim, _n_all) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test_all, _, _nq_all) = read_fvecs("data/sift/sift_query.fvecs");

    // 归一化
    for v in train_all.iter_mut() { *v /= 255.0; }
    for v in test_all.iter_mut() { *v /= 255.0; }

    // 取前 10000 个训练节点
    let n = 10_000usize;
    let train: Vec<f32> = train_all[..n * dim].to_vec();

    // 取前 200 个查询
    let nq = 200usize;
    let test: Vec<f32> = test_all[..nq * dim].to_vec();
    let k = 10usize;

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);

    // 暴力计算 ground truth（在前 10K 节点中）
    println!("计算暴力 ground truth...");
    let mut gt: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let mut dists: Vec<(f32, u32)> = (0..n as u32)
            .map(|i| {
                let v = &train[i as usize * dim..(i as usize + 1) * dim];
                (l2_simd(query, v), i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        gt.push(dists[..k].iter().map(|(_, id)| *id).collect());
    }
    println!("ground truth 计算完毕");

    // === A: 无分层导航（baseline flat Vamana）===
    println!("\n--- A: flat Vamana（无分层导航）---");
    {
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 100,
            r_max: 32,
            r_soft: 48,
            max_iterations: 2,
            saturate: true,
            enable_layered_nav: false,
            nav_m: 16,
            ..Default::default()
        };
        let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
        println!("建图: {:.1}s", t0.elapsed().as_secs_f64());

        for &ef in &[50usize, 100, 200] {
            let mut searcher = GraphSearcher::new(&train, &graph, ef);
            let mut hits = 0;
            let mut total = 0;
            let mut total_visited = 0;
            let t0 = Instant::now();
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let result = searcher.search(query, k);
                total_visited += searcher.last_visited_count();
                let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
                for &g in &gt[q] {
                    if found.contains(&g) { hits += 1; }
                }
                total += k;
            }
            let dt = t0.elapsed().as_secs_f64();
            let recall = hits as f64 / total as f64;
            let qps = nq as f64 / dt;
            let avg_vis = total_visited as f64 / nq as f64;
            println!("  ef={:>3}  recall={:.4}  QPS={:>6.0}  avg_visited={:.1}", ef, recall, qps, avg_vis);
        }
    }

    // === B: v9 分层导航 ===
    println!("\n--- B: v9 分层导航（独立上层图 + 双向 RobustPrune + 多入口）---");
    {
        let t0 = Instant::now();
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 100,
            r_max: 32,
            r_soft: 48,
            max_iterations: 2,
            saturate: true,
            enable_layered_nav: true,
            nav_m: 16,
            ..Default::default()
        };
        let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
        println!("建图: {:.1}s", t0.elapsed().as_secs_f64());

        if let Some(nav) = graph.layered_nav() {
            println!("  max_level={}", nav.max_level());
        }

        for &ef in &[50usize, 100, 200] {
            let mut searcher = GraphSearcher::new(&train, &graph, ef);
            let mut hits = 0;
            let mut total = 0;
            let mut total_visited = 0;
            let t0 = Instant::now();
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let result = searcher.search(query, k);
                total_visited += searcher.last_visited_count();
                let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
                for &g in &gt[q] {
                    if found.contains(&g) { hits += 1; }
                }
                total += k;
            }
            let dt = t0.elapsed().as_secs_f64();
            let recall = hits as f64 / total as f64;
            let qps = nq as f64 / dt;
            let avg_vis = total_visited as f64 / nq as f64;
            println!("  ef={:>3}  recall={:.4}  QPS={:>6.0}  avg_visited={:.1}", ef, recall, qps, avg_vis);
        }
    }

    println!("\n=== 验证完毕 ===");
}
