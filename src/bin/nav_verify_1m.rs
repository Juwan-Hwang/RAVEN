//! v9 导航层 SIFT1M 验证：只跑 ef=50，快速对比 avg_visited
//!
//! 用法：cargo run --release --bin nav_verify_1m

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
            vectors.push(f32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
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
            gt.push(i32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

fn main() {
    println!("=== v9 导航层 SIFT1M 验证 ===");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    println!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;
    let ef = 50usize;

    // GLASS-COMP 配置（200/32/2），与之前基线对齐
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 48,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: true,
        nav_m: 16,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("建图: {:.1}s", build_time);

    let stats = graph.degree_stats();
    println!(
        "[degree] mean={:.1} p95={} p99={} max={} isolated={}",
        stats.mean_degree, stats.p95_degree, stats.p99_degree,
        stats.max_degree, stats.isolated_nodes
    );

    if let Some(nav) = graph.layered_nav() {
        println!("[nav] max_level={}", nav.max_level());
    }

    // 搜索
    let mut searcher = GraphSearcher::new(&train, &graph, ef);
    let mut hits = 0;
    let mut total = 0;
    let mut total_visited = 0;
    let mut visited_counts: Vec<usize> = Vec::with_capacity(nq);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let vc = searcher.last_visited_count();
        total_visited += vc;
        visited_counts.push(vc);

        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_k..q * gt_k + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) { hits += 1; }
        }
        total += k;
    }
    let query_time = t0.elapsed();

    let recall = hits as f64 / total as f64;
    let qps = nq as f64 / query_time.as_secs_f64();
    let avg_visited = total_visited as f64 / nq as f64;

    visited_counts.sort_unstable();
    let p50 = visited_counts[nq / 2];
    let p95 = visited_counts[(nq as f64 * 0.95) as usize];
    let p99 = visited_counts[(nq as f64 * 0.99) as usize];
    let max_vc = visited_counts[nq - 1];

    println!("\n=== GLASS-COMP (200/32/2) + v9 导航 ===");
    println!("ef=50  recall={:.4}  QPS={:.0}  avg_visited={:.1}", recall, qps, avg_visited);
    println!("p50={}  p95={}  p99={}  max={}", p50, p95, p99, max_vc);

    println!("\n=== 对比基线（GLASS-COMP v3）===");
    println!("ef=50  recall=0.9703  QPS=7,501  avg_visited=1,399.5");
    println!("\n=== 对比 Glass HNSW ===");
    println!("avg_visited=1041 (H20 实测)");

    if avg_visited < 300.0 {
        println!("\n✅ avg_visited < 300，分层导航有效！");
    } else if avg_visited < 1000.0 {
        println!("\n⚠️ avg_visited < 1000 但 > 300，有改善但不够");
    } else {
        println!("\n❌ avg_visited >= 1000，分层导航未生效");
    }
}
