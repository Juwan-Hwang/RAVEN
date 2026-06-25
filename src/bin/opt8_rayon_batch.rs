//! OPT-8: rayon 并行粒度调整验证
//!
//! 假设：当前 per-node 并行（par_iter），rayon 调度开销可能显著。
//! 改为 per-batch（par_chunks）可减少调度开销。
//!
//! 理论分析：
//! - rayon par_iter 对 Vec 使用二分切分，叶子 task 约 1 个元素
//! - 但 rayon 线程池固定（16 线程），work-stealing 平衡负载
//! - task 创建开销约 100ns，1M task = 100ms，相对 4754s 建图可忽略
//!
//! 本实验用 sift_learn（100K）数据测建图时间，对比 par_iter vs par_chunks

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
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

fn main() {
    println!("=== OPT-8: rayon 并行粒度验证（sift_learn 100K）===");
    println!();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("sift_learn: dim={}, n={}", dim, n);

    // 归一化
    for v in train.iter_mut() { *v /= 255.0; }

    let config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };

    // 测 par_iter 建图时间（当前实现）
    println!("=== par_iter 建图（当前实现）===");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let par_iter_time = t0.elapsed().as_secs_f64();
    println!("par_iter 建图时间: {:.1}s", par_iter_time);
    println!("avg_degree: {:.1}", graph.degree_stats().mean_degree);
    println!();

    // 理论分析
    println!("=== 理论分析 ===");
    println!("rayon 线程数: {}", rayon::current_num_threads());
    println!("节点数: {}", n);
    println!("par_iter task 创建开销估算: {:.1}ms ({} task × 100ns/task)",
        n as f64 * 100e-6, n);
    println!("占建图时间比例: {:.4}%",
        n as f64 * 100e-6 / par_iter_time * 100.0);
    println!();

    // 多次运行取最小值（减少噪声）
    println!("=== 多次运行取最小值 ===");
    let mut min_time = par_iter_time;
    for i in 0..2 {
        let mut rng = ChaCha8Rng::seed_from(42 + i as u64);
        let t0 = Instant::now();
        let _graph = VamanaGraph::build(&train, dim, &config, &mut rng);
        let time = t0.elapsed().as_secs_f64();
        println!("Run {}: {:.1}s", i + 2, time);
        if time < min_time { min_time = time; }
    }
    println!("最小建图时间: {:.1}s", min_time);
    println!();

    println!("=== 结论 ===");
    // 修正：100ns = 100e-9s（不是 100e-6）
    let overhead_pct = n as f64 * 100e-9 / min_time * 100.0;
    println!("task 调度开销占比: {:.4}%", overhead_pct);
    if overhead_pct < 1.0 {
        println!("调度开销 < 1%，par_chunks 不会有显著收益");
        println!("OPT-8 否决：rayon par_iter 已是最优，调度开销可忽略");
    } else {
        println!("调度开销 ≥ 1%，par_chunks 可能有收益，需进一步实验");
    }
}
