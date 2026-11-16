//! 快速验证：跳跃边是否在最终图中存活
//!
//! 用法：cargo run --release --bin jump_survive_check

use std::fs::File;
use std::io::Read;

use raven::build::ChaCha8Rng;
use raven::graph::{PruneStrategy, VamanaBuildConfig, VamanaGraph};

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

fn build_and_check(train: &[f32], dim: usize, n: usize, r_max: usize, l_build: usize, n_jumps: usize) {
    std::env::set_var("RAVEN_N_JUMPS", n_jumps.to_string());

    let mut rng = ChaCha8Rng::seed_from(42);
    let r_soft = (r_max as f32 * 1.5) as usize;
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build,
        r_max,
        r_soft,
        max_iterations: 2,
        saturate: false,
        enable_layered_nav: true,
        nav_m: 32,
        prune_strategy: PruneStrategy::DirectionalPrune,
        n_jumps,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();

    // 统计度数分布
    let mut total_degree = 0usize;
    let mut max_degree = 0usize;
    let mut degree_hist = [0usize; 41]; // 0..=40
    for node in 0..n {
        let deg = graph.neighbors(node as u32).len();
        total_degree += deg;
        if deg > max_degree { max_degree = deg; }
        let h_idx = deg.min(40);
        degree_hist[h_idx] += 1;
    }
    let avg_degree = total_degree as f64 / n as f64;

    println!("n_jumps={}: avg_degree={:.2}, max={}, build={:.1}s",
        n_jumps, avg_degree, max_degree, build_time);

    // 度数分布直方图
    for d in 0..=40 {
        if degree_hist[d] > 0 {
            print!("  d={:>2}: {} ({:.1}%)", d, degree_hist[d], degree_hist[d] as f64 / n as f64 * 100.0);
            if d % 4 == 0 { println!(); }
        }
    }
    println!();
}

fn main() {
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    for v in train.iter_mut() { *v /= 255.0; }

    let r_max = 40usize;
    let l_build = 384usize;

    println!("=== 跳跃边存活验证 ===");
    println!("n={}, dim={}, R={}, L={}\n", n, dim, r_max, l_build);

    build_and_check(&train, dim, n, r_max, l_build, 0);
    build_and_check(&train, dim, n, r_max, l_build, 4);
    build_and_check(&train, dim, n, r_max, l_build, 8);
}
