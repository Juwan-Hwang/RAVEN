//! 图构建微基准（divan）
//!
//! 设计文档第六层：
//! 建图速度与吞吐基准

use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;

fn main() {
    divan::main();
}

#[divan::bench_group(name = "graph_build", max_time = Duration::from_secs(5))]
mod graph_benches {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| (i as f32).sin()).collect()
    }

    #[divan::bench(args = [(100, 64), (500, 128), (1000, 256)])]
    fn vamana_build(bencher: divan::Bencher, (n, dim): (usize, usize)) {
        let vectors = make_vectors(n, dim);
        bencher.bench_local(move || {
            let mut rng = ChaCha8Rng::new();
            let config = VamanaBuildConfig {
                alpha: 1.2,
                l_build: 100,
                r_max: 32,
                r_soft: 48,
                max_iterations: 1,
            };
            divan::black_box(VamanaGraph::build(&vectors, dim, &config, &mut rng));
        });
    }
}

use std::time::Duration;
