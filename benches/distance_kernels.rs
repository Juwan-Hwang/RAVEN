//! 距离核微基准（divan）
//!
//! 设计文档第六层：
//! 日常迭代用 divan（轻量、上手快）
//! 正式实验与回归用 criterion（统计驱动，适合参数对比和写实验图表）
//!
//! 以下决策须以微基准结果为准：
//! - chunks_exact(8) 的向量化效果
//! - aligned vs unaligned load
//! - parking_lot vs std::sync
//! - GEMM 阈值
//! - AVX-512 的端到端 QPS（含持续稳定性）

use raven::distance::{is_avx2_supported, l2_avx2, l2_dynamic, l2_scalar};

fn main() {
    // 设计文档：divan 日常迭代基准
    divan::main();
}

#[divan::bench_group(name = "l2_distance", max_time = Duration::from_secs(3))]
mod l2_benches {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| i as f32).collect()
    }

    #[divan::bench(args = [64, 128, 256, 768, 960, 1536])]
    fn l2_dynamic_bench(bencher: divan::Bencher, dim: usize) {
        let a = make_vectors(1, dim);
        let b = make_vectors(1, dim);
        bencher.bench_local(move || divan::black_box(l2_dynamic(&a, &b)));
    }

    #[divan::bench(args = [64, 128, 256, 768, 960, 1536])]
    fn l2_scalar_bench(bencher: divan::Bencher, dim: usize) {
        let a = make_vectors(1, dim);
        let b = make_vectors(1, dim);
        bencher.bench_local(move || divan::black_box(l2_scalar(&a, &b)));
    }

    #[divan::bench(args = [64, 128, 256, 768, 960, 1536])]
    fn l2_avx2_bench(bencher: divan::Bencher, dim: usize) {
        if !is_avx2_supported() {
            return;
        }
        let a = make_vectors(1, dim);
        let b = make_vectors(1, dim);
        bencher.bench_local(move || {
            let d = unsafe { l2_avx2(&a, &b) };
            divan::black_box(d)
        });
    }
}

use std::time::Duration;
