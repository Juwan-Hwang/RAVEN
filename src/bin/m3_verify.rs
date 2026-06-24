//! M3 验证实验：sample_neighbor_pairs O(n²) → 随机采样
//!
//! 验证内容：
//! 1. PreSampledNeighborPairs 模式（调用 sample_neighbor_pairs）能正常训练
//! 2. 训练时间合理（M3 后 O(n*1000*dim)，非 O(n²*dim)）
//! 3. AVQ 质量（reconstruction_loss, retrieval_aware_loss）与 BatchHighScorePairs 对比
//!
//! 若 PreSampledNeighborPairs 训练时间合理且质量不劣于 BatchHighScorePairs，则 M3 是优化

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
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

fn main() {
    println!("=== M3 验证实验：sample_neighbor_pairs 优化 ===");
    println!();

    // 1. 加载 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    println!("siftsmall: dim={}, n={}", dim, n);

    // 归一化（SIFT 0-255 → [0,1]）
    for v in train.iter_mut() { *v /= 255.0; }

    // 2. BatchHighScorePairs（默认，baseline）
    println!();
    println!("=== A. BatchHighScorePairs（默认 baseline）===");
    let t0 = Instant::now();
    let mut rng_a = ChaCha8Rng::seed_from(42);
    let cb_a = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, rng_a.inner(),
    );
    let time_a = t0.elapsed().as_secs_f64();
    println!("训练时间: {:.2}s", time_a);
    println!("high_score_pairs: {}", cb_a.high_score_pairs.len());

    // 3. PreSampledNeighborPairs（M3 优化后）
    println!();
    println!("=== B. PreSampledNeighborPairs（M3 优化后）===");
    let t0 = Instant::now();
    let mut rng_b = ChaCha8Rng::seed_from(42);
    let cb_b = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::PreSampledNeighborPairs, 5, 8, 0.30, rng_b.inner(),
    );
    let time_b = t0.elapsed().as_secs_f64();
    println!("训练时间: {:.2}s", time_b);
    println!("high_score_pairs: {}", cb_b.high_score_pairs.len());

    // 4. 对比 AVQ 质量
    println!();
    println!("=== AVQ 质量对比 ===");

    // reconstruction_loss
    let recon_a = AVQCodebook::reconstruction_loss(
        &cb_a.centers, &train, dim, cb_a.m, cb_a.k, cb_a.sub_dim,
    );
    let recon_b = AVQCodebook::reconstruction_loss(
        &cb_b.centers, &train, dim, cb_b.m, cb_b.k, cb_b.sub_dim,
    );
    println!("reconstruction_loss:");
    println!("  A. BatchHighScorePairs: {:.4}", recon_a);
    println!("  B. PreSampledNeighborPairs: {:.4}", recon_b);
    println!("  差异: {:+.4} ({:+.1}%)", recon_b - recon_a, (recon_b - recon_a) / recon_a * 100.0);

    // retrieval_aware_loss
    let ret_a = cb_a.retrieval_aware_loss(&train);
    let ret_b = cb_b.retrieval_aware_loss(&train);
    println!("retrieval_aware_loss:");
    println!("  A. BatchHighScorePairs: {:.4}", ret_a);
    println!("  B. PreSampledNeighborPairs: {:.4}", ret_b);
    println!("  差异: {:+.4} ({:+.1}%)", ret_b - ret_a, (ret_b - ret_a) / ret_a * 100.0);

    // 5. 汇总
    println!();
    println!("=== 汇总 ===");
    println!("{:<25} {:>10} {:>12} {:>12}", "方案", "训练时间", "recon_loss", "ret_loss");
    println!("{:-<59}", "");
    println!("{:<25} {:>10.2} {:>12.4} {:>12.4}", "A. BatchHighScorePairs", time_a, recon_a, ret_a);
    println!("{:<25} {:>10.2} {:>12.4} {:>12.4}", "B. PreSampledNeighborPairs", time_b, recon_b, ret_b);
    println!();

    // 判定
    let time_diff_pct = (time_b - time_a) / time_a * 100.0;
    println!("训练时间差异: {:+.1}%", time_diff_pct);
    println!();

    if time_b < time_a * 1.5 && recon_b <= recon_a * 1.05 {
        println!("结论: M3 优化有效，PreSampledNeighborPairs 训练时间合理，");
        println!("     AVQ 质量不劣于 BatchHighScorePairs（recon_loss 差异 <5%）");
    } else if time_b >= time_a * 1.5 {
        println!("结论: M3 优化后训练时间仍较长，需要进一步优化");
    } else {
        println!("结论: M3 优化后 AVQ 质量下降，需要检查 sample_neighbor_pairs 实现");
    }
}
