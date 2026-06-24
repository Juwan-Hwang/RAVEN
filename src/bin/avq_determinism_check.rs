//! AVQ 训练确定性验证
//!
//! 验证修复 rand::random() → ChaCha8Rng 后，AVQ 训练结果是否确定性
//! 跑两次相同参数的 AVQ 训练，对比 recon loss 是否完全相同

use std::fs::File;
use std::io::Read;
use raven::quant::opq::OPQRotation;
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
    println!("=== AVQ 训练确定性验证 ===");
    println!("验证修复 rand::random() → ChaCha8Rng 后，两次训练结果是否完全相同");
    println!();

    let (mut learn, dim, n) = read_fvecs("data/sift/sift_learn.fvecs");
    for v in learn.iter_mut() { *v /= 255.0; }
    println!("learn: {} vecs, dim={}", n, dim);

    // OPQ 旋转（确定性，无随机性）
    let opq = OPQRotation::train_with_sub_dim(&learn, dim, 8);
    let learn_rot = opq.apply(&learn, dim);

    // 第一次训练
    println!("\n=== 第一次 AVQ 训练 ===");
    let mut rng1 = ChaCha8Rng::seed_from(42);
    let cb1 = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30,
        rng1.inner(),
    );
    // 编码第一个向量，对比
    let v0 = &learn_rot[0..dim];
    let enc1 = cb1.encode(v0);
    let dec1 = cb1.decode(&enc1);
    let err1 = (0..dim).map(|i| (dec1[i] - v0[i]).powi(2)).sum::<f32>().sqrt();
    println!("第一次: enc[0]={:?}, decode_error={:.6}", &enc1[0..4.min(enc1.len())], err1);

    // 第二次训练（相同参数）
    println!("\n=== 第二次 AVQ 训练（相同参数）===");
    let mut rng2 = ChaCha8Rng::seed_from(42);
    let cb2 = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30,
        rng2.inner(),
    );
    let enc2 = cb2.encode(v0);
    let dec2 = cb2.decode(&enc2);
    let err2 = (0..dim).map(|i| (dec2[i] - v0[i]).powi(2)).sum::<f32>().sqrt();
    println!("第二次: enc[0]={:?}, decode_error={:.6}", &enc2[0..4.min(enc2.len())], err2);

    // 对比
    println!("\n=== 对比 ===");
    let enc_match = enc1 == enc2;
    let err_match = (err1 - err2).abs() < 1e-10;
    println!("encode 相同: {}", enc_match);
    println!("decode_error 相同: {}", err_match);
    println!("error diff: {:.10}", (err1 - err2).abs());

    if enc_match && err_match {
        println!("\nPASS: AVQ 训练确定性修复有效，两次结果完全相同");
    } else {
        println!("\nFAIL: AVQ 训练仍存在非确定性");
    }
}
