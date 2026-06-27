//! AVQ 璁粌纭畾鎬ч獙璇?//!
//! 楠岃瘉淇 rand::random() 鈫?ChaCha8Rng 鍚庯紝AVQ 璁粌缁撴灉鏄惁纭畾鎬?//! 璺戜袱娆＄浉鍚屽弬鏁扮殑 AVQ 璁粌锛屽姣?recon loss 鏄惁瀹屽叏鐩稿悓

use std::fs::File;
use std::io::Read;
use raven::quant::opq::OPQRotation;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");
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
    println!("=== AVQ 璁粌纭畾鎬ч獙璇?===");
    println!("楠岃瘉淇 rand::random() 鈫?ChaCha8Rng 鍚庯紝涓ゆ璁粌缁撴灉鏄惁瀹屽叏鐩稿悓");
    println!();

    let (mut learn, dim, n) = read_fvecs("data/sift/sift_learn.fvecs");
    for v in learn.iter_mut() { *v /= 255.0; }
    println!("learn: {} vecs, dim={}", n, dim);

    // OPQ 鏃嬭浆锛堢‘瀹氭€э紝鏃犻殢鏈烘€э級
    let opq = OPQRotation::train_with_sub_dim(&learn, dim, 8);
    let learn_rot = opq.apply(&learn, dim);

    // 绗竴娆¤缁?    println!("\n=== 绗竴娆?AVQ 璁粌 ===");
    let mut rng1 = ChaCha8Rng::seed_from(42);
    let cb1 = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30,
        rng1.inner(),
    );
    // 缂栫爜绗竴涓悜閲忥紝瀵规瘮
    let v0 = &learn_rot[0..dim];
    let enc1 = cb1.encode(v0);
    let dec1 = cb1.decode(&enc1);
    let err1 = (0..dim).map(|i| (dec1[i] - v0[i]).powi(2)).sum::<f32>().sqrt();
    println!("绗竴娆? enc[0]={:?}, decode_error={:.6}", &enc1[0..4.min(enc1.len())], err1);

    // 绗簩娆¤缁冿紙鐩稿悓鍙傛暟锛?    println!("\n=== 绗簩娆?AVQ 璁粌锛堢浉鍚屽弬鏁帮級===");
    let mut rng2 = ChaCha8Rng::seed_from(42);
    let cb2 = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30,
        rng2.inner(),
    );
    let enc2 = cb2.encode(v0);
    let dec2 = cb2.decode(&enc2);
    let err2 = (0..dim).map(|i| (dec2[i] - v0[i]).powi(2)).sum::<f32>().sqrt();
    println!("绗簩娆? enc[0]={:?}, decode_error={:.6}", &enc2[0..4.min(enc2.len())], err2);

    // 瀵规瘮
    println!("\n=== 瀵规瘮 ===");
    let enc_match = enc1 == enc2;
    let err_match = (err1 - err2).abs() < 1e-10;
    println!("encode 鐩稿悓: {}", enc_match);
    println!("decode_error 鐩稿悓: {}", err_match);
    println!("error diff: {:.10}", (err1 - err2).abs());

    if enc_match && err_match {
        println!("\nPASS: AVQ 璁粌纭畾鎬т慨澶嶆湁鏁堬紝涓ゆ缁撴灉瀹屽叏鐩稿悓");
    } else {
        println!("\nFAIL: AVQ 璁粌浠嶅瓨鍦ㄩ潪纭畾鎬?);
    }
}
