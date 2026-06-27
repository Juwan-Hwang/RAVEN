//! M3 楠岃瘉瀹為獙锛歴ample_neighbor_pairs O(n虏) 鈫?闅忔満閲囨牱
//!
//! 楠岃瘉鍐呭锛?//! 1. PreSampledNeighborPairs 妯″紡锛堣皟鐢?sample_neighbor_pairs锛夎兘姝ｅ父璁粌
//! 2. 璁粌鏃堕棿鍚堢悊锛圡3 鍚?O(n*1000*dim)锛岄潪 O(n虏*dim)锛?//! 3. AVQ 璐ㄩ噺锛坮econstruction_loss, retrieval_aware_loss锛変笌 BatchHighScorePairs 瀵规瘮
//!
//! 鑻?PreSampledNeighborPairs 璁粌鏃堕棿鍚堢悊涓旇川閲忎笉鍔ｄ簬 BatchHighScorePairs锛屽垯 M3 鏄紭鍖?
use std::fs::File;
use std::io::Read;
use std::time::Instant;
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
    println!("=== M3 楠岃瘉瀹為獙锛歴ample_neighbor_pairs 浼樺寲 ===");
    println!();

    // 1. 鍔犺浇 siftsmall
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    println!("siftsmall: dim={}, n={}", dim, n);

    // 褰掍竴鍖栵紙SIFT 0-255 鈫?[0,1]锛?    for v in train.iter_mut() { *v /= 255.0; }

    // 2. BatchHighScorePairs锛堥粯璁わ紝baseline锛?    println!();
    println!("=== A. BatchHighScorePairs锛堥粯璁?baseline锛?==");
    let t0 = Instant::now();
    let mut rng_a = ChaCha8Rng::seed_from(42);
    let cb_a = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, rng_a.inner(),
    );
    let time_a = t0.elapsed().as_secs_f64();
    println!("璁粌鏃堕棿: {:.2}s", time_a);
    println!("high_score_pairs: {}", cb_a.high_score_pairs.len());

    // 3. PreSampledNeighborPairs锛圡3 浼樺寲鍚庯級
    println!();
    println!("=== B. PreSampledNeighborPairs锛圡3 浼樺寲鍚庯級===");
    let t0 = Instant::now();
    let mut rng_b = ChaCha8Rng::seed_from(42);
    let cb_b = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::PreSampledNeighborPairs, 5, 8, 0.30, rng_b.inner(),
    );
    let time_b = t0.elapsed().as_secs_f64();
    println!("璁粌鏃堕棿: {:.2}s", time_b);
    println!("high_score_pairs: {}", cb_b.high_score_pairs.len());

    // 4. 瀵规瘮 AVQ 璐ㄩ噺
    println!();
    println!("=== AVQ 璐ㄩ噺瀵规瘮 ===");

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
    println!("  宸紓: {:+.4} ({:+.1}%)", recon_b - recon_a, (recon_b - recon_a) / recon_a * 100.0);

    // retrieval_aware_loss
    let ret_a = cb_a.retrieval_aware_loss(&train);
    let ret_b = cb_b.retrieval_aware_loss(&train);
    println!("retrieval_aware_loss:");
    println!("  A. BatchHighScorePairs: {:.4}", ret_a);
    println!("  B. PreSampledNeighborPairs: {:.4}", ret_b);
    println!("  宸紓: {:+.4} ({:+.1}%)", ret_b - ret_a, (ret_b - ret_a) / ret_a * 100.0);

    // 5. 姹囨€?    println!();
    println!("=== 姹囨€?===");
    println!("{:<25} {:>10} {:>12} {:>12}", "鏂规", "璁粌鏃堕棿", "recon_loss", "ret_loss");
    println!("{:-<59}", "");
    println!("{:<25} {:>10.2} {:>12.4} {:>12.4}", "A. BatchHighScorePairs", time_a, recon_a, ret_a);
    println!("{:<25} {:>10.2} {:>12.4} {:>12.4}", "B. PreSampledNeighborPairs", time_b, recon_b, ret_b);
    println!();

    // 鍒ゅ畾
    let time_diff_pct = (time_b - time_a) / time_a * 100.0;
    println!("璁粌鏃堕棿宸紓: {:+.1}%", time_diff_pct);
    println!();

    if time_b < time_a * 1.5 && recon_b <= recon_a * 1.05 {
        println!("缁撹: M3 浼樺寲鏈夋晥锛孭reSampledNeighborPairs 璁粌鏃堕棿鍚堢悊锛?);
        println!("     AVQ 璐ㄩ噺涓嶅姡浜?BatchHighScorePairs锛坮econ_loss 宸紓 <5%锛?);
    } else if time_b >= time_a * 1.5 {
        println!("缁撹: M3 浼樺寲鍚庤缁冩椂闂翠粛杈冮暱锛岄渶瑕佽繘涓€姝ヤ紭鍖?);
    } else {
        println!("缁撹: M3 浼樺寲鍚?AVQ 璐ㄩ噺涓嬮檷锛岄渶瑕佹鏌?sample_neighbor_pairs 瀹炵幇");
    }
}
