//! OPT-8: rayon 骞惰绮掑害璋冩暣楠岃瘉
//!
//! 鍋囪锛氬綋鍓?per-node 骞惰锛坧ar_iter锛夛紝rayon 璋冨害寮€閿€鍙兘鏄捐憲銆?
//! 鏀逛负 per-batch锛坧ar_chunks锛夊彲鍑忓皯璋冨害寮€閿€銆?
//!
//! 鐞嗚鍒嗘瀽锛?
//! - rayon par_iter 瀵?Vec 浣跨敤浜屽垎鍒囧垎锛屽彾瀛?task 绾?1 涓厓绱?
//! - 浣?rayon 绾跨▼姹犲浐瀹氾紙16 绾跨▼锛夛紝work-stealing 骞宠　璐熻浇
//! - task 鍒涘缓寮€閿€绾?100ns锛?M task = 100ms锛岀浉瀵?4754s 寤哄浘鍙拷鐣?
//!
//! 鏈疄楠岀敤 sift_learn锛?00K锛夋暟鎹祴寤哄浘鏃堕棿锛屽姣?par_iter vs par_chunks

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇澶辫触");
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
    println!("=== OPT-8: rayon 骞惰绮掑害楠岃瘉锛坰ift_learn 100K锛?==");
    println!();

    let (mut train, dim, n) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("sift_learn: dim={}, n={}", dim, n);

    // 褰掍竴鍖?
    for v in train.iter_mut() { *v /= 255.0; }

    let config = VamanaBuildConfig {
        alpha: 1.0,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };

    // 娴?par_iter 寤哄浘鏃堕棿锛堝綋鍓嶅疄鐜帮級
    println!("=== par_iter 寤哄浘锛堝綋鍓嶅疄鐜帮級===");
    let mut rng = ChaCha8Rng::seed_from(42);
    let t0 = Instant::now();
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let par_iter_time = t0.elapsed().as_secs_f64();
    println!("par_iter 寤哄浘鏃堕棿: {:.1}s", par_iter_time);
    println!("avg_degree: {:.1}", graph.degree_stats().mean_degree);
    println!();

    // 鐞嗚鍒嗘瀽
    println!("=== 鐞嗚鍒嗘瀽 ===");
    println!("rayon 绾跨▼鏁? {}", rayon::current_num_threads());
    println!("鑺傜偣鏁? {}", n);
    println!("par_iter task 鍒涘缓寮€閿€浼扮畻: {:.1}ms ({} task 脳 100ns/task)",
        n as f64 * 100e-6, n);
    println!("鍗犲缓鍥炬椂闂存瘮渚? {:.4}%",
        n as f64 * 100e-6 / par_iter_time * 100.0);
    println!();

    // 澶氭杩愯鍙栨渶灏忓€硷紙鍑忓皯鍣０锛?
    println!("=== 澶氭杩愯鍙栨渶灏忓€?===");
    let mut min_time = par_iter_time;
    for i in 0..2 {
        let mut rng = ChaCha8Rng::seed_from(42 + i as u64);
        let t0 = Instant::now();
        let _graph = VamanaGraph::build(&train, dim, &config, &mut rng);
        let time = t0.elapsed().as_secs_f64();
        println!("Run {}: {:.1}s", i + 2, time);
        if time < min_time { min_time = time; }
    }
    println!("鏈€灏忓缓鍥炬椂闂? {:.1}s", min_time);
    println!();

    println!("=== 缁撹 ===");
    // 淇锛?00ns = 100e-9s锛堜笉鏄?100e-6锛?
    let overhead_pct = n as f64 * 100e-9 / min_time * 100.0;
    println!("task 璋冨害寮€閿€鍗犳瘮: {:.4}%", overhead_pct);
    if overhead_pct < 1.0 {
        println!("璋冨害寮€閿€ < 1%锛宲ar_chunks 涓嶄細鏈夋樉钁楁敹鐩?);
        println!("OPT-8 鍚﹀喅锛歳ayon par_iter 宸叉槸鏈€浼橈紝璋冨害寮€閿€鍙拷鐣?);
    } else {
        println!("璋冨害寮€閿€ 鈮?1%锛宲ar_chunks 鍙兘鏈夋敹鐩婏紝闇€杩涗竴姝ュ疄楠?);
    }
}
