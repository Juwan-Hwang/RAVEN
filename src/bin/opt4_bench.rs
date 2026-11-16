//! OPT-4 寰熀鍑嗭細f32 vs f16 SIMD 璺濈璁＄畻鍚炲悙閲忓姣?//!
//! 鐩爣锛氶獙璇?f16 SIMD锛團16C + AVX-512锛夊湪 memory-bound 鍦烘櫙涓嬫槸鍚︽瘮 f32 AVX-512 蹇?//!
//! 鍏抽敭鐐癸細
//! - f16 甯﹀鍑忓崐锛?B vs 4B/鍏冪礌锛夛紝memory-bound 鍦烘櫙涓嬪簲鏈夋敹鐩?//! - 璁＄畻绮惧害淇濇寔 f32锛團16C 鍦ㄥ瘎瀛樺櫒涓浆 f32 鍚庤绠楋級
//!
//! 瀹為獙鏂规锛?//! - 鏂规 A锛歭2_simd锛坒32 AVX-512锛夆€?褰撳墠鍩虹嚎
//! - 鏂规 B锛歭2_f16_mixed_simd锛坒16 + F16C + AVX-512锛夆€?OPT-4
//! - 鏂规 C锛歭2_f16_mixed锛堟爣閲?f16锛夆€?瀵圭収
//!
//! 鏁版嵁锛歋IFT1M base锛?M 脳 dim=128 = 493MB f32 / 246MB f16锛?//! 闅忔満璁块棶妯″紡锛堟ā鎷熷浘鎼滅储鐑矾寰勶級锛屾暟鎹秴鍑?L3 cache

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::{l2_simd, l2_f16_mixed, l2_f16_mixed_simd, f32_to_f16_slice, is_avx512_and_f16c_supported};

/// 璇诲彇 fvecs 鏂囦欢
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 鏂囦欢闀垮害涓嶅榻?);

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
    println!("=== OPT-4 寰熀鍑嗭細f32 vs f16 SIMD 璺濈璁＄畻锛圫IFT1M memory-bound锛?==");
    println!();

    if !is_avx512_and_f16c_supported() {
        eprintln!("褰撳墠 CPU 涓嶆敮鎸?AVX-512 + F16C锛屾棤娉曟祴璇?SIMD f16");
        return;
    }
    println!("CPU 鏀寔: AVX-512 + F16C");
    println!();

    // 鍔犺浇 SIFT1M base锛?M 鍚戦噺锛?93MB f32锛?    println!("鍔犺浇 SIFT1M base 鏁版嵁...");
    let t0 = Instant::now();
    let (db_f32, dim, n_db) = read_fvecs("data/sift/sift_base.fvecs");
    println!("鍔犺浇瀹屾垚: {} vecs, dim={}, {:.1}s, {:.1}MB (f32)",
        n_db, dim, t0.elapsed().as_secs_f64(), db_f32.len() * 4 / 1024 / 1024);

    // 棰勯噺鍖栦负 f16锛?46MB锛?    let t0 = Instant::now();
    let db_f16 = f32_to_f16_slice(&db_f32);
    println!("f16 棰勯噺鍖? {:.1}s, {:.1}MB (f16, 鑺傜渷 50%)",
        t0.elapsed().as_secs_f64(), db_f16.len() * 2 / 1024 / 1024);
    println!();

    // 妯℃嫙鍥炬悳绱㈢殑闅忔満璁块棶妯″紡
    // 鐢ㄥ浐瀹氱殑浼殢鏈虹储寮曞簭鍒楋紝纭繚涓変釜鏂规璁块棶鐩稿悓鐨勮妭鐐?    let n_queries = 100_000;  // 10 涓囨璺濈璁＄畻
    let query_indices: Vec<usize> = (0..n_queries)
        .map(|i| (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) as usize % n_db)
        .collect();

    // 鍥哄畾鏌ヨ鍚戦噺
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) / dim as f32).collect();

    // 棰勭儹
    let mut warmup = 0.0f32;
    for &idx in query_indices.iter().take(1000) {
        warmup += l2_simd(&query, &db_f32[idx * dim..(idx + 1) * dim]);
    }
    println!("棰勭儹瀹屾垚 (sum={:.4})", warmup);
    println!();

    // 鏂规 A锛歠32 AVX-512锛堝熀绾匡級
    let t0 = Instant::now();
    let mut sum_a = 0.0f32;
    for &idx in &query_indices {
        sum_a += l2_simd(&query, &db_f32[idx * dim..(idx + 1) * dim]);
    }
    let time_a = t0.elapsed().as_secs_f64();
    let qps_a = n_queries as f64 / time_a;
    println!("鏂规 A: f32 AVX-512 (l2_simd)");
    println!("  鏃堕棿: {:.4}s, QPS={:.0}, avg_latency={:.2}us, sum={:.4}",
        time_a, qps_a, time_a * 1e6 / n_queries as f64, sum_a);
    println!();

    // 鏂规 B锛歠16 SIMD锛團16C + AVX-512锛?    let t0 = Instant::now();
    let mut sum_b = 0.0f32;
    for &idx in &query_indices {
        sum_b += l2_f16_mixed_simd(&query, &db_f16[idx * dim..(idx + 1) * dim]);
    }
    let time_b = t0.elapsed().as_secs_f64();
    let qps_b = n_queries as f64 / time_b;
    println!("鏂规 B: f16 SIMD (l2_f16_mixed_simd, F16C + AVX-512)");
    println!("  鏃堕棿: {:.4}s, QPS={:.0}, avg_latency={:.2}us, sum={:.4}",
        time_b, qps_b, time_b * 1e6 / n_queries as f64, sum_b);
    println!();

    // 鏂规 C锛歠16 鏍囬噺锛堝鐓э級
    let t0 = Instant::now();
    let mut sum_c = 0.0f32;
    for &idx in &query_indices {
        sum_c += l2_f16_mixed(&query, &db_f16[idx * dim..(idx + 1) * dim]);
    }
    let time_c = t0.elapsed().as_secs_f64();
    let qps_c = n_queries as f64 / time_c;
    println!("鏂规 C: f16 鏍囬噺 (l2_f16_mixed)");
    println!("  鏃堕棿: {:.4}s, QPS={:.0}, avg_latency={:.2}us, sum={:.4}",
        time_c, qps_c, time_c * 1e6 / n_queries as f64, sum_c);
    println!();

    // 姹囨€?    println!("=== 姹囨€?===");
    println!("{:<25} {:>10} {:>12} {:>10}", "鏂规", "QPS", "avg_latency", "鍔犻€熸瘮");
    println!("{:-<60}", "");
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "A: f32 AVX-512", qps_a, time_a * 1e6 / n_queries as f64, 1.0);
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "B: f16 SIMD", qps_b, time_b * 1e6 / n_queries as f64, qps_b / qps_a);
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "C: f16 鏍囬噺", qps_c, time_c * 1e6 / n_queries as f64, qps_c / qps_a);
    println!();

    // 绮惧害瀵规瘮
    let rel_err = (sum_a - sum_b).abs() / sum_a.max(1e-6);
    println!("绮惧害: f16 SIMD vs f32 鐩稿璇樊 = {:.6} (sum_a={:.4}, sum_b={:.4})", rel_err, sum_a, sum_b);
    println!();

    // 缁撹
    let speedup = qps_b / qps_a;
    if speedup >= 1.2 {
        println!("缁撹: f16 SIMD 鍔犻€?{:.2}x (鈮?.2x)锛岃揪鍒?OPT-4 楠屾敹鏍囧噯锛屽彲鎺ュ叆鏌ヨ璺緞", speedup);
    } else if speedup >= 1.05 {
        println!("缁撹: f16 SIMD 鍔犻€?{:.2}x (1.05-1.2x)锛屾敹鐩婃湁闄愶紝闇€璇勪及鎺ュ叆鎴愭湰", speedup);
    } else {
        println!("缁撹: f16 SIMD 鍔犻€?{:.2}x (<1.05x)锛屾棤鏀剁泭锛孫PT-4 鍚﹀喅", speedup);
    }
}
