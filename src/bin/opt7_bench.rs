//! OPT-7 寰熀鍑嗭細鍏ㄩ噺 medoid vs 閲囨牱 medoid
//!
//! 鍙祴 compute_medoid 鐨勮€楁椂锛屼笉璺戝畬鏁村缓鍥?//! 鏁版嵁闆嗭細sift_learn.fvecs锛?00K 鑺傜偣锛宒im=128锛?
use std::fs::File;
use std::io::Read;
use raven::distance::l2_simd;

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

/// 鍏ㄩ噺 medoid锛堟棫瀹炵幇锛?fn compute_medoid_full(vectors: &[f32], dim: usize, n: usize) -> u32 {
    let mut centroid = vec![0.0f32; dim];
    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        for d in 0..dim {
            centroid[d] += v[d];
        }
    }
    for d in 0..dim {
        centroid[d] /= n as f32;
    }
    let mut best_id = 0u32;
    let mut best_dist = f32::MAX;
    for i in 0..n {
        let dist = l2_simd(&centroid, &vectors[i * dim..(i + 1) * dim]);
        if dist < best_dist {
            best_dist = dist;
            best_id = i as u32;
        }
    }
    best_id
}

/// 閲囨牱 medoid锛堟柊瀹炵幇锛?fn compute_medoid_sampled(vectors: &[f32], dim: usize, n: usize, sample_count: usize) -> u32 {
    use raven::build::ChaCha8Rng;
    use rand::seq::SliceRandom;
    let mut rng = ChaCha8Rng::seed_from(42);
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.partial_shuffle(&mut rng, sample_count);
    let sample: Vec<u32> = indices.iter().take(sample_count).copied().collect();

    let mut centroid = vec![0.0f32; dim];
    for &idx in &sample {
        let v = &vectors[idx as usize * dim..(idx as usize + 1) * dim];
        for d in 0..dim {
            centroid[d] += v[d];
        }
    }
    for d in 0..dim {
        centroid[d] /= sample_count as f32;
    }
    let mut best_id = sample[0];
    let mut best_dist = f32::MAX;
    for &idx in &sample {
        let dist = l2_simd(&centroid, &vectors[idx as usize * dim..(idx as usize + 1) * dim]);
        if dist < best_dist {
            best_dist = dist;
            best_id = idx;
        }
    }
    best_id
}

fn main() {
    let path = "data/sift/sift_learn.fvecs";
    let (vectors, dim, n) = read_fvecs(path);
    println!("=== OPT-7 寰熀鍑嗭細medoid 璁＄畻瀵规瘮 ===");
    println!("鏁版嵁闆? {}, dim={}, n={}", path, dim, n);
    println!();

    // 鏂规 A锛氬叏閲?medoid
    let t0 = std::time::Instant::now();
    let medoid_full = compute_medoid_full(&vectors, dim, n);
    let dur_full = t0.elapsed();

    // 鏂规 B锛氶噰鏍?10K
    let t0 = std::time::Instant::now();
    let medoid_10k = compute_medoid_sampled(&vectors, dim, n, 10_000);
    let dur_10k = t0.elapsed();

    // 鏂规 C锛氶噰鏍?1K
    let t0 = std::time::Instant::now();
    let medoid_1k = compute_medoid_sampled(&vectors, dim, n, 1_000);
    let dur_1k = t0.elapsed();

    println!("鏂规 A (鍏ㄩ噺):     {:>8.2}ms,  medoid={}", dur_full.as_secs_f64() * 1000.0, medoid_full);
    println!("鏂规 B (閲囨牱10K):  {:>8.2}ms,  medoid={}", dur_10k.as_secs_f64() * 1000.0, medoid_10k);
    println!("鏂规 C (閲囨牱1K):   {:>8.2}ms,  medoid={}", dur_1k.as_secs_f64() * 1000.0, medoid_1k);
    println!();
    println!("B vs A 鍔犻€熸瘮: {:.2}x (鑰楁椂姣?{:.1}%)", dur_full.as_secs_f64() / dur_10k.as_secs_f64(), dur_10k.as_secs_f64() / dur_full.as_secs_f64() * 100.0);
    println!("C vs A 鍔犻€熸瘮: {:.2}x (鑰楁椂姣?{:.1}%)", dur_full.as_secs_f64() / dur_1k.as_secs_f64(), dur_1k.as_secs_f64() / dur_full.as_secs_f64() * 100.0);

    // 璁＄畻閲囨牱 medoid 涓庡叏閲?medoid 鐨勮窛绂诲樊寮?    let v_full = &vectors[medoid_full as usize * dim..(medoid_full as usize + 1) * dim];
    let v_10k = &vectors[medoid_10k as usize * dim..(medoid_10k as usize + 1) * dim];
    let v_1k = &vectors[medoid_1k as usize * dim..(medoid_1k as usize + 1) * dim];
    let dist_10k = l2_simd(v_full, v_10k).sqrt();
    let dist_1k = l2_simd(v_full, v_1k).sqrt();
    println!();
    println!("medoid 鍚戦噺璺濈宸紓:");
    println!("  鍏ㄩ噺 vs 閲囨牱10K: {:.4}", dist_10k);
    println!("  鍏ㄩ噺 vs 閲囨牱1K:  {:.4}", dist_1k);

    // 楠屾敹鏍囧噯锛氶噰鏍疯€楁椂 < 鍏ㄩ噺鐨?5%锛屼笖 medoid 璺濈宸紓鍙帴鍙?    let ratio_10k = dur_10k.as_secs_f64() / dur_full.as_secs_f64();
    if ratio_10k < 0.05 {
        println!();
        println!("楠屾敹: PASS (鏂规 B 鑰楁椂姣?{:.1}% < 5%)", ratio_10k * 100.0);
    } else {
        println!();
        println!("楠屾敹: FAIL (鏂规 B 鑰楁椂姣?{:.1}% >= 5%)", ratio_10k * 100.0);
    }
}
