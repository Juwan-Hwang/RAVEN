//! OPT-9 寰熀鍑嗭細PQ k-means++ vs 鍙栧墠 k 涓偣鍒濆鍖?//!
//! 瀵规瘮涓ょ鍒濆鍖栫殑 reconstruction loss 鍜岃缁冭€楁椂
//! 鏁版嵁闆嗭細sift_learn.fvecs锛?00K 鑺傜偣锛宒im=128锛?
use std::fs::File;
use std::io::Read;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom;

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

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// 鏃ф柟妗堬細鍙栧墠 k 涓偣鍒濆鍖?fn kmeans_first_k(data: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n);
    let mut centers: Vec<Vec<f32>> = data[..k].to_vec();
    if centers.is_empty() {
        return vec![vec![0.0; dim]];
    }
    for _ in 0..iterations {
        let mut assignments = vec![0usize; n];
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (j, center) in centers.iter().enumerate() {
                let d = l2_sq(point, center);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            assignments[i] = best;
        }
        let mut new_centers = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            for d in 0..dim {
                new_centers[a][d] += data[i][d];
            }
            counts[a] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    new_centers[j][d] /= counts[j] as f32;
                }
            } else {
                new_centers[j] = centers[j].clone();
            }
        }
        centers = new_centers;
    }
    centers
}

/// 鏂版柟妗堬細k-means++ 鍒濆鍖?fn kmeans_pp(data: &[Vec<f32>], k: usize, iterations: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n);

    // k-means++ 鍒濆鍖?    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
    centers.push(data[0].clone());
    for _ in 1..k {
        let mut dists = vec![f32::MAX; n];
        for (i, point) in data.iter().enumerate() {
            for center in &centers {
                let d = l2_sq(point, center);
                if d < dists[i] {
                    dists[i] = d;
                }
            }
        }
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            if centers.len() < n {
                centers.push(data[centers.len()].clone());
            }
            continue;
        }
        let r: f32 = rng.gen();
        let mut cum = 0.0f32;
        let mut chosen = 0;
        for (i, _) in data.iter().enumerate() {
            cum += dists[i] / total;
            if cum >= r {
                chosen = i;
                break;
            }
        }
        centers.push(data[chosen].clone());
    }

    // K-means 杩唬
    for _ in 0..iterations {
        let mut assignments = vec![0usize; n];
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (j, center) in centers.iter().enumerate() {
                let d = l2_sq(point, center);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            assignments[i] = best;
        }
        let mut new_centers = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            for d in 0..dim {
                new_centers[a][d] += data[i][d];
            }
            counts[a] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    new_centers[j][d] /= counts[j] as f32;
                }
            } else {
                new_centers[j] = centers[j].clone();
            }
        }
        centers = new_centers;
    }
    centers
}

/// 璁＄畻 reconstruction loss
fn reconstruction_loss(data: &[Vec<f32>], centers: &[Vec<f32>]) -> f32 {
    let n = data.len();
    let mut total = 0.0f32;
    for point in data {
        let mut best_dist = f32::MAX;
        for center in centers {
            let d = l2_sq(point, center);
            if d < best_dist {
                best_dist = d;
            }
        }
        total += best_dist;
    }
    total / n as f32
}

fn main() {
    let path = "data/sift/sift_learn.fvecs";
    let (vectors, dim, n) = read_fvecs(path);
    println!("=== OPT-9 寰熀鍑嗭細PQ k-means++ vs 鍙栧墠 k 涓偣 ===");
    println!("鏁版嵁闆? {}, dim={}, n={}", path, dim, n);
    println!();

    let m = 8;  // 瀛愮┖闂存暟
    let k = 256; // 鑱氱被涓績鏁?    let sub_dim = dim / m;
    let iterations = 10;

    // 鎻愬彇绗竴涓瓙绌洪棿鐨勬暟鎹敤浜庢祴璇?    let sub_vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let start = i * dim + 0 * sub_dim;
            vectors[start..start + sub_dim].to_vec()
        })
        .collect();

    // 鏂规 A锛氬彇鍓?k 涓偣鍒濆鍖栵紙鏃у疄鐜帮級
    let t0 = std::time::Instant::now();
    let centers_a = kmeans_first_k(&sub_vectors, k, iterations);
    let dur_a = t0.elapsed();
    let loss_a = reconstruction_loss(&sub_vectors, &centers_a);

    // 鏂规 B锛歬-means++ 鍒濆鍖栵紙鏂板疄鐜帮級
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let t0 = std::time::Instant::now();
    let centers_b = kmeans_pp(&sub_vectors, k, iterations, &mut rng);
    let dur_b = t0.elapsed();
    let loss_b = reconstruction_loss(&sub_vectors, &centers_b);

    // 鏂规 C锛氶噰鏍?k-means++锛堝湪 10K 鏍锋湰涓婂仛 k-means++ 鍒濆鍖?+ 杩唬锛?    let mut rng_c = ChaCha8Rng::seed_from_u64(42);
    let sample_count = 10_000.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.partial_shuffle(&mut rng_c, sample_count);
    let sample: Vec<Vec<f32>> = indices.iter().take(sample_count).map(|&i| sub_vectors[i].clone()).collect();
    let t0 = std::time::Instant::now();
    let centers_c_init = kmeans_pp(&sample, k, iterations, &mut rng_c);
    // 鍦ㄥ叏閲忎笂鍋?3 娆″井璋冭凯浠?    let centers_c = kmeans_refine(&sub_vectors, centers_c_init, 3);
    let dur_c = t0.elapsed();
    let loss_c = reconstruction_loss(&sub_vectors, &centers_c);

    println!("瀛愮┖闂?0: sub_dim={}, k={}, iterations={}", sub_dim, k, iterations);
    println!();
    println!("鏂规 A (鍙栧墠k涓偣):     {:>8.2}ms,  loss={:.6}", dur_a.as_secs_f64() * 1000.0, loss_a);
    println!("鏂规 B (k-means++鍏ㄩ噺): {:>8.2}ms,  loss={:.6}", dur_b.as_secs_f64() * 1000.0, loss_b);
    println!("鏂规 C (閲囨牱k-means++): {:>8.2}ms,  loss={:.6}", dur_c.as_secs_f64() * 1000.0, loss_c);
    println!();
    let loss_improvement_b = (loss_a - loss_b) / loss_a * 100.0;
    let loss_improvement_c = (loss_a - loss_c) / loss_a * 100.0;
    let time_ratio_b = dur_b.as_secs_f64() / dur_a.as_secs_f64();
    let time_ratio_c = dur_c.as_secs_f64() / dur_a.as_secs_f64();
    println!("B: loss 鏀瑰杽 {:.2}%, 鑰楁椂姣?{:.1}%", loss_improvement_b, time_ratio_b * 100.0);
    println!("C: loss 鏀瑰杽 {:.2}%, 鑰楁椂姣?{:.1}%", loss_improvement_c, time_ratio_c * 100.0);

    // 楠屾敹鏍囧噯锛歭oss 鏀瑰杽 > 0%锛屼笖鑰楁椂澧炲姞 < 3 鍊?    if loss_c < loss_a && time_ratio_c < 3.0 {
        println!();
        println!("楠屾敹: PASS (鏂规 C loss 鏀瑰杽 {:.2}%, 鑰楁椂姣?{:.1}% < 300%)", loss_improvement_c, time_ratio_c * 100.0);
    } else {
        println!();
        println!("楠屾敹: FAIL (鏂规 C loss 鏀瑰杽 {:.2}%, 鑰楁椂姣?{:.1}%)", loss_improvement_c, time_ratio_c * 100.0);
    }
}

/// K-means 寰皟锛堢敤缁欏畾涓績鍋?iterations 娆¤凯浠ｏ級
fn kmeans_refine(data: &[Vec<f32>], mut centers: Vec<Vec<f32>>, iterations: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || centers.is_empty() {
        return centers;
    }
    let dim = data[0].len();
    let n = data.len();
    let k = centers.len();
    for _ in 0..iterations {
        let mut assignments = vec![0usize; n];
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (j, center) in centers.iter().enumerate() {
                let d = l2_sq(point, center);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            assignments[i] = best;
        }
        let mut new_centers = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            for d in 0..dim {
                new_centers[a][d] += data[i][d];
            }
            counts[a] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    new_centers[j][d] /= counts[j] as f32;
                }
            } else {
                new_centers[j] = centers[j].clone();
            }
        }
        centers = new_centers;
    }
    centers
}
