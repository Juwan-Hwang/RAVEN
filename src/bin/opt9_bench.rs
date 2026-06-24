//! OPT-9 微基准：PQ k-means++ vs 取前 k 个点初始化
//!
//! 对比两种初始化的 reconstruction loss 和训练耗时
//! 数据集：sift_learn.fvecs（100K 节点，dim=128）

use std::fs::File;
use std::io::Read;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom;

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

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// 旧方案：取前 k 个点初始化
fn kmeans_first_k(data: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
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

/// 新方案：k-means++ 初始化
fn kmeans_pp(data: &[Vec<f32>], k: usize, iterations: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n);

    // k-means++ 初始化
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
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

    // K-means 迭代
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

/// 计算 reconstruction loss
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
    println!("=== OPT-9 微基准：PQ k-means++ vs 取前 k 个点 ===");
    println!("数据集: {}, dim={}, n={}", path, dim, n);
    println!();

    let m = 8;  // 子空间数
    let k = 256; // 聚类中心数
    let sub_dim = dim / m;
    let iterations = 10;

    // 提取第一个子空间的数据用于测试
    let sub_vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let start = i * dim + 0 * sub_dim;
            vectors[start..start + sub_dim].to_vec()
        })
        .collect();

    // 方案 A：取前 k 个点初始化（旧实现）
    let t0 = std::time::Instant::now();
    let centers_a = kmeans_first_k(&sub_vectors, k, iterations);
    let dur_a = t0.elapsed();
    let loss_a = reconstruction_loss(&sub_vectors, &centers_a);

    // 方案 B：k-means++ 初始化（新实现）
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let t0 = std::time::Instant::now();
    let centers_b = kmeans_pp(&sub_vectors, k, iterations, &mut rng);
    let dur_b = t0.elapsed();
    let loss_b = reconstruction_loss(&sub_vectors, &centers_b);

    // 方案 C：采样 k-means++（在 10K 样本上做 k-means++ 初始化 + 迭代）
    let mut rng_c = ChaCha8Rng::seed_from_u64(42);
    let sample_count = 10_000.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.partial_shuffle(&mut rng_c, sample_count);
    let sample: Vec<Vec<f32>> = indices.iter().take(sample_count).map(|&i| sub_vectors[i].clone()).collect();
    let t0 = std::time::Instant::now();
    let centers_c_init = kmeans_pp(&sample, k, iterations, &mut rng_c);
    // 在全量上做 3 次微调迭代
    let centers_c = kmeans_refine(&sub_vectors, centers_c_init, 3);
    let dur_c = t0.elapsed();
    let loss_c = reconstruction_loss(&sub_vectors, &centers_c);

    println!("子空间 0: sub_dim={}, k={}, iterations={}", sub_dim, k, iterations);
    println!();
    println!("方案 A (取前k个点):     {:>8.2}ms,  loss={:.6}", dur_a.as_secs_f64() * 1000.0, loss_a);
    println!("方案 B (k-means++全量): {:>8.2}ms,  loss={:.6}", dur_b.as_secs_f64() * 1000.0, loss_b);
    println!("方案 C (采样k-means++): {:>8.2}ms,  loss={:.6}", dur_c.as_secs_f64() * 1000.0, loss_c);
    println!();
    let loss_improvement_b = (loss_a - loss_b) / loss_a * 100.0;
    let loss_improvement_c = (loss_a - loss_c) / loss_a * 100.0;
    let time_ratio_b = dur_b.as_secs_f64() / dur_a.as_secs_f64();
    let time_ratio_c = dur_c.as_secs_f64() / dur_a.as_secs_f64();
    println!("B: loss 改善 {:.2}%, 耗时比 {:.1}%", loss_improvement_b, time_ratio_b * 100.0);
    println!("C: loss 改善 {:.2}%, 耗时比 {:.1}%", loss_improvement_c, time_ratio_c * 100.0);

    // 验收标准：loss 改善 > 0%，且耗时增加 < 3 倍
    if loss_c < loss_a && time_ratio_c < 3.0 {
        println!();
        println!("验收: PASS (方案 C loss 改善 {:.2}%, 耗时比 {:.1}% < 300%)", loss_improvement_c, time_ratio_c * 100.0);
    } else {
        println!();
        println!("验收: FAIL (方案 C loss 改善 {:.2}%, 耗时比 {:.1}%)", loss_improvement_c, time_ratio_c * 100.0);
    }
}

/// K-means 微调（用给定中心做 iterations 次迭代）
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
