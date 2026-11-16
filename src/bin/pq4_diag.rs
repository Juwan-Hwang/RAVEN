//! PQ4 距离质量诊断
//!
//! 对比 PQ4 ADC 距离与 f32 精确距离的排名相关性，
//! 扫描不同 M 值找出可用配置。
//!
//! 用法：cargo run --release --bin pq4_diag

use std::fs::File;
use std::io::{Read, Write};

use raven::quant::PQ4Dataset;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("open fvecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read fvecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            vectors.push(f32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (vectors, dim, n)
}

fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn diagnose_m(
    train: &[f32],
    test: &[f32],
    dim: usize,
    n: usize,
    nq: usize,
    m: usize,
    out: &mut String,
) {
    out.push_str(&format!("\n--- M={} (sub_dim={}), K=16 ---\n", m, dim / m));

    let pq4 = PQ4Dataset::build(train, dim, m);
    out.push_str(&format!(
        "  编码: {} bytes/vector, LUT: {} bytes ({:.1} KB)\n",
        m / 2,
        m * 16 * 4,
        m * 16 * 4 / 1024
    ));

    let mut top10_overlap_sum = 0usize;
    let mut top100_overlap_sum = 0usize;
    let mut spearman_sum = 0.0f64;
    let check_queries = 20usize;

    for q in 0..check_queries.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let lut = pq4.codebook.compute_lut(query);

        let sample_size = 2000usize;
        let mut f32_dists: Vec<(usize, f32)> = Vec::with_capacity(sample_size);
        let mut pq4_dists: Vec<(usize, f32)> = Vec::with_capacity(sample_size);

        for i in 0..sample_size {
            let idx = (i * n / sample_size) % n;
            let v = &train[idx * dim..(idx + 1) * dim];
            f32_dists.push((idx, l2_sq_f32(query, v)));
            pq4_dists.push((idx, PQ4Dataset::adc_distance(&lut, pq4.code(idx), m)));
        }

        f32_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        pq4_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let f32_top10: std::collections::HashSet<usize> =
            f32_dists.iter().take(10).map(|(i, _)| *i).collect();
        let pq4_top10: std::collections::HashSet<usize> =
            pq4_dists.iter().take(10).map(|(i, _)| *i).collect();
        top10_overlap_sum += f32_top10.intersection(&pq4_top10).count();

        let f32_top100: std::collections::HashSet<usize> =
            f32_dists.iter().take(100).map(|(i, _)| *i).collect();
        let pq4_top100: std::collections::HashSet<usize> =
            pq4_dists.iter().take(100).map(|(i, _)| *i).collect();
        top100_overlap_sum += f32_top100.intersection(&pq4_top100).count();

        // Spearman on top-100 by f32
        let f32_rank: std::collections::HashMap<usize, usize> = f32_dists
            .iter()
            .take(100)
            .enumerate()
            .map(|(rank, (idx, _))| (*idx, rank))
            .collect();
        let mut d_sq_sum = 0.0f64;
        let mut n_in_both = 0usize;
        for (rank, (idx, _)) in pq4_dists.iter().take(100).enumerate() {
            if let Some(&f32_r) = f32_rank.get(idx) {
                let d = (rank as f64) - (f32_r as f64);
                d_sq_sum += d * d;
                n_in_both += 1;
            }
        }
        if n_in_both > 1 {
            let np = n_in_both as f64;
            let spearman = 1.0 - 6.0 * d_sq_sum / (np * (np * np - 1.0));
            spearman_sum += spearman;
        }
    }

    let avg_top10 = top10_overlap_sum as f64 / check_queries as f64;
    let avg_top100 = top100_overlap_sum as f64 / check_queries as f64;
    let avg_spearman = spearman_sum / check_queries as f64;

    out.push_str(&format!(
        "  Top-10 overlap:  {}/10 ({:.0}%)\n",
        avg_top10,
        avg_top10 / 10.0 * 100.0
    ));
    out.push_str(&format!(
        "  Top-100 overlap: {}/100 ({:.0}%)\n",
        avg_top100,
        avg_top100 / 100.0 * 100.0
    ));
    out.push_str(&format!("  Spearman ρ (top-100): {:.4}\n", avg_spearman));

    if avg_top10 >= 5.0 && avg_spearman > 0.5 {
        out.push_str("  → ✅ 可用于图导航\n");
    } else if avg_top100 >= 30.0 && avg_spearman > 0.3 {
        out.push_str("  → ⚠️ 勉强可用，需要大 ef 补偿\n");
    } else {
        out.push_str("  → ❌ 距离太噪声，不可用于图导航\n");
    }
}

fn main() {
    let mut out = String::new();

    macro_rules! println_both {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{}", line);
            out.push_str(&line);
            out.push('\n');
        }};
    }

    println_both!("=== PQ4 距离质量诊断 ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    println_both!("数据: n={}, dim={}, nq={}", n, dim, nq);
    println_both!("采样: 每查询 2000 向量, 20 查询");

    for &m in &[8, 16, 32, 64] {
        if dim % m != 0 {
            continue;
        }
        diagnose_m(&train, &test, dim, n, nq, m, &mut out);
        // 打印当前结果
        let lines: Vec<&str> = out.lines().collect();
        for line in lines.iter().rev().take(6).rev() {
            println!("{}", line);
        }
    }

    let mut f = File::create("pq4_diag_result.txt").expect("create result");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 pq4_diag_result.txt");
}
