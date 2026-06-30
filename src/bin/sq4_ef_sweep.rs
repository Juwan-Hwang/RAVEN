//! SQ4 最终 ef sweep — rr=8，生成提交曲线 CSV
//!
//! 与 final_ef_sweep.rs 相同的 benchmark 方法论（10 repeats × 7 rounds median）
//! 但使用 SQ4 量化 + rerank_factor=8
//!
//! 用法: cargo run --release --bin sq4_ef_sweep

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::graph::{GraphSearcher, VamanaGraph};
use raven::memory::serialize::Serializable;
use raven::quant::{SQ4Dataset, SQ8Dataset};

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

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("open ivecs");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read ivecs");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            gt.push(i32::from_le_bytes(
                bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap(),
            ));
        }
    }
    (gt, dim, n)
}

const REPEATS: usize = 10;
const ROUNDS: usize = 7;
const K: usize = 10;

fn main() {
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let sq4 = SQ4Dataset::build(&train, dim);
    let sq8 = SQ8Dataset::build(&train, dim);

    eprintln!("=== SQ4 Final ef sweep (nav_m=32, DirectionalPrune, r_max=32, rr=8) ===");
    eprintln!("data: n={}, dim={}, nq={}", n, dim, nq);
    eprintln!("SQ4 codes: {} bytes ({:.1} MB), SQ8 codes: {} bytes ({:.1} MB)",
        sq4.codes.len(), sq4.codes.len() as f64 / 1e6,
        sq8.codes.len(), sq8.codes.len() as f64 / 1e6);

    let graph = VamanaGraph::load(std::path::Path::new("data/sift/sweep_navm32.bin"))
        .expect("load nav_m=32 graph");

    let mut csv = String::from("ef,qps,recall,cv_pct\n");
    let mut sink: u64 = 0;

    for &ef in &[40, 45, 50, 55, 60, 65, 70, 80, 100] {
        // recall
        let mut searcher = GraphSearcher::new(&train, &graph, ef);
        searcher.with_sq4(&sq4);
        searcher.with_prefetch_offset(8);
        searcher.with_rerank_factor(8);

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = searcher.search_sq4(query, K);
            let gt_slice = &gt[q * gt_k..q * gt_k + K];
            for &g in gt_slice {
                if result.iter().any(|(id, _)| *id == g as u32) {
                    hits += 1;
                }
            }
            total += K;
        }
        let recall = hits as f64 / total as f64;

        // warmup
        for _ in 0..REPEATS {
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let r = searcher.search_sq4(query, K);
                sink = sink.wrapping_add(r[0].0 as u64);
            }
        }

        // QPS rounds
        let mut qps_vals = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let mut s = GraphSearcher::new(&train, &graph, ef);
            s.with_sq4(&sq4);
            s.with_prefetch_offset(8);
            s.with_rerank_factor(8);
            let t0 = Instant::now();
            for _ in 0..REPEATS {
                for q in 0..nq {
                    let query = &test[q * dim..(q + 1) * dim];
                    let r = s.search_sq4(query, K);
                    sink = sink.wrapping_add(r[0].0 as u64);
                }
            }
            let dt = t0.elapsed();
            qps_vals.push((nq * REPEATS) as f64 / dt.as_secs_f64());
        }

        qps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = qps_vals[ROUNDS / 2];
        let mean = qps_vals.iter().sum::<f64>() / ROUNDS as f64;
        let variance = qps_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ROUNDS as f64;
        let cv = variance.sqrt() / mean * 100.0;

        eprintln!("  ef={:>3}  QPS={:>7.0}  recall={:.4}  CV={:.1}%", ef, median, recall, cv);
        csv.push_str(&format!("{},{:.0},{:.4},{:.1}\n", ef, median, recall, cv));
    }

    if sink == u64::MAX { eprintln!("impossible"); }

    std::fs::write("tuning/sq4_ef_sweep_navm32_rr8.csv", csv).expect("write");
    eprintln!("\n结果写入 tuning/sq4_ef_sweep_navm32_rr8.csv");
}
