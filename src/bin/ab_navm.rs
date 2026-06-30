//! nav_m=32 vs nav_m=16 背靠背 A/B 验证
//! 交替运行，消除热节流偏差

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use raven::graph::{GraphSearcher, VamanaGraph};
use raven::memory::serialize::Serializable;
use raven::quant::SQ8Dataset;

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
const ROUNDS: usize = 8;
const K: usize = 10;

fn bench_one(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    sq8: &SQ8Dataset,
    ef: usize,
    po: usize,
    rr: usize,
) -> (f64, f64, f64) {
    let mut sink: u64 = 0;

    // warmup
    let mut s = GraphSearcher::new(train, graph, ef);
    s.with_sq8(sq8);
    s.with_prefetch_offset(po);
    s.with_rerank_factor(rr);
    for _ in 0..REPEATS {
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let r = s.search_sq8(query, K);
            sink = sink.wrapping_add(r[0].0 as u64);
        }
    }

    let mut qps_vals = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let mut s = GraphSearcher::new(train, graph, ef);
        s.with_sq8(sq8);
        s.with_prefetch_offset(po);
        s.with_rerank_factor(rr);
        let t0 = Instant::now();
        for _ in 0..REPEATS {
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let r = s.search_sq8(query, K);
                sink = sink.wrapping_add(r[0].0 as u64);
            }
        }
        let dt = t0.elapsed();
        qps_vals.push((nq * REPEATS) as f64 / dt.as_secs_f64());
    }

    if sink == u64::MAX { eprintln!("impossible"); }

    qps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = qps_vals[ROUNDS / 2];
    let mean = qps_vals.iter().sum::<f64>() / ROUNDS as f64;
    let variance = qps_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ROUNDS as f64;
    let cv = variance.sqrt() / mean * 100.0;
    (median, mean, cv)
}

fn main() {
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    let sq8 = SQ8Dataset::build(&train, dim);

    eprintln!("data: n={}, dim={}, nq={}", n, dim, nq);

    let g16 = VamanaGraph::load(std::path::Path::new("data/sift/sweep_navm16.bin")).expect("load navm16");
    let g32 = VamanaGraph::load(std::path::Path::new("data/sift/sweep_navm32.bin")).expect("load navm32");

    // recall check
    let mut hits16 = 0usize;
    let mut hits32 = 0usize;
    let total = nq * K;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let gt_slice = &gt[q * gt_k..q * gt_k + K];
        let mut s16 = GraphSearcher::new(&train, &g16, 50);
        s16.with_sq8(&sq8);
        s16.with_prefetch_offset(8);
        s16.with_rerank_factor(3);
        let r16 = s16.search_sq8(query, K);
        let mut s32 = GraphSearcher::new(&train, &g32, 50);
        s32.with_sq8(&sq8);
        s32.with_prefetch_offset(8);
        s32.with_rerank_factor(3);
        let r32 = s32.search_sq8(query, K);
        for &g in gt_slice {
            if r16.iter().any(|(id, _)| *id == g as u32) { hits16 += 1; }
            if r32.iter().any(|(id, _)| *id == g as u32) { hits32 += 1; }
        }
    }
    let recall16 = hits16 as f64 / total as f64;
    let recall32 = hits32 as f64 / total as f64;
    eprintln!("recall: nav_m=16 {:.4}  nav_m=32 {:.4}", recall16, recall32);

    // 交替 A/B 测试：A-B-A-B-A-B-A-B
    eprintln!("\n=== 背靠背 A/B 测试 (8 rounds 交替) ===");
    let mut a_vals = Vec::new();
    let mut b_vals = Vec::new();

    for round in 0..ROUNDS {
        let (graph, label, is_a) = if round % 2 == 0 {
            (&g16, "nav_m=16", true)
        } else {
            (&g32, "nav_m=32", false)
        };

        let mut s = GraphSearcher::new(&train, graph, 50);
        s.with_sq8(&sq8);
        s.with_prefetch_offset(8);
        s.with_rerank_factor(3);
        let mut sink: u64 = 0;
        let t0 = Instant::now();
        for _ in 0..REPEATS {
            for q in 0..nq {
                let query = &test[q * dim..(q + 1) * dim];
                let r = s.search_sq8(query, K);
                sink = sink.wrapping_add(r[0].0 as u64);
            }
        }
        let dt = t0.elapsed();
        let qps = (nq * REPEATS) as f64 / dt.as_secs_f64();
        eprintln!("  round {}  {}  QPS={:.0}", round, label, qps);
        if is_a { a_vals.push(qps); } else { b_vals.push(qps); }
    }

    let a_median = {
        let mut v = a_vals.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let b_median = {
        let mut v = b_vals.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let a_mean = a_vals.iter().sum::<f64>() / a_vals.len() as f64;
    let b_mean = b_vals.iter().sum::<f64>() / b_vals.len() as f64;

    eprintln!("\n=== A/B 结果 ===");
    eprintln!("nav_m=16: median={:.0}  mean={:.0}", a_median, a_mean);
    eprintln!("nav_m=32: median={:.0}  mean={:.0}", b_median, b_mean);
    eprintln!("delta:    median={:+.1}%  mean={:+.1}%", 
        (b_median - a_median) / a_median * 100.0,
        (b_mean - a_mean) / a_mean * 100.0);
    eprintln!("recall:   16={:.4}  32={:.4}  delta={:+.4}", recall16, recall32, recall32 - recall16);

    // 写 CSV
    let csv = format!(
        "config,median_qps,mean_qps,recall\nnav_m=16,{:.0},{:.0},{:.4}\nnav_m=32,{:.0},{:.0},{:.4}\n",
        a_median, a_mean, recall16, b_median, b_mean, recall32
    );
    std::fs::write("tuning/ab_navm_result.csv", csv).expect("write");
    eprintln!("\n结果写入 tuning/ab_navm_result.csv");
}
