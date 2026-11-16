//! po=6 vs po=8 确认实验（ef=50）
//!
//! 交替执行 po=6 和 po=8 各 5 轮，输出每轮 QPS + 均值 + 标准差。
//! 用于判断 po=6 比 po=8 快 1.7% 是真实信号还是噪声。
//!
//! 结果自动写入 po_confirm_result.txt
//!
//! 用法：cargo run --release --bin po_confirm

use std::fs::File;
use std::io::{Read, Write};

use raven::build::ChaCha8Rng;
use raven::graph::{GraphSearcher, VamanaBuildConfig, VamanaGraph};
use std::time::Instant;

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

/// 单次 bench：返回 QPS（recall 和 avg_visited 对同一 ef+po 是确定性的，只需测一次）
fn bench_qps(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    dim: usize,
    nq: usize,
    ef: usize,
    po: usize,
    k: usize,
) -> f64 {
    let mut searcher = GraphSearcher::new(train, graph, ef);
    searcher.with_prefetch_offset(po);

    let t0 = Instant::now();
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search(query, k);
    }
    nq as f64 / t0.elapsed().as_secs_f64()
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64]) -> f64 {
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
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

    println_both!("=== po=6 vs po=8 确认实验 (ef=50) ===\n");

    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (_gt, _gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");

    for v in train.iter_mut() {
        *v /= 255.0;
    }
    for v in test.iter_mut() {
        *v /= 255.0;
    }

    println_both!("数据: n={}, dim={}, nq={}", n, dim, nq);
    let k = 10usize;

    // 建图
    println_both!("\n--- 建图 ---");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 32,
        r_soft: 48,
        max_iterations: 2,
        saturate: true,
        enable_layered_nav: true,
        nav_m: 16,
        ..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    println_both!("建图: {:.1}s", t0.elapsed().as_secs_f64());

    // warm-up
    println_both!("\n--- warm-up ---");
    bench_qps(&train, &graph, &test, dim, nq, 50, 8, k);
    bench_qps(&train, &graph, &test, dim, nq, 50, 6, k);
    println_both!("warm-up done");

    // 5 轮交替
    let rounds = 5;
    let ef = 50;
    let po_a = 6;
    let po_b = 8;

    println_both!(
        "\n--- {} 轮交替测量 (ef={}, po={} vs po={}) ---\n",
        rounds,
        ef,
        po_a,
        po_b
    );
    println_both!(
        "  {:>5}  {:>12}  {:>12}  {:>10}",
        "round", "QPS(po=6)", "QPS(po=8)", "diff%"
    );
    println_both!("  {}", "-".repeat(44));

    let mut qps_a_all: Vec<f64> = Vec::new();
    let mut qps_b_all: Vec<f64> = Vec::new();

    for round in 1..=rounds {
        // 交替顺序：奇数轮先 6 后 8，偶数轮先 8 后 6（消除 cache warming 偏差）
        let (first_po, second_po) = if round % 2 == 1 {
            (po_a, po_b)
        } else {
            (po_b, po_a)
        };

        let qps_first = bench_qps(&train, &graph, &test, dim, nq, ef, first_po, k);
        let qps_second = bench_qps(&train, &graph, &test, dim, nq, ef, second_po, k);

        let (qps_6, qps_8) = if round % 2 == 1 {
            (qps_first, qps_second)
        } else {
            (qps_second, qps_first)
        };

        let diff = (qps_6 / qps_8 - 1.0) * 100.0;
        println_both!(
            "  {:>5}  {:>12.0}  {:>12.0}  {:>+9.1}%",
            round, qps_6, qps_8, diff
        );

        qps_a_all.push(qps_6);
        qps_b_all.push(qps_8);
    }

    // 统计
    let mean_6 = mean(&qps_a_all);
    let mean_8 = mean(&qps_b_all);
    let std_6 = stddev(&qps_a_all);
    let std_8 = stddev(&qps_b_all);
    let diff_pct = (mean_6 / mean_8 - 1.0) * 100.0;

    println_both!("\n--- 统计 ---\n");
    println_both!("  po=6: mean={:.0}  std={:.0}  cv={:.2}%", mean_6, std_6, std_6 / mean_6 * 100.0);
    println_both!("  po=8: mean={:.0}  std={:.0}  cv={:.2}%", mean_8, std_8, std_8 / mean_8 * 100.0);
    println_both!("\n  差距: {:+.1}% (po=6 vs po=8)", diff_pct);

    // 判定
    println_both!("\n--- 判定 ---\n");
    let noise_band = (std_6 / mean_6 * 100.0).max(std_8 / mean_8 * 100.0) * 2.0; // 2σ 噪声带
    if diff_pct.abs() > noise_band {
        if diff_pct > 0.0 {
            println_both!(
                "  结论：po=6 稳定优于 po=8（差距 {:+.1}% > 2σ 噪声带 {:.1}%）",
                diff_pct,
                noise_band
            );
            println_both!("  → 将默认 po 从 8 改为 6");
        } else {
            println_both!(
                "  结论：po=8 稳定优于 po=6（差距 {:+.1}% > 2σ 噪声带 {:.1}%）",
                diff_pct,
                noise_band
            );
            println_both!("  → 保留 po=8 不变");
        }
    } else {
        println_both!(
            "  结论：差距 {:+.1}% 在噪声带内（2σ={:.1}%），无法确认统计显著性",
            diff_pct,
            noise_band
        );
        println_both!("  → 保留 po=8 不变");
    }

    // 写文件
    let mut f = File::create("po_confirm_result.txt").expect("create result file");
    f.write_all(out.as_bytes()).expect("write result");
    println!("\n结果已写入 po_confirm_result.txt");
}
