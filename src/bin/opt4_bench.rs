//! OPT-4 微基准：f32 vs f16 SIMD 距离计算吞吐量对比
//!
//! 目标：验证 f16 SIMD（F16C + AVX-512）在 memory-bound 场景下是否比 f32 AVX-512 快
//!
//! 关键点：
//! - f16 带宽减半（2B vs 4B/元素），memory-bound 场景下应有收益
//! - 计算精度保持 f32（F16C 在寄存器中转 f32 后计算）
//!
//! 实验方案：
//! - 方案 A：l2_simd（f32 AVX-512）— 当前基线
//! - 方案 B：l2_f16_mixed_simd（f16 + F16C + AVX-512）— OPT-4
//! - 方案 C：l2_f16_mixed（标量 f16）— 对照
//!
//! 数据：SIFT1M base（1M × dim=128 = 493MB f32 / 246MB f16）
//! 随机访问模式（模拟图搜索热路径），数据超出 L3 cache

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::{l2_simd, l2_f16_mixed, l2_f16_mixed_simd, f32_to_f16_slice, is_avx512_and_f16c_supported};

/// 读取 fvecs 文件
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 文件长度不对齐");

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
    println!("=== OPT-4 微基准：f32 vs f16 SIMD 距离计算（SIFT1M memory-bound）===");
    println!();

    if !is_avx512_and_f16c_supported() {
        eprintln!("当前 CPU 不支持 AVX-512 + F16C，无法测试 SIMD f16");
        return;
    }
    println!("CPU 支持: AVX-512 + F16C");
    println!();

    // 加载 SIFT1M base（1M 向量，493MB f32）
    println!("加载 SIFT1M base 数据...");
    let t0 = Instant::now();
    let (db_f32, dim, n_db) = read_fvecs("data/sift/sift_base.fvecs");
    println!("加载完成: {} vecs, dim={}, {:.1}s, {:.1}MB (f32)",
        n_db, dim, t0.elapsed().as_secs_f64(), db_f32.len() * 4 / 1024 / 1024);

    // 预量化为 f16（246MB）
    let t0 = Instant::now();
    let db_f16 = f32_to_f16_slice(&db_f32);
    println!("f16 预量化: {:.1}s, {:.1}MB (f16, 节省 50%)",
        t0.elapsed().as_secs_f64(), db_f16.len() * 2 / 1024 / 1024);
    println!();

    // 模拟图搜索的随机访问模式
    // 用固定的伪随机索引序列，确保三个方案访问相同的节点
    let n_queries = 100_000;  // 10 万次距离计算
    let query_indices: Vec<usize> = (0..n_queries)
        .map(|i| (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) as usize % n_db)
        .collect();

    // 固定查询向量
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) / dim as f32).collect();

    // 预热
    let mut warmup = 0.0f32;
    for &idx in query_indices.iter().take(1000) {
        warmup += l2_simd(&query, &db_f32[idx * dim..(idx + 1) * dim]);
    }
    println!("预热完成 (sum={:.4})", warmup);
    println!();

    // 方案 A：f32 AVX-512（基线）
    let t0 = Instant::now();
    let mut sum_a = 0.0f32;
    for &idx in &query_indices {
        sum_a += l2_simd(&query, &db_f32[idx * dim..(idx + 1) * dim]);
    }
    let time_a = t0.elapsed().as_secs_f64();
    let qps_a = n_queries as f64 / time_a;
    println!("方案 A: f32 AVX-512 (l2_simd)");
    println!("  时间: {:.4}s, QPS={:.0}, avg_latency={:.2}us, sum={:.4}",
        time_a, qps_a, time_a * 1e6 / n_queries as f64, sum_a);
    println!();

    // 方案 B：f16 SIMD（F16C + AVX-512）
    let t0 = Instant::now();
    let mut sum_b = 0.0f32;
    for &idx in &query_indices {
        sum_b += l2_f16_mixed_simd(&query, &db_f16[idx * dim..(idx + 1) * dim]);
    }
    let time_b = t0.elapsed().as_secs_f64();
    let qps_b = n_queries as f64 / time_b;
    println!("方案 B: f16 SIMD (l2_f16_mixed_simd, F16C + AVX-512)");
    println!("  时间: {:.4}s, QPS={:.0}, avg_latency={:.2}us, sum={:.4}",
        time_b, qps_b, time_b * 1e6 / n_queries as f64, sum_b);
    println!();

    // 方案 C：f16 标量（对照）
    let t0 = Instant::now();
    let mut sum_c = 0.0f32;
    for &idx in &query_indices {
        sum_c += l2_f16_mixed(&query, &db_f16[idx * dim..(idx + 1) * dim]);
    }
    let time_c = t0.elapsed().as_secs_f64();
    let qps_c = n_queries as f64 / time_c;
    println!("方案 C: f16 标量 (l2_f16_mixed)");
    println!("  时间: {:.4}s, QPS={:.0}, avg_latency={:.2}us, sum={:.4}",
        time_c, qps_c, time_c * 1e6 / n_queries as f64, sum_c);
    println!();

    // 汇总
    println!("=== 汇总 ===");
    println!("{:<25} {:>10} {:>12} {:>10}", "方案", "QPS", "avg_latency", "加速比");
    println!("{:-<60}", "");
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "A: f32 AVX-512", qps_a, time_a * 1e6 / n_queries as f64, 1.0);
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "B: f16 SIMD", qps_b, time_b * 1e6 / n_queries as f64, qps_b / qps_a);
    println!("{:<25} {:>10.0} {:>10.2}us {:>10.2}x", "C: f16 标量", qps_c, time_c * 1e6 / n_queries as f64, qps_c / qps_a);
    println!();

    // 精度对比
    let rel_err = (sum_a - sum_b).abs() / sum_a.max(1e-6);
    println!("精度: f16 SIMD vs f32 相对误差 = {:.6} (sum_a={:.4}, sum_b={:.4})", rel_err, sum_a, sum_b);
    println!();

    // 结论
    let speedup = qps_b / qps_a;
    if speedup >= 1.2 {
        println!("结论: f16 SIMD 加速 {:.2}x (≥1.2x)，达到 OPT-4 验收标准，可接入查询路径", speedup);
    } else if speedup >= 1.05 {
        println!("结论: f16 SIMD 加速 {:.2}x (1.05-1.2x)，收益有限，需评估接入成本", speedup);
    } else {
        println!("结论: f16 SIMD 加速 {:.2}x (<1.05x)，无收益，OPT-4 否决", speedup);
    }
}
