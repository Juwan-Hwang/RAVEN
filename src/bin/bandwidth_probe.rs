//! 内存带宽瓶颈分析探针
//!
//! 设计文档 Week 3-4：内存带宽瓶颈分析（LLC miss / bandwidth counters，
//! 判断 compute-bound vs memory-bound）
//!
//! 本探针通过扫描 working set 大小，测量不同 cache 层级下的有效带宽，
//! 从而判断 L2 距离核是 compute-bound 还是 memory-bound。
//!
//! 判定逻辑：
//! - 若带宽随 working set 增大而显著下降（L1→L2→L3→RAM）→ memory-bound
//! - 若带宽在各层级保持稳定 → compute-bound
//! - 对比算术强度：L2 距离 3 FLOPs / 8 bytes = 0.375 FLOPs/byte，
//!   远低于 compute-bound 门槛（~10 FLOPs/byte），理论上应为 memory-bound

use std::time::Instant;
use raven::distance::l2_dynamic;

/// 测量给定 working set 下的 L2 距离吞吐
///
/// n_pairs: 同时计算的向量对数（控制 working set 大小）
/// dim: 维度
/// iterations: 迭代次数（大 working set 时应减少以控制总耗时）
fn measure_bandwidth(n_pairs: usize, dim: usize, iterations: usize) -> BandwidthResult {
    // 准备数据：n_pairs 对向量
    let total_bytes = n_pairs * 2 * dim * std::mem::size_of::<f32>();
    let vectors_a: Vec<f32> = (0..n_pairs * dim).map(|i| i as f32 * 0.001).collect();
    let vectors_b: Vec<f32> = (0..n_pairs * dim).map(|i| (i as f32 * 0.002) + 1.0).collect();

    // 预热：确保数据进入对应 cache 层级
    let mut sink = 0.0f32;
    for _ in 0..1000.min(n_pairs * 10) {
        for p in 0..n_pairs {
            let a = &vectors_a[p * dim..(p + 1) * dim];
            let b = &vectors_b[p * dim..(p + 1) * dim];
            sink += l2_dynamic(a, b);
        }
    }
    std::hint::black_box(sink);

    // 实测
    let mut sink = 0.0f32;
    let start = Instant::now();
    for _ in 0..iterations {
        for p in 0..n_pairs {
            let a = &vectors_a[p * dim..(p + 1) * dim];
            let b = &vectors_b[p * dim..(p + 1) * dim];
            sink += l2_dynamic(a, b);
        }
    }
    let elapsed = start.elapsed();
    std::hint::black_box(sink);

    let total_computations = iterations * n_pairs;
    let total_flops = total_computations * dim * 3; // sub + mul + add per element
    let total_bytes_loaded = total_computations * dim * 2 * std::mem::size_of::<f32>();

    let secs = elapsed.as_secs_f64();
    let bandwidth_gbs = total_bytes_loaded as f64 / secs / 1e9;
    let gflops = total_flops as f64 / secs / 1e9;
    let latency_ns = elapsed.as_nanos() as f64 / total_computations as f64;

    BandwidthResult {
        n_pairs,
        dim,
        working_set_kb: total_bytes as f64 / 1024.0,
        bandwidth_gbs,
        gflops,
        latency_ns,
    }
}

#[derive(Debug, Clone)]
struct BandwidthResult {
    n_pairs: usize,
    dim: usize,
    working_set_kb: f64,
    bandwidth_gbs: f64,
    gflops: f64,
    latency_ns: f64,
}

impl BandwidthResult {
    fn cache_level(&self) -> &'static str {
        // AMD Zen 4: L1d=32KB, L2=1MB, L3=16MB
        if self.working_set_kb <= 32.0 {
            "L1"
        } else if self.working_set_kb <= 1024.0 {
            "L2"
        } else if self.working_set_kb <= 16384.0 {
            "L3"
        } else {
            "RAM"
        }
    }
}

fn main() {
    println!("=== RAVEN 内存带宽瓶颈分析 ===");
    println!("CPU: AMD Ryzen 7 8845H (Zen 4)");
    println!("Cache: L1d=32KB, L2=1MB, L3=16MB");
    println!();
    println!("L2 距离算术强度: 3 FLOPs / 8 bytes = 0.375 FLOPs/byte");
    println!("Compute-bound 门槛: ~10 FLOPs/byte");
    println!();

    // 测试 1：固定 dim=768，扫描 working set 大小
    // 揭示 cache 层级对带宽的影响
    println!("=== 测试 1: 固定 dim=768, 扫描 working set ===");
    println!("{:>10} {:>8} {:>10} {:>12} {:>10} {:>10} {:>8}",
        "pairs", "dim", "WS(KB)", "cache", "BW(GB/s)", "GFLOPs", "lat(ns)");
    println!("{}", "-".repeat(72));

    let dim = 768;
    // working set = n_pairs × 2 × 768 × 4 bytes
    // 1 pair = 6KB (L1), 5 pairs = 30KB (L1), 10 pairs = 60KB (L2),
    // 100 pairs = 600KB (L2), 1000 pairs = 6MB (L3), 5000 pairs = 30MB (RAM)
    let pair_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096];

    let mut l1_bw: f64 = 0.0;
    let mut ram_bw: f64 = 0.0;

    for &n_pairs in &pair_counts {
        // 大 working set 减少迭代次数以控制总耗时
        let iterations = (1_000_000 / n_pairs).max(1000);
        let result = measure_bandwidth(n_pairs, dim, iterations);
        if result.working_set_kb <= 32.0 {
            l1_bw = l1_bw.max(result.bandwidth_gbs);
        }
        if result.working_set_kb > 16384.0 {
            ram_bw = ram_bw.max(result.bandwidth_gbs);
        }
        println!("{:>10} {:>8} {:>10.1} {:>12} {:>10.1} {:>10.1} {:>8.1}",
            result.n_pairs,
            result.dim,
            result.working_set_kb,
            result.cache_level(),
            result.bandwidth_gbs,
            result.gflops,
            result.latency_ns,
        );
    }

    println!();
    println!("=== 测试 2: 固定 working set 在 L1, 扫描维度 ===");
    println!("{:>8} {:>10} {:>12} {:>10} {:>10} {:>8}",
        "dim", "WS(KB)", "cache", "BW(GB/s)", "GFLOPs", "lat(ns)");
    println!("{}", "-".repeat(62));

    // 固定 1 pair，确保都在 L1
    let dims = [64, 128, 256, 384, 512, 768, 960, 1024, 1536, 2048, 3072, 4096];
    for &d in &dims {
        let result = measure_bandwidth(1, d, 1_000_000);
        println!("{:>8} {:>10.1} {:>12} {:>10.1} {:>10.1} {:>8.1}",
            result.dim,
            result.working_set_kb,
            result.cache_level(),
            result.bandwidth_gbs,
            result.gflops,
            result.latency_ns,
        );
    }

    println!();
    println!("=== 分析结论 ===");
    let bw_ratio = if ram_bw > 0.0 { l1_bw / ram_bw } else { 0.0 };
    println!("L1 峰值带宽: {:.1} GB/s", l1_bw);
    println!("RAM 峰值带宽: {:.1} GB/s", ram_bw);
    println!("L1/RAM 带宽比: {:.1}x", bw_ratio);
    println!();

    if bw_ratio > 3.0 {
        println!(">>> 判定: MEMORY-BOUND <<<");
        println!("带宽随 working set 跨 cache 层级显著下降（{:.1}x），", bw_ratio);
        println!("说明数据搬运是瓶颈。AVX2 加宽指令在此场景收益有限，");
        println!("优化重心应先放在：布局优化 + 预取 + 减少数据搬运。");
    } else {
        println!(">>> 判定: COMPUTE-BOUND 或 OVERHEAD-BOUND <<<");
        println!("带宽在各 cache 层级保持稳定，瓶颈在计算或循环开销。");
        println!("AVX2 加宽指令可直接提升吞吐，应优先引入。");
    }
    println!();
    println!("=== 理论参考 ===");
    println!("Zen 4 峰值带宽: L1~288GB/s, L2~144GB/s, L3~72GB/s, RAM~50GB/s");
    println!("当前 L1 有效带宽 {:.1} GB/s = L1 峰值的 {:.1}%", l1_bw, l1_bw / 288.0 * 100.0);
    println!("若有效带宽 << 峰值带宽，说明计算/循环开销是瓶颈（compute-bound）");
    println!("若有效带宽 ≈ 峰值带宽，说明数据搬运是瓶颈（memory-bound）");
}
