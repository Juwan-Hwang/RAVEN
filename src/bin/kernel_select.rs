//! 三阶段内核选择 CLI（含 5 分钟稳定性测试）
//!
//! 设计文档 Week 3-4 硬目标：三阶段内核选择（含 5 分钟稳定性测试）落地
//! 设计文档 F.10：连续运行 5 分钟，每 30 秒采样一次 QPS
//!
//! 用法：
//!   cargo run --release --bin kernel_select -- --dim 768
//!   cargo run --release --bin kernel_select -- --dim 768 --duration 300
//!   cargo run --release --bin kernel_select -- --dim 768 --skip-stability  # 跳过 5 分钟测试
//!
//! 环境变量：
//!   RAVEN_KERNEL=scalar|avx2  强制指定内核（跳过三阶段筛选）
//!   RAVEN_CACHE_DIR=<path>    指定缓存目录

use std::time::{Duration, Instant};
use raven::distance::kernel::{
    available_kernels, fallback_kernel, measure_latency_ns, measure_qps,
    KernelVariant, LATENCY_THRESHOLD_NS,
};

fn main() {
    // 初始化 tracing（简化版，直接 println）
    let args: Vec<String> = std::env::args().collect();
    let mut dim: usize = 768;
    let mut stability_duration_secs: u64 = 300; // 设计文档 F.10：5 分钟
    let mut skip_stability = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dim" => {
                i += 1;
                dim = args[i].parse().expect("invalid dim");
            }
            "--duration" => {
                i += 1;
                stability_duration_secs = args[i].parse().expect("invalid duration");
            }
            "--skip-stability" => skip_stability = true,
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                eprintln!("unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("=== RAVEN 三阶段内核选择 ===");
    println!("维度: dim={}", dim);
    println!("稳定性测试时长: {}s ({:.1}min)", stability_duration_secs, stability_duration_secs as f64 / 60.0);
    println!();

    let candidates = available_kernels();
    println!("可用内核: {:?}", candidates.iter().map(|k| k.name()).collect::<Vec<_>>());
    println!();

    // === 第一阶段：延迟粗筛 ===
    println!("--- 第一阶段：延迟粗筛（阈值 {}ns）---", LATENCY_THRESHOLD_NS);
    let mut passed_stage1: Vec<(KernelVariant, u64)> = Vec::new();
    for k in &candidates {
        let lat = measure_latency_ns(*k, dim);
        let pass = lat < LATENCY_THRESHOLD_NS;
        println!("  {:<8} latency={:>6}ns  {}",
            k.name(), lat,
            if pass { "PASS" } else { "FAIL (淘汰)" });
        if pass {
            passed_stage1.push((*k, lat));
        }
    }
    println!();

    if passed_stage1.is_empty() {
        println!("所有内核延迟超标，降级到 {}", fallback_kernel(dim).name());
        return;
    }

    // === 第二阶段：瞬时 QPS 筛选 ===
    println!("--- 第二阶段：瞬时 QPS 筛选 ---");
    let mut qps_results: Vec<(KernelVariant, u64)> = Vec::new();
    for (k, _lat) in &passed_stage1 {
        let qps = measure_qps(*k, dim);
        qps_results.push((*k, qps));
        println!("  {:<8} qps={:>10}", k.name(), qps);
    }
    qps_results.sort_by(|a, b| b.1.cmp(&a.1));
    let finalist = qps_results[0].0;
    println!();
    println!("决赛选手: {} (qps={})", finalist.name(), qps_results[0].1);
    println!();

    // === 第三阶段：持续稳定性验证 ===
    if skip_stability {
        println!("--- 第三阶段：跳过（--skip-stability）---");
        println!();
        save_cache(dim, finalist);
        print_summary(dim, finalist, &passed_stage1, &qps_results, None);
        return;
    }

    println!("--- 第三阶段：持续稳定性验证（{}s，每 30s 采样）---", stability_duration_secs);
    println!("  运行中...（此阶段耗时 {}s）", stability_duration_secs);

    let stability_start = Instant::now();
    let baseline_qps = measure_qps(finalist, dim);
    println!("  baseline QPS: {}", baseline_qps);

    // 设计文档 F.10：连续运行 5 分钟，每 30 秒采样一次 QPS
    // 关键：内核必须持续运行（不能空闲），以检测 thermal throttling
    let kernel = finalist.build_kernel(raven::distance::DistanceMetric::L2);
    let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002) + 1.0).collect();

    // 预热
    for _ in 0..256 {
        let _ = kernel.distance(&a, &b);
    }

    let sample_interval = Duration::from_secs(30);
    let mut samples: Vec<(u64, u64)> = Vec::new(); // (elapsed_secs, qps)
    let mut total_count: u64 = 0;

    while stability_start.elapsed() < Duration::from_secs(stability_duration_secs) {
        let window_start = Instant::now();
        let mut window_count: u64 = 0;
        // 持续运行 30 秒，不停歇
        while window_start.elapsed() < sample_interval {
            for _ in 0..1024 {
                let _ = kernel.distance(&a, &b);
            }
            window_count += 1024;
        }
        total_count += window_count;
        let elapsed = stability_start.elapsed().as_secs();
        let window_secs = window_start.elapsed().as_secs_f64();
        let window_qps = (window_count as f64 / window_secs) as u64;
        samples.push((elapsed, window_qps));
        println!("  [{:>3}] elapsed={:>3}s  window_qps={:>10}  total_calls={}",
            samples.len(), elapsed, window_qps, total_count);
    }

    let total_elapsed = stability_start.elapsed();
    let sustained_qps = (total_count as f64 / total_elapsed.as_secs_f64()) as u64;
    let ratio = sustained_qps as f64 / baseline_qps as f64;
    let stable = ratio >= 0.95;

    println!();
    println!("  baseline QPS:   {}", baseline_qps);
    println!("  sustained QPS:  {:.0} ({} 个采样平均)", sustained_qps, samples.len());
    println!("  ratio:          {:.4}", ratio);
    println!("  通过标准:       ratio >= 0.95");
    println!("  结果:           {}",
        if stable { "PASS" } else { "FAIL (降级)" });
    println!();

    let selected = if stable {
        finalist
    } else {
        tracing::warn!(
            kernel = finalist.name(),
            ratio,
            "stability test failed, falling back"
        );
        fallback_kernel(dim)
    };

    // 保存到缓存（设计文档：缓存最优配置，首次选定后落盘）
    save_cache(dim, selected);

    print_summary(dim, selected, &passed_stage1, &qps_results, Some((ratio, stable)));
}

/// 保存内核选择到缓存文件
fn save_cache(dim: usize, variant: KernelVariant) {
    use raven::distance::kernel::KernelCache;
    let cache_dir = std::env::var("RAVEN_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            let p = std::path::PathBuf::from(home).join(".cache").join("raven");
            std::fs::create_dir_all(&p).ok();
            p
        });
    let cache_path = cache_dir.join("kernel_cache.toml");
    let mut cache = KernelCache::load(&cache_path).unwrap_or_default();
    cache.set(dim, variant);
    match cache.save(&cache_path) {
        Ok(()) => eprintln!("cache saved to {}", cache_path.display()),
        Err(e) => eprintln!("warning: failed to save cache: {}", e),
    }
}

fn print_summary(
    dim: usize,
    selected: KernelVariant,
    stage1: &[(KernelVariant, u64)],
    stage2: &[(KernelVariant, u64)],
    stability: Option<(f64, bool)>,
) {
    println!("=== 最终选择结果 ===");
    println!("dim={}, kernel={}", dim, selected.name());
    println!();
    println!("阶段 1（延迟粗筛）:");
    for (k, lat) in stage1 {
        println!("  {:<8} {}ns", k.name(), lat);
    }
    println!("阶段 2（瞬时 QPS）:");
    for (k, qps) in stage2 {
        println!("  {:<8} qps={}", k.name(), qps);
    }
    if let Some((ratio, stable)) = stability {
        println!("阶段 3（稳定性）: ratio={:.4} stable={}", ratio, stable);
    }
    println!();
    println!("缓存路径: ~/.cache/raven/kernel_cache.toml");
    println!("（下次同维度选择将直接命中缓存，跳过三阶段筛选）");
}

fn print_help() {
    println!("RAVEN 三阶段内核选择工具");
    println!();
    println!("用法:");
    println!("  kernel_select [OPTIONS]");
    println!();
    println!("选项:");
    println!("  --dim <N>              维度（默认 768）");
    println!("  --duration <S>         稳定性测试时长秒数（默认 300=5min）");
    println!("  --skip-stability       跳过第三阶段稳定性测试");
    println!("  -v, --verbose          详细输出");
    println!("  -h, --help             显示帮助");
    println!();
    println!("环境变量:");
    println!("  RAVEN_KERNEL=scalar|avx2   强制指定内核");
    println!("  RAVEN_CACHE_DIR=<path>     缓存目录");
}
