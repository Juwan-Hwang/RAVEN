//! 涓夐樁娈靛唴鏍搁€夋嫨 CLI锛堝惈 5 鍒嗛挓绋冲畾鎬ф祴璇曪級
//!
//! 璁捐鏂囨。 Week 3-4 纭洰鏍囷細涓夐樁娈靛唴鏍搁€夋嫨锛堝惈 5 鍒嗛挓绋冲畾鎬ф祴璇曪級钀藉湴
//! 璁捐鏂囨。 F.10锛氳繛缁繍琛?5 鍒嗛挓锛屾瘡 30 绉掗噰鏍蜂竴娆?QPS
//!
//! 鐢ㄦ硶锛?//!   cargo run --release --bin kernel_select -- --dim 768
//!   cargo run --release --bin kernel_select -- --dim 768 --duration 300
//!   cargo run --release --bin kernel_select -- --dim 768 --skip-stability  # 璺宠繃 5 鍒嗛挓娴嬭瘯
//!
//! 鐜鍙橀噺锛?//!   RAVEN_KERNEL=scalar|avx2  寮哄埗鎸囧畾鍐呮牳锛堣烦杩囦笁闃舵绛涢€夛級
//!   RAVEN_CACHE_DIR=<path>    鎸囧畾缂撳瓨鐩綍

use std::time::{Duration, Instant};
use raven::distance::kernel::{
    available_kernels, fallback_kernel, measure_latency_ns, measure_qps,
    KernelVariant, LATENCY_THRESHOLD_NS,
};

fn main() {
    // 鍒濆鍖?tracing锛堢畝鍖栫増锛岀洿鎺?println锛?    let args: Vec<String> = std::env::args().collect();
    let mut dim: usize = 768;
    let mut stability_duration_secs: u64 = 300; // 璁捐鏂囨。 F.10锛? 鍒嗛挓
    let mut skip_stability = false;

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

    println!("=== RAVEN 涓夐樁娈靛唴鏍搁€夋嫨 ===");
    println!("缁村害: dim={}", dim);
    println!("绋冲畾鎬ф祴璇曟椂闀? {}s ({:.1}min)", stability_duration_secs, stability_duration_secs as f64 / 60.0);
    println!();

    let candidates = available_kernels();
    println!("鍙敤鍐呮牳: {:?}", candidates.iter().map(|k| k.name()).collect::<Vec<_>>());
    println!();

    // === 绗竴闃舵锛氬欢杩熺矖绛?===
    println!("--- 绗竴闃舵锛氬欢杩熺矖绛涳紙闃堝€?{}ns锛?--", LATENCY_THRESHOLD_NS);
    let mut passed_stage1: Vec<(KernelVariant, u64)> = Vec::new();
    for k in &candidates {
        let lat = measure_latency_ns(*k, dim);
        let pass = lat < LATENCY_THRESHOLD_NS;
        println!("  {:<8} latency={:>6}ns  {}",
            k.name(), lat,
            if pass { "PASS" } else { "FAIL (娣樻卑)" });
        if pass {
            passed_stage1.push((*k, lat));
        }
    }
    println!();

    if passed_stage1.is_empty() {
        println!("鎵€鏈夊唴鏍稿欢杩熻秴鏍囷紝闄嶇骇鍒?{}", fallback_kernel(dim).name());
        return;
    }

    // === 绗簩闃舵锛氱灛鏃?QPS 绛涢€?===
    println!("--- 绗簩闃舵锛氱灛鏃?QPS 绛涢€?---");
    let mut qps_results: Vec<(KernelVariant, u64)> = Vec::new();
    for (k, _lat) in &passed_stage1 {
        let qps = measure_qps(*k, dim);
        qps_results.push((*k, qps));
        println!("  {:<8} qps={:>10}", k.name(), qps);
    }
    qps_results.sort_by(|a, b| b.1.cmp(&a.1));
    let finalist = qps_results[0].0;
    println!();
    println!("鍐宠禌閫夋墜: {} (qps={})", finalist.name(), qps_results[0].1);
    println!();

    // === 绗笁闃舵锛氭寔缁ǔ瀹氭€ч獙璇?===
    if skip_stability {
        println!("--- 绗笁闃舵锛氳烦杩囷紙--skip-stability锛?--");
        println!();
        save_cache(dim, finalist);
        print_summary(dim, finalist, &passed_stage1, &qps_results, None);
        return;
    }

    println!("--- 绗笁闃舵锛氭寔缁ǔ瀹氭€ч獙璇侊紙{}s锛屾瘡 30s 閲囨牱锛?--", stability_duration_secs);
    println!("  杩愯涓?..锛堟闃舵鑰楁椂 {}s锛?, stability_duration_secs);

    let stability_start = Instant::now();
    let baseline_qps = measure_qps(finalist, dim);
    println!("  baseline QPS: {}", baseline_qps);

    // 璁捐鏂囨。 F.10锛氳繛缁繍琛?5 鍒嗛挓锛屾瘡 30 绉掗噰鏍蜂竴娆?QPS
    // 鍏抽敭锛氬唴鏍稿繀椤绘寔缁繍琛岋紙涓嶈兘绌洪棽锛夛紝浠ユ娴?thermal throttling
    let kernel = finalist.build_kernel(raven::distance::DistanceMetric::L2);
    let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002) + 1.0).collect();

    // 棰勭儹
    for _ in 0..256 {
        let _ = kernel.distance(&a, &b);
    }

    let sample_interval = Duration::from_secs(30);
    let mut samples: Vec<(u64, u64)> = Vec::new(); // (elapsed_secs, qps)
    let mut total_count: u64 = 0;

    while stability_start.elapsed() < Duration::from_secs(stability_duration_secs) {
        let window_start = Instant::now();
        let mut window_count: u64 = 0;
        // 鎸佺画杩愯 30 绉掞紝涓嶅仠姝?        while window_start.elapsed() < sample_interval {
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
    println!("  sustained QPS:  {:.0} ({} 涓噰鏍峰钩鍧?", sustained_qps, samples.len());
    println!("  ratio:          {:.4}", ratio);
    println!("  閫氳繃鏍囧噯:       ratio >= 0.95");
    println!("  缁撴灉:           {}",
        if stable { "PASS" } else { "FAIL (闄嶇骇)" });
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

    // 淇濆瓨鍒扮紦瀛橈紙璁捐鏂囨。锛氱紦瀛樻渶浼橀厤缃紝棣栨閫夊畾鍚庤惤鐩橈級
    save_cache(dim, selected);

    print_summary(dim, selected, &passed_stage1, &qps_results, Some((ratio, stable)));
}

/// 淇濆瓨鍐呮牳閫夋嫨鍒扮紦瀛樻枃浠?fn save_cache(dim: usize, variant: KernelVariant) {
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
    println!("=== 鏈€缁堥€夋嫨缁撴灉 ===");
    println!("dim={}, kernel={}", dim, selected.name());
    println!();
    println!("闃舵 1锛堝欢杩熺矖绛涳級:");
    for (k, lat) in stage1 {
        println!("  {:<8} {}ns", k.name(), lat);
    }
    println!("闃舵 2锛堢灛鏃?QPS锛?");
    for (k, qps) in stage2 {
        println!("  {:<8} qps={}", k.name(), qps);
    }
    if let Some((ratio, stable)) = stability {
        println!("闃舵 3锛堢ǔ瀹氭€э級: ratio={:.4} stable={}", ratio, stable);
    }
    println!();
    println!("缂撳瓨璺緞: ~/.cache/raven/kernel_cache.toml");
    println!("锛堜笅娆″悓缁村害閫夋嫨灏嗙洿鎺ュ懡涓紦瀛橈紝璺宠繃涓夐樁娈电瓫閫夛級");
}

fn print_help() {
    println!("RAVEN 涓夐樁娈靛唴鏍搁€夋嫨宸ュ叿");
    println!();
    println!("鐢ㄦ硶:");
    println!("  kernel_select [OPTIONS]");
    println!();
    println!("閫夐」:");
    println!("  --dim <N>              缁村害锛堥粯璁?768锛?);
    println!("  --duration <S>         绋冲畾鎬ф祴璇曟椂闀跨鏁帮紙榛樿 300=5min锛?);
    println!("  --skip-stability       璺宠繃绗笁闃舵绋冲畾鎬ф祴璇?);
    println!("  -v, --verbose          璇︾粏杈撳嚭");
    println!("  -h, --help             鏄剧ず甯姪");
    println!();
    println!("鐜鍙橀噺:");
    println!("  RAVEN_KERNEL=scalar|avx2   寮哄埗鎸囧畾鍐呮牳");
    println!("  RAVEN_CACHE_DIR=<path>     缂撳瓨鐩綍");
}
