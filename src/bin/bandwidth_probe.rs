//! 鍐呭瓨甯﹀鐡堕鍒嗘瀽鎺㈤拡
//!
//! 璁捐鏂囨。 Week 3-4锛氬唴瀛樺甫瀹界摱棰堝垎鏋愶紙LLC miss / bandwidth counters锛?//! 鍒ゆ柇 compute-bound vs memory-bound锛?//!
//! 鏈帰閽堥€氳繃鎵弿 working set 澶у皬锛屾祴閲忎笉鍚?cache 灞傜骇涓嬬殑鏈夋晥甯﹀锛?//! 浠庤€屽垽鏂?L2 璺濈鏍告槸 compute-bound 杩樻槸 memory-bound銆?//!
//! 鍒ゅ畾閫昏緫锛?//! - 鑻ュ甫瀹介殢 working set 澧炲ぇ鑰屾樉钁椾笅闄嶏紙L1鈫扡2鈫扡3鈫扲AM锛夆啋 memory-bound
//! - 鑻ュ甫瀹藉湪鍚勫眰绾т繚鎸佺ǔ瀹?鈫?compute-bound
//! - 瀵规瘮绠楁湳寮哄害锛歀2 璺濈 3 FLOPs / 8 bytes = 0.375 FLOPs/byte锛?//!   杩滀綆浜?compute-bound 闂ㄦ锛垀10 FLOPs/byte锛夛紝鐞嗚涓婂簲涓?memory-bound

use std::time::Instant;
use raven::distance::l2_dynamic;

/// 娴嬮噺缁欏畾 working set 涓嬬殑 L2 璺濈鍚炲悙
///
/// n_pairs: 鍚屾椂璁＄畻鐨勫悜閲忓鏁帮紙鎺у埗 working set 澶у皬锛?/// dim: 缁村害
/// iterations: 杩唬娆℃暟锛堝ぇ working set 鏃跺簲鍑忓皯浠ユ帶鍒舵€昏€楁椂锛?fn measure_bandwidth(n_pairs: usize, dim: usize, iterations: usize) -> BandwidthResult {
    // 鍑嗗鏁版嵁锛歯_pairs 瀵瑰悜閲?    let total_bytes = n_pairs * 2 * dim * std::mem::size_of::<f32>();
    let vectors_a: Vec<f32> = (0..n_pairs * dim).map(|i| i as f32 * 0.001).collect();
    let vectors_b: Vec<f32> = (0..n_pairs * dim).map(|i| (i as f32 * 0.002) + 1.0).collect();

    // 棰勭儹锛氱‘淇濇暟鎹繘鍏ュ搴?cache 灞傜骇
    let mut sink = 0.0f32;
    for _ in 0..1000.min(n_pairs * 10) {
        for p in 0..n_pairs {
            let a = &vectors_a[p * dim..(p + 1) * dim];
            let b = &vectors_b[p * dim..(p + 1) * dim];
            sink += l2_dynamic(a, b);
        }
    }
    std::hint::black_box(sink);

    // 瀹炴祴
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
    println!("=== RAVEN 鍐呭瓨甯﹀鐡堕鍒嗘瀽 ===");
    println!("CPU: AMD Ryzen 7 8845H (Zen 4)");
    println!("Cache: L1d=32KB, L2=1MB, L3=16MB");
    println!();
    println!("L2 璺濈绠楁湳寮哄害: 3 FLOPs / 8 bytes = 0.375 FLOPs/byte");
    println!("Compute-bound 闂ㄦ: ~10 FLOPs/byte");
    println!();

    // 娴嬭瘯 1锛氬浐瀹?dim=768锛屾壂鎻?working set 澶у皬
    // 鎻ず cache 灞傜骇瀵瑰甫瀹界殑褰卞搷
    println!("=== 娴嬭瘯 1: 鍥哄畾 dim=768, 鎵弿 working set ===");
    println!("{:>10} {:>8} {:>10} {:>12} {:>10} {:>10} {:>8}",
        "pairs", "dim", "WS(KB)", "cache", "BW(GB/s)", "GFLOPs", "lat(ns)");
    println!("{}", "-".repeat(72));

    let dim = 768;
    // working set = n_pairs 脳 2 脳 768 脳 4 bytes
    // 1 pair = 6KB (L1), 5 pairs = 30KB (L1), 10 pairs = 60KB (L2),
    // 100 pairs = 600KB (L2), 1000 pairs = 6MB (L3), 5000 pairs = 30MB (RAM)
    let pair_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096];

    let mut l1_bw: f64 = 0.0;
    let mut ram_bw: f64 = 0.0;

    for &n_pairs in &pair_counts {
        // 澶?working set 鍑忓皯杩唬娆℃暟浠ユ帶鍒舵€昏€楁椂
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
    println!("=== 娴嬭瘯 2: 鍥哄畾 working set 鍦?L1, 鎵弿缁村害 ===");
    println!("{:>8} {:>10} {:>12} {:>10} {:>10} {:>8}",
        "dim", "WS(KB)", "cache", "BW(GB/s)", "GFLOPs", "lat(ns)");
    println!("{}", "-".repeat(62));

    // 鍥哄畾 1 pair锛岀‘淇濋兘鍦?L1
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
    println!("=== 鍒嗘瀽缁撹 ===");
    let bw_ratio = if ram_bw > 0.0 { l1_bw / ram_bw } else { 0.0 };
    println!("L1 宄板€煎甫瀹? {:.1} GB/s", l1_bw);
    println!("RAM 宄板€煎甫瀹? {:.1} GB/s", ram_bw);
    println!("L1/RAM 甯﹀姣? {:.1}x", bw_ratio);
    println!();

    if bw_ratio > 3.0 {
        println!(">>> 鍒ゅ畾: MEMORY-BOUND <<<");
        println!("甯﹀闅?working set 璺?cache 灞傜骇鏄捐憲涓嬮檷锛坽:.1}x锛夛紝", bw_ratio);
        println!("璇存槑鏁版嵁鎼繍鏄摱棰堛€侫VX2 鍔犲鎸囦护鍦ㄦ鍦烘櫙鏀剁泭鏈夐檺锛?);
        println!("浼樺寲閲嶅績搴斿厛鏀惧湪锛氬竷灞€浼樺寲 + 棰勫彇 + 鍑忓皯鏁版嵁鎼繍銆?);
    } else {
        println!(">>> 鍒ゅ畾: COMPUTE-BOUND 鎴?OVERHEAD-BOUND <<<");
        println!("甯﹀鍦ㄥ悇 cache 灞傜骇淇濇寔绋冲畾锛岀摱棰堝湪璁＄畻鎴栧惊鐜紑閿€銆?);
        println!("AVX2 鍔犲鎸囦护鍙洿鎺ユ彁鍗囧悶鍚愶紝搴斾紭鍏堝紩鍏ャ€?);
    }
    println!();
    println!("=== 鐞嗚鍙傝€?===");
    println!("Zen 4 宄板€煎甫瀹? L1~288GB/s, L2~144GB/s, L3~72GB/s, RAM~50GB/s");
    println!("褰撳墠 L1 鏈夋晥甯﹀ {:.1} GB/s = L1 宄板€肩殑 {:.1}%", l1_bw, l1_bw / 288.0 * 100.0);
    println!("鑻ユ湁鏁堝甫瀹?<< 宄板€煎甫瀹斤紝璇存槑璁＄畻/寰幆寮€閿€鏄摱棰堬紙compute-bound锛?);
    println!("鑻ユ湁鏁堝甫瀹?鈮?宄板€煎甫瀹斤紝璇存槑鏁版嵁鎼繍鏄摱棰堬紙memory-bound锛?);
}
