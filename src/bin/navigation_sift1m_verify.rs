//! NavigationLayer SIFT1M 闆嗘垚楠岃瘉瀹為獙
//!
//! 楠岃瘉 NavigationLayer 闆嗘垚鍒?GraphSearcher 鍚庡湪 SIFT1M 涓婄殑鎬ц兘
//! 瀵规瘮锛?
//!   A. GraphSearcher::new锛堥粯璁?medoid entry_point锛?
//!   B. GraphSearcher::new_with_navigation锛坈entroid entry_point锛?
//!
//! 鎸囨爣锛歳ecall@10, QPS, avg_latency
//! SIFT1M: 鈭歂=1000 涓?centroid锛屾壘鏈€杩?centroid 寮€閿€ O(1000*128)=128K flops/query

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, NavigationLayer, NavigationConfig};
use raven::build::ChaCha8Rng;

fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");
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

fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 ivecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 ivecs 澶辫触");
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    let mut gt = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for d in 0..dim {
            let v = i32::from_le_bytes(bytes[offset + d * 4..offset + d * 4 + 4].try_into().unwrap());
            gt.push(v);
        }
    }
    (gt, dim, n)
}

fn main() {
    println!("=== NavigationLayer SIFT1M 闆嗘垚楠岃瘉 ===");
    println!();

    // 1. 鍔犺浇 SIFT1M
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.1}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, gt_k={}", dim, n, nq, gt_k);

    // 褰掍竴鍖?
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;
    let gt_stride = gt_k;

    // 2. 鏋勫缓 VamanaGraph
    println!();
    println!("=== 鏋勫缓 VamanaGraph锛埼?1.2, r_max=32, l_build=100, max_iter=2锛?==");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(&train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();
    println!("寤哄浘鏃堕棿: {:.1}s ({:.0} vec/s)", build_time, n as f64 / build_time);
    println!("entry_point (medoid): {}", graph.entry_point());

    // 3. 鏋勫缓 NavigationLayer锛坈entroid overlay, 鈭歂=1000锛?
    println!();
    println!("=== 鏋勫缓 NavigationLayer锛坈entroid overlay, 鈭歂={}锛?==", (n as f64).sqrt() as usize);
    let t0 = Instant::now();
    let nav_config = NavigationConfig {
        enable_centroid_overlay: true,
        centroid_count: None, // 鈭歂
    };
    let nav = NavigationLayer::new(n, &train, dim, nav_config);
    let nav_build_time = t0.elapsed().as_secs_f64();
    println!("NavigationLayer 鏋勫缓: {:.1}s", nav_build_time);
    println!("centroid 鏁伴噺: {}", nav.centroids().len());

    // 4. A. 榛樿 medoid entry
    println!();
    println!("=== A. GraphSearcher::new锛堥粯璁?medoid entry锛?==");
    let t0 = Instant::now();
    let mut searcher_a = GraphSearcher::new(&train, &graph, ef_search);
    let mut hits_a = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher_a.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits_a += 1;
            }
        }
    }
    let time_a = t0.elapsed().as_secs_f64();
    let recall_a = hits_a as f64 / (nq * k) as f64;
    let qps_a = nq as f64 / time_a;
    println!("recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        recall_a, qps_a, time_a * 1000.0 / nq as f64);

    // 5. B. NavigationLayer centroid entry
    println!();
    println!("=== B. GraphSearcher::new_with_navigation锛坈entroid entry锛?==");
    let t0 = Instant::now();
    let mut searcher_b = GraphSearcher::new_with_navigation(&train, &graph, ef_search, &nav);
    let mut hits_b = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher_b.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits_b += 1;
            }
        }
    }
    let time_b = t0.elapsed().as_secs_f64();
    let recall_b = hits_b as f64 / (nq * k) as f64;
    let qps_b = nq as f64 / time_b;
    println!("recall@10={:.4}, QPS={:.0}, avg_latency={:.3}ms",
        recall_b, qps_b, time_b * 1000.0 / nq as f64);

    // 6. 姹囨€?
    println!();
    println!("=== 姹囨€?===");
    println!("{:<25} {:>10} {:>10} {:>12}", "鏂规", "recall@10", "QPS", "latency_ms");
    println!("{:-<57}", "");
    println!("{:<25} {:>10.4} {:>10.0} {:>12.3}", "A. medoid entry", recall_a, qps_a, time_a * 1000.0 / nq as f64);
    println!("{:<25} {:>10.4} {:>10.0} {:>12.3}", "B. centroid entry", recall_b, qps_b, time_b * 1000.0 / nq as f64);
    println!();

    let recall_diff = recall_b - recall_a;
    let qps_diff_pct = (qps_b - qps_a) / qps_a * 100.0;
    println!("宸紓: recall {:+.4}, QPS {:+.1}%", recall_diff, qps_diff_pct);
    println!();

    // 鍒ゅ畾
    if recall_b >= recall_a - 0.001 && qps_b >= qps_a * 1.02 {
        println!("缁撹: NavigationLayer 鍦?SIFT1M 涓婃湁姝ｅ悜鏀剁泭锛岄泦鎴愭垚鍔?);
    } else if recall_b >= recall_a - 0.001 && qps_b >= qps_a * 0.98 {
        println!("缁撹: NavigationLayer 鍦?SIFT1M 涓婃棤鏄庢樉鏀剁泭锛圦PS 宸紓 <2%锛夛紝浣?recall 涓嶅姡");
    } else if recall_b < recall_a - 0.001 {
        println!("缁撹: NavigationLayer 鍦?SIFT1M 涓?recall 涓嬮檷锛屼笉搴旈泦鎴?);
    } else {
        println!("缁撹: NavigationLayer 鍦?SIFT1M 涓?QPS 涓嬮檷锛坈entroid 鏌ユ壘寮€閿€ > visited 鍑忓皯锛夛紝涓嶅簲闆嗘垚");
    }
}
