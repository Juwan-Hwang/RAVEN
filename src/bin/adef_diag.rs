//! AdaptiveEf 诊断工具 — 打印实际 ef 分配分布，确认链路是否真正生效
//!
//! 用法：cargo run --release --bin adef_diag

use std::fs::File;
use std::io::Read;

use raven::build::ChaCha8Rng;
use raven::graph::{AdaptiveEfConfig, GraphSearcher, VamanaBuildConfig, VamanaGraph, PruneStrategy};
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

fn main() {
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    // 加载缓存的 DirectionalPrune 图
    let graph_path = std::path::Path::new("data/sift/graph_cache_dir_rmin4.bin");
    let graph = if graph_path.exists() {
        eprintln!("loading cached DirectionalPrune graph...");
        VamanaGraph::load(graph_path).expect("load")
    } else {
        eprintln!("building DirectionalPrune graph...");
        let mut rng = ChaCha8Rng::seed_from(42);
        let config = VamanaBuildConfig {
            alpha: 1.2, l_build: 200, r_max: 32, r_soft: 48,
            max_iterations: 2, saturate: false,
            enable_layered_nav: true, nav_m: 16,
            prune_strategy: PruneStrategy::DirectionalPrune,
            ..Default::default()
        };
        let g = VamanaGraph::build(&train, dim, &config, &mut rng);
        let _ = g.save(graph_path);
        g
    };

    let sq8 = SQ8Dataset::build(&train, dim);
    let k = 10usize;
    let ef_search = 50usize;

    // 构建 AdaptiveEf 配置（与 opt_bench --adaptive-ef 完全一致）
    let nav = graph.layered_nav().expect("layered nav required");
    let ac = AdaptiveEfConfig::build_with_layered_nav(
        &train, dim, nav, 40, 75, 3.0);

    // 打印距离分布统计
    let (dmin, dp25, dmed, dp75, dmax) = ac.distribution_stats();
    eprintln!("\n=== AdaptiveEf 诊断 ===");
    eprintln!("样本数: {}", ac.sample_count());
    eprintln!("距离分布: min={:.4} p25={:.4} med={:.4} p75={:.4} max={:.4}",
        dmin, dp25, dmed, dp75, dmax);
    eprintln!("参数: min_ef={} max_ef={} gamma={}", 40, 75, 3.0);

    // 对所有查询计算实际分配的 ef
    let mut ef_list: Vec<usize> = Vec::with_capacity(nq);
    let mut ef_counts: [usize; 76] = [0; 76]; // ef 0..75

    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let (ep, dist) = nav.initialize(&train, dim, query);
        let ef = ac.estimate_ef(dist).max(k);
        ef_list.push(ef);
        if ef < 76 { ef_counts[ef] += 1; }
    }

    ef_list.sort();
    let avg_ef = ef_list.iter().sum::<usize>() as f64 / nq as f64;
    let min_ef_actual = ef_list[0];
    let max_ef_actual = ef_list[nq - 1];
    let med_ef = ef_list[nq / 2];

    eprintln!("\n--- 实际 ef 分配（{} 查询）---", nq);
    eprintln!("min={}  median={}  avg={:.1}  max={}", min_ef_actual, med_ef, avg_ef, max_ef_actual);

    // 打印 ef 分布直方图
    eprintln!("\n--- ef 分布直方图 ---");
    for ef in 40..=75 {
        let count = ef_list.iter().filter(|&&e| e == ef).count();
        if count > 0 {
            let bar = "#".repeat((count * 80 / nq).max(1));
            eprintln!("  ef={:>3}  {:>5}  ({:>5.1}%)  {}", ef, count, count as f64 * 100.0 / nq as f64, bar);
        }
    }

    // 关键验证：如果 avg_ef ≈ ef_search(50)，说明 AdaptiveEf 在 DirectionalPrune 图上
    // 基本没有改变 ef 分配，这就是为什么看不到 QPS 差异
    eprintln!("\n--- 结论 ---");
    if (avg_ef - ef_search as f64).abs() < 3.0 {
        eprintln!("⚠️  avg_ef={:.1} ≈ 固定 ef={}", avg_ef, ef_search);
        eprintln!("   AdaptiveEf 在 DirectionalPrune 图上几乎不改变 ef 分配！");
        eprintln!("   原因：DirectionalPrune 图的入口距离分布极窄，");
        eprintln!("   幂律变换后几乎所有查询都落在 ef≈{} 附近。", ef_search);
    } else if avg_ef < ef_search as f64 {
        eprintln!("✅ avg_ef={:.1} < 固定 ef={}，AdaptiveEf 正在降低平均 ef", avg_ef, ef_search);
        eprintln!("   但 QPS 差异小可能是因为 DirectionalPrune 图遍历本身已足够快。");
    } else {
        eprintln!("✅ avg_ef={:.1} > 固定 ef={}，AdaptiveEf 正在增大平均 ef", avg_ef, ef_search);
    }

    // 额外：实际跑一遍搜索，确认 last_ef_used 确实在变
    eprintln!("\n--- 搜索器验证（前 20 查询的 last_ef_used）---");
    let mut searcher = GraphSearcher::new(&train, &graph, ef_search);
    searcher.with_sq8(&sq8);
    searcher.with_adaptive_ef(ac);

    for q in 0..20.min(nq) {
        let query = &test[q * dim..(q + 1) * dim];
        let _ = searcher.search_sq8(query, k);
        eprintln!("  q{:>3}: last_ef_used={}", q, searcher.last_ef_used());
    }
}
