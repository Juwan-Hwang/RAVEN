//! 闄勫綍 A 閫€鍖栧垽瀹氬疄楠?
//!
//! 璁捐鏂囨。闄勫綍 A锛歊P-Tuning 棰濆瀛樺偍鏂规涓夐€変竴
//! 閫€鍖栧垽瀹氶槇鍊硷紙瀹為獙鍓嶉攣瀹氾級锛?
//!   鎸囨爣涓€锛堝浐瀹?QPS锛夛細A 鏂规 recall@10 宸窛 < 0.5%
//!   鎸囨爣浜岋紙鍥哄畾 recall锛夛細A 鏂规 QPS 宸窛 < 3%
//!   瑕嗙洊鑼冨洿锛氳嚦灏?3 涓笉鍚屾暟鎹泦锛屾瘡涓嫭绔嬪垽瀹?
//!
//! 瀹為獙璁捐锛?
//!   1. 瀹屾暣閲嶅缓鐗堟湰锛坆aseline锛夛細鍒嗗埆鐢?伪=1.0/1.5/2.0 浠庨浂鏋勫缓
//!   2. A 鏂规锛氬缓涓€娆?伪=1.2 鍩虹鍥撅紝RP-Tuning 鍚庨獙鐢熸垚 伪=1.0/1.5/2.0 鍙樹綋
//!   3. 瀵规瘮鍚岀瓑 QPS 涓嬬殑 recall@10 宸窛

use std::time::Instant;
use rand::Rng;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::rp_tuning::{RPTuning, RPTuningConfig, RPTuningStorageScheme};
use raven::build::ChaCha8Rng;

/// 瀹為獙缁撴灉
#[derive(Debug, Clone)]
struct ExperimentResult {
    alpha: f32,
    method: String,  // "rebuild" or "rp_tuning_A"
    build_time_s: f64,
    qps: f64,
    recall: f64,
}

/// 鐢熸垚甯﹁仛绫荤粨鏋勭殑鍚堟垚鏁版嵁闆嗭紙妯℃嫙鐪熷疄鍒嗗竷锛?
fn generate_clustered_data(
    n: usize,
    dim: usize,
    nq: usize,
    k: usize,
    n_clusters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from(seed);
    let mut train = vec![0.0f32; n * dim];
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);

    // 鐢熸垚鑱氱被涓績
    for _ in 0..n_clusters {
        let c: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0).collect();
        centroids.push(c);
    }

    // 姣忎釜鍚戦噺 = 鑱氱被涓績 + 楂樻柉鍣０
    for i in 0..n {
        let cluster = i % n_clusters;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            train[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    // 鐢熸垚鏌ヨ鍚戦噺
    let mut test = vec![0.0f32; nq * dim];
    for i in 0..nq {
        let cluster = (i % n_clusters) as usize;
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0;
            test[i * dim + d] = centroids[cluster][d] + noise;
        }
    }

    // 鏆村姏璁＄畻 ground truth
    let mut gt = vec![0i32; nq * k];
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &train[i * dim..(i + 1) * dim];
                let d: f32 = v.iter().zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for j in 0..k {
            gt[q * k + j] = dists[j].0 as i32;
        }
    }

    (train, test, gt)
}

/// 杩愯鏌ヨ骞惰绠?recall
fn run_queries(
    train: &[f32],
    graph: &VamanaGraph,
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    k: usize,
    ef_search: usize,
) -> (f64, f64) {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let start = Instant::now();
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * k..(q + 1) * k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    let elapsed = start.elapsed();
    let qps = nq as f64 / elapsed.as_secs_f64();
    let recall = hits as f64 / (nq * k) as f64;
    (qps, recall)
}

fn main() {
    println!("=== 闄勫綍 A 閫€鍖栧垽瀹氬疄楠?===");
    println!("璁捐鏂囨。闄勫綍 A锛歊P-Tuning 棰濆瀛樺偍鏂规涓夐€変竴");
    println!("閫€鍖栧垽瀹氶槇鍊硷細recall@10 宸窛 < 0.5%, QPS 宸窛 < 3%");
    println!();

    // 3 涓暟鎹泦锛堜笉鍚岃妯?缁村害锛屾弧瓒?鑷冲皯 3 涓笉鍚屾暟鎹泦"瑕佹眰锛?
    let datasets = [
        ("dataset_1", 1000usize, 128usize, 100usize, 10usize, 20),
        ("dataset_2", 2000, 256, 100, 10, 30),
        ("dataset_3", 3000, 128, 100, 10, 40),
    ];

    let mut all_pass = true;

    for (name, n, dim, nq, k, n_clusters) in &datasets {
        println!("=== {} (n={}, dim={}, nq={}, k={}, clusters={}) ===", name, n, dim, nq, k, n_clusters);
        let (train, test, gt) = generate_clustered_data(*n, *dim, *nq, *k, *n_clusters, 42);

        let alpha_points = vec![1.0f32, 1.5, 2.0];
        let ef_search = 100;

        // === Baseline: 瀹屾暣閲嶅缓 ===
        println!("[Baseline] 瀹屾暣閲嶅缓锛堝垎鍒敤涓嶅悓 伪 浠庨浂鏋勫缓锛?);
        let mut rebuild_results: Vec<ExperimentResult> = Vec::new();
        for &alpha in &alpha_points {
            let mut rng = ChaCha8Rng::seed_from(42);
            let config = VamanaBuildConfig {
                alpha,
                l_build: 200,
                r_max: 64,
                r_soft: 96,
                max_iterations: 1,
..Default::default()
            };
            let start = Instant::now();
            let graph = VamanaGraph::build(&train, *dim, &config, &mut rng);
            let build_time = start.elapsed().as_secs_f64();

            let (qps, recall) = run_queries(&train, &graph, &test, &gt, *dim, *nq, *k, ef_search);
            println!("  伪={:.1}: build={:.3}s, QPS={:.0}, recall@{}={:.4}",
                alpha, build_time, qps, k, recall);
            rebuild_results.push(ExperimentResult {
                alpha,
                method: "rebuild".to_string(),
                build_time_s: build_time,
                qps,
                recall,
            });
        }

        // === A 鏂规锛歊P-Tuning ===
        println!("[Scheme A] RP-Tuning锛堝缓涓€娆?伪=1.2 鍩虹鍥撅紝鍚庨獙鐢熸垚鍙樹綋锛?);
        let mut rng = ChaCha8Rng::seed_from(42);
        let base_config = VamanaBuildConfig {
            alpha: 1.2,
            l_build: 200,
            r_max: 64,
            r_soft: 96,
            max_iterations: 1,
..Default::default()
        };
        let start = Instant::now();
        let base_graph = VamanaGraph::build(&train, *dim, &base_config, &mut rng);
        let base_build_time = start.elapsed().as_secs_f64();
        println!("  base graph (伪=1.2): build={:.3}s", base_build_time);

        let rp_config = RPTuningConfig {
            scheme: RPTuningStorageScheme::SchemeA,
            alpha_points: alpha_points.clone(),
            r_max: 64,
        };
        let rp_start = Instant::now();
        let variants = RPTuning::generate_variants(&base_graph, &train, *dim, &rp_config);
        let rp_time = rp_start.elapsed().as_secs_f64();
        println!("  RP-Tuning generate_variants: {:.3}s ({} variants)", rp_time, variants.len());

        let mut rp_results: Vec<ExperimentResult> = Vec::new();
        for variant in &variants {
            let graph = variant.clone().into_graph(*dim);
            let stats = graph.degree_stats();
            println!("  伪={:.1}: degree stats: mean={:.1}, p95={}, p99={}, max={}, isolated={}, overflow_ratio={:.4}",
                variant.alpha, stats.mean_degree, stats.p95_degree, stats.p99_degree,
                stats.max_degree, stats.isolated_nodes, stats.overflow_ratio);
            let (qps, recall) = run_queries(&train, &graph, &test, &gt, *dim, *nq, *k, ef_search);
            println!("  伪={:.1}: QPS={:.0}, recall@{}={:.4}",
                variant.alpha, qps, k, recall);
            rp_results.push(ExperimentResult {
                alpha: variant.alpha,
                method: "rp_tuning_A".to_string(),
                build_time_s: base_build_time + rp_time,
                qps,
                recall,
            });
        }

        // === 閫€鍖栧垽瀹?===
        println!("[閫€鍖栧垽瀹歖");
        let mut dataset_pass = true;
        for i in 0..alpha_points.len() {
            let rebuild = &rebuild_results[i];
            let rp = &rp_results[i];

            // recall 閫€鍖栵細鍙湪 Scheme A recall 浣庝簬 baseline 鏃剁畻閫€鍖?
            // Scheme A recall 鏇撮珮鏄紭鍔匡紝涓嶇畻閫€鍖?
            let recall_drop = if rp.recall < rebuild.recall {
                rebuild.recall - rp.recall
            } else {
                0.0  // Scheme A 鏇村ソ锛屼笉绠楅€€鍖?
            };

            // QPS 閫€鍖栵細鍙湪 Scheme A QPS 浣庝簬 baseline 鏃剁畻閫€鍖?
            let qps_drop = if rp.qps < rebuild.qps {
                (rebuild.qps - rp.qps) / rebuild.qps
            } else {
                0.0  // Scheme A 鏇村揩锛屼笉绠楅€€鍖?
            };

            let recall_ok = recall_drop < 0.005;  // recall 涓嬮檷 < 0.5%
            let qps_ok = qps_drop < 0.03;  // QPS 涓嬮檷 < 3%

            println!("  伪={:.1}: recall_drop={:.4} ({}), qps_drop={:.2}%, build_time: {}={:.3}s vs {}={:.3}s, {}",
                rebuild.alpha,
                recall_drop,
                if recall_ok { "PASS" } else { "FAIL" },
                qps_drop * 100.0,
                rebuild.method, rebuild.build_time_s,
                rp.method, rp.build_time_s,
                if qps_ok { "PASS" } else { "FAIL" }
            );

            if !recall_ok || !qps_ok {
                dataset_pass = false;
            }
        }

        if dataset_pass {
            println!("  鈫?{} PASS锛圓 鏂规涓嶉€€鍖栵級", name);
        } else {
            println!("  鈫?{} FAIL锛圓 鏂规閫€鍖栵紝闇€闄嶇骇 B锛?, name);
            all_pass = false;
        }
        println!();
    }

    // === 鏈€缁堢粨璁?===
    println!("=== 鏈€缁堢粨璁?===");
    if all_pass {
        println!("鎵€鏈夋暟鎹泦鍧囬€氳繃閫€鍖栧垽瀹?);
        println!("鈫?閫夊畾鏂规 A锛歾ero-cost RP-Tuning");
        println!("鈫?璁烘枃浜偣锛歊P-Tuning 鏃犻澶栧瓨鍌ㄤ唬浠凤紝涓嶉€€鍖?);
    } else {
        println!("瀛樺湪鏁版嵁闆嗘湭閫氳繃閫€鍖栧垽瀹?);
        println!("鈫?闄嶇骇鏂规 B锛氭瘡鑺傜偣瀛樺偍琚壀鎺夌殑閭诲眳 ID");
    }
}
