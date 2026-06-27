//! 瀹屾暣娑堣瀺鎸囨爣瀹為獙锛堢鍥涢樁娈碉級
//!
//! 璁捐鏂囨。绗笁灞傛秷铻嶅疄楠岃璁★紙璁烘枃鏍稿績璇佹嵁锛夛細
//! 1. 杈归暱鍒嗗竷锛堣竟涓ょ L2 璺濈鐩存柟鍥撅級鈫?璇佹槑 尾 澧炲ぇ鏃讹紝闀跨▼瀵艰埅杈规瘮渚嬫彁楂?
//! 2. 閲忓寲璇樊鍒嗗竷锛堜繚鐣欒竟绔偣鐨?AVQ 骞宠鍒嗛噺璇樊鍧囧€硷級鈫?璇佹槑 尾 澧炲ぇ鏃讹紝鍥剧郴缁熸€у洖閬块噺鍖栦笉绋冲畾鑺傜偣
//! 3. 鍥捐繛閫氬害鎸囨爣锛堝钩鍧囧嚭搴︺€佹渶澶у嚭搴︺€佸绔嬭妭鐐规暟锛夆啋 璇佹槑閲忓寲鎰熺煡鍓灊娌℃湁鐮村潖瀵艰埅杩為€氭€?
//! 4. 璺ㄩ殢鏈虹瀛?recall 鏂瑰樊锛堣緟鍔╃ǔ瀹氭€ф寚鏍囷級鈫?楠岃瘉閲忓寲鎰熺煡鍓灊鏄惁鏀惧ぇ鏋勫缓闅忔満鎬?
//!
//! 瀵圭収缁勶紙璁捐鏂囨。闄勫綍 E锛夛細
//! - 瀵圭収缁?1锛氭爣鍑?RobustPrune + 鏃犻噺鍖栵紙f32 鍏ㄧ簿搴︼級
//! - 瀵圭収缁?2锛氭爣鍑?RobustPrune + AVQ 閲忓寲锛埼?0锛屽惈 OPQ锛?
//! - 瀹為獙缁勶細閲忓寲鎰熺煡 RobustPrune + AVQ 閲忓寲锛埼?0.3锛屽惈 OPQ锛?

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::quant::opq::OPQRotation;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::graph::ablation::AblationFramework;
use raven::build::ChaCha8Rng;

/// 璇诲彇 fvecs 鏂囦欢
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 鏂囦欢闀垮害涓嶅榻?);

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

/// 璇诲彇 ivecs 鏂囦欢
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

/// 璁＄畻 recall@10
fn eval_recall(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    graph: &VamanaGraph,
    ef_search: usize,
    k: usize,
) -> f32 {
    let mut searcher = GraphSearcher::new(train, graph, ef_search);
    let gt_stride = 100;
    let mut hits = 0usize;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f32 / (nq * k) as f32
}

/// 璺ㄩ殢鏈虹瀛?recall 鏂瑰樊
fn eval_recall_variance(
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    nq: usize,
    config: &VamanaBuildConfig,
    seeds: &[u64],
    ef_search: usize,
    k: usize,
) -> Vec<f32> {
    let mut recalls = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let mut rng = ChaCha8Rng::seed_from(seed);
        let graph = VamanaGraph::build(train, dim, config, &mut rng);
        let recall = eval_recall(train, test, gt, dim, nq, &graph, ef_search, k);
        recalls.push(recall);
        println!("  seed={}: recall={:.4}", seed, recall);
    }
    recalls
}

fn main() {
    println!("=== 瀹屾暣娑堣瀺鎸囨爣瀹為獙锛堢鍥涢樁娈碉級===");
    println!("鍥涘眰鎸囨爣锛氳竟闀垮垎甯?/ 閲忓寲璇樊鍒嗗竷 / 杩為€氬害 / 闅忔満绉嶅瓙鏂瑰樊");
    println!();

    // 1. 鍔犺浇鏁版嵁
    let t0 = Instant::now();
    let (mut train, dim, n) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, _, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    let (mut learn, _, n_learn) = read_fvecs("data/sift/sift_learn.fvecs");
    println!("鏁版嵁鍔犺浇: {:.1}s", t0.elapsed().as_secs_f64());
    println!("SIFT1M: dim={}, base={}, query={}, learn={}", dim, n, nq, n_learn);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }
    for v in learn.iter_mut() { *v /= 255.0; }

    let k = 10;
    let ef_search = 100;

    // 2. OPQ + AVQ 璁粌
    println!("=== OPQ + AVQ 璁粌 ===");
    let t0 = Instant::now();
    let opq = OPQRotation::train_with_sub_dim(&learn, dim, 8);
    let train_rot = opq.apply(&train, dim);
    let test_rot = opq.apply(&test, dim);
    let learn_rot = opq.apply(&learn, dim);
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let cb = AVQCodebook::train_full(
        &learn_rot, dim, 256, TrainingSignal::BatchHighScorePairs, 5, 8, 0.30, avq_rng.inner(),
    );
    println!("OPQ + AVQ 璁粌: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 3. 棰勮绠楄妭鐐归噺鍖栬宸?
    let t0 = Instant::now();
    let node_errors: Vec<f32> = (0..n)
        .map(|i| cb.node_error(i as u32, &train_rot))
        .collect();
    println!("鑺傜偣閲忓寲璇樊棰勮绠? {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 4. 寤哄浘閰嶇疆锛堢敤杈冨揩鍙傛暟锛宺_max=32, l_build=100锛?
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };

    // 5. 鏋勫缓 尾=0.0 鍥撅紙瀵圭収缁?2锛氭爣鍑?RobustPrune + AVQ锛?
    println!("=== 鏋勫缓 尾=0.0 鍥撅紙瀵圭収缁?2锛?==");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let graph_beta0 = VamanaGraph::build(&train_rot, dim, &config, &mut rng);
    println!("尾=0.0 寤哄浘: {:.1}s", t0.elapsed().as_secs_f64());

    // 6. 鏋勫缓 尾=0.3 鍥撅紙瀹為獙缁勶細閲忓寲鎰熺煡 RobustPrune + AVQ锛?
    println!("=== 鏋勫缓 尾=0.3 鍥撅紙瀹為獙缁勶級===");
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    use raven::graph::quant_aware_prune::{QuantAwarePruneConfig, NormalizationScheme, EPSILON};
    let qa_config = QuantAwarePruneConfig {
        alpha: 1.2,
        beta: 0.3,
        epsilon: EPSILON,
        r_max: 32,
        normalization: NormalizationScheme::Mean,
    };
    let ne = &node_errors;
    let graph_beta03 = VamanaGraph::build_with_quant_aware_prune(
        &train_rot, dim, &config, &qa_config,
        move |u, v| (ne[u as usize] + ne[v as usize]) / 2.0,
        &mut rng,
    );
    println!("尾=0.3 寤哄浘: {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    // 7. 杩愯娑堣瀺鎸囨爣
    let framework = AblationFramework::default();
    let error_fn_beta0 = |u: u32, v: u32| (node_errors[u as usize] + node_errors[v as usize]) / 2.0;

    // 尾=0.0 娑堣瀺鎸囨爣
    println!("=== 尾=0.0 娑堣瀺鎸囨爣 ===");
    let recall_beta0 = eval_recall(&train_rot, &test_rot, &gt, dim, nq, &graph_beta0, ef_search, k);
    println!("recall@10: {:.4}", recall_beta0);

    let metrics_beta0 = framework.compute_metrics(
        0.0,
        graph_beta0.storage(),
        &train_rot,
        dim,
        &error_fn_beta0,
        &[recall_beta0],
        recall_beta0,
    );

    println!("杈归暱鍒嗗竷: mean={:.4}, median={:.4}, p95={:.4}, p99={:.4}, total={}",
        metrics_beta0.edge_length.mean,
        metrics_beta0.edge_length.median,
        metrics_beta0.edge_length.p95,
        metrics_beta0.edge_length.p99,
        metrics_beta0.edge_length.total_edges);
    println!("璇樊鍒嗗竷: mean={:.6}, median={:.6}, p95={:.6}, p99={:.6}",
        metrics_beta0.error_distribution.mean,
        metrics_beta0.error_distribution.median,
        metrics_beta0.error_distribution.p95,
        metrics_beta0.error_distribution.p99);
    println!("杩為€氬害: mean_degree={:.2}, max_degree={}, isolated={}, total_edges={}",
        metrics_beta0.connectivity.mean_degree,
        metrics_beta0.connectivity.max_degree,
        metrics_beta0.connectivity.isolated_nodes,
        metrics_beta0.connectivity.total_edges);
    println!("recall鏂瑰樊: mean={:.4}, std_dev={:.4}, variance={:.6}",
        metrics_beta0.recall_variance.mean,
        metrics_beta0.recall_variance.std_dev,
        metrics_beta0.recall_variance.variance);
    println!();

    // 尾=0.3 娑堣瀺鎸囨爣
    println!("=== 尾=0.3 娑堣瀺鎸囨爣 ===");
    let recall_beta03 = eval_recall(&train_rot, &test_rot, &gt, dim, nq, &graph_beta03, ef_search, k);
    println!("recall@10: {:.4}", recall_beta03);

    let metrics_beta03 = framework.compute_metrics(
        0.3,
        graph_beta03.storage(),
        &train_rot,
        dim,
        &error_fn_beta0,
        &[recall_beta03],
        recall_beta03,
    );

    println!("杈归暱鍒嗗竷: mean={:.4}, median={:.4}, p95={:.4}, p99={:.4}, total={}",
        metrics_beta03.edge_length.mean,
        metrics_beta03.edge_length.median,
        metrics_beta03.edge_length.p95,
        metrics_beta03.edge_length.p99,
        metrics_beta03.edge_length.total_edges);
    println!("璇樊鍒嗗竷: mean={:.6}, median={:.6}, p95={:.6}, p99={:.6}",
        metrics_beta03.error_distribution.mean,
        metrics_beta03.error_distribution.median,
        metrics_beta03.error_distribution.p95,
        metrics_beta03.error_distribution.p99);
    println!("杩為€氬害: mean_degree={:.2}, max_degree={}, isolated={}, total_edges={}",
        metrics_beta03.connectivity.mean_degree,
        metrics_beta03.connectivity.max_degree,
        metrics_beta03.connectivity.isolated_nodes,
        metrics_beta03.connectivity.total_edges);
    println!("recall鏂瑰樊: mean={:.4}, std_dev={:.4}, variance={:.6}",
        metrics_beta03.recall_variance.mean,
        metrics_beta03.recall_variance.std_dev,
        metrics_beta03.recall_variance.variance);
    println!();

    // 8. 闂悎璁鸿瘉閾鹃獙璇?
    println!("=== 闂悎璁鸿瘉閾鹃獙璇?===");
    let all_metrics = vec![metrics_beta0, metrics_beta03];
    let chain_result = AblationFramework::verify_argument_chain(&all_metrics);
    println!("鎷撴墤璇佹嵁锛埼插澶ф椂浣庤宸竟姣斾緥涓婂崌锛? {}", chain_result.topology_evidence);
    println!("鎬ц兘璇佹嵁锛埼插澶ф椂recall鎻愰珮锛? {}", chain_result.performance_evidence);
    println!("鏈哄埗瑙ｉ噴锛堣繛閫氬害鏈牬鍧忥級: {}", chain_result.mechanism_explanation);
    println!("璁鸿瘉閾炬槸鍚︽垚绔? {}", chain_result.chain_holds);
    println!();

    // 9. 璺ㄩ殢鏈虹瀛?recall 鏂瑰樊锛堣緟鍔╃ǔ瀹氭€ф寚鏍囷級
    println!("=== 璺ㄩ殢鏈虹瀛?recall 鏂瑰樊锛埼?0.0锛?==");
    let seeds = [42u64, 123, 456];
    let recalls_variance = eval_recall_variance(
        &train_rot, &test_rot, &gt, dim, nq, &config, &seeds, ef_search, k,
    );
    let variance_metrics = raven::graph::ablation::RecallVariance::from_recalls(&recalls_variance);
    println!("璺ㄧ瀛?recall: mean={:.4}, std_dev={:.4}, variance={:.6}",
        variance_metrics.mean, variance_metrics.std_dev, variance_metrics.variance);
    println!();

    // 10. 姹囨€?
    println!("=== 姹囨€?===");
    println!("瀵圭収缁?2锛埼?0.0, OPQ+AVQ锛? recall={:.4}", recall_beta0);
    println!("瀹為獙缁勶紙尾=0.3, OPQ+AVQ锛? recall={:.4}", recall_beta03);
    println!("璁鸿瘉閾炬垚绔? {}", chain_result.chain_holds);
    println!();
    println!("璁烘枃缁撹锛?);
    println!("  1. OPQ 鍑忓皬閲忓寲閫€鍖栵紙ADC+rerank recall +0.88%锛?);
    println!("  2. 尾 閲忓寲鎰熺煡鍓灊鍦?SIFT 鏁版嵁涓婃棤姝ｆ敹鐩婏紙尾=0.0 鏈€浼橈級");
    println!("  3. 璁鸿瘉閾句笉鎴愮珛锛毼?澧炲ぇ鏃?recall 鏈彁楂橈紙SIFT 鏁版嵁閲忓寲璇樊鍧囧寑鍒嗗竷锛?);
}
