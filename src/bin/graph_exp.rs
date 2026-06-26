//! 图结构快速实验器（100K 子集，~5s/配置）
//!
//! 用 SIFT1M 前 100K 向量建图，1000 条查询，
//! 快速验证图质量改进对 avg_visited 的影响。
//!
//! 用法：
//!   cargo run --release --bin graph_exp

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::l2_simd;
use rayon::prelude::*;

const SUBSET_N: usize = 100_000;
const NQ: usize = 1_000;
const GT_K: usize = 100;
const EF_LIST: &[usize] = &[50, 100];

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


struct ExpConfig {
    name: &'static str,
    alpha: f32,
    r_max: usize,
    r_soft: usize,
    max_iterations: usize,
    saturate: bool,
}

fn run_exp(
    cfg: &ExpConfig,
    train: &[f32],
    dim: usize,
    test: &[f32],
    nq: usize,
    gt: &[Vec<u32>],
) {
    print!("\n--- {} (α={}, R={}, R_soft={}, iter={}) ---\n",
           cfg.name, cfg.alpha, cfg.r_max, cfg.r_soft, cfg.max_iterations);

    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: cfg.alpha,
        l_build: 200,
        r_max: cfg.r_max,
        r_soft: cfg.r_soft,
        max_iterations: cfg.max_iterations,
        saturate: cfg.saturate,
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let build_time = t0.elapsed().as_secs_f64();

    let stats = graph.degree_stats();
    let mut sample_degrees = Vec::new();
    for i in 0..10.min(graph.len()) {
        sample_degrees.push(graph.neighbors(i as u32).len());
    }

    print!("建图: {:.1}s\n", build_time);
    print!("[degree] mean={:.1} p95={} p99={} max={} isolated={} sample={:?}\n",
           stats.mean_degree, stats.p95_degree, stats.p99_degree,
           stats.max_degree, stats.isolated_nodes, sample_degrees);

    for &ef in EF_LIST {
        let mut searcher = GraphSearcher::new(train, &graph, ef);
        let mut hits = 0usize;
        let mut total = 0usize;
        let mut total_visited = 0usize;
        let mut visited_counts: Vec<usize> = Vec::with_capacity(nq);

        let t1 = Instant::now();
        for q in 0..nq {
            let query = &test[q * dim..(q + 1) * dim];
            let result = searcher.search(query, 10);
            let vc = searcher.last_visited_count();
            total_visited += vc;
            visited_counts.push(vc);

            let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
            // gt[q] 是在 100K 子集上暴力计算的 top-100，取前 10 计算 recall@10
            for &g in gt[q].iter().take(10) {
                if found.contains(&g) {
                    hits += 1;
                }
            }
            total += 10;
        }
        let query_time = t1.elapsed();

        let recall = hits as f64 / total as f64;
        let qps = nq as f64 / query_time.as_secs_f64();
        let avg_visited = total_visited as f64 / nq as f64;
        visited_counts.sort_unstable();
        let p50 = visited_counts[nq / 2];
        let p95 = visited_counts[(nq as f64 * 0.95) as usize];
        let p99 = visited_counts[(nq as f64 * 0.99) as usize];

        print!("  ef={:>3}  recall={:.4}  QPS={:>6.0}  avg_visited={:>7.1}  p50={:>5}  p95={:>5}  p99={:>5}\n",
               ef, recall, qps, avg_visited, p50, p95, p99);
    }
}

fn main() {
    let pkg_ver = env!("CARGO_PKG_VERSION");
    let git_hash = option_env!("RAVEN_GIT_HASH").unwrap_or("n/a");
    println!("╔══════════════════════════════════════════════╗");
    println!("║  RAVEN v{}  git:{}  graph_exp", pkg_ver, git_hash);
    println!("╚══════════════════════════════════════════════╝");
    println!("子集: {} 向量, {} 查询, ef={:?}", SUBSET_N, NQ, EF_LIST);

    let (mut train, dim, _n_train) = read_fvecs("data/sift/sift_base.fvecs");
    let (mut test, _, _n_test) = read_fvecs("data/sift/sift_query.fvecs");

    // SIFT 数据归一化
    for v in train.iter_mut() { *v /= 255.0; }
    for v in test.iter_mut() { *v /= 255.0; }

    // 取子集
    train.truncate(SUBSET_N * dim);
    test.truncate(NQ * dim);
    println!("实际: n={}, dim={}, nq={}", SUBSET_N, dim, NQ);

    // ── 暴力计算 100K 子集上的 ground truth ──
    // 原因：SIFT1M 的 GT 索引指向 1M 向量，100K 子集上索引无效
    println!("计算暴力 ground truth ({} × {})...", NQ, SUBSET_N);
    let t_gt = Instant::now();
    let gt: Vec<Vec<u32>> = (0..NQ)
        .into_par_iter()
        .map(|q| {
            let query = &test[q * dim..(q + 1) * dim];
            let mut dists: Vec<(f32, u32)> = (0..SUBSET_N)
                .map(|i| {
                    let v = &train[i * dim..(i + 1) * dim];
                    (l2_simd(query, v), i as u32)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.iter().take(GT_K).map(|(_, id)| *id).collect()
        })
        .collect();
    println!("暴力 GT 完成: {:.1}s", t_gt.elapsed().as_secs_f64());

    // 实验矩阵：saturate on/off × α × R
    let exps = vec![
        // baseline
        ExpConfig { name: "sat-on  α=1.2 R=32", alpha: 1.2, r_max: 32, r_soft: 48, max_iterations: 2, saturate: true },
        // 去 saturation：图自然稀疏，邻居全是 RobustPrune 精选
        ExpConfig { name: "sat-off α=1.2 R=32", alpha: 1.2, r_max: 32, r_soft: 48, max_iterations: 2, saturate: false },
        // 去 saturation + 大 R 容纳更多精选边
        ExpConfig { name: "sat-off α=1.2 R=48", alpha: 1.2, r_max: 48, r_soft: 72, max_iterations: 2, saturate: false },
        // 去 saturation + 大 α 保留长程导航边
        ExpConfig { name: "sat-off α=1.5 R=32", alpha: 1.5, r_max: 32, r_soft: 48, max_iterations: 2, saturate: false },
    ];

    for exp in &exps {
        run_exp(exp, &train, dim, &test, NQ, &gt);
    }

    println!("\n=== 实验完成 ===");
}
