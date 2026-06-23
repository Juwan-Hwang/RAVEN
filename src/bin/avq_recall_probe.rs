//! Week 6：AVQ 完整训练循环验证
//!
//! 目标：验证提升 codebook 容量 (K=256) + 增加训练轮次 (25) 后 recall 能否到 0.95+
//!
//! 如果 recall 仍 < 0.3，说明训练信号本身有问题，需要提前介入

use std::fs::File;
use std::io::Read;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use rand::Rng;
use raven::l2_simd;

/// 读取 fvecs 文件（siftsmall 格式）
/// 每个向量：4 字节 int (维度) + dim * 4 字节 float (数据)
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");

    let record_size = 4 + 128 * 4; // dim(4) + 128 floats
    let n = bytes.len() / record_size;
    let mut vectors = vec![0.0f32; n * 128];

    for i in 0..n {
        let offset = i * record_size;
        let dim = i32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        assert_eq!(dim, 128, "维度不是 128");
        for d in 0..128 {
            let off = offset + 4 + d * 4;
            vectors[i * 128 + d] = f32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]
            ]);
        }
    }
    (vectors, 128, n)
}

/// 读取 ivecs 文件（groundtruth 格式）
/// 每个记录：4 字节 int (维度=100) + dim * 4 字节 int (邻居 ID)
fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");

    let record_size = 4 + 100 * 4; // dim(4) + 100 ints
    let n = bytes.len() / record_size;
    let mut gt = vec![0i32; n * 100];

    for i in 0..n {
        let offset = i * record_size;
        let dim = i32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        assert_eq!(dim, 100, "groundtruth 维度不是 100");
        for d in 0..100 {
            let off = offset + 4 + d * 4;
            gt[i * 100 + d] = i32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]
            ]);
        }
    }
    (gt, 100, n)
}

fn quantized_recall(codebook: &AVQCodebook, train: &[f32], test: &[f32], gt: &[i32], dim: usize, n: usize, nq: usize, k: usize) -> f64 {
    let quantized: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_max: 32,
        r_soft: 48,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(&quantized, dim, &config, &mut rng);

    let searcher = GraphSearcher::new(&quantized, &graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100; // siftsmall groundtruth 每个查询 100 个邻居
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
    hits as f64 / (nq * k) as f64
}

/// ADC 验证：图用 f32 构建，搜索时数据库向量用量化重建，query 保持 f32
///
/// 这是 DiskANN/ScaNN 论文里 PQ/AVQ 的正确用法：
///   - 图结构由 f32 原始向量构建（结构正确）
///   - 搜索时距离计算用 query_f32 ↔ decode(encode(db_vec))（ADC 等价）
///   - 只有距离计算有量化误差，图导航本身不受量化影响
fn adc_recall(
    codebook: &AVQCodebook,
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
) -> f64 {
    // 1. 用 f32 原始向量建图（图结构正确）
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_max: 32,
        r_soft: 48,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);

    // 2. 构造量化重建的数据库向量（ADC：query f32 ↔ db 量化重建）
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 3. 搜索时用量化重建向量算距离，query 保持 f32
    let searcher = GraphSearcher::new(&quantized_db, &graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100; // siftsmall groundtruth 每个查询 100 个邻居
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
    hits as f64 / (nq * k) as f64
}

/// ADC + Reranking：量化粗筛 top-N + f32 精排 top-k
///
/// ScaNN/FAISS-IVF 标准做法：
///   1. f32 建图（图结构正确）
///   2. 量化距离搜索 top-N 候选（N >> k，粗筛快）
///   3. 对 top-N 候选用 f32 L2 精排，取 top-k
fn adc_recall_rerank(
    codebook: &AVQCodebook,
    train: &[f32],
    test: &[f32],
    gt: &[i32],
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
    top_n: usize,
) -> f64 {
    // 1. 用 f32 原始向量建图
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);

    // 2. 构造量化重建的数据库向量（ADC 粗筛）
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 3. 搜索 top-N 候选 + f32 rerank
    let searcher = GraphSearcher::new(&quantized_db, &graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        // 粗筛：量化距离搜 top-N
        let candidates = searcher.search(query, top_n);
        // 精排：f32 L2 重算距离（SIMD 加速）
        let mut reranked: Vec<(u32, f32)> = candidates
            .iter()
            .map(|(id, _)| {
                let v = &train[*id as usize * dim..(*id as usize + 1) * dim];
                (*id, l2_simd(query, v))
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let found: Vec<u32> = reranked.iter().take(k).map(|(id, _)| *id).collect();
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

fn main() {
    println!("=== Week 6：siftsmall 真实数据 AVQ recall 验证 ===");
    println!("目标：recall@10 > 0.95");
    println!();

    // 加载 siftsmall 数据集
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_nq={}, gt_k={}",
        dim, n, nq, gt_nq, gt_k);
    // 诊断：检查 gt 是否 1-indexed
    print!("gt[0..10]={}", "");
    for d in 0..10 { print!("{} ", gt[d]); }
    println!();

    // 归一化到 [0,1]（SIFT 原始 0-255，梯度爆炸导致训练发散）
    // L2 距离排序在均匀缩放下不变，f32 recall 不受影响
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }
    println!("数据归一化: /{} → [0,1]", max_val);
    println!();

    // 基线：f32 原始（不量化）
    let recall_f32 = f32_recall(&train, &test, &gt, dim, n, nq, 10);
    println!("f32 baseline（图+搜索全 f32）: recall@10={:.4}", recall_f32);
    println!();
    println!("目标：recall@10 > 0.95（跳过 α 消融加速验证）");
}

/// f32 建图 + f32 查询（不量化）
/// gt_stride: groundtruth 每个查询的邻居数（siftsmall=100），取前 k 个计算 recall
fn f32_recall(train: &[f32], test: &[f32], gt: &[i32], dim: usize, n: usize, nq: usize, k: usize) -> f64 {
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let searcher = GraphSearcher::new(train, &graph, 100);

    // 诊断：打印第一个查询的搜索候选数 + 图平均度数
    if nq > 0 {
        let query = &test[0..dim];
        let (candidates, visited) = VamanaGraph::greedy_search_vec(
            train, dim, graph.storage(), graph.entry_point(), query, 100,
        );
        eprintln!("[diag] ef=100 top={}, visited={}", candidates.len(), visited.len());
        let mut total_deg = 0usize;
        for i in 0..n { total_deg += graph.storage().degree(i as u32); }
        eprintln!("[diag] 图平均度数={:.1}", total_deg as f64 / n as f64);
    }
    // 诊断：打印第一个查询的结果 vs groundtruth
    if nq > 0 {
        let query = &test[0..dim];
        let result = searcher.search(query, k);
        print!("search[0] top-{}: ", k);
        for (id, _) in result.iter().take(k) { print!("{} ", id); }
        println!();
        print!("gt[0] top-{}:     ", k);
        for d in 0..k { print!("{} ", gt[d]); }
        println!();
        eprintln!("graph: n={}, entry={}", graph.len(), graph.entry_point());
    }

    let mut hits = 0usize;
    let gt_stride = 100; // siftsmall groundtruth 每个查询 100 个邻居
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        // gt 每个查询有 gt_stride 个邻居，取前 k 个计算 recall@k
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}
