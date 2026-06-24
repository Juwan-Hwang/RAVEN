//! f16 精度验证实验（第三阶段）
//!
//! 目标：验证 f16 在 SIFT1M 上的精度损失
//!
//! 流程：
//! 1. 加载 SIFT1M 数据
//! 2. 用 f32 暴力搜索得到 groundtruth
//! 3. 用 f16 暴力搜索对比 recall
//! 4. 测量 f16 vs f32 的 QPS 差异
//!
//! 判断标准：
//! - f16 recall@10 相对 f32 损失 < 1% → f16 可用
//! - f16 QPS 提升 > 10% → 带宽收益显著

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::{l2_simd, l2_f16, l2_f16_mixed, f32_to_f16_slice};

/// 读取 fvecs 文件
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 fvecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 fvecs 失败");

    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record_bytes = (4 + dim * 4) as usize;
    let n = bytes.len() / record_bytes;
    assert_eq!(bytes.len() % record_bytes, 0, "fvecs 文件长度不对齐");

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

/// 读取 ivecs 文件
fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("无法打开 ivecs 文件");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("读取 ivecs 失败");

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
    println!("=== f16 精度验证实验（第三阶段）===");
    println!();

    // 1. 加载数据（用 query 集做暴力搜索，因为 base 太大暴力搜索太慢）
    // 用 sift_learn (100K) 作为数据库，sift_query (10K) 作为查询
    let t0 = Instant::now();
    let (mut db, dim, n_db) = read_fvecs("data/sift/sift_learn.fvecs");
    let (mut queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (gt, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("数据加载: {:.1}s", t0.elapsed().as_secs_f64());
    println!("db(learn): {} vecs, queries: {} vecs, dim={}, gt_k={}", n_db, nq, dim, gt_k);
    println!();

    // 归一化到 [0,1]
    for v in db.iter_mut() { *v /= 255.0; }
    for v in queries.iter_mut() { *v /= 255.0; }

    let k = 10;

    // 2. 预量化数据库为 f16
    println!("=== 预量化数据库为 f16 ===");
    let t0 = Instant::now();
    let db_f16 = f32_to_f16_slice(&db);
    println!("f16 预量化: {:.1}s ({} vecs)", t0.elapsed().as_secs_f64(), n_db);
    println!("内存占用: f32={}MB, f16={}MB (节省 50%)",
        db.len() * 4 / 1024 / 1024,
        db_f16.len() * 2 / 1024 / 1024);
    println!();

    // 3. f32 暴力搜索（基线）
    println!("=== f32 暴力搜索（基线）===");
    let t0 = Instant::now();
    let mut f32_hits = 0usize;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let mut best: Vec<(u32, f32)> = (0..n_db)
            .map(|i| {
                let v = &db[i * dim..(i + 1) * dim];
                (i as u32, l2_simd(query, v))
            })
            .collect();
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let found: Vec<u32> = best.iter().take(k).map(|(id, _)| *id).collect();
        // 注意：learn 集不是真正的 groundtruth，这里用 f32 结果作为 groundtruth 对比 f16
        if q == 0 {
            // 只对第一个 query 记录，用于对比
        }
    }
    let f32_time = t0.elapsed().as_secs_f64();
    let f32_qps = nq as f64 / f32_time;
    println!("f32 搜索: {:.2}s, QPS={:.0}", f32_time, f32_qps);
    println!();

    // 4. f16 混合模式搜索（query=f32, db=f16_packed）
    println!("=== f16 混合模式搜索（query=f32, db=f16_packed）===");
    let t0 = Instant::now();
    let mut f16_matches = 0usize;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        // f32 搜索得到 top-k
        let mut f32_best: Vec<(u32, f32)> = (0..n_db)
            .map(|i| {
                let v = &db[i * dim..(i + 1) * dim];
                (i as u32, l2_simd(query, v))
            })
            .collect();
        f32_best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let f32_topk: Vec<u32> = f32_best.iter().take(k).map(|(id, _)| *id).collect();

        // f16 搜索得到 top-k
        let mut f16_best: Vec<(u32, f32)> = (0..n_db)
            .map(|i| {
                let v = &db_f16[i * dim..(i + 1) * dim];
                (i as u32, l2_f16_mixed(query, v))
            })
            .collect();
        f16_best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let f16_topk: Vec<u32> = f16_best.iter().take(k).map(|(id, _)| *id).collect();

        // 计算 f16 相对 f32 的 recall
        for &id in &f16_topk {
            if f32_topk.contains(&id) {
                f16_matches += 1;
            }
        }
    }
    let f16_time = t0.elapsed().as_secs_f64();
    let f16_qps = nq as f64 / f16_time;
    let f16_recall = f16_matches as f64 / (nq * k) as f64;

    println!("f16 混合模式搜索: {:.2}s, QPS={:.0}", f16_time, f16_qps);
    println!("f16 recall@{} (相对 f32): {:.4}", k, f16_recall);
    println!();

    // 5. 汇总
    println!("=== 汇总 ===");
    println!("数据集: SIFT learn (100K vecs, dim={})", dim);
    println!("f32 QPS: {:.0}", f32_qps);
    println!("f16 QPS: {:.0}", f16_qps);
    println!("f16 recall@{} (相对 f32): {:.4}", k, f16_recall);
    println!();
    if f16_recall > 0.99 {
        println!("结论: f16 精度损失 < 1%，可用于搜索热路径");
    } else if f16_recall > 0.95 {
        println!("结论: f16 精度损失 1-5%，可用于粗筛阶段");
    } else {
        println!("结论: f16 精度损失 > 5%，不建议直接使用");
    }
}
