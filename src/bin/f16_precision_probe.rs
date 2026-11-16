//! f16 绮惧害楠岃瘉瀹為獙锛堢涓夐樁娈碉級
//!
//! 鐩爣锛氶獙璇?f16 鍦?SIFT1M 涓婄殑绮惧害鎹熷け
//!
//! 娴佺▼锛?//! 1. 鍔犺浇 SIFT1M 鏁版嵁
//! 2. 鐢?f32 鏆村姏鎼滅储寰楀埌 groundtruth
//! 3. 鐢?f16 鏆村姏鎼滅储瀵规瘮 recall
//! 4. 娴嬮噺 f16 vs f32 鐨?QPS 宸紓
//!
//! 鍒ゆ柇鏍囧噯锛?//! - f16 recall@10 鐩稿 f32 鎹熷け < 1% 鈫?f16 鍙敤
//! - f16 QPS 鎻愬崌 > 10% 鈫?甯﹀鏀剁泭鏄捐憲

use std::fs::File;
use std::io::Read;
use std::time::Instant;
use raven::distance::{l2_simd, l2_f16_mixed, f32_to_f16_slice};

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

fn main() {
    println!("=== f16 绮惧害楠岃瘉瀹為獙锛堢涓夐樁娈碉級===");
    println!();

    // 1. 鍔犺浇鏁版嵁锛堢敤 query 闆嗗仛鏆村姏鎼滅储锛屽洜涓?base 澶ぇ鏆村姏鎼滅储澶參锛?    // 鐢?sift_learn (100K) 浣滀负鏁版嵁搴擄紝sift_query (10K) 浣滀负鏌ヨ
    let t0 = Instant::now();
    let (mut db, dim, n_db) = read_fvecs("data/sift/sift_learn.fvecs");
    let (mut queries, _, nq) = read_fvecs("data/sift/sift_query.fvecs");
    let (_, gt_k, _) = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("鏁版嵁鍔犺浇: {:.1}s", t0.elapsed().as_secs_f64());
    println!("db(learn): {} vecs, queries: {} vecs, dim={}, gt_k={}", n_db, nq, dim, gt_k);
    println!();

    // 褰掍竴鍖栧埌 [0,1]
    for v in db.iter_mut() { *v /= 255.0; }
    for v in queries.iter_mut() { *v /= 255.0; }

    let k = 10;

    // 2. 棰勯噺鍖栨暟鎹簱涓?f16
    println!("=== 棰勯噺鍖栨暟鎹簱涓?f16 ===");
    let t0 = Instant::now();
    let db_f16 = f32_to_f16_slice(&db);
    println!("f16 棰勯噺鍖? {:.1}s ({} vecs)", t0.elapsed().as_secs_f64(), n_db);
    println!("鍐呭瓨鍗犵敤: f32={}MB, f16={}MB (鑺傜渷 50%)",
        db.len() * 4 / 1024 / 1024,
        db_f16.len() * 2 / 1024 / 1024);
    println!();

    // 3. f32 鏆村姏鎼滅储锛堝熀绾匡級
    println!("=== f32 鏆村姏鎼滅储锛堝熀绾匡級===");
    let t0 = Instant::now();
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let mut best: Vec<(u32, f32)> = (0..n_db)
            .map(|i| {
                let v = &db[i * dim..(i + 1) * dim];
                (i as u32, l2_simd(query, v))
            })
            .collect();
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let _found: Vec<u32> = best.iter().take(k).map(|(id, _)| *id).collect();
        // 娉ㄦ剰锛歭earn 闆嗕笉鏄湡姝ｇ殑 groundtruth锛岃繖閲岀敤 f32 缁撴灉浣滀负 groundtruth 瀵规瘮 f16
        if q == 0 {
            // 鍙绗竴涓?query 璁板綍锛岀敤浜庡姣?        }
    }
    let f32_time = t0.elapsed().as_secs_f64();
    let f32_qps = nq as f64 / f32_time;
    println!("f32 鎼滅储: {:.2}s, QPS={:.0}", f32_time, f32_qps);
    println!();

    // 4. f16 娣峰悎妯″紡鎼滅储锛坬uery=f32, db=f16_packed锛?    println!("=== f16 娣峰悎妯″紡鎼滅储锛坬uery=f32, db=f16_packed锛?==");
    let t0 = Instant::now();
    let mut f16_matches = 0usize;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        // f32 鎼滅储寰楀埌 top-k
        let mut f32_best: Vec<(u32, f32)> = (0..n_db)
            .map(|i| {
                let v = &db[i * dim..(i + 1) * dim];
                (i as u32, l2_simd(query, v))
            })
            .collect();
        f32_best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let f32_topk: Vec<u32> = f32_best.iter().take(k).map(|(id, _)| *id).collect();

        // f16 鎼滅储寰楀埌 top-k
        let mut f16_best: Vec<(u32, f32)> = (0..n_db)
            .map(|i| {
                let v = &db_f16[i * dim..(i + 1) * dim];
                (i as u32, l2_f16_mixed(query, v))
            })
            .collect();
        f16_best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let f16_topk: Vec<u32> = f16_best.iter().take(k).map(|(id, _)| *id).collect();

        // 璁＄畻 f16 鐩稿 f32 鐨?recall
        for &id in &f16_topk {
            if f32_topk.contains(&id) {
                f16_matches += 1;
            }
        }
    }
    let f16_time = t0.elapsed().as_secs_f64();
    let f16_qps = nq as f64 / f16_time;
    let f16_recall = f16_matches as f64 / (nq * k) as f64;

    println!("f16 娣峰悎妯″紡鎼滅储: {:.2}s, QPS={:.0}", f16_time, f16_qps);
    println!("f16 recall@{} (鐩稿 f32): {:.4}", k, f16_recall);
    println!();

    // 5. 姹囨€?    println!("=== 姹囨€?===");
    println!("鏁版嵁闆? SIFT learn (100K vecs, dim={})", dim);
    println!("f32 QPS: {:.0}", f32_qps);
    println!("f16 QPS: {:.0}", f16_qps);
    println!("f16 recall@{} (鐩稿 f32): {:.4}", k, f16_recall);
    println!();
    if f16_recall > 0.99 {
        println!("缁撹: f16 绮惧害鎹熷け < 1%锛屽彲鐢ㄤ簬鎼滅储鐑矾寰?);
    } else if f16_recall > 0.95 {
        println!("缁撹: f16 绮惧害鎹熷け 1-5%锛屽彲鐢ㄤ簬绮楃瓫闃舵");
    } else {
        println!("缁撹: f16 绮惧害鎹熷け > 5%锛屼笉寤鸿鐩存帴浣跨敤");
    }
}
