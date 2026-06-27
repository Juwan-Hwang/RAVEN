//! Week 6锛欰VQ 瀹屾暣璁粌寰幆楠岃瘉
//!
//! 鐩爣锛氶獙璇佹彁鍗?codebook 瀹归噺 (K=256) + 澧炲姞璁粌杞 (25) 鍚?recall 鑳藉惁鍒?0.95+
//!
//! 濡傛灉 recall 浠?< 0.3锛岃鏄庤缁冧俊鍙锋湰韬湁闂锛岄渶瑕佹彁鍓嶄粙鍏?

use std::fs::File;
use std::io::Read;
use raven::quant::avq::{AVQCodebook, TrainingSignal};
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;
use raven::l2_simd;

/// 璇诲彇 fvecs 鏂囦欢锛坰iftsmall 鏍煎紡锛?
/// 姣忎釜鍚戦噺锛? 瀛楄妭 int (缁村害) + dim * 4 瀛楄妭 float (鏁版嵁)
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 fvecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 fvecs 澶辫触");

    let record_size = 4 + 128 * 4; // dim(4) + 128 floats
    let n = bytes.len() / record_size;
    let mut vectors = vec![0.0f32; n * 128];

    for i in 0..n {
        let offset = i * record_size;
        let dim = i32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        assert_eq!(dim, 128, "缁村害涓嶆槸 128");
        for d in 0..128 {
            let off = offset + 4 + d * 4;
            vectors[i * 128 + d] = f32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]
            ]);
        }
    }
    (vectors, 128, n)
}

/// 璇诲彇 ivecs 鏂囦欢锛坓roundtruth 鏍煎紡锛?
/// 姣忎釜璁板綍锛? 瀛楄妭 int (缁村害=100) + dim * 4 瀛楄妭 int (閭诲眳 ID)
fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let mut file = File::open(path).expect("鏃犳硶鎵撳紑 ivecs 鏂囦欢");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("璇诲彇 ivecs 澶辫触");

    let record_size = 4 + 100 * 4; // dim(4) + 100 ints
    let n = bytes.len() / record_size;
    let mut gt = vec![0i32; n * 100];

    for i in 0..n {
        let offset = i * record_size;
        let dim = i32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        assert_eq!(dim, 100, "groundtruth 缁村害涓嶆槸 100");
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
..Default::default()
    };
    let graph = VamanaGraph::build(&quantized, dim, &config, &mut rng);

    let mut searcher = GraphSearcher::new(&quantized, &graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100; // siftsmall groundtruth 姣忎釜鏌ヨ 100 涓偦灞?
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

/// ADC 楠岃瘉锛氬浘鐢?f32 鏋勫缓锛屾悳绱㈡椂鏁版嵁搴撳悜閲忕敤閲忓寲閲嶅缓锛宷uery 淇濇寔 f32
///
/// 杩欐槸 DiskANN/ScaNN 璁烘枃閲?PQ/AVQ 鐨勬纭敤娉曪細
///   - 鍥剧粨鏋勭敱 f32 鍘熷鍚戦噺鏋勫缓锛堢粨鏋勬纭級
///   - 鎼滅储鏃惰窛绂昏绠楃敤 query_f32 鈫?decode(encode(db_vec))锛圓DC 绛変环锛?
///   - 鍙湁璺濈璁＄畻鏈夐噺鍖栬宸紝鍥惧鑸湰韬笉鍙楅噺鍖栧奖鍝?
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
    // 1. 鐢?f32 鍘熷鍚戦噺寤哄浘锛堝浘缁撴瀯姝ｇ‘锛?
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_max: 32,
        r_soft: 48,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);

    // 2. 鏋勯€犻噺鍖栭噸寤虹殑鏁版嵁搴撳悜閲忥紙ADC锛歲uery f32 鈫?db 閲忓寲閲嶅缓锛?
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 3. 鎼滅储鏃剁敤閲忓寲閲嶅缓鍚戦噺绠楄窛绂伙紝query 淇濇寔 f32
    let mut searcher = GraphSearcher::new(&quantized_db, &graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100; // siftsmall groundtruth 姣忎釜鏌ヨ 100 涓偦灞?
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

/// ADC + Reranking锛氶噺鍖栫矖绛?top-N + f32 绮炬帓 top-k
///
/// ScaNN/FAISS-IVF 鏍囧噯鍋氭硶锛?
///   1. f32 寤哄浘锛堝浘缁撴瀯姝ｇ‘锛?
///   2. 閲忓寲璺濈鎼滅储 top-N 鍊欓€夛紙N >> k锛岀矖绛涘揩锛?
///   3. 瀵?top-N 鍊欓€夌敤 f32 L2 绮炬帓锛屽彇 top-k
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
    // 1. 鐢?f32 鍘熷鍚戦噺寤哄浘
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 100,
        r_soft: 48,
        r_max: 32,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);

    // 2. 鏋勯€犻噺鍖栭噸寤虹殑鏁版嵁搴撳悜閲忥紙ADC 绮楃瓫锛?
    let quantized_db: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = &train[i * dim..(i + 1) * dim];
            codebook.decode(&codebook.encode(v))
        })
        .collect();

    // 3. 鎼滅储 top-N 鍊欓€?+ f32 rerank
    let mut searcher = GraphSearcher::new(&quantized_db, &graph, 100);
    let mut hits = 0usize;
    let gt_stride = 100;
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        // 绮楃瓫锛氶噺鍖栬窛绂绘悳 top-N
        let candidates = searcher.search(query, top_n);
        // 绮炬帓锛歠32 L2 閲嶇畻璺濈锛圫IMD 鍔犻€燂級
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
    println!("=== Week 6锛歴iftsmall 鐪熷疄鏁版嵁 AVQ recall 楠岃瘉 ===");
    println!("鐩爣锛歳ecall@10 > 0.95");
    println!();

    // 鍔犺浇 siftsmall 鏁版嵁闆?
    let (mut train, dim, n) = read_fvecs("data/siftsmall_base.fvecs");
    let (mut test, _, nq) = read_fvecs("data/siftsmall_query.fvecs");
    let (gt, gt_k, gt_nq) = read_ivecs("data/siftsmall_groundtruth.ivecs");
    println!("siftsmall: dim={}, base={}, query={}, gt_nq={}, gt_k={}",
        dim, n, nq, gt_nq, gt_k);
    // 璇婃柇锛氭鏌?gt 鏄惁 1-indexed
    print!("gt[0..10]={}", "");
    for d in 0..10 { print!("{} ", gt[d]); }
    println!();

    // 褰掍竴鍖栧埌 [0,1]锛圫IFT 鍘熷 0-255锛屾搴︾垎鐐稿鑷磋缁冨彂鏁ｏ級
    // L2 璺濈鎺掑簭鍦ㄥ潎鍖€缂╂斁涓嬩笉鍙橈紝f32 recall 涓嶅彈褰卞搷
    let max_val = 255.0f32;
    for v in train.iter_mut() { *v /= max_val; }
    for v in test.iter_mut() { *v /= max_val; }
    println!("鏁版嵁褰掍竴鍖? /{} 鈫?[0,1]", max_val);
    println!();

    // 鍩虹嚎锛歠32 鍘熷锛堜笉閲忓寲锛?
    let recall_f32 = f32_recall(&train, &test, &gt, dim, n, nq, 10);
    println!("f32 baseline锛堝浘+鎼滅储鍏?f32锛? recall@10={:.4}", recall_f32);
    println!();

    // 璁粌 AVQ codebook锛圞=256, sub_dim=8, 伪=0.30锛?
    println!("=== AVQ 璁粌锛圞=256, sub_dim=8, 伪=0.30, iter=25锛?==");
    let mut avq_rng = ChaCha8Rng::seed_from(42);
    let codebook = AVQCodebook::train_full(
        &train, dim, 256, TrainingSignal::BatchHighScorePairs, 25, 8, 0.30, avq_rng.inner(),
    );
    println!("AVQ codebook 璁粌瀹屾垚");
    println!();

    // 瀵规瘮瀹為獙 1锛歲uantized_recall锛堥敊璇敤娉?鈥?鐢ㄩ噺鍖栧悜閲忓缓鍥撅級
    // 楠岃瘉"鐢ㄩ噺鍖栧悜閲忓缓鍥?鏄惁鐪熺殑宸紙杩濆弽 ADC 鍘熷垯锛?
    let recall_quantized = quantized_recall(&codebook, &train, &test, &gt, dim, n, nq, 10);
    println!("quantized_recall锛堥敊璇細閲忓寲鍚戦噺寤哄浘锛? recall@10={:.4}", recall_quantized);
    println!();

    // 瀵规瘮瀹為獙 2锛歛dc_recall锛堟纭?ADC 鈥?f32 寤哄浘锛岄噺鍖栬窛绂绘悳绱級
    let recall_adc = adc_recall(&codebook, &train, &test, &gt, dim, n, nq, 10);
    println!("adc_recall锛堟纭細f32 寤哄浘 + 閲忓寲璺濈锛? recall@10={:.4}", recall_adc);
    println!();

    // 瀵规瘮瀹為獙 3锛歛dc_recall_rerank锛圓DC + f32 rerank锛?
    let recall_rerank = adc_recall_rerank(&codebook, &train, &test, &gt, dim, n, nq, 10, 100);
    println!("adc_recall_rerank锛圓DC + f32 rerank top-100锛? recall@10={:.4}", recall_rerank);
    println!();

    println!("=== 缁撹 ===");
    println!("f32 baseline:    recall@10={:.4}", recall_f32);
    println!("quantized (閿?:  recall@10={:.4}  鈫?鐢ㄩ噺鍖栧悜閲忓缓鍥撅紝杩濆弽 ADC 鍘熷垯", recall_quantized);
    println!("adc (姝ｇ‘):      recall@10={:.4}  鈫?f32 寤哄浘 + 閲忓寲璺濈", recall_adc);
    println!("adc+rerank:      recall@10={:.4}  鈫?涓ら樁娈碉紝鐩爣 >0.95", recall_rerank);
}

/// f32 寤哄浘 + f32 鏌ヨ锛堜笉閲忓寲锛?
/// gt_stride: groundtruth 姣忎釜鏌ヨ鐨勯偦灞呮暟锛坰iftsmall=100锛夛紝鍙栧墠 k 涓绠?recall
fn f32_recall(train: &[f32], test: &[f32], gt: &[i32], dim: usize, n: usize, nq: usize, k: usize) -> f64 {
    let mut rng = ChaCha8Rng::seed_from(42);
    let config = VamanaBuildConfig {
        alpha: 1.2,
        l_build: 200,
        r_max: 64,
        r_soft: 96,
        max_iterations: 2,
..Default::default()
    };
    let graph = VamanaGraph::build(train, dim, &config, &mut rng);
    let mut searcher = GraphSearcher::new(train, &graph, 100);

    // 璇婃柇锛氭墦鍗扮涓€涓煡璇㈢殑鎼滅储鍊欓€夋暟 + 鍥惧钩鍧囧害鏁?
    if nq > 0 {
        let query = &test[0..dim];
        let (candidates, visited) = VamanaGraph::greedy_search_vec(
            train, dim, graph.storage(), graph.entry_point(), query, 100,
        );
        eprintln!("[diag] ef=100 top={}, visited={}", candidates.len(), visited.len());
        let mut total_deg = 0usize;
        for i in 0..n { total_deg += graph.storage().degree(i as u32); }
        eprintln!("[diag] 鍥惧钩鍧囧害鏁?{:.1}", total_deg as f64 / n as f64);
    }
    // 璇婃柇锛氭墦鍗扮涓€涓煡璇㈢殑缁撴灉 vs groundtruth
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
    let gt_stride = 100; // siftsmall groundtruth 姣忎釜鏌ヨ 100 涓偦灞?
    for q in 0..nq {
        let query = &test[q * dim..(q + 1) * dim];
        let result = searcher.search(query, k);
        let found: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        // gt 姣忎釜鏌ヨ鏈?gt_stride 涓偦灞咃紝鍙栧墠 k 涓绠?recall@k
        let gt_slice = &gt[q * gt_stride..q * gt_stride + k];
        for &g in gt_slice {
            if found.contains(&(g as u32)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}
