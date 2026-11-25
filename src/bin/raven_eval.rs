//! 测量 harness（H01）
//!
//! 用法：
//!   raven_eval --data /workspace/data/sift1m --ef 20,30,...,120 --mode counters|qps \
//!            [--R 32 --L 200 --alpha 1.2 --alpha-core 1.0 --nav-m 32 --rerank 5 --po 16] \
//!            [--repeats 5] [--build-threads 2] [--out file.csv] [--max-base N]
//!
//! 数据目录需含 base.fvecs / query.fvecs / gt.ivecs / gt_dist.fvecs（由数据准备脚本生成）。
//!
//! recall@10 口径与 ann-benchmarks 一致：阈值 = gt_dist[k-1] + 1e-3（官方度量下的距离），
//! 返回结果中真实距离 ≤ 阈值的个数 / k。
//! H14：数据目录可含 `metric` 文件（euclidean | angular，由数据准备脚本写入）。
//! angular 数据集的 base/query 已 L2 归一化，图与搜索全程用 L2（单位向量上 ||x-q||² = 2(1-cos)，排序等价）；
//! recall 按官方 angular 距离 1-cos 计算（gt_dist 为官方 1-cos 原值）。无 metric 文件 = euclidean（旧数据目录）。
//!
//! 图缓存：/workspace/data/graphs/<fnv64(key)>.bin（+ .key 文本），key 含数据集、规模、全部建图参数与种子。
//!
//! mode=build：只建图并写缓存（用无计数器二进制，避免原子计数干扰建图耗时）。
//! mode=counters：需 `--features diag_counters` 编译；每个 ef 单轮全量查询，输出 recall + 计数均值。
//! mode=qps：不开计数器编译；外部 `taskset -c 1` 单核；每个 ef 先预热 1 轮，再交错跑 repeats 轮，
//!           取中位数 QPS 与 CV；另给"稳健 QPS"：每轮按 1000 查询分块，取分块耗时中位数换算，
//!           再对各轮取中位数（抵抗 VM steal 突发；同一批查询分块固定，跨实验可比）。

#[path = "common/a1_prof.rs"]
mod a1_prof;
use raven::build::ChaCha8Rng;
use raven::graph::robust_prune::PruneStrategy;
use raven::graph::navigation::LayeredNavigation;
use raven::graph::vamana::{GraphSearcher, L0Opts, NavOpts, VamanaBuildConfig};
use raven::memory::serialize::Serializable;
use raven::quant::sq4::SQ4Dataset;
use raven::VamanaGraph;
use std::io::Write;
use std::time::Instant;

fn read_vecs_f32(path: &str) -> (Vec<f32>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let d = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let row = 4 * (d + 1);
    let n = bytes.len() / row;
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        let r = &bytes[i * row + 4..(i + 1) * row];
        out.extend(r.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
    }
    (out, d)
}

fn read_vecs_i32(path: &str) -> (Vec<i32>, usize) {
    let (v, d) = read_vecs_f32(path);
    (v.into_iter().map(|x| x.to_bits() as i32).collect(), d)
}

fn fnv64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse::<T>().unwrap_or_else(|_| panic!("bad value for {name}")))
        .unwrap_or(default)
}

fn arg_str(args: &[String], name: &str, default: &str) -> String {
    arg(args, name, default.to_string())
}

/// H14：官方 angular 距离 1 - cos（f64）
fn angular_dist(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y; aa += x * x; bb += y * y;
    }
    1.0 - ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn exact_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| { let d = *x as f64 - *y as f64; d * d }).sum::<f64>().sqrt()
}

struct Row {
    ef: usize,
    recall: f64,
    full: f64,
    quant: f64,
    hops: f64,
    nav: f64,
    /// H07：完整 SQ4 距离次数 / 提前放弃次数 / 提前放弃折算 NDC（已含在 quant 内）
    q_complete: f64,
    q_part_cnt: f64,
    q_part_ndc: f64,
    qps_med: f64,
    qps_cv: f64,
    qps_all: Vec<f64>,
    /// 稳健 QPS：每轮取分块耗时中位数换算，再对各轮取中位数
    qps_rob: f64,
    qps_rob_cv: f64,
}

fn interp(rows: &[Row], target: f64, f: impl Fn(&Row) -> f64) -> Option<f64> {
    for w in rows.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.recall <= target && b.recall >= target {
            if (b.recall - a.recall).abs() < 1e-12 {
                return Some(f(a));
            }
            let t = (target - a.recall) / (b.recall - a.recall);
            return Some(f(a) + t * (f(b) - f(a)));
        }
    }
    None
}

/// H08：按"图邻居对 SQ4 码平方差"降序的维度置换（与主图 --perm near 同算法），供 --alt-R 第二张图使用。
fn near_perm(graph: &VamanaGraph, codes: &[u8], cb: usize, n: usize, dim: usize) -> Vec<usize> {
    let nib = |c: &[u8], d: usize| -> i32 { ((c[d / 2] >> ((d % 2) * 4)) & 0x0F) as i32 };
    let mut e = vec![0f64; dim];
    let step = (n / 20000).max(1);
    let st = graph.storage();
    for i in (0..n).step_by(step) {
        let ci = &codes[i * cb..(i + 1) * cb];
        for &j in st.neighbors(i as u32) {
            let cj = &codes[j as usize * cb..(j as usize + 1) * cb];
            for d in 0..dim { let t = nib(ci, d) - nib(cj, d); e[d] += (t * t) as f64; }
        }
    }
    let mut perm: Vec<usize> = (0..dim).collect();
    perm.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
    perm
}

thread_local! { static O2N: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new()); }
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = arg_str(&args, "--data", "/workspace/data/sift1m");
    let mode = arg_str(&args, "--mode", "counters");
    let efs: Vec<usize> = arg_str(&args, "--ef", "20,25,30,35,40,45,50,55,60,70,80,100,120")
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let r_max: usize = arg(&args, "--R", 32);
    let l_build: usize = arg(&args, "--L", 200);
    let alpha: f32 = arg(&args, "--alpha", 1.2);
    let alpha_core: f32 = arg(&args, "--alpha-core", 1.0);
    let r_soft: usize = arg(&args, "--r-soft", (r_max as f32 * 1.5) as usize);
    let nav_m: usize = arg(&args, "--nav-m", 32);
    let rerank: usize = arg(&args, "--rerank", 5);
    let po: usize = arg(&args, "--po", 16);
    let repeats: usize = arg(&args, "--repeats", 5);
    let build_threads: usize = arg(&args, "--build-threads", 2);
    let seed: u64 = arg(&args, "--seed", 42);
    let k: usize = 10;
    let out = arg_str(&args, "--out", "");
    let cache_dir = arg_str(&args, "--cache-dir", "/workspace/data/graphs");
    // H09 导航层搜索侧选项（默认 = 旧行为）
    let nav_opts = NavOpts {
        dedup: arg::<u8>(&args, "--nav-dedup", 1) != 0,
        sq4: arg::<u8>(&args, "--nav-sq4", 0) != 0,
        carry: arg(&args, "--nav-carry", 1usize),
        stop_level: arg(&args, "--nav-stop", 1u8),
    };
    // H09：只重建上层导航（不重建 Layer 0），0 = 不重建
    let nav_rebuild_m: usize = arg(&args, "--nav-rebuild-m", 0);
    let nav_rebuild_k: usize = arg(&args, "--nav-rebuild-k", r_max / 2);
    eprintln!("nav_opts={nav_opts:?} nav_rebuild_m={nav_rebuild_m} nav_rebuild_k={nav_rebuild_k}");
    // H07 Layer 0 剪枝选项（默认全关 = 旧行为）
    //   --ea 1            SQ4 分段提前放弃（默认精确：阈值 = 池最差距离）
    //   --ea-mask 7       检查点位掩码（bit c = 32·(c+1) 维后检查）
    //   --ea-eps E        有损模式（E≥0）：阈值 = 最差 × frac_c × (1+E)，frac_c 为近邻对能量累计占比
    //   --ea-scale a,b,c  直接给各检查点 scale（覆盖 eps）
    //   --perm none|var|near  SQ4 维度置换（var：全库码方差降序；near：图邻居对平方差降序）
    //   --es-k K --es-pat P   H12 早停：第 K 名距离连续 P 跳未改善即停
    let ea = arg::<u8>(&args, "--ea", 0) != 0;
    let ea_gran: usize = arg(&args, "--ea-gran", 32usize); // 32 或 16 维一段
    let ea_mask: u32 = arg(&args, "--ea-mask", 0xFFu32);
    let ea_eps: f32 = arg(&args, "--ea-eps", -1.0f32);
    let ea_scale_s = arg_str(&args, "--ea-scale", "");
    let perm_mode = arg_str(&args, "--perm", "none");
    let es_k: usize = arg(&args, "--es-k", 0);
    let es_pat: usize = arg(&args, "--es-pat", 0);
    // H08：搜索时度数上限（0 = 不限制，取邻接表前 sd 个；邻接表建图时已按距离升序排序）
    let sd: usize = arg(&args, "--sd", 0);

    if mode == "build" && raven::counters::ENABLED {
        eprintln!("mode=build should use the non-counter binary (build time would be skewed)");
        std::process::exit(2);
    }
    if mode == "counters" && !raven::counters::ENABLED {
        eprintln!("mode=counters requires --features diag_counters");
        std::process::exit(2);
    }

    // H39：oracle 上界（RAVEN_ORACLE=k[,off]：把 GT 第 off..off+k 名作为额外 L0 种子；仅诊断）
    let (oracle_k, oracle_off): (usize, usize) = std::env::var("RAVEN_ORACLE").ok().map(|v| {
        let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); (p[0], *p.get(1).unwrap_or(&0)) }).unwrap_or((0, 0));
    let t0 = Instant::now();
    let (base, dim) = read_vecs_f32(&format!("{data}/base.fvecs"));
    let (queries, qd) = read_vecs_f32(&format!("{data}/query.fvecs"));
    let (gt, gtk) = read_vecs_i32(&format!("{data}/gt.ivecs"));
    let gt_u: Vec<u32> = gt.iter().map(|&x| x as u32).collect();
    let (gt_dist, _) = read_vecs_f32(&format!("{data}/gt_dist.fvecs"));
    assert_eq!(dim, qd);
    let n = base.len() / dim;
    let nq = queries.len() / dim;
    let metric = std::fs::read_to_string(format!("{data}/metric")).map(|s| s.trim().to_string()).unwrap_or_else(|_| "euclidean".into());
    assert!(metric == "euclidean" || metric == "angular", "unknown metric {metric}");
    let angular = metric == "angular";
    eprintln!("data {data}: n={n} nq={nq} dim={dim} gt_k={gtk} metric={metric} (load {:.1}s)", t0.elapsed().as_secs_f64());
    if angular {
        // 归一化校验：angular 数据目录必须是单位向量（prep_dataset.py 已归一化），否则 L2 排序 ≠ cosine 排序
        let bad = (0..n).step_by((n / 1000).max(1)).filter(|&i| {
            let s: f32 = base[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum();
            s > 0.0 && (s - 1.0).abs() > 1e-3 && (s - 9.0).abs() > 1e-2 // 9 = 零向量哨兵（范数 3，见 prep_dataset.py）
        }).count();
        assert_eq!(bad, 0, "angular dataset must be L2-normalised");
    }

    // ── 图：缓存或建图 ──
    let key = format!(
        "v1|data={data}|n={n}|dim={dim}|prune=Directional|R={r_max}|L={l_build}|alpha={alpha}|alpha_core={alpha_core}|r_soft={r_soft}|iters=2|nav_m={nav_m}|n_jumps=0|rmin_div=4|seed={seed}"
    );
    let hash = format!("{:016x}", fnv64(&key));
    let gpath = format!("{cache_dir}/{hash}.bin");
    let mut graph = if std::path::Path::new(&gpath).exists() {
        let t = Instant::now();
        let g = VamanaGraph::load(std::path::Path::new(&gpath)).expect("load graph");
        eprintln!("graph cache hit {gpath} ({:.1}s)", t.elapsed().as_secs_f64());
        // R26 H149 / R28：饱和填充（raven::graph::vamana::fill_default：平台默认，RAVEN_FILL=D,ef 覆盖，=0 关）；填充图缓存为 <hash>.f<D>e<ef>.bin
        let g = if let Some((fd, fe)) = raven::graph::vamana::fill_default() {
            let fpath = format!("{cache_dir}/{hash}.f{fd}e{fe}.bin");
            if std::path::Path::new(&fpath).exists() {
                let g2 = VamanaGraph::load(std::path::Path::new(&fpath)).expect("load filled graph"); eprintln!("R28 fill cache hit {fpath}"); g2
            } else {
                let t = Instant::now(); let g2 = g.saturate_fill(&base, dim, fd, fe);
                eprintln!("R26 H149 fill D={fd} ef={fe} in {:.1}s", t.elapsed().as_secs_f64());
                if g2.save(std::path::Path::new(&fpath)).is_ok() { eprintln!("R28 fill cached {fpath}"); }
                g2
            }
        } else { g };
        // R26 X3：RAVEN_DUMP_L0=path 导出本图 L0 邻接（u32 n×r_max，SENTINEL 空槽）供离线拓扑分析
        if let Ok(dp) = std::env::var("RAVEN_DUMP_L0") {
            let st = g.storage(); let rm = st.r_max();
            let mut out: Vec<u8> = Vec::with_capacity(n * rm * 4);
            for u in 0..n as u32 { let nb = st.neighbors(u); for j in 0..rm { out.extend_from_slice(&nb.get(j).copied().unwrap_or(u32::MAX).to_le_bytes()); } }
            std::fs::write(&dp, &out).expect("dump l0"); eprintln!("R26 dumped L0 r_max={rm} to {dp}");
        }
        // R26 X3：RAVEN_EXT_L0=path（u32 n×K 邻接，u32::MAX 空槽；RAVEN_EXT_K=K）替换 L0 图（入口不变；每行按到本节点的精确距离升序，供 sd 截断）
        if let Ok(ep) = std::env::var("RAVEN_EXT_L0") {
            let k: usize = std::env::var("RAVEN_EXT_K").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
            load_ext_l0(&ep, k, &base, n, dim, g.entry_point())
        } else {
        g
        }
    } else {
        std::fs::create_dir_all(&cache_dir).ok();
        std::env::set_var("RAVEN_ALPHA_CORE", alpha_core.to_string());
        std::env::remove_var("RAVEN_N_JUMPS");
        std::env::remove_var("RAVEN_RMIN_DIVISOR");
        let config = VamanaBuildConfig {
            alpha,
            l_build,
            r_max,
            r_soft,
            max_iterations: 2,
            saturate: false,
            enable_layered_nav: true,
            nav_m,
            prune_strategy: PruneStrategy::DirectionalPrune,
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(build_threads).build().unwrap();
        let mut rng = ChaCha8Rng::seed_from(seed);
        let t = Instant::now();
        if std::env::var("RAVEN_BUILD_CKPT").is_err() {
            std::env::set_var("RAVEN_BUILD_CKPT", format!("{cache_dir}/{hash}.ckpt"));
        }
        let g = pool.install(|| VamanaGraph::build(&base, dim, &config, &mut rng));
        let _ = std::fs::remove_file(format!("{cache_dir}/{hash}.ckpt"));
        let secs = t.elapsed().as_secs_f64();
        eprintln!("graph built in {secs:.1}s ({build_threads} threads)");
        g.save(std::path::Path::new(&gpath)).expect("save graph");
        std::fs::write(format!("{cache_dir}/{hash}.key"), format!("{key}\nbuild_secs={secs:.1}\nbuild_threads={build_threads}\n")).ok();
        g
    };
    // 生产管线验证：--pipeline 1 走 raven::pipeline::prepare（与 python.rs 相同代码），逐 ef 输出 recall/QPS
    if arg::<u8>(&args, "--pipeline", 0) != 0 {
        let mut po_ = raven::pipeline::PipelineOpts::default();
        // R43-3：构建期知道评测 k（RAVEN_K），KL 大 k 计划与 LPX/HR 标定按此 k
        po_.k = std::env::var("RAVEN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        po_.rerank = rerank; po_.po = po;
        let sdp: i64 = arg(&args, "--sd", -1i64);
        if sdp >= 0 { po_.search_degree = Some(sdp as usize); }
        if arg::<u8>(&args, "--reorder-off", 0) != 0 { po_.reorder = false; }
        // R33 HR 模式：k-means 结果缓存到图缓存目录（内部序）；--hr 0/1 覆盖默认（RAVEN_HR 环境变量优先）
        let hrnl = std::env::var("RAVEN_HR_NL").ok().and_then(|v| v.parse().ok()).unwrap_or(raven::graph::cellconfirm::default_nl(n));
        po_.hr_cache = Some(format!("{cache_dir}/{hash}.hrc{hrnl}.r{}.bin", if po_.reorder { 1 } else { 0 }));
        let hrarg: i64 = arg(&args, "--hr", -1i64);
        if hrarg >= 0 { po_.hr = hrarg != 0; }
        let t = Instant::now();
        let mut p = raven::pipeline::prepare(graph, base.clone(), dim, &po_);
        eprintln!("pipeline prepared in {:.1}s sd={}", t.elapsed().as_secs_f64(), p.sd);
        if let Ok(v) = std::env::var("RAVEN_HSEL") {
            // R23：跳内排序选择 k[/epf]（0 = 关）；表在此构建（只读、进程级）
            let mut it = v.split('/'); let k: usize = it.next().unwrap().parse().unwrap();
            raven::graph::vamana::HSEL_K.store(k, std::sync::atomic::Ordering::Relaxed);
            raven::graph::vamana::HSEL_EPF.store(it.next().map(|x| x.parse().unwrap()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
            if k > 0 { let n = p.vectors.len() / dim; let b = raven::graph::vamana::hsel_build(&p.vectors, dim, n, 0xD1B54A32D192ED03, true); eprintln!("R23 HSEL k={k} table {:.1} MB", b as f64 / 1e6); }
        }
        if let Ok(v) = std::env::var("RAVEN_HREL") {
            let mut it = v.split('/'); let k: usize = it.next().unwrap().parse().unwrap();
            raven::graph::vamana::HREL_K.store(k, std::sync::atomic::Ordering::Relaxed);
            raven::graph::vamana::HREL_EPF.store(it.next().map(|x| x.parse().unwrap()).unwrap_or(16), std::sync::atomic::Ordering::Relaxed);
            if k > 0 { let n = p.vectors.len() / dim; let sdv = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
                let info = raven::graph::vamana::hrel_build(&p.vectors, dim, p.graph.storage(), n, sdv, true); eprintln!("R23 HREL k={k} {info}"); }
        }
        eprintln!("H122 effective defaults: arch={} la={} rrpf={} l0ea={} l0mask={:x} vh={}", std::env::consts::ARCH, raven::graph::vamana::la_depth(), raven::graph::refiner::rerank_pf_dist(), p.l0.early_abandon, p.l0.check_mask, raven::graph::vamana::vec_hop());
        eprintln!("H42 align: vectors%64={} adj%64={} sq4%64={} r_max={}", p.vectors.as_ptr() as usize % 64, p.graph.storage().main_block().as_ptr() as usize % 64, p.sq4.codes.as_ptr() as usize % 64, p.graph.storage().r_max());
        // R36：RAVEN_K=k（默认 10）→ 评测 recall@k（k=100 = VIBE 口径）；gt 需 ≥k 列
        let kk: usize = std::env::var("RAVEN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        // H40：学习式终止的逐跳轨迹（需 diag_counters 二进制）。--h40-trace PREFIX --h40-src synth|test --h40-n N --h40-seed S
        raven::graph::vamana::init_pair_from_env();
        eprintln!("defaults vh={} rrl2={} navpf={}", raven::graph::vamana::vec_hop(), raven::graph::refiner::RR_L2.load(std::sync::atomic::Ordering::Relaxed), raven::graph::vamana::NAV_PF.load(std::sync::atomic::Ordering::Relaxed));
        let h40 = arg_str(&args, "--h40-trace", "");
        if !h40.is_empty() {
            use rand::RngCore;
            let src = arg_str(&args, "--h40-src", "synth");
            let nqt: usize = arg(&args, "--h40-n", 2000usize);
            let hseed: u64 = arg(&args, "--h40-seed", 11u64);
            let pv = &p.vectors;
            let n = p.graph.len();
            let (qv, gtn): (Vec<f32>, Vec<u32>) = if src == "test" {
                let mut o2n = vec![0u32; n];
                for i in 0..n { o2n[p.orig_id(i as u32) as usize] = i as u32; }
                let m = nqt.min(nq);
                let gg: Vec<u32> = (0..m).flat_map(|qi| (0..kk).map(move |j| (qi, j))).map(|(qi, j)| o2n[gt_u[qi * gtk + j] as usize]).collect();
                (queries[..m * dim].to_vec(), gg)
            } else {
                let mut rng = ChaCha8Rng::seed_from(hseed);
                let mut qv: Vec<f32> = Vec::with_capacity(nqt * dim);
                for _ in 0..nqt {
                    let qi = (rng.next_u64() % n as u64) as usize;
                    let nb = p.graph.neighbors(qi as u32);
                    let j = if nb.is_empty() { qi } else { nb[(rng.next_u64() as usize) % nb.len()] as usize };
                    if src == "synth" {
                        qv.extend((0..dim).map(|d| 0.5 * (pv[qi * dim + d] + pv[j * dim + d])));
                    } else { // "synth3": 中点再沿垂直方向随机扰动（与邻居距离同尺度的 1/2）
                        let dd: f32 = (0..dim).map(|d| { let t = pv[qi * dim + d] - pv[j * dim + d]; t * t }).sum::<f32>().sqrt();
                        let mut noise: Vec<f32> = (0..dim).map(|_| { let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / 9007199254740993.0; let u2 = (rng.next_u64() >> 11) as f64 / 9007199254740992.0; ((-2.0 * u1.ln()).sqrt() * (6.283185307 * u2).cos()) as f32 }).collect();
                        let nn: f32 = noise.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        for x in noise.iter_mut() { *x *= 0.5 * dd / nn; }
                        qv.extend((0..dim).map(|d| 0.5 * (pv[qi * dim + d] + pv[j * dim + d]) + noise[d]));
                        if angular { let l = qv.len(); let s: f32 = qv[l - dim..].iter().map(|x| x * x).sum::<f32>().sqrt(); for x in qv[l - dim..].iter_mut() { *x /= s; } }
                    }
                }
                let t = Instant::now();
                use rayon::prelude::*;
                let gg: Vec<u32> = (0..nqt).into_par_iter().flat_map_iter(|ci| {
                    let q = &qv[ci * dim..(ci + 1) * dim];
                    let mut top: Vec<(f32, u32)> = Vec::with_capacity(kk + 1);
                    for j in 0..n {
                        let x = &pv[j * dim..(j + 1) * dim];
                        let d: f32 = q.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum();
                        if top.len() < kk { top.push((d, j as u32)); top.sort_by(|a, b| a.0.total_cmp(&b.0)); }
                        else if d < top[kk - 1].0 { top[kk - 1] = (d, j as u32); top.sort_by(|a, b| a.0.total_cmp(&b.0)); }
                    }
                    top.into_iter().map(|x| x.1).collect::<Vec<_>>()
                }).collect();
                eprintln!("h40 synth GT ({nqt} q) in {:.1}s", t.elapsed().as_secs_f64());
                (qv, gg)
            };
            let m = qv.len() / dim;
            raven::graph::stoptrace::enable();
            for &ef in &efs {
                raven::counters::reset();
                let mut s = p.searcher(ef);
                let mut hit = 0usize;
                for qi in 0..m {
                    let g = &gtn[qi * kk..(qi + 1) * kk];
                    raven::graph::stoptrace::begin_query(qi as u32, g);
                    let r = s.search_sq4(&qv[qi * dim..(qi + 1) * dim], kk);
                    hit += r.iter().take(kk).filter(|x| g.contains(&x.0)).count();
                }
                let (f, q, h) = raven::counters::snapshot();
                let rows = raven::graph::stoptrace::take_rows();
                let path = format!("{h40}_{src}_ef{ef}.f32");
                let mut fh = std::fs::File::create(&path).unwrap();
                fh.write_all(bytemuck::cast_slice(&rows)).unwrap();
                let (pc, pdims) = raven::counters::partial_snapshot();
                let _ = pdims; let ntc = (f as f64 + q as f64 + pc as f64) / m as f64;
                eprintln!("h40 src={src} ef={ef} recall(id)={:.5} full={:.1} quant={:.1} hops={:.2} ntc={:.1} rows={} -> {path}", hit as f64 / (m * kk) as f64, f as f64 / m as f64, q as f64 / m as f64, h as f64 / m as f64, ntc, rows.len() / raven::graph::stoptrace::NCOL);
            }
            return;
        }
        // H38：量化图（QG）L0 路径。--qg 1：同一进程内依次测 base（生产管线）与 qg，每个 ef 交替 repeats 轮；
        // counters 二进制下同时输出计数（full=精确距离=块数，quant=1-bit 估计数，hops）。
        if arg::<u8>(&args, "--qg", 0) != 0 {
            let qsd: usize = arg(&args, "--qg-sd", p.sd);
            let t = Instant::now();
            let qg = raven::graph::qg::QgIndex::build(&p, qsd);
            eprintln!("QG built in {:.1}s: dp={} bmax={} stride={}B total={:.0}MB", t.elapsed().as_secs_f64(), qg.dp, qg.bmax, qg.stride, qg.bytes() as f64 / 1e6);
            let qefs: Vec<usize> = arg_str(&args, "--qg-ef", &efs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).split(',').map(|x| x.parse().unwrap()).collect();
            let recall_of_res = |res: &Vec<Vec<u32>>| -> f64 {
                let mut tot = 0usize;
                for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                    let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                    tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                        (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
                tot as f64 / (nq * kk) as f64
            };
            let mut csv = String::from("variant,ef,recall10,qps,full,quant,hops,nav\n");
            let skip_base = arg::<u8>(&args, "--qg-skip-base", 0) != 0;
            for (vi, list) in [(0usize, &efs), (1usize, &qefs)] {
                if vi == 0 && skip_base { continue; }
                for &ef in list.iter() {
                    let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
                    let mut best = 0f64;
                    let (mut cf, mut cq, mut ch, mut cn) = (0f64, 0f64, 0f64, 0f64);
                    for rep in 0..repeats.max(1) {
                        raven::counters::reset();
                        let t = Instant::now();
                        if vi == 0 && std::env::var("RAVEN_PERQ_NEW").map(|v| v == "1").unwrap_or(false) {
                            // P1：模拟 Python 绑定每查询新建 searcher 的开销
                            for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let mut s = p.searcher(ef);
                                let r = s.search_sq4(q, kk); if rep == 0 { res[qi] = r.iter().map(|x| p.orig_id(x.0)).collect(); } }
                        } else if vi == 0 {
                            let mut s = p.searcher(ef);
                            for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                                let r = s.search_sq4(q, kk); if rep == 0 { res[qi] = r.iter().map(|x| p.orig_id(x.0)).collect(); } }
                        } else {
                            let mut s = raven::graph::qg::QgSearcher::new(&p, &qg, ef);
                            for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                                let r = s.search(q, kk); if rep == 0 { res[qi] = r.iter().map(|x| p.orig_id(x.0)).collect(); } }
                        }
                        let qps = nq as f64 / t.elapsed().as_secs_f64();
                        if qps > best { best = qps; }
                        if rep == 0 {
                            let (f, q, h) = raven::counters::snapshot();
                            let nv = raven::counters::nav_snapshot();
                            cf = f as f64 / nq as f64; cq = q as f64 / nq as f64; ch = h as f64 / nq as f64; cn = nv as f64 / nq as f64;
                        }
                    }
                    let rec = recall_of_res(&res);
                    let name = if vi == 0 { "base" } else { "qg" };
                    eprintln!("{name} ef={ef} recall={rec:.5} qps(best of {repeats})={best:.0} full={cf:.1} quant={cq:.1} hops={ch:.1} nav={cn:.1}");
                    csv += &format!("{name},{ef},{rec:.5},{best:.1},{cf:.2},{cq:.2},{ch:.2},{cn:.2}\n");
                }
            }
            if !out.is_empty() { std::fs::write(&out, csv).unwrap(); }
            return;
        }
        if let Ok(taus) = std::env::var("RAVEN_H64D") {
            // H64 动态：预计算前 sd 个邻居的 SQ4 边长，按 τ 列表扫 ef（基线 τ=0 先测），复用 h63_run 的评测（DK 关）
            let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
            let n = p.graph.len();
            let el = raven::graph::vamana::build_edge_len_u8(&p.sq4, p.graph.storage(), n, sd);
            raven::graph::vamana::EL_BASE.store(el.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            raven::graph::vamana::EL_SD.store(sd, std::sync::atomic::Ordering::Relaxed);
            for t in std::iter::once("0").chain(taus.split(',')) {
                let tau: f32 = t.parse().unwrap();
                raven::graph::vamana::EL_TAU.store(tau.to_bits(), std::sync::atomic::Ordering::Relaxed);
                eprintln!("H64D tau={tau}");
                h63_run(&mut p, "", &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &format!("{out}.tau{tau}"));
            }
            raven::graph::vamana::EL_TAU.store(0, std::sync::atomic::Ordering::Relaxed);
            drop(el);
            return;
        }
        if let Ok(k) = std::env::var("RAVEN_G1K") {
            // R24-H146 oracle：计数模式下也可开 G1（作弊方向选邻：每跳从行内全部邻居中按真 cos(v−u, q−u) 取前 k）
            raven::graph::vamana::G1_K.store(k.parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
            eprintln!("H146 G1 k={k}");
        }
        if let Ok(spec) = std::env::var("RAVEN_H147") {
            // R25-H147 门廊上提（doorway hoisting）+ 边产出 LFU 置换：库内伪查询（基点 + 高斯扰动，尺度 = 基底 1-NN 距离 × noise，不用测试集）
            // 跑小 ef 轨迹（A1，需 diag 构建 + 精简路径），对每个漏检 GT g：若 g 未取码且位于某个"已取码未扩展"门廊 w 的前 sd 邻接中，
            // 记 hoist(u→g)，u = 取到 w 的已扩展父节点；同时统计每条边 u→v 的产出（v 后来被扩展或进入结果）。
            // 改写（预算中性）：每个 u 按支持度 ≥ smin 取至多 rmax 个 hoist，替换前 sd 槽中产出最低（且 < 支持度）的边，位置不变。
            // spec = "pq_per_point,ef_tr,ef_gt,smin,rmax,noise"；RAVEN_H147_CHEAT=1 用测试查询（作弊上界）；RAVEN_H147_H144=r 先叠 H144
            let f: Vec<&str> = spec.split(',').collect();
            let ppp: f64 = f[0].parse().unwrap(); let ef_tr: usize = f[1].parse().unwrap(); let ef_gt: usize = f[2].parse().unwrap();
            let smins: Vec<u32> = f[3].split('/').map(|x| x.parse().unwrap()).collect(); let rmax: usize = f[4].parse().unwrap();
            let noise: f32 = f.get(5).map(|x| x.parse().unwrap()).unwrap_or(1.0);
            let cheat = std::env::var("RAVEN_H147_CHEAT").map(|v| v == "1").unwrap_or(false);
            let n = p.vectors.len() / dim; let sdv = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
            let orig: Vec<Vec<u32>> = (0..n).map(|u| p.graph.storage().neighbors(u as u32).to_vec()).collect();
            if let Ok(r) = std::env::var("RAVEN_H147_H144") { let r: usize = r.parse().unwrap(); if r > 0 { let vo: Vec<f32> = p.vectors.to_vec();
                let ch = raven::graph::vamana::h144_rewire(p.graph.storage_mut(), &vo, dim, sdv, r, 32); eprintln!("H147 stacked on H144 r={r}: {ch} edges"); } }
            let g0: Vec<Vec<u32>> = (0..n).map(|u| p.graph.storage().neighbors(u as u32).to_vec()).collect();
            h63_run(&mut p, "", &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &format!("{out}.base"));
            // 伪查询
            let t0 = Instant::now();
            let vec_of = |i: usize| &p.vectors[i * dim..(i + 1) * dim];
            let mut rng: u64 = 0x9E3779B97F4A7C15;
            let mut nxt = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
            let l2 = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
            let pqs: Vec<f32> = if cheat { queries[..nq * dim].to_vec() } else {
                // 尺度：随机 2000 个基点的 1-NN（排除自身）均方距离
                let mut s = p.searcher(64); let mut acc = 0f64; let ns = 2000;
                for _ in 0..ns { let i = (nxt() % n as u64) as usize; let r = s.search_sq4(vec_of(i), 2);
                    let j = r.iter().map(|x| x.0 as usize).find(|&j| j != i).unwrap(); acc += l2(vec_of(i), vec_of(j)) as f64; }
                let sig = ((acc / ns as f64) as f32 / dim as f32).sqrt() * noise;
                let m = (ppp * n as f64) as usize; let mut out = Vec::with_capacity(m * dim);
                for k in 0..m { let i = if m >= n { k % n } else { (nxt() % n as u64) as usize }; let x = vec_of(i);
                    let mut q: Vec<f32> = x.iter().map(|&v| { let u1 = ((nxt() >> 11) as f64 + 1.0) / (1u64 << 53) as f64; let u2 = (nxt() >> 11) as f64 / (1u64 << 53) as f64;
                        v + sig * ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 }).collect();
                    if angular { let nn = q.iter().map(|z| z * z).sum::<f32>().sqrt().max(1e-12); for z in q.iter_mut() { *z /= nn; } }
                    out.extend_from_slice(&q); }
                eprintln!("H147 pseudo-queries m={m} sigma/dim={sig:.3} (noise {noise})");
                out };
            let m = pqs.len() / dim;
            // 统计
            let mut yld = vec![0u32; n * sdv]; let mut expc = vec![0u32; n];
            let mut hoist: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
            let (mut nmiss, mut nb_, mut ndoor, mut nfar) = (0usize, 0usize, 0usize, 0usize);
            let mut sg = p.searcher(ef_gt); let mut st_ = p.searcher(ef_tr);
            let o2n: Vec<u32> = { let mut mm = vec![0u32; n]; for i in 0..n as u32 { mm[p.orig_id(i) as usize] = i; } mm };
            for qi in 0..m {
                let q = &pqs[qi * dim..(qi + 1) * dim];
                let g: Vec<u32> = if cheat { (0..10).map(|j| o2n[gt_u[qi * gtk + j] as usize]).collect() } else { sg.search_sq4(q, 10).iter().take(10).map(|x| x.0).collect() };
                raven::graph::vamana::A1_TRACE.with(|t| t.borrow_mut().clear());
                raven::graph::vamana::A1_TRACE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
                let r: Vec<u32> = st_.search_sq4(q, 10).iter().take(10).map(|x| x.0).collect();
                raven::graph::vamana::A1_TRACE_ON.store(false, std::sync::atomic::Ordering::Relaxed);
                let t = raven::graph::vamana::A1_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
                let tr = a1_prof::parse(&t); let Some(tr) = tr.last() else { continue };
                let mut parent: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
                let mut fd: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
                for &(v, d) in &tr.seeds { fd.insert(v, d); }
                let expanded: std::collections::HashSet<u32> = tr.hops.iter().map(|h| h.node).collect();
                for h in &tr.hops { for (k, &v) in h.ids.iter().enumerate() { parent.insert(v, h.node); fd.insert(v, h.ds[k]); } }
                let rs: std::collections::HashSet<u32> = r.iter().cloned().collect();
                for h in &tr.hops { let u = h.node as usize; expc[u] += 1; let row = &g0[u]; let lim = row.len().min(sdv);
                    for &v in &h.ids { if expanded.contains(&v) || rs.contains(&v) { if let Some(sl) = row[..lim].iter().position(|&x| x == v) { yld[u * sdv + sl] += 1; } } } }
                for &gg in &g { if rs.contains(&gg) { continue; } nmiss += 1;
                    if fd.contains_key(&gg) { nb_ += 1; continue; }
                    // 门廊：已取码未扩展、且前 sd 邻接含 gg，取 SQ4 距离最小者
                    let mut best: Option<(f32, u32)> = None;
                    for (&w, &d) in &fd { if expanded.contains(&w) { continue; } let row = &g0[w as usize];
                        if row[..row.len().min(sdv)].contains(&gg) && best.map_or(true, |b| d < b.0) { best = Some((d, w)); } }
                    match best { Some((_, w)) => { if let Some(&u) = parent.get(&w) { ndoor += 1; *hoist.entry((u, gg)).or_insert(0) += 1; } else { nfar += 1; } } None => nfar += 1 }
                }
            }
            eprintln!("H147 traced m={m} ef_tr={ef_tr} in {:.1}s: misses/q={:.3} fetched-but-ranked-out={:.3} doorway={:.3} other={:.3}; hoist pairs={} (support>=2: {}, >=3: {})",
                t0.elapsed().as_secs_f64(), nmiss as f64 / m as f64, nb_ as f64 / m as f64, ndoor as f64 / m as f64, nfar as f64 / m as f64, hoist.len(),
                hoist.values().filter(|&&c| c >= 2).count(), hoist.values().filter(|&&c| c >= 3).count());
            let zero_frac = { let (mut z, mut t) = (0usize, 0usize); for u in 0..n { if expc[u] == 0 { continue; } for sl in 0..g0[u].len().min(sdv) { t += 1; z += (yld[u * sdv + sl] == 0) as usize; } } z as f64 / t.max(1) as f64 };
            eprintln!("H147 expanded nodes {} / {n}; zero-yield edge frac (expanded nodes) {zero_frac:.3}", expc.iter().filter(|&&c| c > 0).count());
            let mut by_u: std::collections::HashMap<u32, Vec<(u32, u32)>> = std::collections::HashMap::new();
            for (&(u, gg), &c) in &hoist { by_u.entry(u).or_default().push((c, gg)); }
            for &smin in &smins {
                let mut changed = 0usize;
                for (&u, hs) in &by_u {
                    let mut hs: Vec<(u32, u32)> = hs.iter().filter(|x| x.0 >= smin).cloned().collect(); if hs.is_empty() { continue; }
                    hs.sort_by(|a, b| b.0.cmp(&a.0)); hs.truncate(rmax);
                    let mut row = g0[u as usize].clone(); let lim = row.len().min(sdv);
                    let mut slots: Vec<usize> = (0..lim).collect(); slots.sort_by_key(|&sl| (yld[u as usize * sdv + sl], std::cmp::Reverse(sl)));
                    let mut si = 0;
                    for (c, gg) in hs { if row[..lim].contains(&gg) { continue; } if si >= slots.len() { break; } let sl = slots[si];
                        if yld[u as usize * sdv + sl] >= c { break; }
                        // 若 gg 在尾部（sd 之后），与之交换，避免重复
                        if let Some(tp) = row[lim..].iter().position(|&x| x == gg) { row[lim + tp] = row[sl]; }
                        row[sl] = gg; si += 1; changed += 1; }
                    p.graph.storage_mut().set_neighbors(u, &row);
                }
                eprintln!("H147 smin={smin} rmax={rmax}: rewired {changed} edges");
                h63_run(&mut p, "", &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &format!("{out}.s{smin}"));
                for u in 0..n { p.graph.storage_mut().set_neighbors(u as u32, &g0[u]); }
            }
            for u in 0..n { p.graph.storage_mut().set_neighbors(u as u32, &orig[u]); }
            return;
        }
        if let Ok(spec) = std::env::var("RAVEN_H144") {
            // R23-H144 门廊直通边（预算中性、与查询无关的拓扑改写）oracle：见 raven::graph::vamana::h144_rewire。spec = "r1,r2,..."，r=0 为基线
            let n = p.vectors.len() / dim; let sdv = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
            let cpool: usize = std::env::var("RAVEN_H144_C").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
            let orig: Vec<Vec<u32>> = (0..n).map(|u| p.graph.storage().neighbors(u as u32).to_vec()).collect();
            let vecs_owned: Vec<f32> = p.vectors.to_vec();
            for r in spec.split(',').map(|x| x.parse::<usize>().unwrap()) {
                let t = Instant::now();
                let ch = raven::graph::vamana::h144_rewire(p.graph.storage_mut(), &vecs_owned, dim, sdv, r, cpool);
                eprintln!("H144 r={r} c={cpool} rewired {ch} edges in {:.1}s", t.elapsed().as_secs_f64());
                h63_run(&mut p, "", &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &format!("{out}.r{r}"));
                for u in 0..n { p.graph.storage_mut().set_neighbors(u as u32, &orig[u]); }
            }
            return;
        }
        if let Ok(ks) = std::env::var("RAVEN_H69") {
            // H69 方向准入 oracle：DIR_K 列表（基线 0 先测），复用 h63_run 评测（ntc/hops/recall）
            // RAVEN_H69_MODE=2：真实 16 位方向签名（H69b）
            let mode: usize = std::env::var("RAVEN_H69_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            raven::graph::vamana::DIR_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
            if mode == 6 {
                let n = p.vectors.len() / dim; let t = Instant::now();
                let m: usize = std::env::var("RAVEN_PQ_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                raven::graph::vamana::build_pq(&p.vectors, dim, n, m, 12, 65536);
                eprintln!("H70b PQ m={m} built in {:.1}s", t.elapsed().as_secs_f64());
                // 自检：同一节点的邻居（距离集中的情形）——取查询的 SQ4 搜索前 40 个结果，PQ 与真实距离的成对排序一致率
                let mut agg = (0usize, 0usize);
                for qi in 0..50 { let q = &queries[qi * dim..(qi + 1) * dim]; raven::graph::vamana::pq_set_query(q);
                    let mut s2 = p.searcher(60); let r = s2.search_sq4(q, 40);
                    let pr: Vec<(f32, f32)> = r.iter().map(|x| (raven::graph::vamana::pq_est(x.0 as usize), exact_dist(q, &p.vectors[x.0 as usize * dim..(x.0 as usize + 1) * dim]) as f32)).collect();
                    for a in 0..pr.len() { for b in a + 1..pr.len() { agg.1 += 1; if (pr[a].0 < pr[b].0) == (pr[a].1 < pr[b].1) { agg.0 += 1; } } } }
                eprintln!("H70b selfcheck (top-40 near neighbors, 50 q): PQ pairwise order agreement {:.3}", agg.0 as f64 / agg.1 as f64);
            }
            if mode == 5 {
                let n = p.vectors.len() / dim; let t = Instant::now();
                raven::graph::vamana::build_rb(&p.vectors, dim, n, 0xD1B54A32D192ED03);
                eprintln!("H70 rabitq-1bit built n={n} in {:.1}s", t.elapsed().as_secs_f64());
                // 自检：每查询 200 个伪随机点，RB 估计与真实距离的成对排序一致率
                for qi in 0..4 { let q = &queries[qi * dim..(qi + 1) * dim]; raven::graph::vamana::rb_set_query(q);
                    let ids: Vec<usize> = (0..200).map(|j| (j * 7919 + qi * 104729) % n).collect();
                    let mut pr = Vec::new();
                    for &id in &ids { let x = &p.vectors[id * dim..(id + 1) * dim]; pr.push((raven::graph::vamana::rb_est(id), exact_dist(q, x) as f32)); }
                    let (mut ok, mut tot) = (0, 0); for a in 0..pr.len() { for b in a + 1..pr.len() { tot += 1; if (pr[a].0 < pr[b].0) == (pr[a].1 < pr[b].1) { ok += 1; } } }
                    eprintln!("H70 selfcheck q{qi}: pairwise order agreement {:.3} (sample {:?})", ok as f64 / tot as f64, &pr[..3]); }
            }
            if mode == 7 {
                // R23 跳内相对排序：RAVEN_HR_KIND（0 随机 / 1 残差 PCA / 2 局部锚点基 / 3 精确），RAVEN_HR_M（投影维数或锚点数 B）
                let kind: usize = std::env::var("RAVEN_HR_KIND").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                let m: usize = std::env::var("RAVEN_HR_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                let n = p.vectors.len() / dim; let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
                let t = Instant::now();
                let info = raven::graph::vamana::build_hop_rank(&p.vectors, dim, p.graph.storage(), n, sd, kind, m, 0x9E3779B97F4A7C15);
                eprintln!("R23 hop-rank built kind={kind} m={m} n={n} sd={sd} in {:.1}s: {info}", t.elapsed().as_secs_f64());
            }
            if mode == 2 || mode == 3 {
                let nb: usize = std::env::var("RAVEN_H69_NB").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                let n = p.vectors.len() / dim; let sd = p.graph.storage().r_max();
                let t = Instant::now();
                raven::graph::vamana::build_dir_sig(&p.vectors, dim, p.graph.storage(), n, sd, nb, 0x9E3779B97F4A7C15);
                eprintln!("H69b dir sig built n={n} sd={sd} nb={nb} in {:.1}s", t.elapsed().as_secs_f64());
            }
            for k in std::iter::once("0").chain(ks.split(',')) {
                raven::graph::vamana::DIR_K.store(k.parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
                eprintln!("H69 dir_k={k}");
                h63_run(&mut p, "", &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &format!("{out}.k{k}"));
                if mode == 7 { let (h, pa, ov, c) = raven::graph::vamana::hr_stats_take(); eprintln!("R23 stats k={k}: filtered hops={h} pairwise_agree={pa:.4} topk_overlap={ov:.4} mean_cand={c:.1}"); }
            }
            raven::graph::vamana::DIR_K.store(0, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        if let Ok(spec) = std::env::var("RAVEN_H129") {
            // H129 oracle：MINDIST 池键（R-tree 节点球排序迁移）。spec = "rank;λ1,λ2,..."，rank = 用第几个邻居（1 起，0 = sd 末位）的距离作半径
            let mut it = spec.split(';');
            let rank: usize = it.next().unwrap().parse().unwrap();
            let lams: Vec<f32> = it.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
            let n = p.graph.len();
            let sdv = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
            let mut rr = vec![0f32; n];
            for w in 0..n {
                let nb = p.graph.neighbors(w as u32);
                let k = if rank == 0 { sdv.min(nb.len()) } else { rank.min(sdv).min(nb.len()) };
                if k == 0 { continue; }
                let cw = p.sq4.code(w).to_vec();
                rr[w] = p.sq4.distance_raw(&cw, nb[k - 1] as usize).sqrt();
            }
            raven::graph::vamana::MINDIST_R.store(rr.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            for &lam in &lams {
                raven::graph::vamana::MINDIST_LAMBDA.store(lam.to_bits(), std::sync::atomic::Ordering::Relaxed);
                for &ef in &efs {
                    let mut sx = p.searcher(ef);
                    let mut tot = 0usize;
                    let c0 = raven::counters::snapshot();
                    for qi in 0..nq {
                        let q = &queries[qi * dim..(qi + 1) * dim];
                        let r = sx.search_sq4(q, 10);
                        let thr = gt_dist[qi * gtk + 9] as f64 + 1e-3;
                        tot += r.iter().take(10).filter(|x| { let id = p.orig_id(x.0) as usize; let b = &base[id * dim..(id + 1) * dim]; (if angular { angular_dist(q, b) } else { exact_dist(q, b) }) <= thr }).count();
                    }
                    let c1 = raven::counters::snapshot();
                    eprintln!("H129 rank={rank} lambda={lam} ef={ef} recall={:.5} per_query(c0,c1,c2)=({:.1},{:.1},{:.1})", tot as f64 / (nq * 10) as f64, (c1.0 - c0.0) as f64 / nq as f64, (c1.1 - c0.1) as f64 / nq as f64, (c1.2 - c0.2) as f64 / nq as f64);
                }
            }
            drop(rr);
            return;
        }
        if let Ok(spec) = std::env::var("RAVEN_H126") {
            // H126 图集成（LSH 多表思想迁移到图）：同数据不同种子的第二张图，两次半 ef 搜索取并集
            let mut it = spec.split(';');
            let seed_b: u64 = it.next().unwrap().parse().unwrap();
            let key_b = key.replace(&format!("seed={seed}"), &format!("seed={seed_b}"));
            let gp_b = format!("{cache_dir}/{:016x}.bin", fnv64(&key_b));
            let gb = VamanaGraph::load(std::path::Path::new(&gp_b)).expect("graph B missing: build it first with --seed");
            let mut ob = po_.clone(); ob.search_degree = Some(p.sd);
            let pb = raven::pipeline::prepare(gb, base.clone(), dim, &ob);
            let pairs: Vec<(usize, usize)> = it.next().unwrap().split(',').map(|x| { let mut y = x.split('+'); (y.next().unwrap().parse().unwrap(), y.next().unwrap().parse().unwrap()) }).collect();
            h126_run(&p, &pb, &pairs, &queries, nq, dim, &base, &gt_dist, gtk, angular);
            return;
        }
        // A1（第 12 轮）：精简路径轨迹记录（diag 构建）/ 回放分段计时（release 构建）
        if let Ok(path) = std::env::var("RAVEN_A1_DUMP") {
            let na: usize = std::env::var("RAVEN_A1_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2000).min(nq);
            raven::graph::vamana::A1_TRACE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            let mut sx = p.searcher(efs[0]);
            for qi in 0..na { let _ = sx.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10); }
            let t = raven::graph::vamana::A1_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
            let b: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&path, b).unwrap();
            eprintln!("A1 dump: {na} queries ef={} -> {path} ({} words)", efs[0], t.len());
            return;
        }
        // R12-H137 oracle：BFS 序 = 隐式聚类？GT top-10 的内部 id 落在"GT 1-NN 内部 id ± W/2"窗口内的比例（局部扫描替代图遍历的上界）
        if std::env::var("RAVEN_WIN").is_ok() {
            let nn = p.vectors.len() / dim;
            let mut o2n = vec![0u32; nn];
            for i in 0..nn as u32 { o2n[p.orig_id(i) as usize] = i; }
            for &w in &[256i64, 1024, 4096, 16384, 65536] {
                let (mut c1, mut c3) = (0usize, 0usize);
                for qi in 0..nq {
                    let ids: Vec<i64> = (0..10).map(|j| o2n[gt_u[qi * gtk + j] as usize] as i64).collect();
                    c1 += ids.iter().filter(|&&x| (x - ids[0]).abs() <= w / 2).count();
                    c3 += ids.iter().filter(|&&x| ids[..3].iter().any(|&a| (x - a).abs() <= w / 2)).count();
                }
                eprintln!("WIN W={w} cover_top10_by_1NN_window={:.4} by_top3_windows={:.4}", c1 as f64 / (nq * 10) as f64, c3 as f64 / (nq * 10) as f64);
            }
            return;
        }
        // R12-N1 oracle：ef 小搜索漏掉的 GT，在"结果 top-10 各自的精确 kNN 名次"中最靠前是多少（暴力扫描，RAVEN_N1=查询数）
        // => 若漏检者多在 top-k 结果的 K 近邻内（K 小），则"图搜索 + 结果近邻补丁"可用 ef20 + 10×K 个码替代 ef40
        if let Ok(nstr) = std::env::var("RAVEN_N1") {
            let na: usize = nstr.parse::<usize>().unwrap_or(200).min(nq);
            let nn = p.vectors.len() / dim;
            let mut o2n = vec![0u32; nn];
            for i in 0..nn as u32 { o2n[p.orig_id(i) as usize] = i; }
            let vec_of = |i: u32| &p.vectors[i as usize * dim..(i as usize + 1) * dim];
            for &ef in &efs {
                let mut sx = p.searcher(ef);
                let (mut nmiss, mut rec) = (0usize, 0usize);
                let mut ranks: Vec<usize> = Vec::new();
                let mut ranks_top3: Vec<usize> = Vec::new();
                let (mut cos_f, mut cos_m): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
                let noscan = std::env::var("RAVEN_N1_NOSCAN").is_ok();
                for qi in 0..na {
                    let r = sx.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10);
                    let rs: Vec<u32> = r.iter().map(|x| x.0).collect();
                    let gts: Vec<u32> = (0..10).map(|j| o2n[gt_u[qi * gtk + j] as usize]).collect();
                    // N1′：方向性空洞——以结果 top-10 质心 c 为参照，cos(x − q, q − c) > 0 表示 x 在 q 的"另一侧"
                    let qv = &queries[qi * dim..(qi + 1) * dim];
                    let mut c = vec![0f32; dim];
                    for &y in &rs { for (a, b) in c.iter_mut().zip(vec_of(y)) { *a += *b / rs.len() as f32; } }
                    let cosd = |x: u32| { let xv = vec_of(x); let (mut d, mut n1, mut n2) = (0f32, 0f32, 0f32);
                        for k in 0..dim { let a = xv[k] - qv[k]; let b = qv[k] - c[k]; d += a * b; n1 += a * a; n2 += b * b; } d / (n1.sqrt() * n2.sqrt() + 1e-12) };
                    for &x in &gts { let cs = cosd(x); if rs.contains(&x) { cos_f.push(cs) } else { cos_m.push(cs) } }
                    if noscan { for &x in &gts { if rs.contains(&x) { rec += 1 } else { nmiss += 1 } } continue; }
                    for &x in &gts {
                        if rs.contains(&x) { rec += 1; continue; }
                        nmiss += 1;
                        let mut best = usize::MAX; let mut best3 = usize::MAX;
                        for (yi, &y) in rs.iter().enumerate() {
                            let dyx = raven::distance::l2_simd(vec_of(y), vec_of(x));
                            let mut cnt = 0usize;
                            for j in 0..nn as u32 { if j != y && raven::distance::l2_simd(vec_of(y), vec_of(j)) < dyx { cnt += 1; if cnt >= 64 { break; } } }
                            best = best.min(cnt); if yi < 3 { best3 = best3.min(cnt); }
                        }
                        ranks.push(best); ranks_top3.push(best3);
                    }
                }
                let pos = |v: &Vec<f32>| v.iter().filter(|&&x| x > 0.0).count() as f64 / v.len().max(1) as f64;
                let mean = |v: &Vec<f32>| v.iter().sum::<f32>() as f64 / v.len().max(1) as f64;
                eprintln!("N1P ef={ef} cos(x-q,q-c): found GT mean={:.3} frac>0={:.3} | missed GT mean={:.3} frac>0={:.3} (n_missed={})", mean(&cos_f), pos(&cos_f), mean(&cos_m), pos(&cos_m), cos_m.len());
                let frac = |v: &Vec<usize>, k: usize| v.iter().filter(|&&r| r < k).count() as f64 / v.len().max(1) as f64;
                eprintln!("N1 ef={ef} n={na} recall_ids={:.4} missed/q={:.3} | missed x within K-NN of some result top-10: K4 {:.3} K8 {:.3} K16 {:.3} K32 {:.3} K64 {:.3} | of top-3: K8 {:.3} K16 {:.3} K32 {:.3}",
                    rec as f64 / (na * 10) as f64, nmiss as f64 / na as f64, frac(&ranks, 4), frac(&ranks, 8), frac(&ranks, 16), frac(&ranks, 32), frac(&ranks, 64), frac(&ranks_top3, 8), frac(&ranks_top3, 16), frac(&ranks_top3, 32));
            }
            return;
        }
        // R22-H141（diag 构建）：门廊前瞻键（doorway lookahead key）oracle。独立精确 f32 best-first（同一图、同一 sd、种子取 RAVEN 导航种子），
        // 候选队列键 = min(d(q,w), la(w))，结果集仍按 d(q,w)；终止：最优候选键 ≥ 第 ef 名结果距离。
        //   mode 0：la = +∞（基线语义）
        //   mode 1：la = min_{x∈N_sd(w)} d(q,x)（完美前瞻，作弊上界，前瞻本身不计码）
        //   mode 2：la = min_j ( ‖q−c_j‖² + γ·σ_j² )，c_j/σ_j² = w 的 sd 邻居按 k-means 分 k 组的质心与组内均方半径（可实现摘要）
        // RAVEN_H141="n,mode,k,gamma"（k/gamma 只对 mode 2）
        if let Ok(spec) = std::env::var("RAVEN_H141") {
            let f: Vec<&str> = spec.split(',').collect();
            let na: usize = f[0].parse::<usize>().unwrap().min(nq);
            let mode: u32 = f[1].parse().unwrap();
            let kc: usize = f.get(2).map(|x| x.parse().unwrap()).unwrap_or(2);
            let gamma: f32 = f.get(3).map(|x| x.parse().unwrap()).unwrap_or(0.0);
            let st = p.graph.storage();
            let nn = p.vectors.len() / dim;
            let sdx = |l: usize| if p.sd > 0 { p.sd.min(l) } else { l };
            let vec_of = |i: u32| &p.vectors[i as usize * dim..(i as usize + 1) * dim];
            let l2 = |a: &[f32], b: &[f32]| -> f32 { let mut s = 0f32; for k in 0..a.len() { let d = a[k] - b[k]; s += d * d; } s };
            let o2n: Vec<u32> = { let mut m = vec![0u32; nn]; for i in 0..nn as u32 { m[p.orig_id(i) as usize] = i; } m };
            // 种子：RAVEN 导航（A1 轨迹的初始池）
            raven::graph::vamana::A1_TRACE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            let efmax = *efs.iter().max().unwrap();
            { let mut sx = p.searcher(efmax); for qi in 0..na { let _ = sx.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10); } }
            let t = raven::graph::vamana::A1_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
            raven::graph::vamana::A1_TRACE_ON.store(false, std::sync::atomic::Ordering::Relaxed);
            let qs = a1_prof::parse(&t);
            // 摘要（mode 2）：每节点 kc 个质心 + σ²
            let mut cent: Vec<f32> = Vec::new(); let mut sig: Vec<f32> = Vec::new();
            if mode == 2 {
                let tm = Instant::now();
                cent = vec![0f32; nn * kc * dim]; sig = vec![0f32; nn * kc];
                for u in 0..nn as u32 {
                    let nb = st.neighbors(u); let nb = &nb[..sdx(nb.len())];
                    let m = nb.len(); if m == 0 { continue; }
                    let kk = kc.min(m);
                    // 初始化：最远点采样（从离 u 最远的邻居开始）
                    let mut cs: Vec<Vec<f32>> = Vec::new();
                    let mut far = 0usize; let mut fd = -1f32;
                    for (i, &x) in nb.iter().enumerate() { let d = l2(vec_of(x), vec_of(u)); if d > fd { fd = d; far = i; } }
                    cs.push(vec_of(nb[far]).to_vec());
                    while cs.len() < kk { let mut bi = 0; let mut bd = -1f32; for (i, &x) in nb.iter().enumerate() { let d = cs.iter().map(|c| l2(vec_of(x), c)).fold(f32::MAX, f32::min); if d > bd { bd = d; bi = i; } } cs.push(vec_of(nb[bi]).to_vec()); }
                    let mut asg = vec![0usize; m];
                    for _it in 0..6 {
                        for (i, &x) in nb.iter().enumerate() { let mut b = 0; let mut bd = f32::MAX; for (j, c) in cs.iter().enumerate() { let d = l2(vec_of(x), c); if d < bd { bd = d; b = j; } } asg[i] = b; }
                        for (j, c) in cs.iter_mut().enumerate() { let mut cnt = 0f32; let mut acc = vec![0f32; dim]; for (i, &x) in nb.iter().enumerate() { if asg[i] == j { cnt += 1.0; for k in 0..dim { acc[k] += vec_of(x)[k]; } } } if cnt > 0.0 { for k in 0..dim { c[k] = acc[k] / cnt; } } }
                    }
                    for j in 0..kc { let o = (u as usize * kc + j) * dim; if j < kk { cent[o..o + dim].copy_from_slice(&cs[j]); let mut s = 0f32; let mut cnt = 0f32; for (i, &x) in nb.iter().enumerate() { if asg[i] == j { s += l2(vec_of(x), &cs[j]); cnt += 1.0; } } sig[u as usize * kc + j] = if cnt > 0.0 { s / cnt } else { 0.0 }; } else { cent[o..o + dim].copy_from_slice(&cs[0]); sig[u as usize * kc + j] = sig[u as usize * kc]; } }
                }
                eprintln!("H141 summaries k={kc} built in {:.1}s", tm.elapsed().as_secs_f64());
            }
            // mode 3/4（H142 门户/影子）：S(u,w) = u 的精确 K 近邻中不在 N_sd(u) 内、且离 w 最近（在 N_sd(u) 中）的点集（RNG 剪枝丢掉的、"藏在 w 后面"的点）
            let (mut sh_off, mut sh_ids): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
            let sdm = sdx(st.r_max());
            if mode >= 3 {
                let tm = Instant::now();
                let kpath = std::env::var("RAVEN_H142_KNN").expect("RAVEN_H142_KNN");
                let kk: usize = std::env::var("RAVEN_H142_K").ok().map(|x| x.parse().unwrap()).unwrap_or(100);
                let raw = std::fs::read(&kpath).unwrap(); let kf = raw.len() / 4 / nn;
                let knn = |o: usize, j: usize| u32::from_le_bytes(raw[(o * kf + j) * 4..(o * kf + j) * 4 + 4].try_into().unwrap());
                let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nn * sdm];
                let mut nsh = 0usize;
                for u in 0..nn as u32 {
                    let nb = st.neighbors(u); let nb = &nb[..sdx(nb.len())];
                    if nb.is_empty() { continue; }
                    let ou = p.orig_id(u) as usize;
                    for j in 0..kk.min(kf) {
                        let v = o2n[knn(ou, j) as usize];
                        if v == u || nb.contains(&v) { continue; }
                        let mut bi = 0; let mut bd = f32::MAX;
                        for (i, &w) in nb.iter().enumerate() { if w as usize >= nn { continue; } let d = l2(vec_of(w), vec_of(v)); if d < bd { bd = d; bi = i; } }
                        lists[u as usize * sdm + bi].push(v); nsh += 1;
                    }
                }
                sh_off.push(0); for l in &lists { sh_ids.extend_from_slice(l); sh_off.push(sh_ids.len() as u32); }
                eprintln!("H142 shadows K={kk} built in {:.1}s, mean shadow pts/node {:.1}", tm.elapsed().as_secs_f64(), nsh as f64 / nn as f64);
            }
            let mut vis = vec![0u32; nn]; let mut stamp = 0u32;
            for &ef in &efs {
                let (mut hit, mut hops, mut codes, mut la_cost) = (0usize, 0usize, 0usize, 0usize);
                for qi in 0..na {
                    let qv = &queries[qi * dim..(qi + 1) * dim];
                    stamp += 1;
                    let mut cand: std::collections::BinaryHeap<std::cmp::Reverse<(u32, u32)>> = std::collections::BinaryHeap::new(); // (key bits, id)
                    let mut resh: std::collections::BinaryHeap<(u32, u32)> = std::collections::BinaryHeap::new(); // max-heap by d bits
                    let key_of = |w: u32, d: f32, la_cost: &mut usize| -> f32 {
                        match mode {
                            1 => { let nb = st.neighbors(w); let mut m = d; for &x in &nb[..sdx(nb.len())] { m = m.min(l2(qv, vec_of(x))); } m }
                            2 => { *la_cost += 1; let mut m = d; for j in 0..kc { let o = (w as usize * kc + j) * dim; m = m.min(l2(qv, &cent[o..o + dim]) + gamma * sig[w as usize * kc + j]); } m }
                            _ => d,
                        }
                    };
                    let key_sh = |u: u32, slot: usize, d: f32| -> f32 {
                        let (a, b) = (sh_off[u as usize * sdm + slot] as usize, sh_off[u as usize * sdm + slot + 1] as usize);
                        if a == b { return d; }
                        let mem = &sh_ids[a..b];
                        if mode == 3 { let mut m = d; for &v in mem { m = m.min(l2(qv, vec_of(v))); } return m; }
                        let mut c = vec![0f32; dim]; for &v in mem { for k in 0..dim { c[k] += vec_of(v)[k]; } }
                        let cn = mem.len() as f32; for k in 0..dim { c[k] /= cn; }
                        let sg = mem.iter().map(|&v| l2(vec_of(v), &c)).sum::<f32>() / cn;
                        d.min(l2(qv, &c) + gamma * sg)
                    };
                    let push = |w: u32, d: f32, k: f32, cand: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, u32)>>, resh: &mut std::collections::BinaryHeap<(u32, u32)>| {
                        cand.push(std::cmp::Reverse((k.max(0.0).to_bits(), w)));
                        if resh.len() < ef { resh.push((d.to_bits(), w)); } else if d.to_bits() < resh.peek().unwrap().0 { resh.pop(); resh.push((d.to_bits(), w)); }
                    };
                    let seeds = &qs[qi].seeds;
                    for &(s, _) in seeds.iter().take(ef) { if vis[s as usize] == stamp { continue; } vis[s as usize] = stamp; let d = l2(qv, vec_of(s)); codes += 1; let k = key_of(s, d, &mut la_cost); push(s, d, k, &mut cand, &mut resh); }
                    while let Some(std::cmp::Reverse((kb, c))) = cand.pop() {
                        if resh.len() >= ef && kb >= resh.peek().unwrap().0 { break; }
                        hops += 1;
                        let nb = st.neighbors(c);
                        for (slot, &x) in nb[..sdx(nb.len())].iter().enumerate() {
                            if (x as usize) >= nn || vis[x as usize] == stamp { continue; }
                            vis[x as usize] = stamp;
                            let d = l2(qv, vec_of(x)); codes += 1;
                            let worst = if resh.len() >= ef { f32::from_bits(resh.peek().unwrap().0) } else { f32::MAX };
                            let k = if mode >= 3 { la_cost += 1; key_sh(c, slot, d) } else { key_of(x, d, &mut la_cost) };
                            if k < worst || d < worst { push(x, d, k, &mut cand, &mut resh); }
                        }
                    }
                    let mut r: Vec<(u32, u32)> = resh.into_vec(); r.sort();
                    let top: Vec<u32> = r.iter().take(10).map(|x| x.1).collect();
                    for j in 0..10 { if top.contains(&o2n[gt_u[qi * gtk + j] as usize]) { hit += 1; } }
                }
                let n = na as f64;
                eprintln!("H141 mode={mode} k={kc} gamma={gamma} ef={ef} n={na} recall={:.4} hops={:.2} codes={:.1} summaries_touched={:.1}", hit as f64 / (n * 10.0), hops as f64 / n, codes as f64 / n, la_cost as f64 / n);
            }
            return;
        }
        // R22-DECOMP（diag 构建）：漏检 GT top-10 的分解（RAVEN_DECOMP=N）：
        //   a  从未被取码（L0 种子/邻居都没出现过；a_nav = 只在导航层 visited 中出现过）
        //   b  已在 L0 取码（SQ4 估计过）但 SQ4 名次 > ef，被排出最终池（估计器/池宽）
        //   c  在最终池中但 rerank 后不在 top-10
        // 另：a 的拓扑子分类（被驱逐的已取码节点的前 sd 邻居里 = 前沿 1 跳；已扩展节点 sd 之后；其它）与入度（前 sd 邻接）对比
        if let Ok(nstr) = std::env::var("RAVEN_DECOMP") {
            let na: usize = nstr.parse::<usize>().unwrap_or(2000).min(nq);
            let st = p.graph.storage();
            let nn = p.vectors.len() / dim;
            let sdx = |l: usize| if p.sd > 0 { p.sd.min(l) } else { l };
            let mut indeg = vec![0u32; nn];
            for u in 0..nn as u32 { let nb = st.neighbors(u); for &v in &nb[..sdx(nb.len())] { if (v as usize) < nn { indeg[v as usize] += 1; } } }
            let mean_in = indeg.iter().map(|&x| x as f64).sum::<f64>() / nn as f64;
            let o2n: Vec<u32> = { let mut m = vec![0u32; nn]; for i in 0..nn as u32 { m[p.orig_id(i) as usize] = i; } m };
            raven::graph::vamana::A1_TRACE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            let dump = std::env::var("RAVEN_DECOMP_DUMP").ok();
            for &ef in &efs {
                let mut sx = p.searcher(ef);
                let mut res = Vec::new();
                for qi in 0..na { res.push(sx.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10)); }
                let t = raven::graph::vamana::A1_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
                let qs = a1_prof::parse(&t);
                assert_eq!(qs.len(), na);
                if let Some(pre) = &dump {
                    let mut buf: Vec<u8> = Vec::new();
                    for r in &res { for j in 0..10 { let o = r.get(j).map(|x| p.orig_id(x.0)).unwrap_or(u32::MAX); buf.extend_from_slice(&o.to_le_bytes()); } }
                    std::fs::write(format!("{pre}_ef{ef}.u32"), buf).unwrap();
                }
                let (mut n_found, mut a, mut a_nav, mut b, mut c) = (0f64, 0f64, 0f64, 0f64, 0f64);
                let (mut b_exp, mut a_front, mut a_beyond, mut a_far) = (0f64, 0f64, 0f64, 0f64);
                let (mut b_rank_sum, mut front_rank_sum, mut front_rank_n) = (0f64, 0f64, 0f64);
                let (mut in_found, mut in_found_n, mut in_miss, mut in_miss_n) = (0f64, 0f64, 0f64, 0f64);
                let mut hops_s = 0f64; let mut codes_s = 0f64;
                let mut b_ratio: Vec<f32> = Vec::new();
                for (qi, q) in qs.iter().enumerate() {
                    // L0 提供过的节点：内部 id → SQ4 距离（种子 + 每跳新邻居）
                    let mut off: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
                    for &(v, d) in &q.seeds { off.entry(v).or_insert(d); }
                    for h in &q.hops { for (k, &v) in h.ids.iter().enumerate() { off.entry(v).or_insert(h.ds[k]); } }
                    hops_s += q.hops.len() as f64; codes_s += q.hops.iter().map(|h| h.ids.len()).sum::<usize>() as f64;
                    let mut ranked: Vec<(f32, u32)> = off.iter().map(|(&v, &d)| (d, v)).collect();
                    ranked.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let rank_of: std::collections::HashMap<u32, usize> = ranked.iter().enumerate().map(|(i, &(_, v))| (v, i)).collect();
                    let pool: std::collections::HashSet<u32> = ranked.iter().take(ef).map(|x| x.1).collect();
                    let poolw = ranked.get(ef.min(ranked.len()) - 1).map(|x| x.0).unwrap_or(f32::MAX);
                    let exp: std::collections::HashSet<u32> = q.hops.iter().map(|h| h.node).collect();
                    let vis: std::collections::HashSet<u32> = q.vis.iter().copied().collect();
                    let rset: std::collections::HashSet<u32> = res[qi].iter().map(|x| x.0).collect();
                    for j in 0..10 {
                        let v = o2n[gt_u[qi * gtk + j] as usize];
                        let idg = indeg[v as usize] as f64;
                        if rset.contains(&v) { n_found += 1.0; in_found += idg; in_found_n += 1.0; continue; }
                        in_miss += idg; in_miss_n += 1.0;
                        if let Some(&d) = off.get(&v) {
                            if pool.contains(&v) { c += 1.0; } else {
                                b += 1.0; if exp.contains(&v) { b_exp += 1.0; }
                                b_rank_sum += rank_of[&v] as f64; b_ratio.push(d / poolw);
                            }
                        } else {
                            a += 1.0; if vis.contains(&v) { a_nav += 1.0; }
                            // 前沿：某个已取码但未扩展的节点 w 的前 sd 邻接含 v → 记 w 的最小 SQ4 名次
                            let mut best = usize::MAX;
                            for (&w, _) in off.iter() { if exp.contains(&w) { continue; } let nb = st.neighbors(w); if nb[..sdx(nb.len())].contains(&v) { best = best.min(rank_of[&w]); } }
                            if best != usize::MAX { a_front += 1.0; front_rank_sum += best as f64; front_rank_n += 1.0; }
                            else if exp.iter().any(|&u| st.neighbors(u).contains(&v)) { a_beyond += 1.0; }
                            else { a_far += 1.0; }
                        }
                    }
                }
                let n = na as f64; let miss = a + b + c;
                b_ratio.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let pct = |v: &Vec<f32>, p_: f64| if v.is_empty() { f32::NAN } else { v[((v.len() - 1) as f64 * p_) as usize] };
                eprintln!("DECOMP ef={ef} n={na} recall={:.4} hops={:.2} codes={:.1} missed/q={:.3} | a_never_touched={:.3} ({:.1}%; of which nav-only {:.3}) b_sq4_ranked_out={:.3} ({:.1}%; expanded {:.3}; mean sq4 rank {:.1}; d/poolworst p10 {:.3} p50 {:.3}) c_lost_at_rerank={:.3} ({:.1}%)",
                    n_found / (n * 10.0), hops_s / n, codes_s / n, miss / n, a / n, 100.0 * a / miss.max(1.0), a_nav / n, b / n, 100.0 * b / miss.max(1.0), b_exp / n, b_rank_sum / b.max(1.0), pct(&b_ratio, 0.1), pct(&b_ratio, 0.5), c / n, 100.0 * c / miss.max(1.0));
                eprintln!("DECOMP ef={ef} a-split/q: frontier1(in sd-adj of touched-unexpanded w)={:.3} (mean sq4 rank of best w {:.1}) beyond_sd_of_expanded={:.3} farther={:.3} | indeg(sd-adj) mean all {:.1} found {:.1} missed {:.1}",
                    a_front / n, front_rank_sum / front_rank_n.max(1.0), a_beyond / n, a_far / n, mean_in, in_found / in_found_n.max(1.0), in_miss / in_miss_n.max(1.0));
            }
            return;
        }
        // R12-DISC（diag 构建）：发现时间剖析——每个 GT top-10 在第几跳首次作为新邻居出现、由哪个扩展节点发现；
        // 回溯"发现链"（发现者 ← 发现者的发现者 … ← 种子）的并集 = 只扩展必要节点的 oracle 跳数（任何"选谁扩展"机制的上界）
        if let Ok(nstr) = std::env::var("RAVEN_DISC") {
            let na: usize = nstr.parse::<usize>().unwrap_or(2000).min(nq);
            raven::graph::vamana::A1_TRACE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            for &ef in &efs {
                let mut sx = p.searcher(ef);
                let mut res = Vec::new();
                for qi in 0..na { res.push(sx.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10)); }
                let t = raven::graph::vamana::A1_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
                let qs = a1_prof::parse(&t);
                assert_eq!(qs.len(), na);
                let (mut s_h, mut s_found, mut s_last, mut s_chain, mut s_prod, mut s_res) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
                let mut miss = [0f64; 5]; let mut miss2 = 0f64; let mut s_needw = 0f64;
                let mut hist = [0f64; 10]; // 发现跳 / 总跳 的十分位直方图
                for (qi, q) in qs.iter().enumerate() {
                    let gts: Vec<u32> = (0..10).map(|j| gt_u[qi * gtk + j]).collect();
                    // 内部 id → (发现跳, 发现者)；种子/导航 visited 视为第 0 跳、发现者 = MAX
                    let mut disc: std::collections::HashMap<u32, (usize, u32)> = std::collections::HashMap::new();
                    for &(v, _) in &q.seeds { disc.entry(v).or_insert((0, u32::MAX)); }
                    for &v in &q.vis { disc.entry(v).or_insert((0, u32::MAX)); }
                    for (hi, h) in q.hops.iter().enumerate() { for &v in &h.ids { disc.entry(v).or_insert((hi + 1, h.node)); } }
                    let hn = q.hops.len();
                    s_h += hn as f64;
                    let hop_of: std::collections::HashMap<u32, usize> = q.hops.iter().enumerate().map(|(i, h)| (h.node, i + 1)).collect();
                    let mut chain: std::collections::HashSet<u32> = std::collections::HashSet::new();
                    let mut prod: std::collections::HashSet<u32> = std::collections::HashSet::new();
                    let mut last = 0usize;
                    let rset: std::collections::HashSet<u32> = res[qi].iter().map(|x| p.orig_id(x.0)).collect();
                    for (v, &(dh, by)) in disc.iter() {
                        let o = p.orig_id(*v);
                        if !gts.contains(&o) { continue; }
                        if rset.contains(&o) { s_res += 1.0; }
                        s_found += 1.0; last = last.max(dh);
                        hist[((dh as f64 / hn.max(1) as f64) * 10.0).min(9.0) as usize] += 1.0;
                        if by != u32::MAX { prod.insert(by); }
                        let mut b = by;
                        while b != u32::MAX && chain.insert(b) { b = disc.get(&b).map(|x| x.1).unwrap_or(u32::MAX); }
                    }
                    let _ = &hop_of;
                    s_last += last as f64; s_chain += chain.len() as f64; s_prod += prod.len() as f64;
                    // 漏掉的 GT（轨迹中从未出现）分类：a 在某扩展节点的完整邻接（含 sd 之外）中；b 在某已发现 GT 的完整邻接中；
                    // c 自身有出边指向某已发现 GT（反向边可达）；d 都不是
                    let st = p.graph.storage();
                    let n2o: std::collections::HashMap<u32, u32> = disc.keys().map(|&v| (p.orig_id(v), v)).collect();
                    let found_int: Vec<u32> = gts.iter().filter_map(|o| n2o.get(o).copied()).collect();
                    let exp_set: std::collections::HashSet<u32> = q.hops.iter().map(|h| h.node).collect();
                    let mut need_w: std::collections::HashSet<u32> = std::collections::HashSet::new();
                    for &o in &gts {
                        if n2o.contains_key(&o) { continue; }
                        // 找内部 id：线性找太慢 → 用 new_to_old 的逆（首次构建）
                        let vi = O2N.with(|m| { let mut m = m.borrow_mut(); if m.is_empty() { let nn = p.vectors.len() / dim; m.resize(nn, 0); for i in 0..nn as u32 { let oo = p.orig_id(i); m[oo as usize] = i; } } m[o as usize] });
                        let in_exp = exp_set.iter().any(|&u| st.neighbors(u).contains(&vi));
                        let in_gt = found_int.iter().any(|&u| st.neighbors(u).contains(&vi));
                        let rev = st.neighbors(vi).iter().any(|w| found_int.contains(w));
                        let rev_any = st.neighbors(vi).iter().any(|w| exp_set.contains(w));
                        // N1 oracle：x 是否在"结果 top-10 的邻居 w（含未扩展者）"的邻接中（2 跳闭包可达）；记一个需要额外扩展的 w
                        let mut two_hop = false;
                        'outer: for r in res[qi].iter().take(10) {
                            for &w in st.neighbors(r.0) { if st.neighbors(w).contains(&vi) { two_hop = true; if !exp_set.contains(&w) { need_w.insert(w); } break 'outer; } }
                        }
                        if two_hop { miss2 += 1.0; }
                        if in_exp { miss[0] += 1.0 } else if in_gt { miss[1] += 1.0 } else if rev { miss[2] += 1.0 } else if rev_any { miss[3] += 1.0 } else { miss[4] += 1.0 }
                    }
                    s_needw += need_w.len() as f64;
                }
                eprintln!("DISC ef={ef} N1 two-hop-from-result-top10 reachable missed/query={:.3} extra_expansions_needed(oracle-picked w)={:.2}", miss2 / na as f64, s_needw / na as f64);
                eprintln!("DISC ef={ef} missed/query: in_expanded_full_adj(beyond sd)={:.3} in_foundGT_adj={:.3} rev_edge_to_foundGT={:.3} rev_edge_to_expanded={:.3} none={:.3}", miss[0]/na as f64, miss[1]/na as f64, miss[2]/na as f64, miss[3]/na as f64, miss[4]/na as f64);
                let n = na as f64;
                eprintln!("DISC ef={ef} n={na} hops={:.2} gt10_found_in_trace={:.3} gt10_in_result={:.3} last_disc_hop={:.2} productive_hops={:.2} chain_union_hops={:.2} => hop_bound={:.2}x",
                    s_h / n, s_found / n / 10.0, s_res / n / 10.0, s_last / n, s_prod / n, s_chain / n, s_h / s_chain.max(1.0));
                let tot: f64 = hist.iter().sum();
                eprintln!("DISC ef={ef} disc_hop_frac_deciles={}", hist.iter().map(|x| format!("{:.3}", x / tot)).collect::<Vec<_>>().join(","));
            }
            return;
        }
        // R13 MLP 微基准：随机 id 流 × SQ4 距离，滚动预取距离 pd（0 = 不预取）——测"无依赖链"时的每码吞吐成本
        if let Ok(spec) = std::env::var("RAVEN_MLP") {
            let n = p.sq4.n; let qc = p.sq4.code(0).to_vec();
            for pd in spec.split(',').map(|x| x.parse::<usize>().unwrap()) {
                let mut x: u64 = 0x9E3779B97F4A7C15; let m = 2_000_000usize;
                let ids: Vec<u32> = (0..m).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x % n as u64) as u32 }).collect();
                let t = Instant::now(); let mut acc = 0f32;
                for i in 0..m { if pd > 0 && i + pd < m { unsafe { raven::arch::prefetch_l1(p.sq4.code(ids[i + pd] as usize).as_ptr() as *const i8); } }
                    acc += p.sq4.distance_raw(&qc, ids[i] as usize); }
                eprintln!("MLP pd={pd} ns/code={:.2} (acc {acc})", t.elapsed().as_secs_f64() * 1e9 / m as f64);
            }
            return;
        }
        if let Ok(path) = std::env::var("RAVEN_A1_REPLAY") { a1_prof::run(&p, &path, &queries, dim, efs[0], repeats); return; }
        if let Ok(spec) = std::env::var("RAVEN_H125") { h125_run(&mut p, &spec, &queries, nq, dim, &base, &gt_u, &gt_dist, gtk, angular, &efs); return; }
        if let Ok(spec) = std::env::var("RAVEN_H63") { h63_run(&mut p, &spec, &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &out); return; }
        if let Ok(spec) = std::env::var("RAVEN_H64") { h64_run(&p, &spec, &queries, nq, dim); return; }
        if let Ok(spec) = std::env::var("RAVEN_H65") { h65_run(&p, &spec, &queries, nq, dim); return; }
        if let Ok(spec) = std::env::var("RAVEN_H67") { h67_run(&p, &spec, &queries, nq, dim); return; }
        if let Ok(spec) = std::env::var("RAVEN_H66") { h66_run(&p, &spec, &queries, nq, dim); return; }
        if let Ok(spec) = std::env::var("RAVEN_H62") { h62_run(&p, &spec, &queries, nq, dim); return; }
        if let Ok(spec) = std::env::var("RAVEN_H61") {
            h61_run(&mut p, &spec, &queries, nq, dim, &base, &gt_dist, gtk, angular, &efs, &out);
            return;
        }
        // R66：进程内块交替 A/B 计时（RAVEN_ABX=aa | pad16 | pad:N）：A = 本进程已建管线（缺省 = 主线 行为），
        //   B = 同图重载后另建一条管线（aa：同参数；pad:N：sq4_pad_small=Some(N) 程序化设置），计划（sd/HR 门槛/SDX/LPX）从 A 复制。
        if let Ok(spec) = std::env::var("RAVEN_ABX") {
            abx_run(p, &po_, &gpath, &spec, &base, dim, &queries, nq, &gt_dist, gtk, angular, &efs, repeats, kk, &out);
            return;
        }
        // 进程内 A/B（审计/H42 起）：RAVEN_AB=knob:a:b，按 1000 查询块交替 A/B（先后轮换），报告块配对比值中位数与召回。
        //   运行时开关：la、rrpf、h37（1/0）、carry、mis（SQ4 码错位字节）
        //   需第二份 Prepared：reorder（1/0）、clip（SQ4 clip，-1 = MSE，0 = min/max）
        if let Ok(spec) = std::env::var("RAVEN_AB") {
            let f: Vec<&str> = spec.split(':').collect();
            let (knob, va, vb) = (f[0].to_string(), f[1].to_string(), f[2].to_string());
            let mut pa = p;
            let mut pb_opt: Option<raven::pipeline::Prepared> = None;
            if knob == "h144" {
                // R23-H144：B 侧 = 门廊直通改写图（vb = r[/c]），在原始 id 空间改写后再 prepare（重排一致）
                let mut it = vb.split('/'); let r: usize = it.next().unwrap().parse().unwrap(); let c: usize = it.next().map(|x| x.parse().unwrap()).unwrap_or(32);
                let mut g2 = VamanaGraph::load(std::path::Path::new(&gpath)).expect("load graph");
                let sdv = if pa.sd > 0 { pa.sd } else { g2.storage().r_max() };
                let t = Instant::now();
                let ch = raven::graph::vamana::h144_rewire(g2.storage_mut(), &base, dim, sdv, r, c);
                eprintln!("H144 B rewired r={r} c={c} sd={sdv}: {ch} edges in {:.1}s", t.elapsed().as_secs_f64());
                let mut o2 = po_.clone(); o2.search_degree = Some(pa.sd);
                pb_opt = Some(raven::pipeline::prepare(g2, base.clone(), dim, &o2));
            }
            // R28：multi 组合旋钮 —— 每侧值 = "k1=v1+k2=v2+...[@ef]"，逐项按单旋钮语义应用（全局开关会跨侧残留：两侧须列出同一组键）；
            //   B 侧含 ext=sd 时 B = 外部 L0 图（同 ext 旋钮）；rrn 对 B 的 Prepared 单独生效
            let multi_b_ext = if knob == "multi" { multi_get(&vb, "ext") } else { None };
            if knob == "ext" || multi_b_ext.is_some() {
                // R26 X3：B 侧 = 外部 L0 图（RAVEN_EXT_B=path，RAVEN_EXT_K=k），vb = 该侧 sd[@ef]；导航层由 prepare 在 B 图上重建
                let path = std::env::var("RAVEN_EXT_B").expect("RAVEN_EXT_B");
                let k: usize = std::env::var("RAVEN_EXT_K").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
                let sdb: usize = match &multi_b_ext { Some(x) => x.parse().unwrap(), None => vb.split('@').next().unwrap().parse().unwrap() };
                let g2 = load_ext_l0(&path, k, &base, n, dim, pa.graph.entry_point());
                let mut o2 = po_.clone(); o2.search_degree = Some(sdb);
                pb_opt = Some(raven::pipeline::prepare(g2, base.clone(), dim, &o2));
                if let (Some(pb), Some(r)) = (pb_opt.as_mut(), multi_get(&vb, "rrn")) { pb.rerank = r.parse().unwrap(); }
                if let (Some(pb), Some(x)) = (pb_opt.as_mut(), multi_get(&vb, "l0ea")) { pb.l0.early_abandon = x == "1"; } // B 侧 Prepared 的 L0 开关
            }
            if knob == "reorder" || knob == "clip" {
                let mut o2 = po_.clone();
                let mut oa = po_.clone();
                oa.search_degree = Some(pa.sd); o2.search_degree = Some(pa.sd);
                if knob == "reorder" { oa.reorder = va == "1"; o2.reorder = vb == "1"; }
                if knob == "clip" { oa.sq4_clip = va.parse().unwrap(); o2.sq4_clip = vb.parse().unwrap(); }
                // A 也按 va 重建（若 va 与默认不同）
                let need_a = (knob == "reorder" && oa.reorder != po_.reorder) || (knob == "clip" && oa.sq4_clip != po_.sq4_clip);
                if need_a { drop(pa); let g = VamanaGraph::load(std::path::Path::new(&gpath)).unwrap(); pa = raven::pipeline::prepare(g, base.clone(), dim, &oa); }
                let g2 = VamanaGraph::load(std::path::Path::new(&gpath)).expect("load graph");
                pb_opt = Some(raven::pipeline::prepare(g2, base.clone(), dim, &o2));
            }
            let mut alt = if knob == "mis" { Some(raven::quant::sq4::AlignedBytes::from_slice_with_offset(&pa.sq4.codes, vb.parse().unwrap())) } else { None };
            // amis：B = 向量 + 邻接 + SQ4 码全部错位 vb 字节（复现 H42 前布局）
            let mut alt_v = if knob == "amis" { Some(raven::memory::aligned::AlignedBuf::from_slice_with_offset(&pa.vectors, vb.parse().unwrap())) } else { None };
            let mut alt_g = if knob == "amis" { Some(raven::memory::aligned::AlignedBuf::from_slice_with_offset(pa.graph.storage().main_block(), vb.parse().unwrap())) } else { None };
            // H87：pad knob —— B 侧换入补零到 vb 字节倍数的码（码长随之改变；查询侧补位恒 0，距离逐位一致）
            let mut pad_alt: Option<(raven::quant::sq4::AlignedBytes, usize)> = if knob == "pad" {
                let mult: usize = vb.parse().unwrap();
                let cb = pa.sq4.code_bytes; let nb = cb.div_ceil(mult) * mult; let n = pa.sq4.n;
                let mut out = vec![0u8; n * nb];
                for i in 0..n { out[i * nb..i * nb + cb].copy_from_slice(&pa.sq4.codes[i * cb..(i + 1) * cb]); }
                eprintln!("H87 pad: {cb}B -> {nb}B");
                Some((raven::quant::sq4::AlignedBytes::from_slice(&out), nb))
            } else { None };
            if knob == "amis" { alt = Some(raven::quant::sq4::AlignedBytes::from_slice_with_offset(&pa.sq4.codes, vb.parse().unwrap())); }
            // H47 thp：A/B 两侧分别以 va/vb 的大页设置重新拷贝 向量 + 邻接 + SQ4 码
            if knob == "thp" {
                use raven::memory::aligned::AlignedBuf;
                let (ha, hb) = (va == "1", vb == "1");
                pa.vectors = AlignedBuf::from_slice_opts(&pa.vectors, 0, ha);
                pa.sq4.codes = AlignedBuf::from_slice_opts(&pa.sq4.codes, 0, ha);
                let mut ga = AlignedBuf::from_slice_opts(pa.graph.storage().main_block(), 0, ha);
                pa.graph.storage_mut().swap_main_block(&mut ga);
                drop(ga);
                alt_v = Some(AlignedBuf::from_slice_opts(&pa.vectors, 0, hb));
                alt_g = Some(AlignedBuf::from_slice_opts(pa.graph.storage().main_block(), 0, hb));
                alt = Some(AlignedBuf::from_slice_opts(&pa.sq4.codes, 0, hb));
                eprintln!("H47 thp: A huge={} B huge={} (vectors A {} / B {})", ha, hb, pa.vectors.is_huge(), alt_v.as_ref().unwrap().is_huge());
                if let Ok(m) = std::fs::read_to_string("/proc/meminfo") { for l in m.lines().filter(|l| l.starts_with("AnonHugePages")) { eprintln!("H47 {l}"); } }
            }
            // H56 邻居码内嵌：构建内嵌码数组（保活到进程结束）
            let _inl_keep = if knob == "inl" {
                let st = pa.graph.storage();
                let isd = if pa.sd > 0 { pa.sd.min(st.r_max()) } else { st.r_max() };
                let n = pa.vectors.len() / dim;
                let t = Instant::now();
                let b = raven::graph::vamana::build_inline_codes(&pa.sq4, st, n, isd, std::env::var("RAVEN_THP").map(|v| v == "1").unwrap_or(false));
                raven::graph::vamana::INL_BASE.store(b.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
                raven::graph::vamana::INL_SD.store(isd, std::sync::atomic::Ordering::Relaxed);
                eprintln!("H56 inline codes: n={n} sd={isd} cb={} bytes={} ({:.2} GB, {:.1}x codes) built in {:.1}s", pa.sq4.code_bytes, b.len(), b.len() as f64 / 1e9, b.len() as f64 / pa.sq4.codes.len() as f64, t.elapsed().as_secs_f64());
                Some(b)
            } else { None };
            // H64b 边长准入：构建 u8 边长表（保活到进程结束），A/B 两侧以 τ 区分（0 = 关）
            let _el_keep = if knob == "el" {
                let sd = if pa.sd > 0 { pa.sd } else { pa.graph.storage().r_max() };
                let b = raven::graph::vamana::build_edge_len_u8(&pa.sq4, pa.graph.storage(), pa.graph.len(), sd);
                raven::graph::vamana::EL_BASE.store(b.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
                raven::graph::vamana::EL_SD.store(sd, std::sync::atomic::Ordering::Relaxed);
                Some(b)
            } else { None };
            if let Ok(h) = std::env::var("RAVEN_H139") {
                let f: Vec<f32> = h.split(',').map(|x| x.parse().unwrap()).collect();
                let t = Instant::now();
                let mut tb = raven::graph::vamana::h139_build(&pa.sq4, f[0] as usize, 8, 65536);
                tb.gamma = *f.get(1).unwrap_or(&1.0); tb.filter = f.get(2).map(|x| *x > 0.0).unwrap_or(false);
                eprintln!("H139 table k={} gamma={} filter={} built in {:.1}s", tb.k, tb.gamma, tb.filter, t.elapsed().as_secs_f64());
                raven::graph::vamana::H139.with(|h| *h.borrow_mut() = Some(tb));
            }
            // N2（第 13 轮）：rerank 残差估计跳过表（knob rrlb：值 = δ，0 = 关）
            let rrlb_ptr: usize = if knob == "rrlb" {
                let t = Instant::now();
                let e = raven::graph::vamana::rrlb_build(&pa.vectors, dim, &pa.sq4, &pa.perm);
                let mean = e.iter().map(|&x| x as f64).sum::<f64>() / e.len() as f64;
                eprintln!("N2 rrlb table built in {:.1}s, mean e_x={mean:.3}", t.elapsed().as_secs_f64());
                Box::leak(e.into_boxed_slice()).as_ptr() as usize
            } else { 0 };
            if knob == "hrel" {
                let t = Instant::now(); let n = pa.vectors.len() / dim; let sdv = if pa.sd > 0 { pa.sd } else { pa.graph.storage().r_max() };
                let info = raven::graph::vamana::hrel_build(&pa.vectors, dim, pa.graph.storage(), n, sdv, std::env::var("RAVEN_THP").map(|v| v != "0").unwrap_or(true));
                eprintln!("R23 HREL table {info} built in {:.1}s", t.elapsed().as_secs_f64());
            }
            if knob == "hsel" {
                let t = Instant::now(); let n = pa.vectors.len() / dim;
                let bytes = raven::graph::vamana::hsel_build(&pa.vectors, dim, n, 0xD1B54A32D192ED03, std::env::var("RAVEN_THP").map(|v| v != "0").unwrap_or(true));
                eprintln!("R23 HSEL table n={n} bytes={bytes} ({:.1} MB) built in {:.1}s", bytes as f64 / 1e6, t.elapsed().as_secs_f64());
            }
            // R25 CM：串行链随机表（RAVEN_CM_TAB_LOG2，默认 2^25 个 u32 = 128MB，远超 L3）
            let _cm_tab_keep: Option<Vec<u32>> = if knob == "cm" {
                let lg: usize = std::env::var("RAVEN_CM_TAB_LOG2").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
                let mut x: u64 = 0x2545F4914F6CDD1D;
                let t: Vec<u32> = (0..1usize << lg).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u32 }).collect();
                raven::graph::vamana::CM_TAB.store(t.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
                raven::graph::vamana::CM_MASK.store((1usize << lg) - 1, std::sync::atomic::Ordering::Relaxed);
                eprintln!("R25 CM chain table 2^{lg} u32 ({} MB)", (4usize << lg) >> 20);
                Some(t)
            } else { None };
            eprintln!("AB knob={knob} A={va} B={vb}");
            let mut csv = String::from("knob,A,B,ef,recall_A,recall_B,qps_A,qps_B,ratio_A_over_B_median,ratio_mean\n");
            let set_side = |side: usize, pa: &mut raven::pipeline::Prepared, alt: &mut Option<raven::quant::sq4::AlignedBytes>| {
                let v = if side == 0 { &va } else { &vb };
                let v_own = v.split('@').next().unwrap().to_string(); // "值@ef"：@ 后为该侧 ef（R13）
                let kvs: Vec<(String, String)> = if knob == "multi" {
                    v_own.split('+').filter(|x| !x.is_empty()).map(|x| { let (a, b) = x.split_once('=').expect("multi item k=v"); (a.to_string(), b.to_string()) }).collect()
                } else { vec![(knob.clone(), v_own.clone())] };
                for (kn, v_item) in &kvs { let v = v_item;
                if knob == "multi" && kn == "ext" { continue; }
                match kn.as_str() {
                    "la" => raven::graph::vamana::set_la_depth(v.parse().unwrap()),
                    "rrl2" => raven::graph::refiner::RR_L2.store(v.parse().unwrap(), std::sync::atomic::Ordering::Relaxed), // H133
                    "hrel" => { let mut it = v.split('/'); raven::graph::vamana::HREL_K.store(it.next().unwrap().parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::HREL_EPF.store(it.next().map(|x| x.parse().unwrap()).unwrap_or(16), std::sync::atomic::Ordering::Relaxed); } // R23 HREL k[/按分数早码预取数]
                    "hsel" => { let mut it = v.split('/'); raven::graph::vamana::HSEL_K.store(it.next().unwrap().parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::HSEL_EPF.store(it.next().map(|x| x.parse().unwrap()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::HSEL_RPF.store(it.next().map(|x| x != "0").unwrap_or(true), std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::HSEL_LUTONLY.store(it.next().map(|x| x == "1").unwrap_or(false), std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::HSEL_DBG.store(it.next().map(|x| x.parse().unwrap()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed); } // R23 跳内排序选择 k[/早码预取数]
                    "cm" => { // R25 代价模型 oracle：off | f/C/E（f = 被拒者保留比例，C = 每跳串行依赖访存数，E = 每跳额外独立取码数）
                        if v == "off" { raven::graph::vamana::CM_ON.store(false, std::sync::atomic::Ordering::Relaxed); CMW_MASK.store(0, std::sync::atomic::Ordering::Relaxed); } else {
                            let f: Vec<&str> = v.split('/').collect(); let fr: f64 = f[0].parse().unwrap();
                            raven::graph::vamana::CM.with(|c| { let mut c = c.borrow_mut(); c.rec = false;
                                c.keep_thr = if fr >= 1.0 { u32::MAX } else { (fr * u32::MAX as f64) as u32 };
                                c.chain = f.get(1).map(|x| x.parse().unwrap()).unwrap_or(0); c.extra = f.get(2).map(|x: &&str| if let Some(y) = x.strip_prefix('p') { y.parse::<u32>().unwrap() | 0x8000_0000 } else { x.parse().unwrap() }).unwrap_or(0); c.block = f.get(3).map(|x| x.parse().unwrap()).unwrap_or(0); });
                            CMW_MASK.store(f.get(4).map(|x| x.parse().unwrap()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
                            let ah: u32 = f.get(5).map(|x| x.parse().unwrap()).unwrap_or(0); raven::graph::vamana::CM.with(|c| c.borrow_mut().ah = ah);
                            let za: u32 = f.get(6).map(|x| x.parse().unwrap()).unwrap_or(0); raven::graph::vamana::CM.with(|c| c.borrow_mut().za = za); // R27 热态逐项消融掩码
                            raven::graph::vamana::CM_ON.store(true, std::sync::atomic::Ordering::Relaxed); } }
                    "vh" => raven::graph::vamana::set_vec_hop(v.parse().unwrap()), // H130：向量化跳执行位掩码
                    "lla" => raven::graph::vamana::set_lean_la(v.parse().unwrap()), // H128：精简路径前瞻
                    "novis" => raven::graph::vamana::set_novis(v.parse().unwrap()), // R27 S1：H153 无 visited 搜索
                    "rrlb" => { let dl: f32 = v.parse().unwrap(); raven::graph::vamana::RRLB_E.store(if dl > 0.0 { rrlb_ptr } else { 0 }, std::sync::atomic::Ordering::Relaxed); raven::graph::vamana::RRLB_DELTA.store(dl.to_bits(), std::sync::atomic::Ordering::Relaxed); }
                    "rrn" => pa.rerank = v.parse().unwrap(), // N3：rerank 宽度因子（rerank_n = 10 × 值，再截到池大小）
                    "g1" => raven::graph::vamana::G1_K.store(v.parse().unwrap(), std::sync::atomic::Ordering::Relaxed), // G1 方向选邻 oracle（作弊）
                    "n4" => raven::graph::vamana::set_n4(raven::graph::vamana::parse_n4(v.split('@').next().unwrap())), // N4（第 13 轮）：T/T'/H/flags[@ef]
                    "xsimd" => raven::quant::sq4::set_xsimd(v == "1"),
                    "xtail" => raven::quant::sq4::set_xtail(v == "1"),
                    "xab" => raven::quant::sq4::set_xab(v == "1"),
                    "po" => raven::graph::vamana::PO_OVERRIDE.store(v.parse().unwrap(), std::sync::atomic::Ordering::Relaxed),
                    "simple" => raven::graph::vamana::SIMPLE_MODE.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "rrpf" => raven::graph::refiner::set_rerank_pf_dist(v.parse().unwrap()),
                    "h37" => raven::graph::vamana::H37_CACHE.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "vthp" => raven::memory::visited::VIS_HUGE.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "vhash" => raven::memory::visited::VIS_HASH.store(v == "1", std::sync::atomic::Ordering::Relaxed), // H135
                    "vbits" => raven::memory::visited::VIS_BITS.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "rad" => raven::graph::vamana::set_rad(&v.replace('/', ",")),
                    "u8rr" => raven::graph::refiner::U8_RR.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "cpf" => raven::graph::vamana::CODE_PF.store(v.parse().unwrap(), std::sync::atomic::Ordering::Relaxed),
                    "navpf" => raven::graph::vamana::NAV_PF.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "defer" => raven::graph::vamana::DEFER_MODE.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "itp" => raven::graph::vamana::ITP_MODE.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "inl" => raven::graph::vamana::INL_MODE.store(v.parse().unwrap(), std::sync::atomic::Ordering::Relaxed),
                    "pf2" => raven::graph::vamana::CODE_PF2.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "pair" => raven::graph::vamana::PAIR_MODE.store(v == "1", std::sync::atomic::Ordering::Relaxed),
                    "carry" => pa.nav_opts.carry = v.parse().unwrap(),
                    "l0mask" => pa.l0.check_mask = u32::from_str_radix(v, 16).unwrap(), // H112：L0 检查点掩码（hex），0 = 走 L0 内核但不检查
                    "g16" => { pa.l0.gran16 = v == "1"; if v == "1" { pa.l0.check_mask = 0x7F; } else { let nchk = pa.sq4.dim / 32; pa.l0.check_mask = (((1u64 << (nchk - 1).min(32)) - 1) as u32) & 0xFF; } } // H115
                    "hsd" => { let mut it = v.split('/'); let h: usize = it.next().unwrap().parse().unwrap(); let d: usize = it.next().map(|x| x.parse().unwrap()).unwrap_or(0); raven::graph::vamana::HSD_CFG.store((h << 16) | d, std::sync::atomic::Ordering::Relaxed); }
                    "l0ea" => pa.l0.early_abandon = v == "1", // H111：L0 精确分段提前放弃开关
                    // X1/X3（x86 线）：组合旋钮 cfg=l0ea/vh/pair/simple/lla（例 1/0/0/1/0 = x86 旧默认）；"-" = 不改该项
                    "cfg" => { let f: Vec<&str> = v.split('/').collect();
                        let g = |i: usize| f.get(i).copied().filter(|x| *x != "-");
                        if let Some(x) = g(0) { pa.l0.early_abandon = x == "1"; }
                        if let Some(x) = g(1) { raven::graph::vamana::set_vec_hop(x.parse().unwrap()); }
                        if let Some(x) = g(2) { raven::graph::vamana::PAIR_MODE.store(x == "1", std::sync::atomic::Ordering::Relaxed); }
                        if let Some(x) = g(3) { raven::graph::vamana::SIMPLE_MODE.store(x == "1", std::sync::atomic::Ordering::Relaxed); }
                        if let Some(x) = g(4) { raven::graph::vamana::set_lean_la(x.parse().unwrap()); }
                        if let Some(x) = g(5) { raven::graph::refiner::RR_L2.store(x.parse().unwrap(), std::sync::atomic::Ordering::Relaxed); }
                        if let Some(x) = g(6) { raven::graph::vamana::NAV_PF.store(x == "1", std::sync::atomic::Ordering::Relaxed); }
                        if let Some(x) = g(7) { raven::graph::vamana::set_novis(x.parse().unwrap()); } } // R27 S1
                    "el" => raven::graph::vamana::EL_TAU.store(v.parse::<f32>().unwrap().to_bits(), std::sync::atomic::Ordering::Relaxed),
                    "mis" => { if side == 1 || alt.is_some() { let a = alt.as_mut().unwrap(); std::mem::swap(&mut pa.sq4.codes, a); } }
                    "s2" => raven::graph::vamana::set_s2(v.parse().unwrap()), // R28：H154 S2 批量归并池
                    _ => { if knob == "multi" { panic!("multi: unknown knob {kn}"); } }
                }
                }
            };
            let recall_of_res = |res: &Vec<Vec<u32>>| -> f64 {
                let mut tot = 0usize;
                for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                    let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                    tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                        (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
                tot as f64 / (nq * kk) as f64
            };
            // R26：cm 规格任一侧带第 5 字段（warm 掩码）→ 逐查询计时 + 段预热
            let cm_pqt = knob == "cm" && [&va, &vb].iter().any(|v| v.split('@').next().unwrap().split('/').count() >= 5);
            let mut cm_lines: Vec<CmLines> = Vec::new();
            for &ef in &efs {
                let (mut ta, mut tb, mut ratios) = (0f64, 0f64, Vec::new());
                let mut res: [Vec<Vec<u32>>; 2] = [vec![Vec::new(); nq], vec![Vec::new(); nq]];
                // R25 CM 记录轮（不计时）：每查询记下 pool.insert 成功的节点
                let cm_sets: Vec<Vec<u32>> = if knob == "cm" {
                    use raven::graph::vamana::{CM, CM_ON};
                    CM_ON.store(true, std::sync::atomic::Ordering::Relaxed);
                    CM.with(|c| { let mut c = c.borrow_mut(); c.rec = true; c.n_new = 0; c.n_kept = 0; c.hops = 0; });
                    let mut s = pa.searcher(ef);
                    cm_lines.clear();
                    let sets: Vec<Vec<u32>> = (0..nq).map(|qi| { CM.with(|c| { let mut c = c.borrow_mut(); c.acc.clear(); c.r_adj.clear(); c.r_vis.clear(); c.r_new.clear(); c.r_nav.clear(); c.r_rr.clear(); c.r_pop.clear(); c.r_off.clear(); c.r_dist.clear(); c.r_isn.clear(); c.r_acc2.clear(); c.r_aoff.clear(); c.r_fin.clear(); });
                        std::hint::black_box(s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk));
                        CM.with(|c| { let c = c.borrow();
                            if cm_pqt { // R26 段归因：按段保存本查询的行（acc/rej 按记录轮的入池成功划分）
                                let accs: std::collections::HashSet<u32> = c.acc.iter().copied().collect();
                                let (mut a, mut r) = (Vec::new(), Vec::new());
                                for &v in &c.r_new { if accs.contains(&v) { a.push(v) } else { r.push(v) } }
                                let mut off = c.r_off.clone(); off.push(c.r_new.len() as u32);
                                cm_lines.push(CmLines { adj: c.r_adj.clone(), vis: c.r_vis.clone(), acc: a, rej: r, nav: c.r_nav.clone(), rr: c.r_rr.clone(), pop: c.r_pop.clone(), off, newl: c.r_new.clone(),
                                    dist: c.r_dist.clone(), isn: c.r_isn.clone(), acc2: c.r_acc2.clone(), aoff: c.r_aoff.clone(), fin: c.r_fin.clone() }); }
                            c.acc.clone() }) }).collect();
                    if cm_pqt { let m = |f: &dyn Fn(&CmLines) -> usize| cm_lines.iter().map(|x| f(x)).sum::<usize>() as f64 / nq as f64;
                        eprintln!("CMLINES ef={ef} adj_lines/q={:.1} vis_checks/q={:.1} acc_codes/q={:.1} rej_codes/q={:.1} nav_vecs/q={:.1} rr_vecs/q={:.1}",
                            m(&|x| x.adj.len()), m(&|x| x.vis.len()), m(&|x| x.acc.len()), m(&|x| x.rej.len()), m(&|x| x.nav.len()), m(&|x| x.rr.len())); }
                    let (nn, hp) = CM.with(|c| { let mut c = c.borrow_mut(); c.rec = false; (c.n_new, c.hops) });
                    CM_ON.store(false, std::sync::atomic::Ordering::Relaxed);
                    let na: usize = sets.iter().map(|x| x.len()).sum();
                    eprintln!("CMREC ef={ef} new_codes/q={:.1} accepted/q={:.1} hops/q={:.2}", nn as f64 / nq as f64, na as f64 / nq as f64, hp as f64 / nq as f64);
                    sets
                } else { Vec::new() };
                for rep in 0..repeats.max(1) {
                    for c in 0..nq.div_ceil(1000) {
                        let (lo, hi) = (c * 1000, ((c + 1) * 1000).min(nq));
                        let mut tm = [0f64; 2];
                        for step in 0..2 {
                            let side = (step + rep + c) % 2;
                            // mis：B 侧把错位码换入，测完换回
                            if knob == "mis" { if side == 1 { set_side(1, &mut pa, &mut alt); } } else { set_side(side, &mut pa, &mut alt); }
                            if knob == "pad" && side == 1 { let (c, nb) = pad_alt.as_mut().unwrap(); std::mem::swap(&mut pa.sq4.codes, c); std::mem::swap(&mut pa.sq4.code_bytes, nb); }
                            if (knob == "amis" || knob == "thp") && side == 1 {
                                std::mem::swap(&mut pa.vectors, alt_v.as_mut().unwrap());
                                pa.graph.storage_mut().swap_main_block(alt_g.as_mut().unwrap());
                                std::mem::swap(&mut pa.sq4.codes, alt.as_mut().unwrap());
                            }
                            {
                                let pp: &raven::pipeline::Prepared = if side == 1 { pb_opt.as_ref().unwrap_or(&pa) } else { &pa };
                                let ef_side = { let v = if side == 0 { &va } else { &vb }; v.split('@').nth(1).and_then(|x| x.parse().ok()).unwrap_or(ef) };
                                let mut s = pp.searcher(ef_side);
                                if knob == "warmbt" && side == 1 {
                                    // H57b 数据热 + 分支冷：先跑一遍预热数据，再用随机分支冲刷分支预测器（不碰 L1 外内存），再计时第二遍
                                    let mut acc = 0f64;
                                    let mut x: u64 = 0x9E3779B97F4A7C15;
                                    let mut sink = 0u64;
                                    for qi in lo..hi { let q = &queries[qi * dim..(qi + 1) * dim];
                                        std::hint::black_box(s.search_sq4(q, kk));
                                        for _ in 0..20000 {
                                            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                                            if x & 1 != 0 { sink = sink.wrapping_add(x); }
                                            if x & 2 != 0 { sink ^= x >> 3; }
                                            if x & 4 != 0 { sink = sink.rotate_left(5); }
                                            if x & 8 != 0 { sink = sink.wrapping_mul(3); }
                                            if (x >> 9) & 1 != 0 { sink = sink.wrapping_sub(1); }
                                            if (x >> 17) & 3 == 1 { sink ^= 0xFF; }
                                        }
                                        std::hint::black_box(sink);
                                        let t = Instant::now();
                                        let r = s.search_sq4(q, kk);
                                        acc += t.elapsed().as_secs_f64();
                                        if rep == 0 { res[side][qi] = r.iter().map(|x| pp.orig_id(x.0)).collect(); } else { std::hint::black_box(&r); } }
                                    tm[side] = acc;
                                } else if knob == "warm" && side == 1 {
                                    // H57 热缓存 oracle：每个查询先跑一遍（预热其全部访存），只计第二遍耗时 = 零 DRAM 缺失下界
                                    let mut acc = 0f64;
                                    for qi in lo..hi { let q = &queries[qi * dim..(qi + 1) * dim];
                                        std::hint::black_box(s.search_sq4(q, kk));
                                        let t = Instant::now();
                                        let r = s.search_sq4(q, kk);
                                        acc += t.elapsed().as_secs_f64();
                                        if rep == 0 { res[side][qi] = r.iter().map(|x| pp.orig_id(x.0)).collect(); } else { std::hint::black_box(&r); } }
                                    tm[side] = acc;
                                } else if knob == "x2" && side == 1 {
                                    // H86：跨查询双路交错
                                    let mut s2 = pp.searcher(ef);
                                    let t = Instant::now();
                                    let mut qi = lo;
                                    while qi < hi {
                                        if qi + 1 < hi {
                                            let (ra, rb) = raven::graph::vamana::GraphSearcher::search_sq4_x2(&mut s, &mut s2, &queries[qi * dim..(qi + 1) * dim], &queries[(qi + 1) * dim..(qi + 2) * dim], kk);
                                            if rep == 0 { res[side][qi] = ra.iter().map(|x| pp.orig_id(x.0)).collect(); res[side][qi + 1] = rb.iter().map(|x| pp.orig_id(x.0)).collect(); } else { std::hint::black_box((&ra, &rb)); }
                                            qi += 2;
                                        } else {
                                            let r = s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk);
                                            if rep == 0 { res[side][qi] = r.iter().map(|x| pp.orig_id(x.0)).collect(); }
                                            qi += 1;
                                        }
                                    }
                                    tm[side] = t.elapsed().as_secs_f64();
                                } else {
                                if knob == "cm" && cm_pqt {
                                    // R26 段归因：逐查询计时；计时前（不计时）按掩码把某些段的行读入缓存。
                                    // 掩码位：1 邻接行 · 2 visited 字节 · 4 入池码 · 8 被拒码 · 16 导航层向量 · 32 rerank 向量 · 64 整查询预跑（全热）· 128 对照（同行数，改读 CM 随机表）
                                    let w = CMW_MASK.load(std::sync::atomic::Ordering::Relaxed);
                                    let (tab, tmask) = (raven::graph::vamana::CM_TAB.load(std::sync::atomic::Ordering::Relaxed), raven::graph::vamana::CM_MASK.load(std::sync::atomic::Ordering::Relaxed));
                                    let vb = dim * 4; let vbase = pp.vectors.as_ptr() as usize;
                                    let mut acc = 0f64; let mut sink = 0u64; let mut addrs: Vec<usize> = Vec::with_capacity(8192); let mut rng: u64 = 0x9E3779B97F4A7C15 ^ (qi_seed(lo, rep) as u64);
                                    for qi in lo..hi { let q = &queries[qi * dim..(qi + 1) * dim];
                                        let l = &cm_lines[qi]; addrs.clear();
                                        raven::graph::vamana::CM.with(|c| { let mut c = c.borrow_mut(); c.load(&cm_sets[qi]);
                                            if c.ah != 0 || c.za != 0 { c.q_pop.clear(); c.q_pop.extend_from_slice(&l.pop); c.q_off.clear(); c.q_off.extend_from_slice(&l.off); c.q_new.clear(); c.q_new.extend_from_slice(&l.newl); }
                                            if c.za != 0 { c.q_dist.clear(); c.q_dist.extend_from_slice(&l.dist); c.q_isn.clear(); c.q_isn.extend_from_slice(&l.isn);
                                                c.q_acc2.clear(); c.q_acc2.extend_from_slice(&l.acc2); c.q_aoff.clear(); c.q_aoff.extend_from_slice(&l.aoff); c.q_fin.clear(); c.q_fin.extend_from_slice(&l.fin); } });
                                        let zclr = raven::graph::vamana::CM.with(|c| c.borrow().za & 512 != 0);
                                        if w & 16 != 0 { // 导航层：pipeline 默认 nav_opts.sq4 = true → 节点的 SQ4 码 + 其上层邻接表行；否则 f32 向量
                                            let nav = pp.graph.layered_nav();
                                            for &v in &l.nav { if pp.nav_opts.sq4 { addrs.push(pp.sq4.code(v as usize).as_ptr() as usize); } else { let b = vbase + v as usize * vb; for o in (0..vb).step_by(64) { addrs.push(b + o); } }
                                                if let Some(nv) = nav { let (a, n) = nv.upper_row(v); if n > 0 { for o in (0..n).step_by(64) { addrs.push(a + o); } } } } }
                                        if w & 32 != 0 { for &v in &l.rr { let b = vbase + v as usize * vb; for o in (0..vb).step_by(64) { addrs.push(b + o); } } }
                                        if w & 8 != 0 { for &v in &l.rej { addrs.push(pp.sq4.code(v as usize).as_ptr() as usize); } }
                                        if w & 2 != 0 { for &v in &l.vis { addrs.push(s.visited_addr(v)); } }
                                        if w & 4 != 0 { for &v in &l.acc { addrs.push(pp.sq4.code(v as usize).as_ptr() as usize); } }
                                        if w & 1 != 0 { addrs.extend_from_slice(&l.adj); }
                                        if w & 128 != 0 { for a in addrs.iter_mut() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; *a = tab + ((rng as usize & tmask) & !15usize) * 4; } }
                                        for &a in &addrs { sink = sink.wrapping_add(unsafe { std::ptr::read_volatile(a as *const u8) } as u64); }
                                        if w & 64 != 0 { std::hint::black_box(s.search_sq4(q, kk)); if zclr { s.visited_clear_ids(&l.vis); } }
                                        std::hint::black_box(sink);
                                        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                                        let t = Instant::now();
                                        let r = s.search_sq4(q, kk);
                                        acc += t.elapsed().as_secs_f64();
                                        if zclr { s.visited_clear_ids(&l.vis); }
                                        if rep == 0 { res[side][qi] = r.iter().map(|x| pp.orig_id(x.0)).collect(); } else { std::hint::black_box(&r); } }
                                    tm[side] = acc;
                                } else {
                                let t = Instant::now();
                                for qi in lo..hi { let q = &queries[qi * dim..(qi + 1) * dim];
                                    if knob == "cm" { raven::graph::vamana::CM.with(|c| c.borrow_mut().load(&cm_sets[qi])); }
                                    let r = s.search_sq4(q, kk);
                                    if rep == 0 { res[side][qi] = r.iter().map(|x| pp.orig_id(x.0)).collect(); } else { std::hint::black_box(&r); } }
                                tm[side] = t.elapsed().as_secs_f64();
                                }
                                }
                            }
                            if knob == "mis" && side == 1 { set_side(1, &mut pa, &mut alt); }
                            if knob == "pad" && side == 1 { let (c, nb) = pad_alt.as_mut().unwrap(); std::mem::swap(&mut pa.sq4.codes, c); std::mem::swap(&mut pa.sq4.code_bytes, nb); }
                            if (knob == "amis" || knob == "thp") && side == 1 {
                                std::mem::swap(&mut pa.vectors, alt_v.as_mut().unwrap());
                                pa.graph.storage_mut().swap_main_block(alt_g.as_mut().unwrap());
                                std::mem::swap(&mut pa.sq4.codes, alt.as_mut().unwrap());
                            }
                        }
                        ta += tm[0]; tb += tm[1]; ratios.push(tm[1] / tm[0]);
                    }
                }
                if std::env::var("RAVEN_N4_STATS").is_ok() {
                    for side in 0..2 {
                        set_side(side, &mut pa, &mut alt);
                        let v = if side == 0 { &va } else { &vb };
                        let ef_side = v.split('@').nth(1).and_then(|x| x.parse().ok()).unwrap_or(ef);
                        let pp: &raven::pipeline::Prepared = if side == 1 { pb_opt.as_ref().unwrap_or(&pa) } else { &pa };
                        let mut s = pp.searcher(ef_side);
                        raven::graph::vamana::N4_CNT.with(|c| *c.borrow_mut() = [0; 6]);
                        raven::graph::vamana::H139_CNT.with(|c| *c.borrow_mut() = [0; 12]);
                        raven::graph::vamana::N4_STATS.store(true, std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::RRLB_STATS.store(true, std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::RRLB_CNT.with(|c| *c.borrow_mut() = [0; 3]);
                        raven::graph::vamana::N4_NS.with(|c| c.set(0)); raven::graph::vamana::N4_NS2.with(|c| c.set(0));
                        let tq = Instant::now();
                        for qi in 0..nq { std::hint::black_box(s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk)); }
                        let tq = tq.elapsed().as_secs_f64() * 1e6 / nq as f64;
                        raven::graph::vamana::N4_STATS.store(false, std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::RRLB_STATS.store(false, std::sync::atomic::Ordering::Relaxed);
                        eprintln!("N4TIME side={side} us/query={tq:.2} batch_us/query={:.2} (adj+filter {:.2})", raven::graph::vamana::N4_NS.with(|c| c.get()) as f64 / 1e3 / nq as f64, raven::graph::vamana::N4_NS2.with(|c| c.get()) as f64 / 1e3 / nq as f64);
                        let c = raven::graph::vamana::N4_CNT.with(|c| *c.borrow());
                        let n = c[3].max(1) as f64;
                        eprintln!("N4STAT ef={ef} side={side} cfg={v} seq_hops={:.2} batches={:.2} batch_nodes={:.2} batch_codes={:.2} batch_inserted={:.2} total_expanded={:.2}",
                            c[0] as f64 / n, c[4] as f64 / n, c[1] as f64 / n, c[2] as f64 / n, c[5] as f64 / n, (c[0] + c[1]) as f64 / n);
                        let rc = raven::graph::vamana::RRLB_CNT.with(|c| std::mem::take(&mut *c.borrow_mut()));
                        if rc[2] > 0 { eprintln!("N2STAT side={side} candidates/q={:.2} exact/q={:.2}", rc[0] as f64 / rc[2] as f64, rc[1] as f64 / rc[2] as f64); }
                        let h = raven::graph::vamana::H139_CNT.with(|c| *c.borrow());
                        if h[0] > 0 {
                            let t = h[0] as f64; let u = h[1].max(1) as f64;
                            let r = raven::graph::vamana::H139_RATIO.with(|c| std::mem::take(&mut *c.borrow_mut()));
                            eprintln!("H139RATIO side={side} frac batch codes with d_sq4/thr >= 1.1 {:.3} | 1.2 {:.3} | 1.35 {:.3} | 1.5 {:.3} | 2.0 {:.3}", r[0] as f64 / t, r[1] as f64 / t, r[2] as f64 / t, r[3] as f64 / t, r[4] as f64 / t);
                            let hh = raven::graph::vamana::H139_HOT.with(|c| std::mem::take(&mut *c.borrow_mut()));
                            let (ht, hu) = (hh[0].max(1) as f64, hh[1].max(1) as f64);
                            eprintln!("H139HOT side={side} useful={:.4} | beta(d_hot32/thr) 0.3 rej={:.3} lost={:.3} | 0.4 rej={:.3} lost={:.3} | 0.5 rej={:.3} lost={:.3} | 0.6 rej={:.3} lost={:.3} | 0.8 rej={:.3} lost={:.3} | 1.0(exact) rej={:.3} lost={:.3}",
                                hh[1] as f64 / ht, hh[2] as f64 / ht, hh[3] as f64 / hu, hh[4] as f64 / ht, hh[5] as f64 / hu, hh[6] as f64 / ht, hh[7] as f64 / hu, hh[8] as f64 / ht, hh[9] as f64 / hu, hh[10] as f64 / ht, hh[11] as f64 / hu, hh[12] as f64 / ht, hh[13] as f64 / hu);
                            eprintln!("H139STAT side={side} batch_codes={} useful(d<thr)={:.4} | gamma 0.8 rej={:.3} lost={:.3} | 1.0 rej={:.3} lost={:.3} | 1.2 rej={:.3} lost={:.3} | 1.5 rej={:.3} lost={:.3} | 2.0 rej={:.3} lost={:.3}",
                                h[0], h[1] as f64 / t, h[2] as f64 / t, h[3] as f64 / u, h[4] as f64 / t, h[5] as f64 / u, h[6] as f64 / t, h[7] as f64 / u, h[8] as f64 / t, h[9] as f64 / u, h[10] as f64 / t, h[11] as f64 / u);
                        }
                    }
                }
                if knob == "cm" {
                    for side in 0..2 { set_side(side, &mut pa, &mut alt);
                        raven::graph::vamana::CM.with(|c| { let mut c = c.borrow_mut(); c.n_new = 0; c.n_kept = 0; c.hops = 0; });
                        let on = raven::graph::vamana::CM_ON.load(std::sync::atomic::Ordering::Relaxed);
                        raven::graph::vamana::CM.with(|c| c.borrow_mut().za = 0); // R27：计数轮不做消融（q_* 只在逐查询计时轮载入）
                        let mut s = pa.searcher(ef);
                        for qi in 0..nq { raven::graph::vamana::CM.with(|c| c.borrow_mut().load(&cm_sets[qi])); std::hint::black_box(s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk)); }
                        let (nn, nk, hp) = raven::graph::vamana::CM.with(|c| { let c = c.borrow(); (c.n_new, c.n_kept, c.hops) });
                        if on { eprintln!("CMSTAT ef={ef} side={side} new/q={:.1} fetched/q={:.1} hops/q={:.2}", nn as f64 / nq as f64, nk as f64 / nq as f64, hp as f64 / nq as f64); }
                        else { eprintln!("CMSTAT ef={ef} side={side} off"); } }
                    let same = (0..nq).all(|qi| res[0][qi] == res[1][qi]);
                    eprintln!("CMSAME ef={ef} results_identical={same}");
                    raven::graph::vamana::CM_ON.store(false, std::sync::atomic::Ordering::Relaxed);
                }
                if knob == "novis" || ((knob == "cfg" || knob == "multi") && std::env::var("RAVEN_NVCHK").is_ok()) {
                    // R27 S1 逐查询校验：两侧的 rerank 结果、终态池（含已扩展标记与游标）以及（diag 构建）L0 弹出序列逐位比对
                    let mut out: [Vec<(Vec<(u32, f32)>, Vec<(u32, f32)>, usize, Vec<u32>)>; 2] = [Vec::new(), Vec::new()];
                    for side in 0..2 { set_side(side, &mut pa, &mut alt);
                        #[cfg(feature = "diag")] raven::graph::vamana::POPREC_ON.store(true, std::sync::atomic::Ordering::Relaxed);
                        let mut s = pa.searcher(ef);
                        for qi in 0..nq {
                            #[cfg(feature = "diag")] raven::graph::vamana::POPREC.with(|p| p.borrow_mut().clear());
                            let r = s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk);
                            let (pr, cur) = s.pool_raw();
                            #[cfg(feature = "diag")] let pops = raven::graph::vamana::POPREC.with(|p| p.borrow().clone());
                            #[cfg(not(feature = "diag"))] let pops: Vec<u32> = Vec::new();
                            out[side].push((r, pr, cur, pops));
                        }
                        #[cfg(feature = "diag")] raven::graph::vamana::POPREC_ON.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                    let bits = |v: &Vec<(u32, f32)>| v.iter().map(|x| (x.0, x.1.to_bits())).collect::<Vec<_>>();
                    let (mut sr, mut sp, mut so, mut npop) = (0usize, 0usize, 0usize, 0usize);
                    for qi in 0..nq { let (a, b) = (&out[0][qi], &out[1][qi]);
                        sr += (bits(&a.0) == bits(&b.0)) as usize; sp += (bits(&a.1) == bits(&b.1) && a.2 == b.2) as usize; so += (a.3 == b.3) as usize; npop += a.3.len(); }
                    eprintln!("NVCHK ef={ef} results_identical={sr}/{nq} pool_identical={sp}/{nq} popseq_identical={so}/{nq} (pops/q={:.2}, popseq recorded={})", npop as f64 / nq as f64, cfg!(feature = "diag"));
                }
                ratios.sort_by(|a, b| a.total_cmp(b));
                let med = ratios[ratios.len() / 2];
                let n_all = (nq * repeats.max(1)) as f64;
                let (ra, rb) = (recall_of_res(&res[0]), recall_of_res(&res[1]));
                eprintln!("AB ef={ef} recall A={ra:.5} B={rb:.5} qps_A={:.0} qps_B={:.0} A/B(qps) median={med:.4} mean={:.4}", n_all / ta, n_all / tb, tb / ta);
                csv += &format!("{knob},{va},{vb},{ef},{ra:.5},{rb:.5},{:.1},{:.1},{med:.4},{:.4}\n", n_all / ta, n_all / tb, tb / ta);
            }
            if !out.is_empty() { std::fs::write(&out, csv).unwrap(); }
            return;
        }
        if std::env::var("RAVEN_YT").map(|v| v == "1").unwrap_or(false) { raven::graph::vamana::YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed); }
        // H58 dump：每跳 9×u32 行 + new_to_old 映射 + 前 sd 个邻接（新 id）
        let yt_dump = std::env::var("RAVEN_YT_DUMP").ok();
        if let Some(pfx) = &yt_dump {
            raven::graph::vamana::YT.with(|t| t.borrow_mut().dump = Some(Vec::new()));
            let n = p.vectors.len() / dim;
            let m: Vec<u8> = (0..n as u32).flat_map(|i| p.orig_id(i).to_le_bytes()).collect();
            std::fs::write(format!("{pfx}.n2o"), m).unwrap();
            let st = p.graph.storage(); let sdd = if p.sd > 0 { p.sd } else { st.r_max() };
            let mut adj = vec![u32::MAX; n * sdd];
            for u in 0..n { for (i, &v) in st.neighbors(u as u32).iter().take(sdd).enumerate() { adj[u * sdd + i] = v; } }
            std::fs::write(format!("{pfx}.adj{sdd}"), adj.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        }
        if std::env::var("RAVEN_TRACE_ORACLE").map(|v| v == "1").unwrap_or(false) {
            // H59 轨迹回放 oracle：查询 q 的取码集合 T(q) 与其"地标"查询（库内点）的取码集合 T(l) 的覆盖率
            use raven::graph::vamana::{YT, YT_MODE};
            YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
            YT.with(|t| t.borrow_mut().keep_touch = true);
            let ef = efs[0]; let nqo = nq.min(2000);
            let mut s = p.searcher_k(ef, kk);
            let pv = &p.vectors[..];
            let run = |s: &mut raven::graph::vamana::GraphSearcher, q: &[f32]| -> (Vec<u32>, u32) {
                YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); });
                std::hint::black_box(s.search_sq4(q, kk));
                YT.with(|t| { let t = t.borrow(); let mut v: Vec<u32> = t.touched.iter().chain(t.popped.iter()).copied().collect(); v.sort_unstable(); v.dedup(); (v, t.popped[0]) })
            };
            let st = p.graph.storage();
            if std::env::var("RAVEN_TRACE_GROUPS").is_ok() {
                // H60 行共享 oracle：取码 id 按连续 g 个分组（BFS 序），每查询需要的不同组数 / 取码数
                let gs = [2u32, 4, 8, 16];
                let mut acc = [0f64; 4]; let mut tot = 0f64;
                for qi in 0..nqo { let q = &queries[qi * dim..(qi + 1) * dim];
                    YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); });
                    std::hint::black_box(s.search_sq4(q, kk));
                    let mut v = YT.with(|t| t.borrow().touched.clone()); v.sort_unstable(); v.dedup();
                    tot += v.len() as f64;
                    for (gi, &g) in gs.iter().enumerate() { let mut b: Vec<u32> = v.iter().map(|x| x / g).collect(); b.dedup(); acc[gi] += b.len() as f64; }
                }
                for (gi, &g) in gs.iter().enumerate() { eprintln!("H60 groups g={g}: distinct groups/q={:.1} vs codes/q={:.1} ratio={:.3}", acc[gi] / nqo as f64, tot / nqo as f64, acc[gi] / tot); }
                return;
            }
            for &kmod in &[1u32, 8, 32, 128] {
                let (mut tq, mut tl, mut cov, mut cov_cnt) = (0f64, 0f64, 0f64, 0f64);
                for qi in 0..nqo {
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    let (a, first) = run(&mut s, q);
                    // 地标：first ∪ N(first) ∪ N²(first) 中 id % kmod == 0 且离 q 最近者
                    let mut cand: Vec<u32> = vec![first];
                    for &v in st.neighbors(first) { cand.push(v); for &w in st.neighbors(v) { cand.push(w); } }
                    let l = cand.iter().copied().filter(|&c| c % kmod == 0).min_by(|&x, &y| {
                        let dx = raven::distance::l2_simd(q, &pv[x as usize * dim..(x as usize + 1) * dim]);
                        let dy = raven::distance::l2_simd(q, &pv[y as usize * dim..(y as usize + 1) * dim]);
                        dx.total_cmp(&dy) });
                    let Some(l) = l else { continue };
                    let lv: Vec<f32> = pv[l as usize * dim..(l as usize + 1) * dim].to_vec();
                    let (b, _) = run(&mut s, &lv);
                    let inter = { let (mut i, mut j, mut c) = (0, 0, 0); while i < a.len() && j < b.len() { if a[i] == b[j] { c += 1; i += 1; j += 1; } else if a[i] < b[j] { i += 1; } else { j += 1; } } c };
                    tq += a.len() as f64; tl += b.len() as f64; cov += inter as f64 / a.len() as f64; cov_cnt += 1.0;
                }
                eprintln!("H59 trace oracle ef={ef} landmarks=1/{kmod}: |T(q)|={:.1} |T(l)|={:.1} coverage={:.3} (n={})", tq / cov_cnt, tl / cov_cnt, cov / cov_cnt, cov_cnt);
            }
            return;
        }
        if let Ok(spec) = std::env::var("RAVEN_SV3B") {
            sv3b_run(&p, &base, &queries, dim, nq, &gt_dist, gtk, metric == "angular", &spec, repeats, &efs, kk, &cache_dir, &hash);
            return;
        }
        // R31 SV-3（诊断原型，默认不走）：图定位 + 邻近簇流式扫描确认。RAVEN_SV3="NL;E1/E2;P1/P2;T1/T2"，与 --ef 基线在同进程内逐轮交替
        if let Ok(spec) = std::env::var("RAVEN_SV3") {
            sv3_run(&p, &base, &queries, dim, nq, &gt_dist, gtk, metric == "angular", &spec, repeats, &efs, kk, &cache_dir, &hash);
            return;
        }
        let angular = metric == "angular";
        if let Ok(pfx) = std::env::var("RAVEN_PQDET") {
            pqdet_run(&p, &base, &queries, dim, nq, &gt_dist, gtk, angular, &efs, kk, &pfx);
            return;
        }
        if let Ok(pfx) = std::env::var("RAVEN_PQEF") {
            pqef_run(&p, &base, &queries, dim, nq, &gt_dist, gtk, angular, repeats, &efs, kk, &pfx);
            return;
        }
        if let Some(blk) = std::env::var("RAVEN_HR_AB_PAIR").ok().and_then(|v| v.parse::<usize>().ok()) {
            hr_ab_pair_run(&p, &base, &queries, dim, nq, &gt_dist, gtk, angular, repeats, &efs, kk, &out, blk.max(1));
            return;
        }
        if std::env::var("RAVEN_HR_AB").map(|v| v == "1").unwrap_or(false) {
            hr_ab_run(&p, &base, &queries, dim, nq, &gt_dist, gtk, angular, repeats, &efs, kk, &out);
            return;
        }
        // R47 原型：查询内精度切换。RAVEN_TP="ef1:ns"：LP 码 searcher(ef1) 取精排前 ns 个作种子 → 全维码 searcher(--ef 值) seeded 搜索
        if let Ok(spec) = std::env::var("RAVEN_TP") {
            let mut it = spec.split(':'); let ef1: usize = it.next().unwrap().parse().unwrap(); let ns: usize = it.next().map(|x| x.parse().unwrap()).unwrap_or(4);
            let mut csv = String::from("mode,ef,recall10,qps_median,qps_runs\n");
            for &ef in &efs {
                let mut s1 = p.searcher_lp(ef1); let mut s2 = p.searcher_full(ef);
                let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq]; let mut runs = Vec::new(); let mut seeds: Vec<u32> = Vec::with_capacity(ns);
                for _ in 0..repeats.max(1) {
                    let t = Instant::now();
                    for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                        seeds.clear(); seeds.extend(s1.search_auto(q, ns).iter().map(|x| x.0));
                        res[qi] = s2.search_sq4_from_seeds(q, kk, &seeds).iter().map(|x| p.orig_id(x.0)).collect(); }
                    runs.push(nq as f64 / t.elapsed().as_secs_f64());
                }
                let mut tot = 0usize;
                for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                    tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                        (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
                let rec = tot as f64 / (nq * kk) as f64; let mut sr = runs.clone(); sr.sort_by(|a, b| a.total_cmp(b)); let med = sr[sr.len() / 2];
                eprintln!("tp ef1={ef1} ns={ns} ef2={ef} recall={rec:.5} qps={med:.0}");
                csv += &format!("tp{ef1}x{ns},{ef},{rec:.5},{med:.1},{}\n", runs.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(";"));
            }
            if !out.is_empty() { std::fs::write(&out, csv).unwrap(); }
            return;
        }
        let mut csv = String::from("mode,ef,recall10,qps_median,qps_runs\n");
        // R38-6：RAVEN_DUMP_ADJ=path 导出 L0 邻接（原 id 顺序；每行 u32 度 + u32 原 id 邻居），诊断用
        if let Ok(path) = std::env::var("RAVEN_DUMP_ADJ") {
            let st = p.graph.storage(); let nn = p.vectors.len() / dim;
            let mut o2n = vec![0u32; nn]; for i in 0..nn as u32 { o2n[p.orig_id(i) as usize] = i; }
            let mut buf: Vec<u8> = Vec::with_capacity(nn * 50 * 4);
            for o in 0..nn { let nb = st.neighbors(o2n[o]); let nb: Vec<u32> = nb.iter().copied().filter(|&v| (v as usize) < nn).collect();
                buf.extend_from_slice(&(nb.len() as u32).to_le_bytes()); for v in nb { buf.extend_from_slice(&p.orig_id(v).to_le_bytes()); } }
            std::fs::write(&path, buf).unwrap(); eprintln!("R38-6 dumped adjacency to {path}"); return;
        }
        // R38-5：pipeline 路径的 oracle 种子注入（RAVEN_ORACLE=k,off，GT 原 id → 重排后 id）
        let o2n_or: Vec<u32> = if oracle_k > 0 { let nn = p.vectors.len() / dim; let mut m = vec![0u32; nn]; for i in 0..nn as u32 { m[p.orig_id(i) as usize] = i; } m } else { Vec::new() };
        for &ef in &efs {
            let mut s = p.searcher_k(ef, kk);
            let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
            let mut runs = Vec::new();
            raven::counters::prof::reset();
            raven::counters::reset();
            // H72：RAVEN_WARM=1 每个查询先未计时跑一遍（工作集进 cache），只计第二遍——区分访存受限与计算受限
            let warm = std::env::var("RAVEN_WARM").map(|v| v == "1").unwrap_or(false);
            for _ in 0..repeats.max(1) {
                let t = Instant::now();
                let mut tw = 0f64;
                for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                    raven::graph::vamana::YT.with(|t| t.borrow_mut().qid = qi as u32);
                    if oracle_k > 0 { let sd: Vec<u32> = gt_u[qi * gtk + oracle_off..qi * gtk + oracle_off + oracle_k].iter().map(|&g| o2n_or[g as usize]).collect(); raven::graph::vamana::set_oracle_seeds(&sd); }
                    if warm { let _ = p.run_auto(&mut s, q, kk); let t1 = Instant::now(); res[qi] = p.run_auto(&mut s, q, kk).iter().map(|x| p.orig_id(x.0)).collect(); tw += t1.elapsed().as_secs_f64(); }
                    else { res[qi] = p.run_auto(&mut s, q, kk).iter().map(|x| p.orig_id(x.0)).collect(); } }
                runs.push(nq as f64 / if warm { tw } else { t.elapsed().as_secs_f64() });
            }
            let mut tot = 0usize;
            for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
                let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                    (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
            let rec = tot as f64 / (nq * kk) as f64;
            runs.sort_by(|a, b| a.total_cmp(b));
            let med = runs[runs.len() / 2];
            eprintln!("pipeline ef={ef} recall={rec:.5} qps={med:.0}");
            if std::env::var("RAVEN_HR_STAT").map(|v| v == "1").unwrap_or(false) {
                let h = raven::graph::cellconfirm::hr_stat_take(); let c = h[0].max(1) as f64;
                eprintln!("R43 HR stat ef={ef}: confirms {} cells/q {:.1} survivors/q {:.1} inserted/q {:.1} c8pass/q {:.1}", h[0], h[1] as f64 / c, h[2] as f64 / c, h[3] as f64 / c, h[4] as f64 / c);
                if h[5] > 0 { eprintln!("R44 HR block oracle ef={ef}: blocks/q {:.0} blocks-with-survivor {:.3} | pts/q {:.0} tri-prunable {:.3} pts-in-zero-blocks {:.3}", h[5] as f64 / c, h[6] as f64 / h[5] as f64, h[7] as f64 / c, h[8] as f64 / h[7].max(1) as f64, h[9] as f64 / h[7].max(1) as f64); }
                if h[10] + h[11] > 0 || std::env::var("RAVEN_HR_ORM").is_ok() { eprintln!("R45 HR proj oracle ef={ef}: proj-prunable {:.3} pts-in-fully-prunable-blocks {:.3}", h[10] as f64 / h[7].max(1) as f64, h[11] as f64 / h[7].max(1) as f64); }
            }
            if raven::counters::ENABLED {
                // H77c：流水线路径下的取码计数（NTC = 完整 SQ4 + 部分折算），按查询×轮次平均
                let (f, q, h) = raven::counters::snapshot();
                let (_pc, pd) = raven::counters::partial_snapshot();
                let den = (nq * repeats.max(1) * if warm { 2 } else { 1 }) as f64;
                eprintln!("pipeline_counters ef={ef} full={:.1} ntc={:.1} hops={:.2}", f as f64 / den, (q as f64 + pd as f64 / dim as f64) / den, h as f64 / den);
            }
            if raven::graph::vamana::YT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
                let st: Vec<u64> = raven::graph::vamana::YT_STATS.iter().map(|x| x.swap(0, std::sync::atomic::Ordering::Relaxed)).collect();
                let nn = (nq * repeats.max(1)) as f64;
                eprintln!("H58 yield ef={ef}: hops/q={:.1} full-pool hops={:.1} | useless hops={:.1} ({:.1}%) codes in useless hops={:.1}/{:.1} ({:.1}%) | zero-insert hops={:.1} codes={:.1} ({:.1}%)",
                    st[0] as f64 / nn, st[6] as f64 / nn, st[1] as f64 / nn, 100.0 * st[1] as f64 / st[0] as f64, st[4] as f64 / nn, st[3] as f64 / nn, 100.0 * st[4] as f64 / st[3] as f64,
                    st[2] as f64 / nn, st[5] as f64 / nn, 100.0 * st[5] as f64 / st[3] as f64);
            }
            if cfg!(feature = "diag_prof") {
                let pr = raven::counters::prof::snapshot();
                let nn = (nq * repeats.max(1)) as f64;
                eprintln!("prof ns/query: nav={:.0} L0={:.0} rerank={:.0} | L0 hop parts: pop+pf={:.0} visited={:.0} dist+ins={:.0} | peek-miss {}/{} hops", pr[0] as f64 / nn, pr[1] as f64 / nn, pr[2] as f64 / nn, pr[3] as f64 / nn, pr[4] as f64 / nn, pr[5] as f64 / nn, pr[6], pr[7]);
            }
            if let Some(pfx) = &yt_dump { let d = raven::graph::vamana::YT.with(|t| t.borrow_mut().dump.as_mut().map(std::mem::take).unwrap_or_default()); std::fs::write(format!("{pfx}.ef{ef}.rows"), d).unwrap(); }
            // R42：RAVEN_RES_DUMP=prefix 导出每查询结果 id（原 id，u32 LE，nq×k，不足补 u32::MAX）供离线 oracle（SV-2 现实口径）
            if let Ok(pfx) = std::env::var("RAVEN_RES_DUMP") {
                let mut b: Vec<u8> = Vec::with_capacity(nq * kk * 4);
                for r in &res { for j in 0..kk { b.extend_from_slice(&r.get(j).copied().unwrap_or(u32::MAX).to_le_bytes()); } }
                std::fs::write(format!("{pfx}.ef{ef}.ids"), b).unwrap();
            }
            csv += &format!("pipeline,{ef},{rec:.5},{med:.1},{}\n", runs.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(";"));
        }
        let _ = &gt;
        if !out.is_empty() { std::fs::write(&out, csv).unwrap(); }
        return;
    }
    if nav_rebuild_m > 0 {
        let t = Instant::now();
        let nav = LayeredNavigation::build(&base, dim, graph.storage(), graph.entry_point(), nav_rebuild_m, nav_rebuild_k);
        eprintln!("nav rebuilt m={nav_rebuild_m} k={nav_rebuild_k} max_level={} in {:.1}s", nav.max_level(), t.elapsed().as_secs_f64());
        graph.set_layered_nav(nav);
    }
    let ds = graph.degree_stats();
    eprintln!("graph key: {key}\ngraph hash: {hash}\ndegree stats: {ds:?}");
    if mode == "build" {
        println!("# key={key}\n# hash={hash}");
        return;
    }
    {
        // H08：检查邻接表是否按到中心节点的距离升序（search_degree 依赖此前提）
        let st = graph.storage();
        let step = (n / 5000).max(1);
        let (mut tot, mut sorted, mut inv, mut pairs) = (0usize, 0usize, 0usize, 0usize);
        for i in (0..n).step_by(step) {
            let c = &base[i * dim..(i + 1) * dim];
            let nb = st.neighbors(i as u32);
            let ds: Vec<f64> = nb.iter().map(|&j| exact_dist(c, &base[j as usize * dim..(j as usize + 1) * dim])).collect();
            let bad = ds.windows(2).filter(|w| w[1] < w[0]).count();
            inv += bad; pairs += ds.len().saturating_sub(1);
            tot += 1; if bad == 0 { sorted += 1; }
        }
        eprintln!("H08 adjacency sorted: {sorted}/{tot} nodes fully ascending, adjacent inversions {inv}/{pairs}; search_degree={sd}");
    }
    // H17：SQ4 稳健量化范围（每维 [clip, 1-clip] 分位数；0 = min/max 旧行为；<0 = H17b 逐维 MSE 自适应）
    let sq4_clip: f32 = arg(&args, "--sq4-clip", -1.0f32); // H17 accepted 2026-10-30: MSE-adaptive default (0 = old min/max)
    let sq4_plain = SQ4Dataset::build_clipped(&base, dim, sq4_clip);
    eprintln!("H17 sq4_clip={sq4_clip}");

    // ── H07：维度置换 + 检查点阈值标定（仅搜索侧，不改图）──
    let nchk = dim / ea_gran; // 段数（32 维/段 = NEON 16 字节；16 维/段 = 8 字节）
    let (perm, frac): (Vec<usize>, Vec<f64>) = {
        // 每维能量：near = 图邻居对的码平方差；var = 码方差（等价随机对）
        let cb = sq4_plain.code_bytes;
        let nib = |c: &[u8], d: usize| -> i32 { ((c[d / 2] >> ((d % 2) * 4)) & 0x0F) as i32 };
        let mut e = vec![0f64; dim];
        let step = (n / 20000).max(1);
        if perm_mode == "var" {
            let mut s1 = vec![0f64; dim];
            let mut s2 = vec![0f64; dim];
            let mut cnt = 0f64;
            for i in (0..n).step_by(step) {
                let c = &sq4_plain.codes[i * cb..(i + 1) * cb];
                for d in 0..dim { let x = nib(c, d) as f64; s1[d] += x; s2[d] += x * x; }
                cnt += 1.0;
            }
            for d in 0..dim { e[d] = s2[d] / cnt - (s1[d] / cnt).powi(2); }
        } else {
            let st = graph.storage();
            for i in (0..n).step_by(step) {
                let ci = &sq4_plain.codes[i * cb..(i + 1) * cb];
                for &j in st.neighbors(i as u32) {
                    let cj = &sq4_plain.codes[j as usize * cb..(j as usize + 1) * cb];
                    for d in 0..dim { let t = nib(ci, d) - nib(cj, d); e[d] += (t * t) as f64; }
                }
            }
        }
        let mut perm: Vec<usize> = (0..dim).collect();
        if perm_mode != "none" {
            perm.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
        }
        // frac 始终以"近邻对能量"在当前顺序下的累计占比给出（用于有损阈值）
        let mut en = vec![0f64; dim];
        {
            let st = graph.storage();
            for i in (0..n).step_by(step) {
                let ci = &sq4_plain.codes[i * cb..(i + 1) * cb];
                for &j in st.neighbors(i as u32) {
                    let cj = &sq4_plain.codes[j as usize * cb..(j as usize + 1) * cb];
                    for d in 0..dim { let t = nib(ci, d) - nib(cj, d); en[d] += (t * t) as f64; }
                }
            }
        }
        let tot: f64 = en.iter().sum();
        let mut frac = Vec::new();
        let mut acc = 0f64;
        for (j, &d) in perm.iter().enumerate() {
            acc += en[d];
            if (j + 1) % ea_gran == 0 { frac.push(acc / tot); }
        }
        (perm, frac)
    };
    eprintln!("H07 perm={perm_mode} gran={ea_gran} frac={:?} perm_head={:?}", frac.iter().map(|f| format!("{f:.3}")).collect::<Vec<_>>(), &perm[..16.min(dim)]);
    let mut l0 = L0Opts::default();
    if ea {
        l0.early_abandon = true;
        // 检查点只在完整段之间（c+1 < 段数），且库内阈值数组长 8（H14：d=100 → 3 段 + 4 维尾，检查点 32/64；d≥288 截到前 8 个）
        let seg_mask: u32 = if nchk >= 2 { ((1u64 << (nchk - 1).min(32)) - 1) as u32 } else { 0 };
        l0.check_mask = ea_mask & seg_mask & 0xFF;
        l0.gran16 = ea_gran == 16;
        if !ea_scale_s.is_empty() {
            for (c, x) in ea_scale_s.split(',').enumerate() { l0.scale[c] = x.trim().parse().unwrap(); }
        } else if ea_eps >= 0.0 {
            for c in 0..nchk.min(8) { l0.scale[c] = (frac[c] * (1.0 + ea_eps as f64)) as f32; }
        }
    }
    l0.es_k = es_k;
    l0.es_patience = es_pat;
    l0.stop_gamma = arg(&args, "--stop-gamma", 0.0f32);
    l0.stop_k = arg(&args, "--stop-k", 10usize);
    l0.stop_min = arg(&args, "--stop-min", 0usize);
    eprintln!("H07 l0_opts={l0:?}");
    // H08：第二张图（--alt-R R2 [--alt-r-soft] [--alt-nav-rebuild-m 32 --alt-nav-rebuild-k 16]），只读缓存，
    // 用于 --vlist 进程内交错 QPS（跨图对比也能抵消 VM 漂移）。第二张图的维序按其自身邻居重算（near）或 none。
    let alt_r: usize = arg(&args, "--alt-R", 0);
    let alt = if alt_r > 0 {
        let alt_rs: usize = arg(&args, "--alt-r-soft", (alt_r as f32 * 1.5) as usize);
        let akey = format!(
            "v1|data={data}|n={n}|dim={dim}|prune=Directional|R={alt_r}|L={l_build}|alpha={alpha}|alpha_core={alpha_core}|r_soft={alt_rs}|iters=2|nav_m={nav_m}|n_jumps=0|rmin_div=4|seed={seed}"
        );
        let ahash = format!("{:016x}", fnv64(&akey));
        let apath = format!("{cache_dir}/{ahash}.bin");
        let mut g = VamanaGraph::load(std::path::Path::new(&apath)).unwrap_or_else(|e| panic!("alt graph {apath} not cached: {e:?}"));
        let am: usize = arg(&args, "--alt-nav-rebuild-m", 0);
        let ak: usize = arg(&args, "--alt-nav-rebuild-k", alt_r / 2);
        if am > 0 {
            let nav = LayeredNavigation::build(&base, dim, g.storage(), g.entry_point(), am, ak);
            g.set_layered_nav(nav);
        }
        let ap = if perm_mode != "none" { near_perm(&g, &sq4_plain.codes, sq4_plain.code_bytes, n, dim) } else { (0..dim).collect() };
        let asq = if perm_mode != "none" { SQ4Dataset::build_permuted_clipped(&base, dim, &ap, sq4_clip) } else { SQ4Dataset::build_clipped(&base, dim, sq4_clip) };
        eprintln!("alt graph key: {akey}\nalt hash: {ahash} nav_rebuild m={am} k={ak} degree stats: {:?} perm_head={:?}", g.degree_stats(), &ap[..16.min(dim)]);
        Some((g, asq, ap, akey))
    } else { None };
    let sq4 = if perm_mode != "none" { SQ4Dataset::build_permuted_clipped(&base, dim, &perm, sq4_clip) } else { sq4_plain };
    let use_perm = perm_mode != "none";
    // H20：建索引时自动选 search_degree（只用库内样本点做查询，不碰测试集）。
    // 300 个库内点（排除自身）的精确 top-10 → 每个候选 sd 二分找最小 ef 使校准召回 ≥ 目标，计时取最快。
    let sd = if arg::<usize>(&args, "--sd-auto", 0) > 0 {
        let t0 = Instant::now();
        let nc: usize = arg(&args, "--sd-auto-n", 300);
        let target: f64 = arg(&args, "--sd-auto-recall", 0.95f64);
        let cands: Vec<usize> = arg_str(&args, "--sd-auto-cands", "24,32,40,0").split(',').map(|x| x.trim().parse().unwrap()).collect();
        use rand::RngCore;
        let mut rng = ChaCha8Rng::seed_from(7);
        let ids: Vec<usize> = (0..nc).map(|_| (rng.next_u64() % n as u64) as usize).collect();
        let dist = |a: &[f32], b: &[f32]| if angular { angular_dist(a, b) } else { exact_dist(a, b) };
        // H20 v2：校准查询 = 库内点与其随机一个图邻居的中点（angular 归一化），不在库中，难度接近真实查询
        let mut qv: Vec<f32> = Vec::with_capacity(nc * dim);
        for &qi in &ids {
            let nb = graph.neighbors(qi as u32); let j = nb[(rng.next_u64() as usize) % nb.len().max(1)] as usize;
            let mut q: Vec<f32> = (0..dim).map(|d| 0.5 * (base[qi * dim + d] + base[j * dim + d])).collect();
            if angular { let nn = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); for x in q.iter_mut() { *x /= nn; } }
            qv.extend(q);
        }
        let thr: Vec<f64> = (0..nc).map(|ci| {
            let q = &qv[ci * dim..(ci + 1) * dim];
            let mut top: Vec<f64> = Vec::with_capacity(k + 1);
            for j in 0..n {
                let d = dist(q, &base[j * dim..(j + 1) * dim]);
                if top.len() < k { top.push(d); top.sort_by(|a, b| a.total_cmp(b)); } else if d < top[k - 1] { top[k - 1] = d; top.sort_by(|a, b| a.total_cmp(b)); }
            }
            top[k - 1] + 1e-3
        }).collect();
        let eval = |sdv: usize, ef: usize| -> (f64, f64) {
            let mut s = GraphSearcher::new(&base, &graph, ef);
            s.with_prefetch_offset(po); s.with_rerank_factor(rerank); s.with_sq4(&sq4);
            s.with_nav_opts(nav_opts); s.with_l0_opts(l0); s.with_search_degree(sdv);
            if use_perm { s.with_sq4_perm(&perm); }
            let mut hit = 0usize; let mut best = f64::MAX;
            for rep in 0..3 { let t = Instant::now(); let mut h = 0usize;
                for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim];
                    let r = s.search_sq4(q, k);
                    h += r.iter().take(k).filter(|x| dist(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count(); }
                best = best.min(t.elapsed().as_secs_f64()); if rep == 0 { hit = h; } }
            (hit as f64 / (nc * k) as f64, best)
        };
        let mut pick = (sd, f64::MAX);
        for &c in &cands {
            let (mut lo, mut hi) = (k, 4096usize);
            if eval(c, hi).0 < target { eprintln!("H20 sd={c} cannot reach {target}"); continue; }
            while hi - lo > (lo / 16).max(2) { let mid = (lo + hi) / 2; if eval(c, mid).0 >= target { hi = mid } else { lo = mid } }
            let (rc, tm) = eval(c, hi);
            eprintln!("H20 sd={c} ef={hi} calib_recall={rc:.4} time={:.4}s", tm);
            if tm < pick.1 { pick = (c, tm); }
        }
        eprintln!("H20 sd-auto picked sd={} in {:.1}s", pick.0, t0.elapsed().as_secs_f64());
        pick.0
    } else { sd };
    // H02：--reorder bfs|lbfs → vlist 记号 c<sd>：主图按新节点序重排后的副本（结果译回原 ID，逐位等价）
    let reorder_mode = arg_str(&args, "--reorder", "");
    let reo = if !reorder_mode.is_empty() {
        let t = Instant::now();
        let o2n = match reorder_mode.as_str() { "bfs" => raven::bench::util::bfs_order_pub(&graph), "lbfs" => raven::bench::util::level_bfs_order(&graph), m => panic!("reorder {m}") };
        let r = raven::bench::util::reorder_with(&graph, &sq4, &base, dim, &o2n);
        eprintln!("H02 reorder={reorder_mode} in {:.1}s", t.elapsed().as_secs_f64());
        Some(r)
    } else { None };
    // H14：码数组对齐诊断（码跨 cache line 时每个邻居需 2 行；prefetch 只发 1 行）
    {
        let cb = sq4.code_bytes;
        let off = sq4.codes.as_ptr() as usize % 64;
        let straddle = (0..n.min(65536)).filter(|&i| (off + i * cb) % 64 + cb > 64).count() as f64 / n.min(65536) as f64;
        eprintln!("H14 sq4 codes ptr%64={off} code_bytes={cb} frac_codes_straddling_2_lines={straddle:.3}");
    }

    // ── 评测 ──
    // 返回 (总耗时秒, 每 CHUNK 个查询的分块耗时)。分块中位数用于抵抗 VM steal 突发噪声。
    const CHUNK: usize = 1000;
    // H14：变体级开关（--vlist 记号后缀 "-ea" 关掉 H07a 提前放弃，"-nav" 关掉 H09b SQ4 导航+携带，"-dd" 关掉 H09a 去重）
    let nav_legacy_b = NavOpts { dedup: nav_opts.dedup, sq4: false, carry: 1, stop_level: nav_opts.stop_level };
    let run_gvo = |use_alt: u8, ef: usize, sd: usize, no_ea: bool, no_nav: bool, no_dd: bool, results: &mut Vec<Vec<u32>>| -> (f64, Vec<f64>) {
        let (g, sq, pm, bs, n2o): (&VamanaGraph, &SQ4Dataset, &[usize], &[f32], Option<&[u32]>) = match (&alt, &reo, use_alt) {
            (Some((ag, asq, ap, _)), _, 1) => (ag, asq, ap, &base[..], None),
            (_, Some(r), 2) => (&r.graph, &r.sq4, &perm, &r.train[..], Some(&r.new_to_old[..])),
            _ => (&graph, &sq4, &perm, &base[..], None),
        };
        let mut s = GraphSearcher::new(bs, g, ef);
        s.with_prefetch_offset(po);
        s.with_rerank_factor(rerank);
        s.with_sq4(sq);
        let mut nv = if no_nav { nav_legacy_b } else { nav_opts };
        if no_dd { nv.dedup = false; }
        s.with_nav_opts(nv);
        s.with_l0_opts(if no_ea { L0Opts::default() } else { l0 });
        s.with_search_degree(sd);
        if use_perm { s.with_sq4_perm(pm); }
        let mut chunks = Vec::with_capacity(nq / CHUNK + 1);
        let t = Instant::now();
        let mut tc = Instant::now();
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            if oracle_k > 0 { raven::graph::vamana::set_oracle_seeds(&gt_u[qi * gtk + oracle_off..qi * gtk + oracle_off + oracle_k]); }
            let r = s.search_sq4(q, k);
            let slot = &mut results[qi];
            slot.clear();
            match n2o { Some(m) => slot.extend(r.iter().map(|x| m[x.0 as usize])), None => slot.extend(r.iter().map(|x| x.0)) }
            if (qi + 1) % CHUNK == 0 {
                chunks.push(tc.elapsed().as_secs_f64());
                tc = Instant::now();
            }
        }
        (t.elapsed().as_secs_f64(), chunks)
    };
    let run_gv = |use_alt: u8, ef: usize, sd: usize, results: &mut Vec<Vec<u32>>| run_gvo(use_alt, ef, sd, false, false, false, results);
    let run_sd = |ef: usize, sd: usize, results: &mut Vec<Vec<u32>>| run_gv(0, ef, sd, results);
    let run_once = |ef: usize, results: &mut Vec<Vec<u32>>| run_sd(ef, sd, results);
    let recall_of = |results: &[Vec<u32>]| -> f64 {
        let mut tot = 0usize;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let thr = gt_dist[qi * gtk + k - 1] as f64 + 1e-3;
            tot += results[qi]
                .iter()
                .take(k)
                .filter(|&&id| {
                    let x = &base[id as usize * dim..(id as usize + 1) * dim];
                    (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr
                })
                .count();
        }
        let _ = &gt;
        tot as f64 / (nq * k) as f64
    };

    let mut results: Vec<Vec<u32>> = vec![Vec::with_capacity(k); nq];
    let mut rows: Vec<Row> = Vec::new();
    // H08/H11：进程内交错 QPS（--sd-list 0,28,...）：每轮每个 ef 依次跑各 sd（每轮轮换起点），
    // 漂移/VM steal 在毫秒级被各变体平均分摊。每变体输出 recall 与中位数 QPS，并在 0.95 插值。
    let sd_list: Vec<usize> = arg_str(&args, "--sd-list", "").split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse().unwrap()).collect();
    // --vlist a0,b28,b48：a = 主图，b = --alt-R 图，后接 search_degree（0 = 不限）
    // H14：记号可带后缀 -ea / -nav / -dd（见 run_gvo），如 a0-ea、b32-nav-ea
    let mut vars: Vec<(u8, usize, String, bool, bool, bool)> = sd_list.iter().map(|&x| (0u8, x, format!("{x}"), false, false, false)).collect();
    for t in arg_str(&args, "--vlist", "").split(',').filter(|x| !x.is_empty()) {
        let t = t.trim();
        let ua: u8 = match &t[..1] { "a" => 0, "b" => 1, "c" => 2, _ => panic!("vlist entry {t}") };
        assert!(ua != 1 || alt.is_some(), "vlist uses b but no --alt-R");
        assert!(ua != 2 || reo.is_some(), "vlist uses c but no --reorder");
        let mut parts = t[1..].split('-');
        let sdv: usize = parts.next().unwrap().parse().unwrap();
        let (mut ne, mut nn, mut nd) = (false, false, false);
        for f in parts { match f { "ea" => ne = true, "nav" => nn = true, "dd" => nd = true, _ => panic!("vlist flag {f}") } }
        vars.push((ua, sdv, t.to_string(), ne, nn, nd));
    }
    if mode == "qps" && !vars.is_empty() {
        let nv = vars.len();
        let mut rec = vec![vec![0f64; efs.len()]; nv];
        for (vi, (ua, sdv, _, ne, nn, nd)) in vars.iter().enumerate() {
            for (i, &ef) in efs.iter().enumerate() { let _ = run_gvo(*ua, ef, *sdv, *ne, *nn, *nd, &mut results); rec[vi][i] = recall_of(&results); }
        }
        let mut times = vec![vec![Vec::<f64>::new(); efs.len()]; nv];
        for rep in 0..repeats {
            for (i, &ef) in efs.iter().enumerate() {
                for j in 0..nv {
                    let vi = (j + rep + i) % nv;
                    let (secs, _) = run_gvo(vars[vi].0, ef, vars[vi].1, vars[vi].3, vars[vi].4, vars[vi].5, &mut results);
                    times[vi][i].push(nq as f64 / secs);
                }
            }
            eprintln!("repeat {}/{} done", rep + 1, repeats);
        }
        let mut out_s = String::from("mode,sd,ef,recall10,qps_median,qps_cv,qps_runs\n");
        let mut summ = String::new();
        for (vi, (_, _, sdv, _, _, _)) in vars.iter().enumerate() {
            let mut vrows: Vec<Row> = Vec::new();
            for (i, &ef) in efs.iter().enumerate() {
                let mut v = times[vi][i].clone();
                v.sort_by(|a, b| a.total_cmp(b));
                let med = v[v.len() / 2];
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                let cv = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt() / mean;
                out_s += &format!("qpsx,{sdv},{ef},{:.5},{med:.1},{cv:.4},{}\n", rec[vi][i], times[vi][i].iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(";"));
                eprintln!("sd={sdv} ef={ef} recall={:.4} qps_med={med:.0} cv={:.2}%", rec[vi][i], cv * 100.0);
                vrows.push(Row { ef, recall: rec[vi][i], full: f64::NAN, quant: f64::NAN, hops: f64::NAN, nav: f64::NAN,
                    q_complete: f64::NAN, q_part_cnt: f64::NAN, q_part_ndc: f64::NAN, qps_med: med, qps_cv: cv, qps_all: vec![], qps_rob: f64::NAN, qps_rob_cv: f64::NAN });
            }
            summ += &format!("# sd={sdv} @0.95 qps={:?} max_cv={:.4}\n", interp(&vrows, 0.95, |r| r.qps_med), vrows.iter().map(|r| r.qps_cv).fold(0.0, f64::max));
        }
        if let Some((_, _, _, ak)) = &alt { out_s += &format!("# alt(b)={ak} alt_nav_rebuild_k={}\n", arg::<usize>(&args, "--alt-nav-rebuild-k", alt_r / 2)); }
        out_s += &format!("# metric={metric}\n# key={key}\n# hash={hash}\n# nav_opts={nav_opts:?} nav_rebuild_m={nav_rebuild_m} nav_rebuild_k={nav_rebuild_k}\n# l0_opts={l0:?} perm={perm_mode}\n");
        out_s += &summ;
        print!("{out_s}");
        if !out.is_empty() { std::fs::write(&out, out_s.as_bytes()).expect("write out"); eprintln!("wrote {out}"); }
        return;
    }
    if mode == "counters" {
        for &ef in &efs {
            raven::counters::reset();
            let _ = run_once(ef, &mut results);
            let (f, q, h) = raven::counters::snapshot();
            let nv = raven::counters::nav_snapshot();
            let (pc, pd) = raven::counters::partial_snapshot();
            let rec = recall_of(&results);
            // quant = 完整 SQ4 次数 + 提前放弃按"已处理维度 / dim"折算（与基线可比）
            let pndc = pd as f64 / dim as f64 / nq as f64;
            let row = Row {
                ef, recall: rec,
                full: f as f64 / nq as f64, quant: q as f64 / nq as f64 + pndc, hops: h as f64 / nq as f64, nav: nv as f64 / nq as f64,
                q_complete: q as f64 / nq as f64, q_part_cnt: pc as f64 / nq as f64, q_part_ndc: pndc,
                qps_med: f64::NAN, qps_cv: f64::NAN, qps_all: vec![], qps_rob: f64::NAN, qps_rob_cv: f64::NAN,
            };
            eprintln!("ef={ef:4} recall={rec:.4} full={:.1} quant={:.1} (complete {:.1} + partial {:.1}x→{:.1}) hops={:.1} nav={:.1}", row.full, row.quant, row.q_complete, row.q_part_cnt, row.q_part_ndc, row.hops, row.nav);
            rows.push(row);
        }
    } else {
        // 预热 + recall
        let mut recalls = Vec::new();
        for &ef in &efs {
            let _ = run_once(ef, &mut results);
            recalls.push(recall_of(&results));
        }
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); efs.len()];
        let mut robs: Vec<Vec<f64>> = vec![Vec::new(); efs.len()];
        for rep in 0..repeats {
            for (i, &ef) in efs.iter().enumerate() {
                let (secs, mut ch) = run_once(ef, &mut results);
                times[i].push(nq as f64 / secs);
                ch.sort_by(|a, b| a.total_cmp(b));
                robs[i].push(CHUNK as f64 / ch[ch.len() / 2]);
            }
            eprintln!("repeat {}/{} done", rep + 1, repeats);
        }
        let med_cv = |v: &[f64]| -> (f64, f64) {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.total_cmp(b));
            let med = s[s.len() / 2];
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            let sd = (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
            (med, sd / mean)
        };
        for (i, &ef) in efs.iter().enumerate() {
            let (med, cv) = med_cv(&times[i]);
            let (rmed, rcv) = med_cv(&robs[i]);
            let row = Row {
                ef, recall: recalls[i], full: f64::NAN, quant: f64::NAN, hops: f64::NAN, nav: f64::NAN,
                q_complete: f64::NAN, q_part_cnt: f64::NAN, q_part_ndc: f64::NAN,
                qps_med: med, qps_cv: cv, qps_all: times[i].clone(), qps_rob: rmed, qps_rob_cv: rcv,
            };
            eprintln!("ef={ef:4} recall={:.4} qps_med={med:.0} cv={:.2}% | robust={rmed:.0} cv={:.2}%", row.recall, cv * 100.0, rcv * 100.0);
            rows.push(row);
        }
    }

    // ── 输出 ──
    let mut csv = String::from("mode,ef,recall10,full_dist,quant_dist,hops,nav_dist,qps_median,qps_cv,qps_robust_median,qps_robust_cv,qps_runs,quant_complete,quant_partial_cnt,quant_partial_ndc\n");
    for r in &rows {
        csv += &format!(
            "{mode},{},{:.5},{:.2},{:.2},{:.2},{:.2},{:.1},{:.4},{:.1},{:.4},{},{:.2},{:.2},{:.2}\n",
            r.ef, r.recall, r.full, r.quant, r.hops, r.nav, r.qps_med, r.qps_cv, r.qps_rob, r.qps_rob_cv,
            r.qps_all.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(";"),
            r.q_complete, r.q_part_cnt, r.q_part_ndc
        );
    }
    let target = 0.95;
    let summary = if mode == "counters" {
        format!(
            "# @0.95 full_dist={:?} quant_dist={:?} ndc_total={:?} hops={:?} nav_dist={:?} quant_complete={:?} quant_partial_cnt={:?} quant_partial_ndc={:?}\n",
            interp(&rows, target, |r| r.full),
            interp(&rows, target, |r| r.quant),
            interp(&rows, target, |r| r.full + r.quant),
            interp(&rows, target, |r| r.hops),
            interp(&rows, target, |r| r.nav),
            interp(&rows, target, |r| r.q_complete),
            interp(&rows, target, |r| r.q_part_cnt),
            interp(&rows, target, |r| r.q_part_ndc),
        )
    } else {
        format!("# @0.95 qps={:?} max_cv={:.4} | qps_robust={:?} max_robust_cv={:.4}\n",
            interp(&rows, target, |r| r.qps_med),
            rows.iter().map(|r| r.qps_cv).fold(0.0, f64::max),
            interp(&rows, target, |r| r.qps_rob),
            rows.iter().map(|r| r.qps_rob_cv).fold(0.0, f64::max))
    };
    csv += &format!("# key={key}\n# hash={hash}\n# nav_opts={nav_opts:?} nav_rebuild_m={nav_rebuild_m} nav_rebuild_k={nav_rebuild_k}\n# metric={metric} sq4_clip={sq4_clip} l0_opts={l0:?} search_degree={sd} perm={perm_mode} gran={ea_gran} frac={frac:?}\n");
    csv += &summary;
    print!("{csv}");
    if !out.is_empty() {
        let mut f = std::fs::File::create(&out).expect("create out");
        f.write_all(csv.as_bytes()).unwrap();
        eprintln!("wrote {out}");
    }
}

/// H61 去重叠图 oracle（第 9 轮）。RAVEN_H61=src:ntrain:eftrain:policy:params:evalfrom
///   src = test（用测试查询自身统计 = 作弊上界）| synth3（库内点-邻居中点 + 垂直噪声，无需 GT）
///   policy = ratio（按 (U+a)/(F+1) 升序删去已取码边的最低 frac 比例；params = frac 列表 '/' 分隔）
///          | zero（删 U==0 且 F>=fmin 的边；params = fmin 列表）
///   U = 取码后被插入且之后被扩展/进入结果（"有用插入"），F = 取码次数（目标节点首次被访问时经由该边）
///   删边只在前 sd 个邻居内进行并截断（不回填第 sd+1 名），每节点至少保留 floor 个（RAVEN_H61_FLOOR，默认 8）
///   评测：测试查询 evalfrom..nq 在每个 ef 上的 recall（容差口径）/ NTC / hops，插值到 0.95
#[allow(clippy::too_many_arguments)]
fn h61_run(p: &mut raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize, base: &[f32], gt_dist: &[f32], gtk: usize, angular: bool, efs: &[usize], out: &str) {
    use rand::RngCore;
    use raven::graph::vamana::{YT, YT_MODE};
    let f: Vec<&str> = spec.split(':').collect();
    let src = f[0]; let ntrain: usize = f[1].parse().unwrap(); let eft: usize = f[2].parse().unwrap();
    let policy = f[3]; let params: Vec<f64> = f[4].split('/').map(|x| x.parse().unwrap()).collect();
    let evalfrom: usize = f.get(5).map(|x| x.parse().unwrap()).unwrap_or(0);
    let floor: usize = std::env::var("RAVEN_H61_FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let kk = 10usize;
    let n = p.graph.len();
    let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
    let pv: Vec<f32> = p.vectors.to_vec();
    // 训练查询
    let tq: Vec<f32> = if src == "test" { queries[..ntrain.min(nq) * dim].to_vec() } else {
        let mut rng = ChaCha8Rng::seed_from(4242);
        let mut qv = Vec::with_capacity(ntrain * dim);
        for _ in 0..ntrain {
            let qi = (rng.next_u64() % n as u64) as usize;
            let nb = p.graph.neighbors(qi as u32);
            let j = if nb.is_empty() { qi } else { nb[(rng.next_u64() as usize) % nb.len()] as usize };
            let dd: f32 = (0..dim).map(|d| { let t = pv[qi * dim + d] - pv[j * dim + d]; t * t }).sum::<f32>().sqrt();
            let mut noise: Vec<f32> = (0..dim).map(|_| { let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / 9007199254740993.0; let u2 = (rng.next_u64() >> 11) as f64 / 9007199254740992.0; ((-2.0 * u1.ln()).sqrt() * (6.283185307 * u2).cos()) as f32 }).collect();
            let nn: f32 = noise.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in noise.iter_mut() { *x *= 0.5 * dd / nn; }
            let st = qv.len();
            qv.extend((0..dim).map(|d| 0.5 * (pv[qi * dim + d] + pv[j * dim + d]) + noise[d]));
            if angular { let s: f32 = qv[st..].iter().map(|x| x * x).sum::<f32>().sqrt(); for x in qv[st..].iter_mut() { *x /= s; } }
        }
        qv
    };
    let mt = tq.len() / dim;
    // 统计
    let t0 = Instant::now();
    let mut fc = vec![0u32; n * sd]; let mut uc = vec![0u32; n * sd]; let mut ic = vec![0u32; n * sd];
    YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| { let mut t = t.borrow_mut(); t.keep_touch = true; t.keep_all = true; });
    {
        let mut s = p.searcher(eft);
        let st = p.graph.storage();
        for qi in 0..mt {
            YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); t.hops.clear(); t.ins.clear(); t.meta.clear(); });
            let r = s.search_sq4(&tq[qi * dim..(qi + 1) * dim], kk);
            YT.with(|t| {
                let t = t.borrow();
                let mut useful: std::collections::HashSet<u32> = t.popped.iter().copied().collect();
                for x in &r { useful.insert(x.0); }
                let mut off = 0usize;
                for (h, &(es, a, l, _)) in t.hops.iter().enumerate() {
                    let u = t.popped[h];
                    let nb = st.neighbors(u);
                    let ins = &t.ins[a as usize..(a + l) as usize];
                    for &v in &t.touched[off..off + es as usize] {
                        if let Some(i) = nb.iter().take(sd).position(|&x| x == v) {
                            let e = u as usize * sd + i;
                            fc[e] += 1;
                            if ins.contains(&v) { ic[e] += 1; if useful.contains(&v) { uc[e] += 1; } }
                        }
                    }
                    off += es as usize;
                }
            });
        }
    }
    YT_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| { let mut t = t.borrow_mut(); t.keep_touch = false; t.keep_all = false; t.touched.clear(); t.popped.clear(); t.hops.clear(); t.ins.clear(); });
    let fetched = fc.iter().filter(|&&x| x > 0).count();
    let totf: u64 = fc.iter().map(|&x| x as u64).sum(); let toti: u64 = ic.iter().map(|&x| x as u64).sum(); let totu: u64 = uc.iter().map(|&x| x as u64).sum();
    let zf: u64 = fc.iter().zip(&uc).filter(|(_, &u)| u == 0).map(|(&f, _)| f as u64).sum();
    eprintln!("H61 stats src={src} ntrain={mt} ef={eft} ({:.1}s): edges fetched>0 {fetched}/{} ; fetch events {totf} ins {toti} useful {totu} ; fetches on U==0 edges {:.1}%",
        t0.elapsed().as_secs_f64(), n * sd, 100.0 * zf as f64 / totf as f64);
    // 备份原邻接
    let orig: Vec<Vec<u32>> = (0..n).map(|u| p.graph.storage().neighbors(u as u32).to_vec()).collect();
    // 评测
    let eval = |p: &raven::pipeline::Prepared| -> Vec<(f64, f64, f64, f64)> {
        let mut rows = Vec::new();
        for &ef in efs {
            raven::counters::reset();
            let mut s = p.searcher(ef);
            let mut tot = 0usize;
            for qi in evalfrom..nq {
                let q = &queries[qi * dim..(qi + 1) * dim];
                raven::graph::vamana::dir_set_query(q);
                raven::graph::vamana::rb_set_query(q);
                raven::graph::vamana::pq_set_query(q);
                let r = s.search_sq4(q, kk);
                let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                tot += r.iter().take(kk).filter(|x| { let id = p.orig_id(x.0) as usize; let xv = &base[id * dim..(id + 1) * dim];
                    (if angular { angular_dist(q, xv) } else { exact_dist(q, xv) }) <= thr }).count();
            }
            let m = (nq - evalfrom) as f64;
            let (fq, qq, h) = raven::counters::snapshot(); let (pc, _) = raven::counters::partial_snapshot();
            rows.push((tot as f64 / (m * kk as f64), (fq as f64 + qq as f64 + pc as f64) / m, h as f64 / m, ef as f64));
        }
        rows
    };
    let at95 = |rows: &[(f64, f64, f64, f64)], k: usize| -> f64 {
        for w in rows.windows(2) { let (a, b) = (w[0], w[1]); let ga = [a.0, a.1, a.2, a.3]; let gb = [b.0, b.1, b.2, b.3];
            if ga[0] <= 0.95 && gb[0] >= 0.95 { let t = if (gb[0] - ga[0]).abs() < 1e-12 { 0.0 } else { (0.95 - ga[0]) / (gb[0] - ga[0]) }; return ga[k] + t * (gb[k] - ga[k]); } }
        f64::NAN };
    let mut csv = String::from("variant,param,removed_frac,ef,recall,ntc,hops\n");
    let base_rows = eval(p);
    for r in &base_rows { csv += &format!("base,0,0,{},{:.5},{:.1},{:.2}\n", r.3, r.0, r.1, r.2); }
    let (b_ntc, b_h) = (at95(&base_rows, 1), at95(&base_rows, 2));
    eprintln!("H61 base: NTC@0.95={b_ntc:.1} hops@0.95={b_h:.2}");
    // 候选边排序
    let a = 0.05f64;
    let mut cand: Vec<(f64, usize)> = (0..n * sd).filter(|&e| fc[e] > 0).map(|e| ((uc[e] as f64 + a) / (fc[e] as f64 + 1.0), e)).collect();
    cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(fc[y.1].cmp(&fc[x.1])));
    let tot_edges: usize = (0..n).map(|u| orig[u].len().min(sd)).sum();
    for &prm in &params {
        let mut del = vec![false; n * sd];
        match policy {
            "ratio" => { let k = (prm * cand.len() as f64) as usize; for &(_, e) in cand.iter().take(k) { del[e] = true; } }
            "zero" => { for e in 0..n * sd { if uc[e] == 0 && fc[e] as f64 >= prm { del[e] = true; } } }
            _ => panic!("policy"),
        }
        let mut removed = 0usize;
        for u in 0..n {
            let o = &orig[u]; let lim = o.len().min(sd);
            let keep_n = (0..lim).filter(|&i| !del[u * sd + i]).count();
            let mut nl: Vec<u32> = Vec::with_capacity(lim);
            let mut extra = floor.saturating_sub(keep_n);
            for i in 0..lim { if !del[u * sd + i] { nl.push(o[i]); } else if extra > 0 { nl.push(o[i]); extra -= 1; } }
            removed += lim - nl.len();
            p.graph.storage_mut().set_neighbors(u as u32, &nl);
        }
        let rows = eval(p);
        let rf = removed as f64 / tot_edges as f64;
        for r in &rows { csv += &format!("{policy},{prm},{rf:.4},{},{:.5},{:.1},{:.2}\n", r.3, r.0, r.1, r.2); }
        let (ntc, h) = (at95(&rows, 1), at95(&rows, 2));
        eprintln!("H61 {policy}={prm}: removed {:.2}% of first-{sd} edges -> NTC@0.95={ntc:.1} ({:+.1}%) hops@0.95={h:.2} ({:+.1}%) | recall@ef: {}",
            100.0 * rf, 100.0 * (ntc / b_ntc - 1.0), 100.0 * (h / b_h - 1.0), rows.iter().map(|r| format!("{}:{:.4}", r.3, r.0)).collect::<Vec<_>>().join(" "));
        for u in 0..n { p.graph.storage_mut().set_neighbors(u as u32, &orig[u]); }
    }
    if !out.is_empty() { std::fs::write(out, csv).unwrap(); }
}

/// H62/H63 轨迹统计（第 9 轮）。RAVEN_H62=ef[:nq]：在测试查询上跑生产管线，按跳重建（弹出 u, u 的前 sd 个邻居, 新取码集合, 插入集合），输出
///   (1) 相邻两次扩展邻接的 Jaccard、每跳邻接中"已访问"比例（H62 集合级扩展的上界依据）；
///   (2) 每个被取码节点在整条查询中被已扩展节点引用的次数 refs（H63 门卫准入的上界依据），按 有用/仅插入/拒绝 分桶；
///   (3) 被取码时的边名次分布。
fn h62_run(p: &raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize) {
    use raven::graph::vamana::{YT, YT_MODE};
    let f: Vec<&str> = spec.split(':').collect();
    let ef: usize = f[0].parse().unwrap();
    let m: usize = f.get(1).map(|x| x.parse().unwrap()).unwrap_or(nq).min(nq);
    let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
    let st = p.graph.storage();
    YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| { let mut t = t.borrow_mut(); t.keep_touch = true; t.keep_all = true; });
    let mut s = p.searcher(ef);
    let (mut jac_s, mut jac_n, mut vis_s, mut vis_n) = (0f64, 0f64, 0f64, 0f64);
    // refs 桶 1,2,3,4,5+ × 类别 useful/ins_only/rejected
    let mut tab = [[0u64; 5]; 3];
    let mut rank_tab = [[0u64; 4]; 3]; // 名次 0-7, 8-15, 16-23, 24-31
    let mut first_rank_useful_le8 = 0u64;
    let mut nhops = 0u64;
    let mut refs: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for qi in 0..m {
        YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); t.hops.clear(); t.ins.clear(); t.meta.clear(); });
        let r = s.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10);
        YT.with(|t| {
            let t = t.borrow();
            let mut useful: std::collections::HashSet<u32> = t.popped.iter().copied().collect();
            for x in &r { useful.insert(x.0); }
            let insset: std::collections::HashSet<u32> = t.ins.iter().copied().collect();
            refs.clear();
            let mut prev: Vec<u32> = Vec::new();
            let mut off = 0usize;
            let mut fetch_rank: Vec<(u32, u8)> = Vec::new();
            for (h, &(es, _, _, _)) in t.hops.iter().enumerate() {
                let u = t.popped[h];
                let nb: Vec<u32> = st.neighbors(u).iter().take(sd).copied().collect();
                for &v in &nb { *refs.entry(v).or_insert(0) += 1; }
                if !prev.is_empty() {
                    let inter = nb.iter().filter(|x| prev.contains(x)).count();
                    jac_s += inter as f64 / (nb.len() + prev.len() - inter).max(1) as f64; jac_n += 1.0;
                }
                vis_s += 1.0 - es as f64 / nb.len().max(1) as f64; vis_n += 1.0;
                for &v in &t.touched[off..off + es as usize] { let i = nb.iter().position(|&x| x == v).unwrap_or(31); fetch_rank.push((v, i as u8)); }
                off += es as usize;
                prev = nb; nhops += 1;
            }
            for &(v, i) in &fetch_rank {
                let c = if useful.contains(&v) { 0 } else if insset.contains(&v) { 1 } else { 2 };
                let rc = (*refs.get(&v).unwrap_or(&1)).clamp(1, 5) as usize - 1;
                tab[c][rc] += 1;
                rank_tab[c][(i as usize / 8).min(3)] += 1;
                if c == 0 && i < 8 { first_rank_useful_le8 += 1; }
            }
        });
    }
    let _ = first_rank_useful_le8;
    YT_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
    eprintln!("H62 ef={ef} q={m} hops/q={:.1}: mean Jaccard(adj(u_t), adj(u_t-1))={:.3}; mean fraction of popped node's adjacency already visited={:.3}", nhops as f64 / m as f64, jac_s / jac_n, vis_s / vis_n);
    let names = ["useful", "ins_only", "rejected"];
    let tot: u64 = tab.iter().flatten().sum();
    for c in 0..3 { let rs: u64 = tab[c].iter().sum();
        eprintln!("H63 refs[{}] per q={:.1} ({:.1}% of fetches): refs=1 {:.1}%  2 {:.1}%  3 {:.1}%  4 {:.1}%  5+ {:.1}% | edge rank 0-7 {:.1}% 8-15 {:.1}% 16-23 {:.1}% 24-31 {:.1}%",
            names[c], rs as f64 / m as f64, 100.0 * rs as f64 / tot as f64,
            100.0 * tab[c][0] as f64 / rs as f64, 100.0 * tab[c][1] as f64 / rs as f64, 100.0 * tab[c][2] as f64 / rs as f64, 100.0 * tab[c][3] as f64 / rs as f64, 100.0 * tab[c][4] as f64 / rs as f64,
            100.0 * rank_tab[c][0] as f64 / rs as f64, 100.0 * rank_tab[c][1] as f64 / rs as f64, 100.0 * rank_tab[c][2] as f64 / rs as f64, 100.0 * rank_tab[c][3] as f64 / rs as f64); }
    let r1: u64 = (0..3).map(|c| tab[c][0]).sum();
    eprintln!("H63 static upper bound (skip all refs==1 fetches): saves {:.1}% of L0 fetches, loses {:.1}% of useful nodes", 100.0 * r1 as f64 / tot as f64, 100.0 * tab[0][0] as f64 / tab[0].iter().sum::<u64>() as f64);
}

/// H63 门卫准入 sweep（第 9 轮）。RAVEN_H63="k0/k1/need,k0/k1/need,..."（基线 0/0/0 总是先测）
///   每个配置在全部测试查询、每个 ef 上测 recall（容差口径）/ NTC / hops，插值到 0.95。
#[allow(clippy::too_many_arguments)]
fn h63_run(p: &mut raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize, base: &[f32], gt_dist: &[f32], gtk: usize, angular: bool, efs: &[usize], out: &str) {
    let kk = 10usize;
    let eval = |p: &raven::pipeline::Prepared| -> Vec<(f64, f64, f64, f64)> {
        let mut rows = Vec::new();
        for &ef in efs {
            raven::counters::reset();
            let mut s = p.searcher(ef);
            let mut tot = 0usize;
            for qi in 0..nq {
                let q = &queries[qi * dim..(qi + 1) * dim];
                // R23 修复：H69b/H69c/H70/H70b 经 h63_run 评测时从未逐查询设置投影/LUT（只有 h61_run 设置）
                raven::graph::vamana::dir_set_query(q); raven::graph::vamana::rb_set_query(q); raven::graph::vamana::pq_set_query(q); raven::graph::vamana::hr_set_query(q);
                let r = s.search_sq4(q, kk);
                let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                tot += r.iter().take(kk).filter(|x| { let id = p.orig_id(x.0) as usize; let xv = &base[id * dim..(id + 1) * dim];
                    (if angular { angular_dist(q, xv) } else { exact_dist(q, xv) }) <= thr }).count();
            }
            let m = nq as f64;
            let (fq, qq, h) = raven::counters::snapshot(); let (pc, _) = raven::counters::partial_snapshot();
            rows.push((tot as f64 / (m * kk as f64), (fq as f64 + qq as f64 + pc as f64) / m, h as f64 / m, ef as f64));
        }
        rows
    };
    let at95 = |rows: &[(f64, f64, f64, f64)], k: usize| -> f64 {
        for w in rows.windows(2) { let (a, b) = (w[0], w[1]); let ga = [a.0, a.1, a.2, a.3]; let gb = [b.0, b.1, b.2, b.3];
            if ga[0] <= 0.95 && gb[0] >= 0.95 { let t = if (gb[0] - ga[0]).abs() < 1e-12 { 0.0 } else { (0.95 - ga[0]) / (gb[0] - ga[0]) }; return ga[k] + t * (gb[k] - ga[k]); } }
        f64::NAN };
    let mut csv = String::from("k0,k1,need,r,ef,recall,ntc,hops\n");
    let mut cfgs: Vec<[u32; 4]> = vec![[0, 0, 0, 0]];
    for c in spec.split(',').filter(|x| !x.is_empty()) { let v: Vec<u32> = c.split('/').map(|x| x.parse().unwrap()).collect(); cfgs.push([v[0], v[1], v[2], *v.get(3).unwrap_or(&0)]); }
    let mut b = (f64::NAN, f64::NAN);
    for c in cfgs {
        raven::graph::vamana::set_dk(c[0], c[1], c[2], c[3]);
        let rows = eval(p);
        for r in &rows { csv += &format!("{},{},{},{},{},{:.5},{:.1},{:.2}\n", c[0], c[1], c[2], c[3], r.3, r.0, r.1, r.2); }
        let (ntc, h) = (at95(&rows, 1), at95(&rows, 2));
        if c[2] == 0 { b = (ntc, h); }
        eprintln!("H63 k0={} k1={} need={} r={}: NTC@0.95={ntc:.1} ({:+.1}%) hops@0.95={h:.2} ({:+.1}%) | {}", c[0], c[1], c[2], c[3], 100.0 * (ntc / b.0 - 1.0), 100.0 * (h / b.1 - 1.0),
            rows.iter().map(|r| format!("ef{}:{:.4}/{:.0}", r.3, r.0, r.1)).collect::<Vec<_>>().join(" "));
        if !out.is_empty() { std::fs::write(out, &csv).unwrap(); }
    }
    raven::graph::vamana::set_dk(0, 0, 0, 0);
}

/// H64 三角定位准入 oracle（第 9 轮）。RAVEN_H64=ef[:nq]：在测试查询的基线轨迹上，对每次 L0 取码 (u→v) 计算
///   d_u = 弹出时 u 的 SQ4 距离、w = 池最差距离、L = SQ4 码空间边长 |u−v|²（构建期可存于邻接行空闲行，1 字节）、名次 i，
///   估计 est = d_u + L − 2c·sqrt(d_u·L)（假设查询方向与边夹角余弦 = c），按 est/w 从大到小跳过，报告
///   "跳过的 拒绝/仅插入/有用 取码比例" 曲线（静态上界：跳过被拒取码不改变轨迹）。
fn h64_run(p: &raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize) {
    use raven::graph::vamana::{YT, YT_MODE};
    let f: Vec<&str> = spec.split(':').collect();
    let ef: usize = f[0].parse().unwrap();
    let m: usize = f.get(1).map(|x| x.parse().unwrap()).unwrap_or(nq).min(nq);
    let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
    let st = p.graph.storage();
    YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| { let mut t = t.borrow_mut(); t.keep_touch = true; t.keep_all = true; });
    let mut s = p.searcher(ef);
    // (d_u, w, L, rank, pop_rank, class) class 0 useful 1 ins_only 2 rejected
    let mut rows: Vec<(f32, f32, f32, u8, u16, u8)> = Vec::new();
    for qi in 0..m {
        YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); t.hops.clear(); t.ins.clear(); t.meta.clear(); });
        let r = s.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10);
        YT.with(|t| {
            let t = t.borrow();
            let mut useful: std::collections::HashSet<u32> = t.popped.iter().copied().collect();
            for x in &r { useful.insert(x.0); }
            let mut off = 0usize;
            for (h, &(es, a, l, _)) in t.hops.iter().enumerate() {
                let (u, du, w, pr) = t.meta[h];
                let nb = st.neighbors(u);
                let ins = &t.ins[a as usize..(a + l) as usize];
                for &v in &t.touched[off..off + es as usize] {
                    let i = nb.iter().take(sd).position(|&x| x == v).unwrap_or(31);
                    let lv = raven::quant::sq4::l2_sq4_raw(p.sq4.code(u as usize), p.sq4.code(v as usize));
                    let c = if useful.contains(&v) { 0 } else if ins.contains(&v) { 1 } else { 2 };
                    rows.push((du, w, lv, i as u8, pr as u16, c));
                }
                off += es as usize;
            }
        });
    }
    YT_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
    let tot = [0u8, 1, 2].map(|c| rows.iter().filter(|r| r.5 == c).count() as f64);
    eprintln!("H64 ef={ef} q={m}: fetches/q={:.1} useful {:.1} ins_only {:.1} rejected {:.1}", rows.len() as f64 / m as f64, tot[0] / m as f64, tot[1] / m as f64, tot[2] / m as f64);
    // 只在池满时（w 有限）才可能跳过
    for &cc in &[0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let mut sc: Vec<(f32, u8)> = rows.iter().map(|r| {
            let est = if r.1.is_finite() && r.1 < f32::MAX / 2.0 { (r.0 + r.2 - 2.0 * cc * (r.0 * r.2).sqrt()) / r.1 } else { 0.0 };
            (est, r.5) }).collect();
        sc.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out = String::new();
        for &lossu in &[0.001f64, 0.005, 0.01, 0.02, 0.05] {
            let (mut cu, mut ci, mut cr) = (0f64, 0f64, 0f64);
            for &(_, c) in &sc { match c { 0 => { if (cu + 1.0) / tot[0] > lossu { break; } cu += 1.0; } 1 => ci += 1.0, _ => cr += 1.0 } }
            out += &format!(" | useful-loss {:.1}%: skip rej {:.1}% ins {:.1}% => fetches -{:.1}%", lossu * 100.0, 100.0 * cr / tot[2], 100.0 * ci / tot[1], 100.0 * (cr + ci + cu) / rows.len() as f64);
        }
        eprintln!("H64 c={cc}{out}");
    }
    // 对照：仅用名次 / 仅用 d_u/w
    for (name, key) in [("rank", 0usize), ("du/w", 1), ("L/w", 2)] {
        let mut sc: Vec<(f32, u8)> = rows.iter().map(|r| (match key { 0 => r.3 as f32 + 0.001 * r.4 as f32, 1 => if r.1 < f32::MAX / 2.0 { r.0 / r.1 } else { 0.0 }, _ => if r.1 < f32::MAX / 2.0 { r.2 / r.1 } else { 0.0 } }, r.5)).collect();
        sc.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out = String::new();
        for &lossu in &[0.01f64, 0.05] {
            let (mut cu, mut ci, mut cr) = (0f64, 0f64, 0f64);
            for &(_, c) in &sc { match c { 0 => { if (cu + 1.0) / tot[0] > lossu { break; } cu += 1.0; } 1 => ci += 1.0, _ => cr += 1.0 } }
            out += &format!(" | useful-loss {:.0}%: fetches -{:.1}%", lossu * 100.0, 100.0 * (cr + ci + cu) / rows.len() as f64);
        }
        eprintln!("H64 baseline-score {name}{out}");
    }
}

/// H65 L2 常驻 1–2 字节/向量"取码前粗码"oracle（第 9 轮）。RAVEN_H65=ef:nq:K（K = k-means 簇数 ≤ 65536）
///   粗码 = 簇号（K≤256 时 1 B）+ 残差范数 r_v（1 B）；整表 n×2 B（1M = 2 MB）设计为常驻 L2，取码前零 DRAM 访问。
///   估计 est = |q−c_v|² + r_v²（交叉项期望 0），按 est/(池最差距离的 f32 等价) 从大到小跳过，报告 有用损失 vs 取码节省 曲线；
///   另报被拒取码的真实距离/最差距离分位数（判断粗码需要多准）。
fn h65_run(p: &raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize) {
    use raven::graph::vamana::{YT, YT_MODE};
    use rand::RngCore;
    let f: Vec<&str> = spec.split(':').collect();
    let ef: usize = f[0].parse().unwrap();
    let m: usize = f[1].parse::<usize>().unwrap().min(nq);
    let kc: usize = f[2].parse().unwrap();
    let n = p.graph.len();
    let pv = &p.vectors[..];
    let l2 = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    // k-means（样本 50K，8 轮），再全量分配
    let t0 = Instant::now();
    let mut rng = ChaCha8Rng::seed_from(7);
    let ns = 50_000.min(n);
    let samp: Vec<usize> = (0..ns).map(|_| (rng.next_u64() % n as u64) as usize).collect();
    let mut cent: Vec<f32> = (0..kc).flat_map(|j| pv[samp[j % ns] * dim..(samp[j % ns] + 1) * dim].to_vec()).collect();
    use rayon::prelude::*;
    for _ in 0..8 {
        let asg: Vec<usize> = samp.par_iter().map(|&i| { let x = &pv[i * dim..(i + 1) * dim]; (0..kc).min_by(|&a, &b| l2(x, &cent[a * dim..(a + 1) * dim]).total_cmp(&l2(x, &cent[b * dim..(b + 1) * dim]))).unwrap() }).collect();
        let mut sum = vec![0f64; kc * dim]; let mut cnt = vec![0usize; kc];
        for (si, &c) in asg.iter().enumerate() { cnt[c] += 1; for d in 0..dim { sum[c * dim + d] += pv[samp[si] * dim + d] as f64; } }
        for c in 0..kc { if cnt[c] > 0 { for d in 0..dim { cent[c * dim + d] = (sum[c * dim + d] / cnt[c] as f64) as f32; } } }
    }
    let asg: Vec<(u32, f32)> = (0..n).into_par_iter().map(|i| { let x = &pv[i * dim..(i + 1) * dim]; let (c, d) = (0..kc).map(|c| (c, l2(x, &cent[c * dim..(c + 1) * dim]))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap(); (c as u32, d) }).collect();
    eprintln!("H65 kmeans K={kc} in {:.1}s; mean residual² / mean |x|² = {:.3}", t0.elapsed().as_secs_f64(),
        asg.iter().map(|x| x.1 as f64).sum::<f64>() / (0..n).map(|i| pv[i * dim..(i + 1) * dim].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sum::<f64>().max(1e-9));
    // 残差范数 1 字节量化（对数尺度）
    let rmax = asg.iter().map(|x| x.1).fold(0f32, f32::max).max(1e-9);
    let rq: Vec<f32> = asg.iter().map(|x| { let t = (x.1 / rmax).sqrt(); let q = (t * 255.0).round() / 255.0; q * q * rmax }).collect();
    let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
    let st = p.graph.storage();
    YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| { let mut t = t.borrow_mut(); t.keep_touch = true; t.keep_all = true; });
    let mut s = p.searcher(ef);
    let mut rows: Vec<(f32, f32, u8, f32)> = Vec::new(); // (true ratio, est ratio, class, edge-length ratio L/w in SQ4 units)
    for qi in 0..m {
        let q = &queries[qi * dim..(qi + 1) * dim];
        YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); t.hops.clear(); t.ins.clear(); t.meta.clear(); });
        let r = s.search_sq4(q, 10);
        let qc: Vec<f32> = (0..kc).map(|c| l2(q, &cent[c * dim..(c + 1) * dim])).collect();
        YT.with(|t| {
            let t = t.borrow();
            let mut useful: std::collections::HashSet<u32> = t.popped.iter().copied().collect();
            for x in &r { useful.insert(x.0); }
            // SQ4→f32 尺度：弹出节点的 f32 距离 / SQ4 距离 的中位数
            let mut sc: Vec<f32> = t.meta.iter().map(|&(u, du, _, _)| l2(q, &pv[u as usize * dim..(u as usize + 1) * dim]) / du.max(1e-9)).collect();
            sc.sort_by(|a, b| a.total_cmp(b)); let scale = sc[sc.len() / 2];
            let mut off = 0usize;
            for (h, &(es, a, l, _)) in t.hops.iter().enumerate() {
                let (u, _, w, _) = t.meta[h];
                let ins = &t.ins[a as usize..(a + l) as usize];
                let wf = if w < f32::MAX / 2.0 { w * scale } else { f32::INFINITY };
                for &v in &t.touched[off..off + es as usize] {
                    let c = if useful.contains(&v) { 0 } else if ins.contains(&v) { 1 } else { 2 };
                    let dv = l2(q, &pv[v as usize * dim..(v as usize + 1) * dim]);
                    let est = qc[asg[v as usize].0 as usize] + rq[v as usize];
                    let lv = raven::quant::sq4::l2_sq4_raw(p.sq4.code(u as usize), p.sq4.code(v as usize));
                    rows.push((dv / wf, est / wf, c, if w < f32::MAX / 2.0 { lv / w } else { 0.0 }));
                }
                off += es as usize;
            }
            let _ = sd; let _ = st;
        });
    }
    YT_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
    let tot = [0u8, 1, 2].map(|c| rows.iter().filter(|r| r.2 == c).count() as f64);
    let mut rj: Vec<f32> = rows.iter().filter(|r| r.2 == 2 && r.0.is_finite()).map(|r| r.0).collect(); rj.sort_by(|a, b| a.total_cmp(b));
    let qn = |v: &Vec<f32>, x: f64| v[((v.len() as f64 - 1.0) * x) as usize];
    eprintln!("H65 rejected true d/worst quantiles: p10 {:.3} p25 {:.3} p50 {:.3} p75 {:.3} p90 {:.3}", qn(&rj, 0.1), qn(&rj, 0.25), qn(&rj, 0.5), qn(&rj, 0.75), qn(&rj, 0.9));
    for (name, key) in [("coarse est", 1usize), ("oracle true d", 0), ("L/w", 2), ("coarse+0.5L/w", 3), ("coarse+1L/w", 4), ("coarse+2L/w", 5)] {
        let mut sc: Vec<(f32, u8)> = rows.iter().map(|r| (match key { 0 => r.0, 1 => r.1, 2 => r.3, 3 => r.1 + 0.5 * r.3, 4 => r.1 + r.3, _ => r.1 + 2.0 * r.3 }, r.2)).map(|(x, c)| (if x.is_finite() { x } else { 0.0 }, c)).collect();
        sc.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out = String::new();
        for &lossu in &[0.001f64, 0.01, 0.05] {
            let (mut cu, mut ci, mut cr) = (0f64, 0f64, 0f64);
            for &(_, c) in &sc { match c { 0 => { if (cu + 1.0) / tot[0] > lossu { break; } cu += 1.0; } 1 => ci += 1.0, _ => cr += 1.0 } }
            out += &format!(" | useful-loss {:.1}%: skip rej {:.1}% ins {:.1}% => fetches -{:.1}%", lossu * 100.0, 100.0 * cr / tot[2], 100.0 * ci / tot[1], 100.0 * (cr + ci + cu) / rows.len() as f64);
        }
        eprintln!("H65 K={kc} {name}{out}");
    }
}

/// H66 码行即单位 上界（第 9 轮）。RAVEN_H66=ef:nq：贪心几何分组（每个未分组节点与其邻接表中最近的 g−1 个未分组邻居成组，g∈{2,4,8}），
///   统计基线轨迹每查询被取码集合 T(q) 落在多少个不同组 / |T(q)|（=按组取行时的行数比；H60 的 BFS-id 分组对照为 0.963/0.804@g16）。
fn h66_run(p: &raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize) {
    use raven::graph::vamana::{YT, YT_MODE};
    let f: Vec<&str> = spec.split(':').collect();
    let ef: usize = f[0].parse().unwrap();
    let m: usize = f[1].parse::<usize>().unwrap().min(nq);
    let n = p.graph.len();
    let st = p.graph.storage();
    let gs = [2usize, 4, 8];
    let groups: Vec<Vec<u32>> = gs.iter().map(|&g| {
        let mut gid = vec![u32::MAX; n]; let mut next = 0u32;
        for u in 0..n {
            if gid[u] != u32::MAX { continue; }
            gid[u] = next; let mut c = 1;
            for &v in st.neighbors(u as u32) { if c >= g { break; } if gid[v as usize] == u32::MAX { gid[v as usize] = next; c += 1; } }
            next += 1;
        }
        eprintln!("H66 g={g}: {next} groups (mean size {:.2})", n as f64 / next as f64);
        gid }).collect();
    YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| t.borrow_mut().keep_touch = true);
    let mut s = p.searcher(ef);
    let mut acc = [0f64; 3]; let mut useful_acc = [0f64; 3]; let mut tot = 0f64;
    for qi in 0..m {
        YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); });
        std::hint::black_box(s.search_sq4(&queries[qi * dim..(qi + 1) * dim], 10));
        let (mut v, pop) = YT.with(|t| { let t = t.borrow(); (t.touched.clone(), t.popped.clone()) }); v.sort_unstable(); v.dedup();
        tot += v.len() as f64;
        for gi in 0..3 {
            let mut b: Vec<u32> = v.iter().map(|&x| groups[gi][x as usize]).collect(); b.sort_unstable(); b.dedup(); acc[gi] += b.len() as f64;
            // 行内"有用"（被扩展）点的组：每个被扩展点所在组里还有多少被扩展点
            let mut pb: Vec<u32> = pop.iter().map(|&x| groups[gi][x as usize]).collect(); pb.sort_unstable(); let l0 = pb.len(); pb.dedup(); useful_acc[gi] += l0 as f64 / pb.len().max(1) as f64;
        }
    }
    YT_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
    for gi in 0..3 { eprintln!("H66 g={}: distinct groups/q={:.1} vs codes/q={:.1} ratio={:.3}; expanded nodes per touched group={:.2}", gs[gi], acc[gi] / m as f64, tot / m as f64, acc[gi] / tot, useful_acc[gi] / m as f64); }
}

/// H67 查询感知截断精度 oracle（第 10 轮）。RAVEN_H67=ef:nq
///   对基线轨迹中每个在"池满"时被取码的候选 v（SQ4 原始整数距离与池最差 w 比较），用逐 nibble 平方差 e_j 计算：
///   (a) 自然顺序（near 置换=方差序）前缀和首次 > w 所需 nibble 数（确定性拒绝，无损）；
///   (b) 查询感知顺序（按 E_v[(q_j−v_j)²] 降序，由每维码值直方图得到，查询期可算）；
///   (c) oracle 顺序（按 e_j 降序，需已知码，下界）；
///   (d) 位平面：只取高 b 位（b=1,2,3）时的区间下界 > w；
///   并据此算"热段+冷段"方案的取字节节省：bytes = hot·N + cold·N_survive，对照 cb·N。另报有损估计（2-bit 中点 / 前缀外推）在有用损失 0.1/1/5% 时的节省。
fn h67_run(p: &raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize) {
    use raven::graph::vamana::{YT, YT_MODE};
    let f: Vec<&str> = spec.split(':').collect();
    let ef: usize = f[0].parse().unwrap();
    let m: usize = f[1].parse::<usize>().unwrap().min(nq);
    let n = p.graph.len();
    let cb = p.sq4.code_bytes;
    let nd = cb * 2; // nibble 数（dim 为奇数时尾 nibble 恒 0）
    let nib = |c: &[u8], j: usize| -> i32 { let b = c[j / 2]; (if j % 2 == 0 { b & 15 } else { b >> 4 }) as i32 };
    // 每维码值直方图（查询感知顺序用）
    let mut hist = vec![0f64; nd * 16];
    for i in (0..n).step_by(7) { let c = p.sq4.code(i); for j in 0..nd { hist[j * 16 + nib(c, j) as usize] += 1.0; } }
    YT_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
    YT.with(|t| { let mut t = t.borrow_mut(); t.keep_touch = true; t.keep_all = true; });
    let mut s = p.searcher(ef);
    // rows: class, full>w, m_nat, m_qa, m_or, bitLB>w for b=1..3, est2/w, estprefix(nd/4)/w
    struct R { c: u8, rej: bool, mn: u16, mq: u16, mo: u16, bl: [bool; 3], e2: f32, ep: [f32; 3] }
    let mut rows: Vec<R> = Vec::new();
    let mut n_notfull = 0usize; let mut n_all = 0usize;
    let mut qbuf = vec![0f32; dim]; let mut qc = vec![0u8; cb];
    for qi in 0..m {
        let q = &queries[qi * dim..(qi + 1) * dim];
        if p.perm.is_empty() { qbuf.copy_from_slice(q); } else { for (k, &d) in p.perm.iter().enumerate() { qbuf[k] = q[d]; } }
        p.sq4.params.encode_into(&qbuf, &mut qc);
        let qn: Vec<i32> = (0..nd).map(|j| nib(&qc, j)).collect();
        let mut qa: Vec<(f64, usize)> = (0..nd).map(|j| { let h = &hist[j * 16..j * 16 + 16]; let t: f64 = h.iter().sum::<f64>().max(1.0);
            ((0..16).map(|x| h[x] * ((qn[j] - x as i32) as f64).powi(2)).sum::<f64>() / t, j) }).collect();
        qa.sort_by(|a, b| b.0.total_cmp(&a.0));
        let qord: Vec<usize> = qa.iter().map(|x| x.1).collect();
        let mut eexp = vec![0f64; nd]; for x in &qa { eexp[x.1] = x.0; }
        let mut esuf = vec![0f64; nd + 1]; for j in (0..nd).rev() { esuf[j] = esuf[j + 1] + eexp[j]; }
        YT.with(|t| { let mut t = t.borrow_mut(); t.touched.clear(); t.popped.clear(); t.hops.clear(); t.ins.clear(); t.meta.clear(); });
        let r = s.search_sq4(q, 10);
        YT.with(|t| {
            let t = t.borrow();
            let mut useful: std::collections::HashSet<u32> = t.popped.iter().copied().collect();
            for x in &r { useful.insert(x.0); }
            let mut off = 0usize;
            for (h, &(es, a, l, _)) in t.hops.iter().enumerate() {
                let (_, _, w, _) = t.meta[h];
                let ins = &t.ins[a as usize..(a + l) as usize];
                for &v in &t.touched[off..off + es as usize] {
                    n_all += 1;
                    if !(w < f32::MAX / 2.0) { n_notfull += 1; continue; }
                    let c = if useful.contains(&v) { 0 } else if ins.contains(&v) { 1 } else { 2 };
                    let code = p.sq4.code(v as usize);
                    let e: Vec<i32> = (0..nd).map(|j| { let d = qn[j] - nib(code, j); d * d }).collect();
                    let full: i32 = e.iter().sum();
                    let rej = full as f32 > w;
                    let first = |ord: &mut dyn Iterator<Item = usize>| -> u16 { let mut sum = 0i32; let mut k = 0u16; for j in ord { sum += e[j]; k += 1; if sum as f32 > w { return k; } } nd as u16 + 1 };
                    let mn = first(&mut (0..nd));
                    let mq = first(&mut qord.iter().copied());
                    let mut eo = e.clone(); eo.sort_unstable_by(|a, b| b.cmp(a));
                    let mo = { let mut sum = 0i32; let mut k = nd as u16 + 1; for (i, &x) in eo.iter().enumerate() { sum += x; if sum as f32 > w { k = i as u16 + 1; break; } } k };
                    let mut bl = [false; 3]; let mut e2 = 0f32;
                    for b in 1..=3usize {
                        let sh = 4 - b; let mut lb = 0i32; let mut mid = 0f32;
                        for j in 0..nd { let vn = nib(code, j); let lo = (vn >> sh) << sh; let hi = lo + (1 << sh) - 1; let x = qn[j];
                            let d = if x < lo { lo - x } else if x > hi { x - hi } else { 0 }; lb += d * d;
                            if b == 2 { let md = (lo + hi) as f32 / 2.0 - x as f32; mid += md * md + ((1 << sh) * (1 << sh) - 1) as f32 / 12.0; } }
                        bl[b - 1] = lb as f32 > w; if b == 2 { e2 = mid / w; }
                    }
                    let mut ep = [0f32; 3];
                    for (k, &hh) in [nd / 4, nd / 2, nd * 3 / 4].iter().enumerate() {
                        // 前缀外推：前缀 + 其余维的直方图期望（查询感知、与码无关）
                        let pre: i32 = e[..hh].iter().sum();
                        ep[k] = (pre as f64 + esuf[hh]) as f32 / w; }
                    rows.push(R { c, rej, mn, mq, mo, bl, e2, ep });
                }
                off += es as usize;
            }
        });
    }
    YT_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
    let nf = rows.len() as f64;
    let tot = [0u8, 1, 2].map(|c| rows.iter().filter(|r| r.c == c).count() as f64);
    eprintln!("H67 ef={ef} nq={m} cb={cb} nibbles={nd}: fetched {} (pool not full {:.1}%), full-check rows {} useful {:.1}% ins {:.1}% rej {:.1}% (full>w {:.1}%)",
        n_all, 100.0 * n_notfull as f64 / n_all as f64, rows.len(), 100.0 * tot[0] / nf, 100.0 * tot[1] / nf, 100.0 * tot[2] / nf, 100.0 * rows.iter().filter(|r| r.rej).count() as f64 / nf);
    // 无损：前缀所需 nibble 数分布（仅 full>w 的行），以及"热段 h 字节"方案的字节节省（以全部取码 n_all 计，未满池行按全取）
    for (name, sel) in [("natural", 0usize), ("query-aware", 1), ("oracle", 2)] {
        let mut v: Vec<u16> = rows.iter().filter(|r| r.rej).map(|r| [r.mn, r.mq, r.mo][sel]).collect(); v.sort_unstable();
        let qn = |x: f64| v[((v.len() as f64 - 1.0) * x) as usize];
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let mut out = String::new();
        for hb in [cb / 8, cb / 4, cb * 3 / 8, cb / 2, cb * 3 / 4] {
            let hn = (hb * 2) as u16;
            let surv = rows.iter().filter(|r| [r.mn, r.mq, r.mo][sel] > hn).count() as f64 + n_notfull as f64;
            let bytes = hb as f64 * n_all as f64 + (cb - hb) as f64 * surv;
            out += &format!(" | hot {hb}B: survive {:.1}% bytes -{:.1}%", 100.0 * surv / n_all as f64, 100.0 * (1.0 - bytes / (cb as f64 * n_all as f64)));
        }
        eprintln!("H67 lossless {name}: rejected need nibbles p10 {} p25 {} p50 {} p75 {} p90 {} mean {:.1} (of {nd}){out}", qn(0.1), qn(0.25), qn(0.5), qn(0.75), qn(0.9), mean);
    }
    for b in 0..3 {
        let hb = (nd * (b + 1)).div_ceil(8);
        let rj = rows.iter().filter(|r| r.bl[b]).count() as f64;
        let surv = n_all as f64 - rj;
        let bytes = hb as f64 * n_all as f64 + (cb - hb.min(cb)) as f64 * surv;
        eprintln!("H67 lossless bitplane top{}b (hot {hb}B): rejects {:.1}% of full-check rows ({:.1}% of full>w) -> bytes -{:.1}%", b + 1, 100.0 * rj / nf, 100.0 * rj / rows.iter().filter(|r| r.rej).count() as f64, 100.0 * (1.0 - bytes / (cb as f64 * n_all as f64)));
    }
    // 有损：按估计/ w 从大到小拒绝，有用损失预算
    for (name, key, hb) in [("bitplane2 midpoint", 0usize, nd * 2 / 8), ("prefix 1/4 + E[rest]", 1, cb / 4), ("prefix 1/2 + E[rest]", 2, cb / 2), ("prefix 3/4 + E[rest]", 3, cb * 3 / 4)] {
        let mut sc: Vec<(f32, u8)> = rows.iter().map(|r| ([r.e2, r.ep[0], r.ep[1], r.ep[2]][key], r.c)).collect();
        sc.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out = String::new();
        for &lossu in &[0.001f64, 0.01, 0.05] {
            let (mut cu, mut cs) = (0f64, 0f64);
            for &(_, c) in &sc { if c == 0 { if (cu + 1.0) / tot[0] > lossu { break; } cu += 1.0; } cs += 1.0; }
            let surv = n_all as f64 - cs;
            let bytes = hb as f64 * n_all as f64 + (cb - hb) as f64 * surv;
            out += &format!(" | loss {:.1}%: skip {:.1}% of fetches, bytes -{:.1}%", lossu * 100.0, 100.0 * cs / n_all as f64, 100.0 * (1.0 - bytes / (cb as f64 * n_all as f64)));
        }
        eprintln!("H67 lossy {name} (hot {hb}B){out}");
    }
}

/// H125 PGO 式失败驱动补边（图的"剖析引导优化"）：在训练查询上以小 ef 搜索，对每个未找到的真 top-10 近邻 v，
/// 从结果中离 v 最近的节点 u 加边 u→v（插入邻接表首位，挤掉末位），然后在留出查询上重测召回-ef 曲线。
/// spec = "train_lo,train_hi,test_lo,test_hi,ef_train,passes"
fn h125_run(p: &mut raven::pipeline::Prepared, spec: &str, queries: &[f32], nq: usize, dim: usize, base: &[f32], gt: &[u32], gt_dist: &[f32], gtk: usize, angular: bool, efs: &[usize]) {
    let f: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect();
    let (tr_lo, tr_hi, te_lo, te_hi, ef_tr, passes) = (f[0], f[1].min(nq), f[2], f[3].min(nq), f[4], f[5]);
    let kk = 10usize;
    let n = p.graph.len();
    let mut old_to_new = vec![0u32; n];
    for i in 0..n as u32 { old_to_new[p.orig_id(i) as usize] = i; }
    let dist = |a: &[f32], b: &[f32]| if angular { angular_dist(a, b) } else { exact_dist(a, b) };
    let eval = |p: &raven::pipeline::Prepared, tag: &str| {
        for &ef in efs {
            let mut s = p.searcher(ef);
            let mut tot = 0usize;
            let t = Instant::now();
            for qi in te_lo..te_hi {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let r = s.search_sq4(q, kk);
                let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
                tot += r.iter().take(kk).filter(|x| { let id = p.orig_id(x.0) as usize; dist(q, &base[id * dim..(id + 1) * dim]) <= thr }).count();
            }
            let el = t.elapsed().as_secs_f64();
            eprintln!("H125 {tag} ef={ef} recall={:.5} qps={:.0}", tot as f64 / ((te_hi - te_lo) * kk) as f64, (te_hi - te_lo) as f64 / el);
        }
    };
    eval(p, "before");
    let sd = if p.sd > 0 { p.sd } else { p.graph.storage().r_max() };
    let mut added_total = 0usize;
    for pass in 0..passes {
        let mut adds: Vec<(u32, u32)> = Vec::new();
        // H125b：伪查询（索引外的同分布点），近似真值 = 大 ef 搜索；RAVEN_H125_PQ=path,count,ef_gt
        if let Ok(pqs) = std::env::var("RAVEN_H125_PQ") {
            let parts: Vec<&str> = pqs.split(',').collect();
            let (pq_all, pdim) = read_vecs_f32(parts[0]);
            assert_eq!(pdim, dim);
            let cnt: usize = parts[1].parse::<usize>().unwrap().min(pq_all.len() / dim);
            let ef_gt: usize = parts[2].parse().unwrap();
            let mut sg = p.searcher(ef_gt);
            let mut s = p.searcher(ef_tr);
            let vecs = &p.vectors;
            for qi in 0..cnt {
                let q = &pq_all[qi * dim..(qi + 1) * dim];
                let g: Vec<u32> = sg.search_sq4(q, kk).iter().take(kk).map(|x| x.0).collect();
                let r: Vec<u32> = s.search_sq4(q, kk).iter().take(kk).map(|x| x.0).collect();
                for &v in &g {
                    if r.contains(&v) { continue; }
                    let vv = &vecs[v as usize * dim..(v as usize + 1) * dim];
                    let (mut best, mut bd) = (u32::MAX, f64::MAX);
                    for &u in &r { let d = dist(vv, &vecs[u as usize * dim..(u as usize + 1) * dim]); if d < bd { bd = d; best = u; } }
                    if best != u32::MAX { adds.push((best, v)); }
                }
            }
        } else {
            let mut s = p.searcher(ef_tr);
            // H126 诊断：漏掉的真近邻 v 相对结果集的位置：在某结果的前 sd 邻居（已取码、被 SQ4 排出）/ 在 sd 之后的尾部邻居 / 两跳内
            let (mut nmiss, mut in_head, mut in_tail, mut in_2hop) = (0usize, 0usize, 0usize, 0usize);
            for qi in tr_lo..tr_hi {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let r = s.search_sq4(q, kk);
                let fnew: Vec<u32> = r.iter().take(kk).map(|x| x.0).collect();
                for j in 0..kk {
                    let v_old = gt[qi * gtk + j];
                    let v = old_to_new[v_old as usize];
                    if fnew.contains(&v) { continue; }
                    nmiss += 1;
                    let mut h = false; let mut t = false; let mut h2 = false;
                    for &u in &fnew { let nb = p.graph.neighbors(u); let k = sd.min(nb.len());
                        if nb[..k].contains(&v) { h = true; }
                        if nb[k..].contains(&v) { t = true; }
                        if !h2 { for &w in &nb[..k] { let nw = p.graph.neighbors(w); if nw[..sd.min(nw.len())].contains(&v) { h2 = true; break; } } } }
                    if h { in_head += 1; } else if t { in_tail += 1; } else if h2 { in_2hop += 1; }
                }
            }
            eprintln!("H126 diag ef={ef_tr}: misses={nmiss} head(visited,SQ4-ranked-out)={in_head} tail(beyond sd)={in_tail} 2hop={in_2hop} other={}", nmiss - in_head - in_tail - in_2hop);
            for qi in tr_lo..tr_hi {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let r = s.search_sq4(q, kk);
                let found: Vec<u32> = r.iter().take(kk).map(|x| p.orig_id(x.0)).collect();
                for j in 0..kk {
                    let v_old = gt[qi * gtk + j];
                    if found.contains(&v_old) { continue; }
                    let vv = &base[v_old as usize * dim..(v_old as usize + 1) * dim];
                    // 源 = 结果中离 v 最近的节点
                    let (mut best, mut bd) = (u32::MAX, f64::MAX);
                    for &u in &found { let d = dist(vv, &base[u as usize * dim..(u as usize + 1) * dim]); if d < bd { bd = d; best = u; } }
                    if best != u32::MAX { adds.push((old_to_new[best as usize], old_to_new[v_old as usize])); }
                }
            }
        }
        let mut applied = 0usize;
        for (u, v) in adds {
            let cur: Vec<u32> = p.graph.neighbors(u).to_vec();
            if cur.iter().take(sd).any(|&x| x == v) { continue; }
            let mut nl = Vec::with_capacity(cur.len());
            nl.push(v);
            for &x in cur.iter() { if x != v && nl.len() < cur.len().max(1) { nl.push(x); } }
            p.graph.storage_mut().set_neighbors(u, &nl);
            applied += 1;
        }
        added_total += applied;
        eprintln!("H125 pass {pass}: applied {applied} edges (total {added_total}) train=[{tr_lo},{tr_hi}) ef_train={ef_tr}");
        eval(p, &format!("after_pass{pass}"));
    }
}

fn h126_run(pa: &raven::pipeline::Prepared, pb: &raven::pipeline::Prepared, pairs: &[(usize, usize)], queries: &[f32], nq: usize, dim: usize, base: &[f32], gt_dist: &[f32], gtk: usize, angular: bool) {
    let kk = 10usize;
    let dist = |a: &[f32], b: &[f32]| if angular { angular_dist(a, b) } else { exact_dist(a, b) };
    for &(ea, eb) in pairs {
        let mut sa = pa.searcher(ea.max(1));
        let mut sb = pb.searcher(eb.max(1));
        let (mut hit_u, mut hit_a, mut hit_b, mut miss_a, mut miss_b, mut miss_both) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut tsec = 0f64;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let t = Instant::now();
            let ra: Vec<u32> = if ea > 0 { sa.search_sq4(q, kk).iter().take(kk).map(|x| pa.orig_id(x.0)).collect() } else { Vec::new() };
            let rb: Vec<u32> = if eb > 0 { sb.search_sq4(q, kk).iter().take(kk).map(|x| pb.orig_id(x.0)).collect() } else { Vec::new() };
            let mut u: Vec<(f64, u32)> = Vec::with_capacity(2 * kk);
            for &id in ra.iter().chain(rb.iter()) { if !u.iter().any(|x| x.1 == id) { u.push((dist(q, &base[id as usize * dim..(id as usize + 1) * dim]), id)); } }
            u.sort_by(|a, b| a.0.total_cmp(&b.0));
            tsec += t.elapsed().as_secs_f64();
            let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            let ok = |id: u32| dist(q, &base[id as usize * dim..(id as usize + 1) * dim]) <= thr;
            hit_u += u.iter().take(kk).filter(|x| ok(x.1)).count();
            let ha = ra.iter().filter(|&&x| ok(x)).count(); let hb = rb.iter().filter(|&&x| ok(x)).count();
            hit_a += ha; hit_b += hb;
            // 逐近邻独立性：真 top-10 中被 A 漏 / 被 B 漏 / 两者都漏
            let ga: std::collections::HashSet<u32> = ra.iter().copied().collect();
            let gb: std::collections::HashSet<u32> = rb.iter().copied().collect();
            let found_u: std::collections::HashSet<u32> = u.iter().take(kk).map(|x| x.1).collect();
            let _ = found_u;
            let mut good: Vec<u32> = ra.iter().chain(rb.iter()).copied().filter(|&x| ok(x)).collect(); good.sort(); good.dedup();
            let n_ok_total = kk; // 近似：以 10 个真近邻为分母
            let ia = ga.iter().filter(|&&x| ok(x)).count(); let ib = gb.iter().filter(|&&x| ok(x)).count();
            let both = good.iter().filter(|x| ga.contains(x) && gb.contains(x)).count();
            miss_a += n_ok_total - ia; miss_b += n_ok_total - ib;
            miss_both += n_ok_total - (ia + ib - both).min(n_ok_total);
        }
        let d = (nq * kk) as f64;
        let (pa_, pb_, pab) = (miss_a as f64 / d, miss_b as f64 / d, miss_both as f64 / d);
        eprintln!("H126 efA={ea} efB={eb} recall_union={:.5} recall_A={:.5} recall_B={:.5} missA={pa_:.4} missB={pb_:.4} missBoth={pab:.4} indep_pred={:.4} qps={:.0}",
            hit_u as f64 / d, hit_a as f64 / d, hit_b as f64 / d, pa_ * pb_, nq as f64 / tsec);
    }
}

/// R26 段归因：一个查询（记录轮）各段触及的行
struct CmLines { adj: Vec<usize>, vis: Vec<u32>, acc: Vec<u32>, rej: Vec<u32>, nav: Vec<u32>, rr: Vec<u32>, pop: Vec<u32>, off: Vec<u32>, newl: Vec<u32>,
    dist: Vec<f32>, isn: Vec<u8>, acc2: Vec<(u32, f32)>, aoff: Vec<u32>, fin: Vec<(u32, f32)> }
static CMW_MASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn qi_seed(lo: usize, rep: usize) -> usize { lo.wrapping_mul(2654435761) ^ rep }

/// R26 X3：读外部 L0 邻接（u32 n×k，越界/自环 = 空槽），每行按到本节点的精确 L2 升序（供 sd 截断）
/// R28 multi 旋钮：取某侧规格中键 key 的值
fn multi_get(spec: &str, key: &str) -> Option<String> {
    spec.split('@').next().unwrap().split('+').filter_map(|x| x.split_once('=')).find(|(a, _)| *a == key).map(|(_, b)| b.to_string())
}

fn load_ext_l0(ep: &str, k: usize, base: &[f32], n: usize, dim: usize, entry: u32) -> VamanaGraph {
    let raw = std::fs::read(ep).expect("read ext l0");
    assert_eq!(raw.len(), n * k * 4, "ext l0 size");
    let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut st = raven::memory::graph::HybridBlockedCsr::new(n, k);
    let mut deg = 0usize;
    for u in 0..n {
        let x = &base[u * dim..(u + 1) * dim];
        let mut row: Vec<(f32, u32)> = ids[u * k..(u + 1) * k].iter().filter(|&&v| (v as usize) < n && v as usize != u)
            .map(|&v| (base[v as usize * dim..(v as usize + 1) * dim].iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(), v)).collect();
        row.sort_by(|a, b| a.0.total_cmp(&b.0)); row.dedup_by_key(|e| e.1);
        deg += row.len();
        for &(_, v) in &row { st.add_edge(u as u32, v); }
    }
    eprintln!("R26 EXT L0 {ep}: K={k} mean degree {:.2}", deg as f64 / n as f64);
    VamanaGraph::from_storage(st, entry, dim)
}


// ───────────────────────── R31 SV-3 prototype ─────────────────────────
// 图负责定位（ef=E 的常规 RAVEN 搜索），流式扫描负责确认：按查询到 k-means 中心的距离取前 P 个单元，
// 顺序扫描单元内连续存放的 SQ4 码（无依赖链、无 visited），取 SQ4 前 T 名做 f32 rerank，与图的 top-k 合并。
#[allow(clippy::too_many_arguments)]
fn sv3_run(p: &raven::pipeline::Prepared, base: &[f32], queries: &[f32], dim: usize, nq: usize, gt_dist: &[f32], gtk: usize,
           angular: bool, spec: &str, repeats: usize, efs: &[usize], kk: usize, cache_dir: &str, hash: &str) {
    use rayon::prelude::*;
    let f: Vec<&str> = spec.split(';').collect();
    let nl: usize = f[0].parse().unwrap();
    let es: Vec<usize> = f[1].split('/').map(|x| x.parse().unwrap()).collect();
    let ps: Vec<usize> = f[2].split('/').map(|x| x.parse().unwrap()).collect();
    let ts: Vec<usize> = f.get(3).unwrap_or(&"32").split('/').map(|x| x.parse().unwrap()).collect();
    let n = base.len() / dim;
    let l2 = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    // 1. k-means（orig 序，缓存）
    let (cent, asg) = sv3_kmeans(base, dim, n, nl, angular, cache_dir, hash);
    // 2. 单元主序 SQ4 码副本（内部 id）
    let cb = p.sq4.code_bytes;
    let mut o2n = vec![0u32; n]; for i in 0..n as u32 { o2n[p.orig_id(i) as usize] = i; }
    let mut offs = vec![0usize; nl + 1]; for &a in &asg { offs[a as usize + 1] += 1; } for j in 0..nl { offs[j + 1] += offs[j]; }
    let mut fill = offs.clone(); let mut ids = vec![0u32; n]; let mut codes = vec![0u8; n * cb];
    for o in 0..n { let j = asg[o] as usize; let k = fill[j]; fill[j] += 1; let iid = o2n[o]; ids[k] = iid; codes[k * cb..(k + 1) * cb].copy_from_slice(p.sq4.code(iid as usize)); }
    eprintln!("SV3 cells NL={nl} mean {:.0} max {} code_bytes {cb}", n as f64 / nl as f64, (0..nl).map(|j| offs[j + 1] - offs[j]).max().unwrap());
    let perm: Vec<usize> = if p.perm.is_empty() { (0..dim).collect() } else { p.perm.clone() };
    // R31 v2：AVX2 批量核所需的预计算——每点 Σb²（码空间）、int8 中心（dim 填充到 128，查询映射到 [-63,63]+64）
    let kv: u8 = std::env::var("RAVEN_SV3_K").ok().and_then(|v| v.parse().ok()).unwrap_or(if cfg!(target_arch = "x86_64") { 2 } else { 1 });
    let nib = |c: &[u8], d: usize| -> i32 { ((c[d / 2] >> ((d % 2) * 4)) & 15) as i32 };
    let norms: Vec<i32> = (0..n).map(|k| { let c = &codes[k * cb..(k + 1) * cb]; (0..dim).map(|d| { let v = nib(c, d); v * v }).sum() }).collect();
    let cmax = cent.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let cs8 = 127.0 / cmax;
    let mut cent8 = vec![0i8; nl * 128]; let mut csum = vec![0i32; nl]; let mut cn2 = vec![0f32; nl];
    for j in 0..nl { for d in 0..dim { let v = (cent[j * dim + d] * cs8).round().clamp(-127.0, 127.0) as i8; cent8[j * 128 + d] = v; csum[j] += v as i32; }
        cn2[j] = cent[j * dim..(j + 1) * dim].iter().map(|x| x * x).sum(); }
    assert!(dim <= 128 && cb <= 64, "SV3 v2 kernels: dim<=128, code_bytes<=64");
    let mut sc_i = vec![0i32; 4096.max(nl) * 2]; let mut q8 = [0u8; 128]; let mut qlo = [0i8; 64]; let mut qhi = [0i8; 64];
    let recall = |res: &Vec<Vec<u32>>| -> f64 { let mut tot = 0usize;
        for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
        tot as f64 / (nq * kk) as f64 };
    let pv: &[f32] = &p.vectors;
    let mut qc = vec![0u8; cb]; let mut qbuf = vec![0f32; dim];
    let mut cs: Vec<(f32, u32)> = vec![(0.0, 0); nl];
    let mut out = String::new();
    for round in 0..repeats.max(1) {
        for &ef in efs {
            let mut s = p.searcher(ef); let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
            let t = Instant::now();
            for qi in 0..nq { res[qi] = s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk).iter().map(|x| p.orig_id(x.0)).collect(); }
            let qps = nq as f64 / t.elapsed().as_secs_f64(); let rec = recall(&res);
            eprintln!("SV3 r{round} base ef={ef} recall={rec:.5} qps={qps:.0}"); out += &format!("{round},base,{ef},0,0,{rec:.5},{qps:.1}\n");
        }
        for &e in &es { for &pp in &ps { for &tt in &ts {
            let mut s = p.searcher(e); let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
            let mut scanned = 0usize;
            let t = Instant::now();
            for qi in 0..nq {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let g = s.search_sq4(q, kk);
                for (j, &d) in perm.iter().enumerate() { qbuf[j] = q[d]; }
                p.sq4.params.encode_into(&qbuf, &mut qc);
                for i in 0..64 { let b = if i < cb { qc[i] } else { 0 }; qlo[i] = (b & 15) as i8; qhi[i] = (b >> 4) as i8; }
                if kv == 2 {
                    let qm = q.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let qs8 = 63.0 / qm;
                    for d in 0..128 { q8[d] = if d < dim { ((q[d] * qs8).round().clamp(-63.0, 63.0) as i32 + 64) as u8 } else { 64 }; }
                    #[cfg(target_arch = "x86_64")] unsafe { sv3_cdot8_avx2(&cent8, nl, &q8, &mut sc_i); }
                    let inv = 1.0 / (cs8 * qs8);
                    for j in 0..nl { cs[j] = (cn2[j] - 2.0 * ((sc_i[j] - 64 * csum[j]) as f32) * inv, j as u32); }
                } else {
                    for j in 0..nl { cs[j] = (raven::distance::l2_simd(q, &cent[j * dim..(j + 1) * dim]), j as u32); }
                }
                let pp2 = pp.min(nl); cs.select_nth_unstable_by(pp2 - 1, |a, b| a.0.total_cmp(&b.0));
                let mut top: Vec<(f32, u32)> = Vec::with_capacity(tt + 1); let mut thr = f32::MAX;
                for ci in 0..pp2 { let j = cs[ci].1 as usize;
                    if kv == 2 {
                        let (o0, o1) = (offs[j], offs[j + 1]);
                        let ti = if thr == f32::MAX { i32::MAX } else { thr as i32 };
                        let mut ns = 0usize;
                        #[cfg(target_arch = "x86_64")] unsafe { ns = sv3_scan_thr_avx2(&codes[o0 * cb..o1 * cb], cb, o1 - o0, &norms[o0..o1], ti, &qlo, &qhi, &mut sc_i); }
                        for t2 in 0..ns { let k = o0 + (sc_i[2 * t2 + 1] as usize); let d = sc_i[2 * t2] as f32;
                            if d < thr { let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, ids[k])); if top.len() > tt { top.pop(); } if top.len() == tt { thr = top[tt - 1].0; } } }
                    } else {
                    for k in offs[j]..offs[j + 1] { let d = raven::quant::sq4::l2_sq4_raw(&qc, &codes[k * cb..(k + 1) * cb]);
                        if d < thr { let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, ids[k])); if top.len() > tt { top.pop(); } if top.len() == tt { thr = top[tt - 1].0; } } }
                    }
                    scanned += offs[j + 1] - offs[j]; }
                let mut m: Vec<(f32, u32)> = g.iter().map(|x| (raven::distance::l2_simd(q, &pv[x.0 as usize * dim..(x.0 as usize + 1) * dim]), x.0)).collect();
                for &(_, v) in &top { if !m.iter().any(|x| x.1 == v) { m.push((raven::distance::l2_simd(q, &pv[v as usize * dim..(v as usize + 1) * dim]), v)); } }
                m.sort_by(|a, b| a.0.total_cmp(&b.0));
                res[qi] = m.iter().take(kk).map(|x| p.orig_id(x.1)).collect();
            }
            let qps = nq as f64 / t.elapsed().as_secs_f64(); let rec = recall(&res);
            eprintln!("SV3 r{round} sv3 E={e} P={pp} T={tt} recall={rec:.5} qps={qps:.0} scanned/q={:.0}", scanned as f64 / nq as f64);
            out += &format!("{round},sv3,{e},{pp},{tt},{rec:.5},{qps:.1}\n");
        } } }
    }
    let _ = std::fs::write(std::env::var("RAVEN_SV3_OUT").unwrap_or("/tmp/sv3.csv".into()), format!("round,mode,ef,P,T,recall10,qps\n{out}"));
}


// ───────────────────────── R33-2 逐查询 ef 需求 oracle（RAVEN_PQEF=前缀） ─────────────────────────
// 对 --ef 网格逐个 ef 跑全部查询（searcher_plain，原路径），记录每查询命中数（带并列容差）与每 ef 平均耗时（repeats 取中位）。
// 输出 {pfx}.hits（nq×nef u8，行主序）与 {pfx}.cost.csv（ef,us_per_query,recall）。oracle 分配由离线脚本完成。
#[allow(clippy::too_many_arguments)]
fn pqef_run(p: &raven::pipeline::Prepared, base: &[f32], queries: &[f32], dim: usize, nq: usize, gt_dist: &[f32], gtk: usize,
            angular: bool, repeats: usize, efs: &[usize], kk: usize, pfx: &str) {
    let ne = efs.len();
    let mut hits = vec![0u8; nq * ne];
    let mut cost = String::from("ef,us_per_query,recall\n");
    // R34：RAVEN_PQEF_HR=1 → 每个 ef 用强制 HR 模式（plan_force：E=ER·ef 图段 + P=PR·ef 单元确认），供两级 HR oracle
    let pq_hr = std::env::var("RAVEN_PQEF_HR").map(|v| v == "1").unwrap_or(false);
    // RAVEN_PQEF_HR_E=E：固定图段 E，--ef 网格解释为确认单元数 P（逐查询 P 的 oracle：单元扫描可精确续扫）
    let pq_fixe: Option<usize> = std::env::var("RAVEN_PQEF_HR_E").ok().and_then(|v| v.parse().ok());
    if pq_hr { assert!(p.hr.is_some(), "RAVEN_PQEF_HR needs HR built (RAVEN_HR=1 RAVEN_HR_GATE=...)"); }
    // R35：RAVEN_PQEF_LIB=1 → 用库默认 searcher（ef ≥ gate 时 HR v2 含耐心早停），作逐查询 P oracle 的 v2 基线曲线
    let pq_lib = std::env::var("RAVEN_PQEF_LIB").map(|v| v == "1").unwrap_or(false);
    // R36：RAVEN_PQEF_BASE=ef1,ef2,… → 同进程基线曲线：每个重复轮内，网格点与库默认 searcher（p.searcher(ef)）的基线点交替计时，
    // 基线写 {pfx}.base.csv（R35 教训：oracle 的成本曲线与基线必须同进程测量）。RAVEN_PQEF_BASE_A=ER:PR[:pat:pminr] 改用强制 HR 映射作基线。
    let base_efs: Vec<usize> = std::env::var("RAVEN_PQEF_BASE").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let base_spec = std::env::var("RAVEN_PQEF_BASE_A").ok();
    let mk = |ef: usize| {
        if pq_lib { p.searcher_k(ef, kk) } else if pq_hr { let cc = p.hr.as_ref().unwrap(); let (e, pc) = match pq_fixe { Some(fe) => (fe, ef.clamp(1, cc.nl)), None => cc.plan_force(ef) }; let mut s = p.searcher_plain(e); s.with_cell_confirm(cc, pc); s } else { p.searcher_plain(ef) }
    };
    let mk_base = |ef: usize| {
        match (&base_spec, p.hr.as_ref()) {
            (Some(spec), _) if spec == "plain" => p.searcher_plain(ef),
            (Some(spec), Some(cc)) => { let v: Vec<&str> = spec.split(':').collect(); let er: f32 = v[0].parse().unwrap(); let pr: f32 = v[1].parse().unwrap();
                let e = ((er * ef as f32).round() as usize).max(10); let pc = ((pr * ef as f32).round() as usize).clamp(1, cc.nl);
                let mut s = p.searcher_plain(e); s.with_cell_confirm(cc, pc);
                s.set_cc_pat(v.get(2).map(|x| x.parse().unwrap()).unwrap_or(0), v.get(3).map(|x| x.parse().unwrap()).unwrap_or(0.25));
                s.set_cc_stage(v.get(4).map(|x| x.parse().unwrap()).unwrap_or(0), v.get(5).map(|x| x.parse().unwrap()).unwrap_or(0), v.get(6).map(|x| x.parse().unwrap()).unwrap_or(0));
                if let Some(x) = v.get(7) { s.set_cc_sc(x.parse().unwrap()); } s }
            _ => p.searcher(ef),
        }
    };
    let nb = base_efs.len();
    let mut ts = vec![Vec::new(); ne]; let mut tsb = vec![Vec::new(); nb];
    let mut res: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); nq]; ne]; let mut resb: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); nq]; nb];
    // R36 RH oracle：RAVEN_PQEF_RHO=mg → 网格 searcher 每查询初始门槛 = (真第 k 名半径·mg)²（角度：2·d_k；L2：d_k²）
    let rho: Option<f32> = std::env::var("RAVEN_PQEF_RHO").ok().and_then(|v| v.parse().ok());
    let thr_true: Vec<f32> = (0..nq).map(|qi| { let d = gt_dist[qi * gtk + kk - 1]; if angular { 2.0 * d } else { d * d } }).collect();
    let run = |s: &mut raven::graph::vamana::GraphSearcher<'_>, out: &mut Vec<Vec<u32>>, orc: bool| -> f64 {
        let t = Instant::now();
        for qi in 0..nq {
            if orc { if let Some(mg) = rho { s.set_rh_oracle(Some(thr_true[qi] * mg * mg)); } }
            out[qi] = s.search_auto(&queries[qi * dim..(qi + 1) * dim], kk).iter().map(|x| p.orig_id(x.0)).collect(); }
        t.elapsed().as_secs_f64()
    };
    for r in 0..repeats.max(1) {
        // 交替：网格点 i 之后跑一个基线点（循环），偶数轮正序、奇数轮反序
        let order: Vec<usize> = if r % 2 == 0 { (0..ne.max(nb)).collect() } else { (0..ne.max(nb)).rev().collect() };
        for &i in &order {
            if i < ne { let mut s = mk(efs[i]); ts[i].push(run(&mut s, &mut res[i], true)); }
            if i < nb { let mut s = mk_base(base_efs[i]); s.set_cc_rh(0, 1.0); tsb[i].push(run(&mut s, &mut resb[i], false)); }
        }
    }
    let rec_of = |res: &Vec<Vec<u32>>, hcol: Option<(&mut Vec<u8>, usize)>| -> f64 {
        let mut tot = 0usize; let mut hc = hcol;
        for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            let h = res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count();
            if let Some((hv, ei)) = hc.as_mut() { hv[qi * ne + *ei] = h as u8; }
            tot += h; }
        tot as f64 / (nq * kk) as f64
    };
    for ei in 0..ne {
        let mut t = ts[ei].clone(); t.sort_by(|a, b| a.total_cmp(b));
        let us = t[t.len() / 2] * 1e6 / nq as f64; let rec = rec_of(&res[ei], Some((&mut hits, ei)));
        eprintln!("PQEF ef={} recall={rec:.5} us/q={us:.2}", efs[ei]);
        cost += &format!("{},{us:.3},{rec:.5}\n", efs[ei]);
    }
    if nb > 0 {
        let mut bc = String::from("ef,us_per_query,recall\n");
        for bi in 0..nb {
            let mut t = tsb[bi].clone(); t.sort_by(|a, b| a.total_cmp(b));
            let us = t[t.len() / 2] * 1e6 / nq as f64; let rec = rec_of(&resb[bi], None);
            eprintln!("PQEF base ef={} recall={rec:.5} us/q={us:.2}", base_efs[bi]);
            bc += &format!("{},{us:.3},{rec:.5}\n", base_efs[bi]);
        }
        std::fs::write(format!("{pfx}.base.csv"), bc).unwrap();
    }
    std::fs::write(format!("{pfx}.hits"), &hits).unwrap();
    std::fs::write(format!("{pfx}.cost.csv"), cost).unwrap();
}

// ───────────────────────── R35 分段确认检测器诊断（RAVEN_PQDET=前缀） ─────────────────────────
// RAVEN_PQDET_E=E（图段）、RAVEN_PQDET_SA=a（第一段单元数）；--ef 网格解释为总单元数 P（每个 P 一个 csv）。
// 每查询：第一段后 hits_a、特征（u 改进数、s 幸存者数、cov、gap），总续扫到 P 后 hits_b。csv: q,hits_a,hits_b,u,s,cov,gap
#[allow(clippy::too_many_arguments)]
fn pqdet_run(p: &raven::pipeline::Prepared, base: &[f32], queries: &[f32], dim: usize, nq: usize, gt_dist: &[f32], gtk: usize,
             angular: bool, efs: &[usize], kk: usize, pfx: &str) {
    let cc = p.hr.as_ref().expect("RAVEN_PQDET needs HR built");
    let e: usize = std::env::var("RAVEN_PQDET_E").ok().and_then(|v| v.parse().ok()).unwrap_or(81);
    let sa: usize = std::env::var("RAVEN_PQDET_SA").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let hits = |qi: usize, ids: &[u32]| -> usize { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
        ids.iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
            (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count() };
    for &pc in efs {
        let mut s = p.searcher_plain(e); s.with_cell_confirm(cc, pc.clamp(1, cc.nl));
        let mut out = String::from("q,hits_a,hits_b,u,s,cov,gap,rmax,r2\n");
        let (mut ta, mut tb) = (0usize, 0usize);
        for qi in 0..nq {
            let mut f = raven::graph::cellconfirm::StageFeat::default();
            let r = s.search_hr_diag(&queries[qi * dim..(qi + 1) * dim], kk, sa, &mut f);
            let ib: Vec<u32> = r.iter().map(|x| p.orig_id(x.0)).collect();
            let ia: Vec<u32> = f.top_a.iter().map(|&x| p.orig_id(x)).collect();
            let (ha, hb) = (hits(qi, &ia), hits(qi, &ib)); ta += ha; tb += hb;
            out += &format!("{qi},{ha},{hb},{},{},{},{:.4},{},{}\n", f.u, f.s, f.cov, f.gap, f.rmax, f.r2);
        }
        eprintln!("PQDET E={e} sa={sa} P={pc} recall_a={:.5} recall_b={:.5}", ta as f64 / (nq * kk) as f64, tb as f64 / (nq * kk) as f64);
        std::fs::write(format!("{pfx}_P{pc}.csv"), out).unwrap();
    }
}

// ───────────────────────── R34 配对细粒度 A/B（RAVEN_HR_AB_PAIR=块大小） ─────────────────────────
// 同一 ef 下 plain（searcher_plain）与 lib（p.searcher：ef≥gate 时 HR）按查询块交替执行（块内顺序逐块翻转），
// 分别累计耗时 → steal/抢占突发同时落在两侧，比整轮交替稳健。每轮记录 taskset 所在 CPU（RAVEN_AB_CPU，默认 1）的
// /proc/stat steal 占比（st 列，%），用于按 CHARTER 只计 steal <5% 的轮。csv: round,mode,ef,recall10,qps,hr_active,st
fn cpu_ticks(cpu: usize) -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let tag = format!("cpu{cpu} ");
    for l in s.lines() { if l.starts_with(&tag) {
        let v: Vec<u64> = l.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
        let tot: u64 = v.iter().take(8).sum(); let st = v.get(7).copied().unwrap_or(0); return (tot, st); } }
    (0, 0)
}
#[allow(clippy::too_many_arguments)]
fn hr_ab_pair_run(p: &raven::pipeline::Prepared, base: &[f32], queries: &[f32], dim: usize, nq: usize, gt_dist: &[f32], gtk: usize,
                  angular: bool, repeats: usize, efs: &[usize], kk: usize, out: &str, blk: usize) {
    let cpu: usize = std::env::var("RAVEN_AB_CPU").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let recall = |res: &Vec<Vec<u32>>| -> f64 { let mut tot = 0usize;
        for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
        tot as f64 / (nq * kk) as f64 };
    let mut csv = String::from("round,mode,ef,recall10,qps,hr_active,st\n");
    for round in 0..repeats.max(1) {
        let (t0, s0) = cpu_ticks(cpu);
        let mut rows: Vec<(usize, f64, f64, f64, f64, bool)> = Vec::new();
        let efo: Vec<usize> = if round % 2 == 0 { efs.to_vec() } else { efs.iter().rev().copied().collect() };
        for &ef in &efo {
            // RAVEN_HR_PAIR_A="ER:PR[:PAT:PMINR]" → A 侧改为强制 HR（不看门槛），用于 HR 新旧映射配对比较
            let mut sa = match std::env::var("RAVEN_HR_PAIR_A") { Ok(spec) => {
                let cc = p.hr.as_ref().expect("RAVEN_HR_PAIR_A needs HR built");
                let v: Vec<&str> = spec.split(':').collect(); let er: f32 = v[0].parse().unwrap(); let pr: f32 = v[1].parse().unwrap();
                let e = ((er * ef as f32).round() as usize).max(10); let pc = ((pr * ef as f32).round() as usize).clamp(1, cc.nl);
                let mut s = p.searcher_plain(e); s.with_cell_confirm(cc, pc);
                s.set_cc_pat(v.get(2).map(|x| x.parse().unwrap()).unwrap_or(0), v.get(3).map(|x| x.parse().unwrap()).unwrap_or(0.25));
                // R35：A 侧默认不用分段确认（RAVEN_HR_SA 只作用于 lib 侧）；第 5–7 段 SA:SU:SS 可显式给 A 侧
                s.set_cc_stage(v.get(4).map(|x| x.parse().unwrap()).unwrap_or(0), v.get(5).map(|x| x.parse().unwrap()).unwrap_or(0), v.get(6).map(|x| x.parse().unwrap()).unwrap_or(0));
                if let Some(x) = v.get(7) { s.set_cc_sc(x.parse().unwrap()); } s }
                Err(_) => p.searcher_plain(ef) };
            let mut sb = p.searcher(ef); let act = sb.hr_active();
            let mut ra: Vec<Vec<u32>> = vec![Vec::new(); nq]; let mut rb: Vec<Vec<u32>> = vec![Vec::new(); nq];
            let (mut ta, mut tb) = (0f64, 0f64);
            let mut b0 = 0usize; let mut bi = 0usize;
            while b0 < nq {
                let b1 = (b0 + blk).min(nq);
                for side in 0..2 {
                    let first_a = (bi + round) % 2 == 0;
                    let do_a = (side == 0) == first_a;
                    let t = Instant::now();
                    if do_a { for qi in b0..b1 { ra[qi] = sa.search_auto(&queries[qi * dim..(qi + 1) * dim], kk).iter().map(|x| p.orig_id(x.0)).collect(); } ta += t.elapsed().as_secs_f64(); }
                    else { for qi in b0..b1 { rb[qi] = sb.search_auto(&queries[qi * dim..(qi + 1) * dim], kk).iter().map(|x| p.orig_id(x.0)).collect(); } tb += t.elapsed().as_secs_f64(); }
                }
                b0 = b1; bi += 1;
            }
            rows.push((ef, recall(&ra), nq as f64 / ta, recall(&rb), nq as f64 / tb, act));
        }
        let (t1, s1) = cpu_ticks(cpu);
        let st = if t1 > t0 { 100.0 * (s1 - s0) as f64 / (t1 - t0) as f64 } else { -1.0 };
        for (ef, ra, qa, rb, qb, act) in rows {
            eprintln!("HRPAIR r{round} ef={ef} plain {ra:.5} {qa:.0} | lib hr={act} {rb:.5} {qb:.0} | st {st:.1}%");
            csv += &format!("{round},plain,{ef},{ra:.5},{qa:.1},0,{st:.2}\n{round},lib,{ef},{rb:.5},{qb:.1},{},{st:.2}\n", act as u8);
        }
    }
    std::fs::write(out, csv).unwrap();
}

// ───────────────────────── R33 HR 模式进程内 A/B（RAVEN_HR_AB=1） ─────────────────────────
// 每轮：对每个 ef 先跑 plain（searcher_plain），再跑 lib（p.searcher，ef≥gate 时 HR 模式）；轮间正反序交替。
// 同一张图、同一进程（CHARTER R29 规则允许的同图旋钮 A/B）。输出 csv: round,mode,ef,recall10,qps,hr_active
#[allow(clippy::too_many_arguments)]
fn hr_ab_run(p: &raven::pipeline::Prepared, base: &[f32], queries: &[f32], dim: usize, nq: usize, gt_dist: &[f32], gtk: usize,
             angular: bool, repeats: usize, efs: &[usize], kk: usize, out: &str) {
    let recall = |res: &Vec<Vec<u32>>| -> f64 { let mut tot = 0usize;
        for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
        tot as f64 / (nq * kk) as f64 };
    let modes = std::env::var("RAVEN_HR_AB_MODES").unwrap_or("plain,lib".into());
    let modes: Vec<&str> = modes.split(',').collect();
    let mut csv = String::from("round,mode,ef,recall10,qps,hr_active\n");
    for round in 0..repeats.max(1) {
        let order: Vec<&str> = if round % 2 == 0 { modes.clone() } else { modes.iter().rev().copied().collect() };
        for &mode in &order { for &ef in efs {
            // mode "hr:ER:PR" = HR 强制（不看门槛），E = round(ER·ef)、P = round(PR·ef)，用于映射调参
            let mut s = if mode == "plain" { p.searcher_plain(ef) } else if let Some(rest) = mode.strip_prefix("hr:") {
                let cc = p.hr.as_ref().expect("hr: mode needs RAVEN_HR=1 and RAVEN_HR_GATE set");
                let mut it = rest.split(':'); let er: f32 = it.next().unwrap().parse().unwrap(); let pr: f32 = it.next().unwrap().parse().unwrap();
                let e = ((er * ef as f32).round() as usize).max(10); let pc = ((pr * ef as f32).round() as usize).clamp(1, cc.nl);
                let mut s = p.searcher_plain(e); s.with_cell_confirm(cc, pc);
                // R34：hr:ER:PR:PAT:PMINR → 单元扫描耐心早停
                if let Some(pat) = it.next() { let pmr: f32 = it.next().unwrap_or("0.25").parse().unwrap(); s.set_cc_pat(pat.parse().unwrap(), pmr); }
                s
            } else { p.searcher(ef) };
            let act = s.hr_active();
            let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
            if let Some(rest) = mode.strip_prefix("cert:") {
                // R33-2 两级自适应：图 ef=EA → 局部修复（结果 top-k 所在单元的 1-bit 扫描）→ 若找回更近点（不完整证据）则升级到 HR(ef)
                let cc = p.hr.as_ref().expect("cert: needs RAVEN_HR=1");
                let mut it = rest.split(':'); let ea: usize = it.next().unwrap().parse().unwrap(); let mm: f32 = it.next().unwrap_or("1").parse().unwrap();
                let delta: f32 = it.next().unwrap_or("0").parse().unwrap(); let snear: usize = it.next().unwrap_or("1000000").parse().unwrap();
                let (er, pr) = (0.37f32, 0.18f32);
                let mut sa = p.searcher_plain(ea);
                let (e, pc) = (((er * ef as f32).round() as usize).max(10), ((pr * ef as f32).round() as usize).clamp(1, cc.nl));
                let mut sb = p.searcher_plain(e); sb.with_cell_confirm(cc, pc);
                let mut sc = cc.scratch(); let mut cells: Vec<u32> = Vec::with_capacity(16);
                let mut nesc = 0usize;
                let pv: &[f32] = &p.vectors;
                let t = Instant::now();
                for qi in 0..nq {
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    let g = sa.search_sq4(q, kk);
                    sc.top.clear();
                    sc.top.extend(g.iter().map(|x| (raven::distance::l2_simd(q, &pv[x.0 as usize * dim..(x.0 as usize + 1) * dim]), x.0)));
                    sc.top.sort_by(|a, b| a.0.total_cmp(&b.0)); sc.top.truncate(kk);
                    cells.clear(); for x in sc.top.iter() { let c = cc.cell_of[x.1 as usize]; if !cells.contains(&c) { cells.push(c); } }
                    let (nimp, nnear) = cc.repair_cells_near(q, pv, kk, &cells, mm, delta, &mut sc);
                    if nimp > 0 || nnear >= snear {
                        nesc += 1;
                        let r2 = sb.search_auto(q, kk);
                        for x in r2 { if !sc.top.iter().any(|y| y.1 == x.0) { let pos = sc.top.partition_point(|y| y.0 <= x.1); sc.top.insert(pos, (x.1, x.0)); } }
                        sc.top.truncate(kk);
                    }
                    res[qi] = sc.top.iter().map(|x| p.orig_id(x.1)).collect();
                }
                let qps = nq as f64 / t.elapsed().as_secs_f64(); let rec = recall(&res);
                eprintln!("HRAB r{round} {mode} ef={ef} hr=true recall={rec:.5} qps={qps:.0} escalated={:.3}", nesc as f64 / nq as f64);
                csv += &format!("{round},{mode},{ef},{rec:.5},{qps:.1},1\n");
                continue;
            }
            let t = Instant::now();
            for qi in 0..nq { res[qi] = s.search_auto(&queries[qi * dim..(qi + 1) * dim], kk).iter().map(|x| p.orig_id(x.0)).collect(); }
            let qps = nq as f64 / t.elapsed().as_secs_f64(); let rec = recall(&res);
            eprintln!("HRAB r{round} {mode} ef={ef} hr={act} recall={rec:.5} qps={qps:.0}");
            csv += &format!("{round},{mode},{ef},{rec:.5},{qps:.1},{}\n", act as u8);
        } }
    }
    if !out.is_empty() { std::fs::write(out, csv).unwrap(); }
}

// ───────────────────────── R32 SV-3b 级联（诊断开关 RAVEN_SV3B） ─────────────────────────
// 图定位（ef=E）→ 精确 top-k 与第 k 名距离 T → 前 P 个 k-means 单元 → 1-bit 残差码 fastscan，
// 免排序门槛 est < (T − ‖q−c‖²) + m·2‖q−c‖·r/√D（T 随幸存者精算动态收紧）→ 幸存者精确距离并入 top-k。
// RAVEN_SV3B="NL;E1/E2;P1/P2;M1/M2"，与 --ef 基线同进程逐轮交替；输出 RAVEN_SV3B_OUT。
#[allow(clippy::too_many_arguments)]
fn sv3b_run(p: &raven::pipeline::Prepared, base: &[f32], queries: &[f32], dim: usize, nq: usize, gt_dist: &[f32], gtk: usize,
            angular: bool, spec: &str, repeats: usize, efs: &[usize], kk: usize, cache_dir: &str, hash: &str) {
    use raven::quant::bitcell::{BitCells, rotate};
    let f: Vec<&str> = spec.split(';').collect();
    let nl: usize = f[0].parse().unwrap();
    let es: Vec<usize> = f[1].split('/').map(|x| x.parse().unwrap()).collect();
    let ps: Vec<usize> = f[2].split('/').map(|x| x.parse().unwrap()).collect();
    let ms: Vec<f32> = f.get(3).unwrap_or(&"1.0").split('/').map(|x| x.parse().unwrap()).collect();
    // 第 5 字段 C：候选单元数（0 = 全部单元打分；>0 = 只对"图 top-1 所在单元"的 C 个最近单元打分——图定位 → 单元邻域，免 4096 中心扫描）
    let ccs: Vec<usize> = f.get(4).unwrap_or(&"0").split('/').map(|x| x.parse().unwrap()).collect();
    let prof = std::env::var("RAVEN_SV3B_PROF").is_ok();
    // RAVEN_SV3B_PFC：整单元 L2 预取的超前单元数（0 = 关）
    let pfc: usize = std::env::var("RAVEN_SV3B_PFC").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let n = base.len() / dim;
    let (cent, asg) = sv3_kmeans(base, dim, n, nl, angular, cache_dir, hash);
    let mut asg_i = vec![0u32; n]; for i in 0..n as u32 { asg_i[i as usize] = asg[p.orig_id(i) as usize]; }
    let pv: &[f32] = &p.vectors;
    let t0 = Instant::now();
    let bc = BitCells::build(pv, dim, &cent, &asg_i, 0x5eed_32b);
    eprintln!("SV3B build NL={nl} dpad={} slots={} ({:.1}% pad) in {:.1}s", bc.dpad, bc.ids.len(), (bc.ids.len() as f64 / n as f64 - 1.0) * 100.0, t0.elapsed().as_secs_f64());
    // 单元打分：x86 用 int8 中心核（同 R31），其它平台 f32
    let cmax = cent.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let cs8 = 127.0 / cmax;
    let mut cent8 = vec![0i8; nl * 128]; let mut csum = vec![0i32; nl];
    for j in 0..nl { for d in 0..dim.min(128) { let v = (cent[j * dim + d] * cs8).round().clamp(-127.0, 127.0) as i8; cent8[j * 128 + d] = v; csum[j] += v as i32; } }
    let use_i8 = cfg!(target_arch = "x86_64") && dim <= 128 && std::env::var("RAVEN_SV3B_CF32").is_err();
    let ccmax = ccs.iter().copied().max().unwrap_or(0).min(nl);
    let cknn = if ccmax > 0 { let t = Instant::now(); let v = raven::quant::bitcell::cell_knn(&cent, dim, ccmax); eprintln!("SV3B cell_knn C={ccmax} in {:.1}s", t.elapsed().as_secs_f64()); v } else { Vec::new() };
    let recall = |res: &Vec<Vec<u32>>| -> f64 { let mut tot = 0usize;
        for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
        tot as f64 / (nq * kk) as f64 };
    let row = |id: u32| -> &[f32] { &pv[id as usize * dim..(id as usize + 1) * dim] };
    let mut sc_i = vec![0i32; nl + 8]; let mut q8 = [0u8; 128];
    let mut cs: Vec<(f32, u32)> = vec![(0.0, 0); nl];
    let mut qr = vec![0f32; bc.dpad];
    let mut sur: Vec<u32> = Vec::with_capacity(4096);
    let sqd = (bc.dpad as f32).sqrt();
    if std::env::var("RAVEN_SV3B_HOT").is_ok() {
        // 热态核微基准：同一查询反复扫描 ncell 个单元（工作集 ≈ ncell×7 KB，L2 常驻）→ 计算上限 ns/点
        for &ncell in &[8usize, 32, 64] {
            let q = &queries[..dim]; rotate(&bc.rot, bc.dpad, q, &mut qr); let ql = bc.prepare_query(&qr);
            let mut npts = 0usize; for j in 0..ncell { npts += bc.offs[j + 1] - bc.offs[j]; }
            let reps = 2000; let mut ns_tot = 0usize;
            let t = Instant::now();
            for _ in 0..reps { for j in 0..ncell { sur.clear(); bc.scan_cell(j, &ql, -1e9, 0.0, &mut sur); ns_tot += sur.len(); } }
            let el = t.elapsed().as_secs_f64();
            eprintln!("SV3B HOT ncell={ncell} pts={npts} {:.3} ns/pt (survivors/rep {:.1})", el * 1e9 / (reps * npts) as f64, ns_tot as f64 / reps as f64);
        }
    }
    let mut out = String::new();
    for round in 0..repeats.max(1) {
        for &ef in efs {
            let mut s = p.searcher(ef); let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
            let t = Instant::now();
            for qi in 0..nq { res[qi] = s.search_sq4(&queries[qi * dim..(qi + 1) * dim], kk).iter().map(|x| p.orig_id(x.0)).collect(); }
            let qps = nq as f64 / t.elapsed().as_secs_f64(); let rec = recall(&res);
            eprintln!("SV3B r{round} base ef={ef} recall={rec:.5} qps={qps:.0}"); out += &format!("{round},base,{ef},0,0,0,{rec:.5},{qps:.1},0,0\n");
        }
        for &e in &es { for &pp in &ps { for &mm in &ms { for &cc in &ccs {
            let mut s = p.searcher(e); let mut res: Vec<Vec<u32>> = vec![Vec::new(); nq];
            let (mut nsur, mut nscan, mut nimp) = (0usize, 0usize, 0usize);
            let mut tp = [0f64; 5]; // 图 / 旋转+表 / 单元选择 / 扫描 / 精算
            let t = Instant::now();
            for qi in 0..nq {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let t0 = if prof { Some(Instant::now()) } else { None };
                let g = s.search_sq4(q, kk);
                let mut top: Vec<(f32, u32)> = g.iter().map(|x| (raven::distance::l2_simd(q, row(x.0)), x.0)).collect();
                top.sort_by(|a, b| a.0.total_cmp(&b.0)); top.truncate(kk);
                let mut thr = if top.len() == kk { top[kk - 1].0 } else { f32::MAX };
                if let Some(t0) = t0 { tp[0] += t0.elapsed().as_secs_f64(); }
                let t1 = if prof { Some(Instant::now()) } else { None };
                rotate(&bc.rot, bc.dpad, q, &mut qr);
                let ql = bc.prepare_query(&qr);
                if let Some(t1) = t1 { tp[1] += t1.elapsed().as_secs_f64(); }
                let t2 = if prof { Some(Instant::now()) } else { None };
                let ncs;
                if cc > 0 {
                    let c0 = asg_i[top[0].1 as usize] as usize; let cc2 = cc.min(nl);
                    for t in 0..cc2 { let j = cknn[c0 * ccmax + t] as usize; cs[t] = (raven::distance::l2_simd(q, &cent[j * dim..(j + 1) * dim]), j as u32); }
                    ncs = cc2;
                } else if use_i8 {
                    let qm = q.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let qs8 = 63.0 / qm;
                    for d in 0..128 { q8[d] = if d < dim { ((q[d] * qs8).round().clamp(-63.0, 63.0) as i32 + 64) as u8 } else { 64 }; }
                    #[cfg(target_arch = "x86_64")] unsafe { sv3_cdot8_avx2(&cent8, nl, &q8, &mut sc_i); }
                    let inv = 1.0 / (cs8 * qs8);
                    for j in 0..nl { cs[j] = (bc.cn2[j] - 2.0 * ((sc_i[j] - 64 * csum[j]) as f32) * inv, j as u32); }
                    ncs = nl;
                } else {
                    for j in 0..nl { cs[j] = (raven::distance::l2_simd(q, &cent[j * dim..(j + 1) * dim]), j as u32); }
                    ncs = nl;
                }
                let pp2 = pp.min(ncs); cs[..ncs].select_nth_unstable_by(pp2 - 1, |a, b| a.0.total_cmp(&b.0));
                cs[..pp2].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                if let Some(t2) = t2 { tp[2] += t2.elapsed().as_secs_f64(); }
                // 扫描全部选中单元（门槛 = 图给出的 T，不随扫描收紧）→ 幸存者统一批量精算（高访存并行）
                let t3 = if prof { Some(Instant::now()) } else { None };
                sur.clear();
                if pp2 > 0 { bc.prefetch_cell(cs[0].1 as usize, 2); }
                for ci in 1..=pfc.min(pp2.saturating_sub(1)) { bc.prefetch_cell_l2(cs[ci].1 as usize); }
                for ci in 0..pp2 {
                    let j = cs[ci].1 as usize;
                    if ci + 1 < pp2 { bc.prefetch_cell(cs[ci + 1].1 as usize, 2); }
                    if pfc > 0 && ci + 1 + pfc < pp2 { bc.prefetch_cell_l2(cs[ci + 1 + pfc].1 as usize); }
                    let qc2 = if cc > 0 || !use_i8 { cs[ci].0 } else { raven::distance::l2_simd(q, &cent[j * dim..(j + 1) * dim]) };
                    let mc = mm * 2.0 * qc2.sqrt() / sqd;
                    bc.scan_cell(j, &ql, thr - qc2, mc, &mut sur);
                    nscan += bc.offs[j + 1] - bc.offs[j];
                }
                nsur += sur.len();
                if let Some(t3) = t3 { tp[3] += t3.elapsed().as_secs_f64(); }
                let t4 = if prof { Some(Instant::now()) } else { None };
                const PFD: usize = 8;
                for &so in sur.iter().take(PFD) { prefetch_row(row(bc.ids[so as usize])); }
                for i in 0..sur.len() {
                    if i + PFD < sur.len() { prefetch_row(row(bc.ids[sur[i + PFD] as usize])); }
                    let id = bc.ids[sur[i] as usize];
                    let d = raven::distance::l2_simd(q, row(id));
                    if d < thr && !top.iter().any(|x| x.1 == id) {
                        let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, id)); top.truncate(kk); nimp += 1;
                        if top.len() == kk { thr = top[kk - 1].0; }
                    }
                }
                if let Some(t4) = t4 { tp[4] += t4.elapsed().as_secs_f64(); }
                res[qi] = top.iter().map(|x| p.orig_id(x.1)).collect();
            }
            let qps = nq as f64 / t.elapsed().as_secs_f64(); let rec = recall(&res);
            let us = |x: f64| x * 1e6 / nq as f64;
            eprintln!("SV3B r{round} sv3b E={e} P={pp} M={mm} C={cc} recall={rec:.5} qps={qps:.0} scanned/q={:.0} survivors/q={:.1} improved/q={:.2}{}",
                      nscan as f64 / nq as f64, nsur as f64 / nq as f64, nimp as f64 / nq as f64,
                      if prof { format!(" prof_us graph={:.1} rot={:.1} cells={:.1} scan={:.1} exact={:.1}", us(tp[0]), us(tp[1]), us(tp[2]), us(tp[3]), us(tp[4])) } else { String::new() });
            out += &format!("{round},sv3b,{e},{pp},{mm},{cc},{rec:.5},{qps:.1},{:.1},{:.1}\n", nscan as f64 / nq as f64, nsur as f64 / nq as f64);
        } } } }
    }
    let _ = std::fs::write(std::env::var("RAVEN_SV3B_OUT").unwrap_or("/tmp/sv3b.csv".into()), format!("round,mode,ef,P,M,C,recall10,qps,scanned,survivors\n{out}"));
}

#[inline]
fn prefetch_row(x: &[f32]) {
    let p = x.as_ptr() as *const i8; let nb = x.len() * 4;
    let mut o = 0;
    while o < nb {
        #[cfg(target_arch = "x86_64")] unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.add(o)); }
        #[cfg(target_arch = "aarch64")] unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.add(o), options(nostack, readonly, preserves_flags)); }
        o += 64;
    }
}

/// R31/R32：k-means 中心（orig 序分配，缓存为 `<hash>.sv3c<nl>.bin`）
fn sv3_kmeans(base: &[f32], dim: usize, n: usize, nl: usize, angular: bool, cache_dir: &str, hash: &str) -> (Vec<f32>, Vec<u32>) {
    use rayon::prelude::*;
    let l2 = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    let cpath = format!("{cache_dir}/{hash}.sv3c{nl}.bin");
    if let Ok(b) = std::fs::read(&cpath) {
        let c: Vec<f32> = b[..nl * dim * 4].chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        let a: Vec<u32> = b[nl * dim * 4..].chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
        eprintln!("SV3 centroid cache hit {cpath}"); (c, a)
    } else {
        let t = Instant::now();
        let m = (nl * 64).min(n); let step = n / m;
        let samp: Vec<usize> = (0..m).map(|i| i * step).collect();
        let mut c: Vec<f32> = samp[..nl].iter().flat_map(|&i| base[i * dim..(i + 1) * dim].to_vec()).collect();
        let assign = |c: &Vec<f32>, ids: &[usize]| -> Vec<u32> { ids.par_iter().map(|&i| { let x = &base[i * dim..(i + 1) * dim];
            let mut best = (f32::MAX, 0u32); for j in 0..nl { let d = l2(x, &c[j * dim..(j + 1) * dim]); if d < best.0 { best = (d, j as u32); } } best.1 }).collect() };
        for it in 0..10 {
            let a = assign(&c, &samp);
            let mut sum = vec![0f64; nl * dim]; let mut cnt = vec![0usize; nl];
            for (k, &i) in samp.iter().enumerate() { let j = a[k] as usize; cnt[j] += 1; for d in 0..dim { sum[j * dim + d] += base[i * dim + d] as f64; } }
            for j in 0..nl { if cnt[j] > 0 { let mut nv: Vec<f32> = (0..dim).map(|d| (sum[j * dim + d] / cnt[j] as f64) as f32).collect();
                if angular { let s = nv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); for x in nv.iter_mut() { *x /= s; } }
                c[j * dim..(j + 1) * dim].copy_from_slice(&nv); } }
            eprintln!("SV3 kmeans it {it} {:.1}s", t.elapsed().as_secs_f64());
        }
        let all: Vec<usize> = (0..n).collect();
        let a = assign(&c, &all);
        let mut b: Vec<u8> = Vec::with_capacity((nl * dim + n) * 4);
        for x in &c { b.extend_from_slice(&x.to_le_bytes()); } for x in &a { b.extend_from_slice(&x.to_le_bytes()); }
        let _ = std::fs::write(&cpath, &b); eprintln!("SV3 kmeans NL={nl} done in {:.1}s, cached {cpath}", t.elapsed().as_secs_f64());
        (c, a)
    }
}

/// R31 SV-3 v2：n 个连续 SQ4 码（步长 cb≤64，低 4 位 = 偶数维、高 4 位 = 奇数维）与查询码的点积 Σ a·b（maddubs，8 点一批 hadd 归约）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sv3_dot8_avx2(codes: &[u8], cb: usize, n: usize, qlo: &[i8; 64], qhi: &[i8; 64], out: &mut [i32]) {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(15); let one = _mm256_set1_epi16(1);
    let l0 = _mm256_loadu_si256(qlo.as_ptr() as *const __m256i); let l1 = _mm256_loadu_si256(qlo.as_ptr().add(32) as *const __m256i);
    let h0 = _mm256_loadu_si256(qhi.as_ptr() as *const __m256i); let h1 = _mm256_loadu_si256(qhi.as_ptr().add(32) as *const __m256i);
    let dot1 = |x: *const u8| -> __m256i {
        let r0 = _mm256_loadu_si256(x as *const __m256i); let r1 = _mm256_loadu_si256(x.add(32) as *const __m256i);
        let s = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(r0, m4), l0), _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r0, 4), m4), h0)),
                                 _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(r1, m4), l1), _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r1, 4), m4), h1)));
        _mm256_madd_epi16(s, one) };
    let base = codes.as_ptr(); let mut j = 0;
    // 64B 窗口读取：cb<64 时最后一个点可能越过切片末尾 → 只对 j+8 < n 的整批走 SIMD，尾部标量
    while j + 8 < n || (cb == 64 && j + 8 <= n) {
        let mut v = [_mm256_setzero_si256(); 8];
        for k in 0..8 { let x = base.add((j + k) * cb); _mm_prefetch::<_MM_HINT_T0>(x.add(16 * cb) as *const i8); v[k] = dot1(x); }
        let a0 = _mm256_hadd_epi32(v[0], v[1]); let a1 = _mm256_hadd_epi32(v[2], v[3]); let a2 = _mm256_hadd_epi32(v[4], v[5]); let a3 = _mm256_hadd_epi32(v[6], v[7]);
        let b0 = _mm256_hadd_epi32(a0, a1); let b1 = _mm256_hadd_epi32(a2, a3);
        let c0 = _mm256_add_epi32(_mm256_permute2x128_si256(b0, b1, 0x20), _mm256_permute2x128_si256(b0, b1, 0x31));
        _mm256_storeu_si256(out.as_mut_ptr().add(j) as *mut __m256i, c0);
        j += 8;
    }
    while j < n { let c = &codes[j * cb..(j + 1) * cb]; let mut acc = 0i32;
        for i in 0..cb.min(64) { acc += (c[i] & 15) as i32 * qlo[i] as i32 + (c[i] >> 4) as i32 * qhi[i] as i32; } out[j] = acc; j += 1; }
}

/// R31 SV-3 v2：int8 中心（每个 128 字节）与 u8 查询（[-63,63]+64）的点积，8 个中心一批。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sv3_cdot8_avx2(cent8: &[i8], nl: usize, q8: &[u8; 128], out: &mut [i32]) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_epi16(1);
    let q = [0usize, 32, 64, 96].map(|o| _mm256_loadu_si256(q8.as_ptr().add(o) as *const __m256i));
    let mut j = 0;
    while j + 8 <= nl {
        let mut v = [_mm256_setzero_si256(); 8];
        for k in 0..8 { let x = cent8.as_ptr().add((j + k) * 128);
            let mut s = _mm256_setzero_si256();
            for t in 0..4 { s = _mm256_add_epi32(s, _mm256_madd_epi16(_mm256_maddubs_epi16(q[t], _mm256_loadu_si256(x.add(t * 32) as *const __m256i)), one)); }
            v[k] = s; }
        let a0 = _mm256_hadd_epi32(v[0], v[1]); let a1 = _mm256_hadd_epi32(v[2], v[3]); let a2 = _mm256_hadd_epi32(v[4], v[5]); let a3 = _mm256_hadd_epi32(v[6], v[7]);
        let b0 = _mm256_hadd_epi32(a0, a1); let b1 = _mm256_hadd_epi32(a2, a3);
        let c0 = _mm256_add_epi32(_mm256_permute2x128_si256(b0, b1, 0x20), _mm256_permute2x128_si256(b0, b1, 0x31));
        _mm256_storeu_si256(out.as_mut_ptr().add(j) as *mut __m256i, c0); j += 8;
    }
    while j < nl { let mut acc = 0i32; for d in 0..128 { acc += q8[d] as i32 * cent8[j * 128 + d] as i32; } out[j] = acc; j += 1; }
}

/// R31 SV-3 v3：融合核——8 点一批算 Σab，SIMD 内算 score = Σb² − 2Σab 并与门槛比较，只把幸存者 (score, 局部下标) 写出；返回幸存者数。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sv3_scan_thr_avx2(codes: &[u8], cb: usize, n: usize, norms: &[i32], thr: i32, qlo: &[i8; 64], qhi: &[i8; 64], out: &mut [i32]) -> usize {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(15); let one = _mm256_set1_epi16(1);
    let l0 = _mm256_loadu_si256(qlo.as_ptr() as *const __m256i); let l1 = _mm256_loadu_si256(qlo.as_ptr().add(32) as *const __m256i);
    let h0 = _mm256_loadu_si256(qhi.as_ptr() as *const __m256i); let h1 = _mm256_loadu_si256(qhi.as_ptr().add(32) as *const __m256i);
    let tv = _mm256_set1_epi32(thr);
    let base = codes.as_ptr(); let mut j = 0; let mut ns = 0usize;
    while j + 8 <= n && (cb == 64 || j + 8 < n) {
        let mut v = [_mm256_setzero_si256(); 8];
        for k in 0..8 { let x = base.add((j + k) * cb); _mm_prefetch::<_MM_HINT_T0>(x.add(16 * cb) as *const i8);
            let r0 = _mm256_loadu_si256(x as *const __m256i); let r1 = _mm256_loadu_si256(x.add(32) as *const __m256i);
            let s = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(r0, m4), l0), _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r0, 4), m4), h0)),
                                     _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(r1, m4), l1), _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r1, 4), m4), h1)));
            v[k] = _mm256_madd_epi16(s, one); }
        let a0 = _mm256_hadd_epi32(v[0], v[1]); let a1 = _mm256_hadd_epi32(v[2], v[3]); let a2 = _mm256_hadd_epi32(v[4], v[5]); let a3 = _mm256_hadd_epi32(v[6], v[7]);
        let b0 = _mm256_hadd_epi32(a0, a1); let b1 = _mm256_hadd_epi32(a2, a3);
        let c0 = _mm256_add_epi32(_mm256_permute2x128_si256(b0, b1, 0x20), _mm256_permute2x128_si256(b0, b1, 0x31));
        let sv = _mm256_sub_epi32(_mm256_loadu_si256(norms.as_ptr().add(j) as *const __m256i), _mm256_add_epi32(c0, c0));
        let mut m = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(tv, sv))) as u32;
        if m != 0 { let mut t = [0i32; 8]; _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, sv);
            while m != 0 { let b = m.trailing_zeros() as usize; out[2 * ns] = t[b]; out[2 * ns + 1] = (j + b) as i32; ns += 1; m &= m - 1; } }
        j += 8;
    }
    while j < n { let c = &codes[j * cb..(j + 1) * cb]; let mut acc = 0i32;
        for i in 0..cb.min(64) { acc += (c[i] & 15) as i32 * qlo[i] as i32 + (c[i] >> 4) as i32 * qhi[i] as i32; }
        let sv = norms[j] - 2 * acc; if sv < thr { out[2 * ns] = sv; out[2 * ns + 1] = j as i32; ns += 1; } j += 1; }
    ns
}


/// R75：打印一侧管线大块缓冲区的地址（mod 2 MB）与所在 VMA 的 Rss / AnonHugePages（/proc/self/smaps），用于诊断侧别偏差。
fn abx_map(tag: &str, p: &raven::pipeline::Prepared) {
    let sm = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
    let mut parts: Vec<(&str, usize, usize)> = vec![
        ("vectors", p.vectors.as_ptr() as usize, p.vectors.len() * 4),
        ("sq4", p.sq4.codes.as_ptr() as usize, p.sq4.codes.len()),
    ];
    if let Some(f) = p.sq4_full.as_ref() { parts.push(("sq4_full", f.codes.as_ptr() as usize, f.codes.len())); }
    if let Some(u) = p.vec_u8.as_ref() { parts.push(("vec_u8", u.as_ptr() as usize, u.len())); }
    let st = p.graph.storage();
    parts.push(("adj", st.neighbors(0).as_ptr() as usize, st.len() * st.r_max() * 4));
    let mut line = format!("ABX map {tag}:");
    for (name, a, len) in parts {
        let (rss, huge) = raven::abtime::smaps_huge(&sm, a, a + len.max(1));
        line.push_str(&format!(" {name}@{:#x}(mod2M={:#x},{}MB,rss={}MB,thp={}MB)", a, a % (2 << 20), len >> 20, rss >> 10, huge >> 10));
    }
    eprintln!("{line}");
}

/// R66：进程内 A/B 计时工具（见 `raven::abtime`）。
/// RAVEN_ABX=aa | pad16 | pad:N；RAVEN_ABX_BLOCK（缺省 500 查询/块）、RAVEN_ABX_REPS（缺省 --repeats）、RAVEN_ABX_BOOT（缺省 2000）。
/// 流程：A = 已建管线；B = 重载同一图文件 + prepare（search_degree 固定为 A.sd，HR 门槛经 RAVEN_HR_GATE 固定为 A 的值或 RAVEN_HR=0，
/// SDX/LPX/po 从 A 复制）→ 两侧各全量跑一遍（不计时，取结果做召回与逐位比对）→ rep 外层、ef 中层、块内层，每块 A、B 各跑一次（先后轮换）。
/// 输出（--out 为 csv 路径，空则只打 stderr）：逐 ef 中位 Δt/q（B/A−1）、95% CI（按 rep 整簇 bootstrap）、合并比、µs/q、召回与 ids 是否逐位一致；
/// 另写 <out>.pairs.csv（每对原始耗时，供离线复核）。
#[allow(clippy::too_many_arguments)]
fn abx_run(pa: raven::pipeline::Prepared, po_: &raven::pipeline::PipelineOpts, gpath: &str, spec: &str, base: &[f32], dim: usize, queries: &[f32], nq: usize,
           gt_dist: &[f32], gtk: usize, angular: bool, efs: &[usize], repeats: usize, kk: usize, out: &str) {
    use raven::abtime::{ab_stats, first_side, Pair};
    let envu = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d);
    let block = envu("RAVEN_ABX_BLOCK", 500).max(1);
    let reps = envu("RAVEN_ABX_REPS", repeats).max(1);
    let n_boot = envu("RAVEN_ABX_BOOT", 2000);
    // R67b/R68：adc = B 侧打开 ADC16（PipelineOpts.sq4_adc = Some(true)），A 侧必须关（RAVEN_SQ4_ADC=0）
    let adc_b = spec == "adc";
    // R75：pfl:N = B 侧把 L0 邻居码预取行数（进程全局 PF_LINES）设为 N，A 侧 = RAVEN_ABX_PFLA（缺省 = 进入时的进程值）；same = 同对象 A/A
    let pfl_b: Option<usize> = spec.strip_prefix("pfl:").map(|v| v.parse().expect("RAVEN_ABX pfl:N"));
    let pfl_a: usize = std::env::var("RAVEN_ABX_PFLA").ok().and_then(|v| v.parse().ok())
        .unwrap_or_else(|| raven::graph::vamana::PF_LINES.load(std::sync::atomic::Ordering::Relaxed));
    let pad_b: Option<usize> = match spec {
        _ if raven::abtime::runtime_only_spec(spec) => None,
        "aa" | "adc" => None,
        "pad16" => Some(16),
        x => Some(x.strip_prefix("pad:").and_then(|v| v.parse().ok()).expect("RAVEN_ABX must be aa | same | adc | pad16 | pad:N | pfl:N")),
    };
    assert!(std::env::var("RAVEN_SQ4_PAD").is_err(), "RAVEN_ABX: unset RAVEN_SQ4_PAD (it overrides both arms)");
    // R75 侧别偏差修正：构造方式见 raven::abtime::BuildMode（缺省 Same / Sym；RAVEN_ABX_LEGACY=1 = R66–R74 旧行为）
    let mode = raven::abtime::build_mode(spec, std::env::var("RAVEN_ABX_LEGACY").map(|v| v == "1").unwrap_or(false),
        std::env::var("RAVEN_ABX_ORDER").map(|v| v == "ba").unwrap_or(false));
    // 重载同图文件（与 A 同一路径；若启用填充图则用同一填充缓存）
    let gfile = match raven::graph::vamana::fill_default() {
        Some((fd, fe)) => { let f = gpath.replace(".bin", &format!(".f{fd}e{fe}.bin")); if std::path::Path::new(&f).exists() { f } else { gpath.to_string() } }
        None => gpath.to_string(),
    };
    let mut oa = po_.clone();
    oa.search_degree = Some(pa.sd);
    let mut ob = oa.clone();
    // aa：B 与 A 同一解析（po_ 原样 → 同读 RAVEN_SQ4_PAD_SMALL / 缺省）；pad:N：B 程序化设为 N
    if let Some(nb) = pad_b { ob.sq4_pad_small = Some(nb); }
    if adc_b { ob.sq4_adc = Some(true); }
    assert!(!(adc_b && pa.adc16), "ABX adc: A side must have ADC16 off (set RAVEN_SQ4_ADC=0)");
    // 计划（sd/HR 门槛/SDX/LPX/po/rerank）取自进程原管线，复制到每一份新建管线
    let plan = (pa.sdx.clone(), pa.lpx.clone(), pa.lpx_sdx.clone(), pa.po, pa.rerank, pa.hr.as_ref().map(|c| c.params));
    let mk = |o: &raven::pipeline::PipelineOpts| -> raven::pipeline::Prepared {
        let g2 = VamanaGraph::load(std::path::Path::new(&gfile)).expect("ABX: reload graph");
        let (old_gate, old_hr) = (std::env::var("RAVEN_HR_GATE").ok(), std::env::var("RAVEN_HR").ok());
        match plan.5.as_ref() {
            Some(hp) => { std::env::set_var("RAVEN_HR_GATE", hp.gate.to_string()); }
            None => { std::env::set_var("RAVEN_HR", "0"); }
        }
        let mut p = raven::pipeline::prepare(g2, base.to_vec(), dim, o);
        match old_gate { Some(v) => std::env::set_var("RAVEN_HR_GATE", v), None => std::env::remove_var("RAVEN_HR_GATE") }
        match old_hr { Some(v) => std::env::set_var("RAVEN_HR", v), None => std::env::remove_var("RAVEN_HR") }
        p.sdx = plan.0.clone(); p.lpx = plan.1.clone(); p.lpx_sdx = plan.2.clone(); p.po = plan.3; p.rerank = plan.4;
        if let (Some(a), Some(b)) = (plan.5.as_ref(), p.hr.as_mut()) { b.params = *a; }
        p
    };
    let t = Instant::now();
    let (side_a, side_b): (raven::pipeline::Prepared, Option<raven::pipeline::Prepared>) = match mode {
        raven::abtime::BuildMode::Same => (pa, None),
        raven::abtime::BuildMode::Legacy => { let pb = mk(&ob); (pa, Some(pb)) }
        raven::abtime::BuildMode::Sym { b_first } => {
            drop(pa);
            if b_first { let pb = mk(&ob); let a2 = mk(&oa); (a2, Some(pb)) } else { let a2 = mk(&oa); let pb = mk(&ob); (a2, Some(pb)) }
        }
    };
    let sides: [&raven::pipeline::Prepared; 2] = [&side_a, side_b.as_ref().unwrap_or(&side_a)];
    let (pa, pb) = (sides[0], sides[1]);
    // R75：RAVEN_ABX_AUTO=1 → 计时真缺省路径（searcher_k(ef, k) + run_auto：含 HR/k 感知；R72 工具同口径）；缺省 0 = R66 原口径（searcher(ef) + run_sq4）
    let auto = std::env::var("RAVEN_ABX_AUTO").map(|v| v == "1").unwrap_or(false);
    let set_side = |side: usize| { if let Some(nb) = pfl_b {
        raven::graph::vamana::PF_LINES.store(if side == 1 { nb } else { pfl_a }, std::sync::atomic::Ordering::Relaxed); } };
    let hr_s = |p: &raven::pipeline::Prepared| p.hr.as_ref().map(|c| c.params.gate.to_string()).unwrap_or_else(|| "off".into());
    eprintln!("ABX spec={spec} mode={mode:?} auto={} block={block} reps={reps} boot={n_boot} adcA={} adcB={} pfl A={pfl_a} B={pfl_b:?} | A: cb={} sd={} hr_gate={} sdx={} | B: cb={} sd={} hr_gate={} sdx={} (sides prepared in {:.1}s, graph {gfile})",
        auto, pa.adc16, pb.adc16, pa.sq4.code_bytes, pa.sd, hr_s(pa), pa.sdx.as_ref().map(|x| x.ef_max as i64).unwrap_or(-1),
        pb.sq4.code_bytes, pb.sd, hr_s(pb), pb.sdx.as_ref().map(|x| x.ef_max as i64).unwrap_or(-1), t.elapsed().as_secs_f64());
    abx_map("A", pa); if side_b.is_some() { abx_map("B", pb); }
    let recall_of = |res: &Vec<Vec<u32>>| -> f64 {
        let mut tot = 0usize;
        for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim];
            let thr = gt_dist[qi * gtk + kk - 1] as f64 + 1e-3;
            tot += res[qi].iter().take(kk).filter(|&&id| { let x = &base[id as usize * dim..(id as usize + 1) * dim];
                (if angular { angular_dist(q, x) } else { exact_dist(q, x) }) <= thr }).count(); }
        tot as f64 / (nq * kk) as f64
    };
    // 不计时全量一遍：取结果（召回 + 逐位比对），兼作预热
    let mut rec: Vec<(f64, f64, bool)> = Vec::new();
    for &ef in efs {
        let mut res: [Vec<Vec<u32>>; 2] = [Vec::with_capacity(nq), Vec::with_capacity(nq)];
        for side in 0..2 { let pp = sides[side]; let mut s = if auto { pp.searcher_k(ef, kk) } else { pp.searcher(ef) }; set_side(side);
            for qi in 0..nq { let q = &queries[qi * dim..(qi + 1) * dim]; let r = if auto { pp.run_auto(&mut s, q, kk) } else { pp.run_sq4(&mut s, q, kk) }; res[side].push(r.iter().take(kk).map(|x| pp.orig_id(x.0)).collect()); } }
        let same = res[0] == res[1];
        let (ra, rb) = (recall_of(&res[0]), recall_of(&res[1]));
        eprintln!("ABX ef={ef} recall A={ra:.5} B={rb:.5} ids_identical={same}");
        rec.push((ra, rb, same));
    }
    let nblk = nq.div_ceil(block);
    let mut pairs: Vec<Vec<Pair>> = vec![Vec::new(); efs.len()];
    let mut firsts: Vec<Vec<usize>> = vec![Vec::new(); efs.len()];
    // R75：同对象模式（运行期旋钮 spec）两侧共用一个 searcher → 两侧内存与状态完全相同，只差旋钮
    let shared = side_b.is_none();
    let mut raw = String::from("ef,rep,block,first,ta_s,tb_s\n");
    let t0 = Instant::now();
    for rep in 0..reps {
        for (ei, &ef) in efs.iter().enumerate() {
            // R75：searcher（visited 等缓冲区）分配先后逐 rep 交换
            let k_s = if auto { kk } else { 10 }; // searcher(ef) == searcher_k(ef, 10)
            let (mut sa, mut sb) = if raven::abtime::searcher_first(rep) == 0 { let a = pa.searcher_k(ef, k_s); (a, pb.searcher_k(ef, k_s)) } else { let b = pb.searcher_k(ef, k_s); (pa.searcher_k(ef, k_s), b) };
            for c in 0..nblk {
                let (lo, hi) = (c * block, ((c + 1) * block).min(nq));
                let f = first_side(rep, c, ei);
                let mut tm = [0f64; 2];
                for step in 0..2 {
                    let side = f ^ step;
                    let s = if side == 0 || shared { &mut sa } else { &mut sb };
                    set_side(side);
                    let t = Instant::now();
                    let pp = sides[side];
                    if auto { for qi in lo..hi { std::hint::black_box(pp.run_auto(s, &queries[qi * dim..(qi + 1) * dim], kk)); } }
                    else { for qi in lo..hi { std::hint::black_box(pp.run_sq4(s, &queries[qi * dim..(qi + 1) * dim], kk)); } }
                    tm[side] = t.elapsed().as_secs_f64() / (hi - lo) as f64;
                }
                pairs[ei].push(Pair { rep, ta: tm[0], tb: tm[1] }); firsts[ei].push(f);
                raw.push_str(&format!("{ef},{rep},{c},{f},{:.9},{:.9}\n", tm[0], tm[1]));
            }
        }
        eprintln!("ABX rep {}/{reps} done ({:.0}s)", rep + 1, t0.elapsed().as_secs_f64());
    }
    // R75：主统计 = 先后平衡对偶（raven::abtime::couples）；旧的逐块中位数作为 median_pair_delta 一并输出
    let mut csv = String::from("spec,ef,recall_A,recall_B,ids_identical,us_q_A,us_q_B,median_delta,ci_lo,ci_hi,pooled_delta,frac_B_faster,n_pairs,reps,block,median_pair_delta,mode\n");
    let mut all_med = Vec::new();
    for (ei, &ef) in efs.iter().enumerate() {
        let st = ab_stats(&raven::abtime::couples(&pairs[ei], &firsts[ei]), n_boot, 0xAB66 + ef as u64);
        let st_pair = ab_stats(&pairs[ei], n_boot, 0xAB66 + ef as u64);
        let mut ua: Vec<f64> = pairs[ei].iter().map(|p| p.ta * 1e6).collect();
        let mut ub: Vec<f64> = pairs[ei].iter().map(|p| p.tb * 1e6).collect();
        let (ma, mb) = (raven::abtime::median(&mut ua), raven::abtime::median(&mut ub));
        let (ra, rb, same) = rec[ei];
        eprintln!("ABX ef={ef} A {ma:.2}us/q B {mb:.2}us/q  median dt/q {:+.2}%  95%CI [{:+.2}%, {:+.2}%]  pooled {:+.2}%  B faster {:.0}%  (couples={} reps={}; per-block median {:+.2}%)  recall {ra:.5}/{rb:.5} ids_identical={same}",
            st.median * 100.0, st.lo * 100.0, st.hi * 100.0, st.pooled * 100.0, st.frac_b_faster * 100.0, st.n_pairs, st.n_reps, st_pair.median * 100.0);
        csv.push_str(&format!("{spec},{ef},{ra:.5},{rb:.5},{same},{ma:.3},{mb:.3},{:.5},{:.5},{:.5},{:.5},{:.3},{},{},{block},{:.5},{mode:?}\n", st.median, st.lo, st.hi, st.pooled, st.frac_b_faster, st.n_pairs, st.n_reps, st_pair.median));
        all_med.push(st.median);
    }
    let mut am = all_med.clone();
    if pfl_b.is_some() { raven::graph::vamana::PF_LINES.store(pfl_a, std::sync::atomic::Ordering::Relaxed); }
    eprintln!("ABX summary spec={spec}: median over ef of per-ef median dt/q = {:+.2}%  all_ids_identical={}", raven::abtime::median(&mut am) * 100.0, rec.iter().all(|x| x.2));
    if !out.is_empty() {
        std::fs::write(out, &csv).expect("write ABX csv");
        std::fs::write(format!("{out}.pairs.csv"), &raw).expect("write ABX pairs");
    }
    print!("{csv}");
}
