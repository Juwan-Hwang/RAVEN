//! R95 gport：GINGER 建图一致化模式（RAVEN_BUILD=pipnn 且 RAVEN_PIP_G=1；缺省不变）。
//!
//! 对照 GINGER 源码（google/ginger 13c0929：pipnn/pipnn.cc、pipnn/pipnn_impl.cc、turboquant/pipnn/process_leaf.cc、
//! turboquant/pipnn/robust_prune.cc、config_recommender.cc）逐条复现它的推荐配置（gbench = RecommendGingerConfig）：
//! 1. 分区：`num_passes`=2 次独立递归划分；depth 0 随机取 min(1024, n) 个领导者、每点归最近 fanout[0]=10 个；
//!    depth≥1 领导者数 = clamp(⌊size·0.005⌋, 3, 512)、fanout[1]=3，更深 fanout=1；子簇 ≤ max_leaf_size=1000 即为叶；
//!    每层划分后 MergeSmallClusters(min_leaf_size=100)：按大小降序，把最小簇并进次小簇（并集去重）直到 ≥100 或放不下；
//!    卡住（depth>8，或 depth>6 且子簇 > 0.8·父簇）→ 随机切成 ≤1000 的块。
//! 2. 叶内：每点叶内最近 2 个（ProcessLeafTop2），双向：u→b1、u→b2、b1→u、b2→u，按目标点分组。
//! 3. 合并：PRUNING_STRATEGY_ROBUST_PRUNE_ON_THE_FLY（推荐配置的策略，不是 HashPrune）：每点邻接缓冲上限 max_degree=64，
//!    新边去重后追加；放不下时对 当前∪新 做 RobustPrune(α) 剪到 ⌊0.75·64⌋=48（keep_pruned=false）。
//! 4. 收尾：每点对缓冲再做一次 RobustPrune(α) 到 max_degree。剪枝规则：候选按 (d(u,c), id) 升序，
//!    c 保留 ⇔ 对所有已留 s：α·d(s,c) ≥ d(u,c)，d 为**平方** L2（GINGER α=1.1 即本模块 A2=1.1）。
//! 5. 距离：GINGER 建图全程用量化码（TurboQuant 4 bit + int8 点积）。这里用全局对称 int8 标量量化 + AVX-512 VNNI
//!    (vpdpbusd) 点积，d = ‖a‖²+‖b‖²−2a·b 全整数；RAVEN_PIP_GF=1 时收尾剪枝改用 float 精确距离。
//! 入口点 / 分层导航沿用 RAVEN（build_pipnn：medoid），GINGER 的图重排与压缩只影响它自己的搜索，不移植。
//!
//! 环境变量（括号内为 GINGER 推荐值 = 缺省）：RAVEN_PIP_GPASS(2) RAVEN_PIP_GTOP(1024) RAVEN_PIP_GFAN("10,3")
//! RAVEN_PIP_GRATE(0.005) RAVEN_PIP_GMAXL(512) RAVEN_PIP_GLMAX(1000) RAVEN_PIP_GLMIN(100) RAVEN_PIP_GK(2)
//! RAVEN_PIP_GMD(64) RAVEN_PIP_GOTF(0.75) RAVEN_PIP_A2(1.1) RAVEN_PIP_CAP(r_max) RAVEN_PIP_GF(0) RAVEN_PIP_SEED(42)
//! 原创扩展（缺省关）：RAVEN_PIP_GDB=b>0 收尾时按方向分桶保留长边（见 `dir_bucket_fill`）。
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// R99-13：缺省 1（GINGER 一致化建图）；RAVEN_PIP_G=0 恢复 R94 pipnn。
pub fn enabled() -> bool { env::<usize>("RAVEN_PIP_G", 1) > 0 }

#[derive(Debug, Clone)]
pub struct GParams {
    pub passes: usize, pub top: usize, pub fan: Vec<usize>, pub rate: f32, pub maxl: usize, pub lmax: usize, pub lmin: usize,
    pub k: usize, pub md: usize, pub otf: f32, pub a2: f32, pub cap: usize, pub gf: bool, pub seed: u64,
    pub db: usize, pub dbk: usize, pub hp: bool, pub hr: usize, pub hb: usize, pub panel: bool,
    /// R98 H181：pk2 = GK=2 也走泛化核（校验用）；occ = 叶内 α-遮挡预剪系数（0 关），occ_r = 从第几近邻起检查（缺省 1）。
    pub pk2: bool, pub occ: f32, pub occ_r: usize,
    /// R98 H159：cons = 多叶共识门（候选边被 ≥cons 个叶独立拾取则免 α 检查直接保留，0 关）；inc = 增量精确剪枝
    /// （上次剪枝留下的边两两不再互检，只查新旧交叉对；输出与原剪枝逐位相同）；dump = 叶成员表落盘路径（H158 LCF oracle）。
    pub cons: u8, pub inc: bool, pub dump: String,
    /// R99 H192 LSM：lsm = 每遍叶分成几批做「线程本地分片缓冲 → 按 src 排序归约」合并（0 = 原逐边加锁合并）；
    /// buf = OTF 剪枝触发阈值倍数（缓冲 > md·buf 才剪，1.0 = GINGER 原规则；>1 时每点剪枝次数下降，收尾一次剪全集）。
    pub lsm: usize, pub buf: f32,
}

impl GParams {
    pub fn from_env(r_max: usize) -> Self {
        let fan: Vec<usize> = std::env::var("RAVEN_PIP_GFAN").unwrap_or_else(|_| "10,3".into())
            .split(',').filter_map(|s| s.trim().parse().ok()).map(|f: usize| f.max(1)).collect();
        GParams {
            passes: env("RAVEN_PIP_GPASS", 2usize).max(1), top: env("RAVEN_PIP_GTOP", 1024usize).max(2), fan,
            rate: env("RAVEN_PIP_GRATE", 0.005f32), maxl: env("RAVEN_PIP_GMAXL", 512usize).max(3),
            lmax: env("RAVEN_PIP_GLMAX", 1000usize).max(8), lmin: env("RAVEN_PIP_GLMIN", 100usize).max(1),
            k: env("RAVEN_PIP_GK", 2usize).max(1), md: env("RAVEN_PIP_GMD", 64usize).max(4), otf: env("RAVEN_PIP_GOTF", 0.75f32),
            a2: env("RAVEN_PIP_A2", 1.1f32), cap: env::<usize>("RAVEN_PIP_CAP", r_max).min(r_max).max(1),
            gf: env::<usize>("RAVEN_PIP_GF", 0) > 0, seed: env("RAVEN_PIP_SEED", 42u64),
            db: env("RAVEN_PIP_GDB", 0usize), dbk: env("RAVEN_PIP_GDBK", 1usize).max(1),
            hp: env::<usize>("RAVEN_PIP_GHP", 0) > 0, hr: env("RAVEN_PIP_GHR", 96usize).max(4), hb: env("RAVEN_PIP_GHB", 12usize).clamp(1, 16),
            panel: env::<usize>("RAVEN_PIP_GPANEL", 1) > 0,
            pk2: env::<usize>("RAVEN_PIP_GPK2", 0) > 0, occ: env("RAVEN_PIP_GOCC", 0f32), occ_r: env("RAVEN_PIP_GOCCR", 1usize).max(1),
            cons: env::<usize>("RAVEN_PIP_GCONS", 0).min(255) as u8, inc: env::<usize>("RAVEN_PIP_GINC", 1) > 0, // R98 H159：缺省开（输出等价；glove 建图 −12%，ccnews/glove QPS 同级）
            dump: std::env::var("RAVEN_PIP_DUMPLEAVES").unwrap_or_default(),
            lsm: env("RAVEN_PIP_GLSM", 16usize), buf: env("RAVEN_PIP_GBUF", 1.0f32).max(1.0), // R99 H192 缺省 GLSM=16、GBUF=1（GBUF3 在 landmark @0.99 掉 9%；L4B1 landmark @0.99 0.93–0.95×，L16B1 1.01–1.04×）
        }
    }
    fn fanout(&self, depth: usize) -> usize { self.fan.get(depth).copied().unwrap_or(1) }
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03 | 1) }
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

// ---------------------------------------------------------------- int8 码与 VNNI 核
pub struct Q8 { pub dp: usize, pub q: Vec<i8>, pub nrm: Vec<i32>, pub sum: Vec<i32>, pub sc: f32 }

impl Q8 {
    pub fn new(x: &[f32], dim: usize) -> Q8 {
        let n = x.len() / dim;
        let dp = dim.div_ceil(64) * 64;
        // 全局对称标度：|x| 的 99.99% 分位（抽样）
        let mut smp: Vec<f32> = Vec::with_capacity(1 << 20);
        let step = (x.len() / (1 << 20)).max(1);
        let mut i = 0; while i < x.len() { smp.push(x[i].abs()); i += step; }
        smp.sort_unstable_by(|a, b| a.total_cmp(b));
        let clip = smp[((smp.len() as f64 * 0.9999) as usize).min(smp.len() - 1)].max(1e-30);
        let sc = 127.0 / clip;
        let mut q = vec![0i8; n * dp];
        let mut nrm = vec![0i32; n]; let mut sum = vec![0i32; n];
        q.par_chunks_mut(dp).zip(nrm.par_iter_mut()).zip(sum.par_iter_mut()).enumerate().for_each(|(u, ((row, nr), sm))| {
            let xr = &x[u * dim..(u + 1) * dim];
            let (mut a, mut b) = (0i32, 0i32);
            for j in 0..dim { let v = (xr[j] * sc).round().clamp(-127.0, 127.0) as i32; row[j] = v as i8; a += v * v; b += v; }
            *nr = a; *sm = b;
        });
        Q8 { dp, q, nrm, sum, sc }
    }
    /// R96-Q：外部向量（查询）按同一标度量化，返回 (码行[dp], 范数²)
    pub fn quant(&self, v: &[f32]) -> (Vec<i8>, i32) {
        let mut row = vec![0i8; self.dp]; let mut a = 0i32;
        for j in 0..v.len() { let t = (v[j] * self.sc).round().clamp(-127.0, 127.0) as i32; row[j] = t as i8; a += t * t; }
        (row, a)
    }
    /// 查询码到点 b 的整数平方距离
    #[inline(always)] pub fn dq(&self, qr: &[i8], qn: i32, b: u32) -> i32 {
        qn + self.nrm[b as usize] - 2 * (dot1(qr.as_ptr(), self.row(b), self.dp) - 128 * self.sum[b as usize])
    }
    /// 预取点 i 的码行
    #[inline(always)] pub fn prefetch(&self, i: u32) {
        #[cfg(target_arch = "x86_64")]
        unsafe { use std::arch::x86_64::*; let p = self.row(i); let mut o = 0; while o < self.dp { _mm_prefetch(p.add(o) as *const i8, _MM_HINT_T0); o += 64; } }
    }
    #[inline(always)] fn row(&self, i: u32) -> *const i8 { unsafe { self.q.as_ptr().add(i as usize * self.dp) } }
    /// 整数平方距离
    #[inline(always)] pub fn d(&self, a: u32, b: u32) -> i32 {
        self.nrm[a as usize] + self.nrm[b as usize] - 2 * (dot1(self.row(a), self.row(b), self.dp) - 128 * self.sum[b as usize])
    }
}

fn have_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    { static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
      *V.get_or_init(|| std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vnni")) }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// Σ (a⊕0x80 as u8)·b = a·b + 128·Σb
#[inline(always)]
fn dot1(a: *const i8, b: *const i8, dp: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    { if have_vnni() { return unsafe { dot1_vnni(a, b, dp) }; } }
    let mut s = 0i32;
    for c in 0..dp { unsafe { s += ((*a.add(c) as i32) + 128) * (*b.add(c) as i32); } }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot1_vnni(a: *const i8, b: *const i8, dp: usize) -> i32 {
    use std::arch::x86_64::*;
    let x80 = _mm512_set1_epi8(-128);
    let mut acc = _mm512_setzero_si512();
    let mut c = 0;
    while c < dp {
        let va = _mm512_xor_si512(_mm512_loadu_si512(a.add(c) as *const _), x80);
        acc = _mm512_dpbusd_epi32(acc, va, _mm512_loadu_si512(b.add(c) as *const _));
        c += 64;
    }
    _mm512_reduce_add_epi32(acc)
}

/// 4×4 块：out[r*4+t] = Σ (a_r⊕0x80)·b_t
#[inline(always)]
fn dot4x4(a: [*const i8; 4], b: [*const i8; 4], dp: usize, out: &mut [i32; 16]) {
    #[cfg(target_arch = "x86_64")]
    { if have_vnni() { unsafe { dot4x4_vnni(a, b, dp, out); } return; } }
    for r in 0..4 { for t in 0..4 { out[r * 4 + t] = dot1(a[r], b[t], dp); } }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot4x4_vnni(a: [*const i8; 4], b: [*const i8; 4], dp: usize, out: &mut [i32; 16]) {
    use std::arch::x86_64::*;
    let x80 = _mm512_set1_epi8(-128);
    let mut acc = [_mm512_setzero_si512(); 16];
    let mut c = 0;
    while c < dp {
        let x = [_mm512_xor_si512(_mm512_loadu_si512(a[0].add(c) as *const _), x80), _mm512_xor_si512(_mm512_loadu_si512(a[1].add(c) as *const _), x80),
                 _mm512_xor_si512(_mm512_loadu_si512(a[2].add(c) as *const _), x80), _mm512_xor_si512(_mm512_loadu_si512(a[3].add(c) as *const _), x80)];
        let y = [_mm512_loadu_si512(b[0].add(c) as *const _), _mm512_loadu_si512(b[1].add(c) as *const _),
                 _mm512_loadu_si512(b[2].add(c) as *const _), _mm512_loadu_si512(b[3].add(c) as *const _)];
        for r in 0..4 { for t in 0..4 { acc[r * 4 + t] = _mm512_dpbusd_epi32(acc[r * 4 + t], x[r], y[t]); } }
        c += 64;
    }
    for i in 0..16 { out[i] = _mm512_reduce_add_epi32(acc[i]); }
}

/// 1×4：c 对 4 个 s（剪枝内层批处理）
#[inline(always)]
fn dot1x4(a: *const i8, b: [*const i8; 4], dp: usize, out: &mut [i32; 4]) {
    #[cfg(target_arch = "x86_64")]
    { if have_vnni() { unsafe { dot1x4_vnni(a, b, dp, out); } return; } }
    for t in 0..4 { out[t] = dot1(a, b[t], dp); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot1x4_vnni(a: *const i8, b: [*const i8; 4], dp: usize, out: &mut [i32; 4]) {
    use std::arch::x86_64::*;
    let x80 = _mm512_set1_epi8(-128);
    let mut acc = [_mm512_setzero_si512(); 4];
    let mut c = 0;
    while c < dp {
        let x = _mm512_xor_si512(_mm512_loadu_si512(a.add(c) as *const _), x80);
        for t in 0..4 { acc[t] = _mm512_dpbusd_epi32(acc[t], x, _mm512_loadu_si512(b[t].add(c) as *const _)); }
        c += 64;
    }
    for t in 0..4 { out[t] = _mm512_reduce_add_epi32(acc[t]); }
}

// ---------------------------------------------------------------- 划分
/// 每点最近 f 个领导者（按 ‖l‖²−2x·l 排序；x 的范数对排序无影响）
fn nearest_leaders(qd: &Q8, ids: &[u32], lead: &[u32], f: usize) -> Vec<u16> {
    let m = lead.len(); let dp = qd.dp;
    // 领导者码拷进连续缓冲（补到 4 的倍数，补行全 0、范数 +∞）
    let mp = m.div_ceil(4) * 4;
    let mut lb = vec![0i8; mp * dp];
    let mut lc = vec![i64::MAX / 4; mp];
    for (i, &l) in lead.iter().enumerate() {
        unsafe { std::ptr::copy_nonoverlapping(qd.row(l), lb.as_mut_ptr().add(i * dp), dp); }
        lc[i] = qd.nrm[l as usize] as i64 + 256 * qd.sum[l as usize] as i64; // ‖l‖² + 2·128·Σl（抵消 dpbusd 偏置）
    }
    let mut out = vec![0u16; ids.len() * f];
    out.par_chunks_mut(f * 64).zip(ids.par_chunks(64)).for_each(|(o, ch)| {
        let mut best: Vec<(i64, u16)> = vec![(i64::MAX, 0); ch.len() * f];
        let mut tmp = [0i32; 16];
        let mut pi = 0;
        while pi < ch.len() {
            let np = (ch.len() - pi).min(4);
            let a: [*const i8; 4] = std::array::from_fn(|r| qd.row(ch[pi + r.min(np - 1)]));
            let mut li = 0;
            while li < mp {
                let b: [*const i8; 4] = std::array::from_fn(|t| unsafe { lb.as_ptr().add((li + t) * dp) });
                dot4x4(a, b, dp, &mut tmp);
                for r in 0..np {
                    let bb = &mut best[(pi + r) * f..(pi + r + 1) * f];
                    for t in 0..4 {
                        let s = lc[li + t] - 2 * tmp[r * 4 + t] as i64;
                        if s < bb[f - 1].0 {
                            let mut k = f - 1; while k > 0 && bb[k - 1].0 > s { bb[k] = bb[k - 1]; k -= 1; }
                            bb[k] = (s, (li + t) as u16);
                        }
                    }
                }
                li += 4;
            }
            pi += 4;
        }
        for (i, e) in best.iter().enumerate() { o[i] = e.1; }
    });
    out
}

/// GINGER MergeSmallClusters（簇内 id 已升序）
fn merge_small(cl: &mut Vec<Vec<u32>>, lmin: usize, lmax: usize) {
    cl.sort_by(|a, b| b.len().cmp(&a.len())); // 稳定
    while cl.len() > 1 && cl.last().unwrap().len() < lmin {
        let small = cl.pop().unwrap();
        let tgt = cl.len() - 1;
        if cl[tgt].len() + small.len() <= lmax {
            let a = std::mem::take(&mut cl[tgt]);
            let mut m = Vec::with_capacity(a.len() + small.len());
            let (mut i, mut j) = (0, 0);
            while i < a.len() || j < small.len() {
                if j >= small.len() || (i < a.len() && a[i] < small[j]) { m.push(a[i]); i += 1; }
                else if i >= a.len() || small[j] < a[i] { m.push(small[j]); j += 1; }
                else { m.push(a[i]); i += 1; j += 1; }
            }
            cl[tgt] = m;
            let mut c = tgt; while c > 0 && cl[c].len() > cl[c - 1].len() { cl.swap(c, c - 1); c -= 1; }
        } else { cl.push(small); break; }
    }
}

fn split_random(mut ids: Vec<u32>, seed: u64, lmax: usize, out: &Mutex<Vec<Vec<u32>>>) {
    let mut r = Rng::new(seed);
    for i in (1..ids.len()).rev() { let j = r.below(i + 1); ids.swap(i, j); }
    let mut o = out.lock().unwrap();
    for c in ids.chunks(lmax) { o.push(c.to_vec()); }
}

/// 非叶簇 `ids`（depth）→ 划分并递归；叶追加到 out
fn partition(qd: &Q8, ids: Vec<u32>, depth: usize, parent: usize, seed: u64, p: &GParams, out: &Mutex<Vec<Vec<u32>>>) {
    let s = ids.len();
    if depth > 0 && (depth - 1 > 8 || (depth - 1 > 6 && s as f64 > 0.8 * parent as f64)) {
        split_random(ids, seed ^ 0x5bd1e995, p.lmax, out); return;
    }
    let m = if depth == 0 { p.top.min(s) } else { ((s as f32 * p.rate) as usize).clamp(3, p.maxl).min(s) };
    let f = p.fanout(depth).min(m).min(16);
    let mut r = Rng::new(seed ^ ((depth as u64) << 40) ^ (s as u64));
    // 无放回均匀抽样 m 个领导者（部分 Fisher–Yates 于下标）
    let mut idx: Vec<u32> = (0..s as u32).collect();
    let mut lead = Vec::with_capacity(m);
    for i in 0..m { let j = i + r.below(s - i); idx.swap(i, j); lead.push(ids[idx[i] as usize]); }
    drop(idx);
    let asg = nearest_leaders(qd, &ids, &lead, f);
    let mut kids: Vec<Vec<u32>> = vec![Vec::new(); m];
    for (i, &u) in ids.iter().enumerate() { for k in 0..f { kids[asg[i * f + k] as usize].push(u); } }
    drop(asg);
    kids.retain(|k| !k.is_empty());
    for k in kids.iter_mut() { k.sort_unstable(); }
    merge_small(&mut kids, p.lmin, p.lmax);
    let (leaves, inner): (Vec<Vec<u32>>, Vec<Vec<u32>>) = kids.into_iter().partition(|k| k.len() <= p.lmax);
    { let mut o = out.lock().unwrap(); for l in leaves { if l.len() >= 2 { o.push(l); } } }
    inner.into_par_iter().enumerate().for_each(|(ci, kid)| {
        let sub = seed.wrapping_add((ci as u64 + 1).wrapping_mul(0x632BE59BD9B4E019));
        partition(qd, kid, depth + 1, s, sub, p, out);
    });
}

// ---------------------------------------------------------------- 叶内 top-k（双向）
/// 叶内每点最近 k 个（k≤4），返回 (局部下标, 距离) 列表，长度 s*k，不足以 u32::MAX 填
fn leaf_topk(qd: &Q8, leaf: &[u32], k: usize, buf: &mut Vec<i8>, top: &mut Vec<(i32, u32)>) {
    let s = leaf.len(); let dp = qd.dp; let sp = s.div_ceil(4) * 4;
    buf.clear(); buf.resize(sp * dp, 0);
    for (i, &u) in leaf.iter().enumerate() { unsafe { std::ptr::copy_nonoverlapping(qd.row(u), buf.as_mut_ptr().add(i * dp), dp); } }
    top.clear(); top.resize(s * k, (i32::MAX, u32::MAX));
    let nrm: Vec<i32> = leaf.iter().map(|&u| qd.nrm[u as usize]).collect();
    let sm: Vec<i32> = leaf.iter().map(|&u| qd.sum[u as usize]).collect();
    #[inline(always)]
    fn ins(t: &mut [(i32, u32)], d: i32, j: u32) {
        let k = t.len();
        if d < t[k - 1].0 || (d == t[k - 1].0 && j < t[k - 1].1) {
            let mut q = k - 1; while q > 0 && (t[q - 1].0 > d || (t[q - 1].0 == d && t[q - 1].1 > j)) { t[q] = t[q - 1]; q -= 1; }
            t[q] = (d, j);
        }
    }
    let mut tmp = [0i32; 16];
    let mut i = 0;
    while i < s {
        let a: [*const i8; 4] = std::array::from_fn(|r| unsafe { buf.as_ptr().add((i + r) * dp) });
        let mut j = i;
        while j < s {
            let b: [*const i8; 4] = std::array::from_fn(|t| unsafe { buf.as_ptr().add((j + t) * dp) });
            dot4x4(a, b, dp, &mut tmp);
            for r in 0..4 { let ii = i + r; if ii >= s { break; }
                for t in 0..4 { let jj = j + t; if jj >= s { break; } if jj <= ii { continue; }
                    let d = nrm[ii] + nrm[jj] - 2 * (tmp[r * 4 + t] - 128 * sm[jj]);
                    ins(&mut top[ii * k..(ii + 1) * k], d, jj as u32);
                    ins(&mut top[jj * k..(jj + 1) * k], d, ii as u32);
                }
            }
            j += 4;
        }
        i += 4;
    }
}


/// 叶内 top-2 面板核（GINGER ProcessLeafTop2 同构：16 个库点占 16 个 i32 lane、每 4 维一组交错，查询 4 字节广播，
/// 下三角只算一次，查询侧/库侧 top-2 同时更新；无水平归约）。结果写 top[i*2..i*2+2]（局部下标）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn leaf_top2_vnni(qd: &Q8, leaf: &[u32], pan: &mut Vec<u32>, aux: &mut Vec<i32>, top: &mut Vec<(i32, u32)>) {
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn upd(m1: &mut __m512i, x1: &mut __m512i, m2: &mut __m512i, x2: &mut __m512i, d: __m512i, idx: __m512i) {
        let lt1 = _mm512_cmplt_epi32_mask(d, *m1); let lt2 = _mm512_cmplt_epi32_mask(d, *m2);
        *m2 = _mm512_mask_blend_epi32(lt2, *m2, d); *x2 = _mm512_mask_blend_epi32(lt2, *x2, idx);
        *m2 = _mm512_mask_blend_epi32(lt1, *m2, *m1); *x2 = _mm512_mask_blend_epi32(lt1, *x2, *x1);
        *m1 = _mm512_mask_blend_epi32(lt1, *m1, d); *x1 = _mm512_mask_blend_epi32(lt1, *x1, idx);
    }
    let s = leaf.len(); let dp = qd.dp; let ng = dp / 4; let np = s.div_ceil(16);
    let pw = ng * 16; // u32 per panel
    pan.clear(); pan.resize((np + 1) * pw, 0x8080_8080);
    for (i, &u) in leaf.iter().enumerate() {
        let (pp, l) = (i / 16, i % 16);
        let r = qd.row(u) as *const u32;
        let dst = pan.as_mut_ptr().add(pp * pw + l);
        for g in 0..ng { *dst.add(g * 16) = r.add(g).read_unaligned() ^ 0x8080_8080; }
    }
    let n16 = (np + 1) * 16;
    aux.clear(); aux.resize(5 * n16, 0);
    for i in 0..s { aux[i] = qd.nrm[leaf[i] as usize]; }
    for i in 0..n16 { aux[n16 + i] = i32::MAX; aux[2 * n16 + i] = -1; aux[3 * n16 + i] = i32::MAX; aux[4 * n16 + i] = -1; }
    top.clear(); top.resize(s * 2, (i32::MAX, u32::MAX));
    let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let vmax = _mm512_set1_epi32(i32::MAX); let vneg = _mm512_set1_epi32(-1); let vs = _mm512_set1_epi32(s as i32);
    let pb = pan.as_ptr() as *const u8;
    let ap = aux.as_mut_ptr();
    let mut lanes_d = [0i32; 16]; let mut lanes_x = [0i32; 16];
    let mut i0 = 0;
    while i0 < s {
        let nq = (s - i0).min(4);
        let qrow: [*const i32; 4] = std::array::from_fn(|r| qd.row(leaf[i0 + r.min(nq - 1)]) as *const i32);
        let qn: [i32; 4] = std::array::from_fn(|r| qd.nrm[leaf[i0 + r.min(nq - 1)] as usize]);
        let qsc: [i32; 4] = std::array::from_fn(|r| 128 * qd.sum[leaf[i0 + r.min(nq - 1)] as usize]);
        let mut m1 = [vmax; 4]; let mut m2 = [vmax; 4]; let mut x1 = [vneg; 4]; let mut x2 = [vneg; 4];
        let mut p = i0 / 16;
        while p < np {
            let mut acc = [[_mm512_setzero_si512(); 2]; 4];
            let b0 = pb.add(p * pw * 4); let b1 = b0.add(pw * 4);
            for g in 0..ng {
                let v0 = _mm512_loadu_si512(b0.add(g * 64) as *const _); let v1 = _mm512_loadu_si512(b1.add(g * 64) as *const _);
                for r in 0..4 {
                    let qb = _mm512_set1_epi32(qrow[r].add(g).read_unaligned());
                    acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], v0, qb);
                    acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], v1, qb);
                }
            }
            for h in 0..2 {
                let pq = p + h; if pq >= np { break; }
                let jidx = _mm512_add_epi32(iota, _mm512_set1_epi32((pq * 16) as i32));
                let nv = _mm512_loadu_si512(ap.add(pq * 16) as *const _);
                let (a1, a2, a3, a4) = (ap.add(n16 + pq * 16), ap.add(2 * n16 + pq * 16), ap.add(3 * n16 + pq * 16), ap.add(4 * n16 + pq * 16));
                let mut d1 = _mm512_loadu_si512(a1 as *const _); let mut i1 = _mm512_loadu_si512(a2 as *const _);
                let mut d2 = _mm512_loadu_si512(a3 as *const _); let mut i2 = _mm512_loadu_si512(a4 as *const _);
                let inr = _mm512_cmplt_epi32_mask(jidx, vs);
                for r in 0..nq {
                    let qi = _mm512_set1_epi32((i0 + r) as i32);
                    let dot = _mm512_sub_epi32(acc[r][h], _mm512_set1_epi32(qsc[r]));
                    let d = _mm512_sub_epi32(_mm512_add_epi32(nv, _mm512_set1_epi32(qn[r])), _mm512_slli_epi32::<1>(dot));
                    let valid = inr & _mm512_cmpgt_epi32_mask(jidx, qi);
                    let d = _mm512_mask_blend_epi32(valid, vmax, d);
                    upd(&mut m1[r], &mut x1[r], &mut m2[r], &mut x2[r], d, jidx);
                    upd(&mut d1, &mut i1, &mut d2, &mut i2, d, qi);
                }
                _mm512_storeu_si512(a1 as *mut _, d1); _mm512_storeu_si512(a2 as *mut _, i1);
                _mm512_storeu_si512(a3 as *mut _, d2); _mm512_storeu_si512(a4 as *mut _, i2);
            }
            p += 2;
        }
        for r in 0..nq {
            let t = &mut top[(i0 + r) * 2..(i0 + r) * 2 + 2];
            for (mm, xx) in [(m1[r], x1[r]), (m2[r], x2[r])] {
                _mm512_storeu_si512(lanes_d.as_mut_ptr() as *mut _, mm); _mm512_storeu_si512(lanes_x.as_mut_ptr() as *mut _, xx);
                for l in 0..16 { if lanes_x[l] >= 0 { ins2(t, lanes_d[l], lanes_x[l] as u32); } }
            }
        }
        i0 += 4;
    }
    for i in 0..s {
        let t = &mut top[i * 2..i * 2 + 2];
        if aux[2 * n16 + i] >= 0 { ins2(t, aux[n16 + i], aux[2 * n16 + i] as u32); }
        if aux[4 * n16 + i] >= 0 { ins2(t, aux[3 * n16 + i], aux[4 * n16 + i] as u32); }
    }
}

/// 无提前退出的包含判断（可自动向量化）
#[inline(always)]
fn has(ids: &[u32], v: u32) -> bool {
    let mut f = false;
    let mut c = ids.chunks_exact(16);
    for ch in &mut c { let mut m = false; for &x in ch { m |= x == v; } f |= m; }
    for &x in c.remainder() { f |= x == v; }
    f
}

/// R98 H181：叶内 top-K 面板核（K=2/3/4 泛化 leaf_top2_vnni）。每 lane 保 K 个有序槽（插入网络，掩码先算后混合），
/// 查询侧/库侧同时更新；GK=3 不再落到通用 leaf_topk（H180：cpu_gemm 31→62 s）。结果写 top[i*K..i*K+K]。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn leaf_topk_vnni<const K: usize>(qd: &Q8, leaf: &[u32], pan: &mut Vec<u32>, aux: &mut Vec<i32>, top: &mut Vec<(i32, u32)>) {
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn updk<const K: usize>(m: &mut [__m512i; K], x: &mut [__m512i; K], d: __m512i, idx: __m512i) {
        let lt: [__mmask16; K] = std::array::from_fn(|k| _mm512_cmplt_epi32_mask(d, m[k]));
        let mut k = K - 1;
        while k > 0 {
            let nm = _mm512_mask_blend_epi32(lt[k - 1], d, m[k - 1]); let nx = _mm512_mask_blend_epi32(lt[k - 1], idx, x[k - 1]);
            m[k] = _mm512_mask_blend_epi32(lt[k], m[k], nm); x[k] = _mm512_mask_blend_epi32(lt[k], x[k], nx);
            k -= 1;
        }
        m[0] = _mm512_mask_blend_epi32(lt[0], m[0], d); x[0] = _mm512_mask_blend_epi32(lt[0], x[0], idx);
    }
    let s = leaf.len(); let dp = qd.dp; let ng = dp / 4; let np = s.div_ceil(16);
    let pw = ng * 16;
    pan.clear(); pan.resize((np + 1) * pw, 0x8080_8080);
    for (i, &u) in leaf.iter().enumerate() {
        let (pp, l) = (i / 16, i % 16);
        let r = qd.row(u) as *const u32;
        let dst = pan.as_mut_ptr().add(pp * pw + l);
        for g in 0..ng { *dst.add(g * 16) = r.add(g).read_unaligned() ^ 0x8080_8080; }
    }
    let n16 = (np + 1) * 16;
    aux.clear(); aux.resize((1 + 2 * K) * n16, 0);
    for i in 0..s { aux[i] = qd.nrm[leaf[i] as usize]; }
    for k in 0..K { for i in 0..n16 { aux[(1 + 2 * k) * n16 + i] = i32::MAX; aux[(2 + 2 * k) * n16 + i] = -1; } }
    top.clear(); top.resize(s * K, (i32::MAX, u32::MAX));
    let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let vmax = _mm512_set1_epi32(i32::MAX); let vneg = _mm512_set1_epi32(-1); let vs = _mm512_set1_epi32(s as i32);
    let pb = pan.as_ptr() as *const u8;
    let ap = aux.as_mut_ptr();
    let mut lanes_d = [0i32; 16]; let mut lanes_x = [0i32; 16];
    let mut i0 = 0;
    while i0 < s {
        let nq = (s - i0).min(4);
        let qrow: [*const i32; 4] = std::array::from_fn(|r| qd.row(leaf[i0 + r.min(nq - 1)]) as *const i32);
        let qn: [i32; 4] = std::array::from_fn(|r| qd.nrm[leaf[i0 + r.min(nq - 1)] as usize]);
        let qsc: [i32; 4] = std::array::from_fn(|r| 128 * qd.sum[leaf[i0 + r.min(nq - 1)] as usize]);
        let mut mq = [[vmax; K]; 4]; let mut xq = [[vneg; K]; 4];
        let mut p = i0 / 16;
        while p < np {
            let mut acc = [[_mm512_setzero_si512(); 2]; 4];
            let b0 = pb.add(p * pw * 4); let b1 = b0.add(pw * 4);
            for g in 0..ng {
                let v0 = _mm512_loadu_si512(b0.add(g * 64) as *const _); let v1 = _mm512_loadu_si512(b1.add(g * 64) as *const _);
                for r in 0..4 {
                    let qb = _mm512_set1_epi32(qrow[r].add(g).read_unaligned());
                    acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], v0, qb);
                    acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], v1, qb);
                }
            }
            for h in 0..2 {
                let pq = p + h; if pq >= np { break; }
                let jidx = _mm512_add_epi32(iota, _mm512_set1_epi32((pq * 16) as i32));
                let nv = _mm512_loadu_si512(ap.add(pq * 16) as *const _);
                let mut dm: [__m512i; K] = std::array::from_fn(|k| _mm512_loadu_si512(ap.add((1 + 2 * k) * n16 + pq * 16) as *const _));
                let mut dx: [__m512i; K] = std::array::from_fn(|k| _mm512_loadu_si512(ap.add((2 + 2 * k) * n16 + pq * 16) as *const _));
                let inr = _mm512_cmplt_epi32_mask(jidx, vs);
                for r in 0..nq {
                    let qi = _mm512_set1_epi32((i0 + r) as i32);
                    let dot = _mm512_sub_epi32(acc[r][h], _mm512_set1_epi32(qsc[r]));
                    let d = _mm512_sub_epi32(_mm512_add_epi32(nv, _mm512_set1_epi32(qn[r])), _mm512_slli_epi32::<1>(dot));
                    let valid = inr & _mm512_cmpgt_epi32_mask(jidx, qi);
                    let d = _mm512_mask_blend_epi32(valid, vmax, d);
                    updk::<K>(&mut mq[r], &mut xq[r], d, jidx);
                    updk::<K>(&mut dm, &mut dx, d, qi);
                }
                for k in 0..K {
                    _mm512_storeu_si512(ap.add((1 + 2 * k) * n16 + pq * 16) as *mut _, dm[k]);
                    _mm512_storeu_si512(ap.add((2 + 2 * k) * n16 + pq * 16) as *mut _, dx[k]);
                }
            }
            p += 2;
        }
        for r in 0..nq {
            let t = &mut top[(i0 + r) * K..(i0 + r) * K + K];
            for k in 0..K {
                _mm512_storeu_si512(lanes_d.as_mut_ptr() as *mut _, mq[r][k]); _mm512_storeu_si512(lanes_x.as_mut_ptr() as *mut _, xq[r][k]);
                for l in 0..16 { if lanes_x[l] >= 0 { insk(t, lanes_d[l], lanes_x[l] as u32); } }
            }
        }
        i0 += 4;
    }
    for i in 0..s {
        let t = &mut top[i * K..i * K + K];
        for k in 0..K { let x = aux[(2 + 2 * k) * n16 + i]; if x >= 0 { insk(t, aux[(1 + 2 * k) * n16 + i], x as u32); } }
    }
}

#[inline]
fn insk(t: &mut [(i32, u32)], d: i32, j: u32) {
    let k = t.len();
    if (d, j) >= t[k - 1] { return; }
    let mut q = k - 1; while q > 0 && t[q - 1] > (d, j) { t[q] = t[q - 1]; q -= 1; }
    t[q] = (d, j);
}

#[inline(always)]
fn ins2(t: &mut [(i32, u32)], d: i32, j: u32) {
    if (d, j) < t[1] { if (d, j) < t[0] { t[1] = t[0]; t[0] = (d, j); } else { t[1] = (d, j); } }
}

// ---------------------------------------------------------------- HashPrune（PiPNN 论文 / GINGER PRUNING_STRATEGY_HASH_PRUNE）
struct HashPrune { r: usize, b: usize, proj: Vec<f32>, res: Vec<Mutex<Vec<(i32, u32, u16)>>> }
impl HashPrune {
    fn new(x: &[f32], dim: usize, r: usize, b: usize, seed: u64) -> Self {
        let n = x.len() / dim;
        let mut rg = Rng::new(seed ^ 0x4A5);
        let pl: Vec<f32> = (0..b * dim).map(|_| { let mut s = 0f32; for _ in 0..12 { s += (rg.next() >> 40) as f32 / (1u64 << 24) as f32; } s - 6.0 }).collect();
        let mut proj = vec![0f32; n * b];
        proj.par_chunks_mut(b).enumerate().for_each(|(u, o)| { let xu = &x[u * dim..(u + 1) * dim];
            for k in 0..b { let p = &pl[k * dim..(k + 1) * dim]; let mut s = 0f32; for j in 0..dim { s += p[j] * xu[j]; } o[k] = s; } });
        HashPrune { r, b, proj, res: (0..n).map(|_| Mutex::new(Vec::new())).collect() }
    }
    #[inline] fn hash(&self, u: u32, v: u32) -> u16 {
        let (a, c) = (&self.proj[u as usize * self.b..(u as usize + 1) * self.b], &self.proj[v as usize * self.b..(v as usize + 1) * self.b]);
        let mut h = 0u16; for k in 0..self.b { h = (h << 1) | (a[k] > c[k]) as u16; } h
    }
    fn merge(&self, u: u32, edges: &[(i32, u32)]) {
        let mut t = self.res[u as usize].lock().unwrap();
        if t.capacity() == 0 { t.reserve_exact(self.r); }
        let worst = |t: &Vec<(i32, u32, u16)>| -> usize { let mut w = 0; for i in 1..t.len() { if (t[i].0, t[i].1) > (t[w].0, t[w].1) { w = i; } } w };
        for &(d, v) in edges {
            if v == u { continue; }
            let full = t.len() == self.r;
            let mut wi = usize::MAX;
            if full { wi = worst(&t); if (d, v) >= (t[wi].0, t[wi].1) { continue; } }
            let h = self.hash(u, v);
            if let Some(pos) = t.iter().position(|e| e.2 == h) {
                if (d, v) < (t[pos].0, t[pos].1) { t[pos].0 = d; t[pos].1 = v; }
                continue;
            }
            if !full { t.push((d, v, h)); } else { t[wi] = (d, v, h); }
        }
    }
}

// ---------------------------------------------------------------- RobustPrune
/// GINGER AcceleratedRobustPrune（keep_pruned=false）：cand=(d(u,c), c) 任意序 → 按 (d,id) 升序去重后贪心，
/// c 保留 ⇔ ∀ 已留 s：a2·d(s,c) ≥ d(u,c)；最多 maxk 个。返回保留的 (d, id)，按 d 升序。
/// R99 H198：剪枝前把候选行预取进缓存（收尾阶段随机行访问是延迟瓶颈，不是乘加瓶颈）。RAVEN_PIP_GPF=行数上限（0 关）。
#[inline(always)]
fn pf_rows(qd: &Q8, ids: impl Iterator<Item = u32>, lim: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let lines = (qd.dp + 63) / 64;
        for c in ids.take(lim) { let r = qd.row(c) as *const i8; for l in 0..lines { _mm_prefetch::<_MM_HINT_T0>(r.add(l * 64)); } }
    }
}
fn pf_lim() -> usize { static L: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *L.get_or_init(|| std::env::var("RAVEN_PIP_GPF").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) }

fn robust_prune_q(qd: &Q8, u: u32, cand: &mut Vec<(i32, u32)>, a2: f32, maxk: usize, keep: &mut Vec<(i32, u32)>) {
    cand.retain(|e| e.1 != u);
    cand.sort_unstable();
    cand.dedup_by_key(|e| e.1);
    { let l = pf_lim(); if l > 0 { pf_rows(qd, cand.iter().map(|e| e.1), l); } }
    // 去重后同 id 可能不相邻（不同距离）——距离来自同一函数故相同，相邻；保守再按 id 去重
    keep.clear();
    let dp = qd.dp;
    let mut sp: Vec<*const i8> = Vec::with_capacity(maxk + 4);
    let mut tmp = [0i32; 4];
    for &(du, c) in cand.iter() {
        if keep.len() >= maxk { break; }
        if keep.iter().any(|e| e.1 == c) { continue; }
        let cr = qd.row(c);
        let thr = du as f32;
        let mut ok = true;
        let mut si = 0;
        while si < keep.len() {
            let nb = (keep.len() - si).min(4);
            let b: [*const i8; 4] = std::array::from_fn(|t| sp[si + t.min(nb - 1)]);
            dot1x4(cr, b, dp, &mut tmp);
            let nc = qd.nrm[c as usize];
            // d(s,c) = ‖s‖²+‖c‖²−2(Σ(c⊕80)s − 128Σs)… 这里 a=c（异或侧）、b=s：dot = raw − 128·Σs
            for t in 0..nb {
                let s = keep[si + t].1 as usize;
                let dsc = qd.nrm[s] + nc - 2 * (tmp[t] - 128 * qd.sum[s]);
                if a2 * (dsc as f32) < thr { ok = false; break; }
            }
            if !ok { break; }
            si += 4;
        }
        if ok { keep.push((du, c)); sp.push(cr); }
    }
}

/// R98 H159 剪枝：ids/ds/cs 为缓冲（[0,nold) 是上次剪枝留下的边，两两已验证不互相遮挡）。
/// inc（cons=0）：与 robust_prune_q 逐位同输出，只是跳过「旧–旧」对；cons>0：先按距离序直接保留被 ≥cons 个叶拾取的边，
/// 再对其余边做 α 检查（对全部已保留边）。返回 (距离, id, 计数)。
fn prune_x(qd: &Q8, u: u32, ids: &[u32], ds: &[i32], cs: &[u8], nold: usize, p: &GParams, maxk: usize) -> Vec<(i32, u32, u8)> {
    let mut c: Vec<(i32, u32, u8, bool)> = (0..ids.len()).filter(|&t| ids[t] != u).map(|t| (ds[t], ids[t], cs[t], t < nold)).collect();
    c.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    c.dedup_by(|b, a| { if a.1 == b.1 { a.2 = a.2.max(b.2); a.3 &= b.3; true } else { false } });
    { let l = pf_lim(); if l > 0 { pf_rows(qd, c.iter().map(|e| e.1), l); } }
    let mut keep: Vec<(i32, u32, u8, bool)> = Vec::with_capacity(maxk + 1);
    let mut used = vec![false; c.len()];
    if p.cons > 0 {
        for (t, e) in c.iter().enumerate() { if keep.len() >= maxk { break; } if e.2 >= p.cons { keep.push(*e); used[t] = true; } }
        keep.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    }
    let inc = p.inc && p.cons == 0;
    let dp = qd.dp; let mut tmp = [0i32; 4]; let mut chk: Vec<usize> = Vec::with_capacity(maxk);
    for (t, e) in c.iter().enumerate() {
        if keep.len() >= maxk { break; }
        if used[t] { continue; }
        let (du, cid, _, old) = *e;
        chk.clear();
        for (j, s) in keep.iter().enumerate() { if !(inc && old && s.3) { chk.push(j); } }
        let cr = qd.row(cid); let nc = qd.nrm[cid as usize]; let thr = du as f32;
        let mut ok = true; let mut si = 0;
        while si < chk.len() {
            let nb = (chk.len() - si).min(4);
            let b: [*const i8; 4] = std::array::from_fn(|q| qd.row(keep[chk[si + q.min(nb - 1)]].1));
            dot1x4(cr, b, dp, &mut tmp);
            for q in 0..nb {
                let sx = keep[chk[si + q]].1 as usize;
                let dsc = qd.nrm[sx] + nc - 2 * (tmp[q] - 128 * qd.sum[sx]);
                if p.a2 * (dsc as f32) < thr { ok = false; break; }
            }
            if !ok { break; }
            si += 4;
        }
        if ok {
            if p.cons > 0 { let z = keep.partition_point(|s| (s.0, s.1) < (du, cid)); keep.insert(z, *e); } else { keep.push(*e); }
        }
    }
    keep.into_iter().map(|e| (e.0, e.1, e.2)).collect()
}

/// float 精确版（收尾可选 RAVEN_PIP_GF=1）
fn robust_prune_f(x: &[f32], dim: usize, u: u32, cand: &[u32], a2: f32, maxk: usize) -> Vec<(f32, u32)> {
    let xr = |i: u32| &x[i as usize * dim..(i as usize + 1) * dim];
    let mut l: Vec<(f32, u32)> = cand.iter().filter(|&&c| c != u).map(|&c| (df(xr(u), xr(c)), c)).collect();
    l.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    l.dedup_by_key(|e| e.1);
    let mut keep: Vec<(f32, u32)> = Vec::with_capacity(maxk);
    for &(du, c) in l.iter() {
        if keep.len() >= maxk { break; }
        if keep.iter().any(|e| e.1 == c) { continue; }
        if keep.iter().all(|&(_, s)| a2 * df(xr(s), xr(c)) >= du) { keep.push((du, c)); }
    }
    keep
}

#[inline(always)]
fn df(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let n = a.len() / 16 * 16;
    for c in (0..n).step_by(16) { for i in 0..16 { let d = a[c + i] - b[c + i]; acc[i] = d.mul_add(d, acc[i]); } }
    let mut s: f32 = acc.iter().sum();
    for i in n..a.len() { let d = a[i] - b[i]; s += d * d; }
    s
}

// ---------------------------------------------------------------- 原创：方向分桶保留长边（RAVEN_PIP_GDB）
/// 收尾 RNG 之后若度数 < cap：对被剪掉的候选按"相对 u 的方向"分桶（b 个随机超平面符号 → 2^b 桶，
/// 以 RNG 已留边的方向占据的桶为已覆盖），每个未覆盖桶按距离保留最近 dbk 个，追加到列表尾（不挤占 RNG 边）。
fn dir_bucket_fill(x: &[f32], dim: usize, planes: &[f32], b: usize, u: u32, kept: &[u32], pool: &[(i32, u32)], cap: usize, dbk: usize) -> Vec<u32> {
    let xr = |i: u32| &x[i as usize * dim..(i as usize + 1) * dim];
    let code = |v: u32| -> u32 {
        let (xu, xv) = (xr(u), xr(v)); let mut h = 0u32;
        for k in 0..b { let pl = &planes[k * dim..(k + 1) * dim]; let mut s = 0f32; for j in 0..dim { s += pl[j] * (xv[j] - xu[j]); } h = (h << 1) | (s > 0.0) as u32; }
        h
    };
    let mut out: Vec<u32> = kept.to_vec();
    let mut cnt = vec![0usize; 1 << b];
    for &v in kept { cnt[code(v) as usize] += dbk; }
    for &(_, v) in pool.iter() {
        if out.len() >= cap { break; }
        if v == u || out.contains(&v) { continue; }
        let h = code(v) as usize;
        if cnt[h] < dbk { cnt[h] += 1; out.push(v); }
    }
    out
}

// ---------------------------------------------------------------- 主流程
pub fn build(x: &[f32], dim: usize, p: &GParams) -> Vec<Vec<u32>> {
    let n = x.len() / dim;
    let t0 = Instant::now();
    if n < 2 { return vec![Vec::new(); n]; }
    let qd = Q8::new(x, dim);
    let t_q = t0.elapsed().as_secs_f64();
    let md = p.md;
    let otf_deg = ((md as f32 * p.otf) as usize).max(1);
    let thr = ((md as f32 * p.buf) as usize).max(md); // R99 H192：OTF 触发阈值
    let shsh = { let mut b = 0u32; while (n >> b) > 1024 { b += 1; } b }; // LSM 分片：src >> shsh < nsh
    let nsh = (n >> shsh) + 1;
    // 邻接缓冲：(整数距离, id)
    let adj: Vec<Mutex<(Vec<u32>, Vec<i32>, Vec<u8>, usize)>> = (0..n).map(|_| Mutex::new((Vec::with_capacity(md + 8), Vec::with_capacity(md + 8), Vec::with_capacity(md + 8), 0usize))).collect();
    let xp = p.cons > 0 || p.inc;
    let (mut t_part, mut t_leaf, mut nleaf, mut leafpts) = (0f64, 0f64, 0usize, 0usize);
    let nprune = std::sync::atomic::AtomicUsize::new(0);
    let nocc = std::sync::atomic::AtomicUsize::new(0);
    let ns_gemm = std::sync::atomic::AtomicU64::new(0); let ns_merge = std::sync::atomic::AtomicU64::new(0);
    let hpr: Option<HashPrune> = if p.hp { Some(HashPrune::new(x, dim, p.hr, p.hb, p.seed)) } else { None };
    for pass in 0..p.passes {
        let tc = Instant::now();
        let leaves: Mutex<Vec<Vec<u32>>> = Mutex::new(Vec::new());
        let all: Vec<u32> = (0..n as u32).collect();
        let seed = p.seed ^ (pass as u64 + 1).wrapping_mul(0xA24BAED4963EE407);
        if n <= p.lmax { leaves.lock().unwrap().push(all); } else { partition(&qd, all, 0, 0, seed, p, &leaves); }
        let leaves = leaves.into_inner().unwrap();
        if !p.dump.is_empty() { // H158 oracle：每叶 [len u32][ids u32...]，按 pass 追加
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(&p.dump).expect("dump"));
            for l in leaves.iter() { f.write_all(&(l.len() as u32).to_le_bytes()).unwrap(); for &v in l { f.write_all(&v.to_le_bytes()).unwrap(); } }
        }
        t_part += tc.elapsed().as_secs_f64();
        nleaf += leaves.len(); leafpts += leaves.iter().map(|l| l.len()).sum::<usize>();
        let tl = Instant::now();
        let k = p.k.min(4);
        if p.lsm > 0 {
            // R99 H192 LSM：叶任务只把 (src,d,dst) 追加进线程本地分片缓冲（按 src 高位分片），每批叶跑完后每片按 (src,dst)
            // 排序归约：一个点一次加锁、同批重复边合成一条并累计共识计数，缓冲超阈才剪。无逐边加锁、无逐边线性查重。
            let chunk = ((leaves.len() + p.lsm - 1) / p.lsm).max(1);
            for bl in leaves.chunks(chunk) {
                let accs: Vec<Vec<Vec<(u32, i32, u32)>>> = bl.par_iter().fold(
                    || (vec![Vec::<(u32, i32, u32)>::new(); nsh], Vec::<i8>::new(), Vec::<(i32, u32)>::new(), Vec::<u32>::new(), Vec::<i32>::new()),
                    |(mut acc, mut buf, mut top, mut pan, mut aux), leaf| {
                        let s = leaf.len();
                        if s <= 3 { return (acc, buf, top, pan, aux); }
                        let kk = k.min(s - 1);
                        let tg = Instant::now();
                        let mut done = false;
                        #[cfg(target_arch = "x86_64")]
                        { if p.panel && have_vnni() { match kk {
                            2 => { if p.pk2 { unsafe { leaf_topk_vnni::<2>(&qd, leaf, &mut pan, &mut aux, &mut top); } } else { unsafe { leaf_top2_vnni(&qd, leaf, &mut pan, &mut aux, &mut top); } } done = true; }
                            3 => { unsafe { leaf_topk_vnni::<3>(&qd, leaf, &mut pan, &mut aux, &mut top); } done = true; }
                            4 => { unsafe { leaf_topk_vnni::<4>(&qd, leaf, &mut pan, &mut aux, &mut top); } done = true; }
                            _ => {} } } }
                        if !done { leaf_topk(&qd, leaf, kk, &mut buf, &mut top); }
                        ns_gemm.fetch_add(tg.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        let tm = Instant::now();
                        for i in 0..s { for t in 0..kk { let (d, j) = top[i * kk + t]; if j == u32::MAX { continue; }
                            if p.occ > 0.0 && t >= p.occ_r {
                                let vj = leaf[j as usize];
                                if (0..t).any(|tb| { let b = top[i * kk + tb].1; b != u32::MAX && p.occ * (qd.d(leaf[b as usize], vj) as f32) < d as f32 }) { nocc.fetch_add(1, std::sync::atomic::Ordering::Relaxed); continue; }
                            }
                            let (a, b) = (leaf[i], leaf[j as usize]);
                            acc[(a >> shsh) as usize].push((a, d, b)); acc[(b >> shsh) as usize].push((b, d, a)); } }
                        ns_merge.fetch_add(tm.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        (acc, buf, top, pan, aux)
                    }).map(|t| t.0).collect();
                (0..nsh).into_par_iter().for_each_init(|| (Vec::<(u32, i32, u32)>::new(), Vec::<(i32, u32)>::new(), Vec::<(i32, u32)>::new()), |(all, cand, keep), sh| {
                    let tm = Instant::now();
                    all.clear();
                    for a in accs.iter() { all.extend_from_slice(&a[sh]); }
                    all.sort_unstable_by_key(|e| (e.0, e.2));
                    let mut i = 0;
                    while i < all.len() {
                        let u = all[i].0; let mut j = i; while j < all.len() && all[j].0 == u { j += 1; }
                        let mut g = adj[u as usize].lock().unwrap();
                        let (ids, ds, cs, nold) = &mut *g;
                        let mut add = 0; let mut q = i;
                        while q < j {
                            let (_, d, v) = all[q]; let mut c = 1u8; let mut r = q + 1;
                            while r < j && all[r].2 == v { c = c.saturating_add(1); r += 1; }
                            q = r;
                            if v == u { continue; }
                            if let Some(z) = ids.iter().position(|&x| x == v) { cs[z] = cs[z].saturating_add(c); continue; }
                            ids.push(v); ds.push(d); cs.push(c); add += 1;
                        }
                        if add > 0 && ids.len() > thr {
                            if xp {
                                let kx = prune_x(&qd, u, ids, ds, cs, *nold, p, otf_deg);
                                ids.clear(); ds.clear(); cs.clear(); for &(d, v, c) in kx.iter() { ids.push(v); ds.push(d); cs.push(c); }
                                *nold = ids.len();
                            } else {
                                cand.clear(); cand.extend(ds.iter().zip(ids.iter()).map(|(&d, &v)| (d, v)));
                                robust_prune_q(&qd, u, cand, p.a2, otf_deg, keep);
                                ids.clear(); ds.clear(); cs.clear(); for &(d, v) in keep.iter() { ids.push(v); ds.push(d); cs.push(1); }
                            }
                            nprune.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        i = j;
                    }
                    ns_merge.fetch_add(tm.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                });
            }
            t_leaf += tl.elapsed().as_secs_f64();
            continue;
        }
        leaves.par_iter().for_each_init(
            || (Vec::<i8>::new(), Vec::<(i32, u32)>::new(), Vec::<Vec<(i32, u32)>>::new(), Vec::<(i32, u32)>::new(), Vec::<(i32, u32)>::new(), Vec::<u32>::new(), Vec::<i32>::new()),
            |(buf, top, grp, cand, keep, pan, aux), leaf| {
                let s = leaf.len();
                if s <= 3 { return; } // GINGER ForEachLeafEdgeGrouped: cluster.size() <= 3 → 无边
                let kk = k.min(s - 1);
                let tg = Instant::now();
                let mut done = false;
                #[cfg(target_arch = "x86_64")]
                { if p.panel && have_vnni() { match kk {
                    2 => { if p.pk2 { unsafe { leaf_topk_vnni::<2>(&qd, leaf, pan, aux, top); } } else { unsafe { leaf_top2_vnni(&qd, leaf, pan, aux, top); } } done = true; }
                    3 => { unsafe { leaf_topk_vnni::<3>(&qd, leaf, pan, aux, top); } done = true; }
                    4 => { unsafe { leaf_topk_vnni::<4>(&qd, leaf, pan, aux, top); } done = true; }
                    _ => {} } } }
                if !done { let _ = (&pan, &aux); leaf_topk(&qd, leaf, kk, buf, top); }
                ns_gemm.fetch_add(tg.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                let tm = Instant::now();
                struct Acc<'a>(&'a std::sync::atomic::AtomicU64, Instant);
                impl Drop for Acc<'_> { fn drop(&mut self) { self.0.fetch_add(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); } }
                let _acc = Acc(&ns_merge, tm);
                if grp.len() < s { grp.resize(s, Vec::new()); }
                for g in grp.iter_mut().take(s) { g.clear(); }
                for i in 0..s { for t in 0..kk { let (d, j) = top[i * kk + t]; if j == u32::MAX { continue; }
                    // R98 H181 叶内 α-遮挡预剪：第 t(≥occ_r) 近邻若被 i 的更近叶内近邻 b 遮挡（occ·d(b,j) < d(i,j)）则不入 OTF 缓冲。
                    if p.occ > 0.0 && t >= p.occ_r {
                        let vj = leaf[j as usize];
                        if (0..t).any(|tb| { let b = top[i * kk + tb].1; b != u32::MAX && p.occ * (qd.d(leaf[b as usize], vj) as f32) < d as f32 }) { nocc.fetch_add(1, std::sync::atomic::Ordering::Relaxed); continue; }
                    }
                    grp[i].push((d, leaf[j as usize])); grp[j as usize].push((d, leaf[i])); } }
                if let Some(h) = hpr.as_ref() {
                    for i in 0..s { if !grp[i].is_empty() { h.merge(leaf[i], &grp[i]); } }
                    return;
                }
                for i in 0..s {
                    if grp[i].is_empty() { continue; }
                    let u = leaf[i];
                    let mut g = adj[u as usize].lock().unwrap();
                    let (ids, ds, cs, nold) = &mut *g;
                    let mut add = 0;
                    for &(d, v) in grp[i].iter() {
                        if v == u { continue; }
                        if p.cons > 0 { if let Some(z) = ids.iter().position(|&x| x == v) { cs[z] = cs[z].saturating_add(1); continue; } }
                        else if has(ids, v) { continue; }
                        ids.push(v); ds.push(d); cs.push(1); add += 1;
                    }
                    if add == 0 { continue; }
                    if ids.len() > thr {
                        if xp {
                            let kx = prune_x(&qd, u, ids, ds, cs, *nold, p, otf_deg);
                            ids.clear(); ds.clear(); cs.clear(); for &(d, v, c) in kx.iter() { ids.push(v); ds.push(d); cs.push(c); }
                            *nold = ids.len();
                        } else {
                            cand.clear(); cand.extend(ds.iter().zip(ids.iter()).map(|(&d, &v)| (d, v)));
                            robust_prune_q(&qd, u, cand, p.a2, otf_deg, keep);
                            ids.clear(); ds.clear(); cs.clear(); for &(d, v) in keep.iter() { ids.push(v); ds.push(d); cs.push(1); }
                        }
                        nprune.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        t_leaf += tl.elapsed().as_secs_f64();
    }
    let tf = Instant::now();
    let cap = p.cap;
    let planes: Vec<f32> = if p.db > 0 {
        let mut r = Rng::new(p.seed ^ 0xDB);
        (0..p.db * dim).map(|_| { // 近似高斯：12 个均匀和
            let mut s = 0f32; for _ in 0..12 { s += (r.next() >> 40) as f32 / (1u64 << 24) as f32; } s - 6.0 }).collect()
    } else { Vec::new() };
    let fin: Vec<Vec<u32>> = adj.into_par_iter().enumerate().map_init(|| (Vec::<(i32, u32)>::new(), Vec::<(i32, u32)>::new()), |(cand, keep), (u, m)| {
        let (bi, bd, bc, bn) = m.into_inner().unwrap();
        if xp && hpr.is_none() && !p.gf {
            let kx = prune_x(&qd, u as u32, &bi, &bd, &bc, bn, p, md);
            let mut out: Vec<u32> = kx.iter().map(|e| e.1).collect(); out.truncate(cap);
            if p.db > 0 && out.len() < cap {
                let mut pool: Vec<(i32, u32)> = bd.into_iter().zip(bi).collect(); pool.sort_unstable();
                out = dir_bucket_fill(x, dim, &planes, p.db, u as u32, &out, &pool, cap, p.dbk);
            }
            let _ = (&cand, &keep);
            return out;
        }
        let mut l: Vec<(i32, u32)> = bd.into_iter().zip(bi).collect();
        let u = u as u32;
        if let Some(h) = hpr.as_ref() { let mut t = h.res[u as usize].lock().unwrap(); l = t.iter().map(|e| (e.0, e.1)).collect(); t.clear(); t.shrink_to_fit(); }
        let mut out: Vec<u32> = if p.gf {
            let ids: Vec<u32> = l.iter().map(|e| e.1).collect();
            robust_prune_f(x, dim, u, &ids, p.a2, md).into_iter().map(|e| e.1).collect()
        } else {
            cand.clear(); cand.extend_from_slice(&l);
            robust_prune_q(&qd, u, cand, p.a2, md, keep);
            keep.iter().map(|e| e.1).collect()
        };
        out.truncate(cap);
        if p.db > 0 && out.len() < cap {
            let mut pool = l.clone(); pool.sort_unstable();
            out = dir_bucket_fill(x, dim, &planes, p.db, u, &out, &pool, cap, p.dbk);
        }
        out
    }).collect();
    let t_f = tf.elapsed().as_secs_f64();
    let deg: Vec<usize> = fin.iter().map(|l| l.len()).collect();
    let d2 = deg.iter().sum::<usize>() as f64 / n as f64;
    let mx = deg.iter().copied().max().unwrap_or(0);
    let mut indeg = vec![0u32; n]; for l in &fin { for &v in l { indeg[v as usize] += 1; } }
    let in0 = indeg.iter().filter(|&&d| d == 0).count();
    eprintln!("[pipg] RESULT n={n} dim={dim} threads={} vnni={} leaves={nleaf} dup={:.2} avgleaf={:.0} quant_s={t_q:.1} part_s={t_part:.1} leaf_s={t_leaf:.1} otf_prunes={} occ_skips={} cpu_gemm_s={:.1} cpu_merge_s={:.1} final_s={t_f:.1} TOTAL_s={:.1} avgdeg={d2:.2} maxdeg={mx} indeg0={in0} params={p:?}",
        rayon::current_num_threads(), have_vnni(), leafpts as f64 / n as f64, leafpts as f64 / nleaf.max(1) as f64,
        nprune.load(std::sync::atomic::Ordering::Relaxed), nocc.load(std::sync::atomic::Ordering::Relaxed), ns_gemm.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9, ns_merge.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9, t0.elapsed().as_secs_f64());
    fin
}
