//! R94 分块多对多建图（RAVEN_BUILD=pipnn，缺省不变）。
//!
//! 动机（H146）：GINGER 的 PiPNN 图（均度 23–32）被 RAVEN 自己的搜索管线使用时 @0.99 比 RAVEN 自有图快 1.32–1.41×，
//! 且 R93 分解显示优势 = "便宜地得到高度、宽候选的图"，不是剪枝规则。本模块是 RAVEN 自己的实现（不链接、不移植对手代码）：
//! 1. **随机球划分**（重复 `REP` 次，种子不同）：在当前点集里随机取 m 个领导者，每点归到最近 f 个领导者（顶层 f=`FAN`，
//!    更深层 f=`FAN2`），子集 > `CMAX` 继续递归，直到叶 ≤ CMAX。叶间重叠（f>1、多次重复）给出跨块候选。
//! 2. **叶内多对多**：叶内向量拷到连续缓冲，4×4 寄存器分块 Gram 矩阵（float），d=‖a‖²+‖b‖²−2a·b；每点取叶内最近 `K` 个，
//!    正向边 u→v 与反向边 v→u 都进 u/v 的候选表（每点有界表，按距离保留最近 `LC` 个，去重）。
//! 3. **α-RNG 剪枝**到 `CAP`（float L2，候选已带距离）；可选第二步出∪入再剪枝（`RP`=1，α²=`A2B`、上限 `CAP2`）补回反向边。
//!
//! 环境变量：RAVEN_PIP_CMAX=1024 RAVEN_PIP_K=12 RAVEN_PIP_REP=2 RAVEN_PIP_FAN=2 RAVEN_PIP_FAN2=1 RAVEN_PIP_BMAX=64
//! RAVEN_PIP_LC=64 RAVEN_PIP_A2=(config α²) RAVEN_PIP_CAP=r_max RAVEN_PIP_RP=1 RAVEN_PIP_A2B=A2 RAVEN_PIP_CAP2=CAP RAVEN_PIP_CM=96
//! RAVEN_PIP_SEED=42。线程数 = rayon 当前线程池。
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// 是否启用。R99-13：缺省启用（RAVEN_BUILD 未设或 =pipnn）；RAVEN_BUILD=vamana（或任何其它值，如 nnd）关闭 → 旧缺省 vamana 建图。
pub fn enabled() -> bool {
    match std::env::var("RAVEN_BUILD") { Ok(v) => v.eq_ignore_ascii_case("pipnn"), Err(_) => true }
}

#[derive(Debug, Clone)]
pub struct PipParams {
    pub cmax: usize, pub k: usize, pub rep: usize, pub fan: usize, pub fan2: usize, pub fanl: usize, pub nn2: usize, pub lead: usize, pub bmax: usize,
    pub lc: usize, pub a2: f32, pub cap: usize, pub rp: bool, pub a2b: f32, pub cap2: usize, pub cm: usize, pub seed: u64,
}

impl PipParams {
    pub fn from_env(alpha: f32, r_max: usize) -> Self {
        let a2 = env("RAVEN_PIP_A2", alpha * alpha);
        let cap = env::<usize>("RAVEN_PIP_CAP", r_max).min(r_max);
        PipParams {
            cmax: env("RAVEN_PIP_CMAX", 1024usize).max(16), k: env("RAVEN_PIP_K", 12), rep: env("RAVEN_PIP_REP", 2).max(1),
            fan: env("RAVEN_PIP_FAN", 2).max(1), fan2: env("RAVEN_PIP_FAN2", 1).max(1), fanl: env("RAVEN_PIP_FANL", 1).clamp(1, 4), nn2: env("RAVEN_PIP_NN2", 0), lead: env("RAVEN_PIP_LEAD", 0), bmax: env("RAVEN_PIP_BMAX", 64usize).max(2),
            lc: env("RAVEN_PIP_LC", 64), a2, cap, rp: env::<usize>("RAVEN_PIP_RP", 0) > 0, a2b: env("RAVEN_PIP_A2B", a2),
            cap2: env::<usize>("RAVEN_PIP_CAP2", cap).min(r_max), cm: env("RAVEN_PIP_CM", 96), seed: env("RAVEN_PIP_SEED", 42),
        }
    }
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03 | 1) }
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
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

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    for c in (0..a.len()).step_by(16) { for i in 0..16 { acc[i] = a[c + i].mul_add(b[c + i], acc[i]); } }
    acc.iter().sum()
}

const HAVE512: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx512f"));

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn k4x4_512(b: *const f32, i: usize, j: usize, dp: usize, s: usize, g: *mut f32) {
    use std::arch::x86_64::*;
    let mut acc = [_mm512_setzero_ps(); 16];
    let (a0, a1, a2, a3) = (b.add(i * dp), b.add((i + 1) * dp), b.add((i + 2) * dp), b.add((i + 3) * dp));
    let (b0, b1, b2, b3) = (b.add(j * dp), b.add((j + 1) * dp), b.add((j + 2) * dp), b.add((j + 3) * dp));
    let mut c = 0;
    while c < dp {
        let x = [_mm512_loadu_ps(a0.add(c)), _mm512_loadu_ps(a1.add(c)), _mm512_loadu_ps(a2.add(c)), _mm512_loadu_ps(a3.add(c))];
        let y = [_mm512_loadu_ps(b0.add(c)), _mm512_loadu_ps(b1.add(c)), _mm512_loadu_ps(b2.add(c)), _mm512_loadu_ps(b3.add(c))];
        for r in 0..4 { for t in 0..4 { acc[r * 4 + t] = _mm512_fmadd_ps(x[r], y[t], acc[r * 4 + t]); } }
        c += 16;
    }
    for r in 0..4 { for t in 0..4 { *g.add((i + r) * s + j + t) = _mm512_reduce_add_ps(acc[r * 4 + t]); } }
}

/// 叶内 Gram（buf: s×dp，dp 为 16 的倍数，补零）。只算上三角 4×4 块再镜像。
fn gram(buf: &[f32], s: usize, dp: usize, g: &mut [f32]) {
    let row = |i: usize| &buf[i * dp..(i + 1) * dp];
    let mut i = 0;
    while i < s {
        let mut j = i;
        while j < s {
            if i + 4 <= s && j + 4 <= s && HAVE512 {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                unsafe { k4x4_512(buf.as_ptr(), i, j, dp, s, g.as_mut_ptr()); }
            } else if i + 4 <= s && j + 4 <= s {
                let (a0, a1, a2, a3) = (row(i), row(i + 1), row(i + 2), row(i + 3));
                let (b0, b1, b2, b3) = (row(j), row(j + 1), row(j + 2), row(j + 3));
                let mut acc = [[0f32; 16]; 16];
                for c in (0..dp).step_by(16) {
                    let a = [&a0[c..c + 16], &a1[c..c + 16], &a2[c..c + 16], &a3[c..c + 16]];
                    let b = [&b0[c..c + 16], &b1[c..c + 16], &b2[c..c + 16], &b3[c..c + 16]];
                    for r in 0..4 { for t in 0..4 { let o = &mut acc[r * 4 + t]; for l in 0..16 { o[l] = a[r][l].mul_add(b[t][l], o[l]); } } }
                }
                for r in 0..4 { for t in 0..4 { g[(i + r) * s + j + t] = acc[r * 4 + t].iter().sum(); } }
            } else {
                for r in i..(i + 4).min(s) { for t in j..(j + 4).min(s) { g[r * s + t] = dot(row(r), row(t)); } }
            }
            j += 4;
        }
        i += 4;
    }
    for r in 0..s { for t in 0..r { g[r * s + t] = g[t * s + r]; } }
}

/// 递归随机球划分：把 ids 切成 ≤ cmax 的叶（追加到 out）。
fn carve(x: &[f32], dim: usize, ids: Vec<u32>, depth: usize, seed: u64, p: &PipParams, out: &Mutex<Vec<Vec<u32>>>, cand: &[Mutex<Vec<(f32, u32)>>]) {
    let s = ids.len();
    if s <= p.cmax {
        if s >= 2 { out.lock().unwrap().push(ids); }
        return;
    }
    if depth > 16 { // 退化保护：随机切块
        for c in ids.chunks(p.cmax) { if c.len() >= 2 { out.lock().unwrap().push(c.to_vec()); } }
        return;
    }
    // 叶层（子集将成为叶）用 FANL 重叠，其余层 FAN（顶层）/FAN2；叶层按 f·s/(0.75·cmax) 取领导者数，使子集均值 ≈0.75·cmax
    let leaf_level = s.div_ceil(p.cmax) <= p.bmax && p.fanl > 1;
    let (m, f) = if leaf_level {
        let m = (p.fanl * s).div_ceil(p.cmax * 3 / 4).clamp(2, p.bmax * p.fanl);
        (m, p.fanl.min(m))
    } else {
        let m = s.div_ceil(p.cmax).clamp(2, p.bmax);
        (m, if depth == 0 { p.fan } else { p.fan2 }.min(m).min(4))
    };
    let mut r = Rng::new(seed ^ ((depth as u64) << 48) ^ s as u64);
    let mut lead: Vec<u32> = Vec::with_capacity(m);
    while lead.len() < m { let v = ids[r.below(s)]; if !lead.contains(&v) { lead.push(v); } }
    let xr = |i: u32| &x[i as usize * dim..(i as usize + 1) * dim];
    // 每点最近 f 个领导者
    let asg: Vec<([u16; 4], f32)> = ids.par_chunks(2048).flat_map_iter(|ch| {
        ch.iter().map(|&u| {
            let mut best = [(f32::INFINITY, 0u16); 4];
            for (li, &l) in lead.iter().enumerate() {
                let d = df(xr(u), xr(l));
                if d < best[f - 1].0 {
                    let mut k = f - 1; while k > 0 && best[k - 1].0 > d { best[k] = best[k - 1]; k -= 1; }
                    best[k] = (d, li as u16);
                }
            }
            let mut o = [u16::MAX; 4]; for k in 0..f { o[k] = best[k].1; } (o, best[0].0)
        }).collect::<Vec<_>>()
    }).collect();
    let mut kids: Vec<Vec<u32>> = vec![Vec::new(); m];
    for (i, (a, _)) in asg.iter().enumerate() { for &li in a.iter().take(f) { if li != u16::MAX { kids[li as usize].push(ids[i]); } } }
    // RAVEN_PIP_LEAD=1：非叶层每点加一条到最近领导者的候选边（u→leader；=2 再加反向 leader→u），给叶内 kNN 补粗尺度长边
    if p.lead > 0 && !(leaf_level) {
        let lc = p.lc.max(p.cap);
        ids.par_iter().zip(asg.par_iter()).for_each(|(&u, &(a, d))| {
            let l0 = lead[a[0] as usize]; if l0 == u { return; }
            { let mut l = cand[u as usize].lock().unwrap(); l.push((d, l0)); if l.len() >= lc * 2 { compact(&mut l, lc); } }
            if p.lead > 1 { let mut l = cand[l0 as usize].lock().unwrap(); l.push((d, u)); if l.len() >= lc * 2 { compact(&mut l, lc); } }
        });
    }
    drop(asg);
    let s0 = s;
    drop(ids);
    kids.into_par_iter().enumerate().for_each(|(ci, kid)| {
        let sub = seed.wrapping_add((ci as u64 + 1).wrapping_mul(0x632BE59BD9B4E019));
        if kid.len() >= s0 { // 无进展：随机切块
            for c in kid.chunks(p.cmax) { if c.len() >= 2 { out.lock().unwrap().push(c.to_vec()); } }
        } else {
            carve(x, dim, kid, depth + 1, sub, p, out, cand);
        }
    });
}

#[inline]
fn compact(l: &mut Vec<(f32, u32)>, lc: usize) {
    l.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    l.dedup_by_key(|e| e.1);
    l.truncate(lc);
}

/// 返回每点邻接（按 float 距离升序）。
pub fn build(x: &[f32], dim: usize, p: &PipParams) -> Vec<Vec<u32>> {
    let n = x.len() / dim;
    let t0 = Instant::now();
    if n < 2 { return vec![Vec::new(); n]; }
    let dp = dim.div_ceil(16) * 16;
    let lc = p.lc.max(p.cap);
    let cand: Vec<Mutex<Vec<(f32, u32)>>> = (0..n).map(|_| Mutex::new(Vec::with_capacity(lc / 2))).collect();
    let xr = |i: usize| &x[i * dim..(i + 1) * dim];
    let (mut t_carve, mut t_leaf, mut nleaf, mut leafpts) = (0f64, 0f64, 0usize, 0usize);
    for rep in 0..p.rep {
        let tc = Instant::now();
        let leaves: Mutex<Vec<Vec<u32>>> = Mutex::new(Vec::new());
        carve(x, dim, (0..n as u32).collect(), 0, p.seed ^ (rep as u64 + 1).wrapping_mul(0xA24BAED4963EE407), p, &leaves, &cand);
        let leaves = leaves.into_inner().unwrap();
        t_carve += tc.elapsed().as_secs_f64();
        nleaf += leaves.len(); leafpts += leaves.iter().map(|l| l.len()).sum::<usize>();
        let tl = Instant::now();
        leaves.par_iter().for_each_init(|| (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<(f32, u32)>::new()), |(buf, g, row), leaf| {
            let s = leaf.len();
            buf.clear(); buf.resize(s * dp, 0.0);
            for (i, &u) in leaf.iter().enumerate() { buf[i * dp..i * dp + dim].copy_from_slice(xr(u as usize)); }
            g.clear(); g.resize(s * s, 0.0);
            gram(buf, s, dp, g);
            let k = p.k.min(s - 1);
            let nrm: Vec<f32> = (0..s).map(|i| g[i * s + i]).collect();
            for i in 0..s {
                row.clear();
                for j in 0..s { if j != i { row.push(((nrm[i] + nrm[j] - 2.0 * g[i * s + j]).max(0.0), j as u32)); } }
                if row.len() > k { row.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0)); row.truncate(k); }
                let u = leaf[i];
                {
                    let mut l = cand[u as usize].lock().unwrap();
                    for &(d, j) in row.iter() { l.push((d, leaf[j as usize])); }
                    if l.len() >= lc * 2 { compact(&mut l, lc); }
                }
                for &(d, j) in row.iter() {
                    let mut l = cand[leaf[j as usize] as usize].lock().unwrap();
                    l.push((d, u)); if l.len() >= lc * 2 { compact(&mut l, lc); }
                }
            }
        });
        t_leaf += tl.elapsed().as_secs_f64();
    }
    let tp = Instant::now();
    let (a2, cap) = (p.a2, p.cap);
    let mut lists: Vec<Vec<(f32, u32)>> = cand.into_par_iter().enumerate().map(|(u, m)| {
        let mut l = m.into_inner().unwrap(); l.retain(|e| e.1 as usize != u); compact(&mut l, lc); l
    }).collect();
    // RAVEN_PIP_NN2=m>0：一轮"邻居的邻居"精化（候选 ∪ 前 m 个候选各自的前 m 个，精确距离），补叶边界处漏掉的近邻
    let mut t_nn2 = 0f64;
    if p.nn2 > 0 {
        let tn = Instant::now(); let m2 = p.nn2;
        let ls = &lists;
        let nl: Vec<Vec<(f32, u32)>> = (0..n).into_par_iter().map(|u| {
            let mut l = ls[u].clone();
            let mut seen: Vec<u32> = l.iter().map(|e| e.1).collect(); seen.sort_unstable();
            for &(_, v) in ls[u].iter().take(m2) {
                for &(_, w) in ls[v as usize].iter().take(m2) {
                    if w as usize == u || seen.binary_search(&w).is_ok() { continue; }
                    let pos = seen.partition_point(|&z| z < w); seen.insert(pos, w);
                    l.push((df(xr(u), xr(w as usize)), w));
                }
            }
            compact(&mut l, lc); l
        }).collect();
        lists = nl; t_nn2 = tn.elapsed().as_secs_f64();
    }
    // α-RNG：候选向量先拷进线程局部连续缓冲（随机读一次），候选间距离在缓冲上算（L1/L2 命中）
    let mut adj: Vec<Vec<u32>> = lists.into_par_iter().map_init(|| (Vec::<f32>::new(), Vec::<usize>::new()), |(buf, keep), l| {
        buf.clear(); buf.resize(l.len() * dp, 0.0); keep.clear();
        for (i, &(_, v)) in l.iter().enumerate() { buf[i * dp..i * dp + dim].copy_from_slice(xr(v as usize)); }
        let br = |i: usize| &buf[i * dp..(i + 1) * dp];
        for (i, &(du, _)) in l.iter().enumerate() {
            if keep.iter().all(|&w| a2 * df(br(w), br(i)) > du) { keep.push(i); if keep.len() >= cap { break; } }
        }
        keep.iter().map(|&i| l[i].1).collect()
    }).collect();
    let t_p1 = tp.elapsed().as_secs_f64();
    let d1 = adj.iter().map(|l| l.len()).sum::<usize>() as f64 / n as f64;
    let tr = Instant::now();
    if p.rp {
        adj = crate::build::nnd::prune_out_in(x, dim, &adj, p.a2b, p.cm, p.cap2, rayon::current_num_threads());
    }
    let t_rp = tr.elapsed().as_secs_f64();
    let deg: Vec<usize> = adj.iter().map(|l| l.len()).collect();
    let d2 = deg.iter().sum::<usize>() as f64 / n as f64;
    let mx = deg.iter().copied().max().unwrap_or(0);
    let mut indeg = vec![0u32; n]; for l in &adj { for &v in l { indeg[v as usize] += 1; } }
    let in0 = indeg.iter().filter(|&&d| d == 0).count();
    eprintln!("[pipnn] RESULT n={n} dim={dim} params={p:?} threads={} leaves={nleaf} dup={:.2} carve_s={t_carve:.1} leaf_s={t_leaf:.1} nn2_s={t_nn2:.1} prune_s={:.1} (avgdeg {d1:.2}) reprune_s={t_rp:.1} TOTAL_s={:.1} avgdeg={d2:.2} maxdeg={mx} indeg0={in0}",
        rayon::current_num_threads(), leafpts as f64 / n as f64, t_p1 - t_nn2, t0.elapsed().as_secs_f64());
    adj
}
