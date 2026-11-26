//! R95 H148 MDB（Miss-Driven Bridges，漏检驱动桥边；缺省关，RAVEN_MDB_LEARN=<learn.fbin> 开启）。
//!
//! 面向 OOD 集（查询分布 ≠ 库分布，如 VIBE yandex 文本→图像）。原创做法（RoarGraph 只作参考：它把每个训练查询的
//! 近邻投影成二部图后全量重建；H121 是 top-10 两两共现补边）：
//! 1. 对每个训练查询（VIBE `fit_ood` 合法输入 = learn 查询，前 NL 行）在**当前图**上跑真实的 float 束搜索（ef=EFL），
//!    记下被扩展（出堆）的节点序列与结果集；
//! 2. 再以 ef=EFG 搜一次取 top-KM 作近似真值（自真值，免 learn×base 暴力；EFG=0 时不做，需外部真值，Rust 侧不支持）；
//! 3. 对每个**漏检**的近似真近邻 t：投一票给桥边 s→t，s = 被扩展节点中（MODE 含 E）与 t 最近的点
//!    ——"搜索本该在那里转弯"——以及/或者（MODE 含 R）已找到的真近邻中与 t 最近的点（答案集闭包）；
//! 4. 每个源点：候选按票数降序（≥MINV），只在新候选（含前几轮已加的桥边）之间做 α-RNG（A2=α²，平方 L2），
//!    每点累计至多 ADD 条、且不超过 r_max；ROUNDS>1 时在加边后的图上重搜，只修剩余漏检。
//! 5. 每行按 float 距离重排（RAVEN SD 读前 SD 槽）。
//!
//! 只有失败的搜索产生边，增量度数 ≤ ADD（缺省 8）。R109：数组路径（Python `Index.set_learn`/`build_ood`）见 [`augment_with`]，
//! 缺省 = [`MdbParams::array_default`]；文件路径环境变量：RAVEN_MDB_LEARN RAVEN_MDB_NL=200000 RAVEN_MDB_EFL=150
//! RAVEN_MDB_EFG=400 RAVEN_MDB_KM=100 RAVEN_MDB_ADD=8 RAVEN_MDB_MODE=ER RAVEN_MDB_MINV=1 RAVEN_MDB_A2=1.44
//! RAVEN_MDB_ROUNDS=1 RAVEN_MDB_NORM=1（learn 查询单位化；角度/余弦集，库向量已在外部单位化）。
use rayon::prelude::*;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

pub fn enabled() -> bool { std::env::var("RAVEN_MDB_LEARN").map(|v| !v.is_empty()).unwrap_or(false) }

#[inline(always)]
fn df(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let n = a.len() / 16 * 16;
    for c in (0..n).step_by(16) { for i in 0..16 { let d = a[c + i] - b[c + i]; acc[i] = d.mul_add(d, acc[i]); } }
    let mut s: f32 = acc.iter().sum();
    for i in n..a.len() { let d = a[i] - b[i]; s += d * d; }
    s
}

struct Ctx { vis: Vec<u32>, epoch: u32, pool: Vec<(f32, u32, bool)>, exp: Vec<u32> }

/// 经典 best-first 束搜索（float 平方 L2，池 = 有序数组，容量 ef）。返回池（按距离升序）；`exp` = 出堆扩展序列。
fn beam(x: &[f32], dim: usize, adj: &[Vec<u32>], ep: u32, q: &[f32], ef: usize, c: &mut Ctx) {
    let xr = |i: u32| &x[i as usize * dim..(i as usize + 1) * dim];
    beam_g(adj, ep, ef, c, |v| df(xr(v), q), |_| {});
}

/// R96-Q：int8 码 + 邻居预取版（RAVEN_MDB_Q8=1）
fn beam_g<D: Fn(u32) -> f32, P: Fn(u32)>(adj: &[Vec<u32>], ep: u32, ef: usize, c: &mut Ctx, dist: D, pf: P) {
    c.epoch = c.epoch.wrapping_add(1);
    if c.epoch == 0 { c.vis.iter_mut().for_each(|v| *v = 0); c.epoch = 1; }
    c.pool.clear(); c.exp.clear();
    c.vis[ep as usize] = c.epoch; c.pool.push((dist(ep), ep, false));
    let mut cur = 0usize;
    while cur < c.pool.len() {
        // 取池内最近的未扩展项（池有序，cur 之前都已扩展）
        let u = c.pool[cur].1; c.pool[cur].2 = true; c.exp.push(u);
        let mut best_new = usize::MAX;
        for &v in adj[u as usize].iter() { if c.vis[v as usize] != c.epoch { pf(v); } }
        if cur + 1 < c.pool.len() { if let Some(&w) = adj[c.pool[cur + 1].1 as usize].first() { pf(w); } }
        for &v in adj[u as usize].iter() {
            if c.vis[v as usize] == c.epoch { continue; }
            c.vis[v as usize] = c.epoch;
            let d = dist(v);
            if c.pool.len() >= ef && d >= c.pool[ef - 1].0 { continue; }
            let pos = c.pool.partition_point(|e| e.0 < d);
            c.pool.insert(pos, (d, v, false)); if c.pool.len() > ef { c.pool.pop(); }
            if pos < best_new { best_new = pos; }
        }
        if best_new <= cur { cur = best_new; }
        while cur < c.pool.len() && c.pool[cur].2 { cur += 1; }
    }
}

/// R109 vibe-ready：MDB 参数。`from_env()` = 文件路径（RAVEN_MDB_LEARN）的历史缺省（NL 20 万 EFL150 EFG400、f32 束），
/// `array_default()` = 数组路径（Python `Index.set_learn` / `build_ood`）的缺省 = yandex「便宜 MDB」档
/// （R99 y101 n5l / R105 y105 benv：Q8=1 NL=5 万 EFL=100 EFG=250 ROUNDS=1，其余同文件路径）。两者都以同名环境变量为覆盖。
#[derive(Debug, Clone, PartialEq)]
pub struct MdbParams {
    pub nl: usize, pub efl: usize, pub efg: usize, pub km: usize, pub add: usize, pub mode: String,
    pub minv: u32, pub a2: f32, pub rounds: usize, pub norm: bool, pub q8: bool,
}

impl MdbParams {
    /// 文件路径历史缺省（与 R95–R108 `RAVEN_MDB_LEARN` 行为逐位相同）
    pub fn file_default() -> Self {
        MdbParams { nl: 200_000, efl: 150, efg: 400, km: 100, add: 8, mode: "ER".to_string(), minv: 1, a2: 1.44, rounds: 1, norm: true, q8: false }
    }
    /// 数组路径缺省（yandex 便宜 MDB 档，见类型注释）
    pub fn array_default() -> Self {
        MdbParams { nl: 50_000, efl: 100, efg: 250, q8: true, ..Self::file_default() }
    }
    /// 以环境变量覆盖 `self` 的各字段（未设置的保持不变）
    pub fn env_override(mut self) -> Self {
        self.nl = env("RAVEN_MDB_NL", self.nl); self.efl = env("RAVEN_MDB_EFL", self.efl); self.efg = env("RAVEN_MDB_EFG", self.efg);
        self.km = env("RAVEN_MDB_KM", self.km); self.add = env("RAVEN_MDB_ADD", self.add); self.mode = env("RAVEN_MDB_MODE", self.mode.clone());
        self.minv = env("RAVEN_MDB_MINV", self.minv); self.a2 = env("RAVEN_MDB_A2", self.a2); self.rounds = env("RAVEN_MDB_ROUNDS", self.rounds);
        self.norm = env::<u32>("RAVEN_MDB_NORM", self.norm as u32) > 0; self.q8 = env::<u32>("RAVEN_MDB_Q8", self.q8 as u32) > 0;
        self
    }
    /// 文件路径实际使用的参数（历史缺省 + 环境变量）
    pub fn from_env() -> Self { Self::file_default().env_override() }
}

/// 文件路径入口（RAVEN_MDB_LEARN=<learn.fbin>，fbin = u32 行数、u32 维数、行主序 f32）。行为与 R95–R108 逐位相同。
pub fn augment(x: &[f32], dim: usize, adj: &mut [Vec<u32>], ep: u32, r_max: usize) {
    let t0 = Instant::now();
    let lf = std::env::var("RAVEN_MDB_LEARN").unwrap();
    let p = MdbParams::from_env();
    let b = std::fs::read(&lf).expect("RAVEN_MDB_LEARN read");
    let (ln, ld) = (u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize, u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize);
    assert_eq!(ld, dim, "learn dim mismatch");
    let nl = p.nl.min(ln).min((b.len() - 8) / (4 * dim));
    let lq: Vec<f32> = b[8..8 + nl * dim * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    drop(b);
    augment_core(x, dim, adj, ep, r_max, lq, &p, t0);
}

/// R109 数组路径入口：`learn` = 行主序 f32（行数 = learn.len()/dim），只取前 `p.nl` 行；其后与文件路径走同一段代码。
pub fn augment_with(x: &[f32], dim: usize, adj: &mut [Vec<u32>], ep: u32, r_max: usize, learn: &[f32], p: &MdbParams) {
    let t0 = Instant::now();
    assert!(dim > 0 && learn.len() % dim == 0, "learn length must be a multiple of dim");
    let nl = p.nl.min(learn.len() / dim);
    let lq: Vec<f32> = learn[..nl * dim].to_vec();
    augment_core(x, dim, adj, ep, r_max, lq, p, t0);
}

#[allow(clippy::too_many_arguments)]
fn augment_core(x: &[f32], dim: usize, adj: &mut [Vec<u32>], ep: u32, r_max: usize, mut lq: Vec<f32>, p: &MdbParams, t0: Instant) {
    let n = adj.len();
    let nl = lq.len() / dim;
    let (efl, efg, km, add, mode, minv, a2, rounds, norm) = (p.efl, p.efg, p.km, p.add, p.mode.clone(), p.minv, p.a2, p.rounds, p.norm);
    let (me, mr) = (mode.contains('E'), mode.contains('R'));
    if norm { lq.par_chunks_mut(dim).for_each(|v| { let s = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-30); v.iter_mut().for_each(|a| *a /= s); }); }
    let xr = |i: u32| &x[i as usize * dim..(i as usize + 1) * dim];
    let useq = p.q8;
    let tq = Instant::now();
    let q8 = if useq { Some(crate::build::pipg::Q8::new(x, dim)) } else { None };
    let lqc: Vec<(Vec<i8>, i32)> = match &q8 { Some(qd) => lq.par_chunks(dim).map(|v| qd.quant(v)).collect(), None => Vec::new() };
    if useq { eprintln!("[mdb] q8 codes {:.1}s", tq.elapsed().as_secs_f64()); }
    let q8r = q8.as_ref();
    let lqcr = &lqc;
    let run = |qi: usize, q: &[f32], ef: usize, c: &mut Ctx, g: &[Vec<u32>]| {
        match q8r { Some(qd) => { let (qr, qn) = &lqcr[qi]; beam_g(g, ep, ef, c, |v| qd.dq(qr, *qn, v) as f32, |v| qd.prefetch(v)); }
                    None => beam(x, dim, g, ep, q, ef, c) }
    };
    let deg0: Vec<usize> = adj.iter().map(|l| l.len()).collect();
    let mut nadd = vec![0u32; n];
    eprintln!("[mdb] n={n} nl={nl} efl={efl} efg={efg} km={km} add={add} mode={mode} minv={minv} a2={a2} rounds={rounds} r_max={r_max} load_s={:.1}", t0.elapsed().as_secs_f64());
    // R96：自真值只在第 0 轮算一次并缓存（后续轮复用；图只增边，自真值近似不变）
    let mut gtc: Vec<u32> = Vec::new();
    for rd in 0..rounds {
        let ts = Instant::now();
        if rd == 0 {
            let g: &[Vec<u32>] = adj;
            gtc = lq.par_chunks(dim).enumerate().map_init(
                || Ctx { vis: vec![0u32; n], epoch: 0, pool: Vec::with_capacity(efg.max(km) + 64), exp: Vec::with_capacity(4096) },
                |c, (qi, q)| { run(qi, q, efg.max(km), c, g); let mut v: Vec<u32> = c.pool.iter().take(km).map(|e| e.1).collect(); v.resize(km, u32::MAX); v })
                .flatten().collect();
            eprintln!("[mdb] self-gt ef{efg} km{km} gt_s={:.1}", ts.elapsed().as_secs_f64());
        }
        let g: &[Vec<u32>] = adj;
        let gtr: &[u32] = &gtc;
        // 每个 learn 查询 → (源, 目标) 票
        let (mut pairs, miss): (Vec<u64>, usize) = lq.par_chunks(dim).enumerate().map_init(
            || Ctx { vis: vec![0u32; n], epoch: 0, pool: Vec::with_capacity(efg.max(efl) + 64), exp: Vec::with_capacity(4096) },
            |c, (qi, q)| {
                let gt: Vec<u32> = gtr[qi * km..(qi + 1) * km].iter().copied().filter(|&t| t != u32::MAX).collect();
                run(qi, q, efl, c, g);
                let mut res: Vec<u32> = c.pool.iter().map(|e| e.1).collect(); res.sort_unstable();
                let found: Vec<u32> = gt.iter().copied().filter(|t| res.binary_search(t).is_ok()).collect();
                let mut out = Vec::new(); let mut m = 0usize;
                for &t in gt.iter() {
                    if res.binary_search(&t).is_ok() { continue; }
                    m += 1; let xt = xr(t);
                    let pick = |set: &[u32]| -> u32 { let mut bd = f32::MAX; let mut bs = u32::MAX;
                        match q8r { Some(qd) => { for &s in set { let d = qd.d(s, t) as f32; if d < bd { bd = d; bs = s; } } }
                                    None => { for &s in set { let d = df(xr(s), xt); if d < bd { bd = d; bs = s; } } } } bs };
                    if me || found.is_empty() { let s = pick(&c.exp); if s != t { out.push(((s as u64) << 32) | t as u64); } }
                    if mr && !found.is_empty() { let s = pick(&found); if s != t { out.push(((s as u64) << 32) | t as u64); } }
                }
                (out, m)
            }).reduce(|| (Vec::new(), 0), |mut a, b| { a.0.extend(b.0); (a.0, a.1 + b.1) });
        let t_s = ts.elapsed().as_secs_f64();
        pairs.par_sort_unstable();
        // 计票 → 每源点候选 (票数, 目标)
        let mut bysrc: Vec<(u32, Vec<(u32, u32)>)> = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let s = (pairs[i] >> 32) as u32; let mut l = Vec::new();
            while i < pairs.len() && (pairs[i] >> 32) as u32 == s {
                let p = pairs[i]; let mut c = 0u32; while i < pairs.len() && pairs[i] == p { c += 1; i += 1; }
                if c >= minv { l.push((c, p as u32)); }
            }
            if !l.is_empty() { bysrc.push((s, l)); }
        }
        drop(pairs);
        let nsrc = bysrc.len();
        let newe: Vec<(u32, Vec<u32>)> = bysrc.into_par_iter().filter_map(|(s, mut l)| {
            let su = s as usize; let k0 = g[su].len();
            let room = (add as isize - nadd[su] as isize).min(r_max as isize - k0 as isize);
            if room <= 0 { return None; }
            l.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut occ: Vec<u32> = g[su][deg0[su]..].to_vec(); let mut acc: Vec<u32> = Vec::new();
            for &(_, t) in l.iter() {
                if acc.len() as isize >= room { break; }
                if g[su].contains(&t) { continue; }
                let du = df(xr(s), xr(t));
                if occ.iter().any(|&w| a2 * df(xr(w), xr(t)) < du) { continue; }
                acc.push(t); occ.push(t);
            }
            if acc.is_empty() { None } else { Some((s, acc)) }
        }).collect();
        let mut added = 0usize;
        for (s, acc) in newe { added += acc.len(); nadd[s as usize] += acc.len() as u32; adj[s as usize].extend(acc); }
        eprintln!("[mdb] round {rd}: learn recall@{km}(vs ef{efg}) ef{efl} = {:.5} sources {nsrc} added {added} ({:.2}/node) search_s={t_s:.1} total_s={:.1}",
            1.0 - miss as f64 / (nl * km) as f64, added as f64 / n as f64, ts.elapsed().as_secs_f64());
    }
    adj.par_iter_mut().enumerate().for_each(|(u, l)| {
        let xu = &x[u * dim..(u + 1) * dim];
        let mut v: Vec<(f32, u32)> = l.iter().map(|&w| (df(xu, &x[w as usize * dim..(w as usize + 1) * dim]), w)).collect();
        v.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))); *l = v.into_iter().map(|e| e.1).collect();
    });
    let d = adj.iter().map(|l| l.len()).sum::<usize>() as f64 / n as f64;
    eprintln!("[mdb] RESULT total_added={} avgdeg={:.2} maxdeg={} MDB_s={:.1}", nadd.iter().map(|&a| a as usize).sum::<usize>(), d,
        adj.iter().map(|l| l.len()).max().unwrap_or(0), t0.elapsed().as_secs_f64());
}
