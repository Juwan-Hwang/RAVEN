//! R88 码空间建图（R87-D oracle 的库内实现）：SQ4 码空间 NN-descent → 出∪入 kNN 候选上一次 α-RNG 剪枝（float L2）。
//!
//! 开关：`RAVEN_BUILD=nnd`（缺省不变 = Vamana）。参数（环境变量，缺省值 = R87-D 过门配置）：
//! - `RAVEN_NND_K`=40 kNN 列表长，`RAVEN_NND_S`=20 每轮 new/old 采样数，`RAVEN_NND_ITERS`=20，`RAVEN_NND_DELTA`=2e-4（更新数 < δ·N·K 停）
//! - `RAVEN_NND_A2`=α²（缺省 config.alpha²），`RAVEN_NND_CAP`=r_max，`RAVEN_NND_CM`=96（剪枝前按 float 距离取最近 CM 个候选）
//! - `RAVEN_NND_SEED`=42，线程数 = rayon 当前线程数
//! - `RAVEN_NND_LR`=0：补救项（R88-3a），见 `long_range_edges`；`RAVEN_NND_RIN`=0：补救项（R88-3b）入度下限，见 build 末尾。
//!
//! SQ4：逐维 offset = 2 万样本的 p0.1，全局单一步长 max_j(p99.9−p0.1)/15，使码空间 L2 ∝ float L2；码不打包（1 字节/级），
//! 长度补齐到 32 的倍数，距离用 u16 累加（15²·len/32 < 65536 要求 dim ≤ 9300）。
//! 输入向量须已是建图度量下的向量（cosine 数据由调用方先归一化，与 Vamana 路径相同）。
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// 是否启用码空间建图（RAVEN_BUILD=nnd）
pub fn enabled() -> bool {
    std::env::var("RAVEN_BUILD").map(|v| v.eq_ignore_ascii_case("nnd")).unwrap_or(false)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

#[inline(always)]
fn dcode(a: &[u8], b: &[u8]) -> u32 {
    let mut acc = [0u16; 32];
    for (x, y) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        for i in 0..32 { let d = x[i].abs_diff(y[i]) as u16; acc[i] += d * d; }
    }
    acc.iter().map(|&v| v as u32).sum()
}

#[inline(always)]
fn df(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let n = a.len() / 16 * 16;
    for c in (0..n).step_by(16) { for i in 0..16 { let d = a[c + i] - b[c + i]; acc[i] += d * d; } }
    let mut s: f32 = acc.iter().sum();
    for i in n..a.len() { let d = a[i] - b[i]; s += d * d; }
    s
}

#[derive(Clone, Copy)]
struct Nb { d: u32, id: u32, new: bool }

fn par<F: Fn(usize, usize) + Sync>(n: usize, th: usize, chunk: usize, f: F) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..th.max(1) {
            let (next, f) = (&next, &f);
            s.spawn(move || loop {
                let st = next.fetch_add(chunk, Ordering::Relaxed);
                if st >= n { break; }
                f(st, (st + chunk).min(n));
            });
        }
    });
}

/// 构建参数
#[derive(Debug, Clone)]
pub struct NndParams { pub k: usize, pub s: usize, pub iters: usize, pub delta: f64, pub a2: f32, pub cm: usize, pub cap: usize, pub seed: u64, pub threads: usize, pub lr: usize, pub r_max: usize }

impl NndParams {
    pub fn from_env(alpha: f32, r_max: usize) -> Self {
        NndParams {
            k: env("RAVEN_NND_K", 40), s: env("RAVEN_NND_S", 20), iters: env("RAVEN_NND_ITERS", 20),
            delta: env("RAVEN_NND_DELTA", 2e-4), a2: env("RAVEN_NND_A2", alpha * alpha), cm: env("RAVEN_NND_CM", 96),
            cap: env::<usize>("RAVEN_NND_CAP", r_max).min(r_max), seed: env("RAVEN_NND_SEED", 42),
            threads: rayon::current_num_threads(), lr: env("RAVEN_NND_LR", 0), r_max,
        }
    }
}

/// 返回每点的剪枝后邻接（按 float 距离升序，长度 ≤ cap），以及码空间 kNN 前 `hk` 个（扁平 n×hk，不足补 u32::MAX；hk=0 时为空）
pub fn build(x: &[f32], dim: usize, p: &NndParams, hk: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
    let n = x.len() / dim;
    let (k, s_, th, seed) = (p.k.min(n.saturating_sub(1)).max(1), p.s, p.threads, p.seed);
    let t0 = Instant::now();
    if n < 2 { return (vec![Vec::new(); n], Vec::new()); }
    // ---- SQ4（全局步长）
    let dp = dim.div_ceil(32) * 32;
    let mut lo = vec![0f32; dim]; let mut hi = vec![0f32; dim];
    {
        let ns = n.min(20000); let stp = n / ns;
        for j in 0..dim {
            let mut col: Vec<f32> = (0..ns).map(|i| x[i * stp * dim + j]).collect();
            col.sort_by(|a, b| a.total_cmp(b));
            lo[j] = col[ns / 1000]; hi[j] = col[ns - 1 - ns / 1000];
        }
    }
    let step = ((0..dim).map(|j| hi[j] - lo[j]).fold(0f32, f32::max) / 15.0).max(1e-30);
    let mut code = vec![0u8; n * dp];
    {
        let cp = code.as_mut_ptr() as usize;
        par(n, th, 4096, |st, en| { for i in st..en { for j in 0..dim {
            let q = ((x[i * dim + j] - lo[j]) / step + 0.5).floor().clamp(0.0, 15.0) as u8;
            unsafe { *(cp as *mut u8).add(i * dp + j) = q; } } } });
    }
    let tq = t0.elapsed().as_secs_f64();
    let cd = |i: usize| &code[i * dp..(i + 1) * dp];
    // ---- NN-descent（Dong et al.：new/old 标记、反向采样、逐点互斥列表）
    let lists: Vec<Mutex<Vec<Nb>>> = (0..n).map(|_| Mutex::new(Vec::with_capacity(k + 1))).collect();
    par(n, th, 4096, |st, en| {
        let mut r = Rng(seed ^ (st as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        for u in st..en {
            let mut l = lists[u].lock().unwrap();
            while l.len() < k {
                let v = r.below(n) as u32; if v as usize == u || l.iter().any(|e| e.id == v) { continue; }
                l.push(Nb { d: dcode(cd(u), cd(v as usize)), id: v, new: true });
            }
            l.sort_by_key(|e| e.d);
        }
    });
    let ndist = AtomicU64::new(0);
    // R90-3（RAVEN_NND_FAST=1 阈值；=2 再加块化码缓冲；缺省关）：每点"当前第 k 名距离"原子阈值，局部 join 先无锁比阈值，
    // 只有可能入表的提议才加锁（R88 实测 ≈90% 提议被拒，每对 2 次加锁是主成本）；候选码块化拷到线程局部缓冲。
    let fastm: u32 = env("RAVEN_NND_FAST", 0); let fast = fastm > 0; let blk = fastm >= 2;
    let worst: Vec<AtomicU32> = (0..n).map(|u| { let l = lists[u].lock().unwrap(); AtomicU32::new(if l.len() >= k { l[l.len() - 1].d } else { u32::MAX }) }).collect();
    let insert = |u: usize, v: u32, d: u32| -> u64 {
        if fast && d >= worst[u].load(Ordering::Relaxed) { return 0; }
        let mut l = lists[u].lock().unwrap();
        if l.len() >= k && d >= l[l.len() - 1].d { return 0; }
        if l.iter().any(|e| e.id == v) { return 0; }
        let pos = l.partition_point(|e| e.d <= d);
        l.insert(pos, Nb { d, id: v, new: true }); if l.len() > k { l.pop(); }
        if fast && l.len() >= k { worst[u].store(l[l.len() - 1].d, Ordering::Relaxed); }
        1
    };
    let rev = |f: &Vec<Vec<u32>>| -> (Vec<usize>, Vec<u32>) {
        let mut cnt = vec![0usize; n + 1];
        for l in f { for &v in l { cnt[v as usize + 1] += 1; } }
        for i in 0..n { cnt[i + 1] += cnt[i]; }
        let mut pos = cnt.clone(); let mut ids = vec![0u32; cnt[n]];
        for (u, l) in f.iter().enumerate() { for &v in l { ids[pos[v as usize]] = u as u32; pos[v as usize] += 1; } }
        (cnt, ids)
    };
    let mut iters_done = 0;
    for it in 0..p.iters {
        let ti = Instant::now();
        let mut fnew: Vec<Vec<u32>> = vec![Vec::new(); n]; let mut fold: Vec<Vec<u32>> = vec![Vec::new(); n];
        {
            let (pn, po) = (fnew.as_mut_ptr() as usize, fold.as_mut_ptr() as usize);
            par(n, th, 4096, |st, en| {
                let mut r = Rng(seed ^ ((it as u64) << 40) ^ (st as u64 + 7).wrapping_mul(0x9E3779B97F4A7C15));
                for u in st..en {
                    let mut l = lists[u].lock().unwrap();
                    let mut nw: Vec<usize> = (0..l.len()).filter(|&i| l[i].new).collect();
                    for i in 0..nw.len().min(s_) { let j = i + r.below(nw.len() - i); nw.swap(i, j); }
                    nw.truncate(s_);
                    let mut a_ = Vec::with_capacity(nw.len()); for &i in &nw { a_.push(l[i].id); l[i].new = false; }
                    let mut o_: Vec<u32> = Vec::new();
                    for e in l.iter() { if !e.new && !a_.contains(&e.id) { o_.push(e.id); } }
                    unsafe { *(pn as *mut Vec<u32>).add(u) = a_; *(po as *mut Vec<u32>).add(u) = o_; }
                }
            });
        }
        let (rnp, rni) = rev(&fnew); let (rop, roi) = rev(&fold);
        let upd = AtomicU64::new(0);
        par(n, th, 1024, |st, en| {
            let mut r = Rng(seed ^ ((it as u64) << 32) ^ (st as u64 + 13).wrapping_mul(0xD1B54A32D192ED03));
            let (mut loc_u, mut loc_d) = (0u64, 0u64);
            let mut buf: Vec<u8> = Vec::new();
            for u in st..en {
                let mut nw = fnew[u].clone(); let mut od = fold[u].clone();
                let pick = |src: &[u32], dst: &mut Vec<u32>, r: &mut Rng| {
                    if src.len() <= s_ { for &v in src { if !dst.contains(&v) { dst.push(v); } } }
                    else { for _ in 0..s_ { let v = src[r.below(src.len())]; if !dst.contains(&v) { dst.push(v); } } }
                };
                pick(&rni[rnp[u]..rnp[u + 1]], &mut nw, &mut r);
                pick(&roi[rop[u]..rop[u + 1]], &mut od, &mut r);
                od.retain(|v| !nw.contains(v));
                if fast {
                    // 块化：候选码拷到连续缓冲（nw 在前、od 在后），距离从缓冲算，对称一次
                    let (a, b) = (nw.len(), od.len());
                    if blk { buf.clear(); for &v in nw.iter().chain(od.iter()) { buf.extend_from_slice(cd(v as usize)); } }
                    let bc = |i: usize| if blk { &buf[i * dp..(i + 1) * dp] } else { cd(if i < a { nw[i] } else { od[i - a] } as usize) };
                    for i in 0..a {
                        let a_ = nw[i];
                        for j in i + 1..a + b {
                            let b_ = if j < a { nw[j] } else { od[j - a] }; if b_ == a_ { continue; }
                            let d = dcode(bc(i), bc(j)); loc_d += 1;
                            loc_u += insert(a_ as usize, b_, d) + insert(b_ as usize, a_, d);
                        }
                    }
                    continue;
                }
                for i in 0..nw.len() {
                    let a_ = nw[i] as usize;
                    for j in i + 1..nw.len() {
                        let b_ = nw[j] as usize; let d = dcode(cd(a_), cd(b_)); loc_d += 1;
                        loc_u += insert(a_, b_ as u32, d) + insert(b_, a_ as u32, d);
                    }
                    for &b in &od {
                        let b_ = b as usize; if b_ == a_ { continue; } let d = dcode(cd(a_), cd(b_)); loc_d += 1;
                        loc_u += insert(a_, b, d) + insert(b_, a_ as u32, d);
                    }
                }
            }
            upd.fetch_add(loc_u, Ordering::Relaxed); ndist.fetch_add(loc_d, Ordering::Relaxed);
        });
        let u_ = upd.load(Ordering::Relaxed); iters_done = it + 1;
        eprintln!("[nnd] iter {it}: updates {u_} ({:.4} of nK) {:.1}s total {:.1}s", u_ as f64 / (n * k) as f64, ti.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
        if (u_ as f64) < p.delta * (n * k) as f64 { break; }
    }
    let tn = t0.elapsed().as_secs_f64();
    drop(worst);
    let knn: Vec<Vec<u32>> = lists.into_iter().map(|m| m.into_inner().unwrap().into_iter().map(|e| e.id).collect()).collect();
    let mut knnh = vec![u32::MAX; n * hk];
    for (u, l) in knn.iter().enumerate() { for (j, &v) in l.iter().take(hk).enumerate() { knnh[u * hk + j] = v; } }
    let (ip, ii) = rev(&knn);
    // ---- 补救候选（R88-3，缺省关）
    let extra: Vec<Vec<u32>> = if p.lr > 0 { long_range_edges(x, dim, &code, dp, &knn, p) } else { Vec::new() };
    let tl = t0.elapsed().as_secs_f64();
    // ---- 一次 α-RNG：候选 = kNN 出 ∪ kNN 入（∪ 补救候选），float 距离最近 CM 个，贪心 RNG，cap
    let xr = |i: usize| &x[i * dim..(i + 1) * dim];
    let (a2, cm, cap) = (p.a2, p.cm, p.cap);
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let degsum = AtomicU64::new(0);
    {
        let ap = adj.as_mut_ptr() as usize;
        par(n, th, 1024, |st, en| {
            let mut ds_ = 0u64;
            for u in st..en {
                let mut c: Vec<u32> = knn[u].clone(); c.extend_from_slice(&ii[ip[u]..ip[u + 1]]);
                if !extra.is_empty() { c.extend_from_slice(&extra[u]); }
                c.sort_unstable(); c.dedup(); c.retain(|&v| v as usize != u);
                let mut cdv: Vec<(f32, u32)> = c.iter().map(|&v| (df(xr(u), xr(v as usize)), v)).collect();
                cdv.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1))); cdv.truncate(cm.max(1));
                let mut keep: Vec<(f32, u32)> = Vec::with_capacity(cap);
                for &(du, v) in &cdv {
                    if keep.iter().all(|&(_, w)| a2 * df(xr(w as usize), xr(v as usize)) > du) { keep.push((du, v)); }
                    if keep.len() >= cap { break; }
                }
                // 补救候选若被剪掉，仍强制保留最多 lr 条（长程边按定义会被近邻遮挡）
                if p.lr > 0 && !extra.is_empty() && keep.len() < cap {
                    let mut add = 0;
                    for &v in &extra[u] {
                        if add >= p.lr || keep.len() >= cap { break; }
                        if v as usize == u || keep.iter().any(|&(_, w)| w == v) { continue; }
                        keep.push((df(xr(u), xr(v as usize)), v)); add += 1;
                    }
                    keep.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
                }
                ds_ += keep.len() as u64;
                unsafe { *(ap as *mut Vec<u32>).add(u) = keep.iter().map(|t| t.1).collect(); }
            }
            degsum.fetch_add(ds_, Ordering::Relaxed);
        });
    }
    // ---- R88-3b 入度下限（缺省关，RAVEN_NND_RIN=m>0）：入度 < m 的点 v，沿 v 的保留出边 v→w 按距离由近到远补反向边 w→v
    // （不过 RNG、按距离插入 w 的有序表；w 的度数上限 RAVEN_NND_RIN_CAP，缺省 cap+8 且 ≤ r_max 由调用方保证）。
    // 动机：Vamana 的双向插入在度数未满时无条件保留反向边，kNN→一次 RNG 的图没有这一步，孤立/OOD 区的点入度低、难被搜到。
    let rin: usize = env("RAVEN_NND_RIN", 0);
    let (mut in0, mut in2) = (0usize, 0usize);
    {
        let mut indeg = vec![0u32; n];
        for l in &adj { for &v in l { indeg[v as usize] += 1; } }
        for &d in &indeg { if d == 0 { in0 += 1; } if d <= 2 { in2 += 1; } }
        if rin > 0 {
            let rcap: usize = env::<usize>("RAVEN_NND_RIN_CAP", cap + 8).min(p.r_max);
            let mut added = 0usize;
            for v in 0..n {
                if (indeg[v] as usize) >= rin { continue; }
                let outs = adj[v].clone();
                for w in outs {
                    if (indeg[v] as usize) >= rin { break; }
                    let wl = &mut adj[w as usize];
                    if wl.len() >= rcap || wl.contains(&(v as u32)) { continue; }
                    let dv = df(xr(w as usize), xr(v));
                    let pos = wl.partition_point(|&z| df(xr(w as usize), xr(z as usize)) <= dv);
                    wl.insert(pos, v as u32); indeg[v] += 1; added += 1;
                }
            }
            let dsum: usize = adj.iter().map(|l| l.len()).sum();
            degsum.store(dsum as u64, Ordering::Relaxed);
            eprintln!("[nnd] in-degree floor m={rin} rcap={rcap}: added {added} reverse edges");
        }
    }
    let tp = t0.elapsed().as_secs_f64();
    eprintln!("[nnd] indeg0={in0} indeg<=2={in2} (before floor)");
    eprintln!("[nnd] RESULT n={n} dim={dim} K={k} S={s_} A2={a2} CM={cm} CAP={cap} LR={} threads={th} iters={iters_done} sq4_s={tq:.1} nnd_s={:.1} lr_s={:.1} prune_s={:.1} TOTAL_s={tp:.1} codedist={:.2}e9 avgdeg={:.2}",
        p.lr, tn - tq, tl - tn, tp - tl, ndist.load(Ordering::Relaxed) as f64 / 1e9, degsum.load(Ordering::Relaxed) as f64 / n as f64);
    eprintln!("[nnd] fast={fastm}");
    (adj, knnh)
}

/// R88-3 补救候选（缺省关，RAVEN_NND_LR=m>0）：码空间"粗层 kNN 继承"长程边。
/// 动机：kNN 派生的图在 OOD/稀疏区缺 Vamana 搜索路径上的长程边（R87-D：@0.99 evals 1.07×，召回天花板 0.996 vs 0.999）。
/// 做法：随机取 n/γ 个样本点（γ=RAVEN_NND_LR_GAMMA，缺省 64），在样本内做暴力码空间 kNN（每样本 KL=RAVEN_NND_LR_KL 个，缺省 8）——
/// 这是"粗尺度"近邻图，其边长约为 γ^(1/d_intr) 倍的细尺度边。每点 u 取其 kNN 中最近的样本代表 s(u)（u 自身是样本则取自身），
/// 候选补 s(u) 的粗 kNN（共 KL 个），在剪枝里优先过 RNG，过不了时强制保留最近的 m 条。
pub fn long_range_edges(_x: &[f32], _dim: usize, code: &[u8], dp: usize, knn: &[Vec<u32>], p: &NndParams) -> Vec<Vec<u32>> {
    let n = knn.len();
    let gamma: usize = env("RAVEN_NND_LR_GAMMA", 64); let kl: usize = env("RAVEN_NND_LR_KL", 8);
    let cd = |i: usize| &code[i * dp..(i + 1) * dp];
    let mut r = Rng(p.seed ^ 0xA5A5_5A5A_1234_5678);
    let ns = (n / gamma.max(2)).max(kl + 2).min(n);
    let mut is_s = vec![u32::MAX; n]; let mut samp: Vec<u32> = Vec::with_capacity(ns);
    while samp.len() < ns { let v = r.below(n); if is_s[v] == u32::MAX { is_s[v] = samp.len() as u32; samp.push(v as u32); } }
    // 样本内暴力码空间 kNN（ns²/2 码距离；n=1M γ=64 → 1.2e8，≈1–2 s）
    let mut sknn: Vec<Vec<u32>> = vec![Vec::new(); ns];
    {
        let sp = sknn.as_mut_ptr() as usize; let samp = &samp;
        par(ns, p.threads, 64, |st, en| {
            for a in st..en {
                let mut best: Vec<(u32, u32)> = Vec::with_capacity(kl + 1);
                for b in 0..ns {
                    if a == b { continue; }
                    let d = dcode(cd(samp[a] as usize), cd(samp[b] as usize));
                    if best.len() < kl || d < best[best.len() - 1].0 {
                        let pos = best.partition_point(|t| t.0 <= d); best.insert(pos, (d, samp[b])); if best.len() > kl { best.pop(); }
                    }
                }
                unsafe { *(sp as *mut Vec<u32>).add(a) = best.into_iter().map(|t| t.1).collect(); }
            }
        });
    }
    // 每点代表样本：自身 / kNN 中第一个样本点 / 否则 kNN 的 kNN 中第一个样本点（两跳）；都没有则无补边
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    let op = out.as_mut_ptr() as usize;
    par(n, p.threads, 4096, |st, en| {
        for u in st..en {
            let mut s = is_s[u];
            if s == u32::MAX { for &v in &knn[u] { if is_s[v as usize] != u32::MAX { s = is_s[v as usize]; break; } } }
            if s == u32::MAX { 'o: for &v in &knn[u] { for &w in &knn[v as usize] { if is_s[w as usize] != u32::MAX { s = is_s[w as usize]; break 'o; } } } }
            if s == u32::MAX { continue; }
            let mut c: Vec<u32> = Vec::with_capacity(kl + 1);
            if samp[s as usize] as usize != u { c.push(samp[s as usize]); }
            c.extend(sknn[s as usize].iter().copied().filter(|&v| v as usize != u));
            unsafe { *(op as *mut Vec<u32>).add(u) = c; }
        }
    });
    out
}

/// 出∪入候选上的一次 α-RNG 剪枝（float L2，最近 CM 个，cap）；库内 H134 步（R88-3c 用于 Vamana 补搜之后）。
pub fn prune_out_in(x: &[f32], dim: usize, out: &[Vec<u32>], a2: f32, cm: usize, cap: usize, th: usize) -> Vec<Vec<u32>> {
    let n = out.len();
    let mut cnt = vec![0usize; n + 1];
    for l in out { for &v in l { cnt[v as usize + 1] += 1; } }
    for i in 0..n { cnt[i + 1] += cnt[i]; }
    let mut pos = cnt.clone(); let mut ii = vec![0u32; cnt[n]];
    for (u, l) in out.iter().enumerate() { for &v in l { ii[pos[v as usize]] = u as u32; pos[v as usize] += 1; } }
    let xr = |i: usize| &x[i * dim..(i + 1) * dim];
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let ap = adj.as_mut_ptr() as usize; let (ii, cnt) = (&ii, &cnt);
    par(n, th, 1024, |st, en| {
        for u in st..en {
            let mut c: Vec<u32> = out[u].clone(); c.extend_from_slice(&ii[cnt[u]..cnt[u + 1]]);
            c.sort_unstable(); c.dedup(); c.retain(|&v| v as usize != u);
            let mut cdv: Vec<(f32, u32)> = c.iter().map(|&v| (df(xr(u), xr(v as usize)), v)).collect();
            cdv.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1))); cdv.truncate(cm.max(1));
            let mut keep: Vec<(f32, u32)> = Vec::with_capacity(cap);
            for &(du, v) in &cdv {
                if keep.iter().all(|&(_, w)| a2 * df(xr(w as usize), xr(v as usize)) > du) { keep.push((du, v)); }
                if keep.len() >= cap { break; }
            }
            unsafe { *(ap as *mut Vec<u32>).add(u) = keep.iter().map(|t| t.1).collect(); }
        }
    });
    adj
}
