//! R33 HR 模式（SV-3b 库内集成）：高召回档的"图定位 + 单元 1-bit 免排序确认"。
//!
//! 由 [`crate::pipeline::Prepared::searcher`] 按 ef 门控：`ef < gate` 时 searcher 不挂本结构，
//! `search_sq4` 走原路径（逐位一致）；`ef ≥ gate` 时 searcher 以 `E = er·ef` 跑图得到精确 top-k 与
//! 第 k 名距离 T，再按查询到 k-means 中心的距离取前 `P = pr·ef` 个单元，用 [`BitCells`] fastscan
//! 以免排序门槛 `est < T − ‖q−c‖² + m·2‖q−c‖·ē·g/√D` 预筛，幸存者批量精算并入 top-k。
//! 机制与原创点见 design notes R32-2 / R33-1。

use crate::quant::bitcell::{rotate, BitCells, QueryLut};

/// HR 模式参数（ef → 图段 E、单元数 P）。
#[derive(Clone, Copy, Debug)]
pub struct HrParams {
    /// ef ≥ gate 时启用
    pub gate: usize,
    /// 图段 E = max(k, round(er·ef))
    pub er: f32,
    /// 单元数 P = max(1, round(pr·ef))
    pub pr: f32,
    /// 余量系数 m
    pub margin: f32,
    /// R34 单元扫描"耐心"早停：连续 pat 个单元无幸存者（且已扫 ≥ pminr·P 个）即停；0 = 关（逐位同 R33）。
    /// R34 默认映射 v2：P = 0.30·ef 为上限、pat 2、pminr 0.3（x86 同进程对 v1 0.37/0.18 +6…+17%，4/4）
    pub pat: usize,
    pub pminr: f32,
    /// R35 分段确认（S2C）：先扫 sa 个单元并精算幸存者、收紧门槛，再按证据决定是否续扫（sa=0 = 关，逐位同 R34）。
    /// 续扫条件：第一段改进数 ≥ su 或第一段幸存者数 ≥ ss（su/ss = 0 表示该判据不用；两者都 0 = 总续扫）
    pub sa: usize,
    pub su: usize,
    pub ss: usize,
    /// R35 覆盖判据：图段 top-k 落在前 sa 个单元内的个数 < sc 则续扫（0 = 不用）
    pub sc: usize,
    /// R36 第一段单元数随 P 伸缩：sa_eff = max(sa, round(saf·P))（0 = 固定 sa）。高 ef 时保持逐 ef 召回（CHARTER (b) 同 ef 召回差 ≤0.003）
    pub saf: f32,
    /// R36 半径收割（RH，大 k 用，原创）：k > rk 时图段只取 top-rk，按局部内禀维数（LID）MLE 外推第 k 名半径 r̂，
    /// 单元扫描以 (r̂·rmg)² 为初始门槛收割（rk = 0 = 关，逐位同旧路径）
    pub rk: usize,
    pub rmg: f32,
    /// R43 渐进门槛确认（PG，原创）：每扫 pg 个单元就精算已收集的幸存者、收紧门槛，后续单元用收紧后的门槛预筛
    /// （0 = 关，逐位同旧路径：全部单元以图段门槛扫完再精算）。k=100 时图段 top-k 门槛松，幸存者数是主成本。
    pub pg: usize,
    /// R43 诊断：累计单元数/幸存者/精算入榜数（RAVEN_HR_STAT=1；默认关，不进热路径）
    pub stat: bool,
    /// R43 图段 E 下限是否为 k（R36-5；默认 true）。PG 开启时单元段自行填满 top-k，E 可小于 k（RAVEN_HR_EK=0）
    pub ek: bool,
    /// R43 C8（原创，确认段字节）：槽序（单元连续）8-bit 码副本，幸存者先算 8-bit 距离，> thr·(1+c8) 直接弃，
    /// 只对通过者读 f32 行（0 = 关）。k=100 时幸存者 f32 行读取（每点 4·dim B）是确认段主成本。
    pub c8: f32,
}

/// R43 HR 诊断计数：[confirm 调用数, 扫描单元数, 幸存者数, 入榜数]
/// R44 oracle 扩展：[5] 扫描块数 [6] 有幸存者的块数 [7] 扫描点数 [8] 三角下界可剪点数（|d(q,c)−‖x−c‖|² > 当时门槛）[9] 无幸存者块内的点数
/// R45 oracle：[10] 子空间投影下界 ‖P(q−x)‖² > 门槛 的点数 [11] 整块全部可剪的块内点数
pub static HR_STAT: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
pub fn hr_stat_take() -> [u64; 12] { let mut o = [0u64; 12]; for i in 0..12 { o[i] = HR_STAT[i].swap(0, std::sync::atomic::Ordering::Relaxed); } o }

impl Default for HrParams {
    // R35：两平台统一映射 v2（P 上限 0.30·ef + 耐心早停 2/0.3）。x86 R34 证实；ARM R35 配对 v2 对 v1 6/6 +7…+26%（steal<5%）
    // R36 合并候选 B2：S2C sa 8、su 2、sc 5，P 上限 0.64·ef（sc 4 的 c1 在 ef ≥ 280 同 ef 召回比 v2 低 >0.003，sc 5 两平台守住），待档位门控 (a)(b) 逐平台判定
    fn default() -> Self {
        // R41 合并候选：两平台默认回到 v2（sa 0 = 不走 S2C 分段确认，P 上限 0.30·ef），与 main R35 行为相同；
        // B2（sa 8/su 2/sc 5/P 0.64·ef）x86 (a) 五批全负（R40）→ 留在叠加池，RAVEN_HR_SA/SU/SC/PR 可开。
        HrParams { gate: 200, er: 0.37, pr: 0.30, margin: 1.0, pat: 2, pminr: 0.3, sa: 0, su: 2, ss: 0, sc: 5, saf: 0.0, rk: 0, rmg: 1.02, pg: 0, stat: false, ek: true, c8: 0.0 }
    }
}

impl HrParams {
    /// 环境变量覆盖：RAVEN_HR_GATE / RAVEN_HR_ER / RAVEN_HR_PR / RAVEN_HR_M
    pub fn from_env(mut self) -> Self {
        let g = |k: &str| std::env::var(k).ok();
        if let Some(v) = g("RAVEN_HR_GATE").and_then(|v| v.parse().ok()) { self.gate = v; }
        if let Some(v) = g("RAVEN_HR_ER").and_then(|v| v.parse().ok()) { self.er = v; }
        if let Some(v) = g("RAVEN_HR_PR").and_then(|v| v.parse().ok()) { self.pr = v; }
        if let Some(v) = g("RAVEN_HR_M").and_then(|v| v.parse().ok()) { self.margin = v; }
        if let Some(v) = g("RAVEN_HR_PAT").and_then(|v| v.parse().ok()) { self.pat = v; }
        if let Some(v) = g("RAVEN_HR_PMINR").and_then(|v| v.parse().ok()) { self.pminr = v; }
        if let Some(v) = g("RAVEN_HR_SA").and_then(|v| v.parse().ok()) { self.sa = v; }
        if let Some(v) = g("RAVEN_HR_SU").and_then(|v| v.parse().ok()) { self.su = v; }
        if let Some(v) = g("RAVEN_HR_SS").and_then(|v| v.parse().ok()) { self.ss = v; }
        if let Some(v) = g("RAVEN_HR_SC").and_then(|v| v.parse().ok()) { self.sc = v; }
        if let Some(v) = g("RAVEN_HR_SAF").and_then(|v| v.parse().ok()) { self.saf = v; }
        if let Some(v) = g("RAVEN_HR_RK").and_then(|v| v.parse().ok()) { self.rk = v; }
        if let Some(v) = g("RAVEN_HR_RMG").and_then(|v| v.parse().ok()) { self.rmg = v; }
        if let Some(v) = g("RAVEN_HR_PG").and_then(|v| v.parse().ok()) { self.pg = v; }
        if let Some(v) = g("RAVEN_HR_STAT") { self.stat = v == "1"; }
        if let Some(v) = g("RAVEN_HR_EK") { self.ek = v != "0"; }
        if let Some(v) = g("RAVEN_HR_C8").and_then(|v| v.parse().ok()) { self.c8 = v; }
        self
    }
}

pub struct CellConfirm {
    pub bc: BitCells,
    pub nl: usize,
    pub dim: usize,
    pub params: HrParams,
    cent8: Vec<i8>,
    csum: Vec<i32>,
    cs8: f32,
    use_i8: bool,
    /// 内部 id → 单元（R33-2 局部修复/证书用）
    pub cell_of: Vec<u32>,
    /// R43 C8：槽序 8-bit 码（ns × dim；空 = 关）、每维下界、全局步长
    pub c8: Vec<u8>,
    pub c8_min: Vec<f32>,
    pub c8_s: f32,
    /// R45 oracle（仅统计模式且 RAVEN_HR_ORM=m>0）：PCA 子空间投影 P (m×dim) 与全库投影 (内部 id × m)
    pub orp: Vec<f32>,
    pub orproj: Vec<f32>,
    pub orm: usize,
}

/// 每 searcher 的复用缓冲
pub struct CcScratch {
    pub cs: Vec<(f32, u32)>,
    pub sc_i: Vec<i32>,
    pub qr: Vec<f32>,
    pub sur: Vec<u32>,
    pub top: Vec<(f32, u32)>,
    pub lut: QueryLut,
    /// R36：初始门槛上限（平方 L2；f32::MAX = 不用）。top 不足 k 时用它代替 +∞
    pub thr0: f32,
    /// R42b：冷参数
    pub ext: CcExt,
    /// R43-2：上次 confirm 扫描的单元数（确定性代价模型用；cs[..ncell] 即所扫单元）
    pub ncell: u32,
}

/// 一个 searcher 上启用的 HR 计划（Copy，挂在 GraphSearcher 上）。
/// R42b：只留热字段（与 main R35 相同的 5 个），R35/R36 的分段确认/RH 覆盖参数移到 [`CcScratch::ext`]（已装箱，冷数据），
/// 避免 GraphSearcher 变大改变原路径布局（R43 接力测得 x86 GloVe 未启用档 −3…−6%、ARM HR 档 −4%）。
#[derive(Clone, Copy)]
pub struct CcActive<'a> {
    pub cc: &'a CellConfirm,
    pub p: usize,
    pub margin: f32,
    pub pat: usize,
    pub pminr: f32,
}

/// R35/R36 冷参数（可被 searcher setter 覆盖；默认取自 HrParams）
#[derive(Clone, Copy, Debug)]
pub struct CcExt {
    pub sa: usize,
    pub su: usize,
    pub ss: usize,
    pub sc: usize,
    pub saf: f32,
    pub rk: usize,
    pub rmg: f32,
    /// R36 RH oracle 初始门槛（诊断）
    pub rh_oracle: Option<f32>,
}

/// R35 分段确认诊断特征（第一段之后）
#[derive(Clone, Debug, Default)]
pub struct StageFeat {
    /// 第一段精算后进入 top-k 的点数
    pub u: u32,
    /// 第一段 1-bit 幸存者数
    pub s: u32,
    /// 图段 top-k 中落在前 sa 个单元内的个数
    pub cov: u32,
    /// 第 sa+1 个单元的中心距离相对收紧后门槛：sqrt(d_c)/sqrt(thr) − 1
    pub gap: f32,
    /// 第一段结束时的 top（内部 id）
    pub top_a: Vec<u32>,
    /// 第一段结束时 top-k 各成员所在单元在中心排序中的名次（≥ P 记 P）：最大 / 第二大
    pub rmax: u32,
    pub r2: u32,
}

/// 单元数：n/512 向下取 2 的幂，夹在 [64, 8192]
pub fn default_nl(n: usize) -> usize {
    let t = (n / 512).max(64);
    let mut p = 64usize; while p * 2 <= t && p < 8192 { p *= 2; }
    p
}

/// 采样 k-means（与 R31/R32 eval harness sv3_kmeans 同算法：nl·64 个等距采样点、10 轮 Lloyd、angular 时中心归一化），
/// 最后对全部点做一次分配。vectors 为内部序 n×dim。
pub fn kmeans(vectors: &[f32], dim: usize, nl: usize, angular: bool) -> (Vec<f32>, Vec<u32>) {
    use rayon::prelude::*;
    let n = vectors.len() / dim;
    let l2 = |a: &[f32], b: &[f32]| -> f32 { crate::distance::l2_simd(a, b) };
    let m = (nl * 64).min(n).max(nl.min(n)); let step = (n / m).max(1);
    let samp: Vec<usize> = (0..m).map(|i| (i * step).min(n - 1)).collect();
    let mut c: Vec<f32> = samp[..nl.min(m)].iter().flat_map(|&i| vectors[i * dim..(i + 1) * dim].to_vec()).collect();
    let nlr = c.len() / dim;
    let assign = |c: &Vec<f32>, ids: &[usize]| -> Vec<u32> { ids.par_iter().map(|&i| { let x = &vectors[i * dim..(i + 1) * dim];
        let mut best = (f32::MAX, 0u32); for j in 0..nlr { let d = l2(x, &c[j * dim..(j + 1) * dim]); if d < best.0 { best = (d, j as u32); } } best.1 }).collect() };
    for _it in 0..10 {
        let a = assign(&c, &samp);
        let mut sum = vec![0f64; nlr * dim]; let mut cnt = vec![0usize; nlr];
        for (k, &i) in samp.iter().enumerate() { let j = a[k] as usize; cnt[j] += 1; for d in 0..dim { sum[j * dim + d] += vectors[i * dim + d] as f64; } }
        for j in 0..nlr { if cnt[j] > 0 { let mut nv: Vec<f32> = (0..dim).map(|d| (sum[j * dim + d] / cnt[j] as f64) as f32).collect();
            if angular { let s = nv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); for x in nv.iter_mut() { *x /= s; } }
            c[j * dim..(j + 1) * dim].copy_from_slice(&nv); } }
    }
    let all: Vec<usize> = (0..n).collect();
    let a = assign(&c, &all);
    (c, a)
}

/// 数据是否单位范数（抽样判断），用于 k-means 中心归一化
pub fn looks_normalized(vectors: &[f32], dim: usize) -> bool {
    let n = vectors.len() / dim; if n == 0 { return false; }
    let step = (n / 1000).max(1);
    (0..n).step_by(step).all(|i| { let s: f32 = vectors[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum(); (s - 1.0).abs() < 1e-3 })
}

impl CellConfirm {
    /// cent: nl×dim；asg[i] = 内部 id i 的单元
    pub fn from_centroids(vectors: &[f32], dim: usize, cent: &[f32], asg: &[u32], params: HrParams) -> CellConfirm {
        let nl = cent.len() / dim;
        let bc = BitCells::build(vectors, dim, cent, asg, 0x5eed_32b);
        let cmax = cent.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let cs8 = 127.0 / cmax;
        let mut cent8 = vec![0i8; nl * 128]; let mut csum = vec![0i32; nl];
        for j in 0..nl { for d in 0..dim.min(128) { let v = (cent[j * dim + d] * cs8).round().clamp(-127.0, 127.0) as i8; cent8[j * 128 + d] = v; csum[j] += v as i32; } }
        #[cfg(target_arch = "x86_64")]
        let use_i8 = dim <= 128 && std::is_x86_feature_detected!("avx2") && std::env::var("RAVEN_SV3B_CF32").is_err();
        #[cfg(not(target_arch = "x86_64"))]
        let use_i8 = false;
        let (c8, c8_min, c8_s) = if params.c8 > 0.0 { build_c8(vectors, dim, &bc.ids) } else { (Vec::new(), Vec::new(), 0.0) };
        let orm: usize = if params.stat { std::env::var("RAVEN_HR_ORM").ok().and_then(|x| x.parse().ok()).unwrap_or(0) } else { 0 };
        let (orproj, orp) = if orm > 0 && orm < dim {
            let (p, frac) = crate::pipeline::lp_auto(vectors, dim, orm);
            let n = vectors.len() / dim; let mut o = vec![0f32; n * orm];
            for i in 0..n { let x = &vectors[i * dim..(i + 1) * dim]; for r in 0..orm { o[i * orm + r] = p[r * dim..(r + 1) * dim].iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>(); } }
            eprintln!("R45 oracle proj m={orm} varfrac={frac:.3}");
            (p, o)
        } else { (Vec::new(), Vec::new()) };
        CellConfirm { bc, nl, dim, params, cent8, csum, cs8, use_i8, cell_of: asg.to_vec(), c8, c8_min, c8_s, orp, orproj, orm }
    }

    pub fn build(vectors: &[f32], dim: usize, nl: usize, angular: bool, params: HrParams) -> CellConfirm {
        let (c, a) = kmeans(vectors, dim, nl, angular);
        Self::from_centroids(vectors, dim, &c, &a, params)
    }

    /// ef → (E, P)；None = 未启用
    #[inline]
    pub fn plan(&self, ef: usize) -> Option<(usize, usize)> { self.plan_k(ef, 10) }

    /// R36-5：k 感知规划。图段 E 必须 ≥ k，否则图段只返回 E(<k) 个候选，确认段阈值 thr=∞、P 个单元全量精算
    /// （x86 GloVe k=100 实测 ef 200/250 → E 74/92：5.1–6.4 ms/查询，比 ef 300 慢 8–10×）。
    pub fn plan_k(&self, ef: usize, k: usize) -> Option<(usize, usize)> {
        let p = &self.params;
        if p.gate == 0 || ef < p.gate { return None; }
        let (e, pc) = self.plan_force(ef);
        Some((if p.ek { e.max(k) } else { e }, pc))
    }

    /// 不看门槛的 ef → (E, P)（门槛标定用）
    pub fn plan_force(&self, ef: usize) -> (usize, usize) {
        let p = &self.params;
        let e = ((p.er * ef as f32).round() as usize).max(10);
        let pc = ((p.pr * ef as f32).round() as usize).clamp(1, self.nl);
        (e, pc)
    }

    pub fn scratch(&self) -> CcScratch {
        CcScratch { cs: vec![(0.0, 0); self.nl], sc_i: vec![0; self.nl + 8], qr: vec![0.0; self.bc.dpad], sur: Vec::with_capacity(4096),
                    top: Vec::with_capacity(128), lut: QueryLut { lut: vec![0; self.bc.nb * 32], k: 0.0, k2: 0.0 }, thr0: f32::MAX,
                    ext: CcExt { sa: self.params.sa, su: self.params.su, ss: self.params.ss, sc: self.params.sc, saf: self.params.saf, rk: self.params.rk, rmg: self.params.rmg, rh_oracle: None }, ncell: 0 }
    }

    /// R43-2：上次 confirm 的 (幸存者数, 扫描单元数, 扫描槽数, 每槽流式字节)（确定性代价模型）
    pub fn last_counts(&self, sc: &CcScratch) -> (usize, usize, usize, usize) {
        let nc = sc.ncell as usize; let bc = &self.bc;
        let slots: usize = sc.cs[..nc.min(sc.cs.len())].iter().map(|x| { let j = x.1 as usize; bc.offs[j + 1] - bc.offs[j] }).sum();
        (sc.sur.len(), nc, slots, bc.bs * 2 / crate::quant::bitcell::BLK)
    }

    pub fn bytes(&self) -> usize { self.bc.blob.len() * 2 + self.bc.ids.len() * 4 + self.bc.rot.len() * 4 + self.bc.cent.len() * 4 + self.cent8.len() + self.c8.len() }

    /// 确认段：top（精确距离升序、≤k 个，内部 id）就地更新。
    pub fn confirm(&self, q: &[f32], vectors: &[f32], k: usize, pp: usize, mm: f32, sc: &mut CcScratch) { self.confirm_pat(q, vectors, k, pp, mm, self.params.pat, self.params.pminr, sc) }

    pub fn confirm_pat(&self, q: &[f32], vectors: &[f32], k: usize, pp: usize, mm: f32, pat: usize, pminr: f32, sc: &mut CcScratch) {
        let dim = self.dim; let nl = self.nl; let bc = &self.bc;
        let row = |id: u32| -> &[f32] { &vectors[id as usize * dim..(id as usize + 1) * dim] };
        let top = &mut sc.top;
        let mut thr = if top.len() >= k { top[k - 1].0.min(sc.thr0) } else { sc.thr0 };
        rotate(&bc.rot, bc.dpad, q, &mut sc.qr);
        bc.prepare_query_into(&sc.qr, &mut sc.lut);
        // 单元打分
        let cs = &mut sc.cs;
        if self.use_i8 {
            #[cfg(target_arch = "x86_64")] {
                let mut q8 = [0u8; 128];
                let qm = q.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let qs8 = 63.0 / qm;
                for d in 0..128 { q8[d] = if d < dim { ((q[d] * qs8).round().clamp(-63.0, 63.0) as i32 + 64) as u8 } else { 64 }; }
                unsafe { cdot8_avx2(&self.cent8, nl, &q8, &mut sc.sc_i); }
                let inv = 1.0 / (self.cs8 * qs8);
                for j in 0..nl { cs[j] = (bc.cn2[j] - 2.0 * ((sc.sc_i[j] - 64 * self.csum[j]) as f32) * inv, j as u32); }
            }
        } else {
            for j in 0..nl { cs[j] = (crate::distance::l2_simd(q, &bc.cent[j * dim..(j + 1) * dim]), j as u32); }
        }
        let pp2 = pp.min(nl);
        cs[..nl].select_nth_unstable_by(pp2 - 1, |a, b| a.0.total_cmp(&b.0));
        cs[..pp2].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let sqd = (bc.dpad as f32).sqrt();
        let sur = &mut sc.sur; sur.clear();
        bc.prefetch_cell(cs[0].1 as usize, 2);
        let pmin = ((pminr * pp2 as f32).round() as usize).max(1);
        let mut dry = 0usize;
        let pg = self.params.pg;
        let mut done = 0usize; let mut nins = 0u64; let mut ncell = 0usize;
        // 精算 sur[from..] 并入 top，收紧 thr
        #[inline(always)]
        fn conf(sur: &[u32], from: usize, bc: &BitCells, vectors: &[f32], dim: usize, q: &[f32], k: usize, top: &mut Vec<(f32, u32)>, thr: &mut f32, nins: &mut u64) {
            const PFD: usize = 8;
            let row = |id: u32| -> &[f32] { &vectors[id as usize * dim..(id as usize + 1) * dim] };
            for &so in sur[from..].iter().take(PFD) { prefetch_row(row(bc.ids[so as usize])); }
            for i in from..sur.len() {
                if i + PFD < sur.len() { prefetch_row(row(bc.ids[sur[i + PFD] as usize])); }
                let id = bc.ids[sur[i] as usize];
                let d = crate::distance::l2_simd(q, row(id));
                if d < *thr && !top.iter().any(|x| x.1 == id) {
                    let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, id)); top.truncate(k);
                    if top.len() == k { *thr = top[k - 1].0; }
                    *nins += 1;
                }
            }
        }
        // R43 C8：先 8-bit 槽序码过滤（qq = q − min），通过者再精算
        #[inline(always)]
        fn conf8(sur: &[u32], from: usize, cc: &CellConfirm, vectors: &[f32], q16: &[i16], q: &[f32], k: usize, top: &mut Vec<(f32, u32)>, thr: &mut f32, nins: &mut u64, n8: &mut u64) {
            const PFD: usize = 8;
            let dim = cc.dim; let s8 = cc.c8_s; let slack = 1.0 + cc.params.c8;
            let crow = |so: u32| -> &[u8] { &cc.c8[so as usize * dim..(so as usize + 1) * dim] };
            for &so in sur[from..].iter().take(PFD) { prefetch_u8(crow(so)); }
            for i in from..sur.len() {
                if i + PFD < sur.len() { prefetch_u8(crow(sur[i + PFD])); }
                let so = sur[i];
                let c = crow(so);
                // 整数 SSD（可向量化）：d8 = s²·Σ(c − q16)²
                let mut acc = 0i32;
                for t in 0..dim { let e = c[t] as i32 - q16[t] as i32; acc += e * e; }
                let d8 = s8 * s8 * acc as f32;
                if top.len() >= k && d8 > *thr * slack { continue; }
                *n8 += 1;
                let id = cc.bc.ids[so as usize];
                let d = crate::distance::l2_simd(q, &vectors[id as usize * dim..(id as usize + 1) * dim]);
                if d < *thr && !top.iter().any(|x| x.1 == id) {
                    let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, id)); top.truncate(k);
                    if top.len() == k { *thr = top[k - 1].0; }
                    *nins += 1;
                }
            }
        }
        let use8 = !self.c8.is_empty();
        // 旋转查询缓冲在 LUT 准备后已不再需要，复用为 qq = q − min（C8 开时）
        let mut q16: Vec<i16> = Vec::new();
        if use8 { q16.extend((0..dim).map(|t| ((q[t] - self.c8_min[t]) / self.c8_s).round().clamp(-32000.0, 32000.0) as i16)); }
        let mut n8 = 0u64;
        macro_rules! confirm_batch { ($from:expr) => {
            if use8 { conf8(sur, $from, self, vectors, &q16, q, k, top, &mut thr, &mut nins, &mut n8); }
            else { conf(sur, $from, bc, vectors, dim, q, k, top, &mut thr, &mut nins); }
        } }
        for ci in 0..pp2 {
            let j = cs[ci].1 as usize;
            if ci + 1 < pp2 { bc.prefetch_cell(cs[ci + 1].1 as usize, 2); }
            let qc2 = if !self.use_i8 { cs[ci].0 } else { crate::distance::l2_simd(q, &bc.cent[j * dim..(j + 1) * dim]) };
            let mc = mm * 2.0 * qc2.sqrt() / sqd;
            let n0 = sur.len();
            bc.scan_cell(j, &sc.lut, thr - qc2, mc, sur);
            if self.params.stat {
                // R44 oracle（仅统计模式）：块级幸存者分布 + 三角不等式下界的可剪比例
                use std::sync::atomic::Ordering::Relaxed;
                let (a, b) = (bc.offs[j], bc.offs[j + 1]); let blk = crate::quant::bitcell::BLK;
                let nb = (b - a) / blk; let mut hit = vec![false; nb];
                for &so in &sur[n0..] { hit[(so as usize - a) / blk] = true; }
                let nh = hit.iter().filter(|&&h| h).count();
                let dqc = qc2.max(0.0).sqrt(); let c = &bc.cent[j * dim..(j + 1) * dim];
                let (mut npt, mut ntri, mut nzero) = (0u64, 0u64, 0u64);
                for so in a..b { let id = bc.ids[so]; if id == u32::MAX { continue; } npt += 1;
                    let r = crate::distance::l2_simd(row(id), c).max(0.0).sqrt(); let lb = (dqc - r).abs();
                    if lb * lb > thr { ntri += 1; }
                    if !hit[(so - a) / blk] { nzero += 1; } }
                HR_STAT[5].fetch_add(nb as u64, Relaxed); HR_STAT[6].fetch_add(nh as u64, Relaxed); HR_STAT[7].fetch_add(npt, Relaxed);
                HR_STAT[8].fetch_add(ntri, Relaxed); HR_STAT[9].fetch_add(nzero, Relaxed);
                if self.orm > 0 {
                    let m = self.orm; let pq: Vec<f32> = (0..m).map(|r| self.orproj[r * dim..(r + 1) * dim].iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()).collect();
                    let (mut npr, mut nbs) = (0u64, 0u64);
                    for bi in 0..nb { let (mut all, mut cnt) = (true, 0u64);
                        for so in a + bi * blk..a + (bi + 1) * blk { let id = bc.ids[so]; if id == u32::MAX { continue; } cnt += 1;
                            let lb = crate::distance::l2_simd(&pq, &self.orp[id as usize * m..(id as usize + 1) * m]);
                            if lb > thr { npr += 1; } else { all = false; } }
                        if all { nbs += cnt; } }
                    HR_STAT[10].fetch_add(npr, Relaxed); HR_STAT[11].fetch_add(nbs, Relaxed);
                }
            }
            ncell = ci + 1;
            if pg > 0 && (ci + 1) % pg == 0 && sur.len() > done {
                confirm_batch!(done); done = sur.len();
            }
            if pat > 0 {
                if sur.len() == n0 { dry += 1; } else { dry = 0; }
                if dry >= pat && ci + 1 >= pmin { break; }
            }
        }
        confirm_batch!(done);
        sc.ncell = ncell as u32;
        if self.params.stat {
            use std::sync::atomic::Ordering::Relaxed;
            HR_STAT[0].fetch_add(1, Relaxed); HR_STAT[1].fetch_add(ncell as u64, Relaxed);
            HR_STAT[2].fetch_add(sur.len() as u64, Relaxed); HR_STAT[3].fetch_add(nins, Relaxed); HR_STAT[4].fetch_add(n8, Relaxed);
        }
    }
}

/// R36 RH：由升序精确平方距离（top-m）按 LID 的 MLE 外推第 k 名的平方半径（局部均匀模型 F(r) ∝ r^LID）
pub fn lid_radius2(d2: &[f32], k: usize) -> f32 {
    let m = d2.len();
    if m < 3 { return f32::MAX; }
    let rm = (d2[m - 1].max(1e-30) as f64).sqrt();
    let mut acc = 0f64;
    for &x in &d2[..m - 1] { acc += ((x.max(1e-30) as f64).sqrt() / rm).ln(); }
    let mean = acc / (m - 1) as f64;
    if mean >= -1e-9 { return f32::MAX; }
    let lid = -1.0 / mean;
    let r = rm * (k as f64 / m as f64).powf(1.0 / lid);
    (r * r) as f32
}

impl CellConfirm {
    /// R35 分段确认（S2C，原创："先精算、后取证"）：单元按中心分数排序后，第一段只扫前 sa 个单元并**立即精算**其幸存者，
    /// 用精算结果收紧门槛并取证（改进数 u、幸存者数 s）；只有证据显示图段结果不完整（u ≥ su 或 s ≥ ss）才续扫剩余单元
    /// （续扫用收紧后的门槛 + 耐心早停）。feat 非 None 时总续扫并记录特征（oracle/检测器诊断）。返回是否续扫。
    #[allow(clippy::too_many_arguments)]
    pub fn confirm_staged(&self, q: &[f32], vectors: &[f32], k: usize, pp: usize, mm: f32, sa: usize, su: usize, ss: usize, scv: usize,
                          pat: usize, pminr: f32, sc: &mut CcScratch, mut feat: Option<&mut StageFeat>) -> bool {
        let dim = self.dim; let nl = self.nl; let bc = &self.bc;
        let row = |id: u32| -> &[f32] { &vectors[id as usize * dim..(id as usize + 1) * dim] };
        let mut thr = if sc.top.len() >= k { sc.top[k - 1].0.min(sc.thr0) } else { sc.thr0 };
        rotate(&bc.rot, bc.dpad, q, &mut sc.qr);
        bc.prepare_query_into(&sc.qr, &mut sc.lut);
        let cs = &mut sc.cs;
        if self.use_i8 {
            #[cfg(target_arch = "x86_64")] {
                let mut q8 = [0u8; 128];
                let qm = q.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12); let qs8 = 63.0 / qm;
                for d in 0..128 { q8[d] = if d < dim { ((q[d] * qs8).round().clamp(-63.0, 63.0) as i32 + 64) as u8 } else { 64 }; }
                unsafe { cdot8_avx2(&self.cent8, nl, &q8, &mut sc.sc_i); }
                let inv = 1.0 / (self.cs8 * qs8);
                for j in 0..nl { cs[j] = (bc.cn2[j] - 2.0 * ((sc.sc_i[j] - 64 * self.csum[j]) as f32) * inv, j as u32); }
            }
        } else {
            for j in 0..nl { cs[j] = (crate::distance::l2_simd(q, &bc.cent[j * dim..(j + 1) * dim]), j as u32); }
        }
        let pp2 = pp.min(nl);
        cs[..nl].select_nth_unstable_by(pp2 - 1, |a, b| a.0.total_cmp(&b.0));
        cs[..pp2].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let a2 = sa.clamp(1, pp2);
        let sqd = (bc.dpad as f32).sqrt();
        let top = &mut sc.top; let sur = &mut sc.sur;
        let cov = if feat.is_some() || scv > 0 { top.iter().filter(|x| cs[..a2].iter().any(|c| c.1 == self.cell_of[x.1 as usize])).count() as u32 } else { 0 };
        // 精算 sur 中幸存者，返回改进数
        let refine = |sur: &Vec<u32>, top: &mut Vec<(f32, u32)>, thr: &mut f32| -> u32 {
            const PFD: usize = 8; let mut nimp = 0u32;
            for &so in sur.iter().take(PFD) { prefetch_row(row(bc.ids[so as usize])); }
            for i in 0..sur.len() {
                if i + PFD < sur.len() { prefetch_row(row(bc.ids[sur[i + PFD] as usize])); }
                let id = bc.ids[sur[i] as usize];
                let d = crate::distance::l2_simd(q, row(id));
                if d < *thr && !top.iter().any(|x| x.1 == id) {
                    let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, id)); top.truncate(k); nimp += 1;
                    if top.len() == k { *thr = top[k - 1].0; }
                }
            }
            nimp
        };
        let scan = |ci: usize, thr: f32, sur: &mut Vec<u32>| {
            let j = cs[ci].1 as usize;
            if ci + 1 < pp2 { bc.prefetch_cell(cs[ci + 1].1 as usize, 2); }
            let qc2 = if !self.use_i8 { cs[ci].0 } else { crate::distance::l2_simd(q, &bc.cent[j * dim..(j + 1) * dim]) };
            bc.scan_cell(j, &sc.lut, thr - qc2, mm * 2.0 * qc2.sqrt() / sqd, sur);
        };
        sur.clear();
        bc.prefetch_cell(cs[0].1 as usize, 2);
        for ci in 0..a2 { scan(ci, thr, sur); }
        let s1 = sur.len() as u32;
        let u = refine(sur, top, &mut thr);
        let esc = feat.is_some() || (su > 0 && u as usize >= su) || (ss > 0 && s1 as usize >= ss) || (scv > 0 && (cov as usize) < scv) || (su == 0 && ss == 0 && scv == 0);
        if let Some(f) = feat.as_deref_mut() {
            f.u = u; f.s = s1; f.cov = cov;
            f.gap = if a2 < pp2 && thr < f32::MAX && thr > 0.0 { (cs[a2].0.max(0.0) / thr).sqrt() - 1.0 } else { f32::MAX };
            f.top_a = top.iter().map(|x| x.1).collect();
            let mut rk: Vec<u32> = top.iter().map(|x| { let c = self.cell_of[x.1 as usize]; cs[..pp2].iter().position(|y| y.1 == c).unwrap_or(pp2) as u32 }).collect();
            rk.sort_unstable(); f.rmax = rk.last().copied().unwrap_or(0); f.r2 = if rk.len() >= 2 { rk[rk.len() - 2] } else { 0 };
        }
        if !esc || a2 >= pp2 { return false; }
        sur.clear();
        let pmin = ((pminr * pp2 as f32).round() as usize).max(1);
        let mut dry = 0usize;
        for ci in a2..pp2 {
            let n0 = sur.len();
            scan(ci, thr, sur);
            if pat > 0 {
                if sur.len() == n0 { dry += 1; } else { dry = 0; }
                if dry >= pat && ci + 1 >= pmin { break; }
            }
        }
        refine(sur, top, &mut thr);
        true
    }

    /// R33-2 局部修复：只扫给定单元（如图结果 top-k 所在单元，免中心打分），门槛 = 当前第 k 名精确距离；
    /// 返回改进次数（= 有比图结果更近的点被找回 → 图结果不完整的证据）。sc.top 需已含图结果（精确距离升序）。
    pub fn repair_cells(&self, q: &[f32], vectors: &[f32], k: usize, cells: &[u32], mm: f32, sc: &mut CcScratch) -> usize {
        self.repair_cells_near(q, vectors, k, cells, mm, 0.0, sc).0
    }

    /// 同上，另返回“近失”数：精算距离 < (1+delta)·T0 但未进 top-k 的非结果点（T0 = 修复前第 k 名距离）
    pub fn repair_cells_near(&self, q: &[f32], vectors: &[f32], k: usize, cells: &[u32], mm: f32, delta: f32, sc: &mut CcScratch) -> (usize, usize) {
        let dim = self.dim; let bc = &self.bc;
        let row = |id: u32| -> &[f32] { &vectors[id as usize * dim..(id as usize + 1) * dim] };
        let top = &mut sc.top;
        let mut thr = if top.len() >= k { top[k - 1].0.min(sc.thr0) } else { sc.thr0 };
        let t0 = thr; let tn = if thr < f32::MAX { thr * (1.0 + delta) } else { thr };
        rotate(&bc.rot, bc.dpad, q, &mut sc.qr);
        bc.prepare_query_into(&sc.qr, &mut sc.lut);
        let sqd = (bc.dpad as f32).sqrt();
        let sur = &mut sc.sur; sur.clear();
        for (ci, &j) in cells.iter().enumerate() {
            let j = j as usize;
            if ci + 1 < cells.len() { bc.prefetch_cell(cells[ci + 1] as usize, 2); }
            let qc2 = crate::distance::l2_simd(q, &bc.cent[j * dim..(j + 1) * dim]);
            bc.scan_cell(j, &sc.lut, tn - qc2, mm * 2.0 * qc2.sqrt() / sqd, sur);
        }
        let _ = t0;
        let (mut nimp, mut nnear) = (0usize, 0usize);
        for i in 0..sur.len() {
            if i + 8 < sur.len() { prefetch_row(row(bc.ids[sur[i + 8] as usize])); }
            let id = bc.ids[sur[i] as usize];
            let d = crate::distance::l2_simd(q, row(id));
            let intop = top.iter().any(|x| x.1 == id);
            if d < thr && !intop {
                let pos = top.partition_point(|x| x.0 <= d); top.insert(pos, (d, id)); top.truncate(k); nimp += 1;
                if top.len() == k { thr = top[k - 1].0; }
            } else if !intop && d < tn { nnear += 1; }
        }
        (nimp, nnear)
    }
}

#[inline]
/// R43 C8：槽序 8-bit 码（每维 min，全局步长 = 最大维跨度/255）
pub fn build_c8(vectors: &[f32], dim: usize, ids: &[u32]) -> (Vec<u8>, Vec<f32>, f32) {
    let n = vectors.len() / dim;
    let mut mn = vec![f32::MAX; dim]; let mut mx = vec![f32::MIN; dim];
    for i in 0..n { for d in 0..dim { let v = vectors[i * dim + d]; if v < mn[d] { mn[d] = v; } if v > mx[d] { mx[d] = v; } } }
    let span = (0..dim).map(|d| mx[d] - mn[d]).fold(0f32, f32::max).max(1e-12);
    let s = span / 255.0;
    let mut c = vec![0u8; ids.len() * dim];
    for (so, &id) in ids.iter().enumerate() {
        if id == u32::MAX { continue; }
        for d in 0..dim { c[so * dim + d] = ((vectors[id as usize * dim + d] - mn[d]) / s).round().clamp(0.0, 255.0) as u8; }
    }
    (c, mn, s)
}

#[inline(always)]
fn prefetch_u8(x: &[u8]) {
    let p = x.as_ptr() as *const i8; let nb = x.len();
    let mut o = 0;
    while o < nb {
        #[cfg(target_arch = "x86_64")] unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.wrapping_add(o)); }
        #[cfg(target_arch = "aarch64")] unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.wrapping_add(o), options(nostack, readonly, preserves_flags)); }
        o += 64;
    }
}

fn prefetch_row(x: &[f32]) {
    let p = x.as_ptr() as *const i8; let nb = x.len() * 4;
    let mut o = 0;
    while o < nb {
        #[cfg(target_arch = "x86_64")] unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.wrapping_add(o)); }
        #[cfg(target_arch = "aarch64")] unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.wrapping_add(o), options(nostack, readonly, preserves_flags)); }
        o += 64;
    }
}

/// int8 中心（每个 128 字节）与 u8 查询（[-63,63]+64）的点积，8 个中心一批（R31 sv3_cdot8_avx2）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cdot8_avx2(cent8: &[i8], nl: usize, q8: &[u8; 128], out: &mut [i32]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hr_plan_gate() {
        let mut v = vec![0f32; 4096 * 32];
        let mut s = 1u64; for x in v.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *x = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5; }
        let cc = CellConfirm::build(&v, 32, 64, false, HrParams { gate: 100, pat: 0, ..Default::default() });
        assert!(cc.plan(99).is_none());
        let (e, p) = cc.plan(200).unwrap(); assert!(e >= 10 && p >= 1 && p <= 64);
        assert_eq!(cc.bc.ids.iter().filter(|&&x| x != u32::MAX).count(), 4096);
    }
    #[test]
    fn hr_confirm_finds_exact_nn_full_scan() {
        // P = 全部单元、余量很大 → 确认段应等价于全扫描精确 top-k
        let (n, d) = (3000usize, 40usize);
        let mut v = vec![0f32; n * d];
        let mut s = 7u64; for x in v.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *x = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5; }
        let cc = CellConfirm::build(&v, d, 32, false, HrParams { pat: 0, ..Default::default() });
        let mut sc = cc.scratch();
        for qi in 0..20 {
            let q: Vec<f32> = v[qi * 7 * d..(qi * 7 + 1) * d].iter().map(|x| x + 0.01).collect();
            sc.top.clear();
            cc.confirm(&q, &v, 10, 32, 50.0, &mut sc);
            let mut all: Vec<(f32, u32)> = (0..n).map(|i| (crate::distance::l2_simd(&q, &v[i * d..(i + 1) * d]), i as u32)).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            let got: Vec<u32> = sc.top.iter().map(|x| x.1).collect();
            let want: Vec<u32> = all[..10].iter().map(|x| x.1).collect();
            assert_eq!(got, want);
        }
    }
    #[test]
    fn hr_patience_stop_consistent() {
        // R34：耐心早停 pat ≥ P 等价于不早停；pat=1 时结果仍为升序、精确距离、且不劣于起始 top
        let (n, d) = (3000usize, 40usize);
        let mut v = vec![0f32; n * d];
        let mut s = 11u64; for x in v.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *x = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5; }
        let cc = CellConfirm::build(&v, d, 32, false, HrParams { pat: 0, ..Default::default() });
        let mut sc = cc.scratch();
        for qi in 0..20 {
            let q: Vec<f32> = v[qi * 11 * d..(qi * 11 + 1) * d].iter().map(|x| x + 0.01).collect();
            // 起始 top：任意 10 个点（模拟图段结果）
            let seed: Vec<(f32, u32)> = { let mut t: Vec<(f32, u32)> = (0..10u32).map(|i| { let id = (i * 97 + qi as u32) % n as u32; (crate::distance::l2_simd(&q, &v[id as usize * d..(id as usize + 1) * d]), id) }).collect(); t.sort_by(|a, b| a.0.total_cmp(&b.0)); t };
            sc.top.clear(); sc.top.extend_from_slice(&seed);
            cc.confirm_pat(&q, &v, 10, 16, 1.0, 0, 0.3, &mut sc); let full: Vec<u32> = sc.top.iter().map(|x| x.1).collect();
            sc.top.clear(); sc.top.extend_from_slice(&seed);
            cc.confirm_pat(&q, &v, 10, 16, 1.0, 16, 0.3, &mut sc); let big: Vec<u32> = sc.top.iter().map(|x| x.1).collect();
            assert_eq!(full, big);
            sc.top.clear(); sc.top.extend_from_slice(&seed);
            cc.confirm_pat(&q, &v, 10, 16, 1.0, 1, 0.0, &mut sc);
            assert_eq!(sc.top.len(), 10);
            for w in sc.top.windows(2) { assert!(w[0].0 <= w[1].0); }
            for x in sc.top.iter() { assert_eq!(x.0, crate::distance::l2_simd(&q, &v[x.1 as usize * d..(x.1 as usize + 1) * d])); }
            assert!(sc.top[9].0 <= seed[9].0);
        }
    }
}
