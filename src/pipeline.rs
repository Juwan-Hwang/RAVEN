//! Accepted search stack as one library entry point, shared by `raven_eval --pipeline 1`
//! and the Python binding (ann-benchmarks path), so production gets exactly what was measured.
//!
//! Stack (all accepted 2026-10-29/30 on SIFT1M + GloVe-100):
//! - H09a/H09b: nav dedup + SQ4 upper-layer navigation, trajectory carry into Layer 0
//! - H08b: upper-layer nav rebuilt with m=32, k=16
//! - H17: SQ4 per-dim MSE-adaptive range
//! - H07a: dimension permutation by neighbour-pair energy + exact segmented early abandon
//! - H02: BFS node reorder (ids translated back; results bit-identical)
//! - H20v2: build-time search_degree calibration on neighbour-midpoint queries (no test data)
//! - H13a lookahead prefetch is inside the searcher (RAVEN_LA, default 0 since round 8: GloVe gain <3%)

use crate::bench::util::{bfs_order_pub, remap_sq4, reorder_with};
use crate::build::ChaCha8Rng;
use crate::graph::navigation::LayeredNavigation;
use crate::graph::vamana::{GraphSearcher, L0Opts, NavOpts};
use crate::graph::VamanaGraph;
use crate::quant::SQ4Dataset;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct PipelineOpts {
    pub nav_rebuild_m: usize,
    pub nav_rebuild_k: usize,
    pub sq4_clip: f32,
    pub early_abandon: bool,
    pub perm_near: bool,
    pub reorder: bool,
    /// None = auto-calibrate among `sd_cands`
    pub search_degree: Option<usize>,
    pub sd_cands: Vec<usize>,
    pub sd_calib_n: usize,
    pub sd_calib_recall: f64,
    pub rerank: usize,
    pub po: usize,
    pub k: usize,
    /// R33 HR 模式（SV-3b 高召回确认段，ef ≥ gate 时启用；gate 构建期留一标定，无赢档则不构建）；RAVEN_HR=0/1 覆盖。
    /// 默认只在 x86_64 开：x86 过 CHARTER 档位门控规则（GloVe@0.95 +47.1% 4/4，未启用档逐位一致）；ARM 两次 A/B 未达 (b)（R33-1d）
    pub hr: bool,
    /// HR 单元数（0 = 自动 n/512 取 2 的幂）
    pub hr_nl: usize,
    /// HR 只在 n ≥ hr_min_n 时构建
    pub hr_min_n: usize,
    /// HR k-means 结果缓存文件（None = 不缓存）
    pub hr_cache: Option<String>,
    /// R66：SQ4 小码补零门控的程序化覆盖（None = 读环境变量 RAVEN_SQ4_PAD_SMALL，缺省 16；0 = 关 = 旧 pad32）。
    /// 进程内 A/B（raven_eval RAVEN_ABX）用它在同一进程建 pad32 / pad16 两条管线，不靠改环境变量。
    pub sq4_pad_small: Option<usize>,
    /// R67b/R68 ADC16：低维（码长补到 16 B，dim ≤ 32）时 L0 导航用非对称距离（数据 4 bit 码、查询 f32、逐维步长加权）。
    /// None = 读环境变量 RAVEN_SQ4_ADC（x86_64 缺省 1，aarch64 缺省 0）；进程内 A/B 用 Some(true/false) 程序化设定。
    pub sq4_adc: Option<bool>,
}

impl Default for PipelineOpts {
    fn default() -> Self {
        PipelineOpts {
            nav_rebuild_m: 32,
            nav_rebuild_k: 16,
            sq4_clip: -1.0,
            early_abandon: true,
            perm_near: true,
            reorder: true,
            search_degree: None,
            sd_cands: vec![24, 32, 40, 0],
            sd_calib_n: 300,
            sd_calib_recall: 0.95,
            rerank: 10,
            po: 16,
            k: 10,
            hr: true, // R35：两平台默认开 HR（映射 v2）。ARM 档位门控 (a) 8 轮进程级无回退、(b) GloVe@0.95 配对 6/6 +23…+45%（steal<5%）；x86 自 R33/R34
            hr_nl: 0,
            hr_min_n: 200_000,
            hr_cache: None,
            sq4_pad_small: None,
            sq4_adc: None,
        }
    }
}

pub struct Prepared {
    pub graph: VamanaGraph,
    /// H42b：64B 对齐
    pub vectors: crate::memory::aligned::AlignedBuf<f32>,
    pub sq4: SQ4Dataset,
    pub perm: Vec<usize>,
    pub l0: L0Opts,
    pub nav_opts: NavOpts,
    pub sd: usize,
    pub rerank: usize,
    pub po: usize,
    /// new id -> caller's original id (None when not reordered)
    pub new_to_old: Option<Vec<u32>>,
    /// H53：数据全为 0..=255 整数时的 u8 无损 rerank 副本
    pub vec_u8: Option<crate::memory::aligned::AlignedBuf<u8>>,
    /// R37 LP：低秩投影（m×dim 行主序；空 = 关闭）
    pub proj: Vec<f32>,
    pub proj_m: usize,
    /// R33 HR 模式索引（None = 未构建，searcher 永远走原路径）
    pub hr: Option<crate::graph::cellconfirm::CellConfirm>,
    /// R42 LPX：LP 开启时另存全维 SQ4 码（near 维序），按 ef 计划在高召回段切回全维码
    pub sq4_full: Option<SQ4Dataset>,
    pub perm_full: Vec<usize>,
    pub lpx: Option<LpxPlan>,
    pub lpx_k: usize,
    /// R54 SDX（原创"分段度数计划"）：sd-auto 选了截断度数（sd≠0）时，ef > ef_max 段改走全度数（ef 按召回匹配映射）。
    /// 截断度数在高召回段封顶（fashion sd=40 ef60 0.982，全度数 ef32 0.993），单一 sd 只按 0.95 目标标定。
    pub sdx: Option<LpxPlan>,
    /// R104 hrgate：HR 门槛标定的原路径基线是否为旧的截断度数路径（RAVEN_HRG_SDX=0 或 H217B 关；PLAN 里记 hrg=sd）
    pub hrg_legacy: bool,
    /// R54：SDX 全度数段自己的 LPX 计划（全度数下 LP/全维码切换点不同：fashion sd40 标定 ef_max 60，sd0 标定 ≈25）
    pub lpx_sdx: Option<LpxPlan>,
    /// R68 ADC16 v3：true 时用 [`Prepared::run_auto`] / `GraphSearcher::search_auto_adc` 的独立搜索链（构造时一次决定；
    /// 缺省对称 4 bit 搜索函数与 主线 相同，不含任何 ADC 分支）
    pub adc16: bool,
    /// R84 H85 ADQ（非对称 SQ4 内积导航，单位范数数据；RAVEN_SQ4_ADQ）
    pub adq: bool,
    /// R99 H210c（第 16 棒，代码判据版 H210）：Some((e_sw, rr)) 时 ef ≥ e_sw 段不走 HR，改走原路径 + 整池 f32 精排（rerank 宽度 rr）。
    /// 由 HR 门标定在同一批 LOO 查询上判定（ef ≥ gate 的网格上 H210 同召回耗时 ≤ 0.95×HR 且从 e_sw 起连续赢）；判负 = None，与主线逐位相同。
    /// RAVEN_H210=0 关（与主线逐位相同）；RAVEN_H210_RR 改精排宽度（缺省 100）。
    pub h210: Option<(usize, usize)>,
    /// R102 lpxdet：H217b 开关在 prepare 时读一次 env（原 h217b_on() 在每次 searcher_k 构造 HR 档时读 env）
    pub h217b: bool,
    /// R106 H257（原创“查询流 OOD 守卫”）：SDX 截断段的召回是在库内派生查询（中点/LOO）上标定的；跨模态（OOD）查询流
    /// （laion 文本→图像：sd40 截断段 ef200 测试召回 0.845，中点标定 0.985）上截断段几乎无用。前 OOD_W 个查询观测
    /// 截断段一致度（截断 ef_max 与全度数 2·ef_max 的 top-k 重合，H257b，R106（九）取代距离判据：coco 距离判 OOD 但截断段无损），
    /// 比构建期中点查询的一致度低 > RAVEN_OOD_DELTA（0.05）→ 判 OOD，
    /// 此后 searcher_plain 一律走全度数（不经 SDX 截断段）。R106（八）起缺省开；RAVEN_OOD=0 关（= 0361b5f1 逐位相同）。ID 集判 ID 后计划不变。
    pub ood_ref: f32,
    pub ood_state: std::sync::atomic::AtomicU8,
    pub ood_seen: std::sync::atomic::AtomicU32,
    pub ood_far: std::sync::atomic::AtomicU32,
    /// R106 H257d：LPX（LP 导航码）同代价对照参考 gap 与流累计；lp_off = 判 LP 码在本查询流上失效 → 全维码导航。
    /// sdx_off = SDX 截断段关闭（H257c）。二者独立判定。
    pub ood_ref_lp: f32,
    pub ood_far_lp: std::sync::atomic::AtomicU32,
    pub sdx_off: std::sync::atomic::AtomicBool,
    pub lp_off: std::sync::atomic::AtomicBool,
}

pub const OOD_W: u32 = 64;

/// R42 LPX（原创"双码 ef 计划"）：构建期留一标定 LP 码与全维码各自的 (ef → 召回, 耗时) 曲线；
/// ef ≤ ef_max 走 LP 码（低/中召回段每码字节少），ef > ef_max 时按"LP 在该 ef 的标定召回 → 全维码达到同召回的 ef"映射后走全维码，
/// 使对外 ef 扫描得到两条曲线的下包络，而不是在高召回段被粗码导航拖慢（fashion @0.99 −4%、@0.995 −12% 的修复）。
#[derive(Clone, Debug)]
pub struct LpxPlan {
    pub ef_max: usize,
    /// (ef, 单调化召回) 网格
    pub lp: Vec<(usize, f64)>,
    pub full: Vec<(usize, f64)>,
    /// 全维码 ef / LP ef（切换点召回匹配）
    pub scale: f64,
}

impl LpxPlan {
    fn interp(c: &[(usize, f64)], ef: usize) -> f64 {
        if ef <= c[0].0 { return c[0].1; }
        for w in c.windows(2) { if ef <= w[1].0 { let f = (ef - w[0].0) as f64 / (w[1].0 - w[0].0) as f64; return w[0].1 + f * (w[1].1 - w[0].1); } }
        c[c.len() - 1].1
    }
    /// 全维码 ef：达到 LP@ef 标定召回的最小 ef（网格间按召回线性插值）
    pub fn match_ef(&self, ef: usize, k: usize) -> usize {
        let r = Self::interp(&self.lp, ef);
        if r <= self.full[0].1 { return self.full[0].0.max(k); }
        for w in self.full.windows(2) {
            if r <= w[1].1 {
                let f = if w[1].1 > w[0].1 { (r - w[0].1) / (w[1].1 - w[0].1) } else { 1.0 };
                return ((w[0].0 as f64 + f * (w[1].0 - w[0].0) as f64).ceil() as usize).max(k);
            }
        }
        self.full[self.full.len() - 1].0.max(k)
    }
    /// ef > ef_max 时的全维码 ef = ceil(ef × scale)（scale 在切换点按召回匹配求得；LOO 召回在全维码上易饱和，
    /// 故不逐点匹配，以免高 ef 段映射封顶、对外扫描到不了更高召回）
    /// R49c：切换点之后从匹配 ef 起逐步加（m0 = round(ef_max·scale)，之后每 1/scale 个用户 ef 加 1）。
    /// 旧 ceil(ef·scale) 在 ef_max 刚过时跳过匹配点（fashion ef_max 25、scale 0.44：ef26→12，0.967 的全维 ef11 无用户 ef 可达）。
    pub fn map(&self, ef: usize, k: usize) -> usize {
        if self.ef_max == 0 || self.ef_max == usize::MAX || ef <= self.ef_max { return ((ef as f64 * self.scale).ceil() as usize).max(k); }
        let m0 = (self.ef_max as f64 * self.scale).round() as usize;
        (m0 + ((ef - self.ef_max) as f64 * self.scale).floor() as usize).max(k)
    }
}

impl Prepared {
    /// ef ≥ HR 门槛且已构建 HR 时返回 HR 模式 searcher（图段 ef = E），否则与 [`Self::searcher_plain`] 相同。
    pub fn searcher(&self, ef: usize) -> GraphSearcher<'_> { self.searcher_k(ef, 10) }
    /// R68：按管线构造时定下的导航模式分派（ADC16 → 独立搜索链；否则 = `s.search_auto`，原路径原代码）
    #[inline(always)]
    pub fn run_auto(&self, s: &mut GraphSearcher<'_>, q: &[f32], k: usize) -> Vec<(u32, f32)> { if self.adq { s.search_auto_adq(q, k) } else if self.adc16 { s.search_auto_adc(q, k) } else { s.search_auto(q, k) } }
    /// R68：同 [`Self::run_auto`]，不经 HR（= `s.search_sq4`）
    #[inline(always)]
    pub fn run_sq4(&self, s: &mut GraphSearcher<'_>, q: &[f32], k: usize) -> Vec<(u32, f32)> { if self.adq { s.search_sq4_adq(q, k) } else if self.adc16 { s.search_sq4_adc(q, k) } else { s.search_sq4(q, k) } }

    /// R36-5：k 感知入口（HR 图段 E ≥ k；k ≤ 10 时与 [`Self::searcher`] 完全相同）
    pub fn searcher_k(&self, ef: usize, k: usize) -> GraphSearcher<'_> {
        if let (Some((esw, rr)), Some(_)) = (self.h210, &self.hr) {
            if ef >= esw { let mut s = self.searcher_plain(ef); s.with_rerank_factor(rr); return s; }
        }
        if let Some(cc) = &self.hr {
            if let Some((e, pc)) = cc.plan_k(ef, k) {
                // R99-11 H217b：SDX 计划现在在 HR 开启时也标定（供门槛以下档位用）；HR 档位仍走截断度数原路径，与 H217（630f1166）HR 档逐位相同。
                let mut s = if self.h217b { self.searcher_plain_sd(e) } else { self.searcher_plain(e) };
                s.with_cell_confirm(cc, pc);
                return s;
            }
        }
        self.searcher_plain(ef)
    }

    /// R106 H257：OOD 守卫状态（0 关，1 观测中，2 判 ID，3 判 OOD）
    #[inline(always)]
    pub fn ood_pending(&self) -> bool { self.ood_state.load(std::sync::atomic::Ordering::Relaxed) == 1 }
    /// 观测一个查询的结果；返回 true 表示刚判定为 OOD（调用方须丢弃缓存的 searcher）。
    #[cold] #[inline(never)]
    pub fn ood_observe(&self, q: &[f32], _res: &[(u32, f32)]) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        if self.ood_state.load(Relaxed) != 1 { return false; }
        let (ht, hf, _) = sdx_agree(self, q);
        let (lt, lf, _) = lpx_agree(self, q);
        let far = self.ood_far.fetch_add((hf + 4096 - ht) as u32, Relaxed) + (hf + 4096 - ht) as u32;
        let farl = self.ood_far_lp.fetch_add((lf + 4096 - lt) as u32, Relaxed) + (lf + 4096 - lt) as u32;
        let seen = self.ood_seen.fetch_add(1, Relaxed) + 1;
        if seen < OOD_W { return false; }
        let den = (seen as usize * self.lpx_k.max(1)) as f32;
        let obs = (far as f32 - 4096.0 * seen as f32) / den;
        let obl = (farl as f32 - 4096.0 * seen as f32) / den;
        let so = self.sdx.is_some() && obs > self.ood_ref + ood_tau();
        let lo = lpx_probe_ef(self).is_some() && std::env::var("RAVEN_OOD_LP").map(|v| v != "0").unwrap_or(true) && obl > self.ood_ref_lp + ood_tau();
        if self.ood_state.compare_exchange(1, 2, Relaxed, Relaxed).is_ok() {
            self.sdx_off.store(so, Relaxed); self.lp_off.store(lo, Relaxed);
            eprintln!("R106 OOD guard: stream full-trunc gap {obs:.4} vs ref {:.4}; LP gap {obl:.4} vs ref {:.4} (delta {:.3}) over {seen} q -> sdx {} lp {}", self.ood_ref, self.ood_ref_lp, ood_tau(), if so { "OFF (OOD)" } else { "kept" }, if lo { "OFF (OOD, full codes)" } else { "kept" });
            return so || lo;
        }
        false
    }

    /// 原路径 searcher（不挂 HR）
    pub fn searcher_plain(&self, ef: usize) -> GraphSearcher<'_> {
        if self.sdx_off.load(std::sync::atomic::Ordering::Relaxed) { return self.plain_with(ef, 0, self.lpx_sdx.as_ref().or(self.lpx.as_ref())); }
        if let Some(x) = &self.sdx {
            if ef > x.ef_max { return self.plain_with(x.map(ef, self.lpx_k), 0, self.lpx_sdx.as_ref().or(self.lpx.as_ref())); }
        }
        self.plain_with(ef, self.sd, self.lpx.as_ref())
    }

    /// 原路径 searcher，度数 = self.sd（不经 SDX 计划）
    pub fn searcher_plain_sd(&self, ef: usize) -> GraphSearcher<'_> { self.plain_with(ef, self.sd, self.lpx.as_ref()) }

    /// R54：指定度数与 LPX 计划的原路径 searcher（sd = self.sd、lpx = self.lpx 时与 R42–R53 searcher_plain 逐位相同）
    pub fn plain_with(&self, ef: usize, sd: usize, lpx: Option<&LpxPlan>) -> GraphSearcher<'_> {
        if self.sq4_full.is_some() && self.lp_off.load(std::sync::atomic::Ordering::Relaxed) { let mut s = self.searcher_full(ef); s.with_search_degree(sd); return s; }
        let mut s = match (&self.sq4_full, lpx) {
            (Some(_), Some(pl)) if ef > pl.ef_max => self.searcher_full(pl.map(ef, self.lpx_k)),
            _ => self.searcher_lp(ef),
        };
        s.with_search_degree(sd);
        s
    }

    /// R42 LPX：全维码 searcher（LP 开启且保留全维码时；否则等同 searcher_lp）
    pub fn searcher_full(&self, ef: usize) -> GraphSearcher<'_> {
        let Some(f) = &self.sq4_full else { return self.searcher_lp(ef) };
        let mut s = GraphSearcher::new(&self.vectors, &self.graph, ef);
        s.with_prefetch_offset(self.po);
        s.with_rerank_factor(self.rerank);
        s.with_sq4(f);
        s.with_nav_opts(self.nav_opts);
        s.with_l0_opts(self.l0);
        s.with_search_degree(self.sd);
        if !self.perm_full.is_empty() { s.with_sq4_perm(&self.perm_full); }
        if let Some(v) = &self.vec_u8 { s.with_vec_u8(v); }
        s
    }

    /// 原路径（LP 码若开启）searcher，不经 LPX 计划
    pub fn searcher_lp(&self, ef: usize) -> GraphSearcher<'_> {
        let mut s = GraphSearcher::new(&self.vectors, &self.graph, ef);
        s.with_prefetch_offset(self.po);
        // R37 LP：投影空间码排序更粗，rerank 宽度下限 RAVEN_LP_RR（R99 缺省 20→5：mnist Intel @0.99 +12–30%、ccnews @0.95 +4–12% @0.99 +3–8%，maxrec 不变）
        let rr = if self.proj_m > 0 { self.rerank.max(std::env::var("RAVEN_LP_RR").ok().and_then(|x| x.parse().ok()).unwrap_or(5)) } else { self.rerank };
        s.with_rerank_factor(rr);
        s.with_sq4(&self.sq4);
        s.with_nav_opts(self.nav_opts);
        s.with_l0_opts(self.l0);
        s.with_search_degree(self.sd);
        if self.proj_m > 0 {
            // R42c：投影矩阵已挂在 self.sq4.params.proj 上（search_sq4 自行投影）
            // R95 LPRR-C：LP 导航 + 全维码中间级精排（RAVEN_LPRR_C=1；需 LPX 全维码，缺省关）
            if let Some(f) = &self.sq4_full { if self.adq && crate::graph::vamana::adq::lprr_c_cfg().0 { s.with_rr_full(f, &self.perm_full); } }
        } else if !self.perm.is_empty() {
            s.with_sq4_perm(&self.perm);
        }
        if let Some(v) = &self.vec_u8 { s.with_vec_u8(v); }
        s
    }
    #[inline]
    pub fn orig_id(&self, id: u32) -> u32 {
        match &self.new_to_old { Some(m) => m[id as usize], None => id }
    }
}

/// Dimension order by descending SQ4 code energy over graph-neighbour pairs (H07 "near").
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn near_perm(graph: &VamanaGraph, codes: &[u8], cb: usize, n: usize, dim: usize) -> Vec<usize> {
    let nib = |c: &[u8], d: usize| -> i32 { ((c[d / 2] >> ((d % 2) * 4)) & 0x0F) as i32 };
    let mut e = vec![0f64; dim];
    let step = (n / 20000).max(1);
    let st = graph.storage();
    for i in (0..n).step_by(step) {
        let ci = &codes[i * cb..(i + 1) * cb];
        for &j in st.neighbors(i as u32) {
            let cj = &codes[j as usize * cb..(j as usize + 1) * cb];
            for d in 0..dim {
                let t = nib(ci, d) - nib(cj, d);
                e[d] += (t * t) as f64;
            }
        }
    }
    let mut perm: Vec<usize> = (0..dim).collect();
    perm.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
    perm
}

fn l2(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| { let d = (*x - *y) as f64; d * d }).sum()
}

/// Build the accepted stack on top of a built graph. Vectors must already be in the metric's
/// L2 form (angular: L2-normalised).
pub fn prepare(mut graph: VamanaGraph, vectors: Vec<f32>, dim: usize, o: &PipelineOpts) -> Prepared {
    let n = graph.len();
    // H49 visited 位图：默认开（第 8 轮合入：SIFT1M +9.5% 9/9、GloVe +15.7% 9/9）；RAVEN_VBITS=0 关闭
    if std::env::var("RAVEN_VBITS").map(|v| v != "0").unwrap_or(true) {
        crate::memory::visited::VIS_BITS.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // H135（第 12 轮）：visited 走 L1 常驻哈希集合（RAVEN_VHASH=1；默认关，待 A/B）
    if std::env::var("RAVEN_VHASH").map(|v| v == "1").unwrap_or(false) {
        crate::memory::visited::VIS_HASH.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("RAVEN_VTHP").map(|v| v == "1").unwrap_or(false) {
        crate::memory::visited::VIS_HUGE.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let mut o = o.clone();
    if let Some(v) = std::env::var("RAVEN_NAV_M").ok().and_then(|v| v.parse().ok()) { o.nav_rebuild_m = v; }
    if let Some(v) = std::env::var("RAVEN_NAV_K").ok().and_then(|v| v.parse().ok()) { o.nav_rebuild_k = v; }
    let o = &o;
    // R98 H184：建图期可能未建导航层 → nav_rebuild_m=0 时也补建（m=32,k=16）
    if o.nav_rebuild_m > 0 || graph.layered_nav().is_none() {
        let (m, k) = if o.nav_rebuild_m > 0 { (o.nav_rebuild_m, o.nav_rebuild_k) } else { (32, 16) };
        let t = std::time::Instant::now();
        let nav = LayeredNavigation::build(&vectors, dim, graph.storage(), graph.entry_point(), m, k);
        eprintln!("[prepare] nav m={m} k={k} {:.2}s", t.elapsed().as_secs_f64());
        graph.set_layered_nav(nav);
    }
    if false {
        let nav = LayeredNavigation::build(&vectors, dim, graph.storage(), graph.entry_point(), o.nav_rebuild_m, o.nav_rebuild_k);
        graph.set_layered_nav(nav);
    }
    // R37 LP：RAVEN_LP_FILE=<m×dim f32 行主序投影矩阵>，RAVEN_LP_M=m → SQ4 建在投影空间
    let (proj, proj_m) = load_lp_v(dim, &vectors, o.sq4_clip);
    let (perm, sq4) = if proj_m > 0 {
        let pb = project_all(&vectors, dim, &proj, proj_m);
        let rank: usize = std::env::var("RAVEN_LP_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        // R84 H85b：ADQ 的距离按逐维步长加权在查询侧完成 → LP 码不必共用步长（均匀步长让低方差主成分只剩几级），改逐维拟合。
        // R98 H188：未显式设置 RAVEN_SQ4_ADQ 时，单位范数数据缺省开 ADQ（与下方 adq 判定同一抽样口径）。
        let adq_pre = std::env::var("RAVEN_SQ4_ADQ").map(|v| v != "0").unwrap_or_else(|_| unit_norm_sample(&vectors, dim)) && std::env::var("RAVEN_ADQ_LPU").map(|v| v != "1").unwrap_or(true);
        let mut q = if adq_pre { SQ4Dataset::build_clipped(&pb, proj_m, o.sq4_clip) } else { SQ4Dataset::build_uniform_clipped(&pb, proj_m, o.sq4_clip, rank) };
        if adq_pre { eprintln!("R84 ADQ: LP codes per-dim step (RAVEN_ADQ_LPU=1 restores uniform)"); }
        q.params.proj = proj.clone();
        (Vec::new(), q)
    } else {
    let plain = SQ4Dataset::build_clipped(&vectors, dim, o.sq4_clip);
    let perm: Vec<usize> = if o.perm_near { near_perm(&graph, &plain.codes, plain.code_bytes, n, dim) } else { Vec::new() };
    let sq4 = if o.perm_near { drop(plain); SQ4Dataset::build_permuted_clipped(&vectors, dim, &perm, o.sq4_clip) } else { plain };
    (perm, sq4) };
    // R42 LPX：LP 开启时另建全维码（与非 LP 路径相同的 near 维序 + 截断），供高召回段切换（RAVEN_LPX=0 关）
    let lpx_on = proj_m > 0 && !H178_NOLPX.load(std::sync::atomic::Ordering::Relaxed) && std::env::var("RAVEN_LPX").map(|v| v != "0").unwrap_or(true);
    let full_pre: Option<(Vec<usize>, SQ4Dataset)> = if lpx_on {
        let plain = SQ4Dataset::build_clipped(&vectors, dim, o.sq4_clip);
        if o.perm_near {
            let perm = near_perm(&graph, &plain.codes, plain.code_bytes, n, dim); drop(plain);
            let f = SQ4Dataset::build_permuted_clipped(&vectors, dim, &perm, o.sq4_clip); Some((perm, f))
        } else { Some((Vec::new(), plain)) }
    } else { None };
    let mut l0 = L0Opts::default();
    // H111：RAVEN_L0EA=0 关闭 L0 精确分段提前放弃（结果逐位一致）
    // H111b（第 11 轮，口径修订后）：aarch64 默认关闭（SIFT +9.6–12.5%、GloVe +9.6–13.0%，12/12 点，正反序 3 轮，召回/hops 逐位一致）；
    // R28（x86 s1l：l0ea=0 + novis=2 对 x86 旧默认 3 轮 SIFT +8.1…+12.2%、GloVe +11.6…+16.4%，12/12，召回逐位一致）→ x86 也默认关闭。RAVEN_L0EA=1/0 可覆盖
    let l0ea = o.early_abandon && std::env::var("RAVEN_L0EA").map(|v| v != "0").unwrap_or(false);
    if l0ea && o.perm_near {
        let nchk = dim / 32;
        let seg_mask: u32 = if nchk >= 2 { ((1u64 << (nchk - 1).min(32)) - 1) as u32 } else { 0 };
        l0.early_abandon = true;
        l0.check_mask = seg_mask & 0xFF;
        l0.gran16 = false;
    }
    // H46 距离比终止（实验开关，默认关闭）：RAVEN_STOP_GAMMA / RAVEN_STOP_K / RAVEN_STOP_MIN
    if let Some(g) = std::env::var("RAVEN_STOP_GAMMA").ok().and_then(|v| v.parse::<f32>().ok()) {
        l0.stop_gamma = g;
        l0.stop_k = std::env::var("RAVEN_STOP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        l0.stop_min = std::env::var("RAVEN_STOP_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    }
    let envu = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let nav_opts = NavOpts { dedup: true, sq4: true, carry: envu("RAVEN_NAV_CARRY", 100000), stop_level: envu("RAVEN_NAV_STOP", 1) as _ };
    let (graph, vectors, sq4, new_to_old, full_pre) = if o.reorder {
        let o2n = bfs_order_pub(&graph);
        let r = reorder_with(&graph, &sq4, &vectors, dim, &o2n);
        drop(graph);
        let fp = full_pre.map(|(pm, f)| (pm, remap_sq4(&f, &o2n)));
        (r.graph, r.train, r.sq4, Some(r.new_to_old), fp)
    } else {
        (graph, vectors, sq4, None, full_pre)
    };
    let (perm_full, mut sq4_full) = match full_pre { Some((pm, f)) => (pm, Some(f)), None => (Vec::new(), None) };
    // R65：RAVEN_SQ4_PAD 显式给出时全局生效（旧行为）；否则缺省 32，另有门控 RAVEN_SQ4_PAD_SMALL=N（缺省 0=关）：
    // 原码长 <=N 字节时补到 N（glove-25 13B→16B，R64p5 测试机 召回逐位一致、每查询 −8…−14%，待安静机 A/A 标定后的同轮复核）。
    // 只影响码长 <=N 的低维集；gist/glove-100/sift/fashion 等码长 >16B 不受影响。
    // R66：缺省改为 16（只作用于原码长 <=16B 的集，即 dim<=32；ann-b 中只有 glove-25）。依据 R66 进程内 A/B（测试机 单核，40 reps × 20 块）：
    //   glove-25 召回逐位一致，逐 ef 中位 Δt/q −5.0…−7.0%（95% CI 全 <0），同法 A/A CI 在 ±0.9% 内；sift1m（64B 码）门控不生效，|Δ| ≤1.1%。
    //   RAVEN_SQ4_PAD_SMALL=0 复原旧行为（pad32）。
    let pad_small = o.sq4_pad_small.unwrap_or_else(|| envu("RAVEN_SQ4_PAD_SMALL", 16));
    let pad_for = |cb: usize| -> usize {
        match std::env::var("RAVEN_SQ4_PAD").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(v) => v,
            None => crate::quant::sq4::SQ4Dataset::pad_mult_for(cb, 32, pad_small),
        }
    };
    if let Some(f) = sq4_full.as_mut() { let m = pad_for(f.code_bytes); f.pad_codes(m); }
    let mut sq4 = sq4;
    // H87/H89 接受：码长补零到 32B 倍数（GloVe 50B→64B，每码 1 行）。x86 +31–41%（6/6），
    // ARM（本沙箱 aarch64）独立复核 +10–13%（6/6）；两平台分别验证，非移植。RAVEN_SQ4_PAD=0 关闭
    { let m = pad_for(sq4.code_bytes); sq4.pad_codes(m); }
    // R67b/R68 ADC16（见 PipelineOpts::sq4_adc）：只在 16 B 码、无 LP 投影、L0 无提前放弃时生效。
    // R68 合入：x86_64 缺省开（R67b/R68 预登记门控：glove-25 等召回 QPS +25…+35%，CI 下界 >0；sift/fashion 码长 >16 B 不生效，
    // 缺省搜索函数不变）。aarch64 未测，缺省关。RAVEN_SQ4_ADC=0/1 覆盖。
    let adc16 = {
        let want = o.sq4_adc.unwrap_or_else(|| envu("RAVEN_SQ4_ADC", if cfg!(target_arch = "x86_64") { 1 } else { 0 }) != 0);
        let on = want && sq4.code_bytes == 16 && sq4.params.dim <= 32 && sq4.params.proj.is_empty() && !l0.is_active();
        if want && (on || std::env::var("RAVEN_SQ4_ADC").is_ok() || o.sq4_adc.is_some()) { eprintln!("R68 ADC16 requested: active={} (code_bytes={} dim={} proj={} l0_active={})", on, sq4.code_bytes, sq4.params.dim, !sq4.params.proj.is_empty(), l0.is_active()); }
        on
    };
    // R84 H85 ADQ：单位范数数据（抽样 2000 个范数都在 1±1e-3）+ 无 L0 提前放弃 + 未启用 ADC16 时可开。缺省关（RAVEN_SQ4_ADQ=1 开）。
    let adq = {
        // R98 H188：缺省由「关」改为「单位范数数据自动开」（R98 ny98/ny98c 测试机 nytimes：无 ADQ @0.95 6071/6227 vs ADQ 6920–7524，+12…+21%；
        // 所有 VIBE lane 早已显式 RAVEN_SQ4_ADQ=1）。RAVEN_SQ4_ADQ=0 关，=1 强制请求。
        let unit = unit_norm_sample(&vectors, dim);
        let want = std::env::var("RAVEN_SQ4_ADQ").ok().and_then(|v| v.parse::<usize>().ok()).map(|v| v != 0).unwrap_or(unit);
        let on = want && unit && !adc16 && !l0.is_active();
        if want { eprintln!("R84 ADQ requested: active={} (unit_norm={} adc16={} l0_active={} code_bytes={} proj={})", on, unit, adc16, l0.is_active(), sq4.code_bytes, !sq4.params.proj.is_empty()); }
        on
    };
    let integral = vectors.iter().all(|&x| x >= 0.0 && x <= 255.0 && x.fract() == 0.0);
    let vec_u8 = if integral {
        let b: Vec<u8> = vectors.iter().map(|&x| x as u8).collect();
        Some(crate::memory::aligned::AlignedBuf::from_slice(&b))
    } else { None };
    if let Some(l) = std::env::var("RAVEN_PFL").ok().and_then(|v| v.parse::<usize>().ok()) {
        crate::graph::vamana::PF_LINES.store(l, std::sync::atomic::Ordering::Relaxed);
    } else if cfg!(target_arch = "x86_64") {
        // H73'（R74）：x86_64 上 2 行码（65–128 B，如 nytimes-256 / glove-200）预取整码（R73 测试机 跨进程 nytimes ef160 −6.4%，进程内 −10%）。
        // 在准备阶段一次性决定，热搜索函数（vamana.rs 的 pfl 计算）不改 → 与 主线 逐字节相同（H73 在热函数里加判断 → sift 布局回退 +5…+11%）。
        // 写入 PF_ALL（= usize::MAX − 1，走热函数 `l => l.min(cl).max(1)` 分支 = 整码）：1 行码仍 1 行，≥4 行码仍整码（与 auto 相同），
        // 只有 3 行码（193–256 B）会变成整码 → 本索引含 3 行码时改回 auto（usize::MAX）。
        // ⚠ PF_LINES 是进程全局：同进程先后准备多个索引时以最后一次为准；后准备的 3 行码索引会把先前 2 行码索引退回 main 行为（只影响速度，不影响结果）。
        // aarch64 不变（ARM nytimes ef40 整码预取回退 3.9%）。RAVEN_PFL=1 恢复旧行为。
        const PF_ALL: usize = usize::MAX - 1;
        let cls = [Some(sq4.code_bytes), sq4_full.as_ref().map(|f| f.code_bytes)];
        let has = |n: usize| cls.iter().flatten().any(|&b| b.div_ceil(64) == n);
        if has(3) { crate::graph::vamana::PF_LINES.store(usize::MAX, std::sync::atomic::Ordering::Relaxed); }
        else if has(2) { crate::graph::vamana::PF_LINES.store(PF_ALL, std::sync::atomic::Ordering::Relaxed); }
    }
    if std::env::var("RAVEN_U8RR").map(|v| v == "1").unwrap_or(false) {
        crate::graph::refiner::U8_RR.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let vectors: crate::memory::aligned::AlignedBuf<f32> = vectors.into();
    // R84 H85c：ADQ + RAVEN_ADQ_NU=1 → 主码（非 LP）用共享非均匀 16 级重编码（GINGER/TurboQuant 式 Lloyd–Max 级，但不旋转：R84 oracle 旋转无益）。
    let mut sq4 = sq4;
    if adq && sq4.params.proj.is_empty() && nu_wanted(&vectors, dim) {
        let v: &[f32] = &vectors; let pm = &perm;
        sq4.nu_reencode(&|i: usize| { let r = &v[i * dim..(i + 1) * dim]; if pm.is_empty() { r.to_vec() } else { pm.iter().map(|&d| r[d]).collect() } });
    }
    // R84 H86 PNR：ADQ + RAVEN_ADQ_PNR=z（>0）→ 8 bit 码拆高/低 nibble，导航对池门槛附近的候选才读低 nibble。
    // R84 H86c：LP（投影码）也可 PNR：码维序向量 = 投影 proj·v（RAVEN_ADQ_PNR_LP=0 关）。
    let pnr_lp_ok = sq4.params.proj.is_empty() || std::env::var("RAVEN_ADQ_PNR_LP").map(|v| v != "0").unwrap_or(true);
    if adq && pnr_lp_ok && crate::quant::sq4::NU_TABLE.get().is_none() {
        if let Some(z) = std::env::var("RAVEN_ADQ_PNR").ok().and_then(|v| v.parse::<f32>().ok()).filter(|&z| z > 0.0) {
            let v: &[f32] = &vectors; let pm = &perm; let pj: Vec<f32> = sq4.params.proj.clone(); let pmm = if pj.is_empty() { 0 } else { pj.len() / dim };
            let lo = sq4.pnr_reencode(&|i: usize| { let r = &v[i * dim..(i + 1) * dim];
                if pmm > 0 { (0..pmm).map(|j| crate::pipeline::dot_f32(&pj[j * dim..(j + 1) * dim], r)).collect() }
                else if pm.is_empty() { r.to_vec() } else { pm.iter().map(|&d| r[d]).collect() } }, o.sq4_clip);
            let lo: &'static crate::memory::aligned::AlignedBuf<u8> = Box::leak(Box::new(crate::memory::aligned::AlignedBuf::<u8>::from_slice(&lo)));
            let _ = crate::quant::sq4::PNR.set(crate::quant::sq4::PnrState { hi_base: sq4.codes.as_ptr() as usize, lo_base: lo.as_ptr() as usize, z, dim: sq4.params.dim, proj_len: sq4.params.proj.len() });
            eprintln!("R84 PNR active: z={} code_bytes={}", z, sq4.code_bytes);
        }
    }
    // R85 H102 NER：RAVEN_NER=1（缺省关，待门控）→ ADQ 码做范数均衡取整（主码；LPX 全维码同样处理）。在 PNR/ANL 之前不适用 NU。
    if adq && crate::quant::sq4::NU_TABLE.get().is_none() && crate::quant::sq4::PNR.get().is_none() && ner_enabled(dim) {
        let t = std::time::Instant::now();
        let v: &[f32] = &vectors; let pm = &perm; let pj: Vec<f32> = sq4.params.proj.clone(); let pmm = if pj.is_empty() { 0 } else { pj.len() / dim };
        let a = sq4.ner_reencode(&|i: usize| { let r = &v[i * dim..(i + 1) * dim];
            if pmm > 0 { (0..pmm).map(|j| crate::pipeline::dot_f32(&pj[j * dim..(j + 1) * dim], r)).collect() }
            else if pm.is_empty() { r.to_vec() } else { pm.iter().map(|&d| r[d]).collect() } });
        let pf = &perm_full;
        let b = sq4_full.as_mut().map(|f| f.ner_reencode(&|i: usize| { let r = &v[i * dim..(i + 1) * dim]; if pf.is_empty() { r.to_vec() } else { pf.iter().map(|&d| r[d]).collect() } }));
        eprintln!("R85 NER (target, flips/pt, |N-T| before, after): main={:?} full={:?} ({:.1}s)", a, b, t.elapsed().as_secs_f64());
    }
    // R85 H101 ANL：ADQ 距离补上 ‖x̂‖² 项（码空间真非对称 L2），RAVEN_ADQ_L2=1 开（缺省关，待门控）。主码与 LPX 全维码各登记一份。
    if adq && crate::quant::sq4::NU_TABLE.get().is_none() && std::env::var("RAVEN_ADQ_L2").map(|v| v == "1").unwrap_or(false) {
        let t = std::time::Instant::now();
        let lo: Option<&[u8]> = crate::quant::sq4::PNR.get().filter(|p| p.hi_base == sq4.codes.as_ptr() as usize)
            .map(|p| unsafe { std::slice::from_raw_parts(p.lo_base as *const u8, sq4.n * sq4.code_bytes) });
        let a = sq4.anl_register(lo);
        let b = sq4_full.as_ref().map(|f| f.anl_register(None));
        eprintln!("R85 ANL active: main={} full={:?} pnr_lo={} ({:.1}s)", a, b, lo.is_some(), t.elapsed().as_secs_f64());
    }
    let mut p = Prepared { graph, vectors, sq4, perm, l0, nav_opts, sd: 0, rerank: o.rerank, po: o.po, new_to_old, vec_u8, proj, proj_m, hr: None, sq4_full, perm_full, lpx: None, lpx_k: o.k, sdx: None, lpx_sdx: None, adc16, adq, h210: None, h217b: h217b_on(), hrg_legacy: false, ood_ref: 0.0, ood_state: std::sync::atomic::AtomicU8::new(0), ood_seen: std::sync::atomic::AtomicU32::new(0), ood_far: std::sync::atomic::AtomicU32::new(0), ood_ref_lp: 0.0, ood_far_lp: std::sync::atomic::AtomicU32::new(0), sdx_off: std::sync::atomic::AtomicBool::new(false), lp_off: std::sync::atomic::AtomicBool::new(false) };
    // R97 H173 RRQ：紧凑 rerank 码（RAVEN_RRQ=8|12；缺省不装，rerank 路径不变）
    if let Some(bits) = std::env::var("RAVEN_RRQ").ok().and_then(|v| v.parse::<u32>().ok()).filter(|&b| b == 8 || b == 12) {
        let t = std::time::Instant::now();
        let top = envu("RAVEN_RRQ_F32TOP", 0);
        let r = crate::graph::refiner::Rrq::build(&p.vectors, dim, bits, top);
        eprintln!("R97 RRQ active: bits={} f32top={} bytes={} ({:.1}s)", bits, top, r.bytes(), t.elapsed().as_secs_f64());
        crate::graph::refiner::rrq_install(r);
    }
    // H81：po 自动校准（x86 默认开；RAVEN_POAUTO=0/1 覆盖）。与 sd 校准共用中点查询与真值
    let poauto = std::env::var("RAVEN_POAUTO").map(|v| v == "1").unwrap_or(false); // H85 复测撤销 H82b：收益仅 1–5%，默认关
    // R97 NV-3：低维路由图（建图时已按 lowd_route 取 R96），未显式设 RAVEN_SD 时搜索度取 96（满度附近；R97u6b：s128 ≈ s96 −1…−3%）
    let route_sd = if o.search_degree.is_none() && lowd_route(&p.vectors, dim) {
        let nn = p.graph.len(); let st = (nn / 4096).max(1);
        let mx = (0..nn).step_by(st).map(|i| p.graph.neighbors(i as u32).len()).max().unwrap_or(0);
        let sd = envu("RAVEN_LOWD_SD", 96);
        if mx > 64 { eprintln!("R97 NV-3 lowd route: graph max outdeg {mx} -> sd {sd}"); Some(sd) } else { eprintln!("R97 NV-3 lowd route: graph max outdeg {mx} <= 64 (not a routed build) -> sd calibrate"); None }
    } else { None };
    p.sd = match (o.search_degree.or(route_sd), poauto) {
        (Some(s), false) => s,
        (None, false) => calibrate_sd(&p, dim, o),
        (sd, true) => {
            let mut oo = o.clone();
            if let Some(s) = sd { oo.sd_cands = vec![s]; }
            let (sdv, po) = calibrate_sd_po(&p, dim, &oo);
            p.po = po;
            sdv
        }
    };
    if p.sq4_full.is_some() { p.lpx = Some(calibrate_lpx(&p, dim, o, p.sd, None)); }
    // R104 hrgate：SDX 计划先于 HR 门槛标定，门槛的原路径基线 = 实际服务该 ef 的原路径（截断度数 + SDX 全度数段）。
    // 旧顺序下基线是纯截断度数路径（yahoo-384 k=100：sd32 要 ef800 才到 0.990，实际 SDX 路径 ef≈175）→ HR 对稻草人判赢 → gate 130→KL 20，
    // @0.99 0.64–0.66× Glass（HR_GATE=0：1.27–1.35×）。SDX 标定不依赖 HR（plain_with 不挂 HR），所以计划本身不变。
    // RAVEN_HRG_SDX=0 恢复旧顺序（PLAN 记 hrg=sd）；H217B 关时 SDX 只在 HR 关时标定，必须在门槛之后 → 也走旧顺序。
    let sdx_first = p.h217b && std::env::var("RAVEN_HRG_SDX").map(|v| v != "0").unwrap_or(true);
    let sdx_env = std::env::var("RAVEN_SDX").map(|v| v != "0").unwrap_or(true);
    if sdx_first && p.sd != 0 && sdx_env { sdx_stage(&mut p, dim, o); }
    let hr_on = std::env::var("RAVEN_HR").map(|v| v != "0").unwrap_or(o.hr);
    if hr_on && n >= o.hr_min_n {
        p.hr = Some(build_hr(&p.vectors, dim, o));
        if std::env::var("RAVEN_HR_GATE").is_err() {
            p.hrg_legacy = !sdx_first;
            let (g, h210) = calibrate_hr_gate(&p, dim, o);
            p.h210 = if g > 0 && !kl_on(o.k, dim) { h210 } else { None };
            // R43-3 KL：标定只决定开/关（门槛网格 200–600 处 KL 计划是否赢）；开启时 KL 计划从 ef ≥ 20 起生效
            let g = if g > 0 && kl_on(o.k, dim) { 20 } else { g };
            if let Some(cc) = p.hr.as_mut() { cc.params.gate = g; }
            if g == 0 { p.hr = None; }
        }
    }
    // R99-11 H217b：旧条件 p.hr.is_none() 使 HR 一旦保留（门槛 >0），门槛以下档位就丢了 SDX 全度数计划 → 与门槛 0 的旧路径不逐位相同
    // （nytimes H217 gate 250 时 @0.85 0.83–0.91×）。H217b 开时不论 HR 是否保留都标定 SDX；RAVEN_H217B_OFF=1 恢复旧条件。
    if !sdx_first && p.sd != 0 && (p.hr.is_none() || p.h217b) && sdx_env { sdx_stage(&mut p, dim, o); }
    // R102：一行 PLAN（全部影响结果的构建期决策）+ 其 FNV-1a 64 哈希；同一输入两次 prepare 必须相同
    eprintln!("PLAN {} hash={:016x}", p.plan_string(), p.plan_hash());
    p
}

/// R54/R104：SDX 标定段（全度数段 LPX + SDX ef_max/scale）。R104 起缺省在 HR 门槛标定之前执行。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
fn sdx_stage(p: &mut Prepared, dim: usize, o: &PipelineOpts) {
    {
        // 全度数段 LPX：LP 须便宜 ≥10% 才判赢（RAVEN_SDX_LPXM）。中点标定高估粗码（LP）召回：fashion LP 中点 ef40 0.990 → 测试 ≈0.97，
        // 全维码中点 ef14 0.996 → 测试 0.9925；严格判据在 ef25 处 LP/全维 ≈0.91–1.05 随噪声翻转（ef_max 20/25/∞），∞ 时 @0.99 掉 ≈30%
        if p.sq4_full.is_some() { let m = std::env::var("RAVEN_SDX_LPXM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.9); p.lpx_sdx = Some(calibrate_lpx(p, dim, o, 0, Some(m))); }
        p.sdx = calibrate_sdx(p, dim, o);
        if p.sdx.is_none() { p.lpx_sdx = None; }
        if (p.sdx.is_some() || lpx_probe_ef(p).is_some()) && std::env::var("RAVEN_OOD").map(|v| v != "0").unwrap_or(true) { let (r, rl) = ood_reference(p, dim); p.ood_ref = r; p.ood_ref_lp = rl; p.ood_state.store(1, std::sync::atomic::Ordering::Relaxed); }
    }
}

fn h217b_on() -> bool { std::env::var("RAVEN_H217B_OFF").is_err() }

impl Prepared {
    /// R102：构建期计划的规范字符串（sd、po、rerank、LP 维数、HR 门槛、H210c e_sw@rr、LPX/SDX/LPX-SDX 的 ef_max 与 scale、ADC16/ADQ）。
    /// po 只影响预取（结果逐位不变），也列出以便对照。
    #[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
    pub fn plan_string(&self) -> String {
        let lp = |x: &Option<LpxPlan>| match x { Some(l) => format!("{}@{:.6}", if l.ef_max == usize::MAX { "inf".to_string() } else { l.ef_max.to_string() }, l.scale), None => "none".into() };
        let gate = match self.hr.as_ref() { Some(c) => c.params.gate.to_string(), None => "off".into() };
        // R102 RC2：H210c（e_sw@rr；HR 关时 H210 不生效，记 off）
        let h210 = match (self.h210, &self.hr) { (Some((e, rr)), Some(_)) => format!("{e}@{rr}"), _ => "off".into() };
        // R104：旧门槛基线（截断度数）只在显式回退时记 hrg=sd；缺省（门槛基线含 SDX）字符串格式不变
        let hrg = if self.hrg_legacy && self.hr.is_some() { " hrg=sd" } else if self.ood_state.load(std::sync::atomic::Ordering::Relaxed) != 0 { " ood=1" } else { "" };
        format!("n={} sd={} po={} rerank={} proj_m={} hr_gate={} h210={} lpx={} sdx={} lpx_sdx={} adc16={} adq={}{}",
            self.graph.len(), self.sd, self.po, self.rerank, self.proj_m, gate, h210, lp(&self.lpx), lp(&self.sdx), lp(&self.lpx_sdx), self.adc16 as u8, self.adq as u8, hrg)
    }
    #[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
    pub fn plan_hash(&self) -> u64 {
        self.plan_string().bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
    }
}

#[cold] #[inline(never)]
fn ood_tau() -> f32 { std::env::var("RAVEN_OOD_DELTA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.03) }

/// R106 H257c：同代价对照——截断度数 ef_max 与全度数 map(ef_max)（SDX 标定认为二者召回相当）各自与全度数 2·ef_max 结果的
/// top-k 重合数；返回 (截断, 全度数, k)。差值反映"截断段在这些查询上是否比标定时更差"，与查询难度本身无关。
pub fn sdx_agree(p: &Prepared, q: &[f32]) -> (usize, usize, usize) {
    let x = match &p.sdx { Some(x) => x, None => return (0, 0, 0) };
    let k = p.lpx_k.max(1);
    let full = p.lpx_sdx.as_ref().or(p.lpx.as_ref());
    let mut r = p.plain_with((2 * x.ef_max).max(2 * k), 0, full);
    let mut rb: Vec<u32> = p.run_auto(&mut r, q, k).iter().take(k).map(|r| r.0).collect();
    rb.sort_unstable();
    let mut a = p.plain_with(x.ef_max.max(k), p.sd, p.lpx.as_ref());
    let ta = p.run_auto(&mut a, q, k).iter().take(k).filter(|r| rb.binary_search(&r.0).is_ok()).count();
    let mut f = p.plain_with(x.map(x.ef_max, k), 0, full);
    let tf = p.run_auto(&mut f, q, k).iter().take(k).filter(|r| rb.binary_search(&r.0).is_ok()).count();
    (ta, tf, k)
}

/// R106 H257d：LP 码探测 ef——LPX 生效（有全维码、LP 计划 ef_max > 0）时取 LP 段内的一个 ef：ef_max（∞ 时 2k）。
pub fn lpx_probe_ef(p: &Prepared) -> Option<usize> {
    let pl = p.lpx.as_ref()?; p.sq4_full.as_ref()?;
    if pl.ef_max == 0 { return None; }
    let k = p.lpx_k.max(1);
    // R106 H257d-fix：强制 ∞ 常以大有限值给出（RAVEN_LPX_EFMAX=99999999），4·ef_p 会分配数百 GB → 按 ∞ 处理。
    Some(if pl.ef_max == usize::MAX || pl.ef_max >= 1 << 16 { 2 * k } else { pl.ef_max.max(k) })
}

/// R106 H257d：LP 码同代价对照——LP 码 ef_p 与全维码 match_ef(ef_p)（LPX 标定认为同召回）各与全维码 4·ef_p 结果的 top-k 重合数。
/// 跨模态查询（coco 文本→图像，LP = 图像库 PCA192）上 LP 投影丢掉查询的主要能量，LP 导航失效；库内标定看不出来。
pub fn lpx_agree(p: &Prepared, q: &[f32]) -> (usize, usize, usize) {
    let (Some(ep), Some(pl)) = (lpx_probe_ef(p), p.lpx.as_ref()) else { return (0, 0, 0) };
    let k = p.lpx_k.max(1);
    let mut g = p.searcher_full(4 * ep); g.with_search_degree(p.sd);
    let mut rb: Vec<u32> = p.run_auto(&mut g, q, k).iter().take(k).map(|r| r.0).collect();
    rb.sort_unstable();
    let mut a = p.searcher_lp(ep); a.with_search_degree(p.sd);
    let ta = p.run_auto(&mut a, q, k).iter().take(k).filter(|r| rb.binary_search(&r.0).is_ok()).count();
    let mut f = p.searcher_full(pl.match_ef(ep, k)); f.with_search_degree(p.sd);
    let tf = p.run_auto(&mut f, q, k).iter().take(k).filter(|r| rb.binary_search(&r.0).is_ok()).count();
    (ta, tf, k)
}

/// R106 H257b：构建期参考一致度（128 个 seed-106 中点查询，与 SDX 标定同一种库内派生查询）。
#[cold] #[inline(never)]
pub fn ood_reference(p: &Prepared, dim: usize) -> (f32, f32) {
    use rand::RngCore;
    use rayon::prelude::*;
    let t0 = Instant::now();
    let n = p.graph.len(); let base = &p.vectors;
    let mut rng = ChaCha8Rng::seed_from(106);
    let mut qv: Vec<f32> = Vec::with_capacity(128 * dim);
    for _ in 0..128 {
        let qi = (rng.next_u64() % n as u64) as usize;
        let nb = p.graph.neighbors(qi as u32);
        let j = if nb.is_empty() { qi } else { nb[(rng.next_u64() as usize) % nb.len()] as usize };
        qv.extend((0..dim).map(|d| 0.5 * (base[qi * dim + d] + base[j * dim + d])));
    }
    let (ht, hf, t) = (0..128usize).into_par_iter().map(|i| sdx_agree(p, &qv[i * dim..(i + 1) * dim])).reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    let r = if t == 0 { 0.0 } else { (hf as f32 - ht as f32) / t as f32 };
    let (lt, lf, tl) = (0..128usize).into_par_iter().map(|i| lpx_agree(p, &qv[i * dim..(i + 1) * dim])).reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    let rl = if tl == 0 { 0.0 } else { (lf as f32 - lt as f32) / tl as f32 };
    eprintln!("R106 OOD ref: full-trunc agreement gap {r:.4} (trunc {:.4}, full {:.4}); LP gap {rl:.4} (lp {:.4}, full {:.4}, probe ef {:?}) over 128 mid q ({:.2}s)", ht as f32 / t.max(1) as f32, hf as f32 / t.max(1) as f32, lt as f32 / tl.max(1) as f32, lf as f32 / tl.max(1) as f32, lpx_probe_ef(p), t0.elapsed().as_secs_f64());
    (r, rl)
}

/// R54 SDX 标定：中点查询（与 sd/LPX 标定同法，不碰测试数据）上测截断度数 A = searcher_plain_sd 与全度数 B 的 (ef → 召回, 耗时)，
/// 判定规则与 LPX 相同（A 在同召回下耗时 ≤ B 插值耗时则判赢；首个判负处为 ef_max）；全网格都赢 → None（行为与旧版逐位相同）。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn calibrate_sdx(p: &Prepared, dim: usize, o: &PipelineOpts) -> Option<LpxPlan> {
    use rand::RngCore;
    use rayon::prelude::*;
    let t0 = Instant::now();
    let n = p.graph.len();
    let nc: usize = std::env::var("RAVEN_SDX_CALN").ok().and_then(|v| v.parse().ok()).unwrap_or(if n <= 200_000 { 1000 } else { 500 }).min(n);
    let k = o.k.max(1);
    let base = &p.vectors;
    let mut rng = ChaCha8Rng::seed_from(54);
    let mut qv: Vec<f32> = Vec::with_capacity(nc * dim);
    for _ in 0..nc {
        let qi = (rng.next_u64() % n as u64) as usize;
        let nb = p.graph.neighbors(qi as u32);
        let j = if nb.is_empty() { qi } else { nb[(rng.next_u64() as usize) % nb.len()] as usize };
        qv.extend((0..dim).map(|d| 0.5 * (base[qi * dim + d] + base[j * dim + d])));
    }
    let thr: Vec<f32> = (0..nc).into_par_iter().map(|ci| {
        let q = &qv[ci * dim..(ci + 1) * dim];
        let mut top: Vec<f32> = Vec::with_capacity(k + 2);
        for j in 0..n {
            let d = crate::distance::l2_simd(q, &base[j * dim..(j + 1) * dim]);
            if top.len() < k { top.push(d); if top.len() == k { top.sort_by(|a, b| a.total_cmp(b)); } }
            else if d < top[k - 1] { top[k - 1] = d; top.sort_by(|a, b| a.total_cmp(b)); }
        }
        if top.len() < k { top.sort_by(|a, b| a.total_cmp(b)); }
        top[top.len() - 1] * (1.0 + 1e-5) + 1e-6
    }).collect();
    let mk = |ef: usize, full: bool| if full { p.plain_with(ef, 0, p.lpx_sdx.as_ref().or(p.lpx.as_ref())) } else { p.plain_with(ef, p.sd, p.lpx.as_ref()) };
    // R102：召回与 R43-2 计数代价同一遍得到（召回遍 + novis 关计数遍，均为固定 seed-54 中点查询），逐位确定；
    // 旧版用墙钟计时（严格 margin 1.0），yahoo-384 ef150 处 ±2% 噪声使 ef_max 在 150/120 间翻转。RAVEN_CAL=time 恢复旧计时路径。
    let (timed, dump) = cal_mode();
    let cm = CostModel::platform();
    let eval = |ef: usize, full: bool| -> (f64, f64, [f64; COST_FEATS]) {
        let mut s = mk(ef, full); let mut h = 0usize;
        for ci in 0..nc {
            let q = &qv[ci * dim..(ci + 1) * dim];
            h += p.run_auto(&mut s, q, k).iter().take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
        let mut s = mk(ef, full); let mut fs = [0f64; COST_FEATS];
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; let _ = p.run_auto(&mut s, q, k);
            let f = CostModel::feats(&s, k, dim, 0); for i in 0..COST_FEATS { fs[i] += f[i] / nc as f64; } }
        crate::graph::vamana::set_novis(nv);
        (h as f64 / (nc * k) as f64, cm.cost(&fs) * 1e-3, fs)
    };
    let time_of = |ef: usize, full: bool| -> f64 {
        let mut s = mk(ef, full); let mut sink = 0usize;
        let t = Instant::now();
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; sink += p.run_auto(&mut s, q, k).len(); }
        let el = t.elapsed().as_secs_f64(); std::hint::black_box(sink);
        el * 1e6 / nc as f64
    };
    let efs: Vec<usize> = [10usize, 12, 15, 20, 25, 30, 40, 60, 80, 100, 150].iter().map(|&m| (m * k + 9) / 10).collect();
    let mono = |v: Vec<f64>| -> Vec<f64> { let mut m = 0f64; v.into_iter().map(|x| { m = m.max(x); m }).collect() };
    let ea: Vec<(f64, f64, [f64; COST_FEATS])> = efs.iter().map(|&e| eval(e, false)).collect();
    let eb: Vec<(f64, f64, [f64; COST_FEATS])> = efs.iter().map(|&e| eval(e, true)).collect();
    let ar = mono(ea.iter().map(|x| x.0).collect());
    let br = mono(eb.iter().map(|x| x.0).collect());
    let (mut at, mut bt): (Vec<f64>, Vec<f64>) = (ea.iter().map(|x| x.1).collect(), eb.iter().map(|x| x.1).collect());
    if timed || dump {
        let rounds: usize = std::env::var("RAVEN_SDX_CALR").ok().and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
        let (mut ta, mut tb) = (vec![f64::MAX; efs.len()], vec![f64::MAX; efs.len()]);
        for rd in 0..rounds {
            let order: Vec<usize> = if rd % 2 == 0 { (0..efs.len()).collect() } else { (0..efs.len()).rev().collect() };
            for &i in &order { ta[i] = ta[i].min(time_of(efs[i], false)); tb[i] = tb[i].min(time_of(efs[i], true)); }
        }
        if dump {
            for (i, &e) in efs.iter().enumerate() {
                eprintln!("R43 CALDUMP sdx dim={dim} sd{} ef={e} us={:.3} model={:.3} feats={}", p.sd, ta[i], at[i], fmt_feats(&ea[i].2));
                eprintln!("R43 CALDUMP sdx dim={dim} full ef={e} us={:.3} model={:.3} feats={}", tb[i], bt[i], fmt_feats(&eb[i].2));
            }
        }
        if timed { at = ta; bt = tb; }
    }
    let mut btm = bt.clone(); for i in (0..btm.len().saturating_sub(1)).rev() { btm[i] = btm[i].min(btm[i + 1]); }
    let b_t_at = |r: f64| -> Option<f64> {
        if r <= br[0] { return Some(btm[0]); }
        for i in 1..efs.len() { if r <= br[i] { let f = if br[i] > br[i - 1] { (r - br[i - 1]) / (br[i] - br[i - 1]) } else { 1.0 };
            return Some((btm[i - 1].ln() + f * (btm[i].ln() - btm[i - 1].ln())).exp()); } }
        None
    };
    let br_max = br[br.len() - 1];
    // R102：模型模式下 A 须便宜 ≥5% 才判赢（迟滞，边界点不随模型误差翻转）；计时模式保留旧的严格 1.0
    let margin: f64 = std::env::var("RAVEN_SDX_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if timed { 1.0 } else { 0.95 });
    let mut ef_max = 0usize; let mut msg = String::new(); let mut streak = true;
    for i in 0..efs.len() {
        // 两条曲线都饱和（≥0.999 且 A 已达 B 的最高召回）时无区分力，沿用前一判定
        let sat = br_max >= 0.999 && ar[i] >= br_max - 1e-9;
        let w = if sat { streak } else { match b_t_at(ar[i]) { Some(t) => at[i] <= margin * t, None => false } };
        msg += &format!(" ef{}: sd{} {:.4}/{:.1}us full {:.4}/{:.1}us{}", efs[i], p.sd, ar[i], at[i], br[i], bt[i], if w { " W" } else { "" });
        if streak && w { ef_max = efs[i]; } else { streak = false; }
    }
    if streak { ef_max = usize::MAX; }
    if let Some(v) = std::env::var("RAVEN_SDX_EFMAX").ok().and_then(|v| v.parse().ok()) { ef_max = v; }
    eprintln!("R54 SDX calib ({}, {nc} mid q, k={k}, margin {margin}, {:.1}s):{msg} -> ef_max {ef_max}", if timed { "timed" } else { "cost model" }, t0.elapsed().as_secs_f64());
    if ef_max == usize::MAX { return None; }
    let mut pl = LpxPlan { ef_max, lp: efs.iter().cloned().zip(ar).collect(), full: efs.iter().cloned().zip(br).collect(), scale: 1.0 };
    if ef_max > 0 { pl.scale = (pl.match_ef(ef_max, 1) as f64 / ef_max as f64).min(1.0); }
    if let Some(v) = std::env::var("RAVEN_SDX_SCALE").ok().and_then(|v| v.parse().ok()) { pl.scale = v; }
    eprintln!("R54 SDX plan: ef_max {} scale {:.3}", pl.ef_max, pl.scale);
    Some(pl)
}

/// R43-2 确定性构建期代价模型（取代计时：同一输入两次构建得到相同门控）。
/// 每查询代价（ns）= Σ c_i·f_i，特征全部来自搜索本身的计数（与计时无关、逐位确定）：
///   f0 v（图段取码节点数 = visited）、f1 v·码行数（码字节/64）、f2 v·ef/64（线性池插入移位）、
///   f3 n_r（精算行数 = rerank min(k·rf, ef) + HR 幸存者）、f4 n_r·行数（4·dim/64）、
///   f5 HR 单元扫描流式行数、f6 HR 扫描单元数、f7 HR 单元打分 NL·dim/64、f8 LP 查询投影 m·dim/64。
/// 系数按平台由离线回归给出（R43-2 拟合，每数据集自由尺度，只拟相对代价；见 LOG），RAVEN_CAL_COEF 可覆盖。
/// RAVEN_CAL=time 恢复旧的计时标定；RAVEN_CAL_DUMP=1 同时计时并打印特征（拟合用）。
#[derive(Clone, Copy, Debug)]
pub struct CostModel { pub c: [f64; COST_FEATS] }
pub const COST_FEATS: usize = 9;
impl CostModel {
    #[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
    pub fn platform() -> Self {
        #[cfg(target_arch = "x86_64")]
        let c = COEF_X86;
        #[cfg(not(target_arch = "x86_64"))]
        let c = COEF_ARM;
        let mut m = CostModel { c };
        if let Ok(v) = std::env::var("RAVEN_CAL_COEF") {
            let xs: Vec<f64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if xs.len() == COST_FEATS { m.c.copy_from_slice(&xs); }
        }
        m
    }
    /// 一次搜索后的特征（顺序与系数一致）
    #[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
    pub fn feats(s: &GraphSearcher, k: usize, dim: usize, nl: usize) -> [f64; COST_FEATS] {
        let (ef, rf, cb, pe) = s.cost_shape();
        let v = s.last_visited_count() as f64;
        let rl = (4 * dim) as f64 / 64.0;
        let mut nr = (k * rf).min(ef.max(k)) as f64;
        let (mut sl, mut nc, mut nlf) = (0f64, 0f64, 0f64);
        if let Some((sur, ncell, slots, sb)) = s.hr_last_counts() { nr += sur as f64; sl = (slots * sb) as f64 / 64.0; nc = ncell as f64; nlf = (nl * dim) as f64 / 64.0; }
        [v, v * cb as f64 / 64.0, v * ef as f64 / 64.0, nr, nr * rl, sl, nc, nlf, pe as f64 / 64.0]
    }
    pub fn cost(&self, f: &[f64; COST_FEATS]) -> f64 { (0..COST_FEATS).map(|i| self.c[i] * f[i]).sum() }
}
/// R43-2 系数（ns；[v, v·码行, v·ef/64, n_r, n_r·行, 扫描行, 单元, NL·dim/64, 投影行]）。
/// x86：五台测试机 dump（GloVe-100/nytimes HR 门槛、fashion/mnist/gist/arxiv LPX、gist HR）每数据集自由尺度 NNLS，中位相对误差 3.5%；
/// 用它重放计时 dump 的门控决策：GloVe 200、nytimes 200、gist HR 0、gist/mnist LPX ∞ 与计时一致；fashion LPX 40（计时两次分别 ∞ / 60）。
/// aarch64：尚无 ARM dump，暂用同一组相对系数（ARM 上 HR 门控的 HR/原路径比 0.26–0.55，远离 0.95 判据，不敏感）。
pub const COEF_X86: [f64; COST_FEATS] = [0.0, 15.27, 2.703, 0.0, 10.06, 5.206, 0.0, 1.869, 2.697];
pub const COEF_ARM: [f64; COST_FEATS] = COEF_X86;
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
fn cal_mode() -> (bool, bool) {
    let timed = std::env::var("RAVEN_CAL").map(|v| v == "time").unwrap_or(false);
    let dump = std::env::var("RAVEN_CAL_DUMP").map(|v| v == "1").unwrap_or(false);
    (timed, dump)
}
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
fn fmt_feats(f: &[f64; COST_FEATS]) -> String { f.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(",") }

/// R33：HR 门槛标定（不碰测试数据）。留一查询：抽 nc 个库内点作查询、精确 top-(k+1) 去掉自身作真值；
/// 原路径在 ef 网格上的 (召回, µs/查询) 曲线 vs HR 在候选门槛 g 处的 (召回, µs)；
/// HR 在 g 处"赢" = 其耗时 ≤ 0.95 × 原路径在同召回处的插值耗时（召回超出原路径网格时与网格最大 ef 比）。
/// 门槛 = 最小的 g，使 g 及所有更大的候选都赢；没有则返回 0（不启用）。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn calibrate_hr_gate(p: &Prepared, dim: usize, o: &PipelineOpts) -> (usize, Option<(usize, usize)>) {
    use rand::RngCore;
    use rayon::prelude::*;
    let t0 = Instant::now();
    let cc = p.hr.as_ref().expect("hr");
    let n = p.graph.len();
    let nc: usize = std::env::var("RAVEN_HR_CALN").ok().and_then(|v| v.parse().ok()).unwrap_or(500).min(n);
    let k = o.k;
    let base = &p.vectors;
    let mut rng = ChaCha8Rng::seed_from(33);
    let qids: Vec<usize> = (0..nc).map(|_| (rng.next_u64() % n as u64) as usize).collect();
    // 真值：去掉自身后的第 k 名距离（并列容差）
    let thr: Vec<f32> = qids.par_iter().map(|&qi| {
        let q = &base[qi * dim..(qi + 1) * dim];
        let mut top: Vec<f32> = Vec::with_capacity(k + 2);
        for j in 0..n {
            if j == qi { continue; }
            let d = crate::distance::l2_simd(q, &base[j * dim..(j + 1) * dim]);
            if top.len() < k { top.push(d); top.sort_by(|a, b| a.total_cmp(b)); }
            else if d < top[k - 1] { top[k - 1] = d; top.sort_by(|a, b| a.total_cmp(b)); }
        }
        top[k - 1] * (1.0 + 1e-5) + 1e-6
    }).collect();
    // R34：噪声稳健标定。召回逐位确定（只算一次）；耗时按 R 轮（默认 3，RAVEN_HR_CALR）交错测量，
    // 每轮把全部原路径 ef 与全部 HR 候选各测一遍（奇数轮倒序），逐配置取最小耗时（steal/抢占只会加时，最小值稳健）。
    // 另加召回判据：原路径在 ef=200 的 LOO 召回 ≥ 0.999（易数据集，如 SIFT）→ 直接 gate 0，不比耗时。
    let mk = |ef: usize, hr: bool| { if hr { let (e, pc) = cc.plan_force(ef); let e = if cc.params.ek { e.max(k + 1) } else { e }; let mut s = if p.h217b { p.searcher_plain_sd(e) } else { p.searcher_plain(e) }; s.with_cell_confirm(cc, pc); s } else { p.searcher_plain(ef) } };
    let nl_hr = cc.nl;
    let feats_of = std::cell::RefCell::new(std::collections::HashMap::<(usize, bool), [f64; COST_FEATS]>::new());
    let recall_of = |ef: usize, hr: bool| -> f64 {
        let mut s = mk(ef, hr); let mut h = 0usize; let mut fs = [0f64; COST_FEATS];
        for (ci, &qi) in qids.iter().enumerate() {
            let q = &base[qi * dim..(qi + 1) * dim];
            let r = p.run_auto(&mut s, q, k + 1);
            h += r.iter().filter(|x| x.0 as usize != qi).take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        // 计数遍：novis 关（visited 跟踪 = 取码节点数；novis 路径不计数），只取特征、不影响召回
        let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
        let mut s = mk(ef, hr);
        for &qi in &qids { let q = &base[qi * dim..(qi + 1) * dim]; let _ = p.run_auto(&mut s, q, k + 1);
            let f = CostModel::feats(&s, k + 1, dim, nl_hr); for i in 0..COST_FEATS { fs[i] += f[i] / nc as f64; } }
        crate::graph::vamana::set_novis(nv);
        feats_of.borrow_mut().insert((ef, hr), fs);
        h as f64 / (nc * k) as f64
    };
    let time_of = |ef: usize, hr: bool| -> f64 {
        let mut s = mk(ef, hr); let mut sink = 0usize;
        let t = Instant::now();
        for &qi in &qids { let q = &base[qi * dim..(qi + 1) * dim]; sink += p.run_auto(&mut s, q, k + 1).len(); }
        let el = t.elapsed().as_secs_f64(); std::hint::black_box(sink);
        el * 1e6 / nc as f64
    };
    let pefs = [60usize, 80, 100, 130, 160, 200, 250, 300, 400, 500, 600, 800];
    // R62：门槛网格下探到 60（R61n nytimes：网格下限 200 让 plain 占住 0.93–0.95 段，而 HR ef200 已快于 plain ef160 且召回更高）。
    // RAVEN_HR_GATES=逗号列表 覆盖（=200,250,300,400,500,600 复原 R34 网格）。
    let gates: Vec<usize> = std::env::var("RAVEN_HR_GATES").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![130usize, 160, 200]);
    // R99 H182c：两台机器固定门扫描（测试机 Zen4 R98gt182c_z5、另一台测试机 Zen5 gt182c）都是 130 全档最优或同级；200 在 另一台测试机 @0.94–0.95 慢 14–15%，
    // 300/450 在两台 @0.97 慢 8–38%。按 ef 分段的门表没有必要：网格收到 130–200。
    // R98 H182：缺省网格回到 200 起（测试机 Zen4 nytimes R98ny182_z5：LOO 标定选 gate 60/80，测试查询上 @0.90/0.93/0.95 比固定 gate 200
    // 慢 13–17% / 27–37% / 10–22%，g200 在 0.90–0.99 全档 ≥ auto 与 HR 关；LOO 低 ef 召回系统性高估 → 低门槛误判赢）。
    let rounds: usize = std::env::var("RAVEN_HR_CALR").ok().and_then(|v| v.parse().ok()).unwrap_or(3).max(1);
    let p_rec: Vec<f64> = pefs.iter().map(|&ef| recall_of(ef, false)).collect();
    let easy_r = p_rec[pefs.iter().position(|&e| e == 200).unwrap()];
    if easy_r >= 0.999 {
        eprintln!("R34 HR gate calib ({nc} LOO q, {:.1}s): plain recall@ef200 {easy_r:.4} >= 0.999 (easy) -> gate 0", t0.elapsed().as_secs_f64());
        return (0, None);
    }
    let g_rec: Vec<f64> = gates.iter().map(|&g| recall_of(g, true)).collect();
    let (timed, dump) = cal_mode();
    let cm = CostModel::platform();
    let fo = feats_of.borrow().clone();
    let model_us = |ef: usize, hr: bool| -> f64 { cm.cost(&fo[&(ef, hr)]) * 1e-3 };
    let mut p_t: Vec<f64> = pefs.iter().map(|&e| model_us(e, false)).collect();
    let mut g_t: Vec<f64> = gates.iter().map(|&g| model_us(g, true)).collect();
    if timed || dump {
        let mut tp = vec![f64::MAX; pefs.len()]; let mut tg = vec![f64::MAX; gates.len()];
        let mut cfgs: Vec<(bool, usize)> = Vec::new();
        for i in 0..pefs.len().max(gates.len()) { if i < pefs.len() { cfgs.push((false, i)); } if i < gates.len() { cfgs.push((true, i)); } }
        for rd in 0..rounds {
            let it: Vec<&(bool, usize)> = if rd % 2 == 0 { cfgs.iter().collect() } else { cfgs.iter().rev().collect() };
            for &&(hr, i) in &it {
                if hr { tg[i] = tg[i].min(time_of(gates[i], true)); } else { tp[i] = tp[i].min(time_of(pefs[i], false)); }
            }
        }
        if dump {
            for (i, &e) in pefs.iter().enumerate() { eprintln!("R43 CALDUMP hr_gate dim={dim} plain ef={e} us={:.3} model={:.3} feats={}", tp[i], p_t[i], fmt_feats(&fo[&(e, false)])); }
            for (i, &g) in gates.iter().enumerate() { eprintln!("R43 CALDUMP hr_gate dim={dim} hr ef={g} us={:.3} model={:.3} feats={}", tg[i], g_t[i], fmt_feats(&fo[&(g, true)])); }
        }
        if timed { p_t = tp; g_t = tg; }
    }
    let plain: Vec<(f64, f64)> = p_rec.iter().cloned().zip(p_t.iter().cloned()).collect();
    let t_at = |r: f64| -> f64 {
        // 原路径达到召回 r 的插值耗时（召回按网格单调化）
        let mut best_r = 0f64; let mut pts: Vec<(f64, f64)> = Vec::new();
        for &(rr, tt) in &plain { if rr > best_r { best_r = rr; pts.push((rr, tt)); } }
        if r <= pts[0].0 { return pts[0].1; }
        for w in pts.windows(2) { let ((r0, t0), (r1, t1)) = (w[0], w[1]);
            if r <= r1 { let f = (r - r0) / (r1 - r0); return (t0.ln() * (1.0 - f) + t1.ln() * f).exp(); } }
        // R99-10 H217：r 超出原路径网格最高召回时，旧代码返回最后一个网格点的耗时（钳位），等于拿 HR 在更高召回处的耗时
        // 去比原路径在更低召回处的耗时，HR 系统性吃亏（nytimes：g300 r=0.976 1859us 对 plain ef800 r=0.955 1887us 判负 → gate 0）。
        // 改为在 (-ln(1-r), ln t) 空间用最后两个单调点外推；外推不可用时视为原路径到不了（+inf）。RAVEN_H217_OFF=1 恢复旧行为。
        if std::env::var("RAVEN_H217_OFF").is_ok() { return pts.last().unwrap().1; }
        if pts.len() < 2 { return f64::INFINITY; }
        let ((r0, t0), (r1, t1)) = (pts[pts.len() - 2], pts[pts.len() - 1]);
        let (x0, x1, x) = (-(1.0 - r0).max(1e-9).ln(), -(1.0 - r1).max(1e-9).ln(), -(1.0 - r).max(1e-9).ln());
        if x1 - x0 < 1e-6 || t1 <= t0 { return f64::INFINITY; }
        let slope = (t1.ln() - t0.ln()) / (x1 - x0);
        (t1.ln() + slope * (x - x1)).exp()
    };
    let mut wins = Vec::new(); let mut msg = String::new();
    for (i, &g) in gates.iter().enumerate() {
        let (r, t) = (g_rec[i], g_t[i]); let tp = t_at(r);
        let w = t <= 0.95 * tp; wins.push(w);
        msg += &format!(" g{g}: hr r={r:.4} {t:.0}us vs plain {tp:.0}us{}", if w { " WIN" } else { "" });
    }
    let mut gate = 0;
    for i in (0..gates.len()).rev() { if wins[i] { gate = gates[i]; } else { break; } }
    let base_s = match &p.sdx { Some(x) => format!("sd{}+sdx{}@{:.3}", p.sd, x.ef_max, x.scale), None => format!("sd{}", p.sd) };
    eprintln!("R34 HR gate calib ({nc} LOO q, {}, {:.1}s, plain base {base_s}): plain {} |{msg} -> gate {gate}", if timed { "timed" } else { "cost model" }, t0.elapsed().as_secs_f64(),
              pefs.iter().zip(&plain).map(|(e, (r, t))| format!("ef{e}:{r:.4}/{t:.0}us")).collect::<Vec<_>>().join(" "));
    // R99 H210c：门开（gate>0）时，在同一批 LOO 查询上比 ef ≥ gate 段的 HR 与“原路径 + 整池 f32 精排”（H210）。
    // 判据：H210 在网格点 e 的代价 ≤ margin × HR 在同召回处的插值代价（HR 到不了该召回也算赢）；e_sw = 从网格末端往前连续赢的最小 e。
    // R102 RC2：代价缺省改用 R43-2 计数模型（召回遍 + novis 关计数遍，固定 seed-33 LOO 查询 = HR 门同一批），逐位确定；
    // margin 缺省 0.95（RAVEN_H210_MARGIN 覆盖）。RAVEN_CAL=time 恢复旧的实测路径（R 轮交错取最小）；RAVEN_CAL_DUMP=1 同时计时并打印特征。
    // RAVEN_H210_FORCE=1 跳过判定、强制 e_sw = gate（=N>1 时 e_sw = N），用于非恒等冒烟；RAVEN_H210=0 优先（关）。判负 → None（与主线逐位相同）。
    let h210_on = std::env::var("RAVEN_H210").map(|v| v != "0").unwrap_or(true);
    if gate == 0 || !h210_on { return (gate, None); }
    let rr: usize = std::env::var("RAVEN_H210_RR").ok().and_then(|v| v.parse().ok()).unwrap_or(100).max(1);
    if let Some(fv) = std::env::var("RAVEN_H210_FORCE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v >= 1) {
        let esw = if fv > 1 { fv } else { gate };
        eprintln!("R99 H210c calib (rr {rr}): RAVEN_H210_FORCE={fv} -> switch ef {esw} (forced, no calibration)");
        return (gate, Some((esw, rr)));
    }
    let t1 = Instant::now();
    let es: Vec<usize> = [2usize, 3, 4, 6, 8, 12].iter().map(|&m| gate * m / 2).collect();
    let mk_f = |ef: usize| { let mut s = p.searcher_plain(ef); s.with_rerank_factor(rr); s };
    let eval_g = |hr: bool, ef: usize| -> (f64, f64, [f64; COST_FEATS]) {
        let mut h = 0usize;
        let mut s = if hr { mk(ef, true) } else { mk_f(ef) };
        for (ci, &qi) in qids.iter().enumerate() {
            let q = &base[qi * dim..(qi + 1) * dim];
            let r = p.run_auto(&mut s, q, k + 1);
            h += r.iter().filter(|x| x.0 as usize != qi).take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        // 计数遍（同 HR 门 / SDX）：novis 关，只取特征，不影响召回
        let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
        let mut s = if hr { mk(ef, true) } else { mk_f(ef) }; let mut fs = [0f64; COST_FEATS];
        for &qi in &qids { let q = &base[qi * dim..(qi + 1) * dim]; let _ = p.run_auto(&mut s, q, k + 1);
            let f = CostModel::feats(&s, k + 1, dim, if hr { nl_hr } else { 0 }); for i in 0..COST_FEATS { fs[i] += f[i] / nc as f64; } }
        crate::graph::vamana::set_novis(nv);
        (h as f64 / (nc * k) as f64, cm.cost(&fs) * 1e-3, fs)
    };
    let tim_g = |hr: bool, ef: usize| -> f64 {
        let mut s = if hr { mk(ef, true) } else { mk_f(ef) }; let mut sink = 0usize;
        let t = Instant::now();
        for &qi in &qids { let q = &base[qi * dim..(qi + 1) * dim]; sink += p.run_auto(&mut s, q, k + 1).len(); }
        let el = t.elapsed().as_secs_f64(); std::hint::black_box(sink);
        el * 1e6 / nc as f64
    };
    let eh: Vec<(f64, f64, [f64; COST_FEATS])> = es.iter().map(|&e| eval_g(true, e)).collect();
    let ef_: Vec<(f64, f64, [f64; COST_FEATS])> = es.iter().map(|&e| eval_g(false, e)).collect();
    let hr_r: Vec<f64> = eh.iter().map(|x| x.0).collect();
    let f_r: Vec<f64> = ef_.iter().map(|x| x.0).collect();
    let mut hr_t: Vec<f64> = eh.iter().map(|x| x.1).collect();
    let mut f_t: Vec<f64> = ef_.iter().map(|x| x.1).collect();
    if timed || dump {
        let (mut th, mut tf) = (vec![f64::MAX; es.len()], vec![f64::MAX; es.len()]);
        for rd in 0..rounds {
            let idx: Vec<usize> = if rd % 2 == 0 { (0..es.len()).collect() } else { (0..es.len()).rev().collect() };
            for &i in &idx { th[i] = th[i].min(tim_g(true, es[i])); tf[i] = tf[i].min(tim_g(false, es[i])); }
        }
        if dump {
            for (i, &e) in es.iter().enumerate() {
                eprintln!("R43 CALDUMP h210 dim={dim} hr ef={e} us={:.3} model={:.3} feats={}", th[i], hr_t[i], fmt_feats(&eh[i].2));
                eprintln!("R43 CALDUMP h210 dim={dim} f rr={rr} ef={e} us={:.3} model={:.3} feats={}", tf[i], f_t[i], fmt_feats(&ef_[i].2));
            }
        }
        if timed { hr_t = th; f_t = tf; }
    }
    let margin: f64 = std::env::var("RAVEN_H210_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.95);
    let hr_at = |r: f64| -> Option<f64> {
        let mut best_r = 0f64; let mut pts: Vec<(f64, f64)> = Vec::new();
        for i in 0..es.len() { if hr_r[i] > best_r { best_r = hr_r[i]; pts.push((hr_r[i], hr_t[i])); } }
        if pts.is_empty() || r > pts.last().unwrap().0 { return None; }
        if r <= pts[0].0 { return Some(pts[0].1); }
        for w in pts.windows(2) { let ((r0, t0), (r1, t1)) = (w[0], w[1]);
            if r <= r1 { let f = (r - r0) / (r1 - r0); return Some((t0.ln() * (1.0 - f) + t1.ln() * f).exp()); } }
        None
    };
    let mut fw = Vec::new(); let mut m2 = String::new();
    for i in 0..es.len() {
        let th = hr_at(f_r[i]); let w = match th { None => true, Some(t) => f_t[i] <= margin * t };
        fw.push(w);
        m2 += &format!(" ef{}: hr {:.4}/{:.1}us f {:.4}/{:.1}us{}", es[i], hr_r[i], hr_t[i], f_r[i], f_t[i], if w { " WIN" } else { "" });
    }
    let mut esw = 0usize;
    for i in (0..es.len()).rev() { if fw[i] { esw = es[i]; } else { break; } }
    let out = if esw > 0 { Some((esw, rr)) } else { None };
    eprintln!("R99 H210c calib ({}, rr {rr}, margin {margin}, {:.1}s):{m2} -> {}", if timed { "timed" } else { "cost model" }, t1.elapsed().as_secs_f64(), match out { Some((e, _)) => format!("switch ef {e}"), None => "off".into() });
    (gate, out)
}

/// R42 LPX 标定（不碰测试数据）：中点查询（库内点与随机图邻居的中点，精确 top-k），ef 网格 = k×{1,1.2,1.5,2,2.5,3,4,6,8,10}；
/// LP 码与全维码各测召回（一次）与耗时（5 轮交错取最小；1000 查询）。LP 网格点"赢" = 其耗时 ≤ 全维码在同召回处的插值耗时
/// （召回低于全维最小 ef 的召回时与全维最小 ef 的耗时比）。ef_max = 从最小 ef 起连续赢的最大网格 ef（首点即输 → 0，全部走全维；全网格都赢 → 无上限）。
/// RAVEN_LPX_EFMAX 覆盖 ef_max（仍做召回标定以得映射）。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn calibrate_lpx(p: &Prepared, dim: usize, o: &PipelineOpts, sd: usize, margin_ovr: Option<f64>) -> LpxPlan {
    use rand::RngCore;
    use rayon::prelude::*;
    let t0 = Instant::now();
    let n = p.graph.len();
    let nc: usize = std::env::var("RAVEN_LPX_CALN").ok().and_then(|v| v.parse().ok()).unwrap_or(if n <= 200_000 { 1000 } else { 500 }).min(n);
    let k = o.k.max(1);
    let base = &p.vectors;
    let mut rng = ChaCha8Rng::seed_from(42);
    // 中点查询（库内点与随机图邻居的中点；与 sd 标定同法）：比留一库内点更接近测试查询难度，避免全维码召回在 1.0 饱和
    let mut qv: Vec<f32> = Vec::with_capacity(nc * dim);
    for _ in 0..nc {
        let qi = (rng.next_u64() % n as u64) as usize;
        let nb = p.graph.neighbors(qi as u32);
        let j = if nb.is_empty() { qi } else { nb[(rng.next_u64() as usize) % nb.len()] as usize };
        qv.extend((0..dim).map(|d| 0.5 * (base[qi * dim + d] + base[j * dim + d])));
    }
    let thr: Vec<f32> = (0..nc).into_par_iter().map(|ci| {
        let q = &qv[ci * dim..(ci + 1) * dim];
        let mut top: Vec<f32> = Vec::with_capacity(k + 2);
        for j in 0..n {
            let d = crate::distance::l2_simd(q, &base[j * dim..(j + 1) * dim]);
            if top.len() < k { top.push(d); if top.len() == k { top.sort_by(|a, b| a.total_cmp(b)); } }
            else if d < top[k - 1] { top[k - 1] = d; top.sort_by(|a, b| a.total_cmp(b)); }
        }
        if top.len() < k { top.sort_by(|a, b| a.total_cmp(b)); }
        top[top.len() - 1] * (1.0 + 1e-5) + 1e-6
    }).collect();
    let mk = |ef: usize, full: bool| { let mut s = if full { p.searcher_full(ef) } else { p.searcher_lp(ef) }; s.with_search_degree(sd); s };
    let feats_of = std::cell::RefCell::new(std::collections::HashMap::<(usize, bool), [f64; COST_FEATS]>::new());
    let recall_of = |ef: usize, full: bool| -> f64 {
        let mut s = mk(ef, full); let mut h = 0usize; let mut fs = [0f64; COST_FEATS];
        for ci in 0..nc {
            let q = &qv[ci * dim..(ci + 1) * dim];
            let r = p.run_auto(&mut s, q, k);
            h += r.iter().take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
        let mut s = mk(ef, full);
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; let _ = p.run_auto(&mut s, q, k);
            let f = CostModel::feats(&s, k, dim, 0); for i in 0..COST_FEATS { fs[i] += f[i] / nc as f64; } }
        crate::graph::vamana::set_novis(nv);
        feats_of.borrow_mut().insert((ef, full), fs);
        h as f64 / (nc * k) as f64
    };
    let time_of = |ef: usize, full: bool| -> f64 {
        let mut s = mk(ef, full); let mut sink = 0usize;
        let t = Instant::now();
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; sink += p.run_auto(&mut s, q, k).len(); }
        let el = t.elapsed().as_secs_f64(); std::hint::black_box(sink);
        el * 1e6 / nc as f64
    };
    let efs: Vec<usize> = [10usize, 12, 15, 20, 25, 30, 40, 60, 80, 100].iter().map(|&m| (m * k + 9) / 10).collect();
    let mono = |v: Vec<f64>| -> Vec<f64> { let mut m = 0f64; v.into_iter().map(|x| { m = m.max(x); m }).collect() };
    let lr = mono(efs.iter().map(|&e| recall_of(e, false)).collect());
    let fr = mono(efs.iter().map(|&e| recall_of(e, true)).collect());
    let rounds: usize = std::env::var("RAVEN_LPX_CALR").ok().and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    let (timed0, dump) = cal_mode();
    // R48d：全维码整码预取（≥4 行）后，R43-2 代价模型的"每码行"线性代价对全维码失真（fashion 全维实测 ≈ 模型 0.47×、LP ≈0.8×），
    // 单一 β 无法同时贴合 fashion/mnist/arxiv（β=0.08 mnist −20%，β=0.6 arxiv −30%）。LPX 判定改为默认计时（RAVEN_CAL=model 退回模型）。
    let full_pf = { let cl = (dim / 2).div_ceil(64); let p = crate::graph::vamana::PF_LINES.load(std::sync::atomic::Ordering::Relaxed); cl >= 4 && (p == usize::MAX || p >= cl) };
    let model_forced = std::env::var("RAVEN_CAL").map(|v| v == "model").unwrap_or(false);
    // R102 lpxdet：整码预取下不再默认计时（fashion 中点 ef30 处 LP/全维 计时比 0.91–1.0 随噪声翻转 → ef_max 25 / ∞ 逐进程随机，
    // 测试集上 25 在 @0.99 快 23–25%）。改为确定性"预取校正计数模型"（pf_model）：LP 模型代价 × κ（RAVEN_LPX_PFK，缺省 1.45）
    // 再与全维模型代价比，margin 缺省 0.95。κ 来自 测试机 fashion784 CALDUMP：全维 计时/模型 0.58–0.73、LP 0.77–0.86
    // （比值 ≈1.3–1.5；R48d 的 0.47×/0.8× 同向）。fashion：模型比 ef25 0.628、ef30 0.694 → ×1.45 = 0.91 W / 1.006 L → ef_max 25；
    // mnist（sd-auto 24 / sd0）模型比 ≤0.44 → 判定与计时相同。RAVEN_CAL=time 恢复计时；RAVEN_CAL=model 恢复 R43-2 纯模型（margin 0.85、无 κ）。
    // R102 coldcal/kappa：κ 与机器相关（arxiv-300k 同数据 Zen4 需 κ<1.34、Zen5 需 κ>1.52；ccnews Zen5 >1.71），
    // 只对单行 LP 码（proj_m·4/8 ≤ 64 B，即 m ≤ 128：fashion/mnist）用确定性 pf_model；多行 LP 码（m 192/256：
    // gist/landmark/arxiv/ccnews）退回 lpxdet 之前的缺省（整码预取下计时 + R49d 饱和）。
    let (pf_model, timed) = lpx_cal_kind(full_pf, timed0, model_forced, p.proj_m);
    let cm = CostModel::platform();
    let fo = feats_of.borrow().clone();
    let mut lt: Vec<f64> = efs.iter().map(|&e| cm.cost(&fo[&(e, false)]) * 1e-3).collect();
    let mut ft: Vec<f64> = efs.iter().map(|&e| cm.cost(&fo[&(e, true)]) * 1e-3).collect();
    if timed || dump {
        let mut tl = vec![f64::MAX; efs.len()]; let mut tf = vec![f64::MAX; efs.len()];
        for rd in 0..rounds {
            let order: Vec<usize> = if rd % 2 == 0 { (0..efs.len()).collect() } else { (0..efs.len()).rev().collect() };
            for &i in &order { tl[i] = tl[i].min(time_of(efs[i], false)); tf[i] = tf[i].min(time_of(efs[i], true)); }
        }
        if dump {
            for (i, &e) in efs.iter().enumerate() {
                eprintln!("R43 CALDUMP lpx dim={dim} lp ef={e} us={:.3} model={:.3} feats={}", tl[i], lt[i], fmt_feats(&fo[&(e, false)]));
                eprintln!("R43 CALDUMP lpx dim={dim} full ef={e} us={:.3} model={:.3} feats={}", tf[i], ft[i], fmt_feats(&fo[&(e, true)]));
            }
        }
        if timed { lt = tl; ft = tf; }
    }
    let pfk: f64 = if pf_model { std::env::var("RAVEN_LPX_PFK").ok().and_then(|v| v.parse().ok()).unwrap_or(LPX_PF_KAPPA) } else { 1.0 };
    if !timed && pfk != 1.0 { for x in lt.iter_mut() { *x *= pfk; } }
    // 全维码同召回耗时（耗时按召回单调化：更高召回的耗时不低于更低召回的）
    let mut ftm = ft.clone(); for i in (0..ftm.len().saturating_sub(1)).rev() { ftm[i] = ftm[i].min(ftm[i + 1]); }
    let full_t_at = |r: f64| -> Option<f64> {
        if r <= fr[0] { return Some(ftm[0]); }
        for i in 1..efs.len() { if r <= fr[i] { let f = if fr[i] > fr[i - 1] { (r - fr[i - 1]) / (fr[i] - fr[i - 1]) } else { 1.0 };
            return Some((ftm[i - 1].ln() + f * (ftm[i].ln() - ftm[i - 1].ln())).exp()); } }
        None
    };
    let fr_max = fr[fr.len() - 1];
    // R43-2：代价模型误差约 ±10%，模型模式下 LP 须便宜 ≥15% 才判赢（fashion ef80–100 处 LP/全维 ≈0.9 的边界点不再随噪声翻转）
    let margin: f64 = match margin_ovr { Some(m) => m, None => std::env::var("RAVEN_LPX_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if timed { 1.0 } else if pf_model { LPX_PF_MARGIN } else { 0.85 }) };
    let mut ef_max = 0usize; let mut msg = String::new();
    let mut streak = true;
    for i in 0..efs.len() {
        // LP 判赢 = 耗时 ≤ 全维码同召回插值耗时 × RAVEN_LPX_MARGIN（默认 1.0，严格）；
        // LP 召回已达全维码网格最高召回且 ≥0.999（饱和、无区分力）时沿用前一判定
        // R49d：整码预取下默认计时判定时，LP 的 LOO 召回 ≥0.99 也视为饱和（LOO 下全维码比 LP 更早饱和，近 1 的召回差
        // 无区分力；测试机 mnist 计时 ef_max 曾在 20/25/30/∞ 间翻转，实测 ∞ 最优；R49b 的 10% 迟滞把 fashion 推到 25–30，弃用）
        let sat = lpx_sat(fr_max, lr[i], full_pf, model_forced);
        let w = if sat { streak } else { match full_t_at(lr[i]) { Some(t) => lt[i] <= margin * t, None => false } };
        msg += &format!(" ef{}: lp {:.4}/{:.0}us full {:.4}/{:.0}us{}", efs[i], lr[i], lt[i], fr[i], ft[i], if w { " W" } else { "" });
        if streak && w { ef_max = efs[i]; } else { streak = false; }
    }
    // 全网格都赢 → 不设上限（网格外不切换；否则 gist/mnist 这类 LP 全程占优的集在 ef>10k 时会被错误切回全维码）
    if streak { ef_max = usize::MAX; }
    if let Some(v) = std::env::var("RAVEN_LPX_EFMAX").ok().and_then(|v| v.parse().ok()) { ef_max = v; }
    let mode = if timed && !timed0 { format!("timed default, LP m{} multi-line", p.proj_m) } else if timed { "timed".to_string() } else if pf_model { format!("pf count model k{pfk}") } else { "count model".to_string() };
    eprintln!("R42 LPX calib ({mode}, margin {margin}, sd {sd}, {nc} LOO q, k={k}, {:.1}s):{msg} -> ef_max {ef_max}", t0.elapsed().as_secs_f64());
    let mut pl = LpxPlan { ef_max, lp: efs.iter().cloned().zip(lr).collect(), full: efs.iter().cloned().zip(fr).collect(), scale: 1.0 };
    if ef_max > 0 && ef_max < usize::MAX { pl.scale = (pl.match_ef(ef_max, 1) as f64 / ef_max as f64).min(1.0); }
    if let Some(v) = std::env::var("RAVEN_LPX_SCALE").ok().and_then(|v| v.parse().ok()) { pl.scale = v; }
    eprintln!("R42 LPX plan: ef_max {} scale {:.3}", pl.ef_max, pl.scale);
    pl
}

/// R102 coldcal/kappa：LPX 标定方式 → (pf_model, timed)。
/// pf_model（确定性 κ 计数模型）只在整码预取、非 RAVEN_CAL=time/model、且 LP 码只占一行（proj_m·4/8 ≤ 64 B）时用；
/// 整码预取下其余情况（多行 LP 码，或 RAVEN_CAL=time）= 计时（lpxdet 之前的缺省）；RAVEN_CAL=model = 纯 R43-2 模型。
pub(crate) fn lpx_cal_kind(full_pf: bool, timed0: bool, model_forced: bool, proj_m: usize) -> (bool, bool) {
    let pf_model = full_pf && !timed0 && !model_forced && proj_m * 4 / 8 <= 64;
    let timed = timed0 || (full_pf && !model_forced && !pf_model);
    (pf_model, timed)
}
/// R49d 饱和判定（R102 kappa 修正）：全维码网格饱和，或整码预取下（RAVEN_CAL=model 除外）LP LOO 召回 ≥0.99。
/// 旧子句 `full_pf && !timed0` 使 RAVEN_CAL=time 关掉饱和、不等于 lpxdet 之前的缺省（fashion T1=30 即此伪影）。
pub(crate) fn lpx_sat(fr_max: f64, lr: f64, full_pf: bool, model_forced: bool) -> bool {
    (fr_max >= 0.999 && lr >= fr_max - 1e-9) || (full_pf && !model_forced && lr >= 0.99)
}

/// R102 lpxdet：整码预取（≥4 行）下 LPX 计数模型的 LP 代价校正 κ 与判赢 margin（见 calibrate_lpx）
pub const LPX_PF_KAPPA: f64 = 1.45;
pub const LPX_PF_MARGIN: f64 = 0.95;

/// R43-3 KL 默认门槛（k ≥ 此值启用大 k 计划）：x86_64 50；aarch64 0（未在 ARM 验证，按 CHARTER 逐平台门控）
#[cfg(target_arch = "x86_64")]
pub const KL_DEFAULT: usize = 50;
#[cfg(not(target_arch = "x86_64"))]
pub const KL_DEFAULT: usize = 0;
/// R43 接力-3：KL 只在 dim ≥ RAVEN_HR_KL_MINDIM（默认 128）启用。测试机 GloVe-100 k=100 KL/R42 @0.95 三轮 +3.1/+2.9/+1.9%
/// 未过每轮 ≥+3% 门槛；VIBE glove-200（dim 200）三轮 +30…+233% 过。
pub fn kl_on(k: usize, dim: usize) -> bool {
    let kl: usize = std::env::var("RAVEN_HR_KL").ok().and_then(|v| v.parse().ok()).unwrap_or(KL_DEFAULT);
    let mind: usize = std::env::var("RAVEN_HR_KL_MINDIM").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    kl > 0 && k >= kl && dim >= mind
}

/// R33：构建 HR 模式索引（k-means 结果可缓存到 o.hr_cache；缓存按内部序保存中心与分配）
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn build_hr(vectors: &[f32], dim: usize, o: &PipelineOpts) -> crate::graph::cellconfirm::CellConfirm {
    use crate::graph::cellconfirm::*;
    let t = Instant::now();
    let n = vectors.len() / dim;
    let nl = std::env::var("RAVEN_HR_NL").ok().and_then(|v| v.parse().ok()).unwrap_or(if o.hr_nl > 0 { o.hr_nl } else { default_nl(n) });
    let mut params = HrParams::default();
    // R43-3 大 k 计划（KL，原创组合：IVF 主导映射 E=k、P=ef 单元 + 渐进门槛确认 PG + 单元连续 8-bit 幸存者过滤 C8）。
    // 只在 k ≥ RAVEN_HR_KL（默认 50）时启用；k < KL 时参数与 R42 完全相同（未启用档逐位一致）。环境变量仍可逐项覆盖。
    let large_k = kl_on(o.k, dim);
    if large_k { params.er = 0.05; params.pr = 1.0; params.pg = 1; params.c8 = 0.02; }
    let params = params.from_env();
    let cached = o.hr_cache.as_ref().and_then(|path| std::fs::read(path).ok()).filter(|b| b.len() == (nl * dim + n) * 4);
    let (c, a) = if let Some(b) = cached {
        (b[..nl * dim * 4].chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<f32>>(),
         b[nl * dim * 4..].chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<u32>>())
    } else {
        let (c, a) = kmeans(vectors, dim, nl, looks_normalized(vectors, dim));
        if let Some(path) = &o.hr_cache {
            let mut b: Vec<u8> = Vec::with_capacity((nl * dim + n) * 4);
            for x in &c { b.extend_from_slice(&x.to_le_bytes()); } for x in &a { b.extend_from_slice(&x.to_le_bytes()); }
            let _ = std::fs::write(path, &b);
        }
        (c, a)
    };
    let cc = CellConfirm::from_centroids(vectors, dim, &c, &a, params);
    eprintln!("R33 HR built NL={nl} {:?} {:.1} MB in {:.1}s", params, cc.bytes() as f64 / 1e6, t.elapsed().as_secs_f64());
    cc
}

/// H20v2: pick search_degree by timing each candidate at the smallest ef reaching the target
/// recall on midpoint queries (base point + random graph neighbour), never touching test data.
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn calibrate_sd(p: &Prepared, dim: usize, o: &PipelineOpts) -> usize {
    calibrate_inner(p, dim, o, false).0
}

#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn calibrate_sd_po(p: &Prepared, dim: usize, o: &PipelineOpts) -> (usize, usize) {
    calibrate_inner(p, dim, o, true)
}

#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
fn calibrate_inner(p: &Prepared, dim: usize, o: &PipelineOpts, do_po: bool) -> (usize, usize) {
    use rand::RngCore;
    let t0 = Instant::now();
    let n = p.graph.len();
    let (nc, k) = (o.sd_calib_n.min(n), o.k);
    let base = &p.vectors;
    let mut rng = ChaCha8Rng::seed_from(7);
    let mut qv: Vec<f32> = Vec::with_capacity(nc * dim);
    for _ in 0..nc {
        let qi = (rng.next_u64() % n as u64) as usize;
        let nb = p.graph.neighbors(qi as u32);
        let j = if nb.is_empty() { qi } else { nb[(rng.next_u64() as usize) % nb.len()] as usize };
        qv.extend((0..dim).map(|d| 0.5 * (base[qi * dim + d] + base[j * dim + d])));
    }
    let thr: Vec<f64> = (0..nc).map(|ci| {
        let q = &qv[ci * dim..(ci + 1) * dim];
        let mut top: Vec<f64> = Vec::with_capacity(k + 1);
        for j in 0..n {
            let d = l2(q, &base[j * dim..(j + 1) * dim]);
            if top.len() < k { top.push(d); top.sort_by(|a, b| a.total_cmp(b)); }
            else if d < top[k - 1] { top[k - 1] = d; top.sort_by(|a, b| a.total_cmp(b)); }
        }
        top[k - 1] * (1.0 + 1e-6) + 1e-9
    }).collect();
    let eval_po = |sdv: usize, ef: usize, po: usize| -> (f64, f64) {
        let mut s = p.searcher(ef);
        s.with_search_degree(sdv);
        s.with_prefetch_offset(po);
        let (mut hit, mut best) = (0usize, f64::MAX);
        if !cal_mode().0 {
            // R43-2：确定性代价（召回一次 + novis 关计数遍）；RAVEN_CAL=time 恢复计时
            for ci in 0..nc {
                let q = &qv[ci * dim..(ci + 1) * dim];
                hit += p.run_sq4(&mut s, q, k).iter().take(k)
                    .filter(|x| l2(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
            }
            let cm = CostModel::platform();
            let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
            let mut c = 0f64;
            for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; let _ = p.run_sq4(&mut s, q, k); c += cm.cost(&CostModel::feats(&s, k, dim, 0)); }
            crate::graph::vamana::set_novis(nv);
            return (hit as f64 / (nc * k) as f64, c * 1e-9);
        }
        for rep in 0..3 {
            let t = Instant::now();
            let mut h = 0usize;
            for ci in 0..nc {
                let q = &qv[ci * dim..(ci + 1) * dim];
                h += p.run_sq4(&mut s, q, k).iter().take(k)
                    .filter(|x| l2(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
            }
            best = best.min(t.elapsed().as_secs_f64());
            if rep == 0 { hit = h; }
        }
        (hit as f64 / (nc * k) as f64, best)
    };
    let eval = |sdv: usize, ef: usize| eval_po(sdv, ef, p.po);
    let mut pick = (o.sd_cands.first().copied().unwrap_or(0), f64::MAX);
    let mut pick_ef = 0usize;
    for &c in &o.sd_cands {
        let (mut lo, mut hi) = (k, 4096usize);
        if eval(c, hi).0 < o.sd_calib_recall { continue; }
        while hi - lo > (lo / 16).max(2) {
            let mid = (lo + hi) / 2;
            if eval(c, mid).0 >= o.sd_calib_recall { hi = mid } else { lo = mid }
        }
        let tm = eval(c, hi).1;
        eprintln!("[pipeline] sd={c} ef={hi} t={tm:.4}s");
        if tm < pick.1 { pick = (c, tm); pick_ef = hi; }
    }
    eprintln!("[pipeline] sd-auto picked {} in {:.1}s", pick.0, t0.elapsed().as_secs_f64());
    let mut po = p.po;
    if do_po && pick_ef > 0 {
        // 候选 po 交替计时 5 轮取各自最小值；仅当比默认快 ≥3% 才换（滞回，防噪声翻转）
        let cands: Vec<usize> = std::env::var("RAVEN_POAUTO_C").ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![8, 16]);
        let mut best = vec![f64::MAX; cands.len()];
        let rounds: usize = std::env::var("RAVEN_POAUTO_R").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let hyst: f64 = std::env::var("RAVEN_POAUTO_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.97);
        for _ in 0..rounds { for (i, &c) in cands.iter().enumerate() { best[i] = best[i].min(eval_po(pick.0, pick_ef, c).1); } }
        // H82：校准计时太短（SIFT ~5ms）噪声大；对每个 po 循环查询集直到 ≥100ms，取轮内最小
        let timed = |po: usize| -> f64 {
            let mut s = p.searcher(pick_ef);
            s.with_search_degree(pick.0);
            s.with_prefetch_offset(po);
            let t = Instant::now();
            let mut m = 0usize;
            while m == 0 || t.elapsed().as_secs_f64() < 0.1 {
                for ci in 0..nc { std::hint::black_box(p.run_sq4(&mut s, &qv[ci * dim..(ci + 1) * dim], k)); }
                m += 1;
            }
            t.elapsed().as_secs_f64() / m as f64
        };
        let mode = std::env::var("RAVEN_POAUTO_MODE").unwrap_or_else(|_| "cold".into());
        if mode == "long" {
            best = vec![f64::MAX; cands.len()];
            for _ in 0..rounds { for (i, &c) in cands.iter().enumerate() { best[i] = best[i].min(timed(c)); } }
        } else if mode == "cold" {
            // H85：反复跑同一小批查询会把访问到的节点焐热进缓存，压低预取差距。
            // 改为每轮用全新的中点查询（计时不需要真值），轮内两候选交替先后，取每轮比值中位数。
            let nt = 512usize;
            let mut rng2 = ChaCha8Rng::seed_from(11);
            let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
            let mut sum = vec![0f64; cands.len()];
            for r in 0..rounds.max(2) * 2 {
                let mut tq: Vec<f32> = Vec::with_capacity(nt * dim);
                for _ in 0..nt {
                    let qi = (rng2.next_u64() % n as u64) as usize;
                    let nb = p.graph.neighbors(qi as u32);
                    let j = if nb.is_empty() { qi } else { nb[(rng2.next_u64() as usize) % nb.len()] as usize };
                    tq.extend((0..dim).map(|d| 0.5 * (base[qi * dim + d] + base[j * dim + d])));
                }
                let half = nt / 2;
                let mut ts = vec![0f64; cands.len()];
                for (oi, i) in (0..cands.len()).enumerate() {
                    let i = if r % 2 == 0 { i } else { cands.len() - 1 - oi };
                    let mut s = p.searcher(pick_ef);
                    s.with_search_degree(pick.0);
                    s.with_prefetch_offset(cands[i]);
                    // 每个候选用各自不相交的一半查询，避免后跑者吃前者焐热的缓存
                    let off = if (i + r) % 2 == 0 { 0 } else { half };
                    let t = Instant::now();
                    for ci in off..off + half { std::hint::black_box(p.run_sq4(&mut s, &tq[ci * dim..(ci + 1) * dim], k)); }
                    ts[i] = t.elapsed().as_secs_f64();
                }
                let b = base_i0(&cands, p.po);
                for i in 0..cands.len() { ratios[i].push(ts[i] / ts[b]); sum[i] += ts[i]; }
            }
            best = ratios.iter_mut().map(|v| { v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] }).collect();
        }
        let base_i = cands.iter().position(|&c| c == p.po);
        let (bi, bt) = best.iter().enumerate().fold((0, f64::MAX), |a, (i, &t)| if t < a.1 { (i, t) } else { a });
        let base_t = base_i.map(|i| best[i]).unwrap_or(f64::MAX);
        if bt < base_t * hyst { po = cands[bi]; }
        eprintln!("[pipeline] po-auto: {:?} -> po={po}", cands.iter().zip(best.iter()).map(|(c, t)| format!("{c}:{t:.4}")).collect::<Vec<_>>());
    }
    (pick.0, po)
}

fn base_i0(c: &[usize], po: usize) -> usize { c.iter().position(|&x| x == po).unwrap_or(0) }

/// R37 LP：点积（编译器自动向量化）
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let n = a.len().min(b.len());
    let (ca, cb) = (a[..n].chunks_exact(16), b[..n].chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) { for i in 0..16 { acc[i] += x[i] * y[i]; } }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) { s += x * y; }
    s
}

/// R97 H178：缓存行 LP 档选中时置位，prepare 据此不建 LPX 全维码。
pub static H178_NOLPX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn load_lp(dim: usize) -> (Vec<f32>, usize) { load_lp_v(dim, &[], -1.0) }

/// R37 LP：自动投影。RAVEN_LP=auto（或 =m 指定维数）时在库内由样本协方差的子空间迭代求前 m 主子空间，
/// 再乘随机正交阵均衡各维方差；门控：dim ≥ RAVEN_LP_MINDIM(256) 且前 m 维方差占比 ≥ RAVEN_LP_FRAC(0.90)。
/// R40：RAVEN_LP_GATE=fid 时改用"保真度门控"：构建期 LOO（库内样本作查询、子样本上精确 top-C 候选），
/// 比较 LP-m 码与全维 SQ4 码"真 top-k ∈ 近似 top-4k"的比例，相对保真度 ≥ RAVEN_LP_FIDTHR（默认 0.98；Python 校准：fashion LP128 0.9925、gist LP192 0.986、arxiv LP192 0.9925 开，nytimes ≤0.93 关）取最小 m。
/// 方差占比只是代理（VIBE arxiv-768：varfrac(256)=0.896 被 0.90 门槛误关，而 LP192 保真度 0.9925 与 fashion LP128 同档）。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
fn load_lp_v(dim: usize, vectors: &[f32], clip: f32) -> (Vec<f32>, usize) {
    H178_NOLPX.store(false, std::sync::atomic::Ordering::Relaxed);
    // R41：x86_64 默认 RAVEN_LP=auto（方差占比门控；测试机 门控 (b)：fashion +35%、mnist +64…+87%、gist ≈3.7× @0.95；
    // dim<256 的集合与 nytimes 不进入，逐位不变）；aarch64 默认仍关。RAVEN_LP=0 可关。
    let lp_env = std::env::var("RAVEN_LP").ok().or_else(|| if cfg!(target_arch = "x86_64") && std::env::var("RAVEN_LP_M").is_err() { Some("auto".to_string()) } else { None });
    if let Some(v) = lp_env {
        if v != "0" && !vectors.is_empty() {
            let mind: usize = std::env::var("RAVEN_LP_MINDIM").ok().and_then(|x| x.parse().ok()).unwrap_or(256);
            // R42：阈值 0.90→0.88（arxiv-768 m=256 varfrac 0.896、码保真 0.991、测试机 实测 LP256 @0.95 ≈1.6× 被 0.90 误关；
            // 已测集合选择不变：fashion 0.928/mnist 0.937（m=128）、gist m=128 0.865 仍关→m=192 0.909；高召回段由 LPX 双码计划兜底）
            // R85-22：NER 开时全维 4 bit 码导航接近真 L2，LP 只在投影高保真时仍划算 → 阈值 0.90
            // （arxiv m=256 0.897 关：LP0+NER 对 GINGER 1.17×/1.07×；landmark m=256 0.917 开：LP0 ≈0.7× LP auto）。余量很窄，待复测。
            let ner_on = ner_enabled(dim);
            let fth: f64 = std::env::var("RAVEN_LP_FRAC").ok().and_then(|x| x.parse().ok()).unwrap_or(if ner_on { 0.90 } else { 0.88 });
            // auto：按 128→192→256 依次试，要求 m ≤ dim/3（导航码字节至少省 3×），取第一个方差占比过阈的
            let cands: Vec<usize> = if v == "auto" { vec![128, 192, 256] } else { vec![v.parse().unwrap_or(128)] };
            if dim >= mind {
                for m in cands {
                    if v == "auto" && m * 3 > dim { break; }
                    if m >= dim { break; }
                    let t = std::time::Instant::now();
                    let fid_gate = v == "auto" && std::env::var("RAVEN_LP_GATE").map(|g| g == "fid").unwrap_or(false);
                    if fid_gate {
                        let floor: f64 = std::env::var("RAVEN_LP_FIDFLOOR").ok().and_then(|x| x.parse().ok()).unwrap_or(0.70);
                        let thr: f64 = std::env::var("RAVEN_LP_FIDTHR").ok().and_then(|x| x.parse().ok()).unwrap_or(0.98);
                        let (p, frac) = lp_auto(vectors, dim, m);
                        if frac < floor { eprintln!("R40 LP fid-gate m={m} varfrac={frac:.3} < floor {floor} -> skip ({:.1}s)", t.elapsed().as_secs_f64()); continue; }
                        let (fl, ff) = lp_fidelity(vectors, dim, &p, m, clip);
                        let rel = if ff > 0.0 { fl / ff } else { 0.0 };
                        let on = rel >= thr;
                        eprintln!("R40 LP fid-gate m={m} varfrac={frac:.3} fid_lp={fl:.4} fid_full={ff:.4} rel={rel:.4} thr={thr} -> {} ({:.1}s)", if on { "ON" } else { "off" }, t.elapsed().as_secs_f64());
                        if on { return (p, m); }
                        continue;
                    }
                    let (p, frac) = lp_auto(vectors, dim, m);
                    // R97 H179 边界带：方差占比只是代理，|varfrac − 门| < RAVEN_LP_BAND（缺省 0.03）时不按代理硬切，
                    // 改由 R40 保真度（LOO：真 top-k ∈ 近似 top-4k，LP-m 码 / 全维码之比 ≥ RAVEN_LP_FIDTHR 0.98）裁决。
                    // 依据：celeba-2048 m=256 varfrac 0.902 vs 门 0.90（险过；落 LP=0 路径 @0.95 2901 vs LP256 5143 QPS），
                    // arxiv-768 m=256 0.897（险关）。带外（landmark 0.917@门0.5、gist/fashion/mnist、dim<256 集）判定与旧规则逐位相同。
                    let band: f64 = std::env::var("RAVEN_LP_BAND").ok().and_then(|x| x.parse().ok()).unwrap_or(0.03);
                    let on = if band > 0.0 && (frac - fth).abs() < band {
                        let thr: f64 = std::env::var("RAVEN_LP_FIDTHR").ok().and_then(|x| x.parse().ok()).unwrap_or(0.98);
                        let (fl, ff) = lp_fidelity(vectors, dim, &p, m, clip);
                        let rel = if ff > 0.0 { fl / ff } else { 0.0 };
                        eprintln!("R97 LP band m={m} varfrac={frac:.3} thr={fth}±{band} -> fid_lp={fl:.4} fid_full={ff:.4} rel={rel:.4} thr={thr}");
                        rel >= thr
                    } else { frac >= fth };
                    eprintln!("R37 LP auto m={m} varfrac={frac:.3} thr={fth} -> {} ({:.1}s)", if on { "ON" } else { "off" }, t.elapsed().as_secs_f64());
                    if on { return (p, m); }
                }
            }
            // R97 H178：缓存行 LP 档（x86 auto 缺省开，RAVEN_LPCL=0 关）。常规阶梯（m ≤ dim/3）全部未开时，
            // 再试 m 为 128 的倍数（导航码 m/2 B 恰为整 64 B 行）、dim/3 < m ≤ 2·dim/3（每跳码行数至少省 1/3）的投影，
            // 同一方差占比门（fth）。H177/H178：@0.99 档 ~60% 时间是冷访存，每跳取码行数是剩余杠杆
            // （yahoo-384 LP256 = 2 行 vs 全维 3 行：q94 同轮 @0.99 见 LOG「R97 结果 H178」）。此档关 LPX 双码（高 ef 切回全维码会抵消省下的行）。
            let lpcl = v == "auto" && cfg!(target_arch = "x86_64") && std::env::var("RAVEN_LPCL").map(|x| x != "0").unwrap_or(true);
            // RAVEN_LPCL_MINDIM（缺省 384）：dim 256–383（如 nytimes-256）未测，不进此档
            let cl_mind: usize = std::env::var("RAVEN_LPCL_MINDIM").ok().and_then(|x| x.parse().ok()).unwrap_or(384);
            if lpcl && dim >= mind && dim >= cl_mind {
                let mut m = 128usize;
                while m * 3 <= dim { m += 128; }
                // 上限 RAVEN_LPCL_MAXM（缺省 256 = 128 B/2 行）：ccnews-768 同轮 LP384（192 B）@0.99 ≈0.86×、LP256 ≈0.97×（varfrac 0.892 本就不过门）→ 只在 ≤2 行档开
                let maxm: usize = std::env::var("RAVEN_LPCL_MAXM").ok().and_then(|x| x.parse().ok()).unwrap_or(256);
                while m * 3 <= dim * 2 && m < dim && m <= maxm {
                    let t = std::time::Instant::now();
                    let (p, frac) = lp_auto(vectors, dim, m);
                    let on = frac >= fth;
                    eprintln!("R97 H178 LPCL m={m} ({} B/node vs {} B) varfrac={frac:.3} thr={fth} -> {} ({:.1}s)", m / 2, dim / 2, if on { "ON (LPX off)" } else { "off" }, t.elapsed().as_secs_f64());
                    if on { H178_NOLPX.store(true, std::sync::atomic::Ordering::Relaxed); return (p, m); }
                    m += 128;
                }
            }
            return (Vec::new(), 0);
        }
    }
    let m: usize = std::env::var("RAVEN_LP_M").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if m == 0 { return (Vec::new(), 0); }
    let f = std::env::var("RAVEN_LP_FILE").expect("RAVEN_LP_M needs RAVEN_LP_FILE");
    let b = std::fs::read(&f).expect("read LP file");
    let all: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert!(all.len() >= m * dim && all.len() % dim == 0, "LP file must be M×dim with M ≥ m");
    (all[..m * dim].to_vec(), m)
}

/// R40 保真度：返回 (LP-m 码保真度, 全维 SQ4 码保真度)，保真度 = 真 top-k 落入近似 top-4k 的比例（k = RAVEN_K 或 10）。
/// 查询 = 200 个库内点（与候选子样本错位，避开自身）；候选 = ≤10 万行等距子样本上精确 top-C（C = max(20k, 2000)）。确定性。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn lp_fidelity(v: &[f32], dim: usize, proj: &[f32], m: usize, clip: f32) -> (f64, f64) {
    use crate::quant::sq4::SQ4Params;
    let n = v.len() / dim;
    let k: usize = std::env::var("RAVEN_K").ok().and_then(|x| x.parse().ok()).unwrap_or(10);
    let ns = n.min(100_000); let step = (n / ns).max(1);
    let sidx: Vec<usize> = (0..ns).map(|i| i * step).collect();
    let nq = 200usize.min(n / 2);
    let qidx: Vec<usize> = (0..nq).map(|j| (j * (n / nq) + step / 2 + 1) % n).collect();
    let c = (20 * k).max(2000).min(ns.saturating_sub(1));
    let r = (4 * k).min(c);
    let rows = |ix: &[usize]| -> Vec<f32> { let mut o = Vec::with_capacity(ix.len() * dim); for &i in ix { o.extend_from_slice(&v[i * dim..(i + 1) * dim]); } o };
    let xs = rows(&sidx);
    let ps = project_all(&xs, dim, proj, m);
    // 全维 SQ4（与原路径同拟合）与 LP 均匀步长 SQ4（与 build_uniform_clipped 同：步长取最大维）
    let pf = SQ4Params::fit_range(&xs, dim, clip);
    let pl0 = SQ4Params::fit_range(&ps, m, clip);
    let st = pl0.scale.iter().cloned().fold(1e-12f32, f32::max);
    let lmin: Vec<f32> = (0..m).map(|d| 0.5 * (pl0.min[d] + pl0.max[d]) - 7.5 * st).collect();
    let deq = |x: f32, mn: f32, sc: f32| -> f32 { let c = ((x - mn) / sc).round().clamp(0.0, 15.0); mn + c * sc };
    let nt = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1).min(16);
    let per = nq.div_ceil(nt);
    let (hl, hf): (usize, usize) = std::thread::scope(|sc| {
        let hs: Vec<_> = qidx.chunks(per).map(|qs| { let (xs, ps, pf, lmin, sidx) = (&xs, &ps, &pf, &lmin, &sidx); sc.spawn(move || {
            let (mut a, mut b) = (0usize, 0usize);
            for &qi in qs {
                let q = &v[qi * dim..(qi + 1) * dim];
                let mut ex: Vec<(f32, u32)> = (0..ns).filter(|&j| sidx[j] != qi).map(|j| (crate::distance::l2_simd(q, &xs[j * dim..(j + 1) * dim]), j as u32)).collect();
                let cc = c.min(ex.len() - 1);
                ex.select_nth_unstable_by(cc, |x, y| x.0.total_cmp(&y.0)); ex.truncate(cc); ex.sort_by(|x, y| x.0.total_cmp(&y.0));
                let truth: Vec<u32> = ex[..k.min(ex.len())].iter().map(|x| x.1).collect();
                let mut qp = vec![0f32; m]; for (jj, o) in qp.iter_mut().enumerate() { *o = dot_f32(&proj[jj * dim..(jj + 1) * dim], q); }
                let mut af: Vec<(f32, u32)> = ex.iter().map(|&(_, j)| { let x = &xs[j as usize * dim..(j as usize + 1) * dim];
                    let mut s = 0f32; for d in 0..dim { let t = q[d] - deq(x[d], pf.min[d], pf.scale[d].max(1e-12)); s += t * t; } (s, j) }).collect();
                let mut al: Vec<(f32, u32)> = ex.iter().map(|&(_, j)| { let y = &ps[j as usize * m..(j as usize + 1) * m];
                    let mut s = 0f32; for d in 0..m { let t = qp[d] - deq(y[d], lmin[d], st); s += t * t; } (s, j) }).collect();
                let rr = r.min(af.len() - 1);
                af.select_nth_unstable_by(rr, |x, y| x.0.total_cmp(&y.0)); al.select_nth_unstable_by(rr, |x, y| x.0.total_cmp(&y.0));
                let sf: std::collections::HashSet<u32> = af[..rr].iter().map(|x| x.1).collect();
                let sl: std::collections::HashSet<u32> = al[..rr].iter().map(|x| x.1).collect();
                b += truth.iter().filter(|t| sf.contains(t)).count();
                a += truth.iter().filter(|t| sl.contains(t)).count();
            }
            (a, b) }) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).fold((0, 0), |s, x| (s.0 + x.0, s.1 + x.1))
    });
    let den = (nq * k) as f64;
    (hl as f64 / den, hf as f64 / den)
}

fn project_all(v: &[f32], dim: usize, proj: &[f32], m: usize) -> Vec<f32> {
    let n = v.len() / dim;
    let mut out = vec![0f32; n * m];
    let nt = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let rows = n.div_ceil(nt).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(rows * m).enumerate() {
            sc.spawn(move || {
                for (r, o) in chunk.chunks_mut(m).enumerate() {
                    let i = ci * rows + r;
                    let x = &v[i * dim..(i + 1) * dim];
                    for j in 0..m { o[j] = dot_f32(&proj[j * dim..(j + 1) * dim], x); }
                }
            });
        }
    });
    out
}

/// 返回 (m×dim 行主序正交投影, 前 m 子空间方差占比)。确定性（固定种子）。
#[cold] #[inline(never)] // R102 coldcal: prepare-time only, keep out of hot .text
pub fn lp_auto(v: &[f32], dim: usize, m: usize) -> (Vec<f32>, f64) {
    let n = v.len() / dim;
    let ns = n.min(20000);
    let step = (n / ns).max(1);
    let idx: Vec<usize> = (0..ns).map(|i| i * step).collect();
    let mut mu = vec![0f64; dim];
    for &i in &idx { for d in 0..dim { mu[d] += v[i * dim + d] as f64; } }
    for x in mu.iter_mut() { *x /= ns as f64; }
    // 中心化样本（f32）
    let mut x = vec![0f32; ns * dim];
    for (r, &i) in idx.iter().enumerate() { for d in 0..dim { x[r * dim + d] = (v[i * dim + d] as f64 - mu[d]) as f32; } }
    // 协方差 C = XᵀX / ns（按行块并行累加）
    let nt = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
    let rows = ns.div_ceil(nt);
    let parts: Vec<Vec<f64>> = std::thread::scope(|sc| {
        let hs: Vec<_> = x.chunks(rows * dim).map(|blk| sc.spawn(move || {
            let mut c = vec![0f64; dim * dim];
            let mut acc = vec![0f32; dim * dim];
            for (k, row) in blk.chunks(dim).enumerate() {
                for a in 0..dim { let ra = row[a]; if ra == 0.0 { continue; }
                    let o = &mut acc[a * dim..(a + 1) * dim];
                    for b in 0..dim { o[b] += ra * row[b]; } }
                if k % 256 == 255 { for (ci, ai) in c.iter_mut().zip(acc.iter_mut()) { *ci += *ai as f64; *ai = 0.0; } }
            }
            for (ci, ai) in c.iter_mut().zip(acc.iter()) { *ci += *ai as f64; }
            c
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut c = vec![0f64; dim * dim];
    for p in parts { for (a, b) in c.iter_mut().zip(p) { *a += b; } }
    for e in c.iter_mut() { *e /= ns as f64; }
    let tr: f64 = (0..dim).map(|d| c[d * dim + d]).sum();
    // 子空间迭代：Q (m×dim 行 = 基向量)
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut gauss = || { // Box-Muller
        let mut u = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
        let (a, b) = (u(), u()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    };
    let mut q: Vec<f64> = (0..m * dim).map(|_| gauss()).collect();
    let mgs = |q: &mut Vec<f64>| {
        for i in 0..m {
            for j in 0..i { let d: f64 = (0..dim).map(|t| q[i * dim + t] * q[j * dim + t]).sum();
                for t in 0..dim { q[i * dim + t] -= d * q[j * dim + t]; } }
            let nr: f64 = (0..dim).map(|t| q[i * dim + t] * q[i * dim + t]).sum::<f64>().sqrt().max(1e-300);
            for t in 0..dim { q[i * dim + t] /= nr; }
        }
    };
    mgs(&mut q);
    let iters: usize = std::env::var("RAVEN_LP_ITERS").ok().and_then(|x| x.parse().ok()).unwrap_or(12);
    for _ in 0..iters {
        let mut z = vec![0f64; m * dim];
        for i in 0..m { let qi = &q[i * dim..(i + 1) * dim]; let zi = &mut z[i * dim..(i + 1) * dim];
            for a in 0..dim { let ca = &c[a * dim..(a + 1) * dim]; zi[a] = ca.iter().zip(qi).map(|(x, y)| x * y).sum(); } }
        q = z; mgs(&mut q);
    }
    let mut var = 0f64;
    for i in 0..m { let qi = &q[i * dim..(i + 1) * dim];
        for a in 0..dim { let ca = &c[a * dim..(a + 1) * dim]; var += qi[a] * ca.iter().zip(qi).map(|(x, y)| x * y).sum::<f64>(); } }
    // 随机正交 m×m（高斯 + MGS），P = R·Q
    let mut r: Vec<f64> = (0..m * m).map(|_| gauss()).collect();
    for i in 0..m {
        for j in 0..i { let d: f64 = (0..m).map(|t| r[i * m + t] * r[j * m + t]).sum(); for t in 0..m { r[i * m + t] -= d * r[j * m + t]; } }
        let nr: f64 = (0..m).map(|t| r[i * m + t] * r[i * m + t]).sum::<f64>().sqrt(); for t in 0..m { r[i * m + t] /= nr; }
    }
    let mut p = vec![0f32; m * dim];
    for i in 0..m { for k in 0..m { let w = r[i * m + k]; for t in 0..dim { p[i * dim + t] += (w * q[k * dim + t]) as f32; } } }
    (p, var / tr.max(1e-300))
}

/// R97 H177 NU 自动关判据：RAVEN_ADQ_NU=1 现为 auto（数据稀疏或全非负时不做 NU 重编码），=2 强开（= R96 的 =1 行为），其它 = 关。
/// 依据（R97 XVI，测试机，celeba-resnet-2048 ReLU 特征，约 60% 精确 0、全非负）：同图 LP=0 ef100 召回 NU 0.484 vs 关 NU 0.774；
/// NU 的 16 个共享 Lloyd–Max 级按全体分量的近高斯分布定，在"零团 + 正半轴长尾"上浪费级数。
/// 判据：等距抽样 ≤20000 行，零值占比 > RAVEN_NU_ZMAX（缺省 0.30）或负值占比 < 0.001 → 关。稠密有符号集（glove/landmark/imagenet/…）不触发 ⇒ 路径不变。
pub(crate) fn nu_wanted(v: &[f32], dim: usize) -> bool {
    match std::env::var("RAVEN_ADQ_NU").as_deref() {
        Ok("2") => true,
        Ok("1") => {
            let n = if dim == 0 { 0 } else { v.len() / dim };
            if n == 0 { return true; }
            let step = (n / 20000).max(1);
            let (mut z, mut neg, mut tot) = (0usize, 0usize, 0usize);
            let mut i = 0;
            while i < n { for &x in &v[i * dim..(i + 1) * dim] { z += (x == 0.0) as usize; neg += (x < 0.0) as usize; } tot += dim; i += step; }
            let zf = z as f64 / tot as f64; let nf = neg as f64 / tot as f64;
            let zmax: f64 = std::env::var("RAVEN_NU_ZMAX").ok().and_then(|x| x.parse().ok()).unwrap_or(0.30);
            let on = !(zf > zmax || nf < 0.001);
            eprintln!("R97 NU auto: zero_frac={zf:.3} neg_frac={nf:.3} (zmax {zmax}) -> {}", if on { "on" } else { "off" });
            on
        }
        _ => false,
    }
}

/// R97 NV-3 低维路由（glove-200 收口）：128 ≤ dim < RAVEN_LOWD_MAXDIM（256）的单位范数、稠密有符号集
/// → 建图 GMD=128、R≥96（H175 r96 配方），搜索度缺省 96（未设 RAVEN_SD 时）。
/// 依据（R97 NV-2，测试机 两轮）：glove-200 r96 SD96 对 G @0.90 ▲1.44–1.50×、@0.95 ▲1.71–1.80×（g0 SD48 缺省 ▼0.91–0.93×），
/// 缺口来源是图的度而非导航量化（oracle：4 bit NU 码 P500 上界 0.9999）。
/// 判据：x86 缺省开（RAVEN_LOWD_ROUTE=0 关，aarch64 缺省关）；等距抽样 ≤20000 行，|‖x‖−1| < 1e-3、零值占比 ≤ 0.30、负值占比 ≥ 0.001。
/// dim < RAVEN_LOWD_MINDIM（128：glove-25/100、deep-96 未测）不进；sift（全非负）、yandex（非单位范数）不触发。
pub fn lowd_route(v: &[f32], dim: usize) -> bool {
    let on = std::env::var("RAVEN_LOWD_ROUTE").map(|x| x != "0").unwrap_or(cfg!(target_arch = "x86_64"));
    let envu = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    if !on || dim < envu("RAVEN_LOWD_MINDIM", 128) || dim >= envu("RAVEN_LOWD_MAXDIM", 256) { return false; }
    let n = if dim == 0 { 0 } else { v.len() / dim };
    if n == 0 { return false; }
    let step = (n / 20000).max(1);
    let (mut z, mut neg, mut tot, mut badn) = (0usize, 0usize, 0usize, 0usize);
    let mut i = 0;
    while i < n {
        let r = &v[i * dim..(i + 1) * dim]; let mut s = 0f64;
        for &x in r { z += (x == 0.0) as usize; neg += (x < 0.0) as usize; s += (x as f64) * (x as f64); }
        badn += ((s.sqrt() - 1.0).abs() >= 1e-3) as usize; tot += dim; i += step;
    }
    let zf = z as f64 / tot as f64; let nf = neg as f64 / tot as f64;
    let r = badn == 0 && zf <= 0.30 && nf >= 0.001;
    eprintln!("R97 NV-3 lowd route: dim={dim} zero_frac={zf:.3} neg_frac={nf:.3} non_unit_rows={badn} -> {}", if r { "ON (GMD128 R96 SD96)" } else { "off" });
    r
}

/// R85-36 NER 维数门（OOD 门的第一版）：RAVEN_NER=1 强开、=0 强关；缺省 auto = dim ≥ RAVEN_NER_MINDIM（384）才开。
/// 依据（R85 同机对 GINGER）：imagenet-512 ▲1.29–1.37×、landmark/arxiv-768 ≈/▲；yandex-200（OOD）为负、glove-100 中性。
/// 只在 ADQ（单位范数）时才会被用到；阈值是经验门，样本少，待更多集（ccnews-768、yahoo-384）验证。
pub(crate) fn ner_enabled(dim: usize) -> bool {
    match std::env::var("RAVEN_NER").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => dim >= std::env::var("RAVEN_NER_MINDIM").ok().and_then(|x| x.parse().ok()).unwrap_or(384),
    }
}

/// R98 H188：抽样 ≤2000 个向量，范数平方都在 1±2e-3 内即视为单位范数数据。
fn unit_norm_sample(vectors: &[f32], dim: usize) -> bool {
    let nn = vectors.len() / dim.max(1);
    if nn == 0 { return false; }
    let st = (nn / 2000).max(1);
    (0..nn).step_by(st).all(|i| { let s: f32 = vectors[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum(); (s - 1.0).abs() < 2e-3 })
}

/// R102：构建期标定确定性。同一图、同一数据两次 prepare → PLAN 逐字相同、各 ef 的结果 id/距离逐位相同。
#[cfg(test)]
mod r102_detcal_tests {
    use super::*;
    use crate::graph::vamana::VamanaBuildConfig;
    use crate::memory::serialize::Serializable;
    use rand::RngCore;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// 高斯簇合成数据（固定种子）：n 个库点 + nq 个查询
    fn synth(n: usize, nq: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = ChaCha8Rng::seed_from(seed);
        let mut u = || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let nclu = 32usize;
        let cent: Vec<f32> = (0..nclu * dim).map(|_| (u() * 10.0) as f32).collect();
        let mut gen = |m: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(m * dim);
            for _ in 0..m {
                let c = ((u() * nclu as f64) as usize).min(nclu - 1);
                for d in 0..dim {
                    let g = (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos();
                    v.push(cent[c * dim + d] + g as f32);
                }
            }
            v
        };
        let base = gen(n);
        let q = gen(nq);
        (base, q)
    }

    fn graph_bytes(base: &[f32], dim: usize) -> Vec<u8> {
        let cfg = VamanaBuildConfig { l_build: 64, r_max: 32, r_soft: 40, ..Default::default() };
        let mut rng = ChaCha8Rng::seed_from(42);
        VamanaGraph::build(base, dim, &cfg, &mut rng).serialize()
    }

    /// (PLAN 字符串, 各 ef 结果 [(id, dist bits)])
    fn run(gb: &[u8], base: &[f32], q: &[f32], dim: usize, o: &PipelineOpts, efs: &[usize]) -> (String, u64, Vec<Vec<(u32, u32)>>) {
        let g = VamanaGraph::deserialize(gb).expect("graph");
        let p = prepare(g, base.to_vec(), dim, o);
        let nq = q.len() / dim;
        let mut out = Vec::new();
        for &ef in efs {
            let mut s = p.searcher(ef);
            let mut r = Vec::new();
            for i in 0..nq { for x in p.run_auto(&mut s, &q[i * dim..(i + 1) * dim], o.k) { r.push((p.orig_id(x.0), x.1.to_bits())); } }
            out.push(r);
        }
        (p.plan_string(), p.plan_hash(), out)
    }

    fn check(o: &PipelineOpts, n: usize, dim: usize) -> String {
        let (base, q) = synth(n, 200, dim, 102);
        let gb = graph_bytes(&base, dim);
        let efs = [10usize, 20, 40, 80, 160, 300];
        let a = run(&gb, &base, &q, dim, o, &efs);
        let b = run(&gb, &base, &q, dim, o, &efs);
        assert_eq!(a.0, b.0, "PLAN differs between two prepares");
        assert_eq!(a.1, b.1);
        for (i, ef) in efs.iter().enumerate() { assert!(a.2[i] == b.2[i], "result ids/dists differ at ef {ef}"); assert!(!a.2[i].is_empty()); }
        a.0
    }

    #[test]
    fn prepare_twice_identical_sd_auto() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let o = PipelineOpts::default();
        let plan = check(&o, 4000, 32);
        eprintln!("sd-auto plan: {plan}");
    }

    #[test]
    fn prepare_twice_identical_sdx() {
        // 强制截断度数 → SDX 标定必走（R54 原计时路径就是 yahoo 非确定性的来源）
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let o = PipelineOpts { search_degree: Some(12), ..Default::default() };
        let plan = check(&o, 4000, 32);
        assert!(plan.contains("sd=12"), "{plan}");
        eprintln!("sdx plan: {plan}");
    }

    #[test]
    fn hr_gate_after_sdx_r104() {
        // R104 hrgate：截断度数 + HR（小 n 强开）→ SDX 先于门槛标定。两次 prepare 计划/结果逐位相同；
        // SDX 计划与旧顺序（RAVEN_HRG_SDX=0）相同（SDX 标定不依赖 HR）；旧顺序 PLAN 记 hrg=sd，缺省不记。
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dim = 32usize;
        let (base, q) = synth(4000, 200, dim, 104);
        let gb = graph_bytes(&base, dim);
        let o = PipelineOpts { search_degree: Some(12), hr_min_n: 0, ..Default::default() };
        let efs = [10usize, 20, 40, 80, 160, 300];
        let a = run(&gb, &base, &q, dim, &o, &efs);
        let b = run(&gb, &base, &q, dim, &o, &efs);
        assert_eq!(a.0, b.0, "PLAN differs between two prepares");
        assert_eq!(a.1, b.1);
        for (i, ef) in efs.iter().enumerate() { assert!(a.2[i] == b.2[i], "result ids/dists differ at ef {ef}"); assert!(!a.2[i].is_empty()); }
        assert!(a.0.contains("sd=12") && !a.0.contains("hrg="), "{}", a.0);
        std::env::set_var("RAVEN_HRG_SDX", "0");
        let l = run(&gb, &base, &q, dim, &o, &efs);
        std::env::remove_var("RAVEN_HRG_SDX");
        eprintln!("hrgate plan new: {}\nhrgate plan old: {}", a.0, l.0);
        assert!(l.0.ends_with(" hrg=sd"), "{}", l.0);
        let field = |s: &str, k: &str| s.split(' ').find(|t| t.starts_with(k)).map(|t| t.to_string());
        for k in ["sd=", "sdx=", "lpx_sdx=", "lpx=", "proj_m="] { assert_eq!(field(&a.0, k), field(&l.0, k), "{k} must not depend on calibration order"); }
        // 门槛判定相同时，两种顺序的搜索结果逐位相同（重排只改门槛基线，不改搜索路径）
        if field(&a.0, "hr_gate=") == field(&l.0, "hr_gate=") && field(&a.0, "h210=") == field(&l.0, "h210=") {
            for (i, ef) in efs.iter().enumerate() { assert!(a.2[i] == l.2[i], "same plan but results differ at ef {ef}"); }
        }
    }

    #[test]
    fn plan_string_hrg_token_r104() {
        // 未建 HR 时不记 hrg（旧顺序也不记）：未触发门槛标定的数据集 PLAN 字符串与 af8e3514 格式相同
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dim = 32usize;
        let (base, q) = synth(3000, 50, dim, 105);
        let gb = graph_bytes(&base, dim);
        let o = PipelineOpts { search_degree: Some(12), ..Default::default() }; // n < hr_min_n → 无 HR
        std::env::set_var("RAVEN_HRG_SDX", "0");
        let l = run(&gb, &base, &q, dim, &o, &[40]);
        std::env::remove_var("RAVEN_HRG_SDX");
        let a = run(&gb, &base, &q, dim, &o, &[40]);
        assert_eq!(a.0, l.0); assert_eq!(a.1, l.1); assert!(a.2 == l.2);
        assert!(a.0.contains("hr_gate=off") && !a.0.contains("hrg="), "{}", a.0);
    }

    /// 低秩合成数据（dim 512，秩 48 + 小噪声）：x86 上 LP auto 开（m=128）、全维码整码预取（512/2/64 = 4 行）→ 走 LPX pf_model 标定
    fn synth_lowrank(n: usize, nq: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = ChaCha8Rng::seed_from(seed);
        let mut u = || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let r = 48usize;
        let w: Vec<f32> = (0..r * dim).map(|_| (u() * 2.0 - 1.0) as f32).collect();
        let mut gen = |m: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(m * dim);
            for _ in 0..m {
                let z: Vec<f32> = (0..r).map(|_| (u() * 2.0 - 1.0) as f32).collect();
                for d in 0..dim { let mut x = 0.05 * (u() as f32 - 0.5); for j in 0..r { x += z[j] * w[j * dim + d]; } v.push(x); }
            }
            v
        };
        let base = gen(n);
        let q = gen(nq);
        (base, q)
    }

    #[test]
    fn lpx_cal_kind_and_sat_r102_kappa() {
        use super::{lpx_cal_kind, lpx_sat};
        // 单行 LP 码（m ≤ 128，fashion/mnist）+ 整码预取 + 缺省 → 确定性 pf_model
        assert_eq!(lpx_cal_kind(true, false, false, 128), (true, false));
        assert_eq!(lpx_cal_kind(true, false, false, 64), (true, false));
        // 多行 LP 码（m 192/256：landmark/arxiv/ccnews/gist）→ 退回旧缺省：计时
        assert_eq!(lpx_cal_kind(true, false, false, 192), (false, true));
        assert_eq!(lpx_cal_kind(true, false, false, 256), (false, true));
        // RAVEN_CAL=time → 计时；RAVEN_CAL=model → 纯模型；无整码预取 → 纯模型
        assert_eq!(lpx_cal_kind(true, true, false, 128), (false, true));
        assert_eq!(lpx_cal_kind(true, false, true, 128), (false, false));
        assert_eq!(lpx_cal_kind(true, false, true, 256), (false, false));
        assert_eq!(lpx_cal_kind(false, false, false, 128), (false, false));
        assert_eq!(lpx_cal_kind(false, true, false, 256), (false, true));
        // R49d 饱和：整码预取下缺省与 RAVEN_CAL=time 相同（都开）；RAVEN_CAL=model 关；全维网格饱和子句不变
        assert!(lpx_sat(0.98, 0.995, true, false));
        assert!(!lpx_sat(0.98, 0.985, true, false));
        assert!(!lpx_sat(0.98, 0.995, true, true));
        assert!(!lpx_sat(0.98, 0.995, false, false));
        assert!(lpx_sat(1.0, 1.0, false, true));
        assert!(!lpx_sat(1.0, 0.995, false, false));
    }

    #[test]
    fn prepare_twice_identical_lpx_pf() {
        // R102 lpxdet：LP + LPX（整码预取）标定两次 prepare 计划与结果逐位相同（旧默认为计时判定）
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dim = 512usize;
        let (base, q) = synth_lowrank(3000, 100, dim, 1021);
        let gb = graph_bytes(&base, dim);
        let o = PipelineOpts::default();
        let efs = [10usize, 20, 40, 80, 160];
        let a = run(&gb, &base, &q, dim, &o, &efs);
        let b = run(&gb, &base, &q, dim, &o, &efs);
        eprintln!("lpx-pf plan: {}", a.0);
        assert_eq!(a.0, b.0, "PLAN differs between two prepares");
        assert_eq!(a.1, b.1);
        for (i, ef) in efs.iter().enumerate() { assert!(a.2[i] == b.2[i], "result ids/dists differ at ef {ef}"); assert!(!a.2[i].is_empty()); }
        if cfg!(target_arch = "x86_64") {
            assert!(!a.0.contains("proj_m=0 "), "LP expected on for low-rank dim 512: {}", a.0);
            assert!(!a.0.contains("lpx=none"), "LPX plan expected: {}", a.0);
        }
    }

    /// 冒烟（ignored）：单进程一次 prepare 并打印 PLAN 哈希 + 结果哈希；图缓存在 RAVEN_DETCAL_G（缺省 temp 目录），
    /// 两个进程分别运行（可配 RAVEN_CAL=time）比较输出。
    #[test]
    #[ignore]
    fn detcal_smoke_print() {
        let (n, dim) = (std::env::var("RAVEN_DETCAL_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20000usize), 64usize);
        let (base, q) = synth(n, 500, dim, 7);
        let gp = std::env::var("RAVEN_DETCAL_G").unwrap_or_else(|_| std::env::temp_dir().join(format!("raven_detcal_{n}.g")).to_string_lossy().into_owned());
        let gb = match std::fs::read(&gp) { Ok(b) => b, Err(_) => { let b = graph_bytes(&base, dim); std::fs::write(&gp, &b).unwrap(); b } };
        let sd = std::env::var("RAVEN_DETCAL_SD").ok().and_then(|v| v.parse().ok());
        let o = PipelineOpts { search_degree: sd, ..Default::default() };
        let (plan, h, res) = run(&gb, &base, &q, dim, &o, &[10, 20, 40, 80, 160, 300]);
        let rh = res.iter().flatten().fold(0xcbf29ce484222325u64, |h, &(i, d)| ((h ^ i as u64).wrapping_mul(0x100000001b3) ^ d as u64).wrapping_mul(0x100000001b3));
        println!("DETCAL PLAN {plan} hash={h:016x} results={rh:016x}");
    }
}
