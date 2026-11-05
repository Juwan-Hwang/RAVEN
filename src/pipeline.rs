//! Accepted search stack as one library entry point, shared by `eval harness --pipeline 1`
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
    /// R54：SDX 全度数段自己的 LPX 计划（全度数下 LP/全维码切换点不同：fashion sd40 标定 ef_max 60，sd0 标定 ≈25）
    pub lpx_sdx: Option<LpxPlan>,
}

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

    /// R36-5：k 感知入口（HR 图段 E ≥ k；k ≤ 10 时与 [`Self::searcher`] 完全相同）
    pub fn searcher_k(&self, ef: usize, k: usize) -> GraphSearcher<'_> {
        if let Some(cc) = &self.hr {
            if let Some((e, pc)) = cc.plan_k(ef, k) {
                let mut s = self.searcher_plain(e);
                s.with_cell_confirm(cc, pc);
                return s;
            }
        }
        self.searcher_plain(ef)
    }

    /// 原路径 searcher（不挂 HR）
    pub fn searcher_plain(&self, ef: usize) -> GraphSearcher<'_> {
        if let Some(x) = &self.sdx {
            if ef > x.ef_max { return self.plain_with(x.map(ef, self.lpx_k), 0, self.lpx_sdx.as_ref().or(self.lpx.as_ref())); }
        }
        self.plain_with(ef, self.sd, self.lpx.as_ref())
    }

    /// 原路径 searcher，度数 = self.sd（不经 SDX 计划）
    pub fn searcher_plain_sd(&self, ef: usize) -> GraphSearcher<'_> { self.plain_with(ef, self.sd, self.lpx.as_ref()) }

    /// R54：指定度数与 LPX 计划的原路径 searcher（sd = self.sd、lpx = self.lpx 时与 R42–R53 searcher_plain 逐位相同）
    pub fn plain_with(&self, ef: usize, sd: usize, lpx: Option<&LpxPlan>) -> GraphSearcher<'_> {
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
        // R37 LP：投影空间码排序更粗，rerank 宽度下限 RAVEN_LP_RR（默认 20）
        let rr = if self.proj_m > 0 { self.rerank.max(std::env::var("RAVEN_LP_RR").ok().and_then(|x| x.parse().ok()).unwrap_or(20)) } else { self.rerank };
        s.with_rerank_factor(rr);
        s.with_sq4(&self.sq4);
        s.with_nav_opts(self.nav_opts);
        s.with_l0_opts(self.l0);
        s.with_search_degree(self.sd);
        if self.proj_m > 0 {
            // R42c：投影矩阵已挂在 self.sq4.params.proj 上（search_sq4 自行投影）
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
    if o.nav_rebuild_m > 0 {
        let nav = LayeredNavigation::build(&vectors, dim, graph.storage(), graph.entry_point(), o.nav_rebuild_m, o.nav_rebuild_k);
        graph.set_layered_nav(nav);
    }
    // R37 LP：RAVEN_LP_FILE=<m×dim f32 行主序投影矩阵>，RAVEN_LP_M=m → SQ4 建在投影空间
    let (proj, proj_m) = load_lp_v(dim, &vectors, o.sq4_clip);
    let (perm, sq4) = if proj_m > 0 {
        let pb = project_all(&vectors, dim, &proj, proj_m);
        let rank: usize = std::env::var("RAVEN_LP_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut q = SQ4Dataset::build_uniform_clipped(&pb, proj_m, o.sq4_clip, rank);
        q.params.proj = proj.clone();
        (Vec::new(), q)
    } else {
    let plain = SQ4Dataset::build_clipped(&vectors, dim, o.sq4_clip);
    let perm: Vec<usize> = if o.perm_near { near_perm(&graph, &plain.codes, plain.code_bytes, n, dim) } else { Vec::new() };
    let sq4 = if o.perm_near { drop(plain); SQ4Dataset::build_permuted_clipped(&vectors, dim, &perm, o.sq4_clip) } else { plain };
    (perm, sq4) };
    // R42 LPX：LP 开启时另建全维码（与非 LP 路径相同的 near 维序 + 截断），供高召回段切换（RAVEN_LPX=0 关）
    let lpx_on = proj_m > 0 && std::env::var("RAVEN_LPX").map(|v| v != "0").unwrap_or(true);
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
    if let Some(f) = sq4_full.as_mut() { f.pad_codes(envu("RAVEN_SQ4_PAD", 32)); }
    let mut sq4 = sq4;
    // H87/H89 接受：码长补零到 32B 倍数（GloVe 50B→64B，每码 1 行）。x86 +31–41%（6/6），
    // ARM（本沙箱 aarch64）独立复核 +10–13%（6/6）；两平台分别验证，非移植。RAVEN_SQ4_PAD=0 关闭
    sq4.pad_codes(envu("RAVEN_SQ4_PAD", 32));
    let integral = vectors.iter().all(|&x| x >= 0.0 && x <= 255.0 && x.fract() == 0.0);
    let vec_u8 = if integral {
        let b: Vec<u8> = vectors.iter().map(|&x| x as u8).collect();
        Some(crate::memory::aligned::AlignedBuf::from_slice(&b))
    } else { None };
    if let Some(l) = std::env::var("RAVEN_PFL").ok().and_then(|v| v.parse::<usize>().ok()) {
        crate::graph::vamana::PF_LINES.store(l, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("RAVEN_U8RR").map(|v| v == "1").unwrap_or(false) {
        crate::graph::refiner::U8_RR.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let vectors: crate::memory::aligned::AlignedBuf<f32> = vectors.into();
    let mut p = Prepared { graph, vectors, sq4, perm, l0, nav_opts, sd: 0, rerank: o.rerank, po: o.po, new_to_old, vec_u8, proj, proj_m, hr: None, sq4_full, perm_full, lpx: None, lpx_k: o.k, sdx: None, lpx_sdx: None };
    // H81：po 自动校准（x86 默认开；RAVEN_POAUTO=0/1 覆盖）。与 sd 校准共用中点查询与真值
    let poauto = std::env::var("RAVEN_POAUTO").map(|v| v == "1").unwrap_or(false); // H85 复测撤销 H82b：收益仅 1–5%，默认关
    p.sd = match (o.search_degree, poauto) {
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
    let hr_on = std::env::var("RAVEN_HR").map(|v| v != "0").unwrap_or(o.hr);
    if hr_on && n >= o.hr_min_n {
        p.hr = Some(build_hr(&p.vectors, dim, o));
        if std::env::var("RAVEN_HR_GATE").is_err() {
            let g = calibrate_hr_gate(&p, dim, o);
            // R43-3 KL：标定只决定开/关（门槛网格 200–600 处 KL 计划是否赢）；开启时 KL 计划从 ef ≥ 20 起生效
            let g = if g > 0 && kl_on(o.k, dim) { 20 } else { g };
            if let Some(cc) = p.hr.as_mut() { cc.params.gate = g; }
            if g == 0 { p.hr = None; }
        }
    }
    if p.sd != 0 && p.hr.is_none() && std::env::var("RAVEN_SDX").map(|v| v != "0").unwrap_or(true) {
        // 全度数段 LPX：LP 须便宜 ≥10% 才判赢（RAVEN_SDX_LPXM）。中点标定高估粗码（LP）召回：fashion LP 中点 ef40 0.990 → 测试 ≈0.97，
        // 全维码中点 ef14 0.996 → 测试 0.9925；严格判据在 ef25 处 LP/全维 ≈0.91–1.05 随噪声翻转（ef_max 20/25/∞），∞ 时 @0.99 掉 ≈30%
        if p.sq4_full.is_some() { let m = std::env::var("RAVEN_SDX_LPXM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.9); p.lpx_sdx = Some(calibrate_lpx(&p, dim, o, 0, Some(m))); }
        p.sdx = calibrate_sdx(&p, dim, o);
        if p.sdx.is_none() { p.lpx_sdx = None; }
    }
    p
}

/// R54 SDX 标定：中点查询（与 sd/LPX 标定同法，不碰测试数据）上测截断度数 A = searcher_plain_sd 与全度数 B 的 (ef → 召回, 耗时)，
/// 判定规则与 LPX 相同（A 在同召回下耗时 ≤ B 插值耗时则判赢；首个判负处为 ef_max）；全网格都赢 → None（行为与旧版逐位相同）。
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
    let recall_of = |ef: usize, full: bool| -> f64 {
        let mut s = mk(ef, full); let mut h = 0usize;
        for ci in 0..nc {
            let q = &qv[ci * dim..(ci + 1) * dim];
            h += s.search_auto(q, k).iter().take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        h as f64 / (nc * k) as f64
    };
    let time_of = |ef: usize, full: bool| -> f64 {
        let mut s = mk(ef, full); let mut sink = 0usize;
        let t = Instant::now();
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; sink += s.search_auto(q, k).len(); }
        let el = t.elapsed().as_secs_f64(); std::hint::black_box(sink);
        el * 1e6 / nc as f64
    };
    let efs: Vec<usize> = [10usize, 12, 15, 20, 25, 30, 40, 60, 80, 100, 150].iter().map(|&m| (m * k + 9) / 10).collect();
    let mono = |v: Vec<f64>| -> Vec<f64> { let mut m = 0f64; v.into_iter().map(|x| { m = m.max(x); m }).collect() };
    let ar = mono(efs.iter().map(|&e| recall_of(e, false)).collect());
    let br = mono(efs.iter().map(|&e| recall_of(e, true)).collect());
    let rounds: usize = std::env::var("RAVEN_SDX_CALR").ok().and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    let (mut at, mut bt) = (vec![f64::MAX; efs.len()], vec![f64::MAX; efs.len()]);
    for rd in 0..rounds {
        let order: Vec<usize> = if rd % 2 == 0 { (0..efs.len()).collect() } else { (0..efs.len()).rev().collect() };
        for &i in &order { at[i] = at[i].min(time_of(efs[i], false)); bt[i] = bt[i].min(time_of(efs[i], true)); }
    }
    let mut btm = bt.clone(); for i in (0..btm.len().saturating_sub(1)).rev() { btm[i] = btm[i].min(btm[i + 1]); }
    let b_t_at = |r: f64| -> Option<f64> {
        if r <= br[0] { return Some(btm[0]); }
        for i in 1..efs.len() { if r <= br[i] { let f = if br[i] > br[i - 1] { (r - br[i - 1]) / (br[i] - br[i - 1]) } else { 1.0 };
            return Some((btm[i - 1].ln() + f * (btm[i].ln() - btm[i - 1].ln())).exp()); } }
        None
    };
    let br_max = br[br.len() - 1];
    let margin: f64 = std::env::var("RAVEN_SDX_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let mut ef_max = 0usize; let mut msg = String::new(); let mut streak = true;
    for i in 0..efs.len() {
        // 两条曲线都饱和（≥0.999 且 A 已达 B 的最高召回）时无区分力，沿用前一判定
        let sat = br_max >= 0.999 && ar[i] >= br_max - 1e-9;
        let w = if sat { streak } else { match b_t_at(ar[i]) { Some(t) => at[i] <= margin * t, None => false } };
        msg += &format!(" ef{}: sd{} {:.4}/{:.0}us full {:.4}/{:.0}us{}", efs[i], p.sd, ar[i], at[i], br[i], bt[i], if w { " W" } else { "" });
        if streak && w { ef_max = efs[i]; } else { streak = false; }
    }
    if streak { ef_max = usize::MAX; }
    if let Some(v) = std::env::var("RAVEN_SDX_EFMAX").ok().and_then(|v| v.parse().ok()) { ef_max = v; }
    eprintln!("R54 SDX calib ({nc} mid q, k={k}, {:.1}s):{msg} -> ef_max {ef_max}", t0.elapsed().as_secs_f64());
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
/// x86：IBX 五机 dump（GloVe-100/nytimes HR 门槛、fashion/mnist/gist/arxiv LPX、gist HR）每数据集自由尺度 NNLS，中位相对误差 3.5%；
/// 用它重放计时 dump 的门控决策：GloVe 200、nytimes 200、gist HR 0、gist/mnist LPX ∞ 与计时一致；fashion LPX 40（计时两次分别 ∞ / 60）。
/// aarch64：尚无 ARM dump，暂用同一组相对系数（ARM 上 HR 门控的 HR/原路径比 0.26–0.55，远离 0.95 判据，不敏感）。
pub const COEF_X86: [f64; COST_FEATS] = [0.0, 15.27, 2.703, 0.0, 10.06, 5.206, 0.0, 1.869, 2.697];
pub const COEF_ARM: [f64; COST_FEATS] = COEF_X86;
fn cal_mode() -> (bool, bool) {
    let timed = std::env::var("RAVEN_CAL").map(|v| v == "time").unwrap_or(false);
    let dump = std::env::var("RAVEN_CAL_DUMP").map(|v| v == "1").unwrap_or(false);
    (timed, dump)
}
fn fmt_feats(f: &[f64; COST_FEATS]) -> String { f.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(",") }

/// R33：HR 门槛标定（不碰测试数据）。留一查询：抽 nc 个库内点作查询、精确 top-(k+1) 去掉自身作真值；
/// 原路径在 ef 网格上的 (召回, µs/查询) 曲线 vs HR 在候选门槛 g 处的 (召回, µs)；
/// HR 在 g 处"赢" = 其耗时 ≤ 0.95 × 原路径在同召回处的插值耗时（召回超出原路径网格时与网格最大 ef 比）。
/// 门槛 = 最小的 g，使 g 及所有更大的候选都赢；没有则返回 0（不启用）。
pub fn calibrate_hr_gate(p: &Prepared, dim: usize, o: &PipelineOpts) -> usize {
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
    let mk = |ef: usize, hr: bool| { if hr { let (e, pc) = cc.plan_force(ef); let e = if cc.params.ek { e.max(k + 1) } else { e }; let mut s = p.searcher_plain(e); s.with_cell_confirm(cc, pc); s } else { p.searcher_plain(ef) } };
    let nl_hr = cc.nl;
    let feats_of = std::cell::RefCell::new(std::collections::HashMap::<(usize, bool), [f64; COST_FEATS]>::new());
    let recall_of = |ef: usize, hr: bool| -> f64 {
        let mut s = mk(ef, hr); let mut h = 0usize; let mut fs = [0f64; COST_FEATS];
        for (ci, &qi) in qids.iter().enumerate() {
            let q = &base[qi * dim..(qi + 1) * dim];
            let r = s.search_auto(q, k + 1);
            h += r.iter().filter(|x| x.0 as usize != qi).take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        // 计数遍：novis 关（visited 跟踪 = 取码节点数；novis 路径不计数），只取特征、不影响召回
        let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
        let mut s = mk(ef, hr);
        for &qi in &qids { let q = &base[qi * dim..(qi + 1) * dim]; let _ = s.search_auto(q, k + 1);
            let f = CostModel::feats(&s, k + 1, dim, nl_hr); for i in 0..COST_FEATS { fs[i] += f[i] / nc as f64; } }
        crate::graph::vamana::set_novis(nv);
        feats_of.borrow_mut().insert((ef, hr), fs);
        h as f64 / (nc * k) as f64
    };
    let time_of = |ef: usize, hr: bool| -> f64 {
        let mut s = mk(ef, hr); let mut sink = 0usize;
        let t = Instant::now();
        for &qi in &qids { let q = &base[qi * dim..(qi + 1) * dim]; sink += s.search_auto(q, k + 1).len(); }
        let el = t.elapsed().as_secs_f64(); std::hint::black_box(sink);
        el * 1e6 / nc as f64
    };
    let pefs = [60usize, 80, 100, 130, 160, 200, 250, 300, 400, 500, 600, 800];
    let gates = [200usize, 250, 300, 400, 500, 600];
    let rounds: usize = std::env::var("RAVEN_HR_CALR").ok().and_then(|v| v.parse().ok()).unwrap_or(3).max(1);
    let p_rec: Vec<f64> = pefs.iter().map(|&ef| recall_of(ef, false)).collect();
    let easy_r = p_rec[pefs.iter().position(|&e| e == 200).unwrap()];
    if easy_r >= 0.999 {
        eprintln!("R34 HR gate calib ({nc} LOO q, {:.1}s): plain recall@ef200 {easy_r:.4} >= 0.999 (easy) -> gate 0", t0.elapsed().as_secs_f64());
        return 0;
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
        pts.last().unwrap().1
    };
    let mut wins = Vec::new(); let mut msg = String::new();
    for (i, &g) in gates.iter().enumerate() {
        let (r, t) = (g_rec[i], g_t[i]); let tp = t_at(r);
        let w = t <= 0.95 * tp; wins.push(w);
        msg += &format!(" g{g}: hr r={r:.4} {t:.0}us vs plain {tp:.0}us{}", if w { " WIN" } else { "" });
    }
    let mut gate = 0;
    for i in (0..gates.len()).rev() { if wins[i] { gate = gates[i]; } else { break; } }
    eprintln!("R34 HR gate calib ({nc} LOO q, {}, {:.1}s): plain {} |{msg} -> gate {gate}", if timed { "timed" } else { "cost model" }, t0.elapsed().as_secs_f64(),
              pefs.iter().zip(&plain).map(|(e, (r, t))| format!("ef{e}:{r:.4}/{t:.0}us")).collect::<Vec<_>>().join(" "));
    gate
}

/// R42 LPX 标定（不碰测试数据）：中点查询（库内点与随机图邻居的中点，精确 top-k），ef 网格 = k×{1,1.2,1.5,2,2.5,3,4,6,8,10}；
/// LP 码与全维码各测召回（一次）与耗时（5 轮交错取最小；1000 查询）。LP 网格点"赢" = 其耗时 ≤ 全维码在同召回处的插值耗时
/// （召回低于全维最小 ef 的召回时与全维最小 ef 的耗时比）。ef_max = 从最小 ef 起连续赢的最大网格 ef（首点即输 → 0，全部走全维；全网格都赢 → 无上限）。
/// RAVEN_LPX_EFMAX 覆盖 ef_max（仍做召回标定以得映射）。
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
            let r = s.search_auto(q, k);
            h += r.iter().take(k)
                .filter(|x| crate::distance::l2_simd(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
        }
        let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
        let mut s = mk(ef, full);
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; let _ = s.search_auto(q, k);
            let f = CostModel::feats(&s, k, dim, 0); for i in 0..COST_FEATS { fs[i] += f[i] / nc as f64; } }
        crate::graph::vamana::set_novis(nv);
        feats_of.borrow_mut().insert((ef, full), fs);
        h as f64 / (nc * k) as f64
    };
    let time_of = |ef: usize, full: bool| -> f64 {
        let mut s = mk(ef, full); let mut sink = 0usize;
        let t = Instant::now();
        for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; sink += s.search_auto(q, k).len(); }
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
    let timed = timed0 || (full_pf && !model_forced);
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
    let margin: f64 = match margin_ovr { Some(m) => m, None => std::env::var("RAVEN_LPX_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if timed { 1.0 } else { 0.85 }) };
    let mut ef_max = 0usize; let mut msg = String::new();
    let mut streak = true;
    for i in 0..efs.len() {
        // LP 判赢 = 耗时 ≤ 全维码同召回插值耗时 × RAVEN_LPX_MARGIN（默认 1.0，严格）；
        // LP 召回已达全维码网格最高召回且 ≥0.999（饱和、无区分力）时沿用前一判定
        // R49d：整码预取下默认计时判定时，LP 的 LOO 召回 ≥0.99 也视为饱和（LOO 下全维码比 LP 更早饱和，近 1 的召回差
        // 无区分力；IBX-2 mnist 计时 ef_max 曾在 20/25/30/∞ 间翻转，实测 ∞ 最优；R49b 的 10% 迟滞把 fashion 推到 25–30，弃用）
        let sat = (fr_max >= 0.999 && lr[i] >= fr_max - 1e-9) || (full_pf && !timed0 && lr[i] >= 0.99);
        let w = if sat { streak } else { match full_t_at(lr[i]) { Some(t) => lt[i] <= margin * t, None => false } };
        msg += &format!(" ef{}: lp {:.4}/{:.0}us full {:.4}/{:.0}us{}", efs[i], lr[i], lt[i], fr[i], ft[i], if w { " W" } else { "" });
        if streak && w { ef_max = efs[i]; } else { streak = false; }
    }
    // 全网格都赢 → 不设上限（网格外不切换；否则 gist/mnist 这类 LP 全程占优的集在 ef>10k 时会被错误切回全维码）
    if streak { ef_max = usize::MAX; }
    if let Some(v) = std::env::var("RAVEN_LPX_EFMAX").ok().and_then(|v| v.parse().ok()) { ef_max = v; }
    eprintln!("R42 LPX calib (sd {sd}, {nc} LOO q, k={k}, {:.1}s):{msg} -> ef_max {ef_max}", t0.elapsed().as_secs_f64());
    let mut pl = LpxPlan { ef_max, lp: efs.iter().cloned().zip(lr).collect(), full: efs.iter().cloned().zip(fr).collect(), scale: 1.0 };
    if ef_max > 0 && ef_max < usize::MAX { pl.scale = (pl.match_ef(ef_max, 1) as f64 / ef_max as f64).min(1.0); }
    if let Some(v) = std::env::var("RAVEN_LPX_SCALE").ok().and_then(|v| v.parse().ok()) { pl.scale = v; }
    eprintln!("R42 LPX plan: ef_max {} scale {:.3}", pl.ef_max, pl.scale);
    pl
}

/// R43-3 KL 默认门槛（k ≥ 此值启用大 k 计划）：x86_64 50；aarch64 0（未在 ARM 验证，按 CHARTER 逐平台门控）
#[cfg(target_arch = "x86_64")]
pub const KL_DEFAULT: usize = 50;
#[cfg(not(target_arch = "x86_64"))]
pub const KL_DEFAULT: usize = 0;
/// R43 接力-3：KL 只在 dim ≥ RAVEN_HR_KL_MINDIM（默认 128）启用。IBX-2 GloVe-100 k=100 KL/R42 @0.95 三轮 +3.1/+2.9/+1.9%
/// 未过每轮 ≥+3% 门槛；VIBE glove-200（dim 200）三轮 +30…+233% 过。
pub fn kl_on(k: usize, dim: usize) -> bool {
    let kl: usize = std::env::var("RAVEN_HR_KL").ok().and_then(|v| v.parse().ok()).unwrap_or(KL_DEFAULT);
    let mind: usize = std::env::var("RAVEN_HR_KL_MINDIM").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    kl > 0 && k >= kl && dim >= mind
}

/// R33：构建 HR 模式索引（k-means 结果可缓存到 o.hr_cache；缓存按内部序保存中心与分配）
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
pub fn calibrate_sd(p: &Prepared, dim: usize, o: &PipelineOpts) -> usize {
    calibrate_inner(p, dim, o, false).0
}

pub fn calibrate_sd_po(p: &Prepared, dim: usize, o: &PipelineOpts) -> (usize, usize) {
    calibrate_inner(p, dim, o, true)
}

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
                hit += s.search_sq4(q, k).iter().take(k)
                    .filter(|x| l2(q, &base[x.0 as usize * dim..(x.0 as usize + 1) * dim]) <= thr[ci]).count();
            }
            let cm = CostModel::platform();
            let nv = crate::graph::vamana::novis_mode(); crate::graph::vamana::set_novis(0);
            let mut c = 0f64;
            for ci in 0..nc { let q = &qv[ci * dim..(ci + 1) * dim]; let _ = s.search_sq4(q, k); c += cm.cost(&CostModel::feats(&s, k, dim, 0)); }
            crate::graph::vamana::set_novis(nv);
            return (hit as f64 / (nc * k) as f64, c * 1e-9);
        }
        for rep in 0..3 {
            let t = Instant::now();
            let mut h = 0usize;
            for ci in 0..nc {
                let q = &qv[ci * dim..(ci + 1) * dim];
                h += s.search_sq4(q, k).iter().take(k)
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
                for ci in 0..nc { std::hint::black_box(s.search_sq4(&qv[ci * dim..(ci + 1) * dim], k)); }
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
                    for ci in off..off + half { std::hint::black_box(s.search_sq4(&tq[ci * dim..(ci + 1) * dim], k)); }
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

fn load_lp(dim: usize) -> (Vec<f32>, usize) { load_lp_v(dim, &[], -1.0) }

/// R37 LP：自动投影。RAVEN_LP=auto（或 =m 指定维数）时在库内由样本协方差的子空间迭代求前 m 主子空间，
/// 再乘随机正交阵均衡各维方差；门控：dim ≥ RAVEN_LP_MINDIM(256) 且前 m 维方差占比 ≥ RAVEN_LP_FRAC(0.90)。
/// R40：RAVEN_LP_GATE=fid 时改用"保真度门控"：构建期 LOO（库内样本作查询、子样本上精确 top-C 候选），
/// 比较 LP-m 码与全维 SQ4 码"真 top-k ∈ 近似 top-4k"的比例，相对保真度 ≥ RAVEN_LP_FIDTHR（默认 0.98；Python 校准：fashion LP128 0.9925、gist LP192 0.986、arxiv LP192 0.9925 开，nytimes ≤0.93 关）取最小 m。
/// 方差占比只是代理（VIBE arxiv-768：varfrac(256)=0.896 被 0.90 门槛误关，而 LP192 保真度 0.9925 与 fashion LP128 同档）。
fn load_lp_v(dim: usize, vectors: &[f32], clip: f32) -> (Vec<f32>, usize) {
    // R41：x86_64 默认 RAVEN_LP=auto（方差占比门控；IBX 门控 (b)：fashion +35%、mnist +64…+87%、gist ≈3.7× @0.95；
    // dim<256 的集合与 nytimes 不进入，逐位不变）；aarch64 默认仍关。RAVEN_LP=0 可关。
    let lp_env = std::env::var("RAVEN_LP").ok().or_else(|| if cfg!(target_arch = "x86_64") && std::env::var("RAVEN_LP_M").is_err() { Some("auto".to_string()) } else { None });
    if let Some(v) = lp_env {
        if v != "0" && !vectors.is_empty() {
            let mind: usize = std::env::var("RAVEN_LP_MINDIM").ok().and_then(|x| x.parse().ok()).unwrap_or(256);
            // R42：阈值 0.90→0.88（arxiv-768 m=256 varfrac 0.896、码保真 0.991、IBX 实测 LP256 @0.95 ≈1.6× 被 0.90 误关；
            // 已测集合选择不变：fashion 0.928/mnist 0.937（m=128）、gist m=128 0.865 仍关→m=192 0.909；高召回段由 LPX 双码计划兜底）
            let fth: f64 = std::env::var("RAVEN_LP_FRAC").ok().and_then(|x| x.parse().ok()).unwrap_or(0.88);
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
                    let on = frac >= fth;
                    eprintln!("R37 LP auto m={m} varfrac={frac:.3} thr={fth} -> {} ({:.1}s)", if on { "ON" } else { "off" }, t.elapsed().as_secs_f64());
                    if on { return (p, m); }
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
