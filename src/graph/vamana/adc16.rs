//! R68 ADC16 v3：低维（SQ4 码补到 16 B，dim ≤ 32）非对称 L0 导航的**独立搜索链**。
//!
//! 本文件中的函数是 `vamana.rs` 中对应函数的逐字拷贝，仅把 SQ4 对称距离 `distance_raw[_unchecked]`
//! 换成 [`adc16_dist`]（数据 4 bit 码、查询 f32、逐维步长加权）、查询编码换成 [`SQ4Params::encode_adc16_into`]。
//! 分派在管线构造时一次决定（`Prepared::adc16`），调用方选 [`GraphSearcher::search_auto_adc`] 或
//! `search_auto`；`vamana.rs` 中 主线 的缺省搜索函数一个字节都没有改（R67 v1/v2 的跨进程回退来自
//! 缺省路径函数内的分派/距离分支）。
#![allow(clippy::too_many_arguments, unused_unsafe, clippy::needless_return)]
use super::*;
use crate::quant::sq4::adc16_dist;

/// SAFETY: idx < sq4.n；query_code 为 256 B ADC16 查询
#[inline(always)]
unsafe fn adc_u(sq4: &SQ4Dataset, query_code: &[u8], idx: usize) -> f32 { adc16_dist(query_code, sq4.code_unchecked(idx)) }

impl VamanaGraph {
    /// SQ4 标量量化图遍历
    ///
    /// 与 `greedy_search_sq8` 相同的 Two-Pass Prefetch 架构，
    /// 但距离计算用 SQ4 packed 4-bit 码代替 SQ8 u8 码。
    ///
    /// 优势：
    /// - 内存带宽降 2x（64B/vector vs SQ8 的 128B for SIFT-128）
    /// - AVX2 每次处理 64 维（32 bytes packed），SIFT-128 仅 2 次迭代（vs SQ8 的 4 次）
    /// - 1M 向量仅 64MB，部分进入 L3 cache（vs SQ8 的 128MB 完全不进 L3）
    ///
    /// 返回 (节点ID, SQ4距离) 列表，需外部 f32 rerank。
    pub(super) fn greedy_search_sq4_adc(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        entry_point: u32,
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
    ) -> Vec<(u32, f32)> {
        visited.reset();
        pool.reset_for(l);

        // 入口节点距离（SQ4）
        let entry_dist = adc16_dist(query_code, sq4.code(entry_point as usize));
        visited.visit(entry_point);
        pool.insert(entry_point, entry_dist);

        Self::sq4_expand_adc::<false>(sq4, storage, query_code, visited, pool, po, search_degree);
        pool.to_sorted_vec()
    }

    /// H09：多种子 SQ4 图遍历。`seeds` 为 (节点, SQ4 raw 距离)，调用方保证已在 `visited` 中标记
    /// （导航层轨迹节点全部保持 visited，不重复计算）。不 reset `visited`。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn greedy_search_sq4_seeded_adc(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        seeds: &[(u32, f32)],
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
    ) -> Vec<(u32, f32)> {
        Self::greedy_search_sq4_seeded_nv_adc(sq4, storage, seeds, query_code, l, visited, pool, po, search_degree, false)
    }

    /// R27 S1：同 [`greedy_search_sq4_seeded`]，`novis = true` 时 L0 不用 visited（H153）。调用方必须保证：
    /// 所有已在 `visited` 中标记的节点（导航轨迹）都已作为种子提供（插入或被满池拒绝），否则不再逐位等价。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn greedy_search_sq4_seeded_nv_adc(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        seeds: &[(u32, f32)],
        query_code: &[u8],
        l: usize,
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
        novis: bool,
    ) -> Vec<(u32, f32)> {
        pool.reset_for(l);
        for &(v, d) in seeds {
            pool.insert(v, orx_dist(v).unwrap_or(d));
        }
        if novis { Self::sq4_expand_adc::<true>(sq4, storage, query_code, visited, pool, po, search_degree); }
        else { Self::sq4_expand_adc::<false>(sq4, storage, query_code, visited, pool, po, search_degree); }
        pool.to_sorted_vec()
    }

    /// SQ4 图遍历主循环（greedy_search_sq4 / _seeded 共用，inline(always) 保持热路径代码不变）
    #[inline(always)]
    fn sq4_expand_adc<const NV: bool>(
        sq4: &SQ4Dataset,
        storage: &HybridBlockedCsr,
        query_code: &[u8],
        visited: &mut VisitedTracker,
        pool: &mut LinearPool,
        po: usize,
        search_degree: usize,
    ) {
        let mut edge_buf: [u32; 128] = [0; 128];
        let hsel_rec = HSEL_REC_PTR.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let hsel_k: usize = 0; // R68：ADC16 链不支持 R23 HSEL/HREL（实验特性，缺省关；search_sq4_adc 入口断言），不调 hsel_query/hrel_query，保证 main 的 hsel_query 仍是单调用者（objdump 同构）
        let hsel_epf = HSEL_EPF.load(std::sync::atomic::Ordering::Relaxed);
        let hsel_rpf = HSEL_RPF.load(std::sync::atomic::Ordering::Relaxed);
        let hsel_lut: *const f32 = HSEL_LUT.with(|l| l.get() as *const f32);
        // R23 HREL：每边残差 PCA 投影（行内顺序块，提前一跳预取）
        let hrel_base = HREL_PTR.load(std::sync::atomic::Ordering::Relaxed) as *const u8;
        let hrel_k: usize = 0; let _ = hrel_base;
        let hrel_bsz = HREL_BSZ.load(std::sync::atomic::Ordering::Relaxed);
        let hrel_epf = HREL_EPF.load(std::sync::atomic::Ordering::Relaxed);
        let hrel_pq: *const f32 = HREL_PQ.with(|l| l.get() as *const f32);
        let mut hrel_sc: [f32; 128] = [0.0; 128];
        let mut slot_buf: [u8; 128] = [0; 128];
        // H128：前瞻流水线移植到精简路径（第 11 轮：aarch64 默认 L0EA=0 后默认走本路径，原 L0 路径的 la 不再生效）。
        // ARM 4 轮：SIFT +3.2–5.7%（8/8），GloVe +2.0–6.5%（8/8 为正，均值 2.9% <3%）→ 默认关；RAVEN_LEAN_LA=8 可开，AB 旋钮 lla
        let la = lean_la_depth();
        let mut adj_ready: [u32; 2] = [u32::MAX; 2];
        let mindist_lambda = if cfg!(feature = "diag") { f32::from_bits(MINDIST_LAMBDA.load(std::sync::atomic::Ordering::Relaxed)) } else { 0.0 };
        let mindist_r = MINDIST_R.load(std::sync::atomic::Ordering::Relaxed) as *const f32;
        // A1（第 12 轮）：精简路径轨迹记录（仅 diag 构建；release 热路径不变）
        let a1 = cfg!(feature = "diag") && A1_TRACE_ON.load(std::sync::atomic::Ordering::Relaxed);
        if a1 { a1_begin(visited, pool); }
        let vh = vec_hop();
        let orx = cfg!(feature = "diag") && ORX_Q.load(std::sync::atomic::Ordering::Relaxed) != 0;
        let pfl = { let cl = sq4.code_bytes.div_ceil(64); match PF_LINES.load(std::sync::atomic::Ordering::Relaxed) { usize::MAX => if cl >= 4 { cl } else { 1 }, l => l.min(cl).max(1) } };
        #[cfg(target_arch = "aarch64")]
        let qdec = if vh & 4 != 0 && sq4.code_bytes == 64 { Some(crate::quant::sq4::QDec64::new(query_code)) } else { None };
        // R25 CM（代价模型裁决 oracle，仅 diag_cm 构建）：见 CmState
        let cm = cfg!(feature = "diag_cm") && CM_ON.load(std::sync::atomic::Ordering::Relaxed);
        let cmp: *mut CmState = if cm { CM.with(|c| c.as_ptr()) } else { std::ptr::null_mut() };
        let (cm_rec, cm_thr, cm_chain, cm_extra, cm_seed, cm_block) = if cm { let c = unsafe { &*cmp }; (c.rec, c.keep_thr, c.chain, c.extra, c.seed, c.block) } else { (false, u32::MAX, 0, 0, 0, 0) };
        let cm_zero: u32 = if cm { std::hint::black_box(0u32) } else { 0 };
        let cm_ah: u32 = if cm && !cm_rec { unsafe { (*cmp).ah } } else { 0 };
        let mut cm_hi: usize = 0;
        let (cm_tab, cm_mask) = if cm { (CM_TAB.load(std::sync::atomic::Ordering::Relaxed) as *const u32, CM_MASK.load(std::sync::atomic::Ordering::Relaxed)) } else { (std::ptr::null(), 0) };
        let mut cm_sink = 0f32;
        let cmp_u = cmp as usize; let cm_keep = move |v: u32| -> bool { (cm_mix(v ^ cm_seed) <= cm_thr) | unsafe { (*(cmp_u as *const CmState)).contains(v) } };
        // R27 ZA：热态逐项消融（计时轮；全部位都让池/轨迹/结果与基线逐位一致）。位：1 距离核→记录距离（不读码）· 2 距离核→记录距离 + 依赖读码首末字节
        // （理想计算、真实访存）· 4 去掉 L0 全部码预取指令 · 8 门槛比较理想化（不对逐码调 insert，跳末只插入记录的入池者）· 16 入池理想化（不插入，结束时装入记录的池；隐含 8|32）
        // · 32 弹出回放（按记录弹出序列，不调 pool.pop/peek）· 64 visited 回放（按记录的 is_new，不碰位图/history/visited 预取）· 128 邻接回放（直接装入记录的新邻居表；隐含 64）
        // · 256 整个 L0 循环置零（只装入记录的结果池）· 512 visited 不写 history（调用方在计时外按记录清位图）
        let cm_za: u32 = if cm && !cm_rec { unsafe { (*cmp).za } } else { 0 };
        if cm_za & 256 != 0 { pool.load_sorted(unsafe { &(*cmp).q_fin }); return; }
        let (z_kz, z_kl, z_pf, z_th, z_pi, z_pr, z_vr, z_ar, z_hz) = (cm_za & 1 != 0, cm_za & 2 != 0, cm_za & 4 != 0, cm_za & 24 != 0, cm_za & 16 != 0, cm_za & 48 != 0, cm_za & 192 != 0, cm_za & 128 != 0, cm_za & 512 != 0);
        let (mut z_di, mut z_vi) = (0usize, 0usize);
        let (z_qpop, z_npop, z_qisn, z_qdist): (*const u32, usize, *const u8, *const f32) = if cm_za != 0 { let c = unsafe { &*cmp }; (c.q_pop.as_ptr(), c.q_pop.len(), c.q_isn.as_ptr(), c.q_dist.as_ptr()) } else { (std::ptr::null(), 0, std::ptr::null(), std::ptr::null()) };
        let mut z_sink = 0u32;
        let cm_filt = cm && cm_thr != u32::MAX; // R27：f=1 时不做（全保留的）过滤，去掉每邻居一次哈希查表的测量开销

        // N4（第 13 轮）：确认阶段批量化（RAVEN_N4 = T | T'<<8 | H<<16 | flags<<24；0 = 关）
        let n4 = n4_cfg();
        let (n4_t, n4_tp, n4_h, n4_f) = ((n4 & 0xFF) as usize, ((n4 >> 8) & 0xFF) as usize, ((n4 >> 16) & 0xFF) as usize, n4 >> 24);
        let mut n4_hops = 0usize;
        // R27 S1：无 visited 仅在精确等价的纯路径上生效（诊断/选择类旋钮会撤销 visited 或改变候选集 → 回退到 visited）
        let nv = storage.r_max() <= 128 && NV && hsel_k == 0 && hrel_k == 0 && !cm && n4_t == 0 && !a1 && !orx && mindist_lambda == 0.0
            && G1_K.load(std::sync::atomic::Ordering::Relaxed) == 0;
        let nvrf = NvRf::begin(nv);
        let s2 = nv && s2_mode() != 0;
        loop {
            if n4_t > 0 && n4_hops >= n4_h && pool.cursor() >= n4_t && pool.has_next() {
                n4_batch(sq4, storage, query_code, visited, pool, search_degree, n4_tp, n4_f);
                continue;
            }
            let node = if z_pr { if cm_hi >= z_npop { break } unsafe { *z_qpop.add(cm_hi) } } else { let Some((node, _dist)) = pool.pop() else { break }; node };
            pop_rec(node);
            n4_hops += 1;
            crate::diag_count!(HOPS);
            // R25 CM：每跳串行插入 cm_chain 次相互依赖的随机访存，并让本跳邻接行地址依赖其结果（node + (x & 0)）
            let node = if cm && cm_chain > 0 && !cm_rec {
                let mut x = node as usize;
                for c in 0..cm_chain { x = unsafe { *cm_tab.add((cm_mix(x as u32 ^ c) as usize) & cm_mask) } as usize; }
                node + (x as u32 & cm_zero)
            } else { node };
            if cm { unsafe { (*cmp).hops += 1; } }
            if cm_rec { let c = unsafe { &mut *cmp }; c.r_pop.push(node); let l = c.r_new.len() as u32; c.r_off.push(l); let a = c.r_acc2.len() as u32; c.r_aoff.push(a); }
            // R26 H148 oracle：按记录轨迹提前 D 跳预取邻接行、提前 Dc 跳预取码（完美弹出序列预测的上界；预取本身计时）
            if cm_ah != 0 {
                let c = unsafe { &*cmp };
                let (d, dc, md) = ((cm_ah & 15) as usize, ((cm_ah >> 4) & 15) as usize, cm_ah >> 8);
                if md & 1 != 0 { if let Some(&u) = c.q_pop.get(cm_hi + d) {
                    let er = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                    let rp = storage.main_block().as_ptr().wrapping_add(u as usize * storage.r_max()) as *const i8;
                    for l in 0..(er * 4).div_ceil(64) { unsafe { if md & 8 != 0 { crate::arch::prefetch_l2(rp.add(l * 64)); } else { crate::arch::prefetch_l1(rp.add(l * 64)); } } } } }
                // R26 H150（md&8）：远期预取改发 L2 提示（避开 L1 填充缓冲上限）；md&16：同时预取第 i+Dc 跳新邻居的 visited 位图行
                if md & 22 != 0 { let j = cm_hi + dc; if j + 1 < c.q_off.len() {
                    for &v in &c.q_new[c.q_off[j] as usize..c.q_off[j + 1] as usize] {
                        if md & 16 != 0 { unsafe { visited.prefetch(v); } }
                        if md & 4 != 0 || (md & 2 != 0 && c.contains(v)) { let cp = sq4.code(v as usize).as_ptr() as *const i8;
                            unsafe { if md & 8 != 0 { crate::arch::prefetch_l2(cp); } else { crate::arch::prefetch_l1(cp); } } } } } }
            }
            cm_hi += 1;
            // CM 行内块（Descartes 式"邻居码内联"代价模型）：每跳一个由 node 决定位置的连续 cm_block 行块，跳开始时整块预取，
            // 在本跳距离计算之前逐行读取（块是本跳决策的输入 → 在链上）
            let cm_blk: usize = if cm && cm_block > 0 && !cm_rec {
                let b = (cm_mix(node ^ 0xB10C) as usize & cm_mask) & !15usize; let b = b.min(cm_mask + 1 - 16 * cm_block as usize);
                for l in 0..cm_block as usize { unsafe { crate::arch::prefetch_l1(cm_tab.add(b + l * 16) as *const i8); } }
                b } else { usize::MAX };
            // CM 额外取码：位 31 置位时在跳开始（弹出后）一次性预取全部额外码（与真实码的早预取同位置，不在链上等）
            if cm && cm_extra & 0x8000_0000 != 0 && !cm_rec {
                let n = sq4.n as u32; let base = cm_mix(node ^ cm_seed);
                for j in 0..(cm_extra & 0xFFFF) { let v = cm_mix(base.wrapping_add(j)) % n; unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); } }
            }
            if a1 { A1_TRACE.with(|t| { let mut t = t.borrow_mut(); t.push(A1_HOP); t.push(node); }); }
            let mut edge_size = 0usize;
            if z_ar {
                // R27 ZA 128：邻接回放——不读邻接行/度数、不做 visited，直接装入记录的本跳新邻居
                let c = unsafe { &*cmp }; let (a, b) = (c.q_off[cm_hi - 1] as usize, c.q_off[cm_hi] as usize);
                edge_buf[..b - a].copy_from_slice(&c.q_new[a..b]); edge_size = b - a;
            } else {
            if la > 0 {
                let mut nr: [u32; 2] = [u32::MAX; 2];
                if let Some((c, _)) = pool.peek_unchecked() {
                    nr[0] = c;
                    if c == adj_ready[0] || c == adj_ready[1] {
                        let nb = storage.neighbors(c);
                        let lim = la.min(nb.len()).min(if search_degree > 0 { search_degree } else { usize::MAX });
                        for &v in &nb[..lim] {
                            unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); }
                        }
                    }
                }
                if let Some(c2) = pool.peek_second_unchecked() {
                    nr[1] = c2;
                    let ptr = storage.main_block().as_ptr().wrapping_add(c2 as usize * storage.r_max()) as *const i8;
                    let effective_r = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                    unsafe { for l in 0..(effective_r * 4).div_ceil(64) { crate::arch::prefetch_l1(ptr.add(l * 64)); } }
                    // R14（vh 位 64）：度数数组是邻接切片长度的来源，另一次随机访存 → 与邻接行一起预取
                    if vh & 64 != 0 { unsafe { crate::arch::prefetch_l1(storage.degrees_ptr().wrapping_add(c2 as usize) as *const i8); } }
                }
                adj_ready = nr;
            }
            // Multi-line graph prefetch（尊重 search_degree：只预取实际会读取的字节）
            let z_pk = if z_pr { if cm_hi < z_npop { Some((unsafe { *z_qpop.add(cm_hi) }, 0f32)) } else { None } } else { pool.peek_unchecked() };
            if let Some((next_node, _)) = z_pk {
                if vh & 64 != 0 { unsafe { crate::arch::prefetch_l1(storage.degrees_ptr().wrapping_add(next_node as usize) as *const i8); } }
                let start = next_node as usize * storage.r_max();
                let ptr = storage.main_block().as_ptr().wrapping_add(start) as *const i8;
                // search_degree > 0 时只预取前 search_degree 个邻居的字节
                let effective_r = if search_degree > 0 {
                    search_degree.min(storage.r_max())
                } else {
                    storage.r_max()
                };
                let neighbor_bytes = effective_r * 4;
                let graph_lines = neighbor_bytes.div_ceil(64);
                unsafe {
                    crate::arch::prefetch_l1(ptr);
                    for l in 1..graph_lines {
                        crate::arch::prefetch_l1(ptr.add(l * 64));
                    }
                }
                if hrel_k > 0 { let b = hrel_base.wrapping_add(next_node as usize * hrel_bsz) as *const i8; unsafe { for l in 0..hrel_bsz / 64 { crate::arch::prefetch_l1(b.add(l * 64)); } } }
            }

            // R14b（vh 位 128）：不读 degrees（跳链上一次独立随机访存），直接取邻接行前 cap 槽，按 SENTINEL 计数得长度
            // （行按序填充、空槽为 SENTINEL；cap 槽已随邻接行预取进 L1）。与原路径逐位一致。
            let (neighbors, effective_len) = if vh & 128 != 0 {
                let cap = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                let row = unsafe { std::slice::from_raw_parts(storage.main_block().as_ptr().add(node as usize * storage.r_max()), cap) };
                let mut c = 0usize; for &x in row { c += (x != crate::SENTINEL) as usize; }
                (&row[..c], c)
            } else {
                let neighbors = storage.neighbors(node);
                // 搜索时度数上限：只看前 search_degree 个邻居（已按距离升序排列）
                // search_degree=0 表示不限制
                let effective_len = if search_degree > 0 {
                    search_degree.min(neighbors.len())
                } else {
                    neighbors.len()
                };
                (neighbors, effective_len)
            };

            // 第一遍：收集未访问邻居（branchless — A8）
            //
            // Visited 预取：visited 数组 = N bytes（1M = 1MB），超出 L2 cache。
            // 每个邻居的 visited 检查是随机访问，提前 po 步预取隐藏延迟。
            // H136（第 12 轮，A1 冷回放发现 visited 过滤位于"邻接→码取"关键路径上）：RAVEN_VH 位 8/16/32 = 过滤前直接按原始邻接
            // 预取前 8/16/全部 个邻居的码（已访问者的码多半已在缓存，浪费小），让首批码取不等 visited 过滤。结果逐位一致（只加预取）
            // R23 HREL：先对全部邻居打分（行内块已提前一跳预取），早码预取改为按分数取前 hrel_epf 个
            // R26 段归因：记录本跳邻接行（与上方预取同口径：前 effective_r 个槽的行）及度数项
            if cm_rec {
                let c = unsafe { &mut *cmp };
                let er = if search_degree > 0 { search_degree.min(storage.r_max()) } else { storage.r_max() };
                let rp = storage.main_block().as_ptr().wrapping_add(node as usize * storage.r_max()) as usize;
                for l in 0..(er * 4).div_ceil(64) { c.r_adj.push(rp + l * 64); }
                if vh & 128 == 0 { c.r_adj.push(storage.degrees_ptr().wrapping_add(node as usize) as usize); }
            }
            if hrel_k > 0 {
                unsafe { hrel_scores(hrel_base.add(node as usize * hrel_bsz), hrel_pq, effective_len, &mut hrel_sc); }
                if hrel_epf > 0 {
                    let mut tmp: [(f32, u8); 128] = [(0.0, 0); 128];
                    for i in 0..effective_len { tmp[i] = (hrel_sc[i], i as u8); }
                    let e = hrel_epf.min(effective_len);
                    if e < effective_len { tmp[..effective_len].select_nth_unstable_by(e - 1, |a, b| a.0.total_cmp(&b.0)); }
                    for t in &tmp[..e] { unsafe { crate::arch::prefetch_l1(sq4.code(neighbors[t.1 as usize] as usize).as_ptr() as *const i8); } }
                }
            } else if hsel_k > 0 {
                if hsel_rpf { for &v in neighbors.iter().take(effective_len) { unsafe { crate::arch::prefetch_l1(hsel_rec.add(v as usize * HSEL_REC) as *const i8); } } }
                let n_epf = hsel_epf.min(effective_len);
                for &v in neighbors.iter().take(n_epf) { unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); } }
            } else if vh & 56 != 0 && !z_pf {
                let n_epf = if vh & 32 != 0 { effective_len } else if vh & 16 != 0 { 16 } else { 8 }.min(effective_len);
                if cm_filt && !cm_rec {
                    // CM：完美选择下不会取被丢弃者的码 → 早码预取也只预取保留者
                    for &v in neighbors.iter().take(n_epf) { let k = cm_keep(v); let a = if k { sq4.code(v as usize).as_ptr() } else { query_code.as_ptr() }; unsafe { crate::arch::prefetch_l1(a as *const i8); } }
                } else {
                for &v in neighbors.iter().take(n_epf) { unsafe { pf_code_lines(sq4.code(v as usize).as_ptr() as *const i8, pfl); } }
                }
            }
            let pf_vis = if z_vr || nv { 0 } else { po.min(effective_len) };
            for &v in neighbors.iter().take(pf_vis) {
                unsafe { visited.prefetch(v); }
            }

            // G1 oracle（第 13 轮）：只保留与 (q−u) 夹角最小的前 g1k 个邻居（作弊：用真 q 的 f32 方向）
            let g1k = G1_K.load(std::sync::atomic::Ordering::Relaxed);
            let mut g1_buf: [u32; 128] = [0; 128];
            let (neighbors, effective_len) = if g1k > 0 && effective_len > g1k { let n = g1_select(node, &neighbors[..effective_len], g1k, &mut g1_buf); (&g1_buf[..n], n) } else { (neighbors, effective_len) };
            if nv {
                // R27 S1：不过滤，原样取全部邻居（重复者在入池时由门槛/去重拒绝）
                let n = effective_len.min(128);
                edge_size = nvrf.fill(&neighbors[..n], &mut edge_buf);
            } else {
            for (i, &v) in neighbors.iter().take(effective_len).enumerate() {
                if edge_size >= 128 {
                    break;
                }
                if i + po < effective_len && !z_vr {
                    unsafe { visited.prefetch(neighbors[i + po]); }
                }
                if cm_rec { unsafe { (*cmp).r_vis.push(v); } }
                let is_new = if z_vr { let b = unsafe { *z_qisn.add(z_vi) } != 0; z_vi += 1; b }
                    else if z_hz { unsafe { visited.visit_nohist(v) } }
                    else { unsafe { visited.visit_unchecked_branchless(v) } };
                if cm_rec { unsafe { (*cmp).r_isn.push(is_new as u8); } }
                unsafe { *edge_buf.get_unchecked_mut(edge_size) = v; *slot_buf.get_unchecked_mut(edge_size) = i as u8; }
                edge_size += is_new as usize;
            }
            } // R27 S1 nv else 结束
            if hrel_k > 0 && edge_size > hrel_k && pool.is_full() {
                let mut sc: [(f32, u32); 128] = [(0.0, 0); 128];
                for j in 0..edge_size { sc[j] = (unsafe { *hrel_sc.get_unchecked(*slot_buf.get_unchecked(j) as usize) }, edge_buf[j]); }
                sc[..edge_size].select_nth_unstable_by(hrel_k - 1, |a, b| a.0.total_cmp(&b.0));
                for j in hrel_k..edge_size { visited.unvisit(sc[j].1); }
                for j in 0..hrel_k { edge_buf[j] = sc[j].1; }
                edge_size = hrel_k;
            }
            // R23 HSEL：池满且新邻居多于 k 时，用 1-bit RaBitQ 全局记录估计距离，只保留前 k 个取 SQ4（其余撤销 visited，可被再次发现）
            if hsel_k > 0 && edge_size > hsel_k && pool.is_full() {
                edge_size = unsafe { hsel_select(hsel_rec, hsel_lut, &mut edge_buf, edge_size, hsel_k, visited) };
            }

            // R25 CM：固定轨迹的选择 oracle——只保留"记录轮中插入池成功"者 + 被拒者中按哈希比例 f 保留的部分（被拒者不改变池 → 轨迹逐位一致）
            if cm {
                let c = unsafe { &mut *cmp };
                c.n_new += edge_size as u64;
                if cm_rec { c.r_new.extend_from_slice(&edge_buf[..edge_size]); }
                if !cm_rec && cm_filt {
                    let mut k = 0usize;
                    for j in 0..edge_size { let v = edge_buf[j]; edge_buf[k] = v; k += cm_keep(v) as usize; }
                    edge_size = k;
                }
                c.n_kept += edge_size as u64;
                if cm_blk != usize::MAX { let mut acc = 0u32; for l in 0..cm_block as usize { acc = acc.wrapping_add(unsafe { *cm_tab.add(cm_blk + l * 16) }); }
                    edge_size += (acc & cm_zero) as usize; }
            }
            } // R27 ZA 128 else 结束
            // 预取前 po 个邻居的 SQ4 码
            // SIFT-128: 64B = 1 cache line，只需 1 条预取指令
            let prefetch_count = if z_pf { 0 } else { po.min(edge_size) };
            for &node in edge_buf.iter().take(prefetch_count) {
                let v = node as usize;
                let ptr = sq4.code(v).as_ptr() as *const i8;
                unsafe { pf_code_lines(ptr, pfl); }
            }

            // R28 S2（H152 批量归并池，RAVEN_S2 / AB s2，仅 S1 无 visited 生效时）：整跳先算完距离写 dbuf，再一次并入池
            // （跳初门槛压缩 → 稳定排序 → 去重 → 反向原地归并；与逐个 insert_dedup 逐位一致，见 LinearPool::merge_batch_dedup）
            if s2 {
                let mut dbm: [std::mem::MaybeUninit<f32>; 128] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                let dbuf = unsafe { &mut *(dbm.as_mut_ptr() as *mut [f32; 128]) };
                for i in 0..edge_size {
                    if i + po < edge_size && !z_pf {
                        let v = unsafe { *edge_buf.get_unchecked(i + po) } as usize;
                        unsafe { pf_code_lines(sq4.code(v).as_ptr() as *const i8, pfl); }
                    }
                    let nb = unsafe { *edge_buf.get_unchecked(i) } as usize;
                    #[cfg(target_arch = "aarch64")]
                    let d = if let Some(qd) = &qdec { unsafe { qd.dist(sq4.code_unchecked(nb).as_ptr()) } } else { unsafe { adc_u(sq4, query_code, nb) } };
                    #[cfg(not(target_arch = "aarch64"))]
                    let d = unsafe { adc_u(sq4, query_code, nb) };
                    unsafe { *dbuf.get_unchecked_mut(i) = d; }
                }
                pool.merge_batch_dedup_r68(&edge_buf[..edge_size], &dbuf[..edge_size]);
                continue;
            }
            // H130（第 12 轮）：向量化跳执行（RAVEN_VH 位掩码，默认 0）：1 = 无分支插入位置，2 = 先整批算距离再按跳初阈值掩码过滤，
            // 4 = 64B 码查询预解码核。全部与原路径逐位一致（阈值只会单调变小，被掩码过滤者必被原 insert 拒绝）。
            if vh & 7 != 0 && !(cfg!(feature = "diag") && (mindist_lambda > 0.0 || a1 || orx)) {
                let mut dbuf: [f32; 128] = [0.0; 128];
                for i in 0..edge_size {
                    if i + po < edge_size {
                        let v = unsafe { *edge_buf.get_unchecked(i + po) } as usize;
                        unsafe { pf_code_lines(sq4.code(v).as_ptr() as *const i8, pfl); }
                    }
                    let nb = unsafe { *edge_buf.get_unchecked(i) } as usize;
                    #[cfg(target_arch = "aarch64")]
                    let d = if let Some(qd) = &qdec { unsafe { qd.dist(sq4.code_unchecked(nb).as_ptr()) } } else { unsafe { adc_u(sq4, query_code, nb) } };
                    #[cfg(not(target_arch = "aarch64"))]
                    let d = unsafe { adc_u(sq4, query_code, nb) };
                    unsafe { *dbuf.get_unchecked_mut(i) = d; }
                    if vh & 2 == 0 {
                        if nv { pool.insert_dedup(nb as u32, d); } else if vh & 1 != 0 { pool.insert_bl(nb as u32, d); } else { pool.insert(nb as u32, d); }
                    }
                }
                if vh & 2 != 0 {
                    let w = if pool.is_full() { pool.worst_distance() } else { f32::INFINITY };
                    let mut base = 0usize;
                    while base < edge_size {
                        let end = (base + 64).min(edge_size);
                        let mut m: u64 = 0;
                        for j in base..end { m |= ((unsafe { *dbuf.get_unchecked(j) } < w) as u64) << (j - base); }
                        while m != 0 {
                            let j = base + m.trailing_zeros() as usize; m &= m - 1;
                            let (nb, d) = unsafe { (*edge_buf.get_unchecked(j), *dbuf.get_unchecked(j)) };
                            if nv { pool.insert_dedup(nb, d); } else if vh & 1 != 0 { pool.insert_bl(nb, d); } else { pool.insert(nb, d); }
                        }
                        base = end;
                    }
                }
                continue;
            }
            if N4_STATS.load(std::sync::atomic::Ordering::Relaxed) { N4_CNT.with(|c| c.borrow_mut()[2] += edge_size as u64); }
            // 第二遍：SQ4 距离计算 + 前瞻预取
            for i in 0..edge_size {
                if i + po < edge_size && !z_pf {
                    let v = edge_buf[i + po] as usize;
                    let ptr = sq4.code(v).as_ptr() as *const i8;
                    unsafe {
                        crate::arch::prefetch_l1(ptr);
                    }
                }
                let neighbor = edge_buf[i];
                let d = if z_kz || z_kl {
                    let rd = unsafe { *z_qdist.add(z_di) }; z_di += 1;
                    if z_kl { let cp = unsafe { sq4.code_unchecked(neighbor as usize) };
                        let b = unsafe { (*cp.as_ptr() ^ *cp.as_ptr().add(cp.len() - 1)) as u32 }; rd + f32::from_bits(b & cm_zero) } else { rd }
                } else { unsafe { adc_u(sq4, query_code, neighbor as usize) } };
                if cm_rec { unsafe { (*cmp).r_dist.push(d); } }
                let d = if orx { orx_dist(neighbor).unwrap_or(d) } else { d };
                if cfg!(feature = "diag") && mindist_lambda > 0.0 {
                    // H129 oracle：池键 = 到节点邻域球（半径 λ·r_w）的 MINDIST²（R-tree 最佳优先 kNN 的节点排序迁移到图）
                    let r = unsafe { *mindist_r.add(neighbor as usize) } * mindist_lambda;
                    let sd_ = d.sqrt() - r;
                    pool.insert(neighbor, if sd_ > 0.0 { sd_ * sd_ } else { 0.0 });
                } else if z_th {
                    z_sink ^= d.to_bits(); // R27 ZA 8：不逐码调 insert（保留核输出的数据依赖）
                } else {
                    let ins = if nv { pool.insert_dedup(neighbor, d) } else { pool.insert(neighbor, d) };
                    if cm_rec && ins { unsafe { (*cmp).acc.push(neighbor); (*cmp).r_acc2.push((neighbor, d)); } }
                }
                if a1 { A1_TRACE.with(|t| { let mut t = t.borrow_mut(); t.push(neighbor); t.push(d.to_bits()); }); }
            }
            if z_th && !z_pi { // R27 ZA 8：跳末只插入记录的本跳入池者（无门槛分支、无被拒者调用）
                let c = unsafe { &*cmp }; let (a, b) = (c.q_aoff[cm_hi - 1] as usize, c.q_aoff[cm_hi] as usize);
                for &(v, d) in &c.q_acc2[a..b] { pool.insert(v, d); }
            }
            // R25 CM：每跳额外 cm_extra 个独立随机取码 + 距离（地址只依赖本跳 node，不在池依赖链上；结果丢弃）
            if cm && cm_extra > 0 && !cm_rec {
                let n = sq4.n as u32; let base = cm_mix(node ^ cm_seed);
                for j in 0..(cm_extra & 0xFFFF) {
                    if j + po as u32 > j { let v2 = cm_mix(base.wrapping_add(j + po as u32)) % n; unsafe { crate::arch::prefetch_l1(sq4.code(v2 as usize).as_ptr() as *const i8); } }
                    let v = cm_mix(base.wrapping_add(j)) % n;
                    cm_sink += unsafe { adc_u(sq4, query_code, v as usize) };
                }
            }
        }
        if cm { unsafe { (*cmp).sink += cm_sink + f32::from_bits(z_sink & 0x3F00_0000); } }
        if z_pi { pool.load_sorted(unsafe { &(*cmp).q_fin }); }
        if cm_rec { let c = unsafe { &mut *cmp }; c.r_fin = pool.to_sorted_vec(); let a = c.r_acc2.len() as u32; c.r_aoff.push(a); }
        if a1 { A1_TRACE.with(|t| t.borrow_mut().push(A1_END)); }
        if N4_STATS.load(std::sync::atomic::Ordering::Relaxed) { N4_CNT.with(|c| { let mut c = c.borrow_mut(); c[0] += n4_hops as u64; c[3] += 1; }); }
    }
}

impl<'a> GraphSearcher<'a> {
    /// R68：ADC16 管线的统一入口（对应 `search_auto`）。只在 `Prepared::adc16` 为真时由调用方选用。
    #[inline(never)]
    pub fn search_auto_adc(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        match self.cc { Some(act) => self.search_sq4_hr_adc(query, k, act), None => self.search_sq4_adc(query, k) }
    }

    /// H09：带导航层选项的 SQ4 搜索（query_code_buf 已编码）
    fn search_sq4_nav_opts_adc(
        &mut self,
        nav: &LayeredNavigation,
        sq4: &SQ4Dataset,
        query: &[f32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        let dim = self.dim;
        let opts = self.nav_opts;
        let ef = self.ef_search;
        self.last_ef_used = ef;
        let want_trace = opts.carry > 1;
        self.nav_trace.clear();
        self.visited.reset();
        let vectors = self.vectors;
        let code = &self.query_code_buf;
        let use_vis = opts.dedup || opts.sq4;
        let _p0 = crate::counters::prof::tick();
        let navpf = NAV_PF.load(std::sync::atomic::Ordering::Relaxed);
        let cm_nrec = cfg!(feature = "diag_cm") && CM_ON.load(std::sync::atomic::Ordering::Relaxed) && CM.with(|c| c.borrow().rec);
        let (ep, ep_dist) = if opts.sq4 {
            nav.descend_with_pf(
                |v| { if cm_nrec { CM.with(|c| c.borrow_mut().r_nav.push(v)); } unsafe { adc_u(sq4, code, v as usize) } },
                |v| if navpf { unsafe { crate::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); } },
                Some(&mut self.visited),
                opts.stop_level,
                if want_trace { Some(&mut self.nav_trace) } else { None },
            )
        } else {
            nav.descend_with(
                |v| { if cm_nrec { CM.with(|c| c.borrow_mut().r_nav.push(v)); } l2_simd(query, &vectors[v as usize * dim..(v as usize + 1) * dim]) },
                if use_vis { Some(&mut self.visited) } else { None },
                opts.stop_level,
                if want_trace { Some(&mut self.nav_trace) } else { None },
            )
        };
        let _p1 = crate::counters::prof::tick();
        crate::counters::prof::add(0, _p1 - _p0);
        if G1_K.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            G1_Q.store(query.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            G1_V.store(vectors.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            G1_DIM.store(dim, std::sync::atomic::Ordering::Relaxed);
        }
        if cfg!(feature = "diag") && ORX_ON.with(|o| *o) {
            ORX_Q.store(query.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            ORX_V.store(vectors.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            ORX_DIM.store(dim, std::sync::atomic::Ordering::Relaxed);
        }
        let candidates = if opts.sq4 && want_trace {
            // 轨迹全部保持 visited；按 SQ4 距离取前 carry 个作种子
            let t = &mut self.nav_trace;
            // R27 S1：无 visited 的前提——导航层标记过的节点（= 全部轨迹）都作为种子提供（未截断、无 oracle 注入）
            let nv = novis_on_ef(ef) && t.len() <= opts.carry && ORACLE_SEEDS.with(|o| o.borrow().is_empty());
            t.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            t.truncate(opts.carry);
            // H39 oracle 上界实验：注入查询专属种子（raven_eval RAVEN_ORACLE），默认空
            ORACLE_SEEDS.with(|o| {
                for &v in o.borrow().iter() {
                    if self.visited.visit(v) { t.push((v, adc16_dist(code, sq4.code(v as usize)))); }
                }
            });
            debug_assert!(!self.l0_opts.is_active(), "ADC16 is never enabled with L0 early-abandon");
            {
                VamanaGraph::greedy_search_sq4_seeded_nv_adc(
                    sq4, self.graph.storage(), t, code, ef,
                    &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree, nv,
                )
            }
        } else if want_trace {
            // f32 轨迹：取前 carry 个，补算 SQ4，重置 visited 后只标记种子
            let t = &mut self.nav_trace;
            t.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            t.truncate(opts.carry);
            self.visited.reset();
            for s in t.iter_mut() {
                self.visited.visit(s.0);
                s.1 = adc16_dist(code, sq4.code(s.0 as usize));
            }
            VamanaGraph::greedy_search_sq4_seeded_adc(
                sq4, self.graph.storage(), t, code, ef,
                &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
            )
        } else if opts.sq4 {
            // 单入口，入口 SQ4 距离已知；导航轨迹节点保持 visited
            let seeds = [(ep, ep_dist)];
            VamanaGraph::greedy_search_sq4_seeded_adc(
                sq4, self.graph.storage(), &seeds, code, ef,
                &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
            )
        } else {
            VamanaGraph::greedy_search_sq4_adc(
                sq4, self.graph.storage(), ep, code, ef,
                &mut self.visited, &mut self.pool, self.prefetch_offset, self.search_degree,
            )
        };
        self.last_visited_count = self.visited.visited_count();
        let _p2 = crate::counters::prof::tick();
        crate::counters::prof::add(1, _p2 - _p1);
        if cm_nrec {
            let n = (k * self.rerank_factor).max(k).min(candidates.len());
            CM.with(|c| { let mut c = c.borrow_mut(); for x in &candidates[..n] { c.r_rr.push(x.0); } });
        }
        let rrlb = RRLB_E.load(std::sync::atomic::Ordering::Relaxed);
        let r = if rrlb != 0 {
            rerank_lb(self.vectors, dim, query, sq4, self.sq4_perm, unsafe { std::slice::from_raw_parts(rrlb as *const f32, sq4.n) }, candidates, k, self.rerank_factor)
        } else if let (Some(v8), true) = (self.vec_u8, crate::graph::refiner::U8_RR.load(std::sync::atomic::Ordering::Relaxed)) {
            crate::graph::refiner::rerank_u8(v8, dim, query, candidates, k, self.rerank_factor)
        } else if let Some(f16_ds) = self.f16 {
            rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor)
        } else {
            rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor)
        };
        crate::counters::prof::add(2, crate::counters::prof::tick() - _p2);
        r
    }

    #[inline(never)]
    fn search_sq4_hr_adc(&mut self, query: &[f32], k: usize, act: crate::graph::cellconfirm::CcActive<'a>) -> Vec<(u32, f32)> {
        let ext = self.cc_scratch.as_ref().expect("cc scratch").ext;
        let rh = ext.rk > 0 && k > ext.rk;
        let g = self.search_sq4_adc(query, if rh { ext.rk } else { k });
        let mut sc = self.cc_scratch.take().expect("cc scratch");
        let dim = self.dim; let v = self.vectors;
        sc.top.clear();
        sc.top.extend(g.iter().map(|x| (crate::distance::l2_simd(query, &v[x.0 as usize * dim..(x.0 as usize + 1) * dim]), x.0)));
        sc.top.sort_by(|a, b| a.0.total_cmp(&b.0)); sc.top.truncate(k);
        if rh {
            // R36 RH：图段 top-rk 精确距离 → LID 外推第 k 名半径 → 初始门槛（oracle 覆盖时用给定值）
            sc.thr0 = match ext.rh_oracle { Some(t) => t, None => {
                let d2: Vec<f32> = sc.top.iter().map(|x| x.0).collect();
                let r2 = crate::graph::cellconfirm::lid_radius2(&d2, k);
                if r2 == f32::MAX { f32::MAX } else { r2 * ext.rmg * ext.rmg } } };
        } else { sc.thr0 = f32::MAX; }
        if ext.sa > 0 { let sa = ext.sa.max((ext.saf * act.p as f32).round() as usize);
            act.cc.confirm_staged(query, v, k, act.p, act.margin, sa, ext.su, ext.ss, ext.sc, act.pat, act.pminr, &mut sc, None); }
        else { act.cc.confirm_pat_r68(query, v, k, act.p, act.margin, act.pat, act.pminr, &mut sc); }
        let r: Vec<(u32, f32)> = sc.top.iter().map(|x| (x.1, x.0)).collect();
        self.cc_scratch = Some(sc);
        r
    }

    pub fn search_sq4_adc(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let sq4 = self.sq4.expect("search_sq4 requires with_sq4() first");
        let dim = self.dim;

        // 1. 编码查询向量为 SQ4 packed 4-bit（零分配，复用预分配 buffer）
        //    SQ4 码大小 = ceil(dim/2)，可能与 SQ8 的 dim 不同
        let code_bytes = sq4.code_bytes;
        if self.query_code_buf.len() != code_bytes && self.query_code_buf.len() != 256 {
            self.query_code_buf = vec![0u8; code_bytes];
        }
        if !sq4.params.proj.is_empty() {
            // R42c LP：投影矩阵在 SQ4 参数里（m = 码空间维数）
            let (pm, m) = (&sq4.params.proj, sq4.dim);
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear();
            let d = query.len();
            buf.extend((0..m).map(|j| crate::pipeline::dot_f32(&pm[j * d..(j + 1) * d], query)));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf); sq4.params.encode_adc16_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else if let Some(perm) = self.sq4_perm {
            let mut buf = std::mem::take(&mut self.query_perm_buf);
            buf.clear();
            buf.extend(perm.iter().map(|&d| query[d]));
            sq4.params.encode_query_into(&buf, &mut self.query_code_buf); sq4.params.encode_adc16_into(&buf, &mut self.query_code_buf);
            self.query_perm_buf = buf;
        } else {
            sq4.params.encode_query_into(query, &mut self.query_code_buf); sq4.params.encode_adc16_into(query, &mut self.query_code_buf);
        }
        // R68：HSEL/HREL（R23 实验特性，缺省关）在 ADC16 链中不支持
        assert!(HSEL_K.load(std::sync::atomic::Ordering::Relaxed) == 0 && HREL_K.load(std::sync::atomic::Ordering::Relaxed) == 0, "ADC16 chain does not support HSEL/HREL; set RAVEN_SQ4_ADC=0");

        // H09：非默认导航选项走独立路径（默认行为完全不变）
        if !self.nav_opts.is_legacy() && self.upper_ef <= 1 && self.adaptive_ef.is_none() {
            if let Some(nav) = self.graph.layered_nav() {
                return self.search_sq4_nav_opts_adc(nav, sq4, query, k);
            }
        }

        // 2. 选择 entry_point + 捕获 nav.initialize 的 f32 距离
        let (entry_point, nav_entry_dist) = if let Some(nav) = self.graph.layered_nav() {
            if self.upper_ef > 1 {
                // 方案 B：Layer 1 用 ef>1 beam search
                let (ep, dist) = nav.initialize_with_upper_ef_r68(
                    self.vectors, dim, query, self.upper_ef,
                    &mut self.visited, &mut self.upper_pool,
                );
                (ep, Some(dist))
            } else {
                let (ep, dist) = nav.initialize(self.vectors, dim, query);
                (ep, Some(dist))
            }
        } else if let Some(nav) = self.navigation {
            (Self::nearest_centroid(nav.centroids(), self.vectors, dim, query), None)
        } else {
            (self.graph.entry_point(), None)
        };

        // 3. 自适应 ef
        let ef = if let Some(ref adaptive) = self.adaptive_ef {
            let entry_dist = nav_entry_dist.unwrap_or_else(|| {
                l2_simd(
                    query,
                    &self.vectors[entry_point as usize * dim
                        ..(entry_point as usize + 1) * dim],
                )
            });
            adaptive.estimate_ef(entry_dist).max(k)
        } else {
            self.ef_search
        };
        self.last_ef_used = ef;

        // 4. SQ4 图遍历
        let candidates = VamanaGraph::greedy_search_sq4_adc(
            sq4,
            self.graph.storage(),
            entry_point,
            &self.query_code_buf,
            ef,
            &mut self.visited,
            &mut self.pool,
            self.prefetch_offset,
            self.search_degree,
        );

        self.last_visited_count = self.visited.visited_count();

        // 5. f32 rerank（与 SQ8 路径完全相同）
        if let Some(f16_ds) = self.f16 {
            rerank_f16(f16_ds.codes(), dim, query, candidates, k, self.rerank_factor)
        } else {
            rerank_f32(self.vectors, dim, query, candidates, k, self.rerank_factor)
        }
    }
}
