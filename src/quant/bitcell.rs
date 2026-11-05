//! R32 SV-3b：单元主序 1-bit 残差码（随机正交旋转后取符号）+ fastscan 预筛核。
//!
//! 用途（诊断开关，默认搜索路径不使用）：图搜索先给出 top-k 与其精确第 k 名距离 T，
//! 再按查询到 k-means 中心的距离取前 P 个单元，用本模块的 1-bit 估计
//! `est_i = A_i + g_i·(K − K2·t_i)`（t_i = fastscan 整数累加，见 [`BitCells::prepare_query`]）
//! 与 **免排序门槛** `est_i < (T − ‖q−c‖²) + mc·r_i` 比较，只把幸存者交给精确距离。
//!
//! 估计式（RaBitQ 式无偏内积估计的单元残差版，自推导）：
//! x − c = r·o（o 单位向量），ō = sign(R·o)/√D，g = r / ⟨ō, R·o⟩，
//! ‖x−q‖² = r² + ‖q−c‖² − 2⟨x−c, q−c⟩ ≈ r² + ‖q−c‖² − 2g(⟨ō, Rq⟩ − ⟨ō, Rc⟩)
//!        = [r² + 2g·pc] + ‖q−c‖² − 2g⟨ō,Rq⟩，pc = ⟨ō,Rc⟩ 建库时预算 → A = r² + 2g·pc。
//! ⟨ō,Rq⟩ = (Σ sgn_j·q'_j)/√D；fastscan 以 4 维一组（nibble）查 16 项 u8 表：
//! LUT[v] = round((Σ_j (bit? q'_j : −q'_j) + Σ_j|q'_j|)·scale) ∈ [0, 127]，
//! 所以 Σ sgn·q' ≈ t/scale − Σ|q'| → −2g⟨ō,Rq⟩ = g·(K − K2·t)，K = 2Σ|q'|/√D，K2 = 2/(√D·scale)。
//!
//! 布局：每单元补齐到 32 的倍数（块），块内按字节位置转置（32 点的第 b 字节连续 32 B），
//! 副表（A、g，bf16）与 id 按 AVX2/NEON 累加器的自然输出序（偶数点 0,2..30 后奇数点 1,3..31）存放，免洗牌。
//! 补齐槽 id = u32::MAX、A = +∞（永不幸存）。

/// 每个块的点数（fastscan 宽度）。
pub const BLK: usize = 32;

/// 块内槽位 s（输出序）对应的原始块内序号。
#[inline]
pub fn slot_to_lane(s: usize) -> usize { if s < 16 { 2 * s } else { 2 * (s - 16) + 1 } }

pub struct BitCells {
    pub nl: usize,
    pub dim: usize,
    /// 旋转维数（dim 向上补齐到 32 的倍数）
    pub dpad: usize,
    /// 每点字节数 = dpad/8
    pub nb: usize,
    /// 单元 j 的槽范围 [offs[j], offs[j+1])，均为 BLK 的倍数
    pub offs: Vec<usize>,
    /// 槽 → 内部 id（补齐槽 = u32::MAX）
    pub ids: Vec<u32>,
    /// 块交错存储（单一顺序流）：块 b 占 `bs` 个 u16 字 = [码 nb*BLK 字节（第 k 字节位置的 32 B 连续）| A×32 (bf16) | g×32 (bf16)]
    /// R32 v4：去掉每点误差尺度 e，改用 ē·g（ē = 全库 √(1−⟨ō,o⟩²) 均值）→ 余量并入每单元常量；GloVe 128 位码 640 B/块 = 20 B/点
    pub blob: Vec<u16>,
    /// 块步长（u16 字）
    pub bs: usize,
    /// 全库平均误差因子 ē
    pub ebar: f32,
    /// 转置旋转矩阵：rot_t[k*dpad + i] = R[i][k]（k < dim）；rotate = Σ_k x[k]·列 k（可向量化）
    pub rot: Vec<f32>,
    pub cent: Vec<f32>,
    pub cn2: Vec<f32>,
}

/// 查询相关常量
pub struct QueryLut {
    /// nb*2 个 16 项表（字节 k 的低 nibble 表在 2k，高 nibble 表在 2k+1）
    pub lut: Vec<u8>,
    pub k: f32,
    pub k2: f32,
}

/// f32 → bf16（就近偶数舍入；±inf/NaN 保持）
#[inline]
pub fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() { return 0x7fc0; }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
#[inline]
pub fn from_bf16(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }

static PF_BLOCKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// 扫描时软件预取的超前块数（环境变量 RAVEN_SV3B_PFB，默认 2）
#[inline]
pub fn pf_blocks() -> usize {
    let v = PF_BLOCKS.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0 { return v; }
    let v = std::env::var("RAVEN_SV3B_PFB").ok().and_then(|x| x.parse().ok()).unwrap_or(2usize).max(1);
    PF_BLOCKS.store(v, std::sync::atomic::Ordering::Relaxed); v
}

fn xorshift(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
fn gauss(s: &mut u64) -> f32 {
    let u1 = ((xorshift(s) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((xorshift(s) >> 11) as f64) / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// 确定性随机正交矩阵（Gram-Schmidt，f64 累加）
pub fn random_rotation(d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut m: Vec<f64> = (0..d * d).map(|_| gauss(&mut s) as f64).collect();
    for i in 0..d {
        for j in 0..i {
            let dot: f64 = (0..d).map(|k| m[i * d + k] * m[j * d + k]).sum();
            for k in 0..d { m[i * d + k] -= dot * m[j * d + k]; }
        }
        let n: f64 = (0..d).map(|k| m[i * d + k] * m[i * d + k]).sum::<f64>().sqrt().max(1e-30);
        for k in 0..d { m[i * d + k] /= n; }
    }
    m.into_iter().map(|x| x as f32).collect()
}

/// out = R·[x; 0]，rot 为 [`BitCells::rot`] 的转置布局（dim×dpad）
#[inline]
pub fn rotate(rot: &[f32], dpad: usize, x: &[f32], out: &mut [f32]) {
    let out = &mut out[..dpad];
    out.fill(0.0);
    for (k, &xk) in x.iter().enumerate() {
        let col = &rot[k * dpad..(k + 1) * dpad];
        for i in 0..dpad { out[i] += xk * col[i]; }
    }
}

/// 每个单元按中心距离的 nc 个最近单元（含自身，升序）
pub fn cell_knn(cent: &[f32], dim: usize, nc: usize) -> Vec<u32> {
    let nl = cent.len() / dim; let nc = nc.min(nl);
    let mut out = vec![0u32; nl * nc];
    let mut tmp: Vec<(f32, u32)> = vec![(0.0, 0); nl];
    for j in 0..nl {
        let c = &cent[j * dim..(j + 1) * dim];
        for t in 0..nl { tmp[t] = (crate::distance::l2_simd(c, &cent[t * dim..(t + 1) * dim]), t as u32); }
        tmp.select_nth_unstable_by(nc - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        tmp[..nc].sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for t in 0..nc { out[j * nc + t] = tmp[t].1; }
    }
    out
}

impl BitCells {
    /// vectors: 内部序 n×dim；cent: nl×dim；asg[i] = 内部 id i 所属单元。
    pub fn build(vectors: &[f32], dim: usize, cent: &[f32], asg: &[u32], seed: u64) -> BitCells {
        let n = asg.len();
        let nl = cent.len() / dim;
        let dpad = dim.div_ceil(32) * 32;
        let nb = dpad / 8;
        let rfull = random_rotation(dpad, seed);
        let mut rot = vec![0f32; dim * dpad];
        for k in 0..dim { for i in 0..dpad { rot[k * dpad + i] = rfull[i * dpad + k]; } }
        let mut cnt = vec![0usize; nl];
        for &a in asg { cnt[a as usize] += 1; }
        let mut offs = vec![0usize; nl + 1];
        for j in 0..nl { offs[j + 1] = offs[j] + cnt[j].div_ceil(BLK) * BLK; }
        let ns = offs[nl];
        let mut ids = vec![u32::MAX; ns];
        let mut fillp = offs.clone();
        // 先按单元收集（块内原始序），再写成输出序
        let mut lane_ids = vec![u32::MAX; ns];
        for i in 0..n { let j = asg[i] as usize; lane_ids[fillp[j]] = i as u32; fillp[j] += 1; }
        // 中心旋转（用于 pc）
        let mut rc = vec![0f32; nl * dpad];
        for j in 0..nl { rotate(&rot, dpad, &cent[j * dim..(j + 1) * dim], &mut rc[j * dpad..(j + 1) * dpad]); }
        let cn2: Vec<f32> = (0..nl).map(|j| cent[j * dim..(j + 1) * dim].iter().map(|x| x * x).sum()).collect();
        let bs = nb * BLK / 2 + 2 * BLK;
        let nblk = ns / BLK;
        let mut codes = vec![0u8; nblk * nb * BLK];
        let mut a = vec![f32::INFINITY; ns];
        let mut g = vec![0f32; ns];
        let (mut esum, mut ecnt) = (0f64, 0usize);
        let mut res = vec![0f32; dim];
        let mut ro = vec![0f32; dpad];
        let sq = (dpad as f32).sqrt();
        for j in 0..nl {
            let c = &cent[j * dim..(j + 1) * dim];
            for blk in (offs[j]..offs[j + 1]).step_by(BLK) {
                for lane in 0..BLK {
                    let id = lane_ids[blk + lane];
                    // 输出序槽位
                    let slot = if lane % 2 == 0 { lane / 2 } else { 16 + lane / 2 };
                    let so = blk + slot;
                    if id == u32::MAX { continue; }
                    ids[so] = id;
                    let x = &vectors[id as usize * dim..(id as usize + 1) * dim];
                    let mut rr = 0f32;
                    for d in 0..dim { res[d] = x[d] - c[d]; rr += res[d] * res[d]; }
                    let rn = rr.sqrt();
                    rotate(&rot, dpad, &res, &mut ro);
                    // ⟨ō, R·o⟩ = Σ|ro|/(√D·r)；pc = ⟨ō, Rc⟩
                    let mut sabs = 0f32; let mut pc = 0f32;
                    let cb = &mut codes[(blk / BLK) * nb * BLK..(blk / BLK + 1) * nb * BLK];
                    for d in 0..dpad {
                        let bit = ro[d] >= 0.0;
                        sabs += ro[d].abs();
                        let sgn = if bit { 1.0 } else { -1.0 };
                        pc += sgn * rc[j * dpad + d];
                        if bit { cb[(d / 8) * BLK + lane] |= 1 << (d % 8); }
                    }
                    pc /= sq;
                    let oo = if rn > 0.0 { (sabs / (sq * rn)).min(1.0) } else { 1.0 };
                    let gg = if rn > 0.0 { rn / oo.max(1e-6) } else { 0.0 };
                    a[so] = rr + 2.0 * gg * pc;
                    g[so] = gg;
                    // 误差尺度 r·√(1−⟨ō,o⟩²)/⟨ō,o⟩ = g·√(1−⟨ō,o⟩²)；取全库均值 ē
                    if rn > 0.0 { esum += (1.0 - oo * oo).max(0.0).sqrt() as f64; ecnt += 1; }
                }
            }
        }
        let cw = nb * BLK / 2;
        let mut blob = vec![0u16; nblk * bs];
        for b in 0..nblk {
            let w = &mut blob[b * bs..(b + 1) * bs];
            for t in 0..cw { w[t] = u16::from_le_bytes([codes[b * nb * BLK + 2 * t], codes[b * nb * BLK + 2 * t + 1]]); }
            for t in 0..BLK { w[cw + t] = to_bf16(a[b * BLK + t]); w[cw + BLK + t] = to_bf16(g[b * BLK + t]); }
        }
        let ebar = if ecnt > 0 { (esum / ecnt as f64) as f32 } else { 0.0 };
        BitCells { nl, dim, dpad, nb, offs, ids, blob, bs, ebar, rot, cent: cent.to_vec(), cn2 }
    }

    #[inline] fn blk_codes(&self, blk: usize) -> &[u8] {
        let w = &self.blob[(blk / BLK) * self.bs..(blk / BLK) * self.bs + self.nb * BLK / 2];
        unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, self.nb * BLK) } }
    #[inline] fn fld(&self, so: usize, f: usize) -> f32 { from_bf16(self.blob[(so / BLK) * self.bs + self.nb * BLK / 2 + f * BLK + so % BLK]) }
    /// 预取单元 j 的前 nblk 个块（扫描上一单元时调用，隐藏单元起点的 DRAM 延迟）
    #[inline]
    pub fn prefetch_cell(&self, j: usize, nblk: usize) {
        let (o0, o1) = (self.offs[j], self.offs[j + 1]);
        let nbk = ((o1 - o0) / BLK).min(nblk);
        let p = unsafe { self.blob.as_ptr().add((o0 / BLK) * self.bs) } as *const i8;
        let mut o = 0; let end = nbk * self.bs * 2;
        while o < end {
            #[cfg(target_arch = "x86_64")] unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.wrapping_add(o)); }
            #[cfg(target_arch = "aarch64")] unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.wrapping_add(o), options(nostack, readonly, preserves_flags)); }
            o += 64;
        }
    }
    /// 把单元 j 整体预取进 L2（提前一到两个单元发出，增加在途行数；L1 MSHR 不够覆盖 DRAM 延迟时用）
    #[inline]
    pub fn prefetch_cell_l2(&self, j: usize) {
        let (o0, o1) = (self.offs[j], self.offs[j + 1]);
        let p = unsafe { self.blob.as_ptr().add((o0 / BLK) * self.bs) } as *const i8;
        let end = ((o1 - o0) / BLK) * self.bs * 2;
        let mut o = 0;
        while o < end {
            #[cfg(target_arch = "x86_64")] unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(p.wrapping_add(o)); }
            #[cfg(target_arch = "aarch64")] unsafe { std::arch::asm!("prfm pldl2keep, [{0}]", in(reg) p.wrapping_add(o), options(nostack, readonly, preserves_flags)); }
            o += 64;
        }
    }
    /// 槽 so 的 A / g（bf16 还原）
    pub fn a(&self, so: usize) -> f32 { self.fld(so, 0) }
    pub fn g(&self, so: usize) -> f32 { self.fld(so, 1) }

    /// 由已旋转查询 qr（dpad）建 nibble 表；返回常量。
    pub fn prepare_query(&self, qr: &[f32]) -> QueryLut {
        let mut ql = QueryLut { lut: vec![0u8; self.nb * 2 * 16], k: 0.0, k2: 0.0 };
        self.prepare_query_into(qr, &mut ql);
        ql
    }

    /// 同 [`Self::prepare_query`]，复用 `ql` 的表缓冲（R33 库内 HR 模式，零分配）。
    pub fn prepare_query_into(&self, qr: &[f32], ql: &mut QueryLut) {
        let nb = self.nb;
        if ql.lut.len() != nb * 2 * 16 { ql.lut = vec![0u8; nb * 2 * 16]; }
        let lut = &mut ql.lut;
        let mut mx = 1e-30f32; let mut absum = 0f32;
        for k in 0..nb * 2 { let s: f32 = qr[k * 4..k * 4 + 4].iter().map(|x| x.abs()).sum(); mx = mx.max(s); absum += s; }
        // 每表最大值 = 2·Σ|q'| ≤ 127（低+高 ≤ 254，u8 不溢出）
        let scale = 127.0 / (2.0 * mx);
        for k in 0..nb * 2 {
            let q = &qr[k * 4..k * 4 + 4];
            let ab: f32 = q.iter().map(|x| x.abs()).sum();
            for v in 0..16usize {
                let mut s = ab;
                for t in 0..4 { s += if (v >> t) & 1 == 1 { q[t] } else { -q[t] }; }
                lut[k * 16 + v] = (s * scale).round().clamp(0.0, 127.0) as u8;
            }
        }
        let sq = (self.dpad as f32).sqrt();
        ql.k = 2.0 * absum / sq; ql.k2 = 2.0 / (sq * scale);
    }

    /// 标量参考：单元 j 的幸存者槽位（绝对槽号）写入 out；thr = T − ‖q−c‖²，mc = 余量系数（余量 = mc·ē·g_i，并入 u 的常数项）。
    pub fn scan_cell_scalar(&self, j: usize, ql: &QueryLut, thr: f32, mc: f32, out: &mut Vec<u32>) {
        let nb = self.nb;
        for blk in (self.offs[j]..self.offs[j + 1]).step_by(BLK) {
            let cb = self.blk_codes(blk);
            for s in 0..BLK {
                let lane = slot_to_lane(s);
                let mut t = 0u32;
                for k in 0..nb { let b = cb[k * BLK + lane];
                    t += ql.lut[(2 * k) * 16 + (b & 15) as usize] as u32 + ql.lut[(2 * k + 1) * 16 + (b >> 4) as usize] as u32; }
                let so = blk + s;
                let kc = ql.k - mc * self.ebar;
                let u = kc - ql.k2 * t as f32;
                let est = self.a(so) + self.g(so) * u;
                if est < thr { out.push(so as u32); }
            }
        }
    }

    /// 单元 j 扫描（自动选核）。
    #[inline]
    pub fn scan_cell(&self, j: usize, ql: &QueryLut, thr: f32, mc: f32, out: &mut Vec<u32>) {
        #[cfg(target_arch = "x86_64")]
        { if std::is_x86_feature_detected!("avx2") { unsafe { self.scan_cell_avx2(j, ql, thr, mc, out); } return; } }
        #[cfg(target_arch = "aarch64")]
        { unsafe { self.scan_cell_neon(j, ql, thr, mc, out); } return; }
        #[allow(unreachable_code)]
        self.scan_cell_scalar(j, ql, thr, mc, out)
    }

    /// 1-bit 估计值（测试/诊断用，标量）
    pub fn est_slot(&self, so: usize, ql: &QueryLut) -> f32 {
        let nb = self.nb; let blk = so / BLK * BLK; let lane = slot_to_lane(so - blk);
        let cb = self.blk_codes(blk);
        let mut t = 0u32;
        for k in 0..nb { let b = cb[k * BLK + lane];
            t += ql.lut[(2 * k) * 16 + (b & 15) as usize] as u32 + ql.lut[(2 * k + 1) * 16 + (b >> 4) as usize] as u32; }
        self.a(so) + self.g(so) * (ql.k - ql.k2 * t as f32)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_cell_avx2(&self, j: usize, ql: &QueryLut, thr: f32, mc: f32, out: &mut Vec<u32>) {
        use std::arch::x86_64::*;
        let nb = self.nb;
        let m4 = _mm256_set1_epi8(15);
        let m8 = _mm256_set1_epi16(0xff);
        let k2v = _mm256_set1_ps(ql.k2);
        let tv = _mm256_set1_ps(thr); let kv = _mm256_set1_ps(ql.k - mc * self.ebar);
        // nb ≤ 16（dpad ≤ 128）时表常驻寄存器之外也足够快：每次从 L1 加载并广播到两个 128 位 lane
        let lp = ql.lut.as_ptr();
        let (o0, o1) = (self.offs[j], self.offs[j + 1]);
        let mut blk = o0;
        while blk < o1 {
            let bp = self.blob.as_ptr().add((blk / BLK) * self.bs);
            let cb = bp as *const u8;
            let fp = bp.add(nb * BLK / 2);
            { let pfb = pf_blocks(); let nx = (bp as *const i8).wrapping_add(self.bs * 2 * pfb); let mut o = 0; while o < self.bs * 2 { _mm_prefetch::<_MM_HINT_T0>(nx.wrapping_add(o)); o += 64; } }
            let mut acc_e = _mm256_setzero_si256(); let mut acc_o = _mm256_setzero_si256();
            for k in 0..nb {
                let x = _mm256_loadu_si256(cb.add(k * BLK) as *const __m256i);
                let tl = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp.add(2 * k * 16) as *const __m128i));
                let th = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp.add((2 * k + 1) * 16) as *const __m128i));
                let lo = _mm256_and_si256(x, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m4);
                let v = _mm256_add_epi8(_mm256_shuffle_epi8(tl, lo), _mm256_shuffle_epi8(th, hi));
                acc_e = _mm256_add_epi16(acc_e, _mm256_and_si256(v, m8));
                acc_o = _mm256_add_epi16(acc_o, _mm256_srli_epi16(v, 8));
            }
            // 输出序：acc_e 低 128 = 点 0,2..14（槽 0..7），高 128 = 16..30（槽 8..15）；acc_o 同理（槽 16..31）
            let mut m: u32 = 0;
            for (h, acc) in [acc_e, acc_o].into_iter().enumerate() {
                for half in 0..2 {
                    let a16 = if half == 0 { _mm256_castsi256_si128(acc) } else { _mm256_extracti128_si256(acc, 1) };
                    let t = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(a16));
                    let so = h * 16 + half * 8;
                    let av = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(fp.add(so) as *const __m128i)), 16));
                    let gv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(fp.add(BLK + so) as *const __m128i)), 16));
                    let u = _mm256_sub_ps(kv, _mm256_mul_ps(k2v, t));
                    let est = _mm256_add_ps(av, _mm256_mul_ps(gv, u));
                    let c = _mm256_movemask_ps(_mm256_cmp_ps(est, tv, _CMP_LT_OQ)) as u32;
                    m |= c << (h * 16 + half * 8);
                }
            }
            while m != 0 { let b = m.trailing_zeros(); out.push((blk as u32) + b); m &= m - 1; }
            blk += BLK;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn scan_cell_neon(&self, j: usize, ql: &QueryLut, thr: f32, mc: f32, out: &mut Vec<u32>) {
        use std::arch::aarch64::*;
        let nb = self.nb;
        let m4 = vdupq_n_u8(15);
        let lp = ql.lut.as_ptr();
        let kv = vdupq_n_f32(ql.k - mc * self.ebar); let k2v = vdupq_n_f32(ql.k2);
        let tv = vdupq_n_f32(thr);
        let (o0, o1) = (self.offs[j], self.offs[j + 1]);
        let mut blk = o0;
        while blk < o1 {
            let bp = self.blob.as_ptr().add((blk / BLK) * self.bs);
            let cb = bp as *const u8;
            let fp = bp.add(nb * BLK / 2);
            // 两个 16 B 半块：x0 = 点 0..15，x1 = 点 16..31（块内原始序）
            let mut ae0 = vdupq_n_u16(0); let mut ao0 = vdupq_n_u16(0);
            let mut ae1 = vdupq_n_u16(0); let mut ao1 = vdupq_n_u16(0);
            for k in 0..nb {
                let tl = vld1q_u8(lp.add(2 * k * 16)); let th = vld1q_u8(lp.add((2 * k + 1) * 16));
                let x0 = vld1q_u8(cb.add(k * BLK)); let x1 = vld1q_u8(cb.add(k * BLK + 16));
                let v0 = vaddq_u8(vqtbl1q_u8(tl, vandq_u8(x0, m4)), vqtbl1q_u8(th, vshrq_n_u8(x0, 4)));
                let v1 = vaddq_u8(vqtbl1q_u8(tl, vandq_u8(x1, m4)), vqtbl1q_u8(th, vshrq_n_u8(x1, 4)));
                let w0 = vreinterpretq_u16_u8(v0); let w1 = vreinterpretq_u16_u8(v1);
                ae0 = vaddq_u16(ae0, vandq_u16(w0, vdupq_n_u16(0xff))); ao0 = vaddq_u16(ao0, vshrq_n_u16(w0, 8));
                ae1 = vaddq_u16(ae1, vandq_u16(w1, vdupq_n_u16(0xff))); ao1 = vaddq_u16(ao1, vshrq_n_u16(w1, 8));
            }
            // ae0 = 点 0,2..14（槽 0..7）；ae1 = 点 16..30（槽 8..15）；ao0 = 1..15（槽 16..23）；ao1 = 17..31（槽 24..31）
            let mut m: u32 = 0;
            for (gi, acc) in [ae0, ae1, ao0, ao1].into_iter().enumerate() {
                for half in 0..2 {
                    let t32 = if half == 0 { vmovl_u16(vget_low_u16(acc)) } else { vmovl_high_u16(acc) };
                    let t = vcvtq_f32_u32(t32);
                    let so = gi * 8 + half * 4;
                    let av = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(fp.add(so)), 16));
                    let gv = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(fp.add(BLK + so)), 16));
                    let u = vsubq_f32(kv, vmulq_f32(k2v, t));
                    let est = vaddq_f32(av, vmulq_f32(gv, u));
                    let c = vcltq_f32(est, tv);
                    // 4 位掩码
                    let bits = vandq_u32(c, vld1q_u32([1u32, 2, 4, 8].as_ptr()));
                    m |= vaddvq_u32(bits) << (gi * 8 + half * 4);
                }
            }
            while m != 0 { let b = m.trailing_zeros(); out.push((blk as u32) + b); m &= m - 1; }
            blk += BLK;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, dim: usize, nl: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let mut s = seed;
        let x: Vec<f32> = (0..n * dim).map(|_| gauss(&mut s)).collect();
        let cent: Vec<f32> = x[..nl * dim].to_vec();
        let asg: Vec<u32> = (0..n).map(|i| {
            let v = &x[i * dim..(i + 1) * dim];
            (0..nl).min_by(|&a, &b| {
                let da: f32 = v.iter().zip(&cent[a * dim..(a + 1) * dim]).map(|(p, q)| (p - q) * (p - q)).sum();
                let db: f32 = v.iter().zip(&cent[b * dim..(b + 1) * dim]).map(|(p, q)| (p - q) * (p - q)).sum();
                da.total_cmp(&db) }).unwrap() as u32 }).collect();
        (x, cent, asg)
    }

    #[test]
    fn rotation_is_orthogonal() {
        let d = 64; let r = random_rotation(d, 7);
        for i in 0..d { for j in 0..d {
            let dot: f32 = (0..d).map(|k| r[i * d + k] * r[j * d + k]).sum();
            assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-4, "{i} {j} {dot}");
        } }
    }

    #[test]
    fn layout_covers_all_ids_once() {
        let (x, c, a) = data(1000, 37, 7, 3);
        let bc = BitCells::build(&x, 37, &c, &a, 11);
        assert_eq!(bc.dpad, 64);
        let mut seen = vec![0u8; 1000];
        for j in 0..bc.nl { for so in bc.offs[j]..bc.offs[j + 1] { let id = bc.ids[so];
            if id != u32::MAX { assert_eq!(a[id as usize] as usize, j); seen[id as usize] += 1; } else { assert!(bc.a(so).is_infinite()); } } }
        assert!(seen.iter().all(|&s| s == 1));
    }

    #[test]
    fn estimator_tracks_exact_distance() {
        let dim = 100; let (x, c, a) = data(4000, dim, 8, 5);
        let bc = BitCells::build(&x, dim, &c, &a, 13);
        let mut s = 99u64;
        let mut errs = Vec::new(); let mut scale = Vec::new();
        for _ in 0..5 {
            let q: Vec<f32> = (0..dim).map(|_| gauss(&mut s)).collect();
                        let mut qr = vec![0f32; bc.dpad]; rotate(&bc.rot, bc.dpad, &q, &mut qr);
            let ql = bc.prepare_query(&qr);
            for j in 0..bc.nl { let qc2: f32 = q.iter().zip(&c[j * dim..(j + 1) * dim]).map(|(p, r)| (p - r) * (p - r)).sum();
                for so in bc.offs[j]..bc.offs[j + 1] { let id = bc.ids[so]; if id == u32::MAX { continue; }
                    let ex: f32 = q.iter().zip(&x[id as usize * dim..(id as usize + 1) * dim]).map(|(p, r)| (p - r) * (p - r)).sum();
                    let est = bc.est_slot(so, &ql) + qc2;
                    errs.push((est - ex) / ex); scale.push(ex); } }
        }
        let n = errs.len() as f32;
        let mean = errs.iter().sum::<f32>() / n;
        let rms = (errs.iter().map(|e| e * e).sum::<f32>() / n).sqrt();
        // 1-bit 估计：近似无偏（|均值| 小）、相对误差 RMS 有界（随机高斯数据 d=100）
        assert!(mean.abs() < 0.03, "bias {mean}");
        assert!(rms < 0.25, "rms {rms}");
    }

    #[test]
    fn simd_kernel_matches_scalar() {
        for &(dim, nl) in &[(100usize, 5usize), (128, 9), (37, 3)] {
            let (x, c, a) = data(3000, dim, nl, 21 + dim as u64);
            let bc = BitCells::build(&x, dim, &c, &a, 17);
            let mut s = 5u64;
            for _ in 0..4 {
                let q: Vec<f32> = (0..dim).map(|_| gauss(&mut s)).collect();
                let mut qr = vec![0f32; bc.dpad]; rotate(&bc.rot, bc.dpad, &q, &mut qr);
                let ql = bc.prepare_query(&qr);
                for j in 0..bc.nl { for &(thr, mc) in &[(0.0f32, 0.0f32), (-60.0, 1.5), (20.0, 0.3), (f32::MAX / 4.0, 0.0)] {
                    let (mut o1, mut o2) = (Vec::new(), Vec::new());
                    bc.scan_cell_scalar(j, &ql, thr, mc, &mut o1);
                    bc.scan_cell(j, &ql, thr, mc, &mut o2);
                    o1.sort(); o2.sort();
                    assert_eq!(o1, o2, "dim {dim} cell {j} thr {thr} mc {mc}");
                } }
            }
        }
    }
}
