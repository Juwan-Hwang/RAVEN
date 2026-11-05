//! H38：量化图（QG）L0 路径——每节点一个连续块 = [旋转后向量 | 邻居 id | 每边因子 | 邻居 1-bit 码（fast-scan 转置）]。
//!
//! 一次 hop 只取当前节点的一个连续块（硬件/软件流式预取友好），用 NEON `vqtbl1q_u8` 查表一次算 32 个邻居的
//! RaBitQ 式 1-bit 估计距离，弹出节点时用块内向量算精确距离（无需末尾 rerank）。
//!
//! 与 SymphonyQG 的区别（原创部分）：
//! - **查询全局 LUT + 每边中心校正**：邻居码是相对所在节点 u 的残差符号码，但估计式
//!   `‖o−q‖² ≈ ‖q−u‖² + k1 − k2·⟨x̄, Pq⟩` 中 ‖q−u‖ 被消去，`⟨x̄,Pu⟩` 在构建期折进 k1，
//!   因此 LUT 只依赖 Pq：**每个查询建一次 LUT，hop 时零 LUT 构建**（SymphonyQG 每 hop 重建残差 LUT）。
//! - 误差界（RaBitQ）用作可调乐观键：key = est − eps·‖q−u‖·e（eps=0 即纯估计）。
//!
//! 开关：仅 `eval harness --qg 1` 使用；默认路径不变。

use crate::build::ChaCha8Rng;
use crate::graph::linear_pool::LinearPool;
use crate::memory::aligned::AlignedBuf;
use crate::memory::VisitedTracker;
use crate::pipeline::Prepared;
use rand::RngCore;

const SENT: u32 = u32::MAX;

pub struct QgIndex {
    pub n: usize,
    pub dim: usize,
    /// 填充到 4 的倍数的维度
    pub dp: usize,
    /// 4 维一组的组数 = dp/4
    pub g: usize,
    /// 每节点邻居槽数（32 的倍数）
    pub bmax: usize,
    pub stride: usize,
    off_ids: usize,
    off_k1: usize,
    off_k2: usize,
    off_e: usize,
    off_codes: usize,
    pub data: AlignedBuf<u8>,
    /// 随机正交旋转 P（dp×dp，行主序）
    pub rot: Vec<f32>,
}

fn random_orthogonal(dp: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from(seed);
    let gauss = |rng: &mut ChaCha8Rng| -> f64 {
        let u1 = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut m: Vec<f64> = (0..dp * dp).map(|_| gauss(&mut rng)).collect();
    // 修正 Gram-Schmidt（两遍，数值稳）
    for i in 0..dp {
        for _ in 0..2 {
            for j in 0..i {
                let dot: f64 = (0..dp).map(|t| m[i * dp + t] * m[j * dp + t]).sum();
                for t in 0..dp { m[i * dp + t] -= dot * m[j * dp + t]; }
            }
        }
        let nr: f64 = (0..dp).map(|t| m[i * dp + t] * m[i * dp + t]).sum::<f64>().sqrt();
        for t in 0..dp { m[i * dp + t] /= nr; }
    }
    m.into_iter().map(|x| x as f32).collect()
}

#[inline]
fn rotate(rot: &[f32], dp: usize, x: &[f32], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate().take(dp) {
        let row = &rot[i * dp..i * dp + x.len()];
        let mut s = 0f32;
        for (a, b) in row.iter().zip(x) { s += a * b; }
        *o = s;
    }
}

impl QgIndex {
    /// 从生产管线（已重排）构建。`sd` = 每节点使用的邻居数上限（0 = 全部）。
    pub fn build(p: &Prepared, sd: usize) -> Self {
        let n = p.graph.len();
        let dim = p.graph.dim();
        let dp = dim.div_ceil(4) * 4;
        let g = dp / 4;
        let st = p.graph.storage();
        let lim = if sd > 0 { sd } else { usize::MAX };
        let maxdeg = (0..n).map(|i| st.neighbors(i as u32).len().min(lim)).max().unwrap_or(0);
        let bmax = maxdeg.div_ceil(32).max(1) * 32;
        let nb = bmax / 32;
        let off_ids = dp * 4;
        let off_k1 = off_ids + bmax * 4;
        let off_k2 = off_k1 + bmax * 4;
        let off_e = off_k2 + bmax * 4;
        let off_codes = off_e + bmax * 4;
        let stride = (off_codes + nb * g * 16).div_ceil(64) * 64;
        let rot = random_orthogonal(dp, 0x5157_4752);
        // 旋转全部向量
        let mut rv = vec![0f32; n * dp];
        for i in 0..n {
            let (src, dst) = (&p.vectors[i * dim..(i + 1) * dim], &mut rv[i * dp..(i + 1) * dp]);
            rotate(&rot, dp, src, dst);
        }
        let mut data = vec![0u8; n * stride];
        let inv_sqrt_d = 1.0 / (dp as f64).sqrt();
        let sqrt_dm1 = ((dp - 1) as f64).sqrt();
        let mut r = vec![0f64; dp];
        for u in 0..n {
            let blk = &mut data[u * stride..(u + 1) * stride];
            let ru = &rv[u * dp..(u + 1) * dp];
            for t in 0..dp { blk[t * 4..t * 4 + 4].copy_from_slice(&ru[t].to_le_bytes()); }
            let nbrs = st.neighbors(u as u32);
            let deg = nbrs.len().min(lim);
            for j in 0..bmax {
                let id = if j < deg { nbrs[j] } else { SENT };
                blk[off_ids + j * 4..off_ids + j * 4 + 4].copy_from_slice(&id.to_le_bytes());
                if j >= deg {
                    // 填充槽：k1 = +inf 使估计恒为 +inf
                    blk[off_k1 + j * 4..off_k1 + j * 4 + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
                    continue;
                }
                let ro = &rv[id as usize * dp..(id as usize + 1) * dp];
                let mut a2 = 0f64;
                let mut sabs = 0f64;
                let mut cu = 0f64;
                for t in 0..dp {
                    r[t] = ro[t] as f64 - ru[t] as f64;
                    a2 += r[t] * r[t];
                    sabs += r[t].abs();
                    cu += if r[t] > 0.0 { ru[t] as f64 } else { -(ru[t] as f64) };
                }
                cu *= inv_sqrt_d;
                let a = a2.sqrt().max(1e-20);
                let ipx = (sabs * inv_sqrt_d / a).clamp(1e-6, 1.0);
                let k2 = 2.0 * a / ipx;
                let k1 = a2 + k2 * cu;
                let e = 2.0 * a * ((1.0 - ipx * ipx).max(0.0)).sqrt() / ipx / sqrt_dm1;
                blk[off_k1 + j * 4..off_k1 + j * 4 + 4].copy_from_slice(&(k1 as f32).to_le_bytes());
                blk[off_k2 + j * 4..off_k2 + j * 4 + 4].copy_from_slice(&(k2 as f32).to_le_bytes());
                blk[off_e + j * 4..off_e + j * 4 + 4].copy_from_slice(&(e as f32).to_le_bytes());
                // 码：batch b = j/32，lane = j%32；组 gi 的 nibble bit k = r[4gi+k] > 0
                let (b, lane) = (j / 32, j % 32);
                for gi in 0..g {
                    let mut nib = 0u8;
                    for k in 0..4 { if r[4 * gi + k] > 0.0 { nib |= 1 << k; } }
                    let pos = off_codes + (b * g + gi) * 16 + (lane % 16);
                    if lane < 16 { blk[pos] |= nib; } else { blk[pos] |= nib << 4; }
                }
            }
        }
        QgIndex { n, dim, dp, g, bmax, stride, off_ids, off_k1, off_k2, off_e, off_codes, data: data.into(), rot }
    }

    pub fn bytes(&self) -> usize { self.data.len() }

    #[inline(always)]
    fn block(&self, u: u32) -> *const u8 {
        unsafe { self.data.as_ptr().add(u as usize * self.stride) }
    }
}

/// 32 个邻居的 u16 累加（fast-scan）。
#[inline(always)]
unsafe fn fastscan32(codes: *const u8, lut: *const u8, g: usize, out: &mut [u16; 32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let mask = vdupq_n_u8(0x0F);
        let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
        for gi in 0..g {
            let c = vld1q_u8(codes.add(gi * 16));
            let l = vld1q_u8(lut.add(gi * 16));
            let vl = vqtbl1q_u8(l, vandq_u8(c, mask));
            let vh = vqtbl1q_u8(l, vshrq_n_u8::<4>(c));
            a0 = vaddw_u8(a0, vget_low_u8(vl));
            a1 = vaddw_high_u8(a1, vl);
            a2 = vaddw_u8(a2, vget_low_u8(vh));
            a3 = vaddw_high_u8(a3, vh);
        }
        let o = out.as_mut_ptr();
        vst1q_u16(o, a0);
        vst1q_u16(o.add(8), a1);
        vst1q_u16(o.add(16), a2);
        vst1q_u16(o.add(24), a3);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        *out = [0u16; 32];
        for gi in 0..g {
            for i in 0..16 {
                let c = *codes.add(gi * 16 + i);
                out[i] += *lut.add(gi * 16 + (c & 15) as usize) as u16;
                out[i + 16] += *lut.add(gi * 16 + (c >> 4) as usize) as u16;
            }
        }
    }
}

#[inline(always)]
fn l2_f32(a: &[f32], b: *const f32, d: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut t = 0;
        while t + 8 <= d {
            let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(t)), vld1q_f32(b.add(t)));
            let d1 = vsubq_f32(vld1q_f32(a.as_ptr().add(t + 4)), vld1q_f32(b.add(t + 4)));
            s0 = vfmaq_f32(s0, d0, d0);
            s1 = vfmaq_f32(s1, d1, d1);
            t += 8;
        }
        let mut s = vaddvq_f32(vaddq_f32(s0, s1));
        while t < d { let x = a[t] - *b.add(t); s += x * x; t += 1; }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        let mut s = 0f32;
        for t in 0..d { let x = a[t] - *b.add(t); s += x * x; }
        s
    }
}

pub struct QgSearcher<'a> {
    pub p: &'a Prepared,
    pub qg: &'a QgIndex,
    pub ef: usize,
    /// 乐观键系数（RaBitQ 误差界倍数）；0 = 纯估计
    pub eps: f32,
    /// 预取下一候选块的 cache line 数（0 = 整块）
    pub pf_lines: usize,
    /// v2：弹出后用精确距离重设键（RAVEN_QG_REKEY，默认 0：sift100k 上 hops@0.95 62→73.5，拒绝）
    pub rekey: bool,
    /// 诊断：精确距离作池键（RAVEN_QG_EXACT=1）
    pub exact_keys: bool,
    inslog: Vec<(u32, u32)>,
    /// H44 诊断：每查询 (最终 top-k 最后发现 hop, 总 hops)
    pub diag: Vec<(u32, u32)>,
    visited: VisitedTracker,
    pool: LinearPool,
    qrot: Vec<f32>,
    qpad: Vec<f32>,
    lut: AlignedBuf<u8>,
    qcode: Vec<u8>,
    qperm: Vec<f32>,
    trace: Vec<(u32, f32)>,
    res: Vec<(u32, f32)>,
}

impl<'a> QgSearcher<'a> {
    pub fn new(p: &'a Prepared, qg: &'a QgIndex, ef: usize) -> Self {
        let pf = std::env::var("RAVEN_QG_PF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let eps = std::env::var("RAVEN_QG_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        QgSearcher {
            p, qg, ef, eps, pf_lines: pf,
            inslog: Vec::new(), diag: Vec::new(),
            exact_keys: std::env::var("RAVEN_QG_EXACT").map(|v| v == "1").unwrap_or(false),
            rekey: std::env::var("RAVEN_QG_REKEY").map(|v| v == "1").unwrap_or(false),
            visited: VisitedTracker::new(qg.n, ef),
            pool: LinearPool::new(ef),
            qrot: vec![0f32; qg.dp],
            qpad: vec![0f32; qg.dp],
            lut: AlignedBuf::from_slice(&vec![0u8; qg.g * 16]),
            qcode: vec![0u8; p.sq4.code_bytes],
            qperm: Vec::with_capacity(qg.dim),
            trace: Vec::new(),
            res: Vec::new(),
        }
    }

    #[inline(always)]
    fn prefetch_block(&self, u: u32) {
        let b = self.qg.block(u);
        let lines = if self.pf_lines > 0 { self.pf_lines.min(self.qg.stride / 64) } else { self.qg.stride / 64 };
        for l in 0..lines { unsafe { crate::arch::prefetch_l1(b.add(l * 64) as *const i8); } }
    }

    /// 返回 (id（管线重排后空间）, 精确 L2²) 升序前 k 个
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let qg = self.qg;
        let p = self.p;
        let (dp, g, dim) = (qg.dp, qg.g, qg.dim);
        // 1. 旋转查询 + 全局 LUT（每查询一次）
        self.qpad[..dim].copy_from_slice(query);
        rotate(&qg.rot, dp, &self.qpad[..dim], &mut self.qrot);
        let inv = 1.0 / (dp as f32).sqrt();
        let mut lutf = [0f32; 16];
        let mut mins = vec![0f32; g];
        let mut span = 0f32;
        let mut lut_all = vec![0f32; g * 16];
        for gi in 0..g {
            let q4 = &self.qrot[4 * gi..4 * gi + 4];
            for (e, lv) in lutf.iter_mut().enumerate() {
                let mut s = 0f32;
                for (kk, &qv) in q4.iter().enumerate() { s += if e >> kk & 1 == 1 { qv } else { -qv }; }
                *lv = s * inv;
            }
            let mn = lutf.iter().cloned().fold(f32::MAX, f32::min);
            let mx = lutf.iter().cloned().fold(f32::MIN, f32::max);
            mins[gi] = mn;
            span = span.max(mx - mn);
            lut_all[gi * 16..gi * 16 + 16].copy_from_slice(&lutf);
        }
        let delta = (span / 255.0).max(1e-30);
        let mut b0 = 0f32;
        for gi in 0..g {
            b0 += mins[gi];
            for e in 0..16 {
                self.lut[gi * 16 + e] = ((lut_all[gi * 16 + e] - mins[gi]) / delta).round().clamp(0.0, 255.0) as u8;
            }
        }
        // 2. 上层导航（与生产管线相同：SQ4 + 轨迹携带）
        self.visited.reset();
        self.pool.reset_for(self.ef);
        self.res.clear();
        if !p.perm.is_empty() {
            self.qperm.clear();
            self.qperm.extend(p.perm.iter().map(|&d| query[d]));
            p.sq4.params.encode_query_into(&self.qperm, &mut self.qcode);
        } else {
            p.sq4.params.encode_query_into(query, &mut self.qcode);
        }
        let nav = p.graph.layered_nav().expect("QG requires layered nav");
        self.trace.clear();
        {
            let code = &self.qcode;
            let sq4 = &p.sq4;
            nav.descend_with(|v| unsafe { sq4.distance_raw_unchecked(code, v as usize) },
                Some(&mut self.visited), p.nav_opts.stop_level, Some(&mut self.trace));
        }
        self.trace.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        self.trace.truncate(p.nav_opts.carry.max(1));
        for i in 0..self.trace.len() {
            let s = self.trace[i].0;
            crate::diag_count!(FULL_DIST);
            let d = l2_f32(&self.qrot, qg.block(s) as *const f32, dp);
            self.pool.insert(s, d);
        }
        // 3. QG 主循环
        let nb = qg.bmax / 32;
        let mut acc = [0u16; 32];
        let eps = self.eps;
        let rekey = self.rekey;
        let mut hop = 0u32;
        self.inslog.clear();
        for t in self.trace.iter() { self.inslog.push((t.0, 0)); }
        while let Some((u, _, pidx)) = self.pool.pop_idx() {
            hop += 1;
            crate::diag_count!(HOPS);
            if let Some((nx, _)) = self.pool.peek_unchecked() { self.prefetch_block(nx); }
            let blk = qg.block(u);
            crate::diag_count!(FULL_DIST);
            let dq2 = l2_f32(&self.qrot, blk as *const f32, dp);
            self.res.push((u, dq2));
            // v2：已扩展节点改用精确距离作键（池边界 = 精确距离，减少估计噪声对 ef 的放大）
            if rekey { self.pool.rekey_checked(pidx, dq2); }
            let dq = dq2.sqrt();
            let base = dq2 - 0.0;
            let c0 = b0;
            if self.exact_keys {
                // 诊断：池键 = 精确距离（量化噪声为零的 ef/hops 下限）
                let ids = unsafe { blk.add(qg.off_ids) as *const u32 };
                for i in 0..qg.bmax {
                    let id = unsafe { *ids.add(i) };
                    if id == SENT { break; }
                    if self.visited.is_visited(id) { continue; }
                    crate::diag_count!(QUANT_DIST);
                    let d = l2_f32(&self.qrot, qg.block(id) as *const f32, dp);
                    if d < self.pool.worst_distance() || !self.pool.is_full() { self.visited.visit(id); self.pool.insert(id, d); self.inslog.push((id, hop)); }
                }
                continue;
            }
            for bi in 0..nb {
                unsafe { fastscan32(blk.add(qg.off_codes + bi * g * 16), self.lut.as_ptr(), g, &mut acc); }
                crate::diag_count_add!(QUANT_DIST, 32);
                let ids = unsafe { blk.add(qg.off_ids + bi * 128) as *const u32 };
                let k1 = unsafe { blk.add(qg.off_k1 + bi * 128) as *const f32 };
                let k2 = unsafe { blk.add(qg.off_k2 + bi * 128) as *const f32 };
                let ee = unsafe { blk.add(qg.off_e + bi * 128) as *const f32 };
                for i in 0..32 {
                    let id = unsafe { *ids.add(i) };
                    if id == SENT { break; }
                    let x = c0 + delta * acc[i] as f32;
                    let mut est = base + unsafe { *k1.add(i) - *k2.add(i) * x };
                    if eps != 0.0 { est -= eps * dq * unsafe { *ee.add(i) }; }
                    if est < self.pool.worst_distance() || !self.pool.is_full() {
                        if self.visited.visit(id) { self.pool.insert(id, est); }
                    }
                }
            }
        }
        // 4. 结果：已扩展节点的精确距离取前 k
        let r = &mut self.res;
        let kk = k.min(r.len());
        if kk > 0 && kk < r.len() { r.select_nth_unstable_by(kk - 1, |a, b| a.1.total_cmp(&b.1)); r.truncate(kk); }
        r.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        if self.exact_keys {
            // H44 诊断：最终 top-k 中最后一个被发现（入池）的 hop 序号 vs 总 hops
            let mut last = 0u32;
            for x in r.iter() { if let Some(&(_, h)) = self.inslog.iter().find(|e| e.0 == x.0) { last = last.max(h); } }
            self.diag.push((last, hop));
        }
        r.clone()
    }
}

impl Drop for QgSearcher<'_> {
    fn drop(&mut self) {
        if self.diag.is_empty() { return; }
        let mut last: Vec<u32> = self.diag.iter().map(|x| x.0).collect();
        let tot: f64 = self.diag.iter().map(|x| x.1 as f64).sum::<f64>() / self.diag.len() as f64;
        last.sort();
        let q = |f: f64| last[((last.len() as f64 * f) as usize).min(last.len() - 1)];
        let mean = last.iter().map(|&x| x as f64).sum::<f64>() / last.len() as f64;
        eprintln!("H44diag ef={} n={} hops_mean={tot:.1} last_discovery mean={mean:.1} p50={} p90={} p95={} p99={}", self.ef, last.len(), q(0.5), q(0.9), q(0.95), q(0.99));
    }
}
