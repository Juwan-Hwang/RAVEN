//! Refiner 框架：粗量化搜索 + 精量化 rerank 通用管道
//!
//! 设计文档第四层 Refiner 框架：
//! 粗量化搜索（SQ8/PQ4/PQ8）→ 候选集 → 精量化 rerank → 最终结果
//!
//! Glass 有通用的 Refiner 框架，RAVEN 此前是手动拼接。
//! 本模块提取通用 rerank 逻辑，支持任意精量化器组合：
//! - F32Refiner: f32 全精度（当前主线，SIFT-128 默认）
//! - F16Refiner: f16 半精度（带宽减半，高维数据集 768+ 收益显著）

use crate::distance::{l2_simd, f16::{F16, l2_f16_mixed_simd}};

/// 精量化 Refiner trait：对粗搜索候选用精确距离重排序
///
/// 设计文档：粗量化距离排序与精确距离高度一致，
/// 只需对 top-N 候选 rerank 即可覆盖 top-k。
pub trait Refiner {
    /// 对候选集用精确距离重排序，返回 top-k
    ///
    /// `candidates` 已按粗量化距离升序排列
    /// `rerank_factor` 控制重排序的候选数 = k × factor
    fn rerank(&self, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)>;
}

/// f32 全精度 Refiner（默认主线）
///
/// 使用 f32 全精度向量计算精确 L2 距离
/// 适用场景：所有数据集（SIFT-128 等低维数据集的默认选择）
pub struct F32Refiner<'a> {
    /// f32 全精度向量
    vectors: &'a [f32],
    /// 维度
    dim: usize,
}

impl<'a> F32Refiner<'a> {
    /// 创建 f32 全精度 Refiner
    pub fn new(vectors: &'a [f32], dim: usize) -> Self {
        Self { vectors, dim }
    }
}

impl<'a> Refiner for F32Refiner<'a> {
    #[inline]
    fn rerank(&self, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)> {
        let rerank_n = (k * rerank_factor).max(k).min(candidates.len());
        // H27：rerank 软件预取（RAVEN_RR_PF=D，默认 4；0 关闭）：提前 D 个候选预取其整条 f32 向量
        let pf = rerank_pf_dist();
        let vb = std::mem::size_of_val(&self.vectors[..self.dim.min(self.vectors.len())]);
        let pf_vec = |id: u32| {
            let base = self.vectors.as_ptr().wrapping_add(id as usize * self.dim) as *const i8;
            let mut off = 0usize;
            while off < vb { unsafe { crate::arch::prefetch_l1(base.wrapping_add(off)); } off += 64; }
        };
        if pf > 0 { for c in candidates.iter().take(pf.min(rerank_n)) { pf_vec(c.0); } }
        // R12 H133：rerank 起始时把前 RR_L2 个候选的整条向量预取到 L2（L2 请求队列，不占 L1 MSHR）；默认 0 = 关
        let l2n = RR_L2.load(std::sync::atomic::Ordering::Relaxed).min(rerank_n);
        for c in candidates.iter().skip(pf).take(l2n.saturating_sub(pf)) {
            let base = self.vectors.as_ptr().wrapping_add(c.0 as usize * self.dim) as *const i8;
            let mut off = 0usize;
            while off < vb { unsafe { crate::arch::prefetch_l2(base.wrapping_add(off)); } off += 64; }
        }
        if rerank_ea_chunk() > 0 && rerank_n > k && k > 0 {
            // H28：精确 early-abandon rerank——维护 top-k（升序小数组），分块累加部分距离，
            // 部分和已超过当前第 k 名即放弃（部分和单调不减，结果与全量 rerank 逐位一致）
            let ch = rerank_ea_chunk();
            let dim = self.dim;
            let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
            for i in 0..rerank_n {
                if pf > 0 && i + pf < rerank_n { pf_vec(candidates[i + pf].0); }
                let id = candidates[i].0;
                let v = &self.vectors[id as usize * dim..(id as usize + 1) * dim];
                let dist = if top.len() < k {
                    l2_simd(query, v)
                } else {
                    let bound = top[k - 1].1;
                    let mut acc = 0.0f32; let mut o = 0usize; let mut dead = false;
                    while o < dim {
                        let e = (o + ch).min(dim);
                        acc += l2_simd(&query[o..e], &v[o..e]);
                        if acc >= bound { dead = true; break; }
                        o = e;
                    }
                    if dead { continue; }
                    acc
                };
                let pos = top.partition_point(|x| x.1 <= dist);
                if pos < k { top.insert(pos, (id, dist)); top.truncate(k); }
            }
            return top;
        }
        let mut results: Vec<(u32, f32)> = Vec::with_capacity(rerank_n);
        for i in 0..rerank_n {
            if pf > 0 && i + pf < rerank_n { pf_vec(candidates[i + pf].0); }
            let id = candidates[i].0;
            let dist = l2_simd(
                query,
                &self.vectors[id as usize * self.dim..(id as usize + 1) * self.dim],
            );
            results.push((id, dist));
        }
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        results
    }
}

/// f16 半精度 Refiner（带宽优化）
///
/// 数据库预量化为 f16（2B/元素 vs f32 4B/元素），
/// rerank 阶段内存带宽减半，F16C 指令在寄存器中转回 f32 计算
///
/// 适用场景：维度 ≥ 512 的高维数据集（Cohere-768、GIST-960 等）
/// 低维数据集（SIFT-128）：rerank 数据已在 L1 cache，带宽不是瓶颈，收益极小
pub struct F16Refiner<'a> {
    /// f16 预量化向量
    vectors: &'a [F16],
    /// 维度
    dim: usize,
}

impl<'a> F16Refiner<'a> {
    /// 创建 f16 半精度 Refiner
    pub fn new(vectors: &'a [F16], dim: usize) -> Self {
        Self { vectors, dim }
    }
}

impl<'a> Refiner for F16Refiner<'a> {
    #[inline]
    fn rerank(&self, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)> {
        let rerank_n = (k * rerank_factor).max(k).min(candidates.len());
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(rerank_n)
            .map(|(id, _)| {
                let dist = l2_f16_mixed_simd(
                    query,
                    &self.vectors[id as usize * self.dim..(id as usize + 1) * self.dim],
                );
                (id, dist)
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        results
    }
}

/// 通用 rerank 函数（零分配热路径之外的辅助函数）
///
/// 供 batch_search 等无法直接使用 trait 对象的场景调用
#[inline]
pub fn rerank_f32(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rerank_factor: usize,
) -> Vec<(u32, f32)> {
    F32Refiner::new(vectors, dim).rerank(query, candidates, k, rerank_factor)
}

/// f16 rerank 函数（同上）
#[inline]
pub fn rerank_f16(
    f16_vectors: &[F16],
    dim: usize,
    query: &[f32],
    candidates: Vec<(u32, f32)>,
    k: usize,
    rerank_factor: usize,
) -> Vec<(u32, f32)> {
    F16Refiner::new(f16_vectors, dim).rerank(query, candidates, k, rerank_factor)
}

/// H27：rerank 预取距离（RAVEN_RR_PF 覆盖；默认 0 = 关闭（第 8 轮审计），4 = 原设置）
#[inline]
pub fn rerank_pf_dist() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = RR_PF.load(Relaxed);
    if v != usize::MAX { return v; }
    let d = std::env::var("RAVEN_RR_PF").ok().and_then(|x| x.parse().ok()).unwrap_or(if cfg!(target_arch = "aarch64") { 1 } else { 0 }); // H99：ARM pad32 后 rrpf=1 两轮正反序 SIFT +3.5–4.5%、GloVe +1.7–3.6% → ARM 默认 1；x86 H92 未过 → 0。第 8 轮：GloVe 仅 +1.6% <3% → 默认关（SIFT +7.8%，RAVEN_RR_PF=4 可开）
    RR_PF.store(d, Relaxed);
    d
}
/// R12 H133：rerank 起始 L2 批量预取的候选数（RAVEN_RR_L2，默认 0；AB 旋钮 rrl2）
/// R14-C2：aarch64 默认 100（与 navpf、vh144 组合 5 轮 SIFT +13…+16%、GloVe +6…+13%）；x86 C2 3 轮 SIFT +8…+11%、GloVe +2…+4% → 也默认 100
pub static RR_L2: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
static RR_PF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
/// 进程内 A/B：设置 rerank 预取距离
pub fn set_rerank_pf_dist(v: usize) { RR_PF.store(v, std::sync::atomic::Ordering::Relaxed); }

/// H28：rerank early-abandon 分块维度（RAVEN_RR_EA；默认 0 = 关闭）
#[inline]
pub fn rerank_ea_chunk() -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("RAVEN_RR_EA").ok().and_then(|x| x.parse().ok()).unwrap_or(0))
}

/// H53：整数数据集（全部分量为 0..=255 的整数，如 SIFT/Fashion/MNIST）的 u8 无损 rerank 存储：
/// 每条向量 dim 字节（SIFT 128B = 2 行，f32 为 8 行）。结果与 f32 rerank 数学上等价（仅浮点求和顺序不同）。
#[inline]
pub fn l2_u8_f32(q: &[f32], x: &[u8]) -> f32 {
    let mut acc = [0f32; 16];
    let n = q.len().min(x.len());
    let mut i = 0;
    while i + 16 <= n {
        for j in 0..16 { let d = q[i + j] - x[i + j] as f32; acc[j] += d * d; }
        i += 16;
    }
    let mut s: f32 = acc.iter().sum();
    while i < n { let d = q[i] - x[i] as f32; s += d * d; i += 1; }
    s
}

/// R48：u8 精确距离（两侧均为 0..255 整数），i32 累加，LLVM 自动向量化为 vpmaddwd。
#[inline]
pub fn l2_u8_u8(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0u32; 32];
    let mut ca = a.chunks_exact(32); let mut cb = b.chunks_exact(32);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for j in 0..32 { let d = x[j] as i32 - y[j] as i32; acc[j] = acc[j].wrapping_add((d * d) as u32); }
    }
    let mut s: u32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) { let d = *x as i32 - *y as i32; s += (d * d) as u32; }
    s
}

thread_local! { static QU8: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(Vec::new()); }

pub fn rerank_u8(vecs: &[u8], dim: usize, query: &[f32], candidates: Vec<(u32, f32)>, k: usize, rerank_factor: usize) -> Vec<(u32, f32)> {
    let rerank_n = (k * rerank_factor).max(k).min(candidates.len());
    let pf = rerank_pf_dist();
    let pf_vec = |id: u32| {
        let base = vecs.as_ptr().wrapping_add(id as usize * dim) as *const i8;
        let mut off = 0usize;
        while off < dim { unsafe { crate::arch::prefetch_l1(base.wrapping_add(off)); } off += 64; }
    };
    if pf > 0 { for c in candidates.iter().take(pf.min(rerank_n)) { pf_vec(c.0); } }
    // R48：查询也是整数 → 全 u8 精确距离 + 分块 early-abandon（与 f32 全量 rerank 排序一致，整数距离无舍入）
    let qint = query.iter().all(|&x| x >= 0.0 && x <= 255.0 && x.fract() == 0.0);
    if qint && k > 0 {
        return QU8.with(|qb| {
            let mut qb = qb.borrow_mut(); qb.clear(); qb.extend(query.iter().map(|&x| x as u8));
            let ch = 128usize;
            let mut top: Vec<(u32, u32)> = Vec::with_capacity(k + 1);
            for i in 0..rerank_n {
                if pf > 0 && i + pf < rerank_n { pf_vec(candidates[i + pf].0); }
                let id = candidates[i].0;
                let v = &vecs[id as usize * dim..(id as usize + 1) * dim];
                let dist = if top.len() < k { l2_u8_u8(&qb, v) } else {
                    let bound = top[k - 1].1;
                    let mut acc = 0u32; let mut o = 0usize; let mut dead = false;
                    while o < dim { let e = (o + ch).min(dim); acc += l2_u8_u8(&qb[o..e], &v[o..e]); if acc >= bound { dead = true; break; } o = e; }
                    if dead { continue; }
                    acc
                };
                let pos = top.partition_point(|x| x.1 <= dist);
                if pos < k { top.insert(pos, (id, dist)); top.truncate(k); }
            }
            top.into_iter().map(|(i, d)| (i, d as f32)).collect()
        });
    }
    let mut results: Vec<(u32, f32)> = Vec::with_capacity(rerank_n);
    for i in 0..rerank_n {
        if pf > 0 && i + pf < rerank_n { pf_vec(candidates[i + pf].0); }
        let id = candidates[i].0;
        results.push((id, l2_u8_f32(query, &vecs[id as usize * dim..(id as usize + 1) * dim])));
    }
    results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(k);
    results
}

/// H53 开关（RAVEN_U8RR=1；AB knob u8rr）
pub static U8_RR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod r48_tests {
    use super::*;
    #[test]
    fn rerank_u8_int_matches_f32() {
        let dim = 300; let n = 200;
        let mut st = 12345u64; let mut nx = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st % 256) as u8 };
        let v: Vec<u8> = (0..n * dim).map(|_| nx()).collect();
        let vf: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let q: Vec<f32> = (0..dim).map(|_| nx() as f32).collect();
        let cands: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, 0.0)).collect();
        let a = rerank_u8(&v, dim, &q, cands.clone(), 10, 20);
        let mut b: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, l2_u8_f32(&q, &v[i as usize * dim..(i as usize + 1) * dim]))).collect();
        b.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
        let _ = vf;
        assert_eq!(a.iter().map(|x| x.1 as u64).collect::<Vec<_>>(), b[..10].iter().map(|x| x.1 as u64).collect::<Vec<_>>());
    }
}
