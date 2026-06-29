//! SQ8 标量量化器（Phase 1 Step 0）
//!
//! 每维度 1 字节（u8）量化，向量 128B(f32) → 32B(u8)，内存降 4x。
//!
//! 编码：code[i] = round((v[i] - min[i]) / (max[i] - min[i]) * 255) clamp [0,255]
//! 解码：v[i] ≈ code[i] * scale[i] + offset[i]
//!       其中 scale[i] = (max[i] - min[i]) / 255, offset[i] = min[i]
//!
//! L2 距离（量化空间）：
//!   d = Σ (qa[i] - qb[i])² × scale_sq[i]
//! 其中 scale_sq[i] = scale[i]² = ((max[i]-min[i]) / 255)²
//!
//! AVX2 内核：每次处理 16 个 u8 → 2×__m128i(i16) → madd_epi16 → 2×__m128(i32) → cvt f32 → FMA
//! 128 维 = 8 次迭代（每次 16 维）

use std::arch::x86_64::*;

/// SQ8 量化参数（per-dimension min/max/scale）
#[derive(Debug, Clone)]
pub struct SQ8Params {
    /// 每维度最小值
    pub min: Vec<f32>,
    /// 每维度最大值
    pub max: Vec<f32>,
    /// 每维度 scale = (max - min) / 255
    pub scale: Vec<f32>,
    /// 每维度 offset = min
    pub offset: Vec<f32>,
    /// 每维度 scale² （L2 距离用）
    pub scale_sq: Vec<f32>,
    /// 每维度逆尺度 inv_scale = 1/scale（编码用，FMA 友好）
    pub inv_scale: Vec<f32>,
    /// 每维度 -offset/scale（编码用，配合 inv_scale 实现 FMA）
    pub neg_offset_div: Vec<f32>,
    /// 维度
    pub dim: usize,
}

impl SQ8Params {
    /// 从训练数据拟合量化参数（per-dimension min/max）
    pub fn fit(data: &[f32], dim: usize) -> Self {
        let n = data.len() / dim;
        assert!(n > 0, "SQ8Params::fit: empty data");

        let mut min = vec![f32::MAX; dim];
        let mut max = vec![f32::MIN; dim];

        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            for d in 0..dim {
                if row[d] < min[d] {
                    min[d] = row[d];
                }
                if row[d] > max[d] {
                    max[d] = row[d];
                }
            }
        }

        // 防止 max == min（常数维度）
        for d in 0..dim {
            if max[d] <= min[d] {
                max[d] = min[d] + 1.0;
            }
        }

        let scale: Vec<f32> = (0..dim).map(|d| (max[d] - min[d]) / 255.0).collect();
        let offset = min.clone();
        let scale_sq: Vec<f32> = scale.iter().map(|s| s * s).collect();
        // 预计算逆尺度：编码时用 FMA (v * inv_scale + neg_offset_div) 替代除法
        let inv_scale: Vec<f32> = scale.iter().map(|s| 1.0 / s).collect();
        let neg_offset_div: Vec<f32> = (0..dim).map(|d| -offset[d] * inv_scale[d]).collect();

        Self { min, max, scale, offset, scale_sq, inv_scale, neg_offset_div, dim }
    }

    /// 编码单个向量 → u8 codes（堆分配版本，非热路径用）
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim);
        let mut buf = vec![0u8; self.dim];
        self.encode_into(v, &mut buf);
        buf
    }

    /// 编码单个向量到预分配 buffer（零分配，热路径专用）
    ///
    /// `buf` 长度必须等于 `self.dim`
    #[inline(always)]
    pub fn encode_into(&self, v: &[f32], buf: &mut [u8]) {
        assert_eq!(v.len(), self.dim);
        assert_eq!(buf.len(), self.dim);
        for d in 0..self.dim {
            // FMA: v * inv_scale + neg_offset_div ≡ (v - offset) / scale
            let q = (v[d] * self.inv_scale[d] + self.neg_offset_div[d]).round();
            buf[d] = q.clamp(0.0, 255.0) as u8;
        }
    }

    /// 编码整个数据集 → 扁平 u8 数组 (n × dim)
    pub fn encode_all(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len() / self.dim;
        let mut codes = vec![0u8; n * self.dim];
        for i in 0..n {
            let row = &data[i * self.dim..(i + 1) * self.dim];
            for d in 0..self.dim {
                let q = (row[d] * self.inv_scale[d] + self.neg_offset_div[d]).round();
                codes[i * self.dim + d] = q.clamp(0.0, 255.0) as u8;
            }
        }
        codes
    }

    /// 解码单个 u8 code → f32 近似向量
    #[allow(dead_code)]
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(code.len(), self.dim);
        (0..self.dim)
            .map(|d| code[d] as f32 * self.scale[d] + self.offset[d])
            .collect()
    }
}

// ─── AVX2 L2 距离核 ───────────────────────────────────────────────────

/// SQ8 L2 距离（AVX2 256-bit）：每次处理 32 个维度
///
/// 计算 d = Σ (qa[i] - qb[i])² × scale_sq[i]
///
/// 256-bit 流水线（每次 32 维）：
/// 1. _mm256_loadu_si256: 加载 32 个 u8
/// 2. _mm256_unpacklo/hi_epi8(x, zero): u8→i16 零扩展，拆为 2×(16×i16)
///    （unpack 在 128-bit lane 内独立操作，lo 覆盖 bytes 0-7/16-23，
///     hi 覆盖 bytes 8-15/24-31，合起来正好 32 字节，顺序正确）
/// 3. _mm256_sub_epi16: 差值（i16 范围 [-255, 255]，不溢出）
/// 4. _mm256_madd_epi16(diff, diff): 每对 i16 的平方和 → 8 个 i32
/// 5. _mm256_cvtepi32_ps → _mm256_fmadd_ps: 转换 + 加权累加
///
/// 选 unpack 而非 cvtepu8_epi16 的原因：
///   - unpack 可跑 port 0/1/5（throughput 0.5），cvtepu8 只能 port 5（throughput 1）
///   - unpack 无需 extracti128_si256 的 lane-crossing（latency 3, port 5 only）
///   - 实测 QPS +6.6%（11997 → 12786 @ ef=50），recall 完全一致
///
/// SIFT-128: 4 次迭代（vs 128-bit 版 8 次），吞吐翻倍
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn l2_sq8_avx2(qa: &[u8], qb: &[u8], scale_sq: &[f32]) -> f32 {
    let n = qa.len();
    debug_assert_eq!(qb.len(), n);
    debug_assert_eq!(scale_sq.len(), n);

    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_ps(); // 8 个 f32 累加器

    let chunks = n / 32;
    for c in 0..chunks {
        let base = c * 32;

        // 加载 32 u8（一个 256-bit load）
        let a = _mm256_loadu_si256(qa.as_ptr().add(base) as *const __m256i);
        let b = _mm256_loadu_si256(qb.as_ptr().add(base) as *const __m256i);

        // u8→i16 零扩展：interleave with zero = zero-extend
        // unpacklo: lane0[0-7] + lane1[16-23] → 16×i16
        // unpackhi: lane0[8-15] + lane1[24-31] → 16×i16
        let a_lo = _mm256_unpacklo_epi8(a, zero);
        let a_hi = _mm256_unpackhi_epi8(a, zero);
        let b_lo = _mm256_unpacklo_epi8(b, zero);
        let b_hi = _mm256_unpackhi_epi8(b, zero);

        // 差值 (i16, 范围 [-255, 255]，不会溢出)
        let d_lo = _mm256_sub_epi16(a_lo, b_lo);
        let d_hi = _mm256_sub_epi16(a_hi, b_hi);

        // 平方和：madd(diff, diff) = diff[2i]² + diff[2i+1]² → 8 个 i32
        let sq_lo = _mm256_madd_epi16(d_lo, d_lo);
        let sq_hi = _mm256_madd_epi16(d_hi, d_hi);

        // 转 f32
        let sq_lo_f = _mm256_cvtepi32_ps(sq_lo);
        let sq_hi_f = _mm256_cvtepi32_ps(sq_hi);

        // 加载 scale_sq (8 + 8 = 16 个 f32)
        let sc_lo = _mm256_loadu_ps(scale_sq.as_ptr().add(base));
        let sc_hi = _mm256_loadu_ps(scale_sq.as_ptr().add(base + 8));

        // 加权累加
        acc = _mm256_fmadd_ps(sq_lo_f, sc_lo, acc);
        acc = _mm256_fmadd_ps(sq_hi_f, sc_hi, acc);
    }

    // 水平求和 8 个 f32
    let mut result = horizontal_sum_256(acc);

    // 尾部标量处理（32 维对齐剩余）
    let remainder_start = chunks * 32;
    for i in remainder_start..n {
        let diff = qa[i] as i32 - qb[i] as i32;
        result += (diff * diff) as f32 * scale_sq[i];
    }

    result
}

/// __m256 水平求和（8 个 f32 → 1 个 f32）
#[inline(always)]
unsafe fn horizontal_sum_256(v: __m256) -> f32 {
    // [a, b, c, d, e, f, g, h]
    let hi128 = _mm256_extractf128_ps(v, 1); // [e, f, g, h]
    let lo128 = _mm256_castps256_ps128(v);   // [a, b, c, d]
    let sum128 = _mm_add_ps(lo128, hi128);   // [a+e, b+f, c+g, d+h]
    horizontal_sum_128(sum128)
}

/// __m128 水平求和（4 个 f32 → 1 个 f32）
#[inline(always)]
unsafe fn horizontal_sum_128(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v); // [b, b, d, d]
    let sums = _mm_add_ps(v, shuf); // [a+b, _, c+d, _]
    let shuf2 = _mm_movehl_ps(v, sums); // [c+d, _, _, _]
    let sums2 = _mm_add_ss(sums, shuf2); // [a+b+c+d, _, _, _]
    _mm_cvtss_f32(sums2)
}

// ─── i32 水平求和（无加权距离核专用） ──────────────────────────────────

/// __m256i 水平求和（8 个 i32 → 1 个 i32）
#[inline(always)]
unsafe fn horizontal_sum_i32_256(v: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256::<1>(v);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128); // [s0+s4, s1+s5, s2+s6, s3+s7]
    horizontal_sum_i32_128(sum128)
}

/// __m128i 水平求和（4 个 i32 → 1 个 i32）
///
/// 两步 shuffle+add：
///   [s0, s1, s2, s3] → shuffle 0x4E → [s2, s3, s0, s1]
///   add → [s0+s2, s1+s3, ...] → shuffle 0x01 → [s1+s3, s0+s2, ...]
///   add → [s0+s1+s2+s3, ...]
#[inline(always)]
unsafe fn horizontal_sum_i32_128(v: __m128i) -> i32 {
    let shuf = _mm_shuffle_epi32::<0x4E>(v); // _MM_SHUFFLE(1,0,3,2): [s2, s3, s0, s1]
    let sum1 = _mm_add_epi32(v, shuf);      // [s0+s2, s1+s3, ...]
    let shuf2 = _mm_shuffle_epi32::<0x01>(sum1); // _MM_SHUFFLE(0,0,0,1): [s1+s3, ...]
    let sum2 = _mm_add_epi32(sum1, shuf2);  // [s0+s1+s2+s3, ...]
    _mm_cvtsi128_si32(sum2)
}

// ─── 无加权 L2 距离核（OPT-15） ───────────────────────────────────────

/// SQ8 无加权 L2 距离（AVX2 256-bit）：跳过 scale_sq 加权
///
/// 计算 d_raw = Σ (qa[i] - qb[i])²
///
/// 与 `l2_sq8_avx2` 的区别：
/// - i32 累加（跳过 cvtepi32_ps）
/// - 无 scale_sq 加权（跳过 loadu_ps + fmadd_ps）
/// - 每 32 维节省 6 条指令（2× cvtepi32_ps + 2× loadu_ps + 2× fmadd_ps → 2× add_epi32）
/// - SIFT-128 共 4 chunks，净节省 24 条指令
///
/// 用途：图导航只需相对排序，最终 f32 rerank 保证精度
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn l2_sq8_raw_avx2(qa: &[u8], qb: &[u8]) -> f32 {
    let n = qa.len();
    debug_assert_eq!(qb.len(), n);

    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_si256(); // 8 个 i32 累加器

    let chunks = n / 32;
    for c in 0..chunks {
        let base = c * 32;

        let a = _mm256_loadu_si256(qa.as_ptr().add(base) as *const __m256i);
        let b = _mm256_loadu_si256(qb.as_ptr().add(base) as *const __m256i);

        let a_lo = _mm256_unpacklo_epi8(a, zero);
        let a_hi = _mm256_unpackhi_epi8(a, zero);
        let b_lo = _mm256_unpacklo_epi8(b, zero);
        let b_hi = _mm256_unpackhi_epi8(b, zero);

        let d_lo = _mm256_sub_epi16(a_lo, b_lo);
        let d_hi = _mm256_sub_epi16(a_hi, b_hi);

        let sq_lo = _mm256_madd_epi16(d_lo, d_lo);
        let sq_hi = _mm256_madd_epi16(d_hi, d_hi);

        // i32 累加（跳过 cvtepi32_ps + scale_sq load + fmadd）
        acc = _mm256_add_epi32(acc, sq_lo);
        acc = _mm256_add_epi32(acc, sq_hi);
    }

    let mut result = horizontal_sum_i32_256(acc) as f32;

    // 尾部标量处理
    let remainder_start = chunks * 32;
    for i in remainder_start..n {
        let diff = qa[i] as i32 - qb[i] as i32;
        result += (diff * diff) as f32;
    }

    result
}

/// 统一分发：AVX2 → 标量
///
/// 当 `target-cpu=native` 编译且 CPU 支持 AVX2+FMA 时，
/// `cfg!(target_feature = ...)` 在编译期为 true，直接走 AVX2 路径，
/// 消除每次距离计算的 `is_x86_feature_detected!` 运行时分支开销。
#[inline(always)]
pub fn l2_sq8(qa: &[u8], qb: &[u8], scale_sq: &[f32]) -> f32 {
    // 编译期已知 AVX2+FMA：零运行时开销
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        unsafe { l2_sq8_avx2(qa, qb, scale_sq) }
    }
    // 运行时检测（非 native 编译或非 AVX2 CPU）
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    {
        if is_sq8_avx2_supported() {
            unsafe { l2_sq8_avx2(qa, qb, scale_sq) }
        } else {
            l2_sq8_scalar(qa, qb, scale_sq)
        }
    }
}

/// 标量 fallback（target-cpu=native + AVX2 时编译期排除，保留供非 AVX2 路径）
#[allow(dead_code)]
#[inline(always)]
fn l2_sq8_scalar(qa: &[u8], qb: &[u8], scale_sq: &[f32]) -> f32 {
    let n = qa.len();
    let mut sum = 0.0f32;
    for i in 0..n {
        let diff = qa[i] as i32 - qb[i] as i32;
        sum += (diff * diff) as f32 * scale_sq[i];
    }
    sum
}

/// 无加权标量 fallback
#[allow(dead_code)]
#[inline(always)]
fn l2_sq8_raw_scalar(qa: &[u8], qb: &[u8]) -> f32 {
    let n = qa.len();
    let mut sum = 0i32;
    for i in 0..n {
        let diff = qa[i] as i32 - qb[i] as i32;
        sum += diff * diff;
    }
    sum as f32
}

/// 统一分发：AVX2 → 标量（无加权版本）
///
/// 与 `l2_sq8` 相同的编译期分发策略，但跳过 scale_sq 加权。
/// 图导航专用：最终 f32 rerank 保证结果精度。
#[inline(always)]
pub fn l2_sq8_raw(qa: &[u8], qb: &[u8]) -> f32 {
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        unsafe { l2_sq8_raw_avx2(qa, qb) }
    }
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    {
        if is_sq8_avx2_supported() {
            unsafe { l2_sq8_raw_avx2(qa, qb) }
        } else {
            l2_sq8_raw_scalar(qa, qb)
        }
    }
}

/// 检测 AVX2 支持
pub fn is_sq8_avx2_supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

// ─── SQ8 量化数据集 ──────────────────────────────────────────────────

/// SQ8 量化后的完整数据集（codes + params）
pub struct SQ8Dataset {
    /// 扁平 u8 codes: n × dim
    pub codes: Vec<u8>,
    /// 量化参数
    pub params: SQ8Params,
    /// 维度
    pub dim: usize,
    /// 向量数
    pub n: usize,
}

impl SQ8Dataset {
    /// 从 f32 数据集构建 SQ8 量化数据集
    pub fn build(data: &[f32], dim: usize) -> Self {
        let n = data.len() / dim;
        let params = SQ8Params::fit(data, dim);
        let codes = params.encode_all(data);
        Self { codes, params, dim, n }
    }

    /// 获取第 idx 个向量的 SQ8 code 引用
    #[inline(always)]
    pub fn code(&self, idx: usize) -> &[u8] {
        &self.codes[idx * self.dim..(idx + 1) * self.dim]
    }

    /// 获取第 idx 个向量的 SQ8 code 引用（无边界检查，热路径专用）
    ///
    /// SAFETY: idx 必须 < self.n
    #[inline(always)]
    pub unsafe fn code_unchecked(&self, idx: usize) -> &[u8] {
        let start = idx * self.dim;
        self.codes.get_unchecked(start..start + self.dim)
    }

    /// 计算查询 u8 code 与第 idx 个向量之间的 SQ8 L2 距离
    #[inline(always)]
    pub fn distance(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq8(query_code, self.code(idx), &self.params.scale_sq)
    }

    /// 计算查询 u8 code 与第 idx 个向量之间的 SQ8 L2 距离（无边界检查，热路径专用）
    ///
    /// SAFETY: idx 必须 < self.n
    #[inline(always)]
    pub unsafe fn distance_unchecked(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq8(query_code, self.code_unchecked(idx), &self.params.scale_sq)
    }

    /// 计算无加权 SQ8 L2 距离（OPT-15：图导航专用）
    ///
    /// 跳过 scale_sq 加权，返回 Σ (qa[i] - qb[i])²。
    /// 图导航只需相对排序，最终 f32 rerank 保证精度。
    #[inline(always)]
    pub fn distance_raw(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq8_raw(query_code, self.code(idx))
    }

    /// 计算无加权 SQ8 L2 距离（无边界检查，热路径专用）
    ///
    /// SAFETY: idx 必须 < self.n
    #[inline(always)]
    pub unsafe fn distance_raw_unchecked(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq8_raw(query_code, self.code_unchecked(idx))
    }
}

// ─── 测试 ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq8_params_fit_basic() {
        // 2 维, 3 个向量
        let data = [
            0.0, 0.0,
            1.0, 2.0,
            0.5, 1.0,
        ];
        let p = SQ8Params::fit(&data, 2);
        assert!((p.min[0] - 0.0).abs() < 1e-6);
        assert!((p.max[0] - 1.0).abs() < 1e-6);
        assert!((p.min[1] - 0.0).abs() < 1e-6);
        assert!((p.max[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn sq8_encode_decode_roundtrip() {
        let data = [
            0.0, 0.0,
            1.0, 2.0,
            0.5, 1.0,
        ];
        let p = SQ8Params::fit(&data, 2);
        let v = [0.5, 1.0];
        let code = p.encode(&v);
        let decoded = p.decode(&code);
        // SQ8 量化误差应小于 1 个量化级
        let max_err = p.scale[0].max(p.scale[1]);
        assert!((decoded[0] - v[0]).abs() <= max_err);
        assert!((decoded[1] - v[1]).abs() <= max_err);
    }

    #[test]
    fn sq8_distance_matches_scalar() {
        if !is_sq8_avx2_supported() {
            eprintln!("AVX2 not supported, skipping SIMD test");
            return;
        }
        // 128 维
        let dim = 128;
        let data: Vec<f32> = (0..256).flat_map(|i| {
            (0..dim).map(move |d| (i * dim + d) as f32 * 0.01)
        }).collect();
        let ds = SQ8Dataset::build(&data, dim);

        let query: Vec<f32> = (0..dim).map(|d| d as f32 * 0.01 + 0.5).collect();
        let qcode = ds.params.encode(&query);

        for idx in [0, 1, 50, 127, 200].into_iter().filter(|&i| i < ds.n) {
            let d_simd = ds.distance(&qcode, idx);
            let d_scalar = l2_sq8_scalar(&qcode, ds.code(idx), &ds.params.scale_sq);
            let rel_err = (d_simd - d_scalar).abs() / d_scalar.max(1e-10);
            assert!(rel_err < 1e-4, "idx={}: simd={} scalar={} rel_err={}", idx, d_simd, d_scalar, rel_err);
        }
    }

    #[test]
    fn sq8_distance_vs_f32_correlation() {
        // SQ8 距离应与 f32 距离高度相关
        let dim = 128;
        let n = 500;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5).collect();
        let ds = SQ8Dataset::build(&data, dim);

        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.1).cos() * 0.5 + 0.5).collect();
        let qcode = ds.params.encode(&query);

        // 计算前 100 个向量的 SQ8 距离和 f32 距离
        let mut sq8_dists = Vec::new();
        let mut f32_dists = Vec::new();
        for i in 0..100.min(n) {
            let d_sq8 = ds.distance(&qcode, i);
            let d_f32: f32 = (0..dim)
                .map(|d| {
                    let diff = data[i * dim + d] - query[d];
                    diff * diff
                })
                .sum();
            sq8_dists.push(d_sq8);
            f32_dists.push(d_f32);
        }

        // 检查排序一致性：SQ8 top-10 应与 f32 top-10 高度重叠
        let mut sq8_ranked: Vec<(usize, f32)> = (0..sq8_dists.len())
            .map(|i| (i, sq8_dists[i]))
            .collect();
        sq8_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut f32_ranked: Vec<(usize, f32)> = (0..f32_dists.len())
            .map(|i| (i, f32_dists[i]))
            .collect();
        f32_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let sq8_top10: std::collections::HashSet<usize> =
            sq8_ranked.iter().take(10).map(|(i, _)| *i).collect();
        let f32_top10: std::collections::HashSet<usize> =
            f32_ranked.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = sq8_top10.intersection(&f32_top10).count();

        assert!(overlap >= 7, "SQ8 top-10 overlap with f32: {}/10 (expected ≥7)", overlap);
    }

    #[test]
    fn sq8_constant_dimension_handling() {
        // 某维度全为常数 → max == min → 应被安全处理
        let data = [
            1.0, 5.0,
            1.0, 3.0,
            1.0, 7.0,
        ];
        let p = SQ8Params::fit(&data, 2);
        // 常数维度 min[0]=1.0, max[0]=1.0 → 应被强制 max=min+1
        assert!((p.max[0] - 2.0).abs() < 1e-6);
        let code = p.encode(&[1.0, 5.0]);
        // 编码不应 panic
        assert_eq!(code.len(), 2);
    }
}
