//! SQ4 标量量化器（4-bit per dimension）
//!
//! 每维度 4 bit（16 级），向量 128B(f32) → 64B(packed 4-bit)。
//! 相比 SQ8 的 128B，内存带宽降 2x，且 1M 向量仅 64MB（部分进 L3 cache）。
//!
//! 编码：code[i] = round((v[i] - min[i]) / (max[i] - min[i]) * 15) clamp [0,15]
//! 打包：2 个 4-bit code → 1 byte (even dim → low nibble, odd dim → high nibble)
//!
//! L2 距离（量化空间，无加权 — 图导航专用）：
//!   d_raw = Σ (qa[i] - qb[i])²
//!   差值范围 [-15, 15]，平方范围 [0, 225]
//!
//! AVX2 内核：每次处理 64 维（32 bytes packed = 1 cache line）
//!   1. 加载 32 bytes（64 dims packed）
//!   2. 分离 low/high nibbles（and 0x0F / shift right 4 + and 0x0F）
//!   3. u8 减法 + abs_epi8（差值 [-15,15]，abs 后 [0,15]）
//!   4. maddubs_epi16: 相邻 u8 平方和 → u16（每对最大 225+225=450）
//!   5. 水平求和
//! SIFT-128: 2 次迭代（vs SQ8 的 4 次），每次 32 bytes = 1 cache line
//!
//! 与 PQ4 的本质区别：
//! - PQ4：子空间聚类 + LUT 随机查表 → cache miss 杀死性能
//! - SQ4：per-dimension 独立量化 + 顺序 SIMD → 和 SQ8 完全相同的访问模式

use std::arch::x86_64::*;

/// SQ4 量化参数（per-dimension min/max/scale，4-bit 16 级）
#[derive(Debug, Clone)]
pub struct SQ4Params {
    /// 每维度最小值
    pub min: Vec<f32>,
    /// 每维度最大值
    pub max: Vec<f32>,
    /// 每维度 scale = (max - min) / 15
    pub scale: Vec<f32>,
    /// 每维度 inv_scale = 1/scale（编码用，FMA 友好）
    pub inv_scale: Vec<f32>,
    /// 每维度 -min/scale（编码用，配合 inv_scale 实现 FMA）
    pub neg_min_div: Vec<f32>,
    /// 维度
    pub dim: usize,
}

impl SQ4Params {
    /// 从训练数据拟合量化参数（per-dimension min/max）
    pub fn fit(data: &[f32], dim: usize) -> Self {
        let n = data.len() / dim;
        assert!(n > 0, "SQ4Params::fit: empty data");

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

        let scale: Vec<f32> = (0..dim).map(|d| (max[d] - min[d]) / 15.0).collect();
        let inv_scale: Vec<f32> = scale.iter().map(|s| 1.0 / s).collect();
        let neg_min_div: Vec<f32> = (0..dim).map(|d| -min[d] * inv_scale[d]).collect();

        Self { min, max, scale, inv_scale, neg_min_div, dim }
    }

    /// 编码单个向量 → packed 4-bit codes (dim/2 bytes)
    ///
    /// byte[i] = code[2*i] | (code[2*i+1] << 4)
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim);
        let mut buf = vec![0u8; self.dim.div_ceil(2)];
        self.encode_into(v, &mut buf);
        buf
    }

    /// 编码单个向量到预分配 buffer（零分配，热路径专用）
    ///
    /// `buf` 长度必须等于 `(dim + 1) / 2`
    #[inline(always)]
    pub fn encode_into(&self, v: &[f32], buf: &mut [u8]) {
        assert_eq!(v.len(), self.dim);
        for d in 0..self.dim {
            // FMA: v * inv_scale + neg_min_div ≡ (v - min) / scale
            let q = v[d].mul_add(self.inv_scale[d], self.neg_min_div[d]).round();
            let code = q.clamp(0.0, 15.0) as u8;
            if d % 2 == 0 {
                buf[d / 2] = code;
            } else {
                buf[d / 2] |= code << 4;
            }
        }
    }

    /// 编码整个数据集 → 扁平 packed codes (n × ceil(dim/2) bytes)
    pub fn encode_all(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len() / self.dim;
        let code_bytes = self.dim.div_ceil(2);
        let mut codes = vec![0u8; n * code_bytes];
        for i in 0..n {
            let row = &data[i * self.dim..(i + 1) * self.dim];
            for d in 0..self.dim {
                let q = row[d].mul_add(self.inv_scale[d], self.neg_min_div[d]).round();
                let code = q.clamp(0.0, 15.0) as u8;
                if d % 2 == 0 {
                    codes[i * code_bytes + d / 2] = code;
                } else {
                    codes[i * code_bytes + d / 2] |= code << 4;
                }
            }
        }
        codes
    }

    /// 解码单个 packed code → f32 近似向量
    #[allow(dead_code)]
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let code_bytes = self.dim.div_ceil(2);
        assert_eq!(code.len(), code_bytes);
        (0..self.dim)
            .map(|d| {
                let raw = if d % 2 == 0 {
                    code[d / 2] & 0x0F
                } else {
                    (code[d / 2] >> 4) & 0x0F
                };
                (raw as f32).mul_add(self.scale[d], self.min[d])
            })
            .collect()
    }
}

// ─── AVX2 L2 距离核 ───────────────────────────────────────────────────

/// SQ4 无加权 L2 距离（AVX2 256-bit）：每次处理 64 维（32 bytes packed）
///
/// 计算 d_raw = Σ (qa[i] - qb[i])²
///
/// 256-bit 流水线（每次 64 dims = 32 bytes）：
/// 1. _mm256_loadu_si256: 加载 32 bytes（64 dims packed）
/// 2. 分离 low/high nibbles: and 0x0F / srli 4 + and 0x0F
/// 3. _mm256_sub_epi8: u8 差值（i8 范围 [-15, 15]）
/// 4. _mm256_abs_epi8: 绝对值 [0, 15]
/// 5. _mm256_maddubs_epi16: 相邻 u8 平方和 → u16
///    （maddubs 把 (a,b) 对作为 (signed a × unsigned b)，
///     但我们先 abs 了，所以 signed=positive，等价于 a²+b²）
/// 6. _mm256_add_epi16 → 水平求和
///
/// 注意：maddubs_epi16 计算的是 signed×unsigned，
///   abs 后的值 [0,15] 在 i8 正数范围内，
///   所以 (a × a + b × b) 结果正确，最大 15²+15²=450 < 32767
///
/// SIFT-128: 2 次迭代（vs SQ8 的 4 次），吞吐翻倍
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn l2_sq4_raw_avx2(qa: &[u8], qb: &[u8]) -> f32 {
    let n = qa.len(); // packed bytes = ceil(dim/2)
    debug_assert_eq!(qb.len(), n);

    let mask_0f = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_si256(); // 16 × i16 累加器

    // 每次处理 32 bytes = 64 dims
    let chunks = n / 32;
    for c in 0..chunks {
        let base = c * 32;

        // 加载 32 bytes packed codes
        let a = _mm256_loadu_si256(qa.as_ptr().add(base) as *const __m256i);
        let b = _mm256_loadu_si256(qb.as_ptr().add(base) as *const __m256i);

        // 分离 low/high nibbles
        let a_lo = _mm256_and_si256(a, mask_0f);
        let a_hi = _mm256_and_si256(_mm256_srli_epi16(a, 4), mask_0f);
        let b_lo = _mm256_and_si256(b, mask_0f);
        let b_hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask_0f);

        // u8 差值 + abs
        let d_lo = _mm256_abs_epi8(_mm256_sub_epi8(a_lo, b_lo));
        let d_hi = _mm256_abs_epi8(_mm256_sub_epi8(a_hi, b_hi));

        // maddubs: (d[2i] × d[2i]) + (d[2i+1] × d[2i+1]) → i16
        // d 已 abs，范围 [0,15]，作为 signed i8 仍为正数
        // maddubs: signed_epu8 × unsigned_epu8 → i16
        //   = d_lo[2i]² + d_lo[2i+1]²
        let sq_lo = _mm256_maddubs_epi16(d_lo, d_lo);
        let sq_hi = _mm256_maddubs_epi16(d_hi, d_hi);

        acc = _mm256_add_epi16(acc, sq_lo);
        acc = _mm256_add_epi16(acc, sq_hi);
    }

    // 水平求和 16 × i16
    let mut result = horizontal_sum_i16_256(acc) as f32;

    // 尾部标量处理（32 bytes 对齐剩余）
    let remainder_start = chunks * 32;
    for i in remainder_start..n {
        let a_lo = (qa[i] & 0x0F) as i32;
        let a_hi = ((qa[i] >> 4) & 0x0F) as i32;
        let b_lo = (qb[i] & 0x0F) as i32;
        let b_hi = ((qb[i] >> 4) & 0x0F) as i32;
        let d_lo = a_lo - b_lo;
        let d_hi = a_hi - b_hi;
        result += (d_lo * d_lo + d_hi * d_hi) as f32;
    }

    result
}

/// __m256i 水平求和（16 × i16 → 1 个 i32）
#[inline(always)]
unsafe fn horizontal_sum_i16_256(v: __m256i) -> i32 {
    // i16 → i32 扩展求和
    // [16 × i16] → split to [8 × i32 (lo)] + [8 × i32 (hi)]
    let hi128 = _mm256_extracti128_si256::<1>(v);
    let lo128 = _mm256_castsi256_si128(v);

    // madd: pairs of i16 → i32
    let ones = _mm_set1_epi16(1);
    let sum_lo = _mm_madd_epi16(lo128, ones); // 4 × i32
    let sum_hi = _mm_madd_epi16(hi128, ones); // 4 × i32
    let sum = _mm_add_epi32(sum_lo, sum_hi); // 4 × i32

    // 水平求和 4 × i32
    let shuf = _mm_shuffle_epi32::<0x4E>(sum); // [s2, s3, s0, s1]
    let sum2 = _mm_add_epi32(sum, shuf); // [s0+s2, s1+s3, ...]
    let shuf2 = _mm_shuffle_epi32::<0x01>(sum2);
    let sum3 = _mm_add_epi32(sum2, shuf2);
    _mm_cvtsi128_si32(sum3)
}

/// 标量 fallback
#[allow(dead_code)]
#[inline(always)]
fn l2_sq4_raw_scalar(qa: &[u8], qb: &[u8]) -> f32 {
    let n = qa.len();
    let mut sum = 0i32;
    for i in 0..n {
        let a_lo = (qa[i] & 0x0F) as i32;
        let a_hi = ((qa[i] >> 4) & 0x0F) as i32;
        let b_lo = (qb[i] & 0x0F) as i32;
        let b_hi = ((qb[i] >> 4) & 0x0F) as i32;
        let d_lo = a_lo - b_lo;
        let d_hi = a_hi - b_hi;
        sum += d_lo * d_lo + d_hi * d_hi;
    }
    sum as f32
}

/// 统一分发：AVX2 → 标量
#[inline(always)]
pub fn l2_sq4_raw(qa: &[u8], qb: &[u8]) -> f32 {
    #[cfg(target_feature = "avx2")]
    {
        unsafe { l2_sq4_raw_avx2(qa, qb) }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { l2_sq4_raw_avx2(qa, qb) }
        } else {
            l2_sq4_raw_scalar(qa, qb)
        }
    }
}

/// 检测 AVX2 支持
pub fn is_sq4_avx2_supported() -> bool {
    std::is_x86_feature_detected!("avx2")
}

// ─── SQ4 量化数据集 ──────────────────────────────────────────────────

/// SQ4 量化后的完整数据集（packed codes + params）
pub struct SQ4Dataset {
    /// 扁平 packed 4-bit codes: n × ceil(dim/2) bytes
    pub codes: Vec<u8>,
    /// 量化参数
    pub params: SQ4Params,
    /// 维度
    pub dim: usize,
    /// 向量数
    pub n: usize,
    /// 每向量码字节数 = ceil(dim/2)
    pub code_bytes: usize,
}

impl SQ4Dataset {
    /// 从 f32 数据集构建 SQ4 量化数据集
    pub fn build(data: &[f32], dim: usize) -> Self {
        let n = data.len() / dim;
        let params = SQ4Params::fit(data, dim);
        let codes = params.encode_all(data);
        let code_bytes = dim.div_ceil(2);
        Self { codes, params, dim, n, code_bytes }
    }

    /// 获取第 idx 个向量的 packed code 引用
    #[inline(always)]
    pub fn code(&self, idx: usize) -> &[u8] {
        let start = idx * self.code_bytes;
        &self.codes[start..start + self.code_bytes]
    }

    /// 获取第 idx 个向量的 packed code 引用（无边界检查，热路径专用）
    ///
    /// SAFETY: idx 必须 < self.n
    #[inline(always)]
    pub unsafe fn code_unchecked(&self, idx: usize) -> &[u8] {
        let start = idx * self.code_bytes;
        self.codes.get_unchecked(start..start + self.code_bytes)
    }

    /// 计算查询 packed code 与第 idx 个向量之间的 SQ4 L2 距离
    #[inline(always)]
    pub fn distance_raw(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq4_raw(query_code, self.code(idx))
    }

    /// 计算查询 packed code 与第 idx 个向量之间的 SQ4 L2 距离（无边界检查）
    ///
    /// SAFETY: idx 必须 < self.n
    #[inline(always)]
    pub unsafe fn distance_raw_unchecked(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq4_raw(query_code, self.code_unchecked(idx))
    }
}

// ─── 测试 ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq4_params_fit_basic() {
        let data = [0.0, 0.0, 1.0, 2.0, 0.5, 1.0];
        let p = SQ4Params::fit(&data, 2);
        assert!((p.min[0] - 0.0).abs() < 1e-6);
        assert!((p.max[0] - 1.0).abs() < 1e-6);
        assert!((p.min[1] - 0.0).abs() < 1e-6);
        assert!((p.max[1] - 2.0).abs() < 1e-6);
        // scale = (max - min) / 15
        assert!((p.scale[0] - 1.0 / 15.0).abs() < 1e-6);
    }

    #[test]
    fn sq4_encode_decode_roundtrip() {
        let dim = 128;
        let data: Vec<f32> = (0..256)
            .flat_map(|i| (0..dim).map(move |d| (i * dim + d) as f32 * 0.01))
            .collect();
        let p = SQ4Params::fit(&data, dim);
        let code = p.encode(&data[0..dim]);
        assert_eq!(code.len(), (dim + 1) / 2);

        let decoded = p.decode(&code);
        // SQ4 量化误差应小于 1 个量化级
        for d in 0..dim {
            let max_err = p.scale[d];
            assert!(
                (decoded[d] - data[d]).abs() <= max_err,
                "dim {}: decoded={} actual={} scale={}",
                d,
                decoded[d],
                data[d],
                max_err
            );
        }
    }

    #[test]
    fn sq4_distance_matches_scalar() {
        if !is_sq4_avx2_supported() {
            eprintln!("AVX2 not supported, skipping SIMD test");
            return;
        }
        let dim = 128;
        let data: Vec<f32> = (0..256)
            .flat_map(|i| (0..dim).map(move |d| (i * dim + d) as f32 * 0.01))
            .collect();
        let ds = SQ4Dataset::build(&data, dim);

        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.02).sin()).collect();
        let query_code = ds.params.encode(&query);

        for idx in [0, 1, 50, 100, 255].iter() {
            let simd = l2_sq4_raw(&query_code, ds.code(*idx));
            let scalar = l2_sq4_raw_scalar(&query_code, ds.code(*idx));
            let rel_err = (simd - scalar).abs() / scalar.max(1.0);
            assert!(
                rel_err < 1e-5,
                "idx={}: simd={} scalar={} rel_err={}",
                idx,
                simd,
                scalar,
                rel_err
            );
        }
    }

    #[test]
    fn sq4_memory_compression() {
        let dim = 128;
        let n = 1000;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.001).collect();
        let ds = SQ4Dataset::build(&data, dim);

        let f32_bytes = n * dim * 4;
        let sq4_bytes = ds.codes.len();
        let compression = f32_bytes as f64 / sq4_bytes as f64;

        // 128 dims → 64 bytes per vector → 8x compression vs f32
        assert_eq!(sq4_bytes, n * ((dim + 1) / 2));
        assert!((compression - 8.0).abs() < 0.1, "compression = {}", compression);
    }

    #[test]
    fn sq4_distance_correlation_with_f32() {
        let dim = 128;
        let n = 1000;
        let data: Vec<f32> = (0..n * dim)
            .map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5)
            .collect();
        let ds = SQ4Dataset::build(&data, dim);

        let query: Vec<f32> = (0..dim)
            .map(|d| (d as f32 * 0.1).cos() * 0.5 + 0.5)
            .collect();
        let query_code = ds.params.encode(&query);

        // Compute SQ4 and f32 distances for first 200 vectors
        let mut sq4_dists: Vec<(usize, f32)> = Vec::new();
        let mut f32_dists: Vec<(usize, f32)> = Vec::new();
        for i in 0..200.min(n) {
            let d_sq4 = l2_sq4_raw(&query_code, ds.code(i));
            let d_f32: f32 = (0..dim)
                .map(|d| {
                    let diff = data[i * dim + d] - query[d];
                    diff * diff
                })
                .sum();
            sq4_dists.push((i, d_sq4));
            f32_dists.push((i, d_f32));
        }

        // Sort by distance
        sq4_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        f32_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Top-10 overlap should be ≥ 7 (SQ4 is approximate but per-dim, much better than PQ4)
        let sq4_top10: std::collections::HashSet<usize> =
            sq4_dists.iter().take(10).map(|(i, _)| *i).collect();
        let f32_top10: std::collections::HashSet<usize> =
            f32_dists.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = sq4_top10.intersection(&f32_top10).count();
        assert!(
            overlap >= 7,
            "SQ4 top-10 overlap with f32: {}/10 (expected ≥7)",
            overlap
        );
    }
}
