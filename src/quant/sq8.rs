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

        Self { min, max, scale, offset, scale_sq, dim }
    }

    /// 编码单个向量 → u8 codes
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim);
        (0..self.dim)
            .map(|d| {
                let q = ((v[d] - self.offset[d]) / self.scale[d]).round();
                q.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// 编码整个数据集 → 扁平 u8 数组 (n × dim)
    pub fn encode_all(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len() / self.dim;
        let mut codes = vec![0u8; n * self.dim];
        for i in 0..n {
            let row = &data[i * self.dim..(i + 1) * self.dim];
            for d in 0..self.dim {
                let q = ((row[d] - self.offset[d]) / self.scale[d]).round();
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

/// SQ8 L2 距离（AVX2）：每次处理 16 个维度
///
/// 计算 d = Σ (qa[i] - qb[i])² × scale_sq[i]
///
/// 流水线：
/// 1. _mm_loadu_si128: 加载 16 个 u8
/// 2. _mm_unpacklo/hi_epi8(zero): 拆分为 2×8 个 i16
/// 3. _mm_sub_epi16: 差值
/// 4. _mm_madd_epi16(diff, diff): 每对 i16 的平方和 → 4 个 i32
/// 5. _mm_cvtepi32_ps: 转 f32
/// 6. _mm_mul_ps + _mm_add_ps: 加权累加
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn l2_sq8_avx2(qa: &[u8], qb: &[u8], scale_sq: &[f32]) -> f32 {
    let n = qa.len();
    debug_assert_eq!(qb.len(), n);
    debug_assert_eq!(scale_sq.len(), n);

    let zero = _mm_setzero_si128();
    let mut acc_lo = _mm_setzero_ps(); // 前 4 个 f32 累加器
    let mut acc_hi = _mm_setzero_ps(); // 后 4 个 f32 累加器

    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;

        // 加载 16 u8
        let a = _mm_loadu_si128(qa.as_ptr().add(base) as *const __m128i);
        let b = _mm_loadu_si128(qb.as_ptr().add(base) as *const __m128i);

        // 拆分为 i16 (low 8, high 8)
        let a_lo = _mm_unpacklo_epi8(a, zero);
        let a_hi = _mm_unpackhi_epi8(a, zero);
        let b_lo = _mm_unpacklo_epi8(b, zero);
        let b_hi = _mm_unpackhi_epi8(b, zero);

        // 差值 (i16, 范围 [-255, 255]，不会溢出)
        let d_lo = _mm_sub_epi16(a_lo, b_lo);
        let d_hi = _mm_sub_epi16(a_hi, b_hi);

        // 平方和：madd(diff, diff) = diff[2i]² + diff[2i+1]² → 4 个 i32
        let sq_lo = _mm_madd_epi16(d_lo, d_lo);
        let sq_hi = _mm_madd_epi16(d_hi, d_hi);

        // 转 f32
        let sq_lo_f = _mm_cvtepi32_ps(sq_lo);
        let sq_hi_f = _mm_cvtepi32_ps(sq_hi);

        // 加载 scale_sq (4 + 4 = 8 个 f32)
        let sc_lo = _mm_loadu_ps(scale_sq.as_ptr().add(base));
        let sc_hi = _mm_loadu_ps(scale_sq.as_ptr().add(base + 4));

        // 加权累加
        acc_lo = _mm_fmadd_ps(sq_lo_f, sc_lo, acc_lo);
        acc_hi = _mm_fmadd_ps(sq_hi_f, sc_hi, acc_hi);
    }

    // 合并两组累加器
    let mut result = horizontal_sum_128(acc_lo) + horizontal_sum_128(acc_hi);

    // 尾部标量处理
    let remainder_start = chunks * 16;
    for i in remainder_start..n {
        let diff = qa[i] as i32 - qb[i] as i32;
        result += (diff * diff) as f32 * scale_sq[i];
    }

    result
}

/// __m128 水平求和（4 个 f32 → 1 个 f32）
#[inline(always)]
unsafe fn horizontal_sum_128(v: __m128) -> f32 {
    // [a, b, c, d] → [a+b, c+d, _, _]
    let shuf = _mm_movehdup_ps(v); // [b, b, d, d]
    let sums = _mm_add_ps(v, shuf); // [a+b, _, c+d, _]
    let shuf2 = _mm_movehl_ps(v, sums); // [c+d, _, _, _]
    let sums2 = _mm_add_ss(sums, shuf2); // [a+b+c+d, _, _, _]
    _mm_cvtss_f32(sums2)
}

/// 统一分发：AVX2 → 标量
#[inline(always)]
pub fn l2_sq8(qa: &[u8], qb: &[u8], scale_sq: &[f32]) -> f32 {
    if is_sq8_avx2_supported() {
        unsafe { l2_sq8_avx2(qa, qb, scale_sq) }
    } else {
        l2_sq8_scalar(qa, qb, scale_sq)
    }
}

/// 标量 fallback
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

    /// 计算查询 u8 code 与第 idx 个向量之间的 SQ8 L2 距离
    #[inline(always)]
    pub fn distance(&self, query_code: &[u8], idx: usize) -> f32 {
        l2_sq8(query_code, self.code(idx), &self.params.scale_sq)
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
