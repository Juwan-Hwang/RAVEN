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

#[cfg(target_arch = "x86_64")]
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
    /// R42c LP：投影矩阵（m×原维 行主序；空 = 不投影）。放在量化参数里而不是 GraphSearcher 上，
    /// 使原路径 GraphSearcher 布局与 主线 完全相同（R43 接力-2：ARM 上 r42b 多一个 24 B 字段即 −4%）。
    pub proj: Vec<f32>,
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

        Self { min, max, scale, inv_scale, neg_min_div, dim, proj: Vec::new() }
    }

    /// H17：稳健量化范围——每维取 `[clip, 1-clip]` 分位数作为 min/max（超出范围的值在编码时被 clamp 到 0/15）。
    /// `clip = 0` 等价于 [`SQ4Params::fit`]。分位数在至多 200K 行的确定性等距样本上计算。
    /// 动机：glove-100 归一化后各维有长尾，min/max 让 16 级大多浪费在尾部；
    /// clip=0.005 时穷举 SQ4 + 精排 top-20 的 recall 0.852→0.964，sift 0.958→0.970。
    pub fn fit_clipped(data: &[f32], dim: usize, clip: f32) -> Self {
        if clip <= 0.0 {
            return Self::fit(data, dim);
        }
        assert!(clip < 0.5, "clip must be < 0.5");
        let n = data.len() / dim;
        assert!(n > 0, "SQ4Params::fit_clipped: empty data");
        let step = (n / 200_000).max(1);
        let rows: Vec<usize> = (0..n).step_by(step).collect();
        let m = rows.len();
        let lo_i = ((m as f64 - 1.0) * clip as f64).round() as usize;
        let hi_i = ((m as f64 - 1.0) * (1.0 - clip as f64)).round() as usize;
        let mut col = vec![0f32; m];
        let mut min = vec![0f32; dim];
        let mut max = vec![0f32; dim];
        for d in 0..dim {
            for (j, &i) in rows.iter().enumerate() {
                col[j] = data[i * dim + d];
            }
            col.sort_by(|a, b| a.total_cmp(b));
            min[d] = col[lo_i];
            max[d] = col[hi_i];
            if max[d] <= min[d] {
                max[d] = min[d] + 1.0;
            }
        }
        let scale: Vec<f32> = (0..dim).map(|d| (max[d] - min[d]) / 15.0).collect();
        let inv_scale: Vec<f32> = scale.iter().map(|s| 1.0 / s).collect();
        let neg_min_div: Vec<f32> = (0..dim).map(|d| -min[d] * inv_scale[d]).collect();
        Self { min, max, scale, inv_scale, neg_min_div, dim, proj: Vec::new() }
    }

    /// H17b：逐维自适应量化范围——在候选分位数 {0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}（对称截尾）中，
    /// 选使该维 4-bit 均匀量化**重建 MSE** 最小者（样本：至多 200K 行等距抽样）。
    /// 稀疏/长尾的"信号维"（如 fashion-mnist 像素）会自动保留 min/max，稠密带离群值的维（glove、sift）会截尾。
    pub fn fit_mse(data: &[f32], dim: usize) -> Self {
        let n = data.len() / dim;
        assert!(n > 0, "SQ4Params::fit_mse: empty data");
        let step = (n / 200_000).max(1);
        let rows: Vec<usize> = (0..n).step_by(step).collect();
        let m = rows.len();
        const CLIPS: [f64; 8] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2];
        let mut col = vec![0f32; m];
        let mut min = vec![0f32; dim];
        let mut max = vec![0f32; dim];
        for d in 0..dim {
            for (j, &i) in rows.iter().enumerate() {
                col[j] = data[i * dim + d];
            }
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let mut best = (f64::MAX, sorted[0], sorted[m - 1]);
            for &c in CLIPS.iter() {
                let lo = sorted[((m as f64 - 1.0) * c).round() as usize];
                let mut hi = sorted[((m as f64 - 1.0) * (1.0 - c)).round() as usize];
                if hi <= lo { hi = lo + 1.0; }
                let sc = (hi - lo) / 15.0;
                let inv = 1.0 / sc;
                let mut err = 0f64;
                for &x in col.iter() {
                    let q = ((x - lo) * inv).round().clamp(0.0, 15.0);
                    let r = q.mul_add(sc, lo);
                    let e = (x - r) as f64;
                    err += e * e;
                }
                if err < best.0 { best = (err, lo, hi); }
            }
            min[d] = best.1;
            max[d] = best.2;
        }
        let scale: Vec<f32> = (0..dim).map(|d| (max[d] - min[d]) / 15.0).collect();
        let inv_scale: Vec<f32> = scale.iter().map(|s| 1.0 / s).collect();
        let neg_min_div: Vec<f32> = (0..dim).map(|d| -min[d] * inv_scale[d]).collect();
        Self { min, max, scale, inv_scale, neg_min_div, dim, proj: Vec::new() }
    }

    /// H17 统一入口：clip > 0 → 固定分位数截尾；clip < 0 → 逐维 MSE 自适应（[`SQ4Params::fit_mse`]）；0 → min/max。
    pub fn fit_range(data: &[f32], dim: usize, clip: f32) -> Self {
        if clip < 0.0 { Self::fit_mse(data, dim) } else { Self::fit_clipped(data, dim, clip) }
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

    /// R53：编码查询并登记越界修正（码空间维 = self.dim）。
    pub fn encode_query_into(&self, v: &[f32], buf: &mut [u8]) {
        self.encode_into(v, buf);
        if !cfg!(feature = "qoor") || !qoor_enabled() { return; }
        QOOR.with(|c| { let st = unsafe { &mut *c.get() };
            st.n = 0; st.c0 = 0.0; st.qptr = buf.as_ptr() as usize;
            for d in 0..self.dim {
                let t = v[d].mul_add(self.inv_scale[d], self.neg_min_div[d]);
                let (e, hi) = if t < -0.5 { (-t, false) } else if t > 15.5 { (t - 15.0, true) } else { continue };
                if st.n == QOOR_MAX { break; }
                // 截断码 = 0 或 15；真实 t 相对截断点偏 e
                let i = st.n; st.dim[i] = d as u16; st.a[i] = 2.0 * e; st.hi[i] = hi; st.c0 += e * e; st.n += 1;
            }
            if st.n > 0 { QOOR_ANY.store(true, std::sync::atomic::Ordering::Relaxed); }
            if qoor_dbg() { let q = QD_Q.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1; QD_N.fetch_add(st.n as u64, std::sync::atomic::Ordering::Relaxed);
                if q % 50 == 0 { eprintln!("[qoor] q={} avg_oor_dims={:.1} calls={} hits={} dim={} c0={:.1}", q, QD_N.load(std::sync::atomic::Ordering::Relaxed) as f64 / q as f64, QD_CALL.load(std::sync::atomic::Ordering::Relaxed), QD_HIT.load(std::sync::atomic::Ordering::Relaxed), self.dim, st.c0); } }
        });
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
#[cfg(target_arch = "x86_64")]
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

    // H76：剩余 ≥16 字节时先用 128 位 SSSE3 处理一段（GloVe-100：50B = 32+16+2，原先 18B 全走标量）
    let mut remainder_start = chunks * 32;
    if n - remainder_start >= 16 && xtail() {
        let m = _mm_set1_epi8(0x0F);
        let a = _mm_loadu_si128(qa.as_ptr().add(remainder_start) as *const __m128i);
        let b = _mm_loadu_si128(qb.as_ptr().add(remainder_start) as *const __m128i);
        let dl = _mm_abs_epi8(_mm_sub_epi8(_mm_and_si128(a, m), _mm_and_si128(b, m)));
        let dh = _mm_abs_epi8(_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(a, 4), m), _mm_and_si128(_mm_srli_epi16(b, 4), m)));
        let sq = _mm_add_epi16(_mm_maddubs_epi16(dl, dl), _mm_maddubs_epi16(dh, dh));
        let s32 = _mm_madd_epi16(sq, _mm_set1_epi16(1));
        let s64 = _mm_add_epi32(s32, _mm_shuffle_epi32::<0b01_00_11_10>(s32));
        let s = _mm_add_epi32(s64, _mm_shuffle_epi32::<0b10_11_00_01>(s64));
        result += _mm_cvtsi128_si32(s) as f32;
        remainder_start += 16;
    }
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

/// H78：AVX2 提前放弃——每 32B（64 维）一段，段后按 check_mask 第 (2k+1) 位与 thr[2k+1] 检查（与标量 16B 分段的检查点对齐）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn l2_sq4_raw_bounded_avx2(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    let n = qa.len();
    let mask_0f = _mm256_set1_epi8(0x0F);
    let chunks = n / 32;
    let mut sum = 0i32;
    for c in 0..chunks {
        let base = c * 32;
        let a = _mm256_loadu_si256(qa.as_ptr().add(base) as *const __m256i);
        let b = _mm256_loadu_si256(qb.as_ptr().add(base) as *const __m256i);
        let d_lo = _mm256_abs_epi8(_mm256_sub_epi8(_mm256_and_si256(a, mask_0f), _mm256_and_si256(b, mask_0f)));
        let d_hi = _mm256_abs_epi8(_mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(a, 4), mask_0f), _mm256_and_si256(_mm256_srli_epi16(b, 4), mask_0f)));
        let acc = _mm256_add_epi16(_mm256_maddubs_epi16(d_lo, d_lo), _mm256_maddubs_epi16(d_hi, d_hi));
        sum += horizontal_sum_i16_256(acc);
        let cp = 2 * c + 1; // 第 cp 个 16B 段之后
        if base + 32 < n && cp < MAX_CHECKPOINTS && (check_mask >> cp) & 1 == 1 && sum as f32 >= thr[cp] {
            return (sum as f32, ((c + 1) * 64) as u32);
        }
    }
    let rest = chunks * 32;
    if rest < n {
        // 尾部复用全长内核的尾部逻辑：直接对剩余字节算
        sum += l2_sq4_raw_avx2(&qa[rest..], &qb[rest..]) as i32;
    }
    (sum as f32, (n * 2) as u32)
}

/// __m256i 水平求和（16 × i16 → 1 个 i32）
#[cfg(target_arch = "x86_64")]
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

/// SQ4 无加权 L2 距离（NEON）：每次 16 bytes = 32 维
///
/// 1. vandq/vshrq 拆 low/high nibble（各 16 × u8，[0,15]）
/// 2. vabdq_u8 绝对差
/// 3. dotprod：udot 自点积累加到 u32；否则 vmull_u8 + vpadalq_u16
///
/// SIFT-128（64 bytes）= 4 次迭代，双累加器
///
/// # Safety
/// `qa.len() == qb.len()`
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn l2_sq4_raw_neon(qa: &[u8], qb: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let n = qa.len();
    debug_assert_eq!(qb.len(), n);
    let pa = qa.as_ptr();
    let pb = qb.as_ptr();
    let mask = vdupq_n_u8(0x0F);
    let mut acc0 = vdupq_n_u32(0);
    let mut acc1 = vdupq_n_u32(0);
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let a = vld1q_u8(pa.add(base));
        let b = vld1q_u8(pb.add(base));
        let d_lo = vabdq_u8(vandq_u8(a, mask), vandq_u8(b, mask));
        let d_hi = vabdq_u8(vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(b));
        #[cfg(target_feature = "dotprod")]
        {
            acc0 = vdotq_u32(acc0, d_lo, d_lo);
            acc1 = vdotq_u32(acc1, d_hi, d_hi);
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            acc0 = vpadalq_u16(acc0, vmull_u8(vget_low_u8(d_lo), vget_low_u8(d_lo)));
            acc0 = vpadalq_u16(acc0, vmull_high_u8(d_lo, d_lo));
            acc1 = vpadalq_u16(acc1, vmull_u8(vget_low_u8(d_hi), vget_low_u8(d_hi)));
            acc1 = vpadalq_u16(acc1, vmull_high_u8(d_hi, d_hi));
        }
    }
    let mut result = vaddvq_u32(vaddq_u32(acc0, acc1)) as i32;
    for i in chunks * 16..n {
        let d_lo = (qa[i] & 0x0F) as i32 - (qb[i] & 0x0F) as i32;
        let d_hi = (qa[i] >> 4) as i32 - (qb[i] >> 4) as i32;
        result += d_lo * d_lo + d_hi * d_hi;
    }
    result as f32
}

/// H130c（第 12 轮）：64B 码专用核，查询侧解码（and/shr）提到每查询一次、常驻 8 个寄存器；结果与 l2_sq4_raw_neon 逐位一致。
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct QDec64 { lo: [std::arch::aarch64::uint8x16_t; 4], hi: [std::arch::aarch64::uint8x16_t; 4] }
#[cfg(target_arch = "aarch64")]
impl QDec64 {
    #[inline(always)]
    pub fn new(q: &[u8]) -> Self {
        use std::arch::aarch64::*;
        debug_assert!(q.len() >= 64);
        unsafe {
            let mask = vdupq_n_u8(0x0F);
            let mut lo = [vdupq_n_u8(0); 4]; let mut hi = [vdupq_n_u8(0); 4];
            for c in 0..4 { let a = vld1q_u8(q.as_ptr().add(c * 16)); lo[c] = vandq_u8(a, mask); hi[c] = vshrq_n_u8::<4>(a); }
            QDec64 { lo, hi }
        }
    }
    #[inline(always)]
    pub unsafe fn dist(&self, code: *const u8) -> f32 {
        use std::arch::aarch64::*;
        let mask = vdupq_n_u8(0x0F);
        let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0));
        let b0 = vld1q_u8(code); let b1 = vld1q_u8(code.add(16)); let b2 = vld1q_u8(code.add(32)); let b3 = vld1q_u8(code.add(48));
        let l0 = vabdq_u8(self.lo[0], vandq_u8(b0, mask)); let h0 = vabdq_u8(self.hi[0], vshrq_n_u8::<4>(b0));
        let l1 = vabdq_u8(self.lo[1], vandq_u8(b1, mask)); let h1 = vabdq_u8(self.hi[1], vshrq_n_u8::<4>(b1));
        let l2 = vabdq_u8(self.lo[2], vandq_u8(b2, mask)); let h2 = vabdq_u8(self.hi[2], vshrq_n_u8::<4>(b2));
        let l3 = vabdq_u8(self.lo[3], vandq_u8(b3, mask)); let h3 = vabdq_u8(self.hi[3], vshrq_n_u8::<4>(b3));
        #[cfg(target_feature = "dotprod")]
        {
            a0 = vdotq_u32(a0, l0, l0); a1 = vdotq_u32(a1, h0, h0);
            a2 = vdotq_u32(a2, l1, l1); a3 = vdotq_u32(a3, h1, h1);
            a0 = vdotq_u32(a0, l2, l2); a1 = vdotq_u32(a1, h2, h2);
            a2 = vdotq_u32(a2, l3, l3); a3 = vdotq_u32(a3, h3, h3);
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            for (acc, d) in [(&mut a0, l0), (&mut a1, h0), (&mut a2, l1), (&mut a3, h1), (&mut a0, l2), (&mut a1, h2), (&mut a2, l3), (&mut a3, h3)] {
                *acc = vpadalq_u16(*acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
                *acc = vpadalq_u16(*acc, vmull_high_u8(d, d));
            }
        }
        crate::diag_count!(QUANT_DIST);
        vaddvq_u32(vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3))) as f32
    }
}

/// 非 x86 平台占位：AVX2 检测恒为 false，热路径不会选到这里
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn l2_sq4_raw_avx2(qa: &[u8], qb: &[u8]) -> f32 {
    l2_sq4_raw(qa, qb)
}


// ─── R53 QOOR：查询越界稀疏修正（Query Out-Of-Range）────────────────────
// 查询码被截断到 [0,15] 时，真实码空间距离 (t-c)^2 = 截断距离 + 修正：
//   t<0 ：2|t|·c + t^2            t>15：2(t-15)·(15-c) + (t-15)^2
// 修正恒 ≥0，故分段提前放弃（部分和为下界）仍无损。只对越界维（稀疏）逐维加；
// 同分布数据集无越界维 → 全局开关恒 false，热路径只多一次 relaxed 读。
// R56：QOOR 默认关（RAVEN_QOOR=1 开），保证未启用档召回与 主线 逐位一致（门控 a）。
pub static QOOR_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub const QOOR_MAX: usize = 256;
pub static QD_CALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static QD_HIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static QD_Q: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static QD_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn qoor_dbg() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("RAVEN_QOOR_DBG").is_ok()) }
pub struct QoorState { pub qptr: usize, pub n: usize, pub dim: [u16; QOOR_MAX], pub a: [f32; QOOR_MAX], pub hi: [bool; QOOR_MAX], pub c0: f32 }
thread_local! { pub static QOOR: std::cell::UnsafeCell<QoorState> = const { std::cell::UnsafeCell::new(QoorState { qptr: 0, n: 0, dim: [0; QOOR_MAX], a: [0.0; QOOR_MAX], hi: [false; QOOR_MAX], c0: 0.0 }) }; }
pub fn qoor_enabled() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("RAVEN_QOOR").map(|v| v != "0").unwrap_or(false)) }
/// 若 qa 是当前线程登记的查询码，返回对 qb 的越界修正；否则 0。
#[inline(always)]
pub fn qoor_corr(qa: &[u8], qb: &[u8]) -> f32 {
    // R59：QOOR 编译期隔离（cargo feature "qoor"，缺省关）→ 缺省构建热路径与 主线 指令级相同（R58a x86 (a) 测得运行期开关 −2.6%/−4.4%）
    #[cfg(not(feature = "qoor"))]
    { let _ = (qa, qb); return 0.0; }
    #[cfg(feature = "qoor")]
    if !QOOR_ANY.load(std::sync::atomic::Ordering::Relaxed) { return 0.0; }
    qoor_corr_slow(qa.as_ptr() as usize, qb)
}
#[inline(never)]
fn qoor_corr_slow(qp: usize, qb: &[u8]) -> f32 {
    QOOR.with(|c| { let st = unsafe { &*c.get() };
        if qoor_dbg() { QD_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed); if st.qptr == qp { QD_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed); } }
        if st.n == 0 || st.qptr != qp { return 0.0; }
        let mut s = st.c0;
        for i in 0..st.n { let d = st.dim[i] as usize; let b = qb[d >> 1]; let code = if d & 1 == 0 { b & 15 } else { b >> 4 } as f32;
            s += st.a[i] * if st.hi[i] { 15.0 - code } else { code }; }
        s })
}

/// 统一分发：AVX2 → 标量
#[inline(always)]
pub fn l2_sq4_raw(qa: &[u8], qb: &[u8]) -> f32 { l2_sq4_raw_k(qa, qb) + qoor_corr(qa, qb) }
#[inline(always)]
fn l2_sq4_raw_k(qa: &[u8], qb: &[u8]) -> f32 {
    crate::diag_count!(QUANT_DIST);
    #[cfg(target_feature = "avx2")]
    {
        unsafe { l2_sq4_raw_avx2(qa, qb) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { l2_sq4_raw_avx2(qa, qb) }
        } else {
            l2_sq4_raw_scalar(qa, qb)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { l2_sq4_raw_neon(qa, qb) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        l2_sq4_raw_scalar(qa, qb)
    }
}

/// H07/H14：分段检查点上限（`thr` 至少 8 个元素；第 8 段之后不再检查，避免高维（d≥288 / gran16 d≥144）
/// 时 `check_mask >> c` 移位溢出（release 下按 32 取模）导致越界读 `thr`）。
pub const MAX_CHECKPOINTS: usize = 8;

/// H73：x86 上提前放弃内核此前是标量（全部开发在 ARM/NEON 上）。xsimd=1 时改用 AVX2 全长距离：
/// 只少了"部分和超阈值即返回"，而被放弃者本就会被 pool 拒绝，结果逐位等价。RAVEN_XSIMD=0 回退。
static XSIMD: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
#[inline(always)]
pub fn xsimd() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let v = XSIMD.load(Relaxed);
    if v != 2 { return v == 1; }
    let on = std::env::var("RAVEN_XSIMD").map(|x| x != "0").unwrap_or(true);
    XSIMD.store(on as u8, Relaxed);
    on
}
static XTAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[inline(always)]
pub fn xtail() -> bool { XTAIL.load(std::sync::atomic::Ordering::Relaxed) }
pub fn set_xtail(on: bool) { XTAIL.store(on, std::sync::atomic::Ordering::Relaxed) }
static XAB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline(always)]
pub fn xab() -> bool { XAB.load(std::sync::atomic::Ordering::Relaxed) }
pub fn set_xab(on: bool) { XAB.store(on, std::sync::atomic::Ordering::Relaxed) }
pub fn set_xsimd(on: bool) { XSIMD.store(on as u8, std::sync::atomic::Ordering::Relaxed); }

/// H07：带分段阈值的 SQ4 距离（提前放弃 / early abandon）。
///
/// 码按 16 字节（32 维）分段累加；第 c 段（c < 段数-1）结束后若 `check_mask` 第 c 位为 1
/// 且部分和 `>= thr[c]`，立即返回 `(部分和, 已处理维度数)`，否则算完返回 `(完整距离, dim)`。
/// 部分和是完整距离的下界（平方和非负），故当 `thr[c]` = 候选池最差距离时剪枝**无损**。
///
/// 计数：完整计算计入 `QUANT_DIST`；提前放弃计入 `QUANT_PARTIAL` 与 `QUANT_PARTIAL_DIMS`。
/// `thr.len()` 需 ≥ 段数-1（SIFT-128：64 字节 = 4 段，检查点在 32/64/96 维）。
#[inline(always)]
pub fn l2_sq4_raw_bounded(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    let n = qa.len();
    debug_assert_eq!(qb.len(), n);
    let chunks = n / 16;
    #[cfg(target_arch = "aarch64")]
    let r = unsafe { l2_sq4_raw_bounded_neon(qa, qb, thr, check_mask) };
    #[cfg(not(target_arch = "aarch64"))]
    let r = if xsimd() { if xab() { unsafe { l2_sq4_raw_bounded_avx2(qa, qb, thr, check_mask) } } else { (unsafe { l2_sq4_raw_avx2(qa, qb) }, (n * 2) as u32) } } else { l2_sq4_raw_bounded_scalar(qa, qb, thr, check_mask) };
    let full_dims = (n * 2) as u32;
    if r.1 < full_dims {
        crate::diag_count!(QUANT_PARTIAL);
        crate::diag_count_add!(QUANT_PARTIAL_DIMS, r.1);
    } else {
        crate::diag_count!(QUANT_DIST);
    }
    let _ = chunks;
    if r.1 >= full_dims { (r.0 + qoor_corr(qa, qb), r.1) } else { r }
}

/// H07：同 [`l2_sq4_raw_bounded`]，但按 8 字节（16 维）分段，检查点在 16/32/…/112 维（bit c = 第 c 段后）。
#[inline(always)]
pub fn l2_sq4_raw_bounded16(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    let n = qa.len();
    debug_assert_eq!(qb.len(), n);
    #[cfg(target_arch = "aarch64")]
    let r = unsafe { l2_sq4_raw_bounded16_neon(qa, qb, thr, check_mask) };
    #[cfg(not(target_arch = "aarch64"))]
    let r = if xsimd() { (unsafe { l2_sq4_raw_avx2(qa, qb) }, (n * 2) as u32) } else { l2_sq4_raw_bounded_scalar_step(qa, qb, thr, check_mask, 8) };
    if r.1 < (n * 2) as u32 {
        crate::diag_count!(QUANT_PARTIAL);
        crate::diag_count_add!(QUANT_PARTIAL_DIMS, r.1);
    } else {
        crate::diag_count!(QUANT_DIST);
        return (r.0 + qoor_corr(qa, qb), r.1);
    }
    r
}

/// H07 分段距离（标量，任意段长 `step` 字节）
#[allow(dead_code)]
#[inline(always)]
fn l2_sq4_raw_bounded_scalar_step(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32, step: usize) -> (f32, u32) {
    let n = qa.len();
    let chunks = n / step;
    let mut sum = 0i32;
    for c in 0..chunks {
        sum += l2_sq4_raw_scalar(&qa[c * step..(c + 1) * step], &qb[c * step..(c + 1) * step]) as i32;
        if c + 1 < chunks && c < MAX_CHECKPOINTS && (check_mask >> c) & 1 == 1 && sum as f32 >= thr[c] {
            return (sum as f32, ((c + 1) * step * 2) as u32);
        }
    }
    if chunks * step < n {
        sum += l2_sq4_raw_scalar(&qa[chunks * step..], &qb[chunks * step..]) as i32;
    }
    (sum as f32, (n * 2) as u32)
}

/// H07 16 维分段距离（NEON，64 位 D 寄存器）
///
/// # Safety
/// `qa.len() == qb.len()`，`thr.len() >= qa.len()/8 - 1`
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l2_sq4_raw_bounded16_neon(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    use std::arch::aarch64::*;
    let n = qa.len();
    let pa = qa.as_ptr();
    let pb = qb.as_ptr();
    let mask = vdup_n_u8(0x0F);
    let mut acc = vdup_n_u32(0);
    let chunks = n / 8;
    for c in 0..chunks {
        let a = vld1_u8(pa.add(c * 8));
        let b = vld1_u8(pb.add(c * 8));
        let d_lo = vabd_u8(vand_u8(a, mask), vand_u8(b, mask));
        let d_hi = vabd_u8(vshr_n_u8::<4>(a), vshr_n_u8::<4>(b));
        #[cfg(target_feature = "dotprod")]
        {
            acc = vdot_u32(acc, d_lo, d_lo);
            acc = vdot_u32(acc, d_hi, d_hi);
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            let w = vaddq_u16(vmull_u8(d_lo, d_lo), vmull_u8(d_hi, d_hi));
            acc = vadd_u32(acc, vget_low_u32(vpaddlq_u16(w)));
            acc = vadd_u32(acc, vget_high_u32(vpaddlq_u16(w)));
        }
        if c + 1 < chunks && c < MAX_CHECKPOINTS && (check_mask >> c) & 1 == 1 {
            let s = vaddv_u32(acc) as f32;
            if s >= *thr.get_unchecked(c) {
                return (s, ((c + 1) * 16) as u32);
            }
        }
    }
    let mut result = vaddv_u32(acc) as i32;
    for i in chunks * 8..n {
        let d_lo = (qa[i] & 0x0F) as i32 - (qb[i] & 0x0F) as i32;
        let d_hi = (qa[i] >> 4) as i32 - (qb[i] >> 4) as i32;
        result += d_lo * d_lo + d_hi * d_hi;
    }
    (result as f32, (n * 2) as u32)
}

/// H07 分段距离（标量 / 非 aarch64 回退，语义与 NEON 版一致）
#[allow(dead_code)]
#[inline(always)]
fn l2_sq4_raw_bounded_scalar(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    let n = qa.len();
    let chunks = n / 16;
    let mut sum = 0i32;
    for c in 0..chunks {
        sum += l2_sq4_raw_scalar(&qa[c * 16..c * 16 + 16], &qb[c * 16..c * 16 + 16]) as i32;
        if c + 1 < chunks && c < MAX_CHECKPOINTS && (check_mask >> c) & 1 == 1 && sum as f32 >= thr[c] {
            return (sum as f32, ((c + 1) * 32) as u32);
        }
    }
    if chunks * 16 < n {
        sum += l2_sq4_raw_scalar(&qa[chunks * 16..], &qb[chunks * 16..]) as i32;
    }
    (sum as f32, (n * 2) as u32)
}

/// H07 分段距离（NEON）
///
/// # Safety
/// `qa.len() == qb.len()`，`thr.len() >= qa.len()/16 - 1`

/// H35：查询预拆分版分段距离。`qs` = 查询码按 16B 块拆成 [lo(16B), hi(16B)] 交错（长度 2n），
/// 省去每次调用对查询做 and/shift；结果与 [`l2_sq4_raw_bounded`] 逐位一致（要求 n % 16 == 0 时无尾）。
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn l2_sq4_bounded_split_neon(qs: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    use std::arch::aarch64::*;
    let n = qb.len();
    let pq = qs.as_ptr();
    let pb = qb.as_ptr();
    let mask = vdupq_n_u8(0x0F);
    let mut acc0 = vdupq_n_u32(0);
    let mut acc1 = vdupq_n_u32(0);
    let chunks = n / 16;
    for c in 0..chunks {
        let b = vld1q_u8(pb.add(c * 16));
        let alo = vld1q_u8(pq.add(c * 32));
        let ahi = vld1q_u8(pq.add(c * 32 + 16));
        let d_lo = vabdq_u8(alo, vandq_u8(b, mask));
        let d_hi = vabdq_u8(ahi, vshrq_n_u8::<4>(b));
        acc0 = vdotq_u32(acc0, d_lo, d_lo);
        acc1 = vdotq_u32(acc1, d_hi, d_hi);
        if c + 1 < chunks && c < MAX_CHECKPOINTS && (check_mask >> c) & 1 == 1 {
            let s = vaddvq_u32(vaddq_u32(acc0, acc1)) as f32;
            if s >= *thr.get_unchecked(c) {
                return (s, ((c + 1) * 32) as u32);
            }
        }
    }
    let mut result = vaddvq_u32(vaddq_u32(acc0, acc1)) as i32;
    for i in chunks * 16..n {
        let a = *qs.get_unchecked(chunks * 32 + (i - chunks * 16));
        let d_lo = (a & 0x0F) as i32 - (qb[i] & 0x0F) as i32;
        let d_hi = (a >> 4) as i32 - (qb[i] >> 4) as i32;
        result += d_lo * d_lo + d_hi * d_hi;
    }
    (result as f32, (n * 2) as u32)
}

/// H35：构造预拆分查询（每 16B 块 → lo16 + hi16；尾部原样附加）
pub fn split_query(q: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let chunks = q.len() / 16;
    for c in 0..chunks {
        out.extend(q[c * 16..c * 16 + 16].iter().map(|&x| x & 0x0F));
        out.extend(q[c * 16..c * 16 + 16].iter().map(|&x| x >> 4));
    }
    out.extend_from_slice(&q[chunks * 16..]);
}

/// H35 开关（RAVEN_QSPLIT，默认 0）
#[inline]
pub fn qsplit_enabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("RAVEN_QSPLIT").map(|v| v == "1").unwrap_or(false))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l2_sq4_raw_bounded_neon(qa: &[u8], qb: &[u8], thr: &[f32], check_mask: u32) -> (f32, u32) {
    use std::arch::aarch64::*;
    let n = qa.len();
    let pa = qa.as_ptr();
    let pb = qb.as_ptr();
    let mask = vdupq_n_u8(0x0F);
    let mut acc0 = vdupq_n_u32(0);
    let mut acc1 = vdupq_n_u32(0);
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let a = vld1q_u8(pa.add(base));
        let b = vld1q_u8(pb.add(base));
        let d_lo = vabdq_u8(vandq_u8(a, mask), vandq_u8(b, mask));
        let d_hi = vabdq_u8(vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(b));
        #[cfg(target_feature = "dotprod")]
        {
            acc0 = vdotq_u32(acc0, d_lo, d_lo);
            acc1 = vdotq_u32(acc1, d_hi, d_hi);
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            acc0 = vpadalq_u16(acc0, vmull_u8(vget_low_u8(d_lo), vget_low_u8(d_lo)));
            acc0 = vpadalq_u16(acc0, vmull_high_u8(d_lo, d_lo));
            acc1 = vpadalq_u16(acc1, vmull_u8(vget_low_u8(d_hi), vget_low_u8(d_hi)));
            acc1 = vpadalq_u16(acc1, vmull_high_u8(d_hi, d_hi));
        }
        if c + 1 < chunks && c < MAX_CHECKPOINTS && (check_mask >> c) & 1 == 1 {
            let s = vaddvq_u32(vaddq_u32(acc0, acc1)) as f32;
            if s >= *thr.get_unchecked(c) {
                return (s, ((c + 1) * 32) as u32);
            }
        }
    }
    let mut result = vaddvq_u32(vaddq_u32(acc0, acc1)) as i32;
    for i in chunks * 16..n {
        let d_lo = (qa[i] & 0x0F) as i32 - (qb[i] & 0x0F) as i32;
        let d_hi = (qa[i] >> 4) as i32 - (qb[i] >> 4) as i32;
        result += d_lo * d_lo + d_hi * d_hi;
    }
    (result as f32, (n * 2) as u32)
}

/// 检测 AVX2 支持
pub fn is_sq4_avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ─── SQ4 量化数据集 ──────────────────────────────────────────────────

/// H42：SQ4 码数组 64B 对齐（见 `crate::memory::aligned`）。
pub type AlignedBytes = crate::memory::aligned::AlignedBuf<u8>;

/// SQ4 量化后的完整数据集（packed codes + params）
pub struct SQ4Dataset {
    /// 扁平 packed 4-bit codes: n × ceil(dim/2) bytes（H42：64B 对齐）
    pub codes: AlignedBytes,
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
        Self { codes: codes.into(), params, dim, n, code_bytes }
    }

    /// H07：按维度置换 `perm` 构建（新第 j 维 = 原第 perm[j] 维）。
    ///
    /// 量化参数仍在原数据上逐维拟合后再置换，故每维的码与 `build` 完全相同，
    /// SQ4 距离（整数平方和）与未置换版本**逐位一致**；只改变分段累加的维度顺序。
    /// 查询需用 [`SQ4Params::encode_into`] 编码**同样置换后**的查询向量。
    pub fn build_permuted(data: &[f32], dim: usize, perm: &[usize]) -> Self {
        Self::build_permuted_clipped(data, dim, perm, 0.0)
    }

    /// H17：同 [`SQ4Dataset::build`]，量化范围用 [`SQ4Params::fit_range`]（clip=0 与 build 完全相同；clip<0 = 逐维 MSE 自适应）。
    pub fn build_clipped(data: &[f32], dim: usize, clip: f32) -> Self {
        let n = data.len() / dim;
        let params = SQ4Params::fit_range(data, dim, clip);
        let codes = params.encode_all(data);
        let code_bytes = dim.div_ceil(2);
        Self { codes: codes.into(), params, dim, n, code_bytes }
    }

    /// R37 LP：全维共用同一步长（码空间 L2 ∝ 真实 L2；投影空间各维方差差异大时必需）。
    /// 步长取第 `rank` 大的逐维步长（0 = 最大，范围更大的维被截断），中心保持逐维 (min+max)/2。
    pub fn build_uniform_clipped(data: &[f32], dim: usize, clip: f32, rank: usize) -> Self {
        let n = data.len() / dim;
        let p0 = SQ4Params::fit_range(data, dim, clip);
        let mut sc = p0.scale.clone();
        sc.sort_by(|a, b| b.total_cmp(a));
        let st = sc[rank.min(dim - 1)].max(1e-12);
        let mut params = p0.clone();
        for d in 0..dim {
            let c = 0.5 * (p0.min[d] + p0.max[d]);
            let mn = c - 7.5 * st;
            params.min[d] = mn; params.max[d] = mn + 15.0 * st;
            params.scale[d] = st; params.inv_scale[d] = 1.0 / st; params.neg_min_div[d] = -mn / st;
        }
        let codes = params.encode_all(data);
        let code_bytes = dim.div_ceil(2);
        Self { codes: codes.into(), params, dim, n, code_bytes }
    }

    /// H17：同 [`SQ4Dataset::build_permuted`]，量化范围用 [`SQ4Params::fit_clipped`]。
    pub fn build_permuted_clipped(data: &[f32], dim: usize, perm: &[usize], clip: f32) -> Self {
        assert_eq!(perm.len(), dim);
        let n = data.len() / dim;
        let p0 = SQ4Params::fit_range(data, dim, clip);
        let pick = |v: &Vec<f32>| -> Vec<f32> { perm.iter().map(|&d| v[d]).collect() };
        let params = SQ4Params {
            min: pick(&p0.min),
            max: pick(&p0.max),
            scale: pick(&p0.scale),
            inv_scale: pick(&p0.inv_scale),
            neg_min_div: pick(&p0.neg_min_div),
            dim,
            proj: Vec::new(),
        };
        let code_bytes = dim.div_ceil(2);
        let mut codes = vec![0u8; n * code_bytes];
        let mut row = vec![0f32; dim];
        for i in 0..n {
            let src = &data[i * dim..(i + 1) * dim];
            for (j, &d) in perm.iter().enumerate() {
                row[j] = src[d];
            }
            params.encode_into(&row, &mut codes[i * code_bytes..(i + 1) * code_bytes]);
        }
        Self { codes: codes.into(), params, dim, n, code_bytes }
    }

    /// H34：把每向量码长补零到 `mult` 字节的整数倍（补位 nibble 恒为 0，查询侧同样补 0，
    /// 故距离逐位一致；消除 NEON 核的标量尾循环，如 GloVe-100 50B → 64B）。
    /// R65：码长补零倍数选择。`small>0` 且原码长 `cb<=small` 时补到 `small`（低维集，如 glove-25 13B→16B），
    /// 否则用 `default`（x86/ARM 缺省 32，H87b/H89/H91）。`small=0` 时恒等于旧行为。
    pub fn pad_mult_for(cb: usize, default: usize, small: usize) -> usize {
        if small > 0 && cb <= small { small } else { default }
    }

    pub fn pad_codes(&mut self, mult: usize) {
        if mult <= 1 { return; }
        let cb = self.code_bytes;
        let nb = cb.div_ceil(mult) * mult;
        if nb == cb { return; }
        let mut out = vec![0u8; self.n * nb];
        for i in 0..self.n {
            out[i * nb..i * nb + cb].copy_from_slice(&self.codes[i * cb..(i + 1) * cb]);
        }
        self.codes = out.into();
        self.code_bytes = nb;
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

// ─── R67b/R68 ADC16（低维非对称导航；新增函数，不改动上面任何已有函数/属性）──────────────
// 注：R67 v1/v2 把这些函数插在 `#[inline(always)]` 与 `fn l2_sq4_raw_k` 之间，使对称核 `l2_sq4_raw_k` 丢了 inline(always)
// （属性挪给了 adc16_dist）——这是 v2 缺省路径代码生成变化的一个具体来源。R68 把它们放到这里。

impl SQ4Params {
    /// R67b ADC16：把（码维序的）f32 查询写成 64×f32 = p[32] | r[32]（p_j = q_j − min_j，r_j = scale_j；j ≥ dim 补 0），
    /// 供 [`adc16_dist`]：Σ (r_j·c_j − p_j)² = Σ (min_j + scale_j·c_j − q_j)²（非对称：数据 4 bit 码，查询不量化）。
    pub fn encode_adc16_into(&self, v: &[f32], buf: &mut Vec<u8>) {
        assert!(self.dim <= 32 && v.len() == self.dim);
        buf.clear(); buf.resize(256, 0);
        let f = buf.as_mut_ptr() as *mut f32;
        for j in 0..self.dim { unsafe { f.add(j).write_unaligned(v[j] - self.min[j]); f.add(32 + j).write_unaligned(self.scale[j]); } }
    }
}

/// R67b ADC16 距离：`qa` = [`SQ4Params::encode_adc16_into`] 的 256 B 查询，`qb` = 16 B 码（32 个 nibble，补位 nibble 为 0 且 r = 0）。
#[inline(always)]
pub fn adc16_dist(qa: &[u8], qb: &[u8]) -> f32 {
    crate::diag_count!(QUANT_DIST);
    debug_assert!(qa.len() >= 256 && qb.len() >= 16);
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    { unsafe { adc16_avx2(qa.as_ptr() as *const f32, qb.as_ptr()) } }
    #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
    { if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { unsafe { adc16_avx2(qa.as_ptr() as *const f32, qb.as_ptr()) } } else { adc16_scalar(qa, qb) } }
    #[cfg(not(target_arch = "x86_64"))]
    { adc16_scalar(qa, qb) }
}

/// ADC16 标量参考核
pub fn adc16_scalar(qa: &[u8], qb: &[u8]) -> f32 {
    let f = qa.as_ptr() as *const f32;
    let mut acc = 0f32;
    for j in 0..32 {
        let c = ((qb[j / 2] >> ((j % 2) * 4)) & 0x0F) as f32;
        let (p, r) = unsafe { (f.add(j).read_unaligned(), f.add(32 + j).read_unaligned()) };
        let t = r * c - p; acc += t * t;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn adc16_avx2(q: *const f32, code: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let b = _mm_loadu_si128(code as *const __m128i);
    let m = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(b, m);
    let hi = _mm_and_si128(_mm_srli_epi16(b, 4), m);
    let d0 = _mm_unpacklo_epi8(lo, hi); // 维 0..15
    let d1 = _mm_unpackhi_epi8(lo, hi); // 维 16..31
    let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(d0));
    let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(d0, 8)));
    let c2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(d1));
    let c3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(d1, 8)));
    let t0 = _mm256_fmsub_ps(c0, _mm256_loadu_ps(q.add(32)), _mm256_loadu_ps(q));
    let t1 = _mm256_fmsub_ps(c1, _mm256_loadu_ps(q.add(40)), _mm256_loadu_ps(q.add(8)));
    let t2 = _mm256_fmsub_ps(c2, _mm256_loadu_ps(q.add(48)), _mm256_loadu_ps(q.add(16)));
    let t3 = _mm256_fmsub_ps(c3, _mm256_loadu_ps(q.add(56)), _mm256_loadu_ps(q.add(24)));
    let a = _mm256_fmadd_ps(t1, t1, _mm256_mul_ps(t0, t0));
    let b2 = _mm256_fmadd_ps(t3, t3, _mm256_mul_ps(t2, t2));
    let v = _mm256_add_ps(a, b2);
    let x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
    _mm_cvtss_f32(x)
}

#[cfg(test)]
mod adc16_tests {
    use super::*;
    #[test]
    fn adc16_matches_decode() {
        let dim = 25; let n = 200;
        let data: Vec<f32> = (0..n * dim).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1.0 + (i % dim) as f32 * 0.1)).collect();
        let mut ds = SQ4Dataset::build(&data, dim); ds.pad_codes(16); assert_eq!(ds.code_bytes, 16);
        let q: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.37).sin() * 1.3).collect();
        let mut qb = Vec::new(); ds.params.encode_adc16_into(&q, &mut qb);
        for i in 0..n {
            let dec = ds.params.decode(&ds.code(i)[..13]);
            let r: f32 = dec.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum();
            let a = adc16_dist(&qb, ds.code(i)); let s = adc16_scalar(&qb, ds.code(i));
            assert!((a - r).abs() <= 1e-3 * r.max(1.0), "i={i} adc={a} ref={r}");
            assert!((s - r).abs() <= 1e-3 * r.max(1.0), "i={i} scalar={s} ref={r}");
        }
    }
}

// ─── 测试 ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    #[cfg(target_arch = "aarch64")]
    fn neon_rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sq4_neon_matches_scalar() {
        for nbytes in [1usize, 7, 15, 16, 17, 32, 48, 63, 64, 65, 100, 480] {
            for seed in 0..5u64 {
                let a = neon_rand_bytes(nbytes, seed * 31 + nbytes as u64);
                let b = neon_rand_bytes(nbytes, seed * 17 + 1000 + nbytes as u64);
                let s = super::l2_sq4_raw_scalar(&a, &b);
                let v = unsafe { super::l2_sq4_raw_neon(&a, &b) };
                assert_eq!(s, v, "nbytes={nbytes} seed={seed}");
                assert_eq!(super::l2_sq4_raw(&a, &b), s);
            }
        }
        let a = vec![0x00u8; 64];
        let b = vec![0xFFu8; 64];
        assert_eq!(unsafe { super::l2_sq4_raw_neon(&a, &b) }, 128.0 * 225.0);
    }
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

    /// H07：分段距离 —— 阈值不触发时等于完整距离；触发时部分和为下界且 >= 阈值
    #[test]
    fn h07_bounded_distance_is_lower_bound() {
        let mut s = 12345u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 };
        for _ in 0..500 {
            let a: Vec<u8> = (0..64).map(|_| rnd()).collect();
            let b: Vec<u8> = (0..64).map(|_| rnd()).collect();
            let full = l2_sq4_raw_scalar(&a, &b);
            let inf = [f32::MAX; 8];
            assert_eq!(l2_sq4_raw_bounded(&a, &b, &inf, 0xFF), (full, 128));
            assert_eq!(l2_sq4_raw_bounded16(&a, &b, &inf, 0xFF), (full, 128));
            let t = (rnd() as f32) * 20.0;
            let thr = [t; 8];
            for (d, dims) in [l2_sq4_raw_bounded(&a, &b, &thr, 0xFF), l2_sq4_raw_bounded16(&a, &b, &thr, 0xFF),
                              l2_sq4_raw_bounded_scalar(&a, &b, &thr, 0xFF), l2_sq4_raw_bounded_scalar_step(&a, &b, &thr, 0xFF, 8)] {
                if dims < 128 {
                    assert!(d >= t && d <= full);
                    let p = dims as usize / 2;
                    assert_eq!(d, l2_sq4_raw_scalar(&a[..p], &b[..p]));
                } else {
                    assert_eq!(d, full);
                }
            }
        }
    }

    /// H07：置换构建的 SQ4 距离与原始逐位一致
    #[test]
    fn h07_permuted_build_same_distance() {
        let (n, dim) = (200usize, 32usize);
        let data: Vec<f32> = (0..n * dim).map(|i| ((i * 7919) % 1000) as f32 * 0.37).collect();
        let perm: Vec<usize> = (0..dim).map(|j| (j * 5) % dim).collect();
        let a = SQ4Dataset::build(&data, dim);
        let b = SQ4Dataset::build_permuted(&data, dim, &perm);
        let q: Vec<f32> = (0..dim).map(|d| d as f32 * 11.0).collect();
        let qp: Vec<f32> = perm.iter().map(|&d| q[d]).collect();
        let ca = a.params.encode(&q);
        let cb = b.params.encode(&qp);
        for i in 0..n {
            assert_eq!(a.distance_raw(&ca, i), b.distance_raw(&cb, i));
        }
    }


    /// H14：非 128 倍数 / 奇数 / 高维（d=25,100,256,784,960）下分段距离与置换的正确性
    #[test]
    fn h14_bounded_and_perm_non128_dims() {
        let mut s = 999u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 };
        for dim in [25usize, 50, 96, 100, 256, 784, 960] {
            let nb = dim.div_ceil(2);
            for _ in 0..200 {
                let mut a: Vec<u8> = (0..nb).map(|_| rnd()).collect();
                let mut b: Vec<u8> = (0..nb).map(|_| rnd()).collect();
                if dim % 2 == 1 { a[nb - 1] &= 0x0F; b[nb - 1] &= 0x0F; } // 奇数维：尾字节高半字节恒 0（encode 行为）
                let full = l2_sq4_raw_scalar(&a, &b);
                assert_eq!(l2_sq4_raw(&a, &b), full, "dim={dim}");
                let inf = [f32::MAX; 8];
                // 全掩码（含越界位）不得 panic / 越界：检查点被限制在前 MAX_CHECKPOINTS 段
                assert_eq!(l2_sq4_raw_bounded(&a, &b, &inf, u32::MAX), (full, (nb * 2) as u32));
                assert_eq!(l2_sq4_raw_bounded16(&a, &b, &inf, u32::MAX), (full, (nb * 2) as u32));
                let t = (rnd() as f32) * (dim as f32 / 4.0);
                let thr = [t; 8];
                for (d, dims, step) in [
                    { let r = l2_sq4_raw_bounded(&a, &b, &thr, u32::MAX); (r.0, r.1, 16) },
                    { let r = l2_sq4_raw_bounded16(&a, &b, &thr, u32::MAX); (r.0, r.1, 8) },
                    { let r = l2_sq4_raw_bounded_scalar(&a, &b, &thr, u32::MAX); (r.0, r.1, 16) },
                    { let r = l2_sq4_raw_bounded_scalar_step(&a, &b, &thr, u32::MAX, 8); (r.0, r.1, 8) },
                ] {
                    if (dims as usize) < dim {
                        // 提前放弃：部分和是下界、≥ 阈值、只在完整段边界且不超过第 8 个检查点
                        assert!(d >= t && d <= full, "dim={dim}");
                        let p = dims as usize / 2;
                        assert_eq!(p % step, 0);
                        assert!(p / step <= MAX_CHECKPOINTS && p < nb);
                        assert_eq!(d, l2_sq4_raw_scalar(&a[..p], &b[..p]));
                    } else {
                        assert_eq!(d, full, "dim={dim}");
                        assert_eq!(dims as usize, nb * 2);
                    }
                }
            }
            // 置换构建在该维度下与原始逐位一致
            let n = 64usize;
            let data: Vec<f32> = (0..n * dim).map(|i| ((i * 7919 + 13) % 1009) as f32 * 0.01 - 5.0).collect();
            let perm: Vec<usize> = (0..dim).map(|j| (j * 37 + 5) % dim).collect();
            // 37 为素数且不整除上述任何 dim → (37j+5) mod dim 是置换
            let mut chk = perm.clone();
            chk.sort();
            assert_eq!(chk, (0..dim).collect::<Vec<_>>());
            let pa = SQ4Dataset::build(&data, dim);
            let pb = SQ4Dataset::build_permuted(&data, dim, &perm);
            assert_eq!(pa.code_bytes, nb);
            let q: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.37).sin() * 4.0).collect();
            let qp: Vec<f32> = perm.iter().map(|&d| q[d]).collect();
            let (ca, cb) = (pa.params.encode(&q), pb.params.encode(&qp));
            for i in 0..n {
                assert_eq!(pa.distance_raw(&ca, i), pb.distance_raw(&cb, i), "dim={dim} i={i}");
            }
        }
    }

    /// H17：clip=0 与 fit 完全一致；clip>0 时范围收窄、编码 clamp 到 [0,15]，置换版与非置换版距离逐位一致
    #[test]
    fn h17_fit_clipped() {
        let (n, dim) = (5000usize, 20usize);
        let mut s = 77u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 };
        // 长尾：大多数值 ∈[0,1)，0.2% 是 ×50 的离群值（clip=1% 分位数必落在主体内）
        let data: Vec<f32> = (0..n * dim).map(|_| { let x = rnd(); if rnd() < 0.002 { x * 50.0 } else { x } }).collect();
        let a = SQ4Params::fit(&data, dim);
        let b = SQ4Params::fit_clipped(&data, dim, 0.0);
        assert_eq!(a.min, b.min);
        assert_eq!(a.max, b.max);
        let c = SQ4Params::fit_clipped(&data, dim, 0.01);
        for d in 0..dim {
            assert!(c.min[d] >= a.min[d] && c.max[d] <= a.max[d] && c.max[d] > c.min[d]);
            assert!(c.max[d] < 10.0, "clipped max {} should drop the outlier tail", c.max[d]);
        }
        // MSE 自适应：每维 MSE 不劣于 min/max（候选包含 clip=0）
        let e = SQ4Params::fit_mse(&data, dim);
        let mse = |p: &SQ4Params, d: usize| -> f64 { (0..n).map(|i| { let x = data[i * dim + d]; let q = ((x - p.min[d]) / p.scale[d]).round().clamp(0.0, 15.0); let r = q * p.scale[d] + p.min[d]; ((x - r) as f64).powi(2) }).sum() };
        for d in 0..dim { assert!(mse(&e, d) <= mse(&a, d) * (1.0 + 1e-6) + 1e-9); }
        let code = c.encode(&vec![1000.0; dim]);
        assert!(code.iter().all(|&x| x == 0xFF)); // 超出范围 → 15
        let perm: Vec<usize> = (0..dim).rev().collect();
        let p1 = SQ4Dataset::build_clipped(&data, dim, 0.01);
        let p2 = SQ4Dataset::build_permuted_clipped(&data, dim, &perm, 0.01);
        let q: Vec<f32> = (0..dim).map(|d| d as f32 * 0.05).collect();
        let qp: Vec<f32> = perm.iter().map(|&d| q[d]).collect();
        let (c1, c2) = (p1.params.encode(&q), p2.params.encode(&qp));
        for i in 0..200 {
            assert_eq!(p1.distance_raw(&c1, i), p2.distance_raw(&c2, i));
        }
    }
}


#[cfg(test)]
mod r65_pad_tests {
    use super::*;
    #[test]
    fn pad_mult_for_default_unchanged() {
        for cb in [13usize, 16, 25, 48, 50, 64, 128, 196, 384, 392, 480] {
            assert_eq!(SQ4Dataset::pad_mult_for(cb, 32, 0), 32);
        }
        assert_eq!(SQ4Dataset::pad_mult_for(13, 32, 16), 16);
        assert_eq!(SQ4Dataset::pad_mult_for(16, 32, 16), 16);
        assert_eq!(SQ4Dataset::pad_mult_for(17, 32, 16), 32);
        assert_eq!(SQ4Dataset::pad_mult_for(392, 32, 16), 32);
    }
}

// ---------------------------------------------------------------------------------------------
// R84 H85 ADQ：任意码长的非对称 SQ4 内积导航（数据 4 bit 码、查询 int8 权重 w_d = q_d·scale_d）。
// 单位范数数据上 q·x̂ = q·min + Σ w_d c_d，−Σ w_d c_d 即无偏的内积估计（对称 4 bit 码空间 L2 会把查询也量化到 16 级、
// 且不按逐维步长加权）。oracle（yandex-200，难负例排序保真度 top100）：对称 0.751 → ADQ 0.814。
// 查询缓冲布局：[0..4) f32 sc，[4..8) i32 big = 15·Σ|wq|，[32..32+cb) 偶数维权重 wlo，[32+cb..32+2cb) 奇数维权重 whi（i8）。
// 距离 = big − Σ wq·c（≥0，越小越近；只在同一查询内比较）。
impl SQ4Params {
    pub fn encode_adq_into(&self, v: &[f32], cb: usize, buf: &mut Vec<u8>) {
        assert!(v.len() == self.dim && cb * 2 >= self.dim);
        buf.clear(); buf.resize(32 + 2 * cb, 0);
        let mut mx = 0f32;
        for d in 0..self.dim { mx = mx.max((v[d] * self.scale[d]).abs()); }
        let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        let inv = 1.0 / sc; let mut big: i32 = 0;
        for d in 0..self.dim {
            let w = (v[d] * self.scale[d] * inv).round().clamp(-127.0, 127.0) as i32;
            big += 15 * w.abs();
            let off = 32 + if d % 2 == 0 { d / 2 } else { cb + d / 2 };
            buf[off] = w as i8 as u8;
        }
        buf[0..4].copy_from_slice(&sc.to_le_bytes()); buf[4..8].copy_from_slice(&big.to_le_bytes());
        // R84 H85c NU：码是共享非均匀级（数据已按 nu_reencode 重编码）时，头部 [8] = 1，[16..32) = 级表 T[j]∈[1,127]，
        // big' = 63·Σ|w| + 64·Σw，距离 = big' − Σ w·T[c]（= 63Σ|w| − Σ w·(T−64) ≥ 0）。
        if let Some(pn) = PNR.get().filter(|p| p.dim == self.dim && p.proj_len == self.proj.len()) {
            // R84 H86 PNR 头：[4..8) big8 = 255·Σ|w|，[9] = 1，[12..16) Σw，[16..20) z·σ（σ = 4.61·‖w‖，低 nibble 近似均匀 0..15）
            let (mut sa, mut sw, mut ss) = (0i64, 0i64, 0f64);
            for d in 0..self.dim { let off = 32 + if d % 2 == 0 { d / 2 } else { cb + d / 2 }; let w = buf[off] as i8 as i64; sa += w.abs(); sw += w; ss += (w * w) as f64; }
            buf[4..8].copy_from_slice(&((255 * sa) as i32).to_le_bytes()); buf[9] = 1;
            buf[12..16].copy_from_slice(&(sw as i32).to_le_bytes());
            buf[16..20].copy_from_slice(&((pn.z as f64 * 4.61 * ss.sqrt()) as f32).to_le_bytes());
        } else if let Some(t) = NU_TABLE.get() {
            let (mut sa, mut sw) = (0i32, 0i32);
            for d in 0..self.dim { let off = 32 + if d % 2 == 0 { d / 2 } else { cb + d / 2 }; let w = buf[off] as i8 as i32; sa += w.abs(); sw += w; }
            let big2 = 63 * sa + 64 * sw;
            buf[4..8].copy_from_slice(&big2.to_le_bytes()); buf[8] = 1; buf[16..32].copy_from_slice(t);
        }
        // R85 H101 ANL：码空间真非对称 L2 = ‖x̂‖²/(2·sc) − Σw·c（+常数）。[10] = 1，[24..28) = 1/(2·sc)。NU 表占 [16..32)，不与 NU 同用。
        if buf[8] != 1 && ANL.iter().any(|a| a.get().map(|st| st.dim == self.dim && st.proj_len == self.proj.len()).unwrap_or(false)) {
            buf[10] = 1; buf[24..28].copy_from_slice(&(0.5 / sc).to_le_bytes());
        }
    }
}

/// R85 H101 ANL（ADQ norm-corrected L2）：每套码一份 ‖x̂‖²（x̂ = 码重建），按码地址换算行号。
pub struct AnlState { pub base: usize, pub end: usize, pub stride: usize, pub norms: &'static [f32], pub dim: usize, pub proj_len: usize }
pub static ANL: [std::sync::OnceLock<AnlState>; 2] = [std::sync::OnceLock::new(), std::sync::OnceLock::new()];
/// R85-16 ANL 快路径：首套码的区间与 norms 指针放成普通原子（免 OnceLock 遍历），配合邻居码预取一并预取 norms 行（anl_pf）。
pub static ANL_B0: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
pub static ANL_E0: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ANL_SH0: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ANL_N0: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[inline(always)]
fn anl_row(p: usize) -> Option<*const f32> {
    use std::sync::atomic::Ordering::Relaxed;
    let b = ANL_B0.load(Relaxed);
    if p >= b && p < ANL_E0.load(Relaxed) {
        let sh = ANL_SH0.load(Relaxed);
        let i = if sh != 0 { (p - b) >> sh } else { (p - b) / ANL[0].get().map(|st| st.stride).unwrap_or(1) };
        return Some(unsafe { (ANL_N0.load(Relaxed) as *const f32).add(i) });
    }
    for a in ANL.iter().skip(1) {
        if let Some(st) = a.get() {
            if p >= st.base && p < st.end { return Some(unsafe { st.norms.as_ptr().add((p - st.base) / st.stride) }); }
        }
    }
    None
}
/// 邻居码预取时顺带预取其 ‖x̂‖²（ANL 未开时只有一次原子读 + 比较）。
#[inline(always)]
pub unsafe fn anl_pf(p: *const i8) {
    if ANL_B0.load(std::sync::atomic::Ordering::Relaxed) == usize::MAX { return; }
    if let Some(r) = anl_row(p as usize) { crate::arch::prefetch_l1(r as *const i8); }
}
#[inline(always)]
fn anl_add(qa: &[u8], qb: &[u8]) -> f32 {
    if qa[10] != 1 { return 0.0; }
    match anl_row(qb.as_ptr() as usize) {
        Some(r) => (unsafe { *r }) * f32::from_le_bytes([qa[24], qa[25], qa[26], qa[27]]),
        None => 0.0,
    }
}
impl SQ4Dataset {
    /// R85 H101：‖x̂‖²（4 bit：x̂_d = min_d + c_d·scale_d；有 PNR 低 nibble 时 c8 = 16h + l，scale = 8 bit 步长）。
    pub fn anl_norms(&self, lo: Option<&[u8]>) -> Vec<f32> {
        let (dim, cb) = (self.params.dim, self.code_bytes);
        (0..self.n).map(|i| {
            let row = &self.codes[i * cb..(i + 1) * cb];
            let mut s = 0f64;
            for d in 0..dim {
                let b = row[d / 2]; let h = if d % 2 == 0 { b & 15 } else { b >> 4 };
                let c = match lo { Some(l) => { let lb = l[i * cb + d / 2]; let ll = if d % 2 == 0 { lb & 15 } else { lb >> 4 }; 16.0 * h as f64 + ll as f64 } None => h as f64 };
                let x = self.params.min[d] as f64 + c * self.params.scale[d] as f64; s += x * x;
            }
            s as f32
        }).collect()
    }
    /// 登记本码集的 ANL（RAVEN_ADQ_L2=1 时由 pipeline 调用）；返回是否登记成功。
    pub fn anl_register(&self, lo: Option<&[u8]>) -> bool {
        let norms: &'static [f32] = Box::leak(self.anl_norms(lo).into_boxed_slice());
        let st = AnlState { base: self.codes.as_ptr() as usize, end: self.codes.as_ptr() as usize + self.n * self.code_bytes, stride: self.code_bytes, norms, dim: self.params.dim, proj_len: self.params.proj.len() };
        for (j, a) in ANL.iter().enumerate() {
            if a.get().is_none() {
                let (b, e, cb, np) = (st.base, st.end, st.stride, st.norms.as_ptr() as usize);
                let ok = a.set(st).is_ok();
                if ok && j == 0 {
                    use std::sync::atomic::Ordering::SeqCst;
                    ANL_SH0.store(if cb.is_power_of_two() { cb.trailing_zeros() as usize } else { 0 }, SeqCst);
                    ANL_N0.store(np, SeqCst); ANL_E0.store(e, SeqCst); ANL_B0.store(b, SeqCst);
                }
                return ok;
            }
        }
        false
    }
}

impl SQ4Dataset {
    /// R84 H86：按逐维（截断）范围做 8 bit 均匀码，主码换成高 nibble，低 nibble 存入返回的数组（同布局同 code_bytes）。
    /// params.scale ← 8 bit 步长（w = q·st8 使 Σ w·c8 = Σ w·(16h + l)）。调用方须把返回数组常驻并登记 PNR。
    pub fn pnr_reencode(&mut self, rows: &dyn Fn(usize) -> Vec<f32>, clip: f32) -> Vec<u8> {
        let dim = self.params.dim; let n = self.n; let cb = self.code_bytes;
        let st = (n / 50000).max(1);
        let samp: Vec<f32> = (0..n).step_by(st).flat_map(|i| rows(i)).collect();
        let p0 = SQ4Params::fit_range(&samp, dim, clip);
        let mut hi = vec![0u8; n * cb]; let mut lo = vec![0u8; n * cb];
        let st8: Vec<f32> = (0..dim).map(|d| ((p0.max[d] - p0.min[d]) / 255.0).max(1e-12)).collect();
        for i in 0..n {
            let r = rows(i);
            for d in 0..dim {
                let c = ((r[d] - p0.min[d]) / st8[d]).round().clamp(0.0, 255.0) as u8;
                let (h, l) = (c >> 4, c & 15);
                if d % 2 == 0 { hi[i * cb + d / 2] = h; lo[i * cb + d / 2] = l; } else { hi[i * cb + d / 2] |= h << 4; lo[i * cb + d / 2] |= l << 4; }
            }
        }
        self.codes = hi.into();
        for d in 0..dim { self.params.min[d] = p0.min[d]; self.params.scale[d] = st8[d]; }
        lo
    }
}

/// R84 H85c：共享非均匀 16 级（Lloyd–Max，按逐维标准化后的样本拟合）的 7 bit 整数级表；进程全局，`nu_reencode` 设置一次。
pub static NU_TABLE: std::sync::OnceLock<[u8; 16]> = std::sync::OnceLock::new();

impl SQ4Dataset {
    /// R84 H85c：用逐维均值/标准差 + 共享 Lloyd–Max 16 级重编码（只给 ADQ 链用；对称核在此码上无意义）。
    /// `rows(i)` 返回第 i 个（码维序的）向量。params.min ← 均值，params.scale ← 标准差 × 级距单位（使 w = q·scale 与级表一致）。
    /// R85 H102 NER（norm-equalized rounding）：对每点只改少数几维的取整方向（就近 ↔ 次近），使 ‖x̂‖² ≈ 全局目标（样本中位数），
    /// 于是 ADQ 只算内积 −q·x̂ 与码空间真 L2 只差常数——零查询开销替代 ANL。按"每单位范数改变的 MSE 代价"贪心挑维。原地改码（保留补齐）。
    pub fn ner_reencode(&mut self, rows: &(dyn Fn(usize) -> Vec<f32> + Sync)) -> (f32, f64, f64, f64) {
        use rayon::prelude::*;
        let (dim, n, cb) = (self.params.dim, self.n, self.code_bytes);
        let pm = &self.params;
        let code_of = |codes: &[u8], i: usize, d: usize| -> u8 { let b = codes[i * cb + d / 2]; if d % 2 == 0 { b & 15 } else { b >> 4 } };
        let norm = |codes: &[u8], i: usize| -> f64 { (0..dim).map(|d| { let x = pm.min[d] as f64 + code_of(codes, i, d) as f64 * pm.scale[d] as f64; x * x }).sum() };
        let st = (n / 20000).max(1);
        let mut samp: Vec<f64> = (0..n).step_by(st).map(|i| norm(&self.codes, i)).collect();
        samp.sort_by(|a, b| a.total_cmp(b));
        let t = samp[samp.len() / 2];
        let old: Vec<u8> = self.codes.to_vec();
        let mut out = old.clone();
        let stats: (f64, f64, f64) = out.par_chunks_mut(cb).enumerate().map(|(i, row)| {
            let r = rows(i);
            let mut c: Vec<u8> = (0..dim).map(|d| code_of(&old, i, d)).collect();
            let mut nn = 0f64; for d in 0..dim { let x = pm.min[d] as f64 + c[d] as f64 * pm.scale[d] as f64; nn += x * x; }
            let d0 = (t - nn).abs();
            let mut rem = t - nn;
            let mut cand: Vec<(f64, usize, u8, f64)> = Vec::with_capacity(dim);
            for d in 0..dim {
                let u = r[d].mul_add(pm.inv_scale[d], pm.neg_min_div[d]) as f64; let cd = c[d] as f64;
                let alt = if u > cd { cd + 1.0 } else { cd - 1.0 };
                if !(0.0..=15.0).contains(&alt) { continue; }
                let (xa, xh) = (pm.min[d] as f64 + alt * pm.scale[d] as f64, pm.min[d] as f64 + cd * pm.scale[d] as f64);
                let dn = xa * xa - xh * xh;
                if dn == 0.0 || dn.signum() != rem.signum() { continue; }
                let cost = (pm.scale[d] as f64).powi(2) * (1.0 - 2.0 * (u - cd).abs().min(0.5));
                cand.push((cost / dn.abs(), d, alt as u8, dn));
            }
            cand.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut flips = 0f64;
            for &(_, d, alt, dn) in &cand { if (rem - dn).abs() < rem.abs() { rem -= dn; c[d] = alt; flips += 1.0; } }
            for d in 0..dim { let v = c[d]; if d % 2 == 0 { row[d / 2] = (row[d / 2] & 0xF0) | v; } else { row[d / 2] = (row[d / 2] & 0x0F) | (v << 4); } }
            (flips, d0, rem.abs())
        }).reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        self.codes = out.into();
        (t as f32, stats.0 / n as f64, stats.1 / n as f64, stats.2 / n as f64)
    }
    pub fn nu_reencode(&mut self, rows: &dyn Fn(usize) -> Vec<f32>) {
        let dim = self.params.dim; let n = self.n;
        let st = (n / 20000).max(1);
        let mut mean = vec![0f64; dim]; let mut sq = vec![0f64; dim]; let mut cnt = 0f64;
        let samp: Vec<Vec<f32>> = (0..n).step_by(st).map(|i| rows(i)).collect();
        for r in &samp { for d in 0..dim { mean[d] += r[d] as f64; sq[d] += (r[d] as f64).powi(2); } cnt += 1.0; }
        let mean: Vec<f32> = mean.iter().map(|m| (m / cnt) as f32).collect();
        let sd: Vec<f32> = (0..dim).map(|d| ((sq[d] / cnt - (mean[d] as f64).powi(2)).max(1e-24)).sqrt() as f32).collect();
        let mut z: Vec<f32> = Vec::with_capacity(samp.len() * dim);
        for r in &samp { for d in 0..dim { z.push((r[d] - mean[d]) / sd[d]); } }
        z.sort_by(|a, b| a.total_cmp(b));
        let mut c: Vec<f32> = (0..16).map(|j| z[((j as f32 + 0.5) / 16.0 * z.len() as f32) as usize]).collect();
        for _ in 0..40 {
            let b: Vec<f32> = (0..15).map(|j| 0.5 * (c[j] + c[j + 1])).collect();
            let (mut s, mut k) = ([0f64; 16], [0f64; 16]);
            let mut j = 0;
            for &x in &z { while j < 15 && x > b[j] { j += 1; } s[j] += x as f64; k[j] += 1.0; }
            for j in 0..16 { if k[j] > 0.0 { c[j] = (s[j] / k[j]) as f32; } }
        }
        let lmax = c.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-12);
        let mut t = [0u8; 16];
        for j in 0..16 { t[j] = ((c[j] / lmax * 63.0).round() as i32 + 64).clamp(1, 127) as u8; }
        let lv: Vec<f32> = t.iter().map(|&x| (x as f32 - 64.0) * lmax / 63.0).collect(); // 实际使用的级（整数化后）
        let b: Vec<f32> = (0..15).map(|j| 0.5 * (lv[j] + lv[j + 1])).collect();
        let cb = self.code_bytes;
        let mut codes = vec![0u8; n * cb];
        for i in 0..n {
            let r = rows(i);
            for d in 0..dim {
                let zz = (r[d] - mean[d]) / sd[d];
                let code = b.iter().take_while(|&&bb| zz > bb).count() as u8;
                if d % 2 == 0 { codes[i * cb + d / 2] = code; } else { codes[i * cb + d / 2] |= code << 4; }
            }
        }
        self.codes = codes.into();
        // w_d = q_d · sd_d · (lmax/63) 时 Σ w (T−64) = Σ q_d sd_d lv(c_d)
        for d in 0..dim { self.params.min[d] = mean[d]; self.params.scale[d] = sd[d] * lmax / 63.0; }
        let _ = NU_TABLE.set(t);
        eprintln!("R84 NU re-encode: levels {:?}", lv.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
    }
}

/// R84 H86 PNR（渐进嵌套精化导航）：8 bit 逐维码拆成高 nibble（= 导航码，常驻主码数组）与低 nibble（另一数组，地址按偏移对应）。
pub struct PnrState { pub hi_base: usize, pub lo_base: usize, pub z: f32, pub dim: usize, pub proj_len: usize }
// R84-17 修复：PNR 只属于一套码（dim + proj 长度识别）；LPX 两段管线（LP 码 + 全维码）里全维查询码不得打 PNR 头（曾致 landmark 段错误 rc=139）。
pub static PNR: std::sync::OnceLock<PnrState> = std::sync::OnceLock::new();
pub static PNR_CNT: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
#[inline(always)]
fn pnr_lo_(qb: &[u8]) -> &[u8] {
    let p = PNR.get().unwrap();
    unsafe { std::slice::from_raw_parts((p.lo_base + (qb.as_ptr() as usize - p.hi_base)) as *const u8, qb.len()) }
}
#[inline(always)]
fn adq_sum(qa: &[u8], qb: &[u8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    { if adq_vnni() { return unsafe { adq_vnni512(qa.as_ptr(), qb.as_ptr(), qb.len()) }; }
      if is_sq4_avx2_supported() { return unsafe { adq_avx2(qa.as_ptr(), qb.as_ptr(), qb.len()) }; } }
    adq_scalar(qa, qb)
}

/// R84 H85d：AVX-512 VNNI 核（vpdpbusd：u8 码 × i8 权重，4 积累加进 i32；每 64 B 码 2 条）。RAVEN_ADQ_VNNI=0 关。
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn adq_vnni() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vnni")
        && std::is_x86_feature_detected!("avx512vl") && std::env::var("RAVEN_ADQ_VNNI").map(|v| v != "0").unwrap_or(true))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2")]
#[inline]
unsafe fn adq_vnni512(q: *const u8, code: *const u8, n: usize) -> i32 {
    let m = _mm512_set1_epi8(0x0F);
    let (wl, wh) = (q.add(32), q.add(32 + n));
    let mut acc = _mm512_setzero_si512();
    let chunks = n / 64;
    for c in 0..chunks {
        let o = c * 64;
        let b = _mm512_loadu_si512(code.add(o) as *const _);
        let lo = _mm512_and_si512(b, m); let hi = _mm512_and_si512(_mm512_srli_epi16(b, 4), m);
        acc = _mm512_dpbusd_epi32(acc, lo, _mm512_loadu_si512(wl.add(o) as *const _));
        acc = _mm512_dpbusd_epi32(acc, hi, _mm512_loadu_si512(wh.add(o) as *const _));
    }
    let mut s = _mm512_reduce_add_epi32(acc);
    let mut o = chunks * 64;
    if n - o >= 32 {
        let m2 = _mm256_set1_epi8(0x0F);
        let b = _mm256_loadu_si256(code.add(o) as *const __m256i);
        let lo = _mm256_and_si256(b, m2); let hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m2);
        let a = _mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, _mm256_loadu_si256(wl.add(o) as *const __m256i));
        let a = _mm256_dpbusd_epi32(a, hi, _mm256_loadu_si256(wh.add(o) as *const __m256i));
        let x = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
        let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0x4E>(x)); let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0x01>(x));
        s += _mm_cvtsi128_si32(x); o += 32;
    }
    while o < n { let cb = *code.add(o); s += (cb & 0x0F) as i32 * (*wl.add(o) as i8 as i32) + (cb >> 4) as i32 * (*wh.add(o) as i8 as i32); o += 1; }
    s
}
pub fn pnr_lo(qb: &[u8]) -> &[u8] { pnr_lo_(qb) }
#[inline(always)] pub fn pnr_zs(qa: &[u8]) -> f32 { f32::from_le_bytes([qa[16], qa[17], qa[18], qa[19]]) }
/// 阶段 1：(d1 中点估计, S_hi)
#[inline(always)] pub fn pnr_stage1(qa: &[u8], qb: &[u8]) -> (f32, i32) {
    crate::diag_count!(QUANT_DIST);
    let big = i32::from_le_bytes([qa[4], qa[5], qa[6], qa[7]]); let sw = i32::from_le_bytes([qa[12], qa[13], qa[14], qa[15]]);
    let sh = adq_sum(qa, qb); ((big - 16 * sh) as f32 - 7.5 * sw as f32 + anl_add(qa, qb), sh)
}
/// 阶段 2：精确 8 bit 距离
#[inline(always)] pub fn pnr_stage2(qa: &[u8], qb: &[u8], sh: i32) -> f32 {
    let big = i32::from_le_bytes([qa[4], qa[5], qa[6], qa[7]]); (big - 16 * sh - adq_sum(qa, pnr_lo_(qb))) as f32 + anl_add(qa, qb)
}
/// 无条件精化（导航层、种子等少量调用）：big8 − (16·S_hi + S_lo)
#[inline(always)]
fn adq_pnr_full(qa: &[u8], qb: &[u8]) -> f32 {
    let big = i32::from_le_bytes([qa[4], qa[5], qa[6], qa[7]]);
    (big - 16 * adq_sum(qa, qb) - adq_sum(qa, pnr_lo_(qb))) as f32 + anl_add(qa, qb)
}
/// 带门槛：中点估计 d1 = big8 − 16·S_hi − 7.5·Σw；d1 − zσ > thr 时直接返回 d1（必被池拒绝），否则读低 nibble 精化。
#[inline(always)]
pub fn adq_pnr_dist(qa: &[u8], qb: &[u8], thr: f32) -> f32 {
    crate::diag_count!(QUANT_DIST);
    let big = i32::from_le_bytes([qa[4], qa[5], qa[6], qa[7]]);
    let sw = i32::from_le_bytes([qa[12], qa[13], qa[14], qa[15]]);
    let zs = f32::from_le_bytes([qa[16], qa[17], qa[18], qa[19]]);
    let sh = adq_sum(qa, qb);
    let an = anl_add(qa, qb);
    let d1 = (big - 16 * sh) as f32 - 7.5 * sw as f32 + an;
    if d1 - zs > thr { if cfg!(feature = "diag_counters") { PNR_CNT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); } return d1; }
    if cfg!(feature = "diag_counters") { PNR_CNT[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    (big - 16 * sh - adq_sum(qa, pnr_lo_(qb))) as f32 + an
}

#[inline(always)]
pub fn adq_dist(qa: &[u8], qb: &[u8]) -> f32 {
    if qa[9] == 1 { return adq_pnr_full(qa, qb); }
    crate::diag_count!(QUANT_DIST);
    let n = qb.len();
    debug_assert!(qa.len() >= 32 + 2 * n);
    let big = i32::from_le_bytes([qa[4], qa[5], qa[6], qa[7]]);
    #[cfg(target_arch = "x86_64")]
    { if qa[8] != 1 && adq_vnni() { return (big - unsafe { adq_vnni512(qa.as_ptr(), qb.as_ptr(), n) }) as f32 + anl_add(qa, qb); }
      if is_sq4_avx2_supported() { return (big - unsafe { if qa[8] == 1 { adq_nu_avx2(qa.as_ptr(), qb.as_ptr(), n) } else { adq_avx2(qa.as_ptr(), qb.as_ptr(), n) } }) as f32 + anl_add(qa, qb); } }
    (big - if qa[8] == 1 { adq_nu_scalar(qa, qb) } else { adq_scalar(qa, qb) }) as f32 + anl_add(qa, qb)
}

pub fn adq_nu_scalar(qa: &[u8], qb: &[u8]) -> i32 {
    let n = qb.len(); let t = &qa[16..32]; let mut s = 0i32;
    for i in 0..n {
        s += t[(qb[i] & 0x0F) as usize] as i32 * (qa[32 + i] as i8 as i32) + t[(qb[i] >> 4) as usize] as i32 * (qa[32 + n + i] as i8 as i32);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn adq_nu_avx2(q: *const u8, code: *const u8, n: usize) -> i32 {
    let m = _mm256_set1_epi8(0x0F); let ones = _mm256_set1_epi16(1);
    let t128 = _mm_loadu_si128(q.add(16) as *const __m128i);
    let t = _mm256_broadcastsi128_si256(t128);
    let (wl, wh) = (q.add(32), q.add(32 + n));
    let mut acc = _mm256_setzero_si256();
    let chunks = n / 32;
    for c in 0..chunks {
        let o = c * 32;
        let b = _mm256_loadu_si256(code.add(o) as *const __m256i);
        let lo = _mm256_shuffle_epi8(t, _mm256_and_si256(b, m)); let hi = _mm256_shuffle_epi8(t, _mm256_and_si256(_mm256_srli_epi16(b, 4), m));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256(wl.add(o) as *const __m256i)), ones));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256(wh.add(o) as *const __m256i)), ones));
    }
    let mut x = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
    let mut o = chunks * 32;
    if n - o >= 16 {
        let m1 = _mm_set1_epi8(0x0F); let o1 = _mm_set1_epi16(1);
        let b = _mm_loadu_si128(code.add(o) as *const __m128i);
        let lo = _mm_shuffle_epi8(t128, _mm_and_si128(b, m1)); let hi = _mm_shuffle_epi8(t128, _mm_and_si128(_mm_srli_epi16(b, 4), m1));
        x = _mm_add_epi32(x, _mm_madd_epi16(_mm_maddubs_epi16(lo, _mm_loadu_si128(wl.add(o) as *const __m128i)), o1));
        x = _mm_add_epi32(x, _mm_madd_epi16(_mm_maddubs_epi16(hi, _mm_loadu_si128(wh.add(o) as *const __m128i)), o1));
        o += 16;
    }
    let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0x4E>(x));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0x01>(x));
    let mut s = _mm_cvtsi128_si32(x);
    let tb = q.add(16);
    while o < n { let cb = *code.add(o); s += *tb.add((cb & 0x0F) as usize) as i32 * (*wl.add(o) as i8 as i32) + *tb.add((cb >> 4) as usize) as i32 * (*wh.add(o) as i8 as i32); o += 1; }
    s
}

pub fn adq_scalar(qa: &[u8], qb: &[u8]) -> i32 {
    let n = qb.len(); let mut s = 0i32;
    for i in 0..n {
        let lo = (qb[i] & 0x0F) as i32; let hi = (qb[i] >> 4) as i32;
        s += lo * (qa[32 + i] as i8 as i32) + hi * (qa[32 + n + i] as i8 as i32);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn adq_avx2(q: *const u8, code: *const u8, n: usize) -> i32 {
    let m = _mm256_set1_epi8(0x0F); let ones = _mm256_set1_epi16(1);
    let (wl, wh) = (q.add(32), q.add(32 + n));
    let mut acc = _mm256_setzero_si256();
    let chunks = n / 32;
    for c in 0..chunks {
        let o = c * 32;
        let b = _mm256_loadu_si256(code.add(o) as *const __m256i);
        let lo = _mm256_and_si256(b, m); let hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m);
        let p = _mm256_add_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256(wl.add(o) as *const __m256i)),
                                 _mm256_maddubs_epi16(hi, _mm256_loadu_si256(wh.add(o) as *const __m256i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    let x = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
    let mut o = chunks * 32;
    let mut x = x;
    if n - o >= 16 {
        let m1 = _mm_set1_epi8(0x0F);
        let b = _mm_loadu_si128(code.add(o) as *const __m128i);
        let lo = _mm_and_si128(b, m1); let hi = _mm_and_si128(_mm_srli_epi16(b, 4), m1);
        let p = _mm_add_epi16(_mm_maddubs_epi16(lo, _mm_loadu_si128(wl.add(o) as *const __m128i)),
                              _mm_maddubs_epi16(hi, _mm_loadu_si128(wh.add(o) as *const __m128i)));
        x = _mm_add_epi32(x, _mm_madd_epi16(p, _mm_set1_epi16(1)));
        o += 16;
    }
    let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0x4E>(x));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0x01>(x));
    let mut s = _mm_cvtsi128_si32(x);
    while o < n { let cb = *code.add(o); s += (cb & 0x0F) as i32 * (*wl.add(o) as i8 as i32) + (cb >> 4) as i32 * (*wh.add(o) as i8 as i32); o += 1; }
    s
}

#[cfg(test)]
mod adq_tests {
    use super::*;
    #[test]
    fn adq_matches_scalar_and_ip() {
        let dim = 200; let n = 300;
        let mut st = 12345u64; let mut r = || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 33) as f32 / (1u64 << 31) as f32) - 0.5 };
        let mut data: Vec<f32> = (0..n * dim).map(|_| r()).collect();
        for i in 0..n { let s: f32 = data[i*dim..(i+1)*dim].iter().map(|x| x*x).sum::<f32>().sqrt(); for x in &mut data[i*dim..(i+1)*dim] { *x /= s; } }
        let mut ds = SQ4Dataset::build(&data, dim); ds.pad_codes(32);
        let q: Vec<f32> = data[0..dim].to_vec();
        let mut qb = Vec::new(); ds.params.encode_adq_into(&q, ds.code_bytes, &mut qb);
        let sc = f32::from_le_bytes([qb[0], qb[1], qb[2], qb[3]]); let big = i32::from_le_bytes([qb[4], qb[5], qb[6], qb[7]]);
        for i in 0..n {
            let a = adq_dist(&qb, ds.code(i)); let s = (big - adq_scalar(&qb, ds.code(i))) as f32;
            assert_eq!(a, s, "i={i}");
            #[cfg(target_arch = "x86_64")]
            if adq_vnni() { assert_eq!(unsafe { adq_vnni512(qb.as_ptr(), ds.code(i).as_ptr(), ds.code_bytes) }, adq_scalar(&qb, ds.code(i))); }
            let xh = ds.params.decode(&ds.code(i)[..dim.div_ceil(2)]);
            let ip: f32 = (0..dim).map(|d| q[d] * (xh[d] - ds.params.min[d])).sum();
            let est = (big as f32 - a) * sc;
            assert!((est - ip).abs() < 0.02 * ip.abs().max(1.0), "i={i} est={est} ip={ip}");
        }
    }
    #[test]
    fn adq_nu_matches_scalar_and_ip() {
        let dim = 200; let n = 400;
        let mut st = 777u64; let mut r = || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 33) as f32 / (1u64 << 31) as f32) - 0.5 };
        let mut data: Vec<f32> = (0..n * dim).map(|_| r() + r() + r()).collect();
        for i in 0..n { let s: f32 = data[i*dim..(i+1)*dim].iter().map(|x| x*x).sum::<f32>().sqrt(); for x in &mut data[i*dim..(i+1)*dim] { *x /= s; } }
        let mut ds = SQ4Dataset::build(&data, dim); ds.pad_codes(32);
        let dd = data.clone(); ds.nu_reencode(&|i| dd[i*dim..(i+1)*dim].to_vec());
        let t = *NU_TABLE.get().unwrap();
        let q: Vec<f32> = data[dim..2*dim].to_vec();
        let mut qb = Vec::new(); ds.params.encode_adq_into(&q, ds.code_bytes, &mut qb);
        assert_eq!(qb[8], 1);
        let sc = f32::from_le_bytes([qb[0], qb[1], qb[2], qb[3]]); let big = i32::from_le_bytes([qb[4], qb[5], qb[6], qb[7]]);
        let mut sw = 0i32; for i in 0..2*ds.code_bytes { sw += qb[32+i] as i8 as i32; }
        let mut err = 0f64; let mut ipn = 0f64;
        for i in 0..n {
            let a = adq_dist(&qb, ds.code(i)); let s = (big - adq_nu_scalar(&qb, ds.code(i))) as f32;
            assert_eq!(a, s, "i={i}"); assert!(a >= 0.0);
            // 估计内积（去掉 q·mean 常数）vs 真实 q·(x − mean)
            let est = ((adq_nu_scalar(&qb, ds.code(i)) - 64 * sw) as f32) * sc;
            let ip: f32 = (0..dim).map(|d| q[d] * (data[i*dim+d] - ds.params.min[d])).sum();
            err += ((est - ip) as f64).powi(2); ipn += (ip as f64).powi(2);
            let _ = t;
        }
        assert!(err / ipn < 0.05, "rel err {}", err / ipn);
    }
}

// ---------------------------------------------------------------------------------------------
// R97 H176 A（RAVEN_L0B=1）：L0 每跳批量打分。一批最多 16 个邻居码（批内按指针 gather），按 64 B 维块外层、码内层循环：
// 每个维块的查询权重（i8，wl/wh）只装一次、在 16 个独立累加器间复用（GINGER TurboQuantScoringInt8BatchDims 式的"批×维块"顺序），
// 16 个码的访存同时在飞。NU（qa[8]=1）用 vpshufb 查共享级表 T（u8 ∈[1,127]）后 vpdpbusd；非 NU 直接 nibble × i8。
// 整数和精确 → 与 adq_dist 逐位一致（调用方保证 qa[9]≠1（PNR 关）、qa[10]≠1（ANL 关））。
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn adq_batch_ok(qa: &[u8]) -> bool { qa.len() > 10 && qa[9] != 1 && qa[10] != 1 && adq_vnni() }
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn adq_batch_ok(_qa: &[u8]) -> bool { false }

/// SAFETY：codes[..cnt] 指向长 n 的码；q 为 encode_adq_into 缓冲（长 32+2n）；cnt ≤ 16；需 adq_vnni()。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2")]
pub unsafe fn adq_batch16(q: *const u8, codes: &[*const u8; 16], cnt: usize, n: usize, out: &mut [f32; 16]) {
    let m = _mm512_set1_epi8(0x0F);
    let nu = *q.add(8) == 1;
    let t = _mm512_broadcast_i32x4(_mm_loadu_si128(q.add(16) as *const __m128i));
    let (wl, wh) = (q.add(32), q.add(32 + n));
    let big = i32::from_le_bytes([*q.add(4), *q.add(5), *q.add(6), *q.add(7)]);
    let mut acc: [__m512i; 16] = [_mm512_setzero_si512(); 16];
    let mut o = 0usize;
    while o < n {
        let rem = n - o;
        let mk: u64 = if rem >= 64 { u64::MAX } else { (1u64 << rem) - 1 };
        let a = _mm512_maskz_loadu_epi8(mk, wl.add(o) as *const i8);
        let b = _mm512_maskz_loadu_epi8(mk, wh.add(o) as *const i8);
        for j in 0..16 {
            if j >= cnt { break; }
            let c = _mm512_maskz_loadu_epi8(mk, (*codes.get_unchecked(j)).add(o) as *const i8);
            let mut lo = _mm512_and_si512(c, m); let mut hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), m);
            if nu { lo = _mm512_shuffle_epi8(t, lo); hi = _mm512_shuffle_epi8(t, hi); }
            let x = _mm512_dpbusd_epi32(*acc.get_unchecked(j), lo, a);
            *acc.get_unchecked_mut(j) = _mm512_dpbusd_epi32(x, hi, b);
        }
        o += 64;
    }
    for j in 0..cnt { *out.get_unchecked_mut(j) = (big - _mm512_reduce_add_epi32(*acc.get_unchecked(j))) as f32; }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod h176_batch_tests {
    use super::*;
    #[test]
    fn batch16_matches_adq_dist() {
        if !adq_vnni() { return; }
        for &dim in &[100usize, 200, 384, 768] {
            let n = 64;
            let mut st = 99u64 + dim as u64; let mut r = || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 33) as f32 / (1u64 << 31) as f32) - 0.5 };
            let data: Vec<f32> = (0..n * dim).map(|_| r()).collect();
            let mut ds = SQ4Dataset::build(&data, dim); ds.pad_codes(32);
            let q: Vec<f32> = (0..dim).map(|_| r()).collect();
            let mut qb = Vec::new(); ds.params.encode_adq_into(&q, ds.code_bytes, &mut qb);
            for nu in [false, true] {
                let mut qq = qb.clone();
                if nu { qq[8] = 1; for j in 0..16 { qq[16 + j] = (1 + j * 8) as u8; }
                    let (mut sa, mut sw) = (0i32, 0i32); for i in 0..2 * ds.code_bytes { let w = qq[32 + i] as i8 as i32; sa += w.abs(); sw += w; }
                    qq[4..8].copy_from_slice(&(63 * sa + 64 * sw).to_le_bytes()); }
                for cnt in [1usize, 7, 16] {
                    let mut ptrs = [std::ptr::null::<u8>(); 16];
                    for j in 0..cnt { ptrs[j] = ds.code(j * 3).as_ptr(); }
                    let mut out = [0f32; 16];
                    unsafe { adq_batch16(qq.as_ptr(), &ptrs, cnt, ds.code_bytes, &mut out); }
                    for j in 0..cnt { assert_eq!(out[j], adq_dist(&qq, ds.code(j * 3)), "dim={dim} nu={nu} j={j}"); }
                }
            }
        }
    }
}
