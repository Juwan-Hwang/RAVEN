//! f16 半精度距离核（Week 7+ 按需优化）
//!
//! 设计文档第一层精度三层：
//! - f16 带宽优化快路径，减少内存传输，不改变图结构决策
//! - Week 7+ 按需优化，不阻塞主线里程碑
//!
//! 实现：
//! - f32 ↔ f16 转换（IEEE 754 半精度）
//! - l2_f16：接受 f32 输入，内部转 f16 计算距离（验证精度损失）
//! - l2_f16_packed：接受预量化的 f16 数组，真正减少内存带宽
//!
//! f16 格式（IEEE 754-2008 binary16）：
//! - 符号位：1 bit
//! - 指数位：5 bits（bias=15）
//! - 尾数位：10 bits
//! - 范围：±65504，最小正规数约 6.1e-5

use std::arch::x86_64::*;

/// f16 类型（用 u16 存储，便于 SIMD 和序列化）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16(pub u16);

impl F16 {
    /// f32 → f16 转换（IEEE 754 半精度，round-to-nearest-even）
    #[inline(always)]
    pub fn from_f32(f: f32) -> Self {
        // 提取 f32 位
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        // 处理特殊值
        if exp == 0xFF {
            // NaN 或 Inf
            if mant != 0 {
                // NaN：保留符号，尾数非零
                return Self(sign | 0x7E00);
            }
            // Inf
            return Self(sign | 0x7C00);
        }

        // 指数调整：f32 bias=127，f16 bias=15
        let new_exp = exp - 127 + 15;

        if new_exp >= 0x1F {
            // 上溢：返回 Inf
            return Self(sign | 0x7C00);
        }

        if new_exp <= 0 {
            // 下溢：denormal 或零
            if new_exp < -10 {
                // 太小，返回零
                return Self(sign);
            }
            // denormal：尾数需要右移
            let mant = mant | 0x800000; // 加上隐含的 1
            let shift = 14 - new_exp;
            // round-to-nearest-even
            let half = 1 << (shift - 1);
            let mant_rounded = (mant >> shift) + u32::from((mant & (half - 1)) > half || ((mant & half) > 0 && (mant & (half - 1)) >= half));
            return Self(sign | mant_rounded as u16);
        }

        // 正规数
        let mant_f16 = (mant >> 13) as u16;
        // round-to-nearest-even
        let rem = mant & 0x1FFF;
        let half = 0x1000;
        let round = u16::from(rem > half || (rem == half && (mant_f16 & 1) == 1));
        let result = sign | ((new_exp as u16) << 10) | mant_f16 | round;
        // 处理 round 导致的进位
        let result = if (result & 0x7C00) == 0x7C00 {
            sign | 0x7C00 // 上溢到 Inf
        } else {
            result
        };
        Self(result)
    }

    /// f16 → f32 转换
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as u32;

        if exp == 0 {
            if mant == 0 {
                // 零
                return f32::from_bits(sign);
            }
            // denormal：调整到 f32 的 denormal 范围
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let new_exp = 127 + e - 14;
            return f32::from_bits(sign | ((new_exp as u32) << 23) | (m << 13));
        }

        if exp == 0x1F {
            // Inf 或 NaN
            return f32::from_bits(sign | 0x7F800000 | (mant << 13));
        }

        // 正规数
        let new_exp = exp - 15 + 127;
        f32::from_bits(sign | ((new_exp as u32) << 23) | (mant << 13))
    }
}

/// 将 f32 切片转换为 f16 切片（预量化，减少内存占用 50%）
pub fn f32_to_f16_slice(f32_slice: &[f32]) -> Vec<F16> {
    f32_slice.iter().map(|&f| F16::from_f32(f)).collect()
}

/// 将 f16 切片转换回 f32 切片
pub fn f16_to_f32_slice(f16_slice: &[F16]) -> Vec<f32> {
    f16_slice.iter().map(|&f| f.to_f32()).collect()
}

/// f16 L2 距离（接受 f32 输入，内部转 f16 计算）
///
/// 设计文档：f16 带宽优化快路径
/// 此函数用于验证 f16 精度损失，不减少实际带宽（输入仍为 f32）
/// 真正的带宽优化用 l2_f16_packed
#[inline(always)]
pub fn l2_f16(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let af = F16::from_f32(a[i]).to_f32();
        let bf = F16::from_f32(b[i]).to_f32();
        let d = af - bf;
        sum = d.mul_add(d, sum);
    }
    sum
}

/// f16 L2 距离（接受预量化 f16 数组，真正减少内存带宽）
///
/// 设计文档：减少内存传输，不改变图结构决策
/// 带宽降低 50%（f16=2B vs f32=4B），适合高维数据集
#[inline(always)]
pub fn l2_f16_packed(a: &[F16], b: &[F16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let af = a[i].to_f32();
        let bf = b[i].to_f32();
        let d = af - bf;
        sum = d.mul_add(d, sum);
    }
    sum
}

/// f16 L2 距离（混合模式：query 为 f32，数据库为 f16）
///
/// 实际场景：query 在线到达为 f32，数据库预量化为 f16
#[inline(always)]
pub fn l2_f16_mixed(query: &[f32], db_packed: &[F16]) -> f32 {
    debug_assert_eq!(query.len(), db_packed.len());
    let mut sum = 0.0f32;
    for i in 0..query.len() {
        let q = query[i];
        let d = db_packed[i].to_f32();
        let diff = q - d;
        sum = diff.mul_add(diff, sum);
    }
    sum
}

// ============================================================================
// OPT-4: SIMD f16 距离核（F16C + AVX-512）
// ============================================================================
//
// 设计文档：f16 带宽优化快路径，减少内存传输，不改变图结构决策
//
// 核心思路：
// - 数据库预量化为 f16（2B/元素），内存带宽减半
// - 查询时用 F16C 指令（_mm256_cvtph_ps）将 f16 转为 f32 在寄存器中计算
// - 计算精度保持 f32，但内存传输量减半
// - 适用于 memory-bound 场景（图搜索的随机访问模式）
//
// 平台要求：F16C + AVX-512F（Skylake-X / Zen4 及以后）

/// 检测 CPU 是否支持 F16C 指令
pub fn is_f16c_supported() -> bool {
    std::is_x86_feature_detected!("f16c")
}

/// SIMD f16 L2 距离（AVX-512 + F16C，混合模式）
///
/// query 为 f32，db 为预量化 f16。
/// 用 F16C 将 f16 转为 f32 在寄存器中计算，内存带宽减半。
///
/// 每 cycle 处理 16 个元素：
/// - 2 次 _mm_loadu_si128 加载 16 个 f16（32 字节）
/// - 2 次 _mm256_cvtph_ps 将 f16 转为 f32（寄存器内）
/// - _mm512_insertf32x8 合并为 16-wide f32
/// - _mm512_fmadd_ps 做距离累加
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "f16c")]
#[inline]
pub unsafe fn l2_f16_mixed_avx512(query: &[f32], db_packed: &[F16]) -> f32 {
    debug_assert_eq!(query.len(), db_packed.len());
    let n = query.len();
    let chunks = n / 16;
    let remainder = n % 16;

    let q_ptr = query.as_ptr();
    // F16 是 u16，按 u16 指针访问
    let db_ptr = db_packed.as_ptr() as *const u16;

    let mut sum = _mm512_setzero_ps();

    // 主循环：每次处理 16 个元素
    for i in 0..chunks {
        let offset = i * 16;
        // 加载 query 的 16 个 f32
        let vq = _mm512_loadu_ps(q_ptr.add(offset));

        // 加载 db 的 16 个 f16（2 次 8-wide）
        let db_lo = _mm_loadu_si128(db_ptr.add(offset) as *const __m128i);
        let db_hi = _mm_loadu_si128(db_ptr.add(offset + 8) as *const __m128i);
        // F16C: f16 → f32
        let db_lo_f32 = _mm256_cvtph_ps(db_lo);
        let db_hi_f32 = _mm256_cvtph_ps(db_hi);
        // 合并为 16-wide f32
        let vdb = _mm512_insertf32x8(_mm512_castps256_ps512(db_lo_f32), db_hi_f32, 1);

        let d = _mm512_sub_ps(vq, vdb);
        sum = _mm512_fmadd_ps(d, d, sum);
    }

    let mut result = 0.0f32;
    if remainder > 0 {
        let offset = chunks * 16;
        // 尾部逐元素处理（remainder < 16）
        for i in 0..remainder {
            let q = *q_ptr.add(offset + i);
            let d = F16(*db_ptr.add(offset + i)).to_f32();
            let diff = q - d;
            result += diff * diff;
        }
    }

    result + _mm512_reduce_add_ps(sum)
}

/// SIMD f16 L2 距离（AVX2 + F16C，混合模式）
///
/// 8-wide f32 计算，适用于不支持 AVX-512 的平台
#[target_feature(enable = "avx2")]
#[target_feature(enable = "f16c")]
#[inline]
pub unsafe fn l2_f16_mixed_avx2(query: &[f32], db_packed: &[F16]) -> f32 {
    debug_assert_eq!(query.len(), db_packed.len());
    let n = query.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let q_ptr = query.as_ptr();
    let db_ptr = db_packed.as_ptr() as *const u16;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let vq = _mm256_loadu_ps(q_ptr.add(offset));
        let db_f16 = _mm_loadu_si128(db_ptr.add(offset) as *const __m128i);
        let vdb = _mm256_cvtph_ps(db_f16);
        let d = _mm256_sub_ps(vq, vdb);
        sum = _mm256_fmadd_ps(d, d, sum);
    }

    let mut result = 0.0f32;
    if remainder > 0 {
        let offset = chunks * 8;
        for i in 0..remainder {
            let q = *q_ptr.add(offset + i);
            let d = F16(*db_ptr.add(offset + i)).to_f32();
            let diff = q - d;
            result += diff * diff;
        }
    }

    // 水平求和 AVX2 寄存器
    let buf: [f32; 8] = std::mem::transmute(sum);
    result + buf.iter().sum::<f32>()
}

/// 统一 SIMD f16 L2 距离分发（AVX-512 > AVX2 > 标量）
///
/// 设计文档：f16 带宽优化快路径
/// 运行时检测 CPU 特性，优先使用最宽的 SIMD 核
#[inline(always)]
pub fn l2_f16_mixed_simd(query: &[f32], db_packed: &[F16]) -> f32 {
    if is_avx512_and_f16c_supported() {
        unsafe { l2_f16_mixed_avx512(query, db_packed) }
    } else if is_avx2_and_f16c_supported() {
        unsafe { l2_f16_mixed_avx2(query, db_packed) }
    } else {
        l2_f16_mixed(query, db_packed)
    }
}

/// 检测 AVX-512 + F16C 同时支持
pub fn is_avx512_and_f16c_supported() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("f16c")
}

/// 检测 AVX2 + F16C 同时支持
pub fn is_avx2_and_f16c_supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")
}

// ─── F16Dataset：预量化数据集 ─────────────────────────────────────

/// f16 预量化数据集
///
/// 将 f32 向量预量化为 f16 紧凑存储，用于 rerank 阶段带宽优化。
/// 内存占用 = f32 的 50%（2B/元素 vs 4B/元素）
///
/// 适用场景：高维数据集（≥512 维）的 rerank 阶段
/// 低维数据集（SIFT-128）：rerank 数据已在 L1 cache，带宽不是瓶颈
pub struct F16Dataset {
    /// f16 编码的向量（紧凑存储，连续布局）
    codes: Vec<F16>,
    /// 维度
    dim: usize,
}

impl F16Dataset {
    /// 从 f32 向量构建 f16 预量化数据集
    pub fn build(vectors: &[f32], dim: usize) -> Self {
        let codes = f32_to_f16_slice(vectors);
        Self { codes, dim }
    }

    /// 获取原始 f16 码区（用于 rerank 等需要全量访问的场景）
    #[inline(always)]
    pub fn codes(&self) -> &[F16] {
        &self.codes
    }

    /// 获取指定节点的 f16 向量
    #[inline(always)]
    pub fn vector(&self, id: usize) -> &[F16] {
        &self.codes[id * self.dim..(id + 1) * self.dim]
    }

    /// 维度
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 节点总数
    pub fn len(&self) -> usize {
        self.codes.len() / self.dim
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_basic() {
        let values = [1.0f32, 2.0, 3.0, 0.5, -1.5, 100.0, 0.001];
        for &v in &values {
            let f16 = F16::from_f32(v);
            let restored = f16.to_f32();
            assert!((restored - v).abs() / v.max(1e-6) < 0.01,
                "f16 roundtrip failed: {} → {} → {}", v, f16.0, restored);
        }
    }

    #[test]
    fn f16_zero_and_inf() {
        assert_eq!(F16::from_f32(0.0).0, 0);
        assert_eq!(F16::from_f32(-0.0).0, 0x8000);
        assert_eq!(F16::from_f32(f32::INFINITY).0, 0x7C00);
        assert_eq!(F16::from_f32(f32::NEG_INFINITY).0, 0xFC00);
    }

    #[test]
    fn f16_overflow() {
        // 65504 是 f16 最大正规数
        let f16 = F16::from_f32(70000.0);
        assert_eq!(f16.0 & 0x7C00, 0x7C00); // Inf
    }

    #[test]
    fn l2_f16_matches_f32_within_tolerance() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.5f32, 2.5, 3.5, 4.5, 5.5];
        // f32 距离
        let mut f32_dist = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            f32_dist += d * d;
        }
        // f16 距离
        let f16_dist = l2_f16(&a, &b);
        // f16 精度损失应在 1% 以内
        let rel_err = (f16_dist - f32_dist).abs() / f32_dist.max(1e-6);
        assert!(rel_err < 0.01, "f16 precision loss too high: {} vs {}, rel_err={}",
            f16_dist, f32_dist, rel_err);
    }

    #[test]
    fn l2_f16_packed_basic() {
        let a_f32 = [1.0f32, 2.0, 3.0];
        let b_f32 = [4.0f32, 5.0, 6.0];
        let a = f32_to_f16_slice(&a_f32);
        let b = f32_to_f16_slice(&b_f32);
        let dist = l2_f16_packed(&a, &b);
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 27
        assert!((dist - 27.0).abs() < 0.5, "l2_f16_packed: {}", dist);
    }

    #[test]
    fn l2_f16_mixed_basic() {
        let query = [1.0f32, 2.0, 3.0];
        let db_f32 = [4.0f32, 5.0, 6.0];
        let db = f32_to_f16_slice(&db_f32);
        let dist = l2_f16_mixed(&query, &db);
        assert!((dist - 27.0).abs() < 0.5, "l2_f16_mixed: {}", dist);
    }

    #[test]
    fn f16_sift_range_precision() {
        // SIFT 数据范围 [0, 1]（归一化后），测试此范围内的精度
        let values: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let mut max_err = 0.0f32;
        for &v in &values {
            let restored = F16::from_f32(v).to_f32();
            let err = (restored - v).abs();
            max_err = max_err.max(err);
        }
        // [0,1] 范围内 f16 精度应足够（最大误差 < 0.001）
        assert!(max_err < 0.001, "f16 max error in [0,1]: {}", max_err);
    }

    // ===== OPT-4 SIMD f16 距离核测试 =====

    #[test]
    fn l2_f16_mixed_simd_matches_scalar() {
        if !is_avx512_and_f16c_supported() && !is_avx2_and_f16c_supported() {
            eprintln!("F16C not supported, skipping SIMD test");
            return;
        }
        let query: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let db_f32: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) * 0.9).collect();
        let db_f16 = f32_to_f16_slice(&db_f32);

        let scalar = l2_f16_mixed(&query, &db_f16);
        let simd = l2_f16_mixed_simd(&query, &db_f16);
        let rel_err = (scalar - simd).abs() / scalar.max(1e-6);
        assert!(rel_err < 1e-4, "scalar={} simd={} rel_err={}", scalar, simd, rel_err);
    }

    #[test]
    fn l2_f16_mixed_avx512_dim128() {
        if !is_avx512_and_f16c_supported() {
            return;
        }
        // SIFT1M dim=128 = 8 × 16，测试主循环
        let query: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let db_f32: Vec<f32> = (0..128).map(|i| (i as f32) * 1.1).collect();
        let db_f16 = f32_to_f16_slice(&db_f32);

        let result = unsafe { l2_f16_mixed_avx512(&query, &db_f16) };
        // 手动计算 f16 精度下的距离
        let expected: f32 = query.iter().zip(db_f16.iter())
            .map(|(q, d)| { let diff = q - d.to_f32(); diff * diff })
            .sum();
        let rel_err = (result - expected).abs() / expected.max(1.0);
        assert!(rel_err < 1e-5, "result={} expected={} rel_err={}", result, expected, rel_err);
    }

    #[test]
    fn l2_f16_mixed_avx512_small_vec() {
        if !is_avx512_and_f16c_supported() {
            return;
        }
        // 维度 < 16，测试尾部处理
        let query = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let db_f32 = [4.0f32, 5.0, 6.0, 7.0, 8.0];
        let db_f16 = f32_to_f16_slice(&db_f32);

        let result = unsafe { l2_f16_mixed_avx512(&query, &db_f16) };
        let expected: f32 = query.iter().zip(db_f16.iter())
            .map(|(q, d)| { let diff = q - d.to_f32(); diff * diff })
            .sum();
        assert!((result - expected).abs() < 1e-4, "result={} expected={}", result, expected);
    }

    #[test]
    fn l2_f16_mixed_avx2_dim128() {
        if !is_avx2_and_f16c_supported() {
            return;
        }
        let query: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let db_f32: Vec<f32> = (0..128).map(|i| (i as f32) * 1.1).collect();
        let db_f16 = f32_to_f16_slice(&db_f32);

        let result = unsafe { l2_f16_mixed_avx2(&query, &db_f16) };
        let expected: f32 = query.iter().zip(db_f16.iter())
            .map(|(q, d)| { let diff = q - d.to_f32(); diff * diff })
            .sum();
        let rel_err = (result - expected).abs() / expected.max(1.0);
        assert!(rel_err < 1e-5, "result={} expected={} rel_err={}", result, expected, rel_err);
    }
}
