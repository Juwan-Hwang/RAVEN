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

/// f16 类型（用 u16 存储，便于 SIMD 和序列化）
#[derive(Debug, Clone, Copy, PartialEq)]
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
            let mant_rounded = (mant >> shift) + if (mant & (half - 1)) > half
                || ((mant & half) > 0 && (mant & (half - 1)) >= half) {
                1
            } else {
                0
            };
            return Self(sign | mant_rounded as u16);
        }

        // 正规数
        let mant_f16 = (mant >> 13) as u16;
        // round-to-nearest-even
        let rem = mant & 0x1FFF;
        let half = 0x1000;
        let round = if rem > half || (rem == half && (mant_f16 & 1) == 1) {
            1
        } else {
            0
        };
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
        sum += d * d;
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
        sum += d * d;
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
        sum += diff * diff;
    }
    sum
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
}
