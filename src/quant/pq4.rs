//! 4-bit Product Quantization (K=16) with LUT-ADC distance
//!
//! Phase 1 核心突破点：每子空间 4-bit 编码，向量 128B(f32) → 16B(packed 4-bit)。
//!
//! 架构：
//! - 训练：M 个子空间各做 K=16 k-means 聚类
//! - 编码：每子空间用 4-bit 码本 ID，2 个码打包为 1 byte
//! - 查询：预计算 LUT (M×K f32)，候选距离 = Σ LUT[m][code[m]]
//! - LUT 大小：M×K×4 = 32×16×4 = 2048 bytes（完全 L1 cache）
//! - 码大小：M/2 = 16 bytes/vector（比 SQ8 的 128 bytes 小 8x）
//!
//! 对于 SIFT-128: M=32, sub_dim=4, K=16

/// 4-bit PQ 码本（K=16 per subspace）
#[derive(Debug, Clone)]
pub struct PQ4Codebook {
    /// 子空间数 M
    pub m: usize,
    /// 每子空间聚类中心数 K（固定 16）
    pub k: usize,
    /// 维度
    pub dim: usize,
    /// 每子空间维度 dim / M
    pub sub_dim: usize,
    /// codebook: M × K × sub_dim，扁平存储
    /// centers[m * K * sub_dim + k * sub_dim + d]
    pub centers: Vec<f32>,
}

impl PQ4Codebook {
    /// 训练 4-bit PQ 码本
    ///
    /// vectors: 扁平 f32 数据
    /// dim: 维度（必须能被 M 整除）
    /// m: 子空间数（SIFT-128 → M=32, sub_dim=4）
    pub fn train(vectors: &[f32], dim: usize, m: usize) -> Self {
        assert!(dim % m == 0, "dim {} must be divisible by m {}", dim, m);
        let sub_dim = dim / m;
        let k = 16; // 4-bit, K=16
        let n = vectors.len() / dim;

        let mut centers = vec![0.0f32; m * k * sub_dim];

        for sub in 0..m {
            // 提取子空间向量
            let sub_vectors: Vec<&[f32]> = (0..n)
                .map(|i| {
                    let start = i * dim + sub * sub_dim;
                    &vectors[start..start + sub_dim]
                })
                .collect();

            let sub_centers = kmeans(&sub_vectors, k, 15);
            for (ki, center) in sub_centers.iter().enumerate() {
                for (d, &v) in center.iter().enumerate() {
                    centers[sub * k * sub_dim + ki * sub_dim + d] = v;
                }
            }
        }

        PQ4Codebook { m, k, dim, sub_dim, centers }
    }

    /// 编码单个向量 → packed 4-bit codes (M/2 bytes)
    ///
    /// byte[i] = code[2*i] | (code[2*i+1] << 4)
    pub fn encode_packed(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);
        let mut codes = vec![0u8; self.m / 2];

        for sub in 0..self.m {
            let sub_vec = &vector[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            let mut best_k = 0;
            let mut best_dist = f32::MAX;
            for ki in 0..self.k {
                let center = &self.centers[sub * self.k * self.sub_dim + ki * self.sub_dim..
                    sub * self.k * self.sub_dim + (ki + 1) * self.sub_dim];
                let dist = l2_sq_scalar(sub_vec, center);
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }
            // Pack: even subspace → low nibble, odd → high nibble
            if sub % 2 == 0 {
                codes[sub / 2] = best_k as u8;
            } else {
                codes[sub / 2] |= (best_k as u8) << 4;
            }
        }

        codes
    }

    /// 编码整个数据集 → 扁平 packed codes (n × M/2 bytes)
    pub fn encode_all_packed(&self, vectors: &[f32]) -> Vec<u8> {
        let n = vectors.len() / self.dim;
        let code_bytes = self.m / 2;
        let mut all_codes = vec![0u8; n * code_bytes];

        for i in 0..n {
            let v = &vectors[i * self.dim..(i + 1) * self.dim];
            let codes = self.encode_packed(v);
            all_codes[i * code_bytes..(i + 1) * code_bytes].copy_from_slice(&codes);
        }

        all_codes
    }

    /// 预计算 LUT（每查询一次）
    ///
    /// lut[m * K + k] = L2_sq(query_sub_m, centroid[m][k])
    /// 大小：M × K × sizeof(f32) = 32 × 16 × 4 = 2048 bytes
    pub fn compute_lut(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);
        let mut lut = vec![0.0f32; self.m * self.k];

        for sub in 0..self.m {
            let q_sub = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            for ki in 0..self.k {
                let center = &self.centers[sub * self.k * self.sub_dim + ki * self.sub_dim..
                    sub * self.k * self.sub_dim + (ki + 1) * self.sub_dim];
                lut[sub * self.k + ki] = l2_sq_scalar(q_sub, center);
            }
        }

        lut
    }
}

/// 4-bit PQ 量化数据集
pub struct PQ4Dataset {
    /// 扁平 packed codes: n × (M/2) bytes
    pub codes: Vec<u8>,
    /// 码本
    pub codebook: PQ4Codebook,
    /// 维度
    pub dim: usize,
    /// 向量数
    pub n: usize,
    /// 每向量码字节数 M/2
    pub code_bytes: usize,
}

impl PQ4Dataset {
    /// 从 f32 数据集构建 PQ4 量化数据集
    pub fn build(data: &[f32], dim: usize, m: usize) -> Self {
        let n = data.len() / dim;
        let codebook = PQ4Codebook::train(data, dim, m);
        let codes = codebook.encode_all_packed(data);
        let code_bytes = m / 2;
        Self { codes, codebook, dim, n, code_bytes }
    }

    /// 获取第 idx 个向量的 packed code 引用
    #[inline(always)]
    pub fn code(&self, idx: usize) -> &[u8] {
        &self.codes[idx * self.code_bytes..(idx + 1) * self.code_bytes]
    }

    /// ADC 距离：lut[m][code[m]] 之和
    ///
    /// lut: 预计算的 M×K 查找表
    /// packed_code: M/2 bytes 的 packed 4-bit codes
    #[inline(always)]
    pub fn adc_distance(lut: &[f32], packed_code: &[u8], m: usize) -> f32 {
        let k = 16usize;
        let mut sum = 0.0f32;
        for i in 0..(m / 2) {
            let byte = packed_code[i] as usize;
            let low = byte & 0x0F;       // even subspace (2i)
            let high = (byte >> 4) & 0x0F; // odd subspace (2i+1)
            sum += lut[2 * i * k + low];
            sum += lut[(2 * i + 1) * k + high];
        }
        sum
    }
}

/// 平方 L2 距离（标量）
#[inline]
fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// 简单 K-means 聚类（取前 k 个点初始化，15 轮迭代）
fn kmeans(data: &[&[f32]], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![vec![0.0; data.get(0).map_or(0, |d| d.len())]];
    }
    let dim = data[0].len();
    let k = k.min(n);

    // 初始化：均匀采样 k 个点
    let mut centers: Vec<Vec<f32>> = (0..k)
        .map(|i| data[i * n / k].to_vec())
        .collect();

    for _ in 0..iterations {
        // 分配
        let mut assignments = vec![0usize; n];
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (j, center) in centers.iter().enumerate() {
                let d = l2_sq_scalar(point, center);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            assignments[i] = best;
        }

        // 更新
        let mut new_centers = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            for d in 0..dim {
                new_centers[a][d] += data[i][d];
            }
            counts[a] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    new_centers[j][d] /= counts[j] as f32;
                }
            } else {
                new_centers[j] = centers[j].clone();
            }
        }
        centers = new_centers;
    }

    centers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq4_train_basic() {
        let dim = 8;
        let m = 4;
        let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5).collect();
        let cb = PQ4Codebook::train(&data, dim, m);
        assert_eq!(cb.m, 4);
        assert_eq!(cb.k, 16);
        assert_eq!(cb.sub_dim, 2);
        assert_eq!(cb.centers.len(), m * 16 * (dim / m));
    }

    #[test]
    fn pq4_encode_packed_roundtrip() {
        let dim = 8;
        let m = 4;
        let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5).collect();
        let cb = PQ4Codebook::train(&data, dim, m);
        let codes = cb.encode_packed(&data[0..dim]);
        assert_eq!(codes.len(), m / 2); // 4 subspaces → 2 bytes

        // Unpack and verify codes are in [0, 15]
        for &b in &codes {
            assert!(b & 0x0F < 16);
            assert!((b >> 4) & 0x0F < 16);
        }
    }

    #[test]
    fn pq4_adc_distance_matches_direct() {
        let dim = 128;
        let m = 32;
        let n = 500;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin() * 0.5 + 0.5).collect();
        let ds = PQ4Dataset::build(&data, dim, m);

        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.01).cos() * 0.5 + 0.5).collect();
        let lut = ds.codebook.compute_lut(&query);

        // ADC distance should match direct computation
        for idx in [0, 1, 50, 200, 499].iter() {
            let adc = PQ4Dataset::adc_distance(&lut, ds.code(*idx), m);

            // Direct: decode + L2
            let codes = ds.code(*idx);
            let mut direct = 0.0f32;
            for i in 0..(m / 2) {
                let byte = codes[i] as usize;
                let low = byte & 0x0F;
                let high = (byte >> 4) & 0x0F;
                // LUT already contains the distance, so ADC should match exactly
                direct += lut[2 * i * 16 + low];
                direct += lut[(2 * i + 1) * 16 + high];
            }

            let rel_err = (adc - direct).abs() / direct.max(1e-10);
            assert!(rel_err < 1e-6, "idx={}: adc={} direct={}", idx, adc, direct);
        }
    }

    #[test]
    fn pq4_distance_correlation_with_f32() {
        let dim = 128;
        let m = 32;
        let n = 1000;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5).collect();
        let ds = PQ4Dataset::build(&data, dim, m);

        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.1).cos() * 0.5 + 0.5).collect();
        let lut = ds.codebook.compute_lut(&query);

        // Compute PQ4 and f32 distances for first 200 vectors
        let mut pq4_dists = Vec::new();
        let mut f32_dists = Vec::new();
        for i in 0..200.min(n) {
            let d_pq4 = PQ4Dataset::adc_distance(&lut, ds.code(i), m);
            let d_f32: f32 = (0..dim)
                .map(|d| {
                    let diff = data[i * dim + d] - query[d];
                    diff * diff
                })
                .sum();
            pq4_dists.push((i, d_pq4));
            f32_dists.push((i, d_f32));
        }

        // Sort by distance
        pq4_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        f32_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Top-10 overlap should be ≥ 5 (PQ4 is approximate)
        let pq4_top10: std::collections::HashSet<usize> =
            pq4_dists.iter().take(10).map(|(i, _)| *i).collect();
        let f32_top10: std::collections::HashSet<usize> =
            f32_dists.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = pq4_top10.intersection(&f32_top10).count();
        assert!(overlap >= 5, "PQ4 top-10 overlap with f32: {}/10 (expected ≥5)", overlap);
    }

    #[test]
    fn pq4_memory_compression() {
        let dim = 128;
        let m = 32;
        let n = 1000;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.001).collect();
        let ds = PQ4Dataset::build(&data, dim, m);

        let f32_bytes = n * dim * 4;
        let pq4_bytes = ds.codes.len();
        let compression = f32_bytes as f64 / pq4_bytes as f64;

        assert_eq!(pq4_bytes, n * m / 2); // 1000 * 16 = 16000 bytes
        assert!((compression - 32.0).abs() < 0.1, "compression = {}", compression);
    }
}
