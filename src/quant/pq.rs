//! PQ（Product Quantization）基线
//!
//! 设计文档第五层：
//! PQ（基线）：优化 reconstruction loss，作为对照组
//! OPQ：构建前旋转向量使各子空间方差均等，再做 PQ
//!
//! PQ 将向量切分为 M 个子空间，每个子空间独立 K-means 聚类，
//! 用聚类中心 ID 编码向量，实现压缩。

/// PQ codebook
///
/// 设计文档：PQ 优化 reconstruction loss，作为对照组
pub struct PQCodebook {
    /// 子空间数 M
    pub m: usize,
    /// 每个子空间的聚类中心数 K（通常 256）
    pub k: usize,
    /// 维度
    pub dim: usize,
    /// 每个子空间的维度 dim / M
    pub sub_dim: usize,
    /// codebook：M × K × sub_dim，扁平存储
    /// centers[m * K * sub_dim + k * sub_dim + d]
    pub centers: Vec<f32>,
}

impl PQCodebook {
    /// 训练 PQ codebook
    ///
    /// vectors: 扁平存储的向量
    /// dim: 维度
    /// m: 子空间数
    /// k: 每个子空间聚类中心数
    pub fn train(vectors: &[f32], dim: usize, m: usize, k: usize) -> Self {
        assert!(dim % m == 0, "dim {} must be divisible by m {}", dim, m);
        let sub_dim = dim / m;
        let n = vectors.len() / dim;

        let mut centers = vec![0.0f32; m * k * sub_dim];

        // 对每个子空间做 K-means
        // OPT-9 实验结论：k-means++ 在 SIFT 上 loss 仅改善 1.92% 但耗时 15 倍，
        // 采样 k-means++ loss 反而更差（-0.06%）。SIFT 数据分布均匀，"取前 k 个点"已足够。
        // 保持"取前 k 个点"初始化，不接入 k-means++。
        for sub in 0..m {
            // 提取子空间向量
            let sub_vectors: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let start = i * dim + sub * sub_dim;
                    vectors[start..start + sub_dim].to_vec()
                })
                .collect();

            let sub_centers = kmeans(&sub_vectors, k, 10);
            // 写入 codebook
            for (ki, center) in sub_centers.iter().enumerate() {
                for (d, &v) in center.iter().enumerate() {
                    centers[sub * k * sub_dim + ki * sub_dim + d] = v;
                }
            }
        }

        PQCodebook { m, k, dim, sub_dim, centers }
    }

    /// 编码单个向量
    ///
    /// 返回 M 个聚类中心 ID
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);
        let mut codes = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let sub_vec = &vector[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            let mut best_k = 0;
            let mut best_dist = f32::MAX;
            for ki in 0..self.k {
                let center = &self.centers[sub * self.k * self.sub_dim + ki * self.sub_dim..
                    sub * self.k * self.sub_dim + (ki + 1) * self.sub_dim];
                let dist = l2_sq(sub_vec, center);
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }
            codes.push(best_k as u8);
        }
        codes
    }

    /// 解码：从编码恢复近似向量
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.dim];
        for (sub, &code) in codes.iter().enumerate() {
            let ki = code as usize;
            for d in 0..self.sub_dim {
                result[sub * self.sub_dim + d] =
                    self.centers[sub * self.k * self.sub_dim + ki * self.sub_dim + d];
            }
        }
        result
    }

    /// 预计算 LUT（Asymmetric Distance Computation）
    ///
    /// lut[m * K + k] = L2_sq(query_sub_m, centroid[m][k])
    /// 大小：M × K × sizeof(f32)
    /// SIFT-128 M=32 K=256: 32×256×4 = 32KB（L1/L2 边界）
    ///
    /// 每次查询预计算一次，后续所有候选距离只需 table lookup
    pub fn compute_lut(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);
        let mut lut = vec![0.0f32; self.m * self.k];
        for sub in 0..self.m {
            let q_sub = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            for ki in 0..self.k {
                let center = &self.centers[sub * self.k * self.sub_dim + ki * self.sub_dim..
                    sub * self.k * self.sub_dim + (ki + 1) * self.sub_dim];
                lut[sub * self.k + ki] = l2_sq(q_sub, center);
            }
        }
        lut
    }

    /// 计算量化误差（reconstruction loss）
    ///
    /// 设计文档：PQ 优化 reconstruction loss
    pub fn reconstruction_loss(&self, vectors: &[f32]) -> f32 {
        let n = vectors.len() / self.dim;
        let mut total = 0.0f32;
        for i in 0..n {
            let v = &vectors[i * self.dim..(i + 1) * self.dim];
            let codes = self.encode(v);
            let decoded = self.decode(&codes);
            total += l2_sq(v, &decoded);
        }
        total / n as f32
    }
}

/// PQ 量化器
pub struct PQ;

impl PQ {
    /// 训练并编码
    pub fn fit_transform(vectors: &[f32], dim: usize, m: usize, k: usize) -> (PQCodebook, Vec<Vec<u8>>) {
        let codebook = PQCodebook::train(vectors, dim, m, k);
        let n = vectors.len() / dim;
        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| codebook.encode(&vectors[i * dim..(i + 1) * dim]))
            .collect();
        (codebook, codes)
    }
}

/// PQ8 量化数据集（K=256, 8-bit codes, LUT-ADC 距离）
///
/// Phase 1 Step 1：复用 PQCodebook 训练，添加扁平 codes 存储 + LUT-ADC 距离。
///
/// 架构（SIFT-128, M=32, K=256）：
/// - 训练：M 个子空间各做 K=256 k-means 聚类（复用 PQCodebook::train）
/// - 编码：每子空间用 8-bit 码本 ID，1 byte/subspace → M bytes/vector
/// - 查询：预计算 LUT (M×K f32 = 32KB)，候选距离 = Σ LUT[m][code[m]]
/// - 码大小：32 bytes/vector（比 f32 512B 小 16x，比 SQ8 128B 小 4x）
///
/// K=256 精度接近 SQ8，但带宽降低 4x（32B vs 128B per vector），
/// LUT-ADC 仅需 M 次 table lookup（32 次 vs SQ8 128 次）。
pub struct PQ8Dataset {
    /// 扁平 codes: n × M bytes
    pub codes: Vec<u8>,
    /// 码本
    pub codebook: PQCodebook,
    /// 维度
    pub dim: usize,
    /// 向量数
    pub n: usize,
    /// 每向量码字节数 = M
    pub code_bytes: usize,
}

impl PQ8Dataset {
    /// 从 f32 数据集构建 PQ8 量化数据集
    ///
    /// data: 扁平 f32 数据
    /// dim: 维度（必须能被 M 整除）
    /// m: 子空间数（SIFT-128 → M=32, sub_dim=4）
    pub fn build(data: &[f32], dim: usize, m: usize) -> Self {
        let k = 256;
        let codebook = PQCodebook::train(data, dim, m, k);
        let n = data.len() / dim;
        let mut codes = vec![0u8; n * m];
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let encoded = codebook.encode(v);
            codes[i * m..(i + 1) * m].copy_from_slice(&encoded);
        }
        Self { codes, codebook, dim, n, code_bytes: m }
    }

    /// 获取第 idx 个向量的 code 引用
    #[inline(always)]
    pub fn code(&self, idx: usize) -> &[u8] {
        &self.codes[idx * self.code_bytes..(idx + 1) * self.code_bytes]
    }

    /// ADC 距离：lut[m][code[m]] 之和
    ///
    /// lut: 预计算的 M×K 查找表（由 PQCodebook::compute_lut 生成）
    /// codes: M bytes 的 8-bit PQ codes
    #[inline(always)]
    pub fn adc_distance(lut: &[f32], codes: &[u8], m: usize, k: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..m {
            sum += lut[i * k + codes[i] as usize];
        }
        sum
    }
}

/// 简单 K-means 聚类（取前 k 个点初始化）
fn kmeans(data: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n);

    // 初始化：取前 k 个点作为中心
    let mut centers: Vec<Vec<f32>> = data[..k].to_vec();
    if centers.is_empty() {
        return vec![vec![0.0; dim]];
    }

    for _ in 0..iterations {
        // 分配
        let mut assignments = vec![0usize; n];
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (j, center) in centers.iter().enumerate() {
                let d = l2_sq(point, center);
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

/// 平方 L2 距离
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_train_and_encode() {
        // 20 个 8 维向量
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = PQCodebook::train(&vectors, 8, 4, 16);
        assert_eq!(codebook.m, 4);
        assert_eq!(codebook.k, 16);
        assert_eq!(codebook.sub_dim, 2);

        let codes = codebook.encode(&vectors[0..8]);
        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn pq_decode_reconstructs() {
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = PQCodebook::train(&vectors, 8, 4, 16);
        let codes = codebook.encode(&vectors[0..8]);
        let decoded = codebook.decode(&codes);
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn pq_reconstruction_loss_positive() {
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = PQCodebook::train(&vectors, 8, 4, 16);
        let loss = codebook.reconstruction_loss(&vectors);
        assert!(loss >= 0.0);
    }
}
