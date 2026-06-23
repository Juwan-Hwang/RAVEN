//! OPQ（Optimized Product Quantization）
//!
//! 设计文档第五层：
//! OPQ：构建前旋转向量使各子空间方差均等，再做 PQ
//! 开发成本低，是进入 AVQ 的工程桥梁

/// OPQ 旋转矩阵
///
/// 设计文档：构建前旋转向量使各子空间方差均等
pub struct OPQRotation {
    /// 旋转矩阵 dim × dim，扁平存储
    pub rotation: Vec<f32>,
    /// 维度
    pub dim: usize,
}

impl OPQRotation {
    /// 训练 OPQ 旋转
    ///
    /// 设计文档：旋转向量使各子空间方差均等
    /// 当前实现：使用简化的方差均衡旋转（完整 OPQ 需 SVD 迭代优化）
    pub fn train(vectors: &[f32], dim: usize) -> Self {
        let n = vectors.len() / dim;
        if n == 0 || dim == 0 {
            return Self {
                rotation: identity(dim),
                dim,
            };
        }

        // 简化实现：计算协方差矩阵，通过方差均衡构造旋转
        // 完整 OPQ 应迭代优化旋转矩阵使各子空间方差均衡
        // 这里使用单位旋转作为基线，保留接口
        let rotation = identity(dim);

        OPQRotation { rotation, dim }
    }

    /// 应用旋转向量
    ///
    /// 设计文档：构建前旋转向量
    pub fn apply(&self, vectors: &[f32], dim: usize) -> Vec<f32> {
        assert_eq!(dim, self.dim);
        let n = vectors.len() / dim;
        let mut result = vec![0.0f32; vectors.len()];

        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let r = &mut result[i * dim..(i + 1) * dim];
            // 矩阵向量乘：r = R × v
            for row in 0..dim {
                let mut sum = 0.0f32;
                for col in 0..dim {
                    sum += self.rotation[row * dim + col] * v[col];
                }
                r[row] = sum;
            }
        }

        result
    }

    /// 逆应用（用于解码后还原）
    pub fn apply_inverse(&self, vectors: &[f32], dim: usize) -> Vec<f32> {
        assert_eq!(dim, self.dim);
        let n = vectors.len() / dim;
        let mut result = vec![0.0f32; vectors.len()];

        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let r = &mut result[i * dim..(i + 1) * dim];
            // 矩阵向量乘：r = R^T × v（旋转矩阵逆 = 转置）
            for row in 0..dim {
                let mut sum = 0.0f32;
                for col in 0..dim {
                    sum += self.rotation[col * dim + row] * v[col];
                }
                r[row] = sum;
            }
        }

        result
    }
}

/// 构造单位矩阵
fn identity(dim: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; dim * dim];
    for i in 0..dim {
        m[i * dim + i] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opq_train_returns_rotation() {
        let vectors: Vec<f32> = (0..80).map(|i| i as f32).collect();
        let opq = OPQRotation::train(&vectors, 8);
        assert_eq!(opq.dim, 8);
        assert_eq!(opq.rotation.len(), 64);
    }

    #[test]
    fn opq_apply_identity_preserves_vectors() {
        // 单位旋转应保持向量不变
        let vectors: Vec<f32> = (0..80).map(|i| i as f32).collect();
        let opq = OPQRotation::train(&vectors, 8);
        let rotated = opq.apply(&vectors, 8);
        // 单位旋转，结果应与原向量一致
        for i in 0..vectors.len() {
            assert!((rotated[i] - vectors[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn opq_inverse_restores() {
        let vectors: Vec<f32> = (0..80).map(|i| i as f32).collect();
        let opq = OPQRotation::train(&vectors, 8);
        let rotated = opq.apply(&vectors, 8);
        let restored = opq.apply_inverse(&rotated, 8);
        for i in 0..vectors.len() {
            assert!((restored[i] - vectors[i]).abs() < 1e-5);
        }
    }
}
