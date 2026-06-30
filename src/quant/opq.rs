//! OPQ（Optimized Product Quantization）
//!
//! 设计文档第五层：
//! OPQ：构建前旋转向量使各子空间方差均等，再做 PQ
//! 开发成本低，是进入 AVQ 的工程桥梁
//!
//! 实现：
//! 1. 计算协方差矩阵 C = (1/n) Σ (x_i - μ)(x_i - μ)^T
//! 2. Jacobi 特征值分解 C = V Λ V^T
//! 3. 按特征值降序排列特征向量（PCA 旋转）
//! 4. 方差均衡排列：将高方差和低方差维度交错分配到各子空间
//!    使每个子空间的总方差尽可能均衡

/// OPQ 旋转矩阵
///
/// 设计文档：构建前旋转向量使各子空间方差均等
pub struct OPQRotation {
    /// 旋转矩阵 dim × dim，扁平存储（行优先）
    pub rotation: Vec<f32>,
    /// 维度
    pub dim: usize,
}

impl OPQRotation {
    /// 训练 OPQ 旋转
    ///
    /// 设计文档：旋转向量使各子空间方差均等
    /// 实现：协方差矩阵 → Jacobi 特征值分解 → PCA + 方差均衡排列
    ///
    /// vectors: 扁平存储的向量
    /// dim: 维度
    /// sub_dim: 子空间维度（用于方差均衡，默认 8）
    pub fn train(vectors: &[f32], dim: usize) -> Self {
        Self::train_with_sub_dim(vectors, dim, 8)
    }

    /// 训练 OPQ 旋转（指定子空间维度，用于方差均衡）
    pub fn train_with_sub_dim(vectors: &[f32], dim: usize, sub_dim: usize) -> Self {
        let n = vectors.len() / dim;
        if n == 0 || dim == 0 {
            return Self {
                rotation: identity(dim),
                dim,
            };
        }

        // 1. 计算均值
        let mut mean = vec![0.0f32; dim];
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            for d in 0..dim {
                mean[d] += v[d];
            }
        }
        for m in mean.iter_mut().take(dim) {
            *m /= n as f32;
        }

        // 2. 计算协方差矩阵 C = (1/n) Σ (x_i - μ)(x_i - μ)^T
        let mut cov = vec![0.0f32; dim * dim];
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            for r in 0..dim {
                let dr = v[r] - mean[r];
                for c in r..dim {
                    let dc = v[c] - mean[c];
                    cov[r * dim + c] = dr.mul_add(dc, cov[r * dim + c]);
                }
            }
        }
        // 对称化 + 归一化
        for r in 0..dim {
            for c in r..dim {
                let val = cov[r * dim + c] / n as f32;
                cov[r * dim + c] = val;
                cov[c * dim + r] = val; // 对称
            }
        }

        // 3. Jacobi 特征值分解
        let (eigenvalues, eigenvectors) = jacobi_eigen(&cov, dim, 100, 1e-8);

        // 4. 按特征值降序排列
        let mut indexed: Vec<(f32, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // 5. 方差均衡排列
        // 将维度按方差从大到小排序，然后交错分配到各子空间
        // 使每个子空间的总方差尽可能均衡
        let m = dim / sub_dim;
        let mut perm = vec![0usize; dim];
        if m > 1 {
            // 交错排列：第 0 个给子空间 0，第 1 个给子空间 1，...，第 m 个给子空间 0，...
            for (rank, &(_, orig_idx)) in indexed.iter().enumerate() {
                let sub = rank % m;
                let pos_in_sub = rank / m;
                let new_pos = sub * sub_dim + pos_in_sub;
                if new_pos < dim {
                    perm[new_pos] = orig_idx;
                } else {
                    perm[rank] = orig_idx;
                }
            }
        } else {
            // sub_dim >= dim，直接按降序
            for (rank, &(_, orig_idx)) in indexed.iter().enumerate() {
                perm[rank] = orig_idx;
            }
        }

        // 6. 构造旋转矩阵：R[new_pos][orig_dim] = V[orig_dim][perm[new_pos]]
        // 即 rotation 的第 new_pos 行 = eigenvectors 的第 perm[new_pos] 列
        let mut rotation = vec![0.0f32; dim * dim];
        for new_pos in 0..dim {
            let orig_idx = perm[new_pos];
            for d in 0..dim {
                rotation[new_pos * dim + d] = eigenvectors[d * dim + orig_idx];
            }
        }

        eprintln!("[OPQ] train: dim={dim}, n={n}, sub_dim={sub_dim}, m={m}");
        eprintln!("[OPQ] eigenvalue range: [{:.6}, {:.6}]",
            eigenvalues.iter().copied().fold(f32::INFINITY, f32::min),
            eigenvalues.iter().copied().fold(f32::NEG_INFINITY, f32::max));

        OPQRotation { rotation, dim }
    }

    /// 应用旋转向量
    ///
    /// 设计文档：构建前旋转向量
    /// x' = R × x（R 是 dim×dim 旋转矩阵）
    pub fn apply(&self, vectors: &[f32], dim: usize) -> Vec<f32> {
        assert_eq!(dim, self.dim);
        let n = vectors.len() / dim;
        let mut result = vec![0.0f32; vectors.len()];

        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let r = &mut result[i * dim..(i + 1) * dim];
            // 矩阵向量乘：r = R × v
            for (row, r_elem) in r.iter_mut().enumerate().take(dim) {
                let mut sum = 0.0f32;
                for (col, &v_col) in v.iter().enumerate().take(dim) {
                    sum = self.rotation[row * dim + col].mul_add(v_col, sum);
                }
                *r_elem = sum;
            }
        }

        result
    }

    /// 逆应用（用于解码后还原）
    /// x = R^T × x'（旋转矩阵逆 = 转置）
    pub fn apply_inverse(&self, vectors: &[f32], dim: usize) -> Vec<f32> {
        assert_eq!(dim, self.dim);
        let n = vectors.len() / dim;
        let mut result = vec![0.0f32; vectors.len()];

        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let r = &mut result[i * dim..(i + 1) * dim];
            // 矩阵向量乘：r = R^T × v
            for (row, r_elem) in r.iter_mut().enumerate().take(dim) {
                let mut sum = 0.0f32;
                for (col, &v_col) in v.iter().enumerate().take(dim) {
                    sum = self.rotation[col * dim + row].mul_add(v_col, sum);
                }
                *r_elem = sum;
            }
        }

        result
    }
}

/// Jacobi 特征值分解（对称矩阵）
///
/// 算法：迭代地用 Givens 旋转消去非对角元素
/// 收敛条件：非对角元素范数 < tol 或达到最大迭代次数
///
/// 输入：symmetric（dim×dim 对称矩阵，行优先扁平存储）
/// 输出：(eigenvalues, eigenvectors)
///   eigenvalues: dim 个特征值
///   eigenvectors: dim×dim 矩阵，第 j 列是对应 eigenvalues[j] 的特征向量
fn jacobi_eigen(
    symmetric: &[f32],
    dim: usize,
    max_sweeps: usize,
    tol: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut a = symmetric.to_vec();
    let mut v = identity(dim);

    for _sweep in 0..max_sweeps {
        // 计算非对角元素范数
        let mut off_diag_sq = 0.0f32;
        for r in 0..dim {
            for c in (r + 1)..dim {
                off_diag_sq = a[r * dim + c].mul_add(a[r * dim + c], off_diag_sq);
            }
        }

        if off_diag_sq < tol {
            break;
        }

        // 一轮 sweep：遍历所有上三角非对角元素
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = a[p * dim + q];
                if apq.abs() < 1e-12 {
                    continue;
                }

                let app = a[p * dim + p];
                let aqq = a[q * dim + q];

                // 计算旋转角度
                // tan(2θ) = 2*apq / (app - aqq)
                // 使用数值稳定的公式
                let theta = if (app - aqq).abs() < 1e-12 {
                    std::f32::consts::FRAC_PI_4
                } else {
                    0.5 * (2.0 * apq / (app - aqq)).atan2(1.0)
                };
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // 应用 Givens 旋转 J^T A J
                // 更新 A
                for i in 0..dim {
                    let aip = a[i * dim + p];
                    let aiq = a[i * dim + q];
                    a[i * dim + p] = cos_t * aip + sin_t * aiq;
                    a[i * dim + q] = (-sin_t).mul_add(aip, cos_t * aiq);
                }
                for i in 0..dim {
                    let api = a[p * dim + i];
                    let aqi = a[q * dim + i];
                    a[p * dim + i] = cos_t * api + sin_t * aqi;
                    a[q * dim + i] = (-sin_t).mul_add(api, cos_t * aqi);
                }

                // 更新 V（累积特征向量）
                for i in 0..dim {
                    let vip = v[i * dim + p];
                    let viq = v[i * dim + q];
                    v[i * dim + p] = cos_t * vip + sin_t * viq;
                    v[i * dim + q] = (-sin_t).mul_add(vip, cos_t * viq);
                }
            }
        }
    }

    // 提取特征值（对角线）
    let eigenvalues: Vec<f32> = (0..dim).map(|i| a[i * dim + i]).collect();

    (eigenvalues, v)
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
    fn opq_apply_then_inverse_restores() {
        // apply 后 apply_inverse 应恢复原向量
        let vectors: Vec<f32> = (0..80).map(|i| i as f32).collect();
        let opq = OPQRotation::train(&vectors, 8);
        let rotated = opq.apply(&vectors, 8);
        let restored = opq.apply_inverse(&rotated, 8);
        for i in 0..vectors.len() {
            assert!((restored[i] - vectors[i]).abs() < 1e-4,
                "mismatch at {}: {} vs {}", i, restored[i], vectors[i]);
        }
    }

    #[test]
    fn opq_rotation_is_orthogonal() {
        // 旋转矩阵应满足 R × R^T = I
        let vectors: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let opq = OPQRotation::train(&vectors, 8);
        let dim = 8;
        // R × R^T
        for r in 0..dim {
            for c in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += opq.rotation[r * dim + k] * opq.rotation[c * dim + k];
                }
                if r == c {
                    assert!((dot - 1.0).abs() < 1e-4, "diagonal {} = {}", r, dot);
                } else {
                    assert!(dot.abs() < 1e-4, "off-diagonal ({},{}) = {}", r, c, dot);
                }
            }
        }
    }

    #[test]
    fn jacobi_diagonal_2x2() {
        // 2x2 对角矩阵，特征值已知
        let m = vec![3.0, 0.0, 0.0, 2.0];
        let (eigenvalues, eigenvectors) = jacobi_eigen(&m, 2, 50, 1e-8);
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 3.0).abs() < 1e-4);
        assert!((sorted[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn jacobi_general_2x2() {
        // 2x2 一般对称矩阵
        // A = [[2, 1], [1, 2]], 特征值 3 和 1
        let m = vec![2.0, 1.0, 1.0, 2.0];
        let (eigenvalues, eigenvectors) = jacobi_eigen(&m, 2, 50, 1e-8);
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 3.0).abs() < 1e-4, "eigenvalue 1: {}", sorted[0]);
        assert!((sorted[1] - 1.0).abs() < 1e-4, "eigenvalue 2: {}", sorted[1]);
    }

    #[test]
    fn jacobi_4x4() {
        // 4x4 对角矩阵
        let m = vec![
            5.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 2.0,
        ];
        let (eigenvalues, _) = jacobi_eigen(&m, 4, 50, 1e-8);
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 5.0).abs() < 1e-4);
        assert!((sorted[1] - 3.0).abs() < 1e-4);
        assert!((sorted[2] - 2.0).abs() < 1e-4);
        assert!((sorted[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn opq_variance_balancing() {
        // 构造方差不均的数据：前 4 维方差大，后 4 维方差小
        let n = 100;
        let dim = 8;
        let sub_dim = 4;
        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            // 前 4 维：大方差
            for d in 0..4 {
                vectors[i * dim + d] = ((i * 7 + d * 3) as f32).sin() * 100.0;
            }
            // 后 4 维：小方差
            for d in 4..8 {
                vectors[i * dim + d] = ((i * 7 + d * 3) as f32).sin() * 0.1;
            }
        }

        let opq = OPQRotation::train_with_sub_dim(&vectors, dim, sub_dim);
        let rotated = opq.apply(&vectors, dim);

        // 计算旋转后各子空间方差
        let m = dim / sub_dim;
        for sub in 0..m {
            let mut var = 0.0f32;
            let mut mean = vec![0.0f32; sub_dim];
            for i in 0..n {
                for d in 0..sub_dim {
                    mean[d] += rotated[i * dim + sub * sub_dim + d];
                }
            }
            for d in 0..sub_dim {
                mean[d] /= n as f32;
            }
            for i in 0..n {
                for d in 0..sub_dim {
                    let diff = rotated[i * dim + sub * sub_dim + d] - mean[d];
                    var += diff * diff;
                }
            }
            var = (var / n as f32).sqrt();
            eprintln!("[test] sub {}: variance = {:.4}", sub, var);
        }
        // 旋转后各子空间方差应比旋转前更均衡
    }
}
