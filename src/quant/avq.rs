//! AVQ（Anisotropic Vector Quantization）
//!
//! 设计文档第五层：
//! AVQ（参考 ScaNN score-aware quantization 思路）：
//!   优化 retrieval-aware quantization loss
//!   核心洞见：量化误差的平行分量对 inner product 高分候选的
//!             召回率影响远大于正交分量
//!   → codebook 训练时用检索目标加权
//!   → 与量化感知 RobustPrune 的归一化误差项直接联动
//!
//! ⚠️ retrieval-aware 信号来源须在实现阶段明确指定：
//!   AVQ codebook 训练在图构建之前，不依赖图结构；
//!   检索感知信号来自 sampled high-score pairs 或预采样近邻对。
//!   Week 5 同时实现最小双分支，用消融数据决定主线（见附录 F.11）。
//!   须在论文方法节说明选择理由并讨论合理性与局限性。

use crate::distance::l2_simd;
use rand_chacha::ChaCha8Rng;
use rand::{SeedableRng, Rng};

/// 量化模式
///
/// 设计文档：PQ / OPQ / AVQ 三种模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// PQ 基线（优化 reconstruction loss）
    Pq,
    /// OPQ（旋转向量使各子空间方差均等）
    Opq,
    /// AVQ（优化 retrieval-aware quantization loss）
    Avq,
}

/// AVQ 训练信号来源
///
/// 设计文档附录 B：AVQ 训练信号的具体实现（Week 5-6 决策）
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingSignal {
    /// 选项一：批次内高分对采样（推荐先验证）
    /// 设计文档：对训练集做批次内点积，选 high-score 对，无额外索引依赖
    /// 论文方法节："we sample high-inner-product pairs within training batches
    ///             to weight the quantization error"
    BatchHighScorePairs,
    /// 选项二：预采样近邻对
    /// 设计文档：需先用 f32 或粗量化跑一个近似近邻图，再采样
    /// 理论上更接近检索实际分布，但引入额外索引构建依赖
    PreSampledNeighborPairs,
}

impl Default for TrainingSignal {
    fn default() -> Self {
        // 设计文档：选项一推荐先验证
        TrainingSignal::BatchHighScorePairs
    }
}

/// AVQ codebook
///
/// 设计文档：优化 retrieval-aware quantization loss
/// 核心洞见：量化误差的平行分量对 inner product 高分候选的召回率影响远大于正交分量
pub struct AVQCodebook {
    /// 子空间数 M
    pub m: usize,
    /// 每个子空间的聚类中心数 K
    pub k: usize,
    /// 维度
    pub dim: usize,
    /// 每个子空间的维度
    pub sub_dim: usize,
    /// codebook：M × K × sub_dim
    pub centers: Vec<f32>,
    /// 训练信号来源（设计文档附录 B）
    pub training_signal: TrainingSignal,
    /// 训练时使用的高分对权重
    pub high_score_pairs: Vec<(u32, u32, f32)>, // (i, j, weight)
}

impl AVQCodebook {
    /// 训练 AVQ codebook（默认使用批次内高分对采样）
    ///
    /// 设计文档：AVQ codebook 训练在图构建之前，不依赖图结构
    /// 检索感知信号来自 sampled high-score pairs 或预采样近邻对
    pub fn train(
        vectors: &[f32],
        dim: usize,
        k: usize,
        _mode: QuantizationMode,
    ) -> Self {
        Self::train_with_signal(vectors, dim, k, _mode, TrainingSignal::BatchHighScorePairs)
    }

    /// 训练 AVQ codebook（指定训练信号来源）
    ///
    /// 设计文档附录 B：Week 5 同时实现最小双分支，用消融数据决定主线
    pub fn train_with_signal(
        vectors: &[f32],
        dim: usize,
        k: usize,
        _mode: QuantizationMode,
        signal: TrainingSignal,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        Self::train_full(vectors, dim, k, signal, 25, 16, 0.5, &mut rng)
    }

    /// 完整训练（Week 6：混合目标 retrieval-aware loss 对齐）
    ///
    /// 设计文档核心贡献：AVQ 优化 retrieval-aware loss，不是纯 reconstruction loss
    /// 两阶段训练：
    ///   1. weighted_kmeans_pp 初始化（reconstruction loss，保证 codebook 结构合理）
    ///   2. finetune_mixed 微调（α 混合 reconstruction + retrieval-aware）
    ///
    /// 参数：
    ///   - k: 每个子空间的聚类中心数（256 = 8-bit 编码）
    ///   - iterations: K-means 初始化迭代次数
    ///   - target_sub_dim: 目标子空间维度（8 或 16，影响 M）
    ///   - alpha: 混合权重 α∈[0,1]
    ///            α=1.0 纯重建（=PQ），α=0.0 纯内积失真，α=0.5 混合（默认推荐）
    pub fn train_full(
        vectors: &[f32],
        dim: usize,
        k: usize,
        signal: TrainingSignal,
        iterations: usize,
        target_sub_dim: usize,
        alpha: f32,
        rng: &mut ChaCha8Rng,
    ) -> Self {
        // 设计文档：M = dim / target_sub_dim
        let m = (dim / target_sub_dim).max(1);
        let sub_dim = dim / m;
        let n = vectors.len() / dim;

        // 设计文档附录 F.11：同时实现最小双分支
        let high_score_pairs = match signal {
            TrainingSignal::BatchHighScorePairs => {
                Self::sample_high_score_pairs(vectors, dim, n)
            }
            TrainingSignal::PreSampledNeighborPairs => {
                Self::sample_neighbor_pairs(vectors, dim, n)
            }
        };

        // ===== 阶段 1：K-means++ 初始化（reconstruction loss）=====
        let mut centers = vec![0.0f32; m * k * sub_dim];
        for sub in 0..m {
            let sub_vectors: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let start = i * dim + sub * sub_dim;
                    vectors[start..start + sub_dim].to_vec()
                })
                .collect();
            let weights = Self::compute_retrieval_weights(&high_score_pairs, n, sub);
            let sub_centers = weighted_kmeans_pp(&sub_vectors, &weights, k, iterations, rng);
            for (ki, center) in sub_centers.iter().enumerate() {
                for (d, &v) in center.iter().enumerate() {
                    centers[sub * k * sub_dim + ki * sub_dim + d] = v;
                }
            }
        }

        let mut codebook = AVQCodebook {
            m,
            k,
            dim,
            sub_dim,
            centers,
            training_signal: signal,
            high_score_pairs,
        };

        // ===== 阶段 2：混合目标微调（设计文档核心贡献）=====
        // loss = α * reconstruction_loss + (1-α) * retrieval_aware_loss
        codebook.finetune_mixed(vectors, iterations * 10, alpha);

        codebook
    }

    /// retrieval-aware 梯度下降微调（设计文档核心贡献）
    ///
    /// 设计文档：优化 retrieval-aware quantization loss
    /// 核心洞见：量化误差的平行分量对 inner product 高分候选的召回率影响远大于正交分量
    ///
    /// 算法（EM 风格）：
    ///   每轮迭代：
    ///     E 步：固定 centers，重新 encode 所有向量（更新 hard assignment）
    ///     M 步：固定 assignment，对 centers 做梯度下降
    ///           目标 = Σ w * (orig_ip - quant_ip)²
    ///           梯度：∂loss/∂centers[sub, code, d]
    ///                = -2 * w * err * (对端子空间解码向量)[d]
    ///                其中 err = orig_ip - quant_ip
    /// 混合目标梯度下降微调（设计文档核心贡献 + L2 场景适配）
    ///
    /// 设计文档核心洞见：量化误差的平行分量对高分候选召回率影响更大
    /// 在 L2 场景下精确表述：
    ///   L2(x,y)² = ||x||² + ||y||² - 2<x,y>
    ///   三项都有量化误差，但近邻对的 <x,y> 绝对值更大，失真影响权重更高
    ///
    /// 混合目标：
    ///   loss = α * reconstruction_loss + (1-α) * retrieval_aware_loss
    ///   - reconstruction_loss = Σ ||x - decode(encode(x))||²（L2 基础项）
    ///   - retrieval_aware_loss = Σ w * (orig_ip - quant_ip)²（近邻对内积失真加权）
    ///
    /// α 作为超参数，默认 0.5
    /// 论文消融：α=0（纯重建=PQ）、α=1（纯内积，不稳定）、α=0.5 预期最优
    pub fn finetune_mixed(&mut self, vectors: &[f32], iterations: usize, alpha: f32) {
        if self.high_score_pairs.is_empty() && alpha >= 1.0 {
            return;
        }
        let n = vectors.len() / self.dim;
        let lr = 1e-4;

        // 诊断：初始 loss
        let initial_recon = Self::reconstruction_loss(&self.centers, vectors, self.dim, self.m,
            self.k, self.sub_dim);
        let initial_ret = self.retrieval_aware_loss(vectors);
        eprintln!("[finetune α={:.2}] start: recon={:.4}, ret={:.4}",
            alpha, initial_recon, initial_ret);

        for iter in 0..iterations {
            // 两项梯度分别累加，分别归一化，保证 alpha 真正控制权重比例
            let mut grad_recon = vec![0.0f32; self.centers.len()];
            let mut grad_ret = vec![0.0f32; self.centers.len()];

            // ===== reconstruction 梯度项 =====
            // recon_loss = Σ (x - ci)²，∂/∂ci = 2*(ci - x)
            if alpha > 0.0 {
                for i in 0..n {
                    let vi = &vectors[i * self.dim..(i + 1) * self.dim];
                    let codes_i = self.encode(vi);
                    let ci = self.decode(&codes_i);
                    for sub in 0..self.m {
                        let code = codes_i[sub] as usize;
                        let sub_dim = self.sub_dim;
                        let vi_sub = &vi[sub * sub_dim..(sub + 1) * sub_dim];
                        let ci_sub = &ci[sub * sub_dim..(sub + 1) * sub_dim];
                        let grad_base = sub * self.k * sub_dim + code * sub_dim;
                        for d in 0..sub_dim {
                            grad_recon[grad_base + d] += 2.0 * (ci_sub[d] - vi_sub[d]);
                        }
                    }
                }
                // reconstruction 项不归一化：保留 Σ 累加，量级 ~n*2*(ci-v) ≈ 1000
                // retrieval 项也不归一化：量级 ~pairs*2*w*err*cj ≈ 10
                // α 直接控制两项权重比例
            }

            // ===== retrieval-aware 梯度项 =====
            // ret_loss = Σ w*(orig_ip - quant_ip)²，∂/∂ci = -2*w*err*cj
            if alpha < 1.0 && !self.high_score_pairs.is_empty() {
                for &(i, j, w) in &self.high_score_pairs {
                    if i as usize >= n || j as usize >= n {
                        continue;
                    }
                    let vi = &vectors[i as usize * self.dim..(i as usize + 1) * self.dim];
                    let vj = &vectors[j as usize * self.dim..(j as usize + 1) * self.dim];

                    let codes_i = self.encode(vi);
                    let codes_j = self.encode(vj);
                    let ci = self.decode(&codes_i);
                    let cj = self.decode(&codes_j);

                    let orig_ip: f32 = vi.iter().zip(vj.iter()).map(|(a, b)| a * b).sum();
                    let quant_ip: f32 = ci.iter().zip(cj.iter()).map(|(a, b)| a * b).sum();
                    let err = orig_ip - quant_ip;
                    if !err.is_finite() {
                        continue;
                    }

                    let grad_coeff = -2.0 * w * err;
                    for sub in 0..self.m {
                        let code_i = codes_i[sub] as usize;
                        let code_j = codes_j[sub] as usize;
                        let sub_dim = self.sub_dim;
                        let ci_sub = &ci[sub * sub_dim..(sub + 1) * sub_dim];
                        let cj_sub = &cj[sub * sub_dim..(sub + 1) * sub_dim];

                        let grad_i_base = sub * self.k * sub_dim + code_i * sub_dim;
                        for d in 0..sub_dim {
                            grad_ret[grad_i_base + d] += grad_coeff * cj_sub[d];
                        }
                        let grad_j_base = sub * self.k * sub_dim + code_j * sub_dim;
                        for d in 0..sub_dim {
                            grad_ret[grad_j_base + d] += grad_coeff * ci_sub[d];
                        }
                    }
                }
                // 不归一化：保留 Σ 累加，与 recon 梯度量级匹配
                // recon ~ Σ_n 2*(ci-v) ≈ 1000，ret ~ Σ_pairs 2*w*err*cj ≈ 10
                // 去掉 /num_pairs 后 ret 提升 1000x，α 能真正控制权重比例
            }

            // 合并并应用梯度下降
            for (idx, (gr, gret)) in grad_recon.iter().zip(grad_ret.iter()).enumerate() {
                let total = alpha * gr + (1.0 - alpha) * gret;
                if total.is_finite() {
                    self.centers[idx] -= lr * total;
                }
            }

            // 诊断
            if (iter + 1) % 10 == 0 || iter == 0 {
                let recon = Self::reconstruction_loss(&self.centers, vectors, self.dim, self.m,
                    self.k, self.sub_dim);
                let ret = self.retrieval_aware_loss(vectors);
                eprintln!("[finetune α={:.2}] iter={}: recon={:.4}, ret={:.4}",
                    alpha, iter + 1, recon, ret);
            }
        }

        let final_recon = Self::reconstruction_loss(&self.centers, vectors, self.dim, self.m,
            self.k, self.sub_dim);
        let final_ret = self.retrieval_aware_loss(vectors);
        eprintln!("[finetune α={:.2}] end: recon={:.4}(Δ{:+.4}), ret={:.4}(Δ{:+.4})",
            alpha, final_recon, final_recon - initial_recon,
            final_ret, final_ret - initial_ret);
    }

    /// reconstruction loss（Σ ||x - decode(encode(x))||²）
    pub fn reconstruction_loss(
        centers: &[f32],
        vectors: &[f32],
        dim: usize,
        m: usize,
        k: usize,
        sub_dim: usize,
    ) -> f32 {
        let n = vectors.len() / dim;
        let mut total = 0.0f32;
        // 临时 codebook 用于 encode/decode
        let cb = AVQCodebook {
            m, k, dim, sub_dim,
            centers: centers.to_vec(),
            training_signal: TrainingSignal::BatchHighScorePairs,
            high_score_pairs: Vec::new(),
        };
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let decoded = cb.decode(&cb.encode(v));
            for d in 0..dim {
                let diff = v[d] - decoded[d];
                total += diff * diff;
            }
        }
        total
    }

    /// 选项一：批次内高分对采样
    ///
    /// 设计文档附录 B：对训练集做批次内点积，选 high-score 对
    fn sample_high_score_pairs(vectors: &[f32], dim: usize, n: usize) -> Vec<(u32, u32, f32)> {
        let mut pairs = Vec::new();
        let batch_size = 64.min(n);
        let max_pairs = 1000;

        // 批次内采样高分对
        for batch_start in (0..n).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n);
            for i in batch_start..batch_end {
                for j in (i + 1)..batch_end {
                    let vi = &vectors[i * dim..(i + 1) * dim];
                    let vj = &vectors[j * dim..(j + 1) * dim];
                    // 计算内积（MIPS 场景的高分对）
                    let ip: f32 = vi.iter().zip(vj).map(|(a, b)| a * b).sum();
                    if ip > 0.0 {
                        pairs.push((i as u32, j as u32, ip));
                    }
                }
                if pairs.len() >= max_pairs {
                    break;
                }
            }
            if pairs.len() >= max_pairs {
                break;
            }
        }

        // 归一化权重
        let max_weight = pairs.iter().map(|(_, _, w)| *w).fold(0.0f32, f32::max);
        if max_weight > 0.0 {
            for (_, _, w) in &mut pairs {
                *w /= max_weight;
            }
        }

        pairs
    }

    /// 选项二：预采样近邻对
    ///
    /// 设计文档附录 B：需先用 f32 或粗量化跑一个近似近邻图，再采样
    /// 评估报告 M3：原 O(n²) 暴力扫描，改为随机采样候选 O(n * sample_size)
    fn sample_neighbor_pairs(vectors: &[f32], dim: usize, n: usize) -> Vec<(u32, u32, f32)> {
        use rand::seq::index::sample;

        let mut pairs = Vec::new();
        let max_pairs = 1000;
        let k_neighbors = 10;
        // 评估报告 M3：随机采样候选，避免 O(n²) 暴力扫描
        // sample_size = min(n, 1000)，时间复杂度 O(n * 1000 * dim)
        let sample_size = n.min(1000).saturating_sub(1);
        let mut rng = crate::build::ChaCha8Rng::seed_from(42);

        for i in 0..n {
            let vi = &vectors[i * dim..(i + 1) * dim];
            // 用 index::sample 直接采样索引，避免先收集成 Vec
            let indices = sample(&mut rng, n, sample_size);
            let mut dists: Vec<(f32, usize)> = indices
                .iter()
                .filter(|&j| j != i)
                .map(|j| {
                    let vj = &vectors[j * dim..(j + 1) * dim];
                    (l2_simd(vi, vj), j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(dist, j) in dists.iter().take(k_neighbors) {
                let weight = 1.0 / (1.0 + dist);
                pairs.push((i as u32, j as u32, weight));
                if pairs.len() >= max_pairs {
                    return pairs;
                }
            }
        }

        pairs
    }

    /// 计算检索感知权重
    ///
    /// 设计文档：codebook 训练时用检索目标加权
    fn compute_retrieval_weights(
        pairs: &[(u32, u32, f32)],
        n: usize,
        _sub: usize,
    ) -> Vec<f32> {
        let mut weights = vec![1.0f32; n];
        for &(i, j, w) in pairs {
            weights[i as usize] += w;
            weights[j as usize] += w;
        }
        // 归一化
        let max_w = weights.iter().cloned().fold(0.0f32, f32::max);
        if max_w > 0.0 {
            for w in &mut weights {
                *w /= max_w;
            }
        }
        weights
    }

    /// 编码单个向量
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

    /// 解码
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

    /// 计算节点的 AVQ 平行分量量化误差
    ///
    /// 设计文档 F.3：error(u, v) = mean(avq_error(u), avq_error(v))
    /// 即边两端点 AVQ 平行分量量化误差的均值
    pub fn node_error(&self, node_id: u32, vectors: &[f32]) -> f32 {
        let v = &vectors[node_id as usize * self.dim..(node_id as usize + 1) * self.dim];
        let codes = self.encode(v);
        let decoded = self.decode(&codes);
        l2_sq(v, &decoded)
    }

    /// 计算边的量化误差（设计文档 F.3 口径）
    ///
    /// error(u, v) = mean(avq_error(u), avq_error(v))
    pub fn edge_error(&self, u: u32, v: u32, vectors: &[f32]) -> f32 {
        let eu = self.node_error(u, vectors);
        let ev = self.node_error(v, vectors);
        (eu + ev) / 2.0
    }

    /// 计算 retrieval-aware loss
    ///
    /// 设计文档：优化 retrieval-aware quantization loss
    pub fn retrieval_aware_loss(&self, vectors: &[f32]) -> f32 {
        let _n = vectors.len() / self.dim;
        let mut total = 0.0f32;
        let mut total_weight = 0.0f32;

        for &(i, j, w) in &self.high_score_pairs {
            let vi = &vectors[i as usize * self.dim..(i as usize + 1) * self.dim];
            let vj = &vectors[j as usize * self.dim..(j as usize + 1) * self.dim];
            let ci = self.decode(&self.encode(vi));
            let cj = self.decode(&self.encode(vj));

            // 平行分量误差（设计文档核心洞见）
            let orig_ip: f32 = vi.iter().zip(vj).map(|(a, b)| a * b).sum();
            let quant_ip: f32 = ci.iter().zip(cj).map(|(a, b)| a * b).sum();
            let parallel_error = (orig_ip - quant_ip).powi(2);

            total += w * parallel_error;
            total_weight += w;
        }

        if total_weight > 0.0 {
            total / total_weight
        } else {
            0.0
        }
    }
}

/// AVQ 量化器
pub struct AVQ;

impl AVQ {
    /// 训练并编码
    pub fn fit_transform(
        vectors: &[f32],
        dim: usize,
        k: usize,
    ) -> (AVQCodebook, Vec<Vec<u8>>) {
        let codebook = AVQCodebook::train(vectors, dim, k, QuantizationMode::Avq);
        let n = vectors.len() / dim;
        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| codebook.encode(&vectors[i * dim..(i + 1) * dim]))
            .collect();
        (codebook, codes)
    }
}

/// 加权 K-means++（k-means++ 初始化 + 收敛判定）
///
/// Week 6 改进：
///   1. k-means++ 初始化：概率正比于 D(x)² / weight
///   2. 收敛判定：中心移动 < ε 时提前终止
fn weighted_kmeans_pp(
    data: &[Vec<f32>],
    weights: &[f32],
    k: usize,
    iterations: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    let dim = data[0].len();
    let n = data.len();
    let k = k.min(n);

    // k-means++ 初始化
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
    // 第一个中心：选权重最大的点
    let first = (0..n).max_by(|&a, &b| {
        weights.get(a).copied().unwrap_or(1.0)
            .partial_cmp(&weights.get(b).copied().unwrap_or(1.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    }).unwrap_or(0);
    centers.push(data[first].clone());

    // 后续中心：D(x)² × weight 概率选择
    for _ in 1..k {
        let mut dists = vec![f32::MAX; n];
        for (i, point) in data.iter().enumerate() {
            for center in &centers {
                let d = l2_sq(point, center);
                if d < dists[i] {
                    dists[i] = d;
                }
            }
        }
        // 加权概率：D(x)² × weight
        let total: f32 = data.iter().enumerate()
            .map(|(i, _)| dists[i] * weights.get(i).copied().unwrap_or(1.0))
            .sum();
        if total <= 0.0 {
            // 所有点都是中心，顺序选下一个
            if centers.len() < n {
                centers.push(data[centers.len()].clone());
            }
            continue;
        }
        // 设计文档约束：使用固定种子 ChaCha8 RNG，保证确定性
        let r: f32 = rng.gen();
        let mut cum = 0.0f32;
        let mut chosen = 0;
        for (i, _point) in data.iter().enumerate() {
            cum += dists[i] * weights.get(i).copied().unwrap_or(1.0) / total;
            if cum >= r {
                chosen = i;
                break;
            }
        }
        centers.push(data[chosen].clone());
    }

    // K-means 迭代 + 收敛判定
    for _ in 0..iterations {
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

        let mut new_centers = vec![vec![0.0f32; dim]; k];
        let mut weight_sums = vec![0.0f32; k];
        for (i, &a) in assignments.iter().enumerate() {
            let w = weights.get(i).copied().unwrap_or(1.0);
            for d in 0..dim {
                new_centers[a][d] += w * data[i][d];
            }
            weight_sums[a] += w;
        }
        for j in 0..k {
            if weight_sums[j] > 0.0 {
                for d in 0..dim {
                    new_centers[j][d] /= weight_sums[j];
                }
            } else {
                new_centers[j] = centers[j].clone();
            }
        }

        // 收敛判定：中心移动 < ε
        let max_move: f32 = centers.iter().zip(new_centers.iter())
            .map(|(old, new)| l2_sq(old, new).sqrt())
            .fold(0.0f32, f32::max);
        centers = new_centers;
        if max_move < 1e-6 {
            break;
        }
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
    fn avq_train_basic() {
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = AVQCodebook::train(&vectors, 8, 16, QuantizationMode::Avq);
        assert_eq!(codebook.dim, 8);
        assert!(codebook.m > 0);
    }

    #[test]
    fn avq_encode_decode() {
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = AVQCodebook::train(&vectors, 8, 16, QuantizationMode::Avq);
        let codes = codebook.encode(&vectors[0..8]);
        let decoded = codebook.decode(&codes);
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn avq_node_error_positive() {
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = AVQCodebook::train(&vectors, 8, 16, QuantizationMode::Avq);
        let err = codebook.node_error(0, &vectors);
        assert!(err >= 0.0);
    }

    #[test]
    fn avq_edge_error_is_mean() {
        // 设计文档 F.3：error(u, v) = mean(avq_error(u), avq_error(v))
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = AVQCodebook::train(&vectors, 8, 16, QuantizationMode::Avq);
        let eu = codebook.node_error(0, &vectors);
        let ev = codebook.node_error(1, &vectors);
        let edge_err = codebook.edge_error(0, 1, &vectors);
        assert!((edge_err - (eu + ev) / 2.0).abs() < 1e-5);
    }

    #[test]
    fn avq_retrieval_aware_loss_nonneg() {
        let vectors: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let codebook = AVQCodebook::train(&vectors, 8, 16, QuantizationMode::Avq);
        let loss = codebook.retrieval_aware_loss(&vectors);
        assert!(loss >= 0.0);
    }

    #[test]
    fn training_signal_default_is_batch() {
        // 设计文档：选项一推荐先验证
        assert_eq!(TrainingSignal::default(), TrainingSignal::BatchHighScorePairs);
    }
}
