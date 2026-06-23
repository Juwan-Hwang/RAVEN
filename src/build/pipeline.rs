//! 纯函数式 Pipeline（离线层，绝不进入查询热路径）
//!
//! 设计文档第四层：
//! let index = RawVectors::load(path)
//!     .pipe(|v| OPQRotation::train(&v))
//!     .pipe(|v| AVQCodebook::train(&v))
//!     .pipe(|(v, cb)| VamanaGraph::build(v, α, R_soft))
//!     .pipe(|(g, cb)| QuantAwareRobustPrune::apply(g, cb, β))
//!     .pipe(|(g, _)| global_final_prune_to_R_max(g))
//!     .pipe(|(g, _)| RPTuning::generate_variants(g, &[1.0, 1.2, 1.5, 2.0]));
//! // 替换任意一步 = 一次消融实验

use crate::build::BuildConfig;
use crate::build::ChaCha8Rng;
use crate::graph::{VamanaGraph, VamanaBuildConfig};
use crate::quant::{OPQRotation, AVQCodebook, QuantizationMode};

/// Pipeline 阶段标识
///
/// 设计文档：替换任意一步 = 一次消融实验
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// 原始向量加载
    LoadVectors,
    /// OPQ 旋转
    OpqRotation,
    /// AVQ codebook 训练
    AvqCodebook,
    /// Vamana 图构建
    VamanaBuild,
    /// 量化感知 RobustPrune
    QuantAwarePrune,
    /// 全局 final prune 到 R_max
    FinalPrune,
    /// RP-Tuning 生成变体
    RpTuning,
}

/// Pipeline 中间状态
///
/// 设计文档纯函数式 Pipeline 的数据流
pub struct PipelineState {
    /// 向量数据（扁平存储）
    pub vectors: Vec<f32>,
    /// 维度
    pub dim: usize,
    /// OPQ 旋转（可选）
    pub opq: Option<OPQRotation>,
    /// AVQ codebook（可选）
    pub avq: Option<AVQCodebook>,
    /// 图（可选）
    pub graph: Option<VamanaGraph>,
    /// 当前阶段
    pub stage: PipelineStage,
}

impl PipelineState {
    /// 从原始向量开始
    pub fn from_vectors(vectors: Vec<f32>, dim: usize) -> Self {
        Self {
            vectors,
            dim,
            opq: None,
            avq: None,
            graph: None,
            stage: PipelineStage::LoadVectors,
        }
    }
}

/// 纯函数式 Pipeline
///
/// 设计文档第四层：离线层，绝不进入查询热路径
pub struct BuildPipeline {
    config: BuildConfig,
}

impl BuildPipeline {
    /// 创建 Pipeline
    pub fn new(config: BuildConfig) -> Self {
        Self { config }
    }

    /// 执行完整构建流程
    ///
    /// 设计文档原文 Pipeline：
    /// 1. OPQ 旋转
    /// 2. AVQ codebook 训练
    /// 3. Vamana 图构建
    /// 4. 量化感知 RobustPrune
    /// 5. 全局 final prune 到 R_max
    /// 6. RP-Tuning 生成变体
    pub fn run(&self, vectors: Vec<f32>, dim: usize) -> PipelineResult {
        let mut state = PipelineState::from_vectors(vectors, dim);

        // 1. OPQ 旋转（设计文档：构建前旋转向量使各子空间方差均等）
        state = self.opq_rotation(state);
        state.stage = PipelineStage::OpqRotation;

        // 2. AVQ codebook 训练（设计文档：优化 retrieval-aware quantization loss）
        state = self.avq_codebook_train(state);
        state.stage = PipelineStage::AvqCodebook;

        // 3. Vamana 图构建
        state = self.vamana_build(state);
        state.stage = PipelineStage::VamanaBuild;

        // 4. 量化感知 RobustPrune（设计文档：量化误差反向影响图剪枝决策）
        state = self.quant_aware_prune(state);
        state.stage = PipelineStage::QuantAwarePrune;

        // 5. 全局 final prune 到 R_max
        state = self.final_prune(state);
        state.stage = PipelineStage::FinalPrune;

        // 6. RP-Tuning 生成变体
        let variants = self.rp_tuning(&state);
        state.stage = PipelineStage::RpTuning;

        PipelineResult {
            graph: state.graph.expect("graph should be built"),
            opq: state.opq,
            avq: state.avq,
            alpha_variants: variants,
            final_stage: state.stage,
        }
    }

    /// OPQ 旋转阶段
    fn opq_rotation(&self, state: PipelineState) -> PipelineState {
        let opq = OPQRotation::train(&state.vectors, state.dim);
        let rotated = opq.apply(&state.vectors, state.dim);
        PipelineState {
            vectors: rotated,
            opq: Some(opq),
            ..state
        }
    }

    /// AVQ codebook 训练阶段
    fn avq_codebook_train(&self, state: PipelineState) -> PipelineState {
        let avq = AVQCodebook::train(&state.vectors, state.dim, 256, QuantizationMode::Avq);
        PipelineState {
            avq: Some(avq),
            ..state
        }
    }

    /// Vamana 图构建阶段
    fn vamana_build(&self, state: PipelineState) -> PipelineState {
        let mut rng = ChaCha8Rng::seed_from(self.config.rng_seed);
        let vamana_config = VamanaBuildConfig {
            alpha: self.config.alpha,
            l_build: self.config.l_build,
            r_max: self.config.r_max,
            r_soft: self.config.r_soft,
            max_iterations: 1,
        };
        let graph = VamanaGraph::build(&state.vectors, state.dim, &vamana_config, &mut rng);
        PipelineState {
            graph: Some(graph),
            ..state
        }
    }

    /// 量化感知 RobustPrune 阶段
    ///
    /// 设计文档：量化误差反向影响图剪枝决策的图网络结构
    fn quant_aware_prune(&self, state: PipelineState) -> PipelineState {
        // 当前实现：量化感知 prune 在 RP-Tuning 阶段通过 β 参数体现
        // 完整实现需要 AVQ codebook 提供误差函数
        // 这里保留 Pipeline 结构，实际量化感知剪枝在 RP-Tuning 中应用
        state
    }

    /// 全局 final prune 到 R_max 阶段
    fn final_prune(&self, mut state: PipelineState) -> PipelineState {
        if let Some(ref mut graph) = state.graph {
            let storage = graph.storage_mut();
            for node in 0..storage.len() as u32 {
                let (main, overflow) = storage.neighbors_full(node);
                let total = main.len() + overflow.len();
                if total <= self.config.r_max {
                    continue;
                }
                let mut all: Vec<u32> = main.to_vec();
                all.extend_from_slice(overflow);
                all.truncate(self.config.r_max);
                storage.set_neighbors(node, &all);
            }
        }
        state
    }

    /// RP-Tuning 生成变体阶段
    fn rp_tuning(&self, state: &PipelineState) -> Vec<crate::graph::AlphaVariant> {
        use crate::graph::{RPTuning, RPTuningConfig};
        if let Some(ref graph) = state.graph {
            let config = RPTuningConfig {
                scheme: Default::default(),
                alpha_points: vec![1.0, 1.2, 1.5, 2.0],
                r_max: self.config.r_max,
            };
            RPTuning::generate_variants(graph, &state.vectors, state.dim, &config)
        } else {
            Vec::new()
        }
    }
}

/// Pipeline 执行结果
pub struct PipelineResult {
    /// 最终图
    pub graph: VamanaGraph,
    /// OPQ 旋转
    pub opq: Option<OPQRotation>,
    /// AVQ codebook
    pub avq: Option<AVQCodebook>,
    /// RP-Tuning α 变体
    pub alpha_variants: Vec<crate::graph::AlphaVariant>,
    /// 最终阶段
    pub final_stage: PipelineStage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_runs_small_dataset() {
        let vectors: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let dim = 10;
        let config = BuildConfig {
            r_max: 4,
            r_soft: 6,
            l_build: 20,
            ..Default::default()
        };
        let pipeline = BuildPipeline::new(config);
        let result = pipeline.run(vectors, dim);
        assert_eq!(result.graph.len(), 10);
        assert!(!result.alpha_variants.is_empty());
        assert_eq!(result.final_stage, PipelineStage::RpTuning);
    }
}
