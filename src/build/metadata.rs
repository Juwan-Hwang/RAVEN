//! 构建元数据与配置
//!
//! 设计文档第四层 F.7：
//! 以上三项写入索引元数据随索引文件落盘：
//!   rng_algorithm: "chacha8"
//!   rng_seed: 42
//!   shard_strategy: "static_range"
//!   build_version: "0.1.0"

use serde::{Deserialize, Serialize};

/// 构建配置
///
/// 设计文档第四层：构建可复现性约定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// RNG 算法（设计文档：ChaCha8）
    pub rng_algorithm: String,
    /// RNG 种子（设计文档：默认值 42）
    pub rng_seed: u64,
    /// 并行分片策略（设计文档：static_range）
    pub shard_strategy: String,
    /// 库版本号
    pub build_version: String,
    /// 全局 α（构建时固定）
    pub alpha: f32,
    /// L_build 构建期搜索宽度
    pub l_build: usize,
    /// R_max 硬上限
    pub r_max: usize,
    /// R_soft 软上限（设计文档：1.5 × R_max）
    pub r_soft: usize,
    /// β 量化感知剪枝权重（设计文档：β=0.0 为标准 RobustPrune）
    pub beta: f32,
}

impl Default for BuildConfig {
    fn default() -> Self {
        let r_max = 64;
        Self {
            rng_algorithm: "chacha8".to_owned(),
            rng_seed: 42,
            shard_strategy: "static_range".to_owned(),
            build_version: crate::BUILD_VERSION.to_owned(),
            alpha: 1.2,
            l_build: 200,
            r_max,
            r_soft: (r_max as f32 * 1.5) as usize,
            beta: 0.0,
        }
    }
}

/// 构建元数据
///
/// 设计文档 F.7：随索引文件落盘
/// 论文复现实验时直接读取元数据，确保不同机器、不同编译版本下
/// 相同参数构建的图结构完全一致。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildMetadata {
    /// RNG 算法
    pub rng_algorithm: String,
    /// RNG 种子
    pub rng_seed: u64,
    /// 并行分片策略
    pub shard_strategy: String,
    /// 库版本号
    pub build_version: String,
    /// 构建时间戳
    pub build_timestamp: u64,
    /// 节点数
    pub n: usize,
    /// 向量维度
    pub dim: usize,
    /// 全局 α
    pub alpha: f32,
    /// L_build
    pub l_build: usize,
    /// R_max
    pub r_max: usize,
    /// R_soft
    pub r_soft: usize,
}

impl BuildMetadata {
    /// 从构建配置创建元数据
    pub fn from_config(config: &BuildConfig, n: usize, dim: usize) -> Self {
        Self {
            rng_algorithm: config.rng_algorithm.clone(),
            rng_seed: config.rng_seed,
            shard_strategy: config.shard_strategy.clone(),
            build_version: config.build_version.clone(),
            build_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            n,
            dim,
            alpha: config.alpha,
            l_build: config.l_build,
            r_max: config.r_max,
            r_soft: config.r_soft,
        }
    }

    /// 序列化为 TOML
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// 从 TOML 反序列化
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = BuildConfig::default();
        assert_eq!(config.rng_algorithm, "chacha8");
        assert_eq!(config.rng_seed, 42);
        assert_eq!(config.shard_strategy, "static_range");
        assert_eq!(config.r_max, 64);
        assert_eq!(config.r_soft, 96); // 1.5 × 64
    }

    #[test]
    fn metadata_roundtrip_toml() {
        let config = BuildConfig::default();
        let metadata = BuildMetadata::from_config(&config, 1000, 768);
        let toml_str = metadata.to_toml().unwrap();
        let restored = BuildMetadata::from_toml(&toml_str).unwrap();
        assert_eq!(restored.n, 1000);
        assert_eq!(restored.dim, 768);
        assert_eq!(restored.rng_algorithm, "chacha8");
        assert_eq!(restored.rng_seed, 42);
    }
}
