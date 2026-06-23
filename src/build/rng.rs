//! 构建可复现性约定：ChaCha8 RNG
//!
//! 设计文档第四层 F.7：
//! 构建过程中所有随机行为默认使用确定性 RNG：
//!   算法：ChaCha8（速度快、质量足够、跨平台结果一致）
//!   种子入口：配置文件或命令行参数显式指定，默认值 42
//!   并行分片策略：按节点 ID 范围静态划分，不依赖调度顺序

pub use rand_chacha::ChaCha8Rng as InnerRng;
use rand::SeedableRng;

/// ChaCha8 RNG 包装
///
/// 设计文档 F.7：ChaCha8 速度快、质量足够、跨平台结果一致
pub struct ChaCha8Rng(InnerRng);

impl ChaCha8Rng {
    /// 从种子创建
    ///
    /// 设计文档：种子入口默认值 42
    pub fn seed_from(seed: u64) -> Self {
        Self(InnerRng::seed_from_u64(seed))
    }

    /// 默认种子创建（设计文档：默认值 42）
    pub fn new() -> Self {
        Self::seed_from(42)
    }

    /// 获取内部 RNG 可变引用
    pub fn inner(&mut self) -> &mut InnerRng {
        &mut self.0
    }
}

impl Default for ChaCha8Rng {
    fn default() -> Self {
        Self::new()
    }
}

impl rand::RngCore for ChaCha8Rng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.0.try_fill_bytes(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn deterministic_same_seed() {
        let mut r1 = ChaCha8Rng::seed_from(42);
        let mut r2 = ChaCha8Rng::seed_from(42);
        let v1: Vec<u32> = (0..10).map(|_| r1.gen()).collect();
        let v2: Vec<u32> = (0..10).map(|_| r2.gen()).collect();
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_seed_different_output() {
        let mut r1 = ChaCha8Rng::seed_from(42);
        let mut r2 = ChaCha8Rng::seed_from(43);
        let v1: Vec<u32> = (0..10).map(|_| r1.gen()).collect();
        let v2: Vec<u32> = (0..10).map(|_| r2.gen()).collect();
        assert_ne!(v1, v2);
    }

    #[test]
    fn default_seed_is_42() {
        let r = ChaCha8Rng::default();
        let _ = r; // 只验证可创建
    }
}
