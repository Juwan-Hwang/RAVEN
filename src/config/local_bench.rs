//! 本地基准评测
//!
//! 设计文档第六层：
//! 分阶段接入策略：先用纯 Rust 扁平二进制格式把本地基准和核心内核跑通，
//! 再回头接 benchmark 外壳。
//!
//! 微基准工具：
//! - 日常迭代：divan（轻量、上手快）
//! - 正式实验与回归：criterion（统计驱动，适合参数对比和写实验图表）

use crate::distance::l2_simd;
use crate::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use crate::build::ChaCha8Rng;
use std::time::{Duration, Instant};

/// 基准测试结果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// 测试名称
    pub name: String,
    /// 数据集大小
    pub n: usize,
    /// 维度
    pub dim: usize,
    /// recall@10
    pub recall_at_10: f64,
    /// QPS（每秒查询数）
    pub qps: f64,
    /// 平均延迟（微秒）
    pub avg_latency_us: f64,
    /// p95 延迟（微秒）
    pub p95_latency_us: f64,
    /// p99 延迟（微秒）
    pub p99_latency_us: f64,
}

impl BenchmarkResult {
    /// 打印结果
    pub fn print(&self) {
        println!("=== {} ===", self.name);
        println!("  N={}, dim={}", self.n, self.dim);
        println!("  recall@10: {:.4}", self.recall_at_10);
        println!("  QPS:        {:.1}", self.qps);
        println!("  avg latency: {:.2} us", self.avg_latency_us);
        println!("  p95 latency: {:.2} us", self.p95_latency_us);
        println!("  p99 latency: {:.2} us", self.p99_latency_us);
    }
}

/// 本地基准测试
///
/// 设计文档：先用纯 Rust 把本地基准和核心内核跑通
pub struct LocalBenchmark {
    /// 向量数据
    vectors: Vec<f32>,
    /// 维度
    dim: usize,
    /// 真值（ground truth）：每个查询的 top-10 最近邻
    ground_truth: Vec<Vec<u32>>,
    /// 查询向量
    queries: Vec<Vec<f32>>,
}

impl LocalBenchmark {
    /// 创建本地基准
    pub fn new(vectors: Vec<f32>, dim: usize) -> Self {
        Self {
            vectors,
            dim,
            ground_truth: Vec::new(),
            queries: Vec::new(),
        }
    }

    /// 设置查询和真值
    pub fn with_queries(mut self, queries: Vec<Vec<f32>>, ground_truth: Vec<Vec<u32>>) -> Self {
        self.queries = queries;
        self.ground_truth = ground_truth;
        self
    }

    /// 生成随机查询和真值（用于无真值数据集的快速验证）
    pub fn generate_random_queries(&mut self, n_queries: usize, k: usize) {
        let n = self.vectors.len() / self.dim;
        let mut rng = ChaCha8Rng::new();
        use rand::Rng;
        self.queries = (0..n_queries)
            .map(|_| {
                let idx = rng.gen_range(0..n);
                self.vectors[idx * self.dim..(idx + 1) * self.dim].to_vec()
            })
            .collect();

        // 暴力计算真值
        self.ground_truth = self
            .queries
            .iter()
            .map(|q| {
                let mut dists: Vec<(f32, u32)> = (0..n)
                    .map(|i| {
                        let v = &self.vectors[i * self.dim..(i + 1) * self.dim];
                        (l2_simd(q, v), i as u32)
                    })
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(k).map(|(_, id)| *id).collect()
            })
            .collect();
    }

    /// 运行基准测试
    ///
    /// 设计文档：recall@10 at target QPS 为主指标
    pub fn run(&self, config: &VamanaBuildConfig, ef_search: usize) -> BenchmarkResult {
        let n = self.vectors.len() / self.dim;
        let mut rng = ChaCha8Rng::new();
        let graph = VamanaGraph::build(&self.vectors, self.dim, config, &mut rng);
        let mut searcher = GraphSearcher::new(&self.vectors, &graph, ef_search);

        let mut latencies: Vec<Duration> = Vec::with_capacity(self.queries.len());
        let mut recall_sum = 0.0f64;

        let start = Instant::now();
        for (i, query) in self.queries.iter().enumerate() {
            let q_start = Instant::now();
            let results = searcher.search(query, 10);
            latencies.push(q_start.elapsed());

            // 计算 recall@10
            if let Some(gt) = self.ground_truth.get(i) {
                let hits = results.iter().filter(|(id, _)| gt.contains(id)).count();
                recall_sum += hits as f64 / gt.len() as f64;
            }
        }
        let elapsed = start.elapsed();

        let recall_at_10 = if self.queries.is_empty() {
            0.0
        } else {
            recall_sum / self.queries.len() as f64
        };

        let qps = if elapsed.as_secs_f64() > 0.0 {
            self.queries.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let avg_latency_us = latencies.iter().map(|d| d.as_micros() as f64).sum::<f64>()
            / latencies.len().max(1) as f64;

        let mut sorted_latencies: Vec<f64> =
            latencies.iter().map(|d| d.as_micros() as f64).collect();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p95_latency_us = percentile_f64(&sorted_latencies, 0.95);
        let p99_latency_us = percentile_f64(&sorted_latencies, 0.99);

        BenchmarkResult {
            name: format!("raven_n{}_dim{}", n, self.dim),
            n,
            dim: self.dim,
            recall_at_10,
            qps,
            avg_latency_us,
            p95_latency_us,
            p99_latency_us,
        }
    }
}

/// 计算分位数
fn percentile_f64(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_runs_small() {
        let vectors: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let dim = 10;
        let mut bench = LocalBenchmark::new(vectors, dim);
        bench.generate_random_queries(10, 10);
let config = VamanaBuildConfig {
alpha: 1.0,
l_build: 50,
r_max: 8,
r_soft: 12,
max_iterations: 1,
saturate: true,
enable_layered_nav: false,
nav_m: 16,
..Default::default()
};
        let result = bench.run(&config, 50);
        assert!(result.recall_at_10 >= 0.0 && result.recall_at_10 <= 1.0);
        assert!(result.qps > 0.0);
    }
}
