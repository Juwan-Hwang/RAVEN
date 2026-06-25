# RAVEN

**Retrieval-Aware Vector Engine with Navigation** -- a high-performance Rust library for Approximate Nearest Neighbor (ANN) search.

## Overview

RAVEN implements a Vamana/DiskANN-style graph index with three integrated research contributions:

1. **RP-Tuning**: Post-hoc alpha tuning for the Vamana build parameter, optimizing the recall-QPS Pareto frontier
2. **AVQ (Asymmetric Vector Quantization)**: Retrieval-aware quantization that minimizes retrieval loss instead of reconstruction loss
3. **Quantization-Aware Pruning**: Quantization error influences graph pruning decisions via a normalized scoring function

### Architecture (6-layer design)

| Layer | Module | Description |
|---|---|---|
| L1 | `distance/` | SIMD distance kernels (AVX-512, AVX2, scalar), f16 mixed-precision |
| L2 | `memory/` | Hybrid Blocked-CSR graph storage, visited tracker, serialization with CRC32 |
| L3 | `graph/` | Vamana build, RobustPrune, quantization-aware pruning, navigation overlay |
| L4 | `build/` | Rayon parallel build pipeline, configurable alpha/r_max/l_build |
| L5 | `quant/` | OPQ, AVQ, PQ codebook training and encoding |
| L6 | `bench/` | Benchmark harness for ann-benchmarks integration |

## Performance

SIFT1M benchmark (1M vectors, dim=128, 16 threads):

| Configuration | recall@10 | QPS | Build Time |
|---|---|---|---|
| r_max=64, ef=100 (high recall) | 0.9961 | 2,434 | ~78 min |
| r_max=64, ef=50 (fast) | 0.9275 | 7,611 | ~78 min |
| r_max=48, ef=100 (balanced) | 0.9934 | 2,524 | ~47 min |

### Key Optimizations Applied

- **Distance reuse**: `greedy_search_vec_reuse` returns `(id, dist)` pairs, eliminating redundant distance computation (+20% QPS)
- **Prefetch strategy**: Prefetch heap-top node's neighbor list instead of next vector (+28% QPS)
- **Medoid sampling**: 1K-sample approximate medoid for n>10K, 25x faster with no recall impact

## Usage

```rust
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher};
use raven::build::ChaCha8Rng;

// Build index
let config = VamanaBuildConfig {
    alpha: 1.2,
    l_build: 200,
    r_soft: 48,
    r_max: 64,
    max_iterations: 2,
};
let mut rng = ChaCha8Rng::seed_from(42);
let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);

// Search
let mut searcher = GraphSearcher::new(&vectors, &graph, 100);
let results = searcher.search(&query, 10); // top-10
```

## Building

```bash
# Requires nightly Rust (for AVX-512 intrinsics)
cargo build --release

# Run quick recall verification on SIFT1M
cargo run --release --bin quick_recall_check
```

### Hardware Requirements

- x86_64 with AVX2 + F16C (minimum)
- AVX-512 recommended for best performance

## Project Structure

```
src/
  distance/     # SIMD distance kernels (L1)
  memory/       # Graph storage and serialization (L2)
  graph/        # Vamana build, pruning, search (L3)
  build/        # Parallel build pipeline (L4)
  quant/        # OPQ/AVQ/PQ quantization (L5)
  bench/        # Benchmark harness (L6)
  bin/          # Experiment binaries (optimization A/B tests)
```

## Documentation

- [设计文档.md](设计文档.md) -- Full technical design document (Chinese)
- [优化方案.md](优化方案.md) -- Optimization plan with A/B experiment results

## License

Copyright 2026 Juwan Hwang (黄治文). Licensed under [Apache 2.0](LICENSE).
