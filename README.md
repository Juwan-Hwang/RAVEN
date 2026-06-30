# RAVEN

**Retrieval-Aware Vector Engine with Navigation** — a high-performance Rust library for Approximate Nearest Neighbor (ANN) search.

## Overview

RAVEN implements a Vamana/DiskANN-style graph index with four integrated research contributions:

1. **DirectionalPrune** — Directional scan with adaptive connectivity floor, eliminating saturation padding for sparser, higher-quality graphs (+3.4% QPS vs RobustPrune)
2. **SQ4 Quantized Search** — 4-bit scalar quantization for graph navigation with f32 rerank, reducing memory bandwidth by 8× (+11~13% QPS vs SQ8)
3. **AdaptiveEf** — Dynamic ef prediction from query→entry-point distance distribution, filling Pareto gaps between fixed-ef points
4. **Branchless Visited Check** — Eliminates branch misprediction in the hot search loop (+2.7% QPS)

### Performance

SIFT-1M benchmark (1M vectors, dim=128, SQ4, Zen 4):

| ef | recall@10 | QPS |
|---:|---:|---:|
| 40 | 0.9407 | 23,800 |
| 45 | 0.9510 | 21,727 |
| 50 | 0.9590 | 20,017 |
| 55 | 0.9656 | 18,623 |
| 65 | 0.9740 | 16,295 |
| 80 | 0.9824 | 13,891 |
| 100 | 0.9885 | 11,641 |

**Working point**: ef=50, recall=0.9590, QPS≈20,017

**AdaptiveEf Pareto fill** (recall 0.945–0.960):

| Config | recall@10 | QPS |
|---|---:|---:|
| γ3.5(35,65) | 0.9453 | 22,790 |
| γ2.4(35,65) | 0.9493 | 22,040 |
| γ2.0(35,65) | 0.9513 | 21,564 |
| γ2.3(35,75) | 0.9537 | 20,651 |
| γ2.0(35,75) | 0.9558 | 20,297 |
| γ2.0(35,85) | 0.9600 | 19,072 |

**Multi-threaded**: ef=50, recall=0.9590, QPS≈131,510 (6.6× single-thread, 16 threads)

### Architecture (6-layer design)

| Layer | Module | Description |
|---|---|---|
| L1 | `src/distance/` | SIMD distance kernels (AVX-512, AVX2, F16C, scalar) |
| L2 | `src/memory/` | Hybrid Blocked-CSR graph storage, visited tracker, CRC32 serialization |
| L3 | `src/graph/` | Vamana build, RobustPrune / DirectionalPrune, layered navigation, graph search, AdaptiveEf |
| L4 | `src/build/` | Rayon parallel build pipeline, configurable α / R_max / L_build |
| L5 | `src/quant/` | SQ4/SQ8 scalar quantization, PQ4/PQ8 product quantization |
| L6 | `src/bench/` | Benchmark harness |

## Quick Start

### Build (Rust)

```bash
cargo build --release
```

### Run benchmark

```bash
# Download SIFT-1M dataset to data/sift/
cargo run --release --bin submission_bench
```

### Python bindings (ann-benchmarks)

```bash
# Create venv and install maturin
python -m venv .venv
source .venv/bin/activate  # Linux: .venv\Scripts\activate (Windows)
pip install maturin numpy

# Build and install Python extension
maturin develop --release --features python

# Test
python test_pybinding.py
```

### Rust API

```rust
use raven::graph::{VamanaGraph, VamanaBuildConfig, GraphSearcher, PruneStrategy};
use raven::build::ChaCha8Rng;
use raven::quant::SQ4Dataset;

// Build index
let config = VamanaBuildConfig {
    alpha: 1.2,
    l_build: 200,
    r_max: 32,
    r_soft: 48,
    max_iterations: 2,
    saturate: false,
    enable_layered_nav: true,
    nav_m: 32,
    prune_strategy: PruneStrategy::DirectionalPrune,
};
let mut rng = ChaCha8Rng::seed_from(42);
let graph = VamanaGraph::build(&vectors, dim, &config, &mut rng);

// SQ4 quantized search
let sq4 = SQ4Dataset::build(&vectors, dim);
let mut searcher = GraphSearcher::new(&vectors, &graph, 50);
searcher.with_sq4(&sq4);
searcher.with_prefetch_offset(8);
searcher.with_rerank_factor(8);

let results = searcher.search_sq4(&query, 10); // top-10
```

### Python API

```python
import raven
import numpy as np

# Build index (SQ4 quantization)
index = raven.Index("L2", dim=128, r=32, l=200, alpha=1.2, nav_m=32,
                    directional=True, quantization="sq4", rerank_factor=8)
index.build(train_data)  # numpy array (N, 128) float32

# Search
searcher = index.searcher()
searcher.set_ef(50)
results = searcher.search(query, k=10)  # numpy array of int

# AdaptiveEf — dynamic ef prediction
index = raven.Index("L2", dim=128, r=32, l=200, alpha=1.2, nav_m=32,
                    directional=True, quantization="sq4", rerank_factor=8,
                    adaptive_ef=True)
index.build(train_data)
searcher = index.searcher()
searcher.set_adaptive_ef(gamma=2.0, min_ef=35, max_ef=85)
results = searcher.search(query, k=10)

# Batch search (rayon parallel)
batch_results = searcher.batch_search(queries, k=10).reshape(-1, k)
```

## Optimal Configuration

Determined by full parameter sweep on SIFT-1M (see [`tuning/SUMMARY.md`](tuning/SUMMARY.md)):

| Parameter | Value | Notes |
|---|---|---|
| PruneStrategy | DirectionalPrune | +3.4% QPS vs RobustPrune |
| R (r_max) | 32 | Hard upper bound on graph degree |
| L (l_build) | 200 | Build-time search width |
| α (alpha) | 1.2 | Pruning distance factor |
| nav_m | 32 | Navigation layer shrink factor (3 layers) |
| Quantization | SQ4 | 4-bit/dim, 64B/vector, 8× compression |
| ef_search | 50 | Query-time search width |
| po (prefetch_offset) | 8 | Two-pass prefetch distance |
| rerank | 8 | f32 rerank multiplier (SQ4) |

## Project Structure

```
RAVEN/
  src/
    distance/      # SIMD distance kernels (L1)
    memory/        # Graph storage and serialization (L2)
    graph/         # Vamana build, pruning, search, AdaptiveEf (L3)
    build/         # Parallel build pipeline (L4)
    quant/         # SQ4/SQ8/PQ quantization (L5)
    bench/         # Benchmark harness (L6)
    bin/           # Experiment binaries
    python.rs      # PyO3 Python bindings
  benches/         # Rust micro-benchmarks (divan/criterion)
  tuning/          # Parameter sweep results and reports
  experiments/     # Raw experiment output logs
  docs/            # Design documents and analysis
  ann-benchmarks/  # ann-benchmarks integration (module.py, config.yml, Dockerfile)
  pyproject.toml   # maturin build config
```

## Hardware Requirements

- x86_64 with AVX2 + F16C (minimum)
- AVX-512 recommended for best performance
- Tested on AMD Zen 4 (Zen 4 AVX-512 path)


## License

Copyright 2026 Juwan Hwang. Licensed under [Apache-2.0](LICENSE).
