# RAVEN

**Retrieval-Aware Vector Engine with Navigation** — a high-performance Rust library for Approximate Nearest Neighbor (ANN) search.

## Overview

RAVEN implements a Vamana/DiskANN-style graph index with three integrated research contributions:

1. **DirectionalPrune** — Directional scan with adaptive connectivity floor, eliminating saturation padding for sparser, higher-quality graphs (+3.4% QPS vs RobustPrune)
2. **SQ8 Quantized Search** — 8-bit scalar quantization for graph navigation with f32 rerank, reducing memory bandwidth by 4× (+18.1% QPS)
3. **Branchless Visited Check** — Eliminates branch misprediction in the hot search loop (+2.7% QPS)

### Performance

SIFT-1M benchmark (1M vectors, dim=128, SQ8 RAW, Zen 4 AVX-512):

| ef | recall@10 | QPS |
|---:|---:|---:|
| 40 | 0.9463 | 22,387 |
| 50 | 0.9629 | 18,671 |
| 60 | 0.9730 | 16,600 |
| 80 | 0.9840 | 12,680 |
| 100 | 0.9897 | 10,561 |

**Working point**: ef=50, recall=0.9629, QPS≈18,671, CV=1.8%

Full ef-QPS curve: [`tuning/final_ef_sweep_navm32.csv`](tuning/final_ef_sweep_navm32.csv)

### Architecture (6-layer design)

| Layer | Module | Description |
|---|---|---|
| L1 | `src/distance/` | SIMD distance kernels (AVX-512, AVX2, F16C, scalar) |
| L2 | `src/memory/` | Hybrid Blocked-CSR graph storage, visited tracker, CRC32 serialization |
| L3 | `src/graph/` | Vamana build, RobustPrune / DirectionalPrune, layered navigation, graph search |
| L4 | `src/build/` | Rayon parallel build pipeline, configurable α / R_max / L_build |
| L5 | `src/quant/` | SQ8 scalar quantization, PQ4/PQ8 product quantization |
| L6 | `src/bench/` | Benchmark harness |

## Quick Start

### Build (Rust)

```bash
cargo build --release
```

### Run benchmark

```bash
# Download SIFT-1M dataset to data/sift/
cargo run --release --bin flagship_bench
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
use raven::quant::SQ8Dataset;

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

// SQ8 quantized search
let sq8 = SQ8Dataset::build(&vectors, dim);
let mut searcher = GraphSearcher::new(&vectors, &graph, 50);
searcher.with_sq8(&sq8);
searcher.with_prefetch_offset(8);
searcher.with_rerank_factor(3);

let results = searcher.search_sq8(&query, 10); // top-10
```

### Python API

```python
import raven
import numpy as np

# Build index
index = raven.Index("L2", dim=128, r=32, l=200, alpha=1.2, nav_m=32, directional=True)
index.build(train_data)  # numpy array (N, 128) float32

# Search
searcher = index.searcher()
searcher.set_ef(50)
results = searcher.search(query, k=10)  # numpy array of int

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
| ef_search | 50 | Query-time search width |
| po (prefetch_offset) | 8 | Two-pass prefetch distance |
| rerank | 3 | f32 rerank multiplier |

## Project Structure

```
RAVEN/
  src/
    distance/      # SIMD distance kernels (L1)
    memory/        # Graph storage and serialization (L2)
    graph/         # Vamana build, pruning, search (L3)
    build/         # Parallel build pipeline (L4)
    quant/         # SQ8/PQ quantization (L5)
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

## Documentation

- [设计文档.md](docs/设计文档.md) — Full technical design document (Chinese)
- [优化方案.md](docs/优化方案.md) — Optimization plan with A/B experiment results
- [tuning/SUMMARY.md](tuning/SUMMARY.md) — Parameter sweep report

## License

Copyright 2026 Juwan Hwang. Licensed under [Apache-2.0](LICENSE).
