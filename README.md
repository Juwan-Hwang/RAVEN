# RAVEN

**Retrieval-Aware Vector Engine with Navigation** — a high-performance Rust library for Approximate Nearest Neighbor (ANN) search.

## Overview

RAVEN implements a Vamana/DiskANN-style graph index with a unified, battle-tested query pipeline:

1. **DirectionalPrune** — Directional scan with adaptive connectivity floor, eliminating saturation padding for sparser, higher-quality graphs (+3.4% QPS vs RobustPrune)
2. **Unified Query Pipeline** (`src/pipeline.rs`) — Single accepted search stack shared by the eval harness and the Python binding, so production gets exactly what was measured: nav dedup + SQ4 upper-layer navigation, SQ4 per-dim MSE-adaptive range (H17), dimension permutation by neighbour-pair energy with segmented early abandon (H07a), BFS node reorder (H02), and build-time search-degree calibration (H20v2)
3. **Cell-Confirm HR** (`src/graph/cellconfirm.rs`) — High-recall mode: graph-located candidates confirmed via sort-free 1-bit cell fastscan with a distance-margin gate, gated by ef threshold
4. **AdaptiveEf** — Dynamic ef prediction from query→entry-point distance distribution, filling Pareto gaps between fixed-ef points

### Performance

SIFT-1M benchmark (1M vectors, dim=128, SQ4, single thread, 2026-11-05):

| ef | recall@10 | QPS |
|---:|---:|---:|
| 40 | 0.9746 | 30,540 |
| 45 | 0.9789 | 28,804 |
| 50 | 0.9827 | 26,464 |
| 55 | 0.9855 | 24,026 |
| 65 | 0.9892 | 21,408 |
| 70 | 0.9905 | 20,141 |
| 80 | 0.9926 | 18,008 |
| 100 | 0.9952 | 14,640 |

**Working point**: ef=50, recall=0.9827, QPS=26,464. Index size: 1.5 MB (64 B/vector).

**Multi-threaded**: ef=50, recall=0.9827, QPS=153,369 (5.8× single-thread, 16 threads, `batch_search`).

### Comparison (same machine, same day)

ann-benchmarks, sift-128-euclidean (1M × 128-dim), single-threaded QPS at matched recall levels:

| Algorithm | @0.95 | @0.97 | @0.98 | @0.99 | Index size |
|---|---:|---:|---:|---:|---:|
| qsgngt | 34,841 | 29,920 | **29,920** | **22,425** | 5.4 MB |
| descartes | **37,722** | **33,156** | 26,616 | 23,957 | 7.1 MB |
| kgn | 32,185 | 24,161 | 22,225 | 16,484 | 3.3 MB |
| RAVEN | 30,540¹ | 30,540 | 28,804 | 20,141 | **1.5 MB** |
| RAVEN MT16 | — | — | 153,369 (recall 0.983) | — | **1.5 MB** |

¹ RAVEN's pipeline curve starts at recall 0.9746 (ef=40); lower-recall operating points are not part of the tuned envelope.

Notes:

- RAVEN is the smallest index by 2.2–4.7× (SQ4, 64 B/vector) and the only one with a parallel batch path (`batch_query`).
- At recall 0.98, RAVEN single-thread leads descartes by +8.2%; MT16 reaches 153K QPS.

### Architecture

| Layer | Module | Description |
|---|---|---|
| L1 | `src/distance/` | SIMD distance kernels (AVX-512, AVX2, NEON, F16C, scalar) |
| L2 | `src/memory/` | Hybrid Blocked-CSR graph storage, aligned allocator, visited tracker, CRC32 serialization |
| L3 | `src/graph/` | Vamana build, DirectionalPrune, cell-confirm HR, layered navigation, AdaptiveEf, stop-trace |
| L4 | `src/build/` | Rayon parallel build pipeline, delayed prune, configurable α / R_max / L_build |
| L5 | `src/quant/` | SQ4/SQ8 scalar quantization (SQ8 asymmetric distance), BitCells 1-bit cells |
| L6 | `src/pipeline.rs` + `src/bench/` | Unified query pipeline and benchmark harness |

## Quick Start

### Build (Rust)

```bash
cargo build --release
```

### Run benchmark

```bash
# Micro-benchmarks (divan: distance kernels, graph build)
cargo bench

# End-to-end ANN benchmark: use the Python binding (below), or the
# ann-benchmarks integration under ann_benchmarks/ (module.py + config.yml +
# Dockerfile; see ann_benchmarks/algorithms/raven/). SISAP 2026 evaluation
# scripts live in sisap26/.
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
    n_jumps: 0,
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
searcher.with_prefetch_offset(16);
searcher.with_rerank_factor(5);

let results = searcher.search_sq4(&query, 10); // top-10
```

### Python API

```python
import raven
import numpy as np

# Build index (SQ4 quantization) — SQ4 runs the unified query pipeline
index = raven.Index("L2", dim=128, r=32, l=200, alpha=1.2, nav_m=32,
                    directional=True, quantization="sq4", rerank_factor=5)
index.build(train_data)  # numpy array (N, 128) float32

# Search
searcher = index.searcher()
searcher.set_ef(50)
searcher.set_po(16)  # prefetch offset
results = searcher.search(query, k=10)  # numpy array of int

# AdaptiveEf — dynamic ef prediction
searcher.set_adaptive_ef(gamma=2.0, min_ef=35, max_ef=85)
results = searcher.search(query, k=10)

# Batch search (rayon parallel)
batch_results = searcher.batch_search(queries, k=10).reshape(-1, k)

# Reuse a saved graph without rebuilding (graph cache)
index = raven.Index.load_with_vectors("graph.kgraph", train_data,
                                      quantization="sq4", rerank_factor=5)
searcher = index.searcher()
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
| Quantization | SQ4 | 4-bit/dim, 64B/vector, 8× compression, per-dim MSE-adaptive range |
| ef_search | 50 | Query-time search width |
| po (prefetch_offset) | 16 | Two-pass prefetch distance |
| rerank | 5 | f32 rerank multiplier (SQ4) |

## Project Structure

```
RAVEN/
  src/
    distance/      # SIMD distance kernels (L1)
    memory/        # Graph storage, aligned allocator, serialization (L2)
    graph/         # Vamana build, pruning, cell-confirm, search, AdaptiveEf (L3)
    build/         # Parallel build pipeline (L4)
    quant/         # SQ4/SQ8/PQ/BitCells quantization (L5)
    config/        # Runtime config, params, rules, local bench profiles
    pipeline.rs    # Unified accepted query pipeline (L6)
    counters.rs    # Search-path instrumentation counters
    python.rs      # PyO3 Python bindings
    bench/         # Benchmark harness
  benches/         # Rust micro-benchmarks (divan/criterion)
  tuning/          # Parameter sweep results and reports
  experiments/     # Raw experiment output logs
  docs/            # Design documents and analysis
  ann_benchmarks/  # ann-benchmarks integration (module.py, config.yml, Dockerfile)
  sisap26/         # SISAP 2026 evaluation runners and TIRA upload
  pyproject.toml   # maturin build config
```

## Hardware Requirements

- x86_64 with AVX2 + F16C (minimum)
- AVX-512 recommended for best performance
- ARM64 NEON supported
- Tested on AMD Zen 4 (AVX-512 path)

## License

Copyright 2026 Juwan Hwang. Licensed under [Apache-2.0](LICENSE).
