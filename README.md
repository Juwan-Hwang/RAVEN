# RAVEN

**Retrieval-Aware Vector Engine with Navigation** — a high-performance Rust library for Approximate Nearest Neighbor (ANN) search.

## Overview

RAVEN implements a Vamana/DiskANN-style graph index with a unified, battle-tested query pipeline:

1. **DirectionalPrune** — Directional scan with adaptive connectivity floor, eliminating saturation padding for sparser, higher-quality graphs (+3.4% QPS vs RobustPrune)
2. **Unified Query Pipeline** (`src/pipeline.rs`) — Single accepted search stack shared by the eval harness and the Python binding, so production gets exactly what was measured: nav dedup + SQ4 upper-layer navigation, SQ4 per-dim MSE-adaptive range (H17), dimension permutation by neighbour-pair energy with segmented early abandon (H07a), BFS node reorder (H02), and build-time search-degree calibration (H20v2)
3. **Cell-Confirm HR** (`src/graph/cellconfirm.rs`) — High-recall mode: graph-located candidates confirmed via sort-free 1-bit cell fastscan with a distance-margin gate, gated by ef threshold
4. **AdaptiveEf** — Dynamic ef prediction from query→entry-point distance distribution, filling Pareto gaps between fixed-ef points
5. **Fast default graph build** (`src/build/pipnn.rs`, `pipg.rs`) — Partition-based leaf top-k build with VNNI panel kernels and in-leaf occlusion pre-pruning, replacing iterative Vamana insertion as the default (orders of magnitude faster on large sets); NN-Descent and learned bridge edges for out-of-distribution queries are available
6. **Asymmetric int8 navigation** (`src/graph/vamana/adq.rs`) — Query-side int8 against SQ4 codes with AVX-512 VNNI batch scoring, exact-bitmap visited set, and low-dimension projected navigation codes for high-dimensional data, enabled automatically for unit-norm data
7. **Deterministic plan calibration** — Search plans (high-recall gate, truncation and projection switch points) are chosen by count models and leave-one-out calibration rather than wall-clock timing, with an OOD guard that falls back to full-dimension codes when query traffic drifts from the build set

### Performance

Single-thread QPS ratios at matched recall (RAVEN ÷ baseline, equal-recall interpolated, competitors built and measured in alternating runs on the same machine, RAVEN with its code defaults), updated 2026-11-25.

ann-benchmarks (k=10), vs Descartes (the ann-benchmarks leader):

| Dataset | recall 0.95 | recall 0.99 |
|---|---:|---:|
| sift-128-euclidean | 1.18× | 1.13–1.15× |
| nytimes-256-angular | 1.08–1.21× | 1.49–1.61× |
| fashion-mnist-784-euclidean | 1.40–1.41× | 1.22–1.23× |
| mnist-784-euclidean | 1.50–1.57× | 1.45–1.50× |
| glove-25-angular | 1.09–1.13× | 1.22–1.29× |

On sift-128 the engine also measures 1.55–2.58× vs GINGER across 0.95–0.99 (AMD Zen 4 and Zen 5).

VIBE (k=100):

| Dataset | Baseline | Ratio |
|---|---|---:|
| imagenet-clip-512 | GINGER | 1.43–1.68× @0.95 |
| imagenet-clip-512 | Glass | 1.66–1.75× @0.99, on par @0.97–0.98 |
| landmark-nomic-768 | GINGER | 1.46–1.54× @0.99 |
| yahoo-minilm-384 | GINGER | 1.11× @0.95, 1.19–1.20× @0.99 |
| yahoo-minilm-384 | Glass | 1.29–1.33× @0.99, 1.53–1.61× @0.995 |
| yandex-deep-200 (OOD) | GINGER | 1.04–1.06× @0.95, 1.08–1.10× @0.99 |
| yandex-deep-200 (OOD) | Glass | 1.27× @0.95, 1.32× @0.97 |
| laion-clip-512 (OOD) | GINGER | 1.21–1.23× @0.98, 1.12–1.16× @0.99 |
| ccnews-nomic-768 | GINGER | 1.18–1.27× @0.95 |
| deep-96 (1M) | Glass | 1.20–1.27× @0.95, 1.43–1.49× @0.99 |

Known gaps: gist-960 still trails at recall 0.90–0.97 (about 0.69–0.84× of the leader), and index memory is not yet below the leaders'.

### Architecture

| Layer | Module | Description |
|---|---|---|
| L1 | `src/distance/` | SIMD distance kernels (AVX-512, AVX2, NEON, F16C, scalar) |
| L2 | `src/memory/` | Hybrid Blocked-CSR graph storage, aligned allocator, visited tracker, CRC32 serialization |
| L3 | `src/graph/` | Vamana build, DirectionalPrune, cell-confirm HR, layered navigation, AdaptiveEf, stop-trace |
| L4 | `src/build/` | Partition-based default build, NN-Descent, Rayon parallel Vamana build, delayed prune, configurable α / R_max / L_build |
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
- AVX-512 VNNI used when present
- Tested on AMD Zen 4 and Zen 5 (AVX-512 path) and Intel Skylake-SP

## License

Copyright 2026 Juwan Hwang. Licensed under [Apache-2.0](LICENSE).
