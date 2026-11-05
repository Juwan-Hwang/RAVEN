# RAVEN

**Retrieval-Aware Vector Engine with Navigation** — a high-performance Rust library for Approximate Nearest Neighbor (ANN) search.

## Overview

RAVEN implements a Vamana/DiskANN-style graph index with a unified, battle-tested query pipeline:

1. **DirectionalPrune** — Directional scan with adaptive connectivity floor, eliminating saturation padding for sparser, higher-quality graphs (+3.4% QPS vs RobustPrune)
2. **Unified Query Pipeline** (`src/pipeline.rs`) — Single accepted search stack shared by the eval harness and the Python binding, so production gets exactly what was measured: nav dedup + SQ4 upper-layer navigation, SQ4 per-dim MSE-adaptive range (H17), dimension permutation by neighbour-pair energy with segmented early abandon (H07a), BFS node reorder (H02), and build-time search-degree calibration (H20v2)
3. **Cell-Confirm HR** (`src/graph/cellconfirm.rs`) — High-recall mode: graph-located candidates confirmed via sort-free 1-bit cell fastscan with a distance-margin gate, gated by ef threshold
4. **AdaptiveEf** — Dynamic ef prediction from query→entry-point distance distribution, filling Pareto gaps between fixed-ef points

### Performance

All numbers below are single-thread, k=10, ann-benchmarks recall (a returned point counts if its distance is ≤ the true 10th-NN distance + 1e-3).

#### SIFT-1M, same machine, same process (2026-11-05)

Machine: Instabox microVM, AMD EPYC (Zen 5), 4 vCPU / 8 GiB, Debian 13. Both engines run through their Python bindings in one process, search pinned to one core, 10,000 queries, median of 3 passes, two interleaved rounds (RAVEN→Descartes, then Descartes→RAVEN). RAVEN = this repository at `f88ad5d` with the configuration in this README (R=32, L=200, α=1.2, nav_m=32, SQ4, rerank 5, po 16). Descartes = `py01ai` with the ann-benchmarks configuration M=64, S=2, L=384 (ex=0 and ex=1 points).

QPS at a recall level is interpolated linearly along each engine's Pareto front; the range covers the two rounds.

| Algorithm | @0.90 | @0.95 | @0.97 | @0.98 | @0.99 |
|---|---:|---:|---:|---:|---:|
| RAVEN | 44.3–45.5K | 35.0–35.2K | 28.1K | 22.9–23.0K | 16.5K |
| Descartes | 39.4–40.1K | 29.7–30.7K | 24.1–25.2K | 21.3–22.0K | 17.5–17.7K |
| RAVEN / Descartes | 1.10–1.16× | 1.14–1.18× | 1.11–1.16× | 1.04–1.08× | **0.93–0.94×** |

At 0.99 Descartes is ahead on the interpolated front (taking instead the best measured point with recall ≥ 0.99, the two are level: 15.9–16.1K vs 15.6–15.8K). qsgngt and kgn were not run on this machine.

RAVEN points (same run, QPS = range over the two rounds):

| ef | recall@10 | QPS |
|---:|---:|---:|
| 25 | 0.9442 | 37.0–37.3K |
| 30 | 0.9587 | 31.9–32.2K |
| 40 | 0.9746 | 25.4–26.0K |
| 50 | 0.9827 | 21.4–21.8K |
| 60 | 0.9875 | 18.2–18.8K |
| 70 | 0.9905 | 15.9–16.1K |
| 100 | 0.9952 | 11.7–12.1K |

**Index size**: about 1.3 GB for SIFT-1M (process RSS grew by 1,330 MB during the build on this machine; ann-benchmarks reports `index_size` ≈ 1,299,700 kB). This includes the f32 vectors used for reranking; the SQ4 codes alone are 64 B/vector. An earlier version of this README said "1.5 MB" — that was the kB figure read as MB.

#### Earlier table (superseded; not a same-machine comparison)

Kept for reference, with corrections. The rival rows are the ann-benchmarks website values (their own hardware and harness: best run with recall ≥ the level). The RAVEN row was measured with this repository's own harness on the author's desktop (AMD Ryzen 7 8845H) and uses a different rule: the first measured ef point at or above the level, not an interpolated or best-run value. The two recall/QPS conventions differ, so the rows are not directly comparable.

| Algorithm | @0.95 | @0.97 | @0.98 | @0.99 |
|---|---:|---:|---:|---:|
| qsgngt (website) | 34,841 | 29,920 | **29,920** | 22,425 |
| descartes (website) | **37,722** | **33,156** | 26,616 | **23,957** |
| kgn (website) | 32,185 | 24,161 | 22,225 | 16,484 |
| RAVEN (desktop) | 30,540¹ | 30,540¹ | 28,804 | 20,141 |

¹ The @0.95 and @0.97 entries are the same measurement: ef=40, recall 0.9746 (the lowest ef that was measured).

- At 0.98, qsgngt leads this table (29,920 vs RAVEN 28,804).
- The index-size column has been removed. RAVEN's value was a kB/MB unit error (see above), and so were the rival values: ann-benchmarks reports `index_size` in kB, and the current sift-128 page (checked 2026-11-05) lists qsgngt at 5,183,608–6,535,088 kB, i.e. about 5.2–6.5 GB, not "5.4 MB". Descartes and kgn no longer appear on that page, so their old "7.1 MB" / "3.3 MB" entries could not be re-checked; read as kB they would be about 7.1 GB and 3.3 GB. No index-size ratio is claimed here.
- The rival rows were copied from the website when this table was first written; Descartes and kgn are not on the current page, so those two rows cannot be re-checked there.

#### Multi-threaded (separate; not comparable with the single-thread tables)

| Engine | Threads | API | ef | recall@10 | QPS | Machine |
|---|---:|---|---:|---:|---:|---|
| RAVEN | 16 | `batch_search` | 50 | 0.9827 | 153,369 | author's desktop, AMD Ryzen 7 8845H (5.8× its own single-thread figure) |

No rival was measured multi-threaded, so there is no multi-threaded comparison.

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
