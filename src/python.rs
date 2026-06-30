//! PyO3 Python 绑定 — ann-benchmarks 集成接口
//!
//! 暴露 Index (build) 和 Searcher (search) 两个类。
//! 编译：maturin develop --release --features python
//!       cargo build --release --features python

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

use crate::build::ChaCha8Rng;
use crate::graph::{VamanaBuildConfig, VamanaGraph, GraphSearcher, PruneStrategy};
use crate::quant::SQ8Dataset;
use crate::memory::serialize::Serializable;

/// RAVEN 索引（构建 + 序列化）
#[pyclass(name = "Index")]
struct PyIndex {
    graph: Option<VamanaGraph>,
    vectors: Vec<f32>,
    dim: usize,
    // 构建参数（由 __init__ 传入，build() 时使用）
    r: usize,
    l: usize,
    alpha: f32,
    nav_m: usize,
    directional: bool,
}

#[pymethods]
impl PyIndex {
    /// 创建索引配置
    ///
    /// 参数对应全参数扫描后的最优配置：
    ///   R=32, L=200, alpha=1.2, nav_m=32, directional=True
    #[new]
    #[pyo3(signature = (metric, dim, r=32, l=200, alpha=1.2, nav_m=32, directional=true))]
    fn new(metric: &str, dim: usize, r: usize, l: usize, alpha: f32, nav_m: usize, directional: bool) -> PyResult<Self> {
        let _ = metric; // L2 only for now
        Ok(PyIndex {
            graph: None,
            vectors: Vec::new(),
            dim,
            r,
            l,
            alpha,
            nav_m,
            directional,
        })
    }

    /// 构建索引
    /// X: numpy array (n, dim) float32
    fn build(&mut self, x: PyReadonlyArray2<f32>) -> PyResult<()> {
        let array = x.as_array();
        let (n, d) = (array.shape()[0], array.shape()[1]);
        if d != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("dim mismatch: expected {}, got {}", self.dim, d)
            ));
        }

        // 拷贝为连续 f32（numpy 可能不是 C-contiguous）
        self.vectors = array.as_standard_layout().to_owned().into_raw_vec();

        let config = VamanaBuildConfig {
            alpha: self.alpha,
            l_build: self.l,
            r_max: self.r,
            r_soft: (self.r as f32 * 1.5) as usize,
            max_iterations: 2,
            saturate: !self.directional,
            enable_layered_nav: true,
            nav_m: self.nav_m,
            prune_strategy: if self.directional {
                PruneStrategy::DirectionalPrune
            } else {
                PruneStrategy::RobustPrune
            },
        };

        let mut rng = ChaCha8Rng::seed_from(42);
        let graph = VamanaGraph::build(&self.vectors, self.dim, &config, &mut rng);
        self.graph = Some(graph);
        Ok(())
    }

    /// 创建搜索器（移动 graph + vectors 到 Searcher，启用 SQ8 量化）
    ///
    /// 调用后 Index 不再持有数据，Searcher 独占所有权。
    /// 这是为了绕过 GraphSearcher<'a> 的生命周期限制。
    fn searcher(&mut self) -> PyResult<PySearcher> {
        let graph = self.graph.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet, call build() first")
        })?;

        let sq8 = SQ8Dataset::build(&self.vectors, self.dim);

        Ok(PySearcher {
            vectors: std::mem::take(&mut self.vectors),
            dim: self.dim,
            graph,
            sq8,
            ef: 50,
            po: 8,
            rerank: 3,
        })
    }

    /// 保存索引到文件
    fn save(&self, path: &str) -> PyResult<()> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet")
        })?;
        graph.save(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// 从文件加载索引
    #[staticmethod]
    fn load(path: &str, dim: usize) -> PyResult<Self> {
        let graph = VamanaGraph::load(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyIndex {
            graph: Some(graph),
            vectors: Vec::new(),
            dim,
            r: 32,
            l: 200,
            alpha: 1.2,
            nav_m: 32,
            directional: true,
        })
    }
}

/// RAVEN 搜索器（SQ8 量化搜索 + f32 rerank）
///
/// 拥有 graph + vectors + sq8 的完整所有权。
/// 每次 search() 创建临时 GraphSearcher（借用内部数据），搜索后丢弃。
/// VisitedTracker 分配 ~16KB（SIFT-1M），~1µs，相对于 ~50µs 搜索可忽略。
#[pyclass(name = "Searcher")]
struct PySearcher {
    vectors: Vec<f32>,
    dim: usize,
    graph: VamanaGraph,
    sq8: SQ8Dataset,
    ef: usize,
    po: usize,
    rerank: usize,
}

#[pymethods]
impl PySearcher {
    /// 设置 ef_search
    fn set_ef(&mut self, ef: usize) {
        self.ef = ef;
    }

    /// 设置 prefetch offset
    fn set_po(&mut self, po: usize) {
        self.po = po;
    }

    /// 搜索最近邻
    /// q: numpy array (dim,) float32
    /// k: 返回的近邻数
    /// 返回: numpy array of int
    fn search(&self, q: PyReadonlyArray1<f32>, k: usize, py: Python) -> PyResult<PyObject> {
        let query = q.as_slice()?;

        let mut searcher = GraphSearcher::new(&self.vectors, &self.graph, self.ef);
        searcher.with_sq8(&self.sq8);
        searcher.with_prefetch_offset(self.po);
        searcher.with_rerank_factor(self.rerank);

        let results = searcher.search_sq8(query, k);

        let ids: Vec<usize> = results.iter().map(|(id, _)| *id as usize).collect();
        Ok(ids.to_pyarray_bound(py).into())
    }

    /// 批量搜索（多线程，rayon 并行）
    /// queries: numpy array (nq, dim) float32
    /// k: 返回的近邻数
    /// 返回: numpy array (nq, k) int
    fn batch_search(&self, queries: PyReadonlyArray2<f32>, k: usize, py: Python) -> PyResult<PyObject> {
        let array = queries.as_array();
        let (nq, d) = (array.shape()[0], array.shape()[1]);
        if d != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("dim mismatch: expected {}, got {}", self.dim, d)
            ));
        }

        let flat: Vec<f32> = array.as_standard_layout().to_owned().into_raw_vec();
        let query_refs: Vec<&[f32]> = (0..nq)
            .map(|i| &flat[i * self.dim..(i + 1) * self.dim])
            .collect();

        let mut searcher = GraphSearcher::new(&self.vectors, &self.graph, self.ef);
        searcher.with_sq8(&self.sq8);
        let results = searcher.batch_search(&query_refs, k);

        let mut ids = vec![0usize; nq * k];
        for (i, result) in results.iter().enumerate() {
            for (j, (id, _)) in result.iter().enumerate().take(k) {
                ids[i * k + j] = *id as usize;
            }
        }

        Ok(ids.to_pyarray_bound(py).into())
    }
}

/// Python 模块入口
#[pymodule]
fn raven(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_class::<PySearcher>()?;
    Ok(())
}
