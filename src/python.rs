//! PyO3 Python 绑定 — ann-benchmarks 集成接口
//!
//! 暴露 Index (build) 和 Searcher (search) 两个类。
//! 编译：maturin develop --release --features python
//!       cargo build --release --features python

use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::build::ChaCha8Rng;
use crate::graph::{
    AdaptiveEfConfig, GraphSearcher, PruneStrategy, VamanaBuildConfig, VamanaGraph,
};
use crate::memory::serialize::Serializable;
use crate::quant::{SQ4Dataset, SQ8Dataset};

/// 量化方式
#[derive(Clone, Copy, PartialEq)]
enum Quant {
    /// 8-bit per dimension（128B/vector, SIFT-128）
    Sq8,
    /// 4-bit per dimension（64B/vector, SIFT-128），内存减半 + 带宽减半
    Sq4,
}

/// RAVEN 索引（构建 + 序列化）
#[pyclass(name = "Index")]
struct PyIndex {
    graph: Option<VamanaGraph>,
    vectors: Vec<f32>,
    dim: usize,
    // 构建参数
    r: usize,
    l: usize,
    alpha: f32,
    nav_m: usize,
    directional: bool,
    // 搜索参数
    quant: Quant,
    rerank: usize,
    threads: usize,
    adaptive_ef: bool,
}

#[pymethods]
impl PyIndex {
    /// 创建索引配置
    ///
    /// 参数对应全参数扫描后的最优配置：
    ///   R=32, L=200, alpha=1.2, nav_m=32, directional=True
    ///   quantization="sq8" (default) 或 "sq4"
    ///   rerank_factor=5 (默认，SQ4/SQ8 均适用)
    #[new]
    #[pyo3(signature = (metric, dim, r=32, l=200, alpha=1.2, nav_m=32, directional=true, quantization="sq8", rerank_factor=5, threads=0, adaptive_ef=false))]
    fn new(
        metric: &str,
        dim: usize,
        r: usize,
        l: usize,
        alpha: f32,
        nav_m: usize,
        directional: bool,
        quantization: &str,
        rerank_factor: usize,
        threads: usize,
        adaptive_ef: bool,
    ) -> PyResult<Self> {
        let _ = metric; // L2 only
        let quant = match quantization.to_lowercase().as_str() {
            "sq8" => Quant::Sq8,
            "sq4" => Quant::Sq4,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown quantization '{}', expected 'sq8' or 'sq4'",
                    other
                )))
            }
        };
        Ok(PyIndex {
            graph: None,
            vectors: Vec::new(),
            dim,
            r,
            l,
            alpha,
            nav_m,
            directional,
            quant,
            rerank: rerank_factor,
            threads,
            adaptive_ef,
        })
    }

    /// 构建索引
    /// X: numpy array (n, dim) float32
    fn build(&mut self, x: PyReadonlyArray2<f32>) -> PyResult<()> {
        let array = x.as_array();
        let (n, d) = (array.shape()[0], array.shape()[1]);
        if d != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim mismatch: expected {}, got {}",
                self.dim, d
            )));
        }

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

    /// 创建搜索器（移动 graph + vectors 到 Searcher，启用量化搜索）
    ///
    /// 根据 quantization 参数构建 SQ8 或 SQ4 数据集。
    fn searcher(&mut self) -> PyResult<PySearcher> {
        let graph = self.graph.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet, call build() first")
        })?;

        let vectors = std::mem::take(&mut self.vectors);
        let (sq8, sq4) = match self.quant {
            Quant::Sq8 => {
                let ds = SQ8Dataset::build(&vectors, self.dim);
                (Some(ds), None)
            }
            Quant::Sq4 => {
                let ds = SQ4Dataset::build(&vectors, self.dim);
                (None, Some(ds))
            }
        };

        // 若启用 AdaptiveEf，从 layered nav 采样距离分布构建配置
        let adaptive_config = if self.adaptive_ef {
            let nav = graph.layered_nav().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "AdaptiveEf requires layered nav (enable_layered_nav=true)",
                )
            })?;
            Some(AdaptiveEfConfig::build_with_layered_nav(
                &vectors, self.dim, nav, 30, 100, 2.0,
            ))
        } else {
            None
        };

        Ok(PySearcher {
            vectors,
            dim: self.dim,
            graph,
            sq8,
            sq4,
            ef: 50,
            po: 16,
            rerank: self.rerank,
            threads: self.threads,
            adaptive_config,
            use_adaptive: false,
            cur_gamma: 2.0,
            cur_min_ef: 30,
            cur_max_ef: 100,
        })
    }

    /// 保存索引到文件
    fn save(&self, path: &str) -> PyResult<()> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index not built yet"))?;
        graph
            .save(std::path::Path::new(path))
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
            quant: Quant::Sq8,
            rerank: 3,
            threads: 0,
            adaptive_ef: false,
        })
    }
}

/// RAVEN 搜索器（量化搜索 + f32 rerank）
///
/// 拥有 graph + vectors + 量化数据集的完整所有权。
/// 根据 quantization 类型选择 search_sq8 或 search_sq4 路径。
/// 若 adaptive_config 非空，搜索时动态预测 ef（覆盖 set_ef 的值）。
#[pyclass(name = "Searcher")]
struct PySearcher {
    vectors: Vec<f32>,
    dim: usize,
    graph: VamanaGraph,
    sq8: Option<SQ8Dataset>,
    sq4: Option<SQ4Dataset>,
    ef: usize,
    po: usize,
    rerank: usize,
    threads: usize,
    adaptive_config: Option<AdaptiveEfConfig>,
    use_adaptive: bool,
    cur_gamma: f32,
    cur_min_ef: usize,
    cur_max_ef: usize,
}

impl PySearcher {
    /// 统一构建 GraphSearcher（根治配置遗漏：search 与 batch_search 共用此入口）
    ///
    /// 之前 batch_search 缺 with_prefetch_offset/with_rerank_factor 导致 recall 掉 0.6pp，
    /// 根因是两条路径各自独立配置 searcher，新增配置项只改一处漏了另一处。
    /// 现在所有配置集中在此方法，新增配置项只改这一处即可同步生效。
    fn make_searcher(&self) -> GraphSearcher<'_> {
        let mut searcher = GraphSearcher::new(&self.vectors, &self.graph, self.ef);
        searcher.with_prefetch_offset(self.po);
        searcher.with_rerank_factor(self.rerank);
        if let Some(ref sq8) = self.sq8 {
            searcher.with_sq8(sq8);
        } else if let Some(ref sq4) = self.sq4 {
            searcher.with_sq4(sq4);
        }
        if self.use_adaptive {
            if let Some(ref base) = self.adaptive_config {
                let cfg = base.with_params(self.cur_min_ef, self.cur_max_ef, self.cur_gamma);
                searcher.with_adaptive_ef(cfg);
            }
        }
        searcher
    }
}

#[pymethods]
impl PySearcher {
    /// 设置 ef_search（固定 ef 模式，禁用 adaptive）
    fn set_ef(&mut self, ef: usize) {
        self.ef = ef;
        self.use_adaptive = false;
    }

    /// 设置 AdaptiveEf 参数（gamma, min_ef, max_ef）
    ///
    /// ef_search 会被设为 max_ef 以确保 VisitedTracker 容量充足，
    /// 实际 ef 由 estimate_ef 根据 query→entry-point 距离动态决定。
    /// 需先在 Index 构造时传 adaptive_ef=true 预计算 adaptive_config。
    fn set_adaptive_ef(&mut self, gamma: f32, min_ef: usize, max_ef: usize) {
        self.cur_gamma = gamma;
        self.cur_min_ef = min_ef;
        self.cur_max_ef = max_ef;
        self.ef = max_ef; // VisitedTracker 容量上限
        self.use_adaptive = self.adaptive_config.is_some();
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
        let mut searcher = self.make_searcher();
        let results = if self.sq8.is_some() {
            searcher.search_sq8(query, k)
        } else if self.sq4.is_some() {
            searcher.search_sq4(query, k)
        } else {
            searcher.search(query, k)
        };
        let ids: Vec<usize> = results.iter().map(|(id, _)| *id as usize).collect();
        Ok(ids.to_pyarray_bound(py).into())
    }

    /// 批量搜索（多线程，rayon 并行）
    /// queries: numpy array (nq, dim) float32
    /// k: 返回的近邻数
    /// 返回: numpy array (nq, k) int
    fn batch_search(
        &self,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        py: Python,
    ) -> PyResult<PyObject> {
        let array = queries.as_array();
        let (nq, d) = (array.shape()[0], array.shape()[1]);
        if d != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim mismatch: expected {}, got {}",
                self.dim, d
            )));
        }

        let flat: Vec<f32> = array.as_standard_layout().to_owned().into_raw_vec();
        let query_refs: Vec<&[f32]> = (0..nq)
            .map(|i| &flat[i * self.dim..(i + 1) * self.dim])
            .collect();

        let mut searcher = self.make_searcher();
        let results = if self.threads > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pool.install(|| searcher.batch_search(&query_refs, k))
        } else {
            searcher.batch_search(&query_refs, k)
        };

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
