//! PyO3 Python 绑定 — ann-benchmarks 集成接口
//!
//! 暴露 Index (build) 和 Searcher (search) 两个类。
//! 编译：maturin develop --release --features python
//!       cargo build --release --features python

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
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
    /// R98 H186：零拷贝建图时持有调用方 numpy 数组（C 连续 f32），`vectors` 推迟到 searcher() 才拷贝
    src: Option<Py<PyAny>>,
    dim: usize,
    // 构建参数
    r: usize,
    l: usize,
    alpha: f32,
    /// Pass 1 核心方向性判据的遮挡阈值（第十三轮诊断：0.93 最优，3-seed 验证）
    alpha_core: f32,
    nav_m: usize,
    directional: bool,
    // 搜索参数
    quant: Quant,
    rerank: usize,
    threads: usize,
    adaptive_ef: bool,
    /// R109：评测 k（None → RAVEN_K → 10），构建期计划（KL 大 k、HR/LPX/SDX 标定、LP 保真度门）按此 k
    k: Option<usize>,
    /// R109：build()/searcher() 用的线程数（0 = 旧行为：rayon 全局池 + available_parallelism；VIBE 用 1）
    build_threads: usize,
    /// R109：OOD 学习查询（前 nl 行，行主序 f32）与 MDB 参数；Some 时 build() 走数组 MDB
    learn: Option<(Vec<f32>, crate::build::mdb::MdbParams)>,
}

/// R109：评测 k 的解析顺序：显式参数 > 环境变量 RAVEN_K > 10
fn resolve_k(explicit: Option<usize>) -> usize {
    let env_k: Option<usize> = std::env::var("RAVEN_K").ok().and_then(|v| v.parse().ok());
    match (explicit, env_k) {
        (Some(k), Some(e)) if e != k => { eprintln!("[raven] k={k} given explicitly; RAVEN_K={e} ignored"); k }
        (Some(k), _) => k,
        (None, Some(e)) => e,
        (None, None) => 10,
    }
}

/// R109：在固定线程数的 rayon 池里运行 f，并把构建期自管线程数（pipeline::PREP_NT）设为同值；nt = 0 → 旧行为（直接运行）
fn with_threads<R: Send>(nt: usize, f: impl FnOnce() -> R + Send) -> PyResult<R> {
    if nt == 0 { return Ok(f()); }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let prev = crate::pipeline::PREP_NT.swap(nt, std::sync::atomic::Ordering::Relaxed);
    let r = pool.install(f);
    crate::pipeline::PREP_NT.store(prev, std::sync::atomic::Ordering::Relaxed);
    Ok(r)
}

#[pymethods]
impl PyIndex {
    /// 创建索引配置
    ///
    /// 参数对应全参数扫描后的最优配置：
    ///   R=40, L=384, alpha=1.2, nav_m=32, directional=True
    ///   quantization="sq8" (default) 或 "sq4"
    ///   rerank_factor=5 (默认，SQ4/SQ8 均适用)
    #[new]
    ///
    /// R109 新增：`k`（评测 k；None → RAVEN_K → 10）、`build_threads`（build/searcher 线程数；0 = 旧行为，VIBE 用 1）。
    #[pyo3(signature = (metric, dim, r=40, l=384, alpha=1.2, alpha_core=1.0, nav_m=32, directional=true, quantization="sq8", rerank_factor=5, threads=0, adaptive_ef=false, k=None, build_threads=0))]
    fn new(
        metric: &str,
        dim: usize,
        r: usize,
        l: usize,
        alpha: f32,
        alpha_core: f32,
        nav_m: usize,
        directional: bool,
        quantization: &str,
        rerank_factor: usize,
        threads: usize,
        adaptive_ef: bool,
        k: Option<usize>,
        build_threads: usize,
    ) -> PyResult<Self> {
        if k == Some(0) { return Err(pyo3::exceptions::PyValueError::new_err("k must be >= 1")); }
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
            src: None,
            dim,
            r,
            l,
            alpha,
            alpha_core,
            nav_m,
            directional,
            quant,
            rerank: rerank_factor,
            threads,
            adaptive_ef,
            k,
            build_threads,
            learn: None,
        })
    }

    /// R109：设置 OOD 学习查询（VIBE `fit_ood` 的 X_learn），之后的 build() 用它跑 MDB 漏检驱动桥边（与 RAVEN_MDB_LEARN 文件路径同一段代码）。
    /// learn: (nl, dim) float32；只保留前 nl 行。参数解析：显式参数 > 同名环境变量 RAVEN_MDB_* > 数组路径缺省
    /// （yandex 便宜档：nl=50000 efl=100 efg=250 q8=True rounds=1 km=100 add=8 mode="ER" minv=1 a2=1.44 normalize=True）。
    /// normalize=True 时在库内把学习查询单位化（角度/余弦集）；欧氏集应传 normalize=False。
    #[pyo3(signature = (learn, nl=None, efl=None, efg=None, km=None, add=None, mode=None, minv=None, a2=None, rounds=None, normalize=None, q8=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_learn(&mut self, learn: PyReadonlyArray2<f32>, nl: Option<usize>, efl: Option<usize>, efg: Option<usize>, km: Option<usize>, add: Option<usize>,
                 mode: Option<String>, minv: Option<u32>, a2: Option<f32>, rounds: Option<usize>, normalize: Option<bool>, q8: Option<bool>) -> PyResult<()> {
        let a = learn.as_array();
        let (ln, d) = (a.shape()[0], a.shape()[1]);
        if d != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("learn dim mismatch: expected {}, got {}", self.dim, d)));
        }
        let mut p = crate::build::mdb::MdbParams::array_default().env_override();
        if let Some(v) = nl { p.nl = v; } if let Some(v) = efl { p.efl = v; } if let Some(v) = efg { p.efg = v; }
        if let Some(v) = km { p.km = v; } if let Some(v) = add { p.add = v; } if let Some(v) = mode { p.mode = v; }
        if let Some(v) = minv { p.minv = v; } if let Some(v) = a2 { p.a2 = v; } if let Some(v) = rounds { p.rounds = v; }
        if let Some(v) = normalize { p.norm = v; } if let Some(v) = q8 { p.q8 = v; }
        let rows = p.nl.min(ln);
        if rows == 0 { self.learn = None; return Ok(()); }
        let sub = a.slice(numpy::ndarray::s![..rows, ..]);
        let v: Vec<f32> = sub.as_standard_layout().to_owned().into_raw_vec();
        self.learn = Some((v, p));
        Ok(())
    }

    /// R109：清除学习查询（之后的 build() 不跑数组 MDB）
    fn clear_learn(&mut self) { self.learn = None; }

    /// R109：当前学习查询的 (行数, MDB 参数字符串)；未设置时 None
    fn learn_info(&self) -> Option<(usize, String)> {
        self.learn.as_ref().map(|(v, p)| (v.len() / self.dim.max(1), format!("{:?}", p)))
    }

    /// R109：OOD 建图 = set_learn(learn, **kw) + build(x)
    #[pyo3(signature = (x, learn, nl=None, efl=None, efg=None, km=None, add=None, mode=None, minv=None, a2=None, rounds=None, normalize=None, q8=None))]
    #[allow(clippy::too_many_arguments)]
    fn build_ood(&mut self, x: PyReadonlyArray2<f32>, learn: PyReadonlyArray2<f32>, nl: Option<usize>, efl: Option<usize>, efg: Option<usize>, km: Option<usize>, add: Option<usize>,
                 mode: Option<String>, minv: Option<u32>, a2: Option<f32>, rounds: Option<usize>, normalize: Option<bool>, q8: Option<bool>) -> PyResult<()> {
        self.set_learn(learn, nl, efl, efg, km, add, mode, minv, a2, rounds, normalize, q8)?;
        self.build(x)
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

        // R98 H186（imagenet 建图峰值内存 ▼+17%）：C 连续 f32 输入直接借用建图，不再先私拷一份 n×d f32；
        // 原始向量只在 searcher() 时才拷贝（build+save 路径从不需要）。RAVEN_BUILD_ZC=0 恢复旧行为。
        let zc = std::env::var("RAVEN_BUILD_ZC").map(|v| v != "0").unwrap_or(true) && x.is_c_contiguous();
        if zc {
            self.vectors = Vec::new();
            self.src = Some(x.as_any().clone().unbind());
        } else {
            self.vectors = array.as_standard_layout().to_owned().into_raw_vec();
            self.src = None;
        }

        // alpha_core 通过环境变量传入 robust_prune（第十三轮诊断发现 0.93 严格 Pareto 支配 1.0）
        std::env::set_var("RAVEN_ALPHA_CORE", self.alpha_core.to_string());

        let config = VamanaBuildConfig {
            alpha: self.alpha,
            l_build: self.l,
            r_max: self.r,
            r_soft: (self.r as f32 * 1.5) as usize,
            max_iterations: 2,
            n_jumps: 0,
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
        let dim = self.dim;
        let learn = self.learn.as_ref().map(|(v, p)| (v.as_slice(), p));
        let graph = if zc {
            let xs = x.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            with_threads(self.build_threads, || VamanaGraph::build_ex(xs, dim, &config, &mut rng, learn))?
        } else {
            let vs = &self.vectors;
            with_threads(self.build_threads, || VamanaGraph::build_ex(vs, dim, &config, &mut rng, learn))?
        };
        self.graph = Some(graph);
        Ok(())
    }

    /// 创建搜索器（移动 graph + vectors 到 Searcher，启用量化搜索）
    ///
    /// 根据 quantization 参数构建 SQ8 或 SQ4 数据集。
    /// R109：k（评测 k）解析顺序 = 本参数 > Index(k=…) > RAVEN_K > 10。
    #[pyo3(signature = (k=None))]
    fn searcher(&mut self, py: Python<'_>, k: Option<usize>) -> PyResult<PySearcher> {
        if k == Some(0) { return Err(pyo3::exceptions::PyValueError::new_err("k must be >= 1")); }
        if self.vectors.is_empty() {
            if let Some(src) = self.src.take() {
                let a: PyReadonlyArray2<f32> = src.bind(py).extract()?;
                self.vectors = a.as_array().as_standard_layout().to_owned().into_raw_vec();
            }
        }
        let graph = self.graph.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet, call build() first")
        })?;

        let vectors = std::mem::take(&mut self.vectors);
        // 接受栈（raven::pipeline）：SQ4 时默认开启；RAVEN_PIPELINE=0 回到旧路径
        let use_pipe = self.quant == Quant::Sq4
            && std::env::var("RAVEN_PIPELINE").map(|v| v != "0").unwrap_or(true);
        if use_pipe {
            let mut o = crate::pipeline::PipelineOpts::default();
            o.rerank = self.rerank;
            // R43-3：评测 k（VIBE k=100）告知构建期（KL 大 k 计划、HR/LPX 标定）。
            // R109：k 为正式参数（searcher(k) > Index(k) > RAVEN_K > 10），库内一律经 PipelineOpts.k 传递
            o.k = resolve_k(k.or(self.k));
            if let Some(sd) = std::env::var("RAVEN_SD").ok().and_then(|v| v.parse().ok()) { o.search_degree = Some(sd); }
            let dim = self.dim;
            let p = with_threads(self.build_threads, move || {
                // R28 组合 1：平台默认的建图后饱和填充（RAVEN_FILL 覆盖，=0 关）
                let graph = match crate::graph::vamana::fill_default() { Some((d, ef)) => graph.saturate_fill(&vectors, dim, d, ef), None => graph };
                crate::pipeline::prepare(graph, vectors, dim, &o)
            })?;
            return Ok(PySearcher {
                vectors: Vec::new(), dim: self.dim, graph: None, sq8: None, sq4: None,
                ef: 50, po: 16, rerank: self.rerank, threads: self.threads,
                adaptive_config: None, use_adaptive: false, cur_gamma: 2.0, cur_min_ef: 30, cur_max_ef: 100,
                cache: std::sync::Mutex::new(Vec::new()), reuse: true, normalize: false,
                pipe: Some(p),
            });
        }
        let (sq8, sq4) = match self.quant {
            Quant::Sq8 => {
                let ds = SQ8Dataset::build(&vectors, self.dim);
                (Some(ds), None)
            }
            Quant::Sq4 => {
                // H17 (accepted 2026-10-30): per-dim MSE-adaptive range; RAVEN_SQ4_CLIP overrides (0 = old min/max)
                let clip: f32 = std::env::var("RAVEN_SQ4_CLIP").ok().and_then(|v| v.parse().ok()).unwrap_or(-1.0);
                let ds = SQ4Dataset::build_clipped(&vectors, self.dim, clip);
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
            graph: Some(graph),
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
            cache: std::sync::Mutex::new(Vec::new()),
            reuse: true,
            normalize: false,
            pipe: None,
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

    /// R28 D：从图文件 + 原始向量加载（可直接走 SQ4 流水线；用于复用已缓存的图，无需重建）
    #[staticmethod]
    #[pyo3(signature = (path, x, quantization="sq4", rerank_factor=10, threads=0))]
    fn load_with_vectors(path: &str, x: PyReadonlyArray2<f32>, quantization: &str, rerank_factor: usize, threads: usize) -> PyResult<Self> {
        let graph = VamanaGraph::load(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let array = x.as_array();
        let dim = array.shape()[1];
        let quant = if quantization.eq_ignore_ascii_case("sq8") { Quant::Sq8 } else { Quant::Sq4 };
        Ok(PyIndex {
            graph: Some(graph),
            vectors: array.as_standard_layout().to_owned().into_raw_vec(),
            src: None,
            dim, r: 48, l: 200, alpha: 1.2, alpha_core: 1.0, nav_m: 32, directional: true,
            quant, rerank: rerank_factor, threads, adaptive_ef: false,
            k: None, build_threads: 0, learn: None,
        })
    }

    /// 从文件加载索引
    #[staticmethod]
    fn load(path: &str, dim: usize) -> PyResult<Self> {
        let graph = VamanaGraph::load(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyIndex {
            graph: Some(graph),
            vectors: Vec::new(),
            src: None,
            dim,
            r: 32,
            l: 200,
            alpha: 1.2,
            alpha_core: 1.0,
            nav_m: 32,
            directional: true,
            quant: Quant::Sq8,
            rerank: 3,
            threads: 0,
            adaptive_ef: false,
            k: None,
            build_threads: 0,
            learn: None,
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
    graph: Option<VamanaGraph>,
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
    /// R28 D：按 ef 缓存的流水线 searcher（visited 位图 / 池 / history 等每查询复用，不再每次查询重新分配）。
    /// 必须声明在 `pipe` 之前：字段按声明顺序析构，缓存的 searcher 借用 `pipe` 的数据，须先于它释放。
    /// try_lock 失败（另一线程正持有）时退回临时 searcher，语义不变。
    cache: std::sync::Mutex<Vec<((usize, usize), CachedSearcher)>>,
    /// R28 D：false = 旧行为（每次查询新建 searcher），仅用于前后对比；默认 true
    reuse: bool,
    /// R80 N1：search() 内做 L2 行归一化（angular 数据集；替代 Python 侧 `v / np.linalg.norm(v)`，ann-b 每查询省 ≈2 µs）。默认 false。
    normalize: bool,
    pipe: Option<crate::pipeline::Prepared>,
}

/// R28 D：缓存的 searcher。生命周期擦除为 'static：它只借用同一 PySearcher 的 `pipe`（pyclass 对象在 Python 堆上不移动，
/// `pipe` 构造后不再替换，且缓存先于 `pipe` 析构），故借用在缓存存活期间始终有效。
struct CachedSearcher(GraphSearcher<'static>);
// SAFETY：GraphSearcher 只含对只读索引数据的引用与本 searcher 独占的缓冲区；访问由 Mutex 串行化。
unsafe impl Send for CachedSearcher {}

/// 每个 PySearcher 最多缓存的不同 ef 个数
const SEARCHER_CACHE_MAX: usize = 4;

impl PySearcher {
    /// R28 D：用按 ef 缓存的 searcher 执行 `f`（仅流水线路径）
    fn with_pipe_searcher<R>(&self, p: &crate::pipeline::Prepared, ef: usize, k: usize, f: impl FnOnce(&mut GraphSearcher<'_>) -> R) -> R {
        if self.reuse {
            if let Ok(mut c) = self.cache.try_lock() {
                let idx = match c.iter().position(|e| e.0 == (ef, k.max(10))) {
                    Some(i) => i,
                    None => {
                        if c.len() >= SEARCHER_CACHE_MAX { c.remove(0); }
                        // SAFETY：见 CachedSearcher
                        let s: GraphSearcher<'static> = unsafe { std::mem::transmute::<GraphSearcher<'_>, GraphSearcher<'static>>(p.searcher_k(ef, k)) };
                        c.push(((ef, k.max(10)), CachedSearcher(s)));
                        c.len() - 1
                    }
                };
                return f(&mut c[idx].1 .0);
            }
        }
        let mut s = p.searcher_k(ef, k);
        f(&mut s)
    }

    /// 统一构建 GraphSearcher（根治配置遗漏：search 与 batch_search 共用此入口）
    ///
    /// 之前 batch_search 缺 with_prefetch_offset/with_rerank_factor 导致 recall 掉 0.6pp，
    /// 根因是两条路径各自独立配置 searcher，新增配置项只改一处漏了另一处。
    /// 现在所有配置集中在此方法，新增配置项只改这一处即可同步生效。
    fn make_searcher(&self) -> GraphSearcher<'_> {
        let mut searcher = GraphSearcher::new(&self.vectors, self.graph.as_ref().expect("legacy graph"), self.ef);
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

    /// R109：构建期计划 (PLAN 字符串, FNV-1a 64 哈希十六进制)；非流水线路径返回 None。用于核对两次构建/两种入口逐位一致。
    fn plan(&self) -> Option<(String, String)> {
        self.pipe.as_ref().map(|p| (p.plan_string(), format!("{:016x}", p.plan_hash())))
    }

    /// R80 N1：search() 内 L2 归一化开关（angular 数据集用；batch_search 不受影响）
    fn set_normalize(&mut self, on: bool) {
        self.normalize = on;
    }

    /// R28 D：searcher 复用开关（默认开；关 = 旧行为，每次查询新建 searcher，仅供前后对比）
    fn set_searcher_reuse(&mut self, on: bool) {
        self.reuse = on;
        if !on { self.cache.lock().map(|mut c| c.clear()).ok(); }
    }

    /// 搜索最近邻
    /// q: numpy array (dim,) float32
    /// k: 返回的近邻数
    /// 返回: numpy array of int
    fn search(&self, q: PyReadonlyArray1<f32>, k: usize, py: Python) -> PyResult<PyObject> {
        let raw = q.as_slice()?;
        let mut sbuf = [0f32; 1024];
        let mut hbuf: Vec<f32>;
        let query: &[f32] = if self.normalize {
            let n2: f32 = raw.iter().map(|x| x * x).sum();
            let inv = if n2 > 0.0 { 1.0 / n2.sqrt() } else { 1.0 };
            let dst: &mut [f32] = if raw.len() <= sbuf.len() { &mut sbuf[..raw.len()] } else { hbuf = vec![0f32; raw.len()]; &mut hbuf[..] };
            for (d, x) in dst.iter_mut().zip(raw) { *d = x * inv; }
            dst
        } else { raw };
        if let Some(p) = &self.pipe {
            if p.ood_pending() {
                let r = self.with_pipe_searcher(p, self.ef, k, |s| p.run_auto(s, query, k));
                if p.ood_observe(query, &r) { self.cache.lock().map(|mut c| c.clear()).ok(); }
                let ids: Vec<usize> = r.iter().map(|x| p.orig_id(x.0) as usize).collect();
                return Ok(ids.to_pyarray_bound(py).into());
            }
            let ids: Vec<usize> = self.with_pipe_searcher(p, self.ef, k, |s| p.run_auto(s, query, k).iter().map(|x| p.orig_id(x.0) as usize).collect());
            return Ok(ids.to_pyarray_bound(py).into());
        }
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

        if let Some(p) = &self.pipe {
            use rayon::prelude::*;
            let ef = self.ef;
            let run = || -> Vec<Vec<u32>> {
                query_refs.par_iter().map_init(|| p.searcher(ef), |s, q| p.run_auto(s, q, k).iter().map(|x| p.orig_id(x.0)).collect()).collect()
            };
            let res: Vec<Vec<u32>> = if self.threads > 0 {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(self.threads).build()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                pool.install(run)
            } else {
                if self.reuse { self.with_pipe_searcher(p, ef, k, |s| query_refs.iter().map(|q| p.run_auto(s, q, k).iter().map(|x| p.orig_id(x.0)).collect()).collect()) }
                else { query_refs.iter().map(|q| { let mut s = p.searcher(ef); p.run_auto(&mut s, q, k).iter().map(|x| p.orig_id(x.0)).collect() }).collect() }
            };
            let mut ids = vec![0usize; nq * k];
            for (i, r) in res.iter().enumerate() { for (j, &id) in r.iter().enumerate().take(k) { ids[i * k + j] = id as usize; } }
            return Ok(ids.to_pyarray_bound(py).into());
        }
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

/// H173b 诊断：返回 (enabled_counters, enabled_prof, [full, quant, hops, nav, partial, partial_dims, edges], prof[8])；未开 feature 时计数为 0
#[pyfunction]
fn _diag_stats() -> (bool, bool, Vec<u64>, Vec<u64>) {
    let (f, q, h) = crate::counters::snapshot();
    let (pc, pd) = crate::counters::partial_snapshot();
    (crate::counters::ENABLED, cfg!(feature = "diag_prof"), vec![f, q, h, crate::counters::nav_snapshot(), pc, pd, crate::counters::EDGES.load(std::sync::atomic::Ordering::Relaxed)], crate::counters::prof::snapshot().to_vec())
}
/// H173b 诊断：清零计数器与分段计时
#[pyfunction]
fn _diag_reset() { crate::counters::reset(); crate::counters::prof::reset(); }

/// Python 模块入口
#[pymodule]
fn raven(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_class::<PySearcher>()?;
    m.add_function(wrap_pyfunction!(_diag_stats, m)?)?;
    m.add_function(wrap_pyfunction!(_diag_reset, m)?)?;
    Ok(())
}
