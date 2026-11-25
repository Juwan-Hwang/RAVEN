import numpy as np
from ..base.module import BaseANN

import raven


class Raven(BaseANN):
    """RAVEN: Retrieval-Aware Vector Engine with Navigation.

    Vamana/DiskANN-style graph index with DirectionalPrune + quantization.

    模式：
      - sq4 (默认):  4-bit/dim,  64B/vector (SIFT-128), rerank_factor=8
      - sq8:          8-bit/dim, 128B/vector (SIFT-128), rerank_factor=3

    AdaptiveEf:
      启用后，搜索时根据 query→entry-point 距离分布动态预测 ef，
      在固定 ef 曲线的 recall 间隙生成 Pareto 最优点。
      query_args 格式变为 [gamma, min_ef, max_ef]。
    """

    def __init__(self, metric, dim, method_param):
        self.metric = metric  # 'euclidean' or 'angular'
        self.dim = dim
        self.R = method_param.get("R", 32)
        self.L = method_param.get("L", 200)
        self.alpha = method_param.get("alpha", 1.2)
        self.alpha_core = method_param.get("alpha_core", 1.0)
        self.nav_m = method_param.get("nav_m", 32)
        self.directional = method_param.get("directional", True)
        self.quantization = method_param.get("quantization", "sq8")
        self.rerank_factor = method_param.get("rerank_factor", 3)
        self.threads = method_param.get("threads", 0)  # 0 = single-thread
        # adaptive_ef 不再是构造参数：fit 时始终预计算 adaptive_config，
        # set_query_arguments 根据参数个数(1=ef, 3=gamma,min,max)切换模式
        self.name = "raven_(R=%d, L=%d, nav_m=%d, %s, rr=%d)" % (
            self.R, self.L, self.nav_m,
            self.quantization, self.rerank_factor,
        )

    def fit(self, X):
        # ann-benchmarks passes float32 numpy arrays
        if self.metric == "angular":
            # Normalize for inner product
            import sklearn.preprocessing as preprocessing
            X = preprocessing.normalize(X, "l2", axis=1)
            # R37: zero rows (e.g. nytimes-256: 239) have no cosine; at the origin they sit at L2^2 = 1 from every
            # unit query and steal top-k slots. Move them to norm 3 (L2^2 >= 4 to any unit query) so they never rank.
            z = ~X.any(axis=1)
            if z.any():
                X = X.copy(); X[z] = 3.0 / np.sqrt(X.shape[1])

        self.index = raven.Index(
            "L2", self.dim,
            r=self.R, l=self.L,
            alpha=self.alpha, alpha_core=self.alpha_core, nav_m=self.nav_m,
            directional=self.directional,
            quantization=self.quantization,
            rerank_factor=self.rerank_factor,
            threads=self.threads,
            adaptive_ef=True,  # 始终预计算，查询时用 set_ef/set_adaptive_ef 切换
        )
        self.index.build(X)
        self.searcher = self.index.searcher()
        # R80 N1：angular 单查询归一化下沉到原生 search()（省掉每查询 numpy norm+除法 ≈2 µs；零向量同样原样保留；召回逐位一致）
        self._native_norm = self.metric == "angular" and hasattr(self.searcher, "set_normalize")
        if self._native_norm:
            self.searcher.set_normalize(True)

    def set_query_arguments(self, *args):
        if len(args) == 3:
            # adaptive 模式: [gamma, min_ef, max_ef]
            gamma, min_ef, max_ef = args
            self.searcher.set_adaptive_ef(gamma, min_ef, max_ef)
            self.name = "raven_(R=%d, L=%d, nav_m=%d, %s, rr=%d, γ=%.1f(%d,%d))" % (
                self.R, self.L, self.nav_m,
                self.quantization, self.rerank_factor,
                gamma, min_ef, max_ef,
            )
        else:
            # 固定 ef 模式
            ef = args[0]
            self.searcher.set_ef(ef)
            self.name = "raven_(R=%d, L=%d, nav_m=%d, %s, rr=%d, ef=%d)" % (
                self.R, self.L, self.nav_m,
                self.quantization, self.rerank_factor, ef,
            )

    def query(self, v, n):
        if self.metric == "angular" and not self._native_norm:
            nv = np.linalg.norm(v)
            if nv > 0:
                v = v / nv
        return self.searcher.search(v, n)

    def batch_query(self, X, n):
        if self.metric == "angular":
            import sklearn.preprocessing as preprocessing
            X = preprocessing.normalize(X, "l2", axis=1)
        self.res = self.searcher.batch_search(X, n).reshape(-1, n)

    def get_batch_results(self):
        return self.res

    def freeIndex(self):
        del self.index
        del self.searcher
