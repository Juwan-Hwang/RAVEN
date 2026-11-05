import os

import numpy as np
from ..base.module import BaseANN

import raven


class Raven(BaseANN):
    """RAVEN: Retrieval-Aware Vector Engine with Navigation.

    Vamana/DiskANN-style graph index with DirectionalPrune + quantization.

    模式：
      - sq4 (默认):  4-bit/dim,  64B/vector (SIFT-128), rerank_factor=5, po=16
      - sq8:          8-bit/dim, 128B/vector (SIFT-128), rerank_factor=5, po=16

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
        self.nav_m = method_param.get("nav_m", 32)
        self.directional = method_param.get("directional", True)
        self.quantization = method_param.get("quantization", "sq4")
        self.rerank_factor = method_param.get("rerank_factor", 5)
        self.po = method_param.get("po", 16)  # prefetch offset
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

        # 图缓存：ann-benchmarks 每个 run_group 独立 fit，建图参数相同时复用同一张图，
        # 跳过重复建图（3 组共用 1 次）。图文件按建图参数命名，参数变化自动失效。
        # 注：图是 f32 建的，与 quantization 无关；量化数据集在 searcher() 阶段按需构建。
        os.makedirs("indices", exist_ok=True)
        graph_path = "indices/raven_R%d_L%d_a%.2f_navm%d_%s.kgraph" % (
            self.R, self.L, self.alpha, self.nav_m,
            "dir" if self.directional else "sat",
        )
        if os.path.exists(graph_path):
            # 复用已有图：load_with_vectors 填回向量，searcher() 走与新建图完全相同的 pipeline
            self.index = raven.Index.load_with_vectors(
                graph_path, X,
                quantization=self.quantization,
                rerank_factor=self.rerank_factor,
                threads=self.threads,
            )
        else:
            self.index = raven.Index(
                "L2", self.dim,
                r=self.R, l=self.L,
                alpha=self.alpha, nav_m=self.nav_m,
                directional=self.directional,
                quantization=self.quantization,
                rerank_factor=self.rerank_factor,
                threads=self.threads,
                adaptive_ef=True,  # 始终预计算，查询时用 set_ef/set_adaptive_ef 切换
            )
            self.index.build(X)
            self.index.save(graph_path)
        self.searcher = self.index.searcher()
        self.searcher.set_po(self.po)

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
        if self.metric == "angular":
            v = v / np.linalg.norm(v)
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
