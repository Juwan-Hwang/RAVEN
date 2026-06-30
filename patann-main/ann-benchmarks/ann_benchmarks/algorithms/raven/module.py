import numpy as np
from ..base.module import BaseANN

import raven


class Raven(BaseANN):
    """RAVEN: Retrieval-Aware Vector Engine with Navigation.

    Vamana/DiskANN-style graph index with DirectionalPrune + SQ8 quantization.
    """

    def __init__(self, metric, dim, method_param):
        self.metric = metric  # 'euclidean' or 'angular'
        self.dim = dim
        self.R = method_param.get("R", 32)
        self.L = method_param.get("L", 200)
        self.alpha = method_param.get("alpha", 1.2)
        self.nav_m = method_param.get("nav_m", 32)
        self.directional = method_param.get("directional", True)
        self.name = "raven_(R=%d, L=%d, alpha=%.1f, nav_m=%d)" % (
            self.R, self.L, self.alpha, self.nav_m
        )

    def fit(self, X):
        # ann-benchmarks passes float32 numpy arrays
        if self.metric == "angular":
            # Normalize for inner product
            import sklearn.preprocessing as preprocessing
            X = preprocessing.normalize(X, "l2", axis=1)

        self.index = raven.Index(
            "L2", self.dim,
            R=self.R, L=self.L,
            alpha=self.alpha, nav_m=self.nav_m,
            directional=self.directional,
        )
        self.index.build(X)
        self.searcher = self.index.searcher()

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.name = "raven_(R=%d, L=%d, nav_m=%d, ef=%d)" % (
            self.R, self.L, self.nav_m, ef
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
