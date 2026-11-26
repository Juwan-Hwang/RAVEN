"""RAVEN adapter for VIBE (https://github.com/vector-index-bench/vibe).

The build/search protocol here is the exact one used for all RAVEN-vs-rival measurements in the RAVEN development log
(internal evaluation harness): Vamana-style graph (pipg build, r=48, l=200), 4-bit scalar-quantized navigation codes,
f32 re-ranking of rerank_factor*k candidates, build-time plan calibrated for the evaluation k (VIBE @count).

Metrics: euclidean -> L2 on raw vectors. cosine / normalized -> rows L2-normalised into a NEW array (X is never
modified), L2 search on the unit sphere (same ranking as cosine); queries are normalised natively in search().
Zero rows stay zero (no NaN). ip is not supported.

OOD datasets: VIBE calls fit_ood(X_train, X_learn, X_learn_neighbors). The first `mdb_nl` learn queries drive
MDB "miss-driven bridge" edges at build time (RAVEN's own build-time use of the query sample; X_learn_neighbors is
not used). Whether MDB runs depends only on VIBE supplying a learn set, never on the dataset name.
"""
import os

import numpy as np

import raven

from ..base.module import BaseANN


class Raven(BaseANN):
    def __init__(self, metric, count, r, l, quantization, rerank_factor, mdb=True):
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"RAVEN does not support metric {metric!r}")
        self.metric = metric
        self.count = int(count)  # VIBE @count = evaluation k; the build-time plan is calibrated for it
        self.r = int(r)
        self.l = int(l)
        self.quantization = quantization
        self.rerank_factor = int(rerank_factor)
        self.mdb = bool(mdb)
        self.ef = None
        self.ood = False
        self.searcher = None

    def _prepare(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.metric in ("cosine", "normalized"):
            # new array; zero rows stay zero (max(.,1e-30)); identical arithmetic to the evaluation harness
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
        return X

    def _index(self, dim):
        return raven.Index(
            "euclidean",
            dim,
            r=self.r,
            l=self.l,
            quantization=self.quantization,
            rerank_factor=self.rerank_factor,
            k=self.count,
            build_threads=1,  # VIBE: every index is built with a single thread
        )

    def _finish(self, index):
        self.searcher = index.searcher(k=self.count)
        if self.metric in ("cosine", "normalized"):
            self.searcher.set_normalize(True)
        plan = self.searcher.plan()
        self.plan_hash = plan[1] if plan else "none"

    def fit(self, X):
        X = self._prepare(X)
        index = self._index(X.shape[1])
        index.build(X)
        del X
        self._finish(index)
        del index

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        if not self.mdb or X_learn is None or len(X_learn) == 0:
            return self.fit(X_train)
        self.ood = True
        X = self._prepare(X_train)
        index = self._index(X.shape[1])
        learn = np.ascontiguousarray(X_learn, dtype=np.float32)
        # library defaults for the array path (= the MDB configuration behind the yandex measurements):
        # nl=50000 efl=100 efg=250 q8=True rounds=1; learn rows are unit-normalised in the library for angular metrics
        index.build_ood(X, learn, normalize=self.metric in ("cosine", "normalized"))
        self.mdb_info = index.learn_info()
        del X, learn
        self._finish(index)
        del index

    def set_query_arguments(self, ef):
        self.ef = int(ef)
        self.searcher.set_ef(self.ef)

    def query(self, v, n):
        # ids are unique row numbers, at most n of them (fewer when ef < n)
        return self.searcher.search(np.ascontiguousarray(v, dtype=np.float32), n).view(np.int64)[:n]

    def get_additional(self):
        return {"plan_hash": getattr(self, "plan_hash", "none")}

    def __str__(self):
        return "RAVEN(metric=%s, k=%d, r=%d, l=%d, quant=%s, rerank=%d, mdb=%s, ef=%s)" % (
            self.metric,
            self.count,
            self.r,
            self.l,
            self.quantization,
            self.rerank_factor,
            "on" if self.ood else "off",
            self.ef,
        )
