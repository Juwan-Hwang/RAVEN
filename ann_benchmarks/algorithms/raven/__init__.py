"""RAVEN ann-benchmarks wrapper.

设计文档第六层模式一：ann_benchmarks/algorithms/yourlib/
接入：__init__.py / Dockerfile / config.yml

本 wrapper 通过 subprocess 调用 Rust 编译的 raven_ann_bench 二进制，
避免引入 PyO3 依赖。数据通过临时文件传递（raw binary 格式）。

ann-benchmarks 框架期望的 BaseANN 接口：
  - fit(X):       构建索引（X 为 numpy 数组）
  - query(q, k):  查询 k 近邻
  - use_threads(): 是否多线程
"""

import os
import json
import tempfile
import subprocess
import numpy as np


class RAVEN:
    """RAVEN ann-benchmarks wrapper.

    参数对应设计文档第 449-458 行的参数扫描空间：
      M (= r_max):       [16, 32, 64]
      ef_construction (= l_build): [100, 200, 400]
      alpha:             [1.0, 1.2, 1.5]
      kernel:            auto (由三阶段选择决定)
      ef_search:         查询时搜索宽度
    """

    def __init__(self, M, ef_construction, alpha, ef_search):
        """初始化 RAVEN 参数。

        Args:
            M: 最大出度（设计文档 r_max）
            ef_construction: 构建期搜索宽度（设计文档 l_build）
            alpha: RobustPrune 的 α 参数
            ef_search: 查询期搜索宽度
        """
        self.M = M
        self.ef_construction = ef_construction
        self.alpha = alpha
        self.ef_search = ef_search
        self.name = f"RAVEN(M={M}, ef_c={ef_construction}, alpha={alpha}, ef_s={ef_search})"

        # RAVEN 二进制路径（由 Dockerfile 设置）
        self.raven_bin = os.environ.get("RAVEN_BIN", "raven_ann_bench")
        self.dim = 0
        self.n = 0
        self._train_file = None
        self._built = False

    def fit(self, X):
        """构建索引。

        ann-benchmarks 框架调用此方法传入训练数据。
        X 为 numpy 数组，shape=(n, dim)。
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.n, self.dim = X.shape

        # 写入临时文件（raw binary）
        self._train_file = tempfile.mktemp(suffix=".bin", prefix="raven_train_")
        X.tofile(self._train_file)

        # 仅构建索引，不查询（query 阶段单独调用）
        cmd = [
            self.raven_bin,
            "--train", self._train_file,
            "--dim", str(self.dim),
            "--n", str(self.n),
            "--alpha", str(self.alpha),
            "--l-build", str(self.ef_construction),
            "--r-max", str(self.M),
            "--ef-search", str(self.ef_search),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"RAVEN build failed: {result.stderr}")

        self._built = True
        return json.loads(result.stdout)

    def query(self, q, k):
        """查询 k 近邻。

        ann-benchmarks 框架调用此方法进行单次查询。
        通过 --output 参数输出邻居 ID（raw binary, i32），与 query_batch 一致。
        """
        if not self._built:
            raise RuntimeError("index not built")

        q = np.ascontiguousarray(q, dtype=np.float32)
        test_file = tempfile.mktemp(suffix=".bin", prefix="raven_test_")
        output_file = tempfile.mktemp(suffix=".bin", prefix="raven_output_")
        q.tofile(test_file)

        try:
            cmd = [
                self.raven_bin,
                "--train", self._train_file,
                "--test", test_file,
                "--output", output_file,
                "--dim", str(self.dim),
                "--n", str(self.n),
                "--nq", "1",
                "--k", str(k),
                "--alpha", str(self.alpha),
                "--l-build", str(self.ef_construction),
                "--r-max", str(self.M),
                "--ef-search", str(self.ef_search),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"RAVEN query failed: {result.stderr}")

            # 读取邻居 ID（raw binary, i32），与 query_batch 一致
            neighbors = np.fromfile(output_file, dtype=np.int32).reshape(1, k)
            return neighbors[0].tolist()
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def query_batch(self, Q, k):
        """批量查询。

        Args:
            Q: numpy 数组, shape=(nq, dim)
            k: 近邻数

        Returns:
            list of list: 每个查询的邻居 ID 列表
        """
        if not self._built:
            raise RuntimeError("index not built")

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        nq, dim = Q.shape
        assert dim == self.dim

        test_file = tempfile.mktemp(suffix=".bin", prefix="raven_test_")
        output_file = tempfile.mktemp(suffix=".bin", prefix="raven_output_")
        Q.tofile(test_file)

        try:
            cmd = [
                self.raven_bin,
                "--train", self._train_file,
                "--test", test_file,
                "--output", output_file,
                "--dim", str(self.dim),
                "--n", str(self.n),
                "--nq", str(nq),
                "--k", str(k),
                "--alpha", str(self.alpha),
                "--l-build", str(self.ef_construction),
                "--r-max", str(self.M),
                "--ef-search", str(self.ef_search),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(f"RAVEN query_batch failed: {result.stderr}")

            data = json.loads(result.stdout)
            print(f"RAVEN: qps={data.get('qps', '?')}, recall@{k}={data.get('recall@k', '?')}")

            # 读取邻居 ID（raw binary, i32）
            neighbors = np.fromfile(output_file, dtype=np.int32).reshape(nq, k)
            return neighbors.tolist()
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def use_threads(self):
        """是否多线程。RAVEN 当前为单线程查询。"""
        return False

    def __str__(self):
        return self.name

    def __del__(self):
        """清理临时文件。"""
        if self._train_file and os.path.exists(self._train_file):
            os.unlink(self._train_file)
