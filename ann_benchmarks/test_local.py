#!/usr/bin/env python3
"""RAVEN ann-benchmarks 本地端到端测试.

生成小规模合成数据集，调用 raven_ann_bench 二进制，
验证 build + query + recall 计算的完整流程。

用法：
  python3 test_local.py
"""

import os
import sys
import json
import tempfile
import subprocess
import numpy as np


def generate_dataset(n=1000, dim=128, nq=50, k=10, seed=42):
    """生成合成数据集和 ground truth."""
    rng = np.random.default_rng(seed)
    train = rng.random((n, dim), dtype=np.float32)
    test = rng.random((nq, dim), dtype=np.float32)

    # 暴力计算 ground truth（L2 距离，不依赖 scikit-learn）
    gt = np.zeros((nq, k), dtype=np.int32)
    for i in range(nq):
        dists = np.sum((train - test[i]) ** 2, axis=1)
        gt[i] = np.argsort(dists)[:k]
    return train, test, gt


def run_raven(raven_bin, train_file, test_file, neighbors_file, output_file,
              n, dim, nq, k, alpha, l_build, r_max, ef_search):
    """调用 raven_ann_bench 二进制."""
    cmd = [
        raven_bin,
        "--train", train_file,
        "--test", test_file,
        "--neighbors", neighbors_file,
        "--output", output_file,
        "--dim", str(dim),
        "--n", str(n),
        "--nq", str(nq),
        "--k", str(k),
        "--alpha", str(alpha),
        "--l-build", str(l_build),
        "--r-max", str(r_max),
        "--ef-search", str(ef_search),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"raven_ann_bench failed with code {result.returncode}")
    return json.loads(result.stdout), result.stderr


def main():
    raven_bin = os.environ.get("RAVEN_BIN", "./target/release/raven_ann_bench")
    if not os.path.exists(raven_bin):
        print(f"error: raven_ann_bench not found at {raven_bin}")
        sys.exit(1)

    print("=== RAVEN ann-benchmarks 本地端到端测试 ===")
    print()

    # 生成数据集
    print("生成合成数据集...")
    n, dim, nq, k = 1000, 128, 50, 10
    train, test, gt = generate_dataset(n=n, dim=dim, nq=nq, k=k)
    print(f"  train: {train.shape}, test: {test.shape}, gt: {gt.shape}")

    # 写入临时文件
    tmpdir = tempfile.mkdtemp(prefix="raven_test_")
    train_file = os.path.join(tmpdir, "train.bin")
    test_file = os.path.join(tmpdir, "test.bin")
    neighbors_file = os.path.join(tmpdir, "neighbors.bin")
    output_file = os.path.join(tmpdir, "output.bin")

    train.tofile(train_file)
    test.tofile(test_file)
    gt.tofile(neighbors_file)

    # 参数扫描
    configs = [
        {"alpha": 1.0, "l_build": 100, "r_max": 32, "ef_search": 50},
        {"alpha": 1.2, "l_build": 200, "r_max": 64, "ef_search": 100},
        {"alpha": 1.5, "l_build": 200, "r_max": 64, "ef_search": 200},
    ]

    print()
    print(f"{'config':<40} {'build_s':>8} {'query_s':>8} {'QPS':>8} {'recall@10':>10}")
    print("-" * 78)

    results = []
    for cfg in configs:
        stats, stderr = run_raven(
            raven_bin, train_file, test_file, neighbors_file, output_file,
            n, dim, nq, k,
            alpha=cfg["alpha"],
            l_build=cfg["l_build"],
            r_max=cfg["r_max"],
            ef_search=cfg["ef_search"],
        )

        # 读取 RAVEN 输出的邻居 ID
        raven_neighbors = np.fromfile(output_file, dtype=np.int32).reshape(nq, k)

        # 独立计算 recall
        hits = 0
        for q in range(nq):
            gt_set = set(gt[q].tolist())
            found_set = set(raven_neighbors[q].tolist())
            hits += len(gt_set & found_set)
        recall = hits / (nq * k)

        label = f"a={cfg['alpha']},lc={cfg['l_build']},r={cfg['r_max']},ef={cfg['ef_search']}"
        print(f"{label:<40} {stats['build_time_s']:>8.3f} {stats['query_time_s']:>8.3f} "
              f"{stats['qps']:>8.0f} {recall:>10.4f}")

        results.append({
            "config": cfg,
            "stats": stats,
            "recall": recall,
        })

    print()
    print("=== Pareto 曲线（recall vs QPS）===")
    results.sort(key=lambda r: r["recall"])
    for r in results:
        cfg = r["config"]
        print(f"  recall={r['recall']:.4f}  QPS={r['stats']['qps']:.0f}  "
              f"(a={cfg['alpha']}, ef={cfg['ef_search']})")

    # 清理
    import shutil
    shutil.rmtree(tmpdir)
    print()
    print("测试完成。")


if __name__ == "__main__":
    main()
