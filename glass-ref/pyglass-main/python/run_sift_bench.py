#!/usr/bin/env python3
"""
Glass HNSW SIFT1M 基准测试
输出 recall@10, QPS, avg_dist_cmps (≈ avg_visited), 建图时间

关键修正：
1. SIFT groundtruth 是 1-indexed，需减 1 转为 0-indexed
2. 使用 batch_search 获取正确的 avg_dist_cmps（单次 search 不更新此统计）
3. 使用 batch_search 的多线程搜索，更贴近真实场景
"""
import numpy as np
import os
import sys
import time
from pathlib import Path

# 确保 glass 模块可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glass

SIFT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sift"


def read_fvecs(path):
    """读取 .fvecs 文件 → (N, dim) float32 数组"""
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = raw[0]
    assert raw.size % (dim + 1) == 0, f"数据大小 {raw.size} 不是 {dim+1} 的倍数"
    n = raw.size // (dim + 1)
    mat = raw.reshape(n, dim + 1)
    return mat[:, 1:].astype(np.float32, copy=True)


def read_ivecs(path):
    """读取 .ivecs 文件 → (N, dim) int32 数组"""
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = raw[0]
    assert raw.size % (dim + 1) == 0
    n = raw.size // (dim + 1)
    mat = raw.reshape(n, dim + 1)
    return mat[:, 1:].astype(np.int32, copy=True)


def main():
    print("=" * 70)
    print("Glass HNSW SIFT1M Benchmark")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] 加载 SIFT1M 数据...")
    base = read_fvecs(SIFT_DIR / "sift_base.fvecs")
    query = read_fvecs(SIFT_DIR / "sift_query.fvecs")
    gt = read_ivecs(SIFT_DIR / "sift_groundtruth.ivecs")
    # SIFT groundtruth 已是 0-indexed（BF 验证确认）
    print(f"  base:   {base.shape} ({base.nbytes / 1e6:.1f} MB)")
    print(f"  query:  {query.shape}")
    print(f"  gt:     {gt.shape} (min={gt.min()}, max={gt.max()})")

    nq = query.shape[0]
    topk = 10

    # 建图
    R = 32
    L = 200
    build_quant = "FP32"
    search_quant = "FP32"
    graph_file = str(SIFT_DIR.parent / f"glass_hnsw_R{R}_L{L}.glass")

    # 如果图已存在则跳过建图
    if os.path.exists(graph_file):
        print(f"\n[2/4] 图已存在，跳过建图: {graph_file}")
        build_time = 0.0
    else:
        print(f"\n[2/4] 构建 HNSW 索引 (R={R}, L={L}, quant={build_quant})...")
        t0 = time.time()
        glass.build_graph(
            "HNSW", base, graph_file,
            metric="L2", quant=build_quant, R=R, L=L,
        )
        build_time = time.time() - t0
        print(f"  建图耗时: {build_time:.1f}s")

    # 加载图并创建 searcher
    print(f"\n[3/4] 加载索引并优化...")
    g = glass.Graph(graph_file)
    searcher = glass.Searcher(g, base, "L2", search_quant, "")
    searcher.optimize()
    print("  优化完成")

    # 搜索测试 — 使用 batch_search 获取正确的 avg_dist_cmps
    print(f"\n[4/4] 搜索测试 (recall@{topk}, batch_search)...")
    print(f"  {'ef':>6s}  {'recall@10':>10s}  {'QPS':>10s}  {'avg_visited':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

    results = []
    for ef in [16, 24, 32, 48, 64, 80, 96, 128, 200]:
        searcher.set_ef(ef)

        # warmup
        searcher.batch_search(query[:100], topk, 1)

        # 正式测量（单线程 batch_search 避免 OpenMP pool 竞争）
        t0 = time.time()
        res, _ = searcher.batch_search(query, topk, 1)
        elapsed = time.time() - t0

        qps = nq / elapsed
        # recall@10
        recall = sum(
            np.intersect1d(res[i, :topk], gt[i, :topk]).size for i in range(nq)
        ) / nq / topk
        # avg_visited = 平均距离计算次数（仅 batch_search 更新此统计）
        avg_visited = searcher.get_last_search_dist_cmps()

        # Debug: 第一个 ef 值时打印前 3 个查询的对比
        if ef == 16:
            print(f"  [DEBUG] res[0,:5]={res[0,:5]}, gt[0,:5]={gt[0,:5]}")
            print(f"  [DEBUG] res[1,:5]={res[1,:5]}, gt[1,:5]={gt[1,:5]}")
            print(f"  [DEBUG] res.dtype={res.dtype}, res.shape={res.shape}")

        print(f"  {ef:6d}  {recall*100:9.2f}%  {qps:10.1f}  {avg_visited:12.1f}")
        results.append((ef, recall, qps, avg_visited))

    print("\n" + "=" * 70)
    print("Summary (Glass HNSW, SIFT1M, FP32, R=32, L=200)")
    if build_time > 0:
        print(f"  Build time: {build_time:.1f}s")
    print(f"  Recall / QPS / avg_visited trade-off:")
    for ef, recall, qps, avg_visited in results:
        if recall >= 0.95:
            print(f"    @recall≥95%:  ef={ef:3d}  recall={recall*100:.2f}%  QPS={qps:.0f}  avg_visited={avg_visited:.1f}")
            break
    for ef, recall, qps, avg_visited in results:
        if recall >= 0.97:
            print(f"    @recall≥97%:  ef={ef:3d}  recall={recall*100:.2f}%  QPS={qps:.0f}  avg_visited={avg_visited:.1f}")
            break
    for ef, recall, qps, avg_visited in results:
        if recall >= 0.99:
            print(f"    @recall≥99%:  ef={ef:3d}  recall={recall*100:.2f}%  QPS={qps:.0f}  avg_visited={avg_visited:.1f}")
            break
    print("=" * 70)


if __name__ == "__main__":
    main()
