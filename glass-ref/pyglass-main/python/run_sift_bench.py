#!/usr/bin/env python3
"""
Glass HNSW SIFT1M 基准测试
输出 recall@10, QPS, avg_dist_cmps (= avg_visited), 建图时间

Bug 修复记录：
  Bug 1 (groundtruth 索引): SIFT1M groundtruth 是 0-indexed，直接用，不做任何 +1/-1。
    旧代码错误地做了 gt = gt - 1，导致 recall 暴跌到 ~2.8%。
  Bug 2 (avg_visited 计数器): 必须用 batch_search，不能用单次 search。
    C++ 层 Search() 不更新 last_search_avg_dist_cmps，只有 SearchBatch() 才更新。
    旧代码用了 search() 导致 avg_visited 恒为 806.0（不随 ef 变化）。

指标等价性：
  Glass 的 dist_cmps = 每次调用 computer(v) 时 +1（含 entry point 的 initialize_search）
  RAVEN 的 visited_count = visited.visit(v) 返回 true 的次数（含 entry point）
  两者语义完全等价：都统计「距离被计算过的唯一节点数」。

配置（与 RAVEN 可比）：
  R=32, L=200, FP32, 单线程 batch_search(num_threads=1)
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
    """读取 .fvecs 文件 -> (N, dim) float32 数组

    fvecs 格式：每条记录前 4 字节是 int32 维度，后面是 float32 数据。
    整个文件以 int32 读入后，数据部分必须用 .view(np.float32) 重解释位模式，
    不能用 .astype() —— 那会做数值转换而非位模式重解释。
    """
    a = np.fromfile(str(path), dtype=np.int32)
    d = a[0]
    assert a.size % (d + 1) == 0, f"数据大小 {a.size} 不是 {d+1} 的倍数"
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path):
    """读取 .ivecs 文件 -> (N, dim) int32 数组

    SIFT1M groundtruth 是 0-indexed，读进来直接用，不做任何 +1/-1。
    """
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = raw[0]
    assert raw.size % (dim + 1) == 0
    n = raw.size // (dim + 1)
    mat = raw.reshape(n, dim + 1)
    return mat[:, 1:].astype(np.int32, copy=True)


def validate_groundtruth(base, query, gt, n_check=5):
    """暴力验证前 n_check 个查询的 groundtruth 是否正确（0-indexed）"""
    print(f"  [BF验证] 对前 {n_check} 个查询做暴力 L2 搜索...")
    dim = base.shape[1]
    ok = 0
    for qi in range(n_check):
        diffs = np.sum((base - query[qi]) ** 2, axis=1)  # L2^2
        bf_top10 = np.argsort(diffs)[:10]
        gt_top10 = gt[qi, :10]
        match = np.intersect1d(bf_top10, gt_top10).size
        if match == 10:
            ok += 1
        else:
            print(f"    Q{qi}: BF={bf_top10[:5]}, GT={gt_top10[:5]}, match={match}/10")
    if ok == n_check:
        print(f"    全部 {n_check} 个查询 groundtruth 验证通过 (0-indexed 正确)")
    else:
        print(f"    警告: {ok}/{n_check} 通过，groundtruth 索引可能有问题!")
    return ok == n_check


def main():
    print("=" * 70)
    print("Glass HNSW SIFT1M Benchmark")
    print("  Config: R=32, L=200, FP32, single-thread")
    print("=" * 70)

    # ---- [1/5] 加载数据 ----
    print("\n[1/5] 加载 SIFT1M 数据...")
    base = read_fvecs(SIFT_DIR / "sift_base.fvecs")
    query = read_fvecs(SIFT_DIR / "sift_query.fvecs")
    gt = read_ivecs(SIFT_DIR / "sift_groundtruth.ivecs")
    # SIFT1M groundtruth 是 0-indexed，直接用，不做任何 +1/-1
    print(f"  base:   {base.shape} ({base.nbytes / 1e6:.1f} MB)")
    print(f"  query:  {query.shape}")
    print(f"  gt:     {gt.shape} (min={gt.min()}, max={gt.max()})")

    nq = query.shape[0]
    topk = 10

    # ---- [2/5] 暴力验证 groundtruth ----
    print(f"\n[2/5] 验证 groundtruth 索引...")
    validate_groundtruth(base, query, gt)

    # ---- [3/5] 建图 ----
    R = 32
    L = 200
    build_quant = "FP32"
    search_quant = "FP32"
    graph_file = str(SIFT_DIR.parent / f"glass_hnsw_R{R}_L{L}.glass")

    if os.path.exists(graph_file):
        print(f"\n[3/5] 图已存在，跳过建图: {graph_file}")
        build_time = 0.0
    else:
        print(f"\n[3/5] 构建 HNSW 索引 (R={R}, L={L}, quant={build_quant})...")
        t0 = time.time()
        glass.build_graph(
            "HNSW", base, graph_file,
            metric="L2", quant=build_quant, R=R, L=L,
        )
        build_time = time.time() - t0
        print(f"  建图耗时: {build_time:.1f}s")

    # ---- [4/5] 加载索引并优化 ----
    print(f"\n[4/5] 加载索引并优化...")
    g = glass.Graph(graph_file)
    searcher = glass.Searcher(g, base, "L2", search_quant, "")
    searcher.optimize()
    print("  优化完成")

    # ---- [5/5] 搜索测试 ----
    # 使用 batch_search(query, k, num_threads=1) 获取正确的 avg_dist_cmps
    # C++ 层 SearchBatch() 会累加 computer.dist_cmps() 并更新 last_search_avg_dist_cmps
    # 单次 Search() 不会更新此统计，旧代码的 bug 就在这里
    print(f"\n[5/5] 搜索测试 (recall@{topk}, batch_search, single-thread)")
    print(f"  {'ef':>6s}  {'recall@10':>10s}  {'QPS':>10s}  {'avg_visited':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

    results = []
    for ef in [50, 32, 48, 64, 80, 96, 128, 200]:
        searcher.set_ef(ef)

        # warmup（单线程，不污染统计数据）
        searcher.batch_search(query[:100], topk, 1)

        # 正式测量（num_threads=1 确保 OpenMP 不干扰）
        t0 = time.time()
        res, _ = searcher.batch_search(query, topk, 1)
        elapsed = time.time() - t0

        qps = nq / elapsed
        # recall@10
        recall = sum(
            np.intersect1d(res[i, :topk], gt[i, :topk]).size for i in range(nq)
        ) / nq / topk
        # avg_visited = avg_dist_cmps（仅 batch_search 更新此统计）
        avg_visited = searcher.get_last_search_dist_cmps()

        # Debug: ef=50 时打印前 3 个查询的对比
        if ef == 50:
            print(f"  [DEBUG] res[0,:5]={res[0,:5]}, gt[0,:5]={gt[0,:5]}")
            print(f"  [DEBUG] res[1,:5]={res[1,:5]}, gt[1,:5]={gt[1,:5]}")
            print(f"  [DEBUG] res.dtype={res.dtype}, res.shape={res.shape}")

        print(f"  {ef:6d}  {recall*100:9.2f}%  {qps:10.1f}  {avg_visited:12.1f}")
        results.append((ef, recall, qps, avg_visited))

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"Summary (Glass HNSW, SIFT1M, FP32, R={R}, L={L}, single-thread)")
    if build_time > 0:
        print(f"  Build time: {build_time:.1f}s")
    print(f"  Recall / QPS / avg_visited trade-off:")
    for threshold, label in [(0.95, "95%"), (0.97, "97%"), (0.99, "99%")]:
        for ef, recall, qps, avg_visited in results:
            if recall >= threshold:
                print(f"    @recall>={label}:  ef={ef:3d}  recall={recall*100:.2f}%  QPS={qps:.0f}  avg_visited={avg_visited:.1f}")
                break
    print("=" * 70)


if __name__ == "__main__":
    main()
