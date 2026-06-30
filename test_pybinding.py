"""端到端测试：RAVEN PyO3 绑定 → build → search → 验证 recall"""
import numpy as np
import raven
import time

# 生成小型测试数据
np.random.seed(42)
N = 1000
DIM = 64
K = 10

data = np.random.randn(N, DIM).astype(np.float32)
queries = np.random.randn(10, DIM).astype(np.float32)

# 暴力搜索 ground truth
def brute_force(data, query, k):
    dists = np.sum((data - query) ** 2, axis=1)
    return np.argsort(dists)[:k]

print("=== RAVEN PyO3 端到端测试 ===")
print(f"N={N}, DIM={DIM}, K={K}")

# 1. 构建索引
t0 = time.time()
index = raven.Index("L2", DIM, r=16, l=100, alpha=1.2, nav_m=16, directional=True)
index.build(data)
t1 = time.time()
print(f"Build: {t1-t0:.2f}s")

# 2. 创建搜索器
searcher = index.searcher()

# 3. 单查询测试
searcher.set_ef(50)
total_recall = 0
for i in range(10):
    result = searcher.search(queries[i], K)
    gt = brute_force(data, queries[i], K)
    recall = len(set(result.tolist()) & set(gt.tolist())) / K
    total_recall += recall
    print(f"  Query {i}: recall={recall:.2f}, ids={result[:5]}...")

print(f"\nAvg recall: {total_recall/10:.4f}")

# 4. 批量搜索测试
batch_results = searcher.batch_search(queries, K).reshape(-1, K)
print(f"\nBatch search shape: {batch_results.shape}")
batch_recall = 0
for i in range(10):
    gt = brute_force(data, queries[i], K)
    batch_recall += len(set(batch_results[i].tolist()) & set(gt.tolist())) / K
print(f"Batch avg recall: {batch_recall/10:.4f}")

# 5. ef 参数切换测试
for ef in [20, 50, 100]:
    searcher.set_ef(ef)
    t0 = time.time()
    for _ in range(10000):
        searcher.search(queries[0], K)
    dt = time.time() - t0
    print(f"  ef={ef}: {10000/dt:.0f} QPS")

print("\n=== 测试通过 ===")
