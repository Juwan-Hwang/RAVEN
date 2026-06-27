"""Smoke test: 验证 --save/--load 索引复用流程"""
import numpy as np
import subprocess
import os
import json

# 创建小型测试数据
np.random.seed(42)
X = np.random.rand(100, 4).astype(np.float32)
Q = np.random.rand(5, 4).astype(np.float32)

train_file = "test_train.bin"
query_file = "test_query.bin"
index_file = "test_index.idx"
output_file = "test_output.bin"

X.tofile(train_file)
Q.tofile(query_file)
print(f"Created: {X.shape} train, {Q.shape} query")

bin_path = "target/release/raven_ann_bench"

# Step 1: 构建索引并保存
print("\n=== Step 1: Build + Save ===")
cmd_build = [
    bin_path,
    "--train", train_file,
    "--save", index_file,
    "--dim", "4",
    "--n", "100",
    "--alpha", "1.0",
    "--l-build", "10",
    "--r-max", "8",
    "--ef-search", "20",
]
result = subprocess.run(cmd_build, capture_output=True, text=True, timeout=60)
print(f"exit code: {result.returncode}")
print(f"stdout: {result.stdout.strip()}")
if result.stderr:
    print(f"stderr: {result.stderr.strip()[-200:]}")
assert result.returncode == 0, "Build failed"
assert os.path.exists(index_file), "Index file not created"
print(f"Index file size: {os.path.getsize(index_file)} bytes")

# Step 2: 用 --load 查询
print("\n=== Step 2: Load + Query ===")
cmd_query = [
    bin_path,
    "--load", index_file,
    "--train", train_file,
    "--test", query_file,
    "--output", output_file,
    "--dim", "4",
    "--n", "100",
    "--nq", "5",
    "--k", "10",
    "--ef-search", "20",
]
result = subprocess.run(cmd_query, capture_output=True, text=True, timeout=60)
print(f"exit code: {result.returncode}")
print(f"stdout: {result.stdout.strip()}")
if result.stderr:
    print(f"stderr: {result.stderr.strip()[-200:]}")
assert result.returncode == 0, "Query failed"

# 读取结果
neighbors = np.fromfile(output_file, dtype=np.int32).reshape(5, 10)
print(f"\nQuery results shape: {neighbors.shape}")
print(f"First query neighbors: {neighbors[0]}")

# 清理
for f in [train_file, query_file, index_file, output_file]:
    if os.path.exists(f):
        os.unlink(f)

print("\n=== SMOKE TEST PASSED ===")
