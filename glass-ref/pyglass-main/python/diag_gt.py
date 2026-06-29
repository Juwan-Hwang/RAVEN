#!/usr/bin/env python3
"""诊断 groundtruth 与 base 数据是否匹配"""
import numpy as np
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glass

SIFT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sift"

def read_fvecs(path):
    a = np.fromfile(str(path), dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)

def read_ivecs(path):
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = raw[0]
    n = raw.size // (dim + 1)
    mat = raw.reshape(n, dim + 1)
    return mat[:, 1:].astype(np.int32, copy=True)

base = read_fvecs(SIFT_DIR / "sift_base.fvecs")
query = read_fvecs(SIFT_DIR / "sift_query.fvecs")
gt = read_ivecs(SIFT_DIR / "sift_groundtruth.ivecs")

print(f"base shape: {base.shape}")
print(f"gt shape: {gt.shape}")
print(f"gt min/max: {gt.min()}, {gt.max()}")

# Q0 暴力最近邻
diffs = np.sum((base - query[0]) ** 2, axis=1)
bf_nn = np.argsort(diffs)[0]
print(f"Q0 BF nearest: {bf_nn}")
print(f"Q0 GT nearest: {gt[0, 0]}")

# 额外：打印前5个BF和GT的对比
bf_top5 = np.argsort(diffs)[:5]
print(f"Q0 BF top5: {bf_top5}")
print(f"Q0 GT top5: {gt[0, :5]}")
print(f"Q0 BF dists: {diffs[bf_top5]}")
