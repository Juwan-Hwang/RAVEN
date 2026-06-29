#!/usr/bin/env python3
"""诊断 read_fvecs 的 bug"""
import numpy as np
from pathlib import Path

SIFT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sift"

# 当前 read_fvecs（有 bug）
raw = np.fromfile(str(SIFT_DIR / "sift_base.fvecs"), dtype=np.int32)
dim = raw[0]
n = raw.size // (dim + 1)
mat = raw.reshape(n, dim + 1)
data_buggy = mat[:, 1:].astype(np.float32, copy=True)

# 正确做法：view 而非 astype
data_correct = mat[:, 1:].copy().view(np.float32)

print(f"dim = {dim}")
print(f"n = {n}")
print(f"buggy[0,:5]   = {data_buggy[0,:5]}")
print(f"correct[0,:5] = {data_correct[0,:5]}")
print(f"buggy range:   [{data_buggy.min()}, {data_buggy.max()}]")
print(f"correct range: [{data_correct.min()}, {data_correct.max()}]")

# SIFT 数据应该是 0~255 的 uint8 转的 float32
