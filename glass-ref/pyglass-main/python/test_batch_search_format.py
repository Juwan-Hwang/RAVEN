#!/usr/bin/env python3
"""验证 batch_search 返回格式"""
import numpy as np
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glass

SIFT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "sift"

def read_fvecs(path):
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = raw[0]
    n = raw.size // (dim + 1)
    mat = raw.reshape(n, dim + 1)
    return mat[:, 1:].astype(np.float32, copy=True)

print("Loading data...")
base = read_fvecs(SIFT_DIR / "sift_base.fvecs")
query = read_fvecs(SIFT_DIR / "sift_query.fvecs")

graph_file = str(SIFT_DIR.parent / "glass_hnsw_R32_L200.glass")
if not os.path.exists(graph_file):
    print(f"ERROR: graph file not found: {graph_file}")
    print("Need to build graph first. Running build...")
    glass.build_graph("HNSW", base, graph_file, metric="L2", quant="FP32", R=32, L=200)
    print("Build done.")

g = glass.Graph(graph_file)
searcher = glass.Searcher(g, base, "L2", "FP32", "")
searcher.set_ef(50)

test_q = query[:10]
ret = searcher.batch_search(test_q, 10, 1)
print(f"type(ret) = {type(ret)}")
print(f"hasattr __len__ = {hasattr(ret, '__len__')}")
if isinstance(ret, tuple):
    print(f"len(ret) = {len(ret)}")
    for i, x in enumerate(ret):
        print(f"  ret[{i}]: type={type(x)}, dtype={getattr(x, 'dtype', '?')}, shape={getattr(x, 'shape', '?')}")
    print(f"\nret[0] (ids) first row: {ret[0][0]}")
    print(f"ret[1] (distances) first row: {ret[1][0]}")
else:
    print(f"ret.shape = {ret.shape}, ret.dtype = {ret.dtype}")
    print(f"first row: {ret[0]}")
