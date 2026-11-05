#!/usr/bin/env python3
"""SISAP 2026 Indexing Challenge, Task 2 (MIPS, k=30) with RAVEN.

Maximum inner product search is reduced to Euclidean search by the standard
one-dimensional lift x -> [x, sqrt(M - |x|^2)], q -> [q, 0]. RAVEN builds a
Vamana-style graph over the lifted vectors, traverses it with an 8-bit
scalar-quantised asymmetric distance (float query against per-dimension
scaled u8 codes) and reranks the final pool with exact inner products.

Query-distribution balancing (on by default, RAVEN_BALANCE=0 disables): the
llama queries are far out of the base distribution (on the base's weakest
principal directions their variance is >100x the base's), which forces a
graph search to explore deep. Before lifting we apply an inner-product
preserving linear map x -> A x, q -> A^{-T} q with A = diag(g) V^T, where V
is the PCA basis of the base, s its variances, t = diag(V^T C V) the queries'
second moments along V, and g = (t / s)^p, p = 0.25. C is the 128x128 query
second-moment matrix of the public llama-dev queries, shipped with this
script (llama_dev_query_moment.npy); no evaluation query is seen at build
time. q.x is unchanged exactly, so results are exact-MIPS reranked as before.

usage: search.py --input data/<ds>/<file>.h5 --output results/<ds>/ [--task task2]
"""
import argparse, glob, json, os, time

# BLAS threads spinning inside the timed region compete with the search threads.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("RAVEN_SQ8_ADC", "1")

import h5py, numpy as np  # noqa: E402
import raven  # noqa: E402

R = int(os.environ.get("RAVEN_R", 48))
L = int(os.environ.get("RAVEN_L", 200))
RR = int(os.environ.get("RAVEN_RR", 4))
BALANCE = os.environ.get("RAVEN_BALANCE", "1") != "0"
P = float(os.environ.get("RAVEN_BALANCE_P", 0.25))
EFS_BAL = "50,60,70,85,100,115,130,145,155,160,165,170,180,190,210"
EFS_RAW = "100,130,160,180,200,210,220,230,240,250,260,280,300,340,400"
EFS = [int(e) for e in os.environ.get("RAVEN_EFS", EFS_BAL if BALANCE else EFS_RAW).split(",")][:15]
PRIOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_dev_query_moment.npy")
K = 30


def load(path, cfg):
    f = h5py.File(path, "r")
    X = np.asarray(f[cfg.get("data", "train")][()], dtype=np.float32)
    qpath = cfg.get("queries") or ("otest/queries" if "otest" in f else "test/queries")
    Q = np.asarray(f[qpath][()], dtype=np.float32)
    d = Q.shape[1] if Q.shape[0] != X.shape[1] else Q.shape[0]
    if X.shape[0] == d and X.shape[1] != d:
        X = X.T
    if Q.shape[0] == d and Q.shape[1] != d:
        Q = Q.T
    return np.ascontiguousarray(X), np.ascontiguousarray(Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--task-description", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.task_description)) if a.task_description and os.path.exists(a.task_description) else {}
    task = a.task or cfg.get("task", "task2")
    if task != "task2":
        print(f"RAVEN only implements task2, skipping {task}")
        return
    inp = a.input if os.path.isfile(a.input) else sorted(glob.glob(a.input) or glob.glob(os.path.join(a.input, "*.h5")))[0]
    threads = int(os.environ.get("RAVEN_THREADS", os.cpu_count() or 8))
    X, Q = load(inp, cfg)
    n, d = X.shape

    t0 = time.time()
    Wq = None
    Xt = X
    if BALANCE and d == 128 and os.path.exists(PRIOR):
        X64 = X.astype(np.float64)
        Xc = X64[::2] - X64.mean(0)
        sv, V = np.linalg.eigh(Xc.T @ Xc / (len(Xc) - 1))
        sv = np.maximum(sv, 1e-12)
        C = np.load(PRIOR)
        tq = np.maximum(np.einsum("ij,ik,kj->j", V, C, V), 1e-12)
        g = (tq / sv) ** P
        Xt = ((X64 @ V) * g).astype(np.float32)
        Wq = np.ascontiguousarray((V / g).astype(np.float32))
    n2 = (Xt.astype(np.float64) ** 2).sum(1)
    Xa = np.hstack([Xt, np.sqrt(np.maximum(n2.max() - n2, 0))[:, None]]).astype(np.float32)
    del Xt
    idx = raven.Index("euclidean", d + 1, r=R, l=L, quantization="sq8",
                      rerank_factor=RR, threads=threads)
    idx.build(Xa)
    s = idx.searcher()
    buildtime = time.time() - t0
    print(f"n={n} d={d} nq={len(Q)} threads={threads} balance={Wq is not None} build {buildtime:.1f}s", flush=True)

    os.makedirs(a.output, exist_ok=True)
    for ef in EFS:
        s.set_ef(ef)
        t = time.time()
        Qa = np.zeros((len(Q), d + 1), np.float32)
        if Wq is None:
            Qa[:, :d] = Q
        else:
            np.matmul(Q, Wq, out=Qa[:, :d])
        I = np.asarray(s.batch_search(np.ascontiguousarray(Qa), K), dtype=np.int64).reshape(len(Q), K)
        D = -np.einsum("qkd,qd->qk", X[I], Q)
        querytime = time.time() - t
        params = f"r={R},l={L},rr={RR},ef={ef},sq8adc" + (f",balance_p={P}" if Wq is not None else "")
        fn = os.path.join(a.output, f"raven_r{R}_l{L}_ef{ef}.h5")
        with h5py.File(fn, "w") as f:
            f.attrs.update(algo="raven", task=task, dataset=cfg.get("dataset_name", ""), buildtime=buildtime,
                           querytime=querytime, params=params)
            f["knns"] = (I + 1).astype(np.int32)
            f["dists"] = D.astype(np.float32)
        print(f"ef={ef} querytime={querytime:.4f}s qps={len(Q)/querytime:.0f}", flush=True)


if __name__ == "__main__":
    main()
