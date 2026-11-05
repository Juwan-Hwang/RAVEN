"""SISAP 2026 Indexing Challenge, Task 2 (MIPS, k=30) with RAVEN.

Maximum inner product search is reduced to Euclidean search by the standard
one-dimensional lift x -> [x, sqrt(M - |x|^2)], q -> [q, 0]. RAVEN builds a
Vamana-style graph over the lifted vectors, traverses it with an 8-bit
scalar-quantised asymmetric distance (float query against per-dimension
scaled u8 codes) and reranks the final pool with exact inner products.

usage: search.py --input data/<ds>/<file>.h5 --output results/<ds>/ [--task task2]
"""
import argparse, os, time
import h5py, numpy as np

os.environ.setdefault("RAVEN_SQ8_ADC", "1")
import raven  # noqa: E402

R = int(os.environ.get("RAVEN_R", 48))
L = int(os.environ.get("RAVEN_L", 200))
RR = int(os.environ.get("RAVEN_RR", 4))
EFS = [int(e) for e in os.environ.get(
    "RAVEN_EFS", "160,180,190,200,210,220,230,240,250,265,280,300,330,360,400").split(",")][:15]
K = 30


def load(path):
    f = h5py.File(path, "r")
    X = np.asarray(f["train"][:], dtype=np.float32)
    grp = "otest" if "otest" in f else "test"
    Q = np.asarray(f[grp]["queries"][:], dtype=np.float32)
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
    ap.add_argument("--task", default="task2")
    ap.add_argument("--task-description", default=None)
    a = ap.parse_args()
    threads = int(os.environ.get("RAVEN_THREADS", os.cpu_count() or 8))
    X, Q = load(a.input)
    n, d = X.shape

    t0 = time.time()
    n2 = (X.astype(np.float64) ** 2).sum(1)
    Xa = np.hstack([X, np.sqrt(np.maximum(n2.max() - n2, 0))[:, None]]).astype(np.float32)
    idx = raven.Index("euclidean", d + 1, r=R, l=L, quantization="sq8",
                      rerank_factor=RR, threads=threads)
    idx.build(Xa)
    s = idx.searcher()
    buildtime = time.time() - t0
    print(f"n={n} d={d} nq={len(Q)} threads={threads} build {buildtime:.1f}s", flush=True)

    os.makedirs(a.output, exist_ok=True)
    for ef in EFS:
        s.set_ef(ef)
        t = time.time()
        Qa = np.hstack([Q, np.zeros((len(Q), 1), np.float32)])
        I = np.asarray(s.batch_search(np.ascontiguousarray(Qa), K), dtype=np.int64).reshape(len(Q), K)
        D = -np.einsum("qkd,qd->qk", X[I], Q)
        querytime = time.time() - t
        params = f"r={R},l={L},rr={RR},ef={ef},sq8adc"
        fn = os.path.join(a.output, f"raven_r{R}_l{L}_ef{ef}.h5")
        with h5py.File(fn, "w") as f:
            f.attrs.update(algo="raven", task=a.task, buildtime=buildtime,
                           querytime=querytime, params=params)
            f["knns"] = (I + 1).astype(np.int32)
            f["dists"] = D.astype(np.float32)
        print(f"ef={ef} querytime={querytime:.4f}s qps={len(Q)/querytime:.0f}", flush=True)


if __name__ == "__main__":
    main()
