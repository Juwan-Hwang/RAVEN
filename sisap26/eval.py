"""Recall@30 of every result file against the gold standard in the dataset file."""
import glob, sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r"); grp = "otest" if "otest" in f else "test"
G = np.asarray(f[grp]["knns"][:]); nq = f[grp]["queries"].shape
G = G if G.shape[0] in nq[:1] and G.shape[0] != 30 else G.T
for fn in sorted(glob.glob(sys.argv[2] + "/*.h5")):
    r = h5py.File(fn, "r"); I = r["knns"][:]
    rec = np.mean([len(set(I[i, :30]) & set(G[i, :30])) / 30 for i in range(len(I))])
    print(f"{r.attrs['params']},recall={rec:.4f},qps={len(I)/r.attrs['querytime']:.0f}")
