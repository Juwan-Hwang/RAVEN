import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sift_bench import read_fvecs, read_ivecs, SIFT_DIR

base = read_fvecs(SIFT_DIR / "sift_base.fvecs")
query = read_fvecs(SIFT_DIR / "sift_query.fvecs")
gt = read_ivecs(SIFT_DIR / "sift_groundtruth.ivecs") - 1

print("base[0,:5]:", base[0,:5])
print("query[0,:5]:", query[0,:5])
print("gt[0,:5]:", gt[0,:5])

# brute-force top-10 for query 0
dists = np.sum((base - query[0])**2, axis=1)
bf_top10 = np.argsort(dists)[:10]
print("BF top10:", bf_top10)
print("BF dists:", dists[bf_top10])
print("GT dists:", dists[gt[0,:10]])

# Now test glass search
import glass
g = glass.Graph(str(SIFT_DIR.parent / "glass_hnsw_R32_L200.glass"))
s = glass.Searcher(g, base, "L2", "FP32", "")
s.set_ef(200)
ids, dis = s.batch_search(query[:5], 10)
print("Glass top10 for q0:", ids[0])
print("Glass dists for q0:", dis[0])
print("Match BF:", np.array_equal(ids[0], bf_top10))
