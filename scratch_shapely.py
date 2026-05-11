import time
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

polys = []
for i in range(1000):
    p = Polygon([(i, 0), (i+1, 0), (i, 1)])
    polys.append(p)

t0 = time.time()
tree = STRtree(polys)
t1 = time.time()
print(f"STRtree build: {t1-t0:.4f}s")

test_poly = Polygon([(500.5, 0.5), (501.5, 0.5), (500.5, 1.5)])
t0 = time.time()
hits = tree.query(test_poly)
overlap = False
for idx in hits:
    if polys[idx].intersection(test_poly).area > 1e-5:
        overlap = True
        break
t1 = time.time()
print(f"STRtree query: {t1-t0:.4f}s, overlap: {overlap}")
