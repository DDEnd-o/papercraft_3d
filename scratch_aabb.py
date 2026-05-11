import time
import numpy as np
from shapely.geometry import Polygon

polys = []
for i in range(1000):
    p = Polygon([(i, 0), (i+1, 0), (i, 1)])
    polys.append(p)

def check_overlap(new_poly, existing_polys):
    minx1, miny1, maxx1, maxy1 = new_poly.bounds
    for p in existing_polys:
        minx2, miny2, maxx2, maxy2 = p.bounds
        if not (maxx1 <= minx2 or minx1 >= maxx2 or maxy1 <= miny2 or miny1 >= maxy2):
            if new_poly.intersection(p).area > 1e-5:
                return True
    return False

test_poly = Polygon([(500.5, 0.5), (501.5, 0.5), (500.5, 1.5)])
t0 = time.time()
overlap = check_overlap(test_poly, polys)
t1 = time.time()
print(f"AABB + query: {t1-t0:.4f}s, overlap: {overlap}")
