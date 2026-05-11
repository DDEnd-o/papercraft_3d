import numpy as np
from shapely.geometry import Polygon

p2d_a = np.array([0.0, 0.0])
p2d_b = np.array([10.0, 0.0])

ordered1 = np.array([
    p2d_a,
    p2d_b,
    [5.0, 10.0]
])

ordered2 = np.array([
    p2d_a,
    p2d_b,
    [5.0, 10.0]
])

poly1 = Polygon(ordered1)
poly2 = Polygon(ordered2)

print("poly1 valid:", poly1.is_valid)
print("poly2 valid:", poly2.is_valid)

area = poly1.intersection(poly2).area
print("Intersection area:", area)
print("Poly1 area:", poly1.area)

overlap = False
if area > 1e-5 and (area > poly1.area * 0.01 or area > poly2.area * 0.01):
    overlap = True

print("Overlap detected:", overlap)
