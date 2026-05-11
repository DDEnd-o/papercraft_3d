from shapely.geometry import Polygon
import numpy as np

# Simulate two adjacent triangles with slight floating point error
v0 = np.array([0.0, 0.0])
v1 = np.array([10.0, 0.0])
v2 = np.array([5.0, 10.0])
v3 = np.array([5.0, -10.0])

# Add tiny error to v0, v1 for the second triangle
v0_err = v0 + np.array([1e-6, 1e-6])
v1_err = v1 + np.array([-1e-6, 1e-6])

p1 = Polygon([v0, v1, v2])
p2 = Polygon([v0_err, v1_err, v3])

print("Intersection area:", p1.intersection(p2).area)
