import trimesh
import numpy as np

mesh = trimesh.load("D:\MyProject\PapercraftApp\image_demo\Lego compatible bricks bundle (3D printing optimized) - 7344067 - part 1 of 3\\files\\86996_1x1x.6.STL")
print("Bbox:", mesh.bounds)
sizes = mesh.bounds[1] - mesh.bounds[0]
print("Sizes:", sizes)

areas = mesh.area_faces
print("Min face area:", np.min(areas))
print("Max face area:", np.max(areas))
print("Mean face area:", np.mean(areas))
