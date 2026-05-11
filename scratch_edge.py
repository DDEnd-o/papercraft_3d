import sys
sys.path.append('.')
from modules.unfolder import unfold_mesh
from modules.mesh_loader import load_primitive
import numpy as np

mesh_r = load_primitive('cube')
mesh = mesh_r.mesh
unfold_r = unfold_mesh(mesh)
panels = unfold_r.panels

print(f"Total faces: {len(mesh.faces)}")
print(f"Total panels: {len(panels)}")

# Group by island
islands = {}
for p in panels:
    islands.setdefault(p.group_id, []).append(p)
print(f"Number of islands: {len(islands)}")

# Build edge to faces map
# mesh.face_adjacency gives pairs of face indices
# mesh.face_adjacency_edges gives the corresponding shared edge vertices
adj_pairs = mesh.face_adjacency
adj_edges = mesh.face_adjacency_edges

edge_to_faces = {}
for i, (f0, f1) in enumerate(adj_pairs):
    v0, v1 = adj_edges[i]
    edge_key = tuple(sorted((int(v0), int(v1))))
    edge_to_faces[edge_key] = (int(f0), int(f1))

print(f"Total shared edges in 3D: {len(edge_to_faces)}")

panel_dict = {p.face_idx: p for p in panels}

folds = 0
cuts = 0
for edge_key, (f0, f1) in edge_to_faces.items():
    if f0 in panel_dict and f1 in panel_dict:
        p0 = panel_dict[f0]
        p1 = panel_dict[f1]
        
        # Check if they share the 2D edge
        # Find which index v0, v1 correspond to in f0 and f1
        v0, v1 = edge_key
        i0_v0 = list(mesh.faces[f0]).index(v0)
        i0_v1 = list(mesh.faces[f0]).index(v1)
        i1_v0 = list(mesh.faces[f1]).index(v0)
        i1_v1 = list(mesh.faces[f1]).index(v1)
        
        # 2D coords
        p0_v0 = p0.verts_2d[i0_v0]
        p0_v1 = p0.verts_2d[i0_v1]
        p1_v0 = p1.verts_2d[i1_v0]
        p1_v1 = p1.verts_2d[i1_v1]
        
        dist0 = np.linalg.norm(p0_v0 - p1_v0)
        dist1 = np.linalg.norm(p0_v1 - p1_v1)
        
        if dist0 < 1e-4 and dist1 < 1e-4:
            folds += 1
        else:
            cuts += 1
print(f"Folds: {folds}, Cuts: {cuts}")
