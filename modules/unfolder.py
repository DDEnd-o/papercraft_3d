"""
modules/unfolder.py
Thuật toán trải phẳng mesh 3D → các panel 2D dùng Spanning Tree BFS.

Cách hoạt động:
1. Xây dựng spanning tree từ đồ thị liền kề mặt
2. BFS từ face gốc, lần lượt xoay từng mặt ra mặt phẳng 2D
3. Mỗi face con được xoay theo shared edge với face cha
4. Kết quả: list Panel2D chứa tọa độ 2D của từng tam giác
"""
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import trimesh
from shapely.geometry import Polygon


@dataclass
class Panel2D:
    """Một mảnh papercraft — tam giác được trải phẳng ra 2D."""
    face_idx:   int                          # index trong mesh gốc
    verts_2d:   np.ndarray                   # shape (3,2) tọa độ 2D
    color:      tuple = (220, 200, 160)      # màu fill (R,G,B)
    group_id:   int   = 0                    # nhóm liên kết (island)
    neighbors:  list  = field(default_factory=list)  # [(face_idx, edge_verts)]
    fold_edges: set   = field(default_factory=set)   # local edge indices (0,1,2) that are folded

    @property
    def centroid(self):
        return self.verts_2d.mean(axis=0)

    @property
    def bbox(self):
        mn = self.verts_2d.min(axis=0)
        mx = self.verts_2d.max(axis=0)
        return mn, mx

    @property
    def area(self):
        v = self.verts_2d
        return abs((v[1,0]-v[0,0])*(v[2,1]-v[0,1]) -
                   (v[2,0]-v[0,0])*(v[1,1]-v[0,1])) * 0.5


@dataclass
class UnfoldResult:
    panels:     list          # list[Panel2D]
    success:    bool = True
    error:      str  = ""
    num_islands: int = 0      # số vùng rời nhau

    def summary(self):
        return (f"  Panels    : {len(self.panels)}\n"
                f"  Islands   : {self.num_islands}\n"
                f"  Success   : {self.success}")


def _get_edge_index(face_verts, v0, v1):
    for i in range(3):
        a = face_verts[i]
        b = face_verts[(i+1)%3]
        if (a == v0 and b == v1) or (a == v1 and b == v0):
            return i
    return -1

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tri_to_2d(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Chiếu tam giác 3D xuống hệ tọa độ 2D cục bộ."""
    e1 = v1 - v0
    e2 = v2 - v0
    len_e1 = np.linalg.norm(e1)
    if len_e1 < 1e-10:
        return np.zeros((3, 2))
    x_axis = e1 / len_e1
    normal = np.cross(e1, e2)
    y_axis = np.cross(normal, e1)
    len_y = np.linalg.norm(y_axis)
    if len_y < 1e-10:
        return np.zeros((3, 2))
    y_axis /= len_y
    return np.array([
        [0.0,                   0.0],
        [np.dot(e1, x_axis),    0.0],
        [np.dot(e2, x_axis),    np.dot(e2, y_axis)],
    ])


def _build_transform(src_a: np.ndarray, src_b: np.ndarray,
                     dst_a: np.ndarray, dst_b: np.ndarray) -> np.ndarray:
    """
    Tính ma trận affine 3×3 (rotation+translation trong 2D) sao cho:
      src_a → dst_a,  src_b → dst_b
    """
    # Vector hướng
    s = src_b - src_a
    d = dst_b - dst_a
    ls = np.linalg.norm(s)
    ld = np.linalg.norm(d)
    if ls < 1e-10 or ld < 1e-10:
        return np.eye(2), np.zeros(2)

    # Góc xoay
    angle_s = np.arctan2(s[1], s[0])
    angle_d = np.arctan2(d[1], d[0])
    angle   = angle_d - angle_s

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # Translation: sau khi xoay, src_a → dst_a
    t = dst_a - R @ src_a
    return R, t


def _apply_transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T + t


# ── Main unfolder ─────────────────────────────────────────────────────────────

def unfold_mesh(mesh: trimesh.Trimesh,
                max_faces: int = 800,
                on_progress=None) -> UnfoldResult:
    """
    Trải phẳng mesh 3D thành các panel 2D bằng BFS spanning tree.

    Args:
        mesh        : trimesh.Trimesh đã được simplify
        max_faces   : giới hạn số mặt (tránh quá chậm)
        on_progress : callback(msg, pct)

    Returns:
        UnfoldResult chứa list Panel2D
    """
    def progress(msg, pct=0):
        print(f"  [{pct:3d}%] {msg}")
        if on_progress:
            on_progress(msg, pct)

    result = UnfoldResult(panels=[], success=False)

    try:
        # 0. CHUẨN HOÁ SCALE: Đưa về kích thước tiêu chuẩn để các ngưỡng 1e-5 hoạt động ổn định
        v_min = mesh.vertices.min(axis=0)
        v_max = mesh.vertices.max(axis=0)
        scale_factor = 100.0 / max((v_max - v_min).max(), 1e-6)
        mesh.vertices = (mesh.vertices - v_min) * scale_factor

        # ── Simplify nếu quá nhiều mặt ────────────────────────────────────
        if len(mesh.faces) > max_faces:
            progress(f"Simplify {len(mesh.faces)} → {max_faces} mặt...", 5)
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
            mesh.remove_unreferenced_vertices()

        n_faces = len(mesh.faces)
        progress(f"Unfold {n_faces} mặt...", 10)

        # ── Xây adjacency list ────────────────────────────────────────────
        adj       = mesh.face_adjacency          # (E,2) cặp face liền kề
        adj_edges = mesh.face_adjacency_edges    # (E,2) vertex index cạnh chung

        neighbors: dict[int, list] = defaultdict(list)
        for ei, (fi, fj) in enumerate(adj):
            neighbors[fi].append((int(fj), int(ei)))
            neighbors[fj].append((int(fi), int(ei)))

        # ── Tìm connected components rồi unfold từng island ───────────────
        all_panels   = []
        unvisited    = set(range(n_faces))
        island_id    = 0

        while unvisited:
            start = next(iter(unvisited))
            island_panels = _unfold_island(
                mesh, start, neighbors, adj, adj_edges, island_id, unvisited)
            all_panels.extend(island_panels)
            for p in island_panels:
                unvisited.discard(p.face_idx)
            island_id += 1

        progress("Unfold xong!", 100)
        result.panels     = all_panels
        result.success    = True
        result.num_islands = island_id

    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()

    return result


import heapq

def _angle_between_normals(n1, n2):
    return np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

def _unfold_island(mesh, start_face, neighbors, adj, adj_edges, island_id, unvisited):
    """MST unfold (Prim's) ưu tiên mặt phẳng, trả về list Panel2D."""
    panels: dict[int, Panel2D] = {}
    placed_polys = []
    
    pq = []
    heapq.heappush(pq, (0.0, start_face, -1, -1))
    
    island_visited = set()
    
    while pq:
        cost, current, parent, ei_parent = heapq.heappop(pq)
        
        if current in island_visited:
            continue
            
        if current not in unvisited and current not in panels:
            continue
            
        if parent == -1:
            v3d = mesh.triangles[current]
            v2d = _tri_to_2d(*v3d)
            panels[current] = Panel2D(
                face_idx=current,
                verts_2d=v2d,
                group_id=island_id,
            )
            placed_polys.append(Polygon(v2d))
            island_visited.add(current)
        else:
            parent_v2d = panels[parent].verts_2d
            
            shared_v_idx = adj_edges[ei_parent]
            sv0, sv1 = int(shared_v_idx[0]), int(shared_v_idx[1])

            # Đỉnh của parent
            pf = mesh.faces[parent]
            pi0 = int(np.where(pf == sv0)[0][0])
            pi1 = int(np.where(pf == sv1)[0][0])
            p2d_a = parent_v2d[pi0]
            p2d_b = parent_v2d[pi1]

            # Đỉnh của child
            cf = mesh.faces[current]
            ci0 = int(np.where(cf == sv0)[0][0])
            ci1 = int(np.where(cf == sv1)[0][0])
            
            # Tạo tam giác 2D cục bộ giữ nguyên winding order của 3D
            child_local = _tri_to_2d(
                mesh.vertices[cf[0]], 
                mesh.vertices[cf[1]], 
                mesh.vertices[cf[2]]
            )
            
            # Khớp cạnh (sv0, sv1) của child vào đúng vị trí (p2d_a, p2d_b) trên parent
            # Vì 3D mesh là manifold, winding order ngược nhau sẽ tự động làm child gập RA NGOÀI
            cl_a = child_local[ci0]
            cl_b = child_local[ci1]
            
            R, t = _build_transform(cl_a, cl_b, p2d_a, p2d_b)
            child_v2d = _apply_transform(child_local, R, t)

            ordered = child_v2d
            child_poly = Polygon(ordered)
            
            overlap = False
            # Check overlap với tất cả các poly đã đặt
            maxx1, maxy1 = ordered.max(axis=0)
            minx1, miny1 = ordered.min(axis=0)
            
            # Sử dụng buffer âm để tránh bắt nhầm các cạnh chạm nhau
            check_poly = child_poly.buffer(-1e-5)
            if check_poly.is_empty: # Triangle quá mỏng, bỏ qua
                check_poly = child_poly

            for p in placed_polys:
                minx2, miny2, maxx2, maxy2 = p.bounds
                if not (maxx1 <= minx2 or minx1 >= maxx2 or maxy1 <= miny2 or miny1 >= maxy2):
                    if check_poly.intersects(p):
                        overlap = True
                        break
            
            if overlap:
                continue

            placed_polys.append(child_poly)
            p_edge_i = _get_edge_index(mesh.faces[parent], sv0, sv1)
            c_edge_i = _get_edge_index(mesh.faces[current], sv0, sv1)
            
            panels[parent].fold_edges.add(p_edge_i)

            panels[current] = Panel2D(
                face_idx=current,
                verts_2d=ordered,
                group_id=island_id,
            )
            panels[current].fold_edges.add(c_edge_i)
            island_visited.add(current)

        # Enqueue neighbors
        n1 = mesh.face_normals[current]
        for child, ei in neighbors[current]:
            if child in island_visited or child not in unvisited:
                continue
            n2 = mesh.face_normals[child]
            angle = _angle_between_normals(n1, n2)
            heapq.heappush(pq, (angle, child, current, ei))

    return list(panels.values())


