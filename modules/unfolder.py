"""
modules/unfolder.py
Thuật toán trải phẳng mesh 3D → các panel 2D dùng MST trên facet n-gon.

v1.2 — Gộp đồng phẳng trước khi unfold:
  - Các tam giác cùng mặt phẳng (góc < 3°) được gộp thành 1 facet đa giác.
  - Cube 12 triangles → 6 facet vuông; cylinder thân → các quad dọc.
  - MST chạy trên đồ thị facet (ít node hơn, ít cạnh "ảo" đè lên).

Pipeline:
  1. _extract_facets()       : union-find theo cạnh đồng phẳng → polygon outline
  2. _build_facet_adjacency(): map facet ↔ facet qua các cạnh không đồng phẳng
  3. _unfold_facets_island() : MST/BFS xoay từng facet vào mặt phẳng 2D
"""
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import heapq
import trimesh
from shapely.geometry import Polygon


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Panel2D:
    """Một mảnh papercraft — facet đa giác đã được trải phẳng ra 2D.

    Sau v1.2, panel có thể là tam giác (n=3) hoặc đa giác bất kỳ (n≥3) tuỳ
    việc gộp các face đồng phẳng.
    """
    face_idx:     int                          # face đại diện (dùng cho label/màu)
    verts_2d:     np.ndarray                   # shape (n, 2) — n đỉnh polygon outline
    color:        tuple = (220, 200, 160)
    group_id:     int   = 0                    # island id
    fold_edges:   set   = field(default_factory=set)  # local edge idx (MST edges)
    # ── Mới ở v1.2 ────────────────────────────────────────────────────────
    outline_vidx: list  = field(default_factory=list) # global vertex idx cho từng đỉnh outline
    face_indices: list  = field(default_factory=list) # tất cả mesh face idx thuộc facet này
    normal:       Optional[np.ndarray] = None         # facet normal (3,)

    @property
    def n(self) -> int:
        return len(self.verts_2d)

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
        x = self.verts_2d[:, 0]
        y = self.verts_2d[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass
class UnfoldResult:
    panels:      list           # list[Panel2D]
    success:     bool = True
    error:       str  = ""
    num_islands: int  = 0
    num_facets:  int  = 0       # tổng số facet đã sinh ra (bao gồm cả những facet bị skip do overlap)

    def summary(self):
        return (f"  Panels    : {len(self.panels)}\n"
                f"  Facets    : {self.num_facets}\n"
                f"  Islands   : {self.num_islands}\n"
                f"  Success   : {self.success}")


# ── 2D geometry helpers ───────────────────────────────────────────────────────

def _project_polygon_to_2d(verts_3d: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Chiếu polygon phẳng (n đỉnh, 3D) xuống hệ tọa độ 2D cục bộ.

    Trục x = cạnh đầu tiên; trục y = perpendicular trong mặt phẳng (theo normal).
    """
    n_verts = len(verts_3d)
    if n_verts < 3:
        return np.zeros((n_verts, 2))

    origin = verts_3d[0]
    e1 = verts_3d[1] - origin
    len_e1 = np.linalg.norm(e1)
    if len_e1 < 1e-10:
        return np.zeros((n_verts, 2))
    x_axis = e1 / len_e1

    y_axis = np.cross(normal, x_axis)
    len_y = np.linalg.norm(y_axis)
    if len_y < 1e-10:
        # Normal có thể bị suy biến → tự tính lại từ verts
        for k in range(2, n_verts):
            ek = verts_3d[k] - origin
            n_alt = np.cross(e1, ek)
            ln = np.linalg.norm(n_alt)
            if ln > 1e-10:
                n_alt = n_alt / ln
                y_axis = np.cross(n_alt, x_axis)
                len_y = np.linalg.norm(y_axis)
                if len_y > 1e-10:
                    break
        if len_y < 1e-10:
            return np.zeros((n_verts, 2))
    y_axis = y_axis / len_y

    out = np.zeros((n_verts, 2))
    for i in range(n_verts):
        rel = verts_3d[i] - origin
        out[i, 0] = float(np.dot(rel, x_axis))
        out[i, 1] = float(np.dot(rel, y_axis))
    return out


def _build_transform(src_a: np.ndarray, src_b: np.ndarray,
                     dst_a: np.ndarray, dst_b: np.ndarray):
    """Ma trận affine 2D (R, t) sao cho src_a→dst_a, src_b→dst_b."""
    s = src_b - src_a
    d = dst_b - dst_a
    if np.linalg.norm(s) < 1e-10 or np.linalg.norm(d) < 1e-10:
        return np.eye(2), np.zeros(2)

    angle = np.arctan2(d[1], d[0]) - np.arctan2(s[1], s[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    t = dst_a - R @ src_a
    return R, t


def _apply_transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T + t


def _angle_between_normals(n1, n2):
    return float(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))


# ── Facet extraction (gộp đồng phẳng) ─────────────────────────────────────────

def _walk_boundary(directed_edges: List[Tuple[int, int]]) -> List[int]:
    """Nối các cạnh có hướng thành 1 vòng kín. Trả về list[vertex_idx] CCW."""
    if not directed_edges:
        return []
    nxt: Dict[int, int] = {}
    for a, b in directed_edges:
        if a in nxt:
            # Vertex có >1 cạnh ra → boundary không phải simple loop, bỏ qua
            return []
        nxt[a] = b

    start = directed_edges[0][0]
    loop = [start]
    cur = start
    safety = len(directed_edges) + 4
    for _ in range(safety):
        nv = nxt.get(cur)
        if nv is None:
            return []
        if nv == start:
            return loop
        loop.append(nv)
        cur = nv
    return []


def _extract_facets(mesh: trimesh.Trimesh, angle_tol_deg: float = 3.0):
    """Gộp các face đồng phẳng (góc nhị diện < angle_tol_deg) thành facet đa giác.

    Trả về:
        facets: list[dict] với key: face_indices, outline_vidx, verts_3d, normal
        face_to_facet: np.ndarray (n_faces,) → facet index
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return [], np.zeros(0, dtype=int)

    coplanar_tol = float(np.radians(angle_tol_deg))

    # Union-find
    parent = list(range(n_faces))
    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        # Path compression
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    adj         = mesh.face_adjacency
    adj_angles  = mesh.face_adjacency_angles

    for (a, b), ang in zip(adj, adj_angles):
        if ang < coplanar_tol:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb

    groups: Dict[int, List[int]] = defaultdict(list)
    for f in range(n_faces):
        groups[find(f)].append(f)

    facets = []
    face_to_facet = np.full(n_faces, -1, dtype=int)

    for root, face_list in groups.items():
        # Tính normal trung bình theo diện tích
        normals = mesh.face_normals[face_list]
        areas   = mesh.area_faces[face_list]
        n_vec   = (normals * areas[:, None]).sum(axis=0)
        ln = np.linalg.norm(n_vec)
        if ln < 1e-10:
            n_vec = normals[0]
        else:
            n_vec = n_vec / ln

        # Tìm boundary outline:
        #   Mỗi face đóng góp 3 cạnh có hướng (theo winding).
        #   Cạnh nội bộ xuất hiện ở cả 2 chiều (chia bởi 2 face cùng facet).
        #   Cạnh boundary xuất hiện duy nhất 1 chiều.
        edge_owner: Dict[Tuple[int, int], int] = {}
        edge_count: Dict[Tuple[int, int], int] = defaultdict(int)

        for fi in face_list:
            verts = mesh.faces[fi]
            for i in range(3):
                a = int(verts[i])
                b = int(verts[(i + 1) % 3])
                edge_owner[(a, b)] = fi
                edge_count[(min(a, b), max(a, b))] += 1

        boundary_directed = [
            (a, b) for (a, b) in edge_owner
            if edge_count[(min(a, b), max(a, b))] == 1
        ]
        outline = _walk_boundary(boundary_directed)

        # Fallback: nếu boundary không phải simple loop, tách từng face làm 1 facet riêng
        if len(outline) < 3:
            for fi in face_list:
                facet_idx = len(facets)
                fv = mesh.faces[fi]
                facets.append({
                    'face_indices': [fi],
                    'outline_vidx': [int(fv[0]), int(fv[1]), int(fv[2])],
                    'verts_3d':     mesh.vertices[fv].copy(),
                    'normal':       mesh.face_normals[fi].copy(),
                })
                face_to_facet[fi] = facet_idx
            continue

        facet_idx = len(facets)
        for fi in face_list:
            face_to_facet[fi] = facet_idx
        facets.append({
            'face_indices': list(face_list),
            'outline_vidx': [int(v) for v in outline],
            'verts_3d':     mesh.vertices[outline].copy(),
            'normal':       n_vec,
        })

    return facets, face_to_facet


def _build_facet_adjacency(mesh: trimesh.Trimesh, face_to_facet: np.ndarray, facets: list):
    """Xây đồ thị adjacency giữa các facet.

    Một cặp facet (A, B) được nối nếu có ít nhất 1 cạnh mesh chia 2 face thuộc 2 facet khác nhau.
    Với facet đã được gộp đồng phẳng, hầu hết các cặp chỉ share đúng 1 cạnh outline.
    Nếu có nhiều cạnh share (mesh phức tạp), ta chỉ giữ cặp đầu tiên gặp được.

    Trả về:
        adj[facet_idx] = list[(other_facet, ei_in_self, ei_in_other, (v0_g, v1_g))]
    """
    # Bảng tra: (facet_idx, undirected_edge_key) → local edge index trong outline
    edge_to_local: Dict[Tuple[int, Tuple[int, int]], int] = {}
    for fi, fac in enumerate(facets):
        outline = fac['outline_vidx']
        n = len(outline)
        for i in range(n):
            a, b = outline[i], outline[(i + 1) % n]
            edge_to_local[(fi, (min(a, b), max(a, b)))] = i

    adj: Dict[int, list] = defaultdict(list)
    pair_seen = set()  # ((fa, fb), edge_key)

    for (fi_a, fi_b), edge_v in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        fa = int(face_to_facet[fi_a])
        fb = int(face_to_facet[fi_b])
        if fa == fb or fa < 0 or fb < 0:
            continue
        v0, v1 = int(edge_v[0]), int(edge_v[1])
        ek = (min(v0, v1), max(v0, v1))
        ei_a = edge_to_local.get((fa, ek))
        ei_b = edge_to_local.get((fb, ek))
        if ei_a is None or ei_b is None:
            continue
        pkey = (min(fa, fb), max(fa, fb), ek)
        if pkey in pair_seen:
            continue
        pair_seen.add(pkey)
        adj[fa].append((fb, ei_a, ei_b, (v0, v1)))
        adj[fb].append((fa, ei_b, ei_a, (v0, v1)))

    return adj


# ── Unfold MST trên facets ────────────────────────────────────────────────────

def _unfold_facets_island(facets, facet_adj, start_facet, island_id, unvisited):
    """MST/BFS unfold qua các facet, trả về list[Panel2D]."""
    panels: Dict[int, Panel2D] = {}
    placed_polys: List[Polygon] = []
    visited = set()

    # PQ entry: (cost, current, parent, ei_in_current, ei_in_parent)
    pq: list = []
    heapq.heappush(pq, (0.0, start_facet, -1, -1, -1))

    while pq:
        cost, current, parent, ei_cur, ei_par = heapq.heappop(pq)
        if current in visited or current not in unvisited:
            continue

        fac = facets[current]
        outline = fac['outline_vidx']
        n_verts = len(outline)

        if parent == -1:
            v2d = _project_polygon_to_2d(fac['verts_3d'], fac['normal'])
            panel = Panel2D(
                face_idx=fac['face_indices'][0],
                verts_2d=v2d,
                outline_vidx=list(outline),
                face_indices=list(fac['face_indices']),
                normal=np.array(fac['normal'], dtype=float),
                group_id=island_id,
            )
            panels[current] = panel
            placed_polys.append(Polygon(v2d))
            visited.add(current)
        else:
            parent_panel = panels[parent]

            # 2D điểm đích trên parent cho cạnh chia sẻ
            n_par = parent_panel.n
            p2d_start = parent_panel.verts_2d[ei_par]
            p2d_end   = parent_panel.verts_2d[(ei_par + 1) % n_par]

            # Local 2D của child theo normal của nó (giữ winding gốc 3D)
            child_local = _project_polygon_to_2d(fac['verts_3d'], fac['normal'])
            cl_start    = child_local[ei_cur]
            cl_end      = child_local[(ei_cur + 1) % n_verts]

            # Mesh manifold: cạnh share winding ngược ở child so với parent.
            # Khớp cl_start → p2d_end và cl_end → p2d_start ⇒ child gập RA NGOÀI parent.
            R, t = _build_transform(cl_start, cl_end, p2d_end, p2d_start)
            child_v2d = _apply_transform(child_local, R, t)

            child_poly = Polygon(child_v2d)
            if not child_poly.is_valid or child_poly.area < 1e-9:
                continue

            # Overlap check với tất cả poly đã đặt (buffer âm tránh kẹp cạnh chung)
            check_poly = child_poly.buffer(-1e-4)
            if check_poly.is_empty:
                check_poly = child_poly

            cmin = child_v2d.min(axis=0)
            cmax = child_v2d.max(axis=0)
            overlap = False
            for p in placed_polys:
                minx2, miny2, maxx2, maxy2 = p.bounds
                if (cmax[0] <= minx2 or cmin[0] >= maxx2 or
                    cmax[1] <= miny2 or cmin[1] >= maxy2):
                    continue
                if check_poly.intersects(p):
                    overlap = True
                    break
            if overlap:
                # Không unfold được — facet này sẽ trở thành seed của island khác
                continue

            placed_polys.append(child_poly)

            # Đánh dấu cạnh đã gập (MST edge) ở 2 phía
            panels[parent].fold_edges.add(ei_par)
            panel = Panel2D(
                face_idx=fac['face_indices'][0],
                verts_2d=child_v2d,
                outline_vidx=list(outline),
                face_indices=list(fac['face_indices']),
                normal=np.array(fac['normal'], dtype=float),
                group_id=island_id,
            )
            panel.fold_edges.add(ei_cur)
            panels[current] = panel
            visited.add(current)

        # Enqueue neighbors
        for (nf, ei_in_self, ei_in_nb, _edge_g) in facet_adj.get(current, []):
            if nf in visited or nf not in unvisited:
                continue
            cost_n = _angle_between_normals(fac['normal'], facets[nf]['normal'])
            # PQ tie-breaker: include index để tránh so sánh tuple với phần tử không order
            heapq.heappush(pq, (cost_n, nf, current, ei_in_nb, ei_in_self))

    return list(panels.values())


# ── Public API ────────────────────────────────────────────────────────────────

def unfold_mesh(mesh: trimesh.Trimesh,
                max_faces: int = 800,
                coplanar_tol_deg: float = 3.0,
                on_progress=None) -> UnfoldResult:
    """
    Trải phẳng mesh 3D thành các panel 2D.

    v1.2: gộp đồng phẳng trước khi unfold (cube 12 tri → 6 quad).

    Args:
        mesh             : trimesh.Trimesh
        max_faces        : giới hạn số mặt (simplify nếu vượt)
        coplanar_tol_deg : góc nhị diện <= ngưỡng này thì coi 2 face đồng phẳng
        on_progress      : callback(msg, pct)
    """
    def progress(msg, pct=0):
        print(f"  [{pct:3d}%] {msg}")
        if on_progress:
            on_progress(msg, pct)

    result = UnfoldResult(panels=[], success=False)

    try:
        mesh = mesh.copy()

        # Chuẩn hoá scale để các ngưỡng 1e-5 ổn định
        v_min = mesh.vertices.min(axis=0)
        v_max = mesh.vertices.max(axis=0)
        scale_factor = 100.0 / max((v_max - v_min).max(), 1e-6)
        mesh.vertices = (mesh.vertices - v_min) * scale_factor

        if len(mesh.faces) > max_faces:
            progress(f"Simplify {len(mesh.faces)} → {max_faces} mặt...", 5)
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
            mesh.remove_unreferenced_vertices()

        n_faces = len(mesh.faces)
        progress(f"Unfold {n_faces} mặt...", 10)

        # 1. Gộp đồng phẳng
        facets, face_to_facet = _extract_facets(mesh, coplanar_tol_deg)
        progress(f"Facets sau gộp đồng phẳng: {len(facets)}", 25)

        # 2. Adjacency giữa facets
        facet_adj = _build_facet_adjacency(mesh, face_to_facet, facets)

        # 3. Unfold từng island (connected component trên facet graph)
        all_panels: List[Panel2D] = []
        unvisited = set(range(len(facets)))
        island_id = 0

        while unvisited:
            start = next(iter(unvisited))
            island_panels = _unfold_facets_island(
                facets, facet_adj, start, island_id, unvisited)
            all_panels.extend(island_panels)
            for p in island_panels:
                # Đánh dấu facet đã unfold — tìm qua facet_idx (face_idx của panel chỉ là đại diện)
                # Cách an toàn: dùng outline_vidx + face_indices[0] đảo ngược về facet idx
                fi0 = p.face_indices[0]
                fac_idx = int(face_to_facet[fi0])
                unvisited.discard(fac_idx)
            # Nếu vòng lặp không tiến (start vẫn còn) ⇒ skip để tránh loop vô hạn
            if start in unvisited:
                unvisited.discard(start)
            island_id += 1

        progress("Unfold xong!", 100)
        result.panels      = all_panels
        result.success     = True
        result.num_islands = island_id
        result.num_facets  = len(facets)

    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()

    return result
