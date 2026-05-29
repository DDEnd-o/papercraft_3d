"""
modules/layout.py
Sắp xếp các Panel2D lên tờ giấy A4 (bin packing đơn giản).
Thêm tab dán (glue flaps) và đường gấp.

v1.2:
  - Panel2D giờ có thể là n-gon (đã gộp đồng phẳng), iterate edge theo `panel.n`.
  - Pack theo thứ tự FFDH (giảm dần chiều cao) để giảm khả năng island chồng nhau.
  - Overlap check toàn cục giữa các island sau khi pack, có retry với gap nới rộng.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union
from modules.unfolder import Panel2D

# ── Hằng số ──────────────────────────────────────────────────────────────────
MM_TO_PT = 2.8346       # 1mm = 2.8346 points (PDF unit)
A4_W_MM  = 210.0
A4_H_MM  = 297.0
MARGIN   = 12.0         # mm
TAB_SAFETY_MM = 5.0     # buffer cho glue tab khi check overlap (≈ TAB_WIDTH_MM ở pdf_exporter)


@dataclass
class LayoutPanel:
    """Panel đã được đặt lên trang — thêm offset vị trí."""
    panel:      Panel2D
    offset:     np.ndarray   # (2,) translation mm
    page:       int = 0
    label:      str = ""
    glue_edges: list = field(default_factory=list)  # [(edge_idx, target_label)]
    fold_edges: list = field(default_factory=list)  # [edge_idx]
    cut_edges:  list = field(default_factory=list)  # [edge_idx]
    flat_edges: list = field(default_factory=list)  # [edge_idx]

    @property
    def verts_placed(self) -> np.ndarray:
        return self.panel.verts_2d + self.offset


@dataclass
class LayoutResult:
    pages:      list        # list[list[LayoutPanel]] — mỗi trang một list
    scale:      float = 1.0
    success:    bool  = True
    error:      str   = ""
    warnings:   list  = field(default_factory=list)

    @property
    def total_panels(self):
        return sum(len(p) for p in self.pages)

    def summary(self):
        lines = [f"  Tổng panels : {self.total_panels}",
                 f"  Số trang    : {len(self.pages)}",
                 f"  Scale       : {self.scale:.3f}"]
        if self.warnings:
            lines.append(f"  ⚠️  Cảnh báo : {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"     - {w}")
        return "\n".join(lines)


# ── Scale panels ─────────────────────────────────────────────────────────────

def _compute_scale(panels: List[Panel2D], target_size_mm: float = 180.0) -> float:
    """Tính scale để island lớn nhất vừa với khổ giấy A4 (~180mm)."""
    islands = {}
    for p in panels:
        islands.setdefault(p.group_id, []).append(p)

    max_dim = 0.0
    for gid, island_panels in islands.items():
        v_min = np.min([p.bbox[0] for p in island_panels], axis=0)
        v_max = np.max([p.bbox[1] for p in island_panels], axis=0)
        w = v_max[0] - v_min[0] + 10.0
        h = v_max[1] - v_min[1] + 10.0
        max_dim = max(max_dim, w, h)

    if max_dim < 1e-6:
        return 1.0
    limit = min(target_size_mm, 180.0)
    return limit / max_dim


def _scale_panel(panel: Panel2D, scale: float) -> Panel2D:
    from copy import deepcopy
    p = deepcopy(panel)
    p.verts_2d = p.verts_2d * scale
    return p


# ── Bin packing (strip packing, FFDH) ─────────────────────────────────────────

def _pack_islands_on_pages(panels: List[Panel2D],
                           scale: float,
                           page_w: float = A4_W_MM - 2*MARGIN,
                           page_h: float = A4_H_MM - 2*MARGIN,
                           gap: float = 12.0) -> List[List[LayoutPanel]]:
    """First-Fit Decreasing Height strip packing.

    Sắp các island theo chiều cao giảm dần rồi xếp lên các "strip" ngang trên A4.
    Cách này giảm rõ rệt số trang và xác suất chồng nhau so với pack theo thứ tự ngẫu nhiên.
    """
    # Group by island
    islands: dict[int, list[Panel2D]] = {}
    for p in panels:
        islands.setdefault(p.group_id, []).append(p)

    # Scale + normalize + compute bbox cho từng island
    island_items = []
    padding = 5.0
    for gid, island_panels in islands.items():
        scaled = [_scale_panel(p, scale) for p in island_panels]
        mn = np.min([p.bbox[0] for p in scaled], axis=0) - padding
        mx = np.max([p.bbox[1] for p in scaled], axis=0) + padding
        for sp in scaled:
            sp.verts_2d -= mn
        w = float(mx[0] - mn[0])
        h = float(mx[1] - mn[1])
        island_items.append((gid, scaled, w, h))

    # FFDH: sort theo h giảm dần để pack chặt
    island_items.sort(key=lambda it: it[3], reverse=True)

    pages: list[list[LayoutPanel]] = [[]]
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0

    for gid, scaled, w, h in island_items:
        if cursor_x + w + gap > page_w and cursor_x > 0:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        if cursor_y + h + gap > page_h:
            pages.append([])
            cursor_x = 0.0
            cursor_y = 0.0
            row_height = 0.0

        offset = np.array([MARGIN + cursor_x, MARGIN + cursor_y])
        for sp in scaled:
            lp = LayoutPanel(
                panel=sp,
                offset=offset,
                page=len(pages) - 1,
                label=str(sp.face_idx + 1),
            )
            pages[-1].append(lp)

        cursor_x += w + gap
        row_height = max(row_height, h)

    return pages


# ── Glue flaps & fold edges (n-gon aware) ─────────────────────────────────────

def angle_between_normals(n1, n2):
    return np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))


def _assign_glue_and_fold(layout_panels: List[LayoutPanel], mesh):
    """Phân loại từng cạnh outline của panel:
        - fold_edges  : MST edge có góc nhị diện > 3°
        - flat_edges  : coplanar (chỉ xảy ra ở edge case, vì facet đã gộp)
        - glue_edges  : non-MST → thêm tab dán (1 phía)
        - cut_edges   : non-MST → cắt ở phía còn lại

    Dùng `panel.outline_vidx` để tra cứu cạnh global → tìm facet láng giềng.
    """
    if mesh is None:
        # Không có mesh: coi tất cả cạnh là cắt
        for lp in layout_panels:
            for ei in range(lp.panel.n):
                lp.cut_edges.append(ei)
        return

    # mesh edge (sorted) → (faceA, faceB)
    edge_to_faces = {}
    for i, (f0, f1) in enumerate(mesh.face_adjacency):
        v0, v1 = mesh.face_adjacency_edges[i]
        edge_key = (int(min(v0, v1)), int(max(v0, v1)))
        edge_to_faces[edge_key] = (int(f0), int(f1))

    # mesh face_idx → panel index (mỗi panel ôm nhiều face nếu đã gộp đồng phẳng)
    face_to_panel: dict[int, int] = {}
    for pi, lp in enumerate(layout_panels):
        face_list = lp.panel.face_indices or [lp.panel.face_idx]
        for fi in face_list:
            face_to_panel[int(fi)] = pi

    for lp in layout_panels:
        panel = lp.panel
        outline = panel.outline_vidx
        if not outline:
            # Backward compat fallback: dùng face_idx tam giác
            outline = [int(v) for v in mesh.faces[panel.face_idx]]
        n_edges = len(outline)

        our_faces = set(panel.face_indices) if panel.face_indices else {int(panel.face_idx)}

        if panel.normal is not None:
            n1 = np.asarray(panel.normal, dtype=float)
        else:
            n1 = mesh.face_normals[panel.face_idx]

        for ei in range(n_edges):
            v0 = outline[ei]
            v1 = outline[(ei + 1) % n_edges]
            edge_key = (min(int(v0), int(v1)), max(int(v0), int(v1)))
            is_mst_fold = ei in panel.fold_edges

            faces_pair = edge_to_faces.get(edge_key)
            if faces_pair is None:
                # Cạnh biên (open mesh) — không có hàng xóm
                if is_mst_fold:
                    lp.fold_edges.append(ei)
                else:
                    lp.cut_edges.append(ei)
                continue

            fA, fB = faces_pair
            if fA in our_faces and fB in our_faces:
                # Edge nội bộ của facet — lý ra không xuất hiện trong outline; nếu có thì ẩn
                lp.flat_edges.append(ei)
                continue

            neighbor_face = fB if fA in our_faces else fA
            n2 = mesh.face_normals[neighbor_face]
            angle_deg = float(np.degrees(angle_between_normals(n1, n2)))

            if angle_deg < 3.0:
                # Lớp phòng thủ: edge cross-facet vẫn đồng phẳng (do skip union-find ở extract)
                lp.flat_edges.append(ei)
                continue

            neighbor_pi = face_to_panel.get(neighbor_face)
            other_repr = (layout_panels[neighbor_pi].panel.face_idx
                          if neighbor_pi is not None else neighbor_face)

            if is_mst_fold:
                # Chỉ vẽ fold 1 lần (phía có face_idx nhỏ hơn)
                if panel.face_idx <= other_repr:
                    lp.fold_edges.append(ei)
            else:
                # Cạnh phải dán: tab ở phía nhỏ hơn, cắt ở phía còn lại
                if panel.face_idx < other_repr:
                    lp.glue_edges.append((ei, str(other_repr + 1)))
                else:
                    lp.cut_edges.append(ei)


# ── Overlap validation (toàn cục giữa các island) ─────────────────────────────

def _island_safety_polygon(island_lps: List[LayoutPanel]):
    """Polygon chiếm chỗ thực sự của 1 island = union polygon panels + buffer tab dán."""
    polys = []
    for lp in island_lps:
        if lp.panel.area > 1e-6:
            try:
                polys.append(Polygon(lp.verts_placed))
            except Exception:
                pass
    if not polys:
        return None
    return unary_union(polys).buffer(TAB_SAFETY_MM)


def _check_page_overlaps(page_panels: List[LayoutPanel],
                         min_overlap_mm2: float = 0.5) -> list:
    """Trả về list (gid_a, gid_b, overlap_area_mm2) cho mỗi cặp island chồng nhau trên 1 trang."""
    by_island: dict[int, List[LayoutPanel]] = {}
    for lp in page_panels:
        by_island.setdefault(lp.panel.group_id, []).append(lp)

    polys = {}
    for gid, lps in by_island.items():
        p = _island_safety_polygon(lps)
        if p is not None and not p.is_empty:
            polys[gid] = p

    gids = list(polys)
    conflicts = []
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            pa, pb = polys[gids[i]], polys[gids[j]]
            if not pa.intersects(pb):
                continue
            inter = pa.intersection(pb)
            if inter.area > min_overlap_mm2:
                conflicts.append((gids[i], gids[j], float(inter.area)))
    return conflicts


def _validate_global_overlaps(pages: List[List[LayoutPanel]],
                              min_overlap_mm2: float = 0.5) -> list:
    """Quét toàn bộ các trang, trả list (page_idx, conflicts_on_that_page)."""
    out = []
    for pi, page in enumerate(pages):
        c = _check_page_overlaps(page, min_overlap_mm2=min_overlap_mm2)
        if c:
            out.append((pi, c))
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def layout_panels(panels: List[Panel2D],
                  scale: float = None,
                  target_size_mm: float = 70.0,
                  mesh = None) -> LayoutResult:
    """
    Đặt tất cả panels lên các trang A4.

    Args:
        panels        : list Panel2D từ unfolder (có thể là n-gon đã gộp đồng phẳng)
        scale         : nếu None, tự tính từ target_size_mm
        target_size_mm: kích thước panel lớn nhất (mm)
        mesh          : trimesh để xác định glue/fold/flat

    Returns:
        LayoutResult với pages, warnings
    """
    result = LayoutResult(pages=[], success=False)
    try:
        if not panels:
            result.error = "Không có panel nào để layout"
            return result

        if scale is None:
            scale = _compute_scale(panels, target_size_mm)
        result.scale = scale

        # Pack + validate overlap, retry với gap nới rộng nếu cần
        gap = 12.0
        pages: List[List[LayoutPanel]] = []
        last_conflicts: list = []
        for attempt in range(4):
            pages = _pack_islands_on_pages(panels, scale, gap=gap)
            all_lps = [lp for page in pages for lp in page]
            # Reset edge classification mỗi lần retry (vì panel được deepcopy mới)
            for lp in all_lps:
                lp.fold_edges.clear()
                lp.cut_edges.clear()
                lp.glue_edges.clear()
                lp.flat_edges.clear()
            _assign_glue_and_fold(all_lps, mesh)
            last_conflicts = _validate_global_overlaps(pages)
            if not last_conflicts:
                break
            gap *= 1.4

        result.pages = pages
        if last_conflicts:
            for pi, c in last_conflicts:
                worst = max(area for *_, area in c)
                result.warnings.append(
                    f"Trang {pi+1}: {len(c)} cặp island chồng nhau "
                    f"(max {worst:.1f}mm², gap đã thử tới {gap/1.4:.1f}mm)"
                )

        result.success = True

    except Exception as e:
        result.error = str(e)
        import traceback; traceback.print_exc()

    return result
