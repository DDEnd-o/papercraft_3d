"""
modules/layout.py
Sắp xếp các Panel2D lên tờ giấy A4 (bin packing đơn giản).
Thêm tab dán (glue flaps) và đường gấp.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List
from modules.unfolder import Panel2D

# ── Hằng số ──────────────────────────────────────────────────────────────────
MM_TO_PT = 2.8346       # 1mm = 2.8346 points (PDF unit)
A4_W_MM  = 210.0
A4_H_MM  = 297.0
MARGIN   = 12.0         # mm


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

    @property
    def total_panels(self):
        return sum(len(p) for p in self.pages)

    def summary(self):
        return (f"  Tổng panels : {self.total_panels}\n"
                f"  Số trang    : {len(self.pages)}\n"
                f"  Scale       : {self.scale:.3f}")


# ── Scale panels ─────────────────────────────────────────────────────────────

def _compute_scale(panels: List[Panel2D], target_size_mm: float = 180.0) -> float:
    """Tính scale để island lớn nhất vừa với khổ giấy A4 (~180mm)."""
    # Gom theo island
    islands = {}
    for p in panels:
        islands.setdefault(p.group_id, []).append(p)
        
    max_dim = 0.0
    for gid, island_panels in islands.items():
        v_min = np.min([p.bbox[0] for p in island_panels], axis=0)
        v_max = np.max([p.bbox[1] for p in island_panels], axis=0)
        # Cộng thêm padding cho tab dán (~10mm)
        w = v_max[0] - v_min[0] + 10.0
        h = v_max[1] - v_min[1] + 10.0
        max_dim = max(max_dim, w, h)

    if max_dim < 1e-6:
        return 1.0
    
    # Khổ giấy A4 khả dụng là 180mm (trừ lề 15mm mỗi bên)
    limit = min(target_size_mm, 180.0)
    return limit / max_dim


def _scale_panel(panel: Panel2D, scale: float) -> Panel2D:
    from copy import deepcopy
    p = deepcopy(panel)
    p.verts_2d = p.verts_2d * scale
    return p


# ── Bin packing (strip packing) ───────────────────────────────────────────────

def _pack_islands_on_pages(panels: List[Panel2D],
                           scale: float,
                           page_w: float = A4_W_MM - 2*MARGIN,
                           page_h: float = A4_H_MM - 2*MARGIN,
                           gap: float = 12.0) -> List[List[LayoutPanel]]:
    pages: list[list[LayoutPanel]] = [[]]
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0

    # Group by island
    islands = {}
    for p in panels:
        islands.setdefault(p.group_id, []).append(p)
    
    # Process each island
    for gid, island_panels in islands.items():
        scaled_panels = [_scale_panel(p, scale) for p in island_panels]
        
        # Compute island bbox (bao gồm cả lề cho tab dán ~5mm)
        padding = 5.0
        mn = np.min([p.bbox[0] for p in scaled_panels], axis=0) - padding
        mx = np.max([p.bbox[1] for p in scaled_panels], axis=0) + padding
        w = mx[0] - mn[0]
        h = mx[1] - mn[1]
        
        # Normalize all panels so island min is at (0,0)
        for sp in scaled_panels:
            sp.verts_2d -= mn
            
        # Cần dòng mới?
        if cursor_x + w + gap > page_w and cursor_x > 0:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
            
        # Cần trang mới?
        if cursor_y + h + gap > page_h:
            pages.append([])
            cursor_x = 0.0
            cursor_y = 0.0
            row_height = 0.0
            
        offset = np.array([MARGIN + cursor_x, MARGIN + cursor_y])
        for sp in scaled_panels:
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


# ── Glue flaps & fold edges ───────────────────────────────────────────────────

def angle_between_normals(n1, n2):
    return np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

def _assign_glue_and_fold(layout_panels: List[LayoutPanel], mesh):
    """
    Với mỗi panel, xác định:
    - fold_edges: cạnh gấp có góc > 3 độ
    - flat_edges: cạnh gấp có góc <= 3 độ (nằm phẳng)
    - glue_edges: cạnh dán (có tab)
    - cut_edges: cạnh cắt (không có tab)
    """
    edge_to_faces = {}
    if mesh is not None:
        for i, (f0, f1) in enumerate(mesh.face_adjacency):
            v0, v1 = mesh.face_adjacency_edges[i]
            edge_key = tuple(sorted((int(v0), int(v1))))
            edge_to_faces[edge_key] = (int(f0), int(f1))

    for lp in layout_panels:
        face_idx = lp.panel.face_idx
        if mesh is not None:
            face_verts = mesh.faces[face_idx]
            n1 = mesh.face_normals[face_idx]
            for edge_i in range(3):
                v0 = face_verts[edge_i]
                v1 = face_verts[(edge_i + 1) % 3]
                edge_key = tuple(sorted((int(v0), int(v1))))
                
                if hasattr(lp.panel, 'fold_edges') and edge_i in lp.panel.fold_edges:
                    # It's a spanning tree edge
                    if edge_key in edge_to_faces:
                        fA, fB = edge_to_faces[edge_key]
                        neighbor_idx = fB if fA == face_idx else fA
                        n2 = mesh.face_normals[neighbor_idx]
                        angle = angle_between_normals(n1, n2)
                        
                        # Chỉ vẽ cạnh gập 1 lần để tránh các nét đứt đè lên nhau thành nét liền
                        if face_idx < neighbor_idx:
                            if np.degrees(angle) < 3.0:
                                lp.flat_edges.append(edge_i)
                            else:
                                lp.fold_edges.append(edge_i)
                    else:
                        lp.fold_edges.append(edge_i)
                else:
                    # Cut edge
                    if edge_key in edge_to_faces:
                        fA, fB = edge_to_faces[edge_key]
                        neighbor_idx = fB if fA == face_idx else fA
                        # Only one side gets the tab to avoid overlapping tabs
                        if face_idx < neighbor_idx:
                            lp.glue_edges.append((edge_i, str(neighbor_idx + 1)))
                        else:
                            lp.cut_edges.append(edge_i)
                    else:
                        lp.cut_edges.append(edge_i)


# ── Public API ────────────────────────────────────────────────────────────────

def layout_panels(panels: List[Panel2D],
                  scale: float = None,
                  target_size_mm: float = 70.0,
                  mesh = None) -> LayoutResult:
    """
    Đặt tất cả panels lên các trang A4.

    Args:
        panels        : list Panel2D từ unfolder
        scale         : nếu None, tự tính từ target_size_mm
        target_size_mm: kích thước panel lớn nhất (mm)
        adj_pairs     : mesh.face_adjacency để xác định glue/fold

    Returns:
        LayoutResult
    """
    result = LayoutResult(pages=[], success=False)
    try:
        if not panels:
            result.error = "Không có panel nào để layout"
            return result

        # Tính scale
        if scale is None:
            scale = _compute_scale(panels, target_size_mm)
        result.scale = scale

        # Sắp xếp panel lớn trước (giảm waste)
        sorted_panels = sorted(panels, key=lambda p: -p.area)

        # Pack
        pages = _pack_islands_on_pages(panels, scale)
        result.pages = pages

        # Gán glue/fold
        all_lps = [lp for page in pages for lp in page]
        _assign_glue_and_fold(all_lps, mesh)

        result.success = True

    except Exception as e:
        result.error = str(e)
        import traceback; traceback.print_exc()

    return result
