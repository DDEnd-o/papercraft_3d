"""
modules/pdf_exporter.py
Xuất layout papercraft ra file PDF đẹp bằng ReportLab.
- Vẽ các tam giác có màu
- Đường cắt (đen), đường gấp (xanh nét đứt), tab dán (xám)
- Đánh số từng mảnh
- Trang cuối: bảng hướng dẫn lắp ráp
"""
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import platform

from modules.layout import LayoutResult, A4_W_MM, A4_H_MM

def _register_fonts():
    try:
        if platform.system() == "Windows":
            pdfmetrics.registerFont(TTFont('Arial', 'C:\\Windows\\Fonts\\arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:\\Windows\\Fonts\\arialbd.ttf'))
        elif platform.system() == "Darwin":
            pdfmetrics.registerFont(TTFont('Arial', '/Library/Fonts/Arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', '/Library/Fonts/Arial Bold.ttf'))
        else:
            pdfmetrics.registerFont(TTFont('Arial', '/usr/share/fonts/truetype/msttcorefonts/arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', '/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf'))
        return 'Arial', 'Arial-Bold'
    except Exception:
        return 'Helvetica', 'Helvetica-Bold'

FONT_REGULAR, FONT_BOLD = _register_fonts()


# ── Màu sắc ──────────────────────────────────────────────────────────────────
CUT_COLOR   = colors.black
FOLD_COLOR  = HexColor('#1565C0')
GLUE_COLOR  = HexColor('#BBBBBB')
GLUE_FILL   = HexColor('#EEEEEE')
LABEL_COLOR = HexColor('#333333')
BG_COLOR    = HexColor('#FAFAFA')

TAB_WIDTH_MM = 5.0    # chiều rộng tab dán
FOLD_DASH    = [2, 2] # nét đứt đường gấp (mm)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mm_to_pt(v: float) -> float:
    return v * mm

def _pts(arr: np.ndarray):
    """Chuyển tọa độ mm → points cho ReportLab, lật trục Y (PDF gốc dưới trái)."""
    return [(x * mm, (A4_H_MM - y) * mm) for x, y in arr]

def _draw_panel_fill(c: rl_canvas.Canvas, verts_mm: np.ndarray):
    """Vẽ vùng nền trắng (không viền) cho 1 panel — hỗ trợ polygon n-gon."""
    pts = _pts(verts_mm)
    if len(pts) < 3:
        return
    p = c.beginPath()
    p.moveTo(*pts[0])
    for px, py in pts[1:]:
        p.lineTo(px, py)
    p.close()
    c.setFillColorRGB(1, 1, 1, 1.0)
    c.drawPath(p, fill=1, stroke=0)

def _draw_cut_line(c: rl_canvas.Canvas, v0_mm, v1_mm):
    """Vẽ đường cắt: nét liền đen."""
    pts = _pts(np.array([v0_mm, v1_mm]))
    c.setStrokeColor(CUT_COLOR)
    c.setLineWidth(0.8)
    c.setDash()
    c.line(*pts[0], *pts[1])

def _draw_fold_line(c: rl_canvas.Canvas, v0_mm, v1_mm):
    """Vẽ đường gấp: nét đứt xanh."""
    pts = _pts(np.array([v0_mm, v1_mm]))
    c.setStrokeColor(FOLD_COLOR)
    c.setLineWidth(0.6)
    c.setDash(FOLD_DASH[0]*mm, FOLD_DASH[1]*mm)
    c.line(*pts[0], *pts[1])
    c.setDash()  # reset

def _draw_glue_tab(c: rl_canvas.Canvas, v0_mm, v1_mm, width_mm=3.0):
    """Vẽ tab dán: nền trắng có dấu 'x' ở giữa, viền cắt và đường gấp ở chân."""
    v0 = np.array(v0_mm)
    v1 = np.array(v1_mm)
    edge = v1 - v0
    length = np.linalg.norm(edge)
    if length < 1e-6:
        return
    actual_width = min(width_mm, length * 0.3) # Giới hạn tab nhỏ gọn, không quá 30% chiều dài cạnh
    
    # Đổi hướng perp ra ngoài (Exterior)
    perp = np.array([edge[1], -edge[0]]) / length * actual_width
    
    # Làm thon gọn hai bên tab tạo góc vát nhọn hơn (Trapezoid)
    taper = edge * 0.35 
    if np.linalg.norm(taper) * 2 >= length:
        taper = edge * 0.1
        
    p0_out = v0 + perp + taper
    p1_out = v1 + perp - taper
    
    tab = np.array([v0, v1, p1_out, p0_out])
    pts = _pts(tab)
    
    # Fill background (White)
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.close()
    c.setFillColorRGB(1, 1, 1)
    c.drawPath(p, fill=1, stroke=0)
    
    # Draw outline (3 sides)
    c.setStrokeColor(CUT_COLOR)
    c.setLineWidth(0.8)
    c.setDash()
    p = c.beginPath()
    p.moveTo(*pts[1])
    p.lineTo(*pts[2])
    p.lineTo(*pts[3])
    p.lineTo(*pts[0])
    c.drawPath(p, fill=0, stroke=1)
    
    # Draw fold line at base
    _draw_fold_line(c, v0_mm, v1_mm)
    
    # Vẽ chữ 'x' ở giữa tab
    center_mm = (v0 + v1) / 2 + perp * 0.5
    cx, cy = _pts([center_mm])[0]
    pdf_edge_x = edge[0]
    pdf_edge_y = -edge[1]
    angle = np.degrees(np.arctan2(pdf_edge_y, pdf_edge_x))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
        
    c.saveState()
    c.translate(cx, cy)
    c.rotate(angle)
    c.setFillColorRGB(0.2, 0.2, 0.2) # Dark grey for 'x'
    c.setFont(FONT_REGULAR, 4.0)
    c.drawCentredString(0, -1.0, "x")
    c.restoreState()

def _draw_label(c: rl_canvas.Canvas, centroid_mm, text: str, fontsize=6):
    """Đánh số mảnh."""
    x, y = _pts([centroid_mm])[0]
    c.setFillColor(LABEL_COLOR)
    c.setFont(FONT_BOLD, fontsize)
    c.drawCentredString(x, y, text)


def _draw_scale_ruler(c: rl_canvas.Canvas, x_mm: float, y_mm: float,
                      length_mm: float = 100.0):
    """Vẽ thước calibration để người dùng kiểm tra tỉ lệ in.

    Một thước 100mm chia vạch mỗi 10mm. Sau khi in, dùng thước thật đo lại
    đoạn này — nếu khác 100mm thì máy in đang scale sai (chỉnh 'Actual Size'
    / 100% trong dialog in).
    """
    pts = _pts(np.array([[x_mm, y_mm], [x_mm + length_mm, y_mm]]))
    c.setStrokeColor(CUT_COLOR)
    c.setLineWidth(0.6)
    c.setDash()
    c.line(*pts[0], *pts[1])

    # Vạch chia mỗi 10mm
    for i in range(int(length_mm // 10) + 1):
        tick_x = x_mm + i * 10
        tick_h = 2.0 if i % 5 else 3.2   # vạch dài hơn ở 0, 50, 100mm
        tp = _pts(np.array([[tick_x, y_mm], [tick_x, y_mm - tick_h]]))
        c.line(*tp[0], *tp[1])

    # Nhãn "0", "100mm"
    c.setFillColor(LABEL_COLOR)
    c.setFont(FONT_REGULAR, 5.5)
    lbl0 = _pts([np.array([x_mm, y_mm + 1.2])])[0]
    lbl_end = _pts([np.array([x_mm + length_mm, y_mm + 1.2])])[0]
    c.drawString(lbl0[0] - 1*mm, lbl0[1], "0")
    c.drawString(lbl_end[0] - 5*mm, lbl_end[1], f"{int(length_mm)}mm")

    # Hint nhỏ
    c.setFillColor(colors.gray)
    c.setFont(FONT_REGULAR, 5)
    hint = _pts([np.array([x_mm, y_mm - 6])])[0]
    c.drawString(hint[0], hint[1],
                 "Calibrate: đoạn này phải dài đúng 100mm khi in (in 'Actual Size / 100%').")


# ── Page drawing ─────────────────────────────────────────────────────────────

def _draw_page(c: rl_canvas.Canvas, layout_panels: list, page_num: int,
               total_pages: int, show_glue: bool = True):
    """Vẽ một trang A4."""
    # Nền nhẹ
    c.setFillColor(BG_COLOR)
    c.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)

    # Header
    c.setFillColor(colors.HexColor('#1A1A2E'))
    c.setFont(FONT_BOLD, 9)
    c.drawString(10*mm, (A4_H_MM - 7)*mm, "🎨 Papercraft 3D")
    c.setFont(FONT_REGULAR, 7)
    c.setFillColor(colors.gray)
    c.drawRightString((A4_W_MM - 10)*mm, (A4_H_MM - 7)*mm,
                      f"Trang {page_num+1} / {total_pages}")

    # Legend (chỉ trang 1)
    if page_num == 0:
        lx, ly = 10*mm, 8*mm
        c.setFont(FONT_REGULAR, 6)
        c.setStrokeColor(CUT_COLOR);   c.setLineWidth(0.5)
        c.line(lx, ly+1.5*mm, lx+8*mm, ly+1.5*mm)
        c.setFillColor(colors.black);  c.drawString(lx+9*mm, ly+0.8*mm, "Đường cắt")

        c.setStrokeColor(FOLD_COLOR);  c.setLineWidth(0.6)
        c.setDash(2*mm, 2*mm)
        c.line(lx+40*mm, ly+1.5*mm, lx+48*mm, ly+1.5*mm)
        c.setDash()
        c.setFillColor(FOLD_COLOR);    c.drawString(lx+49*mm, ly+0.8*mm, "Đường gấp")

        c.setFillColor(GLUE_FILL);     c.setStrokeColor(GLUE_COLOR)
        c.rect(lx+85*mm, ly, 8*mm, 3*mm, fill=1, stroke=1)
        c.setFillColor(colors.black);  c.drawString(lx+94*mm, ly+0.8*mm, "Tab dán")

        # Thước calibration 100mm (góc trên-phải, dưới header)
        _draw_scale_ruler(c, x_mm=A4_W_MM - 110, y_mm=14, length_mm=100.0)

    # Vẽ các panel
    for lp in layout_panels:
        verts = lp.verts_placed  # mm
        n_v = len(verts)
        if n_v < 3:
            continue

        # Tô nền panel (polygon n-gon)
        _draw_panel_fill(c, verts)

        # Vẽ đường cắt
        for ei in lp.cut_edges:
            v0 = verts[ei]
            v1 = verts[(ei + 1) % n_v]
            _draw_cut_line(c, v0, v1)

        # Vẽ tab dán (bao gồm cả nét cắt viền và nét gấp)
        for ei, _ in lp.glue_edges:
            v0 = verts[ei]
            v1 = verts[(ei + 1) % n_v]
            if show_glue:
                _draw_glue_tab(c, v0, v1)

        # Vẽ đường gấp
        for ei in lp.fold_edges:
            v0 = verts[ei]
            v1 = verts[(ei + 1) % n_v]
            _draw_fold_line(c, v0, v1)

        # flat_edges thì KHÔNG VẼ GÌ (tự động giấu nét)

    # Footer
    c.setFillColor(colors.lightgrey)
    c.setFont(FONT_REGULAR, 5.5)
    c.drawCentredString(A4[0]/2, 5*mm,
                        "Cắt theo đường đen ─ Gấp theo đường xanh đứt ─ Dán tab xám")


def _draw_assembly_guide(c: rl_canvas.Canvas, total_panels: int):
    """Trang hướng dẫn lắp ráp."""
    c.setFillColor(HexColor('#1A1A2E'))
    c.rect(0, (A4_H_MM-25)*mm, A4[0], 25*mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(A4[0]/2, (A4_H_MM-14)*mm, "HƯỚNG DẪN LẮP RÁP")
    c.setFont(FONT_REGULAR, 8)
    c.drawCentredString(A4[0]/2, (A4_H_MM-21)*mm,
                        f"Papercraft 3D — {total_panels} mảnh")

    steps = [
        ("1. Chuẩn bị",
         "In tất cả các trang trên giấy 120–160gsm. Dùng giấy dày hơn để mô hình cứng hơn."),
        ("2. Cắt",
         "Cắt theo đường đen (đường cắt). Hãy cắt cẩn thận, đặc biệt ở các góc nhọn."),
        ("3. Gấp",
         "Gấp theo đường xanh đứt. Gấp vào (mountain fold) — phần màu nằm bên ngoài."),
        ("4. Dán",
         "Dán tab (có dấu 'x') vào mặt bên trong của mảnh liền kề. Dấu 'x' chỉ phần keo bôi."),
        ("5. Lắp ráp",
         "Bắt đầu từ mảnh số 1, lần lượt ghép các mảnh liền kề. Dùng kẹp giấy để giữ trong khi keo khô."),
        ("6. Hoàn thiện",
         "Để keo khô hoàn toàn (15–30 phút). Có thể dùng bút màu để tô thêm chi tiết."),
    ]

    y = A4_H_MM - 35
    for title, desc in steps:
        c.setFillColor(HexColor('#1A1A2E'))
        c.setFont(FONT_BOLD, 9)
        c.drawString(15*mm, y*mm, title)
        y -= 7
        c.setFillColor(HexColor('#444444'))
        c.setFont(FONT_REGULAR, 8)
        # Wrap text đơn giản
        words = desc.split()
        line = ""
        for word in words:
            test = line + word + " "
            if len(test) * 4.5 > (A4_W_MM - 30):
                c.drawString(20*mm, y*mm, line.strip())
                y -= 6
                line = word + " "
            else:
                line = test
        if line:
            c.drawString(20*mm, y*mm, line.strip())
            y -= 10

    # Tips box
    c.setStrokeColor(HexColor('#F5C518'))
    c.setLineWidth(1)
    c.setFillColor(HexColor('#FFFDE7'))
    c.roundRect(12*mm, (y-15)*mm, (A4_W_MM-24)*mm, 18*mm,
                4*mm, fill=1, stroke=1)
    c.setFillColor(HexColor('#E65100'))
    c.setFont(FONT_BOLD, 8)
    c.drawString(17*mm, (y-4)*mm, "💡 Mẹo:")
    c.setFillColor(HexColor('#333333'))
    c.setFont(FONT_REGULAR, 7.5)
    c.drawString(17*mm, (y-11)*mm,
                 "Chụp ảnh mô hình sau khi hoàn thành và chia sẻ kết quả!")


# ── HSL helper ────────────────────────────────────────────────────────────────

def _hsl_to_rgb(h, s, l):
    import colorsys
    return colorsys.hls_to_rgb(h, l, s)


# ── Main export ───────────────────────────────────────────────────────────────

def export_pdf(layout_result: LayoutResult,
               output_path: str | Path,
               title: str = "Papercraft 3D",
               show_glue: bool = True,
               on_progress=None) -> bool:
    """
    Xuất toàn bộ papercraft ra file PDF.

    Args:
        layout_result : kết quả từ layout.layout_panels()
        output_path   : đường dẫn file .pdf
        title         : tiêu đề in trên PDF
        show_glue     : có vẽ tab dán không
        on_progress   : callback(msg, pct)

    Returns:
        True nếu thành công
    """
    def progress(msg, pct=0):
        print(f"  [{pct:3d}%] {msg}")
        if on_progress:
            on_progress(msg, pct)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        c = rl_canvas.Canvas(str(output_path), pagesize=A4)
        c.setTitle(title)
        c.setAuthor("Papercraft 3D App")

        total_pages = len(layout_result.pages) + 1  # +1 cho trang hướng dẫn
        total_panels = layout_result.total_panels

        for pi, page_panels in enumerate(layout_result.pages):
            progress(f"Vẽ trang {pi+1}/{len(layout_result.pages)}...",
                     int(pi / total_pages * 90))
            _draw_page(c, page_panels, pi, total_pages, show_glue)
            c.showPage()

        # Trang hướng dẫn
        progress("Vẽ trang hướng dẫn...", 92)
        _draw_assembly_guide(c, total_panels)
        c.showPage()

        c.save()
        progress("✅ Xuất PDF xong!", 100)
        return True

    except Exception as e:
        print(f"❌ Lỗi xuất PDF: {e}")
        import traceback; traceback.print_exc()
        return False
