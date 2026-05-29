"""
gui/main_window.py
Giao diện chính của Papercraft 3D App.
Layout: Sidebar trái | Preview giữa | Controls phải
"""
import sys
import os
import threading
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QPushButton, QSlider, QComboBox, QFileDialog,
    QProgressBar, QTextEdit, QTabWidget, QFrame, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QScrollArea,
    QSizePolicy, QApplication, QMessageBox, QColorDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QPolygonF,
    QPixmap, QPalette, QFontDatabase,
)
from PyQt5.QtCore import QPointF

import numpy as np

# ── Style sheet ───────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background-color: #0F0F1A;
    color: #E8E8F0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2A2A40;
    border-radius: 8px;
    margin-top: 12px;
    padding: 20px 8px 8px 8px;
    font-weight: bold;
    color: #9090B0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton {
    background: #1E1E30;
    border: 1px solid #3A3A55;
    border-radius: 6px;
    padding: 8px 16px;
    color: #E8E8F0;
    font-size: 13px;
}
QPushButton:hover { background: #2A2A45; border-color: #F5C518; }
QPushButton:pressed { background: #F5C518; color: #000; }
QPushButton:disabled { color: #555; border-color: #222; }
QComboBox {
    background: #1E1E30; border: 1px solid #3A3A55;
    border-radius: 6px; padding: 6px 10px; color: #E8E8F0;
    min-height: 28px;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #1E1E30; border: 1px solid #3A3A55;
    selection-background-color: #F5C518; selection-color: #000;
}
QSlider::groove:horizontal {
    height: 4px; background: #2A2A40; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0;
    background: #F5C518; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #F5C518; border-radius: 2px; }
QSpinBox, QDoubleSpinBox {
    background: #1E1E30; border: 1px solid #3A3A55;
    border-radius: 6px; padding: 5px 8px; color: #E8E8F0;
    min-height: 28px;
}
QProgressBar {
    background: #1E1E30; border: 1px solid #2A2A40;
    border-radius: 4px; height: 8px; text-align: center;
}
QProgressBar::chunk { background: #F5C518; border-radius: 4px; }
QTextEdit {
    background: #0A0A14; border: 1px solid #2A2A40;
    border-radius: 6px; color: #9090B0; font-family: 'Consolas', monospace;
    font-size: 11px; padding: 6px;
}
QTabWidget::pane { border: 1px solid #2A2A40; border-radius: 6px; }
QTabBar::tab {
    background: #1E1E30; color: #9090B0;
    padding: 8px 18px; border-radius: 6px 6px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #2A2A45; color: #F5C518; }
QScrollBar:vertical {
    background: #0F0F1A; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2A2A40; border-radius: 4px; min-height: 20px;
}
QLabel#label_title {
    font-size: 20px; font-weight: bold; color: #F5C518;
    letter-spacing: 1px;
}
QLabel#label_sub { color: #555577; font-size: 11px; }
QFrame#separator {
    background: #2A2A40; max-height: 1px;
}
QCheckBox { color: #A0A0C0; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #3A3A55;
    border-radius: 3px; background: #1E1E30;
}
QCheckBox::indicator:checked { background: #F5C518; border-color: #F5C518; }
"""


# ── Worker thread ─────────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    progress_sig = pyqtSignal(str, int)
    done_sig     = pyqtSignal(object, object, object)  # mesh_r, unfold_r, layout_r
    error_sig    = pyqtSignal(str)

    def __init__(self, mode, source, target_faces, target_size_mm):
        super().__init__()
        self.mode           = mode          # 'primitive' | 'file'
        self.source         = source        # primitive name or file path
        self.target_faces   = target_faces
        self.target_size_mm = target_size_mm

    def run(self):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from modules.mesh_loader import load_primitive, load_file
            from modules.unfolder   import unfold_mesh
            from modules.layout     import layout_panels

            # 1. Load mesh
            self.progress_sig.emit("Đang load model 3D...", 5)
            if self.mode == 'primitive':
                mesh_r = load_primitive(self.source, self.target_faces)
            else:
                mesh_r = load_file(self.source, self.target_faces)

            if not mesh_r.success:
                self.error_sig.emit(f"Load thất bại: {mesh_r.error}")
                return

            self.progress_sig.emit(
                f"Load xong: {mesh_r.final_faces} faces", 25)

            # 2. Unfold
            self.progress_sig.emit("Đang unfold mesh...", 30)
            unfold_r = unfold_mesh(
                mesh_r.mesh, max_faces=self.target_faces * 2,
                on_progress=lambda m, p: self.progress_sig.emit(m, 30 + p//3)
            )
            if not unfold_r.success:
                self.error_sig.emit(f"Unfold thất bại: {unfold_r.error}")
                return
            self.progress_sig.emit(
                f"Unfold xong: {len(unfold_r.panels)} panels", 65)

            # 3. Layout
            self.progress_sig.emit("Đang layout...", 70)
            layout_r = layout_panels(
                unfold_r.panels,
                target_size_mm=self.target_size_mm,
                mesh=mesh_r.mesh,
            )
            self.progress_sig.emit(
                f"Layout xong: {len(layout_r.pages)} trang", 100)

            self.done_sig.emit(mesh_r, unfold_r, layout_r)

        except Exception as e:
            import traceback
            self.error_sig.emit(f"{e}\n{traceback.format_exc()}")


# ── Papercraft Preview Widget ─────────────────────────────────────────────────
class PapercraftView(QWidget):
    """Widget vẽ preview papercraft 2D."""

    def __init__(self):
        super().__init__()
        self.layout_r   = None
        self.current_page = 0
        self.zoom       = 1.0
        self.setMinimumSize(400, 500)
        self.setStyleSheet("background:#0A0A14; border-radius:8px;")

    def set_layout(self, layout_r, page=0):
        self.layout_r    = layout_r
        self.current_page = page
        self.zoom        = 1.0
        self.update()

    def set_page(self, page):
        self.current_page = page
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#0A0A14"))

        if not self.layout_r or not self.layout_r.pages:
            painter.setPen(QColor("#3A3A55"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Chưa có dữ liệu\n\nChọn shape và nhấn ▶ Tạo Papercraft")
            return

        page_panels = self.layout_r.pages[self.current_page]
        if not page_panels:
            return

        # Tính scale để fit vào widget
        from modules.layout import A4_W_MM, A4_H_MM
        w, h = self.width() - 40, self.height() - 40
        scale_x = w / A4_W_MM
        scale_y = h / A4_H_MM
        px_per_mm = min(scale_x, scale_y) * self.zoom

        # Vẽ nền A4
        a4_w_px = A4_W_MM * px_per_mm
        a4_h_px = A4_H_MM * px_per_mm
        off_x = (self.width() - a4_w_px) / 2
        off_y = (self.height() - a4_h_px) / 2

        painter.setBrush(QBrush(QColor("#FAFAFA")))
        painter.setPen(QPen(QColor("#2A2A40"), 1))
        painter.drawRoundedRect(int(off_x), int(off_y),
                                int(a4_w_px), int(a4_h_px), 4, 4)

        # Vẽ từng panel (polygon n-gon sau v1.2)
        import colorsys
        for lp in page_panels:
            verts_mm = lp.verts_placed
            n_v = len(verts_mm)
            if n_v < 3:
                continue

            h_val = (lp.panel.face_idx * 47) % 360 / 360.0
            r, g, b = colorsys.hls_to_rgb(h_val, 0.72, 0.55)

            pts = QPolygonF([
                QPointF(off_x + v[0]*px_per_mm,
                        off_y + v[1]*px_per_mm)
                for v in verts_mm
            ])

            # Fill
            painter.setBrush(QBrush(QColor(int(r*255), int(g*255), int(b*255), 200)))
            painter.setPen(QPen(QColor("#333333"), 0.8))
            painter.drawPolygon(pts)

            # Fold lines (n-gon)
            painter.setPen(QPen(QColor("#1565C0"), 0.7, Qt.DashLine))
            for ei in lp.fold_edges:
                p0 = pts[ei]
                p1 = pts[(ei + 1) % n_v]
                painter.drawLine(p0, p1)

            # Label tại centroid
            cx = sum(p.x() for p in pts) / n_v
            cy = sum(p.y() for p in pts) / n_v
            painter.setPen(QPen(QColor("#222222")))
            painter.setFont(QFont("Arial", max(5, int(7*self.zoom))))
            painter.drawText(int(cx-8), int(cy+4), lp.label)

        # Page info
        painter.setPen(QColor("#555577"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(10, self.height()-10,
                         f"Trang {self.current_page+1}/{len(self.layout_r.pages)}  |  "
                         f"{len(page_panels)} panels  |  "
                         f"Scale: {self.layout_r.scale:.1f}")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom = max(0.5, min(3.0, self.zoom * (1.1 if delta > 0 else 0.9)))
        self.update()


# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Papercraft 3D  —  v1.0")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self.setStyleSheet(STYLE)

        self.mesh_r    = None
        self.unfold_r  = None
        self.layout_r  = None
        self.worker    = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background:#12121E; border-right:1px solid #2A2A40;")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(16, 16, 16, 16)
        sb_layout.setSpacing(12)

        # Logo
        title = QLabel("🎨 Papercraft 3D")
        title.setObjectName("label_title")
        sb_layout.addWidget(title)
        sub = QLabel("Tạo mẫu giấy từ mô hình 3D")
        sub.setObjectName("label_sub")
        sb_layout.addWidget(sub)

        sep = QFrame(); sep.setObjectName("separator")
        sb_layout.addWidget(sep)

        # ── Input source ──────────────────────────────────────────────────
        grp_src = QGroupBox("Nguồn Model 3D")
        grp_src_l = QVBoxLayout(grp_src)

        self.combo_source = QComboBox()
        self.combo_source.addItem("🔷  Primitive Shape", "primitive")
        self.combo_source.addItem("📂  Import file .obj/.stl", "file")
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        grp_src_l.addWidget(self.combo_source)

        # Primitive picker
        self.widget_primitive = QWidget()
        pl = QVBoxLayout(self.widget_primitive)
        pl.setContentsMargins(0, 4, 0, 0)
        self.combo_primitive = QComboBox()
        from modules.mesh_loader import get_primitive_list
        for k, v in get_primitive_list().items():
            icon = {'cube':'⬜','sphere':'⚪','cylinder':'🔵','cone':'🔺',
                    'torus':'💿','tetrahedron':'🔷','capsule':'💊','annulus':'⭕'}.get(k,'▪')
            self.combo_primitive.addItem(f"{icon}  {v}", k)
        pl.addWidget(self.combo_primitive)
        grp_src_l.addWidget(self.widget_primitive)

        # File picker
        self.widget_file = QWidget()
        fl = QVBoxLayout(self.widget_file)
        fl.setContentsMargins(0, 4, 0, 0)
        self.label_file = QLabel("Chưa chọn file")
        self.label_file.setStyleSheet("color:#555577; font-size:11px;")
        self.label_file.setWordWrap(True)
        fl.addWidget(self.label_file)
        btn_browse = QPushButton("📂  Chọn file 3D...")
        btn_browse.clicked.connect(self._browse_file)
        fl.addWidget(btn_browse)
        self.widget_file.hide()
        grp_src_l.addWidget(self.widget_file)

        sb_layout.addWidget(grp_src)

        # ── Settings ──────────────────────────────────────────────────────
        grp_set = QGroupBox("Cài Đặt")
        grp_set_l = QVBoxLayout(grp_set)

        grp_set_l.addWidget(QLabel("Số mặt (faces):"))
        self.spin_faces = QSpinBox()
        self.spin_faces.setRange(20, 800)
        self.spin_faces.setValue(120)
        self.spin_faces.setSingleStep(20)
        grp_set_l.addWidget(self.spin_faces)

        grp_set_l.addWidget(QLabel("Kích thước panel (mm):"))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(20, 150)
        self.spin_size.setValue(65)
        grp_set_l.addWidget(self.spin_size)

        self.chk_glue = QCheckBox("Hiển thị tab dán")
        self.chk_glue.setChecked(True)
        grp_set_l.addWidget(self.chk_glue)

        sb_layout.addWidget(grp_set)

        # ── Run button ────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Tạo Papercraft")
        self.btn_run.setFixedHeight(44)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #F5C518; color: #0F0F1A; font-weight: bold; font-size: 14px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #FFD700; }
            QPushButton:disabled { background-color: #3A3A55; color: #8888AA; }
        """)
        self.btn_run.clicked.connect(self._run_pipeline)
        sb_layout.addWidget(self.btn_run)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        sb_layout.addWidget(self.progress)

        # ── Export ────────────────────────────────────────────────────────
        grp_exp = QGroupBox("Xuất File")
        grp_exp_l = QVBoxLayout(grp_exp)
        self.btn_pdf = QPushButton("⬇  Xuất PDF")
        self.btn_pdf.setFixedHeight(44)
        self.btn_pdf.setStyleSheet("""
            QPushButton {
                background-color: #1565C0; color: white; font-weight: bold; font-size: 14px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #1E1E30; color: #555577; }
        """)
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self._export_pdf)
        grp_exp_l.addWidget(self.btn_pdf)
        sb_layout.addWidget(grp_exp)

        sb_layout.addStretch()

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        self.log.setPlaceholderText("Log...")
        sb_layout.addWidget(self.log)

        root.addWidget(sidebar)

        # ── Main area ──────────────────────────────────────────────────────
        main_area = QWidget()
        main_l = QVBoxLayout(main_area)
        main_l.setContentsMargins(12, 12, 12, 12)
        main_l.setSpacing(8)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget { border:none; }")

        # Tab 1: Papercraft preview
        self.preview = PapercraftView()
        self.tabs.addTab(self.preview, "📄 Papercraft Preview")

        # Tab 2: Từ ảnh (Nhánh 1)
        from gui.depth_panel import DepthPanel
        self.depth_panel = DepthPanel()
        self.depth_panel.layout_ready.connect(self._on_depth_layout_ready)
        self.tabs.addTab(self.depth_panel, "📷 Từ Ảnh (AI)")

        # Tab 3: Info
        self.info_widget = QTextEdit()
        self.info_widget.setReadOnly(True)
        self.info_widget.setPlaceholderText("Thông tin mesh và kết quả sẽ hiển thị ở đây...")
        self.tabs.addTab(self.info_widget, "ℹ️ Thông tin")

        main_l.addWidget(self.tabs)

        # Page navigator
        nav = QWidget()
        nav_l = QHBoxLayout(nav)
        nav_l.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("◀ Trang trước")
        self.btn_prev.clicked.connect(lambda: self._change_page(-1))
        self.btn_prev.setEnabled(False)
        self.lbl_page = QLabel("—")
        self.lbl_page.setAlignment(Qt.AlignCenter)
        self.btn_next = QPushButton("Trang sau ▶")
        self.btn_next.clicked.connect(lambda: self._change_page(1))
        self.btn_next.setEnabled(False)
        nav_l.addWidget(self.btn_prev)
        nav_l.addWidget(self.lbl_page, 1)
        nav_l.addWidget(self.btn_next)
        main_l.addWidget(nav)

        root.addWidget(main_area, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_source_changed(self, idx):
        mode = self.combo_source.currentData()
        self.widget_primitive.setVisible(mode == 'primitive')
        self.widget_file.setVisible(mode == 'file')

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file 3D",
            str(Path.home()),
            "3D Files (*.obj *.stl *.glb *.gltf *.ply *.off)"
        )
        if path:
            self.label_file.setText(Path(path).name)
            self.label_file.setToolTip(path)
            self.label_file.setProperty("path", path)

    def _log(self, msg: str, color="#9090B0"):
        self.log.append(f'<span style="color:{color};">{msg}</span>')

    def _run_pipeline(self):
        if self.worker and self.worker.isRunning():
            return

        mode = self.combo_source.currentData()
        if mode == 'primitive':
            source = self.combo_primitive.currentData()
        else:
            source = self.label_file.property("path")
            if not source:
                QMessageBox.warning(self, "Lỗi", "Chưa chọn file 3D!")
                return

        self.btn_run.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.worker = PipelineWorker(
            mode=mode,
            source=source,
            target_faces=self.spin_faces.value(),
            target_size_mm=self.spin_size.value(),
        )
        self.worker.progress_sig.connect(self._on_progress)
        self.worker.done_sig.connect(self._on_done)
        self.worker.error_sig.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.progress.setValue(pct)
        self._log(f"  {pct}%  {msg}")

    def _on_done(self, mesh_r, unfold_r, layout_r):
        self.mesh_r   = mesh_r
        self.unfold_r = unfold_r
        self.layout_r = layout_r

        self.preview.set_layout(layout_r, page=0)
        self._update_page_nav()

        # Info tab
        info = (f"<b style='color:#F5C518'>Model</b><br>"
                f"Nguồn   : {mesh_r.source_name}<br>"
                f"Faces   : {mesh_r.final_faces}<br><br>"
                f"<b style='color:#F5C518'>Unfold</b><br>"
                f"Panels  : {len(unfold_r.panels)}<br>"
                f"Islands : {unfold_r.num_islands}<br><br>"
                f"<b style='color:#F5C518'>Layout</b><br>"
                f"Trang   : {len(layout_r.pages)}<br>"
                f"Scale   : {layout_r.scale:.2f}")
        self.info_widget.setHtml(
            f"<div style='font-family:Consolas;font-size:12px;color:#C0C0D0;'>{info}</div>")

        self.btn_pdf.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        # Cảnh báo chất lượng papercraft (v1.3)
        self._check_papercraft_quality(unfold_r, layout_r)

        self._log("✅ Hoàn tất! Nhấn 'Xuất PDF' để lưu file.", "#4ECB71")

    def _check_papercraft_quality(self, unfold_r, layout_r):
        """Cảnh báo người dùng nếu mesh không phù hợp papercraft."""
        n_panels  = len(unfold_r.panels)
        n_islands = unfold_r.num_islands
        warnings = []
        if n_panels > 60:
            warnings.append(
                f"Quá nhiều mảnh ({n_panels}). Mesh phù hợp papercraft nên có ≤60 mảnh — "
                "thử Primitive (cube/sphere) hoặc STL khối kín với ít mặt cong hơn."
            )
        elif n_panels > 30:
            warnings.append(
                f"Hơi nhiều mảnh ({n_panels}) — cắt/dán sẽ tỉ mỉ."
            )
        if n_islands > 5:
            warnings.append(
                f"Quá nhiều island rời ({n_islands}). Lắp ráp sẽ khó vì các mảnh không liền nhau — "
                "thường do mesh cong hoặc bị overlap khi unfold."
            )
        elif n_islands > 2:
            warnings.append(
                f"Có {n_islands} island riêng biệt — lưu ý khi lắp ráp."
            )
        if layout_r.warnings:
            warnings.extend(layout_r.warnings)

        if not warnings:
            return

        for w in warnings:
            self._log(f"⚠️ {w}", "#F5C518")
        QMessageBox.warning(
            self, "Lưu ý chất lượng",
            "Mesh có thể không lý tưởng cho papercraft:\n\n• " + "\n• ".join(warnings)
        )

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._log(f"❌ {msg[:200]}", "#E74C3C")
        QMessageBox.critical(self, "Lỗi", msg[:300])

    def _update_page_nav(self):
        if not self.layout_r:
            return
        np_ = len(self.layout_r.pages)
        cp  = self.preview.current_page
        self.lbl_page.setText(f"Trang {cp+1} / {np_}")
        self.btn_prev.setEnabled(cp > 0)
        self.btn_next.setEnabled(cp < np_ - 1)

    def _change_page(self, delta):
        if not self.layout_r:
            return
        np_ = len(self.layout_r.pages)
        cp  = max(0, min(np_-1, self.preview.current_page + delta))
        self.preview.set_page(cp)
        self._update_page_nav()

    def _export_pdf(self):
        if not self.layout_r:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu PDF", str(Path.home() / "papercraft.pdf"),
            "PDF Files (*.pdf)")
        if not path:
            return

        from modules.pdf_exporter import export_pdf
        self._log("Đang xuất PDF...", "#F5C518")
        title_suffix = getattr(self.mesh_r, 'source_name', 'Từ Ảnh') if self.mesh_r else ''
        ok = export_pdf(
            self.layout_r, path,
            title=f"Papercraft 3D — {title_suffix}",
            show_glue=self.chk_glue.isChecked(),
        )
        if ok:
            self._log(f"✅ PDF đã lưu: {path}", "#4ECB71")
            QMessageBox.information(self, "Thành công",
                                    f"PDF đã lưu tại:\n{path}")
        else:
            self._log("❌ Xuất PDF thất bại!", "#E74C3C")

    def _on_depth_layout_ready(self, layout_r, depth_r):
        """Nhận kết quả từ DepthPanel → hiện preview + mở khóa export."""
        self.layout_r = layout_r
        self.mesh_r   = depth_r
        self.preview.set_layout(layout_r, page=0)
        self._update_page_nav()
        self.btn_pdf.setEnabled(True)
        self.tabs.setCurrentIndex(0)
        self._log("Depth → Papercraft xong! Xem preview và xuat PDF.", "#4ECB71")
