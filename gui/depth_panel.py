"""
gui/depth_panel.py
Tab "Từ Ảnh" trong GUI — Nhánh 1: Ảnh → Depth → Mesh → Papercraft
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QSpinBox, QCheckBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui  import QPixmap, QImage
from pathlib import Path
import numpy as np


class DepthWorker(QThread):
    progress_sig = pyqtSignal(str, int)
    done_sig     = pyqtSignal(object, object, object)   # depth_r, unfold_r, layout_r
    error_sig    = pyqtSignal(str)

    def __init__(self, image_path, model_size, target_faces, target_size_mm):
        super().__init__()
        self.image_path    = image_path
        self.model_size    = model_size
        self.target_faces  = target_faces
        self.target_size_mm = target_size_mm

    def run(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from modules.depth_estimator import estimate_depth_and_build_mesh
            from modules.unfolder        import unfold_mesh
            from modules.layout          import layout_panels

            # Depth estimation
            self.progress_sig.emit("Chạy Depth Anything V2...", 5)
            depth_r = estimate_depth_and_build_mesh(
                image_path    = self.image_path,
                model_size    = self.model_size,
                target_faces  = self.target_faces,
                on_progress   = lambda m,p: self.progress_sig.emit(m, int(p*0.6)),
            )
            if not depth_r.success:
                self.error_sig.emit(depth_r.error); return

            self.progress_sig.emit(f"Depth OK! {depth_r.num_faces} faces", 62)

            # Unfold (depth mesh có nhiều mặt cong → nới tol 25° để gom mạnh)
            self.progress_sig.emit("Unfold mesh...", 65)
            unfold_r = unfold_mesh(depth_r.mesh, max_faces=self.target_faces*2,
                coplanar_tol_deg=25.0,
                on_progress=lambda m,p: self.progress_sig.emit(m, 65+p//5))
            if not unfold_r.success:
                self.error_sig.emit(unfold_r.error); return

            self.progress_sig.emit(f"Unfold OK! {len(unfold_r.panels)} panels", 85)

            # Layout
            self.progress_sig.emit("Layout panels...", 88)
            layout_r = layout_panels(
                unfold_r.panels,
                target_size_mm = self.target_size_mm,
                mesh           = depth_r.mesh,
            )
            self.progress_sig.emit("Hoàn tất!", 100)
            self.done_sig.emit(depth_r, unfold_r, layout_r)

        except Exception as e:
            import traceback
            self.error_sig.emit(f"{e}\n{traceback.format_exc()}")


class DepthPanel(QWidget):
    """Panel Tab 'Từ Ảnh' — tích hợp vào MainWindow."""

    # Signal gửi layout_r về MainWindow để preview
    layout_ready = pyqtSignal(object, object)  # layout_r, depth_r

    def __init__(self):
        super().__init__()
        self.image_path = None
        self.depth_r    = None
        self.layout_r   = None
        self.worker     = None
        self._build()

    def _build(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(8, 8, 16, 8)

        # ── Chọn ảnh ──────────────────────────────────────────────────────
        grp_img = QGroupBox("Ảnh đầu vào")
        grp_l   = QVBoxLayout(grp_img)

        self.lbl_img = QLabel("Chưa chọn ảnh")
        self.lbl_img.setStyleSheet("color:#555577; font-size:11px;")
        self.lbl_img.setWordWrap(True)
        grp_l.addWidget(self.lbl_img)

        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedHeight(140)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet(
            "background:#0A0A14; border:1px solid #2A2A40; border-radius:6px;")
        self.lbl_preview.setText("Preview ảnh")
        grp_l.addWidget(self.lbl_preview)

        btn_pick = QPushButton("📷  Chọn ảnh...")
        btn_pick.clicked.connect(self._pick_image)
        grp_l.addWidget(btn_pick)
        layout.addWidget(grp_img)

        # ── Cài đặt ───────────────────────────────────────────────────────
        grp_cfg = QGroupBox("Cài đặt AI")
        cfg_l   = QVBoxLayout(grp_cfg)

        cfg_l.addWidget(QLabel("Model size:"))
        self.combo_model = QComboBox()
        self.combo_model.addItem("🚀 Small (~100MB) — CPU OK", "small")
        self.combo_model.addItem("⚡ Base  (~400MB) — GPU tốt hơn", "base")
        self.combo_model.addItem("🎯 Large (~1.3GB) — GPU 4GB+", "large")
        cfg_l.addWidget(self.combo_model)

        cfg_l.addWidget(QLabel("Số faces mục tiêu:"))
        self.spin_faces = QSpinBox()
        self.spin_faces.setRange(50, 800)
        self.spin_faces.setValue(150)
        self.spin_faces.setSingleStep(25)
        cfg_l.addWidget(self.spin_faces)

        cfg_l.addWidget(QLabel("Kích thước panel (mm):"))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(20, 120)
        self.spin_size.setValue(60)
        cfg_l.addWidget(self.spin_size)

        layout.addWidget(grp_cfg)

        # ── Run ───────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Tạo Papercraft từ Ảnh")
        self.btn_run.setFixedHeight(44)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #F5C518; color: #0F0F1A; font-weight: bold; font-size: 14px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #FFD700; }
            QPushButton:disabled { background-color: #3A3A55; color: #8888AA; }
        """)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── Depth preview ─────────────────────────────────────────────────
        grp_depth = QGroupBox("Depth Map")
        depth_l   = QVBoxLayout(grp_depth)
        self.lbl_depth = QLabel("Depth map sẽ hiện ở đây sau khi chạy")
        self.lbl_depth.setFixedHeight(140)
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setStyleSheet(
            "background:#0A0A14; border:1px solid #2A2A40; border-radius:6px; color:#555577;")
        depth_l.addWidget(self.lbl_depth)
        layout.addWidget(grp_depth)

        # ── Log ───────────────────────────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(100)
        self.log.setPlaceholderText("Log...")
        layout.addWidget(self.log)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self.image_path = path
        self.lbl_img.setText(Path(path).name)

        # Show preview
        pix = QPixmap(path).scaledToHeight(130, Qt.SmoothTransformation)
        self.lbl_preview.setPixmap(pix)
        self.btn_run.setEnabled(True)
        self._log(f"Đã chọn: {Path(path).name}", "#F5C518")

    def _log(self, msg, color="#9090B0"):
        self.log.append(f'<span style="color:{color};">{msg}</span>')

    def _run(self):
        if not self.image_path:
            return
        if self.worker and self.worker.isRunning():
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._log("Bắt đầu pipeline...", "#F5C518")

        self.worker = DepthWorker(
            image_path    = self.image_path,
            model_size    = self.combo_model.currentData(),
            target_faces  = self.spin_faces.value(),
            target_size_mm = self.spin_size.value(),
        )
        self.worker.progress_sig.connect(self._on_progress)
        self.worker.done_sig.connect(self._on_done)
        self.worker.error_sig.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.progress.setValue(pct)
        self._log(f"  {pct}% {msg}")

    def _on_done(self, depth_r, unfold_r, layout_r):
        self.depth_r  = depth_r
        self.layout_r = layout_r

        # Hiện depth map
        if depth_r.depth_img is not None:
            import numpy as np
            from PyQt5.QtGui import QImage
            d = np.array(depth_r.depth_img.convert("L"))
            h, w = d.shape
            qimg = QImage(d.data, w, h, w, QImage.Format_Grayscale8)
            pix  = QPixmap.fromImage(qimg).scaledToHeight(
                130, Qt.SmoothTransformation)
            self.lbl_depth.setPixmap(pix)

        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._log(f"✅ Xong! {len(unfold_r.panels)} panels, "
                  f"{len(layout_r.pages)} trang", "#4ECB71")

        # Gửi về MainWindow để preview + export
        self.layout_ready.emit(layout_r, depth_r)

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._log(f"❌ {msg[:300]}", "#E74C3C")
