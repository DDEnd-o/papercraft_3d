"""
gui/multiview_panel.py
Tab "Từ 4 Ảnh (Multi-view)" — v1.5: 4 ảnh (front/back/left/right) → mesh 3D thật → papercraft.

Pipeline:
  1. User chọn 4 ảnh
  2. CLIP validation (tùy chọn) — check 4 ảnh có cùng object không
  3. 4 depth maps → Poisson reconstruction → mesh watertight
  4. Unfold + Layout + emit layout_ready để MainWindow preview & export PDF
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QScrollArea, QFrame, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui  import QPixmap
from pathlib import Path


def _info_label(text: str, tooltip: str) -> QWidget:
    """Label kèm icon ⓘ — hover vào icon để xem chú thích chi tiết."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    lbl = QLabel(text)
    h.addWidget(lbl)
    info = QLabel("ⓘ")
    info.setStyleSheet(
        "color:#7A7AA0; font-size:13px; padding:0 4px;"
        "background: transparent; border: none;"
    )
    info.setCursor(Qt.WhatsThisCursor)
    info.setToolTip(tooltip)
    h.addWidget(info)
    h.addStretch()
    return w


VIEWS = ['front', 'back', 'left', 'right']
VIEW_LABELS = {
    'front': '⬆ Mặt Trước',
    'back':  '⬇ Mặt Sau',
    'left':  '⬅ Mặt Trái',
    'right': '➡ Mặt Phải',
}


class MultiViewWorker(QThread):
    progress_sig = pyqtSignal(str, int)
    done_sig     = pyqtSignal(object, object, object)   # mv_result, unfold_r, layout_r
    error_sig    = pyqtSignal(str)

    def __init__(self, image_paths, model_size, target_size_mm,
                 use_clip_validation, clip_threshold, use_depth):
        super().__init__()
        self.image_paths         = image_paths
        self.model_size          = model_size
        self.target_size_mm      = target_size_mm
        self.use_clip_validation = use_clip_validation
        self.clip_threshold      = clip_threshold
        self.use_depth           = use_depth

    def run(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from modules.depth_estimator         import estimate_depth_only
            from modules.multiview_reconstructor import reconstruct_from_multiview
            from modules.image_validator         import validate_multiview_images
            from modules.unfolder                import unfold_mesh
            from modules.layout                  import layout_panels

            # 0. CLIP validation
            if self.use_clip_validation:
                self.progress_sig.emit("CLIP validation 4 ảnh...", 3)
                val = validate_multiview_images(
                    self.image_paths, threshold=self.clip_threshold
                )
                if not val.success:
                    self.progress_sig.emit(
                        f"⚠ CLIP fail: {val.error[:80]} — bỏ qua, tiếp tục", 8)
                else:
                    self.progress_sig.emit(
                        f"CLIP: min_sim={val.min_sim:.2f} "
                        f"same_object={val.same_object}", 10)
                    if not val.same_object:
                        self.error_sig.emit(
                            f"4 ảnh có vẻ không cùng object "
                            f"(min similarity={val.min_sim:.2f} < {self.clip_threshold}).\n"
                            "Hãy kiểm tra lại 4 ảnh hoặc tắt CLIP validation rồi thử lại."
                        )
                        return

            # 1. Depth + Poisson reconstruction
            self.progress_sig.emit("Depth estimation cho 4 views + Poisson...", 15)

            def depth_fn(pil_img):
                return estimate_depth_only(pil_img, model_size=self.model_size)

            mv_result = reconstruct_from_multiview(
                image_paths        = self.image_paths,
                depth_estimator_fn = depth_fn,
                bg_threshold       = 0.08,
                scale              = 10.0,
                use_depth          = self.use_depth,
            )
            if not mv_result.success:
                self.error_sig.emit(f"Reconstruction thất bại: {mv_result.error}")
                return

            mesh = mv_result.mesh
            self.progress_sig.emit(
                f"Mesh: {mv_result.num_faces} faces, "
                f"watertight={mesh.is_watertight}", 65)

            # 2. Unfold (mesh cong → coplanar tol 25°)
            self.progress_sig.emit("Unfold mesh...", 70)
            unfold_r = unfold_mesh(
                mesh, max_faces=600, coplanar_tol_deg=25.0,
                on_progress=lambda m, p: self.progress_sig.emit(m, 70 + p//5)
            )
            if not unfold_r.success:
                self.error_sig.emit(f"Unfold thất bại: {unfold_r.error}")
                return
            self.progress_sig.emit(
                f"Unfold OK: {len(unfold_r.panels)} panels, "
                f"{unfold_r.num_islands} islands", 90)

            # 3. Layout
            self.progress_sig.emit("Layout panels...", 92)
            layout_r = layout_panels(
                unfold_r.panels,
                target_size_mm = self.target_size_mm,
                mesh           = mesh,
            )
            self.progress_sig.emit("Hoàn tất!", 100)
            self.done_sig.emit(mv_result, unfold_r, layout_r)

        except Exception as e:
            import traceback
            self.error_sig.emit(f"{e}\n{traceback.format_exc()}")


class MultiViewPanel(QWidget):
    """Tab 'Từ 4 Ảnh' — v1.5 multi-view papercraft."""

    # Tái dùng signal giống DepthPanel cho MainWindow
    layout_ready = pyqtSignal(object, object)   # layout_r, mv_result (đóng vai mesh_r)

    def __init__(self):
        super().__init__()
        self.image_paths = {v: None for v in VIEWS}
        self.previews    = {}    # view → QLabel
        self.path_labels = {}    # view → QLabel
        self.worker      = None
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
        layout.setSpacing(14)
        layout.setContentsMargins(8, 8, 16, 8)

        # ── Hướng dẫn ─────────────────────────────────────────────────────
        info = QLabel(
            "<b style='color:#F5C518;'>v1.5 Multi-View Reconstruction</b><br>"
            "<span style='color:#9090B0; font-size:11px;'>"
            "Cần đủ <b>4 ảnh</b> cùng object chụp từ 4 góc: <b>trước / sau / trái / phải</b>.<br>"
            "Object nên đặt giữa, nền sáng đồng đều. Ảnh càng giống nhau về tỉ lệ/góc nhìn → mesh càng đúng."
            "</span>"
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 6px; background:#12121E; border-radius:6px;")
        layout.addWidget(info)

        # ── 4 ô chọn ảnh (grid 2x2) ───────────────────────────────────────
        grp_imgs = QGroupBox("4 Ảnh đầu vào")
        grid = QGridLayout(grp_imgs)
        grid.setSpacing(10)

        positions = {
            'front': (0, 0),
            'back':  (0, 1),
            'left':  (1, 0),
            'right': (1, 1),
        }
        for view, (r, c) in positions.items():
            cell = self._build_image_cell(view)
            grid.addWidget(cell, r, c)
        layout.addWidget(grp_imgs)

        # ── Cài đặt ───────────────────────────────────────────────────────
        grp_cfg = QGroupBox("Cài đặt")
        cfg_l = QVBoxLayout(grp_cfg)

        # CLIP validation
        self.chk_clip = QCheckBox("Bật CLIP validation (check 4 ảnh cùng object)")
        self.chk_clip.setChecked(True)
        self.chk_clip.setToolTip(
            "Bật/tắt việc kiểm tra 4 ảnh có cùng 1 object hay không, dùng CLIP embedding.\n\n"
            "• Bật: encode 4 ảnh thành vector 512-d, tính cosine similarity giữa các cặp; "
            "nếu mọi cặp ≥ threshold thì coi là cùng object, ngược lại dừng pipeline.\n"
            "• Tắt: bỏ qua check, chạy thẳng (dùng khi 4 ảnh chụp khác bối cảnh "
            "hoặc đã chắc chắn cùng object).\n\n"
            "Lần đầu chạy sẽ tải CLIP model ~600MB từ HuggingFace."
        )
        cfg_l.addWidget(self.chk_clip)

        row_thr = QHBoxLayout()
        row_thr.addWidget(_info_label(
            "CLIP threshold:",
            "Ngưỡng cosine similarity tối thiểu giữa các cặp ảnh để coi là cùng object.\n\n"
            "• 0.50–0.65: lỏng — cho phép 4 ảnh khác góc nhiều hoặc khác bối cảnh.\n"
            "• 0.70–0.80: cân bằng (khuyến nghị 0.75) — đủ chặt nhưng vẫn chấp nhận khác góc.\n"
            "• 0.85–0.95: rất chặt — chỉ pass khi 4 ảnh gần như giống hệt nhau.\n\n"
            "Nếu pipeline báo 'không cùng object' nhưng bạn chắc chắn đúng, giảm threshold."
        ))
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.50, 0.95)
        self.spin_thr.setSingleStep(0.05)
        self.spin_thr.setValue(0.75)
        self.spin_thr.setDecimals(2)
        row_thr.addWidget(self.spin_thr)
        row_thr.addStretch()
        cfg_l.addLayout(row_thr)

        # Depth/silhouette mode
        self.chk_silhouette = QCheckBox(
            "Silhouette only (khuyến nghị — robust hơn khi depth không scale-aligned)")
        self.chk_silhouette.setChecked(True)
        self.chk_silhouette.setToolTip(
            "Chọn cách dùng depth map khi dựng mesh 3D từ 4 ảnh.\n\n"
            "• Silhouette only (bật): dùng MẶT NẠ ảnh (foreground vs background), bỏ qua giá trị depth. "
            "Robust hơn vì Depth Anything normalize 4 ảnh khác nhau → 4 depth không khớp scale. "
            "Mất chi tiết bề mặt nhưng hình dạng tổng thể đúng.\n"
            "• Tắt: dùng cả depth (Z) + silhouette → mesh có chi tiết hơn, "
            "nhưng dễ bị 'gờ' giữa các view do scale lệch."
        )
        cfg_l.addWidget(self.chk_silhouette)

        cfg_l.addWidget(_info_label(
            "Model depth size:",
            "Chọn model Depth Anything V2 chạy trên TỪNG ẢNH trong 4 ảnh.\n\n"
            "• Small (~100MB): nhanh nhất, chạy CPU OK, depth thô.\n"
            "• Base  (~400MB): cân bằng tốc độ & chất lượng, khuyến nghị GPU.\n"
            "• Large (~1.3GB): chất lượng cao nhất, cần GPU 4GB+ RAM.\n\n"
            "Lưu ý: 4 ảnh × thời gian depth từng ảnh — chọn Small nếu CPU."
        ))
        from PyQt5.QtWidgets import QComboBox
        self.combo_model = QComboBox()
        self.combo_model.addItem("🚀 Small (~100MB) — CPU OK", "small")
        self.combo_model.addItem("⚡ Base  (~400MB) — GPU tốt hơn", "base")
        self.combo_model.addItem("🎯 Large (~1.3GB) — GPU 4GB+", "large")
        cfg_l.addWidget(self.combo_model)

        cfg_l.addWidget(_info_label(
            "Kích thước panel (mm):",
            "Kích thước mục tiêu (mm) của mảnh giấy lớn nhất khi in lên A4.\n\n"
            "• 20–40 mm: nhiều mảnh nhỏ trên 1 trang.\n"
            "• 50–80 mm: kích thước phổ biến, dễ cắt dán.\n"
            "• 90–120 mm: mảnh lớn, mô hình thành phẩm to.\n\n"
            "Tất cả mảnh được scale đồng đều → giữ đúng tỉ lệ giữa các mặt."
        ))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(20, 120)
        self.spin_size.setValue(70)
        cfg_l.addWidget(self.spin_size)

        layout.addWidget(grp_cfg)

        # ── Run button ────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Tạo Papercraft từ 4 Ảnh")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #F5C518; color: #0F0F1A; font-weight: bold;
                font-size: 14px; border-radius: 8px; border: none;
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

        # ── Log ───────────────────────────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        self.log.setPlaceholderText("Log...")
        layout.addWidget(self.log)

        layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _build_image_cell(self, view: str) -> QWidget:
        """Cell chọn ảnh cho 1 view."""
        cell = QFrame()
        cell.setStyleSheet(
            "QFrame { background:#12121E; border:1px solid #2A2A40; border-radius:8px; }"
        )
        cl = QVBoxLayout(cell)
        cl.setSpacing(6)

        title = QLabel(VIEW_LABELS[view])
        title.setStyleSheet("color:#F5C518; font-weight:bold; font-size:12px; border:none;")
        title.setAlignment(Qt.AlignCenter)
        cl.addWidget(title)

        preview = QLabel("Chưa chọn")
        preview.setFixedHeight(110)
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet(
            "background:#0A0A14; border:1px dashed #2A2A40; border-radius:4px; color:#555577;"
        )
        cl.addWidget(preview)
        self.previews[view] = preview

        path_lbl = QLabel("—")
        path_lbl.setStyleSheet("color:#555577; font-size:10px; border:none;")
        path_lbl.setWordWrap(True)
        path_lbl.setAlignment(Qt.AlignCenter)
        cl.addWidget(path_lbl)
        self.path_labels[view] = path_lbl

        btn = QPushButton(f"📷 Chọn {view}")
        btn.clicked.connect(lambda _, v=view: self._pick_image(v))
        cl.addWidget(btn)

        return cell

    def _pick_image(self, view: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Chọn ảnh {VIEW_LABELS[view]}",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self.image_paths[view] = path
        self.path_labels[view].setText(Path(path).name)

        pix = QPixmap(path).scaledToHeight(100, Qt.SmoothTransformation)
        self.previews[view].setPixmap(pix)
        self.previews[view].setStyleSheet(
            "background:#0A0A14; border:1px solid #F5C518; border-radius:4px;"
        )

        self._log(f"✓ {view}: {Path(path).name}", "#4ECB71")
        self._refresh_run_state()

    def _refresh_run_state(self):
        ready = all(self.image_paths[v] for v in VIEWS)
        self.btn_run.setEnabled(ready)
        if ready:
            self._log("Đã đủ 4 ảnh. Bấm '▶ Tạo Papercraft từ 4 Ảnh' để chạy.", "#F5C518")

    def _log(self, msg, color="#9090B0"):
        self.log.append(f'<span style="color:{color};">{msg}</span>')

    def _run(self):
        if self.worker and self.worker.isRunning():
            return
        missing = [v for v in VIEWS if not self.image_paths[v]]
        if missing:
            QMessageBox.warning(
                self, "Thiếu ảnh",
                f"Cần đủ 4 ảnh. Còn thiếu: {', '.join(missing)}"
            )
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._log("Bắt đầu pipeline multi-view...", "#F5C518")

        self.worker = MultiViewWorker(
            image_paths         = dict(self.image_paths),
            model_size          = self.combo_model.currentData(),
            target_size_mm      = self.spin_size.value(),
            use_clip_validation = self.chk_clip.isChecked(),
            clip_threshold      = self.spin_thr.value(),
            use_depth           = not self.chk_silhouette.isChecked(),
        )
        self.worker.progress_sig.connect(self._on_progress)
        self.worker.done_sig.connect(self._on_done)
        self.worker.error_sig.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.progress.setValue(pct)
        self._log(f"  {pct}% {msg}")

    def _on_done(self, mv_result, unfold_r, layout_r):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        mesh = mv_result.mesh
        self._log(
            f"✅ Xong! mesh: {mv_result.num_faces} faces, "
            f"watertight={mesh.is_watertight}, "
            f"panels={len(unfold_r.panels)}, "
            f"islands={unfold_r.num_islands}, "
            f"trang={len(layout_r.pages)}",
            "#4ECB71"
        )

        # Tạo shim object có .source_name để MainWindow hiển thị info đẹp
        class _MeshShim:
            def __init__(self, mv, uf):
                self.mesh = mv.mesh
                self.final_faces = mv.num_faces
                self.source_name = "Multi-view (4 ảnh)"
                self.depth_img   = None
        shim = _MeshShim(mv_result, unfold_r)
        self.layout_ready.emit(layout_r, shim)

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._log(f"❌ {msg[:300]}", "#E74C3C")
        QMessageBox.critical(self, "Lỗi", msg[:500])
