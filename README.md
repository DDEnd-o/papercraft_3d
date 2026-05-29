# 🎨 Papercraft 3D App

**Papercraft 3D** là ứng dụng máy tính giúp tự động chuyển đổi mô hình 3D (OBJ/STL/GLB/PLY) hoặc ảnh 2D thành bản vẽ cắt dán giấy (papercraft) xuất ra PDF khổ A4.

---

## ✨ Tính năng

### 1. Nhập mô hình đa dạng
- File 3D tiêu chuẩn: `.obj`, `.stl`, `.glb`, `.gltf`, `.ply`, `.off`
- 8 primitive shapes: Cube, Sphere, Cylinder, Cone, Torus, Tetrahedron, Capsule, Annulus
- **AI từ ảnh** (Depth Anything V2): 1 ảnh 2D → mesh 3D → papercraft

### 2. Thuật toán Unfold & Layout (v1.2)
- **Gộp đồng phẳng** trước khi unfold: cube 12 tri → 6 quad, giảm số mảnh đáng kể
- **MST/BFS unfold** trên facet graph với overlap check toàn cục
- **FFDH packing** (First-Fit Decreasing Height) cho A4
- Retry với gap nới rộng nếu phát hiện chồng nhau giữa các island

### 3. AI Pipeline cho ảnh đơn (v1.3)
- **Solidify mesh**: front layer + back layer (z=0) + side walls → khối kín watertight
- **Coplanar tolerance 25°** cho mặt cong (depth mesh) — gom ít mảnh hơn
- **Cảnh báo chất lượng**: popup nếu mesh không phù hợp papercraft (>60 panels hoặc >5 islands)

### 4. AI back view (v1.4 — experimental)
- **Stable Diffusion inpainting** sinh ảnh mặt sau từ ảnh mặt trước
- Depth lần 2 trên ảnh back → mesh có khối ở cả 2 mặt
- ⚠️ Kết quả không đảm bảo đúng (AI có thể hallucinate mặt trước lần 2)

### 5. Multi-view reconstruction (v1.5 — experimental)
- **4 ảnh** (front/back/left/right) → 4 depth maps → Poisson reconstruction → mesh 3D thật
- **CLIP validation** kiểm tra 4 ảnh có cùng object không (threshold cosine sim 0.75)
- Silhouette-only mode (use_depth=False) robust hơn khi depth không scale-aligned

### 6. Xuất PDF chuyên nghiệp
- Khổ A4, đường cắt (đen), đường gấp (xanh đứt), tab dán (xám có dấu 'x')
- Đánh số mảnh, calibration ruler 100mm, trang hướng dẫn lắp ráp

---

## 📂 Cấu trúc dự án

```text
papercraft_app/
│
├── main.py                       # Entry point: GUI hoặc CLI
├── requirements.txt              # Python dependencies
├── README.md                     # File này
├── V14_README.md                 # Notes cho v1.4 (AI back view)
├── V15_DESIGN.md                 # Design doc cho v1.5 (multi-view)
│
├── gui/
│   ├── main_window.py            # Cửa sổ chính (sidebar + preview + tabs)
│   └── depth_panel.py            # Tab "Từ Ảnh (AI)"
│
├── modules/
│   ├── mesh_loader.py            # Load file 3D + tạo primitive
│   ├── unfolder.py               # MST unfold + coplanar facet merging (v1.2)
│   ├── layout.py                 # FFDH packing + overlap validation
│   ├── pdf_exporter.py           # Xuất PDF (ReportLab) với n-gon panels
│   ├── depth_estimator.py        # Depth Anything V2 + solidify mesh (v1.3+)
│   ├── back_view_generator.py    # v1.4: SD inpainting sinh ảnh mặt sau
│   ├── multiview_reconstructor.py # v1.5: Poisson từ 4 depth maps
│   └── image_validator.py        # v1.5: CLIP check 4 ảnh cùng object
│
├── tests/
│   ├── test_pipeline.py          # 14 pytest (8 primitives + STL + cube quad check)
│   ├── conftest.py
│   └── fixtures/
│       ├── generate_sample_stl.py
│       └── sample_house.stl
│
├── scripts/                      # Demo & test scripts
│   ├── test_duck.py              # v1.3 trên duck-toy.png
│   ├── test_duck_v14.py          # v1.4 SD inpainting
│   ├── test_duck_multiview.py    # v1.5 multi-view
│   └── generate_multiview_images.py  # Render 4 views từ duck mesh
│
├── img/
│   ├── duck-toy.png              # Ảnh sample cho v1.3/v1.4
│   └── multiview_duck/           # 4 ảnh render cho v1.5
│
└── output/                       # PDF + intermediate files
```

---

## 🚀 Cài đặt & Chạy

### 1. Yêu cầu hệ thống
- Python 3.10+
- Internet (để tải AI model lần đầu)
- **Torch ≥ 2.6** nếu dùng v1.4 (SD inpainting) — bắt buộc do CVE-2025-32434
- (Khuyến nghị) GPU NVIDIA 4GB+ cho v1.4 (SD trên CPU mất ~5 phút/ảnh)

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

`requirements.txt` đã bao gồm tất cả: trimesh, numpy, scipy, Pillow, reportlab, PyQt5, shapely, pytest, diffusers, accelerate, safetensors, open3d.

Nếu cần dùng tính năng AI (depth/SD/CLIP), cài thêm:

```bash
pip install transformers torch torchvision opencv-python
```

*(Có GPU NVIDIA, cài torch bản CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`)*

### 3. Chạy

**GUI:**
```bash
python main.py
```

**CLI** (primitive):
```bash
python main.py --shape cube --faces 120 --output output/cube.pdf
```

**CLI** (file 3D):
```bash
python main.py --file model.stl --faces 200 --size 65
```

## 🛠 Hướng dẫn sử dụng

### GUI cơ bản
1. Chọn nguồn ở sidebar trái:
   - **Primitive Shape** (cube/sphere/cylinder/...) — khuyến nghị cho người mới
   - **Import file .obj/.stl** — nếu có sẵn model 3D khối kín
   - **Tab 📷 Từ Ảnh (AI)** — sinh papercraft từ 1 ảnh 2D
2. Điều chỉnh **Số mặt** (20-800) và **Kích thước panel** (20-150mm)
3. Bấm **▶ Tạo Papercraft**, đợi xử lý (~10-30s)
4. Xem preview ở giữa, bấm **⬇ Xuất PDF**

### Scripts experimental

```bash
# v1.3 — duck từ 1 ảnh, mặt sau phẳng (phù điêu)
python scripts/test_duck.py

# v1.4 — duck với AI back view (cần torch ≥ 2.6, chậm)
python scripts/test_duck_v14.py

# v1.5 — duck từ 4 ảnh multi-view (chất lượng cao nhất)
python scripts/generate_multiview_images.py    # tạo 4 ảnh trước
python scripts/test_duck_multiview.py          # rồi chạy pipeline
```

---

## 📊 So sánh các version (test với duck)

| Version | Input | Panels | Islands | Watertight | Use case |
|---|---|---|---|---|---|
| **v1.2** | Primitives/STL khối kín | 6 (cube) / ≥7 (house) | 1-3 | ✅ True | **Production tốt nhất** |
| **v1.3** | 1 ảnh đơn | 33 | 3 | ✅ True | Phù điêu (mặt sau phẳng) |
| v1.4 | 1 ảnh + AI back | 17 | 6 | ❌ False | Demo AI (kết quả khó đoán) |
| **v1.5** | 4 ảnh thật | 63 | 10 | ⚠️ Euler=0 | Khối 3D thật (cần 4 ảnh chuẩn) |

---

## ⚠️ Lưu ý

- **Calibration**: trang đầu PDF có thước 100mm — sau khi in, đo lại đoạn này. Nếu không bằng 100mm, chỉnh máy in về "Actual Size / 100%".
- **Giấy**: dùng giấy 120-160gsm cho cứng cáp. Giấy thường (80gsm) sẽ mềm.
- **AI features**: lần đầu chạy sẽ tải model (~100MB cho depth, ~4GB cho SD inpainting). Lưu trong cache HuggingFace.

---

**Chúc bạn có những mô hình thủ công bằng giấy tuyệt vời! ✂️📜**
