# 🎨 Papercraft 3D App

**Papercraft 3D** là một ứng dụng máy tính (Desktop App) giúp người dùng tự động chuyển đổi các mô hình 3D (Object/STL) hoặc thậm chí là hình ảnh 2D thông thường thành các bản vẽ thiết kế cắt dán giấy (Papercraft) có thể in ra dưới định dạng PDF.

---

## ✨ Tính Năng Nổi Bật

1. **Nhập Mô Hình Đa Dạng**: 
   - Hỗ trợ nhập các file 3D tiêu chuẩn như `.obj`, `.stl`, `.glb`, `.ply`, v.v.
   - Cung cấp sẵn các khối hình học cơ bản (Primitives) như Khối lập phương, Hình cầu, Hình trụ,...
2. **AI Image-to-Papercraft (Từ Ảnh sang Papercraft)**: 
   - Tích hợp mô hình AI **Depth Anything V2** (HuggingFace) để phân tích chiều sâu của một bức ảnh 2D bất kỳ, tái tạo thành mô hình 3D và tự động trải phẳng.
3. **Thuật Toán Unfold & Layout Tự Động**: 
   - Trải phẳng bề mặt 3D (Unfold) thành các mảnh 2D tối ưu, giảm thiểu sự rời rạc.
   - Sắp xếp các mảnh ghép (Layout) lên kích thước giấy A4 một cách tiết kiệm không gian nhất.
4. **Trực Quan & Dễ Sử Dụng**: 
   - Giao diện đồ họa (GUI) hiện đại, trực quan được xây dựng bằng PyQt5.
   - Xem trước (Preview) bản vẽ 2D trực tiếp trên màn hình trước khi xuất file.
5. **Xuất Bản Vẽ PDF Chuyên Nghiệp**: 
   - Xuất file định dạng PDF chuẩn kích thước A4.
   - Tự động đánh số thứ tự (ID) các mặt và vẽ sẵn mép dán (Glue tabs), đường gập (Fold lines).

---

## 📂 Cấu Trúc Dự Án

```text
papercraft_app/
│
├── main.py                    # File khởi chạy chính của ứng dụng
├── requirements.txt           # Danh sách các thư viện Python cần thiết
│
├── gui/                       # Thư mục chứa giao diện người dùng (Frontend)
│   ├── main_window.py         # Cửa sổ ứng dụng chính (Sidebar, Viewport, Tabs)
│   └── depth_panel.py         # Giao diện cho tính năng AI (Từ Ảnh -> Papercraft)
│
├── modules/                   # Thư mục chứa logic lõi (Backend Pipeline)
│   ├── mesh_loader.py         # Xử lý đọc file 3D và tạo các khối cơ bản
│   ├── depth_estimator.py     # Xử lý AI ước lượng chiều sâu từ ảnh 2D
│   ├── unfolder.py            # Thuật toán trải phẳng (Unfold) lưới 3D thành 2D
│   ├── layout.py              # Thuật toán sắp xếp (Pack) các panel 2D lên giấy A4
│   └── pdf_exporter.py        # Xuất dữ liệu đồ họa 2D ra file PDF (ReportLab)
│
└── output/                    # (Thư mục) Mặc định lưu các file PDF xuất ra
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Yêu Cầu Hệ Thống
- Python 3.10 trở lên.
- Có kết nối Internet (để tải model AI ở lần chạy đầu tiên nếu dùng tính năng xử lý từ ảnh).
- (Khuyến nghị) Card đồ họa rời nếu muốn xử lý AI Depth (model Large) nhanh hơn.

### 2. Cài Đặt Thư Viện

Mở terminal/command prompt tại thư mục dự án và chạy lệnh sau để cài đặt các thư viện cơ bản:

```bash
pip install -r requirements.txt
```

Nếu bạn muốn sử dụng tính năng **Từ Ảnh (AI)**, bạn cần cài đặt thêm các thư viện xử lý máy học (Deep Learning):

```bash
pip install transformers torch torchvision opencv-python open3d
```
*(Nếu bạn có GPU NVIDIA, hãy cài đặt `torch` bản có CUDA để tăng tốc AI).*

### 3. Khởi Chạy Ứng Dụng

Chạy lệnh sau tại thư mục dự án:

```bash
python main.py
```

---

## 🛠 Hướng Dẫn Sử Dụng Cơ Bản

1. Mở ứng dụng.
2. Chọn nguồn đầu vào ở thanh Sidebar bên trái:
   - **Primitive Shape / Import file .obj/.stl**: Nếu bạn đã có mô hình 3D.
   - **Tab 📷 Từ Ảnh (AI)**: Nếu bạn chỉ có một bức ảnh 2D.
3. Điều chỉnh **Cài Đặt**:
   - *Số mặt (faces)*: Độ chi tiết của mô hình (càng cao càng đẹp nhưng cắt dán càng lâu).
   - *Kích thước panel (mm)*: Kích thước mong muốn của mảnh giấy.
4. Bấm **▶ Tạo Papercraft** và đợi ứng dụng xử lý.
5. Xem trước bản vẽ ở phần giữa màn hình. Bấm **⬇ Xuất PDF** để lưu file đi in.

---

**Chúc bạn có những mô hình thủ công bằng giấy (Papercraft) tuyệt vời! ✂️📜**
