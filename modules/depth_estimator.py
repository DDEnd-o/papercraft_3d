"""
modules/depth_estimator.py
Tái tạo mesh 3D từ 1 ảnh dùng Depth Anything V2 (AI Depth Estimation).

Pipeline:
  1 ảnh → Depth Anything V2 → Depth Map → Point Cloud → Mesh → Papercraft

Cài đặt (chạy 1 lần trên máy bạn):
    pip install transformers torch torchvision pillow opencv-python open3d

Model tự động tải lần đầu chạy (~100MB), lưu vào cache HuggingFace.
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


# ── Model options theo GPU/CPU ────────────────────────────────────────────────
# Small  : ~100MB  - CPU chạy được (~5-10s)
# Base   : ~400MB  - cần GPU hoặc chờ lâu
# Large  : ~1.3GB  - cần GPU NVIDIA 4GB+
DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",   # khuyến nghị
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


@dataclass
class DepthResult:
    success:    bool
    mesh:       object = None        # trimesh.Trimesh
    depth_map:  object = None        # np.ndarray (H,W) normalized 0-1
    depth_img:  object = None        # PIL.Image depth visualization
    orig_image: object = None        # PIL.Image ảnh gốc
    num_faces:  int    = 0
    error:      str    = ""

    def summary(self):
        if not self.success:
            return f"  ❌ {self.error}"
        return (f"  ✅ Depth estimation thành công\n"
                f"  Mesh faces : {self.num_faces:,}\n"
                f"  Depth map  : {self.depth_map.shape if self.depth_map is not None else 'N/A'}")


def estimate_depth_and_build_mesh(
    image_path: str | Path,
    model_size: str = "small",
    target_faces: int = 200,
    depth_scale: float = 3.0,
    remove_background: bool = True,
    on_progress: Optional[Callable] = None,
) -> DepthResult:
    """
    Từ 1 ảnh → Depth Map → Mesh 3D.

    Args:
        image_path       : đường dẫn ảnh (.jpg/.png)
        model_size       : 'small' / 'base' / 'large'
        target_faces     : số faces mesh output
        depth_scale      : độ sâu tương đối (càng lớn mesh càng dày)
        remove_background: loại bỏ vùng nền phẳng
        on_progress      : callback(msg, pct)

    Returns:
        DepthResult
    """
    def prog(msg, pct=0):
        print(f"  [{pct:3d}%] {msg}")
        if on_progress:
            on_progress(msg, pct)

    result = DepthResult(success=False)

    try:
        # ── Bước 1: Load ảnh ──────────────────────────────────────────────
        prog("Load ảnh...", 5)
        if not PIL_OK:
            result.error = "Pillow chưa cài: pip install Pillow"
            return result

        image_path = Path(image_path)
        if not image_path.exists():
            result.error = f"Không tìm thấy ảnh: {image_path}"
            return result

        orig = Image.open(image_path).convert("RGB")
        result.orig_image = orig
        prog(f"Ảnh: {orig.size[0]}×{orig.size[1]}px", 10)

        # ── Bước 2: Depth Estimation ──────────────────────────────────────
        prog(f"Chạy Depth Anything V2 ({model_size})...", 15)
        prog("(Lần đầu sẽ tải model ~100MB, chờ một chút...)", 16)

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            result.error = "transformers chưa cài: pip install transformers"
            return result

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prog(f"Device: {device}", 18)

        model_id = DEPTH_MODELS.get(model_size, DEPTH_MODELS["small"])
        depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1,
        )
        prog("Model loaded!", 40)

        # Resize ảnh nếu quá lớn (tiết kiệm RAM)
        max_dim = 518   # Depth Anything V2 optimal
        w, h    = orig.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            orig_resized = orig.resize((int(w*scale), int(h*scale)),
                                        Image.LANCZOS)
        else:
            orig_resized = orig

        prog("Chạy inference...", 45)
        depth_output = depth_pipe(orig_resized)
        depth_pil    = depth_output["depth"]           # PIL Image (grayscale)
        depth_np     = np.array(depth_pil).astype(float)

        # Normalize 0-1
        dmin, dmax = depth_np.min(), depth_np.max()
        if dmax - dmin > 1e-6:
            depth_np = (depth_np - dmin) / (dmax - dmin)

        result.depth_map = depth_np
        result.depth_img = depth_pil
        prog("Depth estimation xong!", 65)

        # ── Bước 3: Depth Map → Point Cloud ──────────────────────────────
        prog("Chuyển depth map → point cloud...", 68)
        mesh = _depth_to_mesh(
            depth_np,
            orig_resized,
            depth_scale=depth_scale,
            remove_background=remove_background,
            target_faces=target_faces,
        )
        prog("Point cloud → Mesh xong!", 90)

        result.mesh      = mesh
        result.num_faces = len(mesh.faces)
        result.success   = True
        prog("✅ Hoàn tất!", 100)

    except Exception as e:
        import traceback
        result.error = f"{e}"
        traceback.print_exc()

    return result


def _depth_to_mesh(depth_np: np.ndarray,
                   color_img: "Image",
                   depth_scale: float = 3.0,
                   remove_background: bool = True,
                   target_faces: int = 200) -> "trimesh.Trimesh":
    """
    Chuyển depth map 2D → mesh 3D bằng cách tạo lưới điểm 3D.

    Mỗi pixel (x, y) với depth d → điểm 3D (x_norm, y_norm, d * depth_scale)
    Sau đó nối thành mesh tam giác (grid triangulation).
    """
    import trimesh
    import numpy as np

    H, W = depth_np.shape

    # Downsample để giảm số điểm
    step = max(1, min(H, W) // 80)   # tối đa ~80×80 grid
    rows = np.arange(0, H, step)
    cols = np.arange(0, W, step)
    nr, nc = len(rows), len(cols)

    # Tạo lưới tọa độ 3D
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    xs = cc / W * 10.0 - 5.0                       # [-5, 5]
    ys = -(rr / H * 10.0 - 5.0)                    # [-5, 5] lật trục Y
    zs = depth_np[rr, cc] * depth_scale             # [0, depth_scale]

    # Loại background (depth gần 0 = rất xa = background)
    if remove_background:
        bg_threshold = 0.08
        mask = depth_np[rr, cc] > bg_threshold
    else:
        mask = np.ones((nr, nc), dtype=bool)

    # Tạo vertices
    vertices = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

    # Tạo faces (2 tam giác mỗi ô vuông)
    faces = []
    idx = np.arange(nr * nc).reshape(nr, nc)

    for i in range(nr - 1):
        for j in range(nc - 1):
            # Bỏ qua ô nếu bất kỳ đỉnh nào là background
            if not (mask[i,j] and mask[i+1,j] and
                    mask[i,j+1] and mask[i+1,j+1]):
                continue
            a = idx[i,   j]
            b = idx[i+1, j]
            c = idx[i,   j+1]
            d = idx[i+1, j+1]
            faces.append([a, b, c])
            faces.append([b, d, c])

    if not faces:
        # Fallback: không remove background
        for i in range(nr - 1):
            for j in range(nc - 1):
                a = idx[i,   j];   b = idx[i+1, j]
                c = idx[i,   j+1]; d = idx[i+1, j+1]
                faces.append([a, b, c])
                faces.append([b, d, c])

    faces    = np.array(faces, dtype=np.int32)
    mesh     = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    # Gán màu từ ảnh gốc
    try:
        color_arr = np.array(color_img.resize((W, H), Image.LANCZOS))
        colors_flat = color_arr[rr, cc].reshape(-1, 3)
        alpha = np.full((len(colors_flat), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack([colors_flat, alpha])
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, vertex_colors=vertex_colors)
    except Exception:
        pass  # Không có màu cũng không sao

    # Simplify
    if len(mesh.faces) > target_faces:
        ratio = max(0.01, min(0.99, 1.0 - target_faces / len(mesh.faces)))
        try:
            mesh = mesh.simplify_quadric_decimation(ratio)
        except Exception:
            pass

    mesh.merge_vertices()
    return mesh


def save_depth_visualization(depth_result: DepthResult,
                              output_dir: str | Path = "output") -> Path:
    """Lưu depth map ra ảnh PNG để xem."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if depth_result.depth_img is not None:
        out = output_dir / "depth_map.png"
        depth_result.depth_img.save(str(out))
        print(f"  Depth map: {out}")
        return out
    return None


def check_dependencies() -> dict:
    """Kiểm tra tất cả dependencies cần thiết."""
    status = {}
    libs = [
        ("torch",        "pip install torch"),
        ("transformers", "pip install transformers"),
        ("PIL",          "pip install Pillow"),
        ("cv2",          "pip install opencv-python"),
        ("trimesh",      "pip install trimesh"),
        ("open3d",       "pip install open3d"),
    ]
    for name, install_cmd in libs:
        try:
            __import__(name)
            status[name] = ("✅", "OK")
        except ImportError:
            status[name] = ("❌", install_cmd)
    return status
