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
    solidify: bool = True,
    back_depth_np: Optional[np.ndarray] = None,
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
        solidify         : (v1.3) đóng mặt sau + side walls → mesh watertight
        back_depth_np    : (v1.4) depth map mặt sau đã align (cùng (H, W), cùng hệ XY)
                           với front depth. Nếu None → fallback v1.3 (back phẳng).
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

        # v1.4: nếu back_depth_np không khớp shape với depth_np đã resize,
        # interpolate về shape của depth_np.
        back_for_mesh = back_depth_np
        if back_for_mesh is not None and back_for_mesh.shape != depth_np.shape:
            try:
                from PIL import Image as _PILImg
                back_img = _PILImg.fromarray(
                    (np.clip(back_for_mesh, 0.0, 1.0) * 255).astype(np.uint8),
                    mode="L",
                ).resize((depth_np.shape[1], depth_np.shape[0]), _PILImg.LANCZOS)
                back_for_mesh = np.array(back_img).astype(float) / 255.0
            except Exception:
                back_for_mesh = None

        mesh = _depth_to_mesh(
            depth_np,
            orig_resized,
            depth_scale=depth_scale,
            remove_background=remove_background,
            target_faces=target_faces,
            solidify=solidify,
            back_depth_np=back_for_mesh,
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
                   target_faces: int = 200,
                   solidify: bool = True,
                   back_depth_np: Optional[np.ndarray] = None) -> object:
    """
    Chuyển depth map 2D → mesh 3D.

    v1.3 — Solidify:
        Trước kia chỉ tạo lưới (x, y, z=depth) — mesh hở, dán xong ra "tấm cong"
        chứ không phải khối 3D. Giờ ta:
          1. Layer trước  (front) : z = depth_scale * d(x,y)
          2. Layer sau    (back)  : z = 0 — phẳng (đáy nằm trên giấy lưng)
          3. Side walls          : viền foreground (mask boundary) nối front↔back
        ⇒ mesh thành khối kín (watertight) để unfold ra papercraft thật.

    v1.4 — Back depth từ AI inpainting:
        Nếu truyền `back_depth_np` (đã align về cùng (H,W) và cùng hệ toạ độ XY
        với depth_np), back layer dùng z = -back_depth_np * depth_scale,
        cho ra khối 3D có khối lượng thật ở cả 2 mặt (không còn phẳng tịt).
    """
    import trimesh
    import numpy as np

    H, W = depth_np.shape

    # Downsample để giảm số điểm
    step = max(1, min(H, W) // 80)
    rows = np.arange(0, H, step)
    cols = np.arange(0, W, step)
    nr, nc = len(rows), len(cols)

    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    xs = cc / W * 10.0 - 5.0
    ys = -(rr / H * 10.0 - 5.0)
    zs = depth_np[rr, cc] * depth_scale

    if remove_background:
        bg_threshold = 0.08
        mask = depth_np[rr, cc] > bg_threshold
    else:
        mask = np.ones((nr, nc), dtype=bool)

    # ── Front layer ──────────────────────────────────────────────────────
    front_verts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    front_idx   = np.arange(nr * nc).reshape(nr, nc)

    front_faces = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            if not (mask[i, j] and mask[i+1, j] and
                    mask[i, j+1] and mask[i+1, j+1]):
                continue
            a = front_idx[i,   j]
            b = front_idx[i+1, j]
            c = front_idx[i,   j+1]
            d = front_idx[i+1, j+1]
            front_faces.append([a, b, c])
            front_faces.append([b, d, c])

    # Fallback: không remove background nếu sạch không còn face
    if not front_faces:
        mask = np.ones((nr, nc), dtype=bool)
        for i in range(nr - 1):
            for j in range(nc - 1):
                a = front_idx[i,   j];   b = front_idx[i+1, j]
                c = front_idx[i,   j+1]; d = front_idx[i+1, j+1]
                front_faces.append([a, b, c])
                front_faces.append([b, d, c])

    front_faces = np.array(front_faces, dtype=np.int32)

    if not solidify:
        mesh = trimesh.Trimesh(vertices=front_verts, faces=front_faces, process=True)
    else:
        # ── Back layer ────────────────────────────────────────────────────
        # v1.4: nếu có back_depth_np → back có khối nhô ra phía -z (depth nghịch).
        # Không có → fallback v1.3: phẳng z=0.
        if back_depth_np is not None and back_depth_np.shape == depth_np.shape:
            zs_back = -back_depth_np[rr, cc] * depth_scale
        else:
            zs_back = np.zeros_like(zs)
        back_verts  = np.stack([xs, ys, zs_back], axis=-1).reshape(-1, 3)
        back_offset = len(front_verts)

        # Back faces: winding ngược để normal hướng ra ngoài (về -z)
        back_faces = []
        for i in range(nr - 1):
            for j in range(nc - 1):
                if not (mask[i, j] and mask[i+1, j] and
                        mask[i, j+1] and mask[i+1, j+1]):
                    continue
                a = front_idx[i,   j]   + back_offset
                b = front_idx[i+1, j]   + back_offset
                c = front_idx[i,   j+1] + back_offset
                d = front_idx[i+1, j+1] + back_offset
                back_faces.append([a, c, b])
                back_faces.append([b, c, d])
        back_faces = np.array(back_faces, dtype=np.int32)

        # ── Side walls: dò boundary edges của front mesh ─────────────────
        # Cạnh chỉ thuộc đúng 1 tam giác = nằm trên silhouette → cần wall đi xuống back.
        # Lưu cạnh có hướng để giữ winding cho normal hướng ra ngoài.
        from collections import defaultdict
        edge_count = defaultdict(int)
        edge_dir = {}
        for f in front_faces:
            for k in range(3):
                a, b = int(f[k]), int(f[(k+1) % 3])
                key = (min(a, b), max(a, b))
                edge_count[key] += 1
                # Lưu hướng đầu tiên gặp; khi count=1 thì hướng này là CCW của face
                if key not in edge_dir:
                    edge_dir[key] = (a, b)

        side_faces = []
        for key, count in edge_count.items():
            if count != 1:
                continue
            a, b = edge_dir[key]  # boundary CCW từ front
            ab = a + back_offset
            bb = b + back_offset
            # Quad (a, b, bb, ab) — wall đi xuống mặt back, winding sao cho normal ra ngoài
            side_faces.append([a, b, bb])
            side_faces.append([a, bb, ab])

        if side_faces:
            side_faces = np.array(side_faces, dtype=np.int32)
            all_verts = np.vstack([front_verts, back_verts])
            all_faces = np.vstack([front_faces, back_faces, side_faces])
        else:
            all_verts = np.vstack([front_verts, back_verts])
            all_faces = np.vstack([front_faces, back_faces])

        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
        # process=True đã merge duplicate verts; thêm fix_normals để wall xoay đúng chiều
        try:
            mesh.fix_normals()
        except Exception:
            pass

    # Gán màu từ ảnh gốc (chỉ áp dụng cho front layer)
    try:
        color_arr = np.array(color_img.resize((W, H), Image.LANCZOS))
        colors_flat = color_arr[rr, cc].reshape(-1, 3)
        if solidify:
            colors_back = (colors_flat * 0.5).astype(np.uint8)
            colors_all  = np.vstack([colors_flat, colors_back])
        else:
            colors_all = colors_flat
        alpha = np.full((len(colors_all), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack([colors_all, alpha])
        # Mesh sau process có thể đã đổi số verts; chỉ gán nếu khớp
        if len(vertex_colors) == len(mesh.vertices):
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=vertex_colors)
    except Exception:
        pass

    # Simplify
    if len(mesh.faces) > target_faces:
        ratio = max(0.01, min(0.99, 1.0 - target_faces / len(mesh.faces)))
        try:
            mesh = mesh.simplify_quadric_decimation(ratio)
        except Exception:
            pass

    mesh.merge_vertices()
    return mesh


def estimate_depth_only(pil_img: "Image", model_size: str = "small") -> np.ndarray:
    """v1.4 helper — chạy depth-anything trên 1 PIL.Image, trả depth_np normalize 0-1.

    Khác `estimate_depth_and_build_mesh` ở chỗ: không cần file path, không build mesh,
    dùng cho ảnh back-view do SD sinh ra trong RAM.
    """
    from transformers import pipeline as hf_pipeline
    import torch

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = DEPTH_MODELS.get(model_size, DEPTH_MODELS["small"])
    pipe = hf_pipeline(
        task="depth-estimation",
        model=model_id,
        device=0 if device == "cuda" else -1,
    )

    max_dim = 518
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img_resized = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        img_resized = pil_img

    out = pipe(img_resized.convert("RGB"))
    depth_pil = out["depth"]
    depth_np  = np.array(depth_pil).astype(float)
    dmin, dmax = depth_np.min(), depth_np.max()
    if dmax - dmin > 1e-6:
        depth_np = (depth_np - dmin) / (dmax - dmin)
    return depth_np


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
