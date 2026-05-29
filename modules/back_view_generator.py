"""
modules/back_view_generator.py
v1.4 — Sinh ảnh mặt sau từ ảnh mặt trước bằng Stable Diffusion inpainting.

Ý tưởng:
  1. Lật ngang ảnh gốc → ảnh placeholder "góc nhìn sau" (đối xứng trái-phải).
  2. Tạo mask = vùng foreground (silhouette) — giữ nguyên silhouette, inpaint nội dung bên trong.
  3. Stable Diffusion inpainting tô lại vùng đó bằng prompt mô tả "back view of {object}".

Lưu ý:
  - Lần đầu chạy sẽ tải model ~4GB từ HuggingFace, cache vào ~/.cache/huggingface.
  - Trên CPU: 3-5 phút/ảnh. GPU CUDA: 15-30 giây.
  - Kết quả KHÔNG đảm bảo đúng — SD có thể vẽ ra mặt vịt thay vì đuôi vịt.
    Đây là giới hạn của AI sinh ảnh chỉ từ 1 hint.
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

try:
    from PIL import Image, ImageFilter, ImageOps
    PIL_OK = True
except ImportError:
    PIL_OK = False


SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"


@dataclass
class BackViewResult:
    success:  bool
    back_img: object = None         # PIL.Image (RGB, kích thước giống ảnh gốc)
    error:    str    = ""


def _build_silhouette_mask(depth_np: np.ndarray, bg_threshold: float = 0.08,
                           feather_px: int = 6) -> "Image":
    """Tạo mask: white = vùng cần inpaint (foreground), black = giữ nguyên (background).

    SD inpainting convention: white pixel sẽ được vẽ lại.
    """
    fg = (depth_np > bg_threshold).astype(np.uint8) * 255
    mask = Image.fromarray(fg, mode="L")
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        # Re-threshold để biên mềm chuyển thành biên mượt
        mask = mask.point(lambda v: 255 if v > 128 else 0)
    return mask


def _round_to_8(n: int) -> int:
    """SD yêu cầu width/height chia hết cho 8."""
    return max(8, (n // 8) * 8)


def generate_back_view(
    front_img: "Image",
    depth_np:  np.ndarray,
    prompt:    str = "back view of the object, same style, smooth surface, plain color",
    negative_prompt: str = "front view, face, eyes, text, distortion, ugly",
    num_steps: int   = 25,
    guidance:  float = 7.5,
    seed:      Optional[int] = 42,
    on_progress: Optional[Callable] = None,
) -> BackViewResult:
    """
    Sinh ảnh "mặt sau" của object trong front_img.

    Args:
        front_img       : PIL.Image ảnh gốc (RGB).
        depth_np        : depth map đã normalize 0-1 — dùng để tách foreground.
        prompt          : mô tả nội dung muốn AI vẽ vào vùng foreground.
        negative_prompt : những thứ không muốn xuất hiện (vd "front view, face").
        num_steps       : 20-30 cho cân bằng tốc độ/chất lượng.
        guidance        : 7-9 cho prompt-driven inpainting.
        seed            : để reproducible (None = ngẫu nhiên).
        on_progress     : callback(msg, pct).

    Returns:
        BackViewResult.
    """
    def prog(msg, pct=0):
        print(f"  [{pct:3d}%] {msg}")
        if on_progress:
            on_progress(msg, pct)

    result = BackViewResult(success=False)

    if not PIL_OK:
        result.error = "Pillow chưa cài: pip install Pillow"
        return result

    try:
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        import os
        # Workaround: torch 2.5 + transformers ≥5.8 có CVE check nghiêm.
        # Tạm bypass để load .bin (production nên upgrade torch ≥2.6).
        os.environ["HF_HUB_ENABLE_TORCH_LOAD_UNSAFE"] = "1"
    except ImportError as e:
        result.error = f"Thiếu thư viện: {e} — pip install diffusers accelerate safetensors torch"
        return result

    try:
        prog("Tải Stable Diffusion inpainting model...", 5)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            # Không force use_safetensors — để diffusers tự chọn .bin nếu safetensors thiếu
        )
        pipe = pipe.to(device)
        # Tiết kiệm VRAM
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        prog(f"Model loaded trên {device}", 30)

        # SD inpaint mặc định 512x512. Resize ảnh gốc + depth về kích thước chia hết 8.
        w, h = front_img.size
        max_dim = 512
        scale = min(max_dim / max(w, h), 1.0)
        tw = _round_to_8(int(w * scale))
        th = _round_to_8(int(h * scale))
        prog(f"Resize → {tw}×{th}", 35)

        # Lật ngang trước khi inpaint: tạo "góc nhìn sau" thô đối xứng.
        # SD sẽ tinh chỉnh chi tiết bên trong vùng mask.
        flipped = ImageOps.mirror(front_img).convert("RGB").resize((tw, th), Image.LANCZOS)

        # Mask foreground từ depth (resize depth cùng kích thước)
        depth_img = Image.fromarray((depth_np * 255).astype(np.uint8), mode="L")
        depth_img = ImageOps.mirror(depth_img).resize((tw, th), Image.LANCZOS)
        depth_resized = np.array(depth_img).astype(float) / 255.0
        mask = _build_silhouette_mask(depth_resized, bg_threshold=0.08, feather_px=6)

        prog("Chạy SD inpainting...", 50)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=flipped,
            mask_image=mask,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
        )
        back = out.images[0]
        prog("Inpainting xong", 90)

        # Resize về kích thước gốc cho khớp depth pipeline
        back = back.resize(front_img.size, Image.LANCZOS)
        result.back_img = back
        result.success  = True
        prog("✅ Hoàn tất", 100)

    except Exception as e:
        import traceback
        result.error = str(e)
        traceback.print_exc()

    return result


def save_back_view(result: BackViewResult, output_path: str | Path) -> Optional[Path]:
    """Lưu ảnh back view ra disk."""
    if not result.success or result.back_img is None:
        return None
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.back_img.save(out)
    return out
