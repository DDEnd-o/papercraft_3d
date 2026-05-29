"""
modules/image_validator.py
v1.5 — Validate 4 ảnh multi-view có cùng object hay không bằng CLIP embedding similarity.

Dùng `openai/clip-vit-base-patch32` từ transformers (đã cài). Lần đầu chạy tải ~600MB.

Logic:
  1. Encode 4 ảnh thành 512-d embedding
  2. Tính pairwise cosine similarity giữa các cặp
  3. Nếu mọi cặp đều > threshold (mặc định 0.75) → cùng object
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional  # noqa: F401

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class ValidationResult:
    success:      bool
    same_object:  bool = False
    min_sim:      float = 0.0
    max_sim:      float = 0.0
    avg_sim:      float = 0.0
    pairs:        list = field(default_factory=list)  # [(view_a, view_b, similarity)]
    error:        str  = ""


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-10)
    nb = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(na, nb))


def validate_multiview_images(image_paths: dict,
                              threshold: float = 0.75,
                              model_name: str = CLIP_MODEL) -> ValidationResult:
    """Validate 4 ảnh có cùng object không.

    Args:
        image_paths : dict {'front': path, 'back': path, 'left': path, 'right': path}
        threshold   : min pairwise cosine similarity (0.75 = relax, 0.85 = strict)
        model_name  : CLIP model HuggingFace id

    Returns:
        ValidationResult
    """
    result = ValidationResult(success=False)

    if not PIL_OK:
        result.error = "Pillow chưa cài"
        return result

    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except ImportError as e:
        result.error = f"transformers chưa cài: {e}"
        return result

    try:
        # Load images
        views = ['front', 'back', 'left', 'right']
        images = {}
        for v in views:
            if v not in image_paths:
                result.error = f"Thiếu ảnh {v}"
                return result
            p = Path(image_paths[v])
            if not p.exists():
                result.error = f"File không tồn tại: {p}"
                return result
            images[v] = Image.open(p).convert("RGB")

        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)

        # Encode all 4 images
        embeddings = {}
        with torch.no_grad():
            for v, img in images.items():
                inputs = processor(images=img, return_tensors="pt").to(device)
                out = model.get_image_features(**inputs)
                # transformers 5.x: trả BaseModelOutputWithPooling thay vì tensor
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    emb = out.last_hidden_state.mean(dim=1)  # mean-pool
                elif hasattr(out, "image_embeds"):
                    emb = out.image_embeds
                else:
                    emb = out  # đã là tensor (transformers 4.x)
                arr = emb.cpu().numpy()
                embeddings[v] = arr.reshape(-1)

        # Pairwise similarities
        sims = []
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                va, vb = views[i], views[j]
                s = _cosine_similarity(embeddings[va], embeddings[vb])
                sims.append((va, vb, s))

        result.pairs = sims
        result.min_sim = float(min(s for _, _, s in sims))
        result.max_sim = float(max(s for _, _, s in sims))
        result.avg_sim = float(np.mean([s for _, _, s in sims]))
        result.same_object = result.min_sim >= threshold
        result.success = True

    except Exception as e:
        import traceback
        result.error = str(e)
        traceback.print_exc()

    return result
