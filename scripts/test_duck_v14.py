"""
scripts/test_duck_v14.py — pipeline v1.4 trên duck-toy.png:

  1. depth front  : Depth Anything V2 trên ảnh gốc
  2. AI back view : SD inpainting sinh ảnh mặt sau (lật ngang + inpaint silhouette)
  3. depth back   : Depth Anything V2 lần 2 trên ảnh back AI sinh ra
  4. align        : lật ngang depth_back để khớp toạ độ XY với front
  5. mesh         : front_depth + back_depth → khối kín CÓ KHỐI ở cả 2 mặt
  6. unfold/layout/pdf

Đo:
  - back_view có hợp lý không (lưu ra disk để xem)
  - mesh extent z so với v1.3 (v1.3 z chỉ depth_scale; v1.4 z = 2*depth_scale nếu back đối xứng)
  - panels/islands cuối cùng
  - thời gian: model SD load lần đầu chậm
"""
import sys, io, time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageOps
import numpy as np

from modules.depth_estimator     import estimate_depth_and_build_mesh, estimate_depth_only
from modules.back_view_generator import generate_back_view, save_back_view
from modules.unfolder            import unfold_mesh
from modules.layout              import layout_panels
from modules.pdf_exporter        import export_pdf


def run():
    img_path = ROOT / "img" / "duck-toy.png"
    assert img_path.exists(), f"Khong thay anh: {img_path}"
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    print(f"\n=== INPUT: {img_path.name} ===")

    # ── Step 1: depth front ──────────────────────────────────────────────
    print("\n[1] Depth Anything V2 — front ...")
    t0 = time.perf_counter()
    front_r = estimate_depth_and_build_mesh(
        image_path=str(img_path),
        model_size="small",
        target_faces=200,
        depth_scale=3.0,
        remove_background=True,
        solidify=False,    # chỉ lấy front depth + ảnh gốc, build mesh ở bước 5
    )
    if not front_r.success:
        print(f"  FAIL: {front_r.error}")
        return
    print(f"  depth shape={front_r.depth_map.shape}  time={time.perf_counter()-t0:.1f}s")

    front_depth = front_r.depth_map
    front_img   = front_r.orig_image

    # ── Step 2: SD inpainting → ảnh mặt sau ─────────────────────────────
    print("\n[2] Stable Diffusion inpainting — back view ...")
    t0 = time.perf_counter()
    back_r = generate_back_view(
        front_img=front_img,
        depth_np=front_depth,
        prompt="back view of a rubber duck toy, yellow tail and back, smooth plastic, plain background",
        negative_prompt="front view, face, eyes, beak, text, distortion, ugly, blurry",
        num_steps=25,
        guidance=7.5,
        seed=42,
    )
    if not back_r.success:
        print(f"  FAIL: {back_r.error}")
        return
    saved = save_back_view(back_r, out_dir / "duck_back_ai.png")
    print(f"  back saved -> {saved}  time={time.perf_counter()-t0:.1f}s")

    # ── Step 3: depth lần 2 trên ảnh back ───────────────────────────────
    print("\n[3] Depth Anything V2 — back ...")
    t0 = time.perf_counter()
    # Lật ngang lại để toạ độ XY của back depth khớp với front
    back_for_depth = ImageOps.mirror(back_r.back_img)
    back_depth_raw = estimate_depth_only(back_for_depth, model_size="small")
    print(f"  back depth shape={back_depth_raw.shape}  time={time.perf_counter()-t0:.1f}s")

    # Mask back depth bằng silhouette của front (giữ đúng vùng object)
    H, W = front_depth.shape
    bd_img = Image.fromarray((back_depth_raw * 255).astype(np.uint8), mode="L")
    bd_img = bd_img.resize((W, H), Image.LANCZOS)
    back_depth = np.array(bd_img).astype(float) / 255.0

    front_mask = front_depth > 0.08
    back_depth = back_depth * front_mask     # ngoài silhouette → 0

    # ── Step 4: mesh = front + back depth ───────────────────────────────
    print("\n[4] Build mesh với back AI depth ...")
    t0 = time.perf_counter()
    mesh_r = estimate_depth_and_build_mesh(
        image_path=str(img_path),
        model_size="small",
        target_faces=300,
        depth_scale=3.0,
        remove_background=True,
        solidify=True,
        back_depth_np=back_depth,
    )
    if not mesh_r.success:
        print(f"  FAIL: {mesh_r.error}")
        return
    m = mesh_r.mesh
    print(f"  verts={len(m.vertices)}  faces={len(m.faces)}  "
          f"watertight={m.is_watertight}  euler={m.euler_number}  "
          f"extents={m.extents.round(2).tolist()}  time={time.perf_counter()-t0:.1f}s")

    # ── Step 5: unfold + layout + pdf ───────────────────────────────────
    print("\n[5] Unfold + layout + pdf ...")
    uf = unfold_mesh(m, max_faces=400, coplanar_tol_deg=25.0)
    print(f"  panels={len(uf.panels)}  facets={uf.num_facets}  islands={uf.num_islands}")

    lo = layout_panels(uf.panels, target_size_mm=70.0, mesh=m)
    print(f"  pages={len(lo.pages)}  scale={lo.scale:.3f}  warnings={len(lo.warnings)}")

    pdf_out = out_dir / "duck_v14.pdf"
    ok = export_pdf(lo, pdf_out, title="Papercraft Duck v1.4 (AI back)", show_glue=True)
    print(f"  pdf={ok} -> {pdf_out}  size={pdf_out.stat().st_size/1024:.1f}KB" if ok else "  pdf FAIL")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n=== VERDICT v1.4 ===")
    print(f"  watertight       : {m.is_watertight}")
    print(f"  extent z         : {m.extents[2]:.2f}  (v1.3 ~3.0, v1.4 should be larger if back has volume)")
    print(f"  panels           : {len(uf.panels)}")
    print(f"  islands          : {uf.num_islands}")
    print(f"  has back volume  : {back_depth.max() > 0.05}")


if __name__ == "__main__":
    run()
