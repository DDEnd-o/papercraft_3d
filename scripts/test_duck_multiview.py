"""
scripts/test_duck_multiview.py — v1.5 pipeline test

Input:  4 ảnh (front/back/left/right) từ img/multiview_duck/
Output: mesh 3D watertight → unfold → layout → PDF

So sánh với v1.3/v1.4:
  - v1.3: 1 ảnh, back phẳng → 33 mảnh, 3 islands
  - v1.4: 1 ảnh, AI back → 17 mảnh, 6 islands, back sai
  - v1.5: 4 ảnh thật → watertight đúng, ít mảnh hơn
"""
import sys, io, time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.depth_estimator         import estimate_depth_only
from modules.multiview_reconstructor  import reconstruct_from_multiview
from modules.image_validator          import validate_multiview_images
from modules.unfolder                 import unfold_mesh
from modules.layout                   import layout_panels
from modules.pdf_exporter             import export_pdf


def run():
    img_dir = ROOT / "img" / "multiview_duck"
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    image_paths = {
        'front': img_dir / "duck_front.png",
        'back':  img_dir / "duck_back.png",
        'left':  img_dir / "duck_left.png",
        'right': img_dir / "duck_right.png",
    }

    for v, p in image_paths.items():
        if not p.exists():
            print(f"FAIL: Thiếu ảnh {v} tại {p}")
            print("Chạy trước: python scripts/generate_multiview_images.py")
            return

    print("\n=== v1.5 MULTI-VIEW RECONSTRUCTION ===")
    print(f"Input: {img_dir}")

    # ── Step 0: CLIP validation (4 ảnh cùng object?) ────────────────────
    print("\n[0] CLIP validation 4 ảnh có cùng object không...")
    t0 = time.perf_counter()
    val = validate_multiview_images(image_paths, threshold=0.75)
    if not val.success:
        print(f"  WARN: validate fail: {val.error} — bỏ qua check, tiếp tục")
    else:
        print(f"  min_sim={val.min_sim:.3f}  max_sim={val.max_sim:.3f}  "
              f"avg_sim={val.avg_sim:.3f}  same_object={val.same_object}  "
              f"time={time.perf_counter()-t0:.1f}s")
        for va, vb, s in val.pairs:
            print(f"    {va:<5} <-> {vb:<5} : {s:.3f}")
        if not val.same_object:
            print("  FAIL: 4 ảnh không giống nhau (có thể khác object). Dừng pipeline.")
            return

    # ── Step 1: 4 depth maps + Poisson reconstruction ───────────────────
    print("\n[1] Depth estimation (4 views) + Poisson reconstruction...")
    t0 = time.perf_counter()

    def depth_fn(pil_img):
        return estimate_depth_only(pil_img, model_size="small")

    mv_result = reconstruct_from_multiview(
        image_paths=image_paths,
        depth_estimator_fn=depth_fn,
        bg_threshold=0.08,
        scale=10.0,
        use_depth=False,    # silhouette only - robust hơn khi depth không scale-aligned
    )

    if not mv_result.success:
        print(f"  FAIL: {mv_result.error}")
        return

    m = mv_result.mesh
    print(f"  points={mv_result.num_points}  faces={mv_result.num_faces}  "
          f"watertight={m.is_watertight}  euler={m.euler_number}  "
          f"extents={m.extents.round(2).tolist()}  time={time.perf_counter()-t0:.1f}s")

    # Lưu mesh + point cloud để debug visual
    mesh_out = out_dir / "duck_v15_mesh.obj"
    m.export(mesh_out)
    print(f"  mesh saved -> {mesh_out}")

    # ── Step 2: Unfold + layout + PDF ───────────────────────────────────
    print("\n[2] Unfold + layout + PDF...")
    uf = unfold_mesh(m, max_faces=600, coplanar_tol_deg=25.0)
    print(f"  panels={len(uf.panels)}  facets={uf.num_facets}  islands={uf.num_islands}")

    lo = layout_panels(uf.panels, target_size_mm=70.0, mesh=m)
    print(f"  pages={len(lo.pages)}  scale={lo.scale:.3f}  warnings={len(lo.warnings)}")

    pdf_out = out_dir / "duck_v15_multiview.pdf"
    ok = export_pdf(lo, pdf_out, title="Papercraft Duck v1.5 (Multi-View)", show_glue=True)
    print(f"  pdf={ok} -> {pdf_out}  size={pdf_out.stat().st_size/1024:.1f}KB" if ok else "  pdf FAIL")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n=== VERDICT v1.5 ===")
    print(f"  watertight       : {m.is_watertight}")
    print(f"  euler            : {m.euler_number}  (2 = perfect sphere-like)")
    print(f"  extent           : {m.extents.round(2).tolist()}")
    print(f"  panels           : {len(uf.panels)}")
    print(f"  islands          : {uf.num_islands}")
    print(f"  vs v1.3          : v1.3=33 panels/3 islands, v1.5={len(uf.panels)} panels/{uf.num_islands} islands")
    print(f"  vs v1.4          : v1.4=17 panels/6 islands (AI sai), v1.5={len(uf.panels)} panels/{uf.num_islands} islands (4 ảnh thật)")


if __name__ == "__main__":
    run()
