"""
scripts/test_duck.py — chạy thử pipeline ảnh→papercraft trên duck-toy.png.

Đo các chỉ số "có dùng được không":
    - depth estimation chạy được trên CPU không? thời gian?
    - mesh số faces sau simplify, có watertight không
    - unfold ra bao nhiêu panel, bao nhiêu island (nếu islands quá nhiều thì khó dán)
    - layout ra bao nhiêu trang, có overlap không
    - xuất được PDF không
"""
import sys, time, io
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.depth_estimator import estimate_depth_and_build_mesh
from modules.unfolder        import unfold_mesh
from modules.layout          import layout_panels
from modules.pdf_exporter    import export_pdf


def run():
    img = ROOT / "img" / "duck-toy.png"
    assert img.exists(), f"Khong thay anh: {img}"

    print(f"\n=== INPUT: {img.name} ===")

    t0 = time.perf_counter()
    depth_r = estimate_depth_and_build_mesh(
        image_path=str(img),
        model_size="small",
        target_faces=200,
        depth_scale=3.0,
        remove_background=True,
    )
    t_depth = time.perf_counter() - t0
    print(f"\n[depth] success={depth_r.success}  faces={depth_r.num_faces}  time={t_depth:.1f}s")
    if not depth_r.success:
        print(f"  ERROR: {depth_r.error}")
        return

    m = depth_r.mesh
    print(f"[mesh]  vertices={len(m.vertices)}  faces={len(m.faces)}  "
          f"watertight={m.is_watertight}  euler={m.euler_number}  "
          f"extents={m.extents.round(2).tolist()}")

    t0 = time.perf_counter()
    uf = unfold_mesh(m, max_faces=400, coplanar_tol_deg=25.0)
    t_unfold = time.perf_counter() - t0
    print(f"\n[unfold] success={uf.success}  panels={len(uf.panels)}  "
          f"facets={uf.num_facets}  islands={uf.num_islands}  time={t_unfold:.1f}s")
    if not uf.success:
        print(f"  ERROR: {uf.error}")
        return

    if uf.num_islands > len(uf.panels) // 2:
        print(f"  WARN: nhieu island ({uf.num_islands}) => kho dan")

    t0 = time.perf_counter()
    lo = layout_panels(uf.panels, target_size_mm=70.0, mesh=m)
    t_layout = time.perf_counter() - t0
    print(f"\n[layout] success={lo.success}  pages={len(lo.pages)}  "
          f"panels_placed={lo.total_panels}  scale={lo.scale:.3f}  time={t_layout:.1f}s")
    for w in lo.warnings:
        print(f"  WARN: {w}")

    out_pdf = ROOT / "output" / "duck_v12.pdf"
    out_pdf.parent.mkdir(exist_ok=True)
    ok = export_pdf(lo, out_pdf, title="Papercraft Duck", show_glue=True)
    print(f"\n[pdf] export={ok} -> {out_pdf}  "
          f"size={out_pdf.stat().st_size/1024:.1f}KB" if ok else "")

    # Verdict
    print("\n=== VERDICT ===")
    verdict = []
    if uf.num_facets > 60:
        verdict.append(f"FAIL: quá nhiều mảnh ({uf.num_facets}) - khó cắt/dán")
    elif uf.num_facets > 30:
        verdict.append(f"WARN: nhiều mảnh ({uf.num_facets}) - kha kho")
    else:
        verdict.append(f"OK mảnh: {uf.num_facets}")

    if uf.num_islands > 5:
        verdict.append(f"FAIL: quá nhiều island ({uf.num_islands}) - rời rạc")
    elif uf.num_islands > 2:
        verdict.append(f"WARN: nhiều island ({uf.num_islands})")
    else:
        verdict.append(f"OK islands: {uf.num_islands}")

    if lo.warnings:
        verdict.append(f"FAIL: có overlap warning")
    else:
        verdict.append("OK: khong overlap")

    if not m.is_watertight:
        verdict.append("WARN: mesh khong watertight (kho lap rap thanh khoi kin)")

    for v in verdict:
        print("  " + v)


if __name__ == "__main__":
    run()
