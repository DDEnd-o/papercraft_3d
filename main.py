"""
main.py — Entry point Papercraft 3D App
Chạy GUI : python main.py
Chạy CLI : python main.py --shape cube --faces 120 --output output/cube.pdf
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def run_gui():
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("Papercraft 3D")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


def run_cli(args):
    from modules.mesh_loader  import load_primitive, load_file
    from modules.unfolder      import unfold_mesh
    from modules.layout        import layout_panels
    from modules.pdf_exporter  import export_pdf

    print("\n" + "="*55)
    print("  PAPERCRAFT 3D - CLI Mode")
    print("="*55)

    if args.file:
        print(f"\nLoad file: {args.file}")
        mesh_r = load_file(args.file, target_faces=args.faces)
    else:
        print(f"\nPrimitive: {args.shape}")
        mesh_r = load_primitive(args.shape, target_faces=args.faces)

    print(mesh_r.summary())
    if not mesh_r.success:
        return

    print("\nUnfold mesh...")
    unfold_r = unfold_mesh(mesh_r.mesh, max_faces=args.faces * 2)
    print(unfold_r.summary())
    if not unfold_r.success:
        return

    print("\nLayout panels...")
    layout_r = layout_panels(
        unfold_r.panels,
        target_size_mm=args.size,
        mesh=mesh_r.mesh,
    )
    print(layout_r.summary())

    if args.output:
        out = Path(args.output)
    else:
        stem = Path(args.file).stem if args.file else args.shape
        out = Path("output") / f"{stem}.pdf"
    
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nXuat PDF: {out}")
    ok = export_pdf(layout_r, out,
                    title=f"Papercraft - {args.shape or Path(args.file).stem}")
    if ok:
        print(f"\nXong! File: {out.resolve()}")
    else:
        print("\nXuat PDF that bai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Papercraft 3D")
    parser.add_argument("--gui",    action="store_true")
    parser.add_argument("--shape",  default="cube",
        help="cube/sphere/cylinder/cone/torus/tetrahedron/capsule/annulus")
    parser.add_argument("--file",   default=None)
    parser.add_argument("--faces",  type=int, default=120)
    parser.add_argument("--size",   type=int, default=65)
    parser.add_argument("--output", default=None, help="Output PDF path (default: output/<filename>.pdf)")

    args = parser.parse_args()

    if args.gui or len(sys.argv) == 1:
        run_gui()
    else:
        run_cli(args)
