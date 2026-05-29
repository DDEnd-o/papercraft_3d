"""
scripts/generate_multiview_images.py
Tao 4 anh (front/back/left/right) tu 1 mesh 3D de test v1.5.

Vi khong co model duck 3D, toi dung capsule (hinh vien nang) lam demo.
Trong production, user se tu chup 4 anh object that.
"""
import sys, io
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh
from PIL import Image

def create_simple_duck():
    """Tạo 'duck' đơn giản từ primitives: body (sphere) + head (smaller sphere) + beak (cone)."""
    # Body
    body = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    body.apply_translation([0, 0, 0])

    # Head
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.6)
    head.apply_translation([0, 0.8, 0.8])

    # Beak (cone nhỏ)
    beak = trimesh.creation.cone(radius=0.2, height=0.4, sections=8)
    beak.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    beak.apply_translation([0, 0.8, 1.2])

    # Merge
    duck = trimesh.util.concatenate([body, head, beak])
    duck.merge_vertices()

    # Scale to ~10 units
    extents = duck.extents
    scale = 10.0 / max(extents)
    duck.apply_scale(scale)

    return duck


def render_view(mesh, camera_pos, resolution=(512, 512)):
    """Render mesh từ 1 góc nhìn bằng matplotlib 3D (fallback nếu không có pyrender)."""
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ mesh
    verts = mesh.vertices
    faces = mesh.faces
    poly = Poly3DCollection(verts[faces], alpha=0.8, facecolor='yellow', edgecolor='black', linewidths=0.5)
    ax.add_collection3d(poly)

    # Set view angle
    # camera_pos = [x, y, z] → azim, elev
    x, y, z = camera_pos
    azim = np.degrees(np.arctan2(x, -y))  # front=-Y → azim=0
    elev = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))

    ax.view_init(elev=elev, azim=azim)

    # Axis limits
    scale = mesh.extents.max() * 0.6
    ax.set_xlim([-scale, scale])
    ax.set_ylim([-scale, scale])
    ax.set_zlim([-scale, scale])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')

    # Save to PIL
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGBA → RGB
    img = Image.fromarray(buf).resize(resolution, Image.LANCZOS)
    plt.close(fig)
    return img


def main():
    out_dir = Path(__file__).parent.parent / "img" / "multiview_duck"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Tạo simple duck mesh...")
    duck = create_simple_duck()
    print(f"  vertices={len(duck.vertices)}, faces={len(duck.faces)}")

    # 4 camera positions (orthographic-like, xa object)
    distance = 20.0
    views = {
        "front": [0, -distance, 0],      # nhìn từ phía trước (−Y)
        "back":  [0, distance, 0],       # nhìn từ phía sau (+Y)
        "left":  [-distance, 0, 0],      # nhìn từ bên trái (−X)
        "right": [distance, 0, 0],       # nhìn từ bên phải (+X)
    }

    for name, cam_pos in views.items():
        print(f"Render {name} view...")
        img = render_view(duck, cam_pos, resolution=(512, 512))
        out = out_dir / f"duck_{name}.png"
        img.save(out)
        print(f"  -> {out}")

    print(f"\n✅ 4 ảnh đã lưu vào {out_dir}")
    print("Dùng cho v1.5: python scripts/test_duck_multiview.py")


if __name__ == "__main__":
    main()
