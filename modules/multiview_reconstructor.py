"""
modules/multiview_reconstructor.py
v1.5 — Ghép 4 depth maps (front/back/left/right) thành 1 mesh 3D watertight.

Approach: Orthographic projection + Poisson reconstruction
- 4 ảnh → 4 depth maps → 4 point clouds
- Align 4 point clouds về cùng hệ tọa độ (front=ref, back/left/right rotate)
- Merge → Poisson surface reconstruction → mesh kín
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image

try:
    import open3d as o3d
    O3D_OK = True
except ImportError:
    O3D_OK = False


@dataclass
class MultiViewResult:
    success:    bool
    mesh:       object = None   # trimesh.Trimesh
    num_points: int    = 0
    num_faces:  int    = 0
    error:      str    = ""


def _depth_to_pointcloud(depth_np: np.ndarray, mask: np.ndarray,
                         scale: float = 10.0,
                         use_depth: bool = True) -> np.ndarray:
    """Chuyển depth map 2D → point cloud 3D (N, 3).

    Args:
        depth_np:  (H, W) normalized 0-1
        mask:      (H, W) bool — True = foreground
        scale:     scale factor để mesh có kích thước ~10 units
        use_depth: True = dùng depth làm Z; False = silhouette (Z=0).
                   Silhouette mode tránh được vấn đề Depth Anything normalize
                   khác nhau giữa các view, đổi lại mất chi tiết bề mặt.

    Returns:
        points: (N, 3) array
    """
    H, W = depth_np.shape
    ys, xs = np.where(mask)
    depths = depth_np[ys, xs] if use_depth else np.zeros(len(ys))

    x_norm = (xs / W - 0.5) * scale
    y_norm = -(ys / H - 0.5) * scale
    z_norm = depths * scale * 0.5

    points = np.stack([x_norm, y_norm, z_norm], axis=1)
    return points


def _align_pointcloud(points: np.ndarray, view: str) -> np.ndarray:
    """Align point cloud từ view cụ thể về hệ tọa độ front (reference).

    Coordinate system:
        Front: (x, y, z)         — reference, không đổi
        Back:  (-x, y, -z)       — mirror X, negate Z
        Left:  (-z, y, x)        — rotate 90° CCW quanh Y
        Right: (z, y, -x)        — rotate 90° CW quanh Y

    Args:
        points: (N, 3) point cloud trong local coordinate của view đó
        view:   'front' | 'back' | 'left' | 'right'

    Returns:
        aligned_points: (N, 3) trong hệ tọa độ chung
    """
    if view == 'front':
        return points.copy()
    elif view == 'back':
        # Mirror X, negate Z
        aligned = points.copy()
        aligned[:, 0] = -aligned[:, 0]  # x → -x
        aligned[:, 2] = -aligned[:, 2]  # z → -z
        return aligned
    elif view == 'left':
        # Rotate 90° CCW: (x, y, z) → (-z, y, x)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return np.stack([-z, y, x], axis=1)
    elif view == 'right':
        # Rotate 90° CW: (x, y, z) → (z, y, -x)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return np.stack([z, y, -x], axis=1)
    else:
        raise ValueError(f"Unknown view: {view}")


def _view_normal(view: str) -> np.ndarray:
    """Normal hướng từ object ra camera (theo hệ tọa độ chung sau align)."""
    if view == 'front':
        return np.array([0, 0, 1])    # camera ở +Z
    elif view == 'back':
        return np.array([0, 0, -1])   # camera ở -Z
    elif view == 'left':
        return np.array([-1, 0, 0])   # camera ở -X
    elif view == 'right':
        return np.array([1, 0, 0])    # camera ở +X
    raise ValueError(view)


def _poisson_reconstruction(points: np.ndarray, normals: Optional[np.ndarray] = None,
                            voxel_size: float = 0.3,
                            poisson_depth: int = 6,
                            target_faces: int = 150) -> object:
    """Poisson surface reconstruction từ point cloud → trimesh.

    v1.5 fix: voxel downsample mạnh + Poisson depth nhỏ + simplify sau Poisson.

    Args:
        points:        (N, 3) merged point cloud
        normals:       (N, 3) normals (nếu None, estimate)
        voxel_size:    0.3 = ~3% extent → ~3000 points sau downsample
        poisson_depth: 6 = mesh smoother, ~10K-30K faces
        target_faces: simplify mesh xuống còn target_faces (None = không simplify)

    Returns:
        trimesh.Trimesh
    """
    if not O3D_OK:
        raise ImportError("open3d chưa cài: pip install open3d")

    import trimesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if normals is None or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
    )

    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.scale(1.0, bbox.get_center())
    mesh_o3d = mesh_o3d.crop(bbox)

    densities = np.asarray(densities)
    if len(densities) == len(mesh_o3d.vertices):
        density_threshold = np.quantile(densities, 0.15)
        vertices_to_remove = densities < density_threshold
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Laplacian smoothing để giảm gồ ghề Poisson
    mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=2)
    mesh_o3d.compute_vertex_normals()

    # Simplify mesh xuống target_faces (open3d quadric decimation)
    if target_faces is not None and len(mesh_o3d.triangles) > target_faces:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )

    # Convert to trimesh
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Giữ component lớn nhất (loại các island nhỏ rời)
    try:
        comps = mesh.split(only_watertight=False)
        if len(comps) > 1:
            mesh = max(comps, key=lambda c: len(c.faces))
    except Exception:
        pass

    # Fill holes + fix normals
    try:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    return mesh


def reconstruct_from_multiview(
    image_paths: dict,
    depth_estimator_fn,
    bg_threshold: float = 0.08,
    scale: float = 10.0,
    use_depth: bool = True,
) -> MultiViewResult:
    """Ghép 4 ảnh (front/back/left/right) thành 1 mesh 3D.

    Args:
        image_paths:        dict {'front': path, 'back': path, 'left': path, 'right': path}
        depth_estimator_fn: callable(pil_img) → depth_np (0-1)
        bg_threshold:       ngưỡng tách foreground
        scale:              scale factor cho mesh output
        use_depth:          True = depth + silhouette; False = silhouette only (visual hull).
                            Silhouette mode robust hơn khi 4 depth maps không scale-aligned.

    Returns:
        MultiViewResult
    """
    result = MultiViewResult(success=False)

    try:
        required_views = ['front', 'back', 'left', 'right']
        for v in required_views:
            if v not in image_paths:
                result.error = f"Thiếu ảnh {v}"
                return result

        all_points = []
        all_normals = []

        for view in required_views:
            img_path = Path(image_paths[view])
            if not img_path.exists():
                result.error = f"File không tồn tại: {img_path}"
                return result

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Depth estimation
            depth_np = depth_estimator_fn(img)
            mask = depth_np > bg_threshold

            # Depth → point cloud (local coordinate)
            pc_local = _depth_to_pointcloud(depth_np, mask, scale=scale, use_depth=use_depth)

            # Align về hệ tọa độ chung
            pc_aligned = _align_pointcloud(pc_local, view)
            all_points.append(pc_aligned)

            # Normal: từ surface ra camera (theo view direction)
            n = _view_normal(view)
            all_normals.append(np.tile(n, (len(pc_aligned), 1)))

        # Merge all point clouds + normals
        merged_points  = np.vstack(all_points)
        merged_normals = np.vstack(all_normals)
        result.num_points = len(merged_points)

        # Poisson reconstruction (truyền normals biết trước, không cần estimate)
        mesh = _poisson_reconstruction(merged_points, normals=merged_normals)
        result.mesh = mesh
        result.num_faces = len(mesh.faces)
        result.success = True

    except Exception as e:
        import traceback
        result.error = str(e)
        traceback.print_exc()

    return result
