"""
modules/mesh_loader.py - Load file 3D hoặc tạo primitive shape.
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import trimesh

SUPPORTED_FORMATS = {'.obj', '.stl', '.glb', '.gltf', '.ply', '.off'}

PRIMITIVES = {
    'cube':         lambda: trimesh.creation.box(),
    'sphere':       lambda: trimesh.creation.icosphere(subdivisions=2),
    'cylinder':     lambda: trimesh.creation.cylinder(radius=1, height=2, sections=12),
    'cone':         lambda: trimesh.creation.cone(radius=1, height=2, sections=12),
    'torus':        lambda: trimesh.creation.torus(major_radius=2, minor_radius=0.8),
    'tetrahedron':  lambda: trimesh.creation.icosphere(subdivisions=0),
    'capsule':      lambda: trimesh.creation.capsule(radius=0.8, height=1.5),
    'annulus':      lambda: trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=0.5),
}

PRIMITIVE_LABELS = {
    'cube': 'Hộp lập phương', 'sphere': 'Hình cầu',
    'cylinder': 'Hình trụ',   'cone': 'Hình nón',
    'torus': 'Vành khuyên',   'tetrahedron': 'Tứ diện',
    'capsule': 'Viên nang',   'annulus': 'Ống trụ rỗng',
}

@dataclass
class MeshLoadResult:
    success:     bool
    mesh:        object = None
    source_name: str    = ""
    orig_faces:  int    = 0
    final_faces: int    = 0
    error:       str    = ""

    def summary(self):
        if not self.success:
            return f"  ❌ {self.error}"
        return (f"  ✅ {self.source_name}\n"
                f"  Faces gốc : {self.orig_faces:,}\n"
                f"  Faces sau : {self.final_faces:,}")


def _simplify(mesh, target):
    n = len(mesh.faces)
    if n <= target:
        return mesh
    r = max(0.01, min(0.99, 1.0 - target/n))
    try:
        return mesh.simplify_quadric_decimation(r)
    except Exception:
        return mesh

def _clean(mesh):
    mesh = mesh.process()
    extents = mesh.extents
    mx = np.max(extents)
    if mx > 1e-6:
        mesh.apply_scale(10.0 / mx)
    return mesh

def load_file(path, target_faces=300):
    path = Path(path)
    r = MeshLoadResult(success=False, source_name=path.name)
    if not path.exists():
        r.error = f"File không tồn tại: {path}"; return r
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        r.error = f"Định dạng không hỗ trợ: {path.suffix}"; return r
    try:
        loaded = trimesh.load(str(path), force='mesh')
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                r.error = "File không chứa mesh"; return r
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded
        r.orig_faces = len(mesh.faces)
        mesh = _clean(mesh)
        mesh = _simplify(mesh, target_faces)
        r.mesh = mesh; r.final_faces = len(mesh.faces); r.success = True
    except Exception as e:
        r.error = str(e)
    return r

def load_primitive(name, target_faces=200):
    r = MeshLoadResult(success=False, source_name=f"Primitive: {PRIMITIVE_LABELS.get(name, name)}")
    if name not in PRIMITIVES:
        r.error = f"Không có '{name}'. Có: {list(PRIMITIVES)}"; return r
    try:
        mesh = PRIMITIVES[name]()
        r.orig_faces = len(mesh.faces)
        mesh = _clean(mesh)
        mesh = _simplify(mesh, target_faces)
        r.mesh = mesh; r.final_faces = len(mesh.faces); r.success = True
    except Exception as e:
        r.error = str(e)
    return r

def get_primitive_list():
    return PRIMITIVE_LABELS
