"""
Module WRAP TEXTURE (V18) -- xatlas UV unwrap + ortho bake photo grayscale texture,
write OBJ+MTL+PNG + STL + GLB. QA preview package for Kent to verify face detail
before wasting a crystal blank.

Zero xatlas hang risk: reuses subprocess-timeout guard from module_e_texture.py
(V11-V14 proven pattern).

Texture is CLAHE-normalized GRAYSCALE (SSLE luminance-driven engraving convention).
Ortho projection matches capture -> zero distortion (Fable 5 v3 verdict).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModuleWrapConfig:
    input_mesh: Path
    input_image: Path
    out_stem: Path              # e.g. /workspace/out/person_2_v18 -- module writes .obj, .mtl, _texture.png, .stl, .glb

    target_height_mm: float = 70.0
    atlas_res: int = 4096
    xatlas_target_faces: int = 60_000    # decimate before xatlas (V14 pattern)
    xatlas_timeout_s: int = 180
    front_axis: str = "+Z"
    clahe_clip: float = 3.0
    clahe_tile: int = 16
    dry_run: bool = False

    def __post_init__(self):
        self.input_mesh = Path(self.input_mesh)
        self.input_image = Path(self.input_image)
        self.out_stem = Path(self.out_stem)


def _xatlas_worker(verts, faces, out_queue):
    try:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(verts, faces)
        try:
            co = xatlas.ChartOptions(); co.max_iterations = 1
            po = xatlas.PackOptions(); po.padding = 4
            atlas.generate(chart_options=co, pack_options=po)
        except Exception:
            atlas.generate()
        vmapping, indices, uvs = atlas[0]
        out_queue.put((uvs, indices, vmapping))
    except Exception as e:
        out_queue.put(("ERROR", str(e)))


def unwrap_uvs(mesh, target_faces, timeout_s):
    """Decimate + xatlas UV unwrap with subprocess kill guard."""
    import numpy as np
    import trimesh
    import multiprocessing as mp
    print(f"      original: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces", flush=True)
    try:
        dec = mesh.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:
        dec = mesh.simplify_quadric_decimation(target_faces)
    print(f"      decimated to: {len(dec.vertices):,} verts, {len(dec.faces):,} faces", flush=True)
    dec_verts = np.asarray(dec.vertices, dtype=np.float32)
    dec_faces = np.asarray(dec.faces, dtype=np.uint32)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_xatlas_worker, args=(dec_verts, dec_faces, q))
    p.start()
    # V21h.1 Fable v3 fix: DRAIN queue BEFORE joining. The old
    # `p.join(180); q.get_nowait()` pattern was a classic mp.Queue deadlock —
    # child.put()s a multi-MB result, feeder thread blocks flushing >64KB into
    # the pipe nobody is reading, child never exits, join always times out at
    # 180s. Result: xatlas has NEVER successfully returned here; every render
    # silently fell back to pymeshlab trivial-per-wedge (per-triangle fragmented
    # UV = renders visually identical regardless of bake pixels). Kent's
    # "T1 == T2" bug traced to this.
    try:
        result = q.get(timeout=timeout_s)
    except Exception:
        try:
            if p.is_alive():
                p.terminate()
        finally:
            p.join(5)
        raise TimeoutError(f"xatlas > {timeout_s}s (queue drain)")
    p.join(10)
    if p.is_alive():
        p.terminate(); p.join(5)
    if isinstance(result, tuple) and result and result[0] == "ERROR":
        raise RuntimeError(f"xatlas: {result[1]}")
    uvs, indices, vmapping = result
    return (np.asarray(uvs, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(vmapping, dtype=np.int64),
            dec_verts, dec_faces)


def bake_ortho_texture(photo_pil, verts_world, uvs, faces_uv, vmapping,
                       atlas_res, front_axis="+Z", clahe_clip=3.0, clahe_tile=16):
    """Ortho project photo onto UV atlas. Grayscale + CLAHE."""
    import numpy as np
    from PIL import Image
    import cv2

    photo_np = np.array(photo_pil.convert("RGB"))
    img_h, img_w = photo_np.shape[:2]

    # Grayscale + CLAHE first (face detail boost)
    gray = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    gray_eq = clahe.apply(gray)
    photo_gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

    if front_axis == "-Z":
        world_xy = verts_world[:, :2].copy() * np.array([-1, 1])
    else:
        world_xy = verts_world[:, :2].copy()
    v_min = world_xy.min(0); v_max = world_xy.max(0)
    v_span = float(np.max(v_max - v_min)); v_span = max(v_span, 1e-9)
    v_center = (v_min + v_max) / 2.0
    img_center = np.array([img_w/2.0, img_h/2.0])
    img_scale = min(img_w, img_h) / v_span

    def w2i(pts):
        cent = pts - v_center
        cent[:, 1] *= -1.0
        return cent * img_scale + img_center

    atlas = np.full((atlas_res, atlas_res, 3), 128, dtype=np.uint8)  # neutral grey fill
    coverage = np.zeros((atlas_res, atlas_res), dtype=np.uint8)

    tri_orig_idx = vmapping[faces_uv]
    img_pts_all = w2i(world_xy[tri_orig_idx.reshape(-1)]).reshape(-1, 3, 2)
    atlas_pts_all = (uvs[faces_uv] * atlas_res).astype(np.float32)

    v01 = img_pts_all[:, 1] - img_pts_all[:, 0]
    v02 = img_pts_all[:, 2] - img_pts_all[:, 0]
    cross_z = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
    z_sign = 1.0 if front_axis == "+Z" else -1.0
    front_mask = cross_z * z_sign < 0

    n_baked = 0
    for i in range(len(faces_uv)):
        if not front_mask[i]: continue
        img_tri = img_pts_all[i]; atlas_tri = atlas_pts_all[i]
        x0 = max(int(np.floor(atlas_tri[:, 0].min())) - 1, 0)
        y0 = max(int(np.floor(atlas_tri[:, 1].min())) - 1, 0)
        x1 = min(int(np.ceil(atlas_tri[:, 0].max())) + 1, atlas_res)
        y1 = min(int(np.ceil(atlas_tri[:, 1].max())) + 1, atlas_res)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0: continue
        local_tri = atlas_tri - np.array([x0, y0], dtype=np.float32)
        try:
            M = cv2.getAffineTransform(img_tri.astype(np.float32), local_tri)
            warp = cv2.warpAffine(photo_gray_rgb, M, (w, h))
        except cv2.error:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_tri.astype(np.int32), 255)
        roi = atlas[y0:y1, x0:x1]
        roi[mask > 0] = warp[mask > 0]
        coverage[y0:y1, x0:x1] |= mask
        n_baked += 1
    print(f"      baked {n_baked}/{len(faces_uv)} tris (skipped {int((~front_mask).sum())} back-facing)",
          flush=True)
    return atlas, coverage


def run_module_wrap(cfg: ModuleWrapConfig):
    print("=" * 66)
    print("  Module WRAP TEXTURE (V18) -- QA preview package")
    print("=" * 66)
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  input image  : {cfg.input_image}")
    print(f"  out_stem     : {cfg.out_stem}")
    print(f"  atlas_res    : {cfg.atlas_res}  target_faces: {cfg.xatlas_target_faces}")
    print("=" * 66, flush=True)
    if cfg.dry_run: print("[dry-run] validated"); return

    t0 = time.time()
    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.texture import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    cfg.out_stem.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load mesh + scale to mm
    print("\n[1/5] Load + scale mesh", flush=True)
    obj = trimesh.load(cfg.input_mesh, force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values()); geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    else:
        mesh = obj
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    bbox_min = verts.min(0); bbox_max = verts.max(0); bbox_size = bbox_max - bbox_min
    s = cfg.target_height_mm / max(bbox_size[1], 1e-6)
    center = (bbox_min + bbox_max) / 2.0
    verts_mm = (verts - center) * s
    mesh_mm = trimesh.Trimesh(vertices=verts_mm, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
    print(f"      scale={s:.4f} mm/unit, bbox_mm={(bbox_size*s).round(2).tolist()}", flush=True)

    # 2. xatlas UV unwrap
    print("\n[2/5] xatlas UV unwrap", flush=True)
    try:
        uvs, faces_uv, vmapping, dec_verts, dec_faces = unwrap_uvs(
            mesh_mm, cfg.xatlas_target_faces, cfg.xatlas_timeout_s
        )
    except Exception as e:
        print(f"      xatlas fail ({e}) -- falling back to pymeshlab trivial", flush=True)
        import pymeshlab
        try: dec = mesh_mm.simplify_quadric_decimation(face_count=cfg.xatlas_target_faces)
        except TypeError: dec = mesh_mm.simplify_quadric_decimation(cfg.xatlas_target_faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(dec.vertices, dtype=np.float64),
            face_matrix=np.asarray(dec.faces, dtype=np.int32),
        ))
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=cfg.atlas_res)
        ms.compute_texcoord_transfer_wedge_to_vertex()
        m = ms.current_mesh()
        uvs = m.vertex_tex_coord_matrix().astype(np.float32)
        faces_uv = m.face_matrix().astype(np.int64)
        vmapping = np.arange(len(m.vertex_matrix()), dtype=np.int64)
        dec_verts = m.vertex_matrix().astype(np.float32)
    print(f"      UV atlas verts={len(uvs):,} faces={len(faces_uv):,}", flush=True)

    # 3. Ortho bake grayscale-CLAHE photo -> texture atlas
    print(f"\n[3/5] Ortho bake grayscale texture {cfg.atlas_res}x{cfg.atlas_res}", flush=True)
    photo = Image.open(cfg.input_image).convert("RGB")
    atlas_rgb, coverage = bake_ortho_texture(
        photo, dec_verts, uvs, faces_uv, vmapping,
        cfg.atlas_res, cfg.front_axis, cfg.clahe_clip, cfg.clahe_tile,
    )
    # Flip vertically for glTF UV convention
    import numpy as np
    atlas_for_gltf = atlas_rgb[::-1].copy()

    # 4. Build textured Trimesh
    print("\n[4/5] Build textured mesh", flush=True)
    xatlas_verts = dec_verts[vmapping]
    mat = SimpleMaterial(image=Image.fromarray(atlas_for_gltf))
    tex_mesh = trimesh.Trimesh(
        vertices=xatlas_verts, faces=faces_uv,
        visual=TextureVisuals(uv=uvs, image=Image.fromarray(atlas_for_gltf), material=mat),
        process=False,
    )
    # Geometry-only mesh (STL)
    geo_mesh = trimesh.Trimesh(vertices=xatlas_verts, faces=faces_uv, process=False)

    # 5. Export all formats
    print("\n[5/5] Export OBJ + MTL + PNG + STL + GLB", flush=True)
    stem = cfg.out_stem
    # OBJ writes companion .mtl + texture file
    obj_path = stem.with_suffix(".obj")
    tex_mesh.export(obj_path)
    # trimesh's obj exporter creates material_0.png; rename to our stem
    _obj_extras = list(obj_path.parent.glob("material_*.png"))
    tex_png = stem.parent / (stem.name + "_texture.png")
    if _obj_extras:
        _obj_extras[0].rename(tex_png)
    else:
        # Also write a copy manually to guarantee texture PNG present
        Image.fromarray(atlas_for_gltf).save(tex_png)
    print(f"      OBJ {obj_path} ({obj_path.stat().st_size} B)", flush=True)
    print(f"      texture PNG {tex_png} ({tex_png.stat().st_size} B)", flush=True)

    # STL (geometry only)
    stl_path = stem.with_suffix(".stl")
    geo_mesh.export(stl_path)
    print(f"      STL {stl_path} ({stl_path.stat().st_size} B)", flush=True)

    # GLB (all-in-one)
    glb_path = stem.with_suffix(".glb")
    tex_mesh.export(glb_path)
    print(f"      GLB {glb_path} ({glb_path.stat().st_size} B)", flush=True)

    print(f"\n[DONE] wrap total {time.time()-t0:.1f}s", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(prog="module_wrap_texture")
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--out-stem", required=True, type=Path)
    ap.add_argument("--target-height-mm", type=float, default=70.0)
    ap.add_argument("--atlas-res", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg = ModuleWrapConfig(input_mesh=a.input_mesh, input_image=a.input_image, out_stem=a.out_stem,
                           target_height_mm=a.target_height_mm, atlas_res=a.atlas_res, dry_run=a.dry_run)
    run_module_wrap(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
