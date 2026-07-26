"""
Module WRAP TEXTURE (V21) -- extends V18 module_wrap_texture with:

  1. Explicit wrap_direction: 'front' | 'back' | 'left' | 'right'
     Maps to projection axis (kills V19/V20 wrap_auto_detect_view_dir logic per
     Kent+Fable v3 verdict).

     front -> project X,Y onto -Z view (camera looks -Z at head, sees X-right / Y-up)
     back  -> project -X,Y onto +Z view (mirror X so face texture doesn't invert)
     left  -> project Z,Y onto -X view (side view from person's LEFT cheek)
     right -> project -Z,Y onto +X view (side view from person's RIGHT cheek)

  2. Server-side input image preprocessing:
       - flip_h / flip_v (mirror before bake)
       - brightness (-100..100) applied as PIL ImageEnhance.Brightness (0..2)
       - contrast   (-100..100) applied as PIL ImageEnhance.Contrast   (0..2)

     Frontend also bakes B/C for preview; server re-applies so image_url callers
     work identically.

Interface stays compatible with module_wrap_texture (V18) -- V21 config is a
superset. Callers that don't set wrap_direction get 'front' == old +Z behavior.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Reuse V18 primitives (unwrap_uvs, ortho bake helpers). Import lazily inside
# run_module_wrap_v21 to keep import surface minimal.


@dataclass
class ModuleWrapV21Config:
    input_mesh: Path
    input_image: Path
    out_stem: Path

    target_height_mm: float = 70.0
    atlas_res: int = 8192
    xatlas_target_faces: int = 60_000
    xatlas_timeout_s: int = 180
    clahe_clip: float = 3.0
    clahe_tile: int = 16
    dry_run: bool = False

    # V21 explicit controls (replaces V19 wrap_auto_detect_view_dir).
    wrap_direction: str = "front"   # front | back | left | right
    flip_h: bool = False
    flip_v: bool = False
    brightness: float = 0.0         # -100..100
    contrast: float = 0.0           # -100..100

    def __post_init__(self):
        self.input_mesh = Path(self.input_mesh)
        self.input_image = Path(self.input_image)
        self.out_stem = Path(self.out_stem)
        if self.wrap_direction not in ("front", "back", "left", "right"):
            raise ValueError(f"wrap_direction must be front|back|left|right, got {self.wrap_direction!r}")


def _preprocess_image(pil_img, flip_h: bool, flip_v: bool,
                      brightness: float, contrast: float):
    """Apply flip + brightness + contrast in place-safe fashion. Returns new PIL image."""
    from PIL import ImageOps, ImageEnhance
    img = pil_img
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    # brightness: -100 -> 0 (black), 0 -> 1 (no change), +100 -> 2 (double)
    b_factor = 1.0 + max(-100.0, min(100.0, brightness)) / 100.0
    if abs(b_factor - 1.0) > 1e-3:
        img = ImageEnhance.Brightness(img).enhance(b_factor)
    c_factor = 1.0 + max(-100.0, min(100.0, contrast)) / 100.0
    if abs(c_factor - 1.0) > 1e-3:
        img = ImageEnhance.Contrast(img).enhance(c_factor)
    return img


def _axes_for_direction(direction: str):
    """Return (u_axis_idx, v_axis_idx, u_sign, v_sign, proj_axis_idx, proj_sign).
    u,v are the 2D projection axes; proj is the camera view axis; signs handle mirroring.

      front : project (X, Y) onto -Z          -> u=(0,+1) v=(1,+1) proj=(2,-1)
      back  : project (-X, Y) onto +Z         -> u=(0,-1) v=(1,+1) proj=(2,+1)
      left  : project (Z, Y) onto -X          -> u=(2,+1) v=(1,+1) proj=(0,-1)
      right : project (-Z, Y) onto +X         -> u=(2,-1) v=(1,+1) proj=(0,+1)
    """
    if direction == "front":
        return 0, 1, +1.0, +1.0, 2, -1.0
    if direction == "back":
        return 0, 1, -1.0, +1.0, 2, +1.0
    if direction == "left":
        return 2, 1, +1.0, +1.0, 0, -1.0
    if direction == "right":
        return 2, 1, -1.0, +1.0, 0, +1.0
    raise ValueError(f"bad direction {direction!r}")


def bake_ortho_texture_v21(photo_pil, verts_world, uvs, faces_uv, vmapping,
                           atlas_res, wrap_direction="front",
                           clahe_clip=3.0, clahe_tile=16):
    """V21 ortho bake with explicit direction (no auto-detect).

    Same math as V18 bake_ortho_texture but with wrap_direction -> axis mapping.
    """
    import numpy as np
    from PIL import Image
    import cv2

    photo_np = np.array(photo_pil.convert("RGB"))
    img_h, img_w = photo_np.shape[:2]

    gray = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    gray_eq = clahe.apply(gray)
    photo_gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

    u_i, v_i, u_s, v_s, p_i, p_s = _axes_for_direction(wrap_direction)

    # world_xy analogue for chosen direction
    world_uv = np.stack([verts_world[:, u_i] * u_s,
                          verts_world[:, v_i] * v_s], axis=1).astype(np.float32)
    proj_axis = (verts_world[:, p_i] * p_s).astype(np.float32)

    v_min = world_uv.min(0); v_max = world_uv.max(0)
    v_span = float(np.max(v_max - v_min)); v_span = max(v_span, 1e-9)
    v_center = (v_min + v_max) / 2.0
    img_center = np.array([img_w / 2.0, img_h / 2.0])
    img_scale = min(img_w, img_h) / v_span

    def w2i(pts):
        cent = pts - v_center
        cent[:, 1] *= -1.0
        return cent * img_scale + img_center

    atlas = np.full((atlas_res, atlas_res, 3), 128, dtype=np.uint8)
    coverage = np.zeros((atlas_res, atlas_res), dtype=np.uint8)

    tri_orig_idx = vmapping[faces_uv]
    img_pts_all = w2i(world_uv[tri_orig_idx.reshape(-1)]).reshape(-1, 3, 2)
    atlas_pts_all = (uvs[faces_uv] * atlas_res).astype(np.float32)

    # front-facing selection using proj_axis
    tri_proj = proj_axis[tri_orig_idx].reshape(-1, 3).mean(axis=1)
    front_mask = tri_proj > 0.0    # >0 == facing camera (proj_axis already signed)

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

    print(f"      V21 dir={wrap_direction} baked {n_baked}/{len(faces_uv)} tris "
          f"(skipped {int((~front_mask).sum())} back-facing)", flush=True)
    return atlas, coverage


def run_module_wrap_v21(cfg: ModuleWrapV21Config):
    """V21 wrap: preprocess image (flip/BC) + explicit direction bake + export.

    Output files mirror V18 module_wrap: .obj/.mtl/_texture.png/.stl/.glb next to out_stem.
    """
    print("=" * 66)
    print("  Module WRAP TEXTURE (V21) -- explicit direction + input preproc")
    print("=" * 66)
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  input image  : {cfg.input_image}")
    print(f"  out_stem     : {cfg.out_stem}")
    print(f"  wrap_direction : {cfg.wrap_direction}")
    print(f"  flip_h={cfg.flip_h} flip_v={cfg.flip_v} "
          f"brightness={cfg.brightness} contrast={cfg.contrast}")
    print(f"  atlas_res    : {cfg.atlas_res}  target_faces: {cfg.xatlas_target_faces}")
    print("=" * 66, flush=True)
    if cfg.dry_run: print("[dry-run] validated"); return

    t0 = time.time()
    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.texture import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    # Reuse V18 unwrap primitives
    from module_wrap_texture import unwrap_uvs

    cfg.out_stem.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load + scale mesh
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

    # 2. xatlas UV unwrap (V18 proven pattern)
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

    # 3. Preprocess input image + ortho bake with V21 direction axis
    print(f"\n[3/5] Preprocess + ortho bake dir={cfg.wrap_direction} "
          f"{cfg.atlas_res}x{cfg.atlas_res}", flush=True)
    photo = Image.open(cfg.input_image).convert("RGB")
    photo = _preprocess_image(photo, cfg.flip_h, cfg.flip_v, cfg.brightness, cfg.contrast)
    atlas_rgb, coverage = bake_ortho_texture_v21(
        photo, dec_verts, uvs, faces_uv, vmapping,
        cfg.atlas_res, wrap_direction=cfg.wrap_direction,
        clahe_clip=cfg.clahe_clip, clahe_tile=cfg.clahe_tile,
    )
    atlas_for_gltf = atlas_rgb[::-1].copy()   # glTF UV convention

    # 4. Build textured mesh
    print("\n[4/5] Build textured mesh", flush=True)
    xatlas_verts = dec_verts[vmapping]
    mat = SimpleMaterial(image=Image.fromarray(atlas_for_gltf))
    tex_mesh = trimesh.Trimesh(
        vertices=xatlas_verts, faces=faces_uv,
        visual=TextureVisuals(uv=uvs, image=Image.fromarray(atlas_for_gltf), material=mat),
        process=False,
    )
    geo_mesh = trimesh.Trimesh(vertices=xatlas_verts, faces=faces_uv, process=False)

    # 5. Export (same set as V18)
    print("\n[5/5] Export OBJ + MTL + PNG + STL + GLB", flush=True)
    stem = cfg.out_stem
    obj_path = stem.with_suffix(".obj")
    tex_mesh.export(obj_path)
    stl_path = stem.with_suffix(".stl")
    geo_mesh.export(stl_path)
    glb_path = stem.with_suffix(".glb")
    tex_mesh.export(glb_path)

    print(f"\n[done] V21 wrap in {time.time()-t0:.2f}s", flush=True)
    print(f"       OBJ: {obj_path}")
    print(f"       STL: {stl_path}")
    print(f"       GLB: {glb_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out-stem", required=True, type=Path)
    ap.add_argument("--wrap-direction", default="front",
                    choices=["front", "back", "left", "right"])
    ap.add_argument("--flip-h", action="store_true")
    ap.add_argument("--flip-v", action="store_true")
    ap.add_argument("--brightness", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.0)
    ap.add_argument("--atlas-res", type=int, default=8192)
    ap.add_argument("--target-height-mm", type=float, default=70.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = ModuleWrapV21Config(
        input_mesh=args.mesh, input_image=args.image, out_stem=args.out_stem,
        wrap_direction=args.wrap_direction,
        flip_h=args.flip_h, flip_v=args.flip_v,
        brightness=args.brightness, contrast=args.contrast,
        atlas_res=args.atlas_res, target_height_mm=args.target_height_mm,
        dry_run=args.dry_run,
    )
    run_module_wrap_v21(cfg)
