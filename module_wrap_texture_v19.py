"""
Module WRAP TEXTURE V19 -- camera-space direct UV + neutral back-face + Sobel normal map.

Fixes V18 xatlas misalignment: xatlas packs UV charts by angle heuristics with zero
regard for camera projection. Our texture is one ortho front photo, so xatlas layout
puts random pixels on random triangles.

V19: UV = image space. For every front-facing vertex, uv = normalized (x, 1-y) of
its ortho-front projection = the exact pixel it sits under. Zero misalignment,
mathematically. Back-facing verts route to a neutral-colored corner tile so back
of head shows solid skin/hair tone (no uncanny mirror-face).

Per Fable 5 v3 verdict: skip AO/PBR/displacement (misleading for QA), do 8192 CLAHE +
Sobel normal from luminance (cheap, adds pore/wrinkle detail in viewer). GLB primary
(single-file, no MTL path breakage).

Constraints for reuse on any portrait:
- Photo must be front-facing (subject looks toward camera +Z)
- Whole subject in frame, similar framing to TripoSG mesh
- Resolution ~1-2K works well
- Neutral lighting, no extreme backlight/shadow
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModuleWrapV19Config:
    input_mesh: Path
    input_image: Path
    out_stem: Path                  # writes .obj, .mtl, _texture.png, .stl, .glb, _normal.png
    target_height_mm: float = 70.0
    atlas_res: int = 8192           # per Fable v3 (up from V18's 4096)
    front_axis: str = "+Z"
    auto_detect_view_dir: bool = True     # V20: pick +Z vs -Z by head-normal fraction
    front_normal_threshold: float = 0.15   # v.normal.z >= this = front-facing
    neutral_border_frac: float = 0.05      # right 5% of atlas = neutral color for back-face verts
    clahe_clip: float = 3.0
    clahe_tile: int = 16
    sobel_ksize: int = 3
    sobel_strength: float = 1.0     # normal map amplitude
    dry_run: bool = False

    def __post_init__(self):
        self.input_mesh = Path(self.input_mesh)
        self.input_image = Path(self.input_image)
        self.out_stem = Path(self.out_stem)


def _auto_detect_front_axis(verts, normals, top_frac=0.25, threshold=0.15):
    """V20 fix: pick +Z or -Z by counting how many HEAD verts (top top_frac of Y range)
    have normals pointing that way. Assumes head is at top of mesh (TripoSG convention).
    Returns picked axis + diagnostic (score_pz, score_nz, n_top)."""
    import numpy as np
    y = verts[:, 1]
    y_thresh = y.max() - (y.max() - y.min()) * top_frac
    top_mask = y >= y_thresh
    n_top = int(top_mask.sum())
    if n_top < 10:
        return "+Z", (0.0, 0.0, n_top)  # fallback no clear head
    n_z_top = normals[top_mask, 2]
    score_pz = float((n_z_top > threshold).sum()) / n_top
    score_nz = float((n_z_top < -threshold).sum()) / n_top
    picked = "+Z" if score_pz >= score_nz else "-Z"
    return picked, (score_pz, score_nz, n_top)


def _sobel_normal_map(gray_img, ksize=3, strength=1.0):
    """Build a tangent-space normal map (RGB) from luminance gradient.
    Blue channel = 255 (pointing out); R/G = encoded X/Y gradient."""
    import numpy as np
    import cv2
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=ksize)
    # Normalize gradient to [-1,1]
    m = max(1e-6, float(np.max(np.abs([gx, gy]))))
    gx = np.clip(gx * strength / m, -1.0, 1.0)
    gy = np.clip(gy * strength / m, -1.0, 1.0)
    # Encode to [0,255] with blue=255 (up)
    nx = ((-gx * 0.5 + 0.5) * 255).astype(np.uint8)   # inverted for typical DirectX/OpenGL Y
    ny = ((-gy * 0.5 + 0.5) * 255).astype(np.uint8)
    nz = np.full_like(nx, 255)
    return np.stack([nx, ny, nz], axis=-1)


def run_module_wrap_v19(cfg: ModuleWrapV19Config):
    print("=" * 66)
    print("  Module WRAP TEXTURE V19 -- camera-space direct UV")
    print("=" * 66)
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  input image  : {cfg.input_image}")
    print(f"  out_stem     : {cfg.out_stem}")
    print(f"  atlas_res    : {cfg.atlas_res}  front_axis: {cfg.front_axis}")
    print(f"  front_norm_th: {cfg.front_normal_threshold}")
    print(f"  neutral_frac : {cfg.neutral_border_frac}")
    print(f"  clahe        : clip={cfg.clahe_clip} tile={cfg.clahe_tile}")
    print("=" * 66, flush=True)
    if cfg.dry_run: print("[dry-run] validated"); return

    t0 = time.time()
    import numpy as np
    import trimesh
    import cv2
    from PIL import Image
    from trimesh.visual.texture import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    cfg.out_stem.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load + scale mesh
    print("\n[1/6] Load + scale mesh", flush=True)
    obj = trimesh.load(cfg.input_mesh, force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values()); geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    else: mesh = obj
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bbox_min = verts.min(0); bbox_max = verts.max(0); bbox_size = bbox_max - bbox_min
    s = cfg.target_height_mm / max(bbox_size[1], 1e-6)
    center = (bbox_min + bbox_max) / 2.0
    verts_mm = (verts - center) * s
    mesh_mm = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)
    normals = np.asarray(mesh_mm.vertex_normals, dtype=np.float32)
    print(f"      scale={s:.4f} mm/unit, bbox_mm={(bbox_size*s).round(2).tolist()}", flush=True)

    # V20 auto-detect view axis (fix for V19 misalignment where TripoSG face pointed -Z)
    if cfg.auto_detect_view_dir:
        picked, (score_pz, score_nz, n_top) = _auto_detect_front_axis(
            verts_mm, normals, top_frac=0.25, threshold=cfg.front_normal_threshold
        )
        print(f"      auto-detect: head_verts={n_top:,} score_+Z={score_pz:.3f} score_-Z={score_nz:.3f} -> picked {picked}",
              flush=True)
        if picked != cfg.front_axis:
            print(f"      OVERRIDE: config had front_axis={cfg.front_axis}, using {picked}",
                  flush=True)
            cfg.front_axis = picked

    # 2. Build 8192 texture: photo grayscale-CLAHE in main area, neutral color right strip
    print(f"\n[2/6] Build {cfg.atlas_res}x{cfg.atlas_res} texture (CLAHE + neutral border)", flush=True)
    photo = np.array(Image.open(cfg.input_image).convert("RGB"))
    ph, pw = photo.shape[:2]
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    gray_eq = clahe.apply(gray)
    photo_gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

    # Fit photo into left (1 - neutral_border_frac) of atlas, preserve aspect
    A = cfg.atlas_res
    main_w = int(A * (1.0 - cfg.neutral_border_frac))
    # Scale photo to fit main_w x A (letterbox height if needed)
    photo_scale = min(main_w / pw, A / ph)
    new_w = max(1, int(pw * photo_scale))
    new_h = max(1, int(ph * photo_scale))
    photo_resized = cv2.resize(photo_gray_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Neutral color = median of a large center crop of the photo (skin/hair tone)
    center_crop = photo_gray_rgb[ph//4:3*ph//4, pw//4:3*pw//4]
    neutral_color = tuple(int(x) for x in np.median(center_crop.reshape(-1, 3), axis=0))
    print(f"      neutral (median) color: {neutral_color}", flush=True)

    atlas = np.full((A, A, 3), neutral_color, dtype=np.uint8)
    # Place photo top-left in main area
    atlas[:new_h, :new_w] = photo_resized
    # main_area rect in atlas: [0..new_h, 0..new_w]
    # Photo pixel (px, py) -> atlas pixel (px, py) (0-based)

    # 3. Camera-space UV for front-facing verts, neutral corner for back-facing
    print(f"\n[3/6] Camera-space UV assignment", flush=True)
    z_sign = 1.0 if cfg.front_axis == "+Z" else -1.0
    n_z = normals[:, 2] * z_sign
    is_front = n_z >= cfg.front_normal_threshold
    n_front = int(is_front.sum())
    n_back = len(verts_mm) - n_front
    print(f"      front verts: {n_front:,}/{len(verts_mm):,}  back: {n_back:,}", flush=True)

    # Front UV: project verts_mm.xy to normalized [0,1] within photo bbox, then remap
    # to atlas coords (photo occupies [0, new_w] x [0, new_h] in atlas of size A).
    xy = verts_mm[:, :2].copy()
    if cfg.front_axis == "-Z": xy[:, 0] *= -1.0
    v_min_xy = verts_mm[:, :2].min(0)
    v_max_xy = verts_mm[:, :2].max(0)
    v_range = np.maximum(v_max_xy - v_min_xy, 1e-6)
    uv = np.zeros((len(verts_mm), 2), dtype=np.float32)
    # Front verts: sample from photo region (top-left of atlas)
    # Normalized within photo: [0, new_w/A] x [0, new_h/A], with V flipped (top of photo = top of atlas = uv_v high)
    photo_u_scale = new_w / A
    photo_v_scale = new_h / A
    if n_front > 0:
        nx = (xy[is_front, 0] - v_min_xy[0]) / v_range[0]
        ny = (xy[is_front, 1] - v_min_xy[1]) / v_range[1]
        uv[is_front, 0] = nx * photo_u_scale
        # trimesh/glTF UV origin is bottom-left; our photo top-left is in atlas top-left = uv_v = 1 - new_h/A
        uv[is_front, 1] = (1.0 - photo_v_scale) + ny * photo_v_scale
    # Back verts: point to a single pixel in the neutral strip (right side)
    neutral_u = 1.0 - cfg.neutral_border_frac * 0.5
    neutral_v = 0.5
    if n_back > 0:
        uv[~is_front, 0] = neutral_u
        uv[~is_front, 1] = neutral_v

    # 4. Sobel normal map from grayscale
    print("\n[4/6] Sobel normal map from luminance", flush=True)
    # Build normal map at atlas resolution matching photo region
    normal_atlas = np.full((A, A, 3), (128, 128, 255), dtype=np.uint8)  # neutral flat normal
    gray_resized = cv2.resize(gray_eq, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    normal_region = _sobel_normal_map(gray_resized, cfg.sobel_ksize, cfg.sobel_strength)
    normal_atlas[:new_h, :new_w] = normal_region
    normal_atlas_gltf = normal_atlas[::-1].copy()   # flip V for glTF UV convention

    # 5. Per-vertex color bake (fallback safety) from photo direct sample
    print("\n[5/6] Per-vertex RGB sample", flush=True)
    # Project each vert to photo pixel (same normalization as UV, but into photo pixel space)
    p_ux = np.clip(((xy[:, 0] - v_min_xy[0]) / v_range[0]) * (pw - 1), 0, pw - 1).astype(np.int32)
    p_vy = np.clip(((v_max_xy[1] - xy[:, 1]) / v_range[1]) * (ph - 1), 0, ph - 1).astype(np.int32)
    per_vert_rgb = photo[p_vy, p_ux]  # (V, 3) uint8
    # Back-facing verts: use neutral color to prevent front-photo bleed on occiput
    per_vert_rgb[~is_front] = np.array(neutral_color, dtype=np.uint8)

    # 6. Build meshes + export
    print("\n[6/6] Export GLB (primary) + OBJ+MTL+PNG + STL + normal PNG", flush=True)
    # Atlas flipped for glTF
    atlas_gltf = atlas[::-1].copy()

    # Textured mesh
    mat = SimpleMaterial(image=Image.fromarray(atlas_gltf))
    tex_mesh = trimesh.Trimesh(
        vertices=verts_mm, faces=faces,
        visual=TextureVisuals(uv=uv, image=Image.fromarray(atlas_gltf), material=mat),
        vertex_colors=per_vert_rgb,   # fallback safety
        process=False,
    )
    # Geometry-only mesh
    geo_mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)

    stem = cfg.out_stem
    # GLB PRIMARY per Fable v3
    glb_path = stem.with_suffix(".glb")
    tex_mesh.export(glb_path)
    print(f"      GLB (primary): {glb_path} ({glb_path.stat().st_size} B)", flush=True)

    # OBJ+MTL+PNG (multi-file wavefront)
    obj_path = stem.with_suffix(".obj")
    tex_mesh.export(obj_path)
    tex_png = stem.parent / (stem.name + "_texture.png")
    _obj_extras = list(obj_path.parent.glob("material_*.png"))
    if _obj_extras: _obj_extras[0].rename(tex_png)
    else: Image.fromarray(atlas_gltf).save(tex_png)
    print(f"      OBJ+MTL+PNG: {obj_path} + {tex_png}", flush=True)

    # STL geometry-only
    stl_path = stem.with_suffix(".stl")
    geo_mesh.export(stl_path)
    print(f"      STL: {stl_path}", flush=True)

    # Normal map as separate PNG
    nrm_png = stem.parent / (stem.name + "_normal.png")
    Image.fromarray(normal_atlas_gltf).save(nrm_png)
    print(f"      normal PNG: {nrm_png}", flush=True)

    print(f"\n[DONE V19 wrap] total {time.time()-t0:.1f}s", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(prog="module_wrap_texture_v19")
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--out-stem", required=True, type=Path)
    ap.add_argument("--target-height-mm", type=float, default=70.0)
    ap.add_argument("--atlas-res", type=int, default=8192)
    ap.add_argument("--front-normal-threshold", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg = ModuleWrapV19Config(
        input_mesh=a.input_mesh, input_image=a.input_image, out_stem=a.out_stem,
        target_height_mm=a.target_height_mm, atlas_res=a.atlas_res,
        front_normal_threshold=a.front_normal_threshold, dry_run=a.dry_run,
    )
    run_module_wrap_v19(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
