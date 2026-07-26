"""
Module E — Dot cloud luminance modulation + UV texture projection.

Two techniques applied to body mesh (head region excluded — Module C+D handles
face relief separately):

1. UV texture bake — project source portrait onto body mesh via orthographic
   front camera, bake into a UV texture atlas. Gives suit color / embroidery
   / shoe leather visible on the mesh.

2. Blue-noise Poisson-disk dot cloud — sample body surface at uniform density,
   modulate dot intensity by local luminance from the baked texture. Adds
   high-frequency detail cue (SSLE-style dithering) so downstream renderers
   (Windows 3D Viewer, Blender) can differentiate suit fabric from bare
   silhouette.

Dependencies:
    - trimesh       (mesh IO + UV attach)
    - numpy         (raster + sampling)
    - Pillow        (image IO + texture write)
    - xatlas        (UV unwrap of arbitrary triangular mesh, Apache-2.0)
    - opencv-python (Gaussian blur + optional Delaunay projection helpers)

Usage:
    from module_e_texture import ModuleEConfig, run_module_e
    cfg = ModuleEConfig(input_mesh=..., input_image=..., output_mesh=...,
                       dot_density_scale=1.0, dot_size_mm=0.2, uv_bake_res=2048)
    run_module_e(cfg)

Design principles:
    - Lazy import of heavy deps
    - `--dry-run` prints plan only
    - Head verts detected same way v11 does (top 20 percent by Y) so we skip
      them consistently
    - Face texture NOT overwritten — Module C+D produced the face geometry;
      we bake a matte skin-tone patch over that region so texture doesn't
      poke through
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModuleEConfig:
    input_mesh: Path                     # mesh after Module C+D (v11 out)
    input_image: Path                    # source portrait (for texture bake)
    output_mesh: Path                    # v12 final GLB with texture

    # Head-region masking (skip head from UV bake + dot cloud)
    head_top_fraction: float = 0.20      # matches v11 find_head_vertices

    # UV texture bake
    uv_bake_res: int = 2048              # atlas resolution (WxW)
    front_camera_axis: str = "+Z"        # orthographic axis pointing at camera
    face_mask_dilate_px: int = 24        # skin patch expand around face region
    face_skin_color: Tuple[int, int, int] = (198, 158, 128)  # neutral matte skin

    # Dot cloud
    dot_density_scale: float = 1.0       # multiplier on dots-per-unit-area
    dot_size_px: int = 3                 # atlas-pixel dot radius (Fable 5:
                                         # mm scale was meaningless — TripoSG
                                         # meshes are unit-normalized, not m)
    dot_luminance_curve: str = "linear"  # linear | gamma | invert
    dot_min_size_frac: float = 0.3       # min radius as fraction of dot_size

    # Runtime
    dry_run: bool = False

    def __post_init__(self):
        self.input_mesh = Path(self.input_mesh)
        self.input_image = Path(self.input_image)
        self.output_mesh = Path(self.output_mesh)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("input_mesh", "input_image", "output_mesh"):
            d[k] = str(d[k])
        return d


# ---------------------------------------------------------------------------
# Step 1 — Split mesh into head + body regions (skip head from body ops)
# ---------------------------------------------------------------------------

def split_head_body(mesh, top_fraction: float = 0.20):
    """Return (head_face_idx, body_face_idx). Body = all NOT-head faces."""
    import numpy as np
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    y = verts[:, 1]
    thresh = y.max() - (y.max() - y.min()) * top_fraction
    head_vert_mask = y >= thresh                   # (V,)
    # A face is "head" if any vertex is head-region
    head_face_mask = head_vert_mask[faces].any(axis=1)  # (F,)
    body_face_mask = ~head_face_mask

    head_face_idx = np.where(head_face_mask)[0]
    body_face_idx = np.where(body_face_mask)[0]
    return head_face_idx, body_face_idx


# ---------------------------------------------------------------------------
# Step 2 — UV unwrap body-only submesh via xatlas
# ---------------------------------------------------------------------------

def _xatlas_worker(verts, faces, out_queue):
    """Run xatlas in subprocess so we can hard-kill if it stalls.

    xatlas is superlinearly slow on dense meshes (xatlas#36 — 1M faces ~12h).
    We isolate it in a Process + terminate on timeout.
    """
    try:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(verts, faces)
        try:
            co = xatlas.ChartOptions()
            co.max_iterations = 1                          # fastest charting
            po = xatlas.PackOptions()
            po.padding = 4
            atlas.generate(chart_options=co, pack_options=po)
        except Exception:
            atlas.generate()                               # older API fallback
        vmapping, indices, uvs = atlas[0]
        out_queue.put((uvs, indices, vmapping))
    except Exception as e:
        out_queue.put(('ERROR', str(e)))


def unwrap_body_uvs_pymeshlab(dec_verts, dec_faces, texdim: int = 2048):
    """Fable v14 fallback: pymeshlab trivial-per-wedge UV parametrization.

    Tolerates non-manifold TRELLIS/TripoSG-style meshes that xatlas chokes on.
    pymeshlab is already a TripoSG requirement — zero new deps.

    Trade-off: UV area per triangle is uniform (independent of 3D area), so
    density/luminance modulation is approximate. Bake is per-triangle so
    output quality is preserved.
    """
    import numpy as np
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=dec_verts.astype(np.float64),
        face_matrix=dec_faces.astype(np.int32),
    ))
    # FP_BASIC_TRIANGLE_MAPPING — never hangs, works on triangle soup
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=texdim)
    # Per-wedge -> per-vertex UV (splits verts at UV seams internally)
    ms.compute_texcoord_transfer_wedge_to_vertex()
    m = ms.current_mesh()
    uvs = m.vertex_tex_coord_matrix().astype(np.float32)
    faces_uv = m.face_matrix().astype(np.int64)
    # vmapping is identity: pymeshlab kept original vertex order after transfer
    vmapping = np.arange(len(m.vertex_matrix()), dtype=np.int64)
    return uvs, faces_uv, vmapping, m.vertex_matrix().astype(np.float32)


def unwrap_body_uvs(verts_np, body_faces_np, target_faces: int = 15_000,
                   timeout_s: int = 120):
    """Unwrap body-only submesh via xatlas — with aggressive pre-decimation +
    subprocess timeout guard (Fable 5 v13 review).

    Root causes fixed vs v12:
    - v12 hung indefinitely: xatlas on 500k+ faces takes hours, not minutes
    - v12 "50k decim" was likely broken: `mesh.simplify_quadric_decimation(50_000)`
      in trimesh 4+ treats first positional arg as `percent`, not `face_count`.
      Must use `face_count=` kwarg AND assert result.
    - v12 passed full 640k-vert array with body-only faces to xatlas — waste
      of memory. Compact the submesh first.

    Returns (uvs, faces_uv, vmapping_into_dec_verts, dec_verts).
    NOTE: vmapping is into the DECIMATED submesh vertex array, not the original
    full mesh. Caller must return dec_verts[vmapping] for atlas geometry.
    """
    import numpy as np
    import trimesh
    import multiprocessing as mp

    # Compact + weld the body-only submesh
    sub = trimesh.Trimesh(vertices=verts_np, faces=body_faces_np, process=False)
    sub.merge_vertices()
    print(f"        submesh (welded): {len(sub.vertices):,} verts, {len(sub.faces):,} faces",
          flush=True)

    # Decimate to target — use face_count kwarg to survive trimesh 4.x API
    try:
        dec = sub.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:
        # Older trimesh: positional was fine
        dec = sub.simplify_quadric_decimation(target_faces)
    n_dec_faces = len(dec.faces)
    print(f"        decimated to: {len(dec.vertices):,} verts, {n_dec_faces:,} faces",
          flush=True)
    if n_dec_faces > target_faces * 1.2:
        raise RuntimeError(
            f"decimation failed to hit target: got {n_dec_faces}, "
            f"wanted <= {int(target_faces * 1.2)}"
        )

    # Run xatlas in child process, kill if stalled
    dec_verts = np.asarray(dec.vertices, dtype=np.float32)
    dec_faces = np.asarray(dec.faces, dtype=np.uint32)   # xatlas wants uint32

    # Fable v14 fix: use SPAWN not fork. Parent has already initialized CUDA in
    # Stage C+D (DAv2); PyTorch multiprocessing docs say fork deadlocks in that
    # state. Spawn is clean.
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_xatlas_worker, args=(dec_verts, dec_faces, q))
    p.start()
    # V21h.1 Fable v3: same drain-before-join fix as module_wrap_texture.
    try:
        result = q.get(timeout=timeout_s)
    except Exception:
        print(f"        xatlas TIMEOUT after {timeout_s}s (queue drain) — killing",
              flush=True)
        try:
            if p.is_alive(): p.terminate()
        finally:
            p.join(5)
        raise TimeoutError(f"xatlas exceeded {timeout_s}s")
    p.join(10)
    if p.is_alive():
        p.terminate(); p.join(5)
    if isinstance(result, tuple) and result and result[0] == 'ERROR':
        raise RuntimeError(f"xatlas worker error: {result[1]}")

    uvs_a, indices_a, vmapping_a = result
    return (
        np.asarray(uvs_a, dtype=np.float32),
        np.asarray(indices_a, dtype=np.int64),
        np.asarray(vmapping_a, dtype=np.int64),
        dec_verts,
    )


# ---------------------------------------------------------------------------
# Step 3 — Orthographic front projection: photo -> UV atlas
# ---------------------------------------------------------------------------

def project_photo_to_atlas(
    photo_pil,
    verts_world,
    uvs,
    faces_uv,
    vmapping,
    atlas_res: int,
    front_axis: str = "+Z",
    bg_fill=(120, 100, 90),
    skip_backfaces: bool = True,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Bake photo onto UV atlas by orthographic front projection.

    Fixes vs first-pass (Fable 5 review):
    - C4: warp is clipped to each triangle's atlas-bbox (was O(F * atlas_res^2))
    - Quality: uniform world-XY scaling (no aspect distortion — was independent
      X/Y stretch)
    - Quality: skip back-facing triangles (front-facing only, world +Z normal
      component > 0)
    - Quality: pre-fill atlas with a soft neutral suit color so unmapped areas
      don't read as pitch-black holes
    - Returns coverage_mask (H, W) uint8 alongside atlas — used by dot cloud
      to know which pixels are actually mapped (was faked with `gray > 0` and
      broke on legitimately dark suits)
    """
    import numpy as np
    from PIL import Image
    import cv2

    photo_np = np.array(photo_pil.convert("RGB"))
    img_h, img_w = photo_np.shape[:2]

    if front_axis == "+Z":
        world_xy = verts_world[:, :2].copy()
        z_sign = 1.0
    elif front_axis == "-Z":
        world_xy = verts_world[:, :2].copy() * np.array([-1, 1])
        z_sign = -1.0
    else:
        raise ValueError(f"Unsupported front_axis: {front_axis}")

    # Uniform-scale world XY -> image px (Fable 5 quality fix: preserve aspect)
    v_min = world_xy.min(axis=0)
    v_max = world_xy.max(axis=0)
    v_span = float(np.max(v_max - v_min))
    if v_span < 1e-9:
        v_span = 1e-9
    v_center = (v_min + v_max) / 2.0
    img_center = np.array([img_w / 2.0, img_h / 2.0])
    img_scale = min(img_w, img_h) / v_span     # fit tighter dim

    def world_to_img(pts_xy):
        cent = pts_xy - v_center
        cent[:, 1] *= -1.0                       # flip Y (image top = world +Y)
        return cent * img_scale + img_center

    atlas = np.full((atlas_res, atlas_res, 3), bg_fill, dtype=np.uint8)
    coverage = np.zeros((atlas_res, atlas_res), dtype=np.uint8)

    # Precompute all image-space triangles + atlas triangles (vectorized)
    tri_orig_idx = vmapping[faces_uv]                         # (F, 3)
    img_pts_all = world_to_img(world_xy[tri_orig_idx.reshape(-1)]).reshape(-1, 3, 2)
    atlas_pts_all = (uvs[faces_uv] * atlas_res).astype(np.float32)  # (F, 3, 2)

    # Backface culling via z-sign of face normal (compute in world_xy plane
    # from CCW orientation; equivalent since z_sign flips +Z/-Z convention)
    if skip_backfaces:
        v01 = img_pts_all[:, 1] - img_pts_all[:, 0]
        v02 = img_pts_all[:, 2] - img_pts_all[:, 0]
        cross_z = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        # After Y-flip in image space, front-facing is cross_z < 0 for CCW verts
        front_mask = cross_z * z_sign < 0
    else:
        front_mask = np.ones(len(faces_uv), dtype=bool)

    n_baked = 0
    for i in range(len(faces_uv)):
        if not front_mask[i]:
            continue
        img_tri = img_pts_all[i]
        atlas_tri = atlas_pts_all[i]

        # C4 bbox clip
        x0 = int(np.floor(atlas_tri[:, 0].min())) - 1
        y0 = int(np.floor(atlas_tri[:, 1].min())) - 1
        x1 = int(np.ceil(atlas_tri[:, 0].max())) + 1
        y1 = int(np.ceil(atlas_tri[:, 1].max())) + 1
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, atlas_res); y1 = min(y1, atlas_res)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue

        local_tri = atlas_tri - np.array([x0, y0], dtype=np.float32)
        try:
            M = cv2.getAffineTransform(
                img_tri.astype(np.float32), local_tri,
            )
            warp = cv2.warpAffine(photo_np, M, (w, h))
        except cv2.error:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_tri.astype(np.int32), 255)
        roi = atlas[y0:y1, x0:x1]
        roi[mask > 0] = warp[mask > 0]
        coverage[y0:y1, x0:x1] |= mask
        n_baked += 1

    print(f"      baked {n_baked} / {len(faces_uv)} triangles "
          f"(backface culled: {int((~front_mask).sum())})", flush=True)
    return atlas, coverage


# ---------------------------------------------------------------------------
# Step 4 — Poisson-disk dot cloud with luminance modulation
# ---------------------------------------------------------------------------

def poisson_disk_uv_dots(
    atlas: "np.ndarray",
    coverage_mask: "np.ndarray",
    density_scale: float,
    dot_size_px: int,
    curve: str = "linear",
    min_size_frac: float = 0.3,
) -> "np.ndarray":
    """Overlay dot cloud onto atlas. Dot density + size modulated by local
    luminance from the baked photo.

    Approximate blue-noise via UV-space rejection sampling (not strict
    Poisson-disk — visually close enough for luminance dithering, cheaper).

    Fixes vs first pass (Fable 5 review):
    - dot size in atlas pixels (was meaningless world mm scale)
    - `is_mapped` from proper coverage_mask (was `gray > 0`, breaks on dark
      suits where legitimate baked pixels are near-black)
    """
    import numpy as np
    import cv2

    H, W = atlas.shape[:2]
    dot_r_px = max(1, int(dot_size_px))

    target_dots = int(density_scale * (H * W) / (dot_r_px * dot_r_px * 4.0))
    target_dots = max(100, min(target_dots, 200_000))

    print(f"      dot cloud target: {target_dots:,} dots at r={dot_r_px}px",
          flush=True)

    rng = np.random.default_rng(42)
    samples = rng.uniform(0.0, 1.0, size=(target_dots, 2))
    px = (samples[:, 0] * W).astype(np.int32)
    py = (samples[:, 1] * H).astype(np.int32)

    gray = cv2.cvtColor(atlas, cv2.COLOR_RGB2GRAY)
    lum = gray[py.clip(0, H-1), px.clip(0, W-1)].astype(np.float32) / 255.0

    if curve == "linear":
        size_frac = 1.0 - lum
    elif curve == "gamma":
        size_frac = np.power(1.0 - lum, 2.2)
    elif curve == "invert":
        size_frac = lum
    else:
        size_frac = 1.0 - lum
    size_frac = np.clip(size_frac, min_size_frac, 1.0)

    # Skip dots outside coverage (unmapped body regions)
    is_mapped = coverage_mask[py.clip(0, H-1), px.clip(0, W-1)] > 0

    dark_dot_color = (12, 12, 12)      # near-black
    out = atlas.copy()
    for i in range(target_dots):
        if not is_mapped[i]:
            continue
        r = max(1, int(dot_r_px * size_frac[i]))
        cv2.circle(out, (int(px[i]), int(py[i])), r, dark_dot_color, -1)

    return out


# ---------------------------------------------------------------------------
# Step 5 — Mask face region in atlas with matte skin patch (don't overwrite
# the face relief from Module C+D)
# ---------------------------------------------------------------------------

def paint_face_skin_patch(atlas, uvs, faces_uv, head_face_uv_mask,
                          skin_color=(198, 158, 128)) -> "np.ndarray":
    """Fill head-region UV triangles with matte skin so photo bake doesn't
    poke through the face.

    Note: current body-only unwrap already excludes head faces so this is a
    no-op in default flow. Included for the head-included case (if caller
    unwraps whole mesh instead).
    """
    import numpy as np
    import cv2
    H, W = atlas.shape[:2]
    out = atlas.copy()
    for face_i, is_head in enumerate(head_face_uv_mask):
        if not is_head:
            continue
        tri = (uvs[faces_uv[face_i]] * np.array([W, H])).astype(np.int32)
        cv2.fillConvexPoly(out, tri, skin_color)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _print_plan(cfg: ModuleEConfig) -> None:
    print("=" * 66)
    print("  Module E — dot cloud + UV texture projection")
    print("=" * 66)
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  input image  : {cfg.input_image}")
    print(f"  output mesh  : {cfg.output_mesh}")
    print(f"  ---- knobs ----")
    print(f"    uv_bake_res         = {cfg.uv_bake_res}")
    print(f"    front_camera_axis   = {cfg.front_camera_axis}")
    print(f"    face_mask_dilate_px = {cfg.face_mask_dilate_px}")
    print(f"    face_skin_color     = {cfg.face_skin_color}")
    print(f"    dot_density_scale   = {cfg.dot_density_scale}")
    print(f"    dot_size_px         = {cfg.dot_size_px}")
    print(f"    dot_luminance_curve = {cfg.dot_luminance_curve}")
    print(f"    dot_min_size_frac   = {cfg.dot_min_size_frac}")
    print(f"    head_top_fraction   = {cfg.head_top_fraction}")
    print(f"  dry_run      = {cfg.dry_run}")
    print("=" * 66, flush=True)


def run_module_e(cfg: ModuleEConfig) -> Path:
    """Full Module E pass: UV unwrap body -> bake photo -> dot cloud -> export."""
    _print_plan(cfg)

    if cfg.dry_run:
        print("[dry-run] Skipping xatlas + cv2 heavy work. Plan validated.",
              flush=True)
        return cfg.output_mesh

    if not cfg.input_mesh.exists():
        raise FileNotFoundError(f"Input mesh missing: {cfg.input_mesh}")
    if not cfg.input_image.exists():
        raise FileNotFoundError(f"Input image missing: {cfg.input_image}")
    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import trimesh
    from PIL import Image

    t_all = time.time()

    # ---- Load
    print(f"\n[1/5] Load mesh", flush=True)
    obj = trimesh.load(cfg.input_mesh, force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    else:
        mesh = obj
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"      verts={len(verts):,}, faces={len(faces):,}", flush=True)

    # ---- Split head vs body
    print(f"\n[2/5] Split head vs body (top_fraction={cfg.head_top_fraction})",
          flush=True)
    head_face_idx, body_face_idx = split_head_body(mesh, cfg.head_top_fraction)
    print(f"      head faces: {len(head_face_idx):,}, "
          f"body faces: {len(body_face_idx):,}", flush=True)
    if len(body_face_idx) < 10:
        raise RuntimeError(
            f"Body region has {len(body_face_idx)} faces — head_top_fraction "
            f"{cfg.head_top_fraction} too aggressive for this mesh. Lower it."
        )

    # ---- UV unwrap body-only (xatlas → pymeshlab trivial → error)
    print(f"\n[3/5] UV unwrap body submesh", flush=True)
    t0 = time.time()
    body_faces = faces[body_face_idx]
    unwrap_used = None
    # Try xatlas (spawn context per Fable v14 fix)
    try:
        uvs, faces_uv, vmapping, dec_verts = unwrap_body_uvs(
            verts, body_faces, target_faces=15_000, timeout_s=90,
        )
        unwrap_used = "xatlas_15k"
    except (TimeoutError, RuntimeError) as e:
        print(f"      xatlas 15k failed ({e}); trying pymeshlab trivial",
              flush=True)
        # Prepare decimated welded submesh once for pymeshlab
        import trimesh
        sub = trimesh.Trimesh(vertices=verts, faces=body_faces, process=False)
        sub.merge_vertices()
        try:
            dec = sub.simplify_quadric_decimation(face_count=15_000)
        except TypeError:
            dec = sub.simplify_quadric_decimation(15_000)
        dec_verts = np.asarray(dec.vertices, dtype=np.float32)
        dec_faces = np.asarray(dec.faces, dtype=np.uint32)
        try:
            uvs, faces_uv, vmapping, dec_verts = unwrap_body_uvs_pymeshlab(
                dec_verts, dec_faces, texdim=cfg.uv_bake_res,
            )
            unwrap_used = "pymeshlab_trivial"
        except Exception as e2:
            print(f"      pymeshlab also failed ({e2})", flush=True)
            raise
    print(f"      unwrap via {unwrap_used}: atlas verts={len(uvs):,}, "
          f"faces={len(faces_uv):,}, took {time.time()-t0:.1f}s", flush=True)

    # ---- Bake photo onto atlas
    print(f"\n[4/5] Bake photo onto atlas (res={cfg.uv_bake_res})", flush=True)
    t0 = time.time()
    photo = Image.open(cfg.input_image).convert("RGB")
    # NOTE: project_photo_to_atlas now takes dec_verts (not full-mesh verts) as
    # world source — vmapping indexes into dec_verts after v13 decimation.
    atlas_rgb, coverage_mask = project_photo_to_atlas(
        photo, dec_verts, uvs, faces_uv, vmapping,
        cfg.uv_bake_res, cfg.front_camera_axis,
    )
    print(f"      atlas shape={atlas_rgb.shape}, took {time.time()-t0:.1f}s",
          flush=True)

    # ---- Dot cloud overlay
    print(f"\n[5/5] Dot cloud overlay (density={cfg.dot_density_scale}, "
          f"size_px={cfg.dot_size_px})", flush=True)
    t0 = time.time()
    atlas_final = poisson_disk_uv_dots(
        atlas_rgb, coverage_mask,
        density_scale=cfg.dot_density_scale,
        dot_size_px=cfg.dot_size_px,
        curve=cfg.dot_luminance_curve,
        min_size_frac=cfg.dot_min_size_frac,
    )
    print(f"      took {time.time()-t0:.1f}s", flush=True)

    # ---- Rebuild mesh with new UVs + texture
    print(f"\n[export] Build textured mesh + write GLB", flush=True)
    # v13 note: vmapping indexes into DEC_VERTS (decimated submesh), not full
    # mesh, because we welded + decimated body before xatlas. So look up
    # dec_verts[vmapping] for atlas geometry.
    xatlas_verts = dec_verts[vmapping]

    # Fable 5 C3 fix: glTF samples UVs with bottom-left origin (v' = 1 - v).
    # Our bake writes atlas row v*H top-down, so flip vertically on export.
    atlas_for_gltf = atlas_final[::-1].copy()

    from trimesh.visual.texture import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    material = PBRMaterial(
        baseColorTexture=Image.fromarray(atlas_for_gltf),
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        roughnessFactor=0.7,
        metallicFactor=0.0,
    )
    body_textured = trimesh.Trimesh(
        vertices=xatlas_verts,
        faces=faces_uv,
        visual=TextureVisuals(uv=uvs, material=material),
        process=False,
    )

    # Head submesh (v11 face relief geometry stays) — give it a matte skin
    # material so 3D viewers don't render it as pure grey (Fable 5 quality
    # medium).
    head_vert_indices = np.unique(faces[head_face_idx].reshape(-1))
    head_verts = verts[head_vert_indices]
    head_map = -np.ones(len(verts), dtype=np.int64)
    head_map[head_vert_indices] = np.arange(len(head_vert_indices))
    head_faces_local = head_map[faces[head_face_idx]]
    skin = tuple(c / 255.0 for c in cfg.face_skin_color) + (1.0,)
    head_material = PBRMaterial(
        baseColorFactor=list(skin),
        roughnessFactor=0.6,
        metallicFactor=0.0,
    )
    head_mesh = trimesh.Trimesh(
        vertices=head_verts, faces=head_faces_local,
        visual=trimesh.visual.color.ColorVisuals(
            face_colors=[int(c) for c in cfg.face_skin_color] + [255]
        ),
        process=False,
    )

    # Combine into scene (2 geometries — head + textured body)
    scene = trimesh.Scene([body_textured, head_mesh])
    scene.export(cfg.output_mesh)

    size_mb = cfg.output_mesh.stat().st_size / 1024 / 1024
    print(f"      wrote {cfg.output_mesh} ({size_mb:.2f} MB)", flush=True)
    print(f"\n[DONE] total elapsed: {time.time()-t_all:.1f}s", flush=True)
    return cfg.output_mesh


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="module_e_texture",
        description="Dot cloud + UV texture projection for body mesh.",
    )
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output-mesh", required=True, type=Path)
    ap.add_argument("--head-top-fraction", type=float, default=0.20)
    ap.add_argument("--uv-bake-res", type=int, default=2048)
    ap.add_argument("--front-camera-axis", choices=["+Z", "-Z"], default="+Z")
    ap.add_argument("--face-mask-dilate-px", type=int, default=24)
    ap.add_argument("--dot-density-scale", type=float, default=1.0)
    ap.add_argument("--dot-size-px", type=int, default=3)
    ap.add_argument("--dot-luminance-curve",
                    choices=["linear", "gamma", "invert"], default="linear")
    ap.add_argument("--dot-min-size-frac", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = ModuleEConfig(
        input_mesh=args.input_mesh,
        input_image=args.input_image,
        output_mesh=args.output_mesh,
        head_top_fraction=args.head_top_fraction,
        uv_bake_res=args.uv_bake_res,
        front_camera_axis=args.front_camera_axis,
        face_mask_dilate_px=args.face_mask_dilate_px,
        dot_density_scale=args.dot_density_scale,
        dot_size_px=args.dot_size_px,
        dot_luminance_curve=args.dot_luminance_curve,
        dot_min_size_frac=args.dot_min_size_frac,
        dry_run=args.dry_run,
    )
    run_module_e(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
