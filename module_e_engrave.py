"""
Module E ENGRAVE (V16) -- true 3D engrave via geometry displacement.

Replaces V14/V15 Module E (UV texture bake + dot cloud OVERLAY). V16 modulates
vertex positions along per-vertex normals so the mesh itself carries the tonal
information -- no texture atlas involved. Bumps are ~0.1-0.3mm on a unit mesh.

Algorithm:
  1. Orthographic front projection maps each vertex to a photo pixel.
  2. Luminance sampled at that pixel (front-facing verts only via normal.z > 0).
  3. Blue-noise Poisson-disk sample dot positions in UV space using luminance
     as acceptance probability (dark regions attract more dots).
  4. For each dot, find nearby verts within a UV-space radius.
  5. Displace each nearby vert along its normal with Gaussian falloff.
  6. Manifold check: return same faces (topology unchanged).

Design principles:
  - Lazy imports for heavy deps.
  - Deterministic (seed).
  - Skip back-facing verts (they'd get spurious dots from the far side of the
    photo projection).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModuleEEngraveConfig:
    input_mesh: Path
    input_image: Path
    output_mesh: Path

    # Emboss strength as absolute displacement (mesh-unit; TripoSG normalized to
    # ~2 unit tall so 0.002-0.006 ~= 0.1-0.3 mm on a 60cm-tall figurine).
    emboss_strength: float = 0.004

    # Poisson-disk parameters
    dot_count: int = 20000              # target dot count
    dot_min_radius_px: float = 3.0      # min separation in photo pixels
    influence_radius_px: float = 6.0    # verts within this UV pixel distance
                                        # get displacement from each dot

    # Luminance mapping: how dark pixels weight vs bright ones.
    # curve="linear" -> displacement = 1 - lum (dark = tall bump)
    # curve="gamma"  -> displacement = (1 - lum)^gamma_val
    # curve="invert" -> displacement = lum (bright = tall bump)
    lum_curve: str = "linear"
    lum_gamma: float = 2.2

    # Ortho projection axis (matches Module E V14): +Z means camera looks along -Z
    front_axis: str = "+Z"

    # Random seed for dot placement
    seed: int = 42

    dry_run: bool = False

    def __post_init__(self):
        self.input_mesh = Path(self.input_mesh)
        self.input_image = Path(self.input_image)
        self.output_mesh = Path(self.output_mesh)


def _project_ortho(verts, front_axis: str, photo_w: int, photo_h: int):
    """Return (px, py, normal_z_sign) mapping verts to photo pixel coords."""
    import numpy as np
    if front_axis == "+Z":
        world_xy = verts[:, :2].copy()
        z_sign = 1.0
    elif front_axis == "-Z":
        world_xy = verts[:, :2].copy() * np.array([-1.0, 1.0])
        z_sign = -1.0
    else:
        raise ValueError(f"Unsupported front_axis: {front_axis}")
    v_min = world_xy.min(0); v_max = world_xy.max(0)
    v_span = float(np.max(v_max - v_min))
    if v_span < 1e-9: v_span = 1e-9
    v_center = (v_min + v_max) / 2.0
    img_center = np.array([photo_w/2.0, photo_h/2.0], dtype=np.float32)
    scale = min(photo_w, photo_h) / v_span
    xy_centered = world_xy - v_center
    xy_centered[:, 1] *= -1.0
    px_all = xy_centered * scale + img_center
    return px_all[:, 0], px_all[:, 1], z_sign


def run_module_e_engrave(cfg: ModuleEEngraveConfig) -> Path:
    print("=" * 66)
    print("  Module E ENGRAVE (V16) -- geometry displacement dot cloud")
    print("=" * 66)
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  input image  : {cfg.input_image}")
    print(f"  output mesh  : {cfg.output_mesh}")
    print(f"  emboss       : {cfg.emboss_strength} mesh units")
    print(f"  dots         : {cfg.dot_count} min_r_px={cfg.dot_min_radius_px}")
    print(f"  influence_px : {cfg.influence_radius_px}")
    print(f"  curve        : {cfg.lum_curve} gamma={cfg.lum_gamma}")
    print(f"  seed         : {cfg.seed}")
    print(f"  dry_run      : {cfg.dry_run}")
    print("=" * 66, flush=True)
    if cfg.dry_run:
        print("[dry-run] validated"); return cfg.output_mesh

    t_all = time.time()
    import numpy as np
    import trimesh
    import cv2
    from PIL import Image
    from scipy.spatial import cKDTree

    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load mesh + normals
    print("\n[1/6] Load unified mesh + compute vertex normals", flush=True)
    obj = trimesh.load(cfg.input_mesh, force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    else:
        mesh = obj
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    print(f"      verts={len(verts):,} faces={len(faces):,}", flush=True)

    # ---- 2. Load photo + grayscale
    print("\n[2/6] Load photo + luminance", flush=True)
    photo = np.array(Image.open(cfg.input_image).convert("RGB"))
    photo_h, photo_w = photo.shape[:2]
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    print(f"      photo {photo_w}x{photo_h}", flush=True)

    # ---- 3. Ortho project verts to photo pixels + face-forward mask
    print("\n[3/6] Ortho project + face-forward mask", flush=True)
    px, py, z_sign = _project_ortho(verts, cfg.front_axis, photo_w, photo_h)
    # Face-forward: vert normal has positive component along front_axis
    n_z = normals[:, 2] * z_sign
    forward_mask = n_z > 0.05
    n_forward = int(forward_mask.sum())
    print(f"      forward verts: {n_forward:,}/{len(verts):,}", flush=True)

    # ---- 4. Blue-noise Poisson-disk sample dot positions in photo pixel space
    print(f"\n[4/6] Blue-noise sample {cfg.dot_count} dots", flush=True)
    rng = np.random.default_rng(cfg.seed)
    dot_px = []
    n_attempts = cfg.dot_count * 8
    # Simple UV-grid acceleration for min-radius check
    grid_cell = cfg.dot_min_radius_px
    n_gx = int(np.ceil(photo_w / grid_cell)) + 1
    n_gy = int(np.ceil(photo_h / grid_cell)) + 1
    grid = {}  # (gx, gy) -> list of dot_px indices
    for _ in range(n_attempts):
        if len(dot_px) >= cfg.dot_count:
            break
        u = rng.uniform(0, photo_w)
        v = rng.uniform(0, photo_h)
        xi, yi = int(u), int(v)
        if not (0 <= xi < photo_w and 0 <= yi < photo_h):
            continue
        lum = gray[yi, xi]
        if cfg.lum_curve == "linear":
            prob = 1.0 - lum
        elif cfg.lum_curve == "gamma":
            prob = (1.0 - lum) ** cfg.lum_gamma
        elif cfg.lum_curve == "invert":
            prob = lum
        else:
            prob = 1.0 - lum
        if rng.uniform() > prob:
            continue
        # Min-radius check via grid neighbors
        gx = int(u / grid_cell); gy = int(v / grid_cell)
        collide = False
        for dgx in (-1, 0, 1):
            for dgy in (-1, 0, 1):
                key = (gx+dgx, gy+dgy)
                if key not in grid: continue
                for idx in grid[key]:
                    dp = dot_px[idx]
                    if (dp[0]-u)**2 + (dp[1]-v)**2 < cfg.dot_min_radius_px**2:
                        collide = True; break
                if collide: break
            if collide: break
        if collide: continue
        dot_px.append((u, v))
        grid.setdefault((gx, gy), []).append(len(dot_px) - 1)
    dot_px = np.array(dot_px, dtype=np.float32) if dot_px else np.zeros((0, 2), dtype=np.float32)
    print(f"      sampled: {len(dot_px):,}", flush=True)

    # ---- 5. Build vert-in-photo KDTree, accumulate displacement per vert
    print("\n[5/6] KNN influence -> per-vert displacement accumulator", flush=True)
    forward_idx = np.where(forward_mask)[0]
    if len(forward_idx) == 0:
        raise RuntimeError("No forward-facing verts; front_axis wrong?")
    vert_uv = np.stack([px[forward_idx], py[forward_idx]], axis=1)
    tree = cKDTree(vert_uv)
    displacement = np.zeros(len(verts), dtype=np.float32)
    sigma_px = cfg.influence_radius_px * 0.5
    for du in dot_px:
        near = tree.query_ball_point(du, cfg.influence_radius_px)
        if not near: continue
        for local_i in near:
            global_i = forward_idx[local_i]
            d2 = (vert_uv[local_i, 0]-du[0])**2 + (vert_uv[local_i, 1]-du[1])**2
            falloff = np.exp(-d2 / (2*sigma_px*sigma_px))
            displacement[global_i] += falloff
    # Normalize peak to 1
    d_max = float(displacement.max())
    if d_max > 0:
        displacement /= d_max
    n_displaced = int((displacement > 0.01).sum())
    print(f"      verts with displacement: {n_displaced:,}", flush=True)

    # ---- 6. Displace verts along normals, export
    print(f"\n[6/6] Displace + export (strength={cfg.emboss_strength})", flush=True)
    verts_new = verts + normals * (displacement[:, None] * cfg.emboss_strength)
    result = trimesh.Trimesh(vertices=verts_new, faces=faces, process=False)
    result.export(cfg.output_mesh)
    sz_mb = cfg.output_mesh.stat().st_size / 1024 / 1024
    print(f"      wrote {cfg.output_mesh} ({sz_mb:.2f} MB, took {time.time()-t_all:.1f}s)",
          flush=True)
    return cfg.output_mesh


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="module_e_engrave")
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output-mesh", required=True, type=Path)
    ap.add_argument("--emboss-strength", type=float, default=0.004)
    ap.add_argument("--dot-count", type=int, default=20000)
    ap.add_argument("--dot-min-radius-px", type=float, default=3.0)
    ap.add_argument("--influence-radius-px", type=float, default=6.0)
    ap.add_argument("--lum-curve", choices=["linear","gamma","invert"], default="linear")
    ap.add_argument("--lum-gamma", type=float, default=2.2)
    ap.add_argument("--front-axis", choices=["+Z","-Z"], default="+Z")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    a = build_argparser().parse_args(argv)
    cfg = ModuleEEngraveConfig(
        input_mesh=a.input_mesh, input_image=a.input_image, output_mesh=a.output_mesh,
        emboss_strength=a.emboss_strength, dot_count=a.dot_count,
        dot_min_radius_px=a.dot_min_radius_px, influence_radius_px=a.influence_radius_px,
        lum_curve=a.lum_curve, lum_gamma=a.lum_gamma, front_axis=a.front_axis,
        seed=a.seed, dry_run=a.dry_run,
    )
    run_module_e_engrave(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
