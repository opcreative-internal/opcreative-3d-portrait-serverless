"""
Module E SSLE (V17) -- Subsurface Laser Engraving point cloud writer.

Kent's product is CrystalMe3D Pro: photo -> 3D point cloud -> SSLE laser etches
internal micro-fracture points inside a clear crystal cube.

CRITICAL polarity note (Fable 5 v2): laser fractures SCATTER light; points read
as WHITE on an LED base. Density is proportional to LUMINANCE (bright = dense),
NOT its inverse. CrystalMe's preview shows dark dots on white canvas -- that's
negative-space rendering of the same data.

Algorithm (Yuksel weighted sample elimination = blue-noise with exact count):
  1. Load input mesh (V11 C+D output -- has face relief bumps).
  2. Scale into physical crystal cube (60x60x80 mm default, 5 mm margin).
  3. Ortho project to photo, sample luminance per vertex, compute weights.
  4. Draw N_CANDIDATES surface samples via trimesh.sample.sample_surface
     (area-uniform), interpolate weights per barycentric.
  5. Voxel-hash 3D min-distance greedy elimination sorted by weight.
  6. ±0.15 mm normal jitter to avoid coplanar stress sheet.
  7. Write ASCII PLY (positions only, mm, Y-up, +Z front) + XYZ fallback.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModuleESSLEConfig:
    input_mesh: Path                # V11 C+D output GLB
    input_image: Path               # source photo for luminance
    output_ply: Path                # ASCII PLY point cloud

    # Crystal cube in mm (W x D x H)
    cube_w_mm: float = 60.0
    cube_d_mm: float = 60.0
    cube_h_mm: float = 80.0
    margin_mm: float = 5.0

    # Point count target
    target_count: int = 300_000
    accept_low: int = 150_000
    accept_high: int = 450_000
    n_candidates: int = 2_500_000

    # Blue-noise min-distance
    r_min_mm: float = 0.18
    r_min_floor_mm: float = 0.15

    # Weighting
    w_floor: float = 0.08
    w_gamma: float = 1.6
    w_back: float = 0.30
    face_boost: float = 1.3

    # Jitter
    normal_jitter_mm: float = 0.15

    # Projection
    front_axis: str = "+Z"
    seed: int = 42

    dry_run: bool = False

    def __post_init__(self):
        self.input_mesh = Path(self.input_mesh)
        self.input_image = Path(self.input_image)
        self.output_ply = Path(self.output_ply)


def _project_ortho(pts_xy, photo_w, photo_h, front_axis="+Z"):
    """Return image-space pixel coords for XY of mesh points (normalized to fit photo)."""
    import numpy as np
    if front_axis == "-Z":
        xy = pts_xy * np.array([-1.0, 1.0], dtype=np.float32)
    else:
        xy = pts_xy.copy()
    return xy


def run_module_e_ssle(cfg: ModuleESSLEConfig) -> Path:
    print("=" * 66)
    print("  Module E SSLE (V17) -- CrystalMe3D point cloud")
    print("=" * 66)
    print(f"  input_mesh   : {cfg.input_mesh}")
    print(f"  input_image  : {cfg.input_image}")
    print(f"  output_ply   : {cfg.output_ply}")
    print(f"  cube (mm)    : {cfg.cube_w_mm} x {cfg.cube_d_mm} x {cfg.cube_h_mm}, margin {cfg.margin_mm}")
    print(f"  target       : {cfg.target_count} pts, band [{cfg.accept_low},{cfg.accept_high}]")
    print(f"  candidates   : {cfg.n_candidates}")
    print(f"  r_min        : {cfg.r_min_mm} mm (floor {cfg.r_min_floor_mm})")
    print(f"  weights      : floor={cfg.w_floor} gamma={cfg.w_gamma} back={cfg.w_back} face_boost={cfg.face_boost}")
    print(f"  jitter       : {cfg.normal_jitter_mm} mm normal-jitter")
    print(f"  seed         : {cfg.seed}")
    print("=" * 66, flush=True)
    if cfg.dry_run: print("[dry-run] validated"); return cfg.output_ply

    t0 = time.time()
    import numpy as np
    import trimesh
    import cv2
    from PIL import Image
    from scipy.spatial import cKDTree

    cfg.output_ply.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load mesh
    print("\n[1/7] Load mesh", flush=True)
    obj = trimesh.load(cfg.input_mesh, force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    else:
        mesh = obj
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"      verts={len(verts):,} faces={len(faces):,}", flush=True)

    # ---- 2. Rescale mesh into crystal cube (bbox-centered, uniform-scaled)
    print("\n[2/7] Scale to crystal cube", flush=True)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    usable_w = cfg.cube_w_mm - 2 * cfg.margin_mm
    usable_d = cfg.cube_d_mm - 2 * cfg.margin_mm
    usable_h = cfg.cube_h_mm - 2 * cfg.margin_mm
    scale_w = usable_w / max(bbox_size[0], 1e-6)
    scale_d = usable_d / max(bbox_size[2], 1e-6)
    scale_h = usable_h / max(bbox_size[1], 1e-6)
    s = min(scale_w, scale_d, scale_h)
    center_orig = (bbox_min + bbox_max) / 2.0
    verts_mm = (verts - center_orig) * s
    # Center in cube (cube centered at origin: [-w/2, w/2] etc.)
    verts_mm[:, 0] += 0.0  # X = 0 center
    verts_mm[:, 1] += 0.0  # Y = 0 center
    verts_mm[:, 2] += 0.0  # Z = 0 center
    print(f"      scale factor: {s:.4f} mm/unit  bbox_mm={(bbox_size*s).round(2)}", flush=True)
    mesh_mm = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)

    # ---- 3. Load photo grayscale + face bbox (from V11 MediaPipe if available)
    print("\n[3/7] Load photo + luminance", flush=True)
    photo = np.array(Image.open(cfg.input_image).convert("RGB"))
    photo_h, photo_w = photo.shape[:2]
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    print(f"      photo {photo_w}x{photo_h}", flush=True)

    # Face bbox via mediapipe (best-effort; fallback = no face boost)
    face_bbox = None
    try:
        from pipeline_v11_depth_umeyama import detect_landmarks
        lm = detect_landmarks(cfg.input_image, mp_mode=478)
        face_bbox = lm.get("bbox")  # (x, y, w, h) in pixels
        if face_bbox:
            print(f"      face bbox (photo px): {face_bbox}", flush=True)
    except Exception as e:
        print(f"      face bbox skip: {e}", flush=True)

    # ---- 4. Surface sample candidates + luminance-weighted
    print(f"\n[4/7] Surface sample {cfg.n_candidates} candidates", flush=True)
    rng = np.random.default_rng(cfg.seed)
    cand_pts, cand_face_idx = trimesh.sample.sample_surface(
        mesh_mm, cfg.n_candidates, seed=cfg.seed
    )
    cand_pts = np.asarray(cand_pts, dtype=np.float32)
    cand_face_normals = np.asarray(mesh_mm.face_normals, dtype=np.float32)[cand_face_idx]
    print(f"      sampled {len(cand_pts):,}", flush=True)

    # Ortho project into photo. mesh_mm is bbox-centered so bbox_min/max in mm.
    mm_bbox_min = verts_mm.min(0); mm_bbox_max = verts_mm.max(0)
    xy = cand_pts[:, :2]
    xy_norm = (xy - mm_bbox_min[:2]) / np.maximum(mm_bbox_max[:2] - mm_bbox_min[:2], 1e-6)
    # Y-flip: image row 0 = top = world +Y
    xy_norm[:, 1] = 1.0 - xy_norm[:, 1]
    if cfg.front_axis == "-Z": xy_norm[:, 0] = 1.0 - xy_norm[:, 0]
    px_i = np.clip((xy_norm[:, 0] * photo_w).astype(np.int32), 0, photo_w - 1)
    py_i = np.clip((xy_norm[:, 1] * photo_h).astype(np.int32), 0, photo_h - 1)
    lum = gray[py_i, px_i]

    # Front-facing mask via face normal Z sign
    z_sign = 1.0 if cfg.front_axis == "+Z" else -1.0
    n_z_signed = cand_face_normals[:, 2] * z_sign
    front_mask = n_z_signed > 0.05

    # Weight: front-facing get w_floor + (1-w_floor)*L^gamma; back-facing get w_back
    w = np.full(len(cand_pts), cfg.w_back, dtype=np.float32)
    w[front_mask] = cfg.w_floor + (1.0 - cfg.w_floor) * np.power(lum[front_mask], cfg.w_gamma)

    # Face bbox boost
    if face_bbox:
        fx, fy, fw, fh = face_bbox
        in_face = ((px_i >= fx) & (px_i < fx + fw) &
                   (py_i >= fy) & (py_i < fy + fh) & front_mask)
        w[in_face] *= cfg.face_boost
        print(f"      face-boost applied to {int(in_face.sum()):,} candidates", flush=True)

    # ---- 5. Voxel-hash greedy elimination (blue-noise)
    print(f"\n[5/7] Weighted sample elimination -> target {cfg.target_count}", flush=True)
    r_min = cfg.r_min_mm
    # Attempt with r_min, adjust if band miss
    result_pts = None
    result_normals = None
    for attempt in range(3):
        # Sort by weight desc, drop points within r_min of any accepted
        order = np.argsort(-w)
        cell = r_min
        grid = {}  # (gx,gy,gz) -> list of accepted indices
        accepted = []
        r_min_sq = r_min * r_min
        for i in order:
            p = cand_pts[i]
            gx = int(p[0] / cell); gy = int(p[1] / cell); gz = int(p[2] / cell)
            hit = False
            for dgx in (-1, 0, 1):
                for dgy in (-1, 0, 1):
                    for dgz in (-1, 0, 1):
                        key = (gx + dgx, gy + dgy, gz + dgz)
                        if key not in grid: continue
                        for j in grid[key]:
                            q = cand_pts[j]
                            d2 = (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
                            if d2 < r_min_sq: hit = True; break
                        if hit: break
                    if hit: break
                if hit: break
            if hit: continue
            accepted.append(i)
            grid.setdefault((gx, gy, gz), []).append(i)
            if len(accepted) >= cfg.accept_high: break
        n_acc = len(accepted)
        print(f"      attempt {attempt} r_min={r_min:.3f} -> {n_acc:,} accepted", flush=True)
        if cfg.accept_low <= n_acc <= cfg.accept_high:
            result_pts = cand_pts[accepted]
            result_normals = cand_face_normals[accepted]
            break
        # Bisect: too few -> lower r_min; too many -> raise r_min
        if n_acc < cfg.accept_low:
            r_min *= 0.85
            if r_min < cfg.r_min_floor_mm:
                r_min = cfg.r_min_floor_mm
                # Emit whatever we have on next attempt
        else:
            r_min *= 1.15
    if result_pts is None:
        result_pts = cand_pts[accepted]
        result_normals = cand_face_normals[accepted]
        print(f"      accepting {len(result_pts):,} (outside band)", flush=True)

    # ---- 6. Normal jitter, cube bounds assert
    print(f"\n[6/7] Normal jitter + bounds", flush=True)
    jitter = rng.uniform(-cfg.normal_jitter_mm, cfg.normal_jitter_mm,
                        size=(len(result_pts), 1)).astype(np.float32)
    result_pts_j = result_pts + result_normals * jitter
    half_w = cfg.cube_w_mm / 2 - cfg.margin_mm
    half_d = cfg.cube_d_mm / 2 - cfg.margin_mm
    half_h = cfg.cube_h_mm / 2 - cfg.margin_mm
    inside = ((np.abs(result_pts_j[:, 0]) <= half_w) &
              (np.abs(result_pts_j[:, 1]) <= half_h) &
              (np.abs(result_pts_j[:, 2]) <= half_d))
    result_pts_j = result_pts_j[inside]
    print(f"      after jitter+bounds: {len(result_pts_j):,}", flush=True)

    # NN distance sanity
    if len(result_pts_j) > 100:
        tree = cKDTree(result_pts_j)
        d, _ = tree.query(result_pts_j, k=2)
        nn = d[:, 1]
        p0 = float(nn.min()); p5 = float(np.percentile(nn, 5))
        print(f"      NN min={p0:.4f} p5={p5:.4f} mm (floor {cfg.r_min_floor_mm})",
              flush=True)
        if p0 < cfg.r_min_floor_mm * 0.8:
            print(f"      WARN: NN below floor -- crack risk", flush=True)

    # ---- 7. Write PLY (ASCII) + XYZ fallback
    print(f"\n[7/7] Write PLY + XYZ", flush=True)
    n = len(result_pts_j)
    with open(cfg.output_ply, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment units mm, Y-up, front {cfg.front_axis}, cube {cfg.cube_w_mm}x{cfg.cube_d_mm}x{cfg.cube_h_mm}\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in result_pts_j:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    ply_sz = cfg.output_ply.stat().st_size / 1024 / 1024
    print(f"      wrote {cfg.output_ply} ({n:,} pts, {ply_sz:.2f} MB)", flush=True)

    xyz_path = cfg.output_ply.with_suffix(".xyz")
    with open(xyz_path, "w") as f:
        for p in result_pts_j:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    print(f"      wrote {xyz_path}", flush=True)

    print(f"\n[DONE] total {time.time()-t0:.1f}s", flush=True)
    return cfg.output_ply


def build_argparser():
    ap = argparse.ArgumentParser(prog="module_e_ssle")
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output-ply", required=True, type=Path)
    ap.add_argument("--cube-w-mm", type=float, default=60.0)
    ap.add_argument("--cube-d-mm", type=float, default=60.0)
    ap.add_argument("--cube-h-mm", type=float, default=80.0)
    ap.add_argument("--margin-mm", type=float, default=5.0)
    ap.add_argument("--target-count", type=int, default=300_000)
    ap.add_argument("--n-candidates", type=int, default=2_500_000)
    ap.add_argument("--r-min-mm", type=float, default=0.18)
    ap.add_argument("--front-axis", choices=["+Z","-Z"], default="+Z")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg = ModuleESSLEConfig(
        input_mesh=a.input_mesh, input_image=a.input_image, output_ply=a.output_ply,
        cube_w_mm=a.cube_w_mm, cube_d_mm=a.cube_d_mm, cube_h_mm=a.cube_h_mm,
        margin_mm=a.margin_mm, target_count=a.target_count, n_candidates=a.n_candidates,
        r_min_mm=a.r_min_mm, front_axis=a.front_axis, seed=a.seed, dry_run=a.dry_run,
    )
    run_module_e_ssle(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
