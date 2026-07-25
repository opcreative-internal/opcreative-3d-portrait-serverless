"""
OpCreative Pipeline v11 — Depth-Anything-V2 face relief + Umeyama fit + Laplacian pyramid blend

Implements Module C + Module D per MORNING_PLAN spec (commercial-safe stack).

Pipeline:
    Step 1  MediaPipe FaceMesh (478 landmarks w/ iris refine, or 468 base) on portrait
    Step 2  Depth-Anything-V2 (Small/Base/Large) face-region relief map, normalized
    Step 3  Umeyama similarity transform from photo landmarks -> mesh head landmarks
             (scale clamped to [umeyama_clamp_min, umeyama_clamp_max])
    Step 4  Laplacian pyramid blend of relief depth into head vertices (face_weight mix)

Dependencies (runtime — NOT imported at module top; lazy-loaded per-step):
    torch, torchvision                             (DAv2 + Umeyama numeric)
    numpy, Pillow                                  (image IO)
    mediapipe>=0.10.9                              (FaceMesh 478 + iris)
    trimesh, pygltflib                             (GLB mesh IO)
    scipy                                          (Umeyama fallback + KDTree)
    scikit-image                                   (Laplacian pyramid)
    depth-anything-v2 (from HF: depth-anything/Depth-Anything-V2-{Small,Base,Large})

Usage:
    # Single run with recommended defaults
    python pipeline_v11_depth_umeyama.py \\
        --input-image samples/split/person_2.png \\
        --input-mesh samples/triposg_output/person_2.glb \\
        --output samples/v11_output/person_2_v11.glb

    # 12-config sweep (DAv2 {Base,Large} x face_weight {0.5,0.7,0.9} x emboss {0.008,0.012})
    python pipeline_v11_depth_umeyama.py \\
        --input-image ... --input-mesh ... --output ... --sweep

    # Dry run — validate args + print plan, no torch/mediapipe load
    python pipeline_v11_depth_umeyama.py \\
        --input-image ... --input-mesh ... --output ... --dry-run

Importable API:
    from pipeline_v11_depth_umeyama import Config, run_pipeline
    result_path = run_pipeline(Config(input_image=..., input_mesh=..., output=...))
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DAV2_ENCODERS = {"small": "vits", "base": "vitb", "large": "vitl"}
DAV2_HF_REPO = {
    "small": "depth-anything/Depth-Anything-V2-Small",
    "base":  "depth-anything/Depth-Anything-V2-Base",
    "large": "depth-anything/Depth-Anything-V2-Large",
}


@dataclass
class Config:
    """Full pipeline configuration. All defaults are the recommended starting config."""
    input_image: Path
    input_mesh: Path
    output: Path

    # Knobs
    dav2_model: str = "base"          # small | base | large  (Giant excluded — non-commercial)
    mask_expand: float = 1.2           # face bbox expansion factor
    lap_levels: int = 6                # Laplacian pyramid depth
    face_weight: float = 0.7           # relief-vs-original mixing coefficient (0..1)
    mp_mode: int = 478                 # 468 base, 478 = base + iris refine
    umeyama_clamp_min: float = 0.9     # min similarity-transform scale
    umeyama_clamp_max: float = 1.1     # max similarity-transform scale
    depth_norm: str = "percentile"     # minmax | percentile (5-95)
    emboss_strength: float = 0.010     # Z-scale of relief in world units

    # Modes
    sweep: bool = False
    dry_run: bool = False

    def __post_init__(self):
        # Coerce to Path
        self.input_image = Path(self.input_image)
        self.input_mesh = Path(self.input_mesh)
        self.output = Path(self.output)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("input_image", "input_mesh", "output"):
            d[k] = str(d[k])
        return d


def build_sweep_configs(base: Config) -> List[Tuple[str, Config]]:
    """12-config sweep: DAv2 {base,large} x face_weight {0.5,0.7,0.9} x emboss {0.008,0.012}."""
    configs: List[Tuple[str, Config]] = []
    out_parent = base.output.parent
    stem = base.output.stem  # e.g. person_2_v11
    ext = base.output.suffix or ".glb"

    for dav2 in ("base", "large"):
        for fw in (0.5, 0.7, 0.9):
            for emb in (0.008, 0.012):
                cid = f"dav2-{dav2}_fw{fw:.1f}_emb{emb:.3f}"
                cfg = replace(
                    base,
                    dav2_model=dav2,
                    face_weight=fw,
                    emboss_strength=emb,
                    sweep=False,
                    output=out_parent / f"{stem}_{cid}{ext}",
                )
                configs.append((cid, cfg))
    return configs


# ---------------------------------------------------------------------------
# Step 1 — MediaPipe FaceMesh landmarks
# ---------------------------------------------------------------------------

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def detect_landmarks(image_path: Path, mp_mode: int = 478) -> Dict[str, Any]:
    """Return {bbox, landmarks_2d (N,2), landmarks_3d (N,3), n} for FIRST detected face.

    Uses MediaPipe Tasks API (FaceLandmarker) which returns 478 landmarks by default
    (base 468 + iris refinement built into the task model). The mp_mode arg is
    accepted for API compatibility but Tasks API always returns 478.

    Requires face_landmarker.task at $FACE_LANDMARKER_MODEL (default:
    /workspace/face_landmarker.task).
    """
    import os
    import numpy as np
    from PIL import Image
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )

    if mp_mode not in (468, 478):
        raise ValueError(f"mp_mode must be 468 or 478, got {mp_mode}")

    model_path = os.environ.get(
        "FACE_LANDMARKER_MODEL", "/workspace/face_landmarker.task"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"face_landmarker.task not found at {model_path}.\n"
            f"Download via:\n"
            f"  curl -L -o {model_path} \\\n"
            f"    {FACE_LANDMARKER_URL}"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    detector = FaceLandmarker.create_from_options(options)

    # Load image; if RGBA with transparency, composite over white background so
    # bg-removed / alpha-masked portraits don't feed MediaPipe an all-black bg.
    img_raw = Image.open(image_path)
    print(f"      image mode={img_raw.mode} size={img_raw.size}", flush=True)
    if img_raw.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img_raw.size, (255, 255, 255))
        bg.paste(img_raw, mask=img_raw.split()[-1])
        img_pil = bg
    else:
        img_pil = img_raw.convert("RGB")

    # Full-body portrait heuristic: if image is much taller than wide (H > 1.8*W),
    # the face occupies < 20% of frame area which is at the edge of what MediaPipe
    # FaceLandmarker can detect. Crop upper 45% (face + shoulders) before feeding.
    w0, h0 = img_pil.size
    if h0 > 1.8 * w0:
        upper_h = int(h0 * 0.45)
        img_pil = img_pil.crop((0, 0, w0, upper_h))
        print(f"      cropped tall portrait to upper 45%: {img_pil.size}",
              flush=True)
    w, h = img_pil.size
    img_np = np.array(img_pil)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result = detector.detect(mp_img)
    if not result.face_landmarks:
        raise RuntimeError(
            f"No face detected in {image_path} "
            f"(mode={img_raw.mode}, size={img_raw.size}); "
            f"try a color/higher-contrast frontal portrait."
        )

    face_lm = result.face_landmarks[0]
    # NormalizedLandmark: x, y in [0,1] image coords, z relative to face plane
    pts = np.array(
        [[p.x * w, p.y * h, p.z * w] for p in face_lm],
        dtype=np.float32,
    )
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())

    return {
        "bbox": (x1, y1, x2, y2),
        "landmarks_2d": pts[:, :2].copy(),
        "landmarks_3d": pts.copy(),
        "n": int(pts.shape[0]),
        "image_size": (w, h),
    }


# ---------------------------------------------------------------------------
# Step 2 — Depth-Anything-V2 face relief
# ---------------------------------------------------------------------------

def _dav2_load(model_size: str, device: str):
    """Lazy-load Depth-Anything-V2 checkpoint. Expects HF hub cache to be primed."""
    import torch
    from huggingface_hub import hf_hub_download

    if model_size not in DAV2_ENCODERS:
        raise ValueError(f"dav2_model must be one of {list(DAV2_ENCODERS)}, got {model_size}")
    encoder = DAV2_ENCODERS[model_size]
    repo = DAV2_HF_REPO[model_size]

    # DAv2 official repo (Apache-2.0) — expects a python module `depth_anything_v2`
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as e:
        raise ImportError(
            "depth_anything_v2 not installed. On the pod run:\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2 \\\n"
            "  && pip install -r Depth-Anything-V2/requirements.txt \\\n"
            "  && PYTHONPATH=$PWD/Depth-Anything-V2 python pipeline_v11_...\n"
            f"(underlying error: {e})"
        )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    ckpt_name = f"depth_anything_v2_{encoder}.pth"
    ckpt_path = hf_hub_download(repo_id=repo, filename=ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def dav2_face_relief(
    image_path: Path,
    bbox: Tuple[float, float, float, float],
    mask_expand: float,
    model_size: str,
    depth_norm: str,
    emboss_strength: float,
) -> Dict[str, Any]:
    """Run DAv2 on expanded face crop, return normalized relief map + crop metadata.

    Output relief is a (H, W) float array in world-Z units (already scaled by
    emboss_strength). Larger values -> more forward (out of screen).
    """
    import numpy as np
    from PIL import Image
    import torch
    import cv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _dav2_load(model_size, device)

    # Expand bbox
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * mask_expand
    h = (y2 - y1) * mask_expand
    ex1 = int(max(0, cx - w / 2))
    ey1 = int(max(0, cy - h / 2))
    ex2 = int(cx + w / 2)
    ey2 = int(cy + h / 2)

    img_full = np.array(Image.open(image_path).convert("RGB"))
    H, W = img_full.shape[:2]
    ex2 = min(W, ex2)
    ey2 = min(H, ey2)
    crop = img_full[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        raise RuntimeError(f"Face crop empty: bbox={bbox}, expanded=({ex1},{ey1},{ex2},{ey2})")

    with torch.no_grad():
        # DAv2 has infer_image(bgr_ndarray) -> depth (H, W) numpy
        depth = model.infer_image(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # Normalize
    if depth_norm == "minmax":
        d_min, d_max = float(depth.min()), float(depth.max())
    elif depth_norm == "percentile":
        d_min, d_max = np.percentile(depth, [5, 95]).astype(float).tolist()
    else:
        raise ValueError(f"Unknown depth_norm: {depth_norm}")
    span = max(d_max - d_min, 1e-6)
    depth_norm_arr = np.clip((depth - d_min) / span, 0.0, 1.0)

    # Emboss: relief in world Z units, centered around 0
    relief = (depth_norm_arr - 0.5) * float(emboss_strength) * 2.0

    return {
        "relief": relief.astype("float32"),
        "crop_bbox": (ex1, ey1, ex2, ey2),
        "depth_min": float(d_min),
        "depth_max": float(d_max),
    }


# ---------------------------------------------------------------------------
# Step 3 — Umeyama similarity transform
# ---------------------------------------------------------------------------

def umeyama(
    src: "np.ndarray",
    dst: "np.ndarray",
    with_scale: bool = True,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", float]:
    """Compute similarity transform (c*R, t) minimizing ||c*R @ src + t - dst||^2.

    src, dst: (N, D) matched point sets.
    Returns (R (D,D), t (D,), s (D-vec passthrough), c scalar).
    Reference: Umeyama 1991, "Least-squares estimation of transformation params
    between two point patterns". Uses SVD-based closed form.
    """
    import numpy as np
    assert src.shape == dst.shape, f"src {src.shape} vs dst {dst.shape}"
    n, d = src.shape

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n

    U, S, Vt = np.linalg.svd(cov)
    S_diag = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_diag[-1, -1] = -1

    R = U @ S_diag @ Vt
    c = float((S * np.diag(S_diag)).sum() / (var_s + 1e-12)) if with_scale else 1.0
    t = mu_d - c * (R @ mu_s)
    return R, t, S, c


def apply_similarity(pts: "np.ndarray", R, t, c) -> "np.ndarray":
    """Apply c*R @ pts + t (pts is (N, D))."""
    return c * (pts @ R.T) + t


# ---------------------------------------------------------------------------
# Step 4 — Head vertex detection + landmark projection + Laplacian blend
# ---------------------------------------------------------------------------

def find_head_vertices(vertices: "np.ndarray", top_fraction: float = 0.20) -> "np.ndarray":
    """Return indices of head-region vertices (top `top_fraction` by Y axis)."""
    import numpy as np
    y = vertices[:, 1]
    thresh = y.max() - (y.max() - y.min()) * top_fraction
    return np.where(y >= thresh)[0]


def project_orthographic(vertices: "np.ndarray") -> "np.ndarray":
    """Project 3D verts to 2D (X, Y) — simple orthographic front-facing camera."""
    return vertices[:, :2].copy()


def match_head_landmarks_to_mp(
    head_verts_2d: "np.ndarray",
    mp_landmarks_2d: "np.ndarray",
    mp_key_indices: "np.ndarray",
) -> "np.ndarray":
    """For each mp keypoint, find nearest head vertex in 2D — returns matched
    subset of head vertex indices aligned with mp_key_indices.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    key_pts = mp_landmarks_2d[mp_key_indices]

    # Normalize both sets to [0,1] bbox for scale-invariant nearest-neighbor
    def norm(p):
        pmin = p.min(axis=0)
        pmax = p.max(axis=0)
        span = np.maximum(pmax - pmin, 1e-6)
        return (p - pmin) / span

    head_norm = norm(head_verts_2d)
    key_norm = norm(key_pts)

    tree = cKDTree(head_norm)
    _, nn = tree.query(key_norm, k=1)
    return nn  # (K,) indices into head_verts_2d


# MediaPipe FaceMesh anchor indices for stable Umeyama fit (eyes / nose / mouth / chin)
# Standard MP topology (works for both 468 and 478 modes).
MP_ANCHOR_INDICES = [
    33,   # right eye outer corner
    133,  # right eye inner corner
    362,  # left eye inner corner
    263,  # left eye outer corner
    1,    # nose tip
    61,   # mouth right corner
    291,  # mouth left corner
    17,   # lower lip center
    199,  # chin
    10,   # forehead
]


def laplacian_pyramid_blend_z(
    orig_z_map: "np.ndarray",
    relief_z_map: "np.ndarray",
    mask_map: "np.ndarray",
    levels: int,
) -> "np.ndarray":
    """Multi-band blend two Z rasters using a smooth mask.

    Standard Burt-Adelson: build Gaussian pyramid of mask,
    Laplacian pyramids of both z maps, blend per level, collapse.
    """
    import numpy as np
    from skimage.transform import pyramid_gaussian, pyramid_laplacian, pyramid_expand

    # skimage returns generators — realize them
    g_mask = list(pyramid_gaussian(mask_map, max_layer=levels, downscale=2, channel_axis=None))
    l_orig = list(pyramid_laplacian(orig_z_map, max_layer=levels, downscale=2, channel_axis=None))
    l_relief = list(pyramid_laplacian(relief_z_map, max_layer=levels, downscale=2, channel_axis=None))

    # Ensure equal length
    L = min(len(g_mask), len(l_orig), len(l_relief))
    g_mask = g_mask[:L]
    l_orig = l_orig[:L]
    l_relief = l_relief[:L]

    # Blend per level
    blended_pyr = []
    for m, o, r in zip(g_mask, l_orig, l_relief):
        # Broadcast mask if shape mismatch by one pixel from odd sizes
        if m.shape != o.shape:
            m = m[:o.shape[0], :o.shape[1]]
        blended_pyr.append(m * r + (1.0 - m) * o)

    # Collapse pyramid
    out = blended_pyr[-1]
    for i in range(len(blended_pyr) - 2, -1, -1):
        up = pyramid_expand(out, upscale=2, channel_axis=None)
        target_shape = blended_pyr[i].shape
        up = up[:target_shape[0], :target_shape[1]]
        # pad if smaller
        if up.shape != target_shape:
            padded = np.zeros(target_shape, dtype=up.dtype)
            padded[:up.shape[0], :up.shape[1]] = up
            up = padded
        out = up + blended_pyr[i]
    return out


def rasterize_head_to_grid(
    head_verts_2d: "np.ndarray",
    head_z: "np.ndarray",
    grid_size: int = 256,
) -> Tuple["np.ndarray", "np.ndarray", Tuple[float, float, float, float]]:
    """Rasterize head verts into a 2D grid of Z + mask.
    Returns (z_grid, mask_grid, extent (u_min,u_max,v_min,v_max)).
    """
    import numpy as np
    u_min, u_max = float(head_verts_2d[:, 0].min()), float(head_verts_2d[:, 0].max())
    v_min, v_max = float(head_verts_2d[:, 1].min()), float(head_verts_2d[:, 1].max())
    u_span = max(u_max - u_min, 1e-6)
    v_span = max(v_max - v_min, 1e-6)

    z_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    mask_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for (u, v), z in zip(head_verts_2d, head_z):
        gi = int((v - v_min) / v_span * (grid_size - 1))
        gj = int((u - u_min) / u_span * (grid_size - 1))
        z_grid[gi, gj] = float(z)
        mask_grid[gi, gj] = 1.0
    return z_grid, mask_grid, (u_min, u_max, v_min, v_max)


def sample_grid_at_verts(
    grid: "np.ndarray",
    verts_2d: "np.ndarray",
    extent: Tuple[float, float, float, float],
) -> "np.ndarray":
    """Bilinear-sample a 2D grid at vertex (u,v) locations."""
    import numpy as np
    u_min, u_max, v_min, v_max = extent
    u_span = max(u_max - u_min, 1e-6)
    v_span = max(v_max - v_min, 1e-6)
    H, W = grid.shape
    out = np.zeros(len(verts_2d), dtype=np.float32)
    for k, (u, v) in enumerate(verts_2d):
        gi = (v - v_min) / v_span * (H - 1)
        gj = (u - u_min) / u_span * (W - 1)
        gi_c = int(np.clip(gi, 0, H - 1))
        gj_c = int(np.clip(gj, 0, W - 1))
        out[k] = grid[gi_c, gj_c]
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _print_plan(cfg: Config) -> None:
    print("=" * 66)
    print("  Pipeline v11 — Depth Anything V2 + Umeyama + Laplacian blend")
    print("=" * 66)
    print(f"  Input image : {cfg.input_image}")
    print(f"  Input mesh  : {cfg.input_mesh}")
    print(f"  Output      : {cfg.output}")
    print("  ---- Knobs ----")
    for k in ("dav2_model", "mask_expand", "lap_levels", "face_weight",
              "mp_mode", "umeyama_clamp_min", "umeyama_clamp_max",
              "depth_norm", "emboss_strength"):
        print(f"    {k:22s} = {getattr(cfg, k)}")
    print("  ---- Modes ----")
    print(f"    sweep    = {cfg.sweep}")
    print(f"    dry_run  = {cfg.dry_run}")
    print("=" * 66, flush=True)


def run_pipeline(cfg: Config) -> Path:
    """Execute full Module C + D pipeline. Returns output GLB path."""
    _print_plan(cfg)

    if cfg.dry_run:
        print("[dry-run] Skipping heavy imports + inference. Plan validated.", flush=True)
        return cfg.output

    # Verify inputs exist
    if not cfg.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {cfg.input_image}")
    if not cfg.input_mesh.exists():
        raise FileNotFoundError(f"Input mesh not found: {cfg.input_mesh}")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports
    import numpy as np
    import trimesh

    t_start = time.time()

    # ---- Step 1: MediaPipe landmarks
    print(f"\n[1/4] MediaPipe FaceMesh (mode={cfg.mp_mode})", flush=True)
    t0 = time.time()
    lm = detect_landmarks(cfg.input_image, cfg.mp_mode)
    print(f"      landmarks: {lm['n']}, bbox: {tuple(round(v,1) for v in lm['bbox'])}, "
          f"took {time.time()-t0:.2f}s", flush=True)

    # ---- Step 2: DAv2 face relief
    print(f"\n[2/4] Depth-Anything-V2 [{cfg.dav2_model}] face relief", flush=True)
    t0 = time.time()
    relief = dav2_face_relief(
        cfg.input_image, lm["bbox"], cfg.mask_expand,
        cfg.dav2_model, cfg.depth_norm, cfg.emboss_strength,
    )
    print(f"      relief shape: {relief['relief'].shape}, "
          f"depth range: [{relief['depth_min']:.3f}, {relief['depth_max']:.3f}], "
          f"took {time.time()-t0:.2f}s", flush=True)

    # ---- Step 3: Load mesh + head region + Umeyama
    print(f"\n[3/4] Load mesh + Umeyama fit", flush=True)
    t0 = time.time()
    scene = trimesh.load(cfg.input_mesh)
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    else:
        mesh = scene
    verts = np.asarray(mesh.vertices).copy()

    head_idx = find_head_vertices(verts, top_fraction=0.20)
    head_2d = project_orthographic(verts[head_idx])
    print(f"      mesh verts: {len(verts):,}, head-region verts: {len(head_idx):,}", flush=True)

    # Match MP anchors -> head vertices (nearest-neighbor in per-set-normalized 2D)
    key_indices = np.array([i for i in MP_ANCHOR_INDICES if i < lm["n"]], dtype=np.int64)
    head_key_local = match_head_landmarks_to_mp(head_2d, lm["landmarks_2d"], key_indices)
    img_pts_px = lm["landmarks_2d"][key_indices]        # (K, 2) image pixels
    mesh_pts_world = head_2d[head_key_local]             # (K, 2) mesh world coords

    # ---- Normalize BOTH sets to unit box [-1, 1] by their OWN landmark bbox ----
    # This puts image landmarks AND mesh landmarks each into their own [-1, 1] frame,
    # so the Umeyama scale should be ≈ 1.0 (only capturing residual rotation/shift
    # of landmark topology mismatches — nearest-neighbor matching noise).
    def _norm_to_own_bbox(pts):
        p_min = pts.min(axis=0)
        p_max = pts.max(axis=0)
        p_center = (p_min + p_max) / 2.0
        p_half = float(np.max((p_max - p_min) / 2.0))
        if p_half < 1e-9:
            p_half = 1e-9
        return (pts - p_center) / p_half, p_center, p_half

    img_pts_norm, img_lm_center, img_lm_half = _norm_to_own_bbox(img_pts_px)
    mesh_pts_norm, mesh_lm_center, mesh_lm_half = _norm_to_own_bbox(mesh_pts_world)

    # Fit Umeyama: mesh_norm -> img_norm  (source = mesh, target = image)
    R, t, S, c = umeyama(mesh_pts_norm, img_pts_norm, with_scale=True)
    c_clamped = float(np.clip(c, cfg.umeyama_clamp_min, cfg.umeyama_clamp_max))
    if abs(c - c_clamped) > 1e-6:
        print(f"      Umeyama scale {c:.4f} clamped -> {c_clamped:.4f} "
              f"(WARN: >10% off unity, alignment questionable — "
              f"landmark topology mismatch)", flush=True)
    else:
        print(f"      Umeyama scale: {c:.4f} (bbox-normalized, in-range)",
              flush=True)
    R_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"      Umeyama rotation: {R_deg:.2f} deg, translation (norm): {t}",
          flush=True)
    print(f"      img lm normalizer: center={img_lm_center}, half={img_lm_half:.2f}",
          flush=True)
    print(f"      mesh lm normalizer: center={mesh_lm_center}, half={mesh_lm_half:.4f}",
          flush=True)
    print(f"      took {time.time()-t0:.2f}s", flush=True)

    # ---- Step 4: Per-vertex relief sampling + radial-mask Z blend (no raster)
    # Rationale: Laplacian pyramid on a sparse rasterized head grid creates zero-
    # value holes at unfilled cells; smoothing across those holes pulls sampled Z
    # toward zero and inflates the delta to unusable levels. Directly sampling the
    # relief map per-vertex + radial-falloff mask around the face landmark centroid
    # produces predictable Z changes bounded by emboss_strength.
    print(f"\n[4/4] Per-vertex relief blend (face_weight={cfg.face_weight}, "
          f"radial-mask fall-off)", flush=True)
    t0 = time.time()

    # Project every head vertex to image px via mesh→image Umeyama transform
    head_2d_norm = (head_2d - mesh_lm_center) / mesh_lm_half           # (N, 2)
    photo_pts_norm = apply_similarity(head_2d_norm, R, t, c_clamped)   # (N, 2)
    photo_pts = photo_pts_norm * img_lm_half + img_lm_center           # (N, 2) image px

    # Diagnostic: projected head bbox in image space
    ppmin = photo_pts.min(axis=0)
    ppmax = photo_pts.max(axis=0)
    print(f"      head verts projected to image bbox: "
          f"[{ppmin[0]:.1f},{ppmin[1]:.1f}] -> [{ppmax[0]:.1f},{ppmax[1]:.1f}] "
          f"(face bbox was {tuple(round(v,1) for v in lm['bbox'])})", flush=True)

    # Sample relief map per vertex (crop-local coords via crop_bbox offset)
    ex1, ey1, ex2, ey2 = relief["crop_bbox"]
    relief_map = relief["relief"]  # (Hc, Wc) — signed offsets in world Z units
    Hc, Wc = relief_map.shape
    relief_at_head = np.zeros(len(head_idx), dtype=np.float32)
    for k, (px, py) in enumerate(photo_pts):
        cx = int(px - ex1)
        cy = int(py - ey1)
        if 0 <= cx < Wc and 0 <= cy < Hc:
            relief_at_head[k] = float(relief_map[cy, cx])

    # Radial mask: soft cosine fall-off from the face landmark centroid so the
    # relief blends into the head naturally at the edges, no hard seam.
    face_bbox = lm["bbox"]
    face_cx = (face_bbox[0] + face_bbox[2]) / 2.0
    face_cy = (face_bbox[1] + face_bbox[3]) / 2.0
    # Radius = half of face bbox diagonal × mask_expand → matches DAv2 crop
    face_radius = 0.5 * float(np.hypot(
        face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]
    )) * cfg.mask_expand
    dx = photo_pts[:, 0] - face_cx
    dy = photo_pts[:, 1] - face_cy
    dist = np.sqrt(dx * dx + dy * dy)
    radial = np.clip(1.0 - dist / max(face_radius, 1e-6), 0.0, 1.0)
    # Smooth cosine fall-off (matches Hann window)
    mask = 0.5 - 0.5 * np.cos(np.pi * radial)
    mask_final = mask * cfg.face_weight

    # Blend Z: new_z = orig_z + mask * relief (relief is already in world Z units)
    new_verts = verts.copy()
    new_verts[head_idx, 2] = verts[head_idx, 2] + mask_final * relief_at_head

    delta_z = np.abs(new_verts[head_idx, 2] - verts[head_idx, 2])
    n_touched = int((mask_final > 0.01).sum())
    print(f"      relief-in-bbox verts: {int((relief_at_head != 0).sum())} / {len(head_idx)}",
          flush=True)
    print(f"      mask coverage: {n_touched} verts have mask > 0.01 "
          f"(radial fall-off within {face_radius:.1f}px)", flush=True)
    print(f"      head Z delta: mean={delta_z.mean():.5f}, max={delta_z.max():.5f} "
          f"(should be <= face_weight × emboss_strength = "
          f"{cfg.face_weight * cfg.emboss_strength:.5f})", flush=True)
    print(f"      took {time.time()-t0:.2f}s", flush=True)

    # ---- Export
    print(f"\n[export] Writing GLB -> {cfg.output}", flush=True)
    mesh_out = mesh.copy()
    mesh_out.vertices = new_verts
    mesh_out.export(cfg.output)
    size_kb = cfg.output.stat().st_size / 1024
    print(f"      wrote {size_kb:.1f} KB", flush=True)

    print(f"\n[DONE] total elapsed: {time.time()-t_start:.2f}s", flush=True)
    return cfg.output


def run_sweep(base: Config) -> Path:
    """Execute 12-config sweep, write summary JSON alongside output dir."""
    import numpy as np
    configs = build_sweep_configs(base)
    print(f"[sweep] {len(configs)} configs", flush=True)

    summary = []
    for i, (cid, cfg) in enumerate(configs):
        print(f"\n{'#'*66}\n# [{i+1}/{len(configs)}] config: {cid}\n{'#'*66}", flush=True)
        t0 = time.time()
        try:
            out_path = run_pipeline(cfg)
            summary.append({
                "config_id": cid,
                "config": cfg.to_dict(),
                "output": str(out_path),
                "elapsed_s": round(time.time() - t0, 2),
                "status": "ok",
            })
        except Exception as e:
            summary.append({
                "config_id": cid,
                "config": cfg.to_dict(),
                "error": f"{type(e).__name__}: {e}",
                "status": "fail",
            })

    summary_path = base.output.parent / f"{base.output.stem}_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sweep] summary -> {summary_path}", flush=True)
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pipeline_v11_depth_umeyama",
        description="Module C+D: DAv2 face relief + Umeyama fit + Laplacian pyramid blend.",
    )
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dav2-model", choices=list(DAV2_ENCODERS.keys()), default="base")
    ap.add_argument("--mask-expand", type=float, default=1.2)
    ap.add_argument("--lap-levels", type=int, default=6)
    ap.add_argument("--face-weight", type=float, default=0.7)
    ap.add_argument("--mp-mode", type=int, choices=[468, 478], default=478)
    ap.add_argument("--umeyama-clamp-min", type=float, default=0.9)
    ap.add_argument("--umeyama-clamp-max", type=float, default=1.1)
    ap.add_argument("--depth-norm", choices=["minmax", "percentile"], default="percentile")
    ap.add_argument("--emboss-strength", type=float, default=0.010)
    ap.add_argument("--sweep", action="store_true",
                    help="Run 12-config sweep (DAv2 {base,large} x fw {0.5,0.7,0.9} x emboss {0.008,0.012}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate args + print plan without loading torch/mediapipe.")
    return ap


def parse_args(argv: Optional[List[str]] = None) -> Config:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return Config(
        input_image=args.input_image,
        input_mesh=args.input_mesh,
        output=args.output,
        dav2_model=args.dav2_model,
        mask_expand=args.mask_expand,
        lap_levels=args.lap_levels,
        face_weight=args.face_weight,
        mp_mode=args.mp_mode,
        umeyama_clamp_min=args.umeyama_clamp_min,
        umeyama_clamp_max=args.umeyama_clamp_max,
        depth_norm=args.depth_norm,
        emboss_strength=args.emboss_strength,
        sweep=args.sweep,
        dry_run=args.dry_run,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    if cfg.sweep and cfg.dry_run:
        print("[dry-run + sweep] Listing 12 sweep configs:", flush=True)
        for cid, c in build_sweep_configs(cfg):
            print(f"  {cid} -> {c.output.name}")
        return 0
    if cfg.sweep:
        run_sweep(cfg)
    else:
        run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
