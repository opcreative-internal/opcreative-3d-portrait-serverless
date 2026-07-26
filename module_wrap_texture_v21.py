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
    wrap_direction: str = "front"   # front | back | left | right  (backward compat)
    # V21c free rotation (per Kent real-test feedback: F/B/L/R too rigid).
    # If wrap_azimuth or wrap_elevation set (not None), overrides wrap_direction.
    wrap_azimuth: Optional[float] = None    # 0..360 degrees; 0=front(+Z), 90=right(+X), 180=back, 270=left
    wrap_elevation: Optional[float] = None  # -90..90 degrees; 0=equator (default)
    # V21h Fable v3 Bug C: azimuth offset injected by auto-front detector (defaults 0).
    # The pipeline runs 8-azimuth face-detect on the raw TripoSG mesh and this offset
    # rotates user-space azimuth so preset "front"=0 lands on the mesh's true face.
    wrap_front_offset: float = 0.0
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
        # If neither azimuth nor elevation given, use preset enum
        if self.wrap_azimuth is None and self.wrap_elevation is None:
            preset_to_az = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
            self.wrap_azimuth = preset_to_az.get(self.wrap_direction, 0.0)
            self.wrap_elevation = 0.0
        else:
            if self.wrap_azimuth is None: self.wrap_azimuth = 0.0
            if self.wrap_elevation is None: self.wrap_elevation = 0.0


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


def _view_dir_from_azel(azimuth_deg: float, elevation_deg: float):
    """V21c free rotation: azimuth/elevation -> unit view direction vector.
    Convention: azimuth 0=front(+Z), 90=right(+X), 180=back(-Z), 270=left(-X);
    elevation 0=equator, +90=above, -90=below.
    Returns (dx, dy, dz) unit vector pointing FROM camera TOWARD subject.
    """
    import math
    theta = math.radians(elevation_deg)
    phi = math.radians(azimuth_deg)
    # camera position on unit sphere; view_dir = -camera_pos (looks toward origin)
    cx = math.sin(phi) * math.cos(theta)
    cy = math.sin(theta)
    cz = math.cos(phi) * math.cos(theta)
    return (-cx, -cy, -cz)  # view direction (subject-facing)


def _view_basis_from_azel(azimuth_deg: float, elevation_deg: float):
    """V21h Fable v3 Bug A fix: TRUE orthonormal basis from azimuth/elevation.
    NO axis-snapping. Slider 0-360 produces genuinely arbitrary projection planes.

    Returns (u_vec, v_vec, w_vec) as 3-tuples where:
      - w_vec = unit vector pointing FROM subject TOWARD camera (== -view_dir)
      - u_vec = image "right" in world space (perpendicular to w and world-up)
      - v_vec = image "up" in world space (perpendicular to w and u)

    Convention matches _view_dir_from_azel: az=0 -> cam at +Z, az=90 -> cam at +X.
    """
    import math
    import numpy as np
    theta = math.radians(elevation_deg)
    phi = math.radians(azimuth_deg)
    # camera position on unit sphere
    cx = math.sin(phi) * math.cos(theta)
    cy = math.sin(theta)
    cz = math.cos(phi) * math.cos(theta)
    w = np.array([cx, cy, cz], dtype=np.float32)      # subject -> camera
    # world up = +Y (image height axis). Handle pole degenerate: use +Z if too close to up.
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(w, up))) > 0.995:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # image right = normalize(up x w). For az=0 el=0 gives right=+X.
    right = np.cross(up, w)
    right /= (np.linalg.norm(right) + 1e-9)
    # image up = normalize(w x right). Guaranteed orthonormal.
    true_up = np.cross(w, right)
    true_up /= (np.linalg.norm(true_up) + 1e-9)
    return right, true_up, w


def _dir_axes_from_view(view_dir):
    """V21h DEPRECATED: axis-snap projection (Fable v3 Bug A). Kept for backward
    compat; new code path uses _view_basis_from_azel + project_verts_basis.
    """
    import math
    dx, dy, dz = view_dir
    ax = abs(dx); ay = abs(dy); az = abs(dz)
    if az >= ax and az >= ay:
        return 0, 1, 1.0 if dz < 0 else -1.0, 1.0, 2, 1.0 if dz > 0 else -1.0
    elif ax >= ay:
        return 2, 1, 1.0 if dx > 0 else -1.0, 1.0, 0, 1.0 if dx > 0 else -1.0
    else:
        return 0, 2, 1.0, 1.0 if dy > 0 else -1.0, 1, 1.0 if dy > 0 else -1.0


def _detect_face_landmarks_5pt(img_rgb):
    """V21h.4 Fable v3: MediaPipe FaceLandmarker (478-set) on RGB image.
    Returns np.float32 (5, 2) array of (x, y) pixel coords for
    [eye_outer_L, eye_outer_R, nose_tip, mouth_L, mouth_R], or None on fail.

    Uses the 478-landmark model (`face_landmarker.task` pre-baked at /workspace).
    Head-crop is applied when a face is detected in the top ~50% of the image so
    the detector gets the face at higher effective resolution.
    """
    import os
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except Exception as e:
        print(f"      [align] mediapipe import fail: {e}", flush=True)
        return None
    model_path = os.environ.get(
        "FACE_LANDMARKER_MODEL", "/workspace/face_landmarker.task"
    )
    try:
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        with vision.FaceLandmarker.create_from_options(opts) as lm:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(img_rgb))
            res = lm.detect(mp_img)
            if not res.face_landmarks:
                return None
            face = res.face_landmarks[0]
            h, w = img_rgb.shape[:2]
            # 478-set indices per MediaPipe FaceLandmarker canonical mesh:
            # 33=left eye outer, 263=right eye outer, 1=nose tip, 61=mouth left, 291=mouth right
            idx = [33, 263, 1, 61, 291]
            pts = np.array([[face[i].x * w, face[i].y * h] for i in idx], dtype=np.float32)
            return pts
    except Exception as e:
        print(f"      [align] FaceLandmarker error: {e}", flush=True)
        return None


def _render_mesh_gray_for_align(verts, faces, u_vec, v_vec, w_vec,
                                v_min, v_max, img_h=384, img_w=384):
    """V21h.4: render a skin-tinted Lambert image of the mesh using the SAME
    projection basis as the bake (no fresh recomputation of extents/scale, so
    landmark px in this image map back through render inverse -> mesh_uv -> bake
    photo_px unambiguously). Returns (img_rgb_uint8, render_scale, render_center, v_center).
    """
    verts_np = np.asarray(verts, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32)
    verts_u = verts_np @ u_vec
    verts_v = verts_np @ v_vec
    verts_d = verts_np @ w_vec
    v_center = (v_min + v_max) / 2.0
    # Render at 90% of viewport to add padding
    u_span = max(float(v_max[0] - v_min[0]), 1e-9)
    vv_span = max(float(v_max[1] - v_min[1]), 1e-9)
    span = max(u_span, vv_span)
    render_scale = 0.90 * min(img_h, img_w) / span
    render_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)

    def w2r(pts_uv):
        cent = pts_uv - v_center
        cent[:, 1] *= -1.0
        return cent * render_scale + render_center

    world_uv = np.stack([verts_u, verts_v], axis=1).astype(np.float32)
    tri_pts_uv = world_uv[faces_np]
    tri_x = np.empty((len(faces_np), 3), dtype=np.float32)
    tri_y = np.empty((len(faces_np), 3), dtype=np.float32)
    tri_d = verts_d[faces_np]
    for k in range(3):
        pxy = w2r(tri_pts_uv[:, k, :].copy())
        tri_x[:, k] = pxy[:, 0]; tri_y[:, k] = pxy[:, 1]

    # face normals + offset key light (matches autofront _render_gray)
    v0 = verts_np[faces_np[:, 0]]; v1 = verts_np[faces_np[:, 1]]; v2 = verts_np[faces_np[:, 2]]
    face_norm = np.cross(v1 - v0, v2 - v0)
    face_norm /= (np.linalg.norm(face_norm, axis=1, keepdims=True) + 1e-9)
    light = w_vec + 0.5 * u_vec + 0.6 * v_vec
    light /= (np.linalg.norm(light) + 1e-9)
    lam = np.maximum(0.0, face_norm @ light)
    shade = np.clip(40 + 180.0 * lam, 0, 255).astype(np.uint8)
    front_mask = (face_norm @ w_vec) > 0.02

    depth = np.full((img_h, img_w), -np.inf, dtype=np.float32)
    gray = np.zeros((img_h, img_w), dtype=np.uint8)
    N = len(faces_np)
    for i in range(N):
        if not front_mask[i]: continue
        px = tri_x[i]; py = tri_y[i]; pd = tri_d[i]
        x0 = int(np.floor(px.min())); y0 = int(np.floor(py.min()))
        x1 = int(np.ceil(px.max())) + 1; y1 = int(np.ceil(py.max())) + 1
        x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, img_w); y1 = min(y1, img_h)
        if x1 <= x0 or y1 <= y0: continue
        e1x = px[1] - px[0]; e1y = py[1] - py[0]
        e2x = px[2] - px[0]; e2y = py[2] - py[0]
        denom = e1x * e2y - e1y * e2x
        if abs(denom) < 1e-6: continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs - px[0]; dy = ys - py[0]
        b1 = (dx * e2y - dy * e2x) / denom
        b2 = (e1x * dy - e1y * dx) / denom
        b0 = 1.0 - b1 - b2
        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        z = b0 * pd[0] + b1 * pd[1] + b2 * pd[2]
        win = inside & (z > depth[y0:y1, x0:x1])
        gray[y0:y1, x0:x1] = np.where(win, int(shade[i]), gray[y0:y1, x0:x1])
        depth[y0:y1, x0:x1] = np.where(win, z, depth[y0:y1, x0:x1])

    # Skin tint per Fable v3: FaceLandmarker was trained on skin-toned photos, gray render
    # scores poorly. Tint on channel means R*1.0, G*0.86, B*0.74.
    rgb = np.stack([gray * 1.0, gray * 0.86, gray * 0.74], axis=2)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, float(render_scale), render_center.astype(np.float32), v_center.astype(np.float32)


def _rasterize_zbuffer_owner(tri_img_pts, tri_depths, img_h, img_w, front_mask):
    """V21h Fable v3 Bug B fix: per-pixel z-buffer + owner map.

    For each image pixel, records which front-facing triangle is NEAREST to camera.
    Later during bake, atlas triangle i only samples image pixels where owner == i,
    so occluded surfaces do NOT receive occluder's texture (fixes projection bleed).

    tri_img_pts: (N, 3, 2)   -- image-space triangle vertices
    tri_depths:  (N, 3)      -- world-space depth per vertex (higher = closer to camera)
    front_mask:  (N,) bool
    Returns owner (H, W) int32, -1 = background.
    """
    import numpy as np
    depth_buf = np.full((img_h, img_w), -np.inf, dtype=np.float32)
    owner = np.full((img_h, img_w), -1, dtype=np.int32)
    N = len(tri_img_pts)
    for i in range(N):
        if not front_mask[i]:
            continue
        pts = tri_img_pts[i]
        d0, d1, d2 = tri_depths[i]
        x0 = int(np.floor(pts[:, 0].min()))
        y0 = int(np.floor(pts[:, 1].min()))
        x1 = int(np.ceil(pts[:, 0].max())) + 1
        y1 = int(np.ceil(pts[:, 1].max())) + 1
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, img_w); y1 = min(y1, img_h)
        if x1 <= x0 or y1 <= y0:
            continue
        v0 = pts[0]; e1 = pts[1] - v0; e2 = pts[2] - v0
        denom = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(denom) < 1e-6:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px = xs - v0[0]; py = ys - v0[1]
        b1 = (px * e2[1] - py * e2[0]) / denom
        b2 = (e1[0] * py - e1[1] * px) / denom
        b0 = 1.0 - b1 - b2
        inside = (b0 >= 0.0) & (b1 >= 0.0) & (b2 >= 0.0)
        z = b0 * d0 + b1 * d1 + b2 * d2
        # nearest wins (higher depth == closer to camera in our convention)
        win = inside & (z > depth_buf[y0:y1, x0:x1])
        depth_buf[y0:y1, x0:x1] = np.where(win, z, depth_buf[y0:y1, x0:x1])
        owner[y0:y1, x0:x1] = np.where(win, i, owner[y0:y1, x0:x1])
    return owner


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
                           azimuth_deg=None, elevation_deg=None,
                           clahe_clip=3.0, clahe_tile=16,
                           front_offset_deg: float = 0.0,
                           subject_alpha=None,
                           dec_faces=None):
    """V21h.4 Fable v3: landmark-similarity fold-in for face detail alignment.

    NEW `dec_faces` param (optional): the decimated mesh faces used to render a
    skin-tinted preview for landmark detection. Only used if MediaPipe
    FaceLandmarker fires on both photo AND render — otherwise falls back to
    the V21h.3 subject-alpha-bbox mapping (no regression).
    """
    """V21h Fable v3 rewrite: true orthonormal projection + face-normal cull + z-buffer.

    Bug A fix: uses _view_basis_from_azel (no axis-snapping) so arbitrary az/el actually work.
    Bug B fix: face-normal culling AND per-pixel z-buffer -> no projection bleed.
    Bug C support: front_offset_deg lets caller inject auto-detected front azimuth.

    All callers now go through the (az, el) path — wrap_direction enum is converted to
    the equivalent azimuth internally.
    """
    import math
    import numpy as np
    from PIL import Image
    import cv2

    photo_np = np.array(photo_pil.convert("RGB"))
    img_h, img_w = photo_np.shape[:2]

    gray = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    gray_eq = clahe.apply(gray)
    photo_gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

    # Convert preset enum to azimuth if user did not supply az/el
    if azimuth_deg is None and elevation_deg is None:
        preset_az = {"front": 0.0, "back": 180.0, "left": 270.0, "right": 90.0}
        azimuth_deg = preset_az.get(wrap_direction, 0.0)
        elevation_deg = 0.0
    az = float(azimuth_deg or 0.0) + float(front_offset_deg)
    el = float(elevation_deg or 0.0)
    u_vec, v_vec, w_vec = _view_basis_from_azel(az, el)
    print(f"      V21h bake az={az:.1f} el={el:.1f} (front_offset={front_offset_deg:.1f}) "
          f"u={u_vec.round(3).tolist()} v={v_vec.round(3).tolist()} w={w_vec.round(3).tolist()}",
          flush=True)

    # Project verts onto (u, v) plane; depth along +w (higher = closer to camera).
    verts_np = np.asarray(verts_world, dtype=np.float32)
    verts_u = verts_np @ u_vec
    verts_v = verts_np @ v_vec
    verts_d = verts_np @ w_vec  # depth (subject -> camera direction)

    world_uv = np.stack([verts_u, verts_v], axis=1).astype(np.float32)
    v_min = world_uv.min(0); v_max = world_uv.max(0)
    v_center = (v_min + v_max) / 2.0
    # V21h.3 Fable v3 Bug Q2 fix: map mesh bbox -> subject alpha bbox (per-axis).
    # OLD: `min(img_w, img_h) / max(u_span, v_span)` — a 3:4 portrait mesh mapped mesh
    # HEIGHT to img_w rows leaving big vertical padding; mesh WIDTH used only ~40% of
    # img_w. Half the mesh triangles projected outside the frame → bbox clamped → empty
    # `owner` → skipped by _rasterize_zbuffer_owner → flat-gray-128 fill on lower body.
    # NEW: fit mesh projected bbox to subject-alpha bbox from BG remove. Same-subject
    # same-view means silhouette aspect ≈ mesh aspect by construction, per-axis = registration.
    if subject_alpha is not None and (subject_alpha > 8).any():
        ys, xs = np.where(subject_alpha > 8)
        sx0, sx1 = int(xs.min()), int(xs.max()) + 1
        sy0, sy1 = int(ys.min()), int(ys.max()) + 1
        print(f"      subject_bbox={sx0},{sy0}-{sx1},{sy1} of {img_w}x{img_h}", flush=True)
    else:
        sx0, sy0, sx1, sy1 = 0, 0, img_w, img_h
        print(f"      no alpha - using full image bbox {img_w}x{img_h}", flush=True)
    u_span = max(float(v_max[0] - v_min[0]), 1e-9)
    vv_span = max(float(v_max[1] - v_min[1]), 1e-9)
    scale_xy = np.array([(sx1 - sx0) / u_span, (sy1 - sy0) / vv_span], dtype=np.float32)
    img_center = np.array([(sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0], dtype=np.float32)

    def w2i(pts):
        cent = pts - v_center
        cent[:, 1] *= -1.0   # flip Y (image origin top-left, world Y up)
        return cent * scale_xy + img_center

    # ============================================================
    # V21h.4 Fable v3: landmark-similarity fold-in.
    # Detect 5 face landmarks on photo AND on skin-tinted mesh render.
    # Compute rigid similarity s, t on the shared coord frame -> fold into
    # scale_xy + img_center so photo_center + eye distance land on the mesh's
    # face landmarks. Warn+skip fallback on any of: no MediaPipe, no face on
    # photo, no face on render, sign mirror mismatch, sanity gate rejected.
    # ============================================================
    align_status = 'bbox_fallback:disabled'
    if dec_faces is not None and dec_faces.size > 0:
        photo_rgb_for_detect = photo_np
        if subject_alpha is not None and subject_alpha.size > 0:
            # Composite alpha over neutral gray so face detector sees a clean subject.
            a = subject_alpha.astype(np.float32) / 255.0
            if a.ndim == 2: a = a[..., None]
            photo_rgb_for_detect = (photo_np.astype(np.float32) * a
                                    + 127.0 * (1 - a)).astype(np.uint8)
        P = _detect_face_landmarks_5pt(photo_rgb_for_detect)
        if P is None:
            align_status = 'bbox_fallback:no_face_on_photo'
        else:
            # Render mesh at same projection basis (u_vec, v_vec, w_vec, v_min, v_max)
            render_rgb, render_scale, render_center, _vc_r = _render_mesh_gray_for_align(
                verts_np, dec_faces, u_vec, v_vec, w_vec, v_min, v_max,
                img_h=512, img_w=512,
            )
            R = _detect_face_landmarks_5pt(render_rgb)
            if R is None:
                align_status = 'bbox_fallback:no_face_on_render'
            else:
                # Map render_px -> mesh_uv (invert render mapping), then mesh_uv -> photo_px
                # via the CURRENT (bbox) bake mapping. Result is "where the bake currently
                # thinks the mesh's face landmarks land on the photo".
                # render_px = render_center + render_scale * (mesh_uv - v_center) with Y flip
                # so mesh_uv = ((render_px - render_center) / render_scale) with Y unflip + v_center
                R_centered = R - render_center
                R_centered[:, 1] *= -1.0
                mesh_uv_R = R_centered / render_scale + v_center
                # Bake photo_px = img_center + scale_xy * (mesh_uv - v_center) with Y flip
                cent = mesh_uv_R - v_center
                cent[:, 1] *= -1.0
                Mp = cent * scale_xy + img_center

                # Mirror-sign check: photo eye ordering left→right should match Mp
                sgn_ok = np.sign(P[1, 0] - P[0, 0]) == np.sign(Mp[1, 0] - Mp[0, 0])
                if not sgn_ok:
                    align_status = 'bbox_fallback:mirror_sign_mismatch'
                else:
                    # Similarity: P = s * Mp + t
                    P_c = P.mean(0); M_c = Mp.mean(0)
                    P_r = np.linalg.norm(P - P_c, axis=1)
                    M_r = np.linalg.norm(Mp - M_c, axis=1)
                    s = float(np.median(P_r) / max(np.median(M_r), 1e-6))
                    t = P_c - s * M_c
                    # Sanity gate
                    max_translate = 0.35 * max(img_h, img_w)
                    if not (0.6 < s < 1.6) or np.any(np.abs(t) > max_translate):
                        align_status = f'bbox_fallback:sanity s={s:.2f} t={t.round(1).tolist()}'
                    else:
                        # Fold into mapping: new_scale_xy = s * old; new_img_center = s * old + t
                        scale_xy = scale_xy * s
                        img_center = s * img_center + t
                        align_status = f'landmark_sim s={s:.3f} t={t.round(1).tolist()}'
                        # redefine w2i to use updated params
                        def w2i(pts):
                            cent = pts - v_center
                            cent[:, 1] *= -1.0
                            return cent * scale_xy + img_center
    print(f"      align: {align_status}", flush=True)

    tri_orig_idx = vmapping[faces_uv]                        # (N, 3) vertex indices in original mesh
    tri_world_pts = verts_np[tri_orig_idx]                    # (N, 3, 3)
    tri_img_pts = w2i(world_uv[tri_orig_idx.reshape(-1)]).reshape(-1, 3, 2)
    tri_depths = verts_d[tri_orig_idx]                        # (N, 3)
    atlas_pts_all = (uvs[faces_uv] * atlas_res).astype(np.float32)

    # BUG B FIX #1: face-normal culling.
    # Face normal = normalize((v1-v0) x (v2-v0)); front-facing if normal points somewhat toward camera.
    v0 = tri_world_pts[:, 0]; v1 = tri_world_pts[:, 1]; v2 = tri_world_pts[:, 2]
    face_norm = np.cross(v1 - v0, v2 - v0)
    face_norm /= (np.linalg.norm(face_norm, axis=1, keepdims=True) + 1e-9)
    # dot with w_vec (subject -> camera) > 0 means normal faces the camera.
    # Threshold 0.05 skips grazing triangles that would streak.
    facing = face_norm @ w_vec
    front_mask = facing > 0.05

    # BUG B FIX #2: per-pixel z-buffer to resolve occlusion.
    print(f"      V21h building z-buffer ({int(front_mask.sum())} front-facing tris)", flush=True)
    owner = _rasterize_zbuffer_owner(tri_img_pts, tri_depths, img_h, img_w, front_mask)

    atlas = np.full((atlas_res, atlas_res, 3), 128, dtype=np.uint8)
    coverage = np.zeros((atlas_res, atlas_res), dtype=np.uint8)

    n_baked = 0
    n_occluded_skip = 0
    for i in range(len(faces_uv)):
        if not front_mask[i]:
            continue
        img_tri = tri_img_pts[i]; atlas_tri = atlas_pts_all[i]
        x0 = max(int(np.floor(atlas_tri[:, 0].min())) - 1, 0)
        y0 = max(int(np.floor(atlas_tri[:, 1].min())) - 1, 0)
        x1 = min(int(np.ceil(atlas_tri[:, 0].max())) + 1, atlas_res)
        y1 = min(int(np.ceil(atlas_tri[:, 1].max())) + 1, atlas_res)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        local_tri = atlas_tri - np.array([x0, y0], dtype=np.float32)
        try:
            M = cv2.getAffineTransform(img_tri.astype(np.float32), local_tri)
            warp = cv2.warpAffine(photo_gray_rgb, M, (w, h))
            # Per-pixel visibility mask: build owner-remapped mask by warping
            # a binary "this tri owns" image from source (owner==i) into local atlas frame.
            src_own = (owner == i).astype(np.uint8) * 255
            own_warp = cv2.warpAffine(src_own, M, (w, h), flags=cv2.INTER_NEAREST)
        except cv2.error:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_tri.astype(np.int32), 255)
        # AND with per-pixel visibility mask (only pixels this tri owns in source image)
        effective_mask = np.logical_and(mask > 0, own_warp > 128)
        pixels_visible = int(effective_mask.sum())
        pixels_atlas = int((mask > 0).sum())
        if pixels_visible == 0 and pixels_atlas > 0:
            n_occluded_skip += 1
            continue
        roi = atlas[y0:y1, x0:x1]
        roi[effective_mask] = warp[effective_mask]
        coverage[y0:y1, x0:x1] |= effective_mask.astype(np.uint8) * 255
        n_baked += 1

    n_backface = int((~front_mask).sum())
    print(f"      V21h baked {n_baked}/{len(faces_uv)} tris "
          f"(back-face={n_backface} occluded-skip={n_occluded_skip})", flush=True)
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

    # 3. Preprocess input image + ortho bake with V21 direction axis OR V21c azimuth/elevation
    print(f"\n[3/5] Preprocess + ortho bake dir={cfg.wrap_direction} "
          f"az={cfg.wrap_azimuth} el={cfg.wrap_elevation} "
          f"{cfg.atlas_res}x{cfg.atlas_res}", flush=True)
    # V21h.3 Fable v3: load image preserving RGBA if the BG-remove stage produced alpha.
    # Alpha bbox is what drives the mesh-to-image registration in bake_ortho_texture_v21.
    photo_raw = Image.open(cfg.input_image)
    subject_alpha = None
    if photo_raw.mode == 'RGBA':
        subject_alpha = np.array(photo_raw.split()[-1])
        print(f"      photo has alpha, mean={subject_alpha.mean():.1f}/255", flush=True)
    photo = photo_raw.convert("RGB")
    photo = _preprocess_image(photo, cfg.flip_h, cfg.flip_v, cfg.brightness, cfg.contrast)
    atlas_rgb, coverage = bake_ortho_texture_v21(
        photo, dec_verts, uvs, faces_uv, vmapping,
        cfg.atlas_res, wrap_direction=cfg.wrap_direction,
        azimuth_deg=cfg.wrap_azimuth, elevation_deg=cfg.wrap_elevation,
        clahe_clip=cfg.clahe_clip, clahe_tile=cfg.clahe_tile,
        front_offset_deg=cfg.wrap_front_offset,
        subject_alpha=subject_alpha,
        dec_faces=dec_faces,     # V21h.4: enables landmark-similarity fold-in
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

    # 5. Export (V21h.3 Fable v3: GLB uses 2048² preview texture to fit RunPod ≤3MB
    # inline response cap and avoid catbox truncation "Invalid typed array length: 4".
    # Full 8192² PNG stays with OBJ + as separate _texture.png artifact for engraving.)
    print("\n[5/5] Export OBJ + MTL + PNG + STL + GLB", flush=True)
    stem = cfg.out_stem
    obj_path = stem.with_suffix(".obj")
    tex_mesh.export(obj_path)   # OBJ + MTL + auto-written _material_0.png at full res
    stl_path = stem.with_suffix(".stl")
    geo_mesh.export(stl_path)

    # GLB with downsampled texture — Fable v3 fix for issue #5
    _preview_res = min(2048, cfg.atlas_res)
    if _preview_res < cfg.atlas_res:
        preview_img = Image.fromarray(atlas_for_gltf).resize((_preview_res, _preview_res), Image.LANCZOS)
        print(f"      GLB preview texture: downsampled {cfg.atlas_res}²→{_preview_res}²", flush=True)
    else:
        preview_img = Image.fromarray(atlas_for_gltf)
    mat_preview = SimpleMaterial(image=preview_img)
    tex_mesh_preview = trimesh.Trimesh(
        vertices=xatlas_verts, faces=faces_uv,
        visual=TextureVisuals(uv=uvs, image=preview_img, material=mat_preview),
        process=False,
    )
    glb_path = stem.with_suffix(".glb")
    tex_mesh_preview.export(glb_path)

    # GLB integrity check per Fable v3: header bytes 8-12 declare total length.
    # Mismatch with actual file size means truncated write; fail loud.
    try:
        _raw = open(glb_path, 'rb').read()
        _decl = int.from_bytes(_raw[8:12], 'little')
        _actual = len(_raw)
        print(f"      GLB integrity: declared={_decl} actual={_actual} match={_decl == _actual}", flush=True)
        if _decl != _actual:
            print(f"      WARN glb declared_length != file_length -> truncation. Downstream viewer will hit 'Invalid typed array length'", flush=True)
    except Exception as _e:
        print(f"      GLB integrity check failed: {_e}", flush=True)

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
