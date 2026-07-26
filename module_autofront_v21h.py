"""V21h Bug C fix (Fable v3): auto-front detection for TripoSG output.

Renders the untextured mesh from 8 azimuths (gray Lambert shading in software),
runs MediaPipe FaceDetector on each render, picks the azimuth with the highest
face-detection confidence. Returns that azimuth as `front_offset_deg` — the wrap
baker rotates user-space azimuth by this offset so preset "front" (az=0) lands
on the mesh's ACTUAL front face regardless of TripoSG's internal orientation.

No display / no pyrender / no GPU. Pure numpy software rasterizer + MediaPipe.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


def _view_basis_from_azel(az_deg: float, el_deg: float):
    """Same convention as module_wrap_texture_v21._view_basis_from_azel."""
    theta = math.radians(el_deg)
    phi = math.radians(az_deg)
    cx = math.sin(phi) * math.cos(theta)
    cy = math.sin(theta)
    cz = math.cos(phi) * math.cos(theta)
    w = np.array([cx, cy, cz], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(w, up))) > 0.995:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(up, w); right /= (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(w, right); true_up /= (np.linalg.norm(true_up) + 1e-9)
    return right, true_up, w


def _render_gray(verts: np.ndarray, faces: np.ndarray, az: float, el: float,
                 img_h: int = 384, img_w: int = 384) -> np.ndarray:
    """Software Lambert-shaded gray rasterizer. Returns HxWx3 uint8 RGB image.

    Camera at unit sphere, looking at origin. Ambient=40 + Lambert=180*dot(N, -view_dir).
    """
    u_vec, v_vec, w_vec = _view_basis_from_azel(az, el)
    # depth (higher = closer to camera in w_vec direction)
    verts_u = verts @ u_vec
    verts_v = verts @ v_vec
    verts_d = verts @ w_vec

    # image-space projection with y flip
    v_min = np.array([verts_u.min(), verts_v.min()], dtype=np.float32)
    v_max = np.array([verts_u.max(), verts_v.max()], dtype=np.float32)
    v_center = (v_min + v_max) / 2.0
    v_span = float(np.max(v_max - v_min)); v_span = max(v_span, 1e-9)
    # Fit inside 80% of image so silhouette isn't cut off
    scale = 0.80 * min(img_h, img_w) / v_span
    cx_img = img_w / 2.0; cy_img = img_h / 2.0

    def w2i(uu, vv):
        x = (uu - v_center[0]) * scale + cx_img
        y = -(vv - v_center[1]) * scale + cy_img
        return x, y

    tri_u = verts_u[faces]; tri_v = verts_v[faces]; tri_d = verts_d[faces]
    tri_x = np.empty_like(tri_u); tri_y = np.empty_like(tri_u)
    for k in range(3):
        tri_x[:, k], tri_y[:, k] = w2i(tri_u[:, k], tri_v[:, k])

    # face normals + lambert shade
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    face_norm = np.cross(v1 - v0, v2 - v0)
    face_norm /= (np.linalg.norm(face_norm, axis=1, keepdims=True) + 1e-9)
    lam = np.maximum(0.0, face_norm @ w_vec)   # 0..1
    shade = np.clip(40 + 180.0 * lam, 0, 255).astype(np.uint8)
    front_mask = (face_norm @ w_vec) > 0.02

    # per-pixel z-buffer + shade paint
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    depth = np.full((img_h, img_w), -np.inf, dtype=np.float32)
    N = len(faces)
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
        s = int(shade[i])
        for c in range(3):
            img[y0:y1, x0:x1, c] = np.where(win, s, img[y0:y1, x0:x1, c])
        depth[y0:y1, x0:x1] = np.where(win, z, depth[y0:y1, x0:x1])
    return img


def _mediapipe_face_confidence(img_rgb: np.ndarray) -> float:
    """Run MediaPipe FaceDetector on RGB image. Returns highest detection score, or 0."""
    try:
        import mediapipe as mp
    except Exception as e:
        print(f"      [autofront] mediapipe unavailable: {e}", flush=True)
        return 0.0
    # Use short-range model_selection=0 for close-up portraits (mesh render is full-frame face)
    try:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.05
        ) as det:
            res = det.process(img_rgb)
            if not res.detections:
                return 0.0
            scores = []
            for d in res.detections:
                s = float(d.score[0]) if hasattr(d, 'score') and len(d.score) > 0 else 0.0
                scores.append(s)
            return max(scores) if scores else 0.0
    except Exception as e:
        print(f"      [autofront] mediapipe detect error: {e}", flush=True)
        return 0.0


def auto_detect_front_azimuth(mesh_path: Path,
                              n_azimuths: int = 8,
                              elevation_deg: float = 0.0,
                              img_size: int = 384,
                              debug_dir: Optional[Path] = None) -> float:
    """Render mesh from n_azimuths equally-spaced angles, MediaPipe face-detect each,
    return the azimuth (in module_wrap_texture_v21 convention) that yielded highest confidence.

    Returns 0.0 if detection failed on all angles (fallback: assume mesh front == 0°).
    """
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        mesh = geoms[0]
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    # center + normalize scale
    center = (verts.min(0) + verts.max(0)) / 2.0
    verts_c = verts - center
    scale = 1.0 / max(float(np.max(np.linalg.norm(verts_c, axis=1))), 1e-9)
    verts_n = verts_c * scale

    az_list = [360.0 * i / n_azimuths for i in range(n_azimuths)]
    scores = []
    for az in az_list:
        img = _render_gray(verts_n, faces, az=az, el=elevation_deg,
                           img_h=img_size, img_w=img_size)
        s = _mediapipe_face_confidence(img)
        scores.append(s)
        print(f"      [autofront] az={az:6.1f}° face_conf={s:.3f}", flush=True)
        if debug_dir is not None:
            try:
                from PIL import Image
                debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(debug_dir / f"autofront_az{int(az):03d}.png")
            except Exception:
                pass

    best_i = int(np.argmax(scores))
    best_score = scores[best_i]
    best_az = az_list[best_i]
    if best_score < 0.10:
        print(f"      [autofront] WARN best_score={best_score:.3f} < 0.10 -> "
              f"detection unreliable, using 0° default", flush=True)
        return 0.0
    print(f"      [autofront] BEST az={best_az:.1f}° score={best_score:.3f}", flush=True)
    return float(best_az)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--n-az", type=int, default=8)
    args = ap.parse_args()
    az = auto_detect_front_azimuth(args.mesh, n_azimuths=args.n_az, debug_dir=args.debug_dir)
    print(f"detected_front_az={az}")
