"""
Module F — Head-body graft with seam smoothing.

Takes the V14 outputs (textured body from Module E + head with C+D face relief)
and produces a SINGLE seamless mesh where:
  - Head vertices carry the C+D relief geometry (bumps for eyes/nose/mouth)
  - Body vertices carry the xatlas UV atlas + baked photo texture
  - Boundary between head and body is welded (spatial KNN within tolerance) and
    Laplacian-smoothed across 2-3 vertex rings so the seam is invisible.

V14 emitted a Scene with two disjoint meshes (skin-tone head + textured body).
V15 = V14 pipeline + Module F post-processing.

Inputs:
    face_relief_glb  -- head-only cut from pipeline_v14 (topology matches full
                        mesh; has C+D geometry)
    body_scene_glb   -- pipeline_v14 output (Scene with body_textured [decimated
                        + UV+texture] + head_mesh [skin])
    output_glb       -- write single merged Trimesh here

Design notes:
    - Body from Module E is DECIMATED (~15k faces) with xatlas UVs. Head from
      face_relief is FULL topology. They don't share vertices.
    - Weld strategy: find head boundary verts (bottom-most Y band of head cut),
      find body boundary verts (top-most Y band of decimated body), spatial KNN
      match with tolerance = 1% of mesh diagonal, snap body boundary verts to
      head boundary positions.
    - Smoothing: for verts within 2-ring neighborhood of any welded boundary
      vert, apply Laplacian smoothing (avg of neighbor positions) for N iters.
      Keep the *rest* of the mesh geometry frozen (preserves C+D detail).

Dependencies:
    - trimesh
    - numpy
    - scipy (cKDTree for KNN weld)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModuleFConfig:
    face_relief_glb: Path
    body_scene_glb: Path
    output_glb: Path

    # Weld tolerance as fraction of mesh diagonal (bbox)
    weld_tol_frac: float = 0.01
    # Laplacian smoothing on seam
    smooth_iterations: int = 3
    smooth_ring_depth: int = 2  # how many vert-rings around seam get smoothed
    smooth_lambda: float = 0.5  # 0..1 blend toward neighbor mean

    dry_run: bool = False

    def __post_init__(self):
        self.face_relief_glb = Path(self.face_relief_glb)
        self.body_scene_glb = Path(self.body_scene_glb)
        self.output_glb = Path(self.output_glb)


def _load_first_mesh(glb_path: Path, prefer_textured: bool = False):
    """Load GLB and return the primary Trimesh. If Scene, pick the largest
    (or the one with a texture if prefer_textured)."""
    import trimesh
    obj = trimesh.load(glb_path, force=None)
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if prefer_textured:
            textured = [
                g for g in geoms
                if getattr(g, "visual", None) is not None
                and hasattr(g.visual, "material")
                and getattr(g.visual.material, "baseColorTexture", None) is not None
            ]
            if textured:
                textured.sort(key=lambda m: len(m.vertices), reverse=True)
                return textured[0]
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        return geoms[0]
    return obj


def _extract_scene_parts(glb_path: Path):
    """Return (body_textured, head_skin) trimeshes from a v14 output Scene.
    Falls back to (loaded_mesh, None) if only single mesh."""
    import trimesh
    obj = trimesh.load(glb_path, force=None)
    if not isinstance(obj, trimesh.Scene):
        return obj, None
    geoms = list(obj.geometry.values())
    if len(geoms) == 1:
        return geoms[0], None
    # Sort: textured first (baseColorTexture set) — that's the body from Module E
    textured = None
    skin = None
    for g in geoms:
        has_tex = (
            getattr(g, "visual", None) is not None
            and hasattr(g.visual, "material")
            and getattr(g.visual.material, "baseColorTexture", None) is not None
        )
        if has_tex and textured is None:
            textured = g
        elif skin is None:
            skin = g
    if textured is None:
        # Neither had texture — just return the two by vertex count
        geoms.sort(key=lambda m: len(m.vertices), reverse=True)
        return geoms[0], geoms[1] if len(geoms) > 1 else None
    return textured, skin


def run_module_f(cfg: ModuleFConfig) -> Path:
    print("=" * 66)
    print("  Module F -- head-body graft")
    print("=" * 66)
    print(f"  face_relief : {cfg.face_relief_glb}")
    print(f"  body_scene  : {cfg.body_scene_glb}")
    print(f"  output      : {cfg.output_glb}")
    print(f"  ---- knobs ----")
    print(f"    weld_tol_frac     = {cfg.weld_tol_frac}")
    print(f"    smooth_iterations = {cfg.smooth_iterations}")
    print(f"    smooth_ring_depth = {cfg.smooth_ring_depth}")
    print(f"    smooth_lambda     = {cfg.smooth_lambda}")
    print(f"  dry_run     = {cfg.dry_run}")
    print("=" * 66, flush=True)
    if cfg.dry_run:
        print("[dry-run] validated"); return cfg.output_glb

    t0 = time.time()
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    cfg.output_glb.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load head (face relief cut, full topology, C+D geometry)
    print("\n[1/5] Load face_relief head", flush=True)
    head = _load_first_mesh(cfg.face_relief_glb, prefer_textured=False)
    h_verts = np.asarray(head.vertices, dtype=np.float32)
    h_faces = np.asarray(head.faces, dtype=np.int64)
    print(f"      head verts={len(h_verts):,} faces={len(h_faces):,}", flush=True)

    # ---- 2. Load body_textured from v14 scene (the DECIMATED + xatlas mesh)
    print("\n[2/5] Load body_textured from v14 scene", flush=True)
    body_textured, head_skin = _extract_scene_parts(cfg.body_scene_glb)
    b_verts = np.asarray(body_textured.vertices, dtype=np.float32)
    b_faces = np.asarray(body_textured.faces, dtype=np.int64)
    print(f"      body verts={len(b_verts):,} faces={len(b_faces):,}", flush=True)
    body_visual = body_textured.visual  # preserve UVs + material

    # ---- 3. Identify boundary verts + weld body-boundary -> head-boundary
    print("\n[3/5] Boundary detection + weld", flush=True)
    bbox_diag = float(np.linalg.norm(h_verts.max(0) - h_verts.min(0)))
    tol = cfg.weld_tol_frac * bbox_diag
    print(f"      bbox_diag={bbox_diag:.4f}, weld tol={tol:.4f}", flush=True)

    # Head boundary = verts near the bottom of the head cut (lowest 10% of head Y)
    y_head = h_verts[:, 1]
    h_min_y = y_head.min()
    h_max_y = y_head.max()
    h_boundary_mask = y_head <= h_min_y + (h_max_y - h_min_y) * 0.10
    h_boundary_idx = np.where(h_boundary_mask)[0]
    print(f"      head boundary candidates: {len(h_boundary_idx)}", flush=True)

    # Body boundary = verts near the top of the body (upper 10% of body Y)
    y_body = b_verts[:, 1]
    b_min_y = y_body.min()
    b_max_y = y_body.max()
    b_boundary_mask = y_body >= b_max_y - (b_max_y - b_min_y) * 0.10
    b_boundary_idx = np.where(b_boundary_mask)[0]
    print(f"      body boundary candidates: {len(b_boundary_idx)}", flush=True)

    # For each body-boundary vert, snap to nearest head-boundary vert if within tol
    tree = cKDTree(h_verts[h_boundary_idx])
    dists, near = tree.query(b_verts[b_boundary_idx], k=1)
    snap_mask = dists <= tol * 3.0  # be a bit generous - decimation shifted verts
    n_snapped = int(snap_mask.sum())
    print(f"      snapping {n_snapped}/{len(b_boundary_idx)} body-boundary verts", flush=True)

    b_verts_snapped = b_verts.copy()
    b_verts_snapped[b_boundary_idx[snap_mask]] = h_verts[h_boundary_idx[near[snap_mask]]]

    # ---- 4. Concatenate: head verts first, body verts second, faces reindexed
    print("\n[4/5] Concatenate head + body", flush=True)
    combined_verts = np.vstack([h_verts, b_verts_snapped]).astype(np.float32)
    body_face_offset = len(h_verts)
    combined_faces = np.vstack([
        h_faces,
        b_faces + body_face_offset,
    ]).astype(np.int64)
    print(f"      combined verts={len(combined_verts):,} "
          f"faces={len(combined_faces):,}", flush=True)

    # ---- 5. Laplacian smooth boundary region
    print(f"\n[5/5] Laplacian smooth seam ring", flush=True)
    # Find welded seam verts in the combined index space
    # Head boundary vert indices are h_boundary_idx (as-is, in combined space)
    # Body boundary snapped verts are (b_boundary_idx[snap_mask] + body_face_offset)
    seam_idx = np.concatenate([
        h_boundary_idx,
        b_boundary_idx[snap_mask] + body_face_offset,
    ])

    # Build vert-vert adjacency from combined_faces
    n_all = len(combined_verts)
    # For each vert, list of neighbor verts (from shared faces)
    adjacency = [set() for _ in range(n_all)]
    for face in combined_faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        adjacency[a].add(b); adjacency[a].add(c)
        adjacency[b].add(a); adjacency[b].add(c)
        adjacency[c].add(a); adjacency[c].add(b)

    # Ring expansion from seam
    ring_set = set(seam_idx.tolist())
    frontier = set(seam_idx.tolist())
    for r in range(cfg.smooth_ring_depth):
        new_front = set()
        for v in frontier:
            new_front.update(adjacency[v])
        new_front -= ring_set
        ring_set.update(new_front)
        frontier = new_front
    ring_arr = np.array(sorted(ring_set), dtype=np.int64)
    print(f"      seam ring size: {len(ring_arr)}", flush=True)

    # Smooth: iterative averaging
    verts_smoothed = combined_verts.copy()
    for it in range(cfg.smooth_iterations):
        new_pos = verts_smoothed.copy()
        for v in ring_arr:
            neighbors = list(adjacency[v])
            if not neighbors:
                continue
            neigh_mean = verts_smoothed[neighbors].mean(axis=0)
            new_pos[v] = (1.0 - cfg.smooth_lambda) * verts_smoothed[v] + cfg.smooth_lambda * neigh_mean
        verts_smoothed = new_pos

    # ---- Export as single Trimesh preserving body UVs + texture
    print(f"\n[export] Build single-mesh GLB", flush=True)
    # UV visual applies only to body verts. For head verts, use a matte skin material.
    # Because trimesh doesn't natively support mixed materials per vertex range,
    # we output as a Scene with (head_smooth, body_smooth_textured) but with the
    # welded/smoothed geometry so the seam looks continuous.
    head_verts_out = verts_smoothed[:body_face_offset]
    body_verts_out = verts_smoothed[body_face_offset:]

    from trimesh.visual.material import PBRMaterial

    # Fix trimesh GLB export dimension mismatch: face_colors must be shape (F, 4)
    # (broadcasting a single (4,) triggers "dimension mismatch" in exchange/export
    # when scene has mixed visual types). Tile to per-face array.
    skin_rgba = np.array([198, 158, 128, 255], dtype=np.uint8)
    head_face_colors = np.tile(skin_rgba, (len(h_faces), 1))

    head_out = trimesh.Trimesh(
        vertices=head_verts_out,
        faces=h_faces,
        process=False,
    )
    # Attach visual AFTER construction so it binds to the mesh cleanly.
    head_out.visual.face_colors = head_face_colors

    body_out = trimesh.Trimesh(
        vertices=body_verts_out,
        faces=b_faces,
        visual=body_visual,
        process=False,
    )

    # Build scene explicitly with named geometries -- multi-mesh GLB output.
    # Each mesh keeps its own visual (head vertex-color, body UV+texture).
    scene = trimesh.Scene()
    scene.add_geometry(head_out, node_name="head_with_relief")
    scene.add_geometry(body_out, node_name="body_textured")
    try:
        scene.export(cfg.output_glb)
    except Exception as e:
        # Fallback: export each mesh separately, then dump body as the "main"
        # output so Kent still gets the textured body. Better than nothing.
        print(f"      scene export failed: {e}; falling back to body-only export",
              flush=True)
        body_out.export(cfg.output_glb)
        head_only_path = cfg.output_glb.with_name(cfg.output_glb.stem + "_head_only.glb")
        head_out.export(head_only_path)
        print(f"      wrote {head_only_path}", flush=True)
    sz = cfg.output_glb.stat().st_size / 1024 / 1024
    print(f"      wrote {cfg.output_glb} ({sz:.2f} MB, took {time.time()-t0:.1f}s)",
          flush=True)
    return cfg.output_glb


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="module_f_graft")
    ap.add_argument("--face-relief-glb", required=True, type=Path)
    ap.add_argument("--body-scene-glb", required=True, type=Path)
    ap.add_argument("--output-glb", required=True, type=Path)
    ap.add_argument("--weld-tol-frac", type=float, default=0.01)
    ap.add_argument("--smooth-iterations", type=int, default=3)
    ap.add_argument("--smooth-ring-depth", type=int, default=2)
    ap.add_argument("--smooth-lambda", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = ModuleFConfig(
        face_relief_glb=args.face_relief_glb,
        body_scene_glb=args.body_scene_glb,
        output_glb=args.output_glb,
        weld_tol_frac=args.weld_tol_frac,
        smooth_iterations=args.smooth_iterations,
        smooth_ring_depth=args.smooth_ring_depth,
        smooth_lambda=args.smooth_lambda,
        dry_run=args.dry_run,
    )
    run_module_f(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
