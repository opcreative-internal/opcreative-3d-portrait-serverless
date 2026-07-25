"""
Pipeline v14 — v13 orchestrator + Kent's alignment concerns addressed.

Per Fable 5 v14 review:
- Section 1: emit fit-transform sidecar + face-relief-only cut for easier
  merge into Kent's base. If --base-mesh given, fit our mesh to base's
  head landmark bbox before export.
- Section 3: Module E now uses spawn context (v13 was fork = CUDA-deadlock)
  + pymeshlab trivial-per-wedge fallback (v13 hung both attempts).
- Section 4: --base-mesh skips Module A automatically (Kent's base already
  provides body geometry; TripoSG output would mismatch frame).

Outputs (all in cfg.output.parent):
    person_2_v14.glb                 -- fused GLB (final)
    person_2_v14_face_relief_only.glb  -- head-region cut, aligned to base
                                         if --base-mesh given (Kent grafts
                                         onto his own body)
    person_2_v14_transform.json      -- 4x4 fit matrix + face_landmarks_3d
                                         for downstream auto-align
    (glb extras.face_landmarks_3d also populated)

Usage (Kent brings base):
    python pipeline_v14.py \\
        --input-image /workspace/inputs/person_2.png \\
        --input-mesh /workspace/outputs/person_2.glb \\
        --base-mesh /workspace/inputs/kent_base.glb \\
        --output /workspace/outputs/person_2_v14.glb \\
        --no-run-module-a

Usage (no base — full figure v13-style):
    python pipeline_v14.py \\
        --input-image /workspace/inputs/person_2.png \\
        --input-mesh /workspace/outputs/person_2.glb \\
        --output /workspace/outputs/person_2_v14.glb \\
        --no-run-module-a --run-module-e
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PipelineV14Config:
    input_image: Path
    output: Path
    input_mesh: Optional[Path] = None       # v13 mesh (post-A, or v11 raw)
    base_mesh: Optional[Path] = None        # Kent's base — triggers fit
    workdir: Path = Path("/workspace")

    run_module_a: bool = False              # default OFF per Fable v14 §4
    run_module_cd: bool = True
    run_module_e: bool = True

    # Module A knobs (only used if run_module_a=True)
    a_num_inference_steps: int = 75
    a_guidance_scale: float = 7.0
    a_faces: int = 500_000
    a_seed: int = 42
    a_timeout_s: int = 300

    # Module C+D (locked)
    cd_dav2_model: str = "base"
    cd_mask_expand: float = 1.2
    cd_face_weight: float = 0.7
    cd_umeyama_clamp_min: float = 0.5
    cd_umeyama_clamp_max: float = 1.5
    cd_emboss_strength: float = 0.010

    # Module E
    e_uv_bake_res: int = 2048
    e_dot_density_scale: float = 1.0
    e_dot_size_px: int = 3
    e_dot_luminance_curve: str = "linear"

    dry_run: bool = False


# ---------------------------------------------------------------------------
# Alignment helpers (Fable 5 §1)
# ---------------------------------------------------------------------------

def detect_base_frame(mesh) -> str:
    """Best-effort frame ID for Kent's base."""
    import numpy as np
    n = len(mesh.vertices)
    ext = np.asarray(mesh.extents, dtype=np.float32)
    if n == 6890:  return "SMPL"
    if n == 10475: return "SMPL-X"
    if 1.4 < ext[1] < 2.1 and ext[1] == max(ext):
        return "metric_humanoid"
    if abs(max(ext) - 1.0) < 0.15:
        return "unit_normalized"
    return "unknown"


def _head_anchors(mesh, lm2d, key_indices):
    """Reuse v11's head-anchor extraction. Returns (K, 3) 3D anchor positions
    on the input mesh in its own frame."""
    import numpy as np
    from pipeline_v11_depth_umeyama import (
        find_head_vertices, project_orthographic,
        match_head_landmarks_to_mp,
    )
    v = np.asarray(mesh.vertices, np.float32)
    head_idx = find_head_vertices(v, 0.20)
    head_verts = v[head_idx]
    head_2d = project_orthographic(head_verts)
    local = match_head_landmarks_to_mp(head_2d, lm2d, key_indices)
    return head_verts[local]


def fit_relief_to_base(relief_mesh, base_mesh, lm2d, key_indices):
    """Compute similarity transform mapping our C+D relief -> base frame,
    apply to relief_mesh, return (transformed_mesh, T_4x4, scale)."""
    import numpy as np
    from pipeline_v11_depth_umeyama import umeyama

    src_anchor = _head_anchors(relief_mesh, lm2d, key_indices)   # (K, 3)
    dst_anchor = _head_anchors(base_mesh,   lm2d, key_indices)   # (K, 3)

    R, t, _, c = umeyama(src_anchor, dst_anchor, with_scale=True)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = (c * R).astype(np.float32)
    T[:3, 3] = t.astype(np.float32)

    relief_mesh.apply_transform(T)
    return relief_mesh, T, float(c), src_anchor, dst_anchor


def emit_face_relief_only(source_mesh, output_path: Path,
                          top_fraction: float = 0.20):
    """Extract head-region faces from post-C+D mesh, save as standalone GLB.
    Kent can graft this onto HIS body/base and skip our body entirely."""
    import numpy as np
    import trimesh
    from pipeline_v11_depth_umeyama import find_head_vertices

    v = np.asarray(source_mesh.vertices, np.float32)
    f = np.asarray(source_mesh.faces, np.int64)
    head_v_idx = find_head_vertices(v, top_fraction)
    head_v_mask = np.zeros(len(v), dtype=bool)
    head_v_mask[head_v_idx] = True
    # Head face = any vert in head band
    head_f_mask = head_v_mask[f].any(axis=1)
    head_f = f[head_f_mask]
    keep = np.unique(head_f.reshape(-1))
    remap = -np.ones(len(v), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    head_out = trimesh.Trimesh(
        vertices=v[keep],
        faces=remap[head_f],
        process=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    head_out.export(output_path)
    return output_path, len(keep), len(head_f)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _sidecar_paths(cfg: PipelineV14Config):
    stem = cfg.output.stem
    parent = cfg.output.parent
    return {
        "cd_out":     parent / f"{stem}_stageCD_face.glb",
        "e_out":      cfg.output,
        "relief":     parent / f"{stem}_face_relief_only.glb",
        "transform":  parent / f"{stem}_transform.json",
    }


def run_pipeline_v14(cfg: PipelineV14Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    paths = _sidecar_paths(cfg)

    print("=" * 66)
    print("  Pipeline v14 — alignment-aware, dual-output")
    print("=" * 66)
    print(f"  input image  : {cfg.input_image}")
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  base mesh    : {cfg.base_mesh or '(none — v13-style full figure)'}")
    print(f"  output       : {cfg.output}")
    print(f"  stages: A={cfg.run_module_a} C+D={cfg.run_module_cd} E={cfg.run_module_e}")
    print("=" * 66, flush=True)

    if cfg.dry_run:
        print("[dry-run] validated"); return cfg.output

    import numpy as np
    import trimesh

    t_all = time.time()
    current_mesh: Optional[Path] = cfg.input_mesh
    detected_frame = "n/a"

    # ---- (Optional) Stage A
    if cfg.run_module_a:
        print(f"\n>>>>> STAGE A <<<<<", flush=True)
        try:
            from module_a_retune import ModuleAConfig, run_module_a
            a_cfg = ModuleAConfig(
                input_image=cfg.input_image,
                output_mesh=paths["cd_out"].with_name("stageA.glb"),
                num_inference_steps=cfg.a_num_inference_steps,
                guidance_scale=cfg.a_guidance_scale,
                faces=cfg.a_faces, seed=cfg.a_seed,
                triposg_repo=cfg.workdir / "TripoSG",
                timeout_s=cfg.a_timeout_s,
            )
            current_mesh = run_module_a(a_cfg)
        except Exception as e:
            print(f"[Stage A FAILED: {e}] — using input_mesh as fallback",
                  flush=True)
            if not (cfg.input_mesh and cfg.input_mesh.exists()):
                raise
            current_mesh = cfg.input_mesh

    if current_mesh is None:
        raise ValueError("No mesh for downstream (A off + no --input-mesh)")

    # ---- Stage C+D (face relief)
    if cfg.run_module_cd:
        print(f"\n>>>>> STAGE C+D <<<<<", flush=True)
        from pipeline_v11_depth_umeyama import Config as V11Config, run_pipeline as v11_run
        v11_cfg = V11Config(
            input_image=cfg.input_image,
            input_mesh=current_mesh,
            output=paths["cd_out"],
            dav2_model=cfg.cd_dav2_model,
            mask_expand=cfg.cd_mask_expand,
            face_weight=cfg.cd_face_weight,
            umeyama_clamp_min=cfg.cd_umeyama_clamp_min,
            umeyama_clamp_max=cfg.cd_umeyama_clamp_max,
            emboss_strength=cfg.cd_emboss_strength,
        )
        current_mesh = v11_run(v11_cfg)

    # ---- Alignment fit (if base provided)
    T = np.eye(4, dtype=np.float32)
    scale = 1.0
    src_anchor = None
    dst_anchor = None
    if cfg.base_mesh and cfg.base_mesh.exists():
        print(f"\n>>>>> ALIGNMENT FIT (base = {cfg.base_mesh.name}) <<<<<",
              flush=True)
        # Detect MP landmarks on photo (same as C+D)
        from pipeline_v11_depth_umeyama import detect_landmarks, MP_ANCHOR_INDICES
        lm = detect_landmarks(cfg.input_image, mp_mode=478)
        lm2d = lm["landmarks_2d"]
        key_indices = np.array(
            [i for i in MP_ANCHOR_INDICES if i < lm["n"]], dtype=np.int64
        )
        # Load our relief mesh + Kent's base
        relief = trimesh.load(current_mesh, force="mesh")
        if isinstance(relief, trimesh.Scene):
            relief = list(relief.geometry.values())[0]
        base = trimesh.load(cfg.base_mesh, force="mesh")
        if isinstance(base, trimesh.Scene):
            base = list(base.geometry.values())[0]
        detected_frame = detect_base_frame(base)
        print(f"      detected base frame: {detected_frame}", flush=True)
        relief, T, scale, src_anchor, dst_anchor = fit_relief_to_base(
            relief, base, lm2d, key_indices,
        )
        fitted_out = paths["cd_out"].with_name(
            paths["cd_out"].stem + "_fitted.glb"
        )
        relief.export(fitted_out)
        current_mesh = fitted_out
        print(f"      T scale={scale:.4f}, saved {fitted_out}", flush=True)

    # ---- Emit face-relief-only cut (always)
    print(f"\n>>>>> HEAD-ONLY CUT <<<<<", flush=True)
    relief_mesh_obj = trimesh.load(current_mesh, force="mesh")
    if isinstance(relief_mesh_obj, trimesh.Scene):
        relief_mesh_obj = list(relief_mesh_obj.geometry.values())[0]
    relief_path, n_verts, n_faces = emit_face_relief_only(
        relief_mesh_obj, paths["relief"], top_fraction=0.20,
    )
    print(f"      {relief_path}: {n_verts:,} verts, {n_faces:,} faces",
          flush=True)

    # ---- Stage E (proper UV texture)
    if cfg.run_module_e:
        print(f"\n>>>>> STAGE E <<<<<", flush=True)
        try:
            from module_e_texture import ModuleEConfig, run_module_e
            e_cfg = ModuleEConfig(
                input_mesh=current_mesh,
                input_image=cfg.input_image,
                output_mesh=paths["e_out"],
                uv_bake_res=cfg.e_uv_bake_res,
                dot_density_scale=cfg.e_dot_density_scale,
                dot_size_px=cfg.e_dot_size_px,
                dot_luminance_curve=cfg.e_dot_luminance_curve,
            )
            current_mesh = run_module_e(e_cfg)
        except Exception as e:
            print(f"[Stage E FAILED: {e}] — copying current mesh as output",
                  flush=True)
            shutil.copy2(current_mesh, cfg.output)
            current_mesh = cfg.output
    else:
        if current_mesh != cfg.output:
            shutil.copy2(current_mesh, cfg.output)
            current_mesh = cfg.output

    # ---- Sidecar transform JSON
    sidecar = {
        "matrix_rowmajor": T.tolist(),
        "scale": scale,
        "src_frame": "triposg_unit",
        "dst_frame": detected_frame,
        "base_mesh": str(cfg.base_mesh) if cfg.base_mesh else None,
        "face_landmarks_src_3d": src_anchor.tolist() if src_anchor is not None else None,
        "face_landmarks_dst_3d": dst_anchor.tolist() if dst_anchor is not None else None,
        "notes": (
            "Apply matrix_rowmajor to person_2_v14_face_relief_only.glb "
            "to place it in dst_frame. If base_mesh is null, matrix is identity "
            "(no base was supplied) and the relief cut is in triposg_unit frame."
        ),
    }
    paths["transform"].write_text(json.dumps(sidecar, indent=2))
    print(f"\n[sidecar] wrote {paths['transform']}", flush=True)

    print(f"\n[PIPELINE v14 DONE] total elapsed: {time.time()-t_all:.1f}s")
    print(f"  final -> {current_mesh}", flush=True)
    return Path(current_mesh)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline_v14")
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--input-mesh", type=Path, default=None,
                    help="Existing mesh (v11 raw or v13 output); required if "
                    "--no-run-module-a")
    ap.add_argument("--base-mesh", type=Path, default=None,
                    help="Kent's base model for alignment fit; if given, "
                    "output is aligned to base frame + relief-only cut is "
                    "similarly aligned")
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    for stage, default in [("module_a", False), ("module_cd", True),
                           ("module_e", True)]:
        arg_true = f"--run-{stage.replace('_', '-')}"
        arg_false = f"--no-run-{stage.replace('_', '-')}"
        ap.add_argument(arg_true, dest=f"run_{stage}", action="store_true",
                        default=default)
        ap.add_argument(arg_false, dest=f"run_{stage}", action="store_false")
    ap.add_argument("--a-num-inference-steps", type=int, default=75)
    ap.add_argument("--a-guidance-scale", type=float, default=7.0)
    ap.add_argument("--a-faces", type=int, default=500_000)
    ap.add_argument("--a-seed", type=int, default=42)
    ap.add_argument("--a-timeout-s", type=int, default=300)
    ap.add_argument("--cd-dav2-model", choices=["small","base","large"], default="base")
    ap.add_argument("--cd-mask-expand", type=float, default=1.2)
    ap.add_argument("--cd-face-weight", type=float, default=0.7)
    ap.add_argument("--cd-umeyama-clamp-min", type=float, default=0.5)
    ap.add_argument("--cd-umeyama-clamp-max", type=float, default=1.5)
    ap.add_argument("--cd-emboss-strength", type=float, default=0.010)
    ap.add_argument("--e-uv-bake-res", type=int, default=2048)
    ap.add_argument("--e-dot-density-scale", type=float, default=1.0)
    ap.add_argument("--e-dot-size-px", type=int, default=3)
    ap.add_argument("--e-dot-luminance-curve",
                    choices=["linear","gamma","invert"], default="linear")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = PipelineV14Config(**{k: v for k, v in vars(args).items()})
    run_pipeline_v14(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
